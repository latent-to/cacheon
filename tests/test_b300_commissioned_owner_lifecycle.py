"""One full-authority B300 owner spans screen and FIFO qualification work."""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.test_b300_arena_provider as provider_fixtures
import tests.test_b300_qualification_deployment as deployment_fixtures
import tests.test_b300_remote_qualification_adapter as remote_fixtures
import tests.test_b300_remote_worker_adapter as worker_fixtures
import tests.test_b300_sealed_qualification_commission as authority_fixtures
from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaQualificationWork,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.execution_disposition import (
    ExecutionDisposition,
    resolve_infrastructure_result,
)
from cacheon.chain.remote_evaluation_dispatcher import RemoteQualificationProduct
from cacheon.chain.publication import publish_worker_bundle
from cacheon.chain.remote_worker_spool import RemoteWorkerError
from cacheon.eval import b300_qualification_commission as commission_module
from cacheon.eval import b300_remote_qualification_adapter as qualification_module
from cacheon.eval import b300_remote_worker_adapter as worker_module
from cacheon.eval.b300_arena_provider import (
    B300ArenaServiceProvider,
    B300DeploymentAuthorities,
    B300ResidentScreenFactory,
)
from cacheon.eval.b300_mainnet_worker import B300MainnetWorker
from cacheon.eval.b300_qualification_commission import (
    CommissionedB300QualificationService,
)
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300QualificationDeployment,
    compose_b300_qualification_deployment,
)
from cacheon.eval.b300_screen_qualification_bridge import QUALIFICATION_EXECUTOR_ID
from cacheon.eval.evidence_store import publish_evidence
from cacheon.eval.qualification_continuation import (
    QualificationContinuationError,
    QualificationContinuationStore,
)
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.eval.resident_screen_lane import ResidentServingScreenStage
from cacheon.stack_identity import canonical_digest
from cacheon.target_catalog import default_target_catalog


RMSNORM_TARGET = "norm.rmsnorm"
ALL_REDUCE = "collective.all_reduce"
FUSED_EXPERTS = "moe.fused_experts"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass
class _CompositionReceipt:
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class _OwnerHarness:
    service: CommissionedB300QualificationService
    deployment: B300QualificationDeployment
    construction: B300QualificationConstructionAuthority
    resident: provider_fixtures._ResidentFactory
    composition: _CompositionReceipt
    worker_init_calls: list[tuple[object, object, object]]
    provider_init_calls: list[tuple[object, object]]


def _deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    construction = remote_fixtures._construction(tmp_path)
    candidate_executor = remote_fixtures._executor(
        tmp_path,
        role=QUALIFICATION_EXECUTOR_ID,
        lane="A",
    )
    baseline_executor = remote_fixtures._executor(
        tmp_path,
        role=QUALIFICATION_EXECUTOR_ID,
        lane="B",
    )
    lane_pair = deployment_fixtures._lane_pair(
        candidate_executor,
        baseline_executor,
    )
    screen = deployment_fixtures._screen_authorities(
        construction,
        candidate_executor,
        baseline_executor,
        lane_pair,
    )
    resident = provider_fixtures._ResidentFactory(tmp_path / "resident-screen")
    screen = replace(
        screen,
        resident_screen_factory=B300ResidentScreenFactory(
            screen.resident_screen_factory.identity_digest,
            screen.resident_screen_factory.resource_ids,
            resident,
        ),
    )
    manifest = deployment_fixtures._manifest(screen)
    construction = remote_fixtures._bind_construction(construction, manifest)
    resident_pair_factory, pair_executors = (
        authority_fixtures._resident_pair_factory(
            tmp_path / "pair",
            monkeypatch,
            manifest.digest,
        )
    )
    deployment = compose_b300_qualification_deployment(
        manifest=manifest,
        screen_authorities=screen,
        construction=construction,
        candidate_executor=candidate_executor,
        resident_baseline_executor=baseline_executor,
        resident_pair_factory=resident_pair_factory,
        screen_lane="primary",
    )
    reproduction_deployment = compose_b300_qualification_deployment(
        manifest=manifest,
        screen_authorities=screen,
        construction=construction,
        candidate_executor=baseline_executor,
        resident_baseline_executor=candidate_executor,
        resident_pair_factory=resident_pair_factory,
        screen_lane="reproduction",
    )
    readiness = remote_fixtures._readiness(deployment)
    commission = worker_module.B300RemoteQualificationCommission(
        deployment,
        construction,
        readiness,
    )
    reproduction_commission = worker_module.B300RemoteQualificationCommission(
        reproduction_deployment,
        construction,
        readiness,
    )
    return (
        deployment,
        construction,
        readiness,
        commission,
        reproduction_commission,
        (candidate_executor, baseline_executor),
        pair_executors,
        resident,
    )


@pytest.fixture
def owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _OwnerHarness:
    (
        deployment,
        construction,
        readiness,
        commission,
        reproduction_commission,
        executors,
        pair_executors,
        resident,
    ) = _deployment(tmp_path, monkeypatch)
    composition = _CompositionReceipt()
    init_calls: list[tuple[object, object, object]] = []
    provider_calls: list[tuple[object, object]] = []
    original_init = B300MainnetWorker.__init__
    original_provider_init = B300ArenaServiceProvider.__init__

    def counted_init(self, manifest, authorities, observed_readiness):
        init_calls.append((manifest, authorities, observed_readiness))
        original_init(self, manifest, authorities, observed_readiness)

    def counted_provider_init(self, manifest, authorities):
        provider_calls.append((manifest, authorities))
        original_provider_init(self, manifest, authorities)

    monkeypatch.setattr(B300MainnetWorker, "__init__", counted_init)
    monkeypatch.setattr(
        B300ArenaServiceProvider,
        "__init__",
        counted_provider_init,
    )
    monkeypatch.setattr(
        commission_module.screen_deployment,
        "replay_commissioned_screen_composition",
        lambda _registration, _ready: (object(), composition, readiness),
    )
    monkeypatch.setattr(
        commission_module,
        "compose_commissioned_qualifications",
        lambda _inputs, observed_composition, observed_readiness, _capabilities: (
            ((commission, reproduction_commission), executors)
            if observed_composition is composition
            and observed_readiness is readiness
            else pytest.fail("builder changed replayed commissioning authority")
        ),
    )
    service = commission_module.build_commissioned_b300_qualification_service(
        {},
        {},
        object(),
    )
    harness = _OwnerHarness(
        service,
        deployment,
        construction,
        resident,
        composition,
        init_calls,
        provider_calls,
    )
    yield harness
    service.close()
    for executor in pair_executors:
        executor.manager.close()


def _target_candidate(
    tmp_path: Path,
    *,
    index: int,
    target_id: str,
) -> ArenaCandidateBinding:
    target = default_target_catalog().require(target_id)
    source = tmp_path / "source"
    kernels = source / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "entry.py").write_text(
        "\n".join(
            f"def run_{member_index}(*args):\n    return args[0]"
            for member_index, _member in enumerate(target.members)
        )
        + "\n",
        encoding="utf-8",
    )
    (kernels / "native.cu").write_text(
        'extern "C" __global__ void cacheon_fixture() {}\n',
        encoding="utf-8",
    )
    manifest = [
        f"bundle_id = 'commissioned-owner-{index}'",
        "abi_version = 'cacheon-op-abi-v0'",
        "",
        "[competition]",
        f"target = '{target_id}'",
        f"mode = '{'atomic' if len(target.members) > 1 else 'slot'}'",
    ]
    for member_index, member in enumerate(target.members):
        manifest.extend(
            (
                "",
                "[[ops]]",
                f"slot = '{member}'",
                "source = 'kernels/entry.py'",
                f"entry = 'run_{member_index}'",
                "dtypes = ['bfloat16']",
                "architectures = ['sm103']",
                "cuda_sources = ['kernels/native.cu']",
            )
        )
    (source / "manifest.toml").write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )
    for path in sorted(source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)
    publication = publish_worker_bundle(
        source,
        tmp_path / "publications",
        content_hash(source),
    )
    reservation = QualificationReservation(
        _h(f"{target_id}-reservation-{index}"),
        publication.digest,
        target_id,
        _h(f"{target_id}-delta-{index}"),
        index,
        f"miner-{index}",
        20,
        index,
        0,
        target.members,
    )
    return ArenaCandidateBinding(
        reservation,
        publication,
        1,
    )


def _configured(
    owner: _OwnerHarness,
    candidate: ArenaCandidateBinding,
    continuation_root: Path,
) -> remote_fixtures._Configured:
    receipt = deployment_fixtures._receipt(
        owner.deployment.manifest.digest,
        candidate,
    )
    adapter = owner.service.adapter_for(
        (candidate.publication,),
        QualificationContinuationStore(continuation_root),
        "primary",
    )
    return remote_fixtures._Configured(
        owner.deployment,
        owner.construction,
        owner.service.commission.readiness,
        candidate,
        receipt,
        adapter,
        (),
    )


def _passing_resident_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ResidentServingScreenStage,
        "run_screen",
        lambda _stage, _candidate: ScreenStageResult(
            "abbreviated_serving",
            ScreenGrade.PASS,
            _h("resident-screen-pass"),
            1,
        ),
    )


def test_service_routes_reproduction_to_swapped_lane_owner(
    owner: _OwnerHarness,
    tmp_path: Path,
) -> None:
    candidate = _target_candidate(
        tmp_path / "reproduction-candidate",
        index=99,
        target_id=RMSNORM_TARGET,
    )
    store = QualificationContinuationStore(tmp_path / "reproduction-continuation")
    adapter = owner.service.adapter_for(
        (candidate.publication,),
        store,
        "reproduction",
    )
    worker = adapter.worker
    assert worker is owner.service._reproduction_worker
    assert worker is not owner.service.worker
    assert worker._remote_qualification_lane == "reproduction"
    authorities = worker._provider._authorities
    assert tuple(authorities.executor.device_policy.physical_gpu_ids) == (4, 5, 6, 7)
    assert tuple(
        authorities.resident_baseline_executor.device_policy.physical_gpu_ids
    ) == (0, 1, 2, 3)
    assert owner.service.adapter_for(
        (candidate.publication,),
        store,
        "reproduction",
    ).worker is worker


def test_full_owner_releases_its_screen_resident_and_closes_once(
    owner: _OwnerHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = owner.service.worker
    provider = worker._provider
    assert len(owner.worker_init_calls) == 1
    assert owner.provider_init_calls == [
        (owner.deployment.manifest, owner.deployment.authorities)
    ]
    assert owner.worker_init_calls[0] == (
        owner.deployment.manifest,
        owner.deployment.authorities,
        owner.service.commission.readiness,
    )
    assert type(provider._authorities) is B300DeploymentAuthorities
    assert provider._authorities is owner.deployment.authorities
    assert worker._remote_qualification_lane == "primary"

    _passing_resident_screen(monkeypatch)
    candidate = _target_candidate(
        tmp_path / "screen-candidate",
        index=0,
        target_id=RMSNORM_TARGET,
    )
    receipt = worker.service.screen(candidate)
    assert owner.resident.created == 1
    assert owner.resident.closed == 0
    assert worker.service._provider is provider

    builder = provider_fixtures._FactoryBuilder()
    provider._qualification_capabilities = replace(
        owner.deployment.authorities,
        qualification_factory_builder=builder,
        deadline_provider=lambda _request, _state: time.monotonic() + 60.0,
    )
    work = worker.service.plan_qualification((candidate,), (receipt,))
    assert type(work) is ArenaQualificationWork
    assert builder.calls[0][0].candidates == (candidate,)
    assert owner.resident.closed == 1
    assert worker.service._provider is provider

    worker_close = worker.close
    worker_close_calls: list[object] = []

    def close_worker() -> None:
        worker_close_calls.append(provider)
        worker_close()

    monkeypatch.setattr(worker, "close", close_worker)
    manager_calls: dict[int, int] = {}
    managers = {
        id(executor.manager): executor.manager
        for executor in owner.service._executors
    }.values()
    for manager in managers:
        original_close = manager.close
        manager_calls[id(manager)] = 0

        def close_manager(manager=manager, original_close=original_close) -> None:
            manager_calls[id(manager)] += 1
            original_close()

        monkeypatch.setattr(manager, "close", close_manager)

    owner.service.close()
    owner.service.close()
    assert worker_close_calls == [provider]
    assert owner.composition.close_calls == 1
    assert set(manager_calls.values()) == {1}


def test_one_owner_routes_heterogeneous_singletons(
    owner: _OwnerHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _passing_resident_screen(monkeypatch)
    first = _target_candidate(
        tmp_path / "msa-screen",
        index=10,
        target_id=RMSNORM_TARGET,
    )
    owner.service.worker.service.screen(first)
    provider = owner.service.worker._provider
    assert owner.resident.created == 1

    reference = publish_evidence(
        owner.construction.evidence_root,
        b'{"commissioned":"owner"}',
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="cacheon.qualification.cohort-attempt.v1",
    )
    first_configured = _configured(
        owner,
        first,
        tmp_path / "continuation-msa",
    )
    calls = remote_fixtures._patch_worker_result(
        monkeypatch,
        first_configured,
        reference,
    )
    products: list[RemoteQualificationProduct] = []
    requests = []
    targets = (RMSNORM_TARGET, ALL_REDUCE, FUSED_EXPERTS)
    for index, target_id in enumerate(targets, start=10):
        candidate = (
            first
            if target_id == RMSNORM_TARGET
            else _target_candidate(
                tmp_path / f"target-{index}",
                index=index,
                target_id=target_id,
            )
        )
        configured = _configured(
            owner,
            candidate,
            tmp_path / f"continuation-{index}",
        )
        request = remote_fixtures._request(configured)
        requests.append(request)
        assert configured.adapter.worker is owner.service.worker
        assert not configured.adapter._owns_worker
        products.append(configured.adapter.run(request))
        assert owner.service.worker._provider is provider
        assert not owner.service.worker._closed

    assert len(owner.worker_init_calls) == 1
    assert len(owner.provider_init_calls) == 1
    assert len(calls) == 3
    assert tuple(
        product.authority_manifest.reservations[0].target_id
        for product in products
    ) == targets
    assert all(len(request.body["candidates"]) == 1 for request in requests)
    assert all(len(request.members) == 1 for request in requests)
    assert products[0].authority_manifest.reservations[0].target_members == (
        RMSNORM_TARGET,
    )
    assert products[1].authority_manifest.reservations[0].target_members == (
        ALL_REDUCE,
    )

    assert products[2].authority_manifest.reservations[0].target_members == (
        FUSED_EXPERTS,
    )

    production_source = inspect.getsource(
        qualification_module.B300RemoteQualificationAdapter.run
    ) + inspect.getsource(worker_module.AdapterRuntime.qualification_adapter_for)
    assert all(target_id not in production_source for target_id in targets)

    before_drift = len(calls)
    lane_body = remote_fixtures._body(
        first_configured,
        screen_lane="reproduction",
    )
    with pytest.raises(
        qualification_module.B300RemoteQualificationAdapterError,
        match="differs from deployment",
    ):
        first_configured.adapter.run(
            remote_fixtures._request(first_configured, body=lane_body)
        )
    drifted_readiness = replace(
        first_configured.readiness,
        ready_epoch=first_configured.readiness.ready_epoch + 1,
    )
    with pytest.raises(
        qualification_module.B300RemoteQualificationAdapterError,
        match="differs from deployment",
    ):
        first_configured.adapter.run(
            remote_fixtures._request(
                first_configured,
                readiness=drifted_readiness,
            )
        )
    assert len(calls) == before_drift
    assert not owner.service.worker._closed


def test_pre_entry_refusal_and_post_entry_failure_never_replace_owner(
    owner: _OwnerHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(worker_module.AdapterRuntime)
    runtime.paths = worker_fixtures._adapter_paths(tmp_path)
    runtime.registration = {}
    runtime.ready = {}
    runtime.credential = object()
    runtime.identity = object()
    runtime.worker = owner.service.worker
    runtime.qualification_commission = owner.service.commission
    runtime.qualification_continuation_store = QualificationContinuationStore(
        runtime.paths.continuation_root
    )
    runtime._commissioned_service = owner.service
    runtime.closed = False
    runtime.verify_current = lambda: None

    monkeypatch.setattr(worker_module, "load_json", lambda _path, **_kwargs: {})

    def reject_carrier(*_args, **_kwargs):
        raise RemoteWorkerError("malformed authenticated carrier")

    monkeypatch.setattr(worker_module, "verify_request", reject_carrier)
    with pytest.raises(worker_module.AdapterRequestFailed):
        worker_module.run_with_runtime(
            tmp_path / ("b" * 64),
            tmp_path / "bad-result",
            runtime,
        )
    assert len(owner.worker_init_calls) == 1
    assert not owner.service.worker._closed

    candidate = _target_candidate(
        tmp_path / "delegated-candidate",
        index=30,
        target_id=ALL_REDUCE,
    )
    _passing_resident_screen(monkeypatch)
    receipt = runtime.worker.service.screen(candidate)
    assert receipt.decision is PromotionDecision.PROMOTE
    assert owner.resident.closed == 0
    wire = SimpleNamespace(
        body={
            "candidates": [{"publication": candidate.publication.to_dict()}],
            "screen_lane": "reproduction",
            "incumbent_stack_digest": owner.construction.incumbent_stack.digest,
            "incumbent_tree_digest": owner.construction.incumbent_tree_digest,
        }
    )
    worker_fixtures._patch_authenticated_carrier(
        monkeypatch,
        stage="qualification",
        wire=wire,
    )
    monkeypatch.setattr(
        worker_module,
        "resolve_cohort_publications",
        lambda *_args: (candidate.publication,),
    )
    delegated: list[B300MainnetWorker] = []

    def fail_after_entry(self, observed_wire):
        assert observed_wire is wire
        assert self.worker is owner.service._reproduction_worker
        assert owner.resident.closed == 1
        delegated.append(self.worker)
        raise QualificationContinuationError("durable continuation is ambiguous")

    monkeypatch.setattr(
        qualification_module.B300RemoteQualificationAdapter,
        "run",
        fail_after_entry,
    )
    result_dir = tmp_path / "delegated-result"
    result_dir.mkdir(mode=0o700)
    with pytest.raises(worker_module.AdapterEpochFailed) as captured:
        worker_module.run_with_runtime(
            tmp_path / ("d" * 64),
            result_dir,
            runtime,
        )
    assert isinstance(captured.value.__cause__, QualificationContinuationError)
    assert delegated == [owner.service._reproduction_worker]
    assert len(owner.worker_init_calls) == 2
    assert runtime.worker is owner.service.worker
    assert not owner.service.worker._closed
    assert (result_dir / "RESIDENT_ENTRY_ARMED.json").is_file()

    hold = resolve_infrastructure_result(
        "adapter_epoch_failed",
        None,
        request_id=canonical_digest("commissioned-owner-request", {}),
    )
    assert hold.disposition is ExecutionDisposition.HOLD
    assert hold.decision == "NO_DECISION"
    assert hold.failure_code == "adapter_epoch_failed"


def test_standalone_adapter_closes_only_its_owned_worker_once(
    owner: _OwnerHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _target_candidate(
        tmp_path / "standalone-candidate",
        index=40,
        target_id=RMSNORM_TARGET,
    )
    standalone = qualification_module.B300RemoteQualificationAdapter(
        owner.deployment,
        owner.construction,
        owner.service.commission.readiness,
        qualification_module.B300WorkerBundleResolver((candidate.publication,)),
        QualificationContinuationStore(tmp_path / "standalone-continuation"),
    )
    assert standalone._owns_worker
    assert standalone.worker is not owner.service.worker
    assert len(owner.worker_init_calls) == 2
    assert len(owner.provider_init_calls) == 2
    owned_worker = standalone.worker
    assert owned_worker is not None
    original_close = owned_worker.close
    close_calls: list[B300MainnetWorker] = []

    def close_owned() -> None:
        close_calls.append(owned_worker)
        original_close()

    monkeypatch.setattr(owned_worker, "close", close_owned)
    with standalone as entered:
        assert entered is standalone
    standalone.close()
    assert close_calls == [owned_worker]
    assert standalone._closed
    assert not owner.service.worker._closed


def test_injected_commission_runtime_owns_one_full_worker_and_closes_once(
    owner: _OwnerHarness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = worker_fixtures._adapter_paths(tmp_path / "injected-runtime")
    readiness = owner.service.commission.readiness
    registration = {
        "ready_receipt_digest": readiness.ready_receipt_digest,
        "worker_epoch": f"{readiness.ready_epoch:032x}",
        "worker_readiness": readiness.to_dict(),
        "worker_readiness_digest": readiness.digest,
    }
    ready = {"receipt_digest": readiness.ready_receipt_digest}
    monkeypatch.setattr(
        worker_module,
        "load_json",
        lambda path: registration if path == paths.registration else ready,
    )
    monkeypatch.setattr(worker_module, "verify_registration", lambda value: value)
    monkeypatch.setattr(worker_module, "verify_ready_receipt", lambda value: value)
    monkeypatch.setattr(
        worker_module,
        "registration_credential",
        lambda _registration, _path: object(),
    )
    monkeypatch.setattr(
        worker_module,
        "registration_transport_identity",
        lambda _registration: object(),
    )
    drifted_commission = worker_module.B300RemoteQualificationCommission(
        owner.deployment,
        owner.construction,
        replace(readiness, ready_epoch=readiness.ready_epoch + 1),
    )
    with pytest.raises(
        worker_module.AdapterError,
        match="differs from registered READY",
    ):
        worker_module.AdapterRuntime(
            paths,
            qualification_commission=drifted_commission,
        )
    assert len(owner.worker_init_calls) == 1
    assert len(owner.provider_init_calls) == 1

    runtime = worker_module.AdapterRuntime(
        paths,
        qualification_commission=owner.service.commission,
    )
    assert len(owner.worker_init_calls) == 2
    assert len(owner.provider_init_calls) == 2
    assert type(runtime.worker) is B300MainnetWorker
    assert type(runtime.worker._provider._authorities) is B300DeploymentAuthorities
    assert runtime.worker._provider._authorities is owner.deployment.authorities

    candidate = _target_candidate(
        tmp_path / "injected-candidate",
        index=41,
        target_id=ALL_REDUCE,
    )
    request_adapter = runtime.qualification_adapter_for(
        (candidate.publication,),
        "primary",
    )
    assert request_adapter.worker is runtime.worker
    assert not request_adapter._owns_worker
    request_adapter.close()
    request_adapter.close()
    assert not runtime.worker._closed

    runtime_worker = runtime.worker
    original_close = runtime_worker.close
    close_calls: list[B300MainnetWorker] = []

    def close_runtime_worker() -> None:
        close_calls.append(runtime_worker)
        original_close()

    monkeypatch.setattr(runtime_worker, "close", close_runtime_worker)
    runtime.close()
    runtime.close()
    assert close_calls == [runtime_worker]
    assert runtime_worker._closed
    assert not owner.service.worker._closed

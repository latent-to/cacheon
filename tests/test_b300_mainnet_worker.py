"""CPU-only contracts for the closed B300 mainnet worker boundary."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

import cacheon.eval.b300_mainnet_worker as worker_module
from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaQualificationRequest,
    ArenaQualificationWork,
    ArenaService,
    ArenaServiceManifest,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.evaluation_coordinator import (
    ClaimedQualificationEvaluation,
    WorkerReadiness,
)
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.intake import FinalizedArrival, IntakeReservation
from cacheon.chain.publication import publish_worker_bundle
from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldReason
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.b300_arena_provider import (
    B300ArenaProviderError,
    B300ArenaServiceProvider,
    B300DeclaredAuthorities,
    B300DeploymentAuthorities,
    B300QualificationLanePair,
    b300_arena_provider_digest,
)
from cacheon.eval.b300_mainnet_worker import (
    B300MainnetWorker,
    B300MainnetWorkerError,
    B300RemoteQualificationRun,
)
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeError,
    QualificationIntakeOutcome,
    QualificationPlanFactory,
    QualificationReservation,
    QualificationRetryPlan,
)
from tests.support.b300 import (
    FactoryBuilder as _FactoryBuilder,
    arena_manifest as _manifest,
    deployment_authorities as _authorities,
    executors,
    sha as _h,
)


SLOT = "activation.silu_and_mul"


@pytest.fixture
def executor_factory(tmp_path: Path):
    yield from executors(tmp_path)


def _readiness(
    manifest: ArenaServiceManifest,
    authorities: B300DeploymentAuthorities,
) -> WorkerReadiness:
    provider = B300ArenaServiceProvider(manifest, authorities)
    service = ArenaService(manifest, provider)
    try:
        return WorkerReadiness.for_service(
            service,
            ready_receipt_digest=_h("ready-receipt"),
            ready_epoch=7,
        )
    finally:
        provider.close()


def _bundle(tmp_path: Path, index: int) -> Path:
    source = tmp_path / f"source-{index}"
    kernels = source / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "entry.py").write_text("def run(x, out):\n    return None\n")
    (kernels / "native.cu").write_text(
        'extern "C" __global__ void cacheon_fixture() {}\n'
    )
    (source / "rebuild.json").write_text('{"steps": []}\n')
    (source / "manifest.toml").write_text(
        "\n".join(
            (
                f"bundle_id = 'b300-worker-fixture-{index}'",
                "abi_version = 'cacheon-op-abi-v0'",
                "[[ops]]",
                f"slot = '{SLOT}'",
                "source = 'kernels/entry.py'",
                "entry = 'run'",
                "dtypes = ['bfloat16']",
                "cuda_sources = ['kernels/native.cu']",
            )
        )
        + "\n"
    )
    for path in sorted(source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)
    return source


def _bound_row(tmp_path: Path, manifest: ArenaServiceManifest, index: int):
    source = _bundle(tmp_path, index)
    committed = content_hash(source)
    publication = publish_worker_bundle(
        source,
        tmp_path / "publications",
        committed,
    )
    arrival = FinalizedArrival(
        f"miner-{index}",
        committed,
        f"https://example.invalid/{index}",
        20,
        "0x" + f"{20:064x}",
        index,
    )
    fingerprint = SubmittedDeltaFingerprint(
        "component",
        SLOT,
        _h(f"target-spec-{index}"),
        (SLOT,),
        _h(f"exact-payload-{index}"),
        _h(f"selected-delta-{index}"),
        _h(f"normalized-delta-{index}"),
        (_h(f"contained-{index}"),),
        (_h(f"advisory-{index}"),),
    )
    reservation = IntakeReservation(
        reservation_id=arrival.reservation_id,
        arrival=arrival,
        admission_epoch=0,
        status="published",
        target_id=SLOT,
        target_members=(SLOT,),
        delta_fingerprint=fingerprint,
        transport_attempts=1,
        publication_digest=publication.digest,
        publication_root=publication.root,
        qualification_authority_digest="",
        qualification_evidence_digest="",
        arena_service_digest=manifest.digest,
        screen_lane="primary",
        decision="",
        reason="",
    )
    authority = QualificationReservation(
        reservation.reservation_id,
        publication.digest,
        SLOT,
        fingerprint.selected_delta_digest,
        index,
        arrival.hotkey,
        arrival.block,
        arrival.event_index,
        arrival.event_subindex,
        (SLOT,),
    )
    candidate = ArenaCandidateBinding(authority, publication, 1)
    return reservation, publication, candidate


def _qualification_claim(
    tmp_path: Path,
    manifest: ArenaServiceManifest,
    *,
    count: int = 2,
) -> ClaimedQualificationEvaluation:
    rows = tuple(_bound_row(tmp_path, manifest, index) for index in range(count))
    reservations = tuple(row[0] for row in rows)
    lease = EvaluationLease(
        _h(f"qualification-lease-{count}"),
        1,
        "qualification",
        "b300-worker-test",
        tuple(
            EvaluationLeaseMember(row.reservation_id, "published")
            for row in reservations
        ),
        20,
        40,
        40,
    )
    return ClaimedQualificationEvaluation(
        lease,
        reservations,
        tuple(row[1] for row in rows),
        tuple(row[2] for row in rows),
    )


def _systemic_batch(factory: QualificationPlanFactory) -> QualificationIntakeBatch:
    failure = _h("qualification-plan-failure")
    outcomes = tuple(
        QualificationIntakeOutcome(
            row.reservation_digest,
            row.selected_delta_digest,
            factory.manifest.digest,
            QualificationDecision.NO_DECISION,
            "qualification_plan",
            True,
            failure_digest=failure,
        )
        for row in factory.manifest.reservations
    )
    ids = tuple(row.reservation_digest for row in factory.manifest.reservations)
    groups = ((ids[0],),) if len(ids) == 1 else ((ids[0],), (ids[1],))
    return QualificationIntakeBatch(
        factory.manifest.digest,
        outcomes,
        retry_plan=QualificationRetryPlan(
            factory.manifest.digest,
            "requeue" if len(ids) == 1 else "bisect",
            groups,
            failure,
        ),
    )


def _fake_remote_intake(monkeypatch, batch_for):
    """Route the remote path through an intake that returns ``batch_for(factory)``.

    The prebuilt plan is a sentinel, so the worker's ``_validate_batch`` and
    disposition lines run on a chosen batch without a real engine session.
    """

    sentinel_plan = object()
    intake_calls = []

    def fake_intake(factory, **kwargs):
        assert kwargs["prebuilt_plan"] is sentinel_plan
        intake_calls.append(factory)
        return batch_for(factory)

    monkeypatch.setattr(QualificationPlanFactory, "build", lambda self: sentinel_plan)
    monkeypatch.setattr(worker_module, "run_qualification_intake", fake_intake)
    return intake_calls


def test_remote_qualification_releases_systemic_batch(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)
    intake_calls = _fake_remote_intake(monkeypatch, _systemic_batch)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            screen_lane="primary",
            continuation_store=QualificationContinuationStore(tmp_path / "continuation"),
            request_digest=_h("remote-request"),
        )

        assert type(result) is B300RemoteQualificationRun
        assert type(result.run.payload) is QualificationIntakeBatch
        assert result.run.disposition == "released"
        assert result.run.envelope.payload_kind == "qualification_intake_batch"
        assert tuple(
            row.reservation_digest for row in result.run.payload.outcomes
        ) == claim.lease.reservation_ids
        assert result.authority_manifest is intake_calls[0].manifest
        assert len(intake_calls) == 1
        assert builder.calls[0][1] is None
        result.run.envelope.verify(
            claim.lease, readiness, worker.service, result.run.payload
        )
    finally:
        worker.close()


def test_remote_qualification_refuses_lane_or_cohort_drift(
    tmp_path: Path,
    executor_factory,
) -> None:
    authorities, _builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    continuation = QualificationContinuationStore(tmp_path / "continuation")
    request_digest = _h("remote-request")
    try:
        with pytest.raises(
            B300MainnetWorkerError, match="authenticated request digest"
        ):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                screen_lane="primary",
                continuation_store=continuation,
                request_digest=None,
            )
        with pytest.raises(B300MainnetWorkerError, match="exact leased cohort"):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                screen_lane="reproduction",
                continuation_store=continuation,
                request_digest=request_digest,
            )
        with pytest.raises(B300MainnetWorkerError, match="exact leased cohort"):
            worker.run_remote_qualification(
                claim.lease,
                tuple(reversed(claim.candidates)),
                screen_lane="primary",
                continuation_store=continuation,
                request_digest=request_digest,
            )
    finally:
        worker.close()


def test_resident_hold_preserves_the_root_exception_chain() -> None:
    try:
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as inner:
            raise ValueError("stock restoration failed") from inner
    except ValueError as failure:
        hold = worker_module._resident_evidence_hold(
            _h("request"), _h("authority"), _h("source"), failure
        )
    assert hold.reason is RemoteQualificationHoldReason.RESIDENT_EVIDENCE_UNAVAILABLE
    assert hold.failure_type == "ValueError"
    assert "stock restoration failed" in hold.failure_message
    assert "CUDA out of memory" in hold.failure_message


def test_remote_qualification_stage_is_derived_from_swapped_executor_authority(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    primary, _builder = _authorities(executor_factory)
    manifest = _manifest(primary)
    readiness = _readiness(manifest, primary)
    original = _qualification_claim(tmp_path / "cohort", manifest, count=1)
    claim = dataclasses.replace(
        original,
        reservations=tuple(
            dataclasses.replace(row, screen_lane="reproduction")
            for row in original.reservations
        ),
    )
    reproduction = dataclasses.replace(
        primary,
        executor=executor_factory("candidate", "B"),
        resident_baseline_executor=executor_factory("resident_baseline", "A"),
        qualification_stage="reproduction",
    )
    assert b300_arena_provider_digest(reproduction) == manifest.provider_digest
    continuation = QualificationContinuationStore(tmp_path / "continuation")
    request_digest = _h("remote-request")

    intake_calls = _fake_remote_intake(monkeypatch, _systemic_batch)
    worker = B300MainnetWorker(manifest, reproduction, readiness)
    try:
        assert worker._remote_qualification_lane == "reproduction"
        with pytest.raises(B300MainnetWorkerError, match="exact leased cohort"):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                screen_lane="primary",
                continuation_store=continuation,
                request_digest=request_digest,
            )
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            screen_lane="reproduction",
            continuation_store=continuation,
            request_digest=request_digest,
        )
        assert type(result) is B300RemoteQualificationRun
        assert result.screen_lane == "reproduction"
        assert len(intake_calls) == 1
    finally:
        worker.close()


def test_remote_qualification_refuses_a_plan_it_cannot_reopen(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, _builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)

    def substituted(self):
        raise QualificationIntakeError("qualification plan was substituted")

    monkeypatch.setattr(QualificationPlanFactory, "build", substituted)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        with pytest.raises(
            B300MainnetWorkerError, match="could not reopen the prebuilt"
        ):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                screen_lane="primary",
                continuation_store=QualificationContinuationStore(
                    tmp_path / "continuation"
                ),
                request_digest=_h("remote-request"),
            )
    finally:
        worker.close()


def test_qualification_refuses_result_that_reorders_leased_cohort(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, _builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)

    def reordered(factory):
        batch = _systemic_batch(factory)
        failure = batch.retry_plan.failure_digest
        outcomes = tuple(reversed(batch.outcomes))
        groups = tuple((row.reservation_digest,) for row in outcomes)
        return QualificationIntakeBatch(
            batch.authority_manifest_digest,
            outcomes,
            retry_plan=QualificationRetryPlan(
                batch.authority_manifest_digest,
                "bisect",
                groups,
                failure,
            ),
        )

    _fake_remote_intake(monkeypatch, reordered)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        with pytest.raises(B300MainnetWorkerError, match="exact leased cohort"):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                screen_lane="primary",
                continuation_store=QualificationContinuationStore(
                    tmp_path / "continuation"
                ),
                request_digest=_h("remote-request"),
            )
    finally:
        worker.close()


def test_worker_has_no_dynamic_dispatch_surface() -> None:
    source = inspect.getsource(worker_module)
    for forbidden in (
        "importlib",
        "subprocess",
        "__import__",
        "shell=True",
        "os.system",
        "runpy",
    ):
        assert forbidden not in source


def test_readiness_drift_and_declared_only_authority_are_rejected_before_work(
    executor_factory,
) -> None:
    authorities, builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)

    with pytest.raises(B300MainnetWorkerError, match="readiness differs"):
        B300MainnetWorker(
            manifest,
            authorities,
            dataclasses.replace(readiness, service_digest=_h("different-service")),
        )
    # A commissioning declaration carries the same provider identity but no
    # executors, judge, or deadline, so it cannot become a worker.
    declared = B300DeclaredAuthorities(
        authorities.runtime_identity, authorities.qualification
    )
    with pytest.raises(B300MainnetWorkerError, match="authorities are not exact"):
        B300MainnetWorker(manifest, declared, readiness)
    assert builder.calls == []


def test_qualification_preserves_exact_request_order_and_real_authorities(
    tmp_path: Path, executor_factory
) -> None:
    authorities, builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    first = _bound_row(tmp_path / "first", manifest, 0)[2]
    second = _bound_row(tmp_path / "second", manifest, 1)[2]

    work = service.plan_qualification((first, second), state={"attempt": 1})

    assert type(work) is ArenaQualificationWork
    assert type(work.factory) is QualificationPlanFactory
    assert work.factory.manifest.reservations == (first.reservation, second.reservation)
    assert type(work.executor) is OCIEngineExecutor
    assert work.executor is authorities.executor
    assert work.resident_baseline_executor is authorities.resident_baseline_executor
    assert work.entropy_provider is authorities.entropy_provider
    assert work.hidden_judge is authorities.hidden_judge
    assert builder.calls[0][0].candidates == (first, second)
    assert builder.calls[0][1] == {"attempt": 1}


def test_runtime_model_topology_and_policy_must_match_manifest(executor_factory) -> None:
    authorities, _builder = _authorities(executor_factory)
    runtime_mismatch = dataclasses.replace(
        authorities.runtime_identity, model_content_digest=_h("another-model")
    )
    with pytest.raises(B300ArenaProviderError, match="runtime, model, topology"):
        B300ArenaServiceProvider(_manifest(authorities, runtime=runtime_mismatch), authorities)
    with pytest.raises(B300ArenaProviderError, match="qualification policy"):
        B300ArenaServiceProvider(
            _manifest(authorities, qualification_policy_digest=_h("another-policy")),
            authorities,
        )


def test_lane_pair_rejects_overlap_and_authority_rejects_lane_drift(executor_factory) -> None:
    # Two physical TP4 lanes never share a GPU, and an executor that drifts onto
    # the other lane after construction loses the provider identity.
    primary, _builder = _authorities(executor_factory)
    with pytest.raises(B300ArenaProviderError, match="overlapping"):
        B300QualificationLanePair(
            primary.qualification_lane_pair.lane_a,
            dataclasses.replace(primary.qualification_lane_pair.lane_a, lane_id="B"),
        )
    primary.executor.device_policy = executor_factory("candidate", "B").device_policy
    with pytest.raises(B300ArenaProviderError, match="selected physical TP4 lane"):
        b300_arena_provider_digest(primary)


def test_declared_only_provider_refuses_to_build_qualification(
    tmp_path: Path, executor_factory
) -> None:
    full, builder = _authorities(executor_factory)
    manifest = _manifest(full)
    sealed = B300DeclaredAuthorities(full.runtime_identity, full.qualification)
    request = ArenaQualificationRequest(
        manifest.digest,
        manifest.qualification_policy_digest,
        (_bound_row(tmp_path / "candidate", manifest, 0)[2],),
    )
    with pytest.raises(B300ArenaProviderError, match="declared-only provider"):
        B300ArenaServiceProvider(manifest, sealed).build_qualification(request)
    assert builder.calls == []


def test_closed_provider_refuses_qualification(tmp_path: Path, executor_factory) -> None:
    authorities, builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    provider = B300ArenaServiceProvider(manifest, authorities)
    service = ArenaService(manifest, provider)
    provider.close()

    with pytest.raises(B300ArenaProviderError, match="provider is closed"):
        service.plan_qualification((_bound_row(tmp_path / "candidate", manifest, 0)[2],))
    assert builder.calls == []


def test_factory_exception_stays_a_provider_error(tmp_path: Path, executor_factory) -> None:
    authorities, _builder = _authorities(executor_factory, builder=_FactoryBuilder(fail=True))
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))

    with pytest.raises(B300ArenaProviderError, match="factory construction"):
        service.plan_qualification((_bound_row(tmp_path / "candidate", manifest, 0)[2],))

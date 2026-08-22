"""CPU-only contracts for the closed B300 mainnet worker boundary."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import os
import time
from pathlib import Path

import pytest

import cacheon.eval.b300_mainnet_worker as worker_module
import tests.test_b300_sealed_qualification_commission as authority_fixtures
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    PromotionDecision,
    ScreenGrade,
    ScreenStagePolicy,
    ScreenStageResult,
    Workload,
    WorkloadCell,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.evaluation_coordinator import (
    ClaimedQualificationEvaluation,
    ClaimedScreenEvaluation,
    EvaluationResultEnvelope,
    EvaluationRun,
    WorkerReadiness,
)
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.intake import FinalizedArrival, IntakeReservation
from cacheon.chain.publication import publish_worker_bundle
from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldReason
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.b300_arena_provider import (
    B300ArenaServiceProvider,
    B300DeploymentAuthorities,
    B300QualificationLanePair,
    B300QualificationLanePolicy,
    B300ResidentScreenFactory,
    B300ResidentScreenLifetime,
    B300ScreenStageHandler,
    b300_arena_provider_digest,
)
from cacheon.eval.b300_mainnet_worker import (
    B300MainnetWorker,
    B300MainnetWorkerError,
    B300RemoteQualificationRun,
)
from cacheon.eval.b300_qualification_graph_gate import (
    B300QualificationGraphGateHold,
)
from cacheon.eval.device_state import DeviceStatePolicy, GPUConfiguration
from cacheon.eval.oci_backend import (
    OCIBackendConfig,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
)
from cacheon.eval.oci_prebuild import OCIPrebuildConfig, OCIPrebuildPolicy
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationPlanFactory,
    QualificationReservation,
    QualificationRetryPlan,
)
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.resident_queue import ScreenPolicy
from cacheon.eval.resident_screen_lane import (
    ResidentScreenLane,
    ResidentServingScreenStage,
)


SLOT = "activation.silu_and_mul"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _runtime() -> ArenaRuntimeIdentity:
    return ArenaRuntimeIdentity(
        arena_id="production-b300-tp4",
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base-engine"),
        validator_overlay_digest=_h("validator-overlay"),
        worker_distribution_digest=_h("worker-distribution"),
        model_revision_digest=_h("model-revision"),
        model_manifest_digest=_h("model-manifest"),
        model_content_digest=_h("model-content"),
        target_architecture="sm103",
        topology_class="nvlink-domain",
        topology_digest=_h("topology"),
        gpu_count=4,
        tensor_parallel_size=4,
    )


def _runtime_policy() -> OCIRuntimeResourcePolicy:
    return OCIRuntimeResourcePolicy(
        uid=max(1, os.getuid()),
        gid=max(1, os.getgid()),
        cpu_millis=8_000,
        memory_bytes=32 << 30,
        pids_limit=4_096,
        nofile_limit=65_536,
        cache_bytes=4 << 30,
        cache_inodes=100_000,
        tmpfs_bytes=1 << 30,
        shm_bytes=8 << 30,
        init_timeout_seconds=120.0,
        batch_timeout_seconds=60.0,
        container_python="/usr/local/bin/python3",
    )


def _prebuild_policy(runtime: OCIRuntimeResourcePolicy) -> OCIPrebuildPolicy:
    return OCIPrebuildPolicy(
        uid=runtime.uid,
        gid=runtime.gid,
        cpu_millis=8_000,
        memory_bytes=32 << 30,
        pids_limit=4_096,
        tmpfs_bytes=1 << 30,
        stage_bytes=16 << 30,
        stage_inodes=100_000,
        timeout_seconds=7_200.0,
        native_compile_timeout_seconds=6_000,
        container_python=runtime.container_python,
        build_path=("/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin"),
        build_tmpdir="/tmp",
        pinned_build_roots=("/usr/include", "/usr/lib", "/usr/local/cuda"),
        runtime_policy_digest=runtime.digest,
    )


def _gpu(index: int) -> GPUConfiguration:
    return GPUConfiguration(
        physical_id=index,
        uuid=f"GPU-00000000-{index:04x}-0000-0000-{index:012x}",
        pci_bus_id=f"00000000:{index + 1:02x}:00.0",
        name="NVIDIA B300 SXM6 AC",
        memory_total_mib=288_000,
        driver_version="600.10.01",
        power_limit_mw=1_000_000,
        compute_mode="Default",
        persistence_mode="Enabled",
        application_graphics_clock_mhz=None,
        application_memory_clock_mhz=None,
        max_graphics_clock_mhz=2_500,
        max_memory_clock_mhz=5_000,
    )


@pytest.fixture
def executor_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    executors: list[OCIEngineExecutor] = []
    sequence = 0

    def create(role: str, lane: str = "A") -> OCIEngineExecutor:
        nonlocal sequence
        sequence += 1
        normalized_role = (
            "candidate" if role == "candidate" else "resident_baseline"
        )
        if lane not in {"A", "B"}:
            raise AssertionError("fixture lane must be A or B")
        first_gpu = 0 if lane == "A" else 4
        runtime = _runtime_policy()
        root = tmp_path / f"executor-{sequence}-{role}"
        executor = OCIEngineExecutor(
            OCIBackendConfig(
                OCIPrebuildConfig(
                    docker_binary="/usr/bin/docker",
                    recovery_root=root / "recovery",
                    publication_root=root / "publications",
                    seccomp_profile=root / "seccomp.json",
                    executor_id=f"qualification-{normalized_role}",
                    policy=_prebuild_policy(runtime),
                ),
                runtime,
            ),
            DeviceStatePolicy(
                expected_gpus=tuple(
                    _gpu(index) for index in range(first_gpu, first_gpu + 4)
                ),
                required_consecutive_idle_samples=2,
                poll_interval_s=0.05,
                ready_poll_interval_s=0.05,
                drain_timeout_s=2.0,
                maximum_samples=8,
            ),
        )
        executors.append(executor)
        return executor

    create.managed_executors = executors
    create.monkeypatch = monkeypatch
    yield create
    for executor in executors:
        executor.manager.close()


class _Judge:
    def __init__(self) -> None:
        self.binding = HiddenJudgeBinding(
            _h("hidden-corpus"), _h("hidden-judge"), _h("hidden-policy")
        )

    def __call__(self, **_kwargs):
        raise AssertionError("the CPU contract must not execute the hidden judge")


class _FactoryBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object | None]] = []

    def __call__(self, request, state):
        self.calls.append((request, state))
        reservations = tuple(row.reservation for row in request.candidates)
        manifest = QualificationAuthorityManifest(
            "registered",
            _h("qualification-authority"),
            _h("qualification-source"),
            _h("selection-commitment"),
            _h("selection-secret-reference"),
            tuple(row.selected_delta_digest for row in reservations),
            reservations,
        )
        return QualificationPlanFactory(
            manifest,
            lambda _reference: b"s" * 32,
            lambda _secret: None,
        )


class _ResidentFactory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.created = 0
        self.closed = 0

    def __call__(self) -> B300ResidentScreenLifetime:
        self.created += 1

        def unused_lifetime(_driver):
            raise AssertionError("the non-swappable fixture must not start an engine")

        lane = ResidentScreenLane(
            unused_lifetime,
            prompts=("screen prompt",),
            policy=ScreenPolicy(),
            verdict_timeout_s=5.0,
            close_timeout_s=5.0,
        )
        intake = self.root / f"swap-{self.created}"
        intake.mkdir(parents=True)
        stage = ResidentServingScreenStage(lane, intake)

        def close() -> None:
            self.closed += 1
            lane.close()

        return B300ResidentScreenLifetime(stage, close)


def _authorities(tmp_path: Path, executor_factory):
    def runner(manifest, policy, candidate):
        return ScreenStageResult(
            policy.stage,
            ScreenGrade.PASS,
            _h(f"{manifest.digest}:{policy.stage}:{candidate.digest}"),
            1,
        )

    resident = _ResidentFactory(tmp_path)
    builder = _FactoryBuilder()
    handlers = tuple(
        B300ScreenStageHandler(
            stage,
            _h(f"{stage}-handler"),
            () if stage == "static" else (f"{stage}-resource",),
            runner,
        )
        for stage in SCREEN_STAGES[:-1]
    )
    candidate_executor = executor_factory("candidate", "A")
    baseline_executor = executor_factory("resident_baseline", "B")
    lane_pair = B300QualificationLanePair(
        B300QualificationLanePolicy.from_device_policy(
            "A", candidate_executor.device_policy
        ),
        B300QualificationLanePolicy.from_device_policy(
            "B", baseline_executor.device_policy
        ),
    )
    authority_index = len(executor_factory.managed_executors)
    resident_pair_factory, pair_executors = (
        authority_fixtures._resident_pair_factory(
            tmp_path / f"resident-pair-{authority_index}",
            executor_factory.monkeypatch,
            _h(f"resident-pair-placeholder-{authority_index}"),
        )
    )
    executor_factory.managed_executors.extend(pair_executors)
    resident_count_quality = authority_fixtures._resident_count_quality(
        authority_fixtures.default_target_catalog(),
        tmp_path / f"count-evidence-{authority_index}",
    )
    authorities = B300DeploymentAuthorities(
        runtime_identity=_runtime(),
        screen_handlers=handlers,
        resident_screen_factory=B300ResidentScreenFactory(
            _h("resident-screen-factory"),
            ("resident-screen-resource",),
            resident,
        ),
        qualification_policy_digest=_h("qualification-policy"),
        qualification_builder_digest=_h("qualification-builder"),
        qualification_factory_builder=builder,
        executor=candidate_executor,
        resident_baseline_executor=baseline_executor,
        entropy_provider_digest=_h("entropy-provider"),
        entropy_provider=lambda *_args: None,
        hidden_judge=_Judge(),
        deadline_policy_digest=_h("deadline-policy"),
        deadline_provider=lambda _request, _state: time.monotonic() + 600.0,
        qualification_lane_pair=lane_pair,
        qualification_stage="primary",
        resident_pair_factory=resident_pair_factory,
        resident_count_quality=resident_count_quality,
    )
    manifest = _manifest(authorities)
    authorities = dataclasses.replace(
        authorities,
        resident_pair_factory=authority_fixtures._rebind_resident_pair_factory(
            resident_pair_factory,
            manifest.digest,
        ),
    )
    return authorities, resident, builder


def _manifest(authorities: B300DeploymentAuthorities) -> ArenaServiceManifest:
    return ArenaServiceManifest(
        runtime=authorities.runtime_identity,
        workload=Workload(
            _h("prompt-corpus"),
            "sealed-prompt-seeds-v1",
            (WorkloadCell("s8", 8192, 1024, 64, 8),),
        ),
        capacity=ArenaCapacityPolicy(32, 100, 2, 8, 4, 2, 3, 3),
        screens=NonCrownScreenPolicy(
            tuple(ScreenStagePolicy(stage, 30_000) for stage in SCREEN_STAGES)
        ),
        qualification_policy_digest=authorities.qualification_policy_digest,
        provider_digest=b300_arena_provider_digest(authorities),
    )


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


def _bound_row(
    tmp_path: Path,
    manifest: ArenaServiceManifest,
    index: int,
    *,
    promoted: bool,
):
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
        status="promoted" if promoted else "published",
        target_id=SLOT,
        target_members=(SLOT,),
        delta_fingerprint=fingerprint,
        transport_attempts=1,
        publication_digest=publication.digest,
        publication_root=publication.root,
        qualification_authority_digest="",
        qualification_evidence_digest="",
        arena_service_digest=manifest.digest if promoted else "",
        screen_lane="primary" if promoted else "",
        screen_status="promote" if promoted else "",
        screen_stage_count=len(SCREEN_STAGES) if promoted else 0,
        screen_attempts=1 if promoted else 0,
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


def _screen_claim(
    tmp_path: Path,
    manifest: ArenaServiceManifest,
) -> ClaimedScreenEvaluation:
    reservation, publication, candidate = _bound_row(
        tmp_path, manifest, 0, promoted=False
    )
    lease = EvaluationLease(
        _h("screen-lease"),
        1,
        "screen",
        "b300-worker-test",
        (EvaluationLeaseMember(reservation.reservation_id, "published"),),
        20,
        40,
        40,
    )
    return ClaimedScreenEvaluation(lease, reservation, publication, candidate)


def _promoted_receipt(
    manifest: ArenaServiceManifest,
    candidate: ArenaCandidateBinding,
) -> object:
    results = tuple(
        ScreenStageResult(stage, ScreenGrade.PASS, _h(f"{candidate.digest}:{stage}"), 1)
        for stage in SCREEN_STAGES
    )
    from cacheon.arena_service import ArenaScreenReceipt

    return ArenaScreenReceipt(
        manifest.digest,
        candidate.digest,
        candidate.screen_attempt,
        results,
        PromotionDecision.PROMOTE,
    )


def _qualification_claim(
    tmp_path: Path,
    manifest: ArenaServiceManifest,
    *,
    count: int = 2,
) -> ClaimedQualificationEvaluation:
    rows = tuple(
        _bound_row(tmp_path, manifest, index, promoted=True)
        for index in range(count)
    )
    reservations = tuple(row[0] for row in rows)
    publications = tuple(row[1] for row in rows)
    candidates = tuple(row[2] for row in rows)
    receipts = tuple(_promoted_receipt(manifest, row) for row in candidates)
    lease = EvaluationLease(
        _h(f"qualification-lease-{count}"),
        1,
        "qualification",
        "b300-worker-test",
        tuple(
            EvaluationLeaseMember(row.reservation_id, "promoted")
            for row in reservations
        ),
        20,
        40,
        40,
    )
    return ClaimedQualificationEvaluation(
        lease,
        reservations,
        publications,
        candidates,
        receipts,
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


def test_remote_screen_runs_all_real_provider_stages_and_seals_result(
    tmp_path: Path,
    executor_factory,
) -> None:
    authorities, resident, _builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _screen_claim(tmp_path / "candidate", manifest)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        # The path-free remote DTO needs no CPU intake reservation.
        result = worker.run_remote_screen(claim.lease, claim.candidate)

        assert type(result) is EvaluationRun
        assert result.lease is claim.lease
        assert result.disposition == "completed"
        assert type(result.envelope) is EvaluationResultEnvelope
        assert tuple(row.stage for row in result.payload.results) == SCREEN_STAGES
        assert result.payload.decision is PromotionDecision.PROMOTE
        assert result.payload.candidate_digest == claim.candidate.digest
        assert result.envelope.payload_digest == result.payload.digest
        assert resident.created == 1
        result.envelope.verify(claim.lease, readiness, worker.service, result.payload)
    finally:
        worker.close()
    assert resident.closed == 1


def _fake_gate_fail(monkeypatch, batch_for):
    """Route the remote path through a graph-gate FAIL carrying ``batch_for(factory)``.

    The worker binds gate result types by module global, so a test-local FAIL
    shape reaches the same ``_validate_batch`` / disposition lines production
    takes after a real graph-only FAIL, without a real graph exit artifact.
    """

    @dataclasses.dataclass(frozen=True)
    class FakeGateFail:
        plan: object
        factory: object
        batch: QualificationIntakeBatch
        supporting_evidence_refs: tuple = ()

    sentinel_plan = object()
    gate_calls = []

    def fake_gate(factory, plan, *, evidence_root, candidates, authenticated_request_digest):
        assert plan is sentinel_plan
        gate_calls.append(factory)
        return FakeGateFail(plan, factory, batch_for(factory))

    monkeypatch.setattr(QualificationPlanFactory, "build", lambda self: sentinel_plan)
    monkeypatch.setattr(worker_module, "B300QualificationGraphGateFail", FakeGateFail)
    monkeypatch.setattr(worker_module, "run_b300_qualification_graph_gate", fake_gate)
    return gate_calls


def test_remote_qualification_releases_systemic_gate_fail_batch(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, resident, builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)
    gate_calls = _fake_gate_fail(monkeypatch, _systemic_batch)
    intake_calls = []
    monkeypatch.setattr(
        worker_module,
        "run_qualification_intake",
        lambda *args, **kwargs: intake_calls.append((args, kwargs)),
    )
    worker = B300MainnetWorker(manifest, authorities, readiness)
    worker._bind_remote_qualification_graph_gate_root(tmp_path / "graph-root")
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
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
        assert result.authority_manifest is gate_calls[0].manifest
        assert len(gate_calls) == 1 and intake_calls == []
        assert builder.calls[0][1] is None
        assert resident.created == 0
        result.run.envelope.verify(
            claim.lease, readiness, worker.service, result.run.payload
        )
    finally:
        worker.close()


def test_remote_qualification_without_graph_root_returns_authenticated_hold(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, resident, builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)
    calls = []

    def run(factory, **kwargs):
        calls.append((factory, kwargs))
        return _systemic_batch(factory)

    monkeypatch.setattr(worker_module, "run_qualification_intake", run)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    continuation = QualificationContinuationStore(tmp_path / "continuation")
    request_digest = _h("remote-request")
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
            screen_lane="primary",
            continuation_store=continuation,
            request_digest=request_digest,
        )

        assert type(result) is B300QualificationGraphGateHold
        assert result.reason is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
        assert builder.calls[0][1] is None
        assert calls == []
        assert resident.created == 0
    finally:
        worker.close()


def test_remote_qualification_refuses_lane_or_cohort_drift(
    tmp_path: Path,
    executor_factory,
) -> None:
    authorities, _resident, _builder = _authorities(tmp_path, executor_factory)
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
                claim.screen_receipts,
                screen_lane="primary",
                continuation_store=continuation,
                request_digest=None,
            )
        with pytest.raises(B300MainnetWorkerError, match="exact promoted cohort"):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                claim.screen_receipts,
                screen_lane="reproduction",
                continuation_store=continuation,
                request_digest=request_digest,
            )
        with pytest.raises(B300MainnetWorkerError, match="exact promoted cohort"):
            worker.run_remote_qualification(
                claim.lease,
                tuple(reversed(claim.candidates)),
                claim.screen_receipts,
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
    assert hold.failure_type == "ValueError"
    assert "stock restoration failed" in hold.failure_message
    assert "CUDA out of memory" in hold.failure_message


def test_remote_qualification_stage_is_derived_from_swapped_executor_authority(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    primary, _resident, _builder = _authorities(tmp_path, executor_factory)
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

    intake_calls = []
    monkeypatch.setattr(
        worker_module,
        "run_qualification_intake",
        lambda *args, **kwargs: intake_calls.append((args, kwargs)),
    )
    worker = B300MainnetWorker(manifest, reproduction, readiness)
    try:
        assert worker._remote_qualification_lane == "reproduction"
        with pytest.raises(B300MainnetWorkerError, match="exact promoted cohort"):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                claim.screen_receipts,
                screen_lane="primary",
                continuation_store=continuation,
                request_digest=request_digest,
            )
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
            screen_lane="reproduction",
            continuation_store=continuation,
            request_digest=request_digest,
        )
        assert type(result) is B300QualificationGraphGateHold
        assert result.reason is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
        assert intake_calls == []
    finally:
        worker.close()


def test_remote_qualification_retires_released_pair_and_retries_plan_build(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, _resident, _builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)
    continuation = QualificationContinuationStore(tmp_path / "continuation")
    request_digest = _h("remote-request")

    build_calls: list[object] = []
    sentinel_plan = object()

    def flaky_build(self):
        build_calls.append(self)
        if len(build_calls) == 1:
            raise worker_module.B300QualificationGraphEvidenceHold(
                "capture devices busy"
            )
        return sentinel_plan

    gate_hold = worker_module.qualification_graph_gate_hold(
        RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE,
        authenticated_request_digest=request_digest,
        authority_context_digest=manifest.digest,
        code=worker_module.B300QualificationGraphHoldCode.RAW_EVIDENCE_INCOMPLETE,
    )
    gate_calls = []

    def fake_gate(factory, plan, *, evidence_root, candidates, authenticated_request_digest):
        assert plan is sentinel_plan
        gate_calls.append(factory)
        return gate_hold

    monkeypatch.setattr(QualificationPlanFactory, "build", flaky_build)
    monkeypatch.setattr(
        worker_module, "run_b300_qualification_graph_gate", fake_gate
    )
    worker = B300MainnetWorker(manifest, authorities, readiness)
    retires = []
    monkeypatch.setattr(
        worker._resident_pair_factory,
        "retire_released_pair",
        lambda: retires.append(True) or True,
        raising=False,
    )
    worker._bind_remote_qualification_graph_gate_root(tmp_path / "graph-root")
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
            screen_lane="primary",
            continuation_store=continuation,
            request_digest=request_digest,
        )

        assert result is gate_hold
        assert len(build_calls) == 2
        assert retires == [True]
        assert len(gate_calls) == 1
    finally:
        worker.close()


def test_remote_qualification_holds_when_no_released_pair_can_be_retired(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, _resident, _builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _qualification_claim(tmp_path / "cohort", manifest)
    continuation = QualificationContinuationStore(tmp_path / "continuation")
    request_digest = _h("remote-request")

    build_calls: list[object] = []

    def busy_build(self):
        build_calls.append(self)
        raise worker_module.B300QualificationGraphEvidenceHold(
            "capture devices busy"
        )

    monkeypatch.setattr(QualificationPlanFactory, "build", busy_build)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    worker._bind_remote_qualification_graph_gate_root(tmp_path / "graph-root")
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
            screen_lane="primary",
            continuation_store=continuation,
            request_digest=request_digest,
        )

        assert type(result) is B300QualificationGraphGateHold
        assert (
            result.reason
            is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
        )
        assert len(build_calls) == 1
    finally:
        worker.close()


def test_qualification_refuses_result_that_reorders_leased_cohort(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, _resident, _builder = _authorities(tmp_path, executor_factory)
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

    _fake_gate_fail(monkeypatch, reordered)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    worker._bind_remote_qualification_graph_gate_root(tmp_path / "graph-root")
    try:
        with pytest.raises(B300MainnetWorkerError, match="exact leased cohort"):
            worker.run_remote_qualification(
                claim.lease,
                claim.candidates,
                claim.screen_receipts,
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


def test_readiness_drift_is_rejected_before_work(
    tmp_path: Path,
    executor_factory,
) -> None:
    authorities, resident, _builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = dataclasses.replace(
        _readiness(manifest, authorities),
        service_digest=_h("different-service"),
    )

    with pytest.raises(B300MainnetWorkerError, match="readiness differs"):
        B300MainnetWorker(manifest, authorities, readiness)
    assert resident.created == 0

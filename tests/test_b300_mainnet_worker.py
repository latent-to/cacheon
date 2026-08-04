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
    ServingShape,
    WorkloadMixture,
    WorkloadRegime,
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
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.b300_arena_provider import (
    B300ArenaServiceProvider,
    B300DeploymentAuthorities,
    B300ResidentScreenFactory,
    B300ResidentScreenLifetime,
    B300ScreenStageHandler,
    b300_arena_provider_digest,
)
from cacheon.eval.b300_mainnet_worker import (
    B300MainnetWorker,
    B300MainnetWorkerError,
)
from cacheon.eval.device_state import DeviceStatePolicy, GPUConfiguration
from cacheon.eval.oci_backend import (
    OCIBackendConfig,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
)
from cacheon.eval.oci_prebuild import OCIPrebuildConfig, OCIPrebuildPolicy
from cacheon.eval.qualification import QualificationDecision
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
        arena_id="production-b300-tp8",
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base-engine"),
        validator_overlay_digest=_h("validator-overlay"),
        worker_distribution_digest=_h("worker-distribution"),
        model_revision_digest=_h("model-revision"),
        model_manifest_digest=_h("model-manifest"),
        model_content_digest=_h("model-content"),
        target_architecture="sm120",
        topology_class="nvlink-domain",
        topology_digest=_h("topology"),
        gpu_count=8,
        tensor_parallel_size=8,
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


def _gpu(index: int, role: str) -> GPUConfiguration:
    role_value = 1 if role == "candidate" else 2
    return GPUConfiguration(
        physical_id=index,
        uuid=(
            f"GPU-{role_value:08x}-{index:04x}-0000-0000-"
            f"{(role_value * 100 + index):012x}"
        ),
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
def executor_factory(tmp_path: Path):
    executors: list[OCIEngineExecutor] = []
    sequence = 0

    def create(role: str) -> OCIEngineExecutor:
        nonlocal sequence
        sequence += 1
        runtime = _runtime_policy()
        root = tmp_path / f"executor-{sequence}-{role}"
        executor = OCIEngineExecutor(
            OCIBackendConfig(
                OCIPrebuildConfig(
                    docker_binary="/usr/bin/docker",
                    recovery_root=root / "recovery",
                    publication_root=root / "publications",
                    seccomp_profile=root / "seccomp.json",
                    executor_id=f"{role}-{sequence}",
                    policy=_prebuild_policy(runtime),
                ),
                runtime,
            ),
            DeviceStatePolicy(
                expected_gpus=tuple(_gpu(index, role) for index in range(8)),
                required_consecutive_idle_samples=2,
                poll_interval_s=0.05,
                ready_poll_interval_s=0.05,
                drain_timeout_s=2.0,
                maximum_samples=8,
            ),
        )
        executors.append(executor)
        return executor

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
        executor=executor_factory("candidate"),
        resident_baseline_executor=executor_factory("baseline"),
        entropy_provider_digest=_h("entropy-provider"),
        entropy_provider=lambda *_args: None,
        hidden_judge=_Judge(),
        deadline_policy_digest=_h("deadline-policy"),
        deadline_provider=lambda _request, _state: time.monotonic() + 600.0,
    )
    return authorities, resident, builder


def _manifest(authorities: B300DeploymentAuthorities) -> ArenaServiceManifest:
    return ArenaServiceManifest(
        runtime=authorities.runtime_identity,
        workload=WorkloadMixture(
            _h("prompt-corpus"),
            "sealed-prompt-seeds-v1",
            (
                WorkloadRegime(
                    "decode",
                    "decode",
                    500_000,
                    (ServingShape(256, 128, 8, 8),),
                ),
                WorkloadRegime(
                    "prefill",
                    "long_prefill",
                    500_000,
                    (ServingShape(8192, 16, 1, 8),),
                ),
            ),
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


def test_screen_job_runs_all_real_provider_stages_and_seals_result(
    tmp_path: Path,
    executor_factory,
) -> None:
    authorities, resident, _builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    claim = _screen_claim(tmp_path / "candidate", manifest)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        result = worker.run(claim)

        assert type(result) is EvaluationRun
        assert result.lease is claim.lease
        assert result.disposition == "completed"
        assert type(result.envelope) is EvaluationResultEnvelope
        assert tuple(row.stage for row in result.payload.results) == SCREEN_STAGES
        assert result.payload.decision is PromotionDecision.PROMOTE
        assert resident.created == 1
        result.envelope.verify(claim.lease, readiness, worker.service, result.payload)
    finally:
        worker.close()
    assert resident.closed == 1


def test_qualification_uses_only_sealed_work_and_releases_systemic_batch(
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
        assert type(factory) is QualificationPlanFactory
        assert kwargs["executor"] is authorities.executor
        assert kwargs["resident_baseline_executor"] is authorities.resident_baseline_executor
        assert kwargs["entropy_provider"] is authorities.entropy_provider
        assert kwargs["hidden_judge"] is authorities.hidden_judge
        assert kwargs["deadline"] > time.monotonic()
        return _systemic_batch(factory)

    monkeypatch.setattr(worker_module, "run_qualification_intake", run)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        result = worker.run(claim)

        assert type(result.payload) is QualificationIntakeBatch
        assert result.disposition == "released"
        assert result.envelope.payload_kind == "qualification_intake_batch"
        assert tuple(row.reservation_digest for row in result.payload.outcomes) == (
            claim.lease.reservation_ids
        )
        assert len(calls) == 1
        assert builder.calls[0][1] is None
        assert resident.created == 0
        result.envelope.verify(claim.lease, readiness, worker.service, result.payload)
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

    def reordered(factory, **_kwargs):
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

    monkeypatch.setattr(worker_module, "run_qualification_intake", reordered)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        with pytest.raises(B300MainnetWorkerError, match="exact leased cohort"):
            worker.run(claim)
    finally:
        worker.close()


def test_worker_refuses_untyped_jobs_and_has_no_dynamic_dispatch_surface(
    tmp_path: Path,
    executor_factory,
) -> None:
    authorities, _resident, _builder = _authorities(tmp_path, executor_factory)
    manifest = _manifest(authorities)
    readiness = _readiness(manifest, authorities)
    worker = B300MainnetWorker(manifest, authorities, readiness)
    try:
        with pytest.raises(B300MainnetWorkerError, match="not exactly typed"):
            worker.run(object())
    finally:
        worker.close()

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

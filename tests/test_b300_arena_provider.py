"""CPU-only contracts for the closed B300 arena provider."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from pathlib import Path

import pytest

import tests.test_b300_sealed_qualification_commission as authority_fixtures
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaQualificationWork,
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
from cacheon.chain.publication import publish_worker_bundle
from cacheon.eval.b300_arena_provider import (
    B300ArenaProviderError,
    B300ArenaServiceProvider,
    B300DeclaredQualificationAuthorities,
    B300DeploymentAuthorities,
    B300QualificationLanePair,
    B300QualificationLanePolicy,
    B300ResidentScreenFactory,
    B300ResidentScreenLifetime,
    B300ScreenDeploymentAuthorities,
    B300ScreenStageHandler,
    b300_arena_provider_digest,
)
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
)
from cacheon.eval.device_state import DeviceStatePolicy, GPUConfiguration
from cacheon.eval.oci_backend import (
    OCIBackendConfig,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
)
from cacheon.eval.oci_prebuild import OCIPrebuildConfig, OCIPrebuildPolicy
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
    QualificationReservation,
)
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.resident_queue import ScreenPolicy
from cacheon.eval.resident_screen_lane import (
    ResidentScreenLane,
    ResidentScreenLifetimeFailed,
    ResidentServingScreenStage,
    screen_waiver_result,
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
        uuid=(
            f"GPU-00000000-{index:04x}-0000-0000-{index:012x}"
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
        policy = _prebuild_policy(runtime)
        root = tmp_path / f"executor-{sequence}-{role}"
        config = OCIBackendConfig(
            OCIPrebuildConfig(
                docker_binary="/usr/bin/docker",
                recovery_root=root / "recovery",
                publication_root=root / "publications",
                seccomp_profile=root / "seccomp.json",
                executor_id=f"qualification-{normalized_role}",
                policy=policy,
            ),
            runtime,
        )
        executor = OCIEngineExecutor(
            config,
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
        raise AssertionError("the provider must not execute the hidden judge")


class _FactoryBuilder:
    def __init__(self, *, reverse: bool = False, fail: bool = False) -> None:
        self.reverse = reverse
        self.fail = fail
        self.calls = []

    def __call__(self, request, state):
        self.calls.append((request, state))
        if self.fail:
            raise OSError("private authority store unavailable")
        reservations = tuple(row.reservation for row in request.candidates)
        if self.reverse:
            reservations = tuple(reversed(reservations))
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
            raise AssertionError("an unswappable fixture must not start an engine")

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


class _ScreenRunner:
    def __init__(self, grades=None, *, wrong_stage: str | None = None) -> None:
        self.grades = dict(grades or {})
        self.wrong_stage = wrong_stage
        self.calls = []

    def for_stage(self, expected_stage: str):
        def run(manifest, policy, candidate):
            self.calls.append(
                (manifest.digest, policy.stage, candidate.digest)
            )
            value = self.grades.get(expected_stage, ScreenGrade.PASS)
            if isinstance(value, BaseException):
                raise value
            result_stage = self.wrong_stage or expected_stage
            return ScreenStageResult(
                result_stage,
                value,
                _h(f"{expected_stage}-{value.value}-evidence"),
                1,
            )

        return run


def _authorities(
    tmp_path: Path,
    executor_factory,
    *,
    grades=None,
    wrong_stage: str | None = None,
    builder: _FactoryBuilder | None = None,
):
    runner = _ScreenRunner(grades, wrong_stage=wrong_stage)
    handlers = tuple(
        B300ScreenStageHandler(
            stage,
            _h(f"{stage}-handler"),
            () if stage == "static" else ("build-resource",),
            runner.for_stage(stage),
        )
        for stage in SCREEN_STAGES[:-1]
    )
    resident = _ResidentFactory(tmp_path)
    factory_builder = builder or _FactoryBuilder()
    policy_digest = _h("qualification-policy")
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
            ("screen-resource",),
            resident,
        ),
        qualification_policy_digest=policy_digest,
        qualification_builder_digest=_h("qualification-builder"),
        qualification_factory_builder=factory_builder,
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
    return authorities, runner, resident, factory_builder


def _manifest(
    authorities: B300DeploymentAuthorities,
    **changes,
) -> ArenaServiceManifest:
    workload = WorkloadMixture(
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
    )
    values = {
        "runtime": authorities.runtime_identity,
        "workload": workload,
        "capacity": ArenaCapacityPolicy(32, 100, 2, 8, 4, 2, 3, 3),
        "screens": NonCrownScreenPolicy(
            tuple(ScreenStagePolicy(stage, 30_000) for stage in SCREEN_STAGES)
        ),
        "qualification_policy_digest": authorities.qualification_policy_digest,
        "provider_digest": b300_arena_provider_digest(authorities),
    }
    values.update(changes)
    return ArenaServiceManifest(**values)


def _bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
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
                "bundle_id = 'b300-provider-fixture'",
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


def _binding(tmp_path: Path, index: int = 0) -> ArenaCandidateBinding:
    source = _bundle(tmp_path)
    publication = publish_worker_bundle(
        source,
        tmp_path / "publications",
        content_hash(source),
    )
    reservation = QualificationReservation(
        _h(f"reservation-{index}"),
        publication.digest,
        SLOT,
        _h(f"delta-{index}"),
        index,
        f"miner-{index}",
        20,
        index,
        0,
        (SLOT,),
    )
    return ArenaCandidateBinding(reservation, publication, 1)


def test_all_five_real_screens_run_in_order_and_preserve_pass(
    tmp_path: Path, executor_factory
) -> None:
    authorities, runner, resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))

    receipt = service.screen(_binding(tmp_path / "candidate"))

    assert receipt.decision is PromotionDecision.PROMOTE
    assert tuple(row.stage for row in receipt.results) == SCREEN_STAGES
    assert tuple(row[1] for row in runner.calls) == SCREEN_STAGES[:-1]
    assert resident.created == 1
    assert service._provider.resident_screen_active
    service._provider.close()
    assert resident.closed == 1


def test_fail_is_not_rewritten(tmp_path: Path, executor_factory) -> None:
    authorities, _runner, resident, _builder = _authorities(
        tmp_path,
        executor_factory,
        grades={"abi": ScreenGrade.FAIL},
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))

    receipt = service.screen(_binding(tmp_path / "fail"))

    assert receipt.results[-1].stage == "abi"
    assert receipt.results[-1].grade is ScreenGrade.FAIL
    assert receipt.decision is PromotionDecision.REJECT
    assert resident.created == 0


def test_no_decision_screen_evidence_is_waived_into_qualification(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _runner, resident, _builder = _authorities(
        tmp_path,
        executor_factory,
        grades={"abi": ScreenGrade.NO_DECISION},
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))

    receipt = service.screen(_binding(tmp_path / "no-decision"))

    assert receipt.decision is PromotionDecision.PROMOTE
    assert all(row.grade is ScreenGrade.PASS for row in receipt.results)
    assert resident.created == 1
    service._provider.close()


def test_stage_substitution_is_a_provider_contract_error(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _runner, _resident, _builder = _authorities(
        tmp_path,
        executor_factory,
        wrong_stage="build",
    )
    manifest = _manifest(authorities)
    provider = B300ArenaServiceProvider(manifest, authorities)
    candidate = _binding(tmp_path / "candidate")

    with pytest.raises(B300ArenaProviderError, match="changed the requested stage"):
        provider.run_screen(manifest, manifest.screens.stages[0], candidate)
    with pytest.raises(B300ArenaProviderError, match="policy was substituted"):
        provider.run_screen(
            manifest,
            ScreenStagePolicy("static", 29_999),
            candidate,
        )


def test_handler_exception_is_waived_but_untyped_output_is_not(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _runner, _resident, _builder = _authorities(
        tmp_path,
        executor_factory,
        grades={"build": RuntimeError("worker unavailable")},
    )
    manifest = _manifest(authorities)
    provider = B300ArenaServiceProvider(manifest, authorities)
    candidate = _binding(tmp_path / "candidate")

    result = provider.run_screen(manifest, manifest.screens.stages[1], candidate)

    assert type(result) is ScreenStageResult
    assert result.grade is ScreenGrade.PASS
    assert result.stage == "build"

    bad_handler = dataclasses.replace(
        authorities.screen_handlers[1],
        identity_digest=_h("bad-handler"),
        runner=lambda *_args: object(),
    )
    bad_authorities = dataclasses.replace(
        authorities,
        screen_handlers=(
            authorities.screen_handlers[0],
            bad_handler,
            *authorities.screen_handlers[2:],
        ),
    )
    bad_manifest = _manifest(bad_authorities)
    bad_provider = B300ArenaServiceProvider(bad_manifest, bad_authorities)
    with pytest.raises(B300ArenaProviderError, match="evidence type"):
        bad_provider.run_screen(
            bad_manifest,
            bad_manifest.screens.stages[1],
            candidate,
        )


def test_serving_host_error_does_not_release_or_reload_resident(
    tmp_path: Path, executor_factory, monkeypatch
) -> None:
    authorities, _runner, resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    manifest = _manifest(authorities)
    provider = B300ArenaServiceProvider(manifest, authorities)
    candidate = _binding(tmp_path / "candidate")
    calls = 0

    def run(_stage, _candidate):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("candidate carrier unavailable")
        return ScreenStageResult(
            "abbreviated_serving", ScreenGrade.PASS, _h("serving-pass"), 1
        )

    monkeypatch.setattr(ResidentServingScreenStage, "run_screen", run)
    # A host error is not a candidate verdict: the request fails and may be
    # retried while the healthy resident stays loaded.
    with pytest.raises(B300ArenaProviderError, match="resident screen request failed"):
        provider.run_screen(manifest, manifest.screens.stages[-1], candidate)
    second = provider.run_screen(manifest, manifest.screens.stages[-1], candidate)
    assert second.grade is ScreenGrade.PASS
    assert resident.created == 1
    assert resident.closed == 0
    provider.close()
    assert resident.closed == 1


def test_engine_death_latches_epoch_without_silent_reboot(
    tmp_path: Path, executor_factory, monkeypatch
) -> None:
    authorities, _runner, resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    manifest = _manifest(authorities)
    provider = B300ArenaServiceProvider(manifest, authorities)
    candidate = _binding(tmp_path / "candidate")
    monkeypatch.setattr(
        ResidentServingScreenStage,
        "run_screen",
        lambda _stage, _candidate: (_ for _ in ()).throw(
            ResidentScreenLifetimeFailed("engine lost")
        ),
    )
    for _ in range(2):
        with pytest.raises(B300ArenaProviderError, match="epoch restart required"):
            provider.run_screen(
                manifest, manifest.screens.stages[-1], candidate
            )
    assert resident.created == 1
    assert resident.closed == 0
    provider.close()
    assert resident.closed == 1


def test_canary_bypass_survives_resident_retirement(
    tmp_path: Path, executor_factory, monkeypatch
) -> None:
    authorities, _runner, resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    manifest = _manifest(authorities)
    provider = B300ArenaServiceProvider(manifest, authorities)
    candidate = _binding(tmp_path / "candidate")
    calls = 0

    def bypass(stage, item):
        nonlocal calls
        calls += 1
        stage._bypass_reason = "stock canary unavailable"
        return screen_waiver_result(item, stage._bypass_reason, 1)

    monkeypatch.setattr(ResidentServingScreenStage, "run_screen", bypass)
    assert provider.run_screen(
        manifest, manifest.screens.stages[-1], candidate
    ).grade is ScreenGrade.PASS
    assert (calls, resident.created, resident.closed) == (1, 1, 1)
    assert provider.run_screen(
        manifest, manifest.screens.stages[-1], candidate
    ).grade is ScreenGrade.PASS
    assert (calls, resident.created, resident.closed) == (1, 1, 1)
    provider.close()


def test_qualification_preserves_exact_request_order_and_real_authorities(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _runner, resident, builder = _authorities(
        tmp_path, executor_factory
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    first = _binding(tmp_path / "first", 0)
    second = _binding(tmp_path / "second", 1)
    receipts = (service.screen(first), service.screen(second))

    work = service.plan_qualification(
        (first, second),
        receipts,
        state={"attempt": 1},
    )

    assert type(work) is ArenaQualificationWork
    assert type(work.factory) is QualificationPlanFactory
    assert work.factory.manifest.reservations == (
        first.reservation,
        second.reservation,
    )
    assert type(work.executor) is OCIEngineExecutor
    assert type(work.resident_baseline_executor) is OCIEngineExecutor
    assert work.executor is authorities.executor
    assert work.resident_baseline_executor is authorities.resident_baseline_executor
    assert work.entropy_provider is authorities.entropy_provider
    assert work.hidden_judge is authorities.hidden_judge
    assert builder.calls[0][0].candidates == (first, second)
    assert builder.calls[0][1] == {"attempt": 1}
    assert resident.created == 1
    assert resident.closed == 1
    assert not service._provider.resident_screen_active


def test_reordered_factory_is_refused(
    tmp_path: Path, executor_factory
) -> None:
    builder = _FactoryBuilder(reverse=True)
    authorities, _runner, _resident, _builder = _authorities(
        tmp_path,
        executor_factory,
        builder=builder,
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    first = _binding(tmp_path / "first", 0)
    second = _binding(tmp_path / "second", 1)
    receipts = (service.screen(first), service.screen(second))

    with pytest.raises(B300ArenaProviderError, match="cohort order"):
        service.plan_qualification((first, second), receipts)


def test_runtime_model_topology_and_policy_must_match_manifest(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _runner, _resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    runtime_mismatch = dataclasses.replace(
        authorities.runtime_identity,
        model_content_digest=_h("another-model"),
    )
    with pytest.raises(B300ArenaProviderError, match="runtime, model, topology"):
        B300ArenaServiceProvider(
            _manifest(authorities, runtime=runtime_mismatch),
            authorities,
        )
    with pytest.raises(B300ArenaProviderError, match="qualification policy"):
        B300ArenaServiceProvider(
            _manifest(
                authorities,
                qualification_policy_digest=_h("another-policy"),
            ),
            authorities,
        )


def test_provider_digest_binds_handler_runtime_and_qualification_policy(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _runner, _resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    original = b300_arena_provider_digest(authorities)
    changed_handler = dataclasses.replace(
        authorities.screen_handlers[0],
        identity_digest=_h("changed-static-handler"),
    )
    changed_handlers = dataclasses.replace(
        authorities,
        screen_handlers=(changed_handler, *authorities.screen_handlers[1:]),
    )
    changed_policy = dataclasses.replace(
        authorities,
        qualification_policy_digest=_h("changed-qualification-policy"),
    )
    changed_runtime = dataclasses.replace(
        authorities,
        runtime_identity=dataclasses.replace(
            authorities.runtime_identity,
            topology_digest=_h("changed-topology"),
        ),
    )

    assert len(
        {
            original,
            b300_arena_provider_digest(changed_handlers),
            b300_arena_provider_digest(changed_policy),
            b300_arena_provider_digest(changed_runtime),
        }
    ) == 4


def test_provider_identity_is_stable_across_exact_primary_reproduction_swap(
    tmp_path: Path, executor_factory
) -> None:
    primary, _runner, _resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    reproduction = dataclasses.replace(
        primary,
        executor=executor_factory("candidate", "B"),
        resident_baseline_executor=executor_factory("resident_baseline", "A"),
        qualification_stage="reproduction",
    )

    assert primary.qualification_orientation.stage == "primary"
    assert primary.qualification_orientation.candidate.lane_id == "A"
    assert primary.qualification_orientation.resident_baseline.lane_id == "B"
    assert reproduction.qualification_orientation.stage == "reproduction"
    assert reproduction.qualification_orientation.candidate.lane_id == "B"
    assert reproduction.qualification_orientation.resident_baseline.lane_id == "A"
    assert primary.qualification == reproduction.qualification
    assert primary.qualification_lane_pair.digest == (
        reproduction.qualification_lane_pair.digest
    )
    assert primary.qualification_lane_pair.role_swap_digest == (
        reproduction.qualification_lane_pair.role_swap_digest
    )
    assert b300_arena_provider_digest(primary) == b300_arena_provider_digest(
        reproduction
    )


def test_lane_pair_rejects_overlap_and_full_authority_rejects_orientation_drift(
    tmp_path: Path, executor_factory
) -> None:
    primary, _runner, _resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    with pytest.raises(B300ArenaProviderError, match="overlapping"):
        B300QualificationLanePair(
            primary.qualification_lane_pair.lane_a,
            dataclasses.replace(
                primary.qualification_lane_pair.lane_a,
                lane_id="B",
            ),
        )

    with pytest.raises(B300ArenaProviderError, match="selected physical TP4 lane"):
        dataclasses.replace(
            primary,
            executor=executor_factory("candidate", "B"),
        )
    with pytest.raises(B300ArenaProviderError, match="selected physical TP4 lane"):
        dataclasses.replace(
            primary,
            qualification_stage="reproduction",
        )

    primary.executor.device_policy = executor_factory(
        "candidate", "B"
    ).device_policy
    with pytest.raises(B300ArenaProviderError, match="selected physical TP4 lane"):
        b300_arena_provider_digest(primary)


def test_screen_only_and_full_authorities_share_exact_provider_identity(
    tmp_path: Path, executor_factory
) -> None:
    full, _runner, _resident, _builder = _authorities(
        tmp_path, executor_factory
    )
    declared = full.qualification
    assert type(declared) is B300DeclaredQualificationAuthorities
    screen_only = B300ScreenDeploymentAuthorities(
        full.runtime_identity,
        full.screen_handlers,
        full.resident_screen_factory,
        declared,
    )

    assert b300_arena_provider_digest(screen_only) == b300_arena_provider_digest(
        full
    )
    assert _manifest(full).provider_digest == b300_arena_provider_digest(
        screen_only
    )


def test_factory_exception_stays_a_provider_error(
    tmp_path: Path, executor_factory
) -> None:
    builder = _FactoryBuilder(fail=True)
    authorities, _runner, _resident, _builder = _authorities(
        tmp_path, executor_factory, builder=builder
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    candidate = _binding(tmp_path / "candidate")
    receipt = service.screen(candidate)

    with pytest.raises(B300ArenaProviderError, match="factory construction"):
        service.plan_qualification((candidate,), (receipt,))


def test_graph_evidence_hold_survives_arena_provider(
    tmp_path: Path, executor_factory
) -> None:
    hold = B300QualificationGraphEvidenceHold("graph attempt remains armed")

    class HoldBuilder:
        def __call__(self, _request, _state):
            raise hold

    authorities, _runner, _resident, _builder = _authorities(
        tmp_path,
        executor_factory,
        builder=HoldBuilder(),  # type: ignore[arg-type]
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    candidate = _binding(tmp_path / "candidate")
    receipt = service.screen(candidate)

    with pytest.raises(B300QualificationGraphEvidenceHold) as caught:
        service.plan_qualification((candidate,), (receipt,))
    assert caught.value is hold

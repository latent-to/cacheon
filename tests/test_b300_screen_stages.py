"""CPU-only contracts for the closed B300 static/build/ABI/graph stages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

import cacheon.eval.b300_screen_stages as screen_stages
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    ScreenGrade,
    ScreenStagePolicy,
    Workload,
    WorkloadCell,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.engine_tree import inspect_contribution
from cacheon.eval.b300_screen_stages import (
    B300BuildABIGraphScreenAdapter,
    B300ScreenExecutionPlan,
    B300StaticScreenAdapter,
    compose_b300_non_serving_screen_handlers,
)
from cacheon.eval.device_state import DeviceStatePolicy, GPUConfiguration
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    LogicalHardwareSpec,
    NativeBuildSpec,
    PhysicalHardwareBinding,
    TrustedLaunchBinding,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
)
from cacheon.eval.native_artifact import publish_native_artifact
from cacheon.eval.oci_backend import (
    CandidateFreeRuntimeIdentity,
    EngineExecutionEvidence,
    OCIBackendConfig,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_outer_session import (
    BatchExecutionEvidence,
    OuterSessionCandidateError,
    SessionExecutionEvidence,
    SessionExecutionPlan,
)
from cacheon.eval.oci_prebuild import (
    OCIPrebuildConfig,
    OCIPrebuildPolicy,
    OCIPrebuildResult,
)
from cacheon.eval.oci_session_protocol import (
    AuditReceiptFacts,
    BatchEvidence,
    EngineSessionConfig,
    SlotAuditPolicy,
)
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt
from cacheon.target_catalog import TargetCatalog, default_target_catalog
from tests.support.b300 import gpu as _gpu, prebuild_policy, runtime_policy, sha as _h


def _runtime_policy() -> OCIRuntimeResourcePolicy:
    return runtime_policy("screen")


def _prebuild_policy(runtime: OCIRuntimeResourcePolicy) -> OCIPrebuildPolicy:
    return prebuild_policy(runtime, "screen")


SLOT = "activation.silu_and_mul"


def _workload() -> Workload:
    return Workload(
        _h("prompt-corpus"),
        "sealed-prompts-v1",
        (WorkloadCell("s8", 8192, 1024, 64, 8),),
    )


def _manifest(runtime: ArenaRuntimeIdentity) -> ArenaServiceManifest:
    return ArenaServiceManifest(
        runtime,
        _workload(),
        ArenaCapacityPolicy(8, 32, 1, 4, 4, 2, 2, 2),
        NonCrownScreenPolicy(
            tuple(ScreenStagePolicy(stage, 30_000) for stage in SCREEN_STAGES)
        ),
        _h("qualification-policy"),
        _h("provider"),
    )


def _copy_candidate_source(root: Path, *, hostile: bool = False) -> Path:
    source = root / "source"
    shutil.copytree(
        Path(__file__).parents[1] / "examples" / "miner_silu_torch",
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if hostile:
        with (source / "kernels" / "silu_and_mul.py").open("a") as handle:
            handle.write("\nimport subprocess\n")
    for path in sorted(source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)
    return source


def _candidate(
    root: Path,
    catalog: TargetCatalog,
    *,
    hostile: bool = False,
    quant: str | None = None,
) -> ArenaCandidateBinding:
    source = _copy_candidate_source(root, hostile=hostile)
    if quant is not None:
        metadata_path = source / "metadata" / "silu_and_mul.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["quant"] = [quant]
        metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n")
    inspected = inspect_contribution(source, catalog=catalog)
    publication = publish_worker_bundle(
        source,
        root / "publications",
        content_hash(source),
    )
    target = catalog.require(inspected.target_id)
    reservation = QualificationReservation(
        _h(f"reservation:{root}"),
        publication.digest,
        inspected.target_id,
        inspected.selected_delta_digest,
        0,
        "miner",
        100,
        0,
        0,
        target.members,
    )
    return ArenaCandidateBinding(reservation, publication, 1)


def _static_runtime() -> ArenaRuntimeIdentity:
    return ArenaRuntimeIdentity(
        "production-b300-tp4",
        _h("runtime"),
        _h("base"),
        _h("overlay"),
        _h("worker"),
        _h("model-revision"),
        _h("model-manifest"),
        _h("model-content"),
        "sm103",
        "nvlink-domain",
        _h("topology"),
        4,
        4,
    )


def test_static_reopens_and_regrades_the_exact_worker_publication(
    tmp_path: Path,
) -> None:
    catalog = default_target_catalog()
    candidate = _candidate(tmp_path, catalog)
    result = B300StaticScreenAdapter(catalog).run_screen(
        _manifest(_static_runtime()),
        ScreenStagePolicy("static", 30_000),
        candidate,
    )

    assert result.stage == "static"
    assert result.grade is ScreenGrade.PASS


def test_static_candidate_policy_fault_is_fail_but_mutation_is_no_decision(
    tmp_path: Path,
) -> None:
    catalog = default_target_catalog()
    adapter = B300StaticScreenAdapter(catalog)
    manifest = _manifest(_static_runtime())
    policy = ScreenStagePolicy("static", 30_000)

    hostile = _candidate(tmp_path / "hostile", catalog, hostile=True)
    assert adapter.run_screen(manifest, policy, hostile).grade is ScreenGrade.FAIL

    mutated = _candidate(tmp_path / "mutated", catalog)
    leaf = mutated.publication.root / "manifest.toml"
    leaf.chmod(0o600)
    with leaf.open("a") as handle:
        handle.write("\n# changed after publication\n")
    assert adapter.run_screen(manifest, policy, mutated).grade is ScreenGrade.NO_DECISION


def test_static_rejects_candidate_outside_sealed_runtime_quant(
    tmp_path: Path,
) -> None:
    catalog = default_target_catalog()
    candidate = _candidate(tmp_path, catalog)
    manifest = _manifest(_static_runtime())
    policy = ScreenStagePolicy("static", 30_000)
    requirement = ((candidate.reservation.target_id, "nvfp4"),)
    adapter = B300StaticScreenAdapter(
        catalog,
        required_slot_quant=requirement,
    )

    assert adapter.run_screen(manifest, policy, candidate).grade is ScreenGrade.FAIL
    assert adapter.identity_digest != B300StaticScreenAdapter(catalog).identity_digest

    compatible = _candidate(tmp_path / "compatible", catalog, quant="nvfp4")
    assert adapter.run_screen(manifest, policy, compatible).grade is ScreenGrade.PASS


def _executor(root: Path) -> OCIEngineExecutor:
    runtime = _runtime_policy()
    executor = OCIEngineExecutor(
        OCIBackendConfig(
            OCIPrebuildConfig(
                "/usr/bin/docker",
                root / "recovery",
                root / "native-publications",
                root / "seccomp.json",
                "b300-screen",
                _prebuild_policy(runtime),
            ),
            runtime,
        ),
        DeviceStatePolicy(
            tuple(_gpu(index) for index in range(4)),
            required_consecutive_idle_samples=2,
            poll_interval_s=0.05,
            ready_poll_interval_s=0.05,
            drain_timeout_s=1.0,
            maximum_samples=8,
        ),
    )
    return executor


def _preflight(
    image: str,
    platform: str,
    worker: str,
    runtime: OCIRuntimeResourcePolicy,
) -> RuntimePreflightReceipt:
    return RuntimePreflightReceipt(
        schema="cacheon-runtime-preflight-v2",
        requested_image="registry.example/cacheon@sha256:" + image,
        image_digest=image,
        local_image_id="sha256:" + "a" * 64,
        repo_digests=("registry.example/cacheon@sha256:" + image,),
        oci_platform="linux/amd64",
        platform_digest=platform,
        docker_binary="/usr/bin/docker",
        uid=runtime.uid,
        gid=runtime.gid,
        sglang_version="0.0.0.dev1+g56e290315",
        worker_distribution="cacheon-harness",
        worker_version="0.0.1",
        worker_distribution_digest=worker,
        worker_file_count=100,
        worker_total_bytes=100_000,
        python_implementation="cpython",
        python_executable=runtime.container_python,
        python_version="3.12.0",
        python_abi="cpython-312-x86_64-linux-gnu",
        python_platform="linux-x86_64",
        machine="x86_64",
        package_versions=(),
        cudart_library="libcudart.so.13",
        cuda_visible_devices="",
        nvidia_visible_devices="void",
        security_argv_sha256=_h("preflight-argv"),
    )


def _engine_config(*, eager: bool) -> EngineSessionConfig:
    return EngineSessionConfig(
        model_path="/cacheon/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend="flashinfer",
        disable_cuda_graph=eager,
        mem_fraction_static=0.8,
        log_level="error",
        max_running_requests=32,
        tp_size=4,
        moe_runner_backend="flashinfer_trtllm",
        disable_custom_all_reduce=False,
    )


def _screen_case(tmp_path: Path, catalog: TargetCatalog):
    candidate = _candidate(tmp_path / "candidate", catalog)
    executor = _executor(tmp_path / "executor")
    runtime_policy = executor.config.runtime
    image, platform, worker = _h("image"), _h("platform"), _h("worker")
    preflight = _preflight(image, platform, worker, runtime_policy)
    identity = runtime_identity_from_preflight(preflight)
    runtime = ArenaRuntimeIdentity(
        "production-b300-tp4",
        identity.runtime_digest,
        identity.base_engine_digest,
        identity.validator_overlay_digest,
        worker,
        _h("model-revision"),
        _h("model-manifest"),
        _h("model-content"),
        "sm103",
        "nvlink-domain",
        _h("topology"),
        4,
        4,
    )
    manifest = _manifest(runtime)
    hardware = LogicalHardwareSpec(
        4,
        "sm103",
        runtime.topology_class,
        runtime.topology_digest,
        4,
        1,
        1,
        executor.device_policy.policy_sha256,
    )
    physical = PhysicalHardwareBinding(
        ("0", "1", "2", "3"),
        hardware.architecture,
        hardware.topology_class,
        hardware.topology_digest,
        4,
        1,
        1,
        hardware.device_policy_digest,
    )
    tree = tmp_path / "tree"
    model = tmp_path / "model"
    tree.mkdir(mode=0o700)
    model.mkdir(mode=0o700)
    native = NativeBuildSpec(
        _h("tree"),
        image,
        platform,
        worker,
        native_toolchain_digest(image_digest=image, platform_digest=platform),
        native_patcher_digest(worker_distribution_digest=worker),
        native_compiler_policy_digest(
            image_digest=image,
            worker_distribution_digest=worker,
            dependency_policy_digest=executor.config.prebuild.policy.dependency_policy_digest,
            target_architecture="sm103",
        ),
        "sm103",
        executor.config.prebuild.policy.dependency_policy_digest,
    )
    eager_config, graph_config = _engine_config(eager=True), _engine_config(eager=False)
    common = {
        "runtime_digest": runtime.runtime_digest,
        "base_engine_digest": runtime.base_engine_digest,
        "arena_digest": manifest.digest,
        "stack_digest": _h("stack"),
        "tree_digest": native.tree_digest,
        "image_digest": image,
        "platform_digest": platform,
        "controller_distribution_digest": _h("controller"),
        "worker_distribution_digest": worker,
        "model_revision_digest": runtime.model_revision_digest,
        "model_manifest_digest": runtime.model_manifest_digest,
        "model_content_digest": runtime.model_content_digest,
        "validator_overlay_digest": runtime.validator_overlay_digest,
        "seccomp_policy_digest": _h("seccomp"),
        "resource_policy_digest": executor.config.prebuild.policy.resource_policy_digest,
        "native_build_spec_digest": native.digest,
        "hardware": hardware,
    }
    eager_launch = EngineLaunchSpec(
        **common,
        engine_config_digest=eager_config.digest,
    )
    graph_launch = EngineLaunchSpec(
        **common,
        engine_config_digest=graph_config.digest,
    )
    binding = TrustedLaunchBinding(
        tree,
        eager_launch.controller_distribution_digest,
        native,
        preflight,
        physical,
    )
    mount = TrustedArenaModelMountReceipt.capture(
        model,
        arena_digest=manifest.digest,
        model_revision_digest=runtime.model_revision_digest,
        model_manifest_digest=runtime.model_manifest_digest,
        model_content_digest=runtime.model_content_digest,
    )
    audit = SlotAuditPolicy("1" * 32, 1_000_000, 2, (SLOT,), 4)

    def session(launch: EngineLaunchSpec, config: EngineSessionConfig):
        return SessionExecutionPlan(
            launch.digest,
            config.digest,
            config,
            expected_runtime_preflight(launch, preflight),
            (("warmup",), ("timed",)),
            1,
            1,
            2,
            1,
            0.0,
            audit_policy=audit,
        )

    plan = B300ScreenExecutionPlan(
        manifest.digest,
        candidate.digest,
        candidate.screen_attempt,
        candidate.reservation.selected_delta_digest,
        eager_launch,
        graph_launch,
        binding,
        mount,
        session(eager_launch, eager_config),
        session(graph_launch, graph_config),
        time.monotonic() + 120.0,
    )
    staging = tmp_path / "native-stage"
    staging.mkdir(mode=0o700)
    (staging / "kernel.so").write_bytes(b"sealed native bytes")
    publication = publish_native_artifact(
        staging,
        tmp_path / "native-store",
        build_spec_digest=native.digest,
    )
    prebuild = OCIPrebuildResult(
        graph_launch.digest,
        native.digest,
        publication,
        1.0,
        _h("build-argv"),
    )
    return manifest, candidate, executor, plan, prebuild, identity


def _execution(
    launch: EngineLaunchSpec,
    session_plan: SessionExecutionPlan,
    prebuild: OCIPrebuildResult,
    executor: OCIEngineExecutor,
    identity: CandidateFreeRuntimeIdentity,
    *,
    passing: bool,
) -> EngineExecutionEvidence:
    receipts = tuple(
        AuditReceiptFacts(
            SLOT,
            2,
            0 if passing else (1 if rank == 0 else 0),
            0,
            0,
            1.0 if passing or rank else 0.5,
            1.0,
            "allclose",
            100 + rank,
            rank,
            4,
        )
        for rank in range(4)
    )
    batches = tuple(
        BatchExecutionEvidence(
            index,
            f"{index + 1:032x}",
            f"{index + 3:032x}",
            float(index + 1),
            float(index + 2),
            1,
            BatchEvidence(()),
            receipts if index == 1 else (),
        )
        for index in range(2)
    )
    session = SessionExecutionEvidence(
        "a" * 32,
        launch.digest,
        session_plan.expected_preflight,
        1.0,
        batches,
        session_plan.warmup_count,
        session_plan.conditioning_count,
        1.0,
        2.0,
        1,
        3.0,
        audit_policy_digest=session_plan.audit_policy.digest,
    )
    return EngineExecutionEvidence(
        "cacheon-engine-execution-v1",
        launch.digest,
        identity,
        _h("preflight-receipt"),
        _h("model-receipt"),
        executor.config.runtime.digest,
        replace(prebuild, launch_digest=launch.digest),
        prebuild.publication.publication_digest,
        _h("runtime-argv"),
        (),
        (),  # Device receipts are already enforced by OCIEngineExecutor.
        session,
    )


@pytest.mark.parametrize(
    ("graph_passes", "expected"),
    ((True, ScreenGrade.PASS), (False, ScreenGrade.FAIL)),
)
def test_pipeline_retains_reopens_and_host_regrades_each_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph_passes: bool,
    expected: ScreenGrade,
) -> None:
    catalog = default_target_catalog()
    manifest, candidate, executor, plan, prebuild, identity = _screen_case(
        tmp_path,
        catalog,
    )
    calls: list[str] = []

    monkeypatch.setattr(
        screen_stages,
        "_resolve_candidate_tree",
        lambda *_args, **_kwargs: calls.append("resolve"),
    )
    monkeypatch.setattr(
        screen_stages,
        "run_oci_prebuild",
        lambda *args, **kwargs: (calls.append("prebuild") or prebuild),
    )

    def execute(_self, launch, _binding, _mount, session, *, deadline):
        calls.append("eager" if launch is plan.eager_launch else "graph")
        return _execution(
            launch,
            session,
            prebuild,
            executor,
            identity,
            passing=(True if launch is plan.eager_launch else graph_passes),
        )

    monkeypatch.setattr(OCIEngineExecutor, "execute", execute)
    adapter = B300BuildABIGraphScreenAdapter(
        catalog=catalog,
        executor=executor,
        plan_resolver_digest=_h("plan-resolver"),
        plan_resolver=lambda _manifest, _candidate: plan,
        evidence_policy_digest=_h("evidence-policy"),
        evidence_root=tmp_path / "evidence",
    )
    try:
        results = tuple(
            adapter.run_screen(
                manifest,
                ScreenStagePolicy(stage, 30_000),
                candidate,
            )
            for stage in ("build", "abi", "graph")
        )
    finally:
        adapter.close()
        executor.manager.close()

    assert tuple(row.grade for row in results) == (
        ScreenGrade.PASS,
        ScreenGrade.PASS,
        expected,
    )
    assert "prebuild" in calls
    assert calls.count("eager") == 1
    assert calls.count("graph") == 1


def test_resident_pipeline_builds_then_defers_gpu_checks_to_resident_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = default_target_catalog()
    manifest, candidate, executor, plan, prebuild, _identity = _screen_case(
        tmp_path,
        catalog,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        screen_stages,
        "_resolve_candidate_tree",
        lambda *_args, **_kwargs: calls.append("resolve"),
    )
    monkeypatch.setattr(
        screen_stages,
        "run_oci_prebuild",
        lambda *_args, **_kwargs: (calls.append("prebuild") or prebuild),
    )
    monkeypatch.setattr(
        OCIEngineExecutor,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resident routing must not boot a per-candidate engine")
        ),
    )
    adapter = B300BuildABIGraphScreenAdapter(
        catalog=catalog,
        executor=executor,
        plan_resolver_digest=_h("plan-resolver"),
        plan_resolver=lambda _manifest, _candidate: plan,
        evidence_policy_digest=_h("evidence-policy"),
        evidence_root=tmp_path / "evidence",
        execution_mode="resident",
    )
    try:
        results = tuple(
            adapter.run_screen(
                manifest,
                ScreenStagePolicy(stage, 30_000),
                candidate,
            )
            for stage in ("build", "abi", "graph")
        )
    finally:
        adapter.close()
        executor.manager.close()

    assert tuple(row.grade for row in results) == (
        ScreenGrade.PASS,
        ScreenGrade.PASS,
        ScreenGrade.PASS,
    )
    assert calls.count("prebuild") == 1
    assert calls.count("resolve") >= 3


def test_pipeline_order_or_execution_exception_is_no_decision_and_clears_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = default_target_catalog()
    manifest, candidate, executor, plan, prebuild, _identity = _screen_case(
        tmp_path,
        catalog,
    )
    monkeypatch.setattr(screen_stages, "_resolve_candidate_tree", lambda *_a, **_k: None)
    monkeypatch.setattr(screen_stages, "run_oci_prebuild", lambda *_a, **_k: prebuild)
    monkeypatch.setattr(
        OCIEngineExecutor,
        "execute",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("runtime unavailable")),
    )
    adapter = B300BuildABIGraphScreenAdapter(
        catalog=catalog,
        executor=executor,
        plan_resolver_digest=_h("plan-resolver"),
        plan_resolver=lambda _manifest, _candidate: plan,
        evidence_policy_digest=_h("evidence-policy"),
        evidence_root=tmp_path / "evidence",
    )
    try:
        out_of_order = adapter.run_screen(
            manifest,
            ScreenStagePolicy("abi", 30_000),
            candidate,
        )
        build = adapter.run_screen(
            manifest,
            ScreenStagePolicy("build", 30_000),
            candidate,
        )
        abi = adapter.run_screen(
            manifest,
            ScreenStagePolicy("abi", 30_000),
            candidate,
        )
        graph_after_clear = adapter.run_screen(
            manifest,
            ScreenStagePolicy("graph", 30_000),
            candidate,
        )
    finally:
        adapter.close()
        executor.manager.close()

    assert out_of_order.grade is ScreenGrade.NO_DECISION
    assert build.grade is ScreenGrade.PASS
    assert abi.grade is ScreenGrade.NO_DECISION
    assert graph_after_clear.grade is ScreenGrade.NO_DECISION


def test_typed_candidate_exception_is_exact_terminal_screen_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = default_target_catalog()
    manifest, candidate, executor, plan, prebuild, _identity = _screen_case(
        tmp_path, catalog,
    )
    monkeypatch.setattr(screen_stages, "_resolve_candidate_tree", lambda *_a, **_k: None)
    monkeypatch.setattr(screen_stages, "run_oci_prebuild", lambda *_a, **_k: prebuild)
    failure = (
        "rank 3 RuntimeError in moe.fused_experts during entry at "
        "kernels/moe.py:117: invalid launch geometry"
    )
    monkeypatch.setattr(
        OCIEngineExecutor,
        "execute",
        lambda *_a, **_k: (_ for _ in ()).throw(
            OuterSessionCandidateError(
                "batch: CandidateEngineFailure",
                candidate_failure=failure,
                candidate_failure_type="CandidateEngineFailure",
            )
        ),
    )
    adapter = B300BuildABIGraphScreenAdapter(
        catalog=catalog,
        executor=executor,
        plan_resolver_digest=_h("plan-resolver"),
        plan_resolver=lambda _manifest, _candidate: plan,
        evidence_policy_digest=_h("evidence-policy"),
        evidence_root=tmp_path / "evidence",
    )
    try:
        assert adapter.run_screen(
            manifest, ScreenStagePolicy("build", 30_000), candidate
        ).grade is ScreenGrade.PASS
        abi = adapter.run_screen(
            manifest, ScreenStagePolicy("abi", 30_000), candidate
        )
    finally:
        adapter.close()
        executor.manager.close()

    assert abi.grade is ScreenGrade.FAIL
    assert abi.reason.startswith("candidate_exception (CandidateEngineFailure:")
    assert "kernels/moe.py:117" in abi.reason
    assert "invalid launch geometry" in abi.reason


def test_composition_exposes_only_the_exact_non_serving_order(tmp_path: Path) -> None:
    catalog = default_target_catalog()
    executor = _executor(tmp_path / "executor")
    pipeline = B300BuildABIGraphScreenAdapter(
        catalog=catalog,
        executor=executor,
        plan_resolver_digest=_h("plan-resolver"),
        plan_resolver=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        evidence_policy_digest=_h("evidence-policy"),
        evidence_root=tmp_path / "evidence",
    )
    try:
        handlers = compose_b300_non_serving_screen_handlers(
            B300StaticScreenAdapter(catalog),
            pipeline,
            pipeline_resource_ids=("b300-lane",),
        )
    finally:
        pipeline.close()
        executor.manager.close()

    assert tuple(row.stage for row in handlers) == SCREEN_STAGES[:-1]
    assert handlers[0].resource_ids == ()
    assert all(row.resource_ids == ("b300-lane",) for row in handlers[1:])

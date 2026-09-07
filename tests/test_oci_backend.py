from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import os
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.eval.oci_backend as backend
from cacheon.eval.device_state import (
    DeviceStateActiveReceipt,
    DeviceStatePolicy,
    DeviceStateReceipt,
    GPUConfiguration,
)
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    LogicalHardwareSpec,
    NativeBuildSpec,
    PhysicalHardwareBinding,
    ResolvedEngineLaunch,
    TrustedLaunchBinding,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
)
from cacheon.eval.oci_backend import (
    EngineExecutionEvidence,
    OCIBackendConfig,
    OCIBackendError,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
    TrustedArenaModelMountReceipt,
    build_runtime_argv,
    expected_runtime_preflight,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_outer_session import (
    OuterSessionProtocolError,
    SessionExecutionEvidence,
    SessionExecutionPlan,
)
from cacheon.eval.oci_prebuild import (
    OCIPrebuildConfig,
    OCIPrebuildPolicy,
    OCIPrebuildResult,
)
from cacheon.eval.oci_process import (
    STDERR_ARTIFACT_SCHEMA,
    CommandResult,
    OCIAttachedDiagnostic,
    OCIProcessManager,
    OCIStderrArtifactReceipt,
)
from cacheon.eval.oci_session_protocol import EngineSessionConfig
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt
from tests.support.b300 import gpu, prebuild_policy, runtime_policy, sha as _digest
from tests.support.preflight import preflight_receipt


def _gpu() -> GPUConfiguration:
    return gpu(model="rtx6000")


def _runtime_policy() -> OCIRuntimeResourcePolicy:
    return runtime_policy()


def _prebuild_policy(runtime: OCIRuntimeResourcePolicy) -> OCIPrebuildPolicy:
    return prebuild_policy(runtime)


DOCKER = "/usr/bin/docker"
IMAGE_ID = "sha256:" + "a" * 64


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)
    return path


def _preflight(
    *, image: str, platform: str, worker: str, runtime: OCIRuntimeResourcePolicy
) -> RuntimePreflightReceipt:
    return preflight_receipt(
        image=image,
        platform=platform,
        worker=worker,
        uid=runtime.uid,
        gid=runtime.gid,
        python_executable=runtime.container_python,
    )


def _device_policy() -> DeviceStatePolicy:
    return DeviceStatePolicy(
        expected_gpus=(_gpu(),),
        required_consecutive_idle_samples=2,
        poll_interval_s=0.05,
        ready_poll_interval_s=0.05,
        drain_timeout_s=2.0,
        maximum_samples=8,
    )


def _case(tmp_path: Path) -> SimpleNamespace:
    runtime = _runtime_policy()
    prebuild_policy = _prebuild_policy(runtime)
    image = _digest("image")
    platform = _digest("platform")
    worker = _digest("worker")
    preflight = _preflight(
        image=image, platform=platform, worker=worker, runtime=runtime
    )
    identity = runtime_identity_from_preflight(preflight)
    device_policy = _device_policy()
    topology = _digest("topology")
    hardware = LogicalHardwareSpec(
        visible_gpu_count=1,
        architecture="sm120",
        topology_class="pcie_switch",
        topology_digest=topology,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        device_policy_digest=device_policy.policy_sha256,
    )
    physical = PhysicalHardwareBinding(
        physical_gpu_ids=("0",),
        architecture=hardware.architecture,
        topology_class=hardware.topology_class,
        topology_digest=hardware.topology_digest,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        device_policy_digest=hardware.device_policy_digest,
    )
    tree = _mkdir(tmp_path / "engine-tree")
    model = _mkdir(tmp_path / "model")
    recovery = _mkdir(tmp_path / "recovery")
    publication_base = _mkdir(tmp_path / "native-publications")
    tree_digest = _digest("tree")
    native = NativeBuildSpec(
        tree_digest=tree_digest,
        image_digest=image,
        platform_digest=platform,
        worker_distribution_digest=worker,
        toolchain_digest=native_toolchain_digest(
            image_digest=image, platform_digest=platform
        ),
        patcher_digest=native_patcher_digest(
            worker_distribution_digest=worker
        ),
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=image,
            worker_distribution_digest=worker,
            dependency_policy_digest=prebuild_policy.dependency_policy_digest,
            target_architecture="sm120",
        ),
        target_architecture="sm120",
        dependency_policy_digest=prebuild_policy.dependency_policy_digest,
    )
    engine_config = EngineSessionConfig(
        model_path="/cacheon/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend="flashinfer",
        disable_cuda_graph=False,
        mem_fraction_static=0.82,
        log_level="error",
        max_running_requests=64,
        tp_size=1,
        moe_runner_backend="flashinfer_trtllm",
        disable_custom_all_reduce=False,
        engine_kwargs={},
    )
    seccomp = Path(backend.__file__).with_name("seccomp_moby_v0_2_1.json")
    launch = EngineLaunchSpec(
        runtime_digest=identity.runtime_digest,
        base_engine_digest=identity.base_engine_digest,
        arena_digest=_digest("arena"),
        stack_digest=_digest("stack"),
        tree_digest=tree_digest,
        image_digest=image,
        platform_digest=platform,
        controller_distribution_digest=_digest("controller"),
        worker_distribution_digest=worker,
        model_revision_digest=_digest("model-revision"),
        model_manifest_digest=_digest("model-manifest"),
        model_content_digest=_digest("model-content"),
        validator_overlay_digest=identity.validator_overlay_digest,
        engine_config_digest=engine_config.digest,
        seccomp_policy_digest=hashlib.sha256(seccomp.read_bytes()).hexdigest(),
        resource_policy_digest=prebuild_policy.resource_policy_digest,
        native_build_spec_digest=native.digest,
        hardware=hardware,
    )
    binding = TrustedLaunchBinding(
        materialized_tree_root=tree,
        controller_distribution_digest=launch.controller_distribution_digest,
        native_build_spec=native,
        runtime_preflight_receipt=preflight,
        physical_hardware=physical,
    )
    mount = TrustedArenaModelMountReceipt.capture(
        model,
        arena_digest=launch.arena_digest,
        model_revision_digest=launch.model_revision_digest,
        model_manifest_digest=launch.model_manifest_digest,
        model_content_digest=launch.model_content_digest,
    )
    expected = expected_runtime_preflight(launch, preflight)
    plan = SessionExecutionPlan(
        launch_digest=launch.digest,
        expected_engine_config_digest=engine_config.digest,
        engine_config=engine_config,
        expected_preflight=expected,
        prompt_batches=(("warmup",), ("timed",)),
        warmup_count=1,
        conditioning_count=1,
        max_new_tokens=1,
        top_logprobs_num=1,
        temperature=0.0,
    )
    prebuild = OCIPrebuildConfig(
        docker_binary=DOCKER,
        recovery_root=recovery,
        publication_root=publication_base,
        seccomp_profile=seccomp,
        executor_id="validator-a",
        policy=prebuild_policy,
    )
    config = OCIBackendConfig(prebuild=prebuild, runtime=runtime)
    resolved = ResolvedEngineLaunch(
        launch,
        SimpleNamespace(root=tree, files=(), runtime_manifest=None),
        native,
        physical,
        tuple(sorted(preflight.launch_identity().items())),
    )
    artifact_root = _mkdir(
        publication_base / native.digest[:2] / native.digest
    )
    publication = SimpleNamespace(
        root=artifact_root,
        build_spec_digest=native.digest,
        publication_digest=_digest("native-publication"),
        reused=False,
        files=(),
    )
    return SimpleNamespace(
        runtime=runtime,
        prebuild_policy=prebuild_policy,
        preflight=preflight,
        identity=identity,
        device_policy=device_policy,
        native=native,
        launch=launch,
        binding=binding,
        mount=mount,
        plan=plan,
        config=config,
        resolved=resolved,
        publication=publication,
        tree=tree,
        model=model,
        recovery=recovery,
    )


def _runner(
    argv: tuple[str, ...], *, timeout_s: float, max_output_bytes: int
) -> CommandResult:
    del argv, timeout_s, max_output_bytes
    return CommandResult(0, b"", b"")


def _manager(case: SimpleNamespace) -> OCIProcessManager:
    return OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=case.recovery,
        executor_id="validator-a",
        runner=_runner,
        clock=lambda: 100.0,
    )


def _executor(case: SimpleNamespace) -> OCIEngineExecutor:
    return OCIEngineExecutor(case.config, case.device_policy, manager=_manager(case))


def _executor_with(case, manager, session_runner) -> OCIEngineExecutor:
    return OCIEngineExecutor(
        case.config,
        case.device_policy,
        manager=manager,
        session_runner=session_runner,
    )


def _execute(executor, case):
    return executor.execute(
        case.launch, case.binding, case.mount, case.plan, deadline=200.0
    )


def _execute_reference_runtime(executor, case, run):
    return executor._execute_runtime(
        case.launch,
        case.binding,
        case.mount,
        absolute=200.0,
        resolved=case.resolved,
        preflight=case.preflight,
        model_root=case.model,
        session_protocol="reference",
        run=run,
    )


def _argv(case, lease, cache, resolved, runtime, **over):
    return build_runtime_argv(
        lease=lease,
        resolved=resolved,
        preflight=case.preflight,
        model_root=case.model,
        publication=case.publication,
        cache_root=cache,
        seccomp_path=lease.stage_paths[0],
        runtime=runtime,
        **over,
    )


def _resolved(case: SimpleNamespace, launch: EngineLaunchSpec) -> ResolvedEngineLaunch:
    return replace(case.resolved, spec=launch)


def _idle_receipt(
    case: SimpleNamespace, launch_id: str, phase: str, sequence: int, start: float
) -> DeviceStateReceipt:
    return DeviceStateReceipt(
        "cacheon.device-state.v1",
        sequence,
        launch_id,
        phase,
        (0,),
        case.device_policy.configuration_sha256,
        case.device_policy.policy_sha256,
        start,
        start + 1.0,
        2,
        (),
    )


def _active_receipt(
    case: SimpleNamespace, launch_id: str
) -> DeviceStateActiveReceipt:
    return DeviceStateActiveReceipt(
        "cacheon.device-state-active.v1",
        2,
        launch_id,
        "final-warmup",
        (0,),
        case.device_policy.configuration_sha256,
        case.device_policy.policy_sha256,
        3.0,
        4.0,
        1,
        0,
        1,
        (),
    )


class _DeviceGuard:
    def __init__(self, case: SimpleNamespace) -> None:
        self.case = case
        self.deadlines: list[tuple[str, float]] = []

    def before_launch(self, launch_id: str, *, deadline: float) -> DeviceStateReceipt:
        self.deadlines.append(("pre", deadline))
        return _idle_receipt(self.case, launch_id, "pre", 1, 1.0)

    def condition_active(
        self,
        launch_id: str,
        event: str,
        *,
        deadline: float,
        release,
        wait_for_release,
        cancel,
    ) -> DeviceStateActiveReceipt:
        del release, cancel
        assert event == "final-warmup"
        self.deadlines.append(("active", deadline))
        assert wait_for_release(1.0)
        return _active_receipt(self.case, launch_id)

    def after_launch(self, launch_id: str, *, deadline: float) -> DeviceStateReceipt:
        self.deadlines.append(("post", deadline))
        return _idle_receipt(self.case, launch_id, "post", 3, 5.0)


def _session_evidence(case: SimpleNamespace) -> SessionExecutionEvidence:
    return SessionExecutionEvidence(
        "1" * 32,
        case.launch.digest,
        case.plan.expected_preflight,
        1.0,
        (),
        case.plan.warmup_count,
        case.plan.conditioning_count,
        2.0,
        3.0,
        1,
        4.0,
    )


def _install_execution_fakes(
    case: SimpleNamespace,
    executor: OCIEngineExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> list[float]:
    prebuild_deadlines: list[float] = []

    def fake_prebuild(*args, deadline: float, **kwargs) -> OCIPrebuildResult:
        del args, kwargs
        prebuild_deadlines.append(deadline)
        return OCIPrebuildResult(
            case.launch.digest,
            case.native.digest,
            case.publication,
            1.0,
            _digest("prebuild-argv"),
        )

    monkeypatch.setattr(backend, "run_oci_prebuild", fake_prebuild)
    monkeypatch.setattr(
        backend,
        "resolve_engine_launch",
        lambda launch, binding: _resolved(case, launch),
    )
    monkeypatch.setattr(
        backend, "reopen_native_artifact", lambda *args, **kwargs: case.publication
    )
    monkeypatch.setattr(backend, "reopen_launch_tree", lambda *args, **kwargs: None)

    def mount_tmpfs(lease, path, **kwargs):
        del lease, kwargs
        return _mkdir(path)

    monkeypatch.setattr(executor.manager, "mount_tmpfs", mount_tmpfs)
    executor.device_guard = _DeviceGuard(case)  # type: ignore[assignment]
    return prebuild_deadlines


def test_pristine_native_publication_accepts_only_the_prebuild_receipt() -> None:
    def publication(*paths: str, directories: tuple[str, ...] = ()):
        return SimpleNamespace(
            directories=directories,
            files=tuple(SimpleNamespace(path=path) for path in paths),
        )

    assert backend._reference_publication_is_control_only(
        publication("prebuild.json")
    )
    assert not backend._reference_publication_is_control_only(publication())
    assert not backend._reference_publication_is_control_only(
        publication("renamed-prebuild.json")
    )
    assert not backend._reference_publication_is_control_only(
        publication("prebuild.json", "cuda/kernel.so", directories=("cuda",))
    )
    assert not backend._reference_publication_is_control_only(
        publication("metadata/prebuild.json", directories=("metadata",))
    )


def test_reference_reopens_control_receipt_then_rejects_added_native_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    executor = _executor(case)
    _install_execution_fakes(case, executor, monkeypatch)
    control = SimpleNamespace(
        **dict(
            vars(case.publication),
            directories=(),
            files=(SimpleNamespace(path="prebuild.json"),),
        )
    )
    changed = SimpleNamespace(
        **dict(
            vars(case.publication),
            directories=("cuda",),
            files=(
                SimpleNamespace(path="prebuild.json"),
                SimpleNamespace(path="cuda/kernel.so"),
            ),
        )
    )
    reopened = iter((control, changed))
    monkeypatch.setattr(
        backend,
        "reopen_native_artifact",
        lambda *args, **kwargs: next(reopened),
    )

    with pytest.raises(OCIBackendError, match="acquired contribution state"):
        _execute_reference_runtime(executor, case, lambda *_args: object())
    assert executor.device_guard.deadlines == [  # type: ignore[attr-defined]
        ("pre", 200.0),
        ("post", 200.0),
    ]


def test_reference_runtime_accepts_control_receipt_through_both_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    executor = _executor(case)
    _install_execution_fakes(case, executor, monkeypatch)
    control = SimpleNamespace(
        **dict(
            vars(case.publication),
            directories=(),
            files=(SimpleNamespace(path="prebuild.json"),),
        )
    )
    monkeypatch.setattr(
        backend,
        "reopen_native_artifact",
        lambda *args, **kwargs: control,
    )

    class Transport:
        def __init__(self, *_args, **_kwargs) -> None:
            self.aborted = False

        def abort(self) -> None:
            self.aborted = True

    monkeypatch.setattr(backend, "AttachedReferenceTransport", Transport)
    marker = object()
    raw = _execute_reference_runtime(executor, case, lambda *_args: marker)
    assert raw.value is marker
    assert raw.publication_digest == control.publication_digest
    assert executor.device_guard.deadlines == [  # type: ignore[attr-defined]
        ("pre", 200.0),
        ("post", 200.0),
    ]


def test_runtime_identity_and_backend_policy_are_closed(tmp_path: Path) -> None:
    case = _case(tmp_path)

    assert case.identity == runtime_identity_from_preflight(case.preflight)
    assert case.launch.runtime_digest == case.identity.runtime_digest
    assert case.launch.base_engine_digest == case.identity.base_engine_digest
    assert case.launch.validator_overlay_digest == case.identity.validator_overlay_digest
    assert case.launch.resource_policy_digest == case.prebuild_policy.resource_policy_digest

    bad_policy = replace(
        case.prebuild_policy, runtime_policy_digest=_digest("another-runtime-policy")
    )
    bad_prebuild = replace(case.config.prebuild, policy=bad_policy)
    with pytest.raises(OCIBackendError, match="do not share one identity"):
        OCIBackendConfig(prebuild=bad_prebuild, runtime=case.runtime)


def test_runtime_cpuset_policy_is_canonical_paired_and_digest_bound(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    policy = case.runtime
    isolated = replace(
        policy,
        cpuset_cpus="0-3,8-11",
        cpuset_mems="0",
    )
    assert isolated.cpuset_cpus == "0-3,8-11"
    assert isolated.cpuset_mems == "0"
    assert isolated.digest != policy.digest
    with pytest.raises(OCIBackendError, match="do not share one identity"):
        OCIBackendConfig(prebuild=case.config.prebuild, runtime=isolated)

    invalid = (
        ("0-7", None),
        (None, "0"),
        ("", "0"),
        ("00-7", "0"),
        ("0,1", "0"),
        ("4-7,0-3", "0"),
        ("0-4,4-7", "0"),
        ("7-0", "0"),
        ("0-6", "0"),  # Seven CPUs cannot satisfy an eight-CPU quota.
        ("1048576", "0"),
        ("0-7", "65536"),
    )
    for cpus, mems in invalid:
        with pytest.raises(OCIBackendError, match="runtime resource cpuset|cpu_millis"):
            replace(policy, cpuset_cpus=cpus, cpuset_mems=mems)


def test_model_mount_receipt_rejects_relative_crlf_and_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(OCIBackendError, match="absolute host path"):
        TrustedArenaModelMountReceipt.capture(
            Path("relative-model"),
            arena_digest=_digest("arena"),
            model_revision_digest=_digest("revision"),
            model_manifest_digest=_digest("manifest"),
            model_content_digest=_digest("content"),
        )

    for separator in ("\r", "\n"):
        malicious = _mkdir(tmp_path / f"model{separator}injected")
        with pytest.raises(OCIBackendError, match="closed OCI mount"):
            TrustedArenaModelMountReceipt.capture(
                malicious,
                arena_digest=_digest("arena"),
                model_revision_digest=_digest("revision"),
                model_manifest_digest=_digest("manifest"),
                model_content_digest=_digest("content"),
            )

    case = _case(tmp_path / "stable")
    original = case.mount.model_root
    moved = original.with_name("old-model")
    original.rename(moved)
    _mkdir(original)
    with pytest.raises(OCIBackendError, match="identity changed"):
        case.mount.reopen()


def test_runtime_argv_is_exact_closed_and_mount_minimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    manager = _manager(case)
    lease = manager.register(
        lease_id="runtime-" + "1" * 32,
        container_name="cacheon-runtime-" + "1" * 32,
        mount_relpaths=("runtime-cache",),
        stage_relpaths=("seccomp.json",),
    )
    cache = _mkdir(lease.mount_paths[0])
    lease.stage_paths[0].write_bytes(case.config.prebuild.seccomp_profile.read_bytes())
    argv = _argv(case, lease, cache, case.resolved, case.runtime)

    for exact in (
        "--pull=never",
        "--runtime=runc",
        "--network=none",
        "--ipc=private",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--gpus=device=0",
        f"--user={case.runtime.uid}:{case.runtime.gid}",
        IMAGE_ID,
        "--env=CACHEON_TARGET_GPU_ARCH=sm120",
        f"--env=CACHEON_NATIVE_BUILD_SPEC_DIGEST={case.native.digest}",
        f"--env=CACHEON_NATIVE_ARTIFACT_PUBLICATION_DIGEST={case.publication.publication_digest}",
        "--env=CACHEON_REBUILD_PHASE=load",
        "--env=SGLANG_CACHE_DIR=/cacheon/runtime-cache/sglang",
        "--env=TORCHINDUCTOR_CACHE_DIR=/cacheon/runtime-cache/torchinductor",
        "--env=TRITON_HOME=/cacheon/runtime-cache/triton-home",
    ):
        assert exact in argv
    assert any(row.startswith("--tmpfs=/tmp:rw,nosuid,nodev,exec,") for row in argv)
    assert not any(row.startswith("--cap-add") for row in argv)
    assert "--pid=host" not in argv
    assert not any('"' in row or "'" in row for row in argv)
    assert argv.count("--cap-drop=ALL") == 1
    assert case.preflight.requested_image not in argv
    assert not any(row.startswith("--cpuset-") for row in argv)
    mounts = tuple(row for row in argv if row.startswith("--mount="))
    assert len(mounts) == 6
    assert sum(",readonly" not in row for row in mounts) == 1
    assert f"dst={backend.CONTAINER_ENGINE_WORKER_POLICY}" in mounts[0]
    assert f"dst={backend.CONTAINER_SITE_BOOTSTRAP}" in mounts[1]
    assert f"src={case.model},dst=/cacheon/input/model" in mounts[2]
    assert f"src={case.tree},dst=/cacheon/engine-tree" in mounts[3]
    assert f"src={case.publication.root},dst=/cacheon/native-artifacts/" in mounts[4]
    assert f"src={case.publication.root.parent.parent}," not in mounts[4]
    assert f"src={cache},dst=/cacheon/runtime-cache" in mounts[5]
    encoded = "\n".join(argv).lower()
    for forbidden in (".pass", "credentials", "docker.sock", "result-output"):
        assert forbidden not in encoded

    assert "--read-only" not in argv

    multi_resolved = replace(
        case.resolved,
        physical_hardware=replace(
            case.resolved.physical_hardware,
            physical_gpu_ids=("0", "1"),
        ),
    )
    multi_argv = _argv(case, lease, cache, multi_resolved, case.runtime)
    assert '--gpus="device=0,1"' in multi_argv

    isolated_runtime = replace(
        case.runtime,
        cpuset_cpus="0-3,8-11",
        cpuset_mems="0",
    )
    isolated_argv = _argv(case, lease, cache, case.resolved, isolated_runtime)
    assert "--cpuset-cpus=0-3,8-11" in isolated_argv
    assert "--cpuset-mems=0" in isolated_argv

    reference_argv = _argv(
        case, lease, cache, case.resolved, case.runtime, session_protocol="reference"
    )
    assert "--env=CACHEON_SESSION_PROTOCOL=reference" in reference_argv
    with pytest.raises(OCIBackendError, match="protocol"):
        _argv(
            case,
            lease,
            cache,
            case.resolved,
            case.runtime,
            session_protocol="candidate-chosen",
        )


def test_launch_validation_binds_runtime_model_config_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    executor = _executor(case)
    monkeypatch.setattr(
        backend,
        "resolve_engine_launch",
        lambda launch, binding: _resolved(case, launch),
    )

    resolved, preflight, identity, expected, model_root = executor._validate_launch(
        case.launch, case.binding, case.mount, case.plan
    )
    assert resolved.spec == case.launch
    assert preflight == case.preflight
    assert identity == case.identity
    assert expected == case.plan.expected_preflight
    assert model_root == case.model

    for field in ("runtime_digest", "base_engine_digest", "validator_overlay_digest"):
        with pytest.raises(OCIBackendError, match="unsubstantiated"):
            executor._validate_launch(
                replace(case.launch, **{field: _digest("bad-" + field)}),
                case.binding,
                case.mount,
                case.plan,
            )

    with pytest.raises(OCIBackendError, match="arena/model"):
        executor._validate_launch(
            case.launch,
            case.binding,
            replace(case.mount, arena_digest=_digest("wrong-arena")),
            case.plan,
        )

    changed_config = replace(case.plan.engine_config, dtype="float16")
    changed_plan = SessionExecutionPlan(
        launch_digest=case.launch.digest,
        expected_engine_config_digest=changed_config.digest,
        engine_config=changed_config,
        expected_preflight=replace(
            case.plan.expected_preflight,
            engine_config_digest=changed_config.digest,
        ),
        prompt_batches=case.plan.prompt_batches,
        warmup_count=1,
        conditioning_count=1,
        max_new_tokens=1,
        top_logprobs_num=1,
        temperature=0.0,
    )
    with pytest.raises(OCIBackendError, match="configuration"):
        executor._validate_launch(
            case.launch, case.binding, case.mount, changed_plan
        )

    changed_facts = replace(
        case.plan.expected_preflight, runtime_digest=_digest("worker-lied")
    )
    changed_expected_plan = replace(case.plan, expected_preflight=changed_facts)
    with pytest.raises(OCIBackendError, match="expected preflight"):
        executor._validate_launch(
            case.launch, case.binding, case.mount, changed_expected_plan
        )


def test_execute_shares_deadline_and_returns_raw_triplet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    manager = _manager(case)
    session_calls: list[dict[str, object]] = []

    def session_runner(plan: SessionExecutionPlan, **kwargs) -> SessionExecutionEvidence:
        session_calls.append(kwargs)
        callback = kwargs["boundary_callback"]
        deadline = kwargs["deadline"]
        callback("before_final_warmup", 0, deadline)
        callback("after_final_warmup", 0, deadline)
        callback("before_first_timed", 1, deadline)
        return _session_evidence(case)

    executor = _executor_with(case, manager, session_runner)
    prebuild_deadlines = _install_execution_fakes(case, executor, monkeypatch)
    result = _execute(executor, case)

    assert type(result) is EngineExecutionEvidence
    assert prebuild_deadlines == [200.0]
    assert session_calls[0]["deadline"] == 198.0
    assert session_calls[0]["clock"] is manager.clock
    assert tuple(type(row) for row in result.device_receipts) == (
        DeviceStateReceipt,
        DeviceStateActiveReceipt,
        DeviceStateReceipt,
    )
    assert (
        result.device_receipts[0].phase,
        result.device_receipts[1].event,
        result.device_receipts[2].phase,
    ) == ("pre", "final-warmup", "post")
    assert result.session == _session_evidence(case)
    assert not tuple(manager.leases_root.glob("*.json"))
    assert not tuple(manager.resources_root.iterdir())
    assert executor.device_guard.deadlines == [  # type: ignore[attr-defined]
        ("pre", 200.0),
        ("active", 198.0),
        ("post", 200.0),
    ]


def test_execute_opened_keeps_normal_isolation_and_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    manager = _manager(case)
    opened: list[object] = []

    class FakeOpenedSession:
        def __init__(self, plan: SessionExecutionPlan, **kwargs) -> None:
            assert plan is case.plan
            self.kwargs = kwargs
            self.session_id = "1" * 32
            self.closed = False
            opened.append(self)

        def start(self) -> None:
            callback = self.kwargs["boundary_callback"]
            deadline = self.kwargs["deadline"]
            callback("before_final_warmup", 0, deadline)
            callback("after_final_warmup", 0, deadline)
            callback("before_first_timed", 1, deadline)

        def finish(self, *, require_all: bool = True) -> SessionExecutionEvidence:
            assert require_all is False
            self.closed = True
            return _session_evidence(case)

        def abort(self) -> None:
            self.closed = True

    monkeypatch.setattr(backend, "OpenedOuterSession", FakeOpenedSession)
    executor = OCIEngineExecutor(case.config, case.device_policy, manager=manager)
    _install_execution_fakes(case, executor, monkeypatch)

    def driver(session) -> SessionExecutionEvidence:
        assert session is opened[0]
        return session.finish(require_all=False)

    result = executor.execute_opened(
        case.launch,
        case.binding,
        case.mount,
        case.plan,
        deadline=200.0,
        driver=driver,
    )

    assert result.schema == "cacheon.oci-resident-engine-execution.v1"
    assert result.session == _session_evidence(case)
    assert not tuple(manager.leases_root.glob("*.json"))
    assert not tuple(manager.resources_root.iterdir())
    assert executor.device_guard.deadlines == [  # type: ignore[attr-defined]
        ("pre", 200.0),
        ("active", 198.0),
        ("post", 200.0),
    ]


def test_execute_opened_rejects_driver_that_leaves_engine_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    manager = _manager(case)

    class FakeOpenedSession:
        session_id = "1" * 32

        def __init__(self, _plan: SessionExecutionPlan, **_kwargs) -> None:
            self.closed = False

        def start(self) -> None:
            return None

        def abort(self) -> None:
            self.closed = True

    monkeypatch.setattr(backend, "OpenedOuterSession", FakeOpenedSession)
    executor = OCIEngineExecutor(case.config, case.device_policy, manager=manager)
    _install_execution_fakes(case, executor, monkeypatch)

    with pytest.raises(OCIBackendError, match="without closing"):
        executor.execute_opened(
            case.launch,
            case.binding,
            case.mount,
            case.plan,
            deadline=200.0,
            driver=lambda _session: _session_evidence(case),
        )
    assert not tuple(manager.leases_root.glob("*.json"))
    assert not tuple(manager.resources_root.iterdir())
    assert executor.device_guard.deadlines == [  # type: ignore[attr-defined]
        ("pre", 200.0),
        ("post", 200.0),
    ]


def test_execute_failure_still_releases_lease_and_post_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    manager = _manager(case)

    def fail_session(plan: SessionExecutionPlan, **kwargs) -> SessionExecutionEvidence:
        del plan, kwargs
        raise RuntimeError("session failed")

    executor = _executor_with(case, manager, fail_session)
    _install_execution_fakes(case, executor, monkeypatch)
    with pytest.raises(RuntimeError, match="session failed"):
        _execute(executor, case)

    assert not tuple(manager.leases_root.glob("*.json"))
    assert not tuple(manager.resources_root.iterdir())
    assert executor.device_guard.deadlines == [  # type: ignore[attr-defined]
        ("pre", 200.0),
        ("post", 200.0),
    ]


def test_execute_failure_attaches_finalized_stderr_artifact_after_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    manager = _manager(case)
    raw = b"INITIATING-TP-RANK-ERROR\n"
    artifact_path = tmp_path / ("runtime." + "b" * 32 + ".stderr")
    receipt = OCIStderrArtifactReceipt(
        STDERR_ARTIFACT_SCHEMA,
        "validator-a",
        "runtime",
        artifact_path,
        artifact_path.with_name(artifact_path.name + ".json"),
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        len(raw),
        False,
        os.geteuid(),
        os.getegid(),
        0o600,
    )
    diagnostic = OCIAttachedDiagnostic(
        raw,
        False,
        True,
        client_returncode=1,
        stream_bytes=len(raw),
        stream_sha256=hashlib.sha256(raw).hexdigest(),
        artifact=receipt,
    )

    class DiagnosticTransport:
        def __init__(self, *_args, **_kwargs) -> None:
            self.aborted = False

        def abort(self) -> None:
            self.aborted = True

        def stderr_diagnostic(self) -> OCIAttachedDiagnostic:
            assert self.aborted
            return diagnostic

    failure = OuterSessionProtocolError("worker protocol failed")

    def fail_session(*_args, **_kwargs) -> SessionExecutionEvidence:
        raise failure

    executor = _executor_with(case, manager, fail_session)
    _install_execution_fakes(case, executor, monkeypatch)
    monkeypatch.setattr(backend, "AttachedSessionTransport", DiagnosticTransport)
    with pytest.raises(OuterSessionProtocolError) as raised:
        _execute(executor, case)
    assert raised.value is failure
    assert raised.value.diagnostic == diagnostic
    assert receipt.receipt_sha256 in str(raised.value)


def test_execute_reference_selects_reference_transport_and_binds_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon.eval.oci_reference_session import (
        ReferenceExchangeEvidence,
        ReferenceSessionEvidence,
    )
    from cacheon.eval.reference_protocol import encode_reference_evidence, request_sha256
    from cacheon.stack_identity import sha256_hex
    from tests.test_oci_reference_session import _plan as reference_plan, _raw

    case = _case(tmp_path)
    manager = _manager(case)
    plan = reference_plan()
    launch = SimpleNamespace(digest=plan.reference.pristine_launch_digest)
    exchanges = []
    for index, request in enumerate(plan.requests):
        evidence = _raw(request)
        exchanges.append(ReferenceExchangeEvidence(
            index, request, request_sha256(request),
            sha256_hex(encode_reference_evidence(evidence, request)),
            2.0 + index * 2, 3.0 + index * 2, evidence,
        ))
    session = ReferenceSessionEvidence(
        "cacheon.pristine-reference-session.v1",
        plan.requests[0].session_id,
        launch.digest,
        plan.reference.digest,
        plan.digest,
        plan.request_plan_digest,
        plan.expected_preflight,
        1.0,
        tuple(exchanges),
        7.0,
    )
    executor = OCIEngineExecutor(case.config, case.device_policy, manager=manager)
    monkeypatch.setattr(
        executor,
        "_validate_reference_launch",
        lambda *_args: (
            case.resolved,
            case.preflight,
            case.identity,
            plan.expected_preflight,
            case.model,
        ),
    )
    observed = []

    def execute_runtime(*_args, **kwargs):
        observed.append(kwargs["session_protocol"])
        value = kwargs["run"](
            SimpleNamespace(), kwargs["absolute"] - 2.0, "runtime-" + "1" * 32
        )
        return backend._RawRuntimeExecution(
            "runtime-" + "1" * 32,
            SimpleNamespace(),
            _digest("publication"),
            _digest("argv"),
            _idle_receipt(case, "runtime-" + "1" * 32, "pre", 1, 1.0),
            _idle_receipt(case, "runtime-" + "1" * 32, "post", 3, 8.0),
            value,
        )

    monkeypatch.setattr(executor, "_execute_runtime", execute_runtime)
    monkeypatch.setattr(backend, "AttachedReferenceTransport", SimpleNamespace)
    executor.reference_session_runner = lambda _plan, **_kwargs: session
    result = executor.execute_reference(
        launch,
        SimpleNamespace(),
        SimpleNamespace(digest=_digest("model-mount")),
        plan,
        deadline=200.0,
    )
    assert observed == ["reference"]
    assert result.session == session
    assert tuple(row.phase for row in result.device_receipts) == ("pre", "post")


def test_executors_sharing_manager_share_transaction_lock(tmp_path: Path) -> None:
    case = _case(tmp_path)
    manager = _manager(case)
    first = OCIEngineExecutor(case.config, case.device_policy, manager=manager)
    second = OCIEngineExecutor(case.config, case.device_policy, manager=manager)
    assert first._lock is second._lock is manager.transaction_lock
    with first.exclusive_transaction():
        first.prove_quiescent()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            blocked = pool.submit(second.prove_quiescent)
            with pytest.raises(OCIBackendError, match="active session"):
                blocked.result()


def test_trusted_backend_has_no_economic_fields_or_runtime_authority() -> None:
    names = {field.name for field in fields(EngineExecutionEvidence)}
    for forbidden in ("pass", "score", "crown", "miner", "retry", "arm", "role"):
        assert all(forbidden not in name for name in names)

    source = Path(backend.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for forbidden in ("torch", "sglang", "bittensor", "cacheon.chain", "quality"):
        assert all(forbidden not in name for name in imported)

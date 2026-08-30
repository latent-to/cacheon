from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.engine_tree import (
    materialize_engine_tree,
)
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    LogicalHardwareSpec,
    NativeBuildSpec,
    PhysicalHardwareBinding,
    TrustedLaunchBinding,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
    resolve_engine_launch,
)
from cacheon.eval.oci_prebuild import (
    OCIPrebuildConfig,
    OCIPrebuildError,
    OCIPrebuildPolicy,
    PREBUILD_RECEIPT,
    PREBUILD_SCHEMA,
    _write_compile_profile,
    build_prebuild_argv,
    container_build,
    run_oci_prebuild,
)
from cacheon.eval.native_compile_profile import NativeCuTeCompileProfile
from cacheon.eval.oci_process import (
    CommandResult,
    OCIAttachedDiagnostic,
    OCIProcessManager,
    OCIProcessResult,
)
from cacheon.stack_identity import canonical_json_bytes
from cacheon.stack_manifest import EvaluationStackContext, EvaluationStackManifest
from cacheon.target_catalog import default_target_catalog
from tests.support.preflight import preflight_receipt


DOCKER = "/usr/bin/docker"
IMAGE_ID = "sha256:" + "a" * 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _tree(tmp_path: Path):
    catalog = default_target_catalog()
    snapshot = catalog.snapshot()
    context = EvaluationStackContext(
        runtime_digest=_digest("tree-runtime"),
        base_engine_digest=_digest("tree-base"),
        arena_digest=_digest("tree-arena"),
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in snapshot["targets"]
        },
    )
    stack = EvaluationStackManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        entries={},
    )
    return materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver={},
        destination=tmp_path / "tree",
    )


def _policy(**changes: object) -> OCIPrebuildPolicy:
    values: dict[str, object] = {
        "uid": max(1, os.getuid()),
        "gid": max(1, os.getgid()),
        "cpu_millis": 8_000,
        "memory_bytes": 32 << 30,
        "pids_limit": 4_096,
        "tmpfs_bytes": 512 << 20,
        "stage_bytes": 16 << 30,
        "stage_inodes": 100_000,
        "timeout_seconds": 7_200,
        "native_compile_timeout_seconds": 6_000,
        "container_python": "/usr/local/bin/python3",
        "build_path": (
            "/usr/local/cuda/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ),
        "build_tmpdir": "/tmp",
        "pinned_build_roots": (
            "/usr/include",
            "/usr/lib",
            "/usr/local/cuda",
            "/usr/local/include",
            "/usr/local/lib/python3.12/dist-packages",
        ),
        "runtime_policy_digest": _digest("runtime-policy"),
    }
    values.update(changes)
    return OCIPrebuildPolicy(**values)  # type: ignore[arg-type]


def _hardware() -> LogicalHardwareSpec:
    return LogicalHardwareSpec(
        visible_gpu_count=8,
        architecture="sm120",
        topology_class="pcie_switch",
        topology_digest=_digest("topology"),
        tp_size=8,
        ep_size=1,
        dp_size=1,
        device_policy_digest=_digest("device-policy"),
    )


def _physical() -> PhysicalHardwareBinding:
    return PhysicalHardwareBinding(
        physical_gpu_ids=tuple(f"GPU-{index}" for index in range(8)),
        architecture="sm120",
        topology_class="pcie_switch",
        topology_digest=_digest("topology"),
        tp_size=8,
        ep_size=1,
        dp_size=1,
        device_policy_digest=_digest("device-policy"),
    )


def _preflight(
    *,
    image: str,
    platform: str,
    worker: str,
    policy: OCIPrebuildPolicy,
    sglang_version: str = "0.0.0.dev1",
):
    return preflight_receipt(
        image=image,
        platform=platform,
        worker=worker,
        uid=policy.uid,
        gid=policy.gid,
        sglang_version=sglang_version,
    )


def _case(
    tmp_path: Path,
    *,
    policy: OCIPrebuildPolicy | None = None,
    tree=None,
    sglang_version: str = "0.0.0.dev1",
):
    tree = tree or _tree(tmp_path)
    policy = policy or _policy()
    seccomp = tmp_path / "seccomp.json"
    seccomp.write_text('{"defaultAction":"SCMP_ACT_ERRNO"}\n')
    image = _digest("image")
    platform = _digest("platform")
    worker = _digest("worker")
    native = NativeBuildSpec(
        tree_digest=tree.tree_digest,
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
            dependency_policy_digest=policy.dependency_policy_digest,
            target_architecture="sm120",
        ),
        target_architecture="sm120",
        dependency_policy_digest=policy.dependency_policy_digest,
    )
    launch = EngineLaunchSpec(
        runtime_digest=_digest("runtime"),
        base_engine_digest=_digest("engine"),
        arena_digest=_digest("arena"),
        stack_digest=tree.stack_digest,
        tree_digest=tree.tree_digest,
        image_digest=image,
        platform_digest=platform,
        controller_distribution_digest=_digest("controller"),
        worker_distribution_digest=worker,
        model_revision_digest=_digest("model-revision"),
        model_manifest_digest=_digest("model-manifest"),
        model_content_digest=_digest("model-content"),
        validator_overlay_digest=_digest("validator-overlay"),
        engine_config_digest=_digest("engine-config"),
        seccomp_policy_digest=hashlib.sha256(seccomp.read_bytes()).hexdigest(),
        resource_policy_digest=policy.resource_policy_digest,
        native_build_spec_digest=native.digest,
        hardware=_hardware(),
    )
    preflight = _preflight(
        image=image,
        platform=platform,
        worker=worker,
        policy=policy,
        sglang_version=sglang_version,
    )
    binding = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=launch.controller_distribution_digest,
        native_build_spec=native,
        runtime_preflight_receipt=preflight,
        physical_hardware=_physical(),
    )
    config = OCIPrebuildConfig(
        docker_binary=DOCKER,
        recovery_root=(tmp_path / "recovery").absolute(),
        publication_root=(tmp_path / "publications").absolute(),
        seccomp_profile=seccomp.absolute(),
        executor_id="validator-a",
        policy=policy,
    )
    return tree, launch, binding, preflight, config


def _write_receipt(
    stage: Path,
    *,
    launch: EngineLaunchSpec,
    native: NativeBuildSpec,
) -> None:
    entries = sorted((*[path.name for path in stage.iterdir()], PREBUILD_RECEIPT))
    row = {
        "build_spec_digest": native.digest,
        "rebuild_applied": False,
        "schema": PREBUILD_SCHEMA,
        "stage_entries": entries,
        "target_architecture": native.target_architecture,
        "tree_digest": launch.tree_digest,
    }
    (stage / PREBUILD_RECEIPT).write_bytes(canonical_json_bytes(row) + b"\n")


def _native_with_dependency(native: NativeBuildSpec, dependency: str) -> NativeBuildSpec:
    return replace(
        native,
        dependency_policy_digest=dependency,
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=native.image_digest,
            worker_distribution_digest=native.worker_distribution_digest,
            dependency_policy_digest=dependency,
            target_architecture=native.target_architecture,
        ),
    )


class _Controls:
    def __call__(self, argv, *, timeout_s, max_output_bytes):
        row = tuple(argv)
        if row[1:3] == ("container", "ls"):
            return CommandResult(0, b"", b"")
        return CommandResult(0, b"", b"")


def _manager(config, **over):
    return OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
        **over,
    )


def _prebuild_argv(lease, resolved, preflight, config, **over):
    return build_prebuild_argv(
        lease=lease,
        resolved=resolved,
        preflight=preflight,
        config=config,
        stage_path=lease.mount_paths[0],
        seccomp_path=lease.stage_paths[0],
        **over,
    )


def _compile_profile() -> NativeCuTeCompileProfile:
    return NativeCuTeCompileProfile(
        logical_architecture="sm103",
        compiler_architecture="sm_103a",
        image_digest=_digest("profile-image"),
        platform_digest=_digest("profile-platform"),
        worker_distribution_digest=_digest("profile-worker"),
        logical_hardware_digest=_digest("profile-hardware"),
        device_policy_digest=_digest("profile-device"),
        topology_digest=_digest("profile-topology"),
        visible_gpu_count=8,
        tp_size=4,
        ep_size=1,
        dp_size=2,
        constants={"max_active_clusters.cluster_size_1": 148},
        measurement_digest=_digest("profile-measurement"),
    )


def test_compile_profile_staging_overrides_restrictive_controller_umask(
    tmp_path: Path,
) -> None:
    profile = _compile_profile()
    destination = tmp_path / "compile-profile.json"
    previous = os.umask(0o077)
    try:
        _write_compile_profile(profile, destination)
    finally:
        os.umask(previous)
    assert destination.stat().st_mode & 0o777 == 0o444
    assert destination.read_bytes() == profile.canonical_bytes


def test_policy_binds_resource_and_native_dependency_inputs(tmp_path: Path) -> None:
    policy = _policy()
    resource_changes = {
        "uid": policy.uid + 1,
        "cpu_millis": 9_000,
        "memory_bytes": 33 << 30,
        "stage_bytes": 17 << 30,
        "timeout_seconds": 7_201,
        "container_python": "/usr/bin/python3",
        "runtime_policy_digest": _digest("other runtime"),
    }
    for field, value in resource_changes.items():
        assert (
            replace(policy, **{field: value}).resource_policy_digest
            != policy.resource_policy_digest
        )
    assert (
        replace(
            policy,
            cpuset_cpus="0-3,8-11",
            cpuset_mems="0",
        ).resource_policy_digest
        != policy.resource_policy_digest
    )
    dependency_changes = {
        "build_path": ("/usr/bin", "/bin"),
        "build_tmpdir": "/var/tmp",
        "container_python": "/usr/bin/python3",
        "native_compile_timeout_seconds": 5_999,
        "pinned_build_roots": ("/usr/include", "/usr/lib"),
    }
    for field, value in dependency_changes.items():
        assert (
            replace(policy, **{field: value}).dependency_policy_digest
            != policy.dependency_policy_digest
        )

    _tree_row, _launch, binding, _preflight, _config_row = _case(tmp_path)
    changed_policy = replace(policy, container_python="/usr/bin/python3")
    changed_native = _native_with_dependency(
        binding.native_build_spec, changed_policy.dependency_policy_digest
    )
    assert changed_native.digest != binding.native_build_spec.digest


def test_prebuild_cpuset_policy_rejects_partial_or_noncanonical_sets() -> None:
    policy = _policy()
    for cpus, mems in (
        ("0-7", None),
        (None, "0"),
        ("0,1", "0"),
        ("0-7", "00"),
        ("0-6", "0"),
    ):
        with pytest.raises(OCIPrebuildError, match="cpuset|cpu_millis"):
            replace(policy, cpuset_cpus=cpus, cpuset_mems=mems)


def test_exact_prebuild_argv_has_only_two_mounts_no_gpu_no_egress_no_caps(
    tmp_path: Path,
) -> None:
    isolated = _policy(cpuset_cpus="0-3,8-11", cpuset_mems="0")
    tree, launch, binding, preflight, config = _case(tmp_path, policy=isolated)
    resolved = resolve_engine_launch(launch, binding)
    manager = _manager(config)
    lease = manager.register(
        lease_id="prebuild-test",
        container_name="cacheon-prebuild-test",
        mount_relpaths=("stage",),
        stage_relpaths=("seccomp.json",),
    )
    argv = _prebuild_argv(lease, resolved, preflight, config)

    assert argv[: len(lease.run_prefix(DOCKER))] == lease.run_prefix(DOCKER)
    assert "--network=none" in argv and "--read-only" in argv
    assert "--ipc=none" in argv and not any(value.startswith("--pid=") for value in argv)
    assert argv.count("--cap-drop=ALL") == 1
    assert not any(value.startswith("--cap-add") for value in argv)
    assert "--security-opt=no-new-privileges=true" in argv
    assert f"--security-opt=seccomp={lease.stage_paths[0]}" in argv
    assert "--cpuset-cpus=0-3,8-11" in argv
    assert "--cpuset-mems=0" in argv
    mounts = [value for value in argv if value.startswith("--mount=")]
    assert len(mounts) == 2
    assert str(tree.root) in mounts[0] and "readonly" in mounts[0]
    assert str(lease.mount_paths[0]) in mounts[1] and "readonly" not in mounts[1]
    assert not any(
        "/models" in value or "/root" in value or "docker.sock" in value
        for value in argv
    )
    assert not any("--gpus" in value or "--device" in value for value in argv)
    env_rows = [value for value in argv if value.startswith("--env=")]
    assert any(
        "CACHEON_NATIVE_BUILD_SPEC_DIGEST=" + binding.native_build_spec.digest in value
        for value in env_rows
    )
    assert any("CACHEON_BUILD_PATH=" in value for value in env_rows)
    assert any("CACHEON_BUILD_TMPDIR=/tmp" in value for value in env_rows)
    assert any("CACHEON_NATIVE_COMPILE_TIMEOUT_S=6000" in value for value in env_rows)
    env_keys = {value.split("=", 2)[1] for value in env_rows}
    assert not any(
        key.upper().endswith("PROXY") or key.startswith("LD_") or key == "PYTHONPATH"
        for key in env_keys
    )
    assert argv[-5:] == (
        IMAGE_ID,
        "-I",
        "-m",
        "cacheon.eval.oci_prebuild",
        "--container-build",
    )


def test_profiled_prebuild_adds_one_read_only_profile_mount_and_digest_env(
    tmp_path: Path,
) -> None:
    tree, launch, binding, preflight, config = _case(tmp_path)
    resolved = resolve_engine_launch(launch, binding)
    profile_digest = _digest("cute-profile")
    profiled = replace(
        resolved,
        native_compile_profile=SimpleNamespace(digest=profile_digest),
    )
    manager = _manager(config)
    lease = manager.register(
        lease_id="profiled-prebuild-test",
        container_name="cacheon-profiled-prebuild-test",
        mount_relpaths=("stage",),
        stage_relpaths=("seccomp.json", "cute-profile.json"),
    )
    argv = _prebuild_argv(
        lease, profiled, preflight, config, compile_profile_path=lease.stage_paths[1]
    )
    profile_mounts = [
        value
        for value in argv
        if value.startswith("--mount=") and "cute-compile-profile.json" in value
    ]
    assert len(profile_mounts) == 1
    assert "readonly" in profile_mounts[0]
    assert f"--env=CACHEON_CUTE_COMPILE_PROFILE_DIGEST={profile_digest}" in argv
    assert "--env=CACHEON_CUTE_COMPILE_PROFILE=/cacheon/cute-compile-profile.json" in argv

    with pytest.raises(OCIPrebuildError, match="does not match launch authority"):
        _prebuild_argv(lease, profiled, preflight, config)


@pytest.mark.skipif(sys.platform != "linux", reason="production publication uses Linux renameat2")
def test_run_builds_publishes_reopens_and_then_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = _manager(config)
    stage_holder: list[Path] = []

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        stage_holder.append(Path(path))
        return Path(path)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        assert timeout_s == config.policy.timeout_seconds and stdin_bytes == b""
        _write_receipt(stage_holder[-1], launch=launch, native=binding.native_build_spec)
        return OCIProcessResult(0, 1.25)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    first = run_oci_prebuild(launch, binding, config, manager=manager)
    assert first.container_elapsed_seconds == 1.25
    assert first.publication.root.is_dir()
    assert first.publication.build_spec_digest == binding.native_build_spec.digest
    assert first.publication.root.stat().st_mode & 0o777 == 0o555
    assert not manager.leases_root.joinpath("prebuild-test.json").exists()

    second = run_oci_prebuild(launch, binding, config, manager=manager)
    assert second.reused and second.container_elapsed_seconds is None
    assert second.publication.publication_digest == first.publication.publication_digest


@pytest.mark.parametrize("deadline", (float("nan"), float("inf"), -float("inf"), True, "10"))
def test_prebuild_rejects_nonfinite_or_non_numeric_deadline_before_work(
    tmp_path: Path, deadline
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    with pytest.raises(OCIPrebuildError, match="deadline must be a finite"):
        run_oci_prebuild(launch, binding, config, deadline=deadline)
    assert not config.recovery_root.exists()


def test_prebuild_rejects_expired_deadline_before_binding_or_lease(
    tmp_path: Path,
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = _manager(config, clock=lambda: 10.0)
    with pytest.raises(OCIPrebuildError, match="deadline expired during binding"):
        run_oci_prebuild(launch, binding, config, manager=manager, deadline=10.0)
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_rechecks_deadline_after_binding_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cacheon.eval.oci_prebuild as prebuild_mod

    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    now = {"value": 100.0}
    manager = _manager(config, clock=lambda: now["value"])
    real_validate = prebuild_mod._validate_binding

    def validate(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        now["value"] = 106.0
        return result

    monkeypatch.setattr(prebuild_mod, "_validate_binding", validate)
    with pytest.raises(OCIPrebuildError, match="deadline expired during binding"):
        run_oci_prebuild(launch, binding, config, manager=manager, deadline=105.0)
    assert list(manager.leases_root.iterdir()) == []


@pytest.mark.parametrize(
    "deadline,expected_timeout",
    ((105.0, 5.0), (10_000.0, 7_200.0)),
)
def test_prebuild_container_timeout_is_capped_by_absolute_deadline_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deadline: float,
    expected_timeout: float,
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = _manager(config, clock=lambda: 100.0)
    observed = []

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        return Path(path)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        observed.append((timeout_s, stdin_bytes))
        return OCIProcessResult(9, 0.1)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    with pytest.raises(OCIPrebuildError, match="container exited 9"):
        run_oci_prebuild(
            launch, binding, config, manager=manager, deadline=deadline
        )
    assert observed == [(expected_timeout, b"")]
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_failure_preserves_only_bounded_terminal_safe_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = _manager(config)

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        return Path(path)

    tail = b"compile failed: \x1b[31mTRACEBACK-TAIL\x1b[0m"
    diagnostic = OCIAttachedDiagnostic(tail, True, True, client_returncode=9)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        assert timeout_s == config.policy.timeout_seconds and stdin_bytes == b""
        return OCIProcessResult(9, 0.1, diagnostic)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    with pytest.raises(OCIPrebuildError) as caught:
        run_oci_prebuild(launch, binding, config, manager=manager)
    rendered = str(caught.value)
    assert "container exited 9" in rendered
    assert diagnostic.stderr_sha256 in rendered
    assert "TRACEBACK-TAIL" in rendered
    assert "\x1b" not in rendered and "\\x1b" in rendered
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_expiry_after_container_prevents_publication_and_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = _manager(config)
    now = {"value": 100.0}
    monkeypatch.setattr(manager, "clock", lambda: now["value"])
    stage_holder: list[Path] = []

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        stage_holder.append(Path(path))
        return Path(path)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        assert timeout_s == 5.0 and stdin_bytes == b""
        _write_receipt(
            stage_holder[-1], launch=launch, native=binding.native_build_spec
        )
        now["value"] = 106.0
        return OCIProcessResult(0, 1.0)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    with pytest.raises(OCIPrebuildError, match="deadline expired during container"):
        run_oci_prebuild(
            launch, binding, config, manager=manager, deadline=105.0
        )
    assert not (config.publication_root / binding.native_build_spec.digest[:2]).exists()
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_deadline_fails_closed_on_nonfinite_manager_clock(
    tmp_path: Path,
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = _manager(config, clock=lambda: float("nan"))
    with pytest.raises(OCIPrebuildError, match="clock returned a non-finite"):
        run_oci_prebuild(launch, binding, config, manager=manager, deadline=105.0)


@pytest.mark.parametrize(
    "mutator,match",
    (
        (lambda launch, binding, config: (replace(launch, resource_policy_digest=_digest("bad")), binding, config), "resource policy"),
        (lambda launch, binding, config: (launch, replace(binding, native_build_spec=_native_with_dependency(binding.native_build_spec, _digest("bad"))), config), "native_build_spec_digest"),
        (lambda launch, binding, config: (launch, binding, replace(config, docker_binary="/opt/docker")), "Docker clients differ"),
    ),
)
def test_binding_mismatch_rejects_before_lease(
    tmp_path: Path, mutator, match: str
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    launch, binding, config = mutator(launch, binding, config)
    with pytest.raises((OCIPrebuildError, ValueError), match=match):
        run_oci_prebuild(launch, binding, config)
    assert not config.recovery_root.exists()


def test_publication_and_recovery_roots_must_not_overlap_materialized_tree_or_each_other(
    tmp_path: Path,
) -> None:
    tree, launch, binding, _preflight, config = _case(tmp_path)
    with pytest.raises(OCIPrebuildError, match="must not overlap"):
        run_oci_prebuild(
            launch,
            binding,
            replace(config, publication_root=tree.root / "published"),
        )
    with pytest.raises(OCIPrebuildError, match="must not overlap"):
        run_oci_prebuild(
            launch,
            binding,
            replace(config, publication_root=config.recovery_root / "published"),
        )


def test_container_build_scrubs_ambient_environment_and_applies_build_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cacheon.eval.oci_prebuild as prebuild_mod
    import cacheon.rebuild as rebuild_mod

    tree = _tree(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(prebuild_mod, "CONTAINER_TREE", str(tree.root))
    monkeypatch.setattr(prebuild_mod, "CONTAINER_STAGE", str(stage))
    seen = []
    monkeypatch.setattr(
        rebuild_mod,
        "apply_rebuild_plan",
        lambda path, *, phase: seen.append((Path(path), phase)) or False,
    )
    original_environment = dict(os.environ)
    required = {
        "CACHEON_NATIVE_BUILD_SPEC_DIGEST": _digest("build"),
        "CACHEON_ENGINE_TREE_DIGEST": tree.tree_digest,
        "CACHEON_TARGET_GPU_ARCH": "sm120",
        "CACHEON_NATIVE_ARTIFACT_STAGE": str(stage),
        "CACHEON_PINNED_BUILD_ROOTS": "/usr/include:/usr/lib",
        "CACHEON_BUILD_PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "CACHEON_BUILD_TMPDIR": "/tmp",
        "CACHEON_NATIVE_COMPILE_TIMEOUT_S": "60",
        "CACHEON_REBUILD_CONTAINER": "1",
        "HTTPS_PROXY": "https://must-not-survive.invalid",
        "LD_PRELOAD": "/tmp/evil.so",
        "PYTHONPATH": "/tmp/evil",
    }
    try:
        os.environ.update(required)
        receipt = container_build()
        assert receipt == stage / PREBUILD_RECEIPT
        assert seen == [(tree.root.resolve(), "build")]
        for forbidden in ("HTTPS_PROXY", "LD_PRELOAD", "PYTHONPATH"):
            assert forbidden not in os.environ
        assert os.environ["CACHEON_REBUILD_PHASE"] == "build"
        ordinary = json.loads(receipt.read_text())
        assert ordinary == {
            "build_spec_digest": required["CACHEON_NATIVE_BUILD_SPEC_DIGEST"],
            "rebuild_applied": False,
            "schema": PREBUILD_SCHEMA,
            "stage_entries": [PREBUILD_RECEIPT],
            "target_architecture": "sm120",
            "tree_digest": tree.tree_digest,
        }
        assert receipt.read_bytes() == canonical_json_bytes(ordinary) + b"\n"
    finally:
        os.environ.clear()
        os.environ.update(original_environment)


def test_seccomp_bytes_and_existing_publication_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    config.seccomp_profile.write_text("tampered\n")
    with pytest.raises(OCIPrebuildError, match="seccomp"):
        run_oci_prebuild(launch, binding, config)

    # A destination occupying the canonical address is validated, never repaired.
    config.seccomp_profile.write_text('{"defaultAction":"SCMP_ACT_ERRNO"}\n')
    digest = binding.native_build_spec.digest
    destination = config.publication_root / digest[:2] / digest
    destination.mkdir(parents=True)
    (destination / "garbage").write_text("x")
    with pytest.raises(Exception, match="native artifact|mode|manifest"):
        run_oci_prebuild(launch, binding, config)

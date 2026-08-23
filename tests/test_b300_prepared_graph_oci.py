"""Contracts for the fixed, model-free B300 TP4 graph OCI lifetime."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval import b300_prepared_graph_oci as graph_oci
from cacheon.eval.b300_prepared_graph_probe import PreparedGraphProbeRequest
from cacheon.eval.device_state import DeviceStatePolicy, DeviceStatePolicyError
from cacheon.eval.engine_launch import (
    LogicalHardwareSpec,
    NativeBuildSpec,
    PhysicalHardwareBinding,
    TrustedLaunchBinding,
    native_compiler_policy_digest,
)
from cacheon.eval.marginal_runtime import MaterializedArmBinding, PreparedCandidateRuntime
from cacheon.eval.oci_backend import expected_runtime_preflight
from cacheon.eval.oci_process import CommandResult, OCIProcessManager
from tests.test_b300_prepared_graph_probe import _request
from tests.test_b300_qualification_graph_evidence_store import _artifact
from tests.test_b300_qualification_graph_provider import _profile
from tests.test_marginal_runtime import FUSED, SILU
from tests.test_oci_backend import _case as _backend_case, _gpu


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _gpus():
    return tuple(
        replace(
            _gpu(),
            physical_id=index,
            uuid=f"GPU-0000000{index}-0000-0000-0000-00000000000{index}",
            pci_bus_id=f"00000000:{index + 1:02x}:00.0",
        )
        for index in range(4)
    )


def _tp4(profile, backend, policy: DeviceStatePolicy):
    tree = profile.prepared.binding.tree
    topology = _h("b300-tp4-topology")
    hardware = LogicalHardwareSpec(
        4, "sm120", "nvlink4", topology, 4, 1, 1, policy.policy_sha256
    )
    physical = PhysicalHardwareBinding(
        ("0", "1", "2", "3"), "sm120", "nvlink4", topology, 4, 1, 1,
        policy.policy_sha256,
    )
    original = backend.native
    native = NativeBuildSpec(
        tree_digest=tree.tree_digest,
        image_digest=original.image_digest,
        platform_digest=original.platform_digest,
        worker_distribution_digest=original.worker_distribution_digest,
        toolchain_digest=original.toolchain_digest,
        patcher_digest=original.patcher_digest,
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=original.image_digest,
            worker_distribution_digest=original.worker_distribution_digest,
            dependency_policy_digest=original.dependency_policy_digest,
            target_architecture="sm120",
        ),
        target_architecture="sm120",
        dependency_policy_digest=original.dependency_policy_digest,
    )
    engine = replace(profile.prepared.session_plan.engine_config, tp_size=4)
    launch = replace(
        backend.launch,
        stack_digest=tree.stack_digest,
        tree_digest=tree.tree_digest,
        engine_config_digest=engine.digest,
        native_build_spec_digest=native.digest,
        hardware=hardware,
    )
    trusted = TrustedLaunchBinding(
        tree.root,
        launch.controller_distribution_digest,
        native,
        backend.preflight,
        physical,
    )
    session = replace(
        profile.prepared.session_plan,
        launch_digest=launch.digest,
        expected_engine_config_digest=engine.digest,
        engine_config=engine,
        expected_preflight=expected_runtime_preflight(launch, backend.preflight),
    )
    prepared = PreparedCandidateRuntime(
        profile.prepared.arm, MaterializedArmBinding(tree, trusted), launch, session
    )
    bound = SimpleNamespace(candidate=profile.candidate, prepared=prepared)
    return bound, _request(bound)


@pytest.fixture
def commissioned(tmp_path: Path):
    backend = _backend_case(tmp_path / "backend")
    policy = replace(backend.device_policy, expected_gpus=_gpus())
    rows = []
    for label, source in (("singleton", SILU), ("atomic", FUSED)):
        copied = tmp_path / f"{label}-source"
        shutil.copytree(source, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        rows.append(_tp4(_profile(tmp_path / label, copied, label), backend, policy))
    publication = tmp_path / "native-publication"
    publication.mkdir(mode=0o700)
    source = tmp_path / "controller-source"
    module = source / "cacheon" / "eval" / "b300_prepared_graph_oci.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(Path(graph_oci.__file__).read_bytes())
    initializer = module.parents[1] / "__init__.py"
    initializer.write_bytes(b"")
    for file in (initializer, module):
        file.chmod(0o444)
    for directory in (module.parent, module.parent.parent, source):
        directory.chmod(0o555)
    return backend, policy, tuple(rows), publication, source


class _Guard:
    def __init__(self):
        self.events = []

    # Capture quiesces the device around a graph capture (F-09); a cold lane
    # or a quiescent retained pair both satisfy it.
    def before_capture(self, launch_id, *, deadline):
        self.events.append(("pre", launch_id, deadline))

    def after_capture(self, launch_id, *, deadline):
        self.events.append(("post", launch_id, deadline))


def _control(_argv, *, timeout_s, max_output_bytes):
    assert timeout_s > 0 and max_output_bytes == 4096
    return CommandResult(0, b"", b"")


def _executor(monkeypatch, backend, policy, publication, source, capture, *, digest=None):
    manager = OCIProcessManager(
        docker_binary=backend.config.prebuild.docker_binary,
        recovery_root=backend.config.prebuild.recovery_root / _h(str(source))[:12],
        executor_id=backend.config.prebuild.executor_id,
        runner=_control,
    )
    monkeypatch.setattr(
        manager,
        "mount_tmpfs",
        lambda _lease, path, **_kwargs: (path.mkdir(mode=0o700), path)[1],
    )
    published = SimpleNamespace(
        root=publication,
        build_spec_digest=None,
        publication_digest=_h("published-native"),
    )

    def prebuild(launch, *_args, **_kwargs):
        published.build_spec_digest = launch.native_build_spec_digest
        return SimpleNamespace(
            launch_digest=launch.digest,
            publication=published,
        )

    monkeypatch.setattr(graph_oci, "run_oci_prebuild", prebuild)
    monkeypatch.setattr(graph_oci, "reopen_native_artifact", lambda *_a, **_k: published)
    executor = graph_oci.B300PreparedGraphOCIExecutor(
        backend.config,
        policy,
        controller_source_root=source,
        controller_distribution_digest=digest or backend.launch.controller_distribution_digest,
        manager=manager,
        capture_runner=capture,
    )
    executor.device_guard = _Guard()
    return executor


def test_fixed_argv_round_trips_two_registered_targets_without_model(
    commissioned, monkeypatch
):
    backend, policy, rows, publication, source = commissioned
    artifacts = [_artifact(request.binding, request.policy.verification_policy_digest) for _, request in rows]
    calls = []

    def capture(argv, **limits):
        artifact = artifacts[len(calls)]
        calls.append((argv, limits))
        return CommandResult(0, artifact.canonical_bytes, b"candidate diagnostic")

    executor = _executor(monkeypatch, backend, policy, publication, source, capture)
    observed = [
        executor.execute(request, row.prepared, deadline=time.monotonic() + 30)
        for row, request in rows
    ]

    assert observed == artifacts
    assert len({request.binding.target_id for _, request in rows}) == 2
    for argv, limits in calls:
        assert "--network=none" in argv and "--read-only" in argv
        assert "--cap-drop=ALL" in argv
        assert "--security-opt=no-new-privileges=true" in argv
        assert '--gpus="device=0,1,2,3"' in argv
        assert argv[-6:] == (
            f"--entrypoint={backend.runtime.container_python}",
            backend.preflight.local_image_id,
            "-I", "-c", graph_oci._CONTROLLER_BOOTSTRAP, "worker",
        )
        mounts = tuple(value for value in argv if value.startswith("--mount="))
        assert len(mounts) == 5 and not any("/cacheon/input/model" in value for value in mounts)
        assert any(f"src={source},dst=/cacheon/controller" in value and value.endswith(",readonly") for value in mounts)
        assert not any(rows[0][1].binding.target_id in value for value in argv)
        assert set(limits) == {"timeout_s", "max_stdout_bytes", "max_stderr_bytes"}
        assert 0 < limits["timeout_s"] < 30
        assert limits["max_stdout_bytes"] == 8 << 20
        assert limits["max_stderr_bytes"] == 64 << 10
    assert [event[0] for event in executor.device_guard.events] == ["pre", "post"] * 2
    assert executor.manager.prove_quiescent().container_ids == ()


@pytest.mark.parametrize("outcome", ("timeout", "nonzero", "oversize", "foreign"))
def test_failures_hold_and_prove_cleanup(commissioned, monkeypatch, outcome):
    backend, policy, rows, publication, source = commissioned
    row, request = rows[0]
    foreign = _artifact(rows[1][1].binding, request.policy.verification_policy_digest)

    def capture(argv, **_limits):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        if outcome == "nonzero":
            return CommandResult(7, b"", b"discarded:" + b"x" * 4096 + b"\xff\n")
        if outcome == "oversize":
            return CommandResult(0, b"x" * ((8 << 20) + 1), b"")
        return CommandResult(0, foreign.canonical_bytes, b"")

    executor = _executor(monkeypatch, backend, policy, publication, source, capture)
    with pytest.raises(graph_oci.B300PreparedGraphOCIError) as raised:
        executor.execute(request, row.prepared, deadline=time.monotonic() + 30)
    assert raised.value.decision == "HOLD"
    if outcome == "nonzero":
        assert "discarded:" not in str(raised.value)
        assert r"\\xff\n" in str(raised.value)
    assert [event[0] for event in executor.device_guard.events] == ["pre", "post"]
    assert executor.manager.prove_quiescent().lease_records == ()


def test_tp4_and_device_policy_mismatch_reject_before_container(commissioned, monkeypatch):
    backend, policy, rows, publication, source = commissioned
    row, request = rows[0]
    changed = replace(
        policy.expected_gpus[0],
        uuid="GPU-ffffffff-0000-0000-0000-000000000000",
    )
    mismatched = replace(
        policy, expected_gpus=(changed, *policy.expected_gpus[1:])
    )
    mismatch = _executor(
        monkeypatch, backend, mismatched, publication, source,
        lambda *_a, **_k: pytest.fail("device mismatch reached the container"),
    )
    with pytest.raises(graph_oci.B300PreparedGraphOCIError) as raised:
        mismatch.execute(request, row.prepared, deadline=time.monotonic() + 30)
    assert raised.value.decision == "HOLD"
    assert isinstance(raised.value.__cause__, DeviceStatePolicyError)
    assert "policy digest" in str(raised.value.__cause__)
    original = _profile(Path(row.prepared.binding.tree.root).parent / "tp1", SILU, "tp1")
    tp1_request = _request(original)
    manager = OCIProcessManager(
        docker_binary=backend.config.prebuild.docker_binary,
        recovery_root=backend.config.prebuild.recovery_root / "tp1",
        executor_id=backend.config.prebuild.executor_id,
        runner=_control,
    )
    executor = graph_oci.B300PreparedGraphOCIExecutor(
        backend.config,
        policy,
        controller_source_root=source,
        controller_distribution_digest=backend.launch.controller_distribution_digest,
        manager=manager,
        capture_runner=lambda *_a, **_k: None,
    )
    with pytest.raises(graph_oci.B300PreparedGraphOCIError, match="TP4"):
        executor.execute(tp1_request, original.prepared, deadline=time.monotonic() + 30)


def test_controller_source_authority_is_fixed_and_reopened(commissioned, monkeypatch):
    backend, policy, rows, publication, source = commissioned
    row, request = rows[0]
    unreachable = lambda *_a, **_k: pytest.fail("invalid source reached container")
    foreign = _executor(
        monkeypatch, backend, policy, publication, source, unreachable,
        digest=_h("foreign-controller"),
    )
    with pytest.raises(graph_oci.B300PreparedGraphOCIError, match="executor authority"):
        foreign.execute(request, row.prepared, deadline=time.monotonic() + 30)
    foreign.manager.close()

    linked = publication.parent / "linked-controller"
    linked.symlink_to(source, target_is_directory=True)
    writable = publication.parent / "writable-controller"
    shutil.copytree(source, writable)
    writable.chmod(0o755)
    missing_module = publication.parent / "controller-without-module"
    shutil.copytree(source, missing_module)
    missing_module_eval = missing_module / "cacheon" / "eval"
    missing_module_eval.chmod(0o755)
    (missing_module_eval / "b300_prepared_graph_oci.py").unlink()
    missing_module_eval.chmod(0o555)
    missing_initializer = publication.parent / "controller-without-init"
    shutil.copytree(source, missing_initializer)
    missing_initializer_package = missing_initializer / "cacheon"
    missing_initializer_package.chmod(0o755)
    (missing_initializer_package / "__init__.py").unlink()
    missing_initializer_package.chmod(0o555)
    unreadable = publication.parent / "unreadable-controller"
    shutil.copytree(source, unreadable)
    unreadable.chmod(0o500)
    missing = publication.parent / "missing-controller"
    invalid_sources = (
        linked, writable, missing_module, missing_initializer, unreadable, missing,
        source / "cacheon" / "..",
    )
    for invalid in invalid_sources:
        with pytest.raises(graph_oci.B300PreparedGraphOCIError, match="controller source"):
            _executor(monkeypatch, backend, policy, publication, invalid, unreachable)

    reopened = _executor(monkeypatch, backend, policy, publication, source, unreachable)
    module = source / "cacheon" / "eval" / "b300_prepared_graph_oci.py"
    module.chmod(0o644)
    with pytest.raises(graph_oci.B300PreparedGraphOCIError, match="immutable"):
        reopened.execute(request, row.prepared, deadline=time.monotonic() + 30)


def test_fixed_worker_emits_only_reopened_canonical_artifact(
    commissioned, monkeypatch, capfd, tmp_path
):
    _, _, rows, _, _ = commissioned
    row, request = rows[0]
    artifact = _artifact(request.binding, request.policy.verification_policy_digest)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(request.canonical_bytes)
    monkeypatch.setattr(graph_oci, "CONTAINER_REQUEST", str(request_path))
    monkeypatch.setattr(graph_oci, "CONTAINER_TREE", str(row.prepared.binding.tree.root))

    import cacheon.eval.b300_prepared_graph_probe as probe
    prior_sys_path = list(sys.path)

    def execute(reopened: PreparedGraphProbeRequest, root):
        assert reopened == request and Path(root) == row.prepared.binding.tree.root
        assert sys.path[0] == str(row.prepared.binding.tree.root)
        print("bounded candidate diagnostic")
        return artifact

    monkeypatch.setattr(probe, "execute_prepared_graph_probe", execute)
    assert graph_oci.main(["worker"]) == 0
    assert sys.path == prior_sys_path
    stdout, stderr = capfd.readouterr()
    assert stdout.encode() == artifact.canonical_bytes
    assert "bounded candidate diagnostic" in stderr
    with pytest.raises(graph_oci.B300PreparedGraphOCIError, match="fixed"):
        graph_oci.main(["candidate-selected"])

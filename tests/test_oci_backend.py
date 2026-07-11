"""OCI launch boundary tests; no Docker daemon or GPU is required."""

from __future__ import annotations

import dataclasses
import concurrent.futures
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.arenas import MINIMAX_M3_B300_TP4_DECODE_V1
from optima.eval import oci_backend as backend

from optima.eval.oci_backend import (
    OCIBackendError,
    OCICandidateArtifactError,
    OCIInfrastructureError,
    OCIWatchdogTimeout,
    OCILaunchProfile,
    OCILauncher,
    SubprocessOCIExecutor,
    _freeze_artifact_tree,
    _provision_device_state_policy,
    _write_artifact_publication_manifest,
    verify_production_model_once,
)
from optima.eval.device_state import (
    DeviceStateActiveReceipt,
    DeviceStateCancelledError,
    DeviceStateConfigurationError,
    DeviceStatePolicy,
    DeviceStateReceipt,
    DeviceStateSample,
    DeviceStateTimeoutError,
    GPUConfiguration,
)
from optima.eval.oci_protocol import (
    AUTH_KEY_BYTES,
    CONTAINER_ARTIFACT_PATH,
    CONTAINER_BUNDLE_PATH,
    CONTAINER_JIT_PATH,
    CONTAINER_MODEL_PATH,
    CONTAINER_OUTPUT_PATH,
    CONTAINER_SOURCE_PATH,
    FRAME_MAGIC,
    MAX_REQUEST_BYTES,
    OCIProtocolError,
    decode_request,
    decode_stdin_frame,
    encode_request,
    encode_stdin_frame,
    environment_fingerprint,
    make_request,
    topology_fingerprint,
)
from optima.eval.oci_worker import OCIWorkerError, attest_runtime, run_worker
from optima.eval.runtime_preflight import (
    HOST_RECEIPT_SCHEMA,
    RuntimePreflightError,
    RuntimePreflightReceipt,
)
from optima.eval.throughput_kl import EvalConfig
from optima.ipc import (
    LaunchOutcome,
    load_authenticated_file,
)
from optima.runtime_overlay import RuntimeFileOverlay, runtime_overlay_fingerprint


IMAGE = "registry.example/optima/arena@sha256:" + "a" * 64
SHA_ID = "sha256:" + "b" * 64
MODEL_REVISION = "c" * 40


def _runtime_preflight_receipt(
    *,
    image: str = IMAGE,
    sglang_version: str = "0.5.2",
    uid: int | None = None,
    gid: int | None = None,
    docker_binary: str = "/usr/bin/docker",
) -> RuntimePreflightReceipt:
    uid = os.getuid() if uid is None else uid
    gid = os.getgid() if gid is None else gid
    return RuntimePreflightReceipt(
        schema=HOST_RECEIPT_SCHEMA,
        requested_image=image,
        requested_manifest_digest=image.rsplit("@", 1)[1],
        local_image_id="sha256:" + "f" * 64,
        repo_digests=(image,),
        docker_binary=docker_binary,
        uid=uid,
        gid=gid,
        sglang_version=sglang_version,
        python_implementation="cpython",
        python_version="3.12.3",
        python_abi="cpython-312-x86_64-linux-gnu",
        python_platform="linux-x86_64",
        machine="x86_64",
        package_versions=(("torch", "2.11.0"),),
        cudart_library="libcudart.so.13",
        cuda_visible_devices="",
        nvidia_visible_devices="void",
        security_argv_sha256="1" * 64,
    )


@pytest.fixture
def profile(tmp_path: Path) -> OCILaunchProfile:
    paths = {}
    for name in ("source", "model", "artifacts", "bundle", "scratch"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    for relative in (
        "optima/eval/oci_worker.py",
        "optima/eval/oci_session_worker.py",
        "optima/eval/oci_prebuild.py",
        "optima/eval/oci_site/sitecustomize.py",
    ):
        source_file = paths["source"] / relative
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("# validator fixture\n", encoding="utf-8")
    seccomp_source = (
        Path(__file__).resolve().parents[1]
        / "optima/eval/seccomp_moby_v0_2_1.json"
    )
    seccomp_dest = paths["source"] / "optima/eval/seccomp_moby_v0_2_1.json"
    seccomp_dest.write_bytes(seccomp_source.read_bytes())
    runtime_patch = paths["model"] / "sglang_patch" / "modelopt_quant.py"
    runtime_patch.parent.mkdir(parents=True)
    runtime_patch.write_bytes(b"# sealed stock runtime patch\n")
    (paths["bundle"] / "manifest.toml").write_text(
        "# content-addressed bundle fixture\n", encoding="utf-8"
    )
    result = OCILaunchProfile(
        image=IMAGE,
        source_dir=paths["source"],
        model_dir=paths["model"],
        artifact_dir=paths["artifacts"],
        bundle_dir=paths["bundle"],
        scratch_root=paths["scratch"],
        gpu_devices=(0, 3, 5, 7),
        sglang_version="0.5.2",
        gpu_architecture="sm103",
        referee_source_digest=SHA_ID,
        referee_tree_digest=SHA_ID,
        model_revision=MODEL_REVISION,
        model_manifest_digest=SHA_ID,
        model_content_digest=SHA_ID,
        gpu_name="NVIDIA B300 SXM6 AC",
        gpu_memory_mib=275040,
        driver_version="595.71.05",
        runtime_overlays=(RuntimeFileOverlay(
            source="sglang_patch/modelopt_quant.py",
            target=(
                "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/"
                "modelopt_quant.py"
            ),
            sha256=hashlib.sha256(runtime_patch.read_bytes()).hexdigest(),
            size=runtime_patch.stat().st_size,
        ),),
        environment={"CUDA_DEVICE_MAX_CONNECTIONS": "1"},
    )
    from optima.bundle_hash import content_hash

    stage = paths["artifacts"] / "fixture-stage"
    request_id = "d" * 32
    receipt = stage / "prebuild_receipts" / f"{request_id}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    _write_artifact_publication_manifest(
        stage,
        profile=result,
        request_id=request_id,
        bundle_hash=content_hash(result.bundle_dir),
    )
    _freeze_artifact_tree(stage)
    stage.rename(paths["artifacts"] / "published")
    return result


def _cfg(**changes) -> EvalConfig:
    base = EvalConfig(
        model_path="host-path-is-never-sent",
        num_prompts=2,
        warmup_iters=1,
        conditioning_iters=1,
        timed_iters=1,
        tp_size=4,
        isolate=True,
        allow_unsafe_no_isolation=False,
    )
    return dataclasses.replace(base, **changes)


def _batches(count: int = 2) -> list[list[str]]:
    return [[f"batch-{batch}-prompt-{prompt}" for prompt in range(2)] for batch in range(count)]


def _device_policy_for_profile(
    profile: OCILaunchProfile, **changes
) -> DeviceStatePolicy:
    configurations = tuple(
        GPUConfiguration(
            physical_id=device,
            uuid=f"GPU-00000000-0000-0000-0000-{device:012d}",
            pci_bus_id=f"00000000:{device + 1:02x}:00.0",
            name=profile.gpu_name,
            memory_total_mib=profile.gpu_memory_mib,
            driver_version=profile.driver_version,
            power_limit_mw=1_100_000,
            compute_mode="Default",
            persistence_mode="Disabled",
            application_graphics_clock_mhz=None,
            application_memory_clock_mhz=None,
            max_graphics_clock_mhz=2_032,
            max_memory_clock_mhz=3_996,
        )
        for device in profile.gpu_devices
    )
    return dataclasses.replace(
        DeviceStatePolicy(expected_gpus=configurations), **changes
    )


def _clear_runtime_preflight_cache():
    with backend._RUNTIME_PREFLIGHT_LOCK:
        backend._RUNTIME_PREFLIGHT_CACHE.clear()


def test_docker_binary_is_resolved_to_one_absolute_normalized_executable(
    tmp_path, monkeypatch
):
    docker = tmp_path / "bin" / "docker"
    docker.parent.mkdir()
    docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    monkeypatch.setattr(backend.shutil, "which", lambda value: str(docker))

    assert backend._resolved_docker_binary("docker") == str(docker.resolve())

    wrong = tmp_path / "docker-real"
    wrong.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrong.chmod(0o755)
    with pytest.raises(OCIBackendError, match="named 'docker'"):
        backend._resolved_docker_binary(str(wrong))


def test_runtime_preflight_cache_is_success_only_thread_safe_and_identity_checked(
    monkeypatch,
):
    _clear_runtime_preflight_cache()
    docker = "/usr/bin/docker"
    receipt = _runtime_preflight_receipt(
        uid=65_532,
        gid=65_532,
        docker_binary=docker,
    )
    calls = {"n": 0}

    def attest(config):
        calls["n"] += 1
        time.sleep(0.01)
        return receipt

    monkeypatch.setattr(backend, "run_runtime_preflight", attest)
    kwargs = dict(
        image=IMAGE,
        sglang_version="0.5.2",
        worker_uid=65_532,
        worker_gid=65_532,
        docker_binary=docker,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        observed = list(pool.map(
            lambda _index: backend._runtime_preflight_once(**kwargs),
            range(8),
        ))
    assert calls["n"] == 1
    assert all(value is receipt for value in observed)

    # Cached values are revalidated, not blindly trusted.
    object.__setattr__(receipt, "sglang_version", "tampered")
    with pytest.raises(OCIBackendError, match="integrity mismatch"):
        backend._runtime_preflight_once(**kwargs)
    _clear_runtime_preflight_cache()


def test_runtime_preflight_failure_is_not_cached(monkeypatch):
    _clear_runtime_preflight_cache()
    docker = "/usr/bin/docker"
    receipt = _runtime_preflight_receipt(
        uid=65_532,
        gid=65_532,
        docker_binary=docker,
    )
    calls = {"n": 0}

    def fail_then_pass(_config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimePreflightError("daemon unavailable")
        return receipt

    monkeypatch.setattr(backend, "run_runtime_preflight", fail_then_pass)
    kwargs = dict(
        image=IMAGE,
        sglang_version="0.5.2",
        worker_uid=65_532,
        worker_gid=65_532,
        docker_binary=docker,
    )
    with pytest.raises(OCIBackendError, match="trusted stock runtime preflight failed") as caught:
        backend._runtime_preflight_once(**kwargs)
    assert caught.value.validator_fault is True
    assert not backend._RUNTIME_PREFLIGHT_CACHE
    assert backend._runtime_preflight_once(**kwargs) is receipt
    assert calls["n"] == 2
    _clear_runtime_preflight_cache()


def test_profile_preflight_failure_touches_no_candidate_model_or_gpu_helpers(
    monkeypatch,
):
    _clear_runtime_preflight_cache()
    touched = []
    monkeypatch.setattr(
        backend, "_resolved_docker_binary", lambda _value: "/usr/bin/docker"
    )
    monkeypatch.setattr(
        backend,
        "run_runtime_preflight",
        lambda _config: (_ for _ in ()).throw(
            RuntimePreflightError("stock image invalid")
        ),
    )

    def forbidden(name):
        def call(*_args, **_kwargs):
            touched.append(name)
            raise AssertionError(f"{name} ran before stock preflight")

        return call

    for name in (
        "_resolve_gpu_local_cpuset",
        "_selected_gpu_topology_fingerprint",
        "_provision_device_state_policy",
        "_resolved_directory",
        "verify_production_model_once",
    ):
        monkeypatch.setattr(backend, name, forbidden(name))

    with pytest.raises(OCIBackendError, match="trusted stock runtime preflight failed"):
        backend.profile_for_arena(
            MINIMAX_M3_B300_TP4_DECODE_V1,
            source_dir="/candidate/source",
            model_dir="/candidate/model",
            artifact_dir="/candidate/artifacts",
            scratch_root="/candidate/scratch",
            gpu_devices=(0, 1, 2, 3),
            bundle_dir="/candidate/bundle",
            competition_target="sglang.inference.bundle.v1",
        )
    assert touched == []
    _clear_runtime_preflight_cache()


def test_profile_for_arena_attaches_exact_preflight_receipt_before_host_inputs(
    monkeypatch,
):
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    resources = arena.oci_resources
    docker = "/usr/bin/docker"
    receipt = _runtime_preflight_receipt(
        image=arena.validator_image,
        sglang_version=arena.sglang_version,
        uid=resources.worker_uid,
        gid=resources.worker_gid,
        docker_binary=docker,
    )
    monkeypatch.setattr(backend, "_resolved_docker_binary", lambda _value: docker)
    monkeypatch.setattr(
        backend, "_runtime_preflight_once", lambda **_kwargs: receipt
    )
    monkeypatch.setattr(backend, "_resolve_gpu_local_cpuset", lambda *_a, **_k: "0-3")
    monkeypatch.setattr(
        backend,
        "_selected_gpu_topology_fingerprint",
        lambda _devices: arena.gpu_topology_sha256,
    )
    monkeypatch.setattr(
        backend, "_provision_device_state_policy", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        backend, "_resolved_directory", lambda value, _name: Path(value)
    )
    monkeypatch.setattr(
        backend, "verify_production_model_once", lambda _arena, value: Path(value)
    )
    monkeypatch.setattr(
        "optima.source_release.verify_referee_source_release",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        backend,
        "OCILaunchProfile",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    profile_result = backend.profile_for_arena(
        arena,
        source_dir="/trusted/source",
        model_dir="/trusted/model",
        artifact_dir="/trusted/artifacts",
        scratch_root="/trusted/scratch",
        gpu_devices=(0, 1, 2, 3),
        bundle_dir="/candidate/bundle",
        competition_target="sglang.inference.bundle.v1",
    )
    assert profile_result.runtime_preflight_receipt is receipt
    assert profile_result.docker_binary == docker


def test_profile_and_launcher_retain_exact_runtime_preflight_receipt(profile):
    docker = "/usr/bin/docker"
    receipt = _runtime_preflight_receipt(
        uid=profile.worker_uid,
        gid=profile.worker_gid,
        docker_binary=docker,
    )
    preflighted = dataclasses.replace(
        profile,
        docker_binary=docker,
        runtime_preflight_receipt=receipt,
    )

    launcher = OCILauncher(preflighted)
    assert preflighted.runtime_preflight_receipt is receipt
    assert launcher.runtime_preflight_receipt is receipt
    assert launcher.profile.runtime_preflight_receipt is receipt


def test_command_has_complete_oci_policy_and_only_profile_mounts(profile):
    launch = OCILauncher(profile).prepare(_cfg(), _batches(), mode="candidate")
    try:
        argv = list(launch.argv)
        required = {
            "--pull=never",
            "--runtime=runc",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--cap-add=SYS_NICE",
            "--cap-add=SYS_RESOURCE",
            "--security-opt=no-new-privileges=true",
            "--security-opt=seccomp="
            + str(profile.source_dir / "optima/eval/seccomp_moby_v0_2_1.json"),
            "--entrypoint=python3",
            f"--user={os.getuid()}:{os.getgid()}",
            "--ipc=private",
            "--log-driver=none",
            "--ulimit=core=0:0",
            "--memory=1099511627776",
            "--memory-swap=1099511627776",
            "--cpus=96",
            "--ulimit=nofile=65536:65536",
            '--gpus="device=0,3,5,7"',
        }
        assert required <= set(argv)
        assert "--env=USER=optima" in argv
        assert "--env=LOGNAME=optima" in argv
        assert not any(arg.startswith("--pid=") for arg in argv)
        assert [arg for arg in argv if arg.startswith("--cap-add=")] == [
            "--cap-add=SYS_NICE",
            "--cap-add=SYS_RESOURCE",
        ]
        assert "--privileged" not in argv
        assert not any(
            "host" in arg
            for arg in argv
            if arg.startswith(("--pid", "--ipc", "--network"))
        )
        assert argv[-5:] == [
            IMAGE,
            "-m",
            "optima.eval.oci_worker",
            "--result",
            "/optima/output/result.auth",
        ]

        mounts = [arg for arg in argv if arg.startswith("--mount=")]
        for host, target in (
            (profile.source_dir, CONTAINER_SOURCE_PATH),
            (profile.model_dir, CONTAINER_MODEL_PATH),
            (profile.artifact_dir / "published", CONTAINER_ARTIFACT_PATH),
            (profile.bundle_dir, CONTAINER_BUNDLE_PATH),
        ):
            assert (
                f"--mount=type=bind,src={host},dst={target},readonly" in mounts
            )
        assert (
            f"--mount=type=bind,src={launch.jit_dir},dst={CONTAINER_JIT_PATH}" in mounts
        )
        assert (
            f"--mount=type=bind,src={launch.output_dir},dst={CONTAINER_OUTPUT_PATH}" in mounts
        )
        assert all("docker.sock" not in mount for mount in mounts)
        overlay = profile.runtime_overlays[0]
        assert (
            f"--mount=type=bind,src={profile.model_dir / overlay.source},"
            f"dst={overlay.target},readonly"
        ) in mounts
        assert any(
            arg.startswith("--env=OPTIMA_OCI_EXPECTED_RUNTIME_OVERLAYS_SHA256=")
            for arg in argv
        )
        assert (
            "--env=PYTHONPATH=/optima/input/source/optima/eval/oci_site:"
            "/optima/input/source"
        ) in argv
    finally:
        launch.cleanup()


def test_baseline_cannot_see_bundle_and_every_launch_gets_a_distinct_cache(profile):
    launcher = OCILauncher(profile)
    baseline = launcher.prepare_session(mode="baseline")
    candidate = launcher.prepare_session(mode="candidate")
    bookend = launcher.prepare_session(mode="baseline")
    try:
        bundle_fragment = f"dst={CONTAINER_BUNDLE_PATH}"
        assert not any(bundle_fragment in arg for arg in baseline.argv)
        assert any(bundle_fragment in arg for arg in candidate.argv)
        assert not any(bundle_fragment in arg for arg in bookend.argv)
        assert not any(str(profile.artifact_dir) in arg for arg in baseline.argv)
        assert not any(str(profile.artifact_dir) in arg for arg in bookend.argv)
        assert baseline.stock_artifact_dir is not None
        assert bookend.stock_artifact_dir is not None
        assert baseline.stock_artifact_dir != bookend.stock_artifact_dir
        assert any(str(baseline.stock_artifact_dir) in arg for arg in baseline.argv)
        assert not any("OPTIMA_PREBUILT_ARTIFACTS" in arg for arg in baseline.argv)
        assert not any("OPTIMA_OCI_COMPETITION_TARGET" in arg for arg in baseline.argv)
        assert len({baseline.jit_dir, candidate.jit_dir, bookend.jit_dir}) == 3
        assert len({baseline.container_name, candidate.container_name, bookend.container_name}) == 3
        assert all(path.is_dir() for path in (baseline.jit_dir, candidate.jit_dir, bookend.jit_dir))
        overlay_target = f"dst={profile.runtime_overlays[0].target},readonly"
        assert all(
            any(overlay_target in arg for arg in launch.argv)
            for launch in (baseline, candidate, bookend)
        )
        assert all(
            "--runtime=runc" in launch.argv
            for launch in (baseline, candidate, bookend)
        )
        assert all(
            "--entrypoint=python3" in launch.argv
            for launch in (baseline, candidate, bookend)
        )
        assert all(
            launch.argv[launch.argv.index(IMAGE) + 1 :] == (
                "-m", "optima.eval.oci_session_worker"
            )
            for launch in (baseline, candidate, bookend)
        )
    finally:
        baseline.cleanup()
        candidate.cleanup()
        bookend.cleanup()


def test_system_policy_is_staged_only_in_candidate_from_profile(profile, monkeypatch):
    monkeypatch.setattr(
        "optima.eval.oci_backend._verify_source_release_before_mount", lambda profile: None
    )
    monkeypatch.setattr(
        "optima.eval.oci_backend._mount_worker_tmpfs",
        lambda path, **kwargs: False,
    )
    artifact_dir = profile.artifact_dir.parent / "system-artifacts"
    artifact_dir.mkdir()
    docker_binary = "/usr/bin/docker"
    system_profile = dataclasses.replace(
        profile,
        artifact_dir=artifact_dir,
        arena_name="minimax-m3-b300-tp4-decode-v1",
        arena_fingerprint="e" * 64,
        competition_target="sglang.decode-region.v1",
        device_state_policy=_device_policy_for_profile(profile),
        require_host_tmpfs=True,
        docker_binary=docker_binary,
        runtime_preflight_receipt=_runtime_preflight_receipt(
            uid=profile.worker_uid,
            gid=profile.worker_gid,
            docker_binary=docker_binary,
        ),
    )
    launcher = OCILauncher(
        system_profile, prebuild_executor=_PrebuildFakeExecutor()
    )
    launcher.prebuild_candidate_artifacts()
    baseline = launcher.prepare_session(mode="baseline")
    candidate = launcher.prepare_session(mode="candidate")
    try:
        assert not any("OPTIMA_OCI_ARENA_NAME" in arg for arg in baseline.argv)
        assert not any("OPTIMA_OCI_COMPETITION_TARGET" in arg for arg in baseline.argv)
        assert (
            "--env=OPTIMA_OCI_ARENA_NAME=minimax-m3-b300-tp4-decode-v1"
            in candidate.argv
        )
        assert (
            "--env=OPTIMA_OCI_COMPETITION_TARGET=sglang.decode-region.v1"
            in candidate.argv
        )
        assert not any("OPTIMA_SYSTEM_DRIVER_PID" in arg for arg in candidate.argv)
    finally:
        baseline.cleanup()
        candidate.cleanup()


def test_runtime_overlay_bytes_are_rehashed_before_every_launch(profile):
    overlay = profile.runtime_overlays[0]
    (profile.model_dir / overlay.source).write_bytes(b"tampered after profile creation\n")
    with pytest.raises(OCIBackendError, match="runtime overlay.*failed|hash mismatch"):
        OCILauncher(profile).prepare_session(mode="baseline")


def test_seccomp_profile_bytes_are_rehashed_before_every_launch(profile):
    seccomp = profile.source_dir / "optima/eval/seccomp_moby_v0_2_1.json"
    seccomp.write_text('{"defaultAction":"SCMP_ACT_ALLOW"}\n', encoding="utf-8")
    with pytest.raises(OCIBackendError, match="seccomp profile hash mismatch"):
        OCILauncher(profile).prepare_session(mode="baseline")


class _FakeDeviceGuard:
    def __init__(self, policy, *, fail_before=None, fail_after=None):
        self.policy = policy
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.sequence = 0
        self.calls = []

    def _receipt(self, arm, phase, deadline):
        self.sequence += 1
        self.calls.append((phase, arm, deadline))
        return DeviceStateReceipt(
            schema="optima.device-state-receipt.v1",
            sequence=self.sequence,
            arm=arm,
            phase=phase,
            selected_physical_gpu_ids=self.policy.physical_gpu_ids,
            configuration_sha256=self.policy.configuration_sha256,
            policy_sha256=self.policy.policy_sha256,
            started_monotonic_s=float(self.sequence),
            completed_monotonic_s=float(self.sequence) + 0.25,
            consecutive_idle_samples=self.policy.required_consecutive_idle_samples,
            samples=(),
        )

    def before_arm(self, arm, *, deadline):
        if self.fail_before is not None:
            raise self.fail_before
        return self._receipt(arm, "pre", deadline)

    def after_arm(self, arm, *, deadline):
        if self.fail_after is not None:
            raise self.fail_after
        return self._receipt(arm, "post", deadline)

    def condition_active(
        self, arm, event, *, deadline, release=None, wait_for_release=None,
        cancel=None,
    ):
        while release is not None:
            if cancel is not None and cancel():
                raise DeviceStateCancelledError("fake active observation cancelled")
            if release():
                break
            if time.monotonic() >= deadline:
                raise DeviceStateTimeoutError("fake active release timed out")
            if wait_for_release is not None:
                wait_for_release(min(60.0, deadline - time.monotonic()))
            else:
                time.sleep(0.001)
        self.sequence += 1
        self.calls.append(("active", arm, deadline))
        samples = tuple(
            DeviceStateSample(
                monotonic_s=float(self.sequence) + index / 100.0,
                telemetry=(),
                processes=(),
                idle=False,
                idle_reason="active",
                active_envelope_passed=True,
                active_envelope_reason="pinned",
            )
            for index in range(self.policy.required_consecutive_idle_samples + 1)
        )
        return DeviceStateActiveReceipt(
            schema="optima.device-state-active-receipt.v2",
            sequence=self.sequence,
            arm=arm,
            event=event,
            selected_physical_gpu_ids=self.policy.physical_gpu_ids,
            configuration_sha256=self.policy.configuration_sha256,
            policy_sha256=self.policy.policy_sha256,
            started_monotonic_s=float(self.sequence),
            completed_monotonic_s=float(self.sequence) + 0.25,
            consecutive_active_samples=self.policy.required_consecutive_idle_samples,
            release_sample_index=self.policy.required_consecutive_idle_samples,
            post_release_ready_samples=1,
            samples=samples,
        )


def _complete_fake_outer_boundaries(kwargs):
    boundary = kwargs["warmup_timed_boundary"]
    deadline = time.monotonic() + 30.0
    boundary("before_final_warmup", kwargs["mode"], 0, deadline)
    boundary("after_final_warmup", kwargs["mode"], 0, deadline)
    boundary("before_first_timed", kwargs["mode"], 1, deadline)


def test_launcher_wraps_each_arm_in_canonical_device_receipts_and_reserves_post_drain(
    profile, monkeypatch,
):
    policy = _device_policy_for_profile(profile, drain_timeout_s=10.0)
    guarded_profile = dataclasses.replace(profile, device_state_policy=policy)
    guard = _FakeDeviceGuard(policy)
    captured = []

    def fake_outer(*args, **kwargs):
        captured.append(kwargs)
        _complete_fake_outer_boundaries(kwargs)
        return "mode-result"

    monkeypatch.setattr(
        "optima.eval.oci_outer_session.run_outer_timed_session", fake_outer
    )
    launcher = OCILauncher(
        guarded_profile,
        device_guard=guard,
        session_factory=lambda launch: object(),
    )
    launcher.begin_evaluation(timeout_s=100.0)

    assert launcher.run(_cfg(), _batches(), mode="baseline", arm="baseline") == "mode-result"
    assert launcher.run(_cfg(), _batches(), mode="baseline", arm="bookend") == "mode-result"

    assert [(phase, arm) for phase, arm, _ in guard.calls] == [
        ("pre", "baseline-1"),
        ("active", "baseline-1"),
        ("post", "baseline-1"),
        ("pre", "bookend-1"),
        ("active", "bookend-1"),
        ("post", "bookend-1"),
    ]
    assert guard.calls[0][2] == guard.calls[2][2]
    assert guard.calls[3][2] == guard.calls[5][2]
    assert all(89.0 < call["total_timeout_s"] <= 90.0 for call in captured)
    assert [receipt["schema"] for receipt in launcher.attestation_receipts] == [
        "optima.device-state-receipt.v1",
        "optima.device-state-active-receipt.v2",
        "optima.device-state-receipt.v1",
        "optima.device-state-receipt.v1",
        "optima.device-state-active-receipt.v2",
        "optima.device-state-receipt.v1",
    ]
    assert [receipt["arm"] for receipt in launcher.attestation_receipts] == [
        "baseline-1", "baseline-1", "baseline-1",
        "bookend-1", "bookend-1", "bookend-1",
    ]
    assert isinstance(launcher.attestation_receipts[0]["samples"], list)


def test_failed_attempt_receipts_remain_diagnostic_and_retry_publishes_one_triplet(
    profile, monkeypatch,
):
    from optima.eval.oci_outer_session import OuterSessionInfrastructureError

    policy = _device_policy_for_profile(profile, drain_timeout_s=10.0)
    guard = _FakeDeviceGuard(policy)
    attempts = 0

    def fake_outer(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # This models a runtime-preflight failure: the active boundary was
            # never reached, but pre/post drain evidence still exists for audit.
            raise OuterSessionInfrastructureError("runtime preflight mismatch")
        _complete_fake_outer_boundaries(kwargs)
        return "candidate-result"

    monkeypatch.setattr(
        "optima.eval.oci_outer_session.run_outer_timed_session", fake_outer
    )
    launcher = OCILauncher(
        dataclasses.replace(profile, device_state_policy=policy),
        device_guard=guard,
        session_factory=lambda launch: object(),
    )
    launcher.begin_evaluation(timeout_s=100.0)

    with pytest.raises(OuterSessionInfrastructureError, match="preflight"):
        launcher.run(_cfg(), _batches(), mode="candidate", arm="candidate")
    assert launcher.attestation_receipts == []

    assert (
        launcher.run(_cfg(), _batches(), mode="candidate", arm="candidate")
        == "candidate-result"
    )
    assert [receipt["arm"] for receipt in launcher.attestation_receipts] == [
        "candidate-2", "candidate-2", "candidate-2",
    ]
    assert [receipt["sequence"] for receipt in launcher.attestation_receipts] == [
        3, 4, 5,
    ]
    assert [receipt["arm"] for receipt in launcher.device_diagnostic_receipts] == [
        "candidate-1", "candidate-1",
        "candidate-2", "candidate-2", "candidate-2",
    ]
    assert [
        receipt["schema"] for receipt in launcher.device_diagnostic_receipts
    ] == [
        "optima.device-state-receipt.v1",
        "optima.device-state-receipt.v1",
        "optima.device-state-receipt.v1",
        "optima.device-state-active-receipt.v2",
        "optima.device-state-receipt.v1",
    ]


def test_failed_final_warmup_cancels_conditioner_without_masking_primary(
    profile, monkeypatch,
):
    from optima.eval.oci_outer_session import OuterSessionCandidateError

    policy = _device_policy_for_profile(profile, drain_timeout_s=10.0)
    guard = _FakeDeviceGuard(policy)

    def fail_during_final_warmup(*args, **kwargs):
        boundary = kwargs["warmup_timed_boundary"]
        boundary(
            "before_final_warmup",
            kwargs["mode"],
            0,
            time.monotonic() + 30.0,
        )
        raise OuterSessionCandidateError("candidate warmup protocol failed")

    monkeypatch.setattr(
        "optima.eval.oci_outer_session.run_outer_timed_session",
        fail_during_final_warmup,
    )
    launcher = OCILauncher(
        dataclasses.replace(profile, device_state_policy=policy),
        device_guard=guard,
        session_factory=lambda launch: object(),
    )
    launcher.begin_evaluation(timeout_s=100.0)

    started = time.monotonic()
    with pytest.raises(OuterSessionCandidateError, match="warmup protocol"):
        launcher.run(_cfg(), _batches(), mode="candidate", arm="candidate")
    assert time.monotonic() - started < 2.0
    assert launcher.attestation_receipts == []
    assert [receipt["phase"] for receipt in launcher.device_diagnostic_receipts] == [
        "pre", "post",
    ]


def test_device_drain_failures_are_never_charged_to_candidate(profile, monkeypatch):
    policy = _device_policy_for_profile(profile, drain_timeout_s=10.0)
    guarded_profile = dataclasses.replace(profile, device_state_policy=policy)
    monkeypatch.setattr(
        "optima.eval.oci_outer_session.run_outer_timed_session",
        lambda *args, **kwargs: (
            _complete_fake_outer_boundaries(kwargs) or "candidate-result"
        ),
    )

    timed_out = OCILauncher(
        guarded_profile,
        device_guard=_FakeDeviceGuard(
            policy, fail_before=DeviceStateTimeoutError("GPU still occupied")
        ),
        session_factory=lambda launch: object(),
    )
    timed_out.begin_evaluation(timeout_s=100.0)
    with pytest.raises(OCIInfrastructureError) as timeout:
        timed_out.run(_cfg(), _batches(), mode="candidate", arm="candidate")
    assert timeout.value.retryable is True
    assert timeout.value.validator_fault is False

    bad_configuration = OCILauncher(
        guarded_profile,
        device_guard=_FakeDeviceGuard(
            policy,
            fail_after=DeviceStateConfigurationError("power limit changed"),
        ),
        session_factory=lambda launch: object(),
    )
    bad_configuration.begin_evaluation(timeout_s=100.0)
    with pytest.raises(OCIBackendError) as configuration:
        bad_configuration.run(
            _cfg(), _batches(), mode="candidate", arm="candidate"
        )
    assert type(configuration.value) is OCIBackendError
    assert configuration.value.retryable is False
    assert configuration.value.validator_fault is True


def test_arena_device_class_provisions_host_identity_without_pinning_physical_ids(
    monkeypatch,
):
    from optima.arenas import MINIMAX_M3_B300_TP4_DECODE_V1 as arena

    device_class = arena.device_state
    configurations = tuple(
        GPUConfiguration(
            physical_id=device,
            uuid=f"GPU-00000000-0000-0000-0000-{device:012d}",
            pci_bus_id=f"00000000:{device + 1:02x}:00.0",
            name=arena.gpu_name,
            memory_total_mib=arena.gpu_memory_mib,
            driver_version=arena.driver_version,
            power_limit_mw=device_class.power_limit_mw,
            compute_mode=device_class.compute_mode,
            persistence_mode=device_class.persistence_mode,
            application_graphics_clock_mhz=(
                device_class.application_graphics_clock_mhz
            ),
            application_memory_clock_mhz=(
                device_class.application_memory_clock_mhz
            ),
            max_graphics_clock_mhz=device_class.max_graphics_clock_mhz,
            max_memory_clock_mhz=device_class.max_memory_clock_mhz,
        )
        for device in (4, 5, 6, 7)
    )
    monkeypatch.setattr(
        "optima.eval.oci_backend.provision_gpu_configurations",
        lambda selected, **kwargs: configurations,
    )

    policy = _provision_device_state_policy(arena, (4, 5, 6, 7))

    assert policy.expected_gpus == configurations
    assert policy.physical_gpu_ids == (4, 5, 6, 7)
    assert policy.drain_timeout_s == device_class.drain_timeout_s
    assert policy.maximum_samples == device_class.maximum_samples
    assert policy.allowed_active_pstates == ("P0",)
    assert policy.active_maximum_graphics_clock_mhz == device_class.max_graphics_clock_mhz
    assert policy.active_memory_clock_mhz == device_class.max_memory_clock_mhz


class _PrebuildFakeExecutor:
    def __init__(self):
        self.argv = None
        self.launch_root = None
        self.staging_artifact_dir = None

    def run(self, launch, *, timeout_s):
        from optima.bundle_hash import content_hash

        self.argv = launch.argv
        self.launch_root = launch.launch_root
        self.staging_artifact_dir = launch.build_artifact_dir
        launch.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        launch.receipt_path.write_text(json.dumps({
            "schema": "optima-oci-prebuild-v1",
            "request_id": launch.request_id,
            "bundle_hash": content_hash(launch.profile.bundle_dir),
            "rebuild_plan": True,
            "dep_targets": [],
            "system_cache_key": "",
            "system_dest": "",
        }), encoding="utf-8")
        return 0


def _fresh_artifact_profile(profile, name: str):
    root = profile.artifact_dir.parent / name
    root.mkdir()
    return dataclasses.replace(profile, artifact_dir=root)


def test_same_image_prebuild_has_rw_artifacts_but_ro_untrusted_inputs(profile):
    profile = _fresh_artifact_profile(profile, "fresh-artifacts")
    fake = _PrebuildFakeExecutor()
    launcher = OCILauncher(profile, prebuild_executor=fake)
    receipt = launcher.prebuild_candidate_artifacts(timeout_s=13)
    assert receipt.is_file()
    assert fake.launch_root is not None and not fake.launch_root.exists()
    argv = list(fake.argv)
    assert IMAGE in argv
    assert "optima.eval.oci_prebuild" in argv
    assert "--runtime=runc" in argv
    assert "--entrypoint=python3" in argv
    image_index = argv.index(IMAGE)
    assert argv[image_index + 1 : image_index + 3] == [
        "-m", "optima.eval.oci_prebuild",
    ]
    assert "--network=none" in argv and "--read-only" in argv
    assert (
        f"--mount=type=bind,src={profile.source_dir},"
        f"dst={CONTAINER_SOURCE_PATH},readonly"
    ) in argv
    assert f"--mount=type=bind,src={profile.model_dir},dst={CONTAINER_MODEL_PATH},readonly" in argv
    assert (
        f"--mount=type=bind,src={profile.bundle_dir},"
        f"dst={CONTAINER_BUNDLE_PATH},readonly"
    ) in argv
    artifact_mount = (
        f"--mount=type=bind,src={fake.staging_artifact_dir},dst={CONTAINER_ARTIFACT_PATH}"
    )
    assert artifact_mount in argv and not artifact_mount.endswith("readonly")
    assert "--env=OPTIMA_REBUILD_PHASE=build" in argv
    assert "--env=OPTIMA_TARGET_GPU_ARCH=sm_103a" in argv
    assert "--env=TORCH_CUDA_ARCH_LIST=10.3a" in argv
    assert not any(arg.startswith("--gpus=") for arg in argv)
    assert any(
        f"dst={profile.runtime_overlays[0].target},readonly" in arg for arg in argv
    )
    assert receipt.parent.parent == profile.artifact_dir / "published"
    assert (profile.artifact_dir / "published" / "optima-artifact-publication.json").is_file()
    assert not list(profile.artifact_dir.glob(".stage-*"))


def test_prebuild_strips_only_regular_system_overlay_build_lock(profile):
    profile = _fresh_artifact_profile(profile, "system-overlay-lock-artifacts")

    class SystemOverlayExecutor(_PrebuildFakeExecutor):
        def run(self, launch, *, timeout_s):
            result = super().run(launch, timeout_s=timeout_s)
            cache = launch.build_artifact_dir / "system_overlay" / "aa" / "cache"
            package = cache / "site" / "sglang"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("# overlay\n", encoding="utf-8")
            (cache.parent / ".overlay.lock").write_text("", encoding="utf-8")
            return result

    receipt = OCILauncher(
        profile, prebuild_executor=SystemOverlayExecutor()
    ).prebuild_candidate_artifacts()

    publication = receipt.parent.parent
    assert (publication / "system_overlay/aa/cache/site/sglang/__init__.py").is_file()
    assert not list(publication.rglob(".overlay.lock"))


def test_prebuild_rejects_system_overlay_lock_indirection(profile):
    profile = _fresh_artifact_profile(profile, "system-overlay-lock-symlink")

    class SymlinkLockExecutor(_PrebuildFakeExecutor):
        def run(self, launch, *, timeout_s):
            result = super().run(launch, timeout_s=timeout_s)
            parent = launch.build_artifact_dir / "system_overlay" / "aa"
            parent.mkdir(parents=True)
            (parent / ".overlay.lock").symlink_to(launch.receipt_path)
            return result

    with pytest.raises(OCICandidateArtifactError, match="lock path is unsafe"):
        OCILauncher(
            profile, prebuild_executor=SymlinkLockExecutor()
        ).prebuild_candidate_artifacts()


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_prebuild_receipt_indirections_fail_before_read_or_freeze(profile, attack):
    profile = _fresh_artifact_profile(profile, f"artifacts-{attack}")
    external = profile.source_dir / f"external-{attack}.json"

    class AttackExecutor:
        def run(self, launch, *, timeout_s):
            from optima.bundle_hash import content_hash

            external.write_text(json.dumps({
                "schema": "optima-oci-prebuild-v1",
                "request_id": launch.request_id,
                "bundle_hash": content_hash(launch.profile.bundle_dir),
                "rebuild_plan": True,
                "dep_targets": [],
                "system_cache_key": "",
                "system_dest": "",
            }), encoding="utf-8")
            launch.receipt_path.parent.mkdir(parents=True)
            if attack == "symlink":
                launch.receipt_path.symlink_to(external)
            else:
                os.link(external, launch.receipt_path)
            return 0

    with pytest.raises(OCIBackendError, match="open bounded|shape is unsafe"):
        OCILauncher(profile, prebuild_executor=AttackExecutor()).prebuild_candidate_artifacts()
    assert external.is_file() and os.access(external, os.W_OK)
    assert not (profile.artifact_dir / "published").exists()
    assert not list(profile.artifact_dir.glob(".stage-*"))


def test_prebuild_rejects_empty_hidden_directory_and_over_byte_cap(profile):
    profile = _fresh_artifact_profile(profile, "artifacts-bounds")
    profile = dataclasses.replace(profile, artifact_max_bytes=1 << 20)

    class OversizeExecutor(_PrebuildFakeExecutor):
        def run(self, launch, *, timeout_s):
            result = super().run(launch, timeout_s=timeout_s)
            (launch.build_artifact_dir / "evil" / ".hidden").mkdir(parents=True)
            payload = launch.build_artifact_dir / "cuda_ext" / "oversize.bin"
            payload.parent.mkdir()
            payload.write_bytes(b"x" * ((1 << 20) + 1))
            return result

    with pytest.raises(OCICandidateArtifactError, match="artifact tree is invalid"):
        OCILauncher(
            profile, prebuild_executor=OversizeExecutor()
        ).prebuild_candidate_artifacts()
    assert not (profile.artifact_dir / "published").exists()


def test_prebuild_watchdog_is_terminal_candidate_failure(profile):
    profile = _fresh_artifact_profile(profile, "artifacts-timeout")

    class TimeoutExecutor:
        def run(self, launch, *, timeout_s):
            raise OCIWatchdogTimeout("compiler watchdog")

    with pytest.raises(OCICandidateArtifactError) as caught:
        OCILauncher(
            profile, prebuild_executor=TimeoutExecutor()
        ).prebuild_candidate_artifacts()
    assert caught.value.retryable is False
    assert caught.value.validator_fault is False


def test_production_model_bytes_are_hashed_once_per_daemon(tmp_path):
    import optima.eval.oci_backend as backend

    model = tmp_path / "model-once"
    model.mkdir()
    calls = []

    class Arena:
        model_content_digest = "sha256:" + "9" * 64
        model_revision = "8" * 40

        def verify_model_receipt(self, path, *, verify_bytes):
            calls.append(verify_bytes)

    backend._VERIFIED_MODEL_BYTES.clear()
    arena = Arena()
    assert verify_production_model_once(arena, model) == model.resolve()
    assert verify_production_model_once(arena, model) == model.resolve()
    assert calls == [False, True, False]


def test_request_and_hmac_key_exist_only_on_stdin_not_argv_env_or_files(profile):
    first = OCILauncher(profile).prepare(_cfg(), _batches(), mode="candidate")
    second = OCILauncher(profile).prepare(_cfg(), _batches(), mode="candidate")
    try:
        request, key = decode_stdin_frame(first.stdin_frame)
        _, second_key = decode_stdin_frame(second.stdin_frame)
        assert key == first.auth_key and len(key) == AUTH_KEY_BYTES
        assert key != second_key
        assert request.nonce != second.request.nonce
        assert request.eval_config["model_path"] == CONTAINER_MODEL_PATH
        argv_bytes = "\x00".join(first.argv).encode()
        assert key not in argv_bytes
        assert request.nonce not in argv_bytes
        for path in first.launch_root.rglob("*"):
            if path.is_file():
                assert key not in path.read_bytes()
        assert not any("HMAC" in arg or "AUTH_KEY" in arg for arg in first.argv)
    finally:
        first.cleanup()
        second.cleanup()


def test_protocol_rejects_extra_fields_unknown_engine_options_and_bad_batches():
    request = make_request(
        _cfg(), _batches(), mode="candidate", request_id="1" * 32, nonce=b"n" * 16
    )
    payload = json.loads(encode_request(request))
    payload["unexpected"] = True
    with pytest.raises(OCIProtocolError, match="envelope fields"):
        decode_request(json.dumps(payload).encode())

    with pytest.raises(OCIProtocolError, match="unsupported keys"):
        make_request(
            _cfg(extra_engine_kwargs={"miner_selected_callable": "evil.module"}),
            _batches(),
            mode="candidate",
            request_id="2" * 32,
            nonce=b"m" * 16,
        )
    with pytest.raises(OCIProtocolError, match="expected 2"):
        make_request(
            _cfg(), _batches(1), mode="candidate", request_id="3" * 32, nonce=b"k" * 16
        )
    with pytest.raises(OCIProtocolError, match="tp_size"):
        # bool is deliberately not accepted as Python's integer subtype.
        make_request(
            _cfg(tp_size=True),
            _batches(),
            mode="candidate",
            request_id="4" * 32,
            nonce=b"j" * 16,
        )


def test_protocol_rejects_oversize_header_before_allocating():
    frame = FRAME_MAGIC + struct.pack(">I", MAX_REQUEST_BYTES + 1)
    with pytest.raises(OCIProtocolError, match="hard bound"):
        decode_stdin_frame(frame + b"x" * AUTH_KEY_BYTES)


def test_profile_requires_digest_gpu_list_and_nonsecret_environment(tmp_path):
    dirs = []
    for index in range(4):
        path = tmp_path / str(index)
        path.mkdir()
        dirs.append(path)
    for relative in (
        "optima/eval/oci_worker.py",
        "optima/eval/oci_session_worker.py",
        "optima/eval/oci_prebuild.py",
        "optima/eval/oci_site/sitecustomize.py",
    ):
        source_file = dirs[0] / relative
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("# validator fixture\n", encoding="utf-8")
    seccomp_source = (
        Path(__file__).resolve().parents[1]
        / "optima/eval/seccomp_moby_v0_2_1.json"
    )
    (dirs[0] / "optima/eval/seccomp_moby_v0_2_1.json").write_bytes(
        seccomp_source.read_bytes()
    )
    kwargs = dict(
        source_dir=dirs[0], model_dir=dirs[1], artifact_dir=dirs[2],
        scratch_root=dirs[3], gpu_devices=(0,), sglang_version="0.5.2",
        gpu_architecture="sm103", referee_source_digest=SHA_ID,
        referee_tree_digest=SHA_ID,
        model_revision=MODEL_REVISION, model_manifest_digest=SHA_ID,
        model_content_digest=SHA_ID, gpu_name="NVIDIA B300 SXM6 AC",
        gpu_memory_mib=275040, driver_version="595.71.05",
    )
    with pytest.raises(OCIBackendError, match="sha256"):
        OCILaunchProfile(image="mutable:latest", **kwargs)
    with pytest.raises(OCIBackendError, match="unique"):
        OCILaunchProfile(image=IMAGE, **{**kwargs, "gpu_devices": (0, 0)})
    with pytest.raises(OCIBackendError, match="unsafe.*environment"):
        OCILaunchProfile(
            image=IMAGE, **{**kwargs, "environment": {"OPTIMA_HMAC_KEY": "leak"}}
        )


def test_worker_closes_stdin_before_hook_and_authenticates_result(tmp_path):
    request = make_request(
        _cfg(), _batches(), mode="candidate", request_id="5" * 32, nonce=b"q" * 16
    )
    key = b"s" * AUTH_KEY_BYTES
    frame = encode_stdin_frame(request, auth_key=key)
    read_fd, write_fd = os.pipe()
    os.write(write_fd, frame)
    os.close(write_fd)
    result = tmp_path / "result.auth"
    result.touch()

    def hook(observed):
        assert observed == request
        with pytest.raises(OSError):
            os.fstat(read_fd)
        return {"measured": 123.0}

    assert run_worker(
        result_path=str(result),
        stdin_fd=read_fd,
        execute=hook,
        verify_sandbox=False,
        require_dontfork=False,
    ) == 0
    outcome = load_authenticated_file(result, key=key, nonce=request.nonce)
    assert outcome == LaunchOutcome(
        value={
            "result": {"measured": 123.0},
            "runtime_attestation": {
                "verified": False,
                "test_only_bypass": True,
            },
        },
        error=None,
    )
    assert key not in result.read_bytes()


def test_subprocess_executor_is_argv_only_and_closes_stdin(monkeypatch, profile):
    launch = OCILauncher(profile).prepare(_cfg(), _batches(), mode="candidate")
    observed = {}

    class DummyProcess:
        pid = 999999
        returncode = 0

        def communicate(self, *, input, timeout):
            observed["input"] = input
            observed["timeout"] = timeout

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    try:
        assert SubprocessOCIExecutor().run(launch, timeout_s=9) == 0
        assert observed["argv"] == list(launch.argv)
        assert observed["input"] == launch.stdin_frame
        assert observed["kwargs"]["shell"] is False
        assert observed["kwargs"]["close_fds"] is True
        assert observed["kwargs"]["stdin"] is not None
    finally:
        launch.cleanup()


def test_force_remove_requires_independent_container_absence(monkeypatch, profile):
    launch = OCILauncher(profile).prepare_session(mode="baseline")
    calls = []

    def gone(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("subprocess.run", gone)
    try:
        SubprocessOCIExecutor._force_remove(launch)
        assert [call[0][1] for call in calls] == ["rm", "container"]
        assert all(call[1]["shell"] is False for call in calls)
    finally:
        launch.cleanup()


def test_force_remove_fails_closed_when_container_survives(monkeypatch, profile):
    launch = OCILauncher(profile).prepare_session(mode="baseline")
    def survives(argv, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=b"" if argv[1] == "rm" else b"deadbeef\n",
            stderr=b"",
        )

    monkeypatch.setattr("subprocess.run", survives)
    try:
        with pytest.raises(OCIInfrastructureError, match="still exists"):
            SubprocessOCIExecutor._force_remove(launch)
    finally:
        launch.cleanup()


def test_force_remove_does_not_treat_docker_daemon_failure_as_absence(
    monkeypatch, profile,
):
    launch = OCILauncher(profile).prepare_session(mode="baseline")

    def daemon_failure(argv, **kwargs):
        return SimpleNamespace(
            returncode=0 if argv[1] == "rm" else 125,
            stdout=b"",
            stderr=b"cannot connect to daemon",
        )

    monkeypatch.setattr("subprocess.run", daemon_failure)
    try:
        with pytest.raises(OCIInfrastructureError, match="absence listing"):
            SubprocessOCIExecutor._force_remove(launch)
    finally:
        launch.cleanup()


def test_runtime_attestation_binds_packages_model_source_env_and_gpu(monkeypatch):
    policy_env = {"ARENA_DANGEROUS_KNOB": "exact"}
    monkeypatch.setenv("ARENA_DANGEROUS_KNOB", "exact")
    monkeypatch.setenv("OPTIMA_OCI_ATTEST_ENV_KEYS", "ARENA_DANGEROUS_KNOB")
    monkeypatch.setenv(
        "OPTIMA_OCI_ATTEST_ENV_SHA256", environment_fingerprint(policy_env)
    )
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_SGLANG_VERSION", "0.5.2")
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_GPU_ARCH", "sm103")
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_GPU_COUNT", "4")
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_REFEREE_SOURCE_DIGEST", SHA_ID)
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_MODEL_REVISION", MODEL_REVISION)
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_MODEL_MANIFEST_DIGEST", SHA_ID)
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_MODEL_CONTENT_DIGEST", SHA_ID)
    monkeypatch.setenv("OPTIMA_OCI_RUNTIME_OVERLAYS_JSON", "[]")
    monkeypatch.setenv("OPTIMA_OCI_EXPECTED_RUNTIME_OVERLAY_COUNT", "0")
    monkeypatch.setenv(
        "OPTIMA_OCI_EXPECTED_RUNTIME_OVERLAYS_SHA256",
        runtime_overlay_fingerprint(()),
    )
    topology = "GPU0 GPU1\nGPU0 X NV18\nGPU1 NV18 X\n"
    monkeypatch.setenv(
        "OPTIMA_OCI_EXPECTED_TOPOLOGY_SHA256", topology_fingerprint(topology)
    )
    monkeypatch.setattr("optima.arenas.referee_source_digest", lambda path: SHA_ID)
    monkeypatch.setattr(
        "optima.arenas.huggingface_model_manifest",
        lambda path: (MODEL_REVISION, SHA_ID),
    )
    monkeypatch.setattr(
        "optima.arenas.verify_model_content_seal", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "optima.runtime_overlay.verify_runtime_overlays",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        "optima.runtime_overlay.verify_runtime_overlay_targets",
        lambda *args, **kwargs: (),
    )

    def command(argv):
        return "10.3\n" * 4 if "--query-gpu=compute_cap" in argv else topology

    receipt = attest_runtime(
        version_reader=lambda package: "0.5.2",
        command_reader=command,
    )
    assert receipt["verified"] is True
    assert receipt["gpu_count"] == 4
    assert receipt["gpu_architectures"] == ["sm103"] * 4
    assert receipt["topology_sha256"] == topology_fingerprint(topology)

    monkeypatch.setenv("ARENA_DANGEROUS_KNOB", "changed")
    with pytest.raises(OCIWorkerError, match="environment fingerprint"):
        attest_runtime(
            version_reader=lambda package: "0.5.2",
            command_reader=command,
        )

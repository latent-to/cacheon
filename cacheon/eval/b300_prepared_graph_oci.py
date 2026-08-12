"""Fixed OCI lifetime for one commissioned B300 TP4 graph probe."""

from __future__ import annotations

import math
import os
import secrets
import sys
from pathlib import Path

from cacheon.cute_aot import CUTE_COMPILE_PROFILE_DIGEST_ENV
from cacheon.eval.b300_qualification_graph_provider import (
    B300QualificationGraphArtifact,
)
from cacheon.eval.device_state import (
    DeviceStateGuard,
    DeviceStatePolicy,
    validate_device_state_policy,
)
from cacheon.eval.engine_launch import resolve_engine_launch
from cacheon.eval.marginal_runtime import PreparedCandidateRuntime
from cacheon.eval.native_artifact import reopen_native_artifact
from cacheon.eval.oci_backend import (
    CONTAINER_ARTIFACT_BASE,
    CONTAINER_CACHE,
    CONTAINER_TREE,
    OCIBackendConfig,
    build_bind_mount_arg,
    stage_seccomp_profile,
)
from cacheon.eval.oci_prebuild import run_oci_prebuild
from cacheon.eval.oci_process import CaptureRunner, OCIProcessManager
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt, bounded_argv_runner
from cacheon.stack_identity import require_sha256_hex
from cacheon.stack_plan import MarginalArmPlan

CONTAINER_REQUEST = "/cacheon/input/graph-request/request.json"
_REQUEST_DIR = "/cacheon/input/graph-request"
_CONTROLLER_ROOT = "/cacheon/controller"
# The tree root is appended (never inserted): overlay-materialized trees root
# their delta package (cacheon_c_<digest>/) at the tree root and the entries/
# shims import it absolutely, so the root must be importable in the worker and
# in every spawned rank (spawn propagates sys.path). Appending keeps every real
# package resolving from the image; only the unique delta names fall through.
# Without this, every overlay-tree probe fails all ranks with
# ModuleNotFoundError and the store records an incomplete observation.
#
# Image site still runs under ``python -I`` and executes ``cacheon.pth``
# (``import cacheon.bootstrap``) before ``-c``, pinning the image's incomplete
# dist-packages ``cacheon`` into ``sys.modules``. Evict those entries after the
# controller insert so ``run_module`` resolves the mounted controller source.
_CONTROLLER_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,'/cacheon/controller');"
    "[sys.modules.pop(n) for n in tuple(sys.modules)"
    " if n=='cacheon' or n.startswith('cacheon.')];"
    f"sys.path.append('{CONTAINER_TREE}');"
    "runpy.run_module('cacheon.eval.b300_prepared_graph_oci',run_name='__main__')"
)
_MAX_REQUEST_BYTES = 256 << 10
_MAX_ARTIFACT_BYTES = 8 << 20
_MAX_STDERR_BYTES = 64 << 10


class B300PreparedGraphOCIError(RuntimeError):
    """Graph infrastructure was incomplete; it cannot become candidate FAIL."""

    decision = "HOLD"
    validator_fault = True


def _reopen_controller_source(value: str | Path) -> Path:
    try:
        root = Path(value).expanduser()
        modules = (
            root / "cacheon" / "__init__.py",
            root / "cacheon" / "eval" / "b300_prepared_graph_oci.py",
        )
        carriers = (root, modules[0].parent, modules[1].parent)
        if not root.is_absolute() or root.resolve(strict=True) != root:
            raise B300PreparedGraphOCIError("controller source root is noncanonical")
        if any(
            path.is_symlink()
            or not path.is_dir()
            or path.stat().st_mode & 0o222
            or path.stat().st_mode & 0o055 != 0o055
            for path in carriers
        ):
            raise B300PreparedGraphOCIError(
                "controller source carrier is not immutable and container-readable"
            )
        if any(
            module.is_symlink()
            or not module.is_file()
            or module.stat().st_nlink != 1
            or module.stat().st_mode & 0o222
            or module.stat().st_mode & 0o044 != 0o044
            for module in modules
        ):
            raise B300PreparedGraphOCIError(
                "controller graph module is not immutable and container-readable"
            )
    except (OSError, RuntimeError, TypeError) as exc:
        raise B300PreparedGraphOCIError(f"controller source is unavailable: {exc}") from None
    return root


def _remaining(deadline: float, manager: OCIProcessManager, stage: str) -> float:
    if type(deadline) is not float or not math.isfinite(deadline):
        raise B300PreparedGraphOCIError("deadline must be a finite absolute float")
    remaining = deadline - float(manager.clock())
    if not math.isfinite(remaining) or remaining <= 0:
        raise B300PreparedGraphOCIError(f"graph deadline expired during {stage}")
    return remaining


def _write_request(root: Path, payload: bytes, *, uid: int, gid: int) -> Path:
    if not 1 <= len(payload) <= _MAX_REQUEST_BYTES:
        raise B300PreparedGraphOCIError("graph request exceeds its byte bound")
    root.mkdir(mode=0o700)
    path = root / "request.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        os.fchown(descriptor, uid, gid)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise B300PreparedGraphOCIError("graph request write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chown(root, uid, gid)
    if path.read_bytes() != payload:
        raise B300PreparedGraphOCIError("staged graph request did not reopen exactly")
    return path


def _graph_argv(
    lease,
    resolved,
    preflight,
    publication,
    cache,
    seccomp,
    request,
    policy,
    controller_source,
):
    runtime = resolved.spec
    resources = lease.resource_root  # exact lease containment already manager-owned
    if request.parent != resources / "request":
        raise B300PreparedGraphOCIError("graph request escaped its lease")
    artifact_root = (
        f"{CONTAINER_ARTIFACT_BASE}/{publication.build_spec_digest[:2]}/"
        f"{publication.build_spec_digest}"
    )
    physical = resolved.physical_hardware.physical_gpu_ids
    if len(physical) != 4:
        raise B300PreparedGraphOCIError("graph execution requires exact TP4 devices")
    gpu_request = f'"device={",".join(physical)}"'

    env = {
        "CACHEON_ENGINE_TREE_DIGEST": runtime.tree_digest,
        "CACHEON_ENGINE_WORKER": "1",
        "CACHEON_EXTERNAL_NO_EGRESS": "1",
        "CACHEON_NATIVE_ARTIFACT_PUBLICATION_DIGEST": publication.publication_digest,
        "CACHEON_NATIVE_ARTIFACT_ROOT": artifact_root,
        "CACHEON_NATIVE_BUILD_SPEC_DIGEST": resolved.native_build_spec.digest,
        "CACHEON_PREBUILT_ARTIFACTS": "1",
        "CACHEON_REBUILD_PHASE": "load",
        "CACHEON_TARGET_GPU_ARCH": resolved.native_build_spec.target_architecture,
        "CUDA_CACHE_PATH": f"{CONTAINER_CACHE}/cuda",
        "HOME": f"{CONTAINER_CACHE}/home",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TMPDIR": "/tmp",
        "TORCH_EXTENSIONS_DIR": f"{CONTAINER_CACHE}/torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": f"{CONTAINER_CACHE}/torchinductor",
        "TRITON_CACHE_DIR": f"{CONTAINER_CACHE}/triton",
        "TRITON_HOME": f"{CONTAINER_CACHE}/triton-home",
        "XDG_CACHE_HOME": f"{CONTAINER_CACHE}/xdg",
    }
    if resolved.native_compile_profile is not None:
        env[CUTE_COMPILE_PROFILE_DIGEST_ENV] = resolved.native_compile_profile.digest

    argv = [
        *lease.run_prefix(preflight.docker_binary),
        "--rm",
        "--init",
        "--pull=never",
        f"--platform={preflight.oci_platform}",
        "--runtime=runc",
        "--network=none",
        "--read-only",
        "--ipc=private",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        f"--security-opt=seccomp={seccomp}",
        f"--user={policy.uid}:{policy.gid}",
        f"--cpus={policy.cpu_millis / 1000:g}",
        *(
            ()
            if policy.cpuset_cpus is None
            else (
                f"--cpuset-cpus={policy.cpuset_cpus}",
                f"--cpuset-mems={policy.cpuset_mems}",
            )
        ),
        f"--memory={policy.memory_bytes}",
        f"--memory-swap={policy.memory_bytes}",
        f"--pids-limit={policy.pids_limit}",
        f"--ulimit=nofile={policy.nofile_limit}:{policy.nofile_limit}",
        "--ulimit=core=0:0",
        (
            "--tmpfs=/tmp:rw,nosuid,nodev,exec,"
            f"size={policy.tmpfs_bytes},uid={policy.uid},gid={policy.gid},mode=0700"
        ),
        f"--shm-size={policy.shm_bytes}",
        f"--gpus={gpu_request}",
        "--stop-timeout=1",
        "--no-healthcheck",
        "--log-driver=none",
        "--workdir=/tmp",
        build_bind_mount_arg(
            resolved.materialized_tree_root,
            CONTAINER_TREE,
            readonly=True,
        ),
        build_bind_mount_arg(publication.root, artifact_root, readonly=True),
        build_bind_mount_arg(cache, CONTAINER_CACHE, readonly=False),
        build_bind_mount_arg(request.parent, _REQUEST_DIR, readonly=True),
        build_bind_mount_arg(controller_source, _CONTROLLER_ROOT, readonly=True),
    ]
    argv.extend(f"--env={key}={env[key]}" for key in sorted(env))
    argv.extend(
        (
            f"--entrypoint={policy.container_python}",
            preflight.local_image_id,
            "-I",
            "-c",
            _CONTROLLER_BOOTSTRAP,
            "worker",
        )
    )
    return tuple(argv)


class B300PreparedGraphOCIExecutor:
    """Run a path-free prepared graph request under one sealed executor policy."""

    def __init__(
        self,
        config: OCIBackendConfig,
        device_policy: DeviceStatePolicy,
        *,
        controller_source_root: str | Path,
        controller_distribution_digest: str,
        manager: OCIProcessManager,
        capture_runner: CaptureRunner = bounded_argv_runner,
    ) -> None:
        if (
            type(config) is not OCIBackendConfig
            or type(device_policy) is not DeviceStatePolicy
        ):
            raise B300PreparedGraphOCIError("graph policies are not exactly typed")
        if type(manager) is not OCIProcessManager or not callable(capture_runner):
            raise B300PreparedGraphOCIError("graph lifecycle authority is invalid")
        if (manager.docker_binary, manager.executor_id) != (
            config.prebuild.docker_binary, config.prebuild.executor_id
        ):
            raise B300PreparedGraphOCIError("graph manager differs from backend config")
        try:
            self.controller_distribution_digest = require_sha256_hex(
                controller_distribution_digest, field="controller_distribution_digest"
            )
        except (TypeError, ValueError) as exc:
            raise B300PreparedGraphOCIError(str(exc)) from None
        self.controller_source_root = _reopen_controller_source(controller_source_root)
        self.config = config
        self.device_policy = device_policy
        self.manager = manager
        self.capture_runner = capture_runner
        self.device_guard = DeviceStateGuard(device_policy, clock=manager.clock)

    def execute(self, request, prepared: PreparedCandidateRuntime, *, deadline: float):
        from cacheon.eval.b300_prepared_graph_probe import PreparedGraphProbeRequest

        if (
            type(request) is not PreparedGraphProbeRequest
            or type(prepared) is not PreparedCandidateRuntime
        ):
            raise B300PreparedGraphOCIError("graph request/runtime are not exact types")
        if (
            type(prepared.arm) is not MarginalArmPlan
            or request.launch != prepared.launch
        ):
            raise B300PreparedGraphOCIError("request differs from prepared runtime")
        expected_binding = (
            prepared.arm.digest,
            prepared.binding.launch_binding.native_build_spec.digest,
        )
        request_binding = (
            request.binding.prepared_arm_digest,
            request.binding.native_build_spec_digest,
        )
        if request.launch.controller_distribution_digest != self.controller_distribution_digest:
            raise B300PreparedGraphOCIError("graph controller distribution differs from executor authority")
        if (
            request_binding != expected_binding
            or request.policy.tp_size != 4
            or request.launch.hardware.visible_gpu_count != 4
        ):
            raise B300PreparedGraphOCIError("graph binding is not the prepared TP4 arm")
        if not self.manager.transaction_lock.acquire(blocking=False):
            raise B300PreparedGraphOCIError("graph executor is already active")
        try:
            try:
                return self._execute_locked(request, prepared, deadline)
            except B300PreparedGraphOCIError:
                raise
            except Exception as exc:
                raise B300PreparedGraphOCIError(
                    f"graph OCI execution was incomplete: {exc}"
                ) from exc
        finally:
            self.manager.transaction_lock.release()

    def _execute_locked(self, request, prepared, deadline):
        _remaining(deadline, self.manager, "recovery")
        self.manager.recover_stale()
        binding = prepared.binding.launch_binding
        resolved = resolve_engine_launch(request.launch, binding)
        validate_device_state_policy(
            self.device_policy,
            logical_hardware=request.launch.hardware,
            physical_hardware=resolved.physical_hardware,
        )
        preflight = binding.runtime_preflight_receipt
        policy = self.config.runtime
        if type(preflight) is not RuntimePreflightReceipt or (
            preflight.docker_binary,
            preflight.uid,
            preflight.gid,
            preflight.python_executable,
        ) != (
            self.manager.docker_binary,
            policy.uid,
            policy.gid,
            policy.container_python,
        ):
            raise B300PreparedGraphOCIError("preflight differs from runtime policy")
        if (
            request.launch.resource_policy_digest
            != self.config.prebuild.policy.resource_policy_digest
        ):
            raise B300PreparedGraphOCIError("graph launch differs from resource policy")
        prebuild = run_oci_prebuild(
            request.launch,
            binding,
            self.config.prebuild,
            manager=self.manager,
            limits=self.config.native_limits,
            deadline=deadline,
        )
        if (
            prebuild.launch_digest != request.launch.digest
            or prebuild.discovery_overlay_identity_digest is not None
        ):
            raise B300PreparedGraphOCIError("native publication differs from the arm")
        publication = reopen_native_artifact(
            prebuild.publication.root,
            expected_build_spec_digest=resolved.native_build_spec.digest,
            expected_publication_digest=prebuild.publication.publication_digest,
            limits=self.config.native_limits,
        )
        launch_id = "graph-" + secrets.token_hex(10)
        self.device_guard.before_launch(launch_id, deadline=deadline)
        lease = None
        primary = None
        artifact = None
        cleanup = []
        try:
            lease = self.manager.register(
                lease_id=launch_id,
                container_name="cacheon-" + launch_id,
                mount_relpaths=("runtime-cache",),
                stage_relpaths=("seccomp.json", "request"),
            )
            cache, seccomp, request_root = lease.mount_paths[0], *lease.stage_paths
            stage_seccomp_profile(
                self.config.prebuild.seccomp_profile,
                seccomp,
                expected_digest=request.launch.seccomp_policy_digest,
            )
            self.manager.mount_tmpfs(
                lease,
                cache,
                size_bytes=policy.cache_bytes,
                inode_limit=policy.cache_inodes,
                uid=policy.uid,
                gid=policy.gid,
                executable=True,
            )
            request_path = _write_request(
                request_root, request.canonical_bytes, uid=policy.uid, gid=policy.gid
            )
            resolved = resolve_engine_launch(request.launch, binding)
            publication = reopen_native_artifact(
                publication.root,
                expected_build_spec_digest=resolved.native_build_spec.digest,
                expected_publication_digest=publication.publication_digest,
                limits=self.config.native_limits,
            )
            argv = _graph_argv(
                lease,
                resolved,
                preflight,
                publication,
                cache,
                seccomp,
                request_path,
                policy,
                _reopen_controller_source(self.controller_source_root),
            )
            timeout = _remaining(deadline, self.manager, "worker") - float(
                self.device_policy.drain_timeout_s
            )
            if timeout <= 0:
                raise B300PreparedGraphOCIError("worker lacks mandatory drain time")
            result = self.manager.run_capture(
                lease,
                argv,
                timeout_s=timeout,
                max_stdout_bytes=_MAX_ARTIFACT_BYTES,
                max_stderr_bytes=_MAX_STDERR_BYTES,
                capture_runner=self.capture_runner,
            )
            if result.returncode != 0:
                stderr_tail = result.stderr[-(4 << 10) :].decode(
                    "utf-8", errors="backslashreplace"
                )
                raise B300PreparedGraphOCIError(
                    f"worker exited {result.returncode}; stderr tail={stderr_tail!a}"
                )
            artifact = B300QualificationGraphArtifact.from_canonical_bytes(
                result.stdout
            )
            if (
                artifact.binding != request.binding
                or artifact.verification_policy_digest
                != request.policy.verification_policy_digest
                or artifact.expected_graph_replays
                != request.policy.expected_graph_replays
            ):
                raise B300PreparedGraphOCIError("worker returned a foreign artifact")
        except Exception as exc:
            primary = exc
        finally:
            if lease is not None:
                try:
                    self.manager.release(lease)
                except BaseException as exc:
                    cleanup.append(exc)
            for action in (
                lambda: self.device_guard.after_launch(launch_id, deadline=deadline),
                self.manager.prove_quiescent,
            ):
                try:
                    action()
                except BaseException as exc:
                    cleanup.append(exc)
        if cleanup:
            raise B300PreparedGraphOCIError(
                "graph cleanup/device quiescence was not proven"
            ) from (primary or cleanup[0])
        if primary is not None:
            raise B300PreparedGraphOCIError(
                f"graph OCI execution was incomplete: {primary}"
            ) from primary
        return artifact


def _worker() -> int:
    from cacheon.eval.b300_prepared_graph_probe import (
        PreparedGraphProbeRequest,
        execute_prepared_graph_probe,
    )
    payload = Path(CONTAINER_REQUEST).read_bytes()
    request = PreparedGraphProbeRequest.from_canonical_bytes(payload)
    output = os.dup(sys.stdout.fileno())
    prior_sys_path = list(sys.path)

    try:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.path.insert(0, CONTAINER_TREE)
        artifact = execute_prepared_graph_probe(request, CONTAINER_TREE)
        sys.stdout.flush()
        os.dup2(output, sys.stdout.fileno())
        view = memoryview(artifact.canonical_bytes)
        while view:
            written = os.write(output, view)
            if written <= 0:
                raise B300PreparedGraphOCIError("graph artifact write stalled")
            view = view[written:]
    finally:
        sys.path[:] = prior_sys_path
        os.dup2(output, sys.stdout.fileno())
        os.close(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    if list(sys.argv[1:] if argv is None else argv) != ["worker"]:
        raise B300PreparedGraphOCIError("worker accepts only its fixed entrypoint")
    return _worker()


if __name__ == "__main__":  # pragma: no cover - fixed container entrypoint
    raise SystemExit(main())


__all__ = ["B300PreparedGraphOCIError", "B300PreparedGraphOCIExecutor"]

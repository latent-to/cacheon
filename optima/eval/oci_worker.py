"""Trusted entry point inside one isolated OCI engine-launch container.

Only this process owns the wall clock and result HMAC key.  It reads a strict
request/key frame from stdin, moves the key into anonymous ``MADV_DONTFORK`` memory,
closes stdin, verifies the live sandbox, and only then imports/starts SGLang.  The
fixed dispatch below is validator code; request bytes cannot select an import,
callable, module, or arbitrary bundle path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import mmap
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from optima.eval.oci_protocol import (
    AUTH_KEY_BYTES,
    CONTAINER_ARTIFACT_PATH,
    CONTAINER_BUNDLE_PATH,
    CONTAINER_JIT_PATH,
    CONTAINER_MODEL_PATH,
    CONTAINER_OUTPUT_PATH,
    CONTAINER_RESULT_PATH,
    CONTAINER_SOURCE_PATH,
    OCILaunchRequest,
    OCIProtocolError,
    environment_fingerprint,
    read_stdin_frame,
    topology_fingerprint,
)


class OCIWorkerError(RuntimeError):
    """The container/worker does not satisfy the production trust boundary."""


class _SecretVault:
    """Anonymous key storage excluded from fork children and wiped on close."""

    def __init__(self, key: bytearray, *, require_dontfork: bool) -> None:
        if not isinstance(key, bytearray) or len(key) != AUTH_KEY_BYTES:
            raise OCIWorkerError("worker received an invalid HMAC key buffer")
        self._memory = mmap.mmap(-1, AUTH_KEY_BYTES)
        self._closed = False
        try:
            self._memory[:] = key
            key[:] = b"\x00" * len(key)
            dontfork = getattr(mmap, "MADV_DONTFORK", 10)
            madvise = getattr(self._memory, "madvise", None)
            if madvise is None:
                if require_dontfork:
                    raise OCIWorkerError("Python mmap lacks MADV_DONTFORK support")
            else:
                try:
                    madvise(dontfork)
                except (OSError, ValueError) as exc:
                    if require_dontfork:
                        raise OCIWorkerError(
                            f"could not exclude HMAC memory from fork children: {exc}"
                        ) from None
        except BaseException:
            self.close()
            raise

    def reveal(self) -> bytes:
        if self._closed:
            raise OCIWorkerError("HMAC key vault is closed")
        return bytes(self._memory[:])

    def close(self) -> None:
        if not self._closed:
            try:
                self._memory[:] = b"\x00" * AUTH_KEY_BYTES
            finally:
                self._memory.close()
                self._closed = True


def _is_read_only(path: str) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return False


def _assert_live_sandbox(request: OCILaunchRequest) -> None:
    """Fail closed if the runtime did not actually apply the command policy."""

    # Reuse the validator's live Linux checks: only lo/routes, blocked egress, no
    # privilege escalation, a restricted capability bounding set, and read-only root.
    from optima.eval._launch import (
        _egress_is_blocked,
        _loopback_is_up,
        _network_namespace_is_loopback_only,
        _process_sandbox_is_hardened,
    )

    if not _loopback_is_up():
        raise OCIWorkerError("isolated OCI loopback is down; SGLang IPC cannot start")
    if not _network_namespace_is_loopback_only() or not _egress_is_blocked():
        raise OCIWorkerError("OCI worker is not in a loopback-only no-egress namespace")
    if not _process_sandbox_is_hardened():
        raise OCIWorkerError(
            "OCI worker lacks read-only-root/cap-drop/no-new-privileges hardening"
        )
    read_only_inputs = [
        CONTAINER_SOURCE_PATH,
        CONTAINER_MODEL_PATH,
        CONTAINER_ARTIFACT_PATH,
    ]
    if request.active:
        read_only_inputs.append(CONTAINER_BUNDLE_PATH)
    bad = [path for path in read_only_inputs if not _is_read_only(path)]
    if bad:
        raise OCIWorkerError("OCI input mounts are not read-only: " + ", ".join(bad))
    for path in (CONTAINER_JIT_PATH, CONTAINER_OUTPUT_PATH):
        if not Path(path).is_dir() or _is_read_only(path):
            raise OCIWorkerError(f"OCI private writable mount is unavailable: {path}")
    if request.active:
        from optima.manifest import load_manifest

        manifest = load_manifest(CONTAINER_BUNDLE_PATH)
        if manifest.dep_patches:
            overlay_sources = f"{CONTAINER_JIT_PATH}/dep_overlay/v2"
            if not Path(overlay_sources).is_dir() or not _is_read_only(overlay_sources):
                raise OCIWorkerError(
                    "dependency-patch candidate lacks a nested read-only overlay cache"
                )


def attest_runtime(
    *,
    version_reader: Callable[[str], str] | None = None,
    command_reader: Callable[[list[str]], str] | None = None,
) -> dict[str, Any]:
    """Attest score-affecting container state before candidate code is imported."""

    if version_reader is None:
        import importlib.metadata

        version_reader = importlib.metadata.version
    if command_reader is None:
        def command_reader(argv: list[str]) -> str:
            return subprocess.run(
                argv,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=20,
                shell=False,
            ).stdout

    expected_version = os.environ.get(
        "OPTIMA_OCI_EXPECTED_SGLANG_VERSION", ""
    ).strip()
    expected_arch = os.environ.get("OPTIMA_OCI_EXPECTED_GPU_ARCH", "").strip()
    try:
        expected_count = int(os.environ["OPTIMA_OCI_EXPECTED_GPU_COUNT"])
    except (KeyError, TypeError, ValueError):
        raise OCIWorkerError("OCI GPU-count attestation policy is missing") from None
    installed_version = version_reader("sglang")
    if installed_version != expected_version:
        raise OCIWorkerError(
            f"installed sglang {installed_version!r} != arena {expected_version!r}"
        )

    env_keys_text = os.environ.get("OPTIMA_OCI_ATTEST_ENV_KEYS", "")
    env_keys = env_keys_text.split(",") if env_keys_text else []
    if (
        not env_keys
        or len(env_keys) > 512
        or len(set(env_keys)) != len(env_keys)
        or any(re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", key) is None for key in env_keys)
    ):
        raise OCIWorkerError("OCI environment attestation key set is malformed")
    missing_env = [key for key in env_keys if key not in os.environ]
    if missing_env:
        raise OCIWorkerError(f"OCI environment is missing policy keys: {missing_env}")
    observed_env_sha = environment_fingerprint(
        {key: os.environ[key] for key in env_keys}
    )
    expected_env_sha = os.environ.get("OPTIMA_OCI_ATTEST_ENV_SHA256", "")
    if observed_env_sha != expected_env_sha:
        raise OCIWorkerError(
            f"OCI environment fingerprint {observed_env_sha} != {expected_env_sha}"
        )

    from optima.arenas import (
        huggingface_model_manifest,
        referee_source_digest,
        verify_model_content_seal,
    )

    observed_source = referee_source_digest(Path(CONTAINER_SOURCE_PATH) / "optima")
    expected_source = os.environ.get(
        "OPTIMA_OCI_EXPECTED_REFEREE_SOURCE_DIGEST", ""
    )
    if observed_source != expected_source:
        raise OCIWorkerError(
            f"referee source digest {observed_source} != arena {expected_source}"
        )
    model_revision, model_manifest = huggingface_model_manifest(CONTAINER_MODEL_PATH)
    expected_revision = os.environ.get("OPTIMA_OCI_EXPECTED_MODEL_REVISION", "")
    expected_manifest = os.environ.get(
        "OPTIMA_OCI_EXPECTED_MODEL_MANIFEST_DIGEST", ""
    )
    expected_content = os.environ.get(
        "OPTIMA_OCI_EXPECTED_MODEL_CONTENT_DIGEST", ""
    )
    if model_revision != expected_revision or model_manifest != expected_manifest:
        raise OCIWorkerError(
            "model download receipt differs from arena "
            f"(revision={model_revision!r}, manifest={model_manifest!r})"
        )
    verify_model_content_seal(
        CONTAINER_MODEL_PATH,
        expected_digest=expected_content,
        verify_bytes=False,
    )

    try:
        from optima.runtime_overlay import (
            RuntimeFileOverlay,
            RuntimeOverlayError,
            normalize_runtime_overlays,
            runtime_overlay_fingerprint,
            verify_runtime_overlay_targets,
            verify_runtime_overlays,
        )

        raw_overlays = json.loads(
            os.environ.get("OPTIMA_OCI_RUNTIME_OVERLAYS_JSON", "")
        )
        if not isinstance(raw_overlays, list):
            raise ValueError("overlay policy is not a list")
        overlays = normalize_runtime_overlays(
            RuntimeFileOverlay(**item) for item in raw_overlays
        )
        expected_overlay_count = int(
            os.environ["OPTIMA_OCI_EXPECTED_RUNTIME_OVERLAY_COUNT"]
        )
        expected_overlay_sha = os.environ[
            "OPTIMA_OCI_EXPECTED_RUNTIME_OVERLAYS_SHA256"
        ]
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RuntimeOverlayError,
    ) as exc:
        raise OCIWorkerError(
            f"runtime overlay attestation policy is malformed: {exc}"
        ) from None
    if (
        len(overlays) != expected_overlay_count
        or runtime_overlay_fingerprint(overlays) != expected_overlay_sha
    ):
        raise OCIWorkerError("runtime overlay policy fingerprint/count mismatch")
    try:
        verify_runtime_overlays(CONTAINER_MODEL_PATH, overlays)
        verify_runtime_overlay_targets(overlays)
    except Exception as exc:
        raise OCIWorkerError(f"runtime overlay attestation failed: {exc}") from None

    compute_caps = command_reader([
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader,nounits",
    ])
    cap_lines = [line.strip() for line in compute_caps.splitlines() if line.strip()]
    architectures: list[str] = []
    for cap in cap_lines:
        match = re.fullmatch(r"([0-9]{1,2})\.([0-9])", cap)
        if match is None:
            raise OCIWorkerError(f"unrecognized NVIDIA compute capability {cap!r}")
        architectures.append(f"sm{match.group(1)}{match.group(2)}")
    if len(architectures) != expected_count or set(architectures) != {expected_arch}:
        raise OCIWorkerError(
            "visible GPU count/architecture differs from arena: "
            f"count={len(architectures)}/{expected_count}, arch={architectures}/{expected_arch}"
        )
    topology_text = command_reader(["nvidia-smi", "topo", "-m"])
    topology_sha = topology_fingerprint(topology_text)
    expected_topology = os.environ.get(
        "OPTIMA_OCI_EXPECTED_TOPOLOGY_SHA256", ""
    ).strip()
    if expected_topology and topology_sha != expected_topology:
        raise OCIWorkerError(
            f"GPU topology fingerprint {topology_sha} != arena {expected_topology}"
        )
    return {
        "verified": True,
        "sglang_version": installed_version,
        "referee_source_digest": observed_source,
        "model_revision": model_revision,
        "model_manifest_digest": model_manifest,
        "model_content_digest": expected_content,
        "environment_sha256": observed_env_sha,
        "gpu_count": len(architectures),
        "gpu_architectures": architectures,
        "topology_sha256": topology_sha,
    }


@contextlib.contextmanager
def _launched_system_engine(cfg):
    """Launch a system-overlay candidate without component receipt semantics."""

    import shutil
    import tempfile

    from optima import receipts, seam
    from optima.eval._launch import (
        _wait_gpu_drain,
        engine_kwargs,
        env,
        prepare_candidate_environment,
    )
    from optima.system_overlay import driver_module_is_stock
    from optima.system_patch import read_validated_system_overlay

    bundle = CONTAINER_BUNDLE_PATH
    target = os.environ["OPTIMA_SYSTEM_COMPETITION_TARGET"]
    arena = os.environ["OPTIMA_SYSTEM_ARENA"]
    cache_root = Path(os.environ["OPTIMA_SYSTEM_OVERLAY_ROOT"])
    expected_key = os.environ["OPTIMA_SYSTEM_EXPECTED_CACHE_KEY"]
    _identity, _stamp, overlay_dest = read_validated_system_overlay(
        bundle,
        competition_target=target,
        arena_name=arena,
        cache_root=cache_root,
        require_read_only=True,
        read_only_check=lambda path: _is_read_only(str(path)),
    )

    seam.mark_driver()
    prepare_candidate_environment(cfg, bundle_path=bundle, active=True)
    receipt_dir = tempfile.mkdtemp(prefix="optima_system_receipts_")
    try:
        # Component dispatch remains inactive.  The independently armed process-role
        # hook installs the validated system overlay only in exact scheduler children.
        with env(
            OPTIMA_BUNDLE_PATH="",
            OPTIMA_ACTIVE="0",
            OPTIMA_FRAMEWORK_MODE="1",
            OPTIMA_SEAM_RECEIPT_DIR=receipt_dir,
            SGLANG_PLUGINS="optima",
        ):
            import sglang as sgl

            driver_file = getattr(sgl, "__file__", None)
            if not driver_module_is_stock(sgl, overlay_dest):
                raise OCIWorkerError(
                    f"timing driver imported candidate SGLang overlay: {driver_file}"
                )
            _wait_gpu_drain()
            resolved_kwargs = engine_kwargs(cfg, active=True)
            engine = sgl.Engine(**resolved_kwargs)
            try:
                active = receipts.require(
                    receipt_dir,
                    "system_active",
                    context="system candidate engine launch",
                )
                expected_origin = str(
                    (overlay_dest / "site" / "sglang" / "__init__.py").resolve()
                )
                malformed = [
                    receipt
                    for receipt in active
                    if receipt.get("target") != target
                    or receipt.get("arena") != arena
                    or receipt.get("cache_key") != expected_key
                    or receipt.get("module_origin") != expected_origin
                ]
                pids = {
                    int(receipt["pid"])
                    for receipt in active
                    if isinstance(receipt.get("pid"), int)
                    and not isinstance(receipt.get("pid"), bool)
                    and int(receipt["pid"]) > 0
                }
                expected_members = int(resolved_kwargs.get("tp_size", 1) or 1)
                if malformed or len(pids) < expected_members:
                    raise OCIWorkerError(
                        "system candidate has incomplete/mismatched scheduler activation "
                        f"(members={len(pids)}/{expected_members}, malformed={malformed})"
                    )
                if receipts.collect(receipt_dir, "active"):
                    raise OCIWorkerError(
                        "system candidate unexpectedly activated component registry"
                    )
                yield engine, active
            finally:
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    from sglang.srt.utils import kill_process_tree

                    kill_process_tree(os.getpid(), include_parent=False)
                except Exception:  # noqa: BLE001
                    pass
                # There is intentionally no component completed/fallback gate here:
                # the product is the independently validated scheduler overlay. Its
                # load evidence is system_active; controller-side B/C/B' output
                # fidelity is the authoritative correctness gate.
    finally:
        shutil.rmtree(receipt_dir, ignore_errors=True)


def _run_system_launch(cfg, prompt_batches):
    from optima.eval.throughput_kl import _measure

    with _launched_system_engine(cfg) as (engine, _receipts):
        return _measure(engine, prompt_batches, cfg)


def execute_engine_launch(request: OCILaunchRequest) -> Any:
    """Fixed worker dispatch for one timed or untimed quality engine launch."""

    system_manifest = None
    if request.active:
        from optima.manifest import load_manifest

        manifest = load_manifest(CONTAINER_BUNDLE_PATH)
        if manifest.system is not None:
            system_manifest = manifest
            arena_name = os.environ.get("OPTIMA_OCI_ARENA_NAME", "").strip()
            competition_target = os.environ.get(
                "OPTIMA_OCI_COMPETITION_TARGET", ""
            ).strip()
            cache_root = os.environ.get(
                "OPTIMA_OCI_SYSTEM_OVERLAY_ROOT", ""
            ).strip()
            if not arena_name or not competition_target or not cache_root:
                raise OCIWorkerError(
                    "system candidate lacks profile-owned arena/target/artifact policy"
                )
            from optima.system_patch import system_launch_environment

            # Called inside the container so bundle/cache paths and DRIVER_PID are
            # exactly those the scheduler-role bootstrap will independently verify.
            armed = system_launch_environment(
                CONTAINER_BUNDLE_PATH,
                competition_target=competition_target,
                arena_name=arena_name,
                cache_root=Path(cache_root),
            )
            os.environ.update(armed)

    # ``sitecustomize`` already did this during normal OCI interpreter startup.
    # Import explicitly as a fail-safe for an image whose Python site policy differs;
    # spawned schedulers still receive the same read-only sitecustomize PYTHONPATH.
    import optima.bootstrap  # noqa: F401

    from optima.eval.throughput_kl import (
        EvalConfig,
        _run_launch,
        _run_quality_launch,
    )

    cfg = EvalConfig(**dict(request.eval_config))
    prompt_batches = [list(batch) for batch in request.prompt_batches]
    if system_manifest is not None:
        if not cfg.framework_mode:
            raise OCIWorkerError(
                "system candidate requires framework/external controller fidelity mode"
            )
        if request.mode == "candidate_audit":
            raise OCIWorkerError(
                "system candidate has no component in-engine audit lane; use external B/C/B'"
            )
        return _run_system_launch(cfg, prompt_batches)
    if request.mode == "candidate_audit":
        return _run_quality_launch(
            cfg, prompt_batches, bundle_path=CONTAINER_BUNDLE_PATH
        )
    return _run_launch(
        cfg,
        prompt_batches,
        bundle_path=CONTAINER_BUNDLE_PATH if request.active else "",
        active=request.active,
    )


def run_worker(
    *,
    result_path: str,
    stdin_fd: int = 0,
    execute: Callable[[OCILaunchRequest], Any] = execute_engine_launch,
    verify_sandbox: bool = True,
    require_dontfork: bool | None = None,
) -> int:
    """Read/close stdin, execute the fixed hook, and authenticate one outcome."""

    from optima.ipc import LaunchOutcome, dump_authenticated_file

    if require_dontfork is None:
        require_dontfork = sys.platform.startswith("linux")
    request: OCILaunchRequest
    key_buffer: bytearray
    request, key_buffer = read_stdin_frame(stdin_fd)
    # This happens before sandbox verification, throughput imports, or ``execute``.
    # SGLang and every candidate scheduler descendant therefore inherit no key pipe.
    os.close(stdin_fd)
    vault = _SecretVault(key_buffer, require_dontfork=require_dontfork)
    try:
        try:
            if verify_sandbox:
                _assert_live_sandbox(request)
                runtime_attestation = attest_runtime()
            else:
                runtime_attestation = {"verified": False, "test_only_bypass": True}
            value = execute(request)
            outcome = LaunchOutcome(
                value={
                    "result": value,
                    "runtime_attestation": runtime_attestation,
                },
                error=None,
            )
        except BaseException:  # noqa: BLE001 - authenticated failure, never silent pass
            error = traceback.format_exc()
            outcome = LaunchOutcome(value=None, error=error[-64_000:])
        key = vault.reveal()
        try:
            dump_authenticated_file(
                result_path,
                outcome,
                key=key,
                nonce=request.nonce,
            )
        finally:
            # ``bytes`` cannot be zeroized, but it exists only after engine teardown;
            # the long-lived pre-spawn copy remained in MADV_DONTFORK storage.
            del key
        # The authenticated envelope, not process status, carries worker failures.
        # Returning zero after a durable envelope lets the controller surface that
        # bounded traceback; a zero exit with a missing/invalid envelope still fails.
        return 0
    finally:
        vault.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m optima.eval.oci_worker")
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    if args.result != CONTAINER_RESULT_PATH:
        raise OCIProtocolError(
            f"worker result path must be fixed at {CONTAINER_RESULT_PATH!r}"
        )
    return run_worker(result_path=args.result)


if __name__ == "__main__":
    raise SystemExit(main())

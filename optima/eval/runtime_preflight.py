"""Trusted, candidate-free stock runtime preflight.

This module attests a local immutable validator image before any miner bundle,
model, source tree, artifact directory, or GPU is exposed.  It performs exactly
two argv-only operations:

1. inspect the requested ``name@sha256:...`` image and bind that RepoDigest to
   one local content-addressed image ID;
2. run that local image ID with no network, mounts, GPU runtime, or capabilities
   and emit one small, exact-schema standard-library-only runtime receipt.

Every failure is validator-owned.  Nothing here can disqualify a miner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import secrets
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Protocol, Sequence


INSPECT_SCHEMA_KEYS = frozenset({"Id", "RepoDigests", "Volumes"})
CONTAINER_RECEIPT_SCHEMA = "optima-stock-runtime-container-v1"
HOST_RECEIPT_SCHEMA = "optima-stock-runtime-preflight-v1"
MAX_INSPECT_STDOUT_BYTES = 16 * 1024
MAX_RECEIPT_STDOUT_BYTES = 16 * 1024
MAX_STDERR_BYTES = 8 * 1024

_IMAGE = re.compile(
    r"[a-z0-9][a-z0-9._/:+-]{0,255}@sha256:[0-9a-f]{64}\Z"
)
_SHA_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+-]{0,63}\Z")
_SMALL_TEXT = re.compile(r"[^\x00\r\n]{0,255}\Z")
_CONTAINER_NAME = re.compile(r"optima-stock-preflight-[0-9a-f]{20}\Z")
_PACKAGE_NAMES = (
    "cuda-python",
    "flashinfer-python",
    "nvidia-cuda-runtime-cu12",
    "torch",
    "triton",
)

_INSPECT_FORMAT = (
    '--format={"Id":{{json .Id}},"RepoDigests":{{json .RepoDigests}},'
    '"Volumes":{{json .Config.Volumes}}}'
)

# Fixed source: no requested image/version value is interpolated into container code.
_CONTAINER_SCRIPT = r'''import ctypes.util, importlib.metadata, json, os, platform, re, sys, sysconfig
paths = sorted(set(filter(None, (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")))))
for path in paths:
    if path not in sys.path:
        sys.path.append(path)
def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()
versions = {}
for dist in importlib.metadata.distributions(path=paths):
    name = dist.metadata.get("Name")
    if not name:
        continue
    key = norm(name)
    prior = versions.get(key)
    if prior is not None and prior != dist.version:
        raise RuntimeError("conflicting installed distribution metadata: " + name)
    versions[key] = dist.version
def v(name):
    return versions.get(norm(name))
out = {
    "schema": "optima-stock-runtime-container-v1",
    "sglang_version": v("sglang"),
    "python": {
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "abi": str(sysconfig.get_config_var("SOABI") or ""),
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
    },
    "packages": {
        "cuda-python": v("cuda-python"),
        "flashinfer-python": v("flashinfer-python"),
        "nvidia-cuda-runtime-cu12": v("nvidia-cuda-runtime-cu12"),
        "torch": v("torch"),
        "triton": v("triton"),
    },
    "cuda": {
        "cudart_library": ctypes.util.find_library("cudart"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    },
}
print(json.dumps(out, sort_keys=True, separators=(",", ":")))'''


class RuntimePreflightError(RuntimeError):
    """Trusted validator image/runtime state is invalid or unavailable."""

    validator_fault = True
    retryable = False


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class Runner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> CommandResult: ...


@dataclass(frozen=True)
class RuntimePreflightConfig:
    image: str
    expected_sglang_version: str
    uid: int
    gid: int
    docker_binary: str
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or _IMAGE.fullmatch(self.image) is None:
            raise RuntimePreflightError(
                "preflight image must be immutable lowercase name@sha256"
            )
        if (
            not isinstance(self.expected_sglang_version, str)
            or _VERSION.fullmatch(self.expected_sglang_version) is None
        ):
            raise RuntimePreflightError("expected sglang version is invalid")
        for name in ("uid", "gid"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2_147_483_647:
                raise RuntimePreflightError(
                    f"preflight {name} must be a fixed nonzero integer"
                )
        if not _safe_docker_binary(self.docker_binary):
            raise RuntimePreflightError(
                "docker_binary must be an absolute normalized path ending in /docker"
            )
        if (
            type(self.timeout_s) not in (int, float)
            or not math.isfinite(float(self.timeout_s))
            or not 1.0 <= float(self.timeout_s) <= 300.0
        ):
            raise RuntimePreflightError("preflight timeout must be in [1, 300] seconds")


@dataclass(frozen=True)
class RuntimePreflightReceipt:
    schema: str
    requested_image: str
    requested_manifest_digest: str
    local_image_id: str
    repo_digests: tuple[str, ...]
    docker_binary: str
    uid: int
    gid: int
    sglang_version: str
    python_implementation: str
    python_version: str
    python_abi: str
    python_platform: str
    machine: str
    package_versions: tuple[tuple[str, str | None], ...]
    cudart_library: str | None
    cuda_visible_devices: str
    nvidia_visible_devices: str
    security_argv_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "requested_image": self.requested_image,
            "requested_manifest_digest": self.requested_manifest_digest,
            "local_image_id": self.local_image_id,
            "repo_digests": list(self.repo_digests),
            "docker_binary": self.docker_binary,
            "uid": self.uid,
            "gid": self.gid,
            "sglang_version": self.sglang_version,
            "python": {
                "implementation": self.python_implementation,
                "version": self.python_version,
                "abi": self.python_abi,
                "platform": self.python_platform,
                "machine": self.machine,
            },
            "packages": dict(self.package_versions),
            "cuda": {
                "cudart_library": self.cudart_library,
                "cuda_visible_devices": self.cuda_visible_devices,
                "nvidia_visible_devices": self.nvidia_visible_devices,
            },
            "security_argv_sha256": self.security_argv_sha256,
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("ascii")).hexdigest()


def _safe_docker_binary(value: object) -> bool:
    if (
        not isinstance(value, str)
        or len(value) > 4096
        or "\x00" in value
        or value.startswith("//")
    ):
        return False
    path = PurePosixPath(value)
    return bool(
        path.is_absolute()
        and path.name == "docker"
        and ".." not in path.parts
        and str(path) == value
        and all(re.fullmatch(r"[A-Za-z0-9._+-]+", part) for part in path.parts[1:])
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _bounded_argv_runner(
    argv: tuple[str, ...],
    *,
    timeout_s: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> CommandResult:
    """Run one shell-free argv with an absolute deadline and live output caps."""
    if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise RuntimePreflightError("runner received invalid argv")
    deadline = time.monotonic() + float(timeout_s)
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimePreflightError(f"cannot execute preflight command: {exc}") from None
    assert process.stdout is not None and process.stderr is not None
    selector = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        selector = selectors.DefaultSelector()
        selector.register(
            process.stdout,
            selectors.EVENT_READ,
            ("stdout", max_stdout_bytes),
        )
        selector.register(
            process.stderr,
            selectors.EVENT_READ,
            ("stderr", max_stderr_bytes),
        )
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            events = selector.select(remaining)
            if not events:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            for key, _ in events:
                name, limit = key.data
                chunk = os.read(key.fd, min(4096, limit + 1 - len(buffers[name])))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > limit:
                    raise RuntimePreflightError(
                        f"preflight {name} exceeded its {limit}-byte bound"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout_s)
        returncode = process.wait(timeout=remaining)
    except BaseException:
        _terminate(process)
        raise
    finally:
        if selector is not None:
            selector.close()
    return CommandResult(
        returncode=returncode,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
    )


def _strict_json(raw: bytes, *, max_bytes: int, label: str) -> object:
    if not isinstance(raw, bytes) or not raw or len(raw) > max_bytes:
        raise RuntimePreflightError(f"{label} JSON is empty or exceeds its byte bound")
    try:
        text = raw.decode("utf-8", errors="strict")

        def object_pairs(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError(f"duplicate key {key!r}")
                out[key] = value
            return out

        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RuntimePreflightError(f"{label} emitted malformed JSON: {exc}") from None


def _exact_object(value: object, keys: frozenset[str], *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimePreflightError(
            f"{label} keys/type mismatch: expected={sorted(keys)!r} actual={actual!r}"
        )
    return value


def _small_text(value: object, *, label: str, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or _SMALL_TEXT.fullmatch(value) is None
        or (not allow_empty and not value)
    ):
        raise RuntimePreflightError(f"{label} must be a bounded string")
    return value


def _inspect_argv(config: RuntimePreflightConfig) -> tuple[str, ...]:
    return (
        config.docker_binary,
        "image",
        "inspect",
        _INSPECT_FORMAT,
        config.image,
    )


def _container_argv(
    config: RuntimePreflightConfig, *, local_image_id: str, container_name: str
) -> tuple[str, ...]:
    return (
        config.docker_binary,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--runtime=runc",
        "--ipc=none",
        f"--name={container_name}",
        "--stop-timeout=1",
        "--no-healthcheck",
        f"--user={config.uid}:{config.gid}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        # Explicitly override any daemon-level unconfined default. Scored
        # containers use the stricter source-release-pinned profile; this stock,
        # mountless probe proves the daemon can install its builtin filter.
        "--security-opt=seccomp=builtin",
        "--pids-limit=32",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1.0",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--workdir=/tmp",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=CUDA_VISIBLE_DEVICES=",
        "--log-driver=none",
        "--entrypoint=python3",
        local_image_id,
        "-I",
        "-S",
        "-c",
        _CONTAINER_SCRIPT,
    )


def _new_container_name() -> str:
    try:
        name = "optima-stock-preflight-" + secrets.token_hex(10)
    except Exception as exc:
        raise RuntimePreflightError(
            f"cannot allocate trusted preflight container name: {exc}"
        ) from None
    if _CONTAINER_NAME.fullmatch(name) is None:
        raise RuntimePreflightError("trusted preflight container name is invalid")
    return name


def _cleanup_container(
    runner: Runner,
    config: RuntimePreflightConfig,
    *,
    container_name: str,
    clock: Callable[[], float],
) -> None:
    """Force-remove a daemon-owned container after a failed attached run."""
    cleanup_deadline = _clock_now(clock) + min(5.0, float(config.timeout_s))
    try:
        _invoke(
            runner,
            (
                config.docker_binary,
                "rm",
                "--force",
                "--volumes",
                container_name,
            ),
            deadline=cleanup_deadline,
            clock=clock,
            max_stdout_bytes=1024,
        )
    except RuntimePreflightError as exc:
        raise RuntimePreflightError(
            "stock preflight launch failed and forced container cleanup "
            f"could not be confirmed: {exc}"
        ) from None


def _invoke(
    runner: Runner,
    argv: tuple[str, ...],
    *,
    deadline: float,
    clock: Callable[[], float],
    max_stdout_bytes: int,
) -> CommandResult:
    remaining = deadline - _clock_now(clock)
    if not math.isfinite(remaining) or remaining <= 0:
        raise RuntimePreflightError("stock runtime preflight absolute deadline expired")
    try:
        result = runner(
            argv,
            timeout_s=remaining,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=MAX_STDERR_BYTES,
        )
    except (RuntimePreflightError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, RuntimePreflightError):
            raise
        raise RuntimePreflightError("stock runtime preflight timed out") from None
    except Exception as exc:
        raise RuntimePreflightError(f"stock runtime preflight runner failed: {exc}") from None
    if not isinstance(result, CommandResult):
        raise RuntimePreflightError("preflight runner returned an invalid result type")
    if (
        type(result.returncode) is not int
        or not isinstance(result.stdout, bytes)
        or not isinstance(result.stderr, bytes)
    ):
        raise RuntimePreflightError("preflight runner returned invalid field types")
    if result.returncode != 0:
        detail = result.stderr[:512].decode("utf-8", errors="replace")
        raise RuntimePreflightError(
            f"preflight command exited {result.returncode}: {detail}"
        )
    if result.stderr.strip():
        raise RuntimePreflightError("preflight command emitted unexpected stderr")
    if len(result.stdout) > max_stdout_bytes or len(result.stderr) > MAX_STDERR_BYTES:
        raise RuntimePreflightError("preflight runner violated output bounds")
    return result


def _clock_now(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except Exception as exc:
        raise RuntimePreflightError(f"preflight clock failed: {exc}") from None
    if not math.isfinite(value):
        raise RuntimePreflightError("preflight clock returned a non-finite value")
    return value


def run_runtime_preflight(
    config: RuntimePreflightConfig,
    *,
    runner: Runner = _bounded_argv_runner,
    clock: Callable[[], float] = time.monotonic,
) -> RuntimePreflightReceipt:
    """Attest one local stock image without candidate data, mounts, or GPUs."""
    if not isinstance(config, RuntimePreflightConfig):
        raise RuntimePreflightError("runtime preflight requires a validated config")
    started = _clock_now(clock)
    deadline = started + float(config.timeout_s)

    inspect_result = _invoke(
        runner,
        _inspect_argv(config),
        deadline=deadline,
        clock=clock,
        max_stdout_bytes=MAX_INSPECT_STDOUT_BYTES,
    )
    inspected = _exact_object(
        _strict_json(
            inspect_result.stdout,
            max_bytes=MAX_INSPECT_STDOUT_BYTES,
            label="docker image inspect",
        ),
        INSPECT_SCHEMA_KEYS,
        label="docker image inspect",
    )
    local_image_id = inspected["Id"]
    if not isinstance(local_image_id, str) or _SHA_ID.fullmatch(local_image_id) is None:
        raise RuntimePreflightError("docker inspect returned an invalid local image ID")
    repo_digests_raw = inspected["RepoDigests"]
    if (
        not isinstance(repo_digests_raw, list)
        or not 1 <= len(repo_digests_raw) <= 64
        or any(not isinstance(item, str) or _IMAGE.fullmatch(item) is None
               for item in repo_digests_raw)
        or len(set(repo_digests_raw)) != len(repo_digests_raw)
    ):
        raise RuntimePreflightError("docker inspect returned invalid RepoDigests")
    if config.image not in repo_digests_raw:
        raise RuntimePreflightError(
            "requested manifest digest is not bound to the inspected local image ID"
        )
    repo_digests = tuple(sorted(repo_digests_raw))
    volumes = inspected["Volumes"]
    if volumes not in (None, {}):
        raise RuntimePreflightError(
            "stock preflight image declares Dockerfile volumes"
        )

    container_name = _new_container_name()
    container_argv = _container_argv(
        config,
        local_image_id=local_image_id,
        container_name=container_name,
    )
    try:
        container_result = _invoke(
            runner,
            container_argv,
            deadline=deadline,
            clock=clock,
            max_stdout_bytes=MAX_RECEIPT_STDOUT_BYTES,
        )
    except RuntimePreflightError:
        _cleanup_container(
            runner,
            config,
            container_name=container_name,
            clock=clock,
        )
        raise
    container = _exact_object(
        _strict_json(
            container_result.stdout,
            max_bytes=MAX_RECEIPT_STDOUT_BYTES,
            label="stock runtime container",
        ),
        frozenset({"schema", "sglang_version", "python", "packages", "cuda"}),
        label="stock runtime container",
    )
    if container["schema"] != CONTAINER_RECEIPT_SCHEMA:
        raise RuntimePreflightError("stock runtime container schema mismatch")
    sglang_version = _small_text(
        container["sglang_version"], label="sglang_version"
    )
    if sglang_version != config.expected_sglang_version:
        raise RuntimePreflightError(
            f"installed sglang mismatch: {sglang_version!r} != "
            f"{config.expected_sglang_version!r}"
        )

    python = _exact_object(
        container["python"],
        frozenset({"implementation", "version", "abi", "platform", "machine"}),
        label="python receipt",
    )
    packages = _exact_object(
        container["packages"], frozenset(_PACKAGE_NAMES), label="package receipt"
    )
    cuda = _exact_object(
        container["cuda"],
        frozenset({
            "cudart_library", "cuda_visible_devices", "nvidia_visible_devices",
        }),
        label="CUDA receipt",
    )
    package_versions: list[tuple[str, str | None]] = []
    for name in _PACKAGE_NAMES:
        value = packages[name]
        if value is not None:
            value = _small_text(value, label=f"package {name}")
        package_versions.append((name, value))
    cudart = cuda["cudart_library"]
    if cudart is not None:
        cudart = _small_text(cudart, label="cudart_library")
    cuda_visible = _small_text(
        cuda["cuda_visible_devices"],
        label="cuda_visible_devices",
        allow_empty=True,
    )
    nvidia_visible = _small_text(
        cuda["nvidia_visible_devices"], label="nvidia_visible_devices"
    )
    if cuda_visible != "" or nvidia_visible != "void":
        raise RuntimePreflightError("stock preflight container did not preserve no-GPU policy")

    requested_manifest_digest = config.image.rsplit("@", 1)[1]
    security_argv_sha256 = hashlib.sha256(
        json.dumps(container_argv, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimePreflightReceipt(
        schema=HOST_RECEIPT_SCHEMA,
        requested_image=config.image,
        requested_manifest_digest=requested_manifest_digest,
        local_image_id=local_image_id,
        repo_digests=repo_digests,
        docker_binary=config.docker_binary,
        uid=config.uid,
        gid=config.gid,
        sglang_version=sglang_version,
        python_implementation=_small_text(
            python["implementation"], label="python implementation"
        ),
        python_version=_small_text(python["version"], label="python version"),
        python_abi=_small_text(
            python["abi"], label="python ABI", allow_empty=True
        ),
        python_platform=_small_text(python["platform"], label="python platform"),
        machine=_small_text(python["machine"], label="machine"),
        package_versions=tuple(package_versions),
        cudart_library=cudart,
        cuda_visible_devices=cuda_visible,
        nvidia_visible_devices=nvidia_visible,
        security_argv_sha256=security_argv_sha256,
    )


__all__ = [
    "CommandResult",
    "HOST_RECEIPT_SCHEMA",
    "RuntimePreflightConfig",
    "RuntimePreflightError",
    "RuntimePreflightReceipt",
    "run_runtime_preflight",
]

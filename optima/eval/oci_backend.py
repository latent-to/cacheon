"""Production-shaped OCI backend for one engine launch per container.

This is the controller-side foundation only; the evaluator/CLI chooses when to use
it.  Every call prepares a new container with an immutable image digest, no network,
a read-only root, private PID/IPC namespaces, an explicit GPU set, and only two
capabilities needed by the serving runtime.  The command is always an argv vector
executed with ``shell=False``.

The bundle, model, validator source and prebuilt artifact roots are profile-owned
read-only binds.  The only writable binds are a launch-private JIT/cache tree and a
launch-private output directory.  Thus B, C and B' cannot share compiled cache state
or smuggle state through a persistent container filesystem.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from optima.eval.device_state import (
    DeviceStateClockError,
    DeviceStateCommandError,
    DeviceStateConfigurationError,
    DeviceStateActiveReceipt,
    DeviceStateEnvelopeError,
    DeviceStateEnvelopeTimeoutError,
    DeviceStateError,
    DeviceStateGuard,
    DeviceStateParseError,
    DeviceStatePolicy,
    DeviceStatePolicyError,
    DeviceStateReceipt,
    DeviceStateTimeoutError,
    provision_gpu_configurations,
)
from optima.eval.oci_protocol import (
    AUTH_KEY_BYTES,
    AUTH_NONCE_BYTES,
    CONTAINER_ARTIFACT_PATH,
    CONTAINER_BUNDLE_PATH,
    CONTAINER_JIT_PATH,
    CONTAINER_MODEL_PATH,
    CONTAINER_OUTPUT_PATH,
    CONTAINER_RESULT_PATH,
    CONTAINER_SOURCE_PATH,
    OCILaunchRequest,
    OCIProtocolError,
    encode_stdin_frame,
    environment_fingerprint,
    env_is_safe,
    make_request,
)
from optima.eval.runtime_preflight import (
    HOST_RECEIPT_SCHEMA,
    RuntimePreflightConfig,
    RuntimePreflightError,
    RuntimePreflightReceipt,
    run_runtime_preflight,
)
from optima.runtime_overlay import (
    RuntimeFileOverlay,
    RuntimeOverlayError,
    normalize_runtime_overlays,
    runtime_overlay_fingerprint,
    verify_runtime_overlays,
)


_DIGEST_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_EXECUTABLE = re.compile(r"(?:/[A-Za-z0-9_.+-]+)+|[A-Za-z0-9_.+-]+\Z")
_SIZE = re.compile(r"[1-9][0-9]*(?:[kKmMgG])?\Z")
_CID = re.compile(r"[0-9a-f]{12,64}\Z")
_SAFE_CONTAINER_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_PROFILE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_GPU_ARCH = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_TREE_SCHEMA = "optima-oci-artifact-publication-v1"
_ARTIFACT_TREE_MANIFEST = "optima-artifact-publication.json"
_ARTIFACT_TOP_LEVEL = frozenset({
    "cuda_ext", "dep_overlay", "system_overlay", "device_cubin",
    "prebuild_receipts",
})
_MAX_ARTIFACT_FILES = 16_384
_MAX_ARTIFACT_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ARTIFACT_TOTAL_BYTES = 16 * 1024 * 1024 * 1024
_SECCOMP_PROFILE_RELATIVE = Path("optima/eval/seccomp_moby_v0_2_1.json")
_SECCOMP_PROFILE_SHA256 = (
    "de1f5327ca42b80be02daba8d39c0d087a530dc3c16f7028170fe068c9d66e61"
)


class OCIBackendError(RuntimeError):
    """Validator/profile state is invalid; abort the pass rather than charge a miner."""

    retryable = False
    validator_fault = True


class OCICandidateArtifactError(OCIBackendError):
    """Candidate build/artifact/protocol state is deterministically invalid."""

    retryable = False
    validator_fault = False


class OCIInfrastructureError(OCIBackendError):
    """Runtime/host launch failure eligible for bounded clean retry."""

    retryable = True
    validator_fault = False


class OCIWatchdogTimeout(OCIInfrastructureError):
    """A bounded OCI phase timed out before authoritative completion."""


class _FinalWarmupConditioner:
    """Bridge the outer protocol boundary to trusted-host active telemetry.

    Sampling starts before the final warmup request, runs concurrently with the
    hostile engine, and is released only after its complete response has been
    decoded by the host.  The guard then requires fresh consecutive post-release
    samples before the first timed request may be sent.
    """

    _EVENT = "final-warmup-conditioning"

    def __init__(
        self,
        *,
        guard: object,
        arm: str,
        mode: str,
        evaluation_deadline: float,
        record: Callable[[DeviceStateActiveReceipt], None],
    ) -> None:
        self._guard = guard
        self._arm = arm
        if mode not in {"baseline", "candidate", "candidate_audit"}:
            raise OCIBackendError(f"unsupported conditioned arm mode {mode!r}")
        self._mode = mode
        self._evaluation_deadline = evaluation_deadline
        self._record = record
        self._release = threading.Event()
        self._cancel = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._receipt: object | None = None
        self._error: BaseException | None = None
        self._recorded = False
        self._deadline: float | None = None

    def _run(self) -> None:
        assert self._deadline is not None
        self._started.set()
        try:
            self._receipt = self._guard.condition_active(
                self._arm,
                self._EVENT,
                deadline=self._deadline,
                release=self._release.is_set,
                wait_for_release=self._release.wait,
                cancel=self._cancel.is_set,
            )
        except BaseException as exc:  # noqa: BLE001 - relayed to controller thread
            self._error = exc

    def _join(self) -> None:
        thread = self._thread
        if thread is None:
            raise OCIBackendError(
                "final-warmup conditioning was released before it started"
            )
        assert self._deadline is not None
        remaining = max(0.0, self._deadline - time.monotonic())
        thread.join(timeout=remaining)
        if thread.is_alive():
            raise OCIInfrastructureError(
                "trusted-host final-warmup telemetry exceeded its absolute deadline"
            )

    def boundary(
        self, event: str, mode: str, batch_index: int, session_deadline: float
    ) -> None:
        del batch_index
        if mode != self._mode:
            raise OCIBackendError("final-warmup conditioner mode binding changed")
        if event == "before_final_warmup":
            if self._thread is not None:
                raise OCIBackendError("final-warmup conditioning started twice")
            self._deadline = min(self._evaluation_deadline, session_deadline)
            if self._deadline <= time.monotonic():
                raise OCIInfrastructureError(
                    "final-warmup conditioning started after its absolute deadline"
                )
            self._thread = threading.Thread(
                target=self._run,
                name=f"optima-active-{self._arm}",
                daemon=True,
            )
            self._thread.start()
            remaining = max(0.0, self._deadline - time.monotonic())
            if not self._started.wait(timeout=remaining):
                raise OCIInfrastructureError(
                    "trusted-host final-warmup telemetry thread did not start "
                    "before its absolute deadline"
                )
            return
        if event == "after_final_warmup":
            self._release.set()
            self._join()
            if self._error is not None:
                if isinstance(
                    self._error,
                    (DeviceStateEnvelopeTimeoutError, DeviceStateEnvelopeError),
                ):
                    if self._mode == "baseline":
                        raise OCIInfrastructureError(
                            "trusted stock final-warmup failed the active device "
                            f"envelope: {self._error}"
                        ) from None
                    raise OCICandidateArtifactError(
                        "candidate final-warmup failed the active device envelope: "
                        f"{self._error}"
                    ) from None
                if isinstance(
                    self._error, (DeviceStateError, DeviceStatePolicyError)
                ):
                    raise _mapped_device_state_error(
                        self._error,
                        context="trusted-host final-warmup device conditioning",
                    ) from None
                raise OCIBackendError(
                    "trusted-host final-warmup conditioner failed: "
                    f"{self._error}"
                ) from None
            if type(self._receipt) is not DeviceStateActiveReceipt:
                raise OCIBackendError(
                    "device guard returned an invalid active-conditioning receipt"
                )
            self._record(self._receipt)
            self._recorded = True
            return
        if event == "before_first_timed":
            if not self._recorded:
                raise OCIBackendError(
                    "first timed request was reached without final-warmup conditioning"
                )
            return
        raise OCIBackendError(f"unknown warmup/timed boundary event {event!r}")

    def require_complete(self) -> None:
        if not self._recorded:
            raise OCIBackendError(
                "successful OCI arm lacks final-warmup conditioning evidence"
            )

    def cancel(self) -> None:
        """Cancel and reap a monitor when the protocol arm fails early.

        Cancellation is deliberately distinct from the successful warmup-release
        boundary.  Treating an abort as a release would make the guard wait for a
        post-warmup ready envelope after the container has already disappeared.
        """

        if self._thread is None:
            return
        self._cancel.set()
        # Wake a monitor blocked in the release-aware polling wait. The monitor
        # checks cancellation before release, so this cannot mint success evidence.
        self._release.set()
        self._join()


_MODEL_VERIFY_LOCK = threading.Lock()
_VERIFIED_MODEL_BYTES: set[tuple[str, str, str]] = set()
_RUNTIME_PREFLIGHT_LOCK = threading.Lock()
_RUNTIME_PREFLIGHT_CACHE: dict[
    tuple[str, str, int, int, str], tuple[RuntimePreflightReceipt, str]
] = {}


def _resolved_docker_binary(value: str) -> str:
    """Resolve one executable once; production never relies on a mutable PATH."""
    if not isinstance(value, str) or not value or any(
        char in value for char in ("\x00", "\n", "\r")
    ):
        raise OCIBackendError("docker_binary must be a non-empty executable path/name")
    candidate = value if Path(value).is_absolute() else shutil.which(value)
    if not candidate:
        raise OCIBackendError(f"docker executable {value!r} was not found")
    try:
        resolved = Path(candidate).resolve(strict=True)
        info = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise OCIBackendError(f"docker executable does not resolve: {exc}") from None
    if (
        not resolved.is_file()
        or not stat.S_ISREG(info.st_mode)
        or resolved.name != "docker"
        or not os.access(resolved, os.X_OK)
    ):
        raise OCIBackendError(
            "docker executable must resolve to an executable regular file named 'docker'"
        )
    return str(resolved)


def _verify_runtime_preflight_identity(
    receipt: RuntimePreflightReceipt,
    *,
    image: str,
    sglang_version: str,
    worker_uid: int,
    worker_gid: int,
    docker_binary: str,
) -> None:
    requested_digest = image.rsplit("@", 1)[-1] if "@" in image else ""
    try:
        valid = bool(
            type(receipt) is RuntimePreflightReceipt
            and receipt.schema == HOST_RECEIPT_SCHEMA
            and receipt.requested_image == image
            and receipt.requested_manifest_digest == requested_digest
            and isinstance(receipt.repo_digests, tuple)
            and image in receipt.repo_digests
            and isinstance(receipt.local_image_id, str)
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}", receipt.local_image_id
            ) is not None
            and receipt.sglang_version == sglang_version
            and receipt.uid == worker_uid
            and receipt.gid == worker_gid
            and receipt.docker_binary == docker_binary
            and isinstance(receipt.security_argv_sha256, str)
            and _SHA256.fullmatch(receipt.security_argv_sha256) is not None
        )
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise OCIBackendError(
            "trusted stock runtime preflight receipt identity mismatch"
        )


def _runtime_preflight_once(
    *,
    image: str,
    sglang_version: str,
    worker_uid: int,
    worker_gid: int,
    docker_binary: str,
) -> RuntimePreflightReceipt:
    """Attest/cache one exact stock runtime identity once per validator daemon."""
    key = (image, sglang_version, worker_uid, worker_gid, docker_binary)
    with _RUNTIME_PREFLIGHT_LOCK:
        cached_entry = _RUNTIME_PREFLIGHT_CACHE.get(key)
        if cached_entry is not None:
            cached, cached_sha256 = cached_entry
            try:
                observed_sha256 = cached.sha256
            except Exception as exc:
                raise OCIBackendError(
                    f"cached stock runtime receipt is unreadable: {exc}"
                ) from None
            if observed_sha256 != cached_sha256:
                raise OCIBackendError(
                    "cached stock runtime preflight receipt integrity mismatch"
                )
            _verify_runtime_preflight_identity(
                cached,
                image=image,
                sglang_version=sglang_version,
                worker_uid=worker_uid,
                worker_gid=worker_gid,
                docker_binary=docker_binary,
            )
            return cached
        try:
            receipt = run_runtime_preflight(RuntimePreflightConfig(
                image=image,
                expected_sglang_version=sglang_version,
                uid=worker_uid,
                gid=worker_gid,
                docker_binary=docker_binary,
            ))
        except Exception as exc:
            if isinstance(exc, RuntimePreflightError):
                detail = str(exc)
            else:
                detail = f"unexpected {type(exc).__name__}: {exc}"
            raise OCIBackendError(
                f"trusted stock runtime preflight failed: {detail}"
            ) from None
        _verify_runtime_preflight_identity(
            receipt,
            image=image,
            sglang_version=sglang_version,
            worker_uid=worker_uid,
            worker_gid=worker_gid,
            docker_binary=docker_binary,
        )
        _RUNTIME_PREFLIGHT_CACHE[key] = (receipt, receipt.sha256)
        return receipt


def _mapped_device_state_error(
    exc: BaseException, *, context: str
) -> OCIBackendError:
    """Map trusted-host telemetry failures without charging candidate code."""

    if isinstance(exc, DeviceStateTimeoutError):
        return OCIInfrastructureError(f"{context}: {exc}")
    if isinstance(
        exc,
        (
            DeviceStatePolicyError,
            DeviceStateCommandError,
            DeviceStateParseError,
            DeviceStateConfigurationError,
            DeviceStateClockError,
            DeviceStateError,
        ),
    ):
        return OCIBackendError(f"{context}: {exc}")
    return OCIBackendError(f"{context} failed unexpectedly: {exc}")


def verify_production_model_once(arena, model_dir: str | os.PathLike[str]) -> Path:
    """Fully hash one arena model once per validator daemon, then use cheap seals."""

    model = _resolved_directory(model_dir, "model_dir")
    key = (str(model), str(arena.model_content_digest), str(arena.model_revision))
    # Cheap metadata/filesystem-shape receipt remains per profile/bundle.
    try:
        arena.verify_model_receipt(model, verify_bytes=False)
    except Exception as exc:
        raise OCIBackendError(f"mounted model differs from arena receipt: {exc}") from None
    if key in _VERIFIED_MODEL_BYTES:
        return model
    with _MODEL_VERIFY_LOCK:
        if key not in _VERIFIED_MODEL_BYTES:
            try:
                arena.verify_model_receipt(model, verify_bytes=True)
            except Exception as exc:
                raise OCIBackendError(
                    f"full production model byte verification failed: {exc}"
                ) from None
            _VERIFIED_MODEL_BYTES.add(key)
    return model


def _resolved_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise OCIBackendError(f"{name} must be an absolute host path")
    if any(char in str(raw) for char in ("\x00", "\n", "\r", ",")):
        raise OCIBackendError(f"{name} contains a character unsafe for OCI --mount")
    try:
        result = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OCIBackendError(f"{name} does not resolve: {exc}") from None
    if not result.is_dir():
        raise OCIBackendError(f"{name} must be a directory: {result}")
    return result


def _simple_executable(value: str, name: str) -> str:
    if not isinstance(value, str) or _EXECUTABLE.fullmatch(value) is None:
        raise OCIBackendError(f"{name} must be an argv-safe executable name/path")
    return value


def _expand_cpu_list(value: str) -> set[int]:
    if re.fullmatch(r"[0-9,-]{1,256}", value) is None:
        raise OCIBackendError("host CPU-list is malformed")
    result: set[int] = set()
    for piece in value.split(","):
        bounds = piece.split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        if end < start or end - start > 4096:
            raise OCIBackendError("host CPU-list range is invalid")
        result.update(range(start, end + 1))
    return result


def _resolve_gpu_local_cpuset(
    gpu_devices: tuple[int, ...], *, expected_cpu_model: str, expected_cpu_count: int
) -> str:
    """Resolve one NUMA-local CPU lane from trusted host PCI/NUMA state."""

    try:
        output = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=True,
            shell=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise OCIBackendError(f"cannot resolve host GPU PCI identities: {exc}") from None
    buses: dict[int, str] = {}
    for line in output.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) == 2 and pieces[0].isdigit():
            bus = pieces[1].lower()
            if re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bus):
                bus = bus[4:]
            buses[int(pieces[0])] = bus
    numa_nodes: set[int] = set()
    for device in gpu_devices:
        bus = buses.get(device)
        if bus is None or re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bus) is None:
            raise OCIBackendError(f"GPU {device} lacks a canonical PCI identity")
        try:
            node = int((Path("/sys/bus/pci/devices") / bus / "numa_node").read_text().strip())
        except (OSError, ValueError) as exc:
            raise OCIBackendError(f"cannot resolve NUMA node for GPU {device}: {exc}") from None
        if node < 0:
            raise OCIBackendError(f"GPU {device} has no local NUMA node")
        numa_nodes.add(node)
    if len(numa_nodes) != 1:
        raise OCIBackendError(
            "selected GPUs cross NUMA lanes; arena requires single-numa-local-v1"
        )
    node = next(iter(numa_nodes))
    try:
        cpuset = (Path(f"/sys/devices/system/node/node{node}") / "cpulist").read_text().strip()
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise OCIBackendError(f"cannot read host CPU/NUMA identity: {exc}") from None
    models = {
        line.split(":", 1)[1].strip()
        for line in cpuinfo.splitlines()
        if line.startswith("model name") and ":" in line
    }
    if models != {expected_cpu_model}:
        raise OCIBackendError(
            f"host CPU model differs from arena class: {sorted(models)!r}"
        )
    if len(_expand_cpu_list(cpuset)) != expected_cpu_count:
        raise OCIBackendError("NUMA-local CPU count differs from arena resource policy")
    return cpuset


def _selected_gpu_topology_fingerprint(gpu_devices: tuple[int, ...]) -> str:
    """Canonicalize the selected physical GPU interconnect square on the host."""

    try:
        text = subprocess.run(
            ["/usr/bin/nvidia-smi", "topo", "-m"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=True,
            shell=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise OCIBackendError(f"cannot read selected GPU topology: {exc}") from None
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [line.split() for line in clean.splitlines() if line.strip()]
    header = next(
        (
            [cell for cell in row if re.fullmatch(r"GPU[0-9]+", cell)]
            for row in lines
            if row and all(re.fullmatch(r"GPU[0-9]+", cell) for cell in row)
        ),
        None,
    )
    if not header:
        # nvidia-smi adds NIC/CPU columns to the header, so take its leading GPU run.
        for row in lines:
            run: list[str] = []
            for cell in row:
                if re.fullmatch(r"GPU[0-9]+", cell):
                    run.append(cell)
                elif run:
                    break
            if run:
                header = run
                break
    if not header:
        raise OCIBackendError("host GPU topology table lacks a header")
    rows: dict[str, list[str]] = {}
    for fields in lines:
        if fields and fields[0] in header and len(fields) >= 1 + len(header):
            links = fields[1:1 + len(header)]
            if all(re.fullmatch(r"X|NV[0-9]+|PIX|PXB|PHB|NODE|SYS", value) for value in links):
                rows[fields[0]] = links
    labels = [f"GPU{device}" for device in gpu_devices]
    if any(label not in header or label not in rows for label in labels):
        raise OCIBackendError("selected GPUs are missing from the host topology table")
    positions = [header.index(label) for label in labels]
    matrix = [[rows[label][column] for column in positions] for label in labels]
    if any(row[index] != "X" for index, row in enumerate(matrix)):
        raise OCIBackendError("selected GPU topology diagonal is malformed")
    encoded = json.dumps(
        {"schema": "optima-gpu-topology-v1", "matrix": matrix},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provision_device_state_policy(arena, gpu_devices: tuple[int, ...]) -> DeviceStatePolicy:
    """Bind host-specific GPU identities to one arena-owned device class."""

    device_class = arena.device_state
    try:
        configurations = provision_gpu_configurations(
            gpu_devices,
            deadline=time.monotonic() + float(device_class.drain_timeout_s),
        )
    except (DeviceStateError, DeviceStatePolicyError) as exc:
        raise _mapped_device_state_error(
            exc, context="trusted-host GPU provisioning"
        ) from None

    mismatches: list[str] = []
    for configuration in configurations:
        expected = {
            "name": arena.gpu_name,
            "memory_total_mib": arena.gpu_memory_mib,
            "driver_version": arena.driver_version,
            "power_limit_mw": device_class.power_limit_mw,
            "compute_mode": device_class.compute_mode,
            "persistence_mode": device_class.persistence_mode,
            "application_graphics_clock_mhz": (
                device_class.application_graphics_clock_mhz
            ),
            "application_memory_clock_mhz": (
                device_class.application_memory_clock_mhz
            ),
            "max_graphics_clock_mhz": device_class.max_graphics_clock_mhz,
            "max_memory_clock_mhz": device_class.max_memory_clock_mhz,
        }
        changed = [
            name
            for name, value in expected.items()
            if getattr(configuration, name) != value
        ]
        if changed:
            mismatches.append(
                f"gpu{configuration.physical_id}({','.join(changed)})"
            )
    if mismatches:
        raise OCIBackendError(
            "selected GPUs differ from the arena device class: " + ";".join(mismatches)
        )

    try:
        return DeviceStatePolicy(
            expected_gpus=configurations,
            maximum_temperature_c=device_class.maximum_temperature_c,
            maximum_gpu_utilization_percent=(
                device_class.maximum_gpu_utilization_percent
            ),
            maximum_memory_utilization_percent=(
                device_class.maximum_memory_utilization_percent
            ),
            # The final-warmup monitor is crown authority, not diagnostics: every
            # scored arm must reach the serving P-state and pinned clock/power
            # envelope before the first timed request crosses into the container.
            allowed_active_pstates=("P0",),
            active_maximum_graphics_clock_mhz=(
                device_class.max_graphics_clock_mhz
            ),
            active_memory_clock_mhz=device_class.max_memory_clock_mhz,
            active_maximum_power_draw_mw=device_class.power_limit_mw,
            active_require_process_on_every_gpu=(
                device_class.require_process_on_every_gpu
            ),
            required_consecutive_idle_samples=(
                device_class.required_consecutive_idle_samples
            ),
            poll_interval_s=device_class.poll_interval_s,
            ready_poll_interval_s=device_class.ready_poll_interval_s,
            drain_timeout_s=device_class.drain_timeout_s,
            maximum_samples=device_class.maximum_samples,
        )
    except DeviceStatePolicyError as exc:
        raise OCIBackendError(f"arena device-state policy is invalid: {exc}") from None


def _artifact_publication_path(profile: "OCILaunchProfile") -> Path:
    return profile.artifact_dir / "published"


def _prepare_worker_directory(path: Path, profile: "OCILaunchProfile") -> None:
    """Give only the pinned non-root worker access to a launch-private directory."""

    os.chmod(path, 0o700)
    info = path.stat()
    if (info.st_uid, info.st_gid) == (profile.worker_uid, profile.worker_gid):
        return
    if os.geteuid() != 0:
        raise OCIBackendError(
            "validator must run as root (or the pinned worker identity) to prepare "
            f"private OCI directory ownership for {profile.worker_uid}:{profile.worker_gid}"
        )
    try:
        os.chown(path, profile.worker_uid, profile.worker_gid)
    except OSError as exc:
        raise OCIBackendError(f"cannot assign private OCI directory ownership: {exc}") from None


def _mount_worker_tmpfs(
    path: Path,
    *,
    size: str,
    nr_inodes: int,
    exec_allowed: bool,
    profile: "OCILaunchProfile",
) -> bool:
    """Mount a hard-size-bounded host tmpfs for hostile JIT/build writes."""

    if not profile.require_host_tmpfs:
        _prepare_worker_directory(path, profile)
        return False
    if os.geteuid() != 0:
        raise OCIBackendError("crownable host-tmpfs staging requires a root validator")
    options = (
        f"size={size},mode=0700,uid={profile.worker_uid},gid={profile.worker_gid},"
        f"nr_inodes={nr_inodes},nosuid,nodev"
        + ("" if exec_allowed else ",noexec")
    )
    try:
        subprocess.run(
            ["mount", "-t", "tmpfs", "-o", options, "optima-tmpfs", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OCIBackendError(f"cannot mount quota-backed host tmpfs at {path}: {exc}") from None
    try:
        info = path.stat()
        if (
            (info.st_uid, info.st_gid) != (profile.worker_uid, profile.worker_gid)
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise OCIBackendError("host tmpfs ownership/mode differs from arena policy")
        mount_line = ""
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 10 and fields[4].replace("\\040", " ") == str(path):
                mount_line = line
                break
        if " - tmpfs " not in mount_line:
            raise OCIBackendError("host staging path is not a dedicated tmpfs mount")
        mount_options = set(mount_line.split()[5].split(","))
        required_options = {"rw", "nosuid", "nodev"}
        if not required_options.issubset(mount_options):
            raise OCIBackendError("host tmpfs lacks required mount restrictions")
        if ("noexec" in mount_options) == exec_allowed:
            raise OCIBackendError("host tmpfs exec policy differs from arena purpose")
        units = {"k": 1024, "m": 1024**2, "g": 1024**3}
        suffix = size[-1].lower() if size[-1].isalpha() else ""
        requested_bytes = int(size[:-1] if suffix else size) * units.get(suffix, 1)
        fs = os.statvfs(path)
        if fs.f_blocks * fs.f_frsize > requested_bytes or fs.f_files > nr_inodes:
            raise OCIBackendError("host tmpfs byte/inode quota exceeds arena policy")
    except BaseException:
        subprocess.run(
            ["umount", "--", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
            shell=False,
        )
        raise
    return True


def _unmount_worker_tmpfs(path: Path) -> None:
    try:
        subprocess.run(
            ["umount", "--", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OCIInfrastructureError(f"failed to unmount private host tmpfs {path}: {exc}") from None
    if os.path.ismount(path):
        raise OCIInfrastructureError(f"private host tmpfs remained mounted after umount: {path}")


def _cleanup_tmpfs_mounts(paths: Sequence[Path], *, recovery_root: Path) -> None:
    errors: list[str] = []
    for path in reversed(tuple(paths)):
        try:
            _unmount_worker_tmpfs(path)
        except Exception as exc:  # try every mount before surfacing recovery state
            errors.append(f"{path}: {exc}")
    if errors:
        receipt = recovery_root / "TMPFS_RECOVERY_REQUIRED.json"
        try:
            receipt.write_text(
                json.dumps(
                    {"schema": "optima-tmpfs-recovery-v1", "errors": errors},
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        raise OCIInfrastructureError(
            "one or more private tmpfs mounts require operator recovery: "
            + "; ".join(errors)
        )


def _sha256_regular(path: Path, *, expected_size: int | None = None) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise OCIBackendError(f"cannot open artifact without following links: {path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OCIBackendError(f"artifact is not a regular file: {path}")
        if expected_size is not None and before.st_size != expected_size:
            raise OCIBackendError(f"artifact size differs from publication manifest: {path}")
        if before.st_size > _MAX_ARTIFACT_FILE_BYTES:
            raise OCIBackendError(f"artifact file exceeds its hard bound: {path}")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise OCIBackendError(f"artifact was truncated while hashing: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise OCIBackendError(f"artifact grew while hashing: {path}")
        after = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise OCIBackendError(f"artifact changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _read_regular_bounded(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise OCIBackendError(f"cannot open bounded artifact file safely: {path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > max_bytes
        ):
            raise OCIBackendError(f"bounded artifact file shape is unsafe: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise OCIBackendError(f"bounded artifact file was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise OCIBackendError(f"bounded artifact file grew while reading: {path}")
        after = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise OCIBackendError(f"bounded artifact file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _artifact_rows(
    root: Path,
    *,
    max_files: int = _MAX_ARTIFACT_FILES,
    max_total_bytes: int = _MAX_ARTIFACT_TOTAL_BYTES,
    include_manifest: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    entries = 0
    directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        entries += 1
        if entries > max_files * 4:
            raise OCIBackendError("artifact publication exceeds its entry hard bound")
        if (
            not parts
            or len(parts) > 16
            or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
            or parts[0] not in _ARTIFACT_TOP_LEVEL | {_ARTIFACT_TREE_MANIFEST}
        ):
            raise OCIBackendError(f"artifact publication contains an unsafe path: {relative}")
        try:
            info = path.lstat()
        except OSError as exc:
            raise OCIBackendError(f"cannot inspect artifact publication {relative}: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise OCIBackendError(f"artifact publication contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            if parts[0] not in _ARTIFACT_TOP_LEVEL:
                raise OCIBackendError(
                    f"artifact publication has an unapproved directory: {relative}"
                )
            directories.add(relative)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OCIBackendError(
                f"artifact publication contains a non-regular/hardlinked object: {relative}"
            )
        if relative == _ARTIFACT_TREE_MANIFEST and not include_manifest:
            continue
        if parts[0] not in _ARTIFACT_TOP_LEVEL:
            raise OCIBackendError(f"artifact publication has an unapproved top-level path: {relative}")
        if info.st_size > 0 and info.st_blocks * 512 < info.st_size:
            raise OCIBackendError(f"artifact publication contains a sparse file: {relative}")
        total += info.st_size
        if len(rows) >= max_files or total > max_total_bytes:
            raise OCIBackendError("artifact publication exceeds its file/byte hard bound")
        rows.append({
            "path": relative,
            "sha256": _sha256_regular(path, expected_size=info.st_size),
            "size": info.st_size,
        })
    required_directories: set[str] = set()
    for row in rows:
        parts = Path(row["path"]).parts
        for index in range(1, len(parts)):
            required_directories.add(Path(*parts[:index]).as_posix())
    if directories != required_directories:
        raise OCIBackendError(
            "artifact publication contains empty/unmanifested directories"
        )
    return rows


def _remove_build_locks(stage: Path) -> None:
    for locks in (stage / "cuda_ext", stage / "device_cubin"):
        for path in list(locks.rglob(".locks")) if locks.is_dir() else []:
            if path.is_symlink() or not path.is_dir():
                raise OCIBackendError(f"artifact lock path is unsafe: {path}")
            shutil.rmtree(path)
    system = stage / "system_overlay"
    for path in list(system.rglob(".overlay.lock")) if system.is_dir() else []:
        if path.is_symlink() or not path.is_file():
            raise OCIBackendError(f"system-overlay lock path is unsafe: {path}")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OCIBackendError(f"system-overlay lock is not a regular file: {path}")
        path.unlink()


def _freeze_artifact_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            os.chmod(path, 0o444, follow_symlinks=False)
        elif stat.S_ISDIR(info.st_mode):
            os.chmod(path, 0o555, follow_symlinks=False)
        else:
            raise OCIBackendError(f"cannot freeze unsafe artifact object: {path}")
    os.chmod(root, 0o555, follow_symlinks=False)


def _remove_artifact_tree(root: Path) -> None:
    if not os.path.lexists(root):
        return
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts)):
        if path.is_dir() and not path.is_symlink():
            try:
                os.chmod(path, 0o700, follow_symlinks=False)
            except OSError:
                pass
    try:
        os.chmod(root, 0o700, follow_symlinks=False)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)


def _write_artifact_publication_manifest(
    stage: Path, *, profile: "OCILaunchProfile", request_id: str, bundle_hash: str,
) -> dict[str, Any]:
    _remove_build_locks(stage)
    rows = _artifact_rows(
        stage,
        max_files=profile.artifact_max_files,
        max_total_bytes=profile.artifact_max_bytes,
    )
    receipt_path = f"prebuild_receipts/{request_id}.json"
    if receipt_path not in {row["path"] for row in rows}:
        raise OCIBackendError("artifact stage lacks its exact prebuild receipt")
    payload = {
        "schema": _ARTIFACT_TREE_SCHEMA,
        "request_id": request_id,
        "bundle_hash": bundle_hash,
        "image": profile.image,
        "arena_name": profile.arena_name or "",
        "arena_fingerprint": profile.arena_fingerprint or "",
        "competition_target": profile.competition_target or "",
        "gpu_architecture": profile.gpu_architecture,
        "gpu_topology_sha256": profile.topology_sha256 or "",
        "sglang_version": profile.sglang_version,
        "model_revision": profile.model_revision,
        "model_manifest_digest": profile.model_manifest_digest,
        "model_content_digest": profile.model_content_digest,
        "environment_sha256": environment_fingerprint(profile.environment),
        "referee_source_digest": profile.referee_source_digest,
        "referee_tree_digest": profile.referee_tree_digest,
        "runtime_overlays_sha256": runtime_overlay_fingerprint(profile.runtime_overlays),
        "files": rows,
    }
    destination = stage / _ARTIFACT_TREE_MANIFEST
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii") + b"\n"
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OCIBackendError("artifact publication manifest write stalled")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return payload


def _copy_validated_artifact_tree(
    source: Path, destination: Path, *, profile: "OCILaunchProfile"
) -> None:
    """Copy a bounded hostile tmpfs tree into a controller-only XFS stage."""

    if any(destination.iterdir()):
        raise OCIBackendError("artifact publication stage must start empty")
    rows = _artifact_rows(
        source,
        max_files=profile.artifact_max_files,
        max_total_bytes=profile.artifact_max_bytes,
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for row in rows:
        relative = Path(row["path"])
        src = source / relative
        dst = destination / relative
        dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        src_fd = -1
        dst_fd = -1
        try:
            src_fd = os.open(src, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow)
            dst_fd = os.open(
                dst,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0) | nofollow,
                0o600,
            )
        except OSError as exc:
            if src_fd >= 0:
                os.close(src_fd)
            raise OCIBackendError(f"cannot copy staged artifact safely: {relative}: {exc}") from None
        try:
            before = os.fstat(src_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != row["size"]
            ):
                raise OCIBackendError(f"staged artifact changed before copy: {relative}")
            digest = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                chunk = os.read(src_fd, min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise OCIBackendError(f"staged artifact truncated during copy: {relative}")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(dst_fd, view)
                    if written <= 0:
                        raise OCIBackendError(f"artifact copy made no progress: {relative}")
                    view = view[written:]
                remaining -= len(chunk)
            if os.read(src_fd, 1):
                raise OCIBackendError(f"staged artifact grew during copy: {relative}")
            after = os.fstat(src_fd)
            stable = (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
                "st_size", "st_mtime_ns", "st_ctime_ns",
            )
            if any(getattr(before, key) != getattr(after, key) for key in stable):
                raise OCIBackendError(f"staged artifact changed during copy: {relative}")
            if digest.hexdigest() != row["sha256"]:
                raise OCIBackendError(f"staged artifact hash changed during copy: {relative}")
            os.fsync(dst_fd)
        finally:
            if src_fd >= 0:
                os.close(src_fd)
            if dst_fd >= 0:
                os.close(dst_fd)


def _verify_artifact_publication(
    root: Path, *, profile: "OCILaunchProfile", bundle_hash: str,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise OCIBackendError("candidate artifact publication is missing or unsafe")
    manifest_path = root / _ARTIFACT_TREE_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise OCIBackendError("candidate artifact publication lacks its host manifest")
    try:
        raw = _read_regular_bounded(manifest_path, max_bytes=16 * 1024 * 1024)
        manifest = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as exc:
        raise OCIBackendError(f"artifact publication manifest is malformed: {exc}") from None
    expected_keys = {
        "schema", "request_id", "bundle_hash", "image", "arena_name",
        "arena_fingerprint", "competition_target", "gpu_architecture",
        "gpu_topology_sha256", "sglang_version", "model_revision",
        "model_manifest_digest", "model_content_digest", "environment_sha256",
        "referee_source_digest", "referee_tree_digest",
        "runtime_overlays_sha256", "files",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema") != _ARTIFACT_TREE_SCHEMA
        or manifest.get("bundle_hash") != bundle_hash
        or manifest.get("image") != profile.image
        or manifest.get("arena_name") != (profile.arena_name or "")
        or manifest.get("arena_fingerprint") != (profile.arena_fingerprint or "")
        or manifest.get("competition_target") != (profile.competition_target or "")
        or manifest.get("gpu_architecture") != profile.gpu_architecture
        or manifest.get("gpu_topology_sha256") != (profile.topology_sha256 or "")
        or manifest.get("sglang_version") != profile.sglang_version
        or manifest.get("model_revision") != profile.model_revision
        or manifest.get("model_manifest_digest") != profile.model_manifest_digest
        or manifest.get("model_content_digest") != profile.model_content_digest
        or manifest.get("environment_sha256")
            != environment_fingerprint(profile.environment)
        or manifest.get("referee_source_digest") != profile.referee_source_digest
        or manifest.get("referee_tree_digest") != profile.referee_tree_digest
        or manifest.get("runtime_overlays_sha256")
            != runtime_overlay_fingerprint(profile.runtime_overlays)
        or not re.fullmatch(r"[0-9a-f]{32}", str(manifest.get("request_id", "")))
        or not isinstance(manifest.get("files"), list)
    ):
        raise OCIBackendError("artifact publication manifest identity/schema mismatch")
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii") + b"\n"
    if raw != canonical:
        raise OCIBackendError("artifact publication manifest is not canonical")
    observed = _artifact_rows(
        root,
        max_files=profile.artifact_max_files,
        max_total_bytes=profile.artifact_max_bytes,
    )
    if observed != manifest["files"]:
        raise OCIBackendError("artifact publication file set/hash differs from manifest")
    return manifest


@dataclass(frozen=True)
class OCILaunchProfile:
    """Trusted, immutable inputs/options for a family of B/C/B' launches."""

    image: str
    source_dir: Path
    model_dir: Path
    artifact_dir: Path
    scratch_root: Path
    gpu_devices: tuple[int, ...]
    sglang_version: str
    gpu_architecture: str
    referee_source_digest: str
    referee_tree_digest: str
    model_revision: str
    model_manifest_digest: str
    model_content_digest: str
    gpu_name: str
    gpu_memory_mib: int
    driver_version: str
    runtime_overlays: tuple[RuntimeFileOverlay, ...] = ()
    bundle_dir: Path | None = None
    arena_name: str | None = None
    arena_fingerprint: str | None = None
    competition_target: str | None = None
    topology_sha256: str | None = None
    device_state_policy: DeviceStatePolicy | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    docker_binary: str = "docker"
    runtime_preflight_receipt: RuntimePreflightReceipt | None = None
    worker_python: str = "python3"
    shm_size: str = "256g"
    tmpfs_size: str = "16g"
    pids_limit: int = 65_536
    memory_limit_bytes: int = 1 << 40
    cpu_limit: float = 96.0
    cpuset_cpus: str | None = None
    nofile_limit: int = 65_536
    artifact_tmpfs_size: str = "32g"
    artifact_max_bytes: int = _MAX_ARTIFACT_TOTAL_BYTES
    artifact_max_files: int = _MAX_ARTIFACT_FILES
    scratch_tmpfs_inodes: int = 1_000_000
    artifact_tmpfs_inodes: int = 65_536
    require_host_tmpfs: bool = False
    prebuild_timeout_s: float = 1800.0
    bracket_timeout_s: float = 7200.0
    init_timeout_s: float = 1800.0
    batch_timeout_s: float = 1800.0
    # Match the controller by default so cap-dropped container processes can write
    # the private bind mounts without CAP_DAC_OVERRIDE. Root-run validators stay 0:0.
    worker_uid: int = field(default_factory=os.getuid)
    worker_gid: int = field(default_factory=os.getgid)

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or _DIGEST_IMAGE.fullmatch(self.image) is None:
            raise OCIBackendError(
                "OCI image must be pinned as name@sha256:<64 lowercase hex>"
            )
        if self.runtime_preflight_receipt is not None:
            _verify_runtime_preflight_identity(
                self.runtime_preflight_receipt,
                image=self.image,
                sglang_version=self.sglang_version,
                worker_uid=self.worker_uid,
                worker_gid=self.worker_gid,
                docker_binary=self.docker_binary,
            )
        elif self.arena_fingerprint is not None:
            raise OCIBackendError(
                "crownable OCI profile requires a trusted stock runtime preflight receipt"
            )
        object.__setattr__(
            self, "source_dir", _resolved_directory(self.source_dir, "source_dir")
        )
        for relative in (
            "optima/eval/oci_worker.py",
            "optima/eval/oci_session_worker.py",
            "optima/eval/oci_prebuild.py",
            "optima/eval/oci_site/sitecustomize.py",
            "optima/eval/seccomp_moby_v0_2_1.json",
        ):
            source_file = self.source_dir / relative
            if source_file.is_symlink() or not source_file.is_file():
                raise OCIBackendError(
                    f"source_dir lacks required regular validator source {relative!r}"
                )
        object.__setattr__(
            self, "model_dir", _resolved_directory(self.model_dir, "model_dir")
        )
        object.__setattr__(
            self, "artifact_dir", _resolved_directory(self.artifact_dir, "artifact_dir")
        )
        object.__setattr__(
            self, "scratch_root", _resolved_directory(self.scratch_root, "scratch_root")
        )
        for name in ("artifact_dir", "scratch_root"):
            path = getattr(self, name)
            info = path.stat()
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise OCIBackendError(
                    f"{name} must be validator-owned and not group/world writable"
                )
        if self.bundle_dir is not None:
            object.__setattr__(
                self, "bundle_dir", _resolved_directory(self.bundle_dir, "bundle_dir")
            )
        named_roots = {
            "source_dir": self.source_dir,
            "model_dir": self.model_dir,
            "artifact_dir": self.artifact_dir,
            "scratch_root": self.scratch_root,
        }
        if self.bundle_dir is not None:
            named_roots["bundle_dir"] = self.bundle_dir
        pairs = list(named_roots.items())
        for index, (left_name, left) in enumerate(pairs):
            for right_name, right in pairs[index + 1:]:
                if left == right or left in right.parents or right in left.parents:
                    raise OCIBackendError(
                        "OCI immutable/writable roots must be pairwise disjoint: "
                        f"{left_name}={left} overlaps {right_name}={right}"
                    )
        if (self.arena_name is None) != (self.competition_target is None):
            raise OCIBackendError(
                "arena_name and competition_target must be provided together"
            )
        if (self.arena_name is None) != (self.arena_fingerprint is None):
            raise OCIBackendError(
                "arena_name and arena_fingerprint must be provided together"
            )
        if self.arena_fingerprint is not None and _SHA256.fullmatch(
            self.arena_fingerprint
        ) is None:
            raise OCIBackendError("arena_fingerprint must be 64 lowercase hex")
        for name in ("arena_name", "competition_target"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or _PROFILE_ID.fullmatch(value) is None
            ):
                raise OCIBackendError(f"{name} must be a simple validator-owned ID")
        if (
            not isinstance(self.gpu_devices, tuple)
            or not self.gpu_devices
            or any(
                isinstance(device, bool) or not isinstance(device, int) or device < 0
                for device in self.gpu_devices
            )
            or len(set(self.gpu_devices)) != len(self.gpu_devices)
        ):
            raise OCIBackendError("gpu_devices must be a non-empty tuple of unique IDs")
        if (
            not isinstance(self.sglang_version, str)
            or not self.sglang_version
            or len(self.sglang_version) > 128
            or any(char in self.sglang_version for char in ("\x00", "\n", "\r"))
        ):
            raise OCIBackendError("sglang_version must be an exact profile-owned version")
        if (
            not isinstance(self.gpu_architecture, str)
            or _GPU_ARCH.fullmatch(self.gpu_architecture) is None
        ):
            raise OCIBackendError("gpu_architecture must look like 'sm103'")
        if self.topology_sha256 is not None and (
            not isinstance(self.topology_sha256, str)
            or _SHA256.fullmatch(self.topology_sha256) is None
        ):
            raise OCIBackendError("topology_sha256 must be 64 lowercase hex characters")
        if self.device_state_policy is not None:
            if type(self.device_state_policy) is not DeviceStatePolicy:
                raise OCIBackendError(
                    "device_state_policy must be an exact trusted-host policy"
                )
            if self.device_state_policy.physical_gpu_ids != self.gpu_devices:
                raise OCIBackendError(
                    "device-state physical GPU IDs differ from launch GPU IDs"
                )
        if self.arena_fingerprint is not None and self.device_state_policy is None:
            raise OCIBackendError(
                "crownable OCI profile requires a frozen trusted-host device policy"
            )
        if (
            not isinstance(self.gpu_name, str)
            or not self.gpu_name
            or len(self.gpu_name) > 256
            or type(self.gpu_memory_mib) is not int
            or self.gpu_memory_mib <= 0
            or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", self.driver_version)
        ):
            raise OCIBackendError("GPU name/memory/driver policy is invalid")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.model_revision):
            raise OCIBackendError("model_revision must be an immutable hex revision")
        for name in (
            "referee_source_digest",
            "referee_tree_digest",
            "model_manifest_digest",
            "model_content_digest",
        ):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(self, name)):
                raise OCIBackendError(f"{name} must be a sha256 content identity")
        try:
            runtime_overlays = normalize_runtime_overlays(self.runtime_overlays)
            verify_runtime_overlays(self.model_dir, runtime_overlays)
        except RuntimeOverlayError as exc:
            raise OCIBackendError(f"runtime overlay profile is invalid: {exc}") from None
        object.__setattr__(self, "runtime_overlays", runtime_overlays)
        _simple_executable(self.docker_binary, "docker_binary")
        _simple_executable(self.worker_python, "worker_python")
        if not isinstance(self.shm_size, str) or _SIZE.fullmatch(self.shm_size) is None:
            raise OCIBackendError("shm_size must be a positive OCI size")
        if not isinstance(self.tmpfs_size, str) or _SIZE.fullmatch(self.tmpfs_size) is None:
            raise OCIBackendError("tmpfs_size must be a positive OCI size")
        if (
            isinstance(self.pids_limit, bool)
            or not isinstance(self.pids_limit, int)
            or not 256 <= self.pids_limit <= 1_048_576
        ):
            raise OCIBackendError("pids_limit must be an integer in [256, 1048576]")
        if (
            type(self.memory_limit_bytes) is not int
            or not (1 << 30) <= self.memory_limit_bytes <= (1 << 42)
        ):
            raise OCIBackendError("memory_limit_bytes must be in [1GiB, 4TiB]")
        if (
            isinstance(self.cpu_limit, bool)
            or not isinstance(self.cpu_limit, (int, float))
            or not 1.0 <= float(self.cpu_limit) <= 1024.0
        ):
            raise OCIBackendError("cpu_limit must be in [1, 1024]")
        if self.cpuset_cpus is not None and (
            not isinstance(self.cpuset_cpus, str)
            or re.fullmatch(r"[0-9,-]{1,256}", self.cpuset_cpus) is None
        ):
            raise OCIBackendError("cpuset_cpus must be a numeric CPU-list expression")
        if type(self.nofile_limit) is not int or not 1024 <= self.nofile_limit <= 1_048_576:
            raise OCIBackendError("nofile_limit must be in [1024, 1048576]")
        if (
            not isinstance(self.artifact_tmpfs_size, str)
            or _SIZE.fullmatch(self.artifact_tmpfs_size) is None
            or type(self.artifact_max_bytes) is not int
            or not (1 << 20) <= self.artifact_max_bytes <= _MAX_ARTIFACT_TOTAL_BYTES
            or type(self.artifact_max_files) is not int
            or not 1 <= self.artifact_max_files <= _MAX_ARTIFACT_FILES
        ):
            raise OCIBackendError("artifact tmpfs/file/byte resource policy is invalid")
        if (
            type(self.scratch_tmpfs_inodes) is not int
            or type(self.artifact_tmpfs_inodes) is not int
            or not 1024 <= self.scratch_tmpfs_inodes <= 10_000_000
            or not 1024 <= self.artifact_tmpfs_inodes <= 10_000_000
        ):
            raise OCIBackendError("tmpfs inode resource policy is invalid")
        if type(self.require_host_tmpfs) is not bool:
            raise OCIBackendError("require_host_tmpfs must be boolean")
        if self.arena_fingerprint is not None and not self.require_host_tmpfs:
            raise OCIBackendError("crownable OCI profile requires quota-backed host tmpfs")
        for name in (
            "prebuild_timeout_s", "bracket_timeout_s", "init_timeout_s",
            "batch_timeout_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 1 <= float(value) <= 86_400
            ):
                raise OCIBackendError(f"{name} must be in [1, 86400]")
        for name in ("worker_uid", "worker_gid"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 2_147_483_647
            ):
                raise OCIBackendError(f"{name} must be a nonnegative 32-bit ID")

        protected = set(_base_environment()) | {
            # Device access is selected once by --gpus; a second visibility map can
            # silently change rank-to-device assignment inside the container.
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "OPTIMA_OCI_ARENA_NAME",
            "OPTIMA_OCI_COMPETITION_TARGET",
            "OPTIMA_TARGET_GPU_ARCH",
            "TORCH_CUDA_ARCH_LIST",
        }
        clean_env: dict[str, str] = {}
        if not isinstance(self.environment, Mapping):
            raise OCIBackendError("environment must be a mapping")
        for key, value in self.environment.items():
            if not env_is_safe(key, value):
                raise OCIBackendError(f"unsafe OCI profile environment entry {key!r}")
            if (
                key in protected
                or key.startswith(("OPTIMA_OCI_", "OPTIMA_SYSTEM_"))
                or key in {
                    "OPTIMA_ACTIVE",
                    "OPTIMA_BUNDLE_PATH",
                    "OPTIMA_FRAMEWORK_MODE",
                    "SGLANG_PLUGINS",
                }
            ):
                raise OCIBackendError(f"OCI profile may not override protected env {key!r}")
            clean_env[key] = value
        object.__setattr__(self, "environment", MappingProxyType(dict(sorted(clean_env.items()))))


def profile_for_arena(
    arena,
    *,
    source_dir: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    scratch_root: str | os.PathLike[str],
    gpu_devices: tuple[int, ...],
    bundle_dir: str | os.PathLike[str] | None = None,
    competition_target: str | None = None,
    topology_sha256: str | None = None,
    docker_binary: str = "docker",
    worker_python: str = "python3",
) -> OCILaunchProfile:
    """Build a launch profile from one registered arena and exact host inputs.

    This is a trusted controller preflight, not a merge operation: model/source
    receipts must already match the arena and every GPU ID is explicit.
    """

    from optima.arenas import ArenaProfile
    from optima.source_release import RefereeReleaseError, verify_referee_source_release

    if type(arena) is not ArenaProfile:
        raise OCIBackendError("profile_for_arena requires an exact ArenaProfile")
    resources = arena.oci_resources
    resolved_docker = _resolved_docker_binary(docker_binary)
    runtime_preflight_receipt = _runtime_preflight_once(
        image=arena.validator_image,
        sglang_version=arena.sglang_version,
        worker_uid=resources.worker_uid,
        worker_gid=resources.worker_gid,
        docker_binary=resolved_docker,
    )
    if (
        len(gpu_devices) != resources.gpu_count
        or len(set(gpu_devices)) != len(gpu_devices)
        or any(type(device) is not int or device < 0 for device in gpu_devices)
    ):
        raise OCIBackendError(
            f"arena requires {resources.gpu_count} explicit distinct GPU IDs"
        )
    cpuset_cpus = _resolve_gpu_local_cpuset(
        gpu_devices,
        expected_cpu_model=resources.cpu_model,
        expected_cpu_count=resources.cpu_logical_count,
    )
    actual_selected_topology = _selected_gpu_topology_fingerprint(gpu_devices)
    if actual_selected_topology != arena.gpu_topology_sha256:
        raise OCIBackendError(
            "selected GPU topology differs from the arena class: "
            f"{actual_selected_topology} != {arena.gpu_topology_sha256}"
        )
    if topology_sha256 is not None and topology_sha256 != arena.gpu_topology_sha256:
        raise OCIBackendError(
            "operator topology override differs from the registered arena"
        )
    device_state_policy = _provision_device_state_policy(arena, gpu_devices)
    source = _resolved_directory(source_dir, "source_dir")
    try:
        verify_referee_source_release(
            source,
            expected_tree_digest=arena.referee_tree_digest,
            expected_referee_source_digest=arena.referee_source_digest,
        )
    except RefereeReleaseError as exc:
        raise OCIBackendError(
            f"mounted referee release differs from arena receipt: {exc}"
        ) from None
    model = verify_production_model_once(arena, model_dir)
    return OCILaunchProfile(
        image=arena.validator_image,
        source_dir=source,
        model_dir=model,
        artifact_dir=Path(artifact_dir),
        scratch_root=Path(scratch_root),
        gpu_devices=gpu_devices,
        sglang_version=arena.sglang_version,
        gpu_architecture=arena.gpu_architecture,
        referee_source_digest=arena.referee_source_digest,
        referee_tree_digest=arena.referee_tree_digest,
        model_revision=arena.model_revision,
        model_manifest_digest=arena.model_manifest_digest,
        model_content_digest=arena.model_content_digest,
        gpu_name=arena.gpu_name,
        gpu_memory_mib=arena.gpu_memory_mib,
        driver_version=arena.driver_version,
        runtime_overlays=arena.runtime_overlays,
        bundle_dir=Path(bundle_dir) if bundle_dir is not None else None,
        arena_name=arena.name if competition_target is not None else None,
        arena_fingerprint=arena.fingerprint if competition_target is not None else None,
        competition_target=competition_target,
        topology_sha256=arena.gpu_topology_sha256,
        device_state_policy=device_state_policy,
        cpuset_cpus=cpuset_cpus,
        memory_limit_bytes=resources.memory_limit_bytes,
        cpu_limit=resources.cpu_limit,
        nofile_limit=resources.nofile_limit,
        pids_limit=resources.pids_limit,
        shm_size=resources.shm_size,
        tmpfs_size=resources.scratch_tmpfs_size,
        artifact_tmpfs_size=resources.artifact_tmpfs_size,
        artifact_max_bytes=resources.artifact_max_bytes,
        artifact_max_files=resources.artifact_max_files,
        scratch_tmpfs_inodes=resources.scratch_tmpfs_inodes,
        artifact_tmpfs_inodes=resources.artifact_tmpfs_inodes,
        require_host_tmpfs=resources.require_host_tmpfs,
        prebuild_timeout_s=resources.prebuild_timeout_s,
        bracket_timeout_s=resources.bracket_timeout_s,
        init_timeout_s=resources.init_timeout_s,
        batch_timeout_s=resources.batch_timeout_s,
        worker_uid=resources.worker_uid,
        worker_gid=resources.worker_gid,
        environment=dict(arena.environment),
        docker_binary=resolved_docker,
        runtime_preflight_receipt=runtime_preflight_receipt,
        worker_python=worker_python,
    )


def _base_environment() -> dict[str, str]:
    """Security/cache environment fixed by validator code, not bundle/request data."""

    return {
        "OPTIMA_EXTERNAL_NO_EGRESS": "1",
        "OPTIMA_PREBUILT_ARTIFACTS": "1",
        "OPTIMA_ARTIFACT_ROOT": CONTAINER_ARTIFACT_PATH,
        "OPTIMA_CUDA_EXT_CACHE": f"{CONTAINER_ARTIFACT_PATH}/cuda_ext",
        "OPTIMA_DEVICE_CACHE": f"{CONTAINER_ARTIFACT_PATH}/device_cubin",
        "OPTIMA_OCI_SECCOMP_PROFILE_SHA256": _SECCOMP_PROFILE_SHA256,
        # The trusted materialized v2 subtree is nested-mounted read-only below;
        # jit_workspace remains launch-private and writable beside it.
        "OPTIMA_DEP_OVERLAY_CACHE": f"{CONTAINER_JIT_PATH}/dep_overlay",
        "OPTIMA_OCI_SYSTEM_OVERLAY_ROOT": (
            f"{CONTAINER_ARTIFACT_PATH}/system_overlay"
        ),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": f"{CONTAINER_JIT_PATH}/huggingface",
        "XDG_CACHE_HOME": f"{CONTAINER_JIT_PATH}/xdg",
        "TRITON_CACHE_DIR": f"{CONTAINER_JIT_PATH}/triton",
        "TORCH_EXTENSIONS_DIR": f"{CONTAINER_JIT_PATH}/torch_extensions",
        "CUDA_CACHE_PATH": f"{CONTAINER_JIT_PATH}/cuda",
        "FLASHINFER_WORKSPACE_BASE": f"{CONTAINER_JIT_PATH}/flashinfer",
        "HOME": f"{CONTAINER_JIT_PATH}/home",
        # The pinned numeric worker deliberately need not exist in the arena
        # image's mutable account database. Python's getpass (reached by Torch /
        # Inductor during SGLang import) otherwise falls through to getpwuid and
        # crashes before the engine can start.
        "USER": "optima",
        "LOGNAME": "optima",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        # sitecustomize reproduces the established optima.pth bootstrap without
        # mutating the immutable arena image's site-packages.
        "PYTHONPATH": (
            f"{CONTAINER_SOURCE_PATH}/optima/eval/oci_site:{CONTAINER_SOURCE_PATH}"
        ),
    }


def _mount_arg(host: Path, container: str, *, readonly: bool) -> str:
    suffix = ",readonly" if readonly else ""
    return f"--mount=type=bind,src={host},dst={container}{suffix}"


def _seccomp_security_arg(profile: OCILaunchProfile) -> str:
    """Return a byte-verified, source-release-pinned host seccomp policy."""

    path = profile.source_dir / _SECCOMP_PROFILE_RELATIVE
    try:
        payload = _read_regular_bounded(path, max_bytes=64 * 1024)
    except OCIBackendError as exc:
        raise OCIBackendError(f"pinned seccomp profile is unsafe: {exc}") from None
    actual = hashlib.sha256(payload).hexdigest()
    if actual != _SECCOMP_PROFILE_SHA256:
        raise OCIBackendError(
            f"pinned seccomp profile hash mismatch: {actual}"
        )
    try:
        parsed = json.loads(payload)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise OCIBackendError(f"pinned seccomp profile is malformed: {exc}") from None
    if (
        not isinstance(parsed, dict)
        or parsed.get("defaultAction") != "SCMP_ACT_ERRNO"
        or not isinstance(parsed.get("syscalls"), list)
        or len(parsed["syscalls"]) < 10
    ):
        raise OCIBackendError("pinned seccomp profile has an unsafe policy shape")
    return f"--security-opt=seccomp={path}"


def _runtime_overlay_mount_args(profile: OCILaunchProfile) -> list[str]:
    """Re-hash sealed stock patches immediately before constructing Docker argv."""

    try:
        verified = verify_runtime_overlays(profile.model_dir, profile.runtime_overlays)
    except RuntimeOverlayError as exc:
        raise OCIBackendError(f"sealed runtime overlay preflight failed: {exc}") from None
    return [
        _mount_arg(source, overlay.target, readonly=True)
        for source, overlay in verified
    ]


def _verify_source_release_before_mount(profile: OCILaunchProfile) -> None:
    if profile.arena_fingerprint is None:
        return
    from optima.source_release import RefereeReleaseError, verify_referee_source_release

    try:
        verify_referee_source_release(
            profile.source_dir,
            expected_tree_digest=profile.referee_tree_digest,
            expected_referee_source_digest=profile.referee_source_digest,
        )
    except RefereeReleaseError as exc:
        raise OCIBackendError(
            f"referee release changed before OCI mount: {exc}"
        ) from None


def _runtime_overlay_environment(profile: OCILaunchProfile) -> dict[str, str]:
    overlays = normalize_runtime_overlays(profile.runtime_overlays)
    return {
        "OPTIMA_OCI_EXPECTED_RUNTIME_OVERLAYS_SHA256":
            runtime_overlay_fingerprint(overlays),
        "OPTIMA_OCI_EXPECTED_RUNTIME_OVERLAY_COUNT": str(len(overlays)),
        "OPTIMA_OCI_RUNTIME_OVERLAYS_JSON": json.dumps(
            [dataclasses.asdict(overlay) for overlay in overlays],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    }


def _offline_build_environment(profile: OCILaunchProfile) -> dict[str, str]:
    match = re.fullmatch(r"sm([0-9]{2,3})([a-z]?)", profile.gpu_architecture)
    if match is None:  # profile validation should make this unreachable
        raise OCIBackendError("cannot derive offline build architecture")
    digits, suffix = match.groups()
    # Hopper/Blackwell native features are architecture-specific. Arena pins sm103;
    # the reviewed direct-nvcc builders use NVIDIA spelling sm_103a.
    native_suffix = suffix or "a"
    nvcc_arch = f"sm_{digits}{native_suffix}"
    if len(digits) == 2:
        torch_arch = f"{digits[0]}.{digits[1]}{native_suffix}"
    else:
        torch_arch = f"{digits[:-1]}.{digits[-1]}{native_suffix}"
    return {
        "OPTIMA_TARGET_GPU_ARCH": nvcc_arch,
        "TORCH_CUDA_ARCH_LIST": torch_arch,
    }


def _resource_argv(profile: OCILaunchProfile) -> list[str]:
    args = [
        f"--memory={profile.memory_limit_bytes}",
        # Docker interprets memory-swap == memory as zero swap allowance.
        f"--memory-swap={profile.memory_limit_bytes}",
        f"--cpus={float(profile.cpu_limit):g}",
        f"--ulimit=nofile={profile.nofile_limit}:{profile.nofile_limit}",
    ]
    if profile.cpuset_cpus is not None:
        args.append(f"--cpuset-cpus={profile.cpuset_cpus}")
    return args


@dataclass
class PreparedOCILaunch:
    """One fresh launch's argv, private state, and in-memory authentication data."""

    profile: OCILaunchProfile
    request: OCILaunchRequest
    launch_root: Path
    jit_dir: Path
    output_dir: Path
    result_path: Path
    cid_path: Path
    container_name: str
    argv: tuple[str, ...]
    stdin_frame: bytes = field(repr=False)
    auth_key: bytes = field(repr=False)
    tmpfs_mounts: tuple[Path, ...] = ()
    cleaned: bool = False

    def cleanup(self) -> None:
        if not self.cleaned:
            _cleanup_tmpfs_mounts(self.tmpfs_mounts, recovery_root=self.launch_root)
            shutil.rmtree(self.launch_root, ignore_errors=True)
            self.cleaned = True


@dataclass
class PreparedOCIPrebuild:
    profile: OCILaunchProfile
    request_id: str
    launch_root: Path
    jit_dir: Path
    build_artifact_dir: Path
    staging_artifact_dir: Path
    publication_dir: Path
    cid_path: Path
    receipt_path: Path
    container_name: str
    argv: tuple[str, ...]
    stdin_frame: bytes = field(default=b"", repr=False)
    tmpfs_mounts: tuple[Path, ...] = ()
    cleaned: bool = False

    def cleanup(self) -> None:
        if not self.cleaned:
            _cleanup_tmpfs_mounts(self.tmpfs_mounts, recovery_root=self.launch_root)
            shutil.rmtree(self.launch_root, ignore_errors=True)
            self.cleaned = True


@dataclass
class PreparedOCISession:
    profile: OCILaunchProfile
    mode: str
    launch_root: Path
    jit_dir: Path
    stock_artifact_dir: Path | None
    cid_path: Path
    container_name: str
    argv: tuple[str, ...]
    expected_runtime_attestation: Mapping[str, Any]
    tmpfs_mounts: tuple[Path, ...] = ()
    cleaned: bool = False

    def cleanup(self) -> None:
        if not self.cleaned:
            _cleanup_tmpfs_mounts(self.tmpfs_mounts, recovery_root=self.launch_root)
            shutil.rmtree(self.launch_root, ignore_errors=True)
            self.cleaned = True


def _build_argv(
    profile: OCILaunchProfile,
    request: OCILaunchRequest,
    *,
    launch_root: Path,
    jit_dir: Path,
    output_dir: Path,
    cid_path: Path,
    container_name: str,
) -> tuple[str, ...]:
    _verify_source_release_before_mount(profile)
    if request.active and profile.bundle_dir is None:
        raise OCIBackendError("candidate OCI launch requires a profile-owned bundle_dir")
    if request.eval_config.get("tp_size") not in (None, len(profile.gpu_devices)):
        raise OCIBackendError(
            "EvalConfig.tp_size must equal the profile's explicit GPU device count"
        )
    env = _base_environment()
    env.update(profile.environment)
    env.update(
        OPTIMA_OCI_EXPECTED_SGLANG_VERSION=profile.sglang_version,
        OPTIMA_OCI_EXPECTED_GPU_ARCH=profile.gpu_architecture,
        OPTIMA_OCI_EXPECTED_GPU_COUNT=str(len(profile.gpu_devices)),
        OPTIMA_OCI_EXPECTED_REFEREE_SOURCE_DIGEST=profile.referee_source_digest,
        OPTIMA_OCI_EXPECTED_MODEL_REVISION=profile.model_revision,
        OPTIMA_OCI_EXPECTED_MODEL_MANIFEST_DIGEST=profile.model_manifest_digest,
        OPTIMA_OCI_EXPECTED_MODEL_CONTENT_DIGEST=profile.model_content_digest,
    )
    env.update(_runtime_overlay_environment(profile))
    if profile.topology_sha256 is not None:
        env["OPTIMA_OCI_EXPECTED_TOPOLOGY_SHA256"] = profile.topology_sha256
    if request.active and profile.arena_name is not None:
        assert profile.competition_target is not None
        # Staging names deliberately do not arm the bootstrap at interpreter start.
        # The trusted worker validates the artifact and constructs the real
        # OPTIMA_SYSTEM_* environment after it knows its in-container driver PID.
        env["OPTIMA_OCI_ARENA_NAME"] = profile.arena_name
        env["OPTIMA_OCI_COMPETITION_TARGET"] = profile.competition_target
    # Bind every validator-declared score-affecting env value. Image-baked extras are
    # separately bound by the immutable image digest; NVIDIA's generated visibility
    # state is checked through the GPU attestation below.
    attested_keys = sorted(env)
    env["OPTIMA_OCI_ATTEST_ENV_KEYS"] = ",".join(attested_keys)
    env["OPTIMA_OCI_ATTEST_ENV_SHA256"] = environment_fingerprint(
        {key: env[key] for key in attested_keys}
    )
    argv = [
        profile.docker_binary,
        "run",
        "--rm",
        "--init",
        "--interactive",
        "--pull=never",
        "--runtime=runc",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--cap-add=SYS_NICE",
        "--cap-add=SYS_RESOURCE",
        "--security-opt=no-new-privileges=true",
        _seccomp_security_arg(profile),
        f"--user={profile.worker_uid}:{profile.worker_gid}",
        # Docker's default is a private PID namespace.  There is no portable
        # `--pid=private` spelling (Docker rejects it); only the unsafe
        # alternatives such as `--pid=host` need an explicit flag.
        "--ipc=private",
        "--log-driver=none",
        "--ulimit=core=0:0",
        *_resource_argv(profile),
        f"--shm-size={profile.shm_size}",
        f"--pids-limit={profile.pids_limit}",
        f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={profile.tmpfs_size}",
        # Docker's --gpus value uses CSV syntax.  Literal double quotes keep the
        # comma-separated device list as one CSV field; no shell is involved.
        f"--gpus=\"device={','.join(str(device) for device in profile.gpu_devices)}\"",
        # The pinned image's NVIDIA entrypoint prints a CUDA banner to stdout,
        # which would corrupt the authenticated/framed worker channel before
        # Python can reserve it. Execute the validated worker interpreter directly.
        f"--entrypoint={profile.worker_python}",
        "--workdir=/tmp",
        f"--name={container_name}",
        f"--cidfile={cid_path}",
        _mount_arg(profile.source_dir, CONTAINER_SOURCE_PATH, readonly=True),
        _mount_arg(profile.model_dir, CONTAINER_MODEL_PATH, readonly=True),
        _mount_arg(_artifact_publication_path(profile), CONTAINER_ARTIFACT_PATH, readonly=True),
        _mount_arg(jit_dir, CONTAINER_JIT_PATH, readonly=False),
        _mount_arg(output_dir, CONTAINER_OUTPUT_PATH, readonly=False),
        *_runtime_overlay_mount_args(profile),
    ]
    if request.active:
        assert profile.bundle_dir is not None
        argv.append(_mount_arg(profile.bundle_dir, CONTAINER_BUNDLE_PATH, readonly=True))
    # dep_policy historically stores immutable overlay sources and mutable JIT
    # workspaces under one logical root.  Split them with a nested read-only mount:
    # candidate code cannot rewrite the trusted overlay stamp/tree, while every
    # launch still gets a fresh writable jit_workspace sibling.
    dep_overlay_v2 = _artifact_publication_path(profile) / "dep_overlay" / "v2"
    if dep_overlay_v2.is_dir():
        argv.append(
            _mount_arg(
                dep_overlay_v2,
                f"{CONTAINER_JIT_PATH}/dep_overlay/v2",
                readonly=True,
            )
        )
    for key, value in sorted(env.items()):
        argv.append(f"--env={key}={value}")
    argv.extend(
        (
            profile.image,
            "-m",
            "optima.eval.oci_worker",
            "--result",
            CONTAINER_RESULT_PATH,
        )
    )
    return tuple(argv)


def _session_environment(profile: OCILaunchProfile, *, active: bool) -> dict[str, str]:
    # Common values are arena/stock runtime policy, never candidate state.
    env = {
        key: value
        for key, value in _base_environment().items()
        if key not in {
            "OPTIMA_PREBUILT_ARTIFACTS",
            "OPTIMA_CUDA_EXT_CACHE",
            "OPTIMA_DEP_OVERLAY_CACHE",
            "OPTIMA_OCI_SYSTEM_OVERLAY_ROOT",
        }
    }
    env.update(profile.environment)
    env.update(_runtime_overlay_environment(profile))
    env.update(
        OPTIMA_OCI_EXPECTED_SGLANG_VERSION=profile.sglang_version,
        OPTIMA_OCI_EXPECTED_GPU_ARCH=profile.gpu_architecture,
        OPTIMA_OCI_EXPECTED_GPU_COUNT=str(len(profile.gpu_devices)),
        OPTIMA_OCI_EXPECTED_REFEREE_SOURCE_DIGEST=profile.referee_source_digest,
        OPTIMA_OCI_EXPECTED_MODEL_REVISION=profile.model_revision,
        OPTIMA_OCI_EXPECTED_MODEL_MANIFEST_DIGEST=profile.model_manifest_digest,
        OPTIMA_OCI_EXPECTED_MODEL_CONTENT_DIGEST=profile.model_content_digest,
    )
    if profile.topology_sha256 is not None:
        env["OPTIMA_OCI_EXPECTED_TOPOLOGY_SHA256"] = profile.topology_sha256
    if active:
        env.update(
            OPTIMA_PREBUILT_ARTIFACTS="1",
            OPTIMA_CUDA_EXT_CACHE=f"{CONTAINER_ARTIFACT_PATH}/cuda_ext",
            OPTIMA_DEP_OVERLAY_CACHE=f"{CONTAINER_JIT_PATH}/dep_overlay",
            OPTIMA_OCI_SYSTEM_OVERLAY_ROOT=(
                f"{CONTAINER_ARTIFACT_PATH}/system_overlay"
            ),
        )
        if profile.arena_name is not None:
            assert profile.competition_target is not None
            env["OPTIMA_OCI_ARENA_NAME"] = profile.arena_name
            env["OPTIMA_OCI_COMPETITION_TARGET"] = profile.competition_target
    attested_keys = sorted(env)
    env["OPTIMA_OCI_ATTEST_ENV_KEYS"] = ",".join(attested_keys)
    env["OPTIMA_OCI_ATTEST_ENV_SHA256"] = environment_fingerprint(
        {key: env[key] for key in attested_keys}
    )
    return env


def _expected_session_runtime(
    profile: OCILaunchProfile, environment: Mapping[str, str]
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "verified": True,
        "sglang_version": profile.sglang_version,
        "referee_source_digest": profile.referee_source_digest,
        "model_revision": profile.model_revision,
        "model_manifest_digest": profile.model_manifest_digest,
        "model_content_digest": profile.model_content_digest,
        "environment_sha256": environment["OPTIMA_OCI_ATTEST_ENV_SHA256"],
        "gpu_count": len(profile.gpu_devices),
        "gpu_architectures": [profile.gpu_architecture] * len(profile.gpu_devices),
    }
    if profile.topology_sha256 is not None:
        expected["topology_sha256"] = profile.topology_sha256
    return expected


def _build_session_argv(
    profile: OCILaunchProfile,
    *,
    mode: str,
    jit_dir: Path,
    stock_artifact_dir: Path | None,
    cid_path: Path,
    container_name: str,
) -> tuple[str, ...]:
    _verify_source_release_before_mount(profile)
    active = mode != "baseline"
    if active and profile.bundle_dir is None:
        raise OCIBackendError("candidate OCI session requires bundle_dir")
    if not active and stock_artifact_dir is None:
        raise OCIBackendError("baseline OCI session requires an empty stock artifact root")
    env = _session_environment(profile, active=active)
    artifact_source = _artifact_publication_path(profile) if active else stock_artifact_dir
    assert artifact_source is not None
    argv = [
        profile.docker_binary,
        "run",
        "--rm",
        "--init",
        "--interactive",
        "--pull=never",
        "--runtime=runc",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--cap-add=SYS_NICE",
        "--cap-add=SYS_RESOURCE",
        "--security-opt=no-new-privileges=true",
        _seccomp_security_arg(profile),
        f"--user={profile.worker_uid}:{profile.worker_gid}",
        # Private PID namespace is Docker's fail-closed default; see above.
        "--ipc=private",
        "--log-driver=none",
        "--ulimit=core=0:0",
        *_resource_argv(profile),
        f"--shm-size={profile.shm_size}",
        f"--pids-limit={profile.pids_limit}",
        f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={profile.tmpfs_size}",
        f"--gpus=\"device={','.join(str(device) for device in profile.gpu_devices)}\"",
        f"--entrypoint={profile.worker_python}",
        "--workdir=/tmp",
        f"--name={container_name}",
        f"--cidfile={cid_path}",
        _mount_arg(profile.source_dir, CONTAINER_SOURCE_PATH, readonly=True),
        _mount_arg(profile.model_dir, CONTAINER_MODEL_PATH, readonly=True),
        _mount_arg(artifact_source, CONTAINER_ARTIFACT_PATH, readonly=True),
        _mount_arg(jit_dir, CONTAINER_JIT_PATH, readonly=False),
        *_runtime_overlay_mount_args(profile),
    ]
    if active:
        assert profile.bundle_dir is not None
        argv.append(_mount_arg(profile.bundle_dir, CONTAINER_BUNDLE_PATH, readonly=True))
        dep_overlay_v2 = _artifact_publication_path(profile) / "dep_overlay" / "v2"
        if dep_overlay_v2.is_dir():
            argv.append(_mount_arg(
                dep_overlay_v2,
                f"{CONTAINER_JIT_PATH}/dep_overlay/v2",
                readonly=True,
            ))
    for key, value in sorted(env.items()):
        argv.append(f"--env={key}={value}")
    argv.extend((
        profile.image,
        "-m",
        "optima.eval.oci_session_worker",
    ))
    return tuple(argv)


class OCIExecutor(Protocol):
    def run(self, launch: PreparedOCILaunch, *, timeout_s: float) -> int:
        """Run the prepared argv and return its process exit code."""


class SubprocessOCIExecutor:
    """Docker-compatible executor with bounded watchdog cleanup and no shell."""

    @staticmethod
    def _force_remove(launch: PreparedOCILaunch) -> None:
        identifier = launch.container_name
        try:
            if launch.cid_path.is_file():
                raw = launch.cid_path.read_text(encoding="ascii").strip()
                if _CID.fullmatch(raw):
                    identifier = raw
        except OSError:
            pass
        try:
            subprocess.run(
                [launch.profile.docker_binary, "rm", "--force", identifier],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            # The independent list below is authoritative.  A timed-out rm can
            # still have completed in the daemon; a broken daemon cannot produce
            # the required successful empty list.
            pass
        try:
            listed = subprocess.run(
                [
                    launch.profile.docker_binary,
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{launch.container_name}$",
                    "--format={{.ID}}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OCIInfrastructureError(
                f"could not verify scored container removal: {exc}"
            ) from None
        if (
            type(listed.returncode) is not int
            or listed.returncode != 0
            or not isinstance(listed.stdout, bytes)
            or not isinstance(listed.stderr, bytes)
            or len(listed.stdout) > 4096
            or len(listed.stderr) > 4096
        ):
            raise OCIInfrastructureError(
                "could not obtain a bounded successful container-absence listing"
            )
        if listed.stdout.strip():
            raise OCIInfrastructureError(
                "scored container still exists after forced removal"
            )

    @staticmethod
    def _terminate_client(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=10)

    def run(self, launch: PreparedOCILaunch, *, timeout_s: float) -> int:
        try:
            proc = subprocess.Popen(
                list(launch.argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
        except OSError as exc:
            raise OCIInfrastructureError(f"could not start OCI runtime: {exc}") from None
        try:
            proc.communicate(input=launch.stdin_frame, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._force_remove(launch)
            self._terminate_client(proc)
            raise OCIWatchdogTimeout(
                f"OCI launch timed out after {timeout_s:g}s and was force-removed"
            ) from None
        except BaseException:
            # KeyboardInterrupt/controller cancellation must not strand a model-sized
            # GPU container after its launch directory and cidfile are cleaned up.
            self._force_remove(launch)
            self._terminate_client(proc)
            raise
        return int(proc.returncode)


class OCILauncher:
    """Prepare and execute fresh, authenticated OCI engine launches."""

    def __init__(
        self,
        profile: OCILaunchProfile,
        *,
        executor: OCIExecutor | None = None,
        prebuild_executor: OCIExecutor | None = None,
        session_factory=None,
        device_guard=None,
    ):
        self.profile = profile
        self.runtime_preflight_receipt = profile.runtime_preflight_receipt
        self.executor = executor or SubprocessOCIExecutor()
        self.prebuild_executor = prebuild_executor or SubprocessOCIExecutor()
        self.session_factory = session_factory
        if profile.device_state_policy is None:
            if device_guard is not None:
                raise OCIBackendError(
                    "an injected device guard requires a profile-owned device policy"
                )
            self.device_guard = None
        else:
            self.device_guard = (
                DeviceStateGuard(profile.device_state_policy)
                if device_guard is None else device_guard
            )
            if getattr(self.device_guard, "policy", None) != profile.device_state_policy:
                raise OCIBackendError(
                    "injected device guard differs from the profile-owned policy"
                )
            if not all(
                callable(getattr(self.device_guard, name, None))
                for name in ("before_arm", "condition_active", "after_arm")
            ):
                raise OCIBackendError(
                    "injected device guard lacks the required pre/active/post interface"
                )
        self.attestation_receipts: list[dict[str, Any]] = []
        # Every attempt remains locally auditable, but only complete successful
        # pre/active/post triplets are settlement authority. Failed retries may
        # create sequence gaps in the retained authoritative stream.
        self.device_diagnostic_receipts: list[dict[str, Any]] = []
        self._arm_counts: dict[str, int] = {}
        self._last_device_sequence = 0
        self._evaluation_deadline: float | None = None

    def begin_evaluation(
        self, *, timeout_s: float | None = None, reset: bool = False
    ) -> None:
        """Start one absolute deadline shared by every B/C/B' arm and retry."""

        if self._evaluation_deadline is not None and not reset:
            return
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not 0 < float(timeout_s) <= 86_400
        ):
            raise OCIBackendError("evaluation timeout_s must be in (0, 86400]")
        if timeout_s is None:
            timeout_s = float(self.profile.bracket_timeout_s)
        else:
            # Operator watchdogs may only tighten the consensus arena cap.
            timeout_s = min(float(timeout_s), float(self.profile.bracket_timeout_s))
        self._evaluation_deadline = time.monotonic() + float(timeout_s)
        self.attestation_receipts.clear()
        self.device_diagnostic_receipts.clear()
        self._arm_counts.clear()
        self._last_device_sequence = 0

    def _remaining_evaluation_time(self) -> float:
        if self._evaluation_deadline is None:
            self.begin_evaluation()
        assert self._evaluation_deadline is not None
        remaining = self._evaluation_deadline - time.monotonic()
        if remaining <= 0:
            raise OCIInfrastructureError("absolute OCI evaluation deadline expired")
        return remaining

    def _next_arm_label(self, mode: str, requested: str | None) -> str:
        base = requested or mode.replace("_", "-")
        if not isinstance(base, str) or _PROFILE_ID.fullmatch(base) is None:
            raise OCIBackendError("OCI arm label must be a simple 1..128 character ID")
        count = self._arm_counts.get(base, 0) + 1
        self._arm_counts[base] = count
        label = f"{base}-{count}"
        if _PROFILE_ID.fullmatch(label) is None:
            raise OCIBackendError("numbered OCI arm label exceeds its hard bound")
        return label

    def _record_device_receipt(
        self, receipt: object, *, arm: str, phase: str
    ) -> dict[str, Any]:
        policy = self.profile.device_state_policy
        if policy is None or type(receipt) is not DeviceStateReceipt:
            raise OCIBackendError("device guard returned an invalid receipt type")
        if (
            receipt.schema != "optima.device-state-receipt.v1"
            or receipt.arm != arm
            or receipt.phase != phase
            or receipt.selected_physical_gpu_ids != self.profile.gpu_devices
            or receipt.configuration_sha256 != policy.configuration_sha256
            or receipt.policy_sha256 != policy.policy_sha256
            or receipt.sequence <= self._last_device_sequence
        ):
            raise OCIBackendError("device guard receipt identity/order mismatch")
        try:
            raw = json.dumps(
                dataclasses.asdict(receipt),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            canonical = json.loads(raw)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise OCIBackendError(
                f"device guard receipt is not canonical JSON evidence: {exc}"
            ) from None
        self._last_device_sequence = receipt.sequence
        self.device_diagnostic_receipts.append(canonical)
        return canonical

    def _device_receipt(
        self, *, arm: str, phase: str
    ) -> dict[str, Any] | None:
        if self.device_guard is None:
            return None
        if self._evaluation_deadline is None:
            self.begin_evaluation()
        assert self._evaluation_deadline is not None
        try:
            if phase == "pre":
                receipt = self.device_guard.before_arm(
                    arm, deadline=self._evaluation_deadline
                )
            elif phase == "post":
                receipt = self.device_guard.after_arm(
                    arm, deadline=self._evaluation_deadline
                )
            else:  # pragma: no cover - controller-only call sites are fixed
                raise OCIBackendError(f"unsupported device receipt phase {phase!r}")
        except (DeviceStateError, DeviceStatePolicyError) as exc:
            raise _mapped_device_state_error(
                exc, context=f"trusted-host {phase}-arm device attestation"
            ) from None
        return self._record_device_receipt(receipt, arm=arm, phase=phase)

    def _record_active_device_receipt(
        self, receipt: DeviceStateActiveReceipt, *, arm: str
    ) -> dict[str, Any]:
        policy = self.profile.device_state_policy
        if (
            policy is None
            or type(receipt) is not DeviceStateActiveReceipt
            or receipt.schema != "optima.device-state-active-receipt.v2"
            or receipt.arm != arm
            or receipt.event != _FinalWarmupConditioner._EVENT
            or receipt.selected_physical_gpu_ids != self.profile.gpu_devices
            or receipt.configuration_sha256 != policy.configuration_sha256
            or receipt.policy_sha256 != policy.policy_sha256
            or receipt.sequence <= self._last_device_sequence
            or receipt.consecutive_active_samples
            != policy.required_consecutive_idle_samples
            or receipt.release_sample_index < receipt.consecutive_active_samples
            or receipt.release_sample_index >= len(receipt.samples)
            or receipt.post_release_ready_samples != 1
            or len(receipt.samples) - receipt.release_sample_index
            < receipt.post_release_ready_samples
            or not all(
                sample.active_envelope_passed
                for sample in receipt.samples[
                    receipt.release_sample_index - receipt.consecutive_active_samples:
                    receipt.release_sample_index
                ]
            )
            or not all(
                sample.active_envelope_passed
                for sample in receipt.samples[-receipt.post_release_ready_samples:]
            )
        ):
            raise OCIBackendError(
                "device guard active-conditioning receipt identity/order mismatch"
            )
        try:
            raw = json.dumps(
                dataclasses.asdict(receipt),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            canonical = json.loads(raw)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise OCIBackendError(
                "active-conditioning receipt is not canonical JSON evidence: "
                f"{exc}"
            ) from None
        self._last_device_sequence = receipt.sequence
        self.device_diagnostic_receipts.append(canonical)
        return canonical

    def prepare_prebuild(self) -> PreparedOCIPrebuild:
        """Construct (but do not execute) the same-image trusted build container."""

        if self.profile.bundle_dir is None:
            raise OCIBackendError("artifact prebuild requires bundle_dir")
        _verify_source_release_before_mount(self.profile)
        request_id = secrets.token_hex(16)
        launch_root = Path(tempfile.mkdtemp(
            prefix=f"prebuild-{request_id[:12]}-", dir=self.profile.scratch_root
        ))
        staging_artifact_dir: Path | None = None
        tmpfs_mounts: list[Path] = []
        try:
            os.chmod(launch_root, 0o700)
            jit_dir = launch_root / "jit"
            jit_dir.mkdir(mode=0o700)
            if _mount_worker_tmpfs(
                jit_dir,
                size=self.profile.tmpfs_size,
                nr_inodes=self.profile.scratch_tmpfs_inodes,
                exec_allowed=True,
                profile=self.profile,
            ):
                tmpfs_mounts.append(jit_dir)
            cid_path = launch_root / "container.cid"
            publication_dir = _artifact_publication_path(self.profile)
            if publication_dir.exists():
                raise OCIBackendError(
                    "candidate artifact publication already exists; validate/reuse it "
                    "instead of rebuilding in place"
                )
            staging_artifact_dir = Path(tempfile.mkdtemp(
                prefix=f".stage-{request_id}-", dir=self.profile.artifact_dir
            ))
            os.chmod(staging_artifact_dir, 0o700)
            build_artifact_dir = launch_root / "build_artifacts"
            build_artifact_dir.mkdir(mode=0o700)
            if _mount_worker_tmpfs(
                build_artifact_dir,
                size=self.profile.artifact_tmpfs_size,
                nr_inodes=self.profile.artifact_tmpfs_inodes,
                exec_allowed=False,
                profile=self.profile,
            ):
                tmpfs_mounts.append(build_artifact_dir)
            receipt_path = (
                build_artifact_dir / "prebuild_receipts" / f"{request_id}.json"
            )
            container_name = f"optima-prebuild-{request_id}"
            env = _base_environment()
            env.update(self.profile.environment)
            env.update(
                OPTIMA_ACTIVE="0",
                OPTIMA_BUNDLE_PATH="",
                OPTIMA_REBUILD_PHASE="build",
                OPTIMA_CUDA_EXT_CACHE=f"{CONTAINER_ARTIFACT_PATH}/cuda_ext",
                OPTIMA_DEP_OVERLAY_CACHE=f"{CONTAINER_ARTIFACT_PATH}/dep_overlay",
            )
            env.update(_runtime_overlay_environment(self.profile))
            env.update(_offline_build_environment(self.profile))
            if self.profile.arena_name is not None:
                assert self.profile.competition_target is not None
                env["OPTIMA_OCI_ARENA_NAME"] = self.profile.arena_name
                env["OPTIMA_OCI_COMPETITION_TARGET"] = self.profile.competition_target
            argv = [
                self.profile.docker_binary,
                "run",
                "--rm",
                "--init",
                "--pull=never",
                "--runtime=runc",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--cap-add=SYS_NICE",
                "--cap-add=SYS_RESOURCE",
                "--security-opt=no-new-privileges=true",
                _seccomp_security_arg(self.profile),
                f"--user={self.profile.worker_uid}:{self.profile.worker_gid}",
                # Private PID namespace is Docker's fail-closed default.
                "--ipc=private",
                "--log-driver=none",
                "--ulimit=core=0:0",
                *_resource_argv(self.profile),
                f"--shm-size={self.profile.shm_size}",
                f"--pids-limit={self.profile.pids_limit}",
                f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={self.profile.tmpfs_size}",
                f"--entrypoint={self.profile.worker_python}",
                "--workdir=/tmp",
                f"--name={container_name}",
                f"--cidfile={cid_path}",
                _mount_arg(self.profile.source_dir, CONTAINER_SOURCE_PATH, readonly=True),
                _mount_arg(self.profile.model_dir, CONTAINER_MODEL_PATH, readonly=True),
                _mount_arg(self.profile.bundle_dir, CONTAINER_BUNDLE_PATH, readonly=True),
                _mount_arg(build_artifact_dir, CONTAINER_ARTIFACT_PATH, readonly=False),
                _mount_arg(jit_dir, CONTAINER_JIT_PATH, readonly=False),
                *_runtime_overlay_mount_args(self.profile),
            ]
            for key, value in sorted(env.items()):
                argv.append(f"--env={key}={value}")
            argv.extend((
                self.profile.image,
                "-m",
                "optima.eval.oci_prebuild",
                "--request-id",
                request_id,
            ))
            return PreparedOCIPrebuild(
                profile=self.profile,
                request_id=request_id,
                launch_root=launch_root,
                jit_dir=jit_dir,
                build_artifact_dir=build_artifact_dir,
                staging_artifact_dir=staging_artifact_dir,
                publication_dir=publication_dir,
                cid_path=cid_path,
                receipt_path=receipt_path,
                container_name=container_name,
                argv=tuple(argv),
                tmpfs_mounts=tuple(tmpfs_mounts),
            )
        except BaseException:
            _cleanup_tmpfs_mounts(tmpfs_mounts, recovery_root=launch_root)
            shutil.rmtree(launch_root, ignore_errors=True)
            if staging_artifact_dir is not None:
                shutil.rmtree(staging_artifact_dir, ignore_errors=True)
            raise

    def prebuild_candidate_artifacts(self, *, timeout_s: float | None = None) -> Path:
        """Build candidate artifacts in the pinned image, never in the timer."""

        if self.profile.bundle_dir is None:
            raise OCIBackendError("artifact prebuild requires bundle_dir")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or float(timeout_s) <= 0
        ):
            raise OCIBackendError("prebuild timeout must be positive")
        self.begin_evaluation(timeout_s=timeout_s)
        timeout_s = min(
            float(self.profile.prebuild_timeout_s),
            self._remaining_evaluation_time(),
            float(timeout_s) if timeout_s is not None else float("inf"),
        )
        from optima.bundle_hash import content_hash

        bundle_hash = content_hash(self.profile.bundle_dir)
        publication = _artifact_publication_path(self.profile)
        if publication.exists():
            manifest = _verify_artifact_publication(
                publication, profile=self.profile, bundle_hash=bundle_hash
            )
            receipt = (
                publication / "prebuild_receipts" /
                f"{manifest['request_id']}.json"
            )
            if not receipt.is_file() or receipt.is_symlink():
                raise OCIBackendError("published artifact receipt is missing or unsafe")
            return receipt

        launch = self.prepare_prebuild()
        published = False
        try:
            try:
                returncode = self.prebuild_executor.run(launch, timeout_s=timeout_s)
            except OCIWatchdogTimeout as exc:
                raise OCICandidateArtifactError(
                    f"candidate prebuild exceeded its arena deadline: {exc}"
                ) from None
            if returncode != 0:
                raise OCICandidateArtifactError(
                    f"OCI artifact prebuild exited with status {returncode}"
                )
            try:
                data = _read_regular_bounded(launch.receipt_path, max_bytes=64 * 1024)
            except OCIBackendError as exc:
                raise OCICandidateArtifactError(
                    f"OCI prebuild receipt is missing/unsafe: {exc}"
                ) from None
            try:
                receipt = __import__("json").loads(data)
            except (ValueError, UnicodeError):
                raise OCICandidateArtifactError("OCI prebuild receipt is malformed") from None
            expected = {
                "schema", "request_id", "bundle_hash", "rebuild_plan",
                "dep_targets", "system_cache_key", "system_dest",
            }
            if (
                not isinstance(receipt, dict)
                or set(receipt) != expected
                or receipt.get("schema") != "optima-oci-prebuild-v1"
                or receipt.get("request_id") != launch.request_id
            ):
                raise OCICandidateArtifactError(
                    "OCI prebuild receipt identity/schema mismatch"
                )
            if receipt.get("bundle_hash") != bundle_hash:
                raise OCICandidateArtifactError("OCI prebuild receipt bundle hash mismatch")
            try:
                # Build coordination state is neither executable candidate output
                # nor retained evidence. Remove only exact validator-owned lock
                # shapes before the strict hidden-path/artifact scan.
                _remove_build_locks(launch.build_artifact_dir)
                _copy_validated_artifact_tree(
                    launch.build_artifact_dir,
                    launch.staging_artifact_dir,
                    profile=self.profile,
                )
            except OCIBackendError as exc:
                raise OCICandidateArtifactError(
                    f"candidate artifact tree is invalid: {exc}"
                ) from None
            _write_artifact_publication_manifest(
                launch.staging_artifact_dir,
                profile=self.profile,
                request_id=launch.request_id,
                bundle_hash=bundle_hash,
            )
            _freeze_artifact_tree(launch.staging_artifact_dir)
            if launch.publication_dir.exists():
                raise OCIBackendError(
                    "artifact publication path appeared during prebuild; refusing race"
                )
            try:
                os.rename(launch.staging_artifact_dir, launch.publication_dir)
            except OSError as exc:
                raise OCIBackendError(f"atomic artifact publication failed: {exc}") from None
            published = True
            try:
                parent_fd = os.open(
                    self.profile.artifact_dir,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            except OSError as exc:
                raise OCIBackendError(f"artifact publication fsync failed: {exc}") from None
            _verify_artifact_publication(
                launch.publication_dir, profile=self.profile, bundle_hash=bundle_hash
            )
            return (
                launch.publication_dir / "prebuild_receipts" /
                f"{launch.request_id}.json"
            )
        finally:
            if not published:
                _remove_artifact_tree(launch.staging_artifact_dir)
            launch.cleanup()

    def prebuild_system_overlay(self, *, timeout_s: float | None = None) -> Path:
        """Compatibility-named hook for a system bundle's same-image prebuild."""

        if self.profile.arena_name is None or self.profile.competition_target is None:
            raise OCIBackendError(
                "system prebuild requires arena_name and competition_target"
            )
        return self.prebuild_candidate_artifacts(timeout_s=timeout_s)

    def prepare(
        self,
        config: Any,
        prompt_batches: Sequence[Sequence[str]],
        *,
        mode: str,
    ) -> PreparedOCILaunch:
        """Legacy in-container-timed path; retained for diagnostics, never scoring."""
        if mode != "baseline":
            if self.profile.bundle_dir is None:
                raise OCIBackendError("candidate OCI launch requires bundle_dir")
            from optima.bundle_hash import content_hash

            _verify_artifact_publication(
                _artifact_publication_path(self.profile),
                profile=self.profile,
                bundle_hash=content_hash(self.profile.bundle_dir),
            )
        auth_key = secrets.token_bytes(AUTH_KEY_BYTES)
        nonce = secrets.token_bytes(AUTH_NONCE_BYTES)
        request_id = secrets.token_hex(16)
        try:
            request = make_request(
                config,
                prompt_batches,
                mode=mode,
                request_id=request_id,
                nonce=nonce,
            )
        except OCIProtocolError as exc:
            raise OCIBackendError(str(exc)) from None

        prefix = f"{mode.replace('_', '-')}-{request_id[:12]}-"
        launch_root = Path(tempfile.mkdtemp(prefix=prefix, dir=self.profile.scratch_root))
        tmpfs_mounts: list[Path] = []
        try:
            os.chmod(launch_root, 0o700)
            jit_dir = launch_root / "jit"
            output_dir = launch_root / "output"
            jit_dir.mkdir(mode=0o700)
            output_dir.mkdir(mode=0o700)
            if _mount_worker_tmpfs(
                jit_dir,
                size=self.profile.tmpfs_size,
                nr_inodes=self.profile.scratch_tmpfs_inodes,
                exec_allowed=True,
                profile=self.profile,
            ):
                tmpfs_mounts.append(jit_dir)
            _prepare_worker_directory(output_dir, self.profile)
            # Docker can bind a trusted overlay subtree on this nested target only
            # when the destination exists inside the outer writable JIT bind.
            if (
                _artifact_publication_path(self.profile) / "dep_overlay" / "v2"
            ).is_dir():
                nested = jit_dir / "dep_overlay" / "v2"
                nested.mkdir(parents=True, mode=0o700)
                _prepare_worker_directory(nested.parent, self.profile)
                _prepare_worker_directory(nested, self.profile)
            result_path = output_dir / "result.auth"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(result_path, flags, 0o600)
            os.close(fd)
            if os.geteuid() == 0:
                os.chown(result_path, self.profile.worker_uid, self.profile.worker_gid)
            cid_path = launch_root / "container.cid"
            container_name = f"optima-{mode.replace('_', '-')}-{request_id}"
            if _SAFE_CONTAINER_NAME.fullmatch(container_name) is None:
                raise OCIBackendError("generated OCI container name is invalid")
            argv = _build_argv(
                self.profile,
                request,
                launch_root=launch_root,
                jit_dir=jit_dir,
                output_dir=output_dir,
                cid_path=cid_path,
                container_name=container_name,
            )
            return PreparedOCILaunch(
                profile=self.profile,
                request=request,
                launch_root=launch_root,
                jit_dir=jit_dir,
                output_dir=output_dir,
                result_path=result_path,
                cid_path=cid_path,
                container_name=container_name,
                argv=argv,
                stdin_frame=encode_stdin_frame(request, auth_key=auth_key),
                auth_key=auth_key,
                tmpfs_mounts=tuple(tmpfs_mounts),
            )
        except BaseException:
            _cleanup_tmpfs_mounts(tmpfs_mounts, recovery_root=launch_root)
            shutil.rmtree(launch_root, ignore_errors=True)
            raise

    def prepare_session(self, *, mode: str) -> PreparedOCISession:
        """Prepare one fresh outer-timed B, C, B' or audit container session."""

        if mode not in {"baseline", "candidate", "candidate_audit"}:
            raise OCIBackendError(f"unsupported OCI session mode {mode!r}")
        active = mode != "baseline"
        if active and self.profile.bundle_dir is None:
            raise OCIBackendError("candidate OCI session requires bundle_dir")
        if active:
            assert self.profile.bundle_dir is not None
            from optima.bundle_hash import content_hash

            _verify_artifact_publication(
                _artifact_publication_path(self.profile),
                profile=self.profile,
                bundle_hash=content_hash(self.profile.bundle_dir),
            )
        request_id = secrets.token_hex(16)
        launch_root = Path(tempfile.mkdtemp(
            prefix=f"session-{mode.replace('_', '-')}-{request_id[:12]}-",
            dir=self.profile.scratch_root,
        ))
        tmpfs_mounts: list[Path] = []
        try:
            os.chmod(launch_root, 0o700)
            jit_dir = launch_root / "jit"
            jit_dir.mkdir(mode=0o700)
            if _mount_worker_tmpfs(
                jit_dir,
                size=self.profile.tmpfs_size,
                nr_inodes=self.profile.scratch_tmpfs_inodes,
                exec_allowed=True,
                profile=self.profile,
            ):
                tmpfs_mounts.append(jit_dir)
            stock_artifact_dir = None
            if active:
                if (
                    _artifact_publication_path(self.profile) / "dep_overlay" / "v2"
                ).is_dir():
                    nested = jit_dir / "dep_overlay" / "v2"
                    nested.mkdir(parents=True, mode=0o700)
                    _prepare_worker_directory(nested.parent, self.profile)
                    _prepare_worker_directory(nested, self.profile)
            else:
                # B and B' get a unique, empty stock root. They never see candidate
                # artifacts, overlays, bundle bytes, or candidate activation env.
                stock_artifact_dir = launch_root / "stock_artifacts"
                stock_artifact_dir.mkdir(mode=0o700)
                _prepare_worker_directory(stock_artifact_dir, self.profile)
            cid_path = launch_root / "container.cid"
            container_name = f"optima-session-{mode.replace('_', '-')}-{request_id}"
            argv = _build_session_argv(
                self.profile,
                mode=mode,
                jit_dir=jit_dir,
                stock_artifact_dir=stock_artifact_dir,
                cid_path=cid_path,
                container_name=container_name,
            )
            session_environment = _session_environment(
                self.profile, active=active
            )
            return PreparedOCISession(
                profile=self.profile,
                mode=mode,
                launch_root=launch_root,
                jit_dir=jit_dir,
                stock_artifact_dir=stock_artifact_dir,
                cid_path=cid_path,
                container_name=container_name,
                argv=argv,
                expected_runtime_attestation=MappingProxyType(
                    _expected_session_runtime(
                        self.profile, session_environment
                    )
                ),
                tmpfs_mounts=tuple(tmpfs_mounts),
            )
        except BaseException:
            _cleanup_tmpfs_mounts(tmpfs_mounts, recovery_root=launch_root)
            shutil.rmtree(launch_root, ignore_errors=True)
            raise

    def run(
        self,
        config: Any,
        prompt_batches: Sequence[Sequence[str]],
        *,
        mode: str,
        timeout_s: float | None = None,
        arm: str | None = None,
        posthoc_plan=None,
    ) -> Any:
        """Run one authoritative outer-timed, non-executable evidence session."""

        from optima.eval.oci_outer_session import (
            ContainerSessionTransport,
            OuterSessionCandidateError,
            OuterSessionTimeoutError,
            run_outer_timed_session,
        )

        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not 0 < float(timeout_s) <= 86_400
        ):
            raise OCIBackendError("timeout_s must be in (0, 86400]")
        operator_timeout_s = (
            float(timeout_s) if timeout_s is not None else float("inf")
        )
        self._remaining_evaluation_time()
        arm_label = self._next_arm_label(mode, arm)
        launch = self.prepare_session(mode=mode)
        pre_attested = False
        arm_completed = False
        arm_receipts: list[dict[str, Any]] = []
        conditioner: _FinalWarmupConditioner | None = None
        try:
            pre_receipt = self._device_receipt(arm=arm_label, phase="pre")
            pre_attested = self.device_guard is not None
            if pre_receipt is not None:
                arm_receipts.append(pre_receipt)
            if self.device_guard is not None:
                assert self._evaluation_deadline is not None
                conditioner = _FinalWarmupConditioner(
                    guard=self.device_guard,
                    arm=arm_label,
                    mode=mode,
                    evaluation_deadline=self._evaluation_deadline,
                    record=lambda receipt: arm_receipts.append(
                        self._record_active_device_receipt(
                            receipt, arm=arm_label
                        )
                    ),
                )

            # A protocol watchdog must not consume the mandatory post-arm drain.
            # Recompute after the pre-drain: it is part of the same absolute B/C/B'
            # bracket budget and may have consumed meaningful wall time.
            post_reserve_s = (
                float(self.profile.device_state_policy.drain_timeout_s)
                if self.profile.device_state_policy is not None else 0.0
            )
            session_budget_s = self._remaining_evaluation_time() - post_reserve_s
            if session_budget_s <= 0:
                raise OCIInfrastructureError(
                    "OCI bracket lacks time for a session plus mandatory post-arm drain"
                )
            session_timeout_s = min(operator_timeout_s, session_budget_s)
            if self.session_factory is None:
                transport = ContainerSessionTransport(
                    launch.argv,
                    force_remove=lambda: SubprocessOCIExecutor._force_remove(launch),
                )
            else:
                transport = self.session_factory(launch)
            try:
                try:
                    result = run_outer_timed_session(
                        config,
                        prompt_batches,
                        mode=mode,
                        transport=transport,
                        init_timeout_s=min(
                            session_timeout_s, float(self.profile.init_timeout_s)
                        ),
                        batch_timeout_s=min(
                            session_timeout_s, float(self.profile.batch_timeout_s)
                        ),
                        total_timeout_s=session_timeout_s,
                        warmup_timed_boundary=(
                            conditioner.boundary if conditioner is not None else None
                        ),
                        expected_runtime_attestation=(
                            launch.expected_runtime_attestation
                        ),
                        posthoc_plan=posthoc_plan,
                    )
                except OuterSessionTimeoutError as exc:
                    if mode != "baseline":
                        raise OuterSessionCandidateError(
                            "candidate arm exceeded its arena phase deadline: "
                            f"{exc}"
                        ) from None
                    raise
                if conditioner is not None:
                    conditioner.require_complete()
                arm_completed = True
                return result
            except BaseException as primary:
                if conditioner is not None:
                    try:
                        conditioner.cancel()
                    except BaseException as cleanup_error:  # noqa: BLE001
                        if hasattr(primary, "add_note"):
                            primary.add_note(
                                "final-warmup telemetry cleanup also failed: "
                                f"{cleanup_error}"
                            )
                raise
        finally:
            try:
                launch.cleanup()
            finally:
                if pre_attested:
                    post_receipt = self._device_receipt(
                        arm=arm_label, phase="post"
                    )
                    assert post_receipt is not None
                    arm_receipts.append(post_receipt)
                    if arm_completed:
                        schemas = tuple(
                            receipt.get("schema") for receipt in arm_receipts
                        )
                        if schemas != (
                            "optima.device-state-receipt.v1",
                            "optima.device-state-active-receipt.v2",
                            "optima.device-state-receipt.v1",
                        ) or any(
                            receipt.get("arm") != arm_label
                            for receipt in arm_receipts
                        ):
                            raise OCIBackendError(
                                "successful arm lacks one exact pre/active/post "
                                "device receipt triplet"
                            )
                        self.attestation_receipts.extend(arm_receipts)

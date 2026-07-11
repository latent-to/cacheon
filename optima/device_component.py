"""Validator-owned host launcher for development-only device components.

This module is the enforceable distinction that the original slot loader did
not make.  Importing a miner's Triton/CuTe/Python module into the scheduler gives
that submission arbitrary host control, even when it declares no ``setup()``.
Such a submission can still be evaluated in an isolated system lane, but it is
not a component trust boundary.

The ``validator_device`` lane has a deliberately smaller host surface:

* miner input: inspectable device source plus a kernel symbol and capability data;
* trusted build: offline compilation to a content-addressed cubin (never dlopen);
* trusted runtime: this module loads the cubin with the CUDA driver and constructs
  every launch argument/grid/block from a validator-owned ABI adapter;
* no miner Python, native host extension, static initializer, setup/prepare hook,
  dependency patch, or miner-provided launch expression executes.

That is useful hardening, but it is **not** a component trust boundary.  A CUDA
kernel receives raw device pointers in the model's CUDA context.  CUDA provides no
memory protection between those allocations, so malicious device code can read or
overwrite tensors outside the declared ABI, including downstream model state.  Until
device-memory isolation exists, this lane is verification/development-only and may
not produce a component or atomic crown.  It can still be graded as part of an
externally observed whole-serving product if a future system-lane policy admits it.

The first vertical slice is ``activation.silu_and_mul.cuda.v1``.  Triton and
CuTeDSL remain feasible, but need validator-owned AOT compiler adapters which
emit this same cubin+symbol product; their ordinary JIT Python wrappers belong
to ``untrusted_host`` until then.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Literal, Mapping

if TYPE_CHECKING:
    from optima.manifest import Manifest, OpEntry


DEVICE_CACHE_SCHEMA = "optima-device-cubin-v1"
SILU_CUDA_ABI = "activation.silu_and_mul.cuda.v1"
UNTRUSTED_HOST_SYSTEM_TARGET = "sglang.inference.bundle.v1"
_NVCC_FLAGS = ("--cubin", "-O3", "--use_fast_math", "--std=c++17")


class DeviceComponentError(RuntimeError):
    """A validator-device declaration, artifact, or launch failed closed."""


@dataclass(frozen=True)
class DeviceABI:
    """One validator-owned host/device calling convention.

    ``entry_factory`` receives a validated cubin path and the miner-selected
    device symbol, and returns an ordinary slot entry callable.  The factory is
    validator code; the bundle can only select a registered ``name``.
    """

    name: str
    slot: str
    dtypes: frozenset[str]
    entry_factory: Callable[[Path, str], Callable[..., None]]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _command_output(argv: list[str]) -> str:
    try:
        return subprocess.run(
            argv,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DeviceComponentError(
            f"unable to identify device compiler command {argv!r}: {exc}"
        ) from exc


def _tool_identity(name: str) -> dict[str, str]:
    found = shutil.which(name)
    if found is None:
        raise DeviceComponentError(f"required CUDA tool is missing: {name}")
    path = Path(found).resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "version": _command_output([str(path), "--version"]),
    }


def _cache_root() -> Path:
    raw = os.environ.get("OPTIMA_DEVICE_CACHE", "").strip()
    return Path(raw) if raw else Path.home() / ".cache" / "optima" / "device_cubin"


def _current_arch(*, offline: bool = False) -> str:
    if offline:
        raw = os.environ.get("OPTIMA_TARGET_GPU_ARCH", "").strip()
        if re.fullmatch(r"sm_[0-9]{2,3}a", raw) is None:
            raise DeviceComponentError(
                "offline device build requires validator-owned OPTIMA_TARGET_GPU_ARCH"
            )
        return raw
    try:
        import torch

        if not torch.cuda.is_available():
            raise DeviceComponentError(
                "validator_device artifact preparation requires a CUDA device"
            )
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    except DeviceComponentError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize runtime/toolchain failures
        raise DeviceComponentError(f"unable to resolve CUDA architecture: {exc}") from exc
    # Optima's supported Hopper/Blackwell toolchains use architecture-specific
    # features, so compile the exact accelerated target rather than forward PTX.
    return f"sm_{major}{minor}a"


def _artifact_identity(
    bundle: Path, op: "OpEntry", arch: str
) -> tuple[str, dict]:
    from optima.bundle_hash import content_hash

    nvcc = _tool_identity("nvcc")
    ptxas = _tool_identity("ptxas")
    payload = {
        "schema": DEVICE_CACHE_SCHEMA,
        "bundle_hash": content_hash(bundle),
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "slot": op.slot,
        "variant": op.variant,
        "source": op.source,
        "symbol": op.entry,
        "device_abi": op.device_abi,
        "dtypes": list(op.dtypes),
        "architectures": list(op.architectures),
        "arch": arch,
        "flags": list(_NVCC_FLAGS),
        "nvcc": nvcc,
        "ptxas": ptxas,
    }
    return _canonical_hash(payload), payload


def _artifact_paths(bundle_hash: str, artifact_id: str) -> tuple[Path, Path, Path]:
    root = _cache_root() / "v1" / bundle_hash[:2] / bundle_hash / artifact_id
    return root / "kernel.cubin", root / "artifact.json", root


def _stamp(identity: dict, artifact_id: str, cubin: Path) -> dict:
    return {
        "schema": DEVICE_CACHE_SCHEMA,
        "artifact_id": artifact_id,
        "identity": identity,
        "artifact_sha256": _sha256_file(cubin),
    }


def _validate_artifact(
    cubin: Path, stamp_path: Path, *, identity: dict, artifact_id: str
) -> tuple[bool, str]:
    if cubin.is_symlink() or stamp_path.is_symlink():
        return False, "cubin or stamp is a symlink"
    if not cubin.is_file() or not stamp_path.is_file():
        return False, "cubin or stamp is missing"
    try:
        raw = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"stamp is unreadable: {exc}"
    if not isinstance(raw, dict) or set(raw) != {
        "schema", "artifact_id", "identity", "artifact_sha256"
    }:
        return False, "stamp schema is malformed"
    if raw.get("schema") != DEVICE_CACHE_SCHEMA:
        return False, "stamp schema differs"
    if raw.get("artifact_id") != artifact_id or raw.get("identity") != identity:
        return False, "artifact build identity differs"
    if raw.get("artifact_sha256") != _sha256_file(cubin):
        return False, "cubin hash differs from stamp"
    return True, ""


def _write_json_atomic(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _compile_cubin(
    source: Path, cubin: Path, *, bundle: Path, arch: str, identity: dict,
    artifact_id: str, stamp_path: Path,
) -> None:
    ok, _ = _validate_artifact(
        cubin, stamp_path, identity=identity, artifact_id=artifact_id
    )
    if ok:
        return
    nvcc = identity["nvcc"]["path"]
    tmp = cubin.with_name(f".kernel.{os.getpid()}.{uuid.uuid4().hex}.cubin")
    command = [
        nvcc,
        *_NVCC_FLAGS,
        f"-arch={arch}",
        f"-I{bundle}",
        str(source),
        "-o",
        str(tmp),
    ]
    try:
        subprocess.run(command, check=True)
        if tmp.is_symlink() or not tmp.is_file():
            raise DeviceComponentError(
                f"device compiler did not produce a regular cubin for {source}"
            )
        os.replace(tmp, cubin)
        _write_json_atomic(stamp_path, _stamp(identity, artifact_id, cubin))
    except subprocess.CalledProcessError as exc:
        raise DeviceComponentError(
            f"offline cubin compilation failed for {source} with exit {exc.returncode}"
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)


def _device_ops(manifest: "Manifest") -> tuple["OpEntry", ...]:
    from optima.manifest import VALIDATOR_DEVICE_EXECUTION

    return tuple(
        op for op in manifest.ops if op.execution_class == VALIDATOR_DEVICE_EXECUTION
    )


def device_manifest_rejection(manifest: "Manifest") -> str | None:
    """Explain why a manifest is not a valid validator-device development product.

    This validates only the narrow host ABI/build surface.  It deliberately says
    nothing about settlement authority: even a perfectly valid declaration still
    runs arbitrary device instructions over raw pointers and is therefore not
    memory-safe enough for a component crown.
    """

    from optima.manifest import VALIDATOR_DEVICE_EXECUTION

    if not manifest.ops:
        return "component competition requires at least one op"
    host = [
        f"{op.slot}:{op.variant}"
        for op in manifest.ops
        if op.execution_class != VALIDATOR_DEVICE_EXECUTION
    ]
    if host:
        return (
            "a validator_device development product requires validator_device "
            "execution for every variant; miner-controlled scheduler Python is "
            "untrusted_host and must use the isolated system lane "
            f"(rows: {tuple(host)!r})"
        )
    for op in manifest.ops:
        abi = DEVICE_ABIS.get(str(op.device_abi))
        if abi is None:
            return f"unknown validator-owned device_abi {op.device_abi!r}"
        if abi.slot != op.slot:
            return (
                f"device_abi {abi.name!r} belongs to slot {abi.slot!r}, not "
                f"{op.slot!r}"
            )
        claimed = frozenset(op.dtypes)
        if not claimed or not claimed.issubset(abi.dtypes):
            return (
                f"device_abi {abi.name!r} admits dtypes {tuple(sorted(abi.dtypes))!r}; "
                f"manifest claims {tuple(op.dtypes)!r}"
            )
    return None


def component_crown_rejection(manifest: "Manifest") -> str | None:
    """Explain why an op product cannot claim a component/atomic crown.

    Host-Python products are externally graded system candidates.  Validator-device
    products avoid host-code execution, but raw CUDA pointers are not a device-memory
    sandbox.  Keep both cases fail-closed at the component settlement boundary.
    """

    # A normal Triton/CuTe/Python op bundle is not a malformed
    # ``validator_device`` product.  It is a valid *untrusted-host* product in
    # the wrong settlement lane.  Say that directly and give the exact bounded
    # product identity the miner must request.  In particular, do not silently
    # promote a legacy slot declaration: that would turn arbitrary scheduler
    # Python into a component crown and preserve false per-slot attribution.
    if untrusted_host_system_rejection(manifest) is None:
        return (
            "untrusted_host scheduler Python cannot claim a component or atomic "
            "crown; submit the same inspectable bundle through the isolated "
            "system lane as the whole-serving product with [competition] "
            f"target={UNTRUSTED_HOST_SYSTEM_TARGET!r} and mode='system'"
        )

    invalid = device_manifest_rejection(manifest)
    if invalid is not None:
        return invalid
    return (
        "validator_device is development-only: raw CUDA device pointers share the "
        "model CUDA context and provide no memory-safety boundary; use externally "
        "observed whole-serving qualification until device-memory isolation exists"
    )


def untrusted_host_system_rejection(manifest: "Manifest") -> str | None:
    """Return ``None`` when an op bundle is eligible for whole-serving grading.

    This does *not* make scheduler Python trusted.  It classifies the useful
    middle lane requested by Optima's product goal: retain the inspectable bundle,
    reviewed rebuild/dep-patch policy, immutable arena, and fresh isolated OCI
    B/C/B' external qualification, while settling the result as one serving
    product with no per-slot receipts or component champions.

    The system competition resolver/evaluator owns target registration and must
    still require :data:`UNTRUSTED_HOST_SYSTEM_TARGET`; this hook only answers
    whether the submitted product has the right execution shape for that lane.
    """

    from optima.manifest import UNTRUSTED_HOST_EXECUTION

    if getattr(manifest, "system", None) is not None:
        return "source-patch system products use the pinned-runtime overlay lane"
    if not manifest.ops:
        return "whole-serving host bundles require at least one op declaration"
    wrong = [
        f"{op.slot}:{op.variant}:{op.execution_class}"
        for op in manifest.ops
        if op.execution_class != UNTRUSTED_HOST_EXECUTION
    ]
    if wrong:
        return (
            "whole-serving host bundles must contain only untrusted_host variants; "
            f"found {tuple(wrong)!r}"
        )
    # Host mutation features remain bounded by the existing manifest, scanner,
    # rebuild patcher and dependency-patch policies.  Their presence is why the
    # product needs OCI/external grading, not a reason to discard the optimization.
    return None


def untrusted_host_product_fingerprints(bundle_path: str | Path) -> tuple[str, ...]:
    """Product-level near-copy signals for the whole-serving host lane.

    System settlement has no component members, so its copy identity must not be
    lost merely because the source arrived in [[ops]] rows.  Fold the existing
    normalized bundle identity plus every path-independent per-slot file signal
    into target-level tokens.  Exact bundle identity remains separately committed
    by the chain content hash.
    """

    from optima.copy_fingerprint import (
        bundle_fingerprint,
        bundle_slot_file_fingerprints,
    )
    from optima.manifest import load_manifest

    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    reason = untrusted_host_system_rejection(manifest)
    if reason is not None:
        raise DeviceComponentError(reason)
    out: set[str] = set()
    whole = bundle_fingerprint(bundle)
    if whole:
        out.add(whole)
    for slot, file_fingerprints in bundle_slot_file_fingerprints(bundle).items():
        for fingerprint in file_fingerprints:
            out.add(hashlib.sha256(f"{slot}\0{fingerprint}".encode()).hexdigest())
    return tuple(sorted(out))


class _CudaDriver:
    """Small trusted CUDA-driver binding; no bundle-controlled host library."""

    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise DeviceComponentError(f"cannot load NVIDIA CUDA driver: {exc}") from exc
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p
        ]
        self.lib.cuModuleLoad.restype = ctypes.c_int
        self.lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
        ]
        self.lib.cuModuleGetFunction.restype = ctypes.c_int
        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]
        self.lib.cuLaunchKernel.restype = ctypes.c_int
        self._check(self.lib.cuInit(0), "cuInit")

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if int(code) != 0:
            raise DeviceComponentError(f"{operation} failed with CUDA driver code {code}")

    def load(self, cubin: Path, symbol: str) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        module = ctypes.c_void_p()
        self._check(
            self.lib.cuModuleLoad(ctypes.byref(module), os.fsencode(cubin)),
            "cuModuleLoad",
        )
        function = ctypes.c_void_p()
        self._check(
            self.lib.cuModuleGetFunction(
                ctypes.byref(function), module, symbol.encode("ascii")
            ),
            f"cuModuleGetFunction({symbol})",
        )
        return module, function

    def launch(
        self,
        function: ctypes.c_void_p,
        *,
        grid_x: int,
        block_x: int,
        stream: int,
        params: ctypes.Array,
    ) -> None:
        self._check(
            self.lib.cuLaunchKernel(
                function,
                int(grid_x), 1, 1,
                int(block_x), 1, 1,
                0, ctypes.c_void_p(int(stream)),
                params, None,
            ),
            "cuLaunchKernel",
        )


class _SiluEntry:
    """Trusted adapter for ``activation.silu_and_mul.cuda.v1``.

    Device symbol contract::

        extern "C" __global__ void K(
            const __nv_bfloat16* x, __nv_bfloat16* out,
            unsigned long long n_out, unsigned int d);

    The validator owns the 1-D launch: 256 threads and ceil(n_out / 256)
    blocks.  ``x`` and ``out`` are contiguous; index ``i`` maps to row=i/d,
    column=i%d, with gate/up halves at row*(2*d)+column/(column+d).
    """

    _BLOCK = 256

    def __init__(self, cubin: Path, symbol: str) -> None:
        self.cubin = cubin
        self.symbol = symbol
        self._lock = threading.Lock()
        self._loaded: tuple[_CudaDriver, ctypes.c_void_p, ctypes.c_void_p] | None = None

    def _function(self) -> tuple[_CudaDriver, ctypes.c_void_p]:
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    driver = _CudaDriver()
                    module, function = driver.load(self.cubin, self.symbol)
                    self._loaded = (driver, module, function)
        assert self._loaded is not None
        return self._loaded[0], self._loaded[2]

    def __call__(self, x, out) -> None:
        import torch

        if not x.is_cuda or not out.is_cuda:
            raise DeviceComponentError("validator_device silu requires CUDA tensors")
        if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise DeviceComponentError("silu CUDA v1 ABI is bfloat16-only")
        if not x.is_contiguous() or not out.is_contiguous():
            raise DeviceComponentError("silu CUDA v1 ABI requires contiguous tensors")
        if x.ndim < 1 or x.shape[-1] != 2 * out.shape[-1]:
            raise DeviceComponentError("silu CUDA v1 tensor shapes violate the slot ABI")
        if tuple(x.shape[:-1]) != tuple(out.shape[:-1]):
            raise DeviceComponentError("silu CUDA v1 leading dimensions differ")
        n_out = int(out.numel())
        if n_out == 0:
            return
        d = int(out.shape[-1])
        x_ptr = ctypes.c_uint64(int(x.data_ptr()))
        out_ptr = ctypes.c_uint64(int(out.data_ptr()))
        n_arg = ctypes.c_uint64(n_out)
        d_arg = ctypes.c_uint32(d)
        params = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(x_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(out_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(d_arg), ctypes.c_void_p),
        )
        driver, function = self._function()
        stream = int(torch.cuda.current_stream(x.device).cuda_stream)
        driver.launch(
            function,
            grid_x=(n_out + self._BLOCK - 1) // self._BLOCK,
            block_x=self._BLOCK,
            stream=stream,
            params=params,
        )


_DEVICE_ABIS = {
    SILU_CUDA_ABI: DeviceABI(
        name=SILU_CUDA_ABI,
        slot="activation.silu_and_mul",
        dtypes=frozenset({"bfloat16"}),
        entry_factory=lambda cubin, symbol: _SiluEntry(cubin, symbol),
    ),
}
DEVICE_ABIS: Mapping[str, DeviceABI] = MappingProxyType(_DEVICE_ABIS)


def prepare_device_artifacts(
    bundle_path: str | Path, *, phase: Literal["all", "build", "load"] = "all"
) -> bool:
    """Build or validate all validator-device cubins for one bundle.

    ``build`` is called only in the disposable trusted compiler worker.  ``load``
    is intentionally read-only: it neither creates cache directories nor repairs
    stamps.  Actual ``cuModuleLoad`` is lazy in the trusted entry adapter so the
    scheduler's CUDA context is current before driver loading.
    """

    if phase not in {"all", "build", "load"}:
        raise DeviceComponentError(f"unsupported device artifact phase: {phase!r}")
    from optima.bundle_hash import content_hash
    from optima.manifest import (
        all_declared_cuda_sources,
        all_declared_dep_patches,
        load_manifest,
        resolve_source,
    )
    from optima.sandbox import scan_tree

    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    ops = _device_ops(manifest)
    if not ops:
        return False
    tree = scan_tree(
        bundle,
        declared_cuda_sources=all_declared_cuda_sources(bundle, manifest),
        declared_dep_patches=all_declared_dep_patches(bundle, manifest),
    )
    if not tree.ok:
        raise DeviceComponentError(
            "validator_device bundle failed the complete declared-source scan: "
            f"{tree.violations!r}"
        )
    reason = device_manifest_rejection(manifest)
    if reason is not None:
        raise DeviceComponentError(reason)
    arch = _current_arch(offline=phase == "build")
    bundle_hash = content_hash(bundle)

    for op in ops:
        artifact_id, identity = _artifact_identity(bundle, op, arch)
        cubin, stamp_path, artifact_dir = _artifact_paths(bundle_hash, artifact_id)
        if phase == "load":
            ok, why = _validate_artifact(
                cubin, stamp_path, identity=identity, artifact_id=artifact_id
            )
            if not ok:
                raise DeviceComponentError(
                    f"refusing missing/stale validator_device artifact for "
                    f"{op.slot}:{op.variant}: {why}"
                )
            continue

        import fcntl

        artifact_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = _cache_root() / "v1" / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{artifact_id}.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _compile_cubin(
                resolve_source(bundle, op),
                cubin,
                bundle=bundle,
                arch=arch,
                identity=identity,
                artifact_id=artifact_id,
                stamp_path=stamp_path,
            )
        if phase == "all":
            ok, why = _validate_artifact(
                cubin, stamp_path, identity=identity, artifact_id=artifact_id
            )
            if not ok:
                raise DeviceComponentError(
                    f"new validator_device artifact failed validation: {why}"
                )
    return True


def load_device_entry(bundle_path: str | Path, op: "OpEntry") -> Callable[..., None]:
    """Return a trusted slot callable without importing any bundle host code."""

    from optima.bundle_hash import content_hash
    from optima.manifest import load_manifest

    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    matching = [
        candidate
        for candidate in _device_ops(manifest)
        if candidate.slot == op.slot and candidate.variant == op.variant
    ]
    if len(matching) != 1 or matching[0] != op:
        raise DeviceComponentError(
            f"device op identity changed before load: {op.slot}:{op.variant}"
        )
    reason = device_manifest_rejection(manifest)
    if reason is not None:
        raise DeviceComponentError(reason)
    arch = _current_arch()
    artifact_id, identity = _artifact_identity(bundle, op, arch)
    cubin, stamp_path, _ = _artifact_paths(content_hash(bundle), artifact_id)
    ok, why = _validate_artifact(
        cubin, stamp_path, identity=identity, artifact_id=artifact_id
    )
    if not ok:
        raise DeviceComponentError(
            f"refusing missing/stale validator_device artifact for "
            f"{op.slot}:{op.variant}: {why}"
        )
    abi = DEVICE_ABIS[str(op.device_abi)]
    return abi.entry_factory(cubin, op.entry)


def verify_device_entry_from_bundle(
    bundle_path: str | Path,
    slot_name: str,
    variant: str,
    *,
    dtype_name: str = "bfloat16",
    device: str = "cuda",
    seed: int = 0,
    jitter_seed: int | None = None,
    model_key: str | None = None,
):
    """Build, load, and verify one device variant in a throwaway CLI child.

    This function is module-level so :func:`optima.eval._launch.call_in_subprocess`
    can spawn it without importing or executing candidate code in the trusted CLI.
    The child still treats the CUDA context as hostile and is development-only; a
    passing result is correctness/graph evidence, never settlement authority.
    """

    import torch

    from optima.manifest import load_manifest
    from optima.registry import eligibility_from_metadata
    from optima.slots import slot_for_model
    from optima.verify import verify_entry

    if not isinstance(device, str) or not device.startswith("cuda"):
        raise DeviceComponentError(
            "validator_device verification requires a CUDA device"
        )
    try:
        torch.cuda.set_device(torch.device(device))
    except Exception as exc:  # noqa: BLE001 - normalize the development CLI boundary
        raise DeviceComponentError(
            f"cannot select validator_device verification device {device!r}: {exc}"
        ) from exc
    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    op = manifest.op_for(slot_name, variant)
    if op is None:
        raise DeviceComponentError(
            f"validator_device op disappeared before verify: {slot_name}:{variant}"
        )
    invalid = device_manifest_rejection(manifest)
    if invalid is not None:
        raise DeviceComponentError(invalid)
    prepare_device_artifacts(bundle, phase="all")
    entry = load_device_entry(bundle, op)
    metadata: dict = {}
    if op.metadata:
        path = bundle / op.metadata
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DeviceComponentError(
                f"cannot read validator_device metadata {op.metadata!r}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DeviceComponentError("validator_device metadata must be a JSON object")
        metadata = value
    eligibility = eligibility_from_metadata(
        metadata if "capabilities" in metadata else None,
        op.dtypes,
        op.architectures,
    )
    slot = slot_for_model(slot_name, model_key)
    return verify_entry(
        slot,
        entry,
        dtype=getattr(torch, dtype_name),
        device=device,
        seed=seed,
        jitter_seed=jitter_seed,
        graph_safe=None if slot.kind == "op" else bool(metadata.get("graph_safe", False)),
        eligibility=eligibility,
    )

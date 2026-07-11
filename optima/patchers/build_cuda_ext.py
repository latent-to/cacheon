"""Reviewed rebuild patcher: compile and load DECLARED CUDA source bundles.

Native artifacts are content addressed.  In particular, neither ``bundle_id`` nor a
miner-chosen source stem is trusted as a cache key.  The identity of each artifact
commits to the complete submitted bundle tree, this reviewed patcher's exact bytes,
the compilation unit path, GPU architecture, compiler/toolchain identity and every
compile/link flag.  Build and scheduler-load independently derive that identity and
load refuses a missing, malformed, stale or artifact-hash-mismatched stamp.

The work is split deliberately: ``build`` compiles in a disposable controller child
without dlopening miner native code; ``load`` only validates and dlopens inside an
untrusted scheduler rank.  The extension's real ``PyInit_*`` name is a validator-
generated content-addressed identifier.  Its source stem is installed only as a
compatibility import alias for the bundle's Python shim, so two candidates can never
name the same native module/artifact internally.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path


_CACHE_SCHEMA = "optima-cuda-ext-cache-v2"
_NVCC_FLAGS = ("-O3", "--use_fast_math", "--std=c++17", "-Xcompiler", "-fPIC", "-shared")
_LINK_LIBS = ("torch", "torch_python", "c10", "c10_cuda", "torch_cuda")
_C_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OFFLINE_ARCH = re.compile(r"sm_[0-9]{2,3}a$")


def _log(msg: str) -> None:
    print(f"[optima.build_cuda_ext] {msg}", flush=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_hash(data: object) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_root() -> Path:
    root = os.environ.get("OPTIMA_CUDA_EXT_CACHE", "")
    return Path(root) if root else Path.home() / ".cache" / "optima" / "cuda_ext"


def _cache_dir(bundle_hash: str, artifact_id: str) -> Path:
    # Shard by the trusted tree hash, never by bundle_id/source stem.
    return _cache_root() / "v2" / bundle_hash[:2] / bundle_hash / artifact_id


def _lock_path(artifact_id: str) -> Path:
    return _cache_root() / "v2" / ".locks" / f"{artifact_id}.lock"


def _command_output(argv: list[str]) -> str:
    try:
        return subprocess.run(
            argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to identify CUDA toolchain command {argv!r}: {exc}") from exc


def _binary_identity(executable: str) -> dict[str, str]:
    found = shutil.which(executable)
    if found is None:
        raise RuntimeError(f"required CUDA toolchain executable is missing: {executable}")
    path = Path(found).resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "version": _command_output([str(path), "--version"]),
    }


def _runtime_context(arch: str) -> dict:
    """Everything outside the bundle that can change the emitted extension."""
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    py_inc = Path(sysconfig.get_paths()["include"]).resolve()
    torch_git = getattr(torch.version, "git_version", None)
    return {
        "arch": arch,
        "nvcc": _binary_identity("nvcc"),
        "ptxas": _binary_identity("ptxas"),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_git_version": str(torch_git or ""),
        "torch_build_config": str(torch.__config__.show()),
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
        "python_version": sys.version,
        "python_soabi": str(sysconfig.get_config_var("SOABI") or ""),
        "python_include": str(py_inc),
        "torch_include": str(torch_root / "include"),
        "torch_api_include": str(torch_root / "include" / "torch" / "csrc" / "api" / "include"),
        "torch_lib": str(torch_root / "lib"),
        "nvcc_flags": list(_NVCC_FLAGS),
        "link_libraries": list(_LINK_LIBS),
    }


def _bundle_hash(bundle: Path) -> str:
    from optima.bundle_hash import content_hash

    observed = content_hash(bundle)
    expected = os.environ.get("OPTIMA_BUNDLE_CONTENT_HASH", "").strip()
    if expected and expected != observed:
        raise RuntimeError(
            f"bundle mutated during rebuild: expected {expected}, observed {observed}"
        )
    return observed


def _patcher_hash() -> str:
    return _sha256_file(Path(__file__).resolve())


def _artifact_identity(
    *, bundle_hash: str, source_rel: str, alias: str, context: dict
) -> tuple[str, dict, str]:
    payload = {
        "schema": _CACHE_SCHEMA,
        "bundle_hash": bundle_hash,
        "patcher_sha256": _patcher_hash(),
        "compilation_unit": source_rel,
        "import_alias": alias,
        "toolchain": context,
    }
    artifact_id = _canonical_hash(payload)
    # A Python extension init symbol must be a C identifier.  Keep the compatibility
    # alias miner-friendly, but make the actual native module validator-owned/unique.
    module_name = f"_optima_cuda_{alias}_{artifact_id[:24]}"
    return artifact_id, payload, module_name


def _source_units(bundle: Path, sources: list[str]) -> list[tuple[Path, str, str]]:
    units: list[tuple[Path, str, str]] = []
    aliases: dict[str, str] = {}
    for src_str in sources:
        src = Path(src_str).resolve()
        rel = src.relative_to(bundle).as_posix()
        alias = src.stem
        if not _C_IDENTIFIER.fullmatch(alias):
            raise RuntimeError(
                f"CUDA source stem must be an ASCII C/Python identifier: {rel!r}"
            )
        prior = aliases.get(alias)
        if prior is not None and prior != rel:
            raise RuntimeError(
                f"CUDA compilation units {prior!r} and {rel!r} share import alias "
                f"{alias!r}; source stems must be unique within a bundle"
            )
        aliases[alias] = rel
        units.append((src, rel, alias))
    return units


def _compile(src: Path, so: Path, arch: str, module_name: str, context: dict) -> None:
    cmd = [
        context["nvcc"]["path"], *_NVCC_FLAGS, f"-arch={arch}",
        f"-DTORCH_EXTENSION_NAME={module_name}",
        f"-D_GLIBCXX_USE_CXX11_ABI={context['cxx11_abi']}",
        f"-I{context['torch_include']}",
        f"-I{context['torch_api_include']}",
        f"-I{context['python_include']}",
        str(src), f"-L{context['torch_lib']}",
        *[f"-l{lib}" for lib in _LINK_LIBS], "-o", str(so),
    ]
    _log(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _stamp_payload(identity: dict, artifact_id: str, module_name: str, artifact_sha: str) -> dict:
    return {
        "schema": _CACHE_SCHEMA,
        "artifact_id": artifact_id,
        "module_name": module_name,
        "identity": identity,
        "artifact_sha256": artifact_sha,
    }


def _validate_cached(
    so: Path, stamp_path: Path, *, identity: dict, artifact_id: str, module_name: str
) -> tuple[bool, str]:
    if so.is_symlink() or stamp_path.is_symlink():
        return False, "artifact or stamp is a symlink"
    if not so.is_file() or not stamp_path.is_file():
        return False, "artifact or stamp is missing"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"stamp is unreadable: {exc}"
    if not isinstance(stamp, dict) or set(stamp) != {
        "schema", "artifact_id", "module_name", "identity", "artifact_sha256"
    }:
        return False, "stamp schema is malformed"
    if stamp.get("schema") != _CACHE_SCHEMA:
        return False, "stamp cache schema differs"
    if stamp.get("artifact_id") != artifact_id or stamp.get("module_name") != module_name:
        return False, "stamp artifact identity differs"
    if stamp.get("identity") != identity:
        return False, "stamp build identity differs"
    actual_sha = _sha256_file(so)
    if stamp.get("artifact_sha256") != actual_sha:
        return False, "compiled artifact hash differs from stamp"
    return True, ""


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _build_one(
    src: Path, so: Path, stamp: Path, *, arch: str, module_name: str,
    identity: dict, artifact_id: str, context: dict,
) -> None:
    ok, why = _validate_cached(
        so, stamp, identity=identity, artifact_id=artifact_id, module_name=module_name
    )
    if ok:
        _log(f"cache hit for {src.stem} ({artifact_id[:16]})")
        return
    if so.exists() or stamp.exists():
        _log(f"invalid cache entry for {src.stem} ({why}); rebuilding")

    tmp_so = so.with_name(f".{module_name}.{os.getpid()}.{uuid.uuid4().hex}.so")
    try:
        _compile(src, tmp_so, arch, module_name, context)
        if tmp_so.is_symlink() or not tmp_so.is_file():
            raise RuntimeError(f"compiler did not produce a regular artifact: {tmp_so}")
        artifact_sha = _sha256_file(tmp_so)
        os.replace(tmp_so, so)
        _atomic_write_json(
            stamp, _stamp_payload(identity, artifact_id, module_name, artifact_sha)
        )
    finally:
        tmp_so.unlink(missing_ok=True)


def _load(alias: str, module_name: str, so: Path) -> None:
    import torch  # noqa: F401 - extension links libtorch; import it first

    existing_alias = sys.modules.get(alias)
    existing_native = sys.modules.get(module_name)
    if existing_alias is not None and existing_alias is not existing_native:
        raise RuntimeError(
            f"refusing CUDA import alias collision: {alias!r} is already loaded"
        )
    if existing_native is None:
        spec = importlib.util.spec_from_file_location(module_name, so)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not create extension loader for {so}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    else:
        loaded_path = Path(str(getattr(existing_native, "__file__", ""))).resolve()
        if loaded_path != so.resolve():
            raise RuntimeError(
                f"native module name collision for {module_name!r}: "
                f"loaded {loaded_path}, expected {so.resolve()}"
            )
        mod = existing_native
    sys.modules[alias] = mod
    _log(f"loaded {module_name} as compatibility alias {alias} ({so})")


def main() -> None:
    bundle_raw = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle_raw:
        _log("no OPTIMA_BUNDLE_PATH set; nothing to build")
        return
    bundle = Path(bundle_raw).resolve()

    from optima.manifest import all_declared_cuda_sources, load_manifest

    manifest = load_manifest(bundle)
    phase = os.environ.get("OPTIMA_REBUILD_PHASE", "all").strip().lower()
    if phase not in {"all", "build", "load"}:
        raise RuntimeError(f"unsupported OPTIMA_REBUILD_PHASE: {phase!r}")
    declared = sorted(str(s) for s in all_declared_cuda_sources(bundle, manifest))
    sources = [s for s in declared if Path(s).suffix == ".cu"]
    if not sources:
        _log("bundle declares no .cu compilation units; nothing to build")
        return
    units = _source_units(bundle, sources)
    bundle_hash = _bundle_hash(bundle)

    has_nvcc = shutil.which("nvcc") is not None
    has_ptxas = shutil.which("ptxas") is not None
    if not has_nvcc or not has_ptxas:
        if phase in {"all", "build"}:
            raise RuntimeError("offline CUDA build requires pinned nvcc and ptxas")
        raise RuntimeError(
            "scheduler load cannot validate a CUDA artifact without the pinned "
            "nvcc/ptxas toolchain used to derive its cache identity"
        )
    if phase == "build":
        arch = os.environ.get("OPTIMA_TARGET_GPU_ARCH", "").strip()
        if _OFFLINE_ARCH.fullmatch(arch) is None:
            raise RuntimeError(
                "offline CUDA build requires validator-owned OPTIMA_TARGET_GPU_ARCH"
            )
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA artifact load requires a live scored CUDA device")
        cc = torch.cuda.get_device_capability(torch.cuda.current_device())
        arch = f"sm_{cc[0]}{cc[1]}a"
    context = _runtime_context(arch)

    import fcntl

    for src, rel, alias in units:
        artifact_id, identity, module_name = _artifact_identity(
            bundle_hash=bundle_hash, source_rel=rel, alias=alias, context=context
        )
        cache = _cache_dir(bundle_hash, artifact_id)
        so = cache / f"{module_name}.so"
        stamp = cache / "artifact.json"
        if phase == "load":
            # Candidate scheduler caches are mounted read-only.  Load is a pure
            # validation+dlopen operation: no mkdir, lock creation, stamp repair, or
            # fallback compilation is permitted in the untrusted runtime namespace.
            ok, why = _validate_cached(
                so, stamp, identity=identity, artifact_id=artifact_id,
                module_name=module_name,
            )
            if not ok:
                raise RuntimeError(
                    f"refusing stale/missing CUDA artifact for {rel}: {why}; "
                    "the trusted build worker must complete for this exact bundle "
                    "and toolchain before scheduler load"
                )
            _load(alias, module_name, so)
            continue

        cache.mkdir(parents=True, exist_ok=True)
        lock_path = _lock_path(artifact_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            if phase in {"all", "build"}:
                _build_one(
                    src, so, stamp, arch=arch, module_name=module_name,
                    identity=identity, artifact_id=artifact_id, context=context,
                )
            if phase == "all":
                ok, why = _validate_cached(
                    so, stamp, identity=identity, artifact_id=artifact_id,
                    module_name=module_name,
                )
                if not ok:
                    raise RuntimeError(
                        f"refusing stale/missing CUDA artifact for {rel}: {why}; "
                        "the trusted build worker must complete for this exact bundle "
                        "and toolchain before scheduler load"
                    )
                _load(alias, module_name, so)
            else:
                _log(f"built {module_name} ({so}); load deferred to scheduler")


main()

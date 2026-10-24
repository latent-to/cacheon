"""Name materialized sources while preserving bundle-relative JIT inputs."""

import hashlib
from pathlib import Path, PurePosixPath
import re


def generated_name(prefix: str, relative: str, *, suffix: str) -> str:
    """Keep native module and entry names unique across contribution paths."""
    stem = PurePosixPath(relative).stem
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_") or "module"
    path_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{slug}_{path_hash}{suffix}"


def cuda_source_paths(prefix: str, relative: str, *, prebuild: bool) -> tuple[str, str]:
    """Return the build path and the path a library JIT resolves beside Python.

    A native prebuild can coexist with installed-library JIT. Relocating all
    sources into the build tree broke Muse prefill capture on 2026-10-24.
    """
    path = PurePosixPath(relative)
    runtime = (PurePosixPath(prefix) / path).as_posix()
    if not prebuild:
        return runtime, runtime
    name = generated_name(prefix, relative, suffix=path.suffix) if path.suffix == ".cu" else path.name
    return (PurePosixPath("cuda") / prefix / path.parent / name).as_posix(), runtime


def is_native_source(bundle: Path, path: Path) -> bool:
    """Compile generated units once; the matching runtime copies remain JIT inputs."""
    return path.suffix == ".cu" and (
        not (bundle / "metadata/cacheon_engine_tree.json").is_file()
        or path.relative_to(bundle).parts[0] == "cuda"
    )

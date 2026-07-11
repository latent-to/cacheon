"""Deterministic model-volume sealing and registered-arena provisioning.

Registered arenas pin the hashes of the *actual* model tree, including any
validator-owned runtime overlay sources mounted into the serving image.  This
module turns that policy into a reproducible command instead of relying on an
irreplaceable hand-prepared pod volume.

Usage::

    python -m optima.model_provision provision \
      --arena minimax-m3-b300-tp4-longprefill-v1 \
      --model-root /root/models/MiniMax-M3-NVFP4

The expensive byte pass happens once at provisioning.  The resulting
``.optima-content-sha256.json`` is deterministic and is checked cheaply on each
launch; production startup can request a full byte recheck separately.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Iterable

from optima.runtime_overlay import (
    RuntimeFileOverlay,
    normalize_runtime_overlays,
    verify_runtime_overlays,
)


MODEL_SEAL_NAME = ".optima-content-sha256.json"
_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_PACKAGED_ASSET_ROOTS = {
    "MiniMax-M3-NVFP4": Path(__file__).with_name("arena_assets") / "minimax_m3",
}


class ModelProvisionError(RuntimeError):
    """A model tree or packaged overlay cannot be provisioned safely."""


def _real_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(path)
    try:
        info = raw.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ModelProvisionError(f"{label} must be a concrete directory: {raw}")
        return raw.resolve(strict=True)
    except ModelProvisionError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ModelProvisionError(f"cannot resolve {label} {raw}: {exc}") from None


def _validate_relative(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or path.as_posix() != relative
        or ".." in path.parts
        or any("\x00" in part or "\n" in part or "\r" in part for part in path.parts)
    ):
        raise ModelProvisionError(f"unsafe model-relative path: {relative!r}")


def _model_files(root: Path) -> tuple[tuple[str, Path, os.stat_result], ...]:
    """Enumerate the exact non-cache file set without following symlinks."""

    out: list[tuple[str, Path, os.stat_result]] = []
    folded: dict[str, str] = {}

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ModelProvisionError(f"cannot enumerate model directory {directory}: {exc}") from None
        for entry in entries:
            # Hugging Face download receipts are verified independently and are
            # intentionally excluded by arenas.verify_model_content_seal too.
            if entry.name == ".cache":
                continue
            parts = (*prefix, entry.name)
            relative = PurePosixPath(*parts).as_posix()
            _validate_relative(relative)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ModelProvisionError(f"cannot inspect model path {relative}: {exc}") from None
            if stat.S_ISLNK(info.st_mode):
                raise ModelProvisionError(f"model tree contains a symlink: {relative}")
            if stat.S_ISDIR(info.st_mode):
                visit(Path(entry.path), parts)
                continue
            if relative == MODEL_SEAL_NAME:
                if not stat.S_ISREG(info.st_mode):
                    raise ModelProvisionError("existing model seal is not a regular file")
                continue
            if entry.name == MODEL_SEAL_NAME:
                raise ModelProvisionError(
                    f"reserved model seal name appears below the root: {relative}"
                )
            if not stat.S_ISREG(info.st_mode):
                raise ModelProvisionError(f"model tree contains a non-regular file: {relative}")
            if info.st_nlink != 1:
                raise ModelProvisionError(
                    f"model file has external/aliased hard links: {relative}"
                )
            folded_name = relative.casefold()
            previous = folded.get(folded_name)
            if previous is not None and previous != relative:
                raise ModelProvisionError(
                    f"case-colliding model paths are not portable: {previous!r}, {relative!r}"
                )
            folded[folded_name] = relative
            out.append((relative, Path(entry.path), info))

    visit(root, ())
    if not out:
        raise ModelProvisionError("model tree has no sealable files")
    return tuple(sorted(out, key=lambda row: row[0]))


def _hash_regular(
    row: tuple[str, Path, os.stat_result],
) -> dict[str, object]:
    relative, path, observed = row
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ModelProvisionError("this host lacks O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ModelProvisionError(f"cannot open model file safely {relative}: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or any(
                getattr(before, field) != getattr(observed, field)
                for field in _STABLE_STAT_FIELDS
            )
        ):
            raise ModelProvisionError(f"model file changed before hashing: {relative}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in _STABLE_STAT_FIELDS
        ):
            raise ModelProvisionError(f"model file changed while hashing: {relative}")
        return {
            "path": relative,
            "sha256": digest.hexdigest(),
            "size": before.st_size,
        }
    finally:
        os.close(fd)


def _content_digest(files: Iterable[dict[str, object]]) -> str:
    rows = sorted(
        (str(item["path"]), str(item["sha256"]))
        for item in files
    )
    payload = "".join(f"{relative}\0{sha}\n" for relative, sha in rows).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd: int | None = None
    try:
        fd = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ModelProvisionError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, mode)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise ModelProvisionError(f"cannot publish {path}: {exc}") from None
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def generate_model_content_seal(
    model_root: str | os.PathLike[str],
    *,
    expected_digest: str | None = None,
    workers: int = 8,
) -> str:
    """Hash a model tree and atomically publish its deterministic content seal.

    A wrong ``expected_digest`` fails before replacing any existing seal.  File
    rows are sorted independently of thread completion order, so ``workers``
    changes only throughput, never the resulting bytes.
    """

    root = _real_directory(model_root, label="model root")
    if expected_digest is not None and _SHA256_ID.fullmatch(expected_digest) is None:
        raise ModelProvisionError("expected model digest must be sha256:<64 lowercase hex>")
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ModelProvisionError("workers must be an integer in [1, 64]")
    discovered = _model_files(root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        files = sorted(executor.map(_hash_regular, discovered), key=lambda row: str(row["path"]))
    digest = _content_digest(files)
    if expected_digest is not None and digest != expected_digest:
        raise ModelProvisionError(
            f"model content digest mismatch: {digest!r} != {expected_digest!r}"
        )
    seal = {"content_digest": digest, "files": files, "version": 1}
    payload = (
        json.dumps(seal, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    _atomic_write(root / MODEL_SEAL_NAME, payload)
    return digest


def _read_overlay_asset(path: Path, overlay: RuntimeFileOverlay) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ModelProvisionError("this host lacks O_NOFOLLOW")
    try:
        fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise ModelProvisionError(f"cannot open packaged overlay {path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != overlay.size
        ):
            raise ModelProvisionError(f"packaged overlay has wrong file shape: {path}")
        chunks: list[bytes] = []
        remaining = overlay.size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ModelProvisionError(f"packaged overlay was truncated: {path}")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ModelProvisionError(f"packaged overlay grew while reading: {path}")
        after = os.fstat(fd)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in _STABLE_STAT_FIELDS
        ):
            raise ModelProvisionError(f"packaged overlay changed while reading: {path}")
        actual = digest.hexdigest()
        if actual != overlay.sha256:
            raise ModelProvisionError(
                f"packaged overlay hash mismatch for {overlay.source}: "
                f"{actual} != {overlay.sha256}"
            )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _safe_asset_path(asset_root: Path, relative: str) -> Path:
    _validate_relative(relative)
    lexical = asset_root / PurePosixPath(relative)
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(asset_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelProvisionError(
            f"packaged runtime overlay escapes/is missing: {relative}: {exc}"
        ) from None
    current = lexical
    while current != asset_root:
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ModelProvisionError(
                    f"packaged runtime overlay traverses a symlink: {relative}"
                )
        except OSError as exc:
            raise ModelProvisionError(f"cannot inspect packaged overlay {relative}: {exc}") from None
        current = current.parent
    return resolved


def _ensure_destination_parent(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            current.mkdir(mode=0o755)
        except FileExistsError:
            pass
        try:
            info = current.lstat()
        except OSError as exc:
            raise ModelProvisionError(f"cannot inspect overlay destination {current}: {exc}") from None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ModelProvisionError(f"overlay destination parent is not a real directory: {current}")
    try:
        current.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelProvisionError(f"overlay destination escapes model root: {relative}: {exc}") from None
    return current


def install_runtime_overlay_assets(
    model_root: str | os.PathLike[str],
    overlays: Iterable[RuntimeFileOverlay],
    *,
    assets_root: str | os.PathLike[str],
) -> tuple[Path, ...]:
    """Copy exact, hash-pinned packaged overlay bytes into a model volume."""

    root = _real_directory(model_root, label="model root")
    assets = _real_directory(assets_root, label="arena assets root")
    installed: list[Path] = []
    for overlay in normalize_runtime_overlays(overlays):
        source = _safe_asset_path(assets, overlay.source)
        payload = _read_overlay_asset(source, overlay)
        relative = PurePosixPath(overlay.source)
        parent = _ensure_destination_parent(root, relative)
        destination = parent / relative.name
        _atomic_write(destination, payload)
        installed.append(destination)
    verify_runtime_overlays(root, overlays)
    return tuple(installed)


def provision_registered_arena(
    arena_name: str,
    *,
    model_root: str | os.PathLike[str] | None = None,
    assets_root: str | os.PathLike[str] | None = None,
    workers: int = 8,
) -> str:
    """Install stock overlays and seal one registered arena's model volume."""

    from optima.arenas import get_arena

    arena = get_arena(arena_name)
    root = Path(model_root or arena.model_path)
    if arena.runtime_overlays:
        if assets_root is None:
            assets_root = _PACKAGED_ASSET_ROOTS.get(arena.model_id)
        if assets_root is None:
            raise ModelProvisionError(
                f"no packaged runtime-overlay asset root for model {arena.model_id!r}"
            )
        install_runtime_overlay_assets(
            root, arena.runtime_overlays, assets_root=assets_root
        )
    digest = generate_model_content_seal(
        root, expected_digest=arena.model_content_digest, workers=workers
    )
    # The byte pass immediately above established the expensive fact.  Reuse the
    # seal for the arena's path/revision/receipt check instead of hashing 239 GB twice.
    arena.verify_model_receipt(root, verify_bytes=False)
    verify_runtime_overlays(root, arena.runtime_overlays)
    return digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optima-model-provision",
        description="Generate deterministic Optima model seals and install arena overlays.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    seal = sub.add_parser("seal", help="hash a model tree and atomically publish its seal")
    seal.add_argument("model_root")
    seal.add_argument("--expected-digest", default=None)
    seal.add_argument("--workers", type=int, default=8)

    provision = sub.add_parser(
        "provision",
        help="install packaged runtime overlays and seal a registered arena model",
    )
    provision.add_argument("--arena", required=True)
    provision.add_argument("--model-root", default=None)
    provision.add_argument("--assets-root", default=None)
    provision.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "seal":
            digest = generate_model_content_seal(
                args.model_root,
                expected_digest=args.expected_digest,
                workers=args.workers,
            )
        else:
            digest = provision_registered_arena(
                args.arena,
                model_root=args.model_root,
                assets_root=args.assets_root,
                workers=args.workers,
            )
    except Exception as exc:  # CLI boundary: typed internals, concise operator failure
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

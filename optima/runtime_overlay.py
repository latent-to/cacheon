"""Validator-owned runtime file overlays covered by the arena identity.

Some serving arenas need a small, reviewed patch on top of an immutable base
image.  Those bytes are *stock arena state*, not candidate state: B, C, B' and
prebuild must all see the same files.  Sources live below the sealed model root
and are bind-mounted read-only at exact package paths in the pinned image.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_ROOT = PurePosixPath("/sgl-workspace/sglang/python/sglang")


class RuntimeOverlayError(RuntimeError):
    """A stock runtime overlay is unsafe, missing, or has changed bytes."""


@dataclass(frozen=True, slots=True)
class RuntimeFileOverlay:
    """One model-relative source and exact in-image replacement target."""

    source: str
    target: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not isinstance(self.target, str):
            raise RuntimeOverlayError("runtime overlay source/target must be strings")
        source = PurePosixPath(self.source)
        target = PurePosixPath(self.target)
        if (
            not self.source
            or source.is_absolute()
            or source.as_posix() != self.source
            or ".." in source.parts
            or any(
                not re.fullmatch(r"[A-Za-z0-9_.+-]+", part)
                for part in source.parts
            )
        ):
            raise RuntimeOverlayError(
                "runtime overlay source must be a normalized model-relative path"
            )
        if (
            not self.target.startswith("/")
            or target.as_posix() != self.target
            or ".." in target.parts
            or target == _TARGET_ROOT
            or _TARGET_ROOT not in target.parents
            or target.suffix != ".py"
            or any(char in self.target for char in (",", "\x00", "\n", "\r"))
        ):
            raise RuntimeOverlayError(
                "runtime overlay target must be a normalized Python file below "
                f"{_TARGET_ROOT}"
            )
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise RuntimeOverlayError("runtime overlay sha256 must be 64 lowercase hex")
        if type(self.size) is not int or not 0 < self.size <= 16 * 1024 * 1024:
            raise RuntimeOverlayError("runtime overlay size must be in (0, 16MiB]")


def normalize_runtime_overlays(
    overlays: Iterable[RuntimeFileOverlay],
) -> tuple[RuntimeFileOverlay, ...]:
    """Return a deterministic overlay tuple and reject ambiguous destinations."""

    try:
        values = tuple(overlays)
    except TypeError:
        raise RuntimeOverlayError("runtime_overlays must be an iterable") from None
    if any(type(value) is not RuntimeFileOverlay for value in values):
        raise RuntimeOverlayError(
            "runtime_overlays entries must be exact RuntimeFileOverlay values"
        )
    if len(values) > 32:
        raise RuntimeOverlayError("runtime_overlays exceeds the 32-file hard bound")
    sources = [value.source for value in values]
    targets = [value.target for value in values]
    if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise RuntimeOverlayError("runtime overlay sources and targets must be unique")
    return tuple(sorted(values, key=lambda value: (value.target, value.source)))


def runtime_overlay_fingerprint(overlays: Iterable[RuntimeFileOverlay]) -> str:
    """Canonical identity of the complete source/target/hash/size mapping."""

    payload = [
        {
            "sha256": value.sha256,
            "size": value.size,
            "source": value.source,
            "target": value.target,
        }
        for value in normalize_runtime_overlays(overlays)
    ]
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _read_and_hash_regular(path: Path, *, expected_size: int) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeOverlayError("this host lacks O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeOverlayError(f"cannot open runtime overlay safely: {path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise RuntimeOverlayError(
                f"runtime overlay is not the expected regular file: {path}"
            )
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeOverlayError(f"runtime overlay was truncated: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeOverlayError(f"runtime overlay grew while hashing: {path}")
        after = os.fstat(fd)
        stable = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
            "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise RuntimeOverlayError(f"runtime overlay changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(fd)


def verify_runtime_overlays(
    model_root: str | os.PathLike[str],
    overlays: Iterable[RuntimeFileOverlay],
) -> tuple[tuple[Path, RuntimeFileOverlay], ...]:
    """Resolve and hash every overlay source below the sealed model root."""

    raw_root = Path(model_root)
    try:
        raw_info = raw_root.lstat()
        if stat.S_ISLNK(raw_info.st_mode) or not stat.S_ISDIR(raw_info.st_mode):
            raise RuntimeOverlayError("model root must be a real directory")
        root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeOverlayError(f"cannot resolve model root: {exc}") from None
    result: list[tuple[Path, RuntimeFileOverlay]] = []
    inodes: set[tuple[int, int]] = set()
    for overlay in normalize_runtime_overlays(overlays):
        lexical = root / PurePosixPath(overlay.source)
        try:
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeOverlayError(
                f"runtime overlay source escapes/is missing below the model root: "
                f"{overlay.source}: {exc}"
            ) from None
        # A sealed model receipt must describe concrete files, not indirections.
        current = lexical
        while current != root:
            if current.is_symlink():
                raise RuntimeOverlayError(
                    f"runtime overlay source traverses a symlink: {overlay.source}"
                )
            current = current.parent
        actual = _read_and_hash_regular(resolved, expected_size=overlay.size)
        if actual != overlay.sha256:
            raise RuntimeOverlayError(
                f"runtime overlay hash mismatch for {overlay.source}: "
                f"{actual} != {overlay.sha256}"
            )
        info = resolved.stat()
        inode = (info.st_dev, info.st_ino)
        if inode in inodes:
            raise RuntimeOverlayError("runtime overlay sources alias one inode")
        inodes.add(inode)
        result.append((resolved, overlay))
    return tuple(result)


def verify_runtime_overlay_targets(
    overlays: Iterable[RuntimeFileOverlay],
) -> tuple[Path, ...]:
    """Hash the actual in-image bind targets seen by a scored worker."""

    targets: list[Path] = []
    inodes: set[tuple[int, int]] = set()
    for overlay in normalize_runtime_overlays(overlays):
        target = Path(overlay.target)
        try:
            info = target.lstat()
        except OSError as exc:
            raise RuntimeOverlayError(
                f"runtime overlay target is unavailable: {overlay.target}: {exc}"
            ) from None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeOverlayError(
                f"runtime overlay target is not a concrete file: {overlay.target}"
            )
        actual = _read_and_hash_regular(target, expected_size=overlay.size)
        if actual != overlay.sha256:
            raise RuntimeOverlayError(
                f"runtime overlay target hash mismatch for {overlay.target}: "
                f"{actual} != {overlay.sha256}"
            )
        try:
            if not bool(os.statvfs(target).f_flag & os.ST_RDONLY):
                raise RuntimeOverlayError(
                    f"runtime overlay target is not read-only: {overlay.target}"
                )
        except OSError as exc:
            raise RuntimeOverlayError(
                f"cannot inspect runtime overlay target mount: {overlay.target}: {exc}"
            ) from None
        inode = (info.st_dev, info.st_ino)
        if inode in inodes:
            raise RuntimeOverlayError("runtime overlay targets alias one inode")
        inodes.add(inode)
        targets.append(target)
    return tuple(targets)

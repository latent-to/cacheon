#!/usr/bin/env python3
"""Reject release archives that contain legacy Optima package paths."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


EXPECTED_PACKAGES = frozenset({"cacheon", "cacheon_kernels"})
LEGACY_PACKAGES = frozenset({"optima", "optima_kernels"})


def _archive_members(path: Path) -> tuple[list[str], bool]:
    """Return member names and whether the archive has an sdist root directory."""

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist(), False
    if path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(path, mode="r:*") as archive:
            return archive.getnames(), True
    raise ValueError(f"unsupported distribution archive: {path}")


def _package_roots(path: Path) -> set[str]:
    names, has_sdist_root = _archive_members(path)
    roots: set[str] = set()
    sdist_roots: set[str] = set()

    for name in names:
        member = PurePosixPath(name)
        parts = member.parts
        if member.is_absolute() or ".." in parts:
            raise ValueError(f"unsafe archive member in {path}: {name!r}")
        if not parts:
            continue
        if has_sdist_root:
            sdist_roots.add(parts[0])
            parts = parts[1:]
        if not parts:
            continue
        root = parts[0]
        if root.endswith(".py"):
            root = root.removesuffix(".py")
        roots.add(root)

    if has_sdist_root and len(sdist_roots) != 1:
        raise ValueError(
            f"sdist must have one top-level directory, found {sorted(sdist_roots)!r}"
        )
    return roots


def check_archive(path: Path) -> None:
    roots = _package_roots(path)
    legacy = sorted(roots & LEGACY_PACKAGES)
    missing = sorted(EXPECTED_PACKAGES - roots)
    if legacy:
        raise ValueError(f"{path} contains legacy package roots: {legacy!r}")
    if missing:
        raise ValueError(f"{path} is missing package roots: {missing!r}")
    print(f"{path}: Cacheon package inventory is clean")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.archives:
        check_archive(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

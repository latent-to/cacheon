"""Stable artifact I/O used by crash-safe remote request planning."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import stat
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cacheon.chain.publication import WorkerBundlePublication, reopen_worker_bundle
from cacheon.chain.remote_worker_spool import (
    MAX_ARTIFACT_BYTES,
    NATIVE_ARTIFACT_MANIFEST,
    fail,
    require_closed,
    require_digest,
    require_int,
    spool_canonical_json,
    tar_info,
)
from cacheon.stack_identity import sha256_hex


QUALIFICATION_ARTIFACT_ROLES = (
    "qualification_payload",
    "candidate_publication",
)
_PLANNED_ARTIFACT_FIELDS = frozenset({"role", "sha256", "size"})


@dataclass(frozen=True)
class PlannedQualificationArtifact:
    """One path-free artifact identity retained by a request plan."""

    role: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if self.role not in QUALIFICATION_ARTIFACT_ROLES:
            fail("qualification plan artifact role is not closed")
        require_digest(self.sha256, "qualification plan artifact digest")
        require_int(
            self.size,
            "qualification plan artifact size",
            maximum=MAX_ARTIFACT_BYTES,
        )

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "PlannedQualificationArtifact":
        row = require_closed(value, _PLANNED_ARTIFACT_FIELDS, "planned artifact")
        return cls(row["role"], row["sha256"], row["size"])


def qualification_source_map(
    artifact_inputs: Sequence[tuple[str, Path]],
) -> tuple[tuple[str, Path], ...]:
    """Close qualification inputs to one payload plus its ordered carriers."""

    rows = tuple((role, source) for role, source in artifact_inputs)
    roles = tuple(role for role, _source in rows)
    if (
        len(rows) < 2
        or roles[0] != "qualification_payload"
        or any(role != "candidate_publication" for role in roles[1:])
    ):
        fail("qualification artifact inputs are not one payload plus its carriers")
    return rows


def stable_artifact_identity(path: Path) -> tuple[int, str]:
    """Hash one regular, single-link source while proving stable identity."""

    if not isinstance(path, Path) or not path.is_absolute():
        fail("planned artifact source must be an absolute Path")
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > MAX_ARTIFACT_BYTES
        ):
            fail("planned artifact source shape is unsafe")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1 << 20):
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        fail(f"planned artifact source cannot be reopened: {exc}")
    identity = (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size)
    if identity != (
        opened.st_dev,
        opened.st_ino,
        opened.st_mode,
        opened.st_nlink,
        opened.st_size,
    ) or identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
    ) or before.st_mtime_ns != after.st_mtime_ns:
        fail("planned artifact source changed while hashing")
    return before.st_size, digest.hexdigest()


def copy_stable_artifact(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Copy, fsync, and revalidate an artifact against a planned identity."""

    try:
        before = source.lstat()
        descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        digest = hashlib.sha256()
        size = 0
        try:
            with destination.open("xb") as output:
                while chunk := os.read(descriptor, 1 << 20):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        fail(f"planned artifact cannot be copied: {exc}")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        != (after.st_dev, after.st_ino, after.st_mode, after.st_size)
        or before.st_mtime_ns != after.st_mtime_ns
        or size != expected_size
        or digest.hexdigest() != expected_sha256
    ):
        fail("planned artifact bytes or identity changed")
    os.chmod(destination, 0o400)


def publication_archive(publication: object, destination: Path) -> None:
    """Copy one exact WorkerBundlePublication into a path-free worker archive."""

    if type(publication) is not WorkerBundlePublication:
        fail("remote transport requires an exact WorkerBundlePublication")
    root = publication.root
    if root.is_symlink() or not root.is_dir():
        fail("worker publication root is unavailable or symlinked")
    try:
        reopened = reopen_worker_bundle(
            root,
            publication.content_hash,
            expected_publication_digest=publication.publication_digest,
            expected_receipt_digest=publication.digest,
        )
    except Exception as exc:
        fail(f"worker publication cannot be reopened before transport: {exc}")
    if reopened.to_dict() != publication.to_dict():
        fail("reopened worker publication differs before transport")

    def stable_bytes(path: Path, *, size: int | None, sha256: str | None) -> bytes:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size < 0
            or (size is not None and before.st_size != size)
        ):
            fail("worker publication file shape differs during transport")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_mode != before.st_mode
                or opened.st_nlink != before.st_nlink
                or opened.st_size != before.st_size
            ):
                fail("worker publication file changed while opening")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(4 << 20, remaining))
                if not chunk:
                    fail("worker publication file was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                fail("worker publication file grew during transport")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        data = b"".join(chunks)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mode != before.st_mode
            or after.st_nlink != before.st_nlink
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or (sha256 is not None and sha256_hex(data) != sha256)
        ):
            fail("worker publication bytes differ from retained inventory")
        return data

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    try:
        manifest = (
            spool_canonical_json(
                {
                    "publication": publication.to_dict(),
                    "schema": "cacheon-remote-worker-publication-v1",
                }
            )
            + b"\n"
        )
        with tarfile.open(temporary, "w") as archive:
            archive.addfile(
                tar_info("publication.json", len(manifest)), io.BytesIO(manifest)
            )
            native_manifest = stable_bytes(
                root / NATIVE_ARTIFACT_MANIFEST, size=None, sha256=None
            )
            if len(native_manifest) > 16 << 20:
                fail("worker publication native manifest exceeds its hard bound")
            archive.addfile(
                tar_info(f"bundle/{NATIVE_ARTIFACT_MANIFEST}", len(native_manifest)),
                io.BytesIO(native_manifest),
            )
            for row in publication.files:
                relative = Path(row.path)
                if relative.is_absolute() or ".." in relative.parts:
                    fail("worker publication inventory contains an unsafe path")
                data = stable_bytes(
                    root.joinpath(*relative.parts), size=row.size, sha256=row.sha256
                )
                archive.addfile(
                    tar_info(f"bundle/{relative.as_posix()}", len(data)),
                    io.BytesIO(data),
                )
        if Path(temporary).stat().st_size > MAX_ARTIFACT_BYTES:
            fail("worker publication archive exceeds transport limit")
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


__all__ = [
    "PlannedQualificationArtifact",
    "QUALIFICATION_ARTIFACT_ROLES",
    "copy_stable_artifact",
    "publication_archive",
    "qualification_source_map",
    "stable_artifact_identity",
]

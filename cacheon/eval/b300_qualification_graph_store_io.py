"""Strict filesystem primitives for the durable B300 graph-evidence store."""

from __future__ import annotations

import contextlib
import json
import math
import os
import stat
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from cacheon.stack_identity import canonical_json_bytes


class B300QualificationGraphEvidenceStoreError(RuntimeError):
    """The durable graph-evidence store or an indexed artifact is invalid."""


class B300QualificationGraphEvidenceHold(B300QualificationGraphEvidenceStoreError):
    """An armed expensive attempt has no authenticated terminal result."""


def absolute_path(value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise B300QualificationGraphEvidenceStoreError(
            "graph evidence root must be path-like"
        )
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise B300QualificationGraphEvidenceStoreError(
            "graph evidence root must be canonical and absolute"
        )
    return path


def absolute_deadline(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise B300QualificationGraphEvidenceStoreError(
            "graph evidence deadline must be a finite absolute monotonic float"
        )
    if value <= time.monotonic():
        raise B300QualificationGraphEvidenceHold(
            "graph evidence absolute monotonic deadline expired"
        )
    return value


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise B300QualificationGraphEvidenceHold(
            "graph evidence absolute monotonic deadline expired"
        )


def owner_uid() -> int:
    return os.geteuid()


def directory_identity(path: Path) -> tuple[int, int]:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        after = resolved.stat()
    except OSError as exc:
        raise B300QualificationGraphEvidenceStoreError(
            f"graph evidence directory is unavailable: {exc}"
        ) from None
    if (
        resolved != path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_uid != owner_uid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise B300QualificationGraphEvidenceStoreError(
            "graph evidence directory must be canonical, nonsymlink, "
            "owner-controlled mode 0700"
        )
    return before.st_dev, before.st_ino


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError as exc:
        raise B300QualificationGraphEvidenceStoreError(
            f"cannot fsync graph evidence directory: {exc}"
        ) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                # A close failure must not replace a primary open/fsync failure.
                pass


def mkdir_private(path: Path) -> tuple[int, int]:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        pass
    except OSError as exc:
        raise B300QualificationGraphEvidenceStoreError(
            f"cannot create graph evidence directory: {exc}"
        ) from None
    identity = directory_identity(path)
    fsync_directory(path)
    fsync_directory(path.parent)
    return identity


def canonical_object(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_number(value: str) -> object:
        raise B300QualificationGraphEvidenceStoreError(
            f"{label} contains unsupported number {value!r}"
        )

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise B300QualificationGraphEvidenceStoreError(
                    f"{label} repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=reject_number,
            parse_constant=reject_number,
            object_pairs_hook=pairs,
        )
    except B300QualificationGraphEvidenceStoreError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise B300QualificationGraphEvidenceStoreError(
            f"{label} is malformed: {exc}"
        ) from None
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise B300QualificationGraphEvidenceStoreError(
            f"{label} is not exact canonical JSON"
        )
    return value


def read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise B300QualificationGraphEvidenceStoreError(
            f"{label} is unavailable: {exc}"
        ) from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != owner_uid()
        or stat.S_IMODE(before.st_mode) != 0o400
        or not 1 <= before.st_size <= max_bytes
    ):
        raise B300QualificationGraphEvidenceStoreError(
            f"{label} has an unsafe shape, owner, mode, or size"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(opened, name) for name in stable):
            raise B300QualificationGraphEvidenceStoreError(
                f"{label} changed while opening"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(max_bytes + 1)
        after = os.fstat(descriptor)
        if any(getattr(opened, name) != getattr(after, name) for name in stable):
            raise B300QualificationGraphEvidenceStoreError(
                f"{label} changed while reading"
            )
    except B300QualificationGraphEvidenceStoreError:
        raise
    except OSError as exc:
        raise B300QualificationGraphEvidenceStoreError(
            f"cannot read {label}: {exc}"
        ) from None
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
    if len(payload) != before.st_size:
        raise B300QualificationGraphEvidenceStoreError(
            f"{label} was truncated or grew while reading"
        )
    return payload


def publish_sealed(
    target: Path,
    payload: bytes,
    *,
    kind: str,
    label: str,
    max_bytes: int,
    deadline: float,
    staging_root: Path,
    boundary: Callable[[str, str], None],
) -> None:
    """Publish one sealed file; unexpected target state always fails closed."""

    if not 1 <= len(payload) <= max_bytes:
        raise B300QualificationGraphEvidenceStoreError(
            f"{label} exceeds its size limit"
        )
    check_deadline(deadline)
    temporary = staging_root / f".{kind}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        boundary(kind, "temp_created")
        if os.write(descriptor, payload) != len(payload):
            raise B300QualificationGraphEvidenceStoreError(f"{label} write stalled")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        with contextlib.suppress(OSError):
            os.close(descriptor)
        descriptor = None
        if (
            not stat.S_ISREG(sealed.st_mode)
            or sealed.st_nlink != 1
            or sealed.st_uid != owner_uid()
            or stat.S_IMODE(sealed.st_mode) != 0o400
            or sealed.st_size != len(payload)
        ):
            raise B300QualificationGraphEvidenceStoreError(
                f"staged {label} did not seal safely"
            )
        check_deadline(deadline)
        if os.path.lexists(target):
            existing = read_regular(target, label=label, max_bytes=max_bytes)
            if existing != payload:
                raise B300QualificationGraphEvidenceStoreError(
                    f"existing {label} is divergent"
                )
            check_deadline(deadline)
            return
        os.replace(temporary, target)
        boundary(kind, "renamed")
        fsync_directory(target.parent)
        fsync_directory(staging_root)
        boundary(kind, "parents_fsynced")
        if read_regular(target, label=label, max_bytes=max_bytes) != payload:
            raise B300QualificationGraphEvidenceStoreError(
                f"published {label} did not reopen exactly"
            )
        check_deadline(deadline)
    except B300QualificationGraphEvidenceStoreError:
        raise
    except OSError as exc:
        raise B300QualificationGraphEvidenceStoreError(
            f"cannot publish {label}: {exc}"
        ) from None
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        # The staging file has no authority. Cleanup must preserve any primary
        # failure and cannot invalidate already reopened target bytes.
        with contextlib.suppress(
            FileNotFoundError,
            OSError,
            B300QualificationGraphEvidenceStoreError,
        ):
            temporary.unlink()
            fsync_directory(staging_root)


__all__ = [
    "B300QualificationGraphEvidenceHold",
    "B300QualificationGraphEvidenceStoreError",
    "absolute_deadline",
    "absolute_path",
    "canonical_object",
    "check_deadline",
    "directory_identity",
    "fsync_directory",
    "mkdir_private",
    "owner_uid",
    "publish_sealed",
    "read_regular",
]

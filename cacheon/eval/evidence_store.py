"""Content-addressed storage for semantically opaque referee evidence bytes."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import math
import os
import re
import stat
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from cacheon.stack_identity import canonical_json_bytes, require_sha256_hex


DEFAULT_MAX_EVIDENCE_BYTES = 64 << 20
HARD_MAX_EVIDENCE_BYTES = 1 << 30
_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_SCHEMA = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_MEDIA = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,63}$"
)
_LIBC = ctypes.CDLL(None, use_errno=True)
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_LOCK_POLL_SECONDS = 0.01


class EvidenceStoreError(ValueError):
    """Evidence identity, bytes, or filesystem state is invalid."""


@dataclass(frozen=True)
class EvidenceArtifactRef:
    domain: str
    sha256: str
    size: int
    media_type: str
    schema: str

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or _DOMAIN.fullmatch(self.domain) is None:
            raise EvidenceStoreError("evidence domain is invalid")
        try:
            digest = require_sha256_hex(self.sha256, field="evidence sha256")
        except ValueError as exc:
            raise EvidenceStoreError(str(exc)) from None
        object.__setattr__(self, "sha256", digest)
        if (isinstance(self.size, bool) or not isinstance(self.size, int)
                or not 0 <= self.size <= HARD_MAX_EVIDENCE_BYTES):
            raise EvidenceStoreError("evidence size is invalid")
        if not isinstance(self.media_type, str) or _MEDIA.fullmatch(self.media_type) is None:
            raise EvidenceStoreError("evidence media_type is invalid")
        if not isinstance(self.schema, str) or _SCHEMA.fullmatch(self.schema) is None:
            raise EvidenceStoreError("evidence schema is invalid")

    def to_dict(self) -> dict[str, object]:
        return {"domain": self.domain, "media_type": self.media_type,
                "schema": self.schema, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceArtifactRef":
        fields = {"domain", "media_type", "schema", "sha256", "size"}
        if type(value) is not dict or set(value) != fields:
            raise EvidenceStoreError("evidence reference schema is not closed")
        return cls(**value)  # type: ignore[arg-type]


def _limit(value: object) -> int:
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 1 <= value <= HARD_MAX_EVIDENCE_BYTES):
        raise EvidenceStoreError("max evidence size is invalid")
    return value


def _absolute(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise EvidenceStoreError("evidence root must be a path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise EvidenceStoreError("evidence root must be canonical and absolute")
    return path


def _directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceStoreError(f"evidence directory is unavailable: {exc}") from None
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise EvidenceStoreError("evidence directory has an unsafe owner or mode")
    try:
        if path.resolve(strict=True) != path:
            raise EvidenceStoreError("evidence directory traverses a symlink")
    except OSError as exc:
        raise EvidenceStoreError(f"cannot resolve evidence directory: {exc}") from None


def _close_descriptor(descriptor: int) -> None:
    os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        failure: BaseException | None = None
        try:
            os.fsync(fd)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                _close_descriptor(fd)
            except OSError:
                if failure is None:
                    raise
    except OSError as exc:
        raise EvidenceStoreError(f"cannot fsync evidence directory: {exc}") from None


def _publication_boundary(_phase: str) -> None:
    """Private process-crash injection seam for durability tests."""


def _deadline(value: object) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise EvidenceStoreError(
            "evidence publication deadline must be a finite absolute monotonic float"
        )
    if value <= time.monotonic():
        raise EvidenceStoreError("evidence publication deadline expired")
    return value


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise EvidenceStoreError("evidence publication deadline expired")


def _mkdir_private(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        _fsync_dir(path.parent)
    except FileExistsError:
        pass
    except OSError as exc:
        raise EvidenceStoreError(f"cannot create evidence directory: {exc}") from None
    _directory(path)
    return path


def _atomic_rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename *source* without replacing an existing target."""

    source_bytes, target_bytes = os.fsencode(source), os.fsencode(target)
    if sys.platform == "darwin":
        rename = getattr(_LIBC, "renamex_np", None)
        if rename is None:
            raise EvidenceStoreError("atomic no-replace rename is unavailable")
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, _RENAME_EXCL)
    elif sys.platform.startswith("linux"):
        rename = getattr(_LIBC, "renameat2", None)
        if rename is None:
            raise EvidenceStoreError("atomic no-replace rename is unavailable")
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            target_bytes,
            _RENAME_NOREPLACE,
        )
    else:
        raise EvidenceStoreError("atomic no-replace rename is unavailable")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), target)
    raise OSError(error, os.strerror(error), target)


def prepare_evidence_root(root: str | Path) -> Path:
    path = _absolute(root)
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        pass
    except OSError as exc:
        raise EvidenceStoreError(f"cannot create evidence root: {exc}") from None
    _directory(path)
    _fsync_dir(path)
    return path


def _target(root: Path, reference: EvidenceArtifactRef, *, create: bool) -> Path:
    domain, shard = root / reference.domain, root / reference.domain / reference.sha256[:2]
    if create:
        for directory in (domain, shard):
            try:
                directory.mkdir(mode=0o700, exist_ok=False)
                _fsync_dir(directory.parent)
            except FileExistsError:
                pass
            except OSError as exc:
                raise EvidenceStoreError(f"cannot create evidence directory: {exc}") from None
            _directory(directory)
    else:
        _directory(domain)
        _directory(shard)
    target = shard / reference.sha256
    try:
        target.relative_to(root)
    except ValueError:
        raise EvidenceStoreError("evidence path escapes its store root") from None
    return target


def _staging_paths(
    root: Path, reference: EvidenceArtifactRef
) -> tuple[Path, Path, Path]:
    staging = _mkdir_private(root / ".staging")
    domain = _mkdir_private(staging / reference.domain)
    return (
        domain / f"{reference.sha256}.lock",
        domain / f"{reference.sha256}.stage",
        domain,
    )


@contextlib.contextmanager
def _publication_lock(path: Path, *, deadline: float | None) -> Iterator[None]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise EvidenceStoreError(f"cannot open evidence publication lock: {exc}") from None
    failure: BaseException | None = None
    try:
        before = os.fstat(descriptor)
        try:
            named_before = path.lstat()
        except OSError as exc:
            raise EvidenceStoreError(
                f"evidence publication lock path is unavailable: {exc}"
            ) from None
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != 0
            or (before.st_dev, before.st_ino)
            != (named_before.st_dev, named_before.st_ino)
        ):
            raise EvidenceStoreError("evidence publication lock is unsafe")
        if deadline is None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as exc:
                raise EvidenceStoreError(
                    f"cannot lock evidence publication: {exc}"
                ) from None
        else:
            while True:
                _check_deadline(deadline)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                        raise EvidenceStoreError(
                            f"cannot lock evidence publication: {exc}"
                        ) from None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise EvidenceStoreError(
                            "evidence publication deadline expired"
                        ) from None
                    time.sleep(min(_LOCK_POLL_SECONDS, remaining))
            _check_deadline(deadline)
        after = os.fstat(descriptor)
        try:
            named_after = path.lstat()
        except OSError as exc:
            raise EvidenceStoreError(
                f"evidence publication lock path changed: {exc}"
            ) from None
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_size")
        if (
            any(getattr(before, field) != getattr(after, field) for field in stable)
            or (after.st_dev, after.st_ino)
            != (named_after.st_dev, named_after.st_ino)
        ):
            raise EvidenceStoreError("evidence publication lock changed while acquiring")
        yield
        _check_deadline(deadline)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            _close_descriptor(descriptor)
        except OSError as exc:
            if failure is None:
                raise EvidenceStoreError(
                    f"cannot close evidence publication lock: {exc}"
                ) from None


def _read_sealed(
    path: Path,
    reference: EvidenceArtifactRef,
    *,
    limit: int,
    label: str = "evidence artifact",
) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise EvidenceStoreError(f"{label} is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_size != reference.size
        or before.st_size > limit
    ):
        raise EvidenceStoreError(
            f"{label} has an unsafe shape, owner, mode, link count, or size"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceStoreError(f"cannot open {label}: {exc}") from None
    failure: BaseException | None = None
    try:
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
        if any(getattr(before, field) != getattr(opened, field) for field in stable):
            raise EvidenceStoreError(f"{label} changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        if len(payload) != reference.size:
            raise EvidenceStoreError(f"{label} was truncated or grew while reading")
        after = os.fstat(descriptor)
        if any(getattr(opened, field) != getattr(after, field) for field in stable):
            raise EvidenceStoreError(f"{label} changed while reading")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            _close_descriptor(descriptor)
        except OSError as exc:
            if failure is None:
                raise EvidenceStoreError(f"cannot close {label}: {exc}") from None
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise EvidenceStoreError(f"{label} digest mismatch")
    return payload


def _accept_existing(
    target: Path,
    reference: EvidenceArtifactRef,
    payload: bytes,
    *,
    limit: int,
) -> None:
    if _read_sealed(target, reference, limit=limit) != payload:
        raise EvidenceStoreError("existing evidence is not an exact duplicate")
    _fsync_dir(target.parent)


def _write_stage(path: Path, payload: bytes, reference: EvidenceArtifactRef) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    failure: BaseException | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EvidenceStoreError("evidence artifact write stalled")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
    except OSError as exc:
        failure = exc
        raise EvidenceStoreError(f"cannot stage evidence artifact: {exc}") from None
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if descriptor >= 0:
            try:
                _close_descriptor(descriptor)
            except OSError as exc:
                if failure is None:
                    raise EvidenceStoreError(
                        f"cannot close staged evidence artifact: {exc}"
                    ) from None
    if (
        not stat.S_ISREG(sealed.st_mode)
        or sealed.st_nlink != 1
        or sealed.st_uid != os.geteuid()
        or stat.S_IMODE(sealed.st_mode) != 0o400
        or sealed.st_size != reference.size
    ):
        raise EvidenceStoreError("staged evidence file did not seal safely")


def publish_evidence(
    root: str | Path,
    payload: bytes,
    *,
    domain: str,
    media_type: str,
    schema: str,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
    deadline: float | None = None,
) -> EvidenceArtifactRef:
    """Publish bytes, bounded by an optional absolute monotonic deadline."""

    limit = _limit(max_bytes)
    exact_deadline = _deadline(deadline)
    if not isinstance(payload, bytes):
        raise EvidenceStoreError("evidence payload must be exact bytes")
    if len(payload) > limit:
        raise EvidenceStoreError("evidence payload exceeds its size limit")
    reference = EvidenceArtifactRef(domain, hashlib.sha256(payload).hexdigest(),
                                    len(payload), media_type, schema)
    store = prepare_evidence_root(root)
    target = _target(store, reference, create=True)
    lock, stage, staging_directory = _staging_paths(store, reference)
    with _publication_lock(lock, deadline=exact_deadline):
        _check_deadline(exact_deadline)
        if os.path.lexists(target):
            _accept_existing(target, reference, payload, limit=limit)
            _check_deadline(exact_deadline)
            return reference
        if os.path.lexists(stage):
            if _read_sealed(
                stage,
                reference,
                limit=limit,
                label="staged evidence artifact",
            ) != payload:
                raise EvidenceStoreError("staged evidence bytes differ")
        else:
            work = staging_directory / (
                f".{reference.sha256}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            try:
                _write_stage(work, payload, reference)
                if _read_sealed(
                    work,
                    reference,
                    limit=limit,
                    label="staged evidence artifact",
                ) != payload:
                    raise EvidenceStoreError("staged evidence bytes differ")
                _atomic_rename_noreplace(work, stage)
                _fsync_dir(staging_directory)
            except FileExistsError:
                if _read_sealed(
                    stage,
                    reference,
                    limit=limit,
                    label="staged evidence artifact",
                ) != payload:
                    raise EvidenceStoreError("concurrent staged evidence differs")
            except OSError as exc:
                raise EvidenceStoreError(f"cannot publish evidence artifact: {exc}") from None
            finally:
                primary_active = sys.exc_info()[0] is not None
                try:
                    work.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    if not primary_active:
                        raise EvidenceStoreError(
                            f"cannot remove evidence staging work file: {exc}"
                        ) from None
            _publication_boundary("staged_temp_created")
        if os.path.lexists(target):
            _accept_existing(target, reference, payload, limit=limit)
            _check_deadline(exact_deadline)
            return reference
        try:
            _atomic_rename_noreplace(stage, target)
        except FileExistsError:
            _accept_existing(target, reference, payload, limit=limit)
            _check_deadline(exact_deadline)
            return reference
        except OSError as exc:
            raise EvidenceStoreError(f"cannot publish evidence artifact: {exc}") from None
        _publication_boundary("rename_complete_before_directory_fsync")
        _fsync_dir(target.parent)
        _fsync_dir(staging_directory)
        _publication_boundary("directory_fsync_complete")
        if _read_sealed(target, reference, limit=limit) != payload:
            raise EvidenceStoreError("published evidence bytes differ")
        _check_deadline(exact_deadline)
        return reference


def reopen_evidence(
    root: str | Path,
    reference: EvidenceArtifactRef,
    *,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> bytes:
    """Reopen and authenticate exact bytes without interpreting their schema."""

    limit = _limit(max_bytes)
    if type(reference) is not EvidenceArtifactRef:
        raise EvidenceStoreError("evidence reference must be exact and typed")
    if reference.size > limit:
        raise EvidenceStoreError("evidence reference exceeds its size limit")
    store = _absolute(root)
    _directory(store)
    target = _target(store, reference, create=False)
    return _read_sealed(target, reference, limit=limit)


def publish_canonical_json_evidence(
    root: str | Path,
    value: object,
    *,
    domain: str,
    schema: str,
    media_type: str = "application/json",
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> EvidenceArtifactRef:
    """Canonicalize trusted input, then store it as semantically opaque bytes."""

    return publish_evidence(root, canonical_json_bytes(value), domain=domain,
                            media_type=media_type, schema=schema, max_bytes=max_bytes)


__all__ = ["DEFAULT_MAX_EVIDENCE_BYTES", "EvidenceArtifactRef", "EvidenceStoreError",
           "prepare_evidence_root", "publish_canonical_json_evidence",
           "publish_evidence", "reopen_evidence"]

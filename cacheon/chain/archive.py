"""Private, digest-verified off-pod archives for validator durable state.

The object store is a recovery mirror, never the live SQLite/evidence authority.
Snapshots contain an online SQLite backup, the redacted chain audit journal,
database-referenced worker publications and qualification evidence, plus only
explicitly named sealed-input roots.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from cacheon.chain.intake import IntakeScope
from cacheon.chain.audit_log import (
    ChainAuditLogError,
    MAX_AUDIT_RECORD_BYTES,
    validate_chain_audit_record,
)
from cacheon.chain.publication import reopen_worker_bundle
from cacheon.eval.evidence_store import EvidenceArtifactRef, reopen_evidence
from cacheon.object_store import (
    ObjectStore,
    ObjectStoreError,
    ObjectStoreNotFoundError,
)
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)


# Version-1 archive identities are persistent wire/storage contracts.  The
# product/package rename must not strand snapshots already written below the
# original prefix or change the digests that authenticate their manifests.
ARCHIVE_SCHEMA = "optima.validator-archive.v1"
DEFAULT_VALIDATOR_ARCHIVE_PREFIX = "optima/validator-archive/v1"
_ARCHIVED_TREE_DIGEST_DOMAIN = "optima.validator.archived-tree"
_ARCHIVE_MANIFEST_DIGEST_DOMAIN = "optima.validator.archive-manifest"
_ARCHIVE_SCOPE_DIGEST_DOMAIN = "optima.chain.intake-scope"
_ARCHIVED_EVIDENCE_ROOT_DIGEST_DOMAIN = "optima.validator.archived-evidence-root"
_RESTORE_MAP_SCHEMA = "optima.validator-archive-restore-map.v1"
MAX_MANIFEST_BYTES = 16 << 20
MAX_ARCHIVE_FILE_BYTES = 1 << 30
MAX_SQLITE_BYTES = 1 << 30
MAX_TREE_FILES = 16_384
MAX_TREE_BYTES = 4 << 30
_LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_TABLE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class ValidatorArchiveError(RuntimeError):
    """A validator snapshot could not be created, authenticated, or restored."""


def _archive_scope_digest(scope: IntakeScope) -> str:
    """Return the immutable scope identity embedded by archive schema v1."""

    return canonical_digest(_ARCHIVE_SCOPE_DIGEST_DOMAIN, scope.to_dict())


@dataclass(frozen=True)
class ArchiveBlobRef:
    key: str
    sha256: str
    size: int
    media_type: str

    def __post_init__(self) -> None:
        require_sha256_hex(self.sha256, field="archive blob sha256")
        if (
            not isinstance(self.key, str)
            or self.key != f"blobs/sha256/{self.sha256}"
        ):
            raise ValidatorArchiveError(
                "archive blob key is not its content-derived address"
            )
        if type(self.size) is not int or not 0 <= self.size <= MAX_ARCHIVE_FILE_BYTES:
            raise ValidatorArchiveError("archive blob size is malformed")
        if (
            not isinstance(self.media_type, str)
            or not self.media_type
            or len(self.media_type) > 128
        ):
            raise ValidatorArchiveError("archive blob media type is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArchiveBlobRef":
        if type(value) is not dict or set(value) != {
            "key",
            "media_type",
            "sha256",
            "size",
        }:
            raise ValidatorArchiveError("archive blob reference is not closed")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ArchivedDirectory:
    path: str
    mode: int

    def __post_init__(self) -> None:
        _relative_path(self.path, allow_root=True)
        if type(self.mode) is not int or not 0 <= self.mode <= 0o777:
            raise ValidatorArchiveError("archived directory mode is malformed")

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "path": self.path}

    @classmethod
    def from_dict(cls, value: object) -> "ArchivedDirectory":
        if type(value) is not dict or set(value) != {"mode", "path"}:
            raise ValidatorArchiveError("archived directory is not closed")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ArchivedFile:
    path: str
    mode: int
    blob: ArchiveBlobRef

    def __post_init__(self) -> None:
        _relative_path(self.path)
        if type(self.mode) is not int or not 0 <= self.mode <= 0o777:
            raise ValidatorArchiveError("archived file mode is malformed")
        if type(self.blob) is not ArchiveBlobRef:
            raise ValidatorArchiveError("archived file blob is not typed")
        if self.blob.media_type != "application/octet-stream":
            raise ValidatorArchiveError("archived file blob media type differs")

    def to_dict(self) -> dict[str, object]:
        return {"blob": self.blob.to_dict(), "mode": self.mode, "path": self.path}

    @classmethod
    def from_dict(cls, value: object) -> "ArchivedFile":
        if type(value) is not dict or set(value) != {"blob", "mode", "path"}:
            raise ValidatorArchiveError("archived file is not closed")
        return cls(
            path=value["path"],  # type: ignore[arg-type]
            mode=value["mode"],  # type: ignore[arg-type]
            blob=ArchiveBlobRef.from_dict(value["blob"]),
        )


@dataclass(frozen=True)
class ArchivedTree:
    kind: str
    name: str
    source_root: str
    content_hash: str
    publication_digest: str
    directories: tuple[ArchivedDirectory, ...]
    files: tuple[ArchivedFile, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"admitted_bundle", "sealed_input"}:
            raise ValidatorArchiveError("archived tree kind is unsupported")
        if not isinstance(self.name, str) or _LABEL.fullmatch(self.name) is None:
            raise ValidatorArchiveError("archived tree name is malformed")
        source = Path(self.source_root)
        if not source.is_absolute() or source != Path(os.path.normpath(source)):
            raise ValidatorArchiveError("archived tree source root is not canonical")
        if self.kind == "admitted_bundle":
            require_sha256_hex(self.content_hash, field="bundle content hash")
            require_sha256_hex(
                self.publication_digest, field="bundle publication digest"
            )
        elif self.content_hash or self.publication_digest:
            raise ValidatorArchiveError("sealed input carries bundle identity")
        if (
            not self.directories
            or len(self.directories) > MAX_TREE_FILES + 1
            or len(self.files) > MAX_TREE_FILES
            or sum(item.blob.size for item in self.files) > MAX_TREE_BYTES
            or self.directories[0].path != "."
            or tuple(item.path for item in self.directories)
            != tuple(sorted({item.path for item in self.directories}))
            or tuple(item.path for item in self.files)
            != tuple(sorted({item.path for item in self.files}))
        ):
            raise ValidatorArchiveError("archived tree inventory is not canonical")
        directory_paths = {item.path for item in self.directories}
        for item in self.files:
            parent = PurePosixPath(item.path).parent
            parent_text = "." if str(parent) == "." else str(parent)
            if parent_text not in directory_paths:
                raise ValidatorArchiveError("archived file parent is missing")

    @property
    def digest(self) -> str:
        return canonical_digest(
            _ARCHIVED_TREE_DIGEST_DOMAIN,
            self.to_dict(include_digest=False),
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "content_hash": self.content_hash,
            "directories": [item.to_dict() for item in self.directories],
            "files": [item.to_dict() for item in self.files],
            "kind": self.kind,
            "name": self.name,
            "publication_digest": self.publication_digest,
            "source_root": self.source_root,
        }
        if include_digest:
            value["tree_digest"] = self.digest
        return value

    @classmethod
    def from_dict(cls, value: object) -> "ArchivedTree":
        fields = {
            "content_hash",
            "directories",
            "files",
            "kind",
            "name",
            "publication_digest",
            "source_root",
            "tree_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValidatorArchiveError("archived tree is not closed")
        directories = value["directories"]
        files = value["files"]
        if type(directories) is not list or type(files) is not list:
            raise ValidatorArchiveError("archived tree inventory is malformed")
        tree = cls(
            kind=value["kind"],  # type: ignore[arg-type]
            name=value["name"],  # type: ignore[arg-type]
            source_root=value["source_root"],  # type: ignore[arg-type]
            content_hash=value["content_hash"],  # type: ignore[arg-type]
            publication_digest=value["publication_digest"],  # type: ignore[arg-type]
            directories=tuple(ArchivedDirectory.from_dict(row) for row in directories),
            files=tuple(ArchivedFile.from_dict(row) for row in files),
        )
        if value["tree_digest"] != tree.digest:
            raise ValidatorArchiveError("archived tree digest differs")
        return tree


@dataclass(frozen=True)
class ArchivedEvidence:
    source_root: str
    reference: EvidenceArtifactRef
    blob: ArchiveBlobRef

    def __post_init__(self) -> None:
        source = Path(self.source_root)
        if not source.is_absolute() or source != Path(os.path.normpath(source)):
            raise ValidatorArchiveError("evidence source root is not canonical")
        if type(self.reference) is not EvidenceArtifactRef:
            raise ValidatorArchiveError("evidence reference is not typed")
        if type(self.blob) is not ArchiveBlobRef:
            raise ValidatorArchiveError("evidence blob is not typed")
        if (
            self.reference.sha256 != self.blob.sha256
            or self.reference.size != self.blob.size
            or self.reference.media_type != self.blob.media_type
        ):
            raise ValidatorArchiveError("evidence reference differs from archived blob")

    def to_dict(self) -> dict[str, object]:
        return {
            "blob": self.blob.to_dict(),
            "reference": self.reference.to_dict(),
            "source_root": self.source_root,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArchivedEvidence":
        if type(value) is not dict or set(value) != {
            "blob",
            "reference",
            "source_root",
        }:
            raise ValidatorArchiveError("archived evidence is not closed")
        return cls(
            source_root=value["source_root"],  # type: ignore[arg-type]
            reference=EvidenceArtifactRef.from_dict(value["reference"]),
            blob=ArchiveBlobRef.from_dict(value["blob"]),
        )


@dataclass(frozen=True)
class ValidatorArchiveManifest:
    created_at_ns: int
    scope: IntakeScope
    database_schema: int
    finalized_cursor: tuple[int, str] | None
    table_counts: tuple[tuple[str, int], ...]
    database: ArchiveBlobRef
    audit_log: ArchiveBlobRef | None
    admitted_bundles: tuple[ArchivedTree, ...]
    qualification_evidence: tuple[ArchivedEvidence, ...]
    sealed_inputs: tuple[ArchivedTree, ...]

    def __post_init__(self) -> None:
        if type(self.created_at_ns) is not int or self.created_at_ns < 0:
            raise ValidatorArchiveError("archive timestamp is malformed")
        if type(self.scope) is not IntakeScope:
            raise ValidatorArchiveError("archive chain scope is not typed")
        if type(self.database_schema) is not int or self.database_schema <= 0:
            raise ValidatorArchiveError("archive database schema is malformed")
        if self.finalized_cursor is not None and (
            type(self.finalized_cursor) is not tuple
            or len(self.finalized_cursor) != 2
            or type(self.finalized_cursor[0]) is not int
            or self.finalized_cursor[0] < 0
            or not isinstance(self.finalized_cursor[1], str)
        ):
            raise ValidatorArchiveError("archive finalized cursor is malformed")
        if self.table_counts != tuple(sorted(set(self.table_counts))):
            raise ValidatorArchiveError("archive table counts are not canonical")
        for table, count in self.table_counts:
            if (
                _TABLE.fullmatch(table) is None
                or type(count) is not int
                or count < 0
            ):
                raise ValidatorArchiveError("archive table count is malformed")
        if type(self.database) is not ArchiveBlobRef:
            raise ValidatorArchiveError("archive database reference is not typed")
        if (
            self.database.media_type != "application/vnd.sqlite3"
            or self.database.size <= 0
        ):
            raise ValidatorArchiveError("archive database reference is malformed")
        if self.audit_log is not None and type(self.audit_log) is not ArchiveBlobRef:
            raise ValidatorArchiveError("archive audit reference is not typed")
        if (
            self.audit_log is not None
            and self.audit_log.media_type != "application/x-ndjson"
        ):
            raise ValidatorArchiveError("archive audit reference is malformed")
        for values, expected in (
            (self.admitted_bundles, ArchivedTree),
            (self.qualification_evidence, ArchivedEvidence),
            (self.sealed_inputs, ArchivedTree),
        ):
            if any(type(value) is not expected for value in values):
                raise ValidatorArchiveError("archive manifest child is not typed")

    @property
    def digest(self) -> str:
        return canonical_digest(
            _ARCHIVE_MANIFEST_DIGEST_DOMAIN,
            self.to_dict(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "admitted_bundles": [row.to_dict() for row in self.admitted_bundles],
            "audit_log": None if self.audit_log is None else self.audit_log.to_dict(),
            "created_at_ns": self.created_at_ns,
            "database": self.database.to_dict(),
            "database_schema": self.database_schema,
            "finalized_cursor": (
                None
                if self.finalized_cursor is None
                else [self.finalized_cursor[0], self.finalized_cursor[1]]
            ),
            "qualification_evidence": [
                row.to_dict() for row in self.qualification_evidence
            ],
            "schema": ARCHIVE_SCHEMA,
            "scope": self.scope.to_dict(),
            "scope_digest": _archive_scope_digest(self.scope),
            "sealed_inputs": [row.to_dict() for row in self.sealed_inputs],
            "table_counts": [
                {"count": count, "table": table}
                for table, count in self.table_counts
            ],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(
            {"manifest": self.to_dict(), "manifest_digest": self.digest}
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ValidatorArchiveManifest":
        try:
            return cls._from_bytes_unchecked(payload)
        except ValidatorArchiveError:
            raise
        except Exception as exc:
            # Do not leak attacker-controlled field values through a parser
            # exception. Callers receive one closed archive-domain error.
            raise ValidatorArchiveError(
                "archive manifest fields are malformed "
                f"({type(exc).__name__})"
            ) from None

    @classmethod
    def _from_bytes_unchecked(cls, payload: bytes) -> "ValidatorArchiveManifest":
        if not isinstance(payload, bytes) or len(payload) > MAX_MANIFEST_BYTES:
            raise ValidatorArchiveError("archive manifest bytes are malformed")
        try:
            envelope = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidatorArchiveError(f"archive manifest is invalid JSON: {exc}") from None
        if type(envelope) is not dict or set(envelope) != {
            "manifest",
            "manifest_digest",
        }:
            raise ValidatorArchiveError("archive manifest envelope is not closed")
        value = envelope["manifest"]
        fields = {
            "admitted_bundles",
            "audit_log",
            "created_at_ns",
            "database",
            "database_schema",
            "finalized_cursor",
            "qualification_evidence",
            "schema",
            "scope",
            "scope_digest",
            "sealed_inputs",
            "table_counts",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValidatorArchiveError("archive manifest schema is not closed")
        scope_value = value["scope"]
        if type(scope_value) is not dict or set(scope_value) != {"genesis_hash", "netuid"}:
            raise ValidatorArchiveError("archive scope is malformed")
        scope = IntakeScope(**scope_value)  # type: ignore[arg-type]
        if value["schema"] != ARCHIVE_SCHEMA:
            raise ValidatorArchiveError("archive manifest schema is not closed")
        if value["scope_digest"] != _archive_scope_digest(scope):
            raise ValidatorArchiveError("archive scope digest differs")
        cursor_value = value["finalized_cursor"]
        cursor = None
        if cursor_value is not None:
            if type(cursor_value) is not list or len(cursor_value) != 2:
                raise ValidatorArchiveError("archive finalized cursor is malformed")
            cursor = (cursor_value[0], cursor_value[1])
        counts_value = value["table_counts"]
        if type(counts_value) is not list:
            raise ValidatorArchiveError("archive table counts are malformed")
        counts: list[tuple[str, int]] = []
        for row in counts_value:
            if type(row) is not dict or set(row) != {"count", "table"}:
                raise ValidatorArchiveError("archive table count is not closed")
            counts.append((row["table"], row["count"]))  # type: ignore[arg-type]
        bundles = value["admitted_bundles"]
        evidence = value["qualification_evidence"]
        sealed = value["sealed_inputs"]
        if type(bundles) is not list or type(evidence) is not list or type(sealed) is not list:
            raise ValidatorArchiveError("archive manifest collections are malformed")
        audit = value["audit_log"]
        manifest = cls(
            created_at_ns=value["created_at_ns"],  # type: ignore[arg-type]
            scope=scope,
            database_schema=value["database_schema"],  # type: ignore[arg-type]
            finalized_cursor=cursor,
            table_counts=tuple(counts),
            database=ArchiveBlobRef.from_dict(value["database"]),
            audit_log=None if audit is None else ArchiveBlobRef.from_dict(audit),
            admitted_bundles=tuple(ArchivedTree.from_dict(row) for row in bundles),
            qualification_evidence=tuple(
                ArchivedEvidence.from_dict(row) for row in evidence
            ),
            sealed_inputs=tuple(ArchivedTree.from_dict(row) for row in sealed),
        )
        if envelope["manifest_digest"] != manifest.digest:
            raise ValidatorArchiveError("archive manifest digest differs")
        return manifest


@dataclass(frozen=True)
class ValidatorArchivePublication:
    manifest: ValidatorArchiveManifest
    manifest_key: str
    uploaded_blobs: int
    reused_blobs: int
    stored_bytes: int


@dataclass(frozen=True)
class ValidatorArchiveRestore:
    manifest: ValidatorArchiveManifest
    restore_root: Path
    database_path: Path
    audit_log_path: Path | None
    restore_map_path: Path


def _relative_path(value: str, *, allow_root: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValidatorArchiveError("archive relative path is malformed")
    if allow_root and value == ".":
        return value
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValidatorArchiveError("archive relative path is malformed")
    return value


def _safe_root(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    try:
        before = requested.lstat()
        root = requested.resolve(strict=True)
        after = root.lstat()
    except OSError as exc:
        raise ValidatorArchiveError(f"archive root is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) & 0o022
    ):
        raise ValidatorArchiveError(
            "archive root must be validator-owned and not group/world writable"
        )
    return root


def _read_regular(path: Path, *, max_bytes: int) -> tuple[bytes, int]:
    fd = -1
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size < 0
            or before.st_size > max_bytes
        ):
            raise ValidatorArchiveError(f"archive file has an unsafe shape: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        stable = (
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
        if any(getattr(before, name) != getattr(opened, name) for name in stable):
            raise ValidatorArchiveError(f"archive file changed while opening: {path}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(4 << 20, remaining))
            if not chunk:
                raise ValidatorArchiveError(f"archive file was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1) or any(
            getattr(opened, name) != getattr(os.fstat(fd), name) for name in stable
        ):
            raise ValidatorArchiveError(f"archive file changed while reading: {path}")
        return b"".join(chunks), stat.S_IMODE(opened.st_mode)
    except ValidatorArchiveError:
        raise
    except OSError as exc:
        raise ValidatorArchiveError(f"cannot read archive file {path}: {exc}") from None
    finally:
        if fd >= 0:
            os.close(fd)


def _read_audit_log(path: Path) -> bytes:
    fd = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        fcntl.flock(fd, fcntl.LOCK_SH)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > MAX_ARCHIVE_FILE_BYTES
        ):
            raise ValidatorArchiveError("chain audit log has an unsafe shape")
        payload = b""
        remaining = info.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(fd, min(4 << 20, remaining))
            if not chunk:
                raise ValidatorArchiveError("chain audit log was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if payload and not payload.endswith(b"\n"):
            raise ValidatorArchiveError("chain audit log ends with a partial record")
        for line in payload.splitlines():
            if len(line) + 1 > MAX_AUDIT_RECORD_BYTES:
                raise ValidatorArchiveError("chain audit log record exceeds its bound")
            value = json.loads(line)
            validate_chain_audit_record(value)
            if canonical_json_bytes(value) != line:
                raise ValidatorArchiveError("chain audit log record is not canonical")
        return payload
    except (UnicodeDecodeError, json.JSONDecodeError, ChainAuditLogError) as exc:
        raise ValidatorArchiveError(f"chain audit log is invalid: {exc}") from None
    except ValidatorArchiveError:
        raise
    except OSError as exc:
        raise ValidatorArchiveError(f"cannot read chain audit log: {exc}") from None
    finally:
        if fd >= 0:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _blob_ref(data: bytes, *, media_type: str) -> ArchiveBlobRef:
    digest = sha256_hex(data)
    return ArchiveBlobRef(
        key=f"blobs/sha256/{digest}",
        sha256=digest,
        size=len(data),
        media_type=media_type,
    )


def _get_bounded(store: ObjectStore, key: str, max_bytes: int) -> bytes:
    try:
        payload = store.get_bytes(key, max_bytes=max_bytes)
    except ObjectStoreError:
        raise
    if not isinstance(payload, bytes) or len(payload) > max_bytes:
        raise ValidatorArchiveError("archive object exceeds its retained size bound")
    return payload


def _publish_blob(
    store: ObjectStore,
    data: bytes,
    *,
    media_type: str,
) -> tuple[ArchiveBlobRef, bool]:
    reference = _blob_ref(data, media_type=media_type)
    reused = False
    try:
        existing = _get_bounded(store, reference.key, reference.size + 1)
    except ObjectStoreNotFoundError:
        try:
            store.put_bytes(
                reference.key,
                data,
                content_type=reference.media_type,
            )
        except ObjectStoreError as exc:
            raise ValidatorArchiveError(f"archive object upload failed: {exc}") from None
    else:
        reused = True
        if existing != data:
            raise ValidatorArchiveError("content-addressed archive object conflicts")
    try:
        reopened = _get_bounded(store, reference.key, reference.size + 1)
    except ObjectStoreError as exc:
        raise ValidatorArchiveError(f"archive object reopen failed: {exc}") from None
    if reopened != data:
        raise ValidatorArchiveError("stored archive object differs after reopen")
    return reference, reused


def _inventory_paths(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    directories = ["."]
    files: list[str] = []
    for current, names, leaves in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        names[:] = sorted(names)
        leaves.sort()
        for name in names:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValidatorArchiveError(f"archive tree contains unsafe entry: {path}")
            directories.append(path.relative_to(root).as_posix())
        for name in leaves:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValidatorArchiveError(f"archive tree contains unsafe entry: {path}")
            files.append(path.relative_to(root).as_posix())
    return tuple(sorted(directories)), tuple(sorted(files))


def _archive_tree(
    store: ObjectStore,
    source: str | Path,
    *,
    kind: str,
    name: str,
    content_hash: str = "",
    publication_digest: str = "",
) -> tuple[ArchivedTree, int, int, int]:
    root = _safe_root(source)
    before_directories, before_files = _inventory_paths(root)
    if len(before_files) > MAX_TREE_FILES:
        raise ValidatorArchiveError("archive tree exceeds its file-count bound")
    directories: list[ArchivedDirectory] = []
    for relative in before_directories:
        path = root if relative == "." else root / relative
        info = path.lstat()
        if (
            info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ValidatorArchiveError(
                f"archive directory has unsafe ownership or mode: {path}"
            )
        directories.append(
            ArchivedDirectory(relative, stat.S_IMODE(info.st_mode))
        )
    files: list[ArchivedFile] = []
    total = uploaded = reused = 0
    for relative in before_files:
        payload, mode = _read_regular(
            root / relative,
            max_bytes=MAX_ARCHIVE_FILE_BYTES,
        )
        total += len(payload)
        if total > MAX_TREE_BYTES:
            raise ValidatorArchiveError("archive tree exceeds its byte bound")
        blob, was_reused = _publish_blob(
            store,
            payload,
            media_type="application/octet-stream",
        )
        uploaded += not was_reused
        reused += was_reused
        files.append(ArchivedFile(relative, mode, blob))
    if _inventory_paths(root) != (before_directories, before_files):
        raise ValidatorArchiveError("archive tree changed during inventory")
    return (
        ArchivedTree(
            kind=kind,
            name=name,
            source_root=str(root),
            content_hash=content_hash,
            publication_digest=publication_digest,
            directories=tuple(directories),
            files=tuple(files),
        ),
        uploaded,
        reused,
        total,
    )


def _sqlite_snapshot(
    source_path: str | Path,
    temporary_root: Path,
) -> tuple[bytes, IntakeScope, int, tuple[int, str] | None, tuple[tuple[str, int], ...]]:
    requested = Path(source_path).expanduser()
    try:
        before = requested.lstat()
        source = requested.resolve(strict=True)
        after = source.lstat()
        parent = source.parent.lstat()
    except OSError as exc:
        raise ValidatorArchiveError(f"intake database is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or after.st_nlink != 1
        or after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) != 0o600
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise ValidatorArchiveError("intake database has an unsafe owner/mode/link shape")
    destination = temporary_root / "intake.sqlite3"
    source_db = destination_db = None
    try:
        source_db = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=30)
        destination_db = sqlite3.connect(destination)
        source_db.backup(destination_db)
        destination_db.commit()
    except sqlite3.Error as exc:
        raise ValidatorArchiveError(f"SQLite online backup failed: {exc}") from None
    finally:
        if destination_db is not None:
            destination_db.close()
        if source_db is not None:
            source_db.close()
    os.chmod(destination, 0o600)
    connection = sqlite3.connect(destination)
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if [row[0] for row in integrity] != ["ok"]:
            raise ValidatorArchiveError("SQLite backup failed integrity_check")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValidatorArchiveError("SQLite backup failed foreign_key_check")
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key,value FROM metadata")
        }
        try:
            scope_value = json.loads(metadata["intake_scope"])
            schema = int(metadata["schema"])
            cursor_value = (
                None
                if "finalized_cursor" not in metadata
                else json.loads(metadata["finalized_cursor"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidatorArchiveError(f"SQLite backup metadata is corrupt: {exc}") from None
        if type(scope_value) is not dict or set(scope_value) != {"genesis_hash", "netuid"}:
            raise ValidatorArchiveError("SQLite backup scope is malformed")
        scope = IntakeScope(**scope_value)
        cursor = None
        if cursor_value is not None:
            if type(cursor_value) is not list or len(cursor_value) != 2:
                raise ValidatorArchiveError("SQLite finalized cursor is malformed")
            cursor = (cursor_value[0], cursor_value[1])
        table_names = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        if any(_TABLE.fullmatch(name) is None for name in table_names):
            raise ValidatorArchiveError("SQLite backup contains an unsafe table name")
        counts = tuple(
            (
                name,
                int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{name}"'
                    ).fetchone()[0]
                ),
            )
            for name in table_names
        )
    finally:
        connection.close()
    payload = destination.read_bytes()
    if not 0 < len(payload) <= MAX_SQLITE_BYTES:
        raise ValidatorArchiveError("SQLite backup exceeds its byte bound")
    return payload, scope, schema, cursor, counts


def _database_rows(database_payload: bytes, temporary_root: Path):
    path = temporary_root / "query.sqlite3"
    path.write_bytes(database_payload)
    path.chmod(0o600)
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    return connection


def create_validator_archive(
    intake_db: str | Path,
    store: ObjectStore,
    *,
    audit_log: str | Path | None = None,
    sealed_inputs: Mapping[str, str | Path] | None = None,
    created_at_ns: int | None = None,
) -> ValidatorArchivePublication:
    """Create and verify one private validator recovery snapshot."""

    observed_ns = time.time_ns() if created_at_ns is None else created_at_ns
    if type(observed_ns) is not int or observed_ns < 0:
        raise ValidatorArchiveError("archive creation timestamp is malformed")
    named_sealed = dict(sealed_inputs or {})
    if any(_LABEL.fullmatch(name) is None for name in named_sealed):
        raise ValidatorArchiveError("sealed-input label is malformed")
    if len(named_sealed) != len(set(named_sealed)):
        raise ValidatorArchiveError("sealed-input labels are duplicated")

    uploaded = reused = stored_bytes = 0
    with tempfile.TemporaryDirectory(prefix="cacheon-validator-archive.") as temporary:
        temporary_root = Path(temporary)
        database_payload, scope, schema, cursor, table_counts = _sqlite_snapshot(
            intake_db,
            temporary_root,
        )
        database_ref, database_reused = _publish_blob(
            store,
            database_payload,
            media_type="application/vnd.sqlite3",
        )
        uploaded += not database_reused
        reused += database_reused
        stored_bytes += len(database_payload)

        audit_ref = None
        if audit_log is not None:
            audit_path = Path(audit_log).expanduser()
            if audit_path.exists():
                audit_payload = _read_audit_log(audit_path.resolve(strict=True))
                audit_ref, audit_reused = _publish_blob(
                    store,
                    audit_payload,
                    media_type="application/x-ndjson",
                )
                uploaded += not audit_reused
                reused += audit_reused
                stored_bytes += len(audit_payload)

        bundles: list[ArchivedTree] = []
        evidence_rows: list[ArchivedEvidence] = []
        connection = _database_rows(database_payload, temporary_root)
        try:
            bundle_rows = tuple(
                connection.execute(
                    "SELECT DISTINCT content_hash,publication_digest,publication_root "
                    "FROM reservations WHERE publication_root!='' "
                    "ORDER BY content_hash,publication_digest,publication_root"
                )
            )
            evidence_records = tuple(
                connection.execute(
                    "SELECT DISTINCT evidence_root,attempt_ref_json "
                    "FROM settlement_qualifications WHERE evidence_root!='' "
                    "AND attempt_ref_json!='' ORDER BY evidence_root,attempt_ref_json"
                )
            )
        finally:
            connection.close()

        seen_bundle_roots: set[str] = set()
        for row in bundle_rows:
            publication = reopen_worker_bundle(
                row["publication_root"],
                row["content_hash"],
                expected_receipt_digest=row["publication_digest"],
            )
            root_text = str(publication.root)
            if root_text in seen_bundle_roots:
                continue
            seen_bundle_roots.add(root_text)
            tree, tree_uploaded, tree_reused, tree_bytes = _archive_tree(
                store,
                publication.root,
                kind="admitted_bundle",
                name=publication.root.name,
                content_hash=row["content_hash"],
                publication_digest=row["publication_digest"],
            )
            bundles.append(tree)
            uploaded += tree_uploaded
            reused += tree_reused
            stored_bytes += tree_bytes

        seen_evidence: set[tuple[str, ...]] = set()
        for row in evidence_records:
            try:
                reference = EvidenceArtifactRef.from_dict(
                    json.loads(row["attempt_ref_json"])
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValidatorArchiveError(
                    f"retained evidence reference is corrupt: {exc}"
                ) from None
            root = Path(row["evidence_root"])
            identity = (
                str(root),
                reference.domain,
                reference.media_type,
                reference.schema,
                reference.sha256,
                str(reference.size),
            )
            if identity in seen_evidence:
                continue
            seen_evidence.add(identity)
            payload = reopen_evidence(root, reference)
            blob, evidence_reused = _publish_blob(
                store,
                payload,
                media_type=reference.media_type,
            )
            evidence_rows.append(ArchivedEvidence(str(root), reference, blob))
            uploaded += not evidence_reused
            reused += evidence_reused
            stored_bytes += len(payload)

        sealed_trees: list[ArchivedTree] = []
        for name, root in sorted(named_sealed.items()):
            tree, tree_uploaded, tree_reused, tree_bytes = _archive_tree(
                store,
                root,
                kind="sealed_input",
                name=name,
            )
            sealed_trees.append(tree)
            uploaded += tree_uploaded
            reused += tree_reused
            stored_bytes += tree_bytes

        manifest = ValidatorArchiveManifest(
            created_at_ns=observed_ns,
            scope=scope,
            database_schema=schema,
            finalized_cursor=cursor,
            table_counts=table_counts,
            database=database_ref,
            audit_log=audit_ref,
            admitted_bundles=tuple(bundles),
            qualification_evidence=tuple(evidence_rows),
            sealed_inputs=tuple(sealed_trees),
        )
        manifest_payload = manifest.to_bytes()
        manifest_key = (
            f"snapshots/{_archive_scope_digest(scope)}/"
            f"{observed_ns}-{manifest.digest}.json"
        )
        try:
            existing = _get_bounded(store, manifest_key, MAX_MANIFEST_BYTES)
        except ObjectStoreNotFoundError:
            try:
                store.put_bytes(
                    manifest_key,
                    manifest_payload,
                    content_type="application/json",
                )
            except ObjectStoreError as exc:
                raise ValidatorArchiveError(
                    f"archive manifest upload failed: {exc}"
                ) from None
        else:
            if existing != manifest_payload:
                raise ValidatorArchiveError("archive manifest key conflicts")
        try:
            reopened_manifest = _get_bounded(
                store,
                manifest_key,
                MAX_MANIFEST_BYTES,
            )
        except ObjectStoreError as exc:
            raise ValidatorArchiveError(
                f"archive manifest reopen failed: {exc}"
            ) from None
        if ValidatorArchiveManifest.from_bytes(reopened_manifest) != manifest:
            raise ValidatorArchiveError("archive manifest differs after reopen")
        return ValidatorArchivePublication(
            manifest,
            manifest_key,
            uploaded,
            reused,
            stored_bytes,
        )


def _fetch_blob(store: ObjectStore, reference: ArchiveBlobRef) -> bytes:
    try:
        payload = _get_bounded(store, reference.key, reference.size + 1)
    except ObjectStoreError as exc:
        raise ValidatorArchiveError(f"archive blob fetch failed: {exc}") from None
    if (
        len(payload) != reference.size
        or sha256_hex(payload) != reference.sha256
    ):
        raise ValidatorArchiveError("archive blob differs from its reference")
    return payload


def _write_file(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ValidatorArchiveError("archive restore write stalled")
            offset += written
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)


def _restore_tree(
    store: ObjectStore,
    tree: ArchivedTree,
    destination: Path,
) -> None:
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for directory in tree.directories:
        if directory.path == ".":
            continue
        (destination / directory.path).mkdir(mode=0o700, parents=True, exist_ok=True)
    for archived in tree.files:
        _write_file(
            destination / archived.path,
            _fetch_blob(store, archived.blob),
            archived.mode,
        )
    for directory in sorted(
        tree.directories,
        key=lambda value: len(PurePosixPath(value.path).parts),
        reverse=True,
    ):
        path = destination if directory.path == "." else destination / directory.path
        path.chmod(directory.mode)


def _validate_sqlite_payload(path: Path) -> None:
    # The online backup is a closed, standalone image. Its header retains the
    # source's WAL journal mode, but no WAL/SHM sidecars are required or trusted
    # during restore verification.
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
            raise ValidatorArchiveError("restored SQLite failed integrity_check")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValidatorArchiveError("restored SQLite failed foreign_key_check")
    finally:
        connection.close()


def restore_validator_archive(
    store: ObjectStore,
    manifest_key: str,
    restore_root: str | Path,
) -> ValidatorArchiveRestore:
    """Materialize and authenticate one snapshot into a fresh staging root."""

    try:
        manifest_payload = _get_bounded(store, manifest_key, MAX_MANIFEST_BYTES)
    except ObjectStoreError as exc:
        raise ValidatorArchiveError(f"archive manifest fetch failed: {exc}") from None
    manifest = ValidatorArchiveManifest.from_bytes(manifest_payload)
    expected_manifest_key = (
        f"snapshots/{_archive_scope_digest(manifest.scope)}/"
        f"{manifest.created_at_ns}-{manifest.digest}.json"
    )
    if manifest_key != expected_manifest_key:
        raise ValidatorArchiveError("archive manifest key does not bind its digest")
    destination = Path(restore_root).expanduser()
    if destination.exists() or destination.is_symlink():
        raise ValidatorArchiveError("archive restore root must not already exist")
    requested = (
        Path.cwd() / destination
        if not destination.is_absolute()
        else Path(os.path.normpath(destination))
    )
    try:
        destination = requested.parent.resolve(strict=True) / requested.name
    except OSError as exc:
        raise ValidatorArchiveError(
            f"archive restore parent is unavailable: {exc}"
        ) from None
    previous_umask = os.umask(0o077)
    try:
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        database_path = destination / "intake.sqlite3"
        _write_file(
            database_path,
            _fetch_blob(store, manifest.database),
            0o600,
        )
        _validate_sqlite_payload(database_path)

        audit_path = None
        if manifest.audit_log is not None:
            audit_path = destination / "chain-audit.jsonl"
            _write_file(
                audit_path,
                _fetch_blob(store, manifest.audit_log),
                0o600,
            )
            _read_audit_log(audit_path)

        mappings: list[dict[str, str]] = []
        worker_root = destination / "worker"
        worker_root.mkdir(mode=0o700)
        for tree in manifest.admitted_bundles:
            target = worker_root / tree.name[:2] / tree.name
            _restore_tree(store, tree, target)
            reopen_worker_bundle(
                target,
                tree.content_hash,
                expected_receipt_digest=tree.publication_digest,
            )
            mappings.append(
                {
                    "kind": tree.kind,
                    "restored": str(target),
                    "source": tree.source_root,
                }
            )

        evidence_roots: dict[str, Path] = {}
        evidence_parent = destination / "evidence"
        evidence_parent.mkdir(mode=0o700)
        for item in manifest.qualification_evidence:
            root_id = canonical_digest(
                _ARCHIVED_EVIDENCE_ROOT_DIGEST_DOMAIN,
                {"source_root": item.source_root},
            )
            target_root = evidence_roots.get(item.source_root)
            if target_root is None:
                target_root = evidence_parent / root_id
                target_root.mkdir(mode=0o700)
                evidence_roots[item.source_root] = target_root
                mappings.append(
                    {
                        "kind": "qualification_evidence",
                        "restored": str(target_root),
                        "source": item.source_root,
                    }
                )
            target = (
                target_root
                / item.reference.domain
                / item.reference.sha256[:2]
                / item.reference.sha256
            )
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_file(target, _fetch_blob(store, item.blob), 0o400)
            reopen_evidence(target_root, item.reference)

        sealed_parent = destination / "sealed"
        sealed_parent.mkdir(mode=0o700)
        for tree in manifest.sealed_inputs:
            target = sealed_parent / tree.name
            _restore_tree(store, tree, target)
            mappings.append(
                {
                    "kind": tree.kind,
                    "restored": str(target),
                    "source": tree.source_root,
                }
            )

        restore_map = destination / "restore-map.json"
        _write_file(
            restore_map,
            canonical_json_bytes(
                {
                    "manifest_digest": manifest.digest,
                    "mappings": mappings,
                    "schema": _RESTORE_MAP_SCHEMA,
                }
            ),
            0o600,
        )
        return ValidatorArchiveRestore(
            manifest,
            destination,
            database_path,
            audit_path,
            restore_map,
        )
    except Exception:
        # Preserve a partial staging root for operator inspection. It is fresh,
        # private, and never replaces live validator state.
        raise
    finally:
        os.umask(previous_umask)


def verify_validator_archive(
    store: ObjectStore,
    manifest_key: str,
) -> ValidatorArchiveManifest:
    """Download, stage, and semantically reopen a snapshot without retaining it."""

    with tempfile.TemporaryDirectory(prefix="cacheon-validator-restore.") as temporary:
        root = Path(temporary).resolve(strict=True)
        root.chmod(0o700)
        destination = root / "snapshot"
        return restore_validator_archive(store, manifest_key, destination).manifest


def parse_named_roots(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeated ``NAME=PATH`` sealed-input CLI values."""

    result: dict[str, Path] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ValidatorArchiveError("sealed input must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if _LABEL.fullmatch(name) is None or not raw_path:
            raise ValidatorArchiveError("sealed input must use canonical NAME=PATH")
        if name in result:
            raise ValidatorArchiveError("sealed-input label is duplicated")
        result[name] = Path(raw_path)
    return result


__all__ = [
    "ARCHIVE_SCHEMA",
    "DEFAULT_VALIDATOR_ARCHIVE_PREFIX",
    "ArchiveBlobRef",
    "ArchivedDirectory",
    "ArchivedEvidence",
    "ArchivedFile",
    "ArchivedTree",
    "ValidatorArchiveError",
    "ValidatorArchiveManifest",
    "ValidatorArchivePublication",
    "ValidatorArchiveRestore",
    "create_validator_archive",
    "parse_named_roots",
    "restore_validator_archive",
    "verify_validator_archive",
]

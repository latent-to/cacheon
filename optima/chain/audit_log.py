"""Redacted append-only operational audit records for finalized chain passes.

SQLite remains the authority for transitions.  This journal is deliberately
smaller: it records pass chronology and content-derived identifiers without
URLs, hotkeys, candidate bytes, exception messages, or ambient environment.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import time
from pathlib import Path

from optima.stack_identity import canonical_json_bytes


class ChainAuditLogError(RuntimeError):
    """A configured operational audit record could not be retained safely."""


_SAFE_REASON = re.compile(r"^[a-z0-9_.-]{1,96}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MAX_PASS_ITEMS = 65_536
MAX_AUDIT_RECORD_BYTES = 8 << 20
_PASS_FIELDS = {
    "copies",
    "decisions",
    "event",
    "finalized_block",
    "finalized_block_hash",
    "held",
    "published",
    "rejected",
    "reserved",
    "schema",
    "screens",
    "seen",
    "settlements",
    "timestamp_ns",
}
_FAULT_FIELDS = {
    "consecutive_failures",
    "error_type",
    "event",
    "schema",
    "timestamp_ns",
}


def _reason_class(value: object) -> str:
    """Return a bounded non-sensitive disposition class."""

    if not isinstance(value, str):
        return "redacted"
    candidate = value.partition(":")[0].strip().lower()
    return candidate if _SAFE_REASON.fullmatch(candidate) is not None else "redacted"


def _digest_map(
    value: object,
    *,
    field: str,
    allowed_values: frozenset[str] | None = None,
) -> None:
    if type(value) is not dict or len(value) > _MAX_PASS_ITEMS:
        raise ChainAuditLogError(f"chain audit {field} map is malformed")
    for key, item in value.items():
        if not isinstance(key, str) or _SHA256.fullmatch(key) is None:
            raise ChainAuditLogError(f"chain audit {field} key is malformed")
        if allowed_values is None:
            if not isinstance(item, str) or _SHA256.fullmatch(item) is None:
                raise ChainAuditLogError(
                    f"chain audit {field} digest is malformed"
                )
        elif item not in allowed_values:
            raise ChainAuditLogError(f"chain audit {field} value is malformed")


def _digest_list(value: object, *, field: str, sorted_unique: bool) -> None:
    if (
        type(value) is not list
        or len(value) > _MAX_PASS_ITEMS
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in value
        )
        or len(value) != len(set(value))
        or (sorted_unique and value != sorted(value))
    ):
        raise ChainAuditLogError(f"chain audit {field} list is malformed")


def _reason_map(value: object) -> None:
    if type(value) is not dict or len(value) > _MAX_PASS_ITEMS:
        raise ChainAuditLogError("chain audit rejected map is malformed")
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or _SHA256.fullmatch(key) is None
            or not isinstance(item, str)
            or _SAFE_REASON.fullmatch(item) is None
        ):
            raise ChainAuditLogError("chain audit rejected value is malformed")


def validate_chain_audit_record(record: object) -> None:
    """Validate the closed redacted record schema used on append and restore."""

    if type(record) is not dict or record.get("schema") != "optima.chain-audit.v1":
        raise ChainAuditLogError("chain audit record schema is malformed")
    event = record.get("event")
    if event == "validator_fault":
        if (
            set(record) != _FAULT_FIELDS
            or type(record.get("timestamp_ns")) is not int
            or record["timestamp_ns"] < 0
            or type(record.get("consecutive_failures")) is not int
            or record["consecutive_failures"] <= 0
            or not isinstance(record.get("error_type"), str)
            or _ERROR_TYPE.fullmatch(record["error_type"]) is None
        ):
            raise ChainAuditLogError("chain fault audit record is malformed")
        return
    if event != "pass" or set(record) != _PASS_FIELDS:
        raise ChainAuditLogError("chain pass audit record is not closed")
    if (
        type(record.get("timestamp_ns")) is not int
        or record["timestamp_ns"] < 0
        or type(record.get("finalized_block")) is not int
        or record["finalized_block"] < 0
        or not isinstance(record.get("finalized_block_hash"), str)
        or _BLOCK_HASH.fullmatch(record["finalized_block_hash"]) is None
        or type(record.get("seen")) is not int
        or record["seen"] < 0
    ):
        raise ChainAuditLogError("chain pass audit scalar is malformed")
    _digest_list(record["reserved"], field="reserved", sorted_unique=False)
    _digest_list(record["held"], field="held", sorted_unique=True)
    _digest_map(record["copies"], field="copies")
    _digest_map(record["published"], field="published")
    _digest_map(record["settlements"], field="settlements")
    _digest_map(
        record["decisions"],
        field="decisions",
        allowed_values=frozenset({"PASS", "FAIL", "NO_DECISION"}),
    )
    _digest_map(
        record["screens"],
        field="screens",
        allowed_values=frozenset({"pass", "fail", "no_decision", "hold"}),
    )
    _reason_map(record["rejected"])


def pass_audit_record(result, *, timestamp_ns: int | None = None) -> dict[str, object]:
    """Build one canonical, redacted record from a validator pass result."""

    from optima.chain.validator_loop import PassResult

    if type(result) is not PassResult:
        raise ChainAuditLogError("chain pass audit input is not exactly typed")
    observed_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
    if type(observed_ns) is not int or observed_ns < 0:
        raise ChainAuditLogError("chain pass audit timestamp is malformed")
    record = {
        "copies": dict(sorted(result.copies.items())),
        "decisions": dict(sorted(result.decisions.items())),
        "event": "pass",
        "finalized_block": result.finalized_block,
        "finalized_block_hash": result.finalized_block_hash,
        "held": sorted(set(result.held)),
        "published": dict(sorted(result.published.items())),
        "rejected": {
            reservation: _reason_class(reason)
            for reservation, reason in sorted(result.rejected.items())
        },
        "reserved": list(result.reserved),
        "schema": "optima.chain-audit.v1",
        "screens": dict(sorted(result.screens.items())),
        "seen": result.seen,
        "settlements": dict(sorted(result.settlements.items())),
        "timestamp_ns": observed_ns,
    }
    validate_chain_audit_record(record)
    return record


def fault_audit_record(
    exc: BaseException,
    *,
    consecutive_failures: int,
    timestamp_ns: int | None = None,
) -> dict[str, object]:
    """Build a redacted controller-fault record without the exception message."""

    observed_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
    if (
        type(observed_ns) is not int
        or observed_ns < 0
        or type(consecutive_failures) is not int
        or consecutive_failures <= 0
    ):
        raise ChainAuditLogError("chain fault audit metadata is malformed")
    record = {
        "consecutive_failures": consecutive_failures,
        "error_type": type(exc).__name__[:128],
        "event": "validator_fault",
        "schema": "optima.chain-audit.v1",
        "timestamp_ns": observed_ns,
    }
    validate_chain_audit_record(record)
    return record


def append_chain_audit(path: str | Path, record: dict[str, object]) -> None:
    """Append and fsync one canonical JSON line under an exclusive file lock."""

    validate_chain_audit_record(record)
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = (Path.cwd() / destination).resolve()
    else:
        destination = Path(os.path.normpath(destination))
    parent = destination.parent
    previous_umask = os.umask(0o077)
    fd = -1
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_info = parent.lstat()
        if (
            not stat.S_ISLNK(parent_info.st_mode)
            and stat.S_ISDIR(parent_info.st_mode)
            and parent_info.st_uid == os.geteuid()
            and stat.S_IMODE(parent_info.st_mode) != 0o700
            and parent.resolve(strict=True) == parent
        ):
            # A sibling component (private root, publication root) may have
            # created this directory first under the ambient umask.  mkdir with
            # exist_ok never corrects an existing mode, so heal our own
            # canonical directory to the owner-only contract instead of
            # refusing every append for the life of the process.
            os.chmod(parent, 0o700)
            parent_info = parent.lstat()
        if (
            stat.S_ISLNK(parent_info.st_mode)
            or not stat.S_ISDIR(parent_info.st_mode)
            or parent_info.st_uid != os.geteuid()
            or stat.S_IMODE(parent_info.st_mode) != 0o700
            or parent.resolve(strict=True) != parent
        ):
            raise ChainAuditLogError(
                "chain audit parent must be a canonical owner-only directory"
            )
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(destination, flags, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ChainAuditLogError("chain audit file has an unsafe shape")
        payload = canonical_json_bytes(record) + b"\n"
        if len(payload) > MAX_AUDIT_RECORD_BYTES:
            raise ChainAuditLogError("chain audit record exceeds its byte bound")
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ChainAuditLogError("chain audit append stalled")
            offset += written
        os.fsync(fd)
    except ChainAuditLogError:
        raise
    except OSError as exc:
        raise ChainAuditLogError(f"chain audit append failed: {exc}") from None
    finally:
        if fd >= 0:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        os.umask(previous_umask)


__all__ = [
    "ChainAuditLogError",
    "MAX_AUDIT_RECORD_BYTES",
    "append_chain_audit",
    "fault_audit_record",
    "pass_audit_record",
    "validate_chain_audit_record",
]

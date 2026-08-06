"""Durable proof that a trusted pod adapter crossed resident entry.

The marker is written after request authentication and candidate staging, but
before the commissioned resident worker is called.  Its absence is never proof
that execution did not occur unless the adapter protocol itself returned a
typed pre-resident refusal.  Its presence always forbids a fresh experiment.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Mapping

from cacheon.chain.remote_worker_spool import (
    SCHEMA_REQUEST,
    require_digest,
    require_int,
    require_text,
    spool_canonical_json,
    strict_json_object,
    verify_lease,
)


SCHEMA_RESIDENT_ENTRY = "cacheon-remote-resident-entry-armed-v1"
RESIDENT_ENTRY_MARKER = "RESIDENT_ENTRY_ARMED.json"
_FIELDS = frozenset(
    {
        "generation",
        "lease_id",
        "ready_receipt_digest",
        "remote_request_sha256",
        "request_id",
        "schema",
        "service_identity",
        "stage",
        "worker_epoch",
        "worker_readiness_digest",
    }
)


class RemoteWorkerExecutionMarkerError(RuntimeError):
    """A resident-entry marker is malformed, ambiguous, or not durable."""


def _fail(message: str) -> None:
    raise RemoteWorkerExecutionMarkerError(message)


def _request_artifact(request: Mapping[str, Any], stage: str) -> str:
    role = f"{stage}_payload"
    artifacts = request.get("artifacts")
    if type(artifacts) is not list:
        _fail("resident-entry request artifacts are malformed")
    matches = [
        row
        for row in artifacts
        if type(row) is dict and row.get("role") == role
    ]
    if len(matches) != 1:
        _fail("resident-entry request payload is absent or ambiguous")
    try:
        return require_digest(matches[0].get("sha256"), "remote request sha256")
    except Exception as exc:
        raise RemoteWorkerExecutionMarkerError(str(exc)) from None


def _worker_epoch(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("resident-entry worker epoch is malformed")
    return value


def marker_for_request(request: Mapping[str, Any]) -> dict[str, object]:
    """Derive the closed marker solely from one authenticated spool request."""

    if type(request) is not dict or request.get("schema") != SCHEMA_REQUEST:
        _fail("resident-entry request is not one verified spool request")
    try:
        lease = verify_lease(request.get("lease"))
        stage = lease["stage"]
        if stage not in ("screen", "qualification"):
            _fail("resident-entry stage is unsupported")
        return {
            "generation": require_int(
                lease["generation"], "resident-entry generation", minimum=1
            ),
            "lease_id": require_digest(lease["lease_id"], "resident-entry lease id"),
            "ready_receipt_digest": require_digest(
                request.get("ready_receipt_digest"), "resident-entry READY digest"
            ),
            "remote_request_sha256": _request_artifact(request, stage),
            "request_id": require_digest(
                request.get("request_id"), "resident-entry request id"
            ),
            "schema": SCHEMA_RESIDENT_ENTRY,
            "service_identity": require_text(
                request.get("service_identity"),
                "resident-entry service identity",
                maximum=256,
            ),
            "stage": stage,
            "worker_epoch": _worker_epoch(request.get("worker_epoch")),
            "worker_readiness_digest": require_digest(
                request.get("worker_readiness_digest"),
                "resident-entry worker readiness",
            ),
        }
    except RemoteWorkerExecutionMarkerError:
        raise
    except Exception as exc:
        raise RemoteWorkerExecutionMarkerError(str(exc)) from None


def _private_result_directory(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        _fail("resident-entry result root is not an absolute Path")
    try:
        info = path.lstat()
    except OSError as exc:
        raise RemoteWorkerExecutionMarkerError(
            f"resident-entry result root is unavailable: {exc}"
        ) from None
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
    ):
        _fail("resident-entry result root is not validator-owned mode 0700")


def publish_resident_entry(
    result_root: Path, request: Mapping[str, Any]
) -> dict[str, object]:
    """Publish the immutable marker before any resident worker invocation."""

    _private_result_directory(result_root)
    marker = marker_for_request(request)
    path = result_root / RESIDENT_ENTRY_MARKER
    raw = spool_canonical_json(marker) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(
            result_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise RemoteWorkerExecutionMarkerError(
            f"resident-entry marker cannot publish: {exc}"
        ) from None
    return marker


def reopen_resident_entry(
    result_root: Path, request: Mapping[str, Any]
) -> dict[str, object]:
    """Reopen one canonical marker against the exact authenticated request."""

    _private_result_directory(result_root)
    path = result_root / RESIDENT_ENTRY_MARKER
    try:
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            after = os.fstat(handle.fileno())
            raw = handle.read(64 * 1024 + 1)
    except OSError as exc:
        raise RemoteWorkerExecutionMarkerError(
            f"resident-entry marker cannot reopen: {exc}"
        ) from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or len(raw) > 64 * 1024
    ):
        _fail("resident-entry marker file identity is unsafe")
    try:
        value = strict_json_object(raw.decode("utf-8"))
    except Exception as exc:
        raise RemoteWorkerExecutionMarkerError(
            f"resident-entry marker JSON is invalid: {exc}"
        ) from None
    expected = marker_for_request(request)
    if (
        set(value) != _FIELDS
        or value != expected
        or raw != spool_canonical_json(value) + b"\n"
    ):
        _fail("resident-entry marker differs from the authenticated request")
    return value


__all__ = [
    "RESIDENT_ENTRY_MARKER",
    "RemoteWorkerExecutionMarkerError",
    "marker_for_request",
    "publish_resident_entry",
    "reopen_resident_entry",
]

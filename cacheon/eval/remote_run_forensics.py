"""Request-scoped evidence for the existing remote ``worker_log`` artifact."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import time
import traceback
from collections.abc import Iterator, Mapping
from functools import wraps
from pathlib import Path
from typing import Any


RUN_EVENT_SCHEMA = "cacheon.remote-run-event.v1"
WORKER_LOG_SCHEMA = "cacheon.remote-worker-log.v1"
JOURNAL_NAME = "worker-log.events.jsonl"
JOURNAL_ENV = "CACHEON_REMOTE_RUN_JOURNAL"
REQUEST_ENV = "CACHEON_REMOTE_REQUEST_ID"
_HEX = frozenset("0123456789abcdef")


class RemoteRunForensicsError(RuntimeError):
    """A request's diagnostic record could not be retained or reopened."""


def _request_id(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise RemoteRunForensicsError("remote run request ID is malformed")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RemoteRunForensicsError("remote run journal write made no progress")
        view = view[written:]


def component_for_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if "/swap-intake/" in normalized or "/candidate-publications/" in normalized:
        return "miner_bundle"
    if any(
        marker in normalized
        for marker in (
            "/site-packages/sglang/",
            "/dist-packages/sglang/",
            "/sglang/python/sglang/",
        )
    ):
        return "sglang"
    if any(
        marker in normalized
        for marker in ("/cacheon/integrations/", "/cacheon/seam.py", "/cacheon/slots.py")
    ):
        return "slot_adapter"
    if "/site-packages/" in normalized or "/dist-packages/" in normalized:
        return "dependency"
    if "/cacheon/eval/oci_" in normalized:
        return "oci_host"
    if "/cacheon/chain/" in normalized:
        return "chain_control"
    if "/cacheon/" in normalized:
        return "validator"
    return "unknown"


def exception_record(exc: BaseException) -> dict[str, object]:
    """Retain the complete Python exception chain and classify only observed frames."""

    chain: list[dict[str, object]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames = [
            {
                "component": component_for_path(frame.filename),
                "file": frame.filename,
                "function": frame.name,
                "line": frame.lineno,
            }
            for frame in traceback.extract_tb(current.__traceback__)
        ]
        chain.append(
            {
                "frames": frames,
                "message": str(current)[:65_536],
                "type": type(current).__name__,
            }
        )
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    candidate_types = {
        "CandidateEngineFailure",
        "CandidateExecutionFailure",
        "CandidateNeverExecutedError",
        "OuterSessionCandidateError",
    }
    components = [
        frame["component"]
        for item in chain
        for frame in item["frames"]  # type: ignore[index]
        if frame["component"] != "unknown"  # type: ignore[index]
    ]
    component = (
        "miner_bundle"
        if any(item["type"] in candidate_types for item in chain)
        else components[-1]
        if components
        else "unknown"
    )
    return {"component": component, "exceptions": chain}


def journal_path(result_root: Path) -> Path:
    return Path(result_root) / JOURNAL_NAME


def append_event(
    destination: Path,
    request_id: str,
    phase: str,
    state: str,
    **facts: object,
) -> None:
    request_id = _request_id(request_id)
    if not phase or not state or any(c in phase + state for c in "\x00\r\n"):
        raise RemoteRunForensicsError("remote run phase/state is malformed")
    row = {
        **facts,
        "phase": phase,
        "request_id": request_id,
        "schema": RUN_EVENT_SCHEMA,
        "state": state,
        "time_unix_ns": time.time_ns(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        _write_all(descriptor, _canonical(row) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def bind_request(result_root: Path, request_id: str) -> Iterator[None]:
    """Expose one sequential adapter request to OCI diagnostic publication."""

    request_id = _request_id(request_id)
    journal = str(journal_path(result_root))
    previous = {name: os.environ.get(name) for name in (JOURNAL_ENV, REQUEST_ENV)}
    if any(value not in (None, journal, request_id) for value in previous.values()):
        raise RemoteRunForensicsError("another remote request owns the diagnostic context")
    os.environ[JOURNAL_ENV] = journal
    os.environ[REQUEST_ENV] = request_id
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def record_oci_artifact(receipt: object) -> None:
    """Link every retained OCI stdout/stderr receipt to the active request."""

    destination, request_id = os.environ.get(JOURNAL_ENV), os.environ.get(REQUEST_ENV)
    if destination is None and request_id is None:
        return
    if destination is None or request_id is None or not hasattr(receipt, "to_dict"):
        raise RemoteRunForensicsError("OCI diagnostic context is incomplete")
    append_event(
        Path(destination),
        request_id,
        "oci.output",
        "retained",
        diagnostic=receipt.to_dict(),  # type: ignore[union-attr]
        receipt_sha256=getattr(receipt, "receipt_sha256"),
    )


def append_adapter_stream(
    result_root: Path, request_id: str, source: Path, offset: int
) -> None:
    """Copy exactly the persistent-adapter bytes emitted for this request."""

    with source.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read()
    append_event(
        journal_path(result_root),
        request_id,
        "adapter.output",
        "retained",
        stream={
            "bytes": len(payload),
            "payload_base64": base64.b64encode(payload).decode("ascii"),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )


def capture_adapter_stream(function):
    """Decorate the pod adapter call with its exact global-log byte interval."""

    @wraps(function)
    def wrapped(adapter, request, job_root, result_root, *, deadline):
        request_id = request["request_id"]
        source = adapter.paths.root / "logs" / "persistent-adapter.log"
        offset = source.stat().st_size if source.exists() else 0
        try:
            return function(
                adapter, request, job_root, result_root, deadline=deadline
            )
        finally:
            if adapter.log_handle is not None:
                adapter.log_handle.flush()
            if source.is_file():
                append_adapter_stream(result_root, request_id, source, offset)

    return wrapped


def _oci_streams(events: list[dict[str, Any]]) -> list[dict[str, object]]:
    streams: list[dict[str, object]] = []
    seen: set[str] = set()
    for event in events:
        receipt = event.get("diagnostic")
        if event.get("phase") != "oci.output" or not isinstance(receipt, dict):
            continue
        path = Path(str(receipt.get("artifact_path", "")))
        receipt_path = Path(str(receipt.get("receipt_path", "")))
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not receipt_path.is_absolute()
            or receipt_path.is_symlink()
        ):
            raise RemoteRunForensicsError("linked OCI output path is not retained")
        if not path.is_file() or not receipt_path.is_file():
            # A pod interrupt takes the runtime's --rm resource tree with it, so a
            # recovery seal can outlive the stream its journal points at.  Raising
            # here would abort the only result the request will ever get and leave
            # the next recovery to raise again, so the loss is stated and named.
            absent = str(receipt.get("artifact_sha256", ""))
            if absent in seen:
                continue
            seen.add(absent)
            streams.append(
                {
                    "receipt": receipt,
                    "receipt_sha256": event.get("receipt_sha256"),
                    "state": "not_retained",
                }
            )
            continue
        raw = path.read_bytes()
        receipt_raw = receipt_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != receipt.get("artifact_sha256")
            or len(raw) != receipt.get("artifact_bytes")
            or hashlib.sha256(receipt_raw).hexdigest() != event.get("receipt_sha256")
        ):
            raise RemoteRunForensicsError("linked OCI output differs from its receipt")
        if digest in seen:
            continue
        seen.add(digest)
        streams.append(
            {
                "payload_base64": base64.b64encode(raw).decode("ascii"),
                "receipt": receipt,
                "receipt_sha256": event["receipt_sha256"],
            }
        )
    return streams


def publish_worker_log(result_root: Path, request_id: str) -> dict[str, object]:
    """Seal the incremental journal and linked raw streams into one output artifact."""

    request_id = _request_id(request_id)
    journal = journal_path(result_root)
    try:
        events = [json.loads(line) for line in journal.read_text().splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteRunForensicsError(f"remote run journal is unreadable: {exc}") from None
    if not events or any(
        type(row) is not dict
        or row.get("schema") != RUN_EVENT_SCHEMA
        or row.get("request_id") != request_id
        for row in events
    ):
        raise RemoteRunForensicsError("remote run journal identity differs")
    payload = _canonical(
        {
            "events": events,
            "oci_streams": _oci_streams(events),
            "request_id": request_id,
            "schema": WORKER_LOG_SCHEMA,
        }
    )
    digest = hashlib.sha256(payload).hexdigest()
    blobs = result_root / "blobs"
    blobs.mkdir(mode=0o700, exist_ok=True)
    destination = blobs / digest
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise RemoteRunForensicsError("worker log artifact collision")
    else:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o400)
    return {"role": "worker_log", "sha256": digest, "size": len(payload)}


def reopen_worker_log(result_root: Path, result: Mapping[str, object]) -> dict[str, Any] | None:
    matches = [row for row in result.get("artifacts", []) if row.get("role") == "worker_log"]  # type: ignore[union-attr]
    if not matches:
        return None
    if len(matches) != 1:
        raise RemoteRunForensicsError("result has multiple worker logs")
    row = matches[0]
    path = result_root / "blobs" / str(row["sha256"])
    raw = path.read_bytes()
    if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise RemoteRunForensicsError("worker log differs from its result artifact")
    value = json.loads(raw)
    if (
        type(value) is not dict
        or value.get("schema") != WORKER_LOG_SCHEMA
        or value.get("request_id") != result.get("request_id")
        or type(value.get("events")) is not list
        or type(value.get("oci_streams")) is not list
    ):
        raise RemoteRunForensicsError("worker log payload is malformed")
    return value


def worker_log_retention(value: Mapping[str, object]) -> str:
    """State whether every retained diagnostic stream is downloadable in full."""

    streams = value.get("oci_streams")
    if not isinstance(streams, list):
        return "unreadable"
    for stream in streams:
        if not isinstance(stream, dict) or not isinstance(stream.get("receipt"), dict):
            return "unreadable"
        if stream.get("state") == "not_retained" or "payload_base64" not in stream:
            return "partial"
        if stream["receipt"].get("truncated") is True:
            return "partial"
    return "complete"


def _epoch_dirs(root: Path, kind: str) -> list[Path]:
    """Return one spool kind's current and retired directories by name."""

    return sorted(
        path
        for path in root.glob(f"{kind}*")
        if path.is_dir() and (path.name == kind or path.name.startswith(f"{kind}-"))
    )


def result_dir_for_request(root: Path, request_id: str) -> Path | None:
    """Locate one exact result across current and retired worker epochs.

    Cutover moves a result rather than copying it. Byte-identical duplicate
    ``result.json`` files are therefore one record; differing records for one
    request ID are an integrity error rather than a choice for the caller.
    """

    request_id = _request_id(request_id)
    found: dict[bytes, Path] = {}
    for results in _epoch_dirs(Path(root), "results"):
        try:
            payload = (results / request_id / "result.json").read_bytes()
        except OSError:
            continue
        found.setdefault(payload, results)
    if len(found) > 1:
        raise RemoteRunForensicsError(
            f"request {request_id} has differing retained results in "
            + ", ".join(sorted(path.name for path in found.values()))
        )
    epoch = next(iter(found.values()), None)
    return None if epoch is None else epoch / request_id


def remote_runs(
    spool_roots: tuple[Path, ...], reservation_id: str, request_ids: set[str]
) -> list[dict[str, Any]]:
    """Join retained request carriers, transport events, results, and worker logs."""

    from cacheon.eval.explain import worker_log_lines

    runs: dict[str, dict[str, Any]] = {}
    for root in spool_roots:
        for outbox in _epoch_dirs(root, "outbox"):
            for carrier in outbox.glob("*/request.json"):
                try:
                    request = json.loads(carrier.read_text())
                    members = request["lease"]["members"]
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                request_id = request.get("request_id")
                if isinstance(request_id, str) and any(
                    row.get("reservation_id") == reservation_id for row in members
                ):
                    request_ids.add(request_id)
        event_index: dict[str, list[dict[str, Any]]] = {}
        try:
            event_lines = (root / "events.jsonl").read_text().splitlines()
        except OSError:
            event_lines = []
        for line in event_lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            request_id = event.get("request_id")
            if request_id in request_ids:
                event_index.setdefault(request_id, []).append(event)
        for request_id in sorted(request_ids):
            entry = runs.setdefault(request_id, {"events": [], "request_id": request_id})
            entry["events"].extend(event_index.get(request_id, []))
            result_root = result_dir_for_request(root, request_id)
            if result_root is None:
                continue
            try:
                result = json.loads((result_root / "result.json").read_text())
            except (OSError, ValueError):
                continue
            entry.update(
                failure_code=result.get("failure_code"), result_state=result.get("state")
            )
            if result_root.parent.name != "results":
                entry["epoch"] = result_root.parent.name
            try:
                worker_log = reopen_worker_log(result_root, result)
            except (OSError, ValueError, RemoteRunForensicsError) as exc:
                entry["worker_log_error"] = str(exc)
            else:
                if worker_log is None:
                    entry["worker_log_state"] = (
                        "not retained by the worker generation that ran this request"
                    )
                else:
                    artifact = next(
                        row for row in result["artifacts"] if row["role"] == "worker_log"
                    )
                    entry["worker_log"] = {
                        "explanation": worker_log_lines(worker_log),
                        "retention": worker_log_retention(worker_log),
                        "sha256": artifact["sha256"],
                        "size": artifact["size"],
                    }
    for entry in runs.values():
        entry["events"].sort(key=lambda row: (row.get("time_unix", 0), row.get("event", "")))
    return list(runs.values())


__all__ = [
    "RemoteRunForensicsError",
    "append_adapter_stream",
    "append_event",
    "bind_request",
    "capture_adapter_stream",
    "component_for_path",
    "exception_record",
    "journal_path",
    "publish_worker_log",
    "record_oci_artifact",
    "remote_runs",
    "reopen_worker_log",
    "result_dir_for_request",
    "worker_log_retention",
]

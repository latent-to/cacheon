"""One downloadable raw log assembled from a retained remote worker artifact."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cacheon.eval.remote_run_forensics import (
    RemoteRunForensicsError,
    reopen_worker_log,
    result_dir_for_request,
    worker_log_retention,
)


@dataclass(frozen=True)
class WorkerLogDownload:
    request_id: str
    worker_log_sha256: str
    retention: str
    payload: bytes
    sha256: str


def _worker_artifact(
    result_root: Path, result: Mapping[str, object]
) -> tuple[Mapping[str, object], bytes]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise RemoteRunForensicsError("remote result artifacts are malformed")
    matches = [
        row
        for row in artifacts
        if isinstance(row, dict) and row.get("role") == "worker_log"
    ]
    if len(matches) != 1:
        raise RemoteRunForensicsError("remote result worker log identity is ambiguous")
    row = matches[0]
    digest, size = row.get("sha256"), row.get("size")
    if not isinstance(digest, str) or type(size) is not int:
        raise RemoteRunForensicsError("remote result worker log identity is malformed")
    try:
        payload = (result_root / "blobs" / digest).read_bytes()
    except OSError as exc:
        raise RemoteRunForensicsError(f"worker log is unavailable: {exc}") from None
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise RemoteRunForensicsError("worker log differs from its result identity")
    return row, payload


def _stream(value: object, *, size: object, digest: object, label: str) -> bytes:
    if type(size) is not int or size < 0 or not isinstance(digest, str):
        raise RemoteRunForensicsError(f"{label} identity is malformed")
    try:
        payload = base64.b64decode(value, validate=True)
    except (TypeError, ValueError):
        raise RemoteRunForensicsError(f"{label} payload is malformed") from None
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise RemoteRunForensicsError(f"{label} payload differs from its identity")
    return payload


def _append(parts: list[bytes], heading: str, payload: bytes) -> None:
    parts.append(f"\n===== {heading} =====\n".encode("utf-8"))
    parts.append(payload)
    if payload and not payload.endswith(b"\n"):
        parts.append(b"\n")


def worker_log_download(spool_root: Path, request_id: str) -> WorkerLogDownload:
    """Return exact retained diagnostic bytes with plain section boundaries.

    The worker reserves OCI stdout for framed protocol and redirects ordinary
    Python/native stdout to stderr. Consequently the OCI diagnostic sections
    below contain both ordinary stdout prints and stderr diagnostics.
    """

    result_root = result_dir_for_request(Path(spool_root), request_id)
    if result_root is None:
        raise RemoteRunForensicsError("remote request result is not retained")
    try:
        result = json.loads((result_root / "result.json").read_bytes())
    except (OSError, ValueError) as exc:
        raise RemoteRunForensicsError(f"remote result is unreadable: {exc}") from None
    if not isinstance(result, dict) or result.get("request_id") != request_id:
        raise RemoteRunForensicsError("remote result request identity differs")
    worker_log = reopen_worker_log(result_root, result)
    if worker_log is None:
        raise RemoteRunForensicsError("worker log was not retained for this request")
    worker_row, _worker_raw = _worker_artifact(result_root, result)
    retention = worker_log_retention(worker_log)
    parts = [
        (
            "# Cacheon retained evaluation output\n"
            f"# request_id={request_id}\n"
            f"# worker_log_sha256={worker_row['sha256']}\n"
            f"# retention={retention}\n"
            "# OCI stdout is framed protocol; ordinary Python/native stdout is "
            "redirected into the diagnostic stream below.\n"
        ).encode("utf-8")
    ]
    stream_count = 0
    for index, event in enumerate(worker_log.get("events", [])):
        if not isinstance(event, dict) or event.get("phase") != "adapter.output":
            continue
        record = event.get("stream")
        if not isinstance(record, dict):
            raise RemoteRunForensicsError("adapter stderr record is malformed")
        payload = _stream(
            record.get("payload_base64"),
            size=record.get("bytes"),
            digest=record.get("sha256"),
            label="adapter stderr",
        )
        _append(
            parts,
            f"adapter stderr {index} bytes={len(payload)} sha256={record['sha256']}",
            payload,
        )
        stream_count += 1

    for index, stream in enumerate(worker_log.get("oci_streams", [])):
        if not isinstance(stream, dict) or not isinstance(stream.get("receipt"), dict):
            raise RemoteRunForensicsError("OCI diagnostic record is malformed")
        receipt: dict[str, Any] = stream["receipt"]
        retained = receipt.get("artifact_bytes")
        total = receipt.get("stream_bytes", retained)
        truncated = receipt.get("truncated", False)
        identity = (
            f"executor={receipt.get('executor_id')} lease={receipt.get('lease_id')} "
            f"retained_bytes={retained} stream_bytes={total} truncated={truncated} "
            f"artifact_sha256={receipt.get('artifact_sha256')} "
            f"stream_sha256={receipt.get('stream_sha256', receipt.get('artifact_sha256'))}"
        )
        if stream.get("state") == "not_retained" or "payload_base64" not in stream:
            _append(parts, f"engine stdout/stderr {index} NOT RETAINED {identity}", b"")
            continue
        payload = _stream(
            stream.get("payload_base64"),
            size=retained,
            digest=receipt.get("artifact_sha256"),
            label="OCI diagnostic stream",
        )
        _append(parts, f"engine stdout/stderr {index} {identity}", payload)
        stream_count += 1
    if stream_count == 0:
        parts.append(b"\n# No raw diagnostic stream was retained for this request.\n")
    payload = b"".join(parts)
    return WorkerLogDownload(
        request_id=request_id,
        worker_log_sha256=str(worker_row["sha256"]),
        retention=retention,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = ["WorkerLogDownload", "worker_log_download"]

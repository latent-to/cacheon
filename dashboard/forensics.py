"""Dashboard adapter for the validator's retained remote-run forensics."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from cacheon.chain.baseline_band import qualification_speed, qualification_speed_from_payload
from cacheon.chain.remote_qualification_evidence import RemoteEvidenceArtifact
from cacheon.eval.remote_run_download import worker_log_download
from cacheon.eval.remote_run_forensics import (
    RemoteRunForensicsError, remote_runs, result_dir_for_request,
)


class DashboardForensicsError(RuntimeError):
    """A submission's retained forensics cannot be shown or downloaded."""


class ForensicsNotFound(DashboardForensicsError):
    """The requested run is not bound to the selected submission."""


class ForensicsUnavailable(DashboardForensicsError):
    """The selected run predates or has not produced a worker log."""


@dataclass(frozen=True)
class ForensicsLog:
    payload: bytes
    filename: str
    etag: str
    retention: str


def submission_forensics(
    spool_root: Path, reservation_id: str, *, target_id: str = ""
) -> list[dict[str, object]]:
    """Render the same request explanation as ``chain-miner-report`` for the UI."""

    try:
        runs = remote_runs((Path(spool_root),), reservation_id, set())
    except RemoteRunForensicsError as exc:
        raise DashboardForensicsError(str(exc)) from None
    items: list[dict[str, object]] = []
    for run in runs:
        request_id = run["request_id"]
        item: dict[str, object] = {
            "failure_code": run.get("failure_code"),
            "request_id": request_id,
            "result_state": run.get("result_state"),
        }
        worker_log = run.get("worker_log")
        if isinstance(worker_log, dict):
            item["worker_log"] = {
                "download_url": (
                    f"/api/submissions/{reservation_id}/forensics/{request_id}.log"
                ),
                "explanation": worker_log["explanation"],
                "retention": worker_log["retention"],
                "sha256": worker_log["sha256"],
                "size": worker_log["size"],
            }
        elif run.get("worker_log_error"):
            item["worker_log_state"] = f"unreadable: {run['worker_log_error']}"
        elif run.get("worker_log_state"):
            item["worker_log_state"] = run["worker_log_state"]
        elif run.get("events"):
            item["worker_log_state"] = "result not retained yet"
        if target_id:
            try:
                retained = _retained_qualification(
                    spool_root, request_id, reservation_id, target_id)
                key = "qualification_hold" if retained and retained.get("decision") == "HOLD" else "qualification"
                item[key] = retained
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                item["qualification_error"] = str(exc)
        items.append(item)
    return items


def _retained_qualification(
    spool_root: Path, request_id: str, reservation_id: str, target_id: str
) -> dict[str, object] | None:
    """Read the completed result already bound to this reservation by remote_runs."""
    root = result_dir_for_request(spool_root, request_id)
    if root is None:
        return None
    result = json.loads((root / "result.json").read_bytes())
    if result.get("state") != "completed" or not result.get("response_sha256"):
        return None
    artifact = next((row for row in result["artifacts"] if row["role"] == "adapter_result"), None)
    if artifact is None:
        raise DashboardForensicsError("qualification result lacks its response artifact")
    raw = (root / "blobs" / artifact["sha256"]).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != result["response_sha256"]
            or result["response_sha256"] != artifact["sha256"]
            or len(raw) != artifact["size"]):
        raise DashboardForensicsError("qualification response differs from retained result")
    response = json.loads(raw)
    if response.get("payload_kind") == "remote_qualification_hold":
        payload = response["payload"]
        if reservation_id not in payload["reservation_digests"]:
            raise DashboardForensicsError("qualification HOLD names another reservation")
        return {"decision": "HOLD", "reason": payload["reason"],
                "failure_type": payload.get("failure_type", ""),
                "failure_message": payload.get("failure_message", "")}
    if response.get("payload_kind") != "remote_qualification_product":
        return None
    payload = response["payload"]
    batch = payload["batch"]
    outcome = next((row for row in batch["outcomes"]
                    if row["reservation_digest"] == reservation_id), None)
    if outcome is None:
        return None
    reference = batch["attempt_ref"]
    retained = next((row for row in payload["evidence"]
                     if row["reference"] == reference), None)
    if retained is None:
        raise DashboardForensicsError("qualification result lacks its retained attempt")
    evidence = RemoteEvidenceArtifact.from_dict(retained)
    return {
        "request_id": request_id,
        "artifact_sha256": evidence.reference.sha256,
        "decision": outcome["decision"],
        "reason": outcome["reason"],
        "speed": qualification_speed_from_payload(evidence.payload, target_id),
    }


def submission_qualifications(
    connection: sqlite3.Connection, reservation_id: str, target_id: str,
    evidence_roots: tuple[Path, ...], forensics: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Include completed held attempts absent from the qualification disposition table."""
    attempts = [
        {"attempt": row["attempt_index"], "decision": row["decision"],
         "reason": row["reason"],
         "artifact_sha256": json.loads(row["attempt_ref_json"] or "{}").get("sha256"),
         "speed": qualification_speed(row["attempt_ref_json"], evidence_roots, target_id)}
        for row in connection.execute(
            "SELECT attempt_index, decision, reason, attempt_ref_json "
            "FROM qualification_dispositions WHERE reservation_id=? ORDER BY attempt_index",
            (reservation_id,))
    ]
    seen = {row["artifact_sha256"]: row for row in attempts}
    for run in forensics:
        retained = run.get("qualification")
        if not isinstance(retained, dict):
            continue
        existing = seen.get(retained["artifact_sha256"])
        if existing is not None:
            if existing["speed"] is None:
                existing["speed"] = retained["speed"]
        else:
            item = {**retained, "attempt": max(
                (row["attempt"] for row in attempts), default=-1) + 1}
            attempts.append(item)
            seen[retained["artifact_sha256"]] = item
    return attempts


def forensics_log(
    spool_root: Path, reservation_id: str, request_id: str
) -> ForensicsLog:
    """Return raw diagnostic text only after proving request/submission binding."""

    runs = submission_forensics(spool_root, reservation_id)
    run = next((row for row in runs if row["request_id"] == request_id), None)
    if run is None:
        raise ForensicsNotFound("request does not belong to this submission")
    if not isinstance(run.get("worker_log"), dict):
        raise ForensicsUnavailable(
            str(run.get("worker_log_state") or "worker log is not retained")
        )
    try:
        download = worker_log_download(Path(spool_root), request_id)
    except RemoteRunForensicsError as exc:
        raise DashboardForensicsError(str(exc)) from None
    return ForensicsLog(
        payload=download.payload,
        filename=f"cacheon-evaluation-{reservation_id[:12]}-{request_id[:12]}.log",
        etag=download.sha256,
        retention=download.retention,
    )


__all__ = [
    "DashboardForensicsError",
    "ForensicsLog",
    "ForensicsNotFound",
    "ForensicsUnavailable",
    "forensics_log",
    "submission_forensics",
    "submission_qualifications",
]

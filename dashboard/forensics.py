"""Dashboard adapter for the validator's retained remote-run forensics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cacheon.eval.remote_run_download import worker_log_download
from cacheon.eval.remote_run_forensics import RemoteRunForensicsError, remote_runs


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


def submission_forensics(spool_root: Path, reservation_id: str) -> list[dict[str, object]]:
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
        items.append(item)
    return items


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
]

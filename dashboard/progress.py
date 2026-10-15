"""Display completed work from the active worker's existing CPU-held checkpoint."""

from functools import lru_cache
import json
from pathlib import Path
import re
import threading
import time

from cacheon.eval.qualification_worker_checkpoint import read_snapshot
from dashboard.sources import value, worker_heartbeats

_READ_LOCK = threading.Lock()


@lru_cache(maxsize=16)
def _checkpoint_summary(path, digest, request):
    snapshot = read_snapshot(path, digest)
    if snapshot["origin"]["request"] != request:
        raise ValueError("checkpoint request differs")
    stages = snapshot["stages"]
    # Retain only public counters, never the sealed plan or the large turn records.
    return {"completed": [name for name in ("speed", "audit", "quality") if name in stages],
            "replay_windows_completed": sum(name.startswith("speed_window_") for name in stages),
            "first_orientation_completed": "speed_first" in stages}


def active_progress(spool: Path, heartbeat: dict, now: int) -> dict:
    """Show observed completion milestones, not a guessed percentage or time remaining."""
    result = {"worker_state": heartbeat.get("state", "unknown"),
              "heartbeat_age_s": heartbeat.get("age_s"), "completed": []}
    if not heartbeat.get("fresh"):
        return {**result, "state": "unavailable", "label": "Worker status is stale or unavailable"}
    request = heartbeat.get("active_request_id")
    if not request:
        return {**result, "state": "idle", "label": "No active worker request"}
    result.update(state="active", request_id=request, label="Waiting for the first completed checkpoint")
    try:
        matches = []
        for path in (spool / "checkpoints" / "index").glob("*.json"):
            row = json.loads(path.read_text())
            if not isinstance(row, dict):
                raise ValueError("checkpoint index is not an object")
            if row.get("request_id") == request and row.get("worker_epoch") == heartbeat.get("worker_epoch"):
                matches.append((path, row))
        if not matches:
            return result
        if len(matches) != 1:
            raise ValueError("multiple checkpoints for the active request")
        path, row = matches[0]
        digest = row["sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("invalid checkpoint digest")
        # One reader per new immutable digest; concurrent page requests reuse its summary.
        with _READ_LOCK:
            summary = _checkpoint_summary(spool / "checkpoints" / (digest + ".json.gz"),
                                          digest, row["request_digest"])
        completed = summary["completed"]
        label = ("Evaluation checkpoints complete; waiting for the final result" if "quality" in completed else
                 "Audit checkpoint complete; pristine check pending" if "audit" in completed else
                 "Speed replay complete; audit checkpoint pending" if "speed" in completed else
                 "Speed replay in progress")
        return {**result, **summary, "label": label,
                "checkpoint_age_s": max(0, now - int(path.stat().st_mtime))}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return {**result, "state": "unavailable", "label": "Run is active; progress checkpoint is unreadable"}


def install_progress(app, api):
    """Keep a first checkpoint read off the page's main data request."""
    @app.get("/api/progress")
    def progress():
        now = int(time.time())
        registration = api["load_json"](value("REGISTRATION_PATH", api["REGISTRATION_PATH"]))
        heartbeat = worker_heartbeats(value("HEARTBEAT_PATH", api["HEARTBEAT_PATH"]),
                                     registration, now)["gpu_heartbeat"]
        return active_progress(value("SPOOL", api["SPOOL"]), heartbeat, now)

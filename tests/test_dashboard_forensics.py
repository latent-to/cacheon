from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from cacheon.eval.remote_run_forensics import append_event, journal_path, publish_worker_log
from dashboard.forensics import ForensicsNotFound, forensics_log, submission_forensics


def _retained_run(spool: Path, reservation_id: str, request_id: str) -> None:
    carrier = spool / "outbox" / f"{1:020d}-{request_id}"
    carrier.mkdir(parents=True)
    (carrier / "request.json").write_text(
        json.dumps(
            {
                "lease": {"members": [{"reservation_id": reservation_id}]},
                "request_id": request_id,
            }
        )
    )
    result = spool / "results" / request_id
    result.mkdir(parents=True)
    raw = b"miner diagnostic output\nRuntimeError: broken kernel\n"
    append_event(journal_path(result), request_id, "pod.adapter", "started")
    append_event(
        journal_path(result),
        request_id,
        "adapter.output",
        "retained",
        stream={
            "bytes": len(raw),
            "payload_base64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    append_event(
        journal_path(result),
        request_id,
        "adapter.terminal",
        "failed",
        failure_code="candidate_exception",
    )
    artifact = publish_worker_log(result, request_id)
    (result / "result.json").write_text(
        json.dumps(
            {
                "artifacts": [artifact],
                "failure_code": "candidate_exception",
                "request_id": request_id,
                "state": "completed",
            }
        )
    )


def test_dashboard_keeps_the_summary_concise_and_downloads_raw_output(
    tmp_path: Path,
) -> None:
    reservation_id, request_id = "7" * 64, "8" * 64
    spool = tmp_path / "spool"
    _retained_run(spool, reservation_id, request_id)

    (run,) = submission_forensics(spool, reservation_id)
    worker_log = run["worker_log"]
    assert worker_log["download_url"].endswith(f"/{request_id}.log")
    assert "miner diagnostic output" not in "\n".join(worker_log["explanation"])

    download = forensics_log(spool, reservation_id, request_id)
    assert download.filename.endswith(".log")
    assert download.retention == "complete"
    assert b"miner diagnostic output\nRuntimeError: broken kernel\n" in download.payload
    with pytest.raises(ForensicsNotFound, match="does not belong"):
        forensics_log(spool, reservation_id, "9" * 64)


def test_submission_drawer_offers_the_raw_log_beside_the_existing_explanation() -> None:
    html = (Path(__file__).parents[1] / "dashboard" / "static" / "index.html").read_text()
    assert "What happened in evaluation" in html
    assert "Download ${complete ? \"full\" : \"retained\"} raw output (.log)" in html
    assert "log.explanation" in html

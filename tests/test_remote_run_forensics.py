from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from cacheon.eval.explain import worker_log_lines
from cacheon.eval.remote_run_download import worker_log_download
from cacheon.eval.remote_run_forensics import (
    RemoteRunForensicsError,
    append_event,
    bind_request,
    journal_path,
    publish_worker_log,
    record_oci_artifact,
    remote_runs,
    reopen_worker_log,
)


class _Receipt:
    def __init__(self, value: dict, receipt_sha256: str) -> None:
        self.value = value
        self.receipt_sha256 = receipt_sha256

    def to_dict(self) -> dict:
        return self.value


def test_request_artifact_embeds_and_explains_exact_oci_output(tmp_path: Path) -> None:
    request_id = "1" * 64
    result = tmp_path / "result"
    result.mkdir()
    output = tmp_path / "runtime.stderr"
    output.write_text(
        "miner stdout\nTraceback (most recent call last):\n"
        '  File "/usr/local/lib/python3.12/dist-packages/sglang/runtime.py", '
        "line 77, in capture\n    run()\nRuntimeError: graph capture failed\n"
    )
    receipt_path = Path(str(output) + ".json")
    receipt = {
        "artifact_bytes": output.stat().st_size,
        "artifact_path": str(output),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "executor_id": "qualification-lane",
        "lease_id": "runtime-1",
        "receipt_path": str(receipt_path),
    }
    receipt_raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    receipt_path.write_bytes(receipt_raw)
    append_event(journal_path(result), request_id, "pod.adapter", "started")
    with bind_request(result, request_id):
        record_oci_artifact(_Receipt(receipt, hashlib.sha256(receipt_raw).hexdigest()))
    append_event(journal_path(result), request_id, "adapter.terminal", "failed")
    artifact = publish_worker_log(result, request_id)
    assert publish_worker_log(result, request_id) == artifact
    assert journal_path(result).is_file()
    result_row = {"artifacts": [artifact], "request_id": request_id}

    worker_log = reopen_worker_log(result, result_row)
    assert worker_log is not None
    assert worker_log["oci_streams"][0]["receipt"]["artifact_sha256"] == receipt[
        "artifact_sha256"
    ]
    rendered = "\n".join(worker_log_lines(worker_log))
    assert "OCI OUTPUT" in rendered and "miner stdout" not in rendered
    assert "sglang: RuntimeError: graph capture failed" in rendered
    assert "sglang/runtime.py:77 in capture" in rendered


def _linked_oci_run(root: Path, request_id: str, body: str) -> dict[str, Path]:
    """Journal one request whose OCI output lives outside its result root."""

    root.mkdir(parents=True)
    output = root.parent / f"{request_id[:8]}.stderr"
    output.write_text(body)
    receipt_path = Path(str(output) + ".json")
    receipt = {
        "artifact_bytes": output.stat().st_size,
        "artifact_path": str(output),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "executor_id": "screen-lane",
        "lease_id": "runtime-9",
        "receipt_path": str(receipt_path),
    }
    receipt_raw = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    receipt_path.write_bytes(receipt_raw)
    append_event(journal_path(root), request_id, "pod.adapter", "started")
    with bind_request(root, request_id):
        record_oci_artifact(
            _Receipt(receipt, hashlib.sha256(receipt_raw).hexdigest())
        )
    append_event(journal_path(root), request_id, "adapter.terminal", "failed")
    return {"output": output, "receipt_path": receipt_path}


def test_discarded_runtime_stream_is_stated_but_changed_bytes_stay_fatal(
    tmp_path: Path,
) -> None:
    """A removed --rm tree must not prevent sealing the request's result."""

    request_id = "c" * 64
    result = tmp_path / "discarded"
    linked = _linked_oci_run(result, request_id, "resident load failed\n")
    linked["output"].unlink()
    linked["receipt_path"].unlink()

    artifact = publish_worker_log(result, request_id)
    worker_log = reopen_worker_log(
        result, {"artifacts": [artifact], "request_id": request_id}
    )
    assert worker_log is not None
    stream = worker_log["oci_streams"][0]
    assert stream["state"] == "not_retained"
    assert "payload_base64" not in stream

    changed = tmp_path / "changed"
    other = "d" * 64
    linked = _linked_oci_run(changed, other, "resident load failed\n")
    linked["output"].write_text("something else entirely\n")
    with pytest.raises(RemoteRunForensicsError, match="differs from its receipt"):
        publish_worker_log(changed, other)


def _carrier(outbox: Path, request_id: str, reservation_id: str) -> None:
    root = outbox / f"{1:020d}-{request_id}"
    root.mkdir(parents=True)
    (root / "request.json").write_text(
        json.dumps(
            {
                "lease": {"members": [{"reservation_id": reservation_id}]},
                "request_id": request_id,
            }
        )
    )


def test_download_contains_exact_redirected_stdout_and_adapter_diagnostics(
    tmp_path: Path,
) -> None:
    spool, request_id = tmp_path / "spool", "e" * 64
    result = spool / "results" / request_id
    _linked_oci_run(result, request_id, "miner stdout\nRuntimeError: candidate boom\n")
    adapter = b"adapter started request\nadapter finished request\n"
    append_event(
        journal_path(result),
        request_id,
        "adapter.output",
        "retained",
        stream={
            "bytes": len(adapter),
            "payload_base64": base64.b64encode(adapter).decode("ascii"),
            "sha256": hashlib.sha256(adapter).hexdigest(),
        },
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

    download = worker_log_download(spool, request_id)
    assert download.retention == "complete"
    assert b"ordinary Python/native stdout is redirected" in download.payload
    assert b"miner stdout\nRuntimeError: candidate boom\n" in download.payload
    assert adapter in download.payload
    assert hashlib.sha256(download.payload).hexdigest() == download.sha256


def test_remote_runs_reopens_retired_epochs_and_refuses_differing_results(
    tmp_path: Path,
) -> None:
    spool = tmp_path / "spool"
    reservation_id, request_id = "9" * 64, "a" * 64
    epoch = "retired-epoch-1"
    _carrier(spool / f"outbox-{epoch}", request_id, reservation_id)
    result = spool / f"results-{epoch}" / request_id
    result.mkdir(parents=True)
    append_event(journal_path(result), request_id, "adapter.terminal", "failed")
    artifact = publish_worker_log(result, request_id)
    result_row = {
        "artifacts": [artifact],
        "failure_code": "adapter_epoch_failed",
        "request_id": request_id,
        "state": "no_decision",
    }
    (result / "result.json").write_text(json.dumps(result_row, sort_keys=True))

    (run,) = remote_runs((spool,), reservation_id, set())
    assert run["epoch"] == f"results-{epoch}"
    assert run["worker_log"]["retention"] == "complete"

    conflicting = spool / "results-retired-epoch-2" / request_id
    conflicting.mkdir(parents=True)
    (conflicting / "result.json").write_text(
        json.dumps({**result_row, "state": "completed"}, sort_keys=True)
    )
    with pytest.raises(RemoteRunForensicsError, match="differing retained results"):
        remote_runs((spool,), reservation_id, set())

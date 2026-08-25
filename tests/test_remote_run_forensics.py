from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cacheon.eval.explain import worker_log_lines
from cacheon.eval.remote_run_forensics import (
    append_event,
    bind_request,
    journal_path,
    publish_worker_log,
    record_oci_artifact,
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

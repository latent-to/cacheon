from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cacheon.eval.explain import worker_log_lines
from cacheon.eval.remote_run_forensics import (
    RemoteRunForensicsError,
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

"""Run progress exposes only completed checkpoints belonging to the observed worker."""

import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from cacheon.eval.qualification_worker_checkpoint import SCHEMA
from cacheon.stack_identity import canonical_json_bytes
from dashboard import progress


@pytest.fixture
def checkpoint(tmp_path):
    progress._checkpoint_summary.cache_clear()
    root = tmp_path / "checkpoints"
    (root / "index").mkdir(parents=True)
    heartbeat = {"state": "evaluating", "fresh": True, "age_s": 4,
                 "active_request_id": "a" * 64, "worker_epoch": "current"}
    def publish(stages, *, request="b" * 64, worker="current"):
        snapshot = {"schema": SCHEMA, "origin": {"request": request, "selection_secret": "PRIVATE"},
                    "stages": {stage: {"records": "PRIVATE"} for stage in stages}}
        data = gzip.compress(canonical_json_bytes(snapshot))
        digest = hashlib.sha256(data).hexdigest()
        (root / (digest + ".json.gz")).write_bytes(data)
        pointer = {"request_id": "a" * 64, "worker_epoch": worker,
                   "sha256": digest, "request_digest": "b" * 64}
        (root / "index" / "work.json").write_text(json.dumps(pointer))
        return root / (digest + ".json.gz")
    return tmp_path, heartbeat, publish


@pytest.mark.parametrize("stages,label", [
    (["speed_window_1", "speed_window_2"], "Speed replay in progress"),
    (["speed"], "Speed replay complete; audit checkpoint pending"),
    (["speed", "audit"], "Audit checkpoint complete; pristine check pending"),
    (["speed", "audit", "quality"], "Evaluation checkpoints complete; waiting for the final result"),
])
def test_completed_work_and_no_private_payload(checkpoint, stages, label):
    spool, heartbeat, publish = checkpoint
    publish(stages)
    value = progress.active_progress(spool, heartbeat, 100)
    assert value["state"] == "active" and value["label"] == label
    assert value["completed"] == [stage for stage in stages if not stage.startswith("speed_window_")]
    assert value["replay_windows_completed"] == sum(stage.startswith("speed_window_") for stage in stages)
    assert "PRIVATE" not in json.dumps(value) and "selection_secret" not in json.dumps(value)
    assert "percent" not in value


def test_stale_idle_and_different_worker_do_not_claim_live_progress(checkpoint):
    spool, heartbeat, publish = checkpoint
    publish(["speed"], worker="previous")
    assert progress.active_progress(spool, heartbeat, 100)["completed"] == []
    publish(["speed"])
    assert progress.active_progress(spool, {**heartbeat, "fresh": False}, 100)["state"] == "unavailable"
    assert progress.active_progress(spool, {**heartbeat, "active_request_id": None}, 100)["state"] == "idle"
    assert progress.active_progress(spool, {**heartbeat, "active_request_id": "c" * 64}, 100)["completed"] == []


@pytest.mark.parametrize("bad", ["request", "archive", "digest"])
def test_unreadable_checkpoint_is_explicit(checkpoint, bad):
    spool, heartbeat, publish = checkpoint
    archive = publish(["speed"], request="wrong" if bad == "request" else "b" * 64)
    if bad == "archive":
        archive.write_bytes(b"corrupt")
    if bad == "digest":
        pointer = spool / "checkpoints/index/work.json"
        row = json.loads(pointer.read_text())
        row["sha256"] = "../private"
        pointer.write_text(json.dumps(row))
    value = progress.active_progress(spool, heartbeat, 100)
    assert value["state"] == "unavailable" and "unreadable" in value["label"]
    assert value["completed"] == []


def test_concurrent_views_read_one_archive_and_new_checkpoint_advances(checkpoint, monkeypatch):
    spool, heartbeat, publish = checkpoint
    publish(["speed_window_1"])
    reads = []
    read = progress.read_snapshot
    def counted(path, digest):
        reads.append(digest)
        return read(path, digest)
    monkeypatch.setattr(progress, "read_snapshot", counted)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: progress.active_progress(spool, heartbeat, 100), range(16)))
    assert all(value["replay_windows_completed"] == 1 for value in values)
    assert len(reads) == 1
    publish(["speed", "audit"])
    assert progress.active_progress(spool, heartbeat, 100)["completed"] == ["speed", "audit"]
    assert len(reads) == 2

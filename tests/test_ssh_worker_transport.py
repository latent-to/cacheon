from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import nullcontext
from pathlib import Path

import pytest

from cacheon.chain import ssh_worker_transport as transport


@pytest.mark.parametrize("stage", ["qualification", "screen"])
def test_result_deadline_distinguishes_qualification_wait_from_screen_failure(tmp_path, monkeypatch, stage):
    from tests.test_remote_worker_request_plan import _authority

    worker = _authority(tmp_path).transport()
    worker.response_timeout_seconds = 1
    ticks = iter((0, 2))
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(ticks))
    with pytest.raises(transport.RemoteEvaluationDispatcherError) as caught:
        worker._await_completed_result("request", tmp_path / "job", stage=stage)
    if stage == "qualification":
        assert type(caught.value) is transport.RemoteQualificationWaitTimeout
        assert caught.value.request_id == "request"
    else:
        assert type(caught.value) is transport.RemoteEvaluationDispatcherError


@pytest.mark.parametrize("copy_operation", ["transfer_request", "pull_result"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_cpu_relay_heartbeat_survives_long_copy_and_stops_on_exit(
    tmp_path, monkeypatch, copy_operation, interrupted,
):
    registration = {
        "worker_epoch": "a" * 32, "ready_receipt_digest": "b" * 64,
        "worker_readiness_digest": "c" * 64, "credential_path": "unused",
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration))
    monkeypatch.setattr(transport, "verify_registration", lambda row: row)
    monkeypatch.setattr(transport, "registration_transport_identity", lambda _row: None)
    monkeypatch.setattr(transport, "registration_credential", lambda *_args: None)
    current = iter([True, False])
    monkeypatch.setattr(transport, "registration_is_current", lambda *_args: next(current))
    monkeypatch.setattr(transport, "DEFAULT_HEARTBEAT_SECONDS", 0.01)
    now = [1_000]
    monkeypatch.setattr(transport.time, "time", lambda: now[0])
    request_id = "d" * 64
    request = {"request_id": request_id, "deadline_unix": 2_000}
    job = tmp_path / "job"
    job.mkdir()
    monkeypatch.setattr(transport, "iter_queue", lambda *_args, **_kwargs: [(job, request)])
    monkeypatch.setattr(
        transport, "remote_heartbeat",
        lambda *_args: {"state": "idle", "active_request_id": None},
    )
    monkeypatch.setattr(transport, "pull_result", lambda *_args, **_kwargs: False)
    heartbeat_path = tmp_path / "state" / "heartbeat.json"
    heartbeat_path.parent.mkdir()
    updated = threading.Event()
    original_write = transport.atomic_json

    def write_heartbeat(path, payload, **kwargs):
        original_write(path, payload, **kwargs)
        if path == heartbeat_path:
            updated.set()

    monkeypatch.setattr(transport, "atomic_json", write_heartbeat)
    transport.atomic_json(heartbeat_path, transport.heartbeat_payload(registration, "running", None))
    observed = []

    def slow_copy(*_args, **_kwargs):
        for _ in range(8):
            updated.clear()
            now[0] += 10
            assert updated.wait(1), "relay stopped heartbeating during the copy"
            heartbeat = transport.verify_heartbeat(
                transport.load_json(heartbeat_path), registration, 20
            )
            observed.append(heartbeat["time_unix"])
        if interrupted:
            raise RuntimeError("copy interrupted")
        return copy_operation == "pull_result"

    monkeypatch.setattr(transport, copy_operation, slow_copy)
    with pytest.raises(RuntimeError, match="copy interrupted") if interrupted else nullcontext():
        transport.cpu_serve(
            registration_path=registration_path, current_registration_path=registration_path,
            spool_root=tmp_path, site=transport.RemotePodSite("/pod", "/pod/service.py"),
            poll_seconds=0,
        )
    assert len(observed) == 8
    assert max(observed) - min(observed) >= 60
    after_exit = heartbeat_path.read_bytes()
    updated.clear()
    now[0] += 100
    assert not updated.wait(0.03)
    assert heartbeat_path.read_bytes() == after_exit


@pytest.mark.parametrize("status,active,ready,archived", [
    ("failed", 0, True, True), ("expired", 0, True, True),
    ("qualified", 0, True, True), ("held", 0, True, False),
    ("no_decision", 0, True, False), ("qualifying", 1, True, False),
    ("failed", 1, True, False), ("failed", 0, False, False),
    (None, 0, True, False),
])
def test_archive_only_terminal_unleased_results(tmp_path, status, active, ready, archived):
    db = tmp_path / "intake.sqlite3"
    with sqlite3.connect(db) as con:
        con.executescript("CREATE TABLE reservations(reservation_id, status); "
                          "CREATE TABLE evaluation_lease_members(reservation_id, active);")
        if status:
            con.execute("INSERT INTO reservations VALUES('r', ?)", (status,))
        con.execute("INSERT INTO evaluation_lease_members VALUES('r', ?)", (active,))
    job = tmp_path / "outbox" / "123-request"
    job.mkdir(parents=True)
    request = {"request_id": "request", "lease": {"members": [{"reservation_id": "r"}]}}
    payload = json.dumps(request)
    (job / "request.json").write_text(payload)
    result = tmp_path / "results" / "request"
    result.mkdir(parents=True)
    if ready:
        (result / "RESULT_READY").touch()
    transport._archive_completed_request(job, request, str(db))
    assert job.exists() is not archived
    assert result.exists()
    if archived:
        from cacheon.eval.remote_run_forensics import _epoch_dirs
        archive = tmp_path / "outbox-archive"
        assert (archive / job.name / "request.json").read_text() == payload
        assert archive in _epoch_dirs(tmp_path, "outbox")
        assert '"request_archived"' in (tmp_path / "events.jsonl").read_text()


def _heartbeat(state: str, failures: int) -> dict[str, object]:
    return {
        "state": state,
        "adapter_start_count": 1,
        "consecutive_adapter_failures": failures,
    }


def test_worker_hold_states_never_exit_and_event_once_per_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        transport,
        "append_event",
        lambda _root, event, **fields: events.append((event, fields)),
    )
    registration = {"worker_epoch": "b" * 32}

    state = transport._observe_worker_hold(
        None, _heartbeat("epoch_failed", 3), tmp_path, registration
    )
    assert state == "epoch_failed"
    state = transport._observe_worker_hold(
        state, _heartbeat("epoch_failed", 3), tmp_path, registration
    )
    assert state == "epoch_failed"
    assert len(events) == 1

    state = transport._observe_worker_hold(
        state, _heartbeat("adapter_cooldown", 3), tmp_path, registration
    )
    assert state == "adapter_cooldown"
    assert len(events) == 2

    state = transport._observe_worker_hold(
        state, _heartbeat("idle", 0), tmp_path, registration
    )
    assert state is None
    assert len(events) == 2

    # A fresh burst after recovery is evented again.
    state = transport._observe_worker_hold(
        state, _heartbeat("adapter_cooldown", 3), tmp_path, registration
    )
    assert state == "adapter_cooldown"
    assert len(events) == 3

    assert all(event == "dispatcher_worker_cooldown" for event, _fields in events)
    assert events[0][1]["worker_state"] == "epoch_failed"
    assert events[1][1]["worker_state"] == "adapter_cooldown"
    assert all(
        fields["worker_epoch"] == registration["worker_epoch"]
        for _event, fields in events
    )

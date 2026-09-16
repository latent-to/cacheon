from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from cacheon.chain import ssh_worker_transport as transport


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

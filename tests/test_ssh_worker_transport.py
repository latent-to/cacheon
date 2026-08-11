from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.chain import ssh_worker_transport as transport


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

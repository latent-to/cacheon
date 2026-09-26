"""Stopping a CPU waiter releases its renewal thread without replacing GPU work."""

import threading

import pytest

from cacheon.chain.intake import IntakeError
from cacheon.chain.qualification_wait import _RecoveryHeartbeat
from cacheon.chain.recoverable_qualification_dispatcher import RecoverableQualificationDispatcherError
from tests.test_recoverable_qualification_dispatcher import (
    _Transport, _authority_for, _dispatcher, _fixtures, _store,
)


def _pending(tmp_path, profile):
    fixtures = _fixtures()
    authority = _authority_for(fixtures, tmp_path, profile)
    transport = _Transport(authority, fixtures, fail_resume=True)
    dispatcher = _dispatcher(authority, transport)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        dispatcher.dispatch_once()
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
    return authority, transport, dispatcher, recovery


@pytest.mark.parametrize("profile", ["collective-stop", "block-stop"])
def test_stop_cancels_contended_renewal_and_preserves_same_request(tmp_path, profile):
    authority, transport, dispatcher, recovery = _pending(tmp_path, profile)
    coordinator = authority.coordinator
    coordinator.heartbeat_interval_s = 0.001
    coordinator.heartbeat_join_timeout_s = 0.5
    coordinator.lock_retry_delay_s = 0.01
    attempted = threading.Event()
    release = threading.Event()
    original = coordinator._store_factory

    def contended(*args, **kwargs):
        if not release.is_set():
            attempted.set()
            raise IntakeError("another intake controller owns this database")
        return original(*args, **kwargs)

    coordinator._store_factory = contended
    heartbeat = _RecoveryHeartbeat(dispatcher, recovery)
    heartbeat.start()
    try:
        assert attempted.wait(2)
        latest, error = heartbeat.stop()
        stopped = not heartbeat._thread.is_alive()
    finally:
        release.set()
        heartbeat.stop()
        coordinator._store_factory = original
    assert stopped
    assert error is None
    assert latest == recovery
    with _store(authority) as store:
        assert store.pending_qualification_recovery() == recovery
    authority.coordinator.heartbeat_interval_s = 10
    transport.plan_fixtures._write_completed_result(authority, transport.plan)
    transport.fail_resume = False
    result = dispatcher.dispatch_once()
    assert result.disposition == "completed"
    assert (transport.plans, transport.materializations, transport.publications) == (1, 1, 1)


@pytest.mark.parametrize("profile", ["collective-error", "block-error"])
def test_stop_keeps_real_renewal_error(tmp_path, profile):
    authority, _transport, dispatcher, recovery = _pending(tmp_path, profile)
    coordinator = authority.coordinator
    coordinator.heartbeat_interval_s = 0.001
    attempted = threading.Event()
    original = coordinator._store_factory

    def broken(*args, **kwargs):
        attempted.set()
        raise IntakeError("database is corrupt")

    coordinator._store_factory = broken
    heartbeat = _RecoveryHeartbeat(dispatcher, recovery)
    heartbeat.start()
    try:
        assert attempted.wait(2)
        heartbeat._thread.join(2)
        latest, error = heartbeat.stop()
    finally:
        coordinator._store_factory = original
        heartbeat.stop()
    assert not heartbeat._thread.is_alive()
    assert "database is corrupt" in str(error)
    assert latest == recovery

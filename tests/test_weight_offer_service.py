"""Weight-offer producer loop accounting and publication-record chronology.

The sealed config, its refusals, the busy skip before a reward checkpoint, the failure breaker
and the pushed/idle reports run end to end in ``tests/test_validator_weight_service.py``, and its
static arena allocation in ``tests/test_validator_arena_allocation.py``.
"""

from __future__ import annotations

import threading

import pytest

from cacheon.chain.weight_offer_service import WeightOfferBusyError, run_forever


def test_busy_database_is_a_skipped_pass_not_a_failure() -> None:
    stop = threading.Event()
    calls: list[int] = []
    events: list[str] = []

    def publish() -> object:
        calls.append(1)
        if len(calls) >= 3:
            stop.set()
        raise WeightOfferBusyError("busy")

    # max_consecutive_failures=2 would trip on the second pass if a busy
    # database were counted as a failure.
    run_forever(
        publish,
        stop,
        poll_s=0.0,
        max_consecutive_failures=2,
        wait=lambda _s: stop.is_set(),
        on_event=lambda kind, _detail: events.append(kind),
    )
    assert calls == [1, 1, 1]
    assert events == ["busy", "busy", "busy"]


def test_a_successful_pass_resets_the_failure_count() -> None:
    stop = threading.Event()
    events: list[str] = []
    script: list[object] = [
        RuntimeError("a"),
        RuntimeError("b"),
        None,
        RuntimeError("c"),
        RuntimeError("d"),
    ]

    def publish() -> object:
        if not script:
            stop.set()
            return None
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return None

    run_forever(
        publish,
        stop,
        poll_s=0.0,
        restart_initial_backoff_s=0.0,
        restart_max_backoff_s=0.0,
        max_consecutive_failures=3,
        wait=lambda _s: stop.is_set(),
        on_event=lambda kind, _detail: events.append(kind),
    )
    assert events == ["failed", "failed", "idle", "failed", "failed", "idle"]


@pytest.mark.parametrize("reason,last_update,confirmed,valid", [
    ("block_inclusion", 0, 104, True),
    ("block_inclusion", 0, 99, False),
    ("block_inclusion", 105, 104, False),
    ("post_submit_authoritative_readback", 0, 104, False),
    ("post_submit_authoritative_readback", 102, 104, True),
])
def test_reopen_included_weight_record_before_active_readback(reason, last_update, confirmed, valid):
    from cacheon.chain.weights import WeightPublicationError, WeightPublicationRecord
    from cacheon.stack_identity import canonical_digest

    wire = dict(projection_digest="a" * 64, status="confirmed", prior_record_digest=None,
                submit_block=100, retry_after_block=700, reveal_round=23,
                confirmed_block=confirmed, confirmed_last_update=last_update, reason=reason)
    if not valid:
        with pytest.raises(WeightPublicationError, match="chronology"):
            WeightPublicationRecord.from_dict(wire)
        return
    record = WeightPublicationRecord.from_dict(wire)
    assert record.to_dict() == wire
    assert record.digest == canonical_digest("cacheon.chain.weight-publication", wire)

"""Cumulative retry budgets and persist-before-GPU attempt leases."""

from pathlib import Path

import pytest

from optima.commit_reveal import (
    Ledger,
    RETRY_KIND_INFRASTRUCTURE,
    RETRY_KIND_NO_DECISION,
    RETRY_STATE_AUTOMATIC,
    RETRY_STATE_HELD,
    RETRY_STATE_IN_PROGRESS,
)


CHAIN_SCOPE = "genesis-netuid-v1:sha256:" + "a" * 64
ARENA_BRACKET = "arena:regime@" + "b" * 64
HOTKEY = "miner1"
BUNDLE_HASH = "c" * 64
POLICY = {
    "max_automatic_infrastructure_attempts": 3,
    "max_automatic_no_decision_attempts": 4,
    "max_total_attempts": 6,
}
BACKOFF = {"base_backoff_blocks": 10, "max_backoff_blocks": 100}


def _ledger() -> Ledger:
    led = Ledger()
    led.bind_chain_scope(CHAIN_SCOPE)
    return led


def _begin(led: Ledger, block: int, *, policy=POLICY):
    return led.begin_retry_attempt(
        hotkey=HOTKEY,
        bundle_hash=BUNDLE_HASH,
        arena_bracket=ARENA_BRACKET,
        current_block=block,
        reason="trusted evaluator lease",
        **policy,
    )


def _complete(led: Ledger, lease, block: int, kind: str, *, policy=POLICY):
    return led.complete_retry_attempt(
        hotkey=HOTKEY,
        bundle_hash=BUNDLE_HASH,
        arena_bracket=ARENA_BRACKET,
        lease_id=lease.lease_id,
        kind=kind,
        current_block=block,
        reason=f"completed as {kind}",
        **BACKOFF,
        **policy,
    )


def test_begin_is_persisted_before_gpu_and_crash_lease_survives(tmp_path: Path):
    path = tmp_path / "ledger.json"
    led = _ledger()
    lease = _begin(led, 10)

    assert lease.state == RETRY_STATE_IN_PROGRESS
    assert lease.attempts == 1
    assert lease.no_decision_attempts == 0
    assert lease.infrastructure_attempts == 0
    assert len(lease.lease_id) == 64
    led.save(path)  # the required persist-before-GPU boundary

    restored = Ledger.load(path)
    crashed = restored.retry_for(HOTKEY, BUNDLE_HASH, arena_bracket=ARENA_BRACKET)
    assert crashed == lease
    with pytest.raises(ValueError, match="already in progress"):
        _begin(restored, 100)

    cleared = restored.complete_retry_terminal(
        hotkey=HOTKEY,
        bundle_hash=BUNDLE_HASH,
        arena_bracket=ARENA_BRACKET,
        lease_id=lease.lease_id,
    )
    assert cleared == lease
    assert restored.retry_for(
        HOTKEY, BUNDLE_HASH, arena_bracket=ARENA_BRACKET
    ) is None


def test_completion_categorizes_same_attempt_without_double_counting():
    led = _ledger()
    first_lease = _begin(led, 10)
    first = _complete(led, first_lease, 10, RETRY_KIND_NO_DECISION)
    assert first.attempts == 1
    assert first.no_decision_attempts == 1
    assert first.infrastructure_attempts == 0
    assert first.state == RETRY_STATE_AUTOMATIC

    second_lease = _begin(led, first.next_block)
    assert second_lease.attempts == 2
    assert second_lease.no_decision_attempts == 1
    assert second_lease.infrastructure_attempts == 0
    second = _complete(
        led,
        second_lease,
        second_lease.lease_block,
        RETRY_KIND_INFRASTRUCTURE,
    )
    assert second.attempts == 2
    assert second.no_decision_attempts == 1
    assert second.infrastructure_attempts == 1

    with pytest.raises(ValueError, match="active lease"):
        _complete(
            led,
            second_lease,
            second_lease.lease_block,
            RETRY_KIND_INFRASTRUCTURE,
        )
    assert led.retry_for(
        HOTKEY, BUNDLE_HASH, arena_bracket=ARENA_BRACKET
    ).attempts == 2


@pytest.mark.parametrize(
    "kind,limit",
    [
        (RETRY_KIND_INFRASTRUCTURE, 3),
        (RETRY_KIND_NO_DECISION, 4),
    ],
)
def test_per_kind_caps_hold_on_the_exact_failed_attempt(kind: str, limit: int):
    led = _ledger()
    block = 10
    for attempt in range(1, limit + 1):
        lease = _begin(led, block)
        assert lease.attempts == attempt
        retry = _complete(led, lease, block, kind)
        if attempt < limit:
            assert retry.state == RETRY_STATE_AUTOMATIC
            block = retry.next_block
        else:
            assert retry.state == RETRY_STATE_HELD
    assert retry.attempts == limit


def test_total_cap_stops_alternating_kinds_without_resetting_budget():
    # Per-kind caps are four, so a 3/3 alternating split reaches only the total cap.
    policy = {
        "max_automatic_infrastructure_attempts": 4,
        "max_automatic_no_decision_attempts": 4,
        "max_total_attempts": 6,
    }
    led = _ledger()
    block = 10
    kinds = (
        RETRY_KIND_NO_DECISION,
        RETRY_KIND_INFRASTRUCTURE,
    ) * 3
    for attempt, kind in enumerate(kinds, start=1):
        lease = _begin(led, block, policy=policy)
        retry = _complete(led, lease, block, kind, policy=policy)
        assert retry.attempts == attempt
        if attempt < 6:
            assert retry.state == RETRY_STATE_AUTOMATIC
            block = retry.next_block
    assert retry.state == RETRY_STATE_HELD
    assert retry.no_decision_attempts == 3
    assert retry.infrastructure_attempts == 3


def test_compatibility_record_retry_is_cumulative_across_kind_changes():
    led = _ledger()
    block = 10
    retry = None
    for attempt, kind in enumerate(
        (
            RETRY_KIND_NO_DECISION,
            RETRY_KIND_INFRASTRUCTURE,
            RETRY_KIND_NO_DECISION,
        ),
        start=1,
    ):
        retry = led.record_retry(
            hotkey=HOTKEY,
            bundle_hash=BUNDLE_HASH,
            arena_bracket=ARENA_BRACKET,
            kind=kind,
            current_block=block,
            reason=f"legacy completion {kind}",
            **BACKOFF,
            **POLICY,
        )
        assert retry.attempts == attempt
        block = retry.next_block
    assert retry is not None
    assert retry.no_decision_attempts == 2
    assert retry.infrastructure_attempts == 1
    assert retry.state == RETRY_STATE_AUTOMATIC


def test_retry_attempt_budgets_must_be_positive_and_coherent():
    led = _ledger()
    with pytest.raises(ValueError, match="attempt budgets"):
        _begin(
            led,
            10,
            policy={
                "max_automatic_infrastructure_attempts": 3,
                "max_automatic_no_decision_attempts": 4,
                "max_total_attempts": 3,
            },
        )

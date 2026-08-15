"""Bounded hold re-entry and the same-reason circuit breaker."""

import pytest

from cacheon.chain.qualification_hold_requeue import (
    QualificationHoldRequeueError,
    QualificationHoldRequeuePolicy,
)


class _FakeStore:
    def __init__(
        self,
        fail_for: set[str] | None = None,
        parked: tuple[str, ...] = (),
    ) -> None:
        self.released: list[tuple[str, str]] = []
        self.fail_for = fail_for or set()
        self.parked = parked

    def release_hold(self, reservation_id: str, *, reason: str) -> None:
        if reservation_id in self.fail_for:
            raise RuntimeError("release refused")
        self.released.append((reservation_id, reason))

    def auto_requeueable_holds(self, *, limit: int = 64) -> tuple[str, ...]:
        return self.parked[:limit]


def test_policy_rejects_malformed_budgets() -> None:
    for kwargs in (
        {"max_attempts": 0},
        {"max_attempts": True},
        {"breaker_threshold": 0},
        {"breaker_threshold": 2.0},
    ):
        with pytest.raises(QualificationHoldRequeueError):
            QualificationHoldRequeuePolicy(**kwargs)


def test_hold_releases_until_the_attempt_cap_then_stays_held() -> None:
    policy = QualificationHoldRequeuePolicy(max_attempts=3, breaker_threshold=99)
    store = _FakeStore()
    reason = "remote_qualification_hold:graph_evidence_unavailable"

    first = policy.after_hold(store, reservation_ids=("r1",), reason=reason)
    second = policy.after_hold(store, reservation_ids=("r1",), reason=reason)
    capped = policy.after_hold(store, reservation_ids=("r1",), reason=reason)

    assert first == ("r1",) and second == ("r1",) and capped == ()
    assert [row[0] for row in store.released] == ["r1", "r1"]
    assert store.released[0][1] == f"auto_requeue_attempt_2_of_3:{reason}"
    assert store.released[1][1] == f"auto_requeue_attempt_3_of_3:{reason}"


def test_release_failure_is_contained_and_row_stays_held() -> None:
    policy = QualificationHoldRequeuePolicy(breaker_threshold=99)
    store = _FakeStore(fail_for={"bad"})
    released = policy.after_hold(
        store, reservation_ids=("bad", "ok"), reason="remote_qualification_hold:x"
    )
    assert released == ("ok",)
    assert [row[0] for row in store.released] == ["ok"]


def test_breaker_opens_on_consecutive_identical_reasons() -> None:
    policy = QualificationHoldRequeuePolicy(max_attempts=99, breaker_threshold=3)
    store = _FakeStore()
    reason = "remote_qualification_hold:graph_evidence_unavailable"

    assert policy.after_hold(store, reservation_ids=("r1",), reason=reason)
    assert policy.after_hold(store, reservation_ids=("r2",), reason=reason)
    assert not policy.breaker_open
    tripped = policy.after_hold(store, reservation_ids=("r3",), reason=reason)

    assert tripped == ()
    assert policy.breaker_open
    assert policy.refuse_fresh_claim()
    # r3 stayed held: only the first two rows were released.
    assert [row[0] for row in store.released] == ["r1", "r2"]


def test_distinct_reasons_and_terminals_reset_the_streak() -> None:
    policy = QualificationHoldRequeuePolicy(max_attempts=99, breaker_threshold=2)
    store = _FakeStore()

    policy.after_hold(
        store, reservation_ids=("r1",), reason="remote_qualification_hold:a"
    )
    policy.after_hold(
        store, reservation_ids=("r2",), reason="remote_qualification_hold:b"
    )
    assert not policy.breaker_open

    policy.after_hold(
        store, reservation_ids=("r3",), reason="remote_qualification_hold:b"
    )
    assert policy.breaker_open

    policy.note_terminal()
    assert not policy.breaker_open
    assert not policy.refuse_fresh_claim()
    released = policy.after_hold(
        store, reservation_ids=("r4",), reason="remote_qualification_hold:b"
    )
    assert released == ("r4",)


def test_reconcile_releases_rows_parked_by_an_earlier_lifetime() -> None:
    policy = QualificationHoldRequeuePolicy()
    store = _FakeStore(parked=("r1", "r2"))

    released = policy.reconcile_parked(store)

    assert released == ("r1", "r2")
    assert [row[1] for row in store.released] == [
        "auto_requeue_reconciled_at_start"
    ] * 2


def test_reconcile_contains_a_failed_release_and_keeps_the_rest() -> None:
    policy = QualificationHoldRequeuePolicy()
    store = _FakeStore(fail_for={"bad"}, parked=("bad", "ok"))

    assert policy.reconcile_parked(store) == ("ok",)
    assert [row[0] for row in store.released] == ["ok"]


def test_reconcile_restores_a_full_attempt_budget_for_the_reopened_row() -> None:
    policy = QualificationHoldRequeuePolicy(max_attempts=2, breaker_threshold=99)
    store = _FakeStore(parked=("r1",))
    reason = "remote_qualification_hold:legacy_no_decision"

    assert policy.after_hold(store, reservation_ids=("r1",), reason=reason) == ("r1",)
    assert policy.after_hold(store, reservation_ids=("r1",), reason=reason) == ()
    assert policy.reconcile_parked(store) == ("r1",)
    assert policy.after_hold(store, reservation_ids=("r1",), reason=reason) == ("r1",)


def test_reconcile_refuses_an_untyped_parked_listing() -> None:
    class _Untyped(_FakeStore):
        def auto_requeueable_holds(self, *, limit: int = 64) -> tuple[str, ...]:
            return ["r1"]  # type: ignore[return-value]

    with pytest.raises(QualificationHoldRequeueError):
        QualificationHoldRequeuePolicy().reconcile_parked(_Untyped())

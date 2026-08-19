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

        self.exhausted: list[str] = []

    def release_hold(self, reservation_id: str, *, reason: str) -> None:
        if reservation_id in self.fail_for:
            raise RuntimeError("release refused")
        self.released.append((reservation_id, reason))

    def mark_hold_retry_exhausted(self, reservation_id: str) -> None:
        self.exhausted.append(reservation_id)
        self.parked = tuple(p for p in self.parked if p != reservation_id)

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


def test_a_retry_still_within_budget_never_counts_toward_the_breaker() -> None:
    """max_attempts=99: no row can exhaust, so no number of holds may trip it."""

    policy = QualificationHoldRequeuePolicy(max_attempts=99, breaker_threshold=3)
    store = _FakeStore()
    reason = "remote_qualification_hold:graph_evidence_unavailable"

    for rid in ("r1", "r2", "r3", "r4", "r5"):
        assert policy.after_hold(store, reservation_ids=(rid,), reason=reason) == (rid,)

    assert not policy.breaker_open
    assert not policy.refuse_fresh_claim()
    assert store.exhausted == []


def test_distinct_reasons_and_terminals_reset_the_streak() -> None:
    policy = QualificationHoldRequeuePolicy(max_attempts=1, breaker_threshold=2)
    store = _FakeStore()

    policy.after_hold(
        store, reservation_ids=("r1",), reason="remote_qualification_hold:a"
    )
    policy.after_hold(
        store, reservation_ids=("r2",), reason="remote_qualification_hold:b"
    )
    # A different reason restarts the count, so one 'b' is not two.
    assert not policy.breaker_open

    policy.after_hold(
        store, reservation_ids=("r3",), reason="remote_qualification_hold:b"
    )
    assert policy.breaker_open

    policy.note_terminal()
    assert not policy.breaker_open
    assert not policy.refuse_fresh_claim()


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


def test_reconcile_gives_a_fresh_budget_only_to_a_row_that_never_exhausted_one() -> None:
    policy = QualificationHoldRequeuePolicy(max_attempts=3, breaker_threshold=99)
    store = _FakeStore(parked=("r1",))
    reason = "remote_qualification_hold:graph_evidence_unavailable"

    # One retry consumed, budget not spent: a restart may legitimately retry it.
    assert policy.after_hold(store, reservation_ids=("r1",), reason=reason) == ("r1",)
    assert store.exhausted == []

    restarted = QualificationHoldRequeuePolicy(max_attempts=3, breaker_threshold=99)
    assert restarted.reconcile_parked(store) == ("r1",)
    assert restarted.after_hold(
        store, reservation_ids=("r1",), reason=reason
    ) == ("r1",)


def test_reconcile_refuses_an_untyped_parked_listing() -> None:
    class _Untyped(_FakeStore):
        def auto_requeueable_holds(self, *, limit: int = 64) -> tuple[str, ...]:
            return ["r1"]  # type: ignore[return-value]

    with pytest.raises(QualificationHoldRequeueError):
        QualificationHoldRequeuePolicy().reconcile_parked(_Untyped())


def test_one_reservations_own_retries_never_open_the_breaker() -> None:
    """The 2026-08-15 outage: two bad rows halted every qualification claim.

    Counting hold events made one reservation's three bounded retries plus a
    second reservation's two look like five systemic faults.
    """

    policy = QualificationHoldRequeuePolicy(max_attempts=3, breaker_threshold=5)
    store = _FakeStore()
    reason = "remote_qualification_hold:graph_evidence_unavailable"

    for _ in range(3):
        policy.after_hold(store, reservation_ids=("r1",), reason=reason)
    for _ in range(2):
        policy.after_hold(store, reservation_ids=("r2",), reason=reason)

    assert store.exhausted == ["r1"]
    assert not policy.breaker_open
    assert not policy.refuse_fresh_claim()


def test_breaker_opens_on_distinct_exhausted_reservations() -> None:
    policy = QualificationHoldRequeuePolicy(max_attempts=1, breaker_threshold=3)
    store = _FakeStore()
    reason = "remote_qualification_hold:graph_evidence_unavailable"

    for rid in ("r1", "r2"):
        policy.after_hold(store, reservation_ids=(rid,), reason=reason)
    assert not policy.breaker_open

    policy.after_hold(store, reservation_ids=("r3",), reason=reason)
    assert policy.breaker_open
    assert policy.refuse_fresh_claim()
    assert sorted(store.exhausted) == ["r1", "r2", "r3"]

    policy.note_terminal()
    assert not policy.breaker_open


def test_exhausted_row_is_marked_durably_and_not_rehydrated_by_reconcile() -> None:
    """A spent budget must survive a restart, or the watchdog loops forever."""

    reason = "remote_qualification_hold:graph_evidence_unavailable"
    store = _FakeStore(parked=("r1",))

    first = QualificationHoldRequeuePolicy(max_attempts=1, breaker_threshold=99)
    first.after_hold(store, reservation_ids=("r1",), reason=reason)
    assert store.exhausted == ["r1"]

    # A restart forgets in-memory attempt counts; the durable mark must not be
    # forgotten with them.
    restarted = QualificationHoldRequeuePolicy(max_attempts=1, breaker_threshold=99)
    assert restarted.reconcile_parked(store) == ()
    assert store.released == []


def test_a_mark_failure_is_contained_and_still_counts_toward_the_breaker() -> None:
    class _MarkFails(_FakeStore):
        def mark_hold_retry_exhausted(self, reservation_id: str) -> None:
            raise RuntimeError("stamp refused")

    policy = QualificationHoldRequeuePolicy(max_attempts=1, breaker_threshold=2)
    store = _MarkFails()
    reason = "remote_qualification_hold:graph_evidence_unavailable"

    policy.after_hold(store, reservation_ids=("r1",), reason=reason)
    assert not policy.breaker_open
    policy.after_hold(store, reservation_ids=("r2",), reason=reason)
    assert policy.breaker_open


def test_a_non_systemic_reason_retries_but_never_opens_the_breaker() -> None:
    """Native bundles must not halt the queue for every other bundle.

    `legacy_no_decision` means the runner got no resident pair, which happens
    exactly when the bundle is not hot-swappable. That is one bundle class, not
    a fault burning the queue. On 2026-08-16 five such bundles opened the
    breaker eleven times; each time every qualification stopped -- including the
    swappable bundles producing verdicts -- until the stall watchdog restarted
    the supervisor, whereupon the same five re-exhausted and reopened it.

    They keep their bounded retries, because a transient `raw_speed_evidence`
    failure reports the same reason and does deserve them, and they are still
    durably marked so a restart does not rehydrate the budget.
    """

    policy = QualificationHoldRequeuePolicy(max_attempts=1, breaker_threshold=2)
    store = _FakeStore()
    reason = "remote_qualification_hold:legacy_no_decision"

    for rid in ("native1", "native2", "native3", "native4", "native5"):
        policy.after_hold(store, reservation_ids=(rid,), reason=reason)

    # Five exhausted against a threshold of two, and claims still flow.
    assert not policy.breaker_open
    assert not policy.refuse_fresh_claim()
    # Still durably marked, so a restart does not hand them a fresh budget.
    assert sorted(store.exhausted) == [
        "native1", "native2", "native3", "native4", "native5"
    ]


def test_non_systemic_holds_do_not_blind_the_breaker_to_a_real_fault() -> None:
    """Excluding one reason must not weaken the breaker for the others."""

    policy = QualificationHoldRequeuePolicy(max_attempts=1, breaker_threshold=2)
    store = _FakeStore()

    for rid in ("native1", "native2", "native3"):
        policy.after_hold(
            store,
            reservation_ids=(rid,),
            reason="remote_qualification_hold:legacy_no_decision",
        )
    assert not policy.breaker_open

    systemic = "remote_qualification_hold:graph_evidence_unavailable"
    for rid in ("r1", "r2"):
        policy.after_hold(store, reservation_ids=(rid,), reason=systemic)
    assert policy.breaker_open
    assert policy.refuse_fresh_claim()

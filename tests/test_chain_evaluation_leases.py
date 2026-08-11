from __future__ import annotations

import sqlite3

import pytest

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain.intake import (
    EvaluationLease,
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakeError,
    IntakePolicy,
    IntakeScope,
)
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationRetryPlan,
)
from cacheon.stack_identity import sha256_hex


SCOPE = IntakeScope("0x" + "0" * 64, 14)
AUTHORITY = {"schema": "lease-test-authority"}


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _arrival(index: int, *, block: int = 10, hotkey: str | None = None):
    return FinalizedArrival(
        hotkey=hotkey or f"miner-{index}",
        content_hash=_h(f"content:{index}"),
        url=f"https://example.com/{index}.tar.gz",
        block=block,
        block_hash="0x" + f"{block:064x}",
        event_index=index,
    )


def _store(tmp_path, **policy) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(**policy),
        scope=SCOPE,
    )


def _advance(store: FinalizedIntakeStore, block: int) -> None:
    cursor = store.finalized_cursor()
    assert cursor is not None and block >= cursor[0]
    if block == cursor[0]:
        return
    store.reserve_finalized(
        (),
        finalized_block=block,
        finalized_block_hash="0x" + f"{block:064x}",
    )


def _publish(store: FinalizedIntakeStore, row, marker: str):
    store.mark_fetching(row.reservation_id)
    return store.mark_published(
        row.reservation_id,
        delta_fingerprint=SubmittedDeltaFingerprint(
            "component",
            f"target.{marker}",
            _h(f"base:{marker}"),
            (f"slot.{marker}",),
            _h(f"archive:{marker}"),
            _h(f"selected:{marker}"),
            _h(f"exact:{marker}"),
            (_h(f"source:{marker}"),),
            (_h(f"binary:{marker}"),),
        ),
        publication_digest=_h(f"publication:{marker}"),
        publication_root=f"/published/{marker}",
    )


def _promote(store: FinalizedIntakeStore, reservation_id: str) -> None:
    service = _h("service")
    active = store.begin_screen(reservation_id, service_digest=service)
    candidate = _h(f"candidate:{reservation_id}:{active.screen_attempts}")
    receipt = ArenaScreenReceipt(
        service,
        candidate,
        active.screen_attempts,
        tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(stage), 1)
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )
    store.apply_screen_receipt(
        reservation_id, candidate_digest=candidate, receipt=receipt
    )


def _complete_screen(store: FinalizedIntakeStore, lease: EvaluationLease) -> None:
    assert len(lease.members) == 1
    reservation_id = lease.members[0].reservation_id
    service = _h("leased-service")
    active = store.begin_screen(reservation_id, service_digest=service)
    candidate = _h(f"leased-candidate:{reservation_id}:{active.screen_attempts}")
    receipt = ArenaScreenReceipt(
        service,
        candidate,
        active.screen_attempts,
        tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(f"leased:{stage}"), 1)
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )
    store.apply_screen_receipt(
        reservation_id, candidate_digest=candidate, receipt=receipt
    )


def _published_rows(store: FinalizedIntakeStore, count: int = 2):
    rows = store.reserve_finalized(
        tuple(_arrival(index) for index in range(count)),
        finalized_block=10,
        finalized_block_hash="0x" + f"{10:064x}",
    )
    return tuple(_publish(store, row, chr(ord("a") + index)) for index, row in enumerate(rows))


def test_additive_schema_migrates_a_legacy_database(tmp_path):
    with _store(tmp_path) as store:
        path = store.path
    db = sqlite3.connect(path)
    try:
        db.execute("DROP TABLE evaluation_lease_events")
        db.execute("DROP TABLE evaluation_lease_members")
        db.execute("DROP TABLE evaluation_leases")
        db.execute("DELETE FROM metadata WHERE key='evaluation_lease_schema'")
        db.commit()
    finally:
        db.close()

    with _store(tmp_path) as migrated:
        assert migrated._db.execute(
            "SELECT value FROM metadata WHERE key='evaluation_lease_schema'"
        ).fetchone()["value"] == "1"
        assert migrated.active_evaluation_leases() == ()


def test_active_lease_survives_reopen_and_hides_legacy_queue_reader(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        lease = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10, lease_blocks=20
        )
        assert lease is not None
        assert lease.reservation_ids == (row.reservation_id,)
        assert store.screenable() == ()

    with _store(tmp_path) as reopened:
        assert reopened.active_evaluation_leases() == (lease,)
        assert reopened.get(row.reservation_id).status == "published"
        assert reopened.screenable() == ()


def test_preview_and_claim_use_fifo_with_reproduction_priority(tmp_path):
    with _store(tmp_path) as store:
        first, second, third = _published_rows(store, 3)
        # Existing product policy gives a pending independent reproduction
        # priority over primary FIFO.  This direct setup isolates ordering from
        # the much larger settlement fixture.
        store._db.execute(
            "UPDATE reservations SET status='reproduction_pending',"
            "screen_lane='reproduction' WHERE reservation_id=?",
            (third.reservation_id,),
        )
        assert store.preview_evaluation_claim(stage="screen") == (
            third.reservation_id,
        )
        reproduction = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10
        )
        assert reproduction is not None
        assert reproduction.reservation_ids == (third.reservation_id,)
        assert store.preview_evaluation_claim(stage="screen") == (
            first.reservation_id,
        )
        assert second.reservation_id != first.reservation_id


def test_expiry_requeues_exact_status_without_attempt_and_advances_generation(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        before = store.get(row.reservation_id)
        first = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10, lease_blocks=2
        )
        assert first is not None
        _advance(store, 11)
        assert store.expire_evaluation_leases(current_block=11) == ()
        _advance(store, 12)
        assert store.expire_evaluation_leases(current_block=12) == (first,)
        requeued = store.get(row.reservation_id)
        assert (requeued.status, requeued.screen_attempts) == (
            before.status,
            before.screen_attempts,
        )
        second = store.claim_evaluation_lease(
            stage="screen", owner="worker-b", current_block=12, lease_blocks=2
        )
        assert second is not None
        assert second.generation == first.generation + 1
        assert second.lease_id != first.lease_id
        assert [event.event_type for event in store.evaluation_lease_events(
            reservation_id=row.reservation_id
        )] == ["claimed", "expired", "claimed"]


def test_late_result_durably_expires_before_rejecting(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        lease = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10, lease_blocks=2
        )
        assert lease is not None
        _advance(store, 12)
        with pytest.raises(IntakeError, match="after lease expiry"):
            with store.accept_evaluation_result(
                lease, current_block=12, result_digest=_h("late-result")
            ):
                raise AssertionError("expired result entered mutation context")
        assert store.active_evaluation_leases() == ()
        requeued = store.get(row.reservation_id)
        assert (requeued.status, requeued.screen_attempts) == ("published", 0)
        assert store.screenable() == (requeued,)
        assert [event.event_type for event in store.evaluation_lease_events(
            lease_id=lease.lease_id
        )] == ["claimed", "expired"]


def test_heartbeat_is_cas_and_stale_completion_is_rejected(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        original = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10, lease_blocks=10
        )
        assert original is not None
        _advance(store, 15)
        extended = store.heartbeat_evaluation_lease(
            original, current_block=15, lease_blocks=10
        )
        assert extended.expires_block == 25
        _advance(store, 16)
        with pytest.raises(IntakeError, match="stale"):
            with store.accept_evaluation_result(
                original, current_block=16, result_digest=_h("stale-result")
            ):
                raise AssertionError("stale lease entered its mutation context")
        with store.accept_evaluation_result(
            extended, current_block=16, result_digest=_h("screen-result")
        ) as members:
            assert tuple(row.reservation_id for row in members) == (
                row.reservation_id,
            )
            _complete_screen(store, extended)
        assert store.get(row.reservation_id).status == "promoted"
        assert [event.event_type for event in store.evaluation_lease_events(
            lease_id=extended.lease_id
        )] == ["claimed", "heartbeat", "completed"]


def test_only_one_claimer_can_own_one_queue_row(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        first = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10
        )
        second = store.claim_evaluation_lease(
            stage="screen", owner="worker-b", current_block=10
        )
        assert first is not None and first.reservation_ids == (row.reservation_id,)
        assert second is None
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_lease_members WHERE "
            "reservation_id=? AND active=1",
            (row.reservation_id,),
        ).fetchone()["n"] == 1


def test_active_leases_report_full_finalized_arrival_order(tmp_path):
    with _store(tmp_path, max_cohort=4) as store:
        rows = _published_rows(store, 4)
        leases = []
        leases.append(store.claim_evaluation_lease(
            stage="screen", owner="worker-0", current_block=10
        ))
        # Force later rows through the contract's reproduction-priority lane so
        # claim order differs from finalized event order.
        for index in (3, 2):
            store._db.execute(
                "UPDATE reservations SET status='reproduction_pending',"
                "screen_lane='reproduction' WHERE reservation_id=?",
                (rows[index].reservation_id,),
            )
            leases.append(store.claim_evaluation_lease(
                stage="screen", owner=f"worker-{index}", current_block=10
            ))
        leases.append(store.claim_evaluation_lease(
            stage="screen", owner="worker-1", current_block=10
        ))
        assert all(lease is not None for lease in leases)
        assert tuple(
            lease.reservation_ids[0] for lease in store.active_evaluation_leases()
        ) == tuple(row.reservation_id for row in rows)


def test_legacy_mutation_is_fenced_but_exact_accept_context_is_authorized(tmp_path):
    with _store(tmp_path) as store:
        row, unrelated = _published_rows(store, 2)
        lease = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10
        )
        assert lease is not None
        with pytest.raises(IntakeError, match="fences"):
            store.mark_held(row.reservation_id, "operator_race")
        assert store.get(row.reservation_id).status == "published"
        _advance(store, 11)
        with store.accept_evaluation_result(
            lease, current_block=11, result_digest=_h("authorized-result")
        ):
            with pytest.raises(IntakeError, match="non-member"):
                store.mark_held(unrelated.reservation_id, "widened_result")
            with pytest.raises(
                sqlite3.IntegrityError, match="fences reservation"
            ):
                store._db.execute(
                    "UPDATE reservations SET reason='raw_widened_result' "
                    "WHERE reservation_id=?",
                    (unrelated.reservation_id,),
                )
            _complete_screen(store, lease)
        assert store.get(row.reservation_id).status == "promoted"
        assert store.get(unrelated.reservation_id).status == "published"


def test_one_active_qualification_globally_fences_claim_and_settlement(tmp_path):
    with _store(tmp_path, max_cohort=3) as store:
        rows = _published_rows(store, 3)
        for row in rows[:2]:
            _promote(store, row.reservation_id)
        active = store.claim_evaluation_lease(
            stage="qualification",
            owner="worker-a",
            current_block=10,
            max_members=1,
        )
        assert active is not None
        assert store.preview_evaluation_claim(
            stage="qualification", max_members=1
        ) == ()
        assert store.claim_evaluation_lease(
            stage="qualification",
            owner="worker-b",
            current_block=10,
            max_members=1,
        ) is None

        # A retained candidate is enough to prove the settlement availability
        # fence without constructing an unrelated full settlement fixture.
        candidate = rows[2]
        store._db.execute(
            "UPDATE reservations SET status='qualified' WHERE reservation_id=?",
            (candidate.reservation_id,),
        )
        store._db.execute(
            "INSERT INTO settlement_candidates(reservation_id,authority_digest,"
            "candidate_digest,candidate_json,evidence_root,status) "
            "VALUES(?,?,?,?,?,'pending')",
            (
                candidate.reservation_id,
                _h("settlement-authority"),
                _h("settlement-candidate"),
                "{}",
                "/evidence",
            ),
        )
        assert store.has_pending_settlement() is False
        assert store.lease_settlement_cohort(current_block=10) is None
        _advance(store, 11)
        store.release_evaluation_lease(
            active, current_block=11, reason="operator_release"
        )
        assert store.has_pending_settlement() is True


def test_systemic_release_retains_diagnostic_and_consumes_no_attempt(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        lease = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10
        )
        assert lease is not None
        failure = _h("oci-backend-failure")
        _advance(store, 11)
        store.release_evaluation_lease(
            lease,
            current_block=11,
            reason="oci_backend",
            result_digest=failure,
        )
        retained = store.get(row.reservation_id)
        assert (retained.status, retained.screen_attempts) == ("published", 0)
        event = store.evaluation_lease_events(lease_id=lease.lease_id)[-1]
        assert (event.event_type, event.reason, event.result_digest) == (
            "released",
            "oci_backend",
            failure,
        )


def test_qualification_cohort_is_claimed_and_completed_atomically(tmp_path):
    with _store(tmp_path, max_cohort=2) as store:
        rows = _published_rows(store, 2)
        for row in rows:
            _promote(store, row.reservation_id)
        preview = store.preview_evaluation_claim(
            stage="qualification", max_members=2
        )
        assert preview == tuple(row.reservation_id for row in rows)
        lease = store.claim_evaluation_lease(
            stage="qualification",
            owner="worker-a",
            current_block=10,
            max_members=2,
        )
        assert lease is not None and lease.reservation_ids == preview

        # A partial result rolls back both the reservation transition and lease.
        _advance(store, 11)
        with pytest.raises(IntakeError, match="exact cohort"):
            with store.accept_evaluation_result(
                lease, current_block=11, result_digest=_h("partial")
            ):
                first = rows[0]
                store.mark_qualifying(first.reservation_id, _h("authority"), AUTHORITY)
                outcome = QualificationIntakeOutcome(
                    first.reservation_id,
                    first.delta_fingerprint.selected_delta_digest,
                    _h("authority"),
                    QualificationDecision.NO_DECISION,
                    "outer_session",
                    True,
                    failure_digest=_h("failure"),
                )
                # A singleton requeue plan is valid; it is deliberately partial
                # relative to the two-member lease and must not commit.
                retry = QualificationRetryPlan(
                    _h("authority"),
                    "requeue",
                    ((first.reservation_id,),),
                    _h("failure"),
                )
                store.apply_qualification_batch(
                    QualificationIntakeBatch(
                        _h("authority"), (outcome,), retry_plan=retry
                    ),
                    current_finalized_block=11,
                )
        assert all(store.get(row.reservation_id).status == "promoted" for row in rows)
        assert store.active_evaluation_leases() == (lease,)

        authority = _h("cohort-authority")
        failure = _h("cohort-failure")
        _advance(store, 12)
        with store.accept_evaluation_result(
            lease, current_block=12, result_digest=_h("cohort-result")
        ) as members:
            assert tuple(row.reservation_id for row in members) == preview
            for row in rows:
                store.mark_qualifying(row.reservation_id, authority, AUTHORITY)
            outcomes = tuple(
                QualificationIntakeOutcome(
                    row.reservation_id,
                    row.delta_fingerprint.selected_delta_digest,
                    authority,
                    QualificationDecision.NO_DECISION,
                    "outer_session",
                    True,
                    failure_digest=failure,
                )
                for row in rows
            )
            retry = QualificationRetryPlan(
                authority,
                "bisect",
                tuple((row.reservation_id,) for row in rows),
                failure,
            )
            store.apply_qualification_batch(
                QualificationIntakeBatch(authority, outcomes, retry_plan=retry),
                current_finalized_block=12,
            )
        assert store.active_evaluation_leases() == ()
        assert all(store.get(row.reservation_id).status == "published" for row in rows)


def test_no_plan_no_decision_is_not_mapped_to_candidate_failure(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        _promote(store, row.reservation_id)
        authority = _h("authority")
        failure = _h("infrastructure-failure")
        store.mark_qualifying(row.reservation_id, authority, AUTHORITY)
        outcome = QualificationIntakeOutcome(
            row.reservation_id,
            row.delta_fingerprint.selected_delta_digest,
            authority,
            QualificationDecision.NO_DECISION,
            "qualification_runner",
            True,
            failure_digest=failure,
        )
        retry = QualificationRetryPlan(
            authority, "requeue", ((row.reservation_id,),), failure
        )
        batch = QualificationIntakeBatch(
            authority, (outcome,), retry_plan=retry
        )
        # Exercise intake's defensive terminal branch for a retained legacy
        # batch whose retry plan was not available at commit time.
        object.__setattr__(batch, "retry_plan", None)
        result = store.apply_qualification_batch(
            batch, current_finalized_block=10
        )[0]
        assert (result.status, result.decision, result.reason) == (
            "no_decision",
            "NO_DECISION",
            "qualification_runner",
        )


def test_capacity_backpressure_is_deferred_not_candidate_failure(tmp_path):
    with _store(tmp_path, max_pending=1, max_cohort=1) as store:
        first, second = store.reserve_finalized(
            (_arrival(0), _arrival(1)),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )
        assert (first.status, second.status, second.decision, second.reason) == (
            "reserved",
            "deferred",
            "",
            "pending_queue_deferred",
        )
        store.mark_fetching(first.reservation_id)
        store.mark_failed(first.reservation_id, "invalid_bundle")
        assert store.pending() == (store.get(second.reservation_id),)
        assert store.get(second.reservation_id).status == "reserved"


def test_stale_capacity_promotes_older_deferred_before_new_arrival(tmp_path):
    with _store(
        tmp_path, max_pending=1, max_cohort=1, expiry_blocks=20
    ) as store:
        first = store.reserve_finalized(
            (_arrival(0, block=10),),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        second = store.reserve_finalized(
            (_arrival(1, block=11),),
            finalized_block=11,
            finalized_block_hash="0x" + f"{11:064x}",
        )[0]
        assert (first.status, second.status) == ("reserved", "deferred")
        third = store.reserve_finalized(
            (_arrival(2, block=30),),
            finalized_block=30,
            finalized_block_hash="0x" + f"{30:064x}",
        )[0]
        assert store.get(first.reservation_id).status == "expired"
        assert store.get(second.reservation_id).status == "reserved"
        assert (third.status, third.reason) == (
            "deferred",
            "pending_queue_deferred",
        )


def test_lease_clock_rejects_unretained_future_block(tmp_path):
    with _store(tmp_path) as store:
        _published_rows(store, 1)
        with pytest.raises(IntakeError, match="durable finalized cursor"):
            store.claim_evaluation_lease(
                stage="screen", owner="worker-a", current_block=11
            )


def test_event_reader_recomputes_canonical_identity(tmp_path):
    with _store(tmp_path) as store:
        _published_rows(store, 1)
        lease = store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=10
        )
        assert lease is not None
        store._db.execute("DROP TRIGGER evaluation_lease_events_reject_update")
        store._db.execute(
            "UPDATE evaluation_lease_events SET event_id=? WHERE lease_id=?",
            ("f" * 64, lease.lease_id),
        )
        with pytest.raises(IntakeError, match="event identity"):
            store.evaluation_lease_events(lease_id=lease.lease_id)


def test_systemic_release_cap_parks_reservation_held(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        clock = 10
        for round_number in (1, 2, 3):
            _advance(store, clock)
            lease = store.claim_evaluation_lease(
                stage="screen", owner="worker-a", current_block=clock
            )
            assert lease is not None, f"round {round_number} could not claim"
            _advance(store, clock + 1)
            store.release_evaluation_lease(
                lease,
                current_block=clock + 1,
                reason="systemic_qualification:worker_dead",
            )
            clock += 2
            retained = store.get(row.reservation_id)
            if round_number < 3:
                assert retained.status == "published"
            else:
                assert retained.status == "held"
                assert retained.decision == ""
                assert retained.reason == "systemic_release_cap:3"
        _advance(store, clock)
        assert store.claim_evaluation_lease(
            stage="screen", owner="worker-a", current_block=clock
        ) is None


def test_non_systemic_releases_never_trip_the_cap(tmp_path):
    with _store(tmp_path) as store:
        row = _published_rows(store, 1)[0]
        clock = 10
        for _ in range(4):
            _advance(store, clock)
            lease = store.claim_evaluation_lease(
                stage="screen", owner="worker-a", current_block=clock
            )
            assert lease is not None
            _advance(store, clock + 1)
            store.release_evaluation_lease(
                lease, current_block=clock + 1, reason="operator_release"
            )
            clock += 2
        assert store.get(row.reservation_id).status == "published"

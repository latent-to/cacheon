"""Independent worker ownership and ordered economics over one intake database."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from cacheon.chain.evaluation_recovery import EvaluationRecoveryHoldError
from tests import test_evaluation_recovery_store as recovery
from tests import test_chain_intake as intake


def test_simultaneous_coordinators_claim_four_distinct_oldest_jobs(tmp_path):
    from cacheon.arena_service import ArenaService
    from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
    from tests import test_evaluation_coordinator as fixture

    rows = fixture._published_rows(tmp_path, 6)
    service = ArenaService(fixture._manifest(), fixture._Provider())
    cursor = fixture._CursorAuthority((fixture.BLOCK, fixture._block_hash(fixture.BLOCK)))
    workers = [fixture._coordinator(tmp_path, service, cursor, owner=f"pair-{index}",
               store_factory=RecoverableFinalizedIntakeStore) for index in range(4)]
    for _ in rows:
        fixture._run_screen(workers[0])
    start = Barrier(4)

    def claim(worker):
        start.wait(timeout=10)
        store, point = worker._open_at_durable_cursor()
        with store:
            return store.claim_recoverable_qualification(
                owner=worker.owner, current_block=point[0], max_members=1,
                max_active=worker.service.manifest.capacity.max_active_qualifications,
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(claim, workers))
    assert {item.lease.reservation_ids[0] for item in claimed} == {row.reservation_id for row in rows[:4]}
    assert len({item.lease.lease_id for item in claimed}) == 4


@pytest.mark.parametrize("profile", recovery.PROFILES)
def test_four_workers_reopen_only_their_own_fifo_claim(tmp_path, profile):
    with recovery._store(tmp_path, profile, max_cohort=1) as store:
        rows = [recovery._promoted(store, profile, index) for index in range(6)]
        claims = [store.claim_recoverable_qualification(
            owner=f"pair-{index}", current_block=10, max_active=4,
        ) for index in range(4)]
        assert [claim.lease.reservation_ids for claim in claims] == [
            (row.reservation_id,) for row in rows[:4]
        ]
        assert store.claim_recoverable_qualification(
            owner="pair-4", current_block=10, max_active=4,
        ) is None
        assert store.claim_evaluation_lease(
            stage="screen", owner="pair-0", current_block=10, max_active=4,
        ) is None
        held = store.hold_recovery(claims[0], current_block=10, reason="inspection")
        with pytest.raises(EvaluationRecoveryHoldError, match="multiple active"):
            store.pending_qualification_recovery()
    with recovery._store(tmp_path, profile, max_cohort=1) as store:
        assert store.pending_qualification_recovery(owner="pair-0") == held
        assert store.pending_qualification_recovery(owner="unknown") is None
        for index in range(1, 4):
            assert store.pending_qualification_recovery(owner=f"pair-{index}") == claims[index]
            assert store.claim_recoverable_qualification(
                owner=f"pair-{index}", current_block=10, max_active=4,
            ) is None
        renewed, _ = store.renew_recovery_lease(claims[2], current_block=10, lease_blocks=50)
        assert store.pending_qualification_recovery(owner="pair-2") == renewed
        assert store.pending_qualification_recovery(owner="pair-0") == held


def test_version_two_migration_retains_active_lease_and_recovery_events(tmp_path):
    profile = recovery.PROFILES[0]
    with recovery._store(tmp_path, profile, max_cohort=1) as store:
        recovery._promoted(store, profile)
        recovery._promoted(store, profile, 1)
        original = store.claim_recoverable_qualification(owner="pair-0", current_block=10)
        events = store.evaluation_recovery_events(original)
        path = store.path
    with sqlite3.connect(path) as db:
        db.execute("UPDATE metadata SET value='2' WHERE key IN ('evaluation_lease_schema','evaluation_recovery_schema')")
        db.execute("DROP INDEX evaluation_leases_one_active_qualification")
        db.execute("CREATE UNIQUE INDEX evaluation_leases_one_active_qualification ON evaluation_leases(competition_arena) WHERE state='active' AND stage='qualification'")
        db.execute("DROP INDEX evaluation_recoveries_one_unresolved")
        db.execute("CREATE UNIQUE INDEX evaluation_recoveries_one_unresolved ON evaluation_recoveries(competition_arena) WHERE resolution=''")
    with recovery._store(tmp_path, profile, max_cohort=1) as store:
        assert store.pending_qualification_recovery(owner="pair-0") == original
        assert store.evaluation_recovery_events(original) == events
        second = store.claim_recoverable_qualification(owner="pair-1", current_block=10, max_active=2)
        assert second is not None and second.lease.lease_id != original.lease.lease_id
        assert store.evaluation_recovery_events(original) == events


@pytest.mark.parametrize("target", ["activation.silu_and_mul", "norm.rmsnorm"])
def test_out_of_order_import_releases_only_completed_arrival_prefix(tmp_path, monkeypatch, target):
    with intake._store(tmp_path) as store:
        pending = []
        apply = store.apply_qualification_batch
        with monkeypatch.context() as patch:
            patch.setattr(store, "apply_qualification_batch", lambda batch, **kwargs: pending.append((batch, kwargs)))
            candidates = [intake._qualified_settlement_candidate(
                store, index=index, marker=str(index), target=target, check_single_pass=False,
            ) for index in range(4)]
        expected_prefixes = [[], [0], [0], [0, 1, 2, 3]]
        settled = []
        for index, prefix in zip((2, 0, 3, 1), expected_prefixes, strict=True):
            batch, kwargs = pending[index]
            apply(batch, **kwargs)
            assert [claim.hotkey for claim in store.passed_reward_claims()] == [
                candidates[i].hotkey for i in prefix
            ]
            while (lease := store.lease_settlement_cohort(current_block=11)) is not None:
                plan, evidence = intake._settlement_plan(store, lease)
                store.commit_settlement(lease, plan, evidence, current_block=11)
                settled.extend(candidate.reservation_digest for candidate in lease.candidates)
            assert settled == [candidates[i].reservation_digest for i in prefix]
        evidence = [tuple(row) for row in store._db.execute("SELECT * FROM settlement_qualifications")]
        claims = store.passed_reward_claims()
    with intake._store(tmp_path) as store:
        assert store.passed_reward_claims() == claims
        assert [tuple(row) for row in store._db.execute("SELECT * FROM settlement_qualifications")] == evidence
        assert store.lease_settlement_cohort(current_block=11) is None


def test_held_other_target_blocks_new_rewards_but_not_previous_earnings(tmp_path):
    from dashboard.winners import qualified_winners
    with intake._store(tmp_path) as store:
        first = intake._qualified_settlement_candidate(store, marker="first", index=0)
        held = intake._reserve_one(store, index=1, hotkey="held")
        store.mark_held(held.reservation_id, "inspection")
        later = intake._qualified_settlement_candidate(store, marker="later", index=2, target="norm.rmsnorm")
        assert [claim.hotkey for claim in store.passed_reward_claims()] == [first.hotkey]
        assert [row["hotkey"] for row in qualified_winners(store._db)] == [first.hotkey]
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (first,)
        plan, evidence = intake._settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        assert store.lease_settlement_cohort(current_block=11) is None
        store.expire(held.reservation_id, current_block=500010, reason="operator_terminal_expiry")
        assert [claim.hotkey for claim in store.passed_reward_claims()] == [first.hotkey, later.hotkey]
        assert {row["hotkey"] for row in qualified_winners(store._db)} == {first.hotkey, later.hotkey}


def test_migration_preserves_earned_rewards_and_read_only_legacy_dashboard(tmp_path):
    from dashboard.winners import qualified_winners
    with intake._store(tmp_path) as store:
        earlier = intake._reserve_one(store, index=0, hotkey="held")
        store.mark_held(earlier.reservation_id, "inspection")
        earned = intake._qualified_settlement_candidate(store, marker="legacy", index=1)
        # Before pooling, every complete PASS earned even across a HOLD.
        store._db.execute("ALTER TABLE settlement_candidates DROP COLUMN reward_eligible")
        assert [row["hotkey"] for row in qualified_winners(store._db)] == [earned.hotkey]
    with intake._store(tmp_path) as store:
        assert [claim.hotkey for claim in store.passed_reward_claims()] == [earned.hotkey]
        new = intake._qualified_settlement_candidate(store, marker="new", index=2)
        assert [claim.hotkey for claim in store.passed_reward_claims()] == [earned.hotkey]
        assert new.hotkey not in {row["hotkey"] for row in qualified_winners(store._db)}

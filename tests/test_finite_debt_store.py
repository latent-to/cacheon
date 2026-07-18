from __future__ import annotations

import json

import pytest

from optima.chain.finite_debt_store import SeededFamilyClock, reward_family_id
from optima.chain.intake import IntakeError
from optima.finite_debt import (
    IMPROVEMENT_GROSS,
    PPM,
    DebtClaimBalance,
    FamilyBudgetShare,
    FiniteDebtPolicyManifest,
    cancel_claim_balance,
    issue_innovation_claim,
    pay_claim_balance,
)
from optima.settlement import SettlementCandidate, plan_settlement
from tests.test_chain_intake import (
    _arrival,
    _h,
    _qualified_settlement_candidate,
    _store,
)


def _family(candidate: SettlementCandidate) -> str:
    assert candidate.candidate_manifest is not None
    contribution = candidate.candidate_manifest.entries[candidate.target_id]
    return reward_family_id(
        candidate.arena_digest,
        candidate.target_id,
        contribution.target_spec_digest,
    )


def _policy(
    family_id: str,
    *,
    epoch_blocks: int = 10,
    lifetime_blocks: int = 100,
    beta_ppm: int = 100_000,
) -> FiniteDebtPolicyManifest:
    return FiniteDebtPolicyManifest(
        family_budget_shares=(FamilyBudgetShare(family_id, PPM),),
        reserve_hotkey="reserve",
        reserve_ppm=100_000,
        epoch_blocks=epoch_blocks,
        beta_ppm=beta_ppm,
        tau_blocks=648_000,
        lifetime_blocks=lifetime_blocks,
        k_ppm=PPM,
        improvement_basis=IMPROVEMENT_GROSS,
        clock_reset_threshold_log_units_ppm=1,
    )


def _activate(store, candidate, *, policy=None, seeds=()):
    selected = policy or _policy(_family(candidate))
    block_hash = "0x" + f"{10:064x}"
    activation = store.activate_finite_debt_policy(
        selected,
        activation_block=10,
        activation_block_hash=block_hash,
        seeded_family_clocks=seeds,
    )
    return selected, activation


def _commit(store, candidate, *, current_block: int = 12, with_hash: bool = True):
    lease = store.lease_settlement_cohort(current_block=11)
    assert lease is not None
    assert lease.candidates == (candidate,)
    plan = plan_settlement(
        lease.candidates,
        current_manifest=lease.stack.manifest,
        current_tree_digest=lease.stack.tree_digest,
        initial_event_sequence=lease.initial_event_sequence,
        previous_event_digest=lease.previous_event_digest,
    )
    evidence = tuple(
        store.reopen_settlement_evidence(row) for row in lease.candidates
    )
    block_hash = "0x" + f"{current_block:064x}"
    store.reserve_finalized(
        (),
        finalized_block=current_block,
        finalized_block_hash=block_hash,
    )
    state = store.commit_settlement(
        lease,
        plan,
        evidence,
        current_block=current_block,
        **({"current_block_hash": block_hash} if with_hash else {}),
    )
    return state, plan


def _drop_immutable_guard(store, table: str, action: str) -> None:
    store._db.execute(f"DROP TRIGGER {table}_reject_{action}")


def _restore_immutable_guard(store, table: str, action: str) -> None:
    store._db.execute(
        f"CREATE TRIGGER {table}_reject_{action} BEFORE {action.upper()} ON {table} "
        "BEGIN SELECT RAISE(ABORT,'finite-debt rows are immutable'); END"
    )


def test_reward_family_id_matches_v1_standing_claim_identity(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _commit(store, candidate, with_hash=False)
        standing = store.active_reward_claims()[0][0]
        assert _family(candidate) == standing.family_id


def test_schema3_reopen_creates_no_retro_debt_and_preserves_v1_bytes(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _commit(store, candidate, with_hash=False)
        emissions_digest = _h("retained-v1-emissions-policy")
        store._db.execute(
            "INSERT INTO metadata(key,value) VALUES('emissions_policy_digest',?)",
            (emissions_digest,),
        )
        settlement_before = tuple(
            tuple(row)
            for row in store._db.execute(
                "SELECT * FROM settlement_events ORDER BY sequence"
            )
        )
        standing_before = tuple(
            tuple(row)
            for row in store._db.execute(
                "SELECT * FROM standing_reward_claims ORDER BY arena_id,target_id"
            )
        )
        for table in (
            "finite_debt_epoch_allocations",
            "finite_debt_reward_epochs",
            "finite_debt_claim_balances",
            "finite_debt_claims",
            "finite_debt_family_clocks",
            "finite_debt_policy_activations",
            "finite_debt_reward_events",
        ):
            store._db.execute(f"DROP TABLE {table}")
        store._db.execute("UPDATE metadata SET value='3' WHERE key='schema'")

    with _store(tmp_path) as reopened:
        assert reopened._db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()["value"] == "5"
        assert reopened._db.execute(
            "SELECT value FROM metadata WHERE key='emissions_policy_digest'"
        ).fetchone()["value"] == emissions_digest
        assert tuple(
            tuple(row)
            for row in reopened._db.execute(
                "SELECT * FROM settlement_events ORDER BY sequence"
            )
        ) == settlement_before
        assert tuple(
            tuple(row)
            for row in reopened._db.execute(
                "SELECT * FROM standing_reward_claims ORDER BY arena_id,target_id"
            )
        ) == standing_before
        assert reopened.finite_debt_claim_states() == ()
        assert reopened.finite_debt_family_clocks() == ()
        assert reopened.finite_debt_reward_events() == ()
        assert reopened.finite_debt_reward_epochs() == ()


def test_active_policy_crown_atomically_issues_claim_clock_and_v1_claim(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, activation = _activate(store, candidate)
        state, plan = _commit(store, candidate)

        assert state.generation == 1
        assert len(store.active_reward_claims()[0]) == 1
        claims = store.finite_debt_claim_states()
        assert len(claims) == 1
        claim = claims[0].claim
        assert claim.policy_digest == policy.digest
        assert claim.family_id == _family(candidate)
        assert claim.prior_accepted_crown_block is None
        assert claim.time_multiplier_ppm == PPM
        assert claim.settlement_block == 12
        assert claim.expires_block == 112
        assert claims[0].balance.remaining_units == claim.principal_units
        assert (
            claims[0].balance.paid_units
            + claims[0].balance.forfeited_units
            + claims[0].balance.remaining_units
            == claim.principal_units
        )
        clocks = store.finite_debt_family_clocks(
            policy_digest=policy.digest,
            family_id=claim.family_id,
        )
        assert len(clocks) == 1
        assert clocks[0].source == "crown"
        assert clocks[0].claim_digest == claim.digest
        assert clocks[0].finalized_order == candidate.finalized_order
        assert [row["event_type"] for row in store.finite_debt_reward_events()] == [
            "policy_activated",
            "claim_issued",
        ]
        assert activation.policy.digest == policy.digest
        assert any(event.event_type.value == "CROWN" for event in plan.events)


def test_reopen_rejects_missing_required_crown_clock(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _activate(store, candidate)
        _commit(store, candidate)
        with store._transaction():
            _drop_immutable_guard(store, "finite_debt_family_clocks", "delete")
            store._db.execute(
                "DELETE FROM finite_debt_family_clocks WHERE source='crown'"
            )
            _restore_immutable_guard(store, "finite_debt_family_clocks", "delete")
        with pytest.raises(IntakeError, match="exact reward authority"):
            store.finite_debt_claim_states()


def test_reopen_rejects_extra_crown_clock_reusing_old_claim_event(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _activate(store, candidate)
        _commit(store, candidate)
        clock = store.finite_debt_family_clocks()[0]
        with store._transaction():
            store._db.execute(
                "INSERT INTO finite_debt_family_clocks(policy_digest,family_id,"
                "accepted_crown_block,accepted_crown_block_hash,event_index,"
                "event_subindex,reservation_digest,source,claim_digest,"
                "reward_event_digest) VALUES(?,?,?,?,?,?,?,'crown',?,?)",
                (
                    clock.policy_digest,
                    clock.family_id,
                    clock.accepted_crown_block + 1,
                    "0x" + f"{clock.accepted_crown_block + 1:064x}",
                    clock.event_index,
                    clock.event_subindex,
                    _h("forged later crown clock"),
                    clock.claim_digest,
                    clock.reward_event_digest,
                ),
            )
        with pytest.raises(IntakeError, match="exact reward authority"):
            store.finite_debt_family_clocks()


def test_reopen_rejects_seed_not_listed_by_activation(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, _activation = _activate(store, candidate)
        activation_event = store.finite_debt_reward_events()[0]
        with store._transaction():
            store._db.execute(
                "INSERT INTO finite_debt_family_clocks(policy_digest,family_id,"
                "accepted_crown_block,accepted_crown_block_hash,event_index,"
                "event_subindex,reservation_digest,source,claim_digest,"
                "reward_event_digest) VALUES(?,?,?,?,?,?,?,'seed','',?)",
                (
                    policy.digest,
                    _family(candidate),
                    9,
                    "0x" + f"{9:064x}",
                    0,
                    0,
                    _h("unlisted activation seed"),
                    activation_event["event_digest"],
                ),
            )
        with pytest.raises(IntakeError, match="exact reward authority"):
            store.finite_debt_family_clocks()


def test_reopen_rejects_self_consistent_claim_speed_principal_inflation(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, activation = _activate(store, candidate)
        _commit(store, candidate)
        real = store.finite_debt_claim_states()[0]
        forged_claim = issue_innovation_claim(
            policy,
            family_id=real.claim.family_id,
            candidate_digest=real.claim.candidate_digest,
            retained_evidence_digest=real.claim.retained_evidence_digest,
            hotkey=real.claim.hotkey,
            settled_speedup="2",
            threshold_speedup="1",
            accepted_crown_block=real.claim.accepted_crown_block,
            prior_accepted_crown_block=real.claim.prior_accepted_crown_block,
            settlement_block=real.claim.settlement_block,
        )
        assert forged_claim.principal_units > real.claim.principal_units * 10
        forged_balance = DebtClaimBalance.open(forged_claim)
        issuance = store.finite_debt_reward_events()[-1]
        forged_payload = dict(issuance["payload"])
        forged_payload["activation_digest"] = activation.digest
        forged_payload["claim"] = forged_claim.to_dict()
        forged_payload["claim_digest"] = forged_claim.digest
        forged_event_digest = store._finite_debt._event_digest(
            chain_scope_digest=store._finite_debt.chain_scope_digest,
            sequence=issuance["sequence"],
            previous_event_digest=issuance["previous_event_digest"],
            event_type="claim_issued",
            block=issuance["block"],
            block_hash=issuance["block_hash"],
            payload=forged_payload,
        )
        guarded_tables = (
            "finite_debt_reward_events",
            "finite_debt_claims",
            "finite_debt_claim_balances",
            "finite_debt_family_clocks",
        )
        with store._transaction():
            store._db.execute("PRAGMA defer_foreign_keys=ON")
            for table in guarded_tables:
                _drop_immutable_guard(store, table, "update")
            store._db.execute(
                "UPDATE finite_debt_claims SET claim_digest=?,principal_units=?,"
                "claim_json=?,issuance_reward_event_digest=? WHERE claim_digest=?",
                (
                    forged_claim.digest,
                    forged_claim.principal_units,
                    json.dumps(
                        forged_claim.to_dict(), separators=(",", ":"), sort_keys=True
                    ),
                    forged_event_digest,
                    real.claim.digest,
                ),
            )
            store._db.execute(
                "UPDATE finite_debt_claim_balances SET claim_digest=?,balance_digest=?,"
                "principal_units=?,remaining_units=?,balance_json=?,"
                "reward_event_digest=? WHERE claim_digest=?",
                (
                    forged_claim.digest,
                    forged_balance.digest,
                    forged_balance.principal_units,
                    forged_balance.remaining_units,
                    json.dumps(
                        forged_balance.to_dict(),
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    forged_event_digest,
                    real.claim.digest,
                ),
            )
            store._db.execute(
                "UPDATE finite_debt_family_clocks SET claim_digest=?,"
                "reward_event_digest=? WHERE claim_digest=?",
                (
                    forged_claim.digest,
                    forged_event_digest,
                    real.claim.digest,
                ),
            )
            store._db.execute(
                "UPDATE finite_debt_reward_events SET event_digest=?,payload_json=? "
                "WHERE event_digest=?",
                (
                    forged_event_digest,
                    json.dumps(
                        forged_payload, separators=(",", ":"), sort_keys=True
                    ),
                    issuance["event_digest"],
                ),
            )
            for table in guarded_tables:
                _restore_immutable_guard(store, table, "update")

        with pytest.raises(IntakeError, match="claim authority differs"):
            store.finite_debt_claim_states()


def test_missing_family_rolls_back_entire_crown_and_non_crowns_issue_nothing(
    tmp_path,
) -> None:
    crown_root = tmp_path / "crown"
    with _store(crown_root) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        missing_family_policy = _policy(_h("absent-family"))
        _activate(store, candidate, policy=missing_family_policy)
        with pytest.raises(IntakeError, match="absent from the policy budget"):
            _commit(store, candidate)
        assert store.finite_debt_claim_states() == ()
        assert store.finite_debt_family_clocks() == ()
        assert [row["event_type"] for row in store.finite_debt_reward_events()] == [
            "policy_activated"
        ]
        assert store.active_reward_claims() == ((), ())
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM settlement_events"
        ).fetchone()["n"] == 0
        assert store.evaluation_stack(candidate.arena_digest).generation == 0

    noncrown_root = tmp_path / "noncrown"
    with _store(noncrown_root) as store:
        activation_hash = "0x" + f"{10:064x}"
        store.reserve_finalized(
            (), finalized_block=10, finalized_block_hash=activation_hash
        )
        store.activate_finite_debt_policy(
            _policy(_h("unused-family")),
            activation_block=10,
            activation_block_hash=activation_hash,
        )
        rows = store.reserve_finalized(
            (_arrival(0, block=11), _arrival(1, block=11)),
            finalized_block=11,
            finalized_block_hash="0x" + f"{11:064x}",
        )
        store.mark_fetching(rows[0].reservation_id)
        store.mark_failed(rows[0].reservation_id, "failed before settlement")
        store.mark_held(rows[1].reservation_id, "operator hold")
        assert store.finite_debt_claim_states() == ()
        assert store.finite_debt_family_clocks() == ()


def test_seeded_same_block_clock_uses_full_finalized_order_and_multiplier_one(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        family = _family(candidate)
        seed = SeededFamilyClock(
            family,
            candidate.finalized_block,
            "0x" + f"{candidate.finalized_block:064x}",
            candidate.event_index,
            candidate.event_subindex,
            "0" * 63 + "1",
        )
        policy, _activation = _activate(store, candidate, seeds=(seed,))
        _commit(store, candidate)
        claim = store.finite_debt_claim_states()[0].claim
        assert claim.prior_accepted_crown_block == candidate.finalized_block
        assert claim.time_multiplier_ppm == PPM
        clocks = store.finite_debt_family_clocks(
            policy_digest=policy.digest,
            family_id=family,
        )
        assert [row.source for row in clocks] == ["seed", "crown"]
        assert clocks[0].finalized_order < clocks[1].finalized_order


def test_lifecycle_forfeits_departure_and_expiry_and_guards_policy_upgrade(
    tmp_path,
) -> None:
    departure_root = tmp_path / "departure"
    with _store(departure_root) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, _activation = _activate(store, candidate)
        _commit(store, candidate)
        next_policy = _policy(_family(candidate), beta_ppm=100_001)
        block13 = "0x" + f"{13:064x}"
        store.reserve_finalized(
            (), finalized_block=13, finalized_block_hash=block13
        )
        with pytest.raises(IntakeError, match="open debt"):
            store.activate_finite_debt_policy(
                next_policy,
                activation_block=13,
                activation_block_hash=block13,
            )
        changed = store.reconcile_finite_debt_lifecycle(
            current_block=13,
            current_block_hash=block13,
            eligible_hotkeys=("reserve",),
        )
        assert len(changed) == 1
        assert changed[0].balance.status == "cancelled"
        assert changed[0].balance.forfeited_units == changed[0].claim.principal_units
        assert store.reconcile_finite_debt_lifecycle(
            current_block=13,
            current_block_hash=block13,
            eligible_hotkeys=("reserve",),
        ) == ()
        block14 = "0x" + f"{14:064x}"
        store.reserve_finalized(
            (), finalized_block=14, finalized_block_hash=block14
        )
        upgraded = store.activate_finite_debt_policy(
            next_policy,
            activation_block=14,
            activation_block_hash=block14,
        )
        assert upgraded.previous_policy_digest == policy.digest

    expiry_root = tmp_path / "expiry"
    with _store(expiry_root) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _activate(
            store,
            candidate,
            policy=_policy(_family(candidate), lifetime_blocks=15),
        )
        _commit(store, candidate)
        expiry_hash = "0x" + f"{27:064x}"
        store.reserve_finalized(
            (), finalized_block=27, finalized_block_hash=expiry_hash
        )
        changed = store.reconcile_finite_debt_lifecycle(
            current_block=27,
            current_block_hash=expiry_hash,
            eligible_hotkeys=("miner", "reserve"),
        )
        assert len(changed) == 1
        assert changed[0].balance.status == "expired"
        assert changed[0].balance.terminal_block == 27
        assert changed[0].balance.forfeited_units == changed[0].claim.principal_units


def test_policy_upgrade_reopens_terminal_balance_before_acceptance(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, _activation = _activate(store, candidate)
        _commit(store, candidate)
        block13 = "0x" + f"{13:064x}"
        store.reserve_finalized((), finalized_block=13, finalized_block_hash=block13)
        changed = store.reconcile_finite_debt_lifecycle(
            current_block=13,
            current_block_hash=block13,
            eligible_hotkeys=("reserve",),
        )
        assert len(changed) == 1 and changed[0].balance.status == "cancelled"
        with store._transaction():
            _drop_immutable_guard(store, "finite_debt_claim_balances", "update")
            store._db.execute(
                "UPDATE finite_debt_claim_balances SET balance_json='{}' "
                "WHERE claim_digest=? AND revision=1",
                (changed[0].claim.digest,),
            )
            _restore_immutable_guard(store, "finite_debt_claim_balances", "update")
        block14 = "0x" + f"{14:064x}"
        store.reserve_finalized((), finalized_block=14, finalized_block_hash=block14)
        with pytest.raises(IntakeError, match="balance is corrupt"):
            store.activate_finite_debt_policy(
                _policy(_family(candidate), beta_ppm=policy.beta_ppm + 1),
                activation_block=14,
                activation_block_hash=block14,
            )


def test_runtime_family_invalidation_cancels_debt_resets_clock_and_is_idempotent(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, _activation = _activate(store, candidate)
        _commit(store, candidate)
        before = store.finite_debt_claim_states()[0]
        block13 = "0x" + f"{13:064x}"
        store.reserve_finalized((), finalized_block=13, finalized_block_hash=block13)
        invalidation = _h("runtime invalidation authority")
        changed = store.invalidate_finite_debt_family(
            policy_digest=policy.digest,
            family_id=before.claim.family_id,
            invalidation_digest=invalidation,
            current_block=13,
            current_block_hash=block13,
        )
        assert len(changed) == 1
        assert changed[0].claim == before.claim
        assert changed[0].balance.status == "cancelled"
        assert changed[0].balance.terminal_reason == "runtime_invalidation"
        assert changed[0].balance.forfeited_units == before.balance.remaining_units
        clocks = store.finite_debt_family_clocks(
            policy_digest=policy.digest,
            family_id=before.claim.family_id,
        )
        assert [clock.source for clock in clocks] == ["crown", "invalidation"]
        assert clocks[-1].claim_digest == ""
        assert clocks[-1].reservation_digest == invalidation
        assert [
            row["event_type"] for row in store.finite_debt_reward_events()
        ][-1] == "family_invalidated"
        assert store.invalidate_finite_debt_family(
            policy_digest=policy.digest,
            family_id=before.claim.family_id,
            invalidation_digest=invalidation,
            current_block=13,
            current_block_hash=block13,
        ) == changed

        block14 = "0x" + f"{14:064x}"
        store.reserve_finalized((), finalized_block=14, finalized_block_hash=block14)
        with pytest.raises(IntakeError, match="retry differs"):
            store.invalidate_finite_debt_family(
                policy_digest=policy.digest,
                family_id=before.claim.family_id,
                invalidation_digest=invalidation,
                current_block=14,
                current_block_hash=block14,
            )

    with _store(tmp_path) as reopened:
        assert reopened.finite_debt_claim_states()[0] == changed[0]
        assert reopened.invalidate_finite_debt_family(
            policy_digest=policy.digest,
            family_id=before.claim.family_id,
            invalidation_digest=invalidation,
            current_block=13,
            current_block_hash=block13,
        ) == changed
        marker = reopened._finite_debt._latest_clock(
            policy.digest, before.claim.family_id
        )
        assert marker is not None and marker.source == "invalidation"
        # The issuance path deliberately maps this marker to no prior crown, so
        # a later accepted crown receives the first-crown multiplier M=1.
        from optima.finite_debt import issue_innovation_claim

        next_claim = issue_innovation_claim(
            policy,
            family_id=before.claim.family_id,
            candidate_digest=_h("post-invalidation candidate"),
            retained_evidence_digest=_h("post-invalidation evidence"),
            hotkey="miner",
            settled_speedup="1.01",
            threshold_speedup="1",
            accepted_crown_block=14,
            prior_accepted_crown_block=(
                None if marker.source == "invalidation" else marker.accepted_crown_block
            ),
            settlement_block=14,
        )
        assert next_claim.prior_accepted_crown_block is None
        assert next_claim.time_multiplier_ppm == PPM


def test_reopen_rejects_invalidation_that_leaves_family_debt_open(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, activation = _activate(store, candidate)
        _commit(store, candidate)
        state = store.finite_debt_claim_states()[0]
        prior = store.finite_debt_family_clocks()[-1]
        block_hash = "0x" + f"{13:064x}"
        invalidation = _h("empty-transition invalidation")
        store.reserve_finalized(
            (), finalized_block=13, finalized_block_hash=block_hash
        )
        with store._transaction():
            event_digest = store._finite_debt._append_event(
                "family_invalidated",
                block=13,
                block_hash=block_hash,
                payload={
                    "activation_digest": activation.digest,
                    "claim_balance_transitions": [],
                    "family_id": state.claim.family_id,
                    "invalidation_digest": invalidation,
                    "policy_digest": policy.digest,
                    "prior_family_clock": store._finite_debt._clock_identity(prior),
                },
            )
            store._db.execute(
                "INSERT INTO finite_debt_family_clocks(policy_digest,family_id,"
                "accepted_crown_block,accepted_crown_block_hash,event_index,"
                "event_subindex,reservation_digest,source,claim_digest,"
                "reward_event_digest) VALUES(?,?,?,?,9223372036854775807,"
                "9223372036854775807,?,'invalidation','',?)",
                (
                    policy.digest,
                    state.claim.family_id,
                    13,
                    block_hash,
                    invalidation,
                    event_digest,
                ),
            )
        with pytest.raises(IntakeError, match="balance set differs"):
            store.finite_debt_claim_states()


def test_zero_principal_crown_commits_and_advances_clock_without_debt(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        family = _family(candidate)
        other_family = _h("large reference-pool family")
        base = _policy(family)
        policy = FiniteDebtPolicyManifest(
            family_budget_shares=(
                FamilyBudgetShare(family, 2),
                FamilyBudgetShare(other_family, PPM - 2),
            ),
            reserve_hotkey=base.reserve_hotkey,
            reserve_ppm=base.reserve_ppm,
            epoch_blocks=base.epoch_blocks,
            beta_ppm=base.beta_ppm,
            tau_blocks=base.tau_blocks,
            lifetime_blocks=base.lifetime_blocks,
            k_ppm=1,
            improvement_basis=base.improvement_basis,
            clock_reset_threshold_log_units_ppm=(
                base.clock_reset_threshold_log_units_ppm
            ),
        )
        _activate(store, candidate, policy=policy)
        state, _plan = _commit(store, candidate)
        assert state.generation == 1
        assert store.finite_debt_claim_states() == ()
        clocks = store.finite_debt_family_clocks(
            policy_digest=policy.digest,
            family_id=_family(candidate),
        )
        assert len(clocks) == 1
        marker = clocks[0]
        assert marker.source == "crown_no_debt"
        assert marker.accepted_crown_block == candidate.finalized_block
        assert marker.claim_digest == ""
        assert [
            row["event_type"] for row in store.finite_debt_reward_events()
        ][-1] == "claim_not_issued"

    later = issue_innovation_claim(
        _policy(_family(candidate)),
        family_id=_family(candidate),
        candidate_digest=_h("later candidate after no-debt crown"),
        retained_evidence_digest=_h("later evidence after no-debt crown"),
        hotkey="miner",
        settled_speedup="1.01",
        threshold_speedup="1",
        accepted_crown_block=candidate.finalized_block + policy.tau_blocks,
        prior_accepted_crown_block=marker.accepted_crown_block,
        settlement_block=candidate.finalized_block + policy.tau_blocks,
    )
    assert later.prior_accepted_crown_block == candidate.finalized_block
    assert later.time_multiplier_ppm == PPM + policy.beta_ppm // 2


def test_family_invalidation_event_rejects_cross_family_balance_forgery(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        family_a = _family(candidate)
        family_b = _h("unrelated invalidation family")
        policy = _policy(family_a)
        policy = FiniteDebtPolicyManifest(
            family_budget_shares=(
                FamilyBudgetShare(family_a, 500_000),
                FamilyBudgetShare(family_b, 500_000),
            ),
            reserve_hotkey=policy.reserve_hotkey,
            reserve_ppm=policy.reserve_ppm,
            epoch_blocks=policy.epoch_blocks,
            beta_ppm=policy.beta_ppm,
            tau_blocks=policy.tau_blocks,
            lifetime_blocks=policy.lifetime_blocks,
            k_ppm=policy.k_ppm,
            improvement_basis=policy.improvement_basis,
            clock_reset_threshold_log_units_ppm=(
                policy.clock_reset_threshold_log_units_ppm
            ),
        )
        _activate(store, candidate, policy=policy)
        _commit(store, candidate)
        state = store.finite_debt_claim_states()[0]
        block13 = "0x" + f"{13:064x}"
        store.reserve_finalized((), finalized_block=13, finalized_block_hash=block13)
        store.invalidate_finite_debt_family(
            policy_digest=policy.digest,
            family_id=family_b,
            invalidation_digest=_h("family-b invalidation"),
            current_block=13,
            current_block_hash=block13,
        )
        invalidation_event = store.finite_debt_reward_events()[-1]["event_digest"]
        forged = cancel_claim_balance(
            state.claim,
            state.balance,
            at_block=13,
            reason="runtime_invalidation",
        )
        with store._transaction():
            store._finite_debt._insert_balance(
                forged,
                revision=1,
                reward_event_digest=invalidation_event,
            )
        with pytest.raises(IntakeError, match="crossed policy or family"):
            store.finite_debt_claim_states()


def test_inflight_pre_invalidation_crown_settles_without_debt_or_retry_poison(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, _activation = _activate(store, candidate)
        block11 = "0x" + f"{11:064x}"
        store.reserve_finalized((), finalized_block=11, finalized_block_hash=block11)
        store.invalidate_finite_debt_family(
            policy_digest=policy.digest,
            family_id=_family(candidate),
            invalidation_digest=_h("in-flight invalidation"),
            current_block=11,
            current_block_hash=block11,
        )
        state, _plan = _commit(store, candidate, current_block=12)
        assert state.generation == 1
        assert store.finite_debt_claim_states() == ()
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "crowned"
        last = store.finite_debt_reward_events()[-1]
        assert last["event_type"] == "claim_not_issued"
        assert last["payload"]["reason"] == "accepted_before_family_invalidation"
        assert [
            clock.source
            for clock in store.finite_debt_family_clocks(
                policy_digest=policy.digest,
                family_id=_family(candidate),
            )
        ] == ["invalidation"]


def test_no_debt_clock_rejects_unrelated_claim_not_issued_event(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy = _policy(_family(candidate))
        block11 = "0x" + f"{11:064x}"
        store.reserve_finalized((), finalized_block=11, finalized_block_hash=block11)
        store.activate_finite_debt_policy(
            policy, activation_block=11, activation_block_hash=block11
        )
        _commit(store, candidate, current_block=12)
        event = store.finite_debt_reward_events()[-1]
        assert event["event_type"] == "claim_not_issued"
        assert event["payload"]["reason"] == "accepted_before_policy_activation"
        with store._transaction():
            store._db.execute(
                "INSERT INTO finite_debt_family_clocks(policy_digest,family_id,"
                "accepted_crown_block,accepted_crown_block_hash,event_index,"
                "event_subindex,reservation_digest,source,claim_digest,"
                "reward_event_digest) VALUES(?,?,?,?,?,?,?,'crown_no_debt','',?)",
                (
                    policy.digest,
                    _family(candidate),
                    candidate.finalized_block,
                    "0x" + f"{candidate.finalized_block:064x}",
                    candidate.event_index,
                    candidate.event_subindex,
                    candidate.reservation_digest,
                    event["event_digest"],
                ),
            )
        with pytest.raises(IntakeError, match="exact reward authority"):
            store.finite_debt_family_clocks(
                policy_digest=policy.digest,
                family_id=_family(candidate),
            )


def test_projection_is_read_only_and_confirmed_close_is_exactly_once(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        policy, _activation = _activate(store, candidate)
        _commit(store, candidate)
        before = store.finite_debt_claim_states()[0]

        with pytest.raises(IntakeError, match="reserve hotkey"):
            store.project_finite_debt_epoch(
                effective_block=20,
                eligible_hotkeys=("miner",),
            )
        with pytest.raises(IntakeError, match="ineligible positive miner"):
            store.project_finite_debt_epoch(
                effective_block=20,
                eligible_hotkeys=("reserve",),
            )
        projection = store.project_finite_debt_epoch(
            effective_block=20,
            eligible_hotkeys=("miner", "reserve"),
        )
        assert store.finite_debt_claim_states()[0] == before
        assert projection.policy_digest == policy.digest
        assert projection.payout_units + projection.reserve_units == PPM
        assert sum(row.units for row in projection.weights) == PPM

        boundary_hash = "0x" + f"{20:064x}"
        store.reserve_finalized(
            (), finalized_block=20, finalized_block_hash=boundary_hash
        )
        publication = _h("confirmed-debt-publication")
        with pytest.raises(IntakeError, match="expected digest"):
            store.close_confirmed_debt_epoch(
                projection,
                expected_projection_digest=_h("wrong-projection"),
                finalized_block=20,
                finalized_block_hash=boundary_hash,
                publication_record_digest=publication,
                eligible_hotkeys=("miner", "reserve"),
            )
        epoch = store.close_confirmed_debt_epoch(
            projection,
            expected_projection_digest=projection.digest,
            finalized_block=20,
            finalized_block_hash=boundary_hash,
            publication_record_digest=publication,
            eligible_hotkeys=("miner", "reserve"),
        )
        after = store.finite_debt_claim_states()[0]
        assert after.balance.paid_units - before.balance.paid_units == projection.payout_units
        assert epoch.projection.digest == projection.digest
        assert store.finite_debt_reward_epochs() == (epoch,)

        later_hash = "0x" + f"{21:064x}"
        store.reserve_finalized(
            (), finalized_block=21, finalized_block_hash=later_hash
        )
        event_count = len(store.finite_debt_reward_events())
        retry = store.close_confirmed_debt_epoch(
            projection,
            expected_projection_digest=projection.digest,
            finalized_block=20,
            finalized_block_hash=boundary_hash,
            publication_record_digest=publication,
            eligible_hotkeys=("miner", "reserve"),
        )
        assert retry == epoch
        assert store.finite_debt_claim_states()[0] == after
        assert len(store.finite_debt_reward_events()) == event_count
        assert store.project_finite_debt_epoch(
            effective_block=20,
            eligible_hotkeys=("miner", "reserve"),
        ) == projection
        with pytest.raises(IntakeError, match="retry differs"):
            store.close_confirmed_debt_epoch(
                projection,
                expected_projection_digest=projection.digest,
                finalized_block=20,
                finalized_block_hash=boundary_hash,
                publication_record_digest=_h("another-publication"),
                eligible_hotkeys=("miner", "reserve"),
            )

    with _store(tmp_path) as reopened:
        assert reopened.finite_debt_reward_epochs() == (epoch,)
        assert reopened.finite_debt_claim_states()[0] == after
        assert reopened.project_finite_debt_epoch(
            effective_block=20,
            eligible_hotkeys=("miner", "reserve"),
        ) == projection


def test_epoch_reopen_rejects_extra_balance_revision_reusing_payout_event(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _activate(store, candidate)
        _commit(store, candidate)
        projection = store.project_finite_debt_epoch(
            effective_block=20,
            eligible_hotkeys=("miner", "reserve"),
        )
        boundary_hash = "0x" + f"{20:064x}"
        store.reserve_finalized(
            (), finalized_block=20, finalized_block_hash=boundary_hash
        )
        epoch = store.close_confirmed_debt_epoch(
            projection,
            expected_projection_digest=projection.digest,
            finalized_block=20,
            finalized_block_hash=boundary_hash,
            publication_record_digest=_h("extra-revision-publication"),
            eligible_hotkeys=("miner", "reserve"),
        )
        after = store.finite_debt_claim_states()[0]
        forged = pay_claim_balance(
            after.claim,
            after.balance,
            1,
            at_block=20,
        )
        revision = store._db.execute(
            "SELECT MAX(revision) AS value FROM finite_debt_claim_balances "
            "WHERE claim_digest=?",
            (after.claim.digest,),
        ).fetchone()["value"] + 1
        with store._transaction():
            store._finite_debt._insert_balance(
                forged,
                revision=revision,
                reward_event_digest=epoch.payout_event_digest,
            )

        with pytest.raises(IntakeError, match="not exactly authorized"):
            store.finite_debt_claim_states()
        with pytest.raises(IntakeError, match="balance revision set differs"):
            store.finite_debt_reward_epochs()

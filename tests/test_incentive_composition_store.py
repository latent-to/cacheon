from __future__ import annotations

import pytest

from optima.chain.finite_debt_store import reward_family_id
from optima.chain.incentive_composition_store import (
    IncentiveCompositionStoreError,
    SELECTED_SELECTION_REPORT_DIGEST,
)
from optima.chain.intake import IntakeError
from optima.finite_debt import (
    IMPROVEMENT_GROSS,
    PPM,
    FamilyBudgetShare,
    FiniteDebtPolicyManifest,
    pay_claim_balance,
    project_debt_epoch,
)
from optima.incentive_composition import (
    DISCOVERY_BOUNTY_ONLY,
    DISCOVERY_REGISTERED_PROMOTION,
    IncentiveCompositionPolicyManifest,
    pay_discovery_balance,
    review_discovery_disposition,
)
from optima.settlement import SettlementCandidate
from tests.test_chain_intake import (
    _h,
    _qualified_discovery_candidate,
    _qualified_settlement_candidate,
    _store,
)
from tests.test_finite_debt_store import _commit, _family


def _selected_core(family_id: str) -> FiniteDebtPolicyManifest:
    return FiniteDebtPolicyManifest(
        family_budget_shares=(FamilyBudgetShare(family_id, PPM),),
        reserve_hotkey="reserve",
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        beta_ppm=100_000,
        tau_blocks=648_000,
        lifetime_blocks=648_000,
        k_ppm=PPM,
        improvement_basis=IMPROVEMENT_GROSS,
        clock_reset_threshold_log_units_ppm=1,
    )


def _selected_composition(
    core: FiniteDebtPolicyManifest,
) -> IncentiveCompositionPolicyManifest:
    return IncentiveCompositionPolicyManifest(
        innovation_policy_digest=core.digest,
        selection_report_digest=SELECTED_SELECTION_REPORT_DIGEST,
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        discovery_cap_units=50_000,
        per_award_principal_cap_epochs=1,
        discovery_lifetime_blocks=648_000,
    )


def _activate_selected(store, candidate):
    core = _selected_core(_family(candidate))
    block_hash = "0x" + f"{10:064x}"
    core_activation = store.activate_finite_debt_policy(
        core,
        activation_block=10,
        activation_block_hash=block_hash,
    )
    composition = _selected_composition(core)
    activation = store.activate_incentive_composition(
        composition,
        activation_block=10,
        activation_block_hash=block_hash,
    )
    assert activation.core_activation_digest == core_activation.digest
    return core, composition, activation


def _review(
    policy,
    win,
    *,
    marker: str,
    block: int,
    decision: str,
    hotkey: str | None = None,
):
    return review_discovery_disposition(
        policy,
        win_digest=win.digest,
        proposal_digest=win.proposal_digest,
        retained_evidence_digest=win.retained_evidence_digest,
        review_digest=_h(f"review:{marker}"),
        hotkey=win.hotkey if hotkey is None else hotkey,
        win_block=win.settlement_block,
        authority_block=block,
        decision=decision,
        requested_principal_epochs=7 if decision == DISCOVERY_BOUNTY_ONLY else 0,
        promoted_target_digest=(
            _h(f"target:{marker}")
            if decision == DISCOVERY_REGISTERED_PROMOTION
            else ""
        ),
    )


def _retain_discovery_win(store, *, marker: str):
    from optima.settlement import plan_settlement

    candidate = _qualified_discovery_candidate(
        store,
        index=1,
        proposal_digest=_h(f"lifecycle:{marker}"),
        hotkey="lifecycle-discoverer",
    )
    core = _selected_core(_h(f"unused lifecycle family:{marker}"))
    block10 = "0x" + f"{10:064x}"
    store.activate_finite_debt_policy(
        core,
        activation_block=10,
        activation_block_hash=block10,
    )
    policy = _selected_composition(core)
    store.activate_incentive_composition(
        policy,
        activation_block=10,
        activation_block_hash=block10,
    )
    lease = store.lease_settlement_cohort(current_block=11)
    assert lease is not None and lease.candidates == (candidate,)
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
    block11 = "0x" + f"{11:064x}"
    store.reserve_finalized(
        (), finalized_block=11, finalized_block_hash=block11
    )
    store.commit_settlement(
        lease,
        plan,
        evidence,
        current_block=11,
        current_block_hash=block11,
    )
    win = store.review_pending_discovery_wins()[0]
    return candidate, policy, win


def test_schema4_to5_is_empty_no_retro_and_immutable(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        for table in (
            "incentive_composed_allocations",
            "incentive_composed_epochs",
            "incentive_discovery_balances",
            "incentive_discovery_claims",
            "incentive_discovery_dispositions",
            "incentive_discovery_wins",
            "incentive_composition_activations",
        ):
            store._db.execute(f"DROP TABLE {table}")
        store._db.execute("UPDATE metadata SET value='4' WHERE key='schema'")

    with _store(tmp_path) as reopened:
        assert reopened._db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()["value"] == "5"
        assert reopened.reviewed_discovery_dispositions() == ()
        assert reopened.discovery_debt_claim_states() == ()
        assert reopened.incentive_composition_reward_epochs() == ()
        assert reopened.active_incentive_composition(at_block=10) is None
        triggers = {
            row["name"]
            for row in reopened._db.execute(
                "SELECT name FROM sqlite_schema WHERE type='trigger' "
                "AND name LIKE 'incentive_%_reject_%'"
            )
        }
        assert len(triggers) == 14


def test_activation_is_exact_and_legacy_discovery_fails_closed(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        core = _selected_core(_family(candidate))
        block_hash = "0x" + f"{10:064x}"
        store.activate_finite_debt_policy(
            core,
            activation_block=10,
            activation_block_hash=block_hash,
        )
        wrong = IncentiveCompositionPolicyManifest(
            innovation_policy_digest=core.digest,
            selection_report_digest=_h("wrong selection report"),
            reserve_ppm=100_000,
            epoch_blocks=7_200,
            discovery_cap_units=50_000,
            per_award_principal_cap_epochs=1,
            discovery_lifetime_blocks=648_000,
        )
        with pytest.raises(IntakeError, match="exact D-013"):
            store.activate_incentive_composition(
                wrong,
                activation_block=10,
                activation_block_hash=block_hash,
            )
        store._db.execute(
            "INSERT INTO discovery_bounty_claims(claim_digest,proposal_digest,"
            "claim_json,status,event_id) VALUES(?,?,?,'active',?)",
            (_h("legacy claim"), _h("legacy proposal"), "{}", _h("legacy event")),
        )
        with pytest.raises(IntakeError, match="legacy discovery"):
            store.activate_incentive_composition(
                _selected_composition(core),
                activation_block=10,
                activation_block_hash=block_hash,
            )
        store._db.execute(
            "UPDATE discovery_bounty_claims SET status='forged_terminal'"
        )
        with pytest.raises(IntakeError, match="legacy discovery"):
            store.activate_incentive_composition(
                _selected_composition(core),
                activation_block=10,
                activation_block_hash=block_hash,
            )

    unequal_root = tmp_path / "unequal-family-budget"
    with _store(unequal_root) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        unequal = FiniteDebtPolicyManifest(
            family_budget_shares=(
                FamilyBudgetShare(_family(candidate), 600_000),
                FamilyBudgetShare(_h("second family"), 400_000),
            ),
            reserve_hotkey="reserve",
            reserve_ppm=100_000,
            epoch_blocks=7_200,
            beta_ppm=100_000,
            tau_blocks=648_000,
            lifetime_blocks=648_000,
            k_ppm=PPM,
            improvement_basis=IMPROVEMENT_GROSS,
            clock_reset_threshold_log_units_ppm=1,
        )
        block_hash = "0x" + f"{10:064x}"
        store.activate_finite_debt_policy(
            unequal,
            activation_block=10,
            activation_block_hash=block_hash,
        )
        with pytest.raises(IntakeError, match="exact D-013"):
            store.activate_incentive_composition(
                _selected_composition(unequal),
                activation_block=10,
                activation_block_hash=block_hash,
            )


def test_legacy_standing_title_survives_composition_without_retro_debt(tmp_path) -> None:
    from optima.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
    )

    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _commit(store, candidate, with_hash=False)
        standing_before = store.active_reward_claims()[0]
        crown_before = store.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        )
        assert len(standing_before) == 1

        core = _selected_core(_family(candidate))
        block12 = "0x" + f"{12:064x}"
        store.activate_finite_debt_policy(
            core, activation_block=12, activation_block_hash=block12
        )
        store.activate_incentive_composition(
            _selected_composition(core),
            activation_block=12,
            activation_block_hash=block12,
        )
        assert store.active_reward_claims()[0] == standing_before
        assert store.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        ) == crown_before
        assert store.finite_debt_claim_states() == ()

        context = GlobalRewardProjectionContext(
            store.scope.digest,
            "validator",
            12,
            block12,
            (MetagraphMember(0, "validator"),),
        )
        with pytest.raises(IntakeError, match="legacy V1 weight projection"):
            store.build_weight_projection(
                policy=EmissionsPolicyManifest(100, 20, 100_000),
                context=context,
                catalogs={},
                netuid=store.scope.netuid,
            )


def test_composed_disposition_projection_close_and_restart(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        core, _composition, activation = _activate_selected(store, candidate)
        _commit(store, candidate, current_block=12)

        core_before = store.finite_debt_claim_states()[0]
        core_only_projection = project_debt_epoch(
            core,
            effective_block=7_210,
            states=(core_before,),
        )
        projection = store.project_incentive_composition_epoch(
            effective_block=7_210,
            eligible_hotkeys=("miner", "reserve"),
        )
        assert store.finite_debt_claim_states()[0] == core_before
        assert store.discovery_debt_claim_states() == ()
        assert projection.discovery_payout_units == 0
        assert projection.innovation_payout_units == 900_000
        assert projection.reserve_units == 100_000
        assert sum(row.units for row in projection.weights) == PPM
        with pytest.raises(IntakeError, match="core-only projection"):
            store.project_finite_debt_epoch(
                effective_block=7_210,
                eligible_hotkeys=("miner", "reserve"),
            )

        boundary_hash = "0x" + f"{7_210:064x}"
        store.reserve_finalized(
            (), finalized_block=7_210, finalized_block_hash=boundary_hash
        )
        publication = _h("externally confirmed composed publication")
        epoch = store.close_confirmed_composed_epoch(
            projection,
            expected_projection_digest=projection.digest,
            finalized_block=7_210,
            finalized_block_hash=boundary_hash,
            publication_record_digest=publication,
            eligible_hotkeys=("miner", "reserve"),
        )
        core_after = store.finite_debt_claim_states()[0]
        assert (
            core_after.balance.paid_units - core_before.balance.paid_units
            == 900_000
        )
        assert epoch.activation_digest == activation.digest
        assert store.close_confirmed_composed_epoch(
            projection,
            expected_projection_digest=projection.digest,
            finalized_block=7_210,
            finalized_block_hash=boundary_hash,
            publication_record_digest=publication,
            eligible_hotkeys=("miner", "reserve"),
        ) == epoch
        with pytest.raises(IntakeError, match="core-only close"):
            store.close_confirmed_debt_epoch(
                core_only_projection,
                expected_projection_digest=core_only_projection.digest,
                finalized_block=7_210,
                finalized_block_hash=boundary_hash,
                publication_record_digest=publication,
                eligible_hotkeys=("miner", "reserve"),
            )
        events = store.finite_debt_reward_events()
        assert [row["event_type"] for row in events] == [
            "policy_activated",
            "composition_policy_activated",
            "claim_issued",
            "composed_epoch_paid",
        ]

    with _store(tmp_path) as reopened:
        assert reopened.incentive_composition_reward_epochs() == (epoch,)
        assert reopened.finite_debt_claim_states()[0] == core_after
        assert reopened.discovery_debt_claim_states() == ()
        assert reopened.project_incentive_composition_epoch(
            effective_block=7_210,
            eligible_hotkeys=("miner", "reserve"),
        ) == projection


@pytest.mark.parametrize("reward_class", ("core", "discovery"))
def test_composed_epoch_reopen_rejects_extra_revision_reusing_payout_event(
    tmp_path, reward_class: str,
) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        discoveries = (
            _qualified_discovery_candidate(
                store,
                index=0,
                proposal_digest=_h("extra-revision-discovery-a"),
                hotkey="discoverer-a",
            ),
            _qualified_discovery_candidate(
                store,
                index=1,
                proposal_digest=_h("extra-revision-discovery-b"),
                hotkey="discoverer-b",
            ),
        )
        _core, composition, _activation = _activate_selected(store, candidate)

        from optima.settlement import plan_settlement

        # Keep the registered candidate from advancing the incumbent between
        # the two discovery-only settlements.  This is test fixture state, not
        # an economic transition; restore it before its own leased settlement.
        store._db.execute(
            "UPDATE settlement_candidates SET status='held',reason='test_fixture' "
            "WHERE candidate_digest=?",
            (candidate.digest,),
        )
        for block, expected in ((11, discoveries[0]), (12, discoveries[1])):
            lease = store.lease_settlement_cohort(current_block=block)
            assert lease is not None and lease.candidates == (expected,)
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
            block_hash = "0x" + f"{block:064x}"
            store.reserve_finalized(
                (), finalized_block=block, finalized_block_hash=block_hash
            )
            store.commit_settlement(
                lease,
                plan,
                evidence,
                current_block=block,
                current_block_hash=block_hash,
            )
        store._db.execute(
            "UPDATE settlement_candidates SET status='pending',reason='' "
            "WHERE candidate_digest=? AND status='held'",
            (candidate.digest,),
        )
        _commit(store, candidate, current_block=13)

        wins = store.review_pending_discovery_wins()
        assert len(wins) == 2
        for index, win in enumerate(wins, start=14):
            block_hash = "0x" + f"{index:064x}"
            store.reserve_finalized(
                (), finalized_block=index, finalized_block_hash=block_hash
            )
            store.record_reviewed_discovery_disposition(
                _review(
                    composition,
                    win,
                    marker=f"extra-revision-{index}",
                    block=index,
                    decision=DISCOVERY_BOUNTY_ONLY,
                ),
                authority_block_hash=block_hash,
            )

        eligible = ("discoverer-a", "discoverer-b", "miner", "reserve")
        projection = store.project_incentive_composition_epoch(
            effective_block=7_210,
            eligible_hotkeys=eligible,
        )
        assert projection.discovery_payout_units == 50_000
        assert projection.innovation_payout_units == 850_000
        boundary_hash = "0x" + f"{7_210:064x}"
        store.reserve_finalized(
            (), finalized_block=7_210, finalized_block_hash=boundary_hash
        )
        epoch = store.close_confirmed_composed_epoch(
            projection,
            expected_projection_digest=projection.digest,
            finalized_block=7_210,
            finalized_block_hash=boundary_hash,
            publication_record_digest=_h("extra-composed-revision-publication"),
            eligible_hotkeys=eligible,
        )
        if reward_class == "core":
            after = store.finite_debt_claim_states()[0]
            forged = pay_claim_balance(
                after.claim,
                after.balance,
                1,
                at_block=7_210,
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
        else:
            after = store.discovery_debt_claim_states()[0]
            forged = pay_discovery_balance(
                after.claim,
                after.balance,
                1,
                at_block=7_210,
            )
            revision = store._db.execute(
                "SELECT MAX(revision) AS value FROM incentive_discovery_balances "
                "WHERE claim_digest=?",
                (after.claim.digest,),
            ).fetchone()["value"] + 1
            with store._transaction():
                store._incentive_composition._insert_discovery_balance(
                    forged,
                    revision=revision,
                    reward_event_digest=epoch.payout_event_digest,
                )
            with pytest.raises(IntakeError, match="not exactly authorized"):
                store.discovery_debt_claim_states()
        with pytest.raises(IntakeError, match="balance revision set differs"):
            store.incentive_composition_reward_epochs()


def test_active_composition_retains_review_pending_wins_and_binds_dispositions(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        first = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=_h("post-composition discovery one"),
            hotkey="discoverer-one",
        )
        second = _qualified_discovery_candidate(
            store,
            index=2,
            proposal_digest=_h("post-composition discovery two"),
            hotkey="discoverer-two",
        )
        core = _selected_core(_h("unused selected family"))
        block10 = "0x" + f"{10:064x}"
        store.activate_finite_debt_policy(
            core,
            activation_block=10,
            activation_block_hash=block10,
        )
        store.activate_incentive_composition(
            _selected_composition(core),
            activation_block=10,
            activation_block_hash=block10,
        )
        from optima.settlement import plan_settlement

        for block, expected in ((11, first), (12, second)):
            lease = store.lease_settlement_cohort(current_block=block)
            assert lease is not None and lease.candidates == (expected,)
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
            block_hash = "0x" + f"{block:064x}"
            store.reserve_finalized(
                (), finalized_block=block, finalized_block_hash=block_hash
            )
            store.commit_settlement(
                lease,
                plan,
                evidence,
                current_block=block,
                current_block_hash=block_hash,
            )

        wins = store.review_pending_discovery_wins()
        assert tuple(win.candidate_digest for win in wins) == (
            first.digest,
            second.digest,
        )
        assert store.active_reward_claims()[1] == ()
        assert store.reviewed_discovery_dispositions() == ()
        assert store.discovery_debt_claim_states() == ()

        block13 = "0x" + f"{13:064x}"
        store.reserve_finalized((), finalized_block=13, finalized_block_hash=block13)
        bounty = _review(
            _selected_composition(core),
            wins[0],
            marker="bounty",
            block=13,
            decision=DISCOVERY_BOUNTY_ONLY,
        )
        bounty_record = store.record_reviewed_discovery_disposition(
            bounty, authority_block_hash=block13
        )
        assert bounty_record.claim_digest
        assert bounty_record.disposition.decision == "bounty_only"
        assert bounty_record.disposition.review_digest == _h("review:bounty")
        claim = store.discovery_debt_claim_states()[0].claim
        assert claim.principal_units == 50_000
        assert claim.requested_principal_epochs == 7
        assert claim.capped_principal_epochs == 1
        assert claim.awarded_block == wins[0].settlement_block
        assert claim.expires_block == (
            wins[0].settlement_block + 648_000
        )

        block14 = "0x" + f"{14:064x}"
        store.reserve_finalized((), finalized_block=14, finalized_block_hash=block14)
        promotion = _review(
            _selected_composition(core),
            wins[1],
            marker="promotion",
            block=14,
            decision=DISCOVERY_REGISTERED_PROMOTION,
        )
        with pytest.raises(
            IntakeError, match="DiscoveryWinRecord/DiscoveryPromotion"
        ):
            store.record_reviewed_discovery_disposition(
                promotion, authority_block_hash=block14
            )
        assert len(store.reviewed_discovery_dispositions()) == 1
        assert len(store.discovery_debt_claim_states()) == 1

        forged = type(bounty)(
            policy_digest=bounty.policy_digest,
            win_digest=_h("forged varied win"),
            proposal_digest=_h("forged varied proposal"),
            retained_evidence_digest=wins[0].retained_evidence_digest,
            review_digest=_h("forged varied review"),
            hotkey=wins[0].hotkey,
            win_block=wins[0].settlement_block,
            authority_block=14,
            decision=DISCOVERY_BOUNTY_ONLY,
            requested_principal_epochs=1,
            promoted_target_digest="",
        )
        with pytest.raises(IntakeError, match="no retained discovery win"):
            store.record_reviewed_discovery_disposition(
                forged, authority_block_hash=block14
            )

    with _store(tmp_path) as reopened:
        assert reopened.review_pending_discovery_wins() == (wins[1],)
        assert len(reopened.reviewed_discovery_dispositions()) == 1
        assert len(reopened.discovery_debt_claim_states()) == 1


def test_discovery_bounty_cannot_refresh_or_outlive_retained_win(tmp_path) -> None:
    from optima.settlement import plan_settlement

    with _store(tmp_path) as store:
        candidate = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=_h("bounded discovery win"),
            hotkey="bounded-discoverer",
        )
        core = _selected_core(_h("unused bounded family"))
        block10 = "0x" + f"{10:064x}"
        store.activate_finite_debt_policy(
            core, activation_block=10, activation_block_hash=block10
        )
        policy = _selected_composition(core)
        store.activate_incentive_composition(
            policy, activation_block=10, activation_block_hash=block10
        )
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
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
        block11 = "0x" + f"{11:064x}"
        store.reserve_finalized((), finalized_block=11, finalized_block_hash=block11)
        store.commit_settlement(
            lease,
            plan,
            evidence,
            current_block=11,
            current_block_hash=block11,
        )
        win = store.review_pending_discovery_wins()[0]
        expiry = win.settlement_block + policy.discovery_lifetime_blocks
        expiry_hash = "0x" + f"{expiry:064x}"
        store.reserve_finalized(
            (), finalized_block=expiry, finalized_block_hash=expiry_hash
        )

        refreshed = review_discovery_disposition(
            policy,
            win_digest=win.digest,
            proposal_digest=win.proposal_digest,
            retained_evidence_digest=win.retained_evidence_digest,
            review_digest=_h("forged refreshed win block"),
            hotkey=win.hotkey,
            win_block=expiry,
            authority_block=expiry,
            decision=DISCOVERY_BOUNTY_ONLY,
            requested_principal_epochs=1,
        )
        with pytest.raises(IntakeError, match="differs from retained candidate/evidence"):
            store.record_reviewed_discovery_disposition(
                refreshed, authority_block_hash=expiry_hash
            )

        expired = _review(
            policy,
            win,
            marker="expired bounded win",
            block=expiry,
            decision=DISCOVERY_BOUNTY_ONLY,
        )
        with pytest.raises(IntakeError, match="at or after.*expiry"):
            store.record_reviewed_discovery_disposition(
                expired, authority_block_hash=expiry_hash
            )
        assert store.reviewed_discovery_dispositions() == ()
        assert store.discovery_debt_claim_states() == ()
        assert store.expire_review_pending_discovery_wins(
            current_block=expiry,
            current_block_hash=expiry_hash,
        ) == (win,)
        assert store.expire_review_pending_discovery_wins(
            current_block=expiry,
            current_block_hash=expiry_hash,
        ) == ()
        assert store.review_pending_discovery_wins() == ()
        with pytest.raises(IntakeError, match="at or after.*expiry"):
            store.record_reviewed_discovery_disposition(
                expired, authority_block_hash=expiry_hash
            )
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "review_expired"
        assert [
            row["event_type"] for row in store.finite_debt_reward_events()
        ][-1] == "discovery_review_expired"

    with _store(tmp_path) as reopened:
        assert reopened.review_pending_discovery_wins() == ()
        assert reopened._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "review_expired"


@pytest.mark.parametrize(
    "corruption",
    ("pending_as_bounty", "bounty_as_pending", "expired_without_event"),
)
def test_retained_win_reopen_validates_lifecycle_before_status_filter(
    tmp_path, corruption: str,
) -> None:
    with _store(tmp_path) as store:
        candidate, policy, win = _retain_discovery_win(
            store,
            marker=corruption,
        )
        if corruption == "pending_as_bounty":
            store._db.execute(
                "UPDATE settlement_candidates SET status='reviewed_bounty',"
                "reason='reviewed_bounty' WHERE candidate_digest=?",
                (candidate.digest,),
            )
        elif corruption == "bounty_as_pending":
            block12 = "0x" + f"{12:064x}"
            store.reserve_finalized(
                (), finalized_block=12, finalized_block_hash=block12
            )
            store.record_reviewed_discovery_disposition(
                _review(
                    policy,
                    win,
                    marker="lifecycle-cardinality",
                    block=12,
                    decision=DISCOVERY_BOUNTY_ONLY,
                ),
                authority_block_hash=block12,
            )
            store._db.execute(
                "UPDATE settlement_candidates SET status='review_pending',"
                "reason='review_pending' WHERE candidate_digest=?",
                (candidate.digest,),
            )
        else:
            store._db.execute(
                "UPDATE settlement_candidates SET status='review_expired',reason=? "
                "WHERE candidate_digest=?",
                (
                    f"review_expired:{_h('missing lifecycle expiry event')}",
                    candidate.digest,
                ),
            )

    with _store(tmp_path) as reopened:
        expected = (
            "reviewed discovery bounty lifecycle cardinality"
            if corruption == "pending_as_bounty"
            else "review-pending discovery lifecycle cardinality"
            if corruption == "bounty_as_pending"
            else "expired discovery review event authority differs"
        )
        with pytest.raises(IntakeError, match=expected):
            reopened.review_pending_discovery_wins()
        with pytest.raises(IncentiveCompositionStoreError, match=expected):
            reopened._incentive_composition._win_by_digest(win.digest)
        if corruption == "bounty_as_pending":
            with pytest.raises(IntakeError, match=expected):
                reopened.reviewed_discovery_dispositions()


def test_review_pending_win_reopens_exact_typed_settlement_event(tmp_path) -> None:
    from optima.settlement import plan_settlement

    with _store(tmp_path) as store:
        candidate = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=_h("event-bound discovery win"),
            hotkey="event-bound-discoverer",
        )
        core = _selected_core(_h("unused event-bound family"))
        block10 = "0x" + f"{10:064x}"
        store.activate_finite_debt_policy(
            core, activation_block=10, activation_block_hash=block10
        )
        store.activate_incentive_composition(
            _selected_composition(core),
            activation_block=10,
            activation_block_hash=block10,
        )
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
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
        block11 = "0x" + f"{11:064x}"
        store.reserve_finalized((), finalized_block=11, finalized_block_hash=block11)
        store.commit_settlement(
            lease,
            plan,
            evidence,
            current_block=11,
            current_block_hash=block11,
        )
        win = store.review_pending_discovery_wins()[0]
        store._db.execute(
            "UPDATE settlement_events SET event_json='{}' WHERE event_digest=?",
            (win.settlement_event_digest,),
        )
        with pytest.raises(IntakeError, match="settlement event is corrupt"):
            store.review_pending_discovery_wins()


def test_core_policy_upgrade_and_legacy_v1_projection_publication_are_fenced(
    tmp_path,
) -> None:
    from optima.chain.intake import SQLiteWeightPublicationJournal
    from optima.chain.weights import WeightProjection, WeightPublicationRecord
    from optima.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
    )
    from optima.target_catalog import default_target_catalog

    with _store(tmp_path / "active") as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _core, _composition, _activation = _activate_selected(store, candidate)
        block20 = "0x" + f"{20:064x}"
        store.reserve_finalized((), finalized_block=20, finalized_block_hash=block20)
        with pytest.raises(IntakeError, match="core policy upgrades are disabled"):
            store.activate_finite_debt_policy(
                _selected_core(_h("different selected family")),
                activation_block=20,
                activation_block_hash=block20,
            )

        context = GlobalRewardProjectionContext(
            store.scope.digest,
            "validator",
            20,
            block20,
            (MetagraphMember(0, "validator"),),
        )
        with pytest.raises(IntakeError, match="legacy V1 weight projection"):
            store.build_weight_projection(
                policy=EmissionsPolicyManifest(100, 20, 100_000),
                context=context,
                catalogs={candidate.arena_digest: default_target_catalog()},
                netuid=store.scope.netuid,
            )

        projection = WeightProjection(
            _h("scope"),
            307,
            "validator",
            _h("legacy policy"),
            _h("legacy settlement"),
            _h("legacy evaluation"),
            _h("legacy metagraph"),
            (_h("legacy arena"),),
            0,
            20,
            0,
            (),
            (("reserve", PPM),),
        )
        with pytest.raises(IntakeError, match="legacy V1 weight publication"):
            SQLiteWeightPublicationJournal(store, projection)

    with _store(tmp_path / "retained-object") as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        core = _selected_core(_family(candidate))
        block10 = "0x" + f"{10:064x}"
        store.activate_finite_debt_policy(
            core, activation_block=10, activation_block_hash=block10
        )
        projection = WeightProjection(
            _h("object scope"),
            307,
            "validator",
            _h("object policy"),
            _h("object settlement"),
            _h("object evaluation"),
            _h("object metagraph"),
            (_h("object arena"),),
            0,
            10,
            0,
            (),
            (("reserve", PPM),),
        )
        retained_journal = SQLiteWeightPublicationJournal(store, projection)
        store.activate_incentive_composition(
            _selected_composition(core),
            activation_block=10,
            activation_block_hash=block10,
        )
        intent = WeightPublicationRecord(
            projection.digest,
            "intent",
            submit_block=10,
            retry_after_block=20,
            reason="must be fenced after cutover",
        )
        with pytest.raises(IntakeError, match="legacy V1 weight publication"):
            retained_journal.compare_and_swap(None, intent)
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM weight_publications"
        ).fetchone()["n"] == 0

    with _store(tmp_path / "existing-journal") as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        core = _selected_core(_family(candidate))
        block10 = "0x" + f"{10:064x}"
        store.activate_finite_debt_policy(
            core, activation_block=10, activation_block_hash=block10
        )
        projection = WeightProjection(
            _h("existing scope"),
            307,
            "validator",
            _h("existing policy"),
            _h("existing settlement"),
            _h("existing evaluation"),
            _h("existing metagraph"),
            (_h("existing arena"),),
            0,
            10,
            0,
            (),
            (("reserve", PPM),),
        )
        journal = SQLiteWeightPublicationJournal(store, projection)
        journal.compare_and_swap(
            None,
            WeightPublicationRecord(
                projection.digest,
                "pending",
                submit_block=10,
                retry_after_block=20,
                reason="unresolved legacy publication",
            ),
        )
        with pytest.raises(IntakeError, match="explicit cutover"):
            store.activate_incentive_composition(
                _selected_composition(core),
                activation_block=10,
                activation_block_hash=block10,
            )


def test_cutover_terminally_handles_pre_activation_candidates_without_retry_poison(
    tmp_path,
) -> None:
    from optima.settlement import plan_settlement

    def activate_late(store, family_id: str, *, core_block: int = 10) -> int:
        core = _selected_core(family_id)
        core_hash = "0x" + f"{core_block:064x}"
        store.reserve_finalized(
            (), finalized_block=core_block, finalized_block_hash=core_hash
        )
        store.activate_finite_debt_policy(
            core, activation_block=core_block, activation_block_hash=core_hash
        )
        cutover = core_block + 7_200
        cutover_hash = "0x" + f"{cutover:064x}"
        store.reserve_finalized(
            (), finalized_block=cutover, finalized_block_hash=cutover_hash
        )
        store.activate_incentive_composition(
            _selected_composition(core),
            activation_block=cutover,
            activation_block_hash=cutover_hash,
        )
        return cutover

    def commit_at_cutover(store, expected: SettlementCandidate, cutover: int):
        settlement_block = cutover + 1
        lease = store.lease_settlement_cohort(current_block=settlement_block)
        assert lease is not None and lease.candidates == (expected,)
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
        block_hash = "0x" + f"{settlement_block:064x}"
        store.reserve_finalized(
            (), finalized_block=settlement_block, finalized_block_hash=block_hash
        )
        store.commit_settlement(
            lease,
            plan,
            evidence,
            current_block=settlement_block,
            current_block_hash=block_hash,
        )
        return settlement_block

    with _store(tmp_path / "core", expiry_blocks=10_000) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        cutover = activate_late(store, _family(candidate), core_block=11)
        commit_at_cutover(store, candidate, cutover)
        assert store.evaluation_stack(candidate.arena_digest).generation == 1
        assert store.finite_debt_claim_states() == ()
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "crowned"
        assert [
            row["event_type"] for row in store.finite_debt_reward_events()
        ][-1] == "claim_not_issued"

    with _store(tmp_path / "core-before-composition", expiry_blocks=10_000) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        cutover = activate_late(store, _family(candidate))
        commit_at_cutover(store, candidate, cutover)
        assert len(store.finite_debt_claim_states()) == 1
        assert store.finite_debt_claim_states()[0].claim.accepted_crown_block == 10

    with _store(tmp_path / "discovery", expiry_blocks=10_000) as store:
        candidate = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=_h("pre-cutover discovery"),
            hotkey="discoverer",
        )
        cutover = activate_late(store, _h("unused late family"))
        commit_at_cutover(store, candidate, cutover)
        assert store.review_pending_discovery_wins() == ()
        assert store.reviewed_discovery_dispositions() == ()
        assert store.active_reward_claims() == ((), ())
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "review_ineligible"
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM settlement_events WHERE event_type='DISCOVERY_BOUNTY'"
        ).fetchone()["n"] == 1

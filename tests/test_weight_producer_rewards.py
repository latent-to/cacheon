"""Retained PASS attribution and publication clocks in the production producer."""

from dataclasses import replace

import pytest

from cacheon.chain import qualification_settlement as rewards
from cacheon.chain.intake import (
    IntakeError, SQLiteFollowerWeightPublicationJournal, SQLiteWeightPublicationJournal,
)
from cacheon.chain.weight_share import CurrentWeightOffer
from cacheon.chain.weights import WeightPublicationRecord
from cacheon.economics import ArenaRewardAuthority, EconomicsError, project_global_rewards
from cacheon.target_catalog import TargetCatalog
from tests import test_chain_intake as intake
from tests.test_economics import _catalog, _claim, _d, _global_context, _policy, _slot, _stack


@pytest.mark.parametrize("changed", [None, "artifact_digest", "attribution_digest", "contract"])
def test_rebuilt_payload_keeps_original_pass_across_arenas(changed):
    catalog = _catalog()
    old = _stack(catalog, ("slot.a",))
    original = old.entries["slot.a"]
    earned = _claim(old, "slot.a", "alice", 1_100_000)
    before = earned.to_dict()
    if changed == "contract":
        slot = _slot("slot.a")
        catalog = TargetCatalog((replace(slot, contract_ref=replace(
            slot.contract_ref, reference_id="new.reference.v1"
        )), _slot("slot.b")))
    new = _stack(catalog, arena="e")
    carried = replace(original, selected_payload_digest=_d("f"),
                      target_spec_digest=catalog.target_spec_digest("slot.a"))
    if changed in {"artifact_digest", "attribution_digest"}:
        carried = replace(carried, **{changed: _d("9")})
    new = new.with_contribution(carried)
    latest = _claim(new, "slot.b", "bob", 1_200_000, evidence="7")
    authorities = (ArenaRewardAuthority(old, 1, (earned,)), ArenaRewardAuthority(new, 1, (latest,)))
    args = (_policy(), _global_context(), authorities, (earned, latest))
    if changed:
        with pytest.raises(EconomicsError, match="standing claim"):
            project_global_rewards(*args, earned_contributions=(original,))
    else:
        with pytest.raises(EconomicsError, match="standing claim"):
            project_global_rewards(*args)
        projection = project_global_rewards(*args, earned_contributions=(original,))
        assert len(projection.standing) == 2
        assert {row.claim_digest for row in projection.standing} == {earned.digest, latest.digest}
        held = {row.digest: None for row in (earned, latest)}
        credits = [project_global_rewards(
            _policy(), _global_context(block), authorities, (earned, latest),
            earned_contributions=(original,), decay_start_blocks=held,
        ).standing for block in (200, 300)]
        assert credits[0] == credits[1]
    assert earned.to_dict() == before


@pytest.mark.parametrize("reason", ["", "block_inclusion"])
@pytest.mark.parametrize("channel", ["direct", "follower"])
def test_decay_waits_for_fresh_confirmation_and_survives_restart(tmp_path, channel, reason):
    with intake._store(tmp_path) as store, intake._store(tmp_path / "signer") as signer:
        candidate = intake._qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = intake._settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        claims = store.passed_reward_claims()
        retained = [row[0] for row in store._db.execute("SELECT qualification_json FROM settlement_qualifications")]
        context = intake._context("validator", candidate.hotkey)
        projection = store.build_weight_projection(policy=intake.POLICY, context=context, netuid=intake.SCOPE.netuid)
        assert rewards.reward_decay_adjustments(store)[-1]["start_block"] is None
        if channel == "direct":
            journal = SQLiteWeightPublicationJournal(store, projection)
        else:
            journal = SQLiteFollowerWeightPublicationJournal(signer, CurrentWeightOffer.from_legacy_projection(projection))
        previous = None
        for status, update, expected in [("pending", 0, None), ("confirmed", 11, None), ("confirmed", 20, 20), ("confirmed", 30, 20)]:
            included = reason and status == "confirmed"  # the live signer: confirmed_block set, last_update 0
            record = WeightPublicationRecord(
                projection.digest, status, prior_record_digest=previous, submit_block=10, retry_after_block=10,
                confirmed_block=update, confirmed_last_update=0 if included else update, reason=reason if included else "",
            )
            journal.compare_and_swap(previous, record)
            previous = record.digest
            if channel == "follower":
                rewards.reconcile_follower_reward_decay(store, signer.path, validator_hotkey="validator")
            assert rewards.reward_decay_adjustments(store)[-1]["start_block"] == expected
        assert store.passed_reward_claims() == claims
        assert [row[0] for row in store._db.execute("SELECT qualification_json FROM settlement_qualifications")] == retained
    with intake._store(tmp_path) as store:
        if channel == "follower":
            rewards.reconcile_follower_reward_decay(store, signer.path, validator_hotkey="validator")
        assert len(rewards.reward_decay_adjustments(store)) == 2
        claim = store.passed_reward_claims()[0]
        assert claim == claims[0] and claim.crowned_block == candidate.finalized_block
        assert claim.credit_at(120, _policy(), decay_start_block=20) == claim.credit_at(20, _policy(), decay_start_block=20) // 2
        with pytest.raises(IntakeError, match="already fixed"):
            rewards.record_reward_decay_start(store, claim_digest=claim.digest, start_block=30, reason="retry")


def test_legacy_reward_clocks_are_preserved_once(tmp_path):
    with intake._store(tmp_path) as store:
        intake._qualified_settlement_candidate(store)
        rewards.preserve_existing_reward_clocks(store)
        first = store._db.execute("SELECT value FROM metadata WHERE key='reward_decay_legacy_claims'").fetchone()[0]
        intake._qualified_settlement_candidate(store, index=1, marker="next", arena_marker="next")
        rewards.preserve_existing_reward_clocks(store)
        assert store._db.execute("SELECT value FROM metadata WHERE key='reward_decay_legacy_claims'").fetchone()[0] == first
        rewards._hold_unpublished_claims(store, store.passed_reward_claims())
        assert len(rewards.reward_decay_adjustments(store)) == 1
        assert rewards.reward_decay_adjustments(store)[0]["start_block"] is None


def test_wrong_signer_journal_preserves_pending_clock_and_cursor(tmp_path):
    with intake._store(tmp_path) as store, intake._store(tmp_path / "signer") as signer:
        candidate = intake._qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = intake._settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        projection = store.build_weight_projection(
            policy=intake.POLICY, context=intake._context("validator", candidate.hotkey), netuid=intake.SCOPE.netuid,
        )
        journal = SQLiteFollowerWeightPublicationJournal(signer, CurrentWeightOffer.from_legacy_projection(projection))
        journal.compare_and_swap(None, WeightPublicationRecord(
            projection.digest, "confirmed", confirmed_block=12, confirmed_last_update=12,
        ))
        with pytest.raises(IntakeError, match="signer or chain authority"):
            rewards.reconcile_follower_reward_decay(store, signer.path, validator_hotkey="other")
        assert rewards.reward_decay_adjustments(store)[-1]["start_block"] is None
        assert store._db.execute("SELECT value FROM metadata WHERE key='reward_decay_confirmation_cursor'").fetchone() is None
        rewards.reconcile_follower_reward_decay(store, signer.path, validator_hotkey="validator")
        assert rewards.reward_decay_adjustments(store)[-1]["start_block"] == 12

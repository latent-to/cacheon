"""Retained PASS attribution and publication clocks in the production producer.

Decay starting at the first confirmed publication, from the signer's own journal or the
follower's, runs end to end in ``tests/test_validator_weights.py`` and
``tests/test_validator_weight_service.py``.
"""

import json
from dataclasses import replace

import pytest

from cacheon.chain import qualification_settlement as rewards
from cacheon.chain.intake import IntakeError, SQLiteFollowerWeightPublicationJournal
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


def test_migrated_legacy_claims_keep_their_clock_and_new_claims_wait(tmp_path):
    with intake._store(tmp_path) as store:
        intake._qualified_settlement_candidate(store)
        legacy = json.dumps([row.digest for row in store.passed_reward_claims()])
        store._db.execute("INSERT INTO metadata(key,value) VALUES('reward_decay_legacy_claims',?)", (legacy,))
        intake._qualified_settlement_candidate(store, index=1, marker="next", arena_marker="next")
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

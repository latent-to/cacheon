"""A retained complete PASS can earn without changing the evaluation incumbent."""

import pytest

from cacheon.chain.intake import IntakeError
from cacheon.economics import EmissionsPolicyManifest
from cacheon.stack_identity import canonical_digest
from tests.test_chain_intake import (
    _qualified_settlement_candidate, _reserve_one, _settlement_plan, _store,
)


def test_pending_complete_pass_earns_without_settlement_or_stack_change(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        before = store.evaluation_stack(candidate.arena_digest)
        claims = store.passed_reward_claims()
        assert len(claims) == 1
        assert claims[0].hotkey == candidate.hotkey
        assert store.evaluation_stack(candidate.arena_digest) == before
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE reservation_id=?",
            (candidate.reservation_digest,),
        ).fetchone()[0] == "pending"
        assert store.active_reward_claims() == ((), ())


def test_one_pass_earns(tmp_path):
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store, primary_only=True)
        assert len(store.passed_reward_claims()) == 1


def test_missing_pass_evidence_stops_reward_projection(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        store._db.execute(
            "DELETE FROM settlement_qualifications WHERE reservation_id=?",
            (candidate.reservation_digest,),
        )
        with pytest.raises(IntakeError):
            store.passed_reward_claims()


@pytest.mark.parametrize("target", ("forward_pass", "prefix_cache"))
def test_completed_no_decision_does_not_block_later_rewards_or_settlement(tmp_path, target):
    with _store(tmp_path) as store:
        held = _reserve_one(store, index=99, block=9, hotkey="inconclusive")
        store.mark_held(held.reservation_id, "remote_qualification_hold:legacy_no_decision")
        # Persist the result shape produced by commit_remote_qualification_hold.
        store._db.execute(
            "UPDATE reservations SET qualification_evidence_digest=? WHERE reservation_id=?",
            ("e" * 64, held.reservation_id),
        )
        held = store.get(held.reservation_id)
        candidate = _qualified_settlement_candidate(store, target=target)
        assert [claim.hotkey for claim in store.passed_reward_claims()] == [candidate.hotkey]
        assert store.has_pending_settlement()
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        assert store.get(held.reservation_id) == held

    with _store(tmp_path) as reopened:
        assert reopened.get(held.reservation_id) == held
        assert [claim.hotkey for claim in reopened.passed_reward_claims()] == [candidate.hotkey]


@pytest.mark.parametrize("reason,evidence", (
    ("remote_qualification_hold:legacy_no_decision", ""),
    ("operator_hold", "e" * 64),
))
def test_unresolved_hold_still_blocks_later_rewards(tmp_path, reason, evidence):
    with _store(tmp_path) as store:
        held = _reserve_one(store, index=99, block=9)
        store.mark_held(held.reservation_id, reason)
        store._db.execute(
            "UPDATE reservations SET qualification_evidence_digest=? WHERE reservation_id=?",
            (evidence, held.reservation_id),
        )
        _qualified_settlement_candidate(store)
        assert store.passed_reward_claims() == ()
        assert not store.has_pending_settlement()
        assert store.lease_settlement_cohort(current_block=11) is None


@pytest.mark.parametrize("version", ["v1.1", "v1.3", "v1.4", "v1.5", "v1.6"])
def test_policy_upgrade_preserves_numeric_configuration(tmp_path, version):
    policy = EmissionsPolicyManifest(7200, 2160, 100000)
    predecessor = policy.to_dict() | {"policy_version": "cacheon.emissions." + version}
    with _store(tmp_path) as store:
        store._db.execute(
            "INSERT INTO metadata(key,value) VALUES('emissions_policy_digest',?)",
            (canonical_digest("cacheon.economics.policy", predecessor),),
        )
        with pytest.raises(IntakeError, match="policy differs"):
            store._bind_emissions_policy(EmissionsPolicyManifest(7201, 2160, 100000))
        store._bind_emissions_policy(policy)
        assert store._db.execute(
            "SELECT value FROM metadata WHERE key='emissions_policy_digest'",
        ).fetchone()[0] == policy.digest

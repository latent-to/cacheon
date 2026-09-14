"""A retained complete PASS can earn without changing the evaluation incumbent."""

import pytest

from cacheon.chain.intake import IntakeError
from cacheon.economics import EmissionsPolicyManifest
from cacheon.stack_identity import canonical_digest
from tests.test_chain_intake import _qualified_settlement_candidate, _store


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


@pytest.mark.parametrize("version", ["v1.1", "v1.3", "v1.4", "v1.5", "v1.6", "v1.7"])
def test_policy_upgrade_preserves_numeric_configuration(tmp_path, version):
    policy = EmissionsPolicyManifest(7200, 2160, 100000)
    predecessor = policy.to_dict() | {"policy_version": "cacheon.emissions." + version}
    predecessor.pop("frontier_awards_from_block")
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


def test_waiting_for_acceptance_does_not_consume_decay_or_renew_on_baseline_change(tmp_path):
    from cacheon.stack_manifest import EvaluationStackManifest
    from cacheon.chain.qualification_settlement import passed_reward_claims
    policy = EmissionsPolicyManifest(100, 50, 0)
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store, retained_block=410, submission_block=10)
        claims, accepted, baselines, order = passed_reward_claims(store)
        claim = claims[0]
        assert claim.crowned_block == 10
        assert accepted[claim.digest] == 410
        full = claim.credit_at(10, policy)
        assert claim.credit_at(410, policy, accepted_block=accepted[claim.digest]) == full
        assert claim.credit_at(510, policy, accepted_block=accepted[claim.digest]) == full // 2
        store.initialize_evaluation_stack(EvaluationStackManifest.from_dict(store.evaluation_stacks()[0].manifest.to_dict() | {"arena_digest": "e" * 64}), tree_digest="f" * 64)
        assert passed_reward_claims(store) == (claims, accepted, baselines, order)

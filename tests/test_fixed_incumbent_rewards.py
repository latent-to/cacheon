"""A retained PASS pair can earn without changing the evaluation incumbent."""

import pytest

from cacheon.chain.intake import IntakeError
from cacheon.economics import EmissionsPolicyManifest
from cacheon.stack_identity import canonical_digest
from tests.test_chain_intake import _qualified_settlement_candidate, _store


def test_pending_pair_earns_without_settlement_or_stack_change(tmp_path):
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


def test_one_pass_does_not_earn(tmp_path):
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store, primary_only=True)
        assert store.passed_reward_claims() == ()


def test_missing_pair_evidence_stops_reward_projection(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        store._db.execute(
            "DELETE FROM settlement_qualifications WHERE reservation_id=?",
            (candidate.reservation_digest,),
        )
        with pytest.raises(IntakeError):
            store.passed_reward_claims()


@pytest.mark.parametrize("version", ["v1.1", "v1.3", "v1.4"])
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

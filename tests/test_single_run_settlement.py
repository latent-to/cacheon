"""One complete qualification settles, rewards, and survives a controller restart."""

import json
from pathlib import Path

import pytest

from cacheon.chain.intake import IntakeError
from cacheon.settlement import SettlementCandidate, SettlementEvidence
from tests.test_chain_intake import _qualified_settlement_candidate, _settlement_plan, _store


@pytest.mark.parametrize("target", ["activation.silu_and_mul", "norm.rmsnorm"])
def test_one_attempt_reaches_settlement_and_rewards_without_reproduction(tmp_path, target):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store, target=target)
        assert candidate.reproduction is None
        assert candidate.speedup == candidate.primary.speedup
        assert store.get(candidate.reservation_digest).status == "qualified"
        assert store.preview_evaluation_claim(stage="screen", max_members=1) == ()
        assert store.preview_evaluation_claim(stage="qualification", max_members=1) == ()
        assert store._db.execute("SELECT COUNT(*) FROM qualification_dispositions").fetchone()[0] == 1
        receipt = store.reopen_settlement_evidence(candidate)
        assert receipt.reproduction_attempt_ref is None
        assert SettlementCandidate.from_dict(candidate.to_dict()) == candidate
        assert SettlementEvidence.from_dict(receipt.to_dict()) == receipt
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        assert store._db.execute("SELECT status FROM settlement_candidates").fetchone()[0] == "crowned"
        claims = store.passed_reward_claims()
        assert len(claims) == 1 and claims[0].hotkey == candidate.hotkey
        assert store._db.execute("SELECT COUNT(*) FROM settlement_qualifications").fetchone()[0] == 1


@pytest.mark.parametrize("target", ["activation.silu_and_mul", "norm.rmsnorm"])
def test_restart_accepts_old_controller_primary_without_repeating_it(tmp_path, target):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store, target=target)
        original = tuple(store._db.execute("SELECT * FROM settlement_qualifications").fetchone())
        # The previous controller had retained this PASS but queued reproduction.
        store._db.execute("DELETE FROM settlement_candidates")
        store._db.execute(
            "UPDATE reservations SET status='reproduction_pending',decision='',reason='reproduction_pending'"
        )
    with _store(tmp_path) as reopened:
        assert reopened.get(candidate.reservation_digest).status == "qualified"
        assert tuple(reopened._db.execute("SELECT * FROM settlement_qualifications").fetchone()) == original
        assert reopened.preview_evaluation_claim(stage="qualification", max_members=1) == ()
        restored = SettlementCandidate.from_dict(json.loads(
            reopened._db.execute("SELECT candidate_json FROM settlement_candidates").fetchone()[0]
        ))
        assert restored == candidate
        assert len(reopened.passed_reward_claims()) == 1
    with _store(tmp_path) as reopened:
        assert reopened._db.execute("SELECT COUNT(*) FROM settlement_candidates").fetchone()[0] == 1


def test_restart_cannot_credit_a_primary_whose_raw_evidence_is_missing(tmp_path):
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store)
        root, ref = store._db.execute(
            "SELECT evidence_root,attempt_ref_json FROM settlement_qualifications"
        ).fetchone()
        value = json.loads(ref)
        (Path(root) / value["domain"] / value["sha256"][:2] / value["sha256"]).unlink()
        store._db.execute("DELETE FROM settlement_candidates")
        store._db.execute("UPDATE reservations SET status='reproduction_pending',decision=''")
    with pytest.raises(IntakeError, match="cannot reopen"):
        _store(tmp_path)

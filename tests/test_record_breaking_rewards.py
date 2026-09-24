"""Arrival-ordered reward records preserve PASS evidence and pre-policy claims."""

import json

import pytest

from cacheon.chain.intake import IntakeError
from dashboard.winners import qualified_winners
from tests.test_chain_intake import _qualified_settlement_candidate, _store


@pytest.mark.parametrize("target", ["activation.silu_and_mul", "norm.rmsnorm"])
def test_only_threshold_records_earn_in_submission_order(tmp_path, target):
    with _store(tmp_path) as store:
        candidates = [
            _qualified_settlement_candidate(
                store, index=i, marker=str(i), target=target,
                speedups=(score, score), retained_block=100-i,
            ) for i, score in enumerate(("1.0277661067436534", "1.0256515334199241",
                                        "1.0277661067436534", "1.03", "1.0403"))
        ]
        retained = list(store._db.execute("SELECT * FROM settlement_qualifications"))
        expected = {candidates[0].hotkey, candidates[4].hotkey}
        assert {c.hotkey for c in store.passed_reward_claims()} == expected
        assert {r["hotkey"] for r in qualified_winners(store._db)} == expected
        assert all(store.get(c.reservation_digest).decision == "PASS" for c in candidates)
        assert list(store._db.execute("SELECT * FROM settlement_qualifications")) == retained
    with _store(tmp_path) as store:
        assert {c.hotkey for c in store.passed_reward_claims()} == expected


@pytest.mark.parametrize("margin,score,earns", [
    ("0.01", "1.060499999999", False), ("0.01", "1.0605", True),
    ("0.02", "1.0605", False), ("0.02", "1.071", True),
])
def test_sealed_margin_and_exact_threshold(tmp_path, margin, score, earns):
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store)
        payload = json.dumps({"speed_witness": {"resident_policy": {"min_margin": margin}}}).encode()
        later = _qualified_settlement_candidate(
            store, index=1, marker="later", speedups=(score, score),
            attempt_payloads=(payload, payload),
        )
        assert (later.hotkey in {c.hotkey for c in store.passed_reward_claims()}) == earns


def test_pre_policy_runtime_keeps_all_claims_and_clocks(tmp_path):
    with _store(tmp_path) as store:
        first = _qualified_settlement_candidate(store)
        later = _qualified_settlement_candidate(store, index=1, marker="old", speedups=("1.02", "1.02"))
        store._db.execute("INSERT INTO metadata(key,value) VALUES('reward_grandfathered_runtimes',?)",
                          (json.dumps([first.incumbent_manifest.runtime_digest]),))
        assert {c.hotkey for c in store.passed_reward_claims()} == {first.hotkey, later.hotkey}
        assert {r["hotkey"] for r in qualified_winners(store._db)} == {first.hotkey, later.hotkey}
        assert {c.crowned_block for c in store.passed_reward_claims()} == {10}


def test_another_baseline_does_not_compete_and_missing_margin_is_an_error(tmp_path):
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store, speedups=("1.5", "1.5"))
        other = _qualified_settlement_candidate(store, index=1, marker="other", arena_marker="other")
        assert other.hotkey in {c.hotkey for c in store.passed_reward_claims()}
        _qualified_settlement_candidate(store, index=2, marker="broken", arena_marker="other",
                                        speedups=("1.1", "1.1"), attempt_payloads=(b"{}", b"{}"))
        with pytest.raises(IntakeError, match="retained margin"):
            store.passed_reward_claims()


def test_grandfathering_does_not_exempt_a_new_runtime(tmp_path, monkeypatch):
    from tests import test_chain_intake as intake

    with _store(tmp_path) as store:
        old = _qualified_settlement_candidate(store)
        store._db.execute("INSERT INTO metadata VALUES('reward_grandfathered_runtimes',?)",
                          (json.dumps([old.incumbent_manifest.runtime_digest]),))
        original_hash = intake._h
        monkeypatch.setattr(intake, "_h", lambda value: original_hash(
            "mtp-runtime" if value == "runtime" else value))
        current = _qualified_settlement_candidate(store, index=1, marker="mtp", arena_marker="mtp")
        slower = _qualified_settlement_candidate(store, index=2, marker="slower", arena_marker="mtp",
                                                 speedups=("1.02", "1.02"))
        assert {c.hotkey for c in store.passed_reward_claims()} == {old.hotkey, current.hotkey}
        assert slower.hotkey not in {r["hotkey"] for r in qualified_winners(store._db)}

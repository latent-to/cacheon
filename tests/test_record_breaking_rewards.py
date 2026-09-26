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
        from cacheon.chain.qualification_settlement import passed_reward_evidence
        scores = {}
        claims, _ = passed_reward_evidence(store, score_speedups=scores)
        assert scores == {claim.digest: claim.speedup_ppm for claim in claims}


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


def test_dashboard_does_not_reinterpret_retained_audit_schemas(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert len(store.passed_reward_claims()) == 1
        payload = candidate.to_dict()
        payload["primary"]["audit_policy"] = {"retained_schema": "another-runtime"}
        store._db.execute("UPDATE settlement_candidates SET candidate_json=?",
                          (json.dumps(payload),))
        assert [r["hotkey"] for r in qualified_winners(store._db)] == [candidate.hotkey]
        # The write-side evidence authority still rejects altered candidate bytes.
        with pytest.raises((IntakeError, ValueError)):
            store.passed_reward_claims()


@pytest.mark.parametrize("target", ["activation.silu_and_mul", "norm.rmsnorm"])
def test_scoring_uses_queue_record_without_rewriting_claims_or_clocks(tmp_path, target):
    from dataclasses import replace
    from decimal import Decimal
    from cacheon.chain.evaluation_order import reward_comparisons
    from cacheon.chain.qualification_settlement import _reward_projection_inputs, record_reward_decay_start
    from cacheon.economics import project_global_rewards
    from tests.test_chain_intake import POLICY, SCOPE, _context, _settlement_plan

    with _store(tmp_path) as store:
        first = _qualified_settlement_candidate(store, target=target, speedups=("1.1", "1.1"), retained_block=12)
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        later = _qualified_settlement_candidate(
            store, index=1, marker="later", target=target, initialize_stack=False,
            speedups=("1.12", "1.12"), retained_block=11)
        claims = store.passed_reward_claims()
        for claim in claims:
            record_reward_decay_start(store, claim_digest=claim.digest, start_block=11, reason="Published")
        before = [dict(r) for r in store._db.execute("SELECT * FROM settlement_candidates")]
        comparisons = reward_comparisons(store._db)
        comparison = comparisons[later.reservation_digest]
        assert comparison["previous_best_reservation_id"] == first.reservation_digest
        assert comparison["relative_speedup"] == Decimal("1.12") / Decimal("1.1")
        inputs = _reward_projection_inputs(store)
        scores = inputs["score_speedups"]
        by_key = {c.hotkey: c for c in claims}
        assert scores[by_key[first.hotkey].digest] == 1_100_000
        assert scores[by_key[later.hotkey].digest] == 1_018_181
        context = _context("validator", first.hotkey, later.hotkey)
        for key in ("states", "standing_claims", "adjustments", "grandfathered_runtimes"):
            inputs.pop(key)
        projection = project_global_rewards(POLICY, context, **inputs)
        credits = {row.claim_digest: row.credit for row in projection.standing}
        for claim in claims:
            expected = replace(claim, speedup_ppm=scores[claim.digest]).credit_at(12, POLICY, decay_start_block=11)
            assert credits[claim.digest] == expected
        assert projection.weights_by_hotkey[first.hotkey] > projection.weights_by_hotkey[later.hotkey]
        served = store.build_weight_projection(policy=POLICY, context=context, netuid=SCOPE.netuid)
        assert dict(served.weights_ppm) == projection.weights_by_hotkey
        assert store.passed_reward_claims() == claims
        assert [dict(r) for r in store._db.execute("SELECT * FROM settlement_candidates")] == before
        display = {r["reservation_id"]: r for r in qualified_winners(store._db)}[later.reservation_digest]
        assert display["relative_improvement_pct"] == pytest.approx(100 * (1.12 / 1.10 - 1))
        assert display["score_improvement_pct"] == 1.8181
    with _store(tmp_path) as store:
        assert store.passed_reward_claims() == claims
        assert _reward_projection_inputs(store)["score_speedups"] == scores


def test_scoring_compares_all_slots_and_subthreshold_records_but_not_later_rows(tmp_path):
    from decimal import Decimal
    from cacheon.chain.evaluation_order import reward_comparisons

    with _store(tmp_path) as store:
        candidates = [_qualified_settlement_candidate(
            store, index=i, marker=str(i), target=target, speedups=(score, score),
        ) for i, (target, score) in enumerate([
            ("activation.silu_and_mul", "1.1"), ("norm.rmsnorm", "1.105"),
            ("activation.silu_and_mul", "1.12"), ("norm.rmsnorm", "1.5")])]
        store.passed_reward_claims()
        comparisons = reward_comparisons(store._db)
        second, third = (comparisons[c.reservation_digest] for c in candidates[1:3])
        assert not second["reward_eligible"]
        assert third["previous_best_reservation_id"] == candidates[1].reservation_digest
        assert third["score_speedup"] == Decimal("1.12") / Decimal("1.105")
        assert third["reward_eligible"]
        assert comparisons[candidates[0].reservation_digest]["score_speedup"] == Decimal("1.1")

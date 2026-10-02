"""Arrival-ordered reward records preserve PASS evidence and pre-policy claims."""

import json

import pytest

from cacheon.chain.intake import IntakeError
from dashboard.winners import qualified_winners
from tests.test_chain_intake import _qualified_settlement_candidate, _store


@pytest.mark.parametrize("static", [False, True])
def test_unpaid_crown_keeps_pass_authority_without_blocking_rewards(tmp_path, monkeypatch, static):
    from dataclasses import replace
    from types import SimpleNamespace
    from cacheon.arena_allocation import ArenaAllocation
    from cacheon.chain import arena_weight_projection as allocation
    from cacheon.chain.qualification_settlement import build_weight_projection
    from cacheon.chain.reward_checkpoint import RewardCheckpoint
    from tests.test_chain_intake import POLICY, SCOPE, _context, _settlement_plan

    context = _context("validator", "minerpaid", "minerunpaid")
    schedule = ArenaAllocation.from_dict({
        "activation_block": 12, "burn_hotkey": "validator",
        "sources": {"only": "/configs/only.json"},
        "history": [{"from_block": block, "weights_ppm": {"only": 1_000_000}}
                    for block in (0, 12)],
    })
    stage = SimpleNamespace(attribution_hotkey="validator", burn_hotkey="validator",
                            arena_allocation_path=tmp_path / "allocation.json" if static else None,
                            confirmation_journal=None)
    monkeypatch.setattr(allocation, "load_allocation", lambda path: schedule)
    with _store(tmp_path / "signer") as signer:
        journal = signer.path
    stage.confirmation_journal = journal
    checkpoint = RewardCheckpoint(tmp_path / "rewards.json", policy=POLICY, scope=SCOPE, stage=stage)
    with _store(tmp_path) as store:
        paid = _qualified_settlement_candidate(store, marker="paid", speedups=("1.1", "1.1"))
        unpaid = _qualified_settlement_candidate(store, marker="unpaid", index=1,
                    initialize_stack=False, speedups=("1.105", "1.105"))
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        assert store.active_reward_claims()[0][0].hotkey == unpaid.hotkey
        assert {c.hotkey for c in store.passed_reward_claims()} == {paid.hotkey}
        monkeypatch.setattr(allocation, "load_config", lambda path: SimpleNamespace(
            intake_db=store.path, policy=store.policy, scope=store.scope, digest="a" * 64))
        if static:
            allocation.build_static_projection(store, allocation=schedule, policy=POLICY,
                context=replace(context, current_block=11), netuid=SCOPE.netuid,
                confirmation_journal=journal)
        def project(capture=None):
            if static:
                return allocation.build_static_projection(store, allocation=schedule, policy=POLICY,
                    context=context, netuid=SCOPE.netuid, confirmation_journal=journal, capture=capture)
            return build_weight_projection(store, policy=POLICY, context=context,
                                           netuid=SCOPE.netuid, capture=capture)
        result = project(checkpoint.capture)
        checkpoint.save()
        assert dict(result.weights_ppm) == {paid.hotkey: 1_000_000}
        assert result.crown_count == 1
        # Missing actual PASS authority still fails, even though the crown is unpaid.
        store._db.execute("UPDATE reservations SET decision='FAIL' WHERE reservation_id=?",
                          (unpaid.reservation_digest,))
        with pytest.raises(IntakeError, match="no longer has standing authority"):
            project()
        store._db.execute("UPDATE reservations SET decision='PASS' WHERE reservation_id=?",
                          (unpaid.reservation_digest,))
        # An unpaid PASS is still reopened, rather than trusted from the crown row.
        store._db.execute("UPDATE settlement_candidates SET settlement_evidence_digest=? WHERE reservation_id=?",
                          ("0" * 64, unpaid.reservation_digest))
        with pytest.raises(IntakeError, match="retained evidence"):
            project()
    # Recovery reloads the same validation/payment separation without an intake store.
    checkpoint = RewardCheckpoint(checkpoint.path, policy=POLICY, scope=SCOPE, stage=stage)
    retained = checkpoint.project(replace(context, current_block=100))
    assert dict(retained.weights_ppm) == {paid.hotkey: 1_000_000}
    assert retained.crown_count == 1


@pytest.mark.parametrize("target", ["forward_pass", "prefix_cache"])
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


@pytest.mark.parametrize("target", ["forward_pass", "prefix_cache"])
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


def test_scoring_compares_all_slots_without_advancing_unpaid_or_later_records(tmp_path):
    from decimal import Decimal
    from cacheon.chain.evaluation_order import reward_comparisons

    with _store(tmp_path) as store:
        candidates = [_qualified_settlement_candidate(
            store, index=i, marker=str(i), target=target, speedups=(score, score),
        ) for i, (target, score) in enumerate([
            ("forward_pass", "1.1"), ("prefix_cache", "1.105"),
            ("forward_pass", "1.12"), ("prefix_cache", "1.5")])]
        store.passed_reward_claims()
        comparisons = reward_comparisons(store._db)
        second, third = (comparisons[c.reservation_digest] for c in candidates[1:3])
        assert not second["reward_eligible"]
        assert third["previous_best_reservation_id"] == candidates[0].reservation_digest
        assert third["score_speedup"] == Decimal("1.12") / Decimal("1.1")
        assert third["reward_eligible"]
        assert comparisons[candidates[0].reservation_digest]["score_speedup"] == Decimal("1.1")


def test_statistical_records_pay_one_and_a_half_percent_over_the_previous_best(tmp_path):
    from decimal import Decimal
    from cacheon.chain.evaluation_order import reward_comparisons
    from cacheon.eval.goodput_runtime import GoodputPolicy, GoodputReadSet
    from tests.test_service_statistics import CONTRACT, WORK, _read

    policy = GoodputPolicy(CONTRACT, 1.0, 0.0003, 0.0, 0.0, 0.01, 0.0001)
    with _store(tmp_path) as store:
        candidates = []
        for index, score in enumerate((1.05, 1.06, 1.07)):
            baseline, candidate = [], []
            for window, lane in enumerate(("A", "A", "B", "B"), 1):
                baseline.append(_read("incumbent", lane, window, 100))
                candidate.append(_read("candidate", "B" if lane == "A" else "A", window, 100/score))
            reads = GoodputReadSet(tuple(baseline), tuple(candidate),
                                   tuple((root, *counts) for root, counts in WORK.items()), 4)
            payload = json.dumps({"speed_witness": {
                "resident_policy": {"version": 17, "goodput": policy.to_dict()},
                "goodput": reads.to_dict(),
            }}).encode()
            candidates.append(_qualified_settlement_candidate(
                store, index=index, marker=str(index), speedups=(str(score), str(score)),
                attempt_payloads=(payload, payload),
            ))
        store.passed_reward_claims()
        comparisons = reward_comparisons(store._db)
        first, unpaid, paid = (comparisons[c.reservation_digest] for c in candidates)
        assert first["reward_eligible"] and not unpaid["reward_eligible"] and paid["reward_eligible"]
        assert paid["previous_best_reservation_id"] == candidates[0].reservation_digest
        assert paid["score_speedup"] == Decimal("1.07") / Decimal("1.05")
        assert {c.hotkey for c in store.passed_reward_claims()} == {candidates[i].hotkey for i in (0, 2)}

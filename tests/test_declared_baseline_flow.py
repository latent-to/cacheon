"""CPU-only public intake, mocked evaluator completion, durable ranking and weight flow."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cacheon.chain.declared_baseline import commission_baseline, current_baseline
from cacheon.chain.eval_cost import EvalCostPolicy
from cacheon.chain.intake import IntakeError
from cacheon.chain.payload import encode_payload
from cacheon.chain.submission_ranking import current_winners
from cacheon.chain.validator_loop import _finalized_arrivals
from dashboard.baseline_api import baseline_head, competition_details
from tests import test_chain_intake as fixtures


def arrive(store, *, index=0, hotkey="miner", block=10, baseline_ref=None):
    """Exercise the actual wire decoder and finalized-arrival admission owner."""

    head = current_baseline(store)
    ref = head.manifest.digest if baseline_ref is None and head else baseline_ref or ""
    data = encode_payload(f"{index + 1:064x}", f"https://example.com/{index}.tar.gz", baseline_ref=ref)
    reveal = SimpleNamespace(hotkey=hotkey, block=block, block_hash=fixtures._bh(block),
                             event_index=index, data=data)
    rows = _finalized_arrivals(SimpleNamespace(reveals=(reveal,)), netuid=307,
                               eval_cost_policy=EvalCostPolicy(amount_rao=0))
    return store.reserve_finalized(rows, finalized_block=block, finalized_block_hash=fixtures._bh(block))[0]


@pytest.fixture
def evaluator(monkeypatch):
    """Mock only GPU measurements; all intake, ranking, DB and projection code is real."""

    measurements = {"rate": Decimal("110"), "required": Decimal("1.01"), "calls": 0}

    def reserve(store, **kwargs):
        if current_baseline(store) is None:
            state, = store.evaluation_stacks()
            commission_baseline(store, state.manifest, state.tree_digest)
        return arrive(store, **kwargs)

    original = fixtures._qualified_settlement_candidate

    def complete(store, **kwargs):
        measurements["calls"] += 1
        return original(store, measured_rate=str(measurements["rate"]), **kwargs)

    monkeypatch.setattr(fixtures, "_reserve_one", reserve)
    monkeypatch.setattr(fixtures, "_qualified_settlement_candidate", complete)
    return measurements


@pytest.mark.parametrize("target", ["activation.silu_and_mul", "norm.rmsnorm"])
def test_unfinalized_competition_persists_and_earns_before_commission(tmp_path, evaluator, target):
    with fixtures._store(tmp_path) as store:
        first = fixtures._qualified_settlement_candidate(store, target=target)
        head = current_baseline(store)
        assert head.manifest.entries == {}
        assert store.active_reward_claims() == ((), ())
        projection = store.build_weight_projection(
            policy=fixtures.POLICY, context=fixtures._context("miner", "validator"), netuid=307)
        assert dict(projection.weights)["miner"] == 1.0
        assert store.evaluation_stack(first.arena_digest).generation == 0

        evaluator["rate"] = Decimal("110.5")
        loser = fixtures._qualified_settlement_candidate(
            store, index=1, marker="near", target=target, speedups=("1.2", "1.2"))
        assert competition_details(store._db, loser.reservation_digest)["won"] is False
        assert "minernear" not in {row.hotkey for row in store.passed_reward_claims()}

        evaluator["rate"] = Decimal("115")
        winner = fixtures._qualified_settlement_candidate(
            store, index=2, marker="fast", target=target, speedups=("1.15", "1.15"))
        details = competition_details(store._db, winner.reservation_digest)
        assert details["won"] and details["stale"]
        assert details["baseline_ref"] == head.manifest.digest
        assert details["competitor_id"] == loser.reservation_digest
        assert current_winners(store)[0]["reservation_id"] == winner.reservation_digest
        assert evaluator["calls"] == 3  # No A-versus-C evaluator run.
        projection = store.build_weight_projection(
            policy=fixtures.POLICY, context=fixtures._context("miner", "minerfast", "validator"), netuid=307)
        assert dict(projection.weights)["minerfast"] > 0
        assert current_baseline(store) == head
        saved = [tuple(row) for row in store._db.execute("SELECT * FROM submission_rankings")]
    with fixtures._store(tmp_path) as reopened:
        assert [tuple(row) for row in reopened._db.execute("SELECT * FROM submission_rankings")] == saved
        assert competition_details(reopened._db, winner.reservation_digest) == details
        assert evaluator["calls"] == 3


def test_baseline_advance_rejects_new_stale_arrivals_and_preserves_queued_work(tmp_path, evaluator):
    with fixtures._store(tmp_path) as store:
        winner = fixtures._qualified_settlement_candidate(store)
        old = current_baseline(store)
        queued = arrive(store, index=1, hotkey="queued")
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan, evidence = fixtures._settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        accepted = store.evaluation_stack(winner.arena_digest)
        commission_baseline(store, accepted.manifest, accepted.tree_digest)
        head = current_baseline(store)
        assert head.generation > old.generation
        assert baseline_head(store._db)["baseline_ref"] == head.manifest.digest
        assert store.reservation_baseline_segment(queued.reservation_id) == old
        stale = arrive(store, index=2, hotkey="stale", baseline_ref=old.manifest.digest)
        assert stale.status == "failed" and stale.reason == "stale_baseline_ref"
        assert store.reservation_baseline_segment(stale.reservation_id) is None
        fresh = arrive(store, index=3, hotkey="fresh")
        assert fresh.status == "reserved"
        assert store.reservation_baseline_segment(fresh.reservation_id) == head
        with pytest.raises(IntakeError, match="cannot be rebound"):
            store._bind_reservation_baseline_segment(queued.reservation_id, head, reason="test")
        assert not competition_details(store._db, winner.reservation_digest)["stale"]


def test_missing_baseline_never_enters_queue(tmp_path):
    with fixtures._store(tmp_path) as store:
        row = arrive(store, baseline_ref="")
        assert row.status == "failed" and row.reason == "missing_baseline_ref"
        assert store.pending() == ()


def test_commissioned_import_does_not_invent_a_reward_claim(tmp_path, evaluator):
    from cacheon.chain.ranked_rewards import reward_authorities
    with fixtures._store(tmp_path) as store:
        candidate = fixtures._qualified_settlement_candidate(store)
        state = store.evaluation_stack(candidate.arena_digest)
        manifest = type(candidate.candidate_manifest).from_dict(
            candidate.candidate_manifest.to_dict() | {"arena_digest": "a" * 64})
        imported = replace(state, arena_digest=manifest.arena_digest, manifest=manifest)
        assert imported.generation == 0 and imported.manifest.entries
        assert reward_authorities(store, (imported,), store.passed_reward_claims()) == []


def test_quote_and_payment_remark_bind_baseline():
    from cacheon.chain.eval_cost import EvalCostRequest, quote_eval_cost, encode_payment_remark, EvalCostError
    request = EvalCostRequest(307, "miner", "a" * 64, baseline_ref="b" * 64)
    quote = quote_eval_cost(request, policy=EvalCostPolicy(destination="owner"))
    assert quote.baseline_ref == request.baseline_ref
    assert '"baseline_ref":"' + "b" * 64 in encode_payment_remark(request, quote)
    with pytest.raises(EvalCostError, match="baseline differs"):
        encode_payment_remark(replace(request, baseline_ref="c" * 64), quote)


def test_standalone_producer_reopens_scores_after_evaluator_exits(tmp_path, evaluator, monkeypatch):
    from cacheon import chain
    from cacheon.chain import weight_share
    from cacheon.chain.standing_weights_stage import WeightsStageConfig, compose_weight_offer_push
    from cacheon.chain.weight_push_auth import PushCredentialSet, mint_push_credential, write_push_credentials

    with fixtures._store(tmp_path) as store:
        candidate = fixtures._qualified_settlement_candidate(store)
        score = dict(store._db.execute("SELECT * FROM submission_rankings").fetchone())
        assert score["candidate_score"] == "110"
        assert score["required_ratio"] == "1.01"
        assert score["won"] == 1
        assert store.active_reward_claims() == ((), ())
    credentials = tmp_path / "push.json"
    write_push_credentials(credentials, PushCredentialSet((mint_push_credential(credential_id="test"),)))
    monkeypatch.setattr(chain, "connect", lambda *args, **kwargs: object())
    monkeypatch.setattr(chain, "read_finalized_head", lambda _: (12, fixtures._bh(12)))
    monkeypatch.setattr(chain, "fetch_metagraph", lambda *args: SimpleNamespace(
        uids=[0, 1], hotkeys=["miner", "validator"], block=12, block_hash=fixtures._bh(12)))
    offers = []
    monkeypatch.setattr(weight_share, "push_current_weights", lambda url, offer, **kwargs:
                        offers.append(offer) or {"status": "accepted"})
    stage = WeightsStageConfig("wss://example.invalid", "", "http://example.invalid", credentials,
                               "validator", 100, 20, 100_000, 1, "validator")
    publish = compose_weight_offer_push(stage, store_factory=lambda: fixtures._store(tmp_path),
                                        scope=fixtures.SCOPE)
    result = publish()
    assert result.status == "accepted"
    assert len(offers) == 1
    assert dict(offers[0].projection.weights_ppm)["miner"] == 1_000_000
    assert evaluator["calls"] == 1
    with fixtures._store(tmp_path) as store:
        assert dict(store._db.execute("SELECT * FROM submission_rankings").fetchone()) == score
        assert store.evaluation_stack(candidate.arena_digest).generation == 0


def test_comparison_context_survives_baseline_advance_but_separates_measurement_contracts():
    from cacheon.chain.submission_ranking import comparison_context
    from cacheon.eval.calibration import CalibrationContext
    from tests.test_crossover_runtime import _policy_v8

    context = CalibrationContext(**{field: fixtures._h(field) for field in CalibrationContext.__dataclass_fields__})
    policy = _policy_v8(calibration_context_digest=context.digest)
    witness = SimpleNamespace(calibration_context_digest=context.digest, resident_policy=policy)
    original = comparison_context(context, witness)

    def identity(updated, updated_policy=policy):
        return comparison_context(updated, SimpleNamespace(calibration_context_digest=updated.digest,
            resident_policy=replace(updated_policy, calibration_context_digest=updated.digest)))

    assert identity(replace(context, arena_digest="a" * 64, reference_manifest_digest="b" * 64)) == original
    for field in ("logical_hardware_digest", "model_content_digest", "workload_digest", "runtime_digest"):
        assert identity(replace(context, **{field: "c" * 64})) != original
    assert identity(context, replace(policy, min_margin=0.02)) == original
    assert identity(context, replace(policy, min_windows=4)) != original
    with pytest.raises(ValueError, match="calibration context differs"):
        comparison_context(replace(context, workload_digest="f" * 64), witness)


def test_incorporated_winner_is_compared_directly_not_again_using_its_old_rate(tmp_path, evaluator):
    with fixtures._store(tmp_path) as store:
        first = fixtures._qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = fixtures._settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        head = store.evaluation_stack(first.arena_digest)
        commission_baseline(store, head.manifest, head.tree_digest)
        evaluator["rate"] = Decimal("108")
        winner = fixtures._qualified_settlement_candidate(store, index=1, marker="new-baseline",
            incumbent_state=head, initialize_stack=False, speedups=("1.1", "1.1"))
        details = competition_details(store._db, winner.reservation_digest)
        assert details["won"] and not details["stale"]
        assert current_winners(store)[0]["reservation_id"] == winner.reservation_digest
        assert evaluator["calls"] == 2


def test_shared_intake_policy_keeps_the_sealed_dispatch_shape() -> None:
    from cacheon.chain.intake import IntakePolicy
    from cacheon.chain import mainnet_screen_dispatcher as dispatcher_module

    fields = (
        "epoch_blocks",
        "cutoff_blocks",
        "max_pending",
        "max_per_hotkey_epoch",
        "max_per_target_epoch",
        "max_transport_retries",
        "max_qualification_retries",
        "max_cohort",
        "expiry_blocks",
    )
    assert tuple(IntakePolicy.__dataclass_fields__) == fields
    assert dispatcher_module._POLICY_FIELDS == frozenset(fields)


@pytest.mark.parametrize("target,duration", [("activation.silu_and_mul", 0.90), ("norm.rmsnorm", 0.85)])
def test_operator_quality_correction_uses_original_competitive_speed(tmp_path, target, duration):
    from cacheon.chain.submission_ranking import measured_speed
    from cacheon.eval.evidence_store import publish_evidence
    from cacheon.eval.qualification_runner import ResidentSpeedWitness
    from cacheon.stack_identity import canonical_digest, canonical_json_bytes
    from tests.test_crossover_runtime import _rig, _speed, _policy_v8

    plan, baseline, candidate, mount, *_ = _rig(tmp_path / "runtime", (duration,),
                                                policy=_policy_v8(), timed_batches=3)
    witness = ResidentSpeedWitness.from_evidence(_speed(plan, baseline, candidate, mount), plan)
    schema = "cacheon.qualification.operator-quality-correction.v1"
    reservation_id = fixtures._h(target)
    summary = {"decision": "PASS", "quality_decision": "PASS", "speedup": witness.accepted_speedup(),
               "selected_delta_digest": witness.selected_delta_digest,
               "speed_evidence_digest": witness.evidence_digest}
    digest = canonical_digest(f"{schema}.report", summary)
    payload = {"report": summary, "report_digest": digest, "reservation_id": reservation_id,
               "speed_witness": witness.to_dict()}
    root = tmp_path / "evidence"
    ref = publish_evidence(root, canonical_json_bytes(payload),
                           domain="qualification.operator-quality-correction",
                           media_type="application/json", schema=schema)
    qualification = SimpleNamespace(qualification_report_digest=digest, reservation_digest=reservation_id,
        selected_delta_digest=witness.selected_delta_digest, speedup=witness.accepted_speedup(),
        comparison_context_digest=fixtures._h(target + "context"))
    rate, required, context = measured_speed(qualification, ref, root)
    assert float(rate) == pytest.approx(witness.resident_policy.scored_tokens_per_second(witness.rates[1]))
    assert required > 1 and context == qualification.comparison_context_digest
    qualification.qualification_report_digest = "f" * 64
    with pytest.raises(IntakeError, match="correction differs"):
        measured_speed(qualification, ref, root)


def test_upgrade_does_not_reward_historical_pass_that_lost_same_slot(tmp_path, evaluator, monkeypatch):
    from cacheon.chain import submission_ranking
    with fixtures._store(tmp_path) as store:
        first = fixtures._qualified_settlement_candidate(store, speedups=("1.07", "1.07"))
        evaluator["rate"] = Decimal("109")
        loser = fixtures._qualified_settlement_candidate(store, index=1, marker="old-loser",
                                                        speedups=("1.06", "1.06"))
        # Simulate the pre-upgrade DB, where both retained A/B PASSes were rewarded.
        store._db.execute("DELETE FROM submission_rankings")
    rates = {first.reservation_digest: Decimal("110"), loser.reservation_digest: Decimal("109")}
    monkeypatch.setattr(submission_ranking, "measured_speed", lambda q, *args:
                        (rates[q.reservation_digest], Decimal("1.01"), fixtures._h("measurement-context")))
    with fixtures._store(tmp_path) as store:
        assert competition_details(store._db, first.reservation_digest)["won"]
        assert not competition_details(store._db, loser.reservation_digest)["won"]
        assert competition_details(store._db, loser.reservation_digest)["stale"]
        assert {claim.hotkey for claim in store.passed_reward_claims()} == {first.hotkey}
        assert evaluator["calls"] == 2


def test_faster_prior_competitive_loss_still_sets_the_bar(tmp_path, evaluator):
    with fixtures._store(tmp_path) as store:
        fixtures._qualified_settlement_candidate(store)
        evaluator["rate"] = Decimal("111.9")
        prior = fixtures._qualified_settlement_candidate(store, index=1, marker="noisy", measured_required="1.02")
        assert not competition_details(store._db, prior.reservation_digest)["won"]
        evaluator["rate"] = Decimal("111.5")
        slower = fixtures._qualified_settlement_candidate(store, index=2, marker="slower")
        details = competition_details(store._db, slower.reservation_digest)
        assert not details["won"] and details["competitor_id"] == prior.reservation_digest
        assert slower.hotkey not in {claim.hotkey for claim in store.passed_reward_claims()}


@pytest.mark.parametrize("decode,prefill,expected,margin", [
    (102.0, 20.0, 102.0, 1.01), (100.5, 50.0, 112.5, 1.025),
])
def test_v12_competition_uses_existing_credited_score(decode, prefill, expected, margin):
    from cacheon.chain.submission_ranking import _competitive_score
    from tests.test_prefill_lane import _policy, _rates

    score, required = _competitive_score(_policy(), _rates(decode, prefill))
    assert float(score) == pytest.approx(expected)
    assert float(required) == pytest.approx(margin)
    with pytest.raises(IntakeError, match="does not retain its PASS"):
        _competitive_score(_policy(), _rates(95.0, 60.0))


@pytest.mark.parametrize("bookend", [False, True])
def test_historical_acceptance_reopens_original_two_or_three_reads(tmp_path, bookend):
    from cacheon.chain.submission_ranking import measured_speed
    from cacheon.eval.evidence_store import publish_evidence
    from cacheon.eval.qualification_runner import ResidentSpeedWitness
    from cacheon.stack_identity import canonical_digest, canonical_json_bytes
    from tests.test_crossover_runtime import _rig, _speed, _policy_v8

    plan, baseline, candidate, mount, *_ = _rig(tmp_path / "runtime", (0.90,),
                                                policy=_policy_v8(), timed_batches=3)
    witness = ResidentSpeedWitness.from_evidence(_speed(plan, baseline, candidate, mount), plan)
    raw = witness.to_dict()
    raw["resident_policy"]["version"] = 6
    if not bookend:
        raw["rates"] = raw["rates"][:2]
    schema = "cacheon.qualification.stage-exit.v3"
    payload = {"stage": "resident_accept", "decision": "PASS", "speed_witness": raw,
               "authority_digest": fixtures._h("plan"), "selected_delta_digest": witness.selected_delta_digest}
    root = tmp_path / "evidence"
    ref = publish_evidence(root, canonical_json_bytes(payload), domain="qualification.stage-exit",
                          media_type="application/json", schema=schema)
    qualification = SimpleNamespace(qualification_report_digest=canonical_digest(schema, payload),
        qualification_plan_digest=fixtures._h("plan"), selected_delta_digest=witness.selected_delta_digest,
        speedup=witness.accepted_speedup(), comparison_context_digest=fixtures._h("context"))
    score, required, context = measured_speed(qualification, ref, root)
    assert score > 0 and required > 1 and context == fixtures._h("context")
    qualification.qualification_report_digest = "f" * 64
    with pytest.raises(IntakeError, match="historical acceptance differs"):
        measured_speed(qualification, ref, root)

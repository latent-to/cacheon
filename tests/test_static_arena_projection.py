"""One production projector combines retained evidence and frozen offer terms."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from cacheon.arena_allocation import ArenaAllocation
from cacheon.chain import arena_weight_projection as allocation
from cacheon.chain import qualification_settlement as rewards
from cacheon.chain.intake import FinalizedIntakeStore, IntakeError, SQLiteFollowerWeightPublicationJournal
from cacheon.chain.weight_share import CurrentWeightOffer
from cacheon.chain.weights import WeightProjection, WeightPublicationRecord
from cacheon.eval.evidence_store import reopen_evidence
from tests import test_chain_intake as intake


def _fixture(tmp_path, monkeypatch, *, shared_hotkey=False, weights=(600_000, 400_000)):
    monkeypatch.setattr(intake, "FinalizedIntakeStore", allocation.RecoverableFinalizedIntakeStore)
    if shared_hotkey:
        reserve = intake._reserve_one
        monkeypatch.setattr(intake, "_reserve_one", lambda store, **kw:
                            reserve(store, **{**kw, "hotkey": "minera"}))
    configs = {}
    for index, name in enumerate(("a", "b")):
        with intake._store(tmp_path / name) as store:
            intake._qualified_settlement_candidate(store, marker=name, arena_marker=name,
                                                   index=index)
            if name == "a":
                lease = store.lease_settlement_cohort(current_block=11)
                plan, evidence = intake._settlement_plan(store, lease)
                store.commit_settlement(lease, plan, evidence, current_block=11)
            configs[f"/configs/{name}.json"] = SimpleNamespace(
                intake_db=store.path, policy=store.policy, scope=store.scope,
                digest=intake._h(name),
            )
    with FinalizedIntakeStore(tmp_path / "signer" / "intake.sqlite3", intake.IntakePolicy(), scope=intake.SCOPE) as signer:
        journal = signer.path
    monkeypatch.setattr(allocation, "load_config", lambda path: configs[path])
    schedule = ArenaAllocation.from_dict({
        "activation_block": 20, "burn_hotkey": "validator",
        "sources": {name: f"/configs/{name}.json" for name in ("a", "b")},
        "history": [{"from_block": 0, "weights_ppm": {"a": 1_000_000, "b": 0}},
                    {"from_block": 20, "weights_ppm": dict(zip(("a", "b"), weights))}],
    })
    return schedule, journal, configs


def _project(primary, schedule, journal, block=20):
    context = replace(intake._context("validator", "minera", "minerb", "minerc", "minerd"),
                      current_block=block, current_block_hash=intake._bh(block))
    return allocation.build_static_projection(
        primary, allocation=schedule, policy=intake.POLICY, context=context,
        netuid=intake.SCOPE.netuid, confirmation_journal=journal, max_lag_blocks=2,
    )


def _register(tmp_path, schedule, journal):
    with intake._store(tmp_path / "a") as primary:
        baseline = primary.build_weight_projection(
            policy=intake.POLICY, context=intake._context("validator", "minera", "minerb", "minerc", "minerd"),
            netuid=intake.SCOPE.netuid,
        )
        before = _project(primary, schedule, journal, 12)
        assert before == baseline
        assert "allocation_evidence" not in before.to_dict()
        assert WeightProjection.from_dict(before.to_dict()).digest == baseline.digest
    return before


def _advance(tmp_path, block=20):
    for name in ("a", "b"):
        with intake._store(tmp_path / name) as store:
            intake._reserve(store, (), block=block)


def test_combined_projection_retains_terms_report_and_restart_history(tmp_path, monkeypatch):
    schedule, journal, _ = _fixture(tmp_path, monkeypatch, weights=(200_000, 300_000))
    _register(tmp_path, schedule, journal)
    with intake._store(tmp_path / "b") as secondary:
        intake._qualified_settlement_candidate(secondary, marker="c", arena_marker="c",
                                               index=2, submission_block=20, retained_block=20)
    with intake._store(tmp_path / "a") as primary:
        intake._qualified_settlement_candidate(primary, marker="d", arena_marker="d",
                                               index=3, submission_block=20, retained_block=20)
    _advance(tmp_path)
    with intake._store(tmp_path / "a") as primary:
        first = _project(primary, schedule, journal)
        assert dict(first.weights_ppm) == {"minera": 500_000, "minerd": 100_000,
                                          "minerc": 300_000, "validator": 100_000}
        report = json.loads(reopen_evidence(primary.path.parent / "weight-allocation-evidence",
                                           first.allocation_evidence))
        assert sorted(value[1] for value in report["submission_terms"].values()) == [0, 200_000, 300_000, 1_000_000]
        assert report["arena_weights_ppm"] == {"a": 600_000, "b": 300_000}
        assert report["burned_ppm"] == 100_000
        assert WeightProjection.from_dict(first.to_dict()) == first
        with pytest.raises(IntakeError, match="requires its configured producer"):
            primary.build_weight_projection(policy=intake.POLICY,
                context=replace(intake._context("validator", "minera"), current_block=20),
                netuid=intake.SCOPE.netuid)
    with intake._store(tmp_path / "a") as primary:
        assert _project(primary, schedule, journal) == first
        raw = schedule.to_dict()
        raw["history"].append({"from_block": 30, "weights_ppm": {"a": 100_000, "b": 900_000}})
        updated = _project(primary, ArenaAllocation.from_dict(raw), journal)
        assert updated.weights_ppm == first.weights_ppm
        assert updated.allocation_evidence != first.allocation_evidence
        with pytest.raises(IntakeError, match="history or source authority changed"):
            _project(primary, schedule, journal)


def test_quarter_bonus_uses_arrival_even_when_old_pass_qualifies_later(tmp_path, monkeypatch):
    from cacheon.economics import project_global_rewards

    schedule, journal, _ = _fixture(tmp_path, monkeypatch)
    raw = schedule.to_dict()
    raw["history"][1]["stall_bonus_ppm"] = 250_000
    schedule = ArenaAllocation.from_dict(raw)
    _register(tmp_path, schedule, journal)
    with intake._store(tmp_path / "a") as primary:
        for marker, index, arrival in (("c", 2, 19), ("d", 3, 7219)):
            intake._qualified_settlement_candidate(primary, marker=marker, arena_marker="a",
                index=index, initialize_stack=False, submission_block=arrival, retained_block=7220)
    _advance(tmp_path, 7220)
    with intake._store(tmp_path / "a") as primary:
        result = _project(primary, schedule, journal, 7220)
        report = json.loads(reopen_evidence(primary.path.parent / "weight-allocation-evidence",
                                           result.allocation_evidence))
        data = rewards._reward_projection_inputs(primary, include_uncrowned=True)
        claims = {claim.hotkey: claim for claim in data["earning_claims"]}
        assert report["submission_stall_bonus_ppm"][claims["minerc"].digest] == 1_000_000
        assert report["submission_stall_bonus_ppm"][claims["minerd"].digest] == 250_000
        for key in ("standing_claims", "states", "adjustments"):
            data.pop(key)
        context = replace(intake._context("validator", "minera", "minerb", "minerc", "minerd"),
                          current_block=7220, current_block_hash=intake._bh(7220))
        full = project_global_rewards(intake.POLICY, context, **data,
            allocation_terms={c.digest: ("a", schedule.terms_at(c.crowned_block)["a"])
                              for c in data["earning_claims"]}, allocation_burn_hotkey="validator")
        assert dict(result.weights_ppm)["minerd"] < full.weights_by_hotkey["minerd"]
        assert dict(result.weights_ppm).get("validator", 0) == full.weights_by_hotkey.get("validator", 0)
    with intake._store(tmp_path / "a") as primary:
        assert _project(primary, schedule, journal, 7220) == result
        raw["history"].append({"from_block": 8000, "weights_ppm": raw["history"][1]["weights_ppm"]})
        future = _project(primary, ArenaAllocation.from_dict(raw), journal, 7220)
        assert future.weights_ppm == result.weights_ppm
        raw["history"][1]["stall_bonus_ppm"] = 500_000
        with pytest.raises(IntakeError, match="history or source authority changed"):
            _project(primary, ArenaAllocation.from_dict(raw), journal, 7220)


@pytest.mark.parametrize("failure", ["stale", "ahead", "hash", "missing", "duplicate", "evidence"])
def test_bad_secondary_prevents_offer_and_rolls_back_primary(tmp_path, monkeypatch, failure):
    schedule, journal, configs = _fixture(tmp_path, monkeypatch)
    _register(tmp_path, schedule, journal)
    _advance(tmp_path)
    with intake._store(tmp_path / "b") as secondary:
        if failure in {"stale", "ahead", "hash"}:
            block = {"stale": 10, "ahead": 21, "hash": 20}[failure]
            secondary._db.execute("UPDATE metadata SET value=? WHERE key='finalized_cursor'",
                                  (json.dumps([block, intake._bh(99 if failure == "hash" else block)]),))
        elif failure == "duplicate":
            with intake._store(tmp_path / "a") as primary:
                reservation = primary._db.execute("SELECT reservation_id FROM reservations").fetchone()[0]
            # Every listener admits every paid reveal; only the same reservation
            # passing in two stores is a duplicate reward owner.
            intake._reserve(secondary, (intake._arrival(0, hotkey="minera", block=10),), block=20)
            assert secondary.get(reservation) is not None
            secondary._db.execute("UPDATE reservations SET decision='PASS' WHERE reservation_id=?",
                                  (reservation,))
        elif failure == "evidence":
            for path in (secondary.path.parent / "evidence").rglob("*"):
                if path.is_file():
                    path.chmod(0o600)
                    path.write_bytes(b"broken retained evidence")
    if failure == "missing":
        configs["/configs/b.json"].intake_db = tmp_path / "missing.sqlite3"
    with intake._store(tmp_path / "a") as primary:
        before = primary._db.execute("SELECT value FROM metadata WHERE key='static_arena_allocation'").fetchone()[0]
        with pytest.raises((IntakeError, ValueError),
                           match="snapshot" if failure in {"stale", "ahead", "hash"} else None):
            _project(primary, schedule, journal)
        assert primary._db.execute("SELECT value FROM metadata WHERE key='static_arena_allocation'").fetchone()[0] == before


@pytest.mark.parametrize("change", ["past", "edit", "source"])
def test_schedule_cannot_reprice_existing_submissions(tmp_path, monkeypatch, change):
    schedule, journal, configs = _fixture(tmp_path, monkeypatch)
    _register(tmp_path, schedule, journal)
    raw = schedule.to_dict()
    if change == "past":
        raw["history"].append({"from_block": 21, "weights_ppm": {"a": 500_000, "b": 500_000}})
        _advance(tmp_path, 22)
    elif change == "edit":
        raw["history"][1]["weights_ppm"] = {"a": 500_000, "b": 500_000}
        _advance(tmp_path, 22)
    else:
        configs["/configs/b.json"].digest = intake._h("changed-authority")
        _advance(tmp_path, 22)
    with intake._store(tmp_path / "a") as primary:
        with pytest.raises(IntakeError):
            _project(primary, ArenaAllocation.from_dict(raw), journal, 22)


@pytest.mark.parametrize("shared_hotkey", [False, True])
def test_confirmation_starts_only_rewarded_claim_clocks(tmp_path, monkeypatch, shared_hotkey):
    schedule, journal_path, _ = _fixture(tmp_path, monkeypatch, shared_hotkey=shared_hotkey)
    _register(tmp_path, schedule, journal_path)
    _advance(tmp_path)
    with intake._store(tmp_path / "a") as primary:
        projection = _project(primary, schedule, journal_path)
        assert dict(projection.weights_ppm) == {"minera": 1_000_000}
    with FinalizedIntakeStore(journal_path, intake.IntakePolicy(), scope=intake.SCOPE) as signer:
        journal = SQLiteFollowerWeightPublicationJournal(signer, CurrentWeightOffer.from_legacy_projection(projection))
        journal.compare_and_swap(None, WeightPublicationRecord(
            projection.digest, "confirmed", confirmed_block=21, confirmed_last_update=21))
    for name, expected in (("a", 21), ("b", None)):
        with intake._store(tmp_path / name) as store:
            rewards.reconcile_follower_reward_decay(store, journal_path, validator_hotkey="validator")
            assert rewards.reward_decay_adjustments(store)[-1]["start_block"] == expected
            before = rewards.reward_decay_adjustments(store)
            rewards.reconcile_follower_reward_decay(store, journal_path, validator_hotkey="validator")
            assert rewards.reward_decay_adjustments(store) == before


def test_busy_secondary_uses_existing_service_skip(tmp_path, monkeypatch):
    from cacheon.chain.weight_offer_service import WeightOfferBusyError

    schedule, journal, _ = _fixture(tmp_path, monkeypatch)
    _register(tmp_path, schedule, journal)
    _advance(tmp_path)
    with intake._store(tmp_path / "a") as primary, intake._store(tmp_path / "b"):
        with pytest.raises(WeightOfferBusyError):
            _project(primary, schedule, journal)


@pytest.mark.parametrize("invalid_reason", ["missing_eval_cost_payment", ""])
def test_duplicate_arrival_without_a_pass_does_not_claim_reward_ownership(tmp_path, monkeypatch, invalid_reason):
    schedule, journal, _ = _fixture(tmp_path, monkeypatch)
    _register(tmp_path, schedule, journal)
    _advance(tmp_path)
    with intake._store(tmp_path / "a") as primary:
        before = _project(primary, schedule, journal)
    with intake._store(tmp_path / "b") as secondary:
        arrival = replace(intake._arrival(0, hotkey="minera", block=10), invalid_reason=invalid_reason)
        observed = intake._reserve(secondary, (arrival,), block=20)[0]
        assert (observed.status == "failed") == bool(invalid_reason) and observed.decision != "PASS"
    with intake._store(tmp_path / "a") as primary:
        after = _project(primary, schedule, journal)
        assert after.weights_ppm == before.weights_ppm
        assert after.rewarded_evidence_digests == before.rewarded_evidence_digests
        assert dict(after.weights_ppm) == {"minera": 1_000_000}


def test_existing_push_stage_reloads_allocation_and_emits_one_offer(tmp_path, monkeypatch):
    from pathlib import Path
    from cacheon import chain
    from cacheon.chain import weight_share
    from cacheon.chain.standing_weights_stage import compose_weight_offer_push, load_weights_config
    from tests.test_weight_offer_service import _setup, _rewrite

    schedule, journal, _ = _fixture(tmp_path, monkeypatch)
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps(schedule.to_dict()))
    allocation_path.chmod(0o400)
    _, config = _setup(tmp_path)
    stage_path = Path(config["weights_stage_config"])
    raw = json.loads(stage_path.read_text())
    _rewrite(stage_path, {**raw, "schema": "cacheon-standing-weights-config-v2",
                         "confirmation_journal": str(journal),
                         "arena_allocation_path": str(allocation_path),
                         "half_life_blocks": intake.POLICY.half_life_blocks,
                         "discovery_lifetime_blocks": intake.POLICY.discovery_lifetime_blocks,
                         "refresh_blocks": 1})
    stage = load_weights_config(stage_path)
    head = [12]
    monkeypatch.setattr(chain, "connect", lambda *a, **kw: object())
    monkeypatch.setattr(chain, "read_finalized_head", lambda st: (head[0], intake._bh(head[0])))
    monkeypatch.setattr(chain, "fetch_metagraph", lambda st, netuid: SimpleNamespace(
        uids=[0, 1, 2], hotkeys=["validator", "minera", "minerb"],
        block=head[0], block_hash=intake._bh(head[0])))
    offers = []

    def push(url, offer, **kw):
        offers.append(CurrentWeightOffer.from_dict(offer.to_dict()))
        return {"status": "accepted"}

    monkeypatch.setattr(weight_share, "push_current_weights", push)
    publish = compose_weight_offer_push(stage, store_factory=lambda: intake._store(tmp_path / "a"),
                                        scope=intake.SCOPE)
    publish()
    head[0] = 20
    _advance(tmp_path)
    publish()
    assert len(offers) == 2 and offers[0].projection.allocation_evidence is None
    assert offers[1].projection.allocation_evidence is not None
    raw = schedule.to_dict()
    raw["history"].append({"from_block": 30, "weights_ppm": {"a": 200_000, "b": 300_000}})
    _rewrite(allocation_path, raw)
    head[0] = 21
    _advance(tmp_path, 21)
    publish()
    assert len(offers) == 3
    assert offers[2].projection.weights_ppm == offers[1].projection.weights_ppm
    assert offers[2].projection.policy_digest != offers[1].projection.policy_digest
    with intake._store(tmp_path / "b") as secondary:
        intake._qualified_settlement_candidate(secondary, marker="c", arena_marker="c",
                                               index=2, submission_block=21, retained_block=21)
    head[0] = 22
    _advance(tmp_path, 22)
    publish()
    assert len(offers) == 4
    assert dict(offers[-1].projection.weights_ppm) == {"minera": 714_286, "validator": 285_714}
    with intake._store(tmp_path / "a") as primary:
        report = json.loads(reopen_evidence(primary.path.parent / "weight-allocation-evidence",
                                           offers[-1].projection.allocation_evidence))
    assert sorted(value[1] for value in report["submission_terms"].values()) == [0, 400_000, 1_000_000]
    assert report["arena_weights_ppm"] == {"a": 714_286, "b": 285_714}
    assert report["burned_ppm"] == 0

"""Retained production reward inputs keep decaying without an intake database."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from cacheon.chain import qualification_settlement as rewards
from cacheon.chain.arena_weight_projection import build_static_projection
from cacheon.chain.intake import FinalizedIntakeStore, IntakeError, SQLiteFollowerWeightPublicationJournal
from cacheon.chain.reward_checkpoint import RewardCheckpoint
from cacheon.chain.weight_share import CurrentWeightOffer
from cacheon.chain.weights import WeightPublicationRecord
from cacheon.eval.evidence_store import reopen_evidence
from tests import test_chain_intake as intake
from tests.test_static_arena_projection import _fixture, _register, _advance, _project


def _checkpoint(tmp_path, journal, monkeypatch, schedule=None):
    from cacheon.chain import arena_weight_projection

    if schedule is not None:
        monkeypatch.setattr(arena_weight_projection, "load_allocation", lambda path: schedule)
    stage = SimpleNamespace(attribution_hotkey="validator", burn_hotkey="validator",
                            arena_allocation_path=None if schedule is None else tmp_path / "allocation.json",
                            confirmation_journal=journal)
    return RewardCheckpoint(tmp_path / "rewards.json", policy=intake.POLICY, scope=intake.SCOPE, stage=stage)


def _context(block):
    return replace(intake._context("validator", "minera", "minerb", "minerc", "minerd"),
                   current_block=block, current_block_hash=intake._bh(block))


def _confirm(path, projection, block):
    with FinalizedIntakeStore(path, scope=intake.SCOPE) as store:
        journal = SQLiteFollowerWeightPublicationJournal(store, CurrentWeightOffer.from_legacy_projection(projection))
        prior = journal.load()
        journal.compare_and_swap(None if prior is None else prior.digest, WeightPublicationRecord(
            projection.digest, "confirmed", confirmed_block=block, confirmed_last_update=block,
            prior_record_digest=None if prior is None else prior.digest))


@pytest.mark.parametrize("static", [False, True])
def test_checkpoint_recomputes_exact_economics_and_confirmation_clocks(tmp_path, monkeypatch, static):
    schedule, journal, _ = _fixture(tmp_path, monkeypatch)
    if static:
        _register(tmp_path, schedule, journal)
    _advance(tmp_path)
    checkpoint = _checkpoint(tmp_path, journal, monkeypatch, schedule if static else None)
    with intake._store(tmp_path / "a") as primary:
        if static:
            first = build_static_projection(primary, allocation=schedule, policy=intake.POLICY,
                context=_context(20), netuid=intake.SCOPE.netuid, confirmation_journal=journal,
                capture=checkpoint.capture)
        else:
            first = rewards.build_weight_projection(primary, policy=intake.POLICY,
                context=_context(20), netuid=intake.SCOPE.netuid, capture=checkpoint.capture)
    checkpoint.save()
    _confirm(journal, first, 21)
    block = 21 + intake.POLICY.half_life_blocks * 2
    # Restart the checkpoint reader; every intake database is exclusively locked.
    checkpoint = _checkpoint(tmp_path, journal, monkeypatch, schedule if static else None)
    with intake._store(tmp_path / "a"), intake._store(tmp_path / "b"):
        later = checkpoint.project(_context(block))
    data = json.loads(checkpoint.path.read_text())
    assert 21 in data["inputs"]["decay_start_blocks"].values()
    assert later.effective_block == block and later.crown_count == first.crown_count
    assert later.evaluation_state_digest != first.evaluation_state_digest
    with intake._store(tmp_path / "a") as primary:
        if static:
            expected = _project(primary, schedule, journal, block)
        else:
            rewards.reconcile_follower_reward_decay(primary, journal, validator_hotkey="validator")
            expected = primary.build_weight_projection(policy=intake.POLICY, context=_context(block), netuid=intake.SCOPE.netuid)
    assert later.weights_ppm == expected.weights_ppm
    assert later.evaluation_state_digest == expected.evaluation_state_digest
    if static:
        root = primary.path.parent / "weight-allocation-evidence"
        actual_report = json.loads(reopen_evidence(root, later.allocation_evidence))
        expected_report = json.loads(reopen_evidence(root, expected.allocation_evidence))
        assert actual_report["submission_weights_ppm"] == expected_report["submission_weights_ppm"]
        assert later.rewarded_evidence_digests == expected.rewarded_evidence_digests
    assert checkpoint.project(_context(block)) == later


def test_checkpoint_rejects_changed_policy_and_corruption(tmp_path, monkeypatch):
    _, journal, _ = _fixture(tmp_path, monkeypatch)
    checkpoint = _checkpoint(tmp_path, journal, monkeypatch)
    with intake._store(tmp_path / "a") as primary:
        rewards.build_weight_projection(primary, policy=intake.POLICY, context=_context(20),
                                        netuid=intake.SCOPE.netuid, capture=checkpoint.capture)
    checkpoint.save()
    checkpoint.policy = replace(intake.POLICY, half_life_blocks=999)
    with pytest.raises(IntakeError, match="configured authority"):
        checkpoint.project(_context(30))
    checkpoint.path.write_text("broken")
    with pytest.raises(ValueError):
        checkpoint.project(_context(30))


@pytest.mark.parametrize("unavailable", ["primary_lock", "secondary_lock", "primary_missing", "secondary_missing"])
def test_production_service_pushes_checkpoint_then_adopts_new_rewards(tmp_path, monkeypatch, unavailable):
    from contextlib import ExitStack
    from pathlib import Path
    from cacheon import chain
    from cacheon.chain import weight_share, mainnet_screen_dispatcher
    from cacheon.chain.weight_offer_service import build_offer_publisher, load_offer_service_config
    from tests.test_weight_offer_service import _setup, _rewrite

    schedule, journal, configs = _fixture(tmp_path, monkeypatch)
    _register(tmp_path, schedule, journal)
    _advance(tmp_path)
    service_path, raw = _setup(tmp_path)
    stage_path = Path(raw["weights_stage_config"])
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps(schedule.to_dict()))
    allocation_path.chmod(0o400)
    stage = json.loads(stage_path.read_text())
    _rewrite(stage_path, {**stage, "arena_allocation_path": str(allocation_path),
                         "confirmation_journal": str(journal), "refresh_blocks": 2,
                         "half_life_blocks": intake.POLICY.half_life_blocks,
                         "discovery_lifetime_blocks": intake.POLICY.discovery_lifetime_blocks,
                         "discovery_pool_ppm": intake.POLICY.discovery_pool_ppm})
    monkeypatch.setattr(mainnet_screen_dispatcher, "load_config", lambda path: configs["/configs/a.json"])
    head = [20]
    monkeypatch.setattr(chain, "connect", lambda *a, **kw: object())
    monkeypatch.setattr(chain, "read_finalized_head", lambda st: (head[0], intake._bh(head[0])))
    monkeypatch.setattr(chain, "fetch_metagraph", lambda st, netuid: SimpleNamespace(
        uids=[0, 1, 2, 3], hotkeys=["validator", "minera", "minerb", "minerc"],
        block=head[0], block_hash=intake._bh(head[0])))
    offers = []
    monkeypatch.setattr(weight_share, "push_current_weights",
                        lambda url, offer, **kw: offers.append(offer) or {"status": "accepted"})
    config = load_offer_service_config(service_path)
    publish = build_offer_publisher(config)
    publish()
    original = offers[-1]
    _confirm(journal, original.projection, 21)
    head[0] = 221
    source = "a" if unavailable.startswith("primary") else "b"
    db = configs[f"/configs/{source}.json"].intake_db
    with ExitStack() as lifetime:
        if unavailable.endswith("lock"):
            lifetime.enter_context(intake._store(tmp_path / source))
        else:
            db.rename(db.with_suffix(".unavailable"))
            lifetime.callback(db.with_suffix(".unavailable").rename, db)
        # A newly composed service, not a surviving in-memory input cache.
        publish = build_offer_publisher(config)
        publish()
        assert offers[-1].projection.effective_block == 221
        assert offers[-1].projection.evaluation_state_digest != original.projection.evaluation_state_digest
        assert offers[-1].projection.crown_count == original.projection.crown_count
        publish = build_offer_publisher(config)
        publish()
        assert offers[-1] == offers[-2]
    # Recovery at the same block must not conflict with the gateway's last offer.
    publish = build_offer_publisher(config)
    publish()
    assert offers[-1] == offers[-2]
    with intake._store(tmp_path / "b") as secondary:
        intake._qualified_settlement_candidate(secondary, marker="c", arena_marker="c",
                                               index=2, submission_block=222, retained_block=222)
    _advance(tmp_path, 222)
    head[0] = 222
    publish()
    assert "minerc" in dict(offers[-1].projection.weights_ppm)
    assert offers[-1].projection.weights_ppm != offers[-2].projection.weights_ppm

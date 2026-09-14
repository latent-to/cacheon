"""Fixed awards buy new measured progress without diluting outstanding winners."""

from dataclasses import replace
import math

import pytest

from cacheon.economics import ArenaRewardAuthority, EconomicsError, project_global_rewards
from tests.test_economics import _catalog, _claim, _d, _discovery, _global_context, _policy, _stack


def _history(speeds, *, blocks=None, hotkeys=None):
    stack = _stack(_catalog(), ("slot.a",))
    first = _claim(stack, "slot.a", "alice", speeds[0], 100)
    claims = tuple(replace(
        first, speedup_ppm=speed, crowned_block=(blocks or [100] * len(speeds))[i],
        hotkey=(hotkeys or ["alice"] * len(speeds))[i],
        contribution_digest=first.contribution_digest if i == 0 else _d(hex(i)[2:]),
        retained_evidence_digest=_d(hex(i + 4)[2:]),
    ) for i, speed in enumerate(speeds))
    return stack, claims


def _replay(stack, claims, *, block=100, starts=None, policy=None, members=None, bounties=(), baselines=None):
    return project_global_rewards(
        policy or _policy(frontier_awards_from_block=0, discovery_pool_ppm=0),
        _global_context(block, members), (ArenaRewardAuthority(stack, 1, (claims[0],)),),
        claims, bounties,
        accepted_blocks=starts or {c.digest: c.crowned_block for c in claims},
        comparison_baselines=baselines or {c.digest: _d("b") for c in claims},
        acceptance_order={c.digest: i for i, c in enumerate(claims)},
    )


@pytest.mark.parametrize("speed", [1_010_000, 1_018_188, 1_030_000, 1_050_000, 10_000_000])
def test_award_matches_independent_exponential_oracle_and_returns_decay(speed):
    stack, claims = _history([speed])
    fraction = -math.expm1(-math.log(2) * math.log(speed / 1_000_000) / math.log(1.01))
    for age in (0, 100, 400, 500):
        result = _replay(stack, claims, block=100 + age)
        expected = math.floor(1_000_000 * fraction * 2 ** (-age / 100))
        assert abs(result.weights_by_hotkey["alice"] - expected) <= 1
        assert sum(result.weights_by_hotkey.values()) == 1_000_000
        assert result.weights_by_hotkey.get("validator", 0) == 1_000_000 - result.weights_by_hotkey["alice"]
    if speed == 1_010_000:
        assert _replay(stack, claims).weights_by_hotkey == {"alice": 500_000, "validator": 500_000}


def test_splitting_compounded_progress_at_one_acceptance_time_earns_no_extra():
    stack, split = _history([1_010_000, 1_020_100], blocks=[99, 100])
    starts = {c.digest: 100 for c in split}
    split_result = _replay(stack, split, starts=starts)
    whole = replace(split[0], speedup_ppm=1_020_100)
    assert split_result.weights_by_hotkey == _replay(stack, (whole,), starts={whole.digest: 100}).weights_by_hotkey
    assert split_result.weights_by_hotkey == {"alice": 750_000, "validator": 250_000}


def test_stale_baseline_pays_only_incremental_records_and_keeps_older_award():
    stack, claims = _history([1_010_000, 1_005_000, 1_010_000, 1_020_100],
                             blocks=[100, 110, 120, 130], hotkeys=["alice", "bob", "bob", "carol"])
    before = _replay(stack, claims[:1], block=130)
    result = _replay(stack, claims, block=130)
    assert result.weights_by_hotkey["alice"] == before.weights_by_hotkey["alice"]
    assert "bob" not in result.weights_by_hotkey
    assert result.weights_by_hotkey["carol"] > 250_000
    assert result.weights_by_hotkey["carol"] < 500_000
    assert sorted(c.credit for c in result.standing)[:2] == [0, 0]


def test_acceptance_clock_ignores_queue_wait_and_stall_bonus():
    stack, claims = _history([1_010_000])
    queued = replace(claims[0], crowned_block=1)
    fresh = _replay(stack, claims, block=200, starts={claims[0].digest: 200})
    late = _replay(stack, (queued,), block=200, starts={queued.digest: 200})
    assert fresh.weights_by_hotkey == late.weights_by_hotkey == {"alice": 500_000, "validator": 500_000}


def test_migration_preserves_old_awards_and_reserves_absent_miners_liability():
    stack, claims = _history([1_010_000, 1_020_100], blocks=[100, 200], hotkeys=["alice", "bob"])
    policy = _policy(frontier_awards_from_block=200, discovery_pool_ppm=0)
    result = _replay(stack, claims, block=200, policy=policy)
    legacy = _replay(stack, claims[:1], block=200,
                     policy=replace(policy, frontier_awards_from_block=201))
    assert result.weights_by_hotkey["alice"] == legacy.weights_by_hotkey["alice"]
    assert 490_000 < result.weights_by_hotkey["bob"] < 500_000
    members = tuple(m for m in _global_context().metagraph_members if m.hotkey != "alice")
    absent = _replay(stack, claims, block=200, policy=policy, members=members)
    assert absent.weights_by_hotkey["bob"] == result.weights_by_hotkey["bob"]
    assert absent.weights_by_hotkey["validator"] == result.weights_by_hotkey["validator"] + result.weights_by_hotkey["alice"]


def test_discovery_reserve_does_not_dilute_an_existing_award():
    stack, claims = _history([1_010_000])
    policy = _policy(frontier_awards_from_block=0)
    before = _replay(stack, claims, block=200, policy=policy)
    bounty = _replay(stack, claims, block=200, policy=policy, bounties=(_discovery(),))
    assert before.weights_by_hotkey["alice"] == bounty.weights_by_hotkey["alice"] == 200_000
    assert bounty.weights_by_hotkey["carol"] == 200_000
    assert before.weights_by_hotkey["validator"] - bounty.weights_by_hotkey["validator"] == 200_000


def test_recommissioning_does_not_renew_the_same_contribution():
    stack, claims = _history([1_010_000])
    copy = replace(claims[0], arena_digest=_d("f"), crowned_block=200)
    before = _replay(stack, claims, block=200)
    after = _replay(stack, (*claims, copy), block=200,
                    baselines={claims[0].digest: _d("b"), copy.digest: _d("a")})
    assert after.weights_by_hotkey == before.weights_by_hotkey
    assert next(c.credit for c in after.standing if c.claim_digest == copy.digest) == 0


def test_new_award_requires_its_retained_baseline_and_acceptance():
    stack, claims = _history([1_010_000])
    for kwargs, error in (({"comparison_baselines": {}}, "comparison baseline"),
                          ({"accepted_blocks": {}}, "acceptance block")):
        with pytest.raises(EconomicsError, match=error):
            project_global_rewards(
                _policy(frontier_awards_from_block=0), _global_context(),
                (ArenaRewardAuthority(stack, 1, claims),), claims, **kwargs,
            )


def test_commissioned_baselines_keep_distinct_measured_progress():
    stack, claims = _history([1_010_000, 1_010_000], blocks=[100, 200], hotkeys=["alice", "bob"])
    same = _replay(stack, claims, block=200)
    advanced = _replay(stack, claims, block=200,
                       baselines={claims[0].digest: _d("a"), claims[1].digest: _d("b")})
    assert "bob" not in same.weights_by_hotkey
    assert advanced.weights_by_hotkey == {"alice": 250_000, "bob": 375_000, "validator": 375_000}


def test_real_store_reopens_evidence_and_keeps_the_cutover_after_restart(tmp_path):
    from tests import test_chain_intake as intake

    policy = replace(intake.POLICY, frontier_awards_from_block=20)
    with intake._store(tmp_path) as store:
        for index, (marker, accepted) in enumerate((("old", 10), ("new", 20))):
            intake._qualified_settlement_candidate(
                store, index=index, marker=marker, arena_marker=marker,
                retained_block=accepted, speedups=("1.01", "1.01"),
            )
            lease = store.lease_settlement_cohort(current_block=accepted + 1)
            plan, evidence = intake._settlement_plan(store, lease)
            store.commit_settlement(lease, plan, evidence, current_block=accepted + 1)
        context = replace(intake._context("validator", "minerold", "minernew"), current_block=20)
        result = store.build_weight_projection(policy=policy, context=context, netuid=intake.SCOPE.netuid)
        assert dict(result.weights_ppm)["minernew"] > 440_000
        assert 9000 < dict(result.weights_ppm)["minerold"] < 10_000
    with intake._store(tmp_path) as reopened:
        assert reopened.build_weight_projection(policy=policy, context=context, netuid=intake.SCOPE.netuid) == result
        with pytest.raises(intake.IntakeError, match="bound validator consensus"):
            reopened.build_weight_projection(
                policy=replace(policy, frontier_awards_from_block=21), context=context,
                netuid=intake.SCOPE.netuid,
            )
        assert len(reopened.passed_reward_claims()) == 2
        original = reopened._db.execute("SELECT value FROM metadata WHERE key='emissions_reward_history'").fetchone()[0]
        reopened._db.execute("UPDATE settlement_qualifications SET retained_block=19 WHERE retained_block=20")
        with pytest.raises(intake.IntakeError, match="history changed"):
            reopened.build_weight_projection(policy=policy, context=context, netuid=intake.SCOPE.netuid)
        assert reopened._db.execute("SELECT value FROM metadata WHERE key='emissions_reward_history'").fetchone()[0] == original


def test_later_acceptance_in_the_same_block_cannot_change_an_earlier_award():
    stack, claims = _history([1_010_000, 1_020_100], blocks=[100, 99], hotkeys=["alice", "bob"])
    starts = {c.digest: 100 for c in claims}
    first = _replay(stack, claims[:1], starts=starts)
    both = _replay(stack, claims, starts=starts)
    assert both.weights_by_hotkey == {"alice": 500_000, "bob": 250_000, "validator": 250_000}
    assert both.weights_by_hotkey["alice"] == first.weights_by_hotkey["alice"]

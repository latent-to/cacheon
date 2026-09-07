"""Adversarial stock brackets must not mint a price or convict a useful candidate."""

from types import SimpleNamespace

import pytest

from cacheon.eval.speed_verdict import SpeedStageDecision, fail_reason, speed_grade
from tests.test_crossover_runtime import _policy_v3


def _read(rates, *, tokens=None, conditioned=True):
    tokens = tokens or [100] * len(rates)
    windows = tuple(SimpleNamespace(tokens=t, seconds=t / r)
                    for t, r in zip(tokens, rates, strict=True))
    return SimpleNamespace(
        windows=windows, timed_tokens=sum(tokens),
        timed_seconds=sum(window.seconds for window in windows),
        first_batch_index=0, first_timed_batch_index=int(conditioned),
        conditioning_tokens=100 if conditioned else 0,
    )


def _grade(b, c, bp, version=10):
    policy = _policy_v3(version=version, min_windows=5, min_margin=0.01,
                        max_window_scatter=0.05)
    return speed_grade(policy, [b, bp], [c], concluding=True)


@pytest.mark.parametrize("version", [10, 11])
def test_price_uses_the_faster_observed_stock_and_keeps_both_brackets(version):
    verdict, decision = _grade(_read([100] * 5), _read([110] * 5), _read([101] * 5), version)
    assert decision is SpeedStageDecision.PASS
    assert verdict.speedup == pytest.approx(110 / 101)
    assert verdict.speedup < 110 / 100.5  # A mean stock rate would overstate the measured gain.
    assert verdict.n_baselines == 2 and verdict.confident and verdict.passed_speedup


@pytest.mark.parametrize("before,after,candidate", [(80, 100, 95), (100, 80, 95), (80, 100, 30)])
def test_invalid_baseline_never_crowns_or_fails_even_an_apparent_large_loss(before, after, candidate):
    verdict, decision = _grade(_read([before] * 5), _read([candidate] * 5), _read([after] * 5))
    assert decision is SpeedStageDecision.NO_DECISION
    assert not verdict.confident and not verdict.passed_speedup
    assert verdict.n_baselines == 2
    assert "no baseline was discarded" in verdict.detail


def test_matching_aggregate_rates_cannot_hide_unstable_baseline_windows():
    verdict, decision = _grade(_read([80, 120, 100, 100, 100]), _read([110] * 5),
                               _read([120, 80, 100, 100, 100]))
    assert decision is SpeedStageDecision.NO_DECISION
    assert "stability" in verdict.detail


def test_mixed_prefill_decode_windows_are_compared_to_their_corresponding_windows():
    tokens = [131072, 131072, 98304, 98304, 98304]
    baseline = [2000, 2000, 700, 700, 700]
    verdict, decision = _grade(
        _read(baseline, tokens=tokens), _read([r * 1.1 for r in baseline], tokens=tokens),
        _read([r * 1.005 for r in baseline], tokens=tokens), version=11,
    )
    assert decision is SpeedStageDecision.PASS
    assert verdict.speedup == pytest.approx(1.1 / 1.005)


def test_missing_conditioning_cannot_produce_a_speed_verdict():
    verdict, decision = _grade(_read([100] * 5, conditioned=False), _read([110] * 5), _read([100] * 5))
    assert decision is SpeedStageDecision.NO_DECISION
    assert "conditioning" in verdict.detail


def test_uncertainty_is_not_relabelled_as_a_candidate_failure():
    verdict, decision = _grade(_read([100] * 5), _read([101.5] * 5), _read([101] * 5))
    assert decision is SpeedStageDecision.NO_DECISION
    assert "uncertainty" in verdict.detail


def test_a_measured_loss_still_fails_against_valid_baselines():
    verdict, decision = _grade(_read([100] * 5), _read([90] * 5), _read([100] * 5))
    assert decision is SpeedStageDecision.FAIL
    assert fail_reason(verdict) == "candidate_slower"

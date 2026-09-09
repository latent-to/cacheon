"""Retained version-8 grading and its historical mainnet incident fixtures.

Version 3 refuses to produce a rate for a read whose own window scatter
exceeds the sealed bound, which converts settled results into non-answers.
Two mainnet runs are retained here as regression fixtures because both were
recorded as NO_DECISION while their evidence already determined the verdict.
The version-3 scorer and adaptive graders were retired with MiniMax-M3.
These fixtures exercise only the retained v8 arithmetic, never a pre-v8 policy.

Version 5 adds the owner's bracket-drift ruling (2026-08-10): brackets that
disagree beyond the sealed noise ceiling exclude the drifted later brackets,
and the candidate is compared against the earliest bracket B alone. Version
8 grades with the same arithmetic over its precommitted B/C/B-prime reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cacheon.eval.crossover_runtime import (
    ResidentSpeedPolicy,
    TimedWindow,
)
from cacheon.eval.speed_verdict import (
    SpeedStageDecision,
    fail_reason,
    invariant_decision,
    speed_grade,
)

TOKENS = 65_536


def _policy(version: int, **overrides) -> ResidentSpeedPolicy:
    kwargs = {
        "max_stage_seconds": 60,
        "min_margin": 0.005,
        "noise_multiplier": 2.0,
        "max_noise": 0.02,
        "calibration_digest": "8" * 64,
        "calibration_context_digest": "9" * 64,
        "version": version,
        "min_windows": 3,
        "max_window_scatter": 0.05,
        "max_conditioning_slowdown": 1.25,
    }
    kwargs.update(overrides)
    return ResidentSpeedPolicy(**kwargs)


def _read(*rates: float) -> SimpleNamespace:
    """A read whose per-window rates are exactly ``rates``."""

    return SimpleNamespace(
        windows=tuple(
            TimedWindow(index, TOKENS, TOKENS / rate)
            for index, rate in enumerate(rates)
        ),
        conditioning_tokens=TOKENS,
        conditioning_seconds=1.0,
    )


def _steady(rate: float) -> SimpleNamespace:
    return _read(rate, rate, rate)


def test_invariance_decides_both_retained_mainnet_no_decisions() -> None:
    # Bundle 20478659: the candidate lost to BOTH stock bookends, but the
    # 2.38% bookend drift exceeded the 2% ceiling and the run was recorded
    # NO_DECISION. The verdict does not depend on which bookend you believe.
    assert invariant_decision([7477.155, 7303.009], [7085.971], 1.005) is (
        SpeedStageDecision.FAIL
    )

    # Bundle 6d8d62ad: raw ungraded speedup 0.58197 withheld because window
    # scatter was 0.0526643 against a 0.05 ceiling. No drift of that size
    # reaches a 42% deficit.
    assert invariant_decision([7400.0, 7400.0], [7400.0 * 0.58197], 1.005) is (
        SpeedStageDecision.FAIL
    )


def test_invariance_still_crowns_a_winner_and_still_hesitates_when_it_should() -> None:
    bookends = [7477.155, 7303.009]
    assert invariant_decision(bookends, [7900.0], 1.005) is SpeedStageDecision.PASS
    assert invariant_decision(bookends, [6900.0], 1.005) is SpeedStageDecision.FAIL
    # Wins against the low bookend, loses against the high one: the verdict
    # flips inside the observed spread, which is real ambiguity.
    assert invariant_decision(bookends, [7400.0], 1.005) is None


def test_ambiguity_concludes_as_not_proven_rather_than_no_decision() -> None:
    # Ambiguity needs TIGHT bookends. The required margin scales with the
    # observed bookend spread, so widely drifting bookends raise the bar until
    # a marginal candidate is an invariant FAIL rather than an open question.
    policy = _policy(8)
    baselines = [_steady(7400.0), _steady(7420.0)]
    candidates = [_steady(7450.0)]

    # Taking more reads only widens the observed spread, so the concluding
    # grade must terminate. The burden of proof sits with the candidate.
    _, concluded = speed_grade(policy, baselines, candidates)
    assert concluded is SpeedStageDecision.FAIL


def test_retained_v8_drift_excludes_the_late_bracket_and_c_against_b_decides() -> None:
    # The retained 2.36% bookend drift (over the 2% ceiling). Under v5 the
    # earliest bracket is the baseline, so the same candidate rate passes or
    # fails depending on which bracket was measured first -- and either way
    # the run decides immediately instead of escalating or punting.
    policy = _policy(8)
    candidate = [_steady(7400.0)]

    verdict, decision = speed_grade(
        policy, [_steady(7303.009), _steady(7477.155)], candidate
    )
    assert decision is SpeedStageDecision.PASS
    assert verdict.n_baselines == 1
    assert "C against B decides" in verdict.detail
    assert verdict.required == pytest.approx(1.005)

    _, decision = speed_grade(
        policy, [_steady(7477.155), _steady(7303.009)], candidate
    )
    assert decision is SpeedStageDecision.FAIL



def test_a_gross_loss_measured_on_an_unstable_box_still_fails() -> None:
    # The shape that produced the 6d8d62ad NO_DECISION: an unfit candidate read
    # plus a deficit far outside any plausible drift. Note the statistic is a
    # median absolute deviation, so breaching the bound requires dispersion
    # across most of the windows -- a lone outlier cannot do it.
    policy = _policy(8)
    baselines = [_steady(7477.155), _steady(7303.009)]
    unstable_candidate = _read(4400.0, 4800.0, 4000.0)
    assert policy.read_window_scatter(unstable_candidate) > policy.max_window_scatter

    _, decision = speed_grade(
        policy, baselines, [unstable_candidate]
    )
    assert decision is SpeedStageDecision.FAIL



def test_fail_reason_splits_the_band_miss_from_the_measured_slowdown() -> None:
    # The retained defect this pins: a 1.003 speedup against a 1.005 bar was
    # reported as a "regression". Inside the band -- above the mirrored bound
    # 1-u for a bar of 1+u -- a FAIL proves only that the bar was not cleared.
    policy = _policy(8)
    in_band, decision = speed_grade(
        policy, [_steady(1000.0), _steady(1000.0)], [_steady(1003.0)],
    )
    assert decision is SpeedStageDecision.FAIL
    assert fail_reason(in_band) == "speed_threshold_not_met"

    slower, decision = speed_grade(
        policy, [_steady(1000.0), _steady(1000.0)], [_steady(900.0)],
    )
    assert decision is SpeedStageDecision.FAIL
    assert fail_reason(slower) == "candidate_slower"

    # A conditioning regression is a measured slowdown in its own right,
    # whatever the timed band says.
    assert fail_reason(in_band, conditioning_failed=True) == "candidate_slower"

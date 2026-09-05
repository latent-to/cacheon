"""Version-4/5 resident speed grading: a decidable run never returns NO_DECISION.

Version 3 refuses to produce a rate for a read whose own window scatter
exceeds the sealed bound, which converts settled results into non-answers.
Two mainnet runs are retained here as regression fixtures because both were
recorded as NO_DECISION while their evidence already determined the verdict.
The version-3 scatter refusal still lives in the policy's read scorer; its
concluding grader was deleted with the MiniMax-M3 history seal.

Version 5 adds the owner's bracket-drift ruling (2026-08-10): brackets that
disagree beyond the sealed noise ceiling exclude the drifted later brackets,
and the candidate is compared against the earliest bracket B alone. Version
8 grades with the same arithmetic over its precommitted B/C/B-prime reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cacheon.eval.crossover_runtime import (
    CrossoverRuntimeError,
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


def test_version_four_is_sealable_with_a_wider_advisory_scatter_bound() -> None:
    # v3's bound doubles as a ceiling on what may be sealed, because v3 refuses
    # to grade past it. v4 carries scatter as recorded evidence, so a looser
    # advisory bound is admissible -- but still bounded.
    assert _policy(4, max_window_scatter=0.25).max_window_scatter == 0.25
    with pytest.raises(CrossoverRuntimeError, match=r"v4 .*in \(0, 0.25\]"):
        _policy(4, max_window_scatter=0.26)
    with pytest.raises(CrossoverRuntimeError, match=r"v3 .*in \(0, 0.05\]"):
        _policy(3, max_window_scatter=0.06)


def test_version_three_refuses_an_unfit_read_and_version_four_grades_it() -> None:
    unfit = _read(100.0, 110.0, 90.0)  # relative MAD 0.10, over the 0.05 bound
    assert _policy(3).read_window_scatter(unfit) == pytest.approx(0.10)

    with pytest.raises(CrossoverRuntimeError, match="window scatter exceeds"):
        _policy(3).scored_tokens_per_second(unfit)

    assert _policy(4).scored_tokens_per_second(unfit) == pytest.approx(100.0)

    # The sealed window count stays a structural requirement under both.
    for version in (3, 4):
        with pytest.raises(CrossoverRuntimeError, match="required timed windows"):
            _policy(version).scored_tokens_per_second(_read(100.0, 100.0))


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
    policy = _policy(4)
    baselines = [_steady(7400.0), _steady(7420.0)]
    candidates = [_steady(7450.0)]

    # Before the extension an ambiguous read escalates rather than deciding.
    verdict, decision = speed_grade(
        policy, baselines, candidates, concluding=False
    )
    assert decision is None and verdict.required > 1.0

    # Taking more reads only widens the observed spread, so the concluding
    # grade must terminate. The burden of proof sits with the candidate.
    _, concluded = speed_grade(policy, baselines, candidates, concluding=True)
    assert concluded is SpeedStageDecision.FAIL


def test_version_five_drift_excludes_the_late_bracket_and_c_against_b_decides() -> None:
    # The retained 2.36% bookend drift (over the 2% ceiling). Under v5 the
    # earliest bracket is the baseline, so the same candidate rate passes or
    # fails depending on which bracket was measured first -- and either way
    # the run decides immediately instead of escalating or punting.
    policy = _policy(5)
    candidate = [_steady(7400.0)]

    verdict, decision = speed_grade(
        policy, [_steady(7303.009), _steady(7477.155)], candidate, concluding=False
    )
    assert decision is SpeedStageDecision.PASS
    assert verdict.n_baselines == 1
    assert "C against B decides" in verdict.detail
    assert verdict.required == pytest.approx(1.005)

    _, decision = speed_grade(
        policy, [_steady(7477.155), _steady(7303.009)], candidate, concluding=False
    )
    assert decision is SpeedStageDecision.FAIL

    # The identical evidence under v4 flips inside the spread and concludes
    # FAIL. v5 is the only rule that credits a candidate measured against its
    # adjacent bracket.
    _, v4_concluded = speed_grade(
        _policy(4), [_steady(7303.009), _steady(7477.155)], candidate, concluding=True
    )
    assert v4_concluded is SpeedStageDecision.FAIL


def test_version_five_without_drift_matches_version_four_exactly() -> None:
    baselines = [_steady(7400.0), _steady(7420.0)]
    candidates = [_steady(7450.0)]
    for concluding in (False, True):
        assert speed_grade(
            _policy(5), baselines, candidates, concluding=concluding
        ) == speed_grade(_policy(4), baselines, candidates, concluding=concluding)


def test_version_five_candidate_straddle_concludes_fail_never_no_decision() -> None:
    # With the drifted bracket excluded, the candidate's own repeat reads
    # straddle the required bar against B: the miner's jitter is the miner's
    # problem, and the conclusion is "not proven faster", not a non-answer.
    policy = _policy(5)
    baselines = [_steady(7303.009), _steady(7477.155)]
    candidates = [_steady(7330.0), _steady(7400.0)]
    _, concluded = speed_grade(policy, baselines, candidates, concluding=True)
    assert concluded is SpeedStageDecision.FAIL


def test_a_gross_loss_measured_on_an_unstable_box_still_fails() -> None:
    # The shape that produced the 6d8d62ad NO_DECISION: an unfit candidate read
    # plus a deficit far outside any plausible drift. Note the statistic is a
    # median absolute deviation, so breaching the bound requires dispersion
    # across most of the windows -- a lone outlier cannot do it.
    policy = _policy(4)
    baselines = [_steady(7477.155), _steady(7303.009)]
    unstable_candidate = _read(4400.0, 4800.0, 4000.0)
    assert policy.read_window_scatter(unstable_candidate) > policy.max_window_scatter

    _, decision = speed_grade(
        policy, baselines, [unstable_candidate], concluding=False
    )
    assert decision is SpeedStageDecision.FAIL

    # The same evidence under v3 produces no verdict at all.
    with pytest.raises(CrossoverRuntimeError, match="window scatter exceeds"):
        speed_grade(
            _policy(3), baselines, [unstable_candidate], concluding=False
        )


def test_fail_reason_splits_the_band_miss_from_the_measured_slowdown() -> None:
    # The retained defect this pins: a 1.003 speedup against a 1.005 bar was
    # reported as a "regression". Inside the band -- above the mirrored bound
    # 1-u for a bar of 1+u -- a FAIL proves only that the bar was not cleared.
    policy = _policy(8)
    in_band, decision = speed_grade(
        policy, [_steady(1000.0), _steady(1000.0)], [_steady(1003.0)],
        concluding=True,
    )
    assert decision is SpeedStageDecision.FAIL
    assert fail_reason(in_band) == "speed_threshold_not_met"

    slower, decision = speed_grade(
        policy, [_steady(1000.0), _steady(1000.0)], [_steady(900.0)],
        concluding=True,
    )
    assert decision is SpeedStageDecision.FAIL
    assert fail_reason(slower) == "candidate_slower"

    # A conditioning regression is a measured slowdown in its own right,
    # whatever the timed band says.
    assert fail_reason(in_band, conditioning_failed=True) == "candidate_slower"

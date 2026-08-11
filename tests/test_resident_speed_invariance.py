"""Version-4 resident speed grading: a decidable run never returns NO_DECISION.

Version 3 refuses to produce a rate for a read whose own window scatter
exceeds the sealed bound, which converts settled results into non-answers.
Two mainnet runs are retained here as regression fixtures because both were
recorded as NO_DECISION while their evidence already determined the verdict.
Version 3 semantics must stay byte-identical -- sealed policies and retained
evidence depend on them -- so the new behavior is reachable only through a
prospectively sealed version-4 policy.

Version 5 layers the drift rule on top of v4: baseline brackets that drift
past the sealed ``max_noise`` are excluded and the candidate is compared
against the earliest baseline only, under the same terminating arithmetic.
Versions 3 and 4 must keep grading the same evidence identically.
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
        )
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


def test_version_three_concluding_behavior_is_unchanged() -> None:
    policy = _policy(3)
    baselines = [_steady(7477.155), _steady(7303.009)]
    _, concluded = speed_grade(
        policy, baselines, [_steady(7400.0)], concluding=True
    )
    # The v3 bookends drift 2.36% against a 2% ceiling, so the verdict is not
    # confident and v3 answers NO_DECISION. Sealed v3 policies keep this
    # meaning exactly.
    assert concluded is SpeedStageDecision.NO_DECISION


def test_version_five_is_sealable_and_round_trips_canonically() -> None:
    policy = _policy(5)
    assert policy.version == 5
    reopened = type(policy).from_dict(policy.to_dict())
    assert reopened == policy
    assert policy.digest != _policy(4).digest

    # v5 shares v4's advisory scatter ceiling, no wider.
    assert _policy(5, max_window_scatter=0.25).max_window_scatter == 0.25
    with pytest.raises(CrossoverRuntimeError, match=r"v5 .*in \(0, 0.25\]"):
        _policy(5, max_window_scatter=0.26)


def test_v5_drift_rule_decides_from_the_earliest_baseline() -> None:
    # Bookends drift 2.36% against the sealed 2% ceiling. Under v4 the drift
    # inflates the required bar to ~1.047 and a genuine ~1% win against the
    # bracket the candidate was actually measured beside is graded FAIL.
    # Under v5 the later bracket is excluded: C vs earliest B, bar 1.005.
    drifted = [_steady(7477.155), _steady(7303.009)]
    winner = [_steady(7550.0)]

    verdict, decision = speed_grade(_policy(5), drifted, winner, concluding=False)
    assert decision is SpeedStageDecision.PASS
    assert "v5 drift rule" in verdict.detail
    assert verdict.required == pytest.approx(1.005)
    assert verdict.speedup == pytest.approx(7550.0 / 7477.155)

    _, under_v4 = speed_grade(_policy(4), drifted, winner, concluding=False)
    assert under_v4 is SpeedStageDecision.FAIL

    # A candidate that loses against the earliest bracket is a FAIL under v5
    # without escalation.
    loser = [_steady(7400.0)]
    _, decision = speed_grade(_policy(5), drifted, loser, concluding=False)
    assert decision is SpeedStageDecision.FAIL

    # A ~3% win sits inside v4's drift-inflated window (bar ~1.047), so v4
    # escalates for more reads it can never make decisive; v5 crowns it
    # against the bracket it was measured beside, no escalation.
    strong = [_steady(7700.0)]
    _, decision = speed_grade(_policy(5), drifted, strong, concluding=False)
    assert decision is SpeedStageDecision.PASS
    _, under_v4 = speed_grade(_policy(4), drifted, strong, concluding=False)
    assert under_v4 is None


def test_v5_candidate_spread_stays_disqualifying_and_concludes_fail() -> None:
    # The drift rule excludes later BASELINE brackets only. Candidate reads
    # that straddle the bar keep the verdict undetermined, and an undetermined
    # conclusion terminates FAIL exactly as under v4 -- never NO_DECISION.
    drifted = [_steady(7477.155), _steady(7303.009)]
    straddling = [_steady(7550.0), _steady(7450.0)]

    _, escalate = speed_grade(_policy(5), drifted, straddling, concluding=False)
    assert escalate is None
    _, concluded = speed_grade(_policy(5), drifted, straddling, concluding=True)
    assert concluded is SpeedStageDecision.FAIL


def test_v5_without_drift_is_byte_identical_to_v4() -> None:
    # Inside the sealed noise ceiling the drift rule never engages: verdict
    # and decision match v4 exactly, for the clear win, the clear loss, and
    # the genuinely ambiguous read.
    tight = [_steady(7400.0), _steady(7420.0)]
    for candidate, concluding in (
        ([_steady(7900.0)], False),
        ([_steady(6900.0)], False),
        ([_steady(7450.0)], False),
        ([_steady(7450.0)], True),
    ):
        v4_verdict, v4_decision = speed_grade(
            _policy(4), tight, candidate, concluding=concluding
        )
        v5_verdict, v5_decision = speed_grade(
            _policy(5), tight, candidate, concluding=concluding
        )
        assert v5_verdict == v4_verdict
        assert v5_decision is v4_decision


def test_sealed_v3_and_v4_arithmetic_is_untouched_by_the_drift_rule() -> None:
    # The exact drifted-bookend evidence the drift rule exists for must keep
    # regrading identically under the sealed earlier versions.
    drifted = [_steady(7477.155), _steady(7303.009)]
    _, v3_concluded = speed_grade(
        _policy(3), drifted, [_steady(7400.0)], concluding=True
    )
    assert v3_concluded is SpeedStageDecision.NO_DECISION

    v4_verdict, v4_concluded = speed_grade(
        _policy(4), drifted, [_steady(7400.0)], concluding=True
    )
    assert v4_concluded is SpeedStageDecision.FAIL
    assert "v5" not in v4_verdict.detail


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

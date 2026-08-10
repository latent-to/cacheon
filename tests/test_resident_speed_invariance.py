"""Version-4 resident speed grading: a decidable run never returns NO_DECISION.

Version 3 refuses to produce a rate for a read whose own window scatter
exceeds the sealed bound, which converts settled results into non-answers.
Two mainnet runs are retained here as regression fixtures because both were
recorded as NO_DECISION while their evidence already determined the verdict.
Version 3 semantics must stay byte-identical -- sealed policies and retained
evidence depend on them -- so the new behavior is reachable only through a
prospectively sealed version-4 policy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cacheon.eval.crossover_runtime import (
    CrossoverRuntimeError,
    ResidentSpeedPolicy,
    SpeedStageDecision,
    TimedWindow,
    _invariant_decision,
    _speed_grade,
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
    assert _invariant_decision([7477.155, 7303.009], [7085.971], 1.005) is (
        SpeedStageDecision.FAIL
    )

    # Bundle 6d8d62ad: raw ungraded speedup 0.58197 withheld because window
    # scatter was 0.0526643 against a 0.05 ceiling. No drift of that size
    # reaches a 42% deficit.
    assert _invariant_decision([7400.0, 7400.0], [7400.0 * 0.58197], 1.005) is (
        SpeedStageDecision.FAIL
    )


def test_invariance_still_crowns_a_winner_and_still_hesitates_when_it_should() -> None:
    bookends = [7477.155, 7303.009]
    assert _invariant_decision(bookends, [7900.0], 1.005) is SpeedStageDecision.PASS
    assert _invariant_decision(bookends, [6900.0], 1.005) is SpeedStageDecision.FAIL
    # Wins against the low bookend, loses against the high one: the verdict
    # flips inside the observed spread, which is real ambiguity.
    assert _invariant_decision(bookends, [7400.0], 1.005) is None


def test_ambiguity_concludes_as_not_proven_rather_than_no_decision() -> None:
    # Ambiguity needs TIGHT bookends. The required margin scales with the
    # observed bookend spread, so widely drifting bookends raise the bar until
    # a marginal candidate is an invariant FAIL rather than an open question.
    policy = _policy(4)
    baselines = [_steady(7400.0), _steady(7420.0)]
    candidates = [_steady(7450.0)]

    # Before the extension an ambiguous read escalates rather than deciding.
    verdict, decision = _speed_grade(
        policy, baselines, candidates, concluding=False
    )
    assert decision is None and verdict.required > 1.0

    # Taking more reads only widens the observed spread, so the concluding
    # grade must terminate. The burden of proof sits with the candidate.
    _, concluded = _speed_grade(policy, baselines, candidates, concluding=True)
    assert concluded is SpeedStageDecision.FAIL


def test_version_three_concluding_behavior_is_unchanged() -> None:
    policy = _policy(3)
    baselines = [_steady(7477.155), _steady(7303.009)]
    _, concluded = _speed_grade(
        policy, baselines, [_steady(7400.0)], concluding=True
    )
    # The v3 bookends drift 2.36% against a 2% ceiling, so the verdict is not
    # confident and v3 answers NO_DECISION. Sealed v3 policies keep this
    # meaning exactly.
    assert concluded is SpeedStageDecision.NO_DECISION


def test_a_gross_loss_measured_on_an_unstable_box_still_fails() -> None:
    # The shape that produced the 6d8d62ad NO_DECISION: an unfit candidate read
    # plus a deficit far outside any plausible drift. Note the statistic is a
    # median absolute deviation, so breaching the bound requires dispersion
    # across most of the windows -- a lone outlier cannot do it.
    policy = _policy(4)
    baselines = [_steady(7477.155), _steady(7303.009)]
    unstable_candidate = _read(4400.0, 4800.0, 4000.0)
    assert policy.read_window_scatter(unstable_candidate) > policy.max_window_scatter

    _, decision = _speed_grade(
        policy, baselines, [unstable_candidate], concluding=False
    )
    assert decision is SpeedStageDecision.FAIL

    # The same evidence under v3 produces no verdict at all.
    with pytest.raises(CrossoverRuntimeError, match="window scatter exceeds"):
        _speed_grade(
            _policy(3), baselines, [unstable_candidate], concluding=False
        )

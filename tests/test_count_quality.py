from dataclasses import replace

import pytest

from cacheon.eval.count_quality import (
    CountQualityError,
    CountQualityEvidence,
    CountQualityPolicy,
    CountQualityVerdict,
    score_count_quality,
)


def _digest(character: str) -> str:
    return character * 64


def _evidence(*, stock: int = 62, candidate: int = 59) -> CountQualityEvidence:
    return CountQualityEvidence(
        stock_observation_digest=_digest("a"),
        candidate_observation_digest=_digest("b"),
        stock_correct=stock,
        candidate_correct=candidate,
        total=64,
    )


def test_drop_three_passes_under_first_failing_drop_ten() -> None:
    policy = CountQualityPolicy(regression_threshold_drop=10)
    verdict = score_count_quality(_evidence(), policy)

    assert verdict.decision == "PASS"
    assert verdict.observed_drop == 3
    assert verdict.regression_threshold_drop == 10


@pytest.mark.parametrize(
    ("candidate", "decision"),
    [
        (53, "PASS"),  # drop 9 is below the regression threshold
        (52, "FAIL"),  # drop 10 reaches the regression threshold
        (64, "PASS"),  # an improvement is never represented as a drop
    ],
)
def test_regression_threshold_boundary(candidate: int, decision: str) -> None:
    verdict = score_count_quality(
        _evidence(candidate=candidate),
        CountQualityPolicy(regression_threshold_drop=10),
    )

    assert verdict.decision == decision
    assert verdict.observed_drop == max(0, 62 - candidate)


def test_verdict_cannot_claim_pass_at_the_failure_boundary() -> None:
    evidence = _evidence(candidate=52)
    policy = CountQualityPolicy(regression_threshold_drop=10)
    verdict = score_count_quality(evidence, policy)

    with pytest.raises(CountQualityError, match="disagrees with policy arithmetic"):
        replace(verdict, decision="PASS")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CountQualityPolicy(regression_threshold_drop=0),
        lambda: CountQualityPolicy(regression_threshold_drop=True),
        lambda: CountQualityEvidence(_digest("a"), _digest("b"), 65, 59, 64),
        lambda: CountQualityEvidence(_digest("a"), _digest("b"), 62, -1, 64),
        lambda: CountQualityVerdict("MAYBE", 3, 10, _digest("a"), _digest("b")),
    ],
)
def test_invalid_policy_evidence_and_verdict_fail_closed(factory) -> None:
    with pytest.raises(CountQualityError):
        factory()

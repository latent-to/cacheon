"""The speed-failure reason vocabulary: graded with the verdict, never invented.

Pins the split introduced after retained report 162dee6a: a 1.00338x speedup
against a 1.01024x bar -- a miss inside the noise band -- was published as
"speed_regression", accusing an in-band candidate of slowing the stack down.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_runner import (
    QualificationRunnerError,
    _report_reason,
    _retained_reason,
)

_PASSING_GRAPH = SimpleNamespace(decision=QualificationDecision.PASS, reason=None)
_PASSING_AUDIT = SimpleNamespace(decision=QualificationDecision.PASS)


def test_report_carries_the_graded_band_reason() -> None:
    for reason in ("speed_threshold_not_met", "candidate_slower"):
        assert (
            _report_reason(
                _PASSING_GRAPH,
                QualificationDecision.FAIL,
                None,
                _PASSING_AUDIT,
                speed_reason=reason,
            )
            == reason
        )


def test_report_refuses_a_speed_fail_without_its_graded_reason() -> None:
    # The coarse legacy token is retained-report vocabulary; a new report may
    # never record it, and a missing reason is a plumbing fault, not a default.
    for reason in (None, "speed_regression", "made_up"):
        with pytest.raises(QualificationRunnerError, match="graded reason"):
            _report_reason(
                _PASSING_GRAPH,
                QualificationDecision.FAIL,
                None,
                _PASSING_AUDIT,
                speed_reason=reason,
            )


def test_retained_reports_revalidate_under_their_own_coarse_vocabulary() -> None:
    assert (
        _retained_reason("candidate_slower", "speed_regression")
        == "speed_regression"
    )
    assert (
        _retained_reason("speed_threshold_not_met", "speed_regression")
        == "speed_regression"
    )
    # The alias never runs the other way, and never covers non-band reasons.
    assert _retained_reason("candidate_slower", "candidate_slower") == "candidate_slower"
    assert _retained_reason("speed_noise", "speed_regression") == "speed_noise"
    assert _retained_reason("qualified", "speed_regression") == "qualified"

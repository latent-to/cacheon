"""Severe token losses contribute to quality while old attempts retain their original digest."""

from decimal import Decimal

import pytest

from cacheon.eval.calibration import MetricCalibration
from cacheon.eval.reference_quality import (
    PromptQualityEvidence, RawRolloutEvidence, RawTokenEvidence,
    ReferenceQualityError, TeacherNLLEvidence, _metric, _rollout,
)


def test_equal_means_and_maxima_can_have_different_excess_loss():
    def rollout(losses, measured):
        raw = RawRolloutEvidence(tuple(RawTokenEvidence(i, i, str(loss), None, None)
                                       for i, loss in enumerate(losses)), ())
        return _rollout(raw, Decimal(5), measure_excess=measured)

    baseline = rollout((20, 4, 4, 4), True)
    candidate = rollout((20, 12, 0, 0), True)
    assert baseline.teacher_nll.mean == candidate.teacher_nll.mean == 8
    assert baseline.teacher_nll.worst == candidate.teacher_nll.worst == 20
    prompt = PromptQualityEvidence("a" * 64, baseline, candidate, baseline, 4, 4)
    metric = MetricCalibration("tail_excess_nll", "lower", "1", "1")
    assert _metric(prompt, metric) == (3.75, 5.5, 3.75)
    assert TeacherNLLEvidence.from_dict(candidate.teacher_nll.to_dict()) == candidate.teacher_nll
    historic = rollout((20, 4, 4, 4), False)
    assert historic.teacher_nll.to_dict() == {"token_count": 4, "nll_sum": "32", "nll_max": "20", "tail_count": 1}
    assert TeacherNLLEvidence.from_dict(historic.teacher_nll.to_dict()) == historic.teacher_nll
    with pytest.raises(ReferenceQualityError, match="sealed per-token excess"):
        _metric(PromptQualityEvidence("b" * 64, historic, historic, historic, 4, 4), metric)

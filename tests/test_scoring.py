"""Unit tests for the noise-robust speedup scorer (cacheon/eval/scoring.py).

The whole point of this module is to make a sub-10% real win resolvable on a box
whose clocks can't be locked, and to refuse to crown on measurement noise. These
tests pin both halves: a genuine win passes, and noise alone never does.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from cacheon.eval.oci_outer_session import (
    OuterSessionInfrastructureError,
    require_decode_dominant_plan,
)
from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.scoring import (
    ChargedExecutionRate,
    RawSpeedEvidenceError,
    SpeedupVerdict,
    relative_spread,
    score_speedup,
)
from tests.test_marginal_runtime import _case as runtime_case
from tests.test_marginal_runtime import _prepared as prepared_runtime


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(label: str) -> str:
    return _digest(label)[:32]


def test_relative_spread_two_reads_is_range_over_mean():
    # The default bookend has exactly 2 baseline reads; spread is the honest gap.
    assert relative_spread([100.0, 110.0]) == (10.0 / 105.0)


def test_relative_spread_unmeasurable_below_two():
    assert relative_spread([100.0]) == float("inf")
    assert relative_spread([]) == float("inf")


def test_genuine_win_on_stable_box_passes():
    # Baselines agree (1% spread), candidate is a clean 12% faster -> real win.
    v = score_speedup([100.0, 101.0], 113.0, min_margin=0.02, k=2.0, max_noise=0.10)
    assert v.confident
    assert v.passed_speedup
    assert v.speedup > 1.11


def test_noise_alone_does_not_crown():
    # No real improvement (candidate ~= baseline mean) but candidate happens to read
    # a hair high; the noise-derived bar must reject it.
    v = score_speedup([100.0, 108.0], 106.0, min_margin=0.02, k=2.0, max_noise=0.10)
    # baseline spread = 8/104 ~= 7.7% -> required ~= 1 + 2*0.077 = 1.154; speedup ~1.019.
    assert not v.passed_speedup
    assert v.required > 1.15


def test_too_noisy_is_no_decision_not_a_pass():
    # Bracketing baselines disagree by >max_noise: untrustworthy round, never crown,
    # even if the raw ratio looks huge.
    v = score_speedup([100.0, 140.0], 150.0, max_noise=0.10)
    assert not v.confident
    assert not v.passed_speedup
    assert "NO-DECISION" in v.detail


def test_single_baseline_cannot_be_confident():
    # The legacy 2-launch shape (one baseline) can't measure noise -> not crownable.
    v = score_speedup([100.0], 130.0)
    assert not v.confident
    assert not v.passed_speedup
    assert v.noise == 0.0
    assert "single baseline" in v.detail
    codec = ContinuationCodec((SpeedupVerdict,))
    assert codec.decode(codec.encode(v)) == v


def test_min_margin_floor_applies_on_a_perfectly_stable_box():
    # Zero measured noise still requires clearing the floor margin.
    v = score_speedup([100.0, 100.0], 101.0, min_margin=0.02, k=2.0)
    assert v.noise == 0.0
    assert v.required == 1.02
    assert not v.passed_speedup  # 1.01 < 1.02
    v2 = score_speedup([100.0, 100.0], 103.0, min_margin=0.02, k=2.0)
    assert v2.passed_speedup  # 1.03 >= 1.02 on a stable box


def test_a_real_loss_is_a_loss_not_no_decision():
    v = score_speedup([100.0, 101.0], 90.0, max_noise=0.10)
    assert v.confident  # the box was stable; we trust the verdict
    assert not v.passed_speedup
    assert v.speedup < 1.0


def test_multi_candidate_reads_score_on_the_mean():
    # B C B' C' B'' shape: two candidate reads average before the ratio.
    v = score_speedup([100.0, 101.0, 99.0], [113.0, 111.0], min_margin=0.02, k=2.0, max_noise=0.10)
    assert v.n_candidates == 2
    assert v.confident
    assert v.passed_speedup
    assert abs(v.speedup - (112.0 / 100.0)) < 1e-9


def test_single_candidate_read_keeps_legacy_verdict():
    # The historical B/C/B' shape must be bit-identical through the new path.
    legacy = score_speedup([100.0, 108.0], 106.0, min_margin=0.02, k=2.0, max_noise=0.10)
    wrapped = score_speedup([100.0, 108.0], [106.0], min_margin=0.02, k=2.0, max_noise=0.10)
    assert legacy.n_candidates == wrapped.n_candidates == 1
    assert (legacy.speedup, legacy.noise, legacy.required, legacy.passed_speedup,
            legacy.confident, legacy.detail) == (
        wrapped.speedup, wrapped.noise, wrapped.required, wrapped.passed_speedup,
        wrapped.confident, wrapped.detail)


def test_noisy_candidate_reads_are_no_decision():
    # 2026-07-16 forensics: two honest candidate legs spread 7.2% on a boot draw.
    # With tight baselines, that spread alone must block the crown at max_noise 5%.
    v = score_speedup([100.0, 100.5], [107.2, 100.0], max_noise=0.05)
    assert not v.confident
    assert not v.passed_speedup
    assert "candidate drift" in v.detail


def test_candidate_spread_raises_the_required_bar():
    # Within the noise ceiling, a spread candidate raises the bar exactly like a
    # spread baseline: noise = max(baseline, candidate) feeds 1 + k*noise.
    v = score_speedup([100.0, 100.0], [104.0, 100.0], min_margin=0.005, k=2.0, max_noise=0.10)
    assert v.confident
    assert abs(v.noise - (4.0 / 102.0)) < 1e-9
    assert abs(v.required - (1.0 + 2.0 * 4.0 / 102.0)) < 1e-9
    assert not v.passed_speedup  # mean 102 -> 1.020 < required ~1.078


@pytest.mark.parametrize(
    ("baselines", "candidate"),
    (
        ([True, 100.0], 110.0),
        ([0.0, 100.0], 110.0),
        ([-1.0, 100.0], 110.0),
        ([float("nan"), 100.0], 110.0),
        ([float("inf"), 100.0], 110.0),
        ([100.0, 101.0], False),
        ([100.0, 101.0], 0.0),
        ([100.0, 101.0], float("inf")),
        ([100.0, 101.0], [110.0, 0.0]),
        ([100.0, 101.0], [110.0, float("nan")]),
        ([100.0, 101.0], [110.0, True]),
    ),
)
def test_speed_samples_fail_closed_without_filtering(baselines, candidate):
    with pytest.raises(RawSpeedEvidenceError):
        score_speedup(baselines, candidate)


@pytest.mark.parametrize(
    "policy",
    (
        {"min_margin": 0.0},
        {"min_margin": 1.0},
        {"min_margin": True},
        {"k": 0.0},
        {"k": float("inf")},
        {"max_noise": -0.1},
        {"max_noise": 1.0},
        {"max_noise": float("nan")},
    ),
)
def test_speed_policy_fails_closed(policy):
    with pytest.raises(RawSpeedEvidenceError):
        score_speedup([100.0, 101.0], 110.0, **policy)


def test_speed_witness_shape_decides_policy_and_recomputes():
    # The settlement byte contract: 3 rates = the historical B/C/B-prime shape,
    # 5 rates = repeat reads in run order, anything else refuses; the evidence
    # digest must recompute from the rates or construction fails closed.
    from cacheon.eval.qualification_runner import (
        QualificationRunnerError,
        SpeedEvidencePolicy,
        SpeedWitness,
    )
    from cacheon.eval.scoring import _projection_digest

    def rate(label: str) -> ChargedExecutionRate:
        return ChargedExecutionRate(
            _digest("launch:" + label),
            _binding("session:" + label),
            10, 20, 30, 1.0, 2.0, 3.0, 10.0,
        )

    heads = tuple(
        _digest("witness:" + name)
        for name in (
            "delta", "candidate-launch", "calibration",
            "context", "workload", "runtime-policy",
        )
    )
    legacy_rates = tuple(rate(role) for role in ("B", "C", "B-prime"))
    witness = SpeedWitness(
        *heads, _projection_digest(*heads, legacy_rates), legacy_rates
    )
    assert witness.policy == SpeedEvidencePolicy.legacy()
    assert SpeedWitness.from_dict(witness.to_dict()) == witness

    repeat_rates = legacy_rates + tuple(
        rate(role) for role in ("C-prime", "B-double-prime")
    )
    repeat = SpeedWitness(
        *heads, _projection_digest(*heads, repeat_rates), repeat_rates
    )
    assert repeat.policy == SpeedEvidencePolicy.repeat()
    assert SpeedWitness.from_dict(repeat.to_dict()) == repeat

    with pytest.raises(QualificationRunnerError, match="B/C/B-prime"):
        SpeedWitness(
            *heads, _projection_digest(*heads, legacy_rates[:2]), legacy_rates[:2]
        )
    with pytest.raises(QualificationRunnerError, match="does not recompute"):
        SpeedWitness(*heads, _digest("forged evidence"), legacy_rates)
    # A 3-rate witness can never be regraded under the repeat authority (and
    # vice versa): the policy check refuses before any calibration is read.
    with pytest.raises(QualificationRunnerError, match="policy differs"):
        witness.regrade(None, None, expected_policy=SpeedEvidencePolicy.repeat())
    with pytest.raises(QualificationRunnerError, match="policy differs"):
        repeat.regrade(None, None, expected_policy=SpeedEvidencePolicy.legacy())


def test_decode_dominant_plan_gate(tmp_path):
    case = runtime_case(tmp_path)
    case.session = replace(
        case.session,
        prompt_batches=(("warmup",), ("t1",), ("t2",)),
        max_new_tokens=10,
        top_logprobs_num=1,
    )
    plan = prepared_runtime(case).baseline_session_plan
    count_tokens = len  # chars-as-tokens keeps the gate arithmetic transparent
    charged = plan.prompt_batches[plan.warmup_count - plan.conditioning_count :]
    prompt_tokens = sum(len(prompt) for batch in charged for prompt in batch)
    decode_tokens = sum(len(batch) * plan.max_new_tokens for batch in charged)
    expected = decode_tokens / (decode_tokens + prompt_tokens)
    share = require_decode_dominant_plan(
        plan, count_tokens=count_tokens, min_decode_share=expected * 0.9
    )
    assert abs(share - expected) < 1e-12
    with pytest.raises(OuterSessionInfrastructureError, match="prefill-heavy"):
        require_decode_dominant_plan(plan, count_tokens=count_tokens, min_decode_share=0.99)
    with pytest.raises(OuterSessionInfrastructureError, match="min_decode_share"):
        require_decode_dominant_plan(plan, count_tokens=count_tokens, min_decode_share=1.0)
    with pytest.raises(OuterSessionInfrastructureError, match="positive ints"):
        require_decode_dominant_plan(
            plan, count_tokens=lambda prompt: 0, min_decode_share=0.5
        )

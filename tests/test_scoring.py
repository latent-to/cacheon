"""Unit tests for inference_engine.scoring — no GPU, no model download.

Tests the score() function with hand-crafted RunResult objects using
small vocab tensors.

Run with: pytest tests/test_scoring.py -v
"""

import math

import pytest
import torch

from inference_engine.harness import RunResult
from inference_engine.scoring import (
    MEMORY_WEIGHT,
    LATENCY_WEIGHT,
    QUALITY_THRESHOLD,
    ScoreResult,
    score,
    _compute_kl_from_logits,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    logits: list[torch.Tensor] | None = None,
    latency_s: float = 10.0,
    peak_memory_bytes: int = 1_000_000_000,
    policy_memory_bytes: int = 100_000_000,
) -> RunResult:
    """Build a RunResult with sensible defaults for testing."""
    if logits is None:
        logits = [torch.randn(5, 32)]
    n_prompts = len(logits)
    return RunResult(
        output_texts=["test"] * n_prompts,
        output_ids=[[1, 2, 3]] * n_prompts,
        all_logits=logits,
        latency_s=latency_s,
        peak_memory_bytes=peak_memory_bytes,
        policy_memory_bytes=policy_memory_bytes,
    )


def _uniform_logits(n_tokens: int, vocab_size: int) -> torch.Tensor:
    """Return uniform logits — produces a flat softmax distribution."""
    return torch.zeros(n_tokens, vocab_size)


def _shifted_logits(base_logits: torch.Tensor, noise_scale: float) -> torch.Tensor:
    """Add Gaussian noise to logits to create a controlled KL divergence."""
    return base_logits + torch.randn_like(base_logits) * noise_scale


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


class TestKLDivergence:
    def test_identical_logits_kl_zero(self):
        logits = torch.randn(10, 64)
        bl = _make_result(logits=[logits.clone()])
        mn = _make_result(logits=[logits.clone()])
        kl = _compute_kl_from_logits(bl.all_logits, mn.all_logits)
        assert abs(kl) < 1e-5, f"KL should be ~0 for identical logits, got {kl}"

    def test_different_logits_kl_positive(self):
        base = torch.randn(10, 64)
        shifted = base + torch.randn_like(base) * 5.0
        kl = _compute_kl_from_logits([base], [shifted])
        assert kl > 0.0, f"KL should be positive for different logits, got {kl}"

    def test_kl_averaged_across_tokens(self):
        """KL should be averaged per-token, not summed."""
        base = torch.randn(20, 64)
        shifted = base + torch.randn_like(base) * 2.0
        kl = _compute_kl_from_logits([base], [shifted])
        assert kl < 100.0, f"KL seems summed not averaged: {kl}"

    def test_kl_handles_token_count_mismatch(self):
        """Truncate to shorter when token counts differ."""
        base = torch.randn(10, 32)
        shorter = torch.randn(7, 32)
        kl = _compute_kl_from_logits([base], [shorter])
        assert isinstance(kl, float)

    def test_kl_multiple_prompts(self):
        """KL is averaged across all tokens from all prompts."""
        base1 = torch.randn(5, 32)
        base2 = torch.randn(8, 32)
        kl = _compute_kl_from_logits(
            [base1.clone(), base2.clone()],
            [base1.clone(), base2.clone()],
        )
        assert abs(kl) < 1e-5

    def test_kl_empty_logits(self):
        empty = torch.randn(0, 32)
        kl = _compute_kl_from_logits([empty], [empty])
        assert kl == 0.0


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------


class TestInvalidLogits:
    def test_nan_in_miner_logits_disqualifies(self):
        base = torch.randn(5, 32)
        bad = torch.randn(5, 32)
        bad[2, 10] = float("nan")
        result = score(
            _make_result(logits=[base]),
            _make_result(logits=[bad]),
        )
        assert result.disqualified
        assert "NaN" in result.disqualify_reason
        assert result.score == 0.0
        assert result.kl_divergence == float("inf")

    def test_inf_in_miner_logits_disqualifies(self):
        base = torch.randn(5, 32)
        bad = torch.randn(5, 32)
        bad[0, 0] = float("inf")
        result = score(
            _make_result(logits=[base]),
            _make_result(logits=[bad]),
        )
        assert result.disqualified
        assert "Inf" in result.disqualify_reason
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Quality gate (KL threshold)
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_low_kl_passes_gate(self):
        logits = torch.randn(10, 64)
        result = score(
            _make_result(logits=[logits.clone()]),
            _make_result(logits=[logits.clone()]),
        )
        assert not result.disqualified
        assert result.disqualify_reason is None
        assert result.kl_divergence < QUALITY_THRESHOLD

    def test_high_kl_fails_gate(self):
        base = torch.randn(10, 64)
        # Large noise guarantees KL >> threshold
        noisy = base + torch.randn_like(base) * 20.0
        result = score(
            _make_result(logits=[base]),
            _make_result(logits=[noisy]),
        )
        assert result.disqualified
        assert result.score == 0.0
        assert "KL divergence" in result.disqualify_reason


# ---------------------------------------------------------------------------
# Memory reduction
# ---------------------------------------------------------------------------


class TestMemoryReduction:
    def test_miner_uses_less_memory(self):
        result = score(
            _make_result(
                logits=[torch.zeros(5, 32)], policy_memory_bytes=1_000_000_000
            ),
            _make_result(logits=[torch.zeros(5, 32)], policy_memory_bytes=500_000_000),
        )
        assert result.memory_reduction == pytest.approx(0.5)

    def test_miner_uses_more_memory(self):
        result = score(
            _make_result(
                logits=[torch.zeros(5, 32)], policy_memory_bytes=1_000_000_000
            ),
            _make_result(
                logits=[torch.zeros(5, 32)], policy_memory_bytes=1_500_000_000
            ),
        )
        assert result.memory_reduction == pytest.approx(-0.5)

    def test_identical_memory(self):
        result = score(
            _make_result(
                logits=[torch.zeros(5, 32)], policy_memory_bytes=1_000_000_000
            ),
            _make_result(
                logits=[torch.zeros(5, 32)], policy_memory_bytes=1_000_000_000
            ),
        )
        assert result.memory_reduction == pytest.approx(0.0)

    def test_zero_baseline_memory_handled(self):
        result = score(
            _make_result(logits=[torch.zeros(5, 32)], policy_memory_bytes=0),
            _make_result(logits=[torch.zeros(5, 32)], policy_memory_bytes=500),
        )
        assert result.memory_reduction == 0.0

    def test_memory_reduction_clamped(self):
        result = score(
            _make_result(logits=[torch.zeros(5, 32)], policy_memory_bytes=100),
            _make_result(logits=[torch.zeros(5, 32)], policy_memory_bytes=500),
        )
        assert result.memory_reduction >= -1.0


# ---------------------------------------------------------------------------
# Latency improvement
# ---------------------------------------------------------------------------


class TestLatencyImprovement:
    def test_miner_faster(self):
        result = score(
            _make_result(logits=[torch.zeros(5, 32)], latency_s=10.0),
            _make_result(logits=[torch.zeros(5, 32)], latency_s=6.0),
        )
        assert result.latency_improvement == pytest.approx(0.4)

    def test_miner_slower(self):
        result = score(
            _make_result(logits=[torch.zeros(5, 32)], latency_s=10.0),
            _make_result(logits=[torch.zeros(5, 32)], latency_s=15.0),
        )
        assert result.latency_improvement == pytest.approx(-0.5)

    def test_identical_latency(self):
        result = score(
            _make_result(logits=[torch.zeros(5, 32)], latency_s=10.0),
            _make_result(logits=[torch.zeros(5, 32)], latency_s=10.0),
        )
        assert result.latency_improvement == pytest.approx(0.0)

    def test_zero_baseline_latency_handled(self):
        result = score(
            _make_result(logits=[torch.zeros(5, 32)], latency_s=0.0),
            _make_result(logits=[torch.zeros(5, 32)], latency_s=5.0),
        )
        assert result.latency_improvement == 0.0


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------


class TestScoreFormula:
    def test_hand_computed_score(self):
        """memory_reduction=0.5, latency_improvement=0.2 → 0.6*0.5 + 0.4*0.2 = 0.38"""
        result = score(
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=1_000_000_000,
                latency_s=10.0,
            ),
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=500_000_000,
                latency_s=8.0,
            ),
        )
        assert not result.disqualified
        expected = MEMORY_WEIGHT * 0.5 + LATENCY_WEIGHT * 0.2
        assert result.score == pytest.approx(expected, abs=1e-6)

    def test_hand_computed_score_2(self):
        """memory_reduction=0.3, latency_improvement=0.5 → 0.6*0.3 + 0.4*0.5 = 0.38"""
        result = score(
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=1_000_000_000,
                latency_s=10.0,
            ),
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=700_000_000,
                latency_s=5.0,
            ),
        )
        assert not result.disqualified
        expected = MEMORY_WEIGHT * 0.3 + LATENCY_WEIGHT * 0.5
        assert result.score == pytest.approx(expected, abs=1e-6)

    def test_hand_computed_score_3(self):
        """Both worse: memory_reduction=-0.5, latency_improvement=-0.5 → -0.5"""
        result = score(
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=1_000_000_000,
                latency_s=10.0,
            ),
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=1_500_000_000,
                latency_s=15.0,
            ),
        )
        assert not result.disqualified
        expected = MEMORY_WEIGHT * (-0.5) + LATENCY_WEIGHT * (-0.5)
        assert result.score == pytest.approx(expected, abs=1e-6)

    def test_disqualified_score_is_zero(self):
        """Even if memory/latency are great, KL failure → score = 0."""
        base = torch.randn(10, 64)
        noisy = base + torch.randn_like(base) * 20.0
        result = score(
            _make_result(
                logits=[base],
                policy_memory_bytes=1_000_000_000,
                latency_s=10.0,
            ),
            _make_result(
                logits=[noisy],
                policy_memory_bytes=100_000_000,
                latency_s=1.0,
            ),
        )
        assert result.disqualified
        assert result.score == 0.0

    def test_score_can_be_negative(self):
        """A miner that's worse on both axes gets a negative score."""
        result = score(
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=1_000_000_000,
                latency_s=10.0,
            ),
            _make_result(
                logits=[torch.zeros(5, 32)],
                policy_memory_bytes=2_000_000_000,
                latency_s=20.0,
            ),
        )
        assert not result.disqualified
        assert result.score < 0.0


# ---------------------------------------------------------------------------
# Teacher-forced logits
# ---------------------------------------------------------------------------


class TestTeacherForcedLogits:
    def test_tf_logits_used_for_kl(self):
        """When teacher_forced_logits is passed, KL uses those instead
        of miner.all_logits."""
        base = torch.randn(10, 64)
        miner_auto = base + torch.randn_like(base) * 20.0  # large KL
        miner_tf = base.clone()  # ~zero KL

        result = score(
            _make_result(logits=[base]),
            _make_result(logits=[miner_auto]),
            teacher_forced_logits=[miner_tf],
        )
        assert not result.disqualified
        assert result.kl_divergence < 1e-5

    def test_tf_logits_none_falls_back_to_autoregressive(self):
        """With teacher_forced_logits=None, the old path is used."""
        base = torch.randn(10, 64)
        miner_auto = base + torch.randn_like(base) * 20.0

        r1 = score(
            _make_result(logits=[base]),
            _make_result(logits=[miner_auto]),
        )
        r2 = score(
            _make_result(logits=[base]),
            _make_result(logits=[miner_auto]),
            teacher_forced_logits=None,
        )
        assert r1.kl_divergence == r2.kl_divergence

    def test_tf_logits_nan_disqualifies(self):
        base = torch.randn(5, 32)
        bad_tf = torch.randn(5, 32)
        bad_tf[1, 5] = float("nan")
        result = score(
            _make_result(logits=[base]),
            _make_result(logits=[base.clone()]),
            teacher_forced_logits=[bad_tf],
        )
        assert result.disqualified
        assert "NaN" in result.disqualify_reason

    def test_tf_logits_inf_disqualifies(self):
        base = torch.randn(5, 32)
        bad_tf = torch.randn(5, 32)
        bad_tf[0, 0] = float("inf")
        result = score(
            _make_result(logits=[base]),
            _make_result(logits=[base.clone()]),
            teacher_forced_logits=[bad_tf],
        )
        assert result.disqualified
        assert "Inf" in result.disqualify_reason

    def test_tf_logits_memory_latency_from_miner_run(self):
        """Memory and latency come from the autoregressive miner run,
        not the teacher-forced logits."""
        base = torch.zeros(5, 32)
        result = score(
            _make_result(
                logits=[base], policy_memory_bytes=1_000_000_000, latency_s=10.0
            ),
            _make_result(logits=[base], policy_memory_bytes=500_000_000, latency_s=6.0),
            teacher_forced_logits=[base.clone()],
        )
        assert result.memory_reduction == pytest.approx(0.5)
        assert result.latency_improvement == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Latency overrides (interleaved measurement)
# ---------------------------------------------------------------------------


class TestLatencyOverrides:
    def test_overrides_used_when_provided(self):
        """Latency override params replace RunResult.latency_s."""
        base = torch.zeros(5, 32)
        result = score(
            _make_result(logits=[base], latency_s=100.0),
            _make_result(logits=[base], latency_s=100.0),
            baseline_latency_s=10.0,
            miner_latency_s=6.0,
        )
        assert result.latency_improvement == pytest.approx(0.4)

    def test_none_overrides_fall_back_to_run_result(self):
        """None overrides use RunResult.latency_s (backward compat)."""
        base = torch.zeros(5, 32)
        r1 = score(
            _make_result(logits=[base], latency_s=10.0),
            _make_result(logits=[base], latency_s=6.0),
        )
        r2 = score(
            _make_result(logits=[base], latency_s=10.0),
            _make_result(logits=[base], latency_s=6.0),
            baseline_latency_s=None,
            miner_latency_s=None,
        )
        assert r1.latency_improvement == r2.latency_improvement

    def test_overrides_affect_score_not_memory_or_kl(self):
        """Latency overrides change score but not memory_reduction or kl."""
        base = torch.zeros(5, 32)
        r_default = score(
            _make_result(
                logits=[base], policy_memory_bytes=1_000_000_000, latency_s=10.0
            ),
            _make_result(logits=[base], policy_memory_bytes=500_000_000, latency_s=8.0),
        )
        r_override = score(
            _make_result(
                logits=[base], policy_memory_bytes=1_000_000_000, latency_s=10.0
            ),
            _make_result(logits=[base], policy_memory_bytes=500_000_000, latency_s=8.0),
            baseline_latency_s=10.0,
            miner_latency_s=5.0,
        )
        assert r_default.memory_reduction == r_override.memory_reduction
        assert r_default.kl_divergence == r_override.kl_divergence
        assert r_default.score != r_override.score
        assert r_override.latency_improvement == pytest.approx(0.5)

    def test_partial_override_baseline_only(self):
        """Only baseline_latency_s overridden; miner uses RunResult."""
        base = torch.zeros(5, 32)
        result = score(
            _make_result(logits=[base], latency_s=999.0),
            _make_result(logits=[base], latency_s=6.0),
            baseline_latency_s=10.0,
        )
        assert result.latency_improvement == pytest.approx(0.4)

    def test_partial_override_miner_only(self):
        """Only miner_latency_s overridden; baseline uses RunResult."""
        base = torch.zeros(5, 32)
        result = score(
            _make_result(logits=[base], latency_s=10.0),
            _make_result(logits=[base], latency_s=999.0),
            miner_latency_s=6.0,
        )
        assert result.latency_improvement == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# ScoreResult structure
# ---------------------------------------------------------------------------


class TestScoreResultStructure:
    def test_all_fields_present(self):
        logits = torch.zeros(5, 32)
        result = score(
            _make_result(logits=[logits]),
            _make_result(logits=[logits]),
        )
        assert isinstance(result, ScoreResult)
        assert isinstance(result.kl_divergence, float)
        assert isinstance(result.memory_reduction, float)
        assert isinstance(result.latency_improvement, float)
        assert isinstance(result.score, float)
        assert isinstance(result.disqualified, bool)

"""Unit tests for validator.scoring -- pure math, no I/O."""

from __future__ import annotations

import pytest

from validator.scoring import (
    CorrectnessVerdict,
    check_logprob_sanity,
    compute_correctness,
    compute_improvements,
    compute_token_match_rate,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# compute_token_match_rate
# --------------------------------------------------------------------------- #


class TestTokenMatchRate:
    def test_exact_match(self):
        tokens = ["hello", "world", "foo"]
        assert compute_token_match_rate(tokens, tokens) == 1.0

    def test_one_mismatch_in_100(self):
        base = [f"t{i}" for i in range(100)]
        miner = list(base)
        miner[50] = "WRONG"
        assert compute_token_match_rate(base, miner) == pytest.approx(0.99)

    def test_all_different(self):
        base = ["a", "b", "c"]
        miner = ["x", "y", "z"]
        assert compute_token_match_rate(base, miner) == pytest.approx(0.0)

    def test_empty_both(self):
        assert compute_token_match_rate([], []) == 1.0

    def test_miner_shorter(self):
        base = ["a", "b", "c", "d"]
        miner = ["a", "b"]
        assert compute_token_match_rate(base, miner) == pytest.approx(0.5)

    def test_miner_longer(self):
        base = ["a"]
        miner = ["a", "b", "c"]
        assert compute_token_match_rate(base, miner) == pytest.approx(1 / 3)

    def test_single_token_match(self):
        assert compute_token_match_rate(["x"], ["x"]) == 1.0

    def test_single_token_mismatch(self):
        assert compute_token_match_rate(["x"], ["y"]) == 0.0


# --------------------------------------------------------------------------- #
# check_logprob_sanity
# --------------------------------------------------------------------------- #


class TestLogprobSanity:
    def test_baseline_in_top1(self):
        lp = [{"token": "the", "logprob": -0.01}, {"token": "a", "logprob": -0.03}]
        assert check_logprob_sanity("the", lp) is True

    def test_baseline_in_top2(self):
        lp = [{"token": "a", "logprob": -0.10}, {"token": "the", "logprob": -0.12}]
        assert check_logprob_sanity("the", lp) is True

    def test_baseline_in_top5(self):
        lp = [
            {"token": "a", "logprob": -0.10},
            {"token": "b", "logprob": -0.20},
            {"token": "c", "logprob": -0.30},
            {"token": "d", "logprob": -0.40},
            {"token": "the", "logprob": -0.45},
        ]
        assert check_logprob_sanity("the", lp) is True

    def test_baseline_not_in_top5(self):
        lp = [
            {"token": "a", "logprob": -0.10},
            {"token": "b", "logprob": -0.20},
            {"token": "c", "logprob": -0.30},
            {"token": "d", "logprob": -0.40},
            {"token": "e", "logprob": -0.50},
        ]
        assert check_logprob_sanity("the", lp) is False

    def test_gap_within_threshold(self):
        lp = [{"token": "a", "logprob": -0.10}, {"token": "the", "logprob": -0.50}]
        assert check_logprob_sanity("the", lp) is True

    def test_gap_too_large(self):
        lp = [{"token": "a", "logprob": -0.10}, {"token": "the", "logprob": -0.70}]
        assert check_logprob_sanity("the", lp) is False

    def test_gap_exactly_at_threshold(self):
        lp = [{"token": "a", "logprob": -0.10}, {"token": "the", "logprob": -0.60}]
        assert check_logprob_sanity("the", lp, max_gap=0.5) is True

    def test_gap_just_over_threshold(self):
        lp = [{"token": "a", "logprob": -0.10}, {"token": "the", "logprob": -0.61}]
        assert check_logprob_sanity("the", lp, max_gap=0.5) is False

    def test_empty_logprobs(self):
        assert check_logprob_sanity("x", []) is False

    def test_single_entry_is_baseline(self):
        lp = [{"token": "the", "logprob": -0.01}]
        assert check_logprob_sanity("the", lp) is True

    def test_tp_noise_real_example(self):
        """Based on real TP=2 logs: ' reveal' vs ' fully' with identical logprobs."""
        lp = [
            {"token": " fully", "logprob": -0.843},
            {"token": " reveal", "logprob": -0.843},
            {"token": " disclose", "logprob": -2.468},
        ]
        assert check_logprob_sanity(" reveal", lp) is True

    def test_quantized_model_fails(self):
        """Quantized model: baseline token far from top-1."""
        lp = [
            {"token": "wrong", "logprob": -0.10},
            {"token": "also_wrong", "logprob": -0.50},
            {"token": "nope", "logprob": -1.00},
            {"token": "nah", "logprob": -1.50},
            {"token": "the", "logprob": -2.00},
        ]
        assert check_logprob_sanity("the", lp) is False


# --------------------------------------------------------------------------- #
# compute_correctness
# --------------------------------------------------------------------------- #


class TestComputeCorrectness:
    def test_perfect_match_passes(self):
        tokens = ["a", "b", "c", "d", "e"]
        v = compute_correctness(tokens, tokens, None)
        assert v.passed is True
        assert v.token_match_rate == 1.0
        assert v.first_mismatch_index is None

    def test_tp_noise_divergence_passes(self):
        """Sequence diverges at index 2 due to TP noise -- should pass."""
        base = ["a", "b", "c", "d", "e"]
        miner = ["a", "b", "X", "Y", "Z"]
        lp = [
            [{"token": "a", "logprob": -0.01}],
            [{"token": "b", "logprob": -0.01}],
            [{"token": "X", "logprob": -0.50}, {"token": "c", "logprob": -0.52}],
            [{"token": "Y", "logprob": -0.01}],
            [{"token": "Z", "logprob": -0.01}],
        ]
        v = compute_correctness(base, miner, lp)
        assert v.passed is True
        assert v.first_mismatch_index == 2
        assert v.token_match_rate == pytest.approx(2 / 5)

    def test_wrong_model_fails(self):
        """Baseline token not in miner's top-5 -- wrong model."""
        base = ["a", "b", "c"]
        miner = ["a", "WRONG", "c"]
        lp = [
            [{"token": "a", "logprob": -0.01}],
            [
                {"token": "WRONG", "logprob": -0.01},
                {"token": "x", "logprob": -0.50},
                {"token": "y", "logprob": -1.00},
                {"token": "z", "logprob": -1.50},
                {"token": "w", "logprob": -2.00},
            ],
            [{"token": "c", "logprob": -0.01}],
        ]
        v = compute_correctness(base, miner, lp)
        assert v.passed is False
        assert v.first_mismatch_index == 1
        assert "first_mismatch_fail" in (v.reason or "")

    def test_gap_too_large_fails(self):
        """Baseline token in top-5 but logprob gap too large -- quantized model."""
        base = ["a", "b", "c"]
        miner = ["a", "X", "c"]
        lp = [
            [{"token": "a", "logprob": -0.01}],
            [{"token": "X", "logprob": -0.10}, {"token": "b", "logprob": -0.80}],
            [{"token": "c", "logprob": -0.01}],
        ]
        v = compute_correctness(base, miner, lp)
        assert v.passed is False
        assert v.first_mismatch_index == 1

    def test_logprob_sanity_pass_at_first_mismatch(self):
        base = ["a", "b", "c"]
        miner = ["a", "X", "c"]
        lp = [
            [{"token": "a", "logprob": -0.01}],
            [{"token": "X", "logprob": -0.10}, {"token": "b", "logprob": -0.12}],
            [{"token": "c", "logprob": -0.01}],
        ]
        v = compute_correctness(base, miner, lp)
        assert v.passed is True
        assert v.first_mismatch_index == 1

    def test_no_logprobs_with_divergence_fails(self):
        """Miner omitting logprobs on a divergent response is non-compliant."""
        base = ["a", "b", "c"]
        miner = ["a", "X", "c"]
        v = compute_correctness(base, miner, None)
        assert v.passed is False
        assert v.first_mismatch_index == 1
        assert "logprobs missing" in (v.reason or "")

    def test_no_logprobs_perfect_match_passes(self):
        """Perfect match without logprobs is fine -- no divergence to check."""
        tokens = ["a", "b", "c"]
        v = compute_correctness(tokens, tokens, None)
        assert v.passed is True

    def test_empty_miner_with_logprobs_fails(self):
        base = ["a", "b"]
        miner: list[str] = []
        v = compute_correctness(base, miner, [])
        assert v.passed is False
        assert v.first_mismatch_index == 0
        assert v.baseline_token_at_mismatch == "a"
        assert v.miner_token_at_mismatch == ""

    def test_empty_miner_without_logprobs_fails(self):
        base = ["a", "b"]
        miner: list[str] = []
        v = compute_correctness(base, miner, None)
        assert v.passed is False
        assert "logprobs missing" in (v.reason or "")

    def test_non_greedy_miner_fails(self):
        """Miner's chosen token != its own top-1 -- sampling detected."""
        base = ["a", "b", "c"]
        miner = ["a", "X", "c"]
        lp = [
            [{"token": "a", "logprob": -0.01}],
            [{"token": "b", "logprob": -0.10}, {"token": "X", "logprob": -0.12}],
            [{"token": "c", "logprob": -0.01}],
        ]
        v = compute_correctness(base, miner, lp)
        assert v.passed is False
        assert "non_greedy" in (v.reason or "")

    def test_real_tp_noise_example(self):
        """Real data from TP=2 eval: 'you' vs 'provided' with gap 0.125."""
        base = ["The", " passage", " you"]
        miner = ["The", " passage", " provided"]
        lp = [
            [{"token": "The", "logprob": -0.01}],
            [{"token": " passage", "logprob": -0.01}],
            [
                {"token": " provided", "logprob": -1.051},
                {"token": " you", "logprob": -1.176},
                {"token": " from", "logprob": -1.551},
                {"token": " is", "logprob": -3.176},
                {"token": " has", "logprob": -4.051},
            ],
        ]
        v = compute_correctness(base, miner, lp)
        assert v.passed is True


# --------------------------------------------------------------------------- #
# compute_improvements
# --------------------------------------------------------------------------- #


class TestComputeImprovements:
    def test_miner_faster_on_both_axes(self):
        bl_ttft = [1.0, 1.0, 1.0]
        mn_ttft = [0.5, 0.5, 0.5]
        bl_tps = [100.0, 100.0, 100.0]
        mn_tps = [150.0, 150.0, 150.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == pytest.approx(0.5)
        assert score == pytest.approx(0.5)

    def test_miner_slower_floors_at_zero(self):
        bl_ttft = [0.5]
        mn_ttft = [1.0]
        bl_tps = [100.0]
        mn_tps = [50.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == 0.0
        assert tps_imp == 0.0
        assert score == 0.0

    def test_mixed_axes(self):
        bl_ttft = [1.0]
        mn_ttft = [0.8]
        bl_tps = [100.0]
        mn_tps = [80.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == pytest.approx(0.2)
        assert tps_imp == 0.0
        assert score == pytest.approx(0.1)

    def test_median_with_outlier(self):
        bl_ttft = [1.0, 1.0, 1.0, 1.0, 100.0]
        mn_ttft = [0.5, 0.5, 0.5, 0.5, 0.5]
        bl_tps = [100.0, 100.0, 100.0, 100.0, 100.0]
        mn_tps = [120.0, 120.0, 120.0, 120.0, 120.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == pytest.approx(0.2)

    def test_single_prompt(self):
        score, ttft_imp, tps_imp = compute_improvements([2.0], [1.0], [50.0], [75.0])
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == pytest.approx(0.5)
        assert score == pytest.approx(0.5)

    def test_even_number_of_prompts(self):
        score, _, _ = compute_improvements(
            [1.0, 2.0], [0.5, 1.0], [100.0, 100.0], [150.0, 150.0]
        )
        assert score == pytest.approx(0.5)

    def test_empty_lists_return_zero(self):
        assert compute_improvements([], [], [], []) == (0.0, 0.0, 0.0)

    def test_empty_baseline_ttft(self):
        assert compute_improvements([], [1.0], [100.0], [150.0]) == (0.0, 0.0, 0.0)

    def test_zero_baseline_ttft(self):
        score, ttft_imp, tps_imp = compute_improvements([0.0], [0.5], [100.0], [150.0])
        assert ttft_imp == 0.0
        assert tps_imp == pytest.approx(0.5)

    def test_zero_baseline_tps(self):
        score, ttft_imp, tps_imp = compute_improvements([1.0], [0.5], [0.0], [150.0])
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == 0.0

    def test_identical_performance(self):
        score, ttft_imp, tps_imp = compute_improvements([1.0], [1.0], [100.0], [100.0])
        assert ttft_imp == 0.0
        assert tps_imp == 0.0
        assert score == 0.0

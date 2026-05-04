"""Correctness checking and scoring for containerized evaluation.

Pure math -- no I/O, no Docker, no bittensor. All inputs are lists of
floats or strings produced by the HTTP client in ``docker_eval``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CorrectnessVerdict:
    """Result of the greedy-token agreement gate + logprob sanity check.

    On failure, ``first_mismatch_*`` fields capture the first divergent
    position so operators can debug false DQs without re-running the eval.
    """

    passed: bool
    token_match_rate: float
    reason: str | None = None
    first_mismatch_index: int | None = None
    baseline_token_at_mismatch: str | None = None
    miner_token_at_mismatch: str | None = None
    miner_logprobs_at_mismatch: list[dict[str, Any]] | None = None


def compute_token_match_rate(
    baseline_tokens: list[str],
    miner_tokens: list[str],
) -> float:
    """Fraction of positions where tokens agree (0.0 -- 1.0).

    Length mismatch: positions beyond the shorter list count as mismatches.
    """
    if not baseline_tokens and not miner_tokens:
        return 1.0
    total = max(len(baseline_tokens), len(miner_tokens))
    matches = sum(b == m for b, m in zip(baseline_tokens, miner_tokens))
    return matches / total


def check_logprob_sanity(
    baseline_token: str,
    miner_top_logprobs: list[dict[str, Any]],
    max_gap: float = 0.05,
) -> bool:
    """Check that a divergent position is a legitimate numerical tie.

    Returns True (sane) when:
      1. The baseline's chosen token appears in the miner's top-2, AND
      2. The logprob gap between the miner's top-1 and the baseline token
         is <= ``max_gap``.
    """
    if not miner_top_logprobs:
        return False
    top2_tokens = [entry.get("token", "") for entry in miner_top_logprobs[:2]]
    if baseline_token not in top2_tokens:
        return False
    top1_lp = float(miner_top_logprobs[0].get("logprob", float("-inf")))
    baseline_lp = top1_lp
    for entry in miner_top_logprobs[:2]:
        if entry.get("token") == baseline_token:
            baseline_lp = float(entry.get("logprob", float("-inf")))
            break
    return abs(top1_lp - baseline_lp) <= max_gap


def compute_correctness(
    baseline_tokens: list[str],
    miner_tokens: list[str],
    miner_top_logprobs: list[list[dict[str, Any]]] | None,
    threshold: float = 0.99,
) -> CorrectnessVerdict:
    """Greedy-token agreement gate with logprob sanity at divergent positions.

    Steps:
      1. Compute token match rate.
      2. If below ``threshold``, fail immediately with mismatch details.
      3. At each divergent position, run ``check_logprob_sanity``. Fail on
         the first position that doesn't pass.
    """
    rate = compute_token_match_rate(baseline_tokens, miner_tokens)

    first_mm_idx: int | None = None
    first_mm_baseline: str | None = None
    first_mm_miner: str | None = None
    first_mm_lp: list[dict[str, Any]] | None = None

    total = max(len(baseline_tokens), len(miner_tokens))
    for i in range(total):
        bt = baseline_tokens[i] if i < len(baseline_tokens) else ""
        mt = miner_tokens[i] if i < len(miner_tokens) else ""
        if bt != mt and first_mm_idx is None:
            first_mm_idx = i
            first_mm_baseline = bt
            first_mm_miner = mt
            if miner_top_logprobs and i < len(miner_top_logprobs):
                first_mm_lp = miner_top_logprobs[i]
            break

    if rate < threshold:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=rate,
            reason=(
                f"token_match_rate {rate:.4f} < {threshold} "
                f"(first mismatch at index {first_mm_idx})"
            ),
            first_mismatch_index=first_mm_idx,
            baseline_token_at_mismatch=first_mm_baseline,
            miner_token_at_mismatch=first_mm_miner,
            miner_logprobs_at_mismatch=first_mm_lp,
        )

    if miner_top_logprobs is not None:
        for i in range(min(len(baseline_tokens), len(miner_tokens))):
            if baseline_tokens[i] == miner_tokens[i]:
                continue
            lp = miner_top_logprobs[i] if i < len(miner_top_logprobs) else []
            if not check_logprob_sanity(baseline_tokens[i], lp):
                return CorrectnessVerdict(
                    passed=False,
                    token_match_rate=rate,
                    reason=(
                        f"logprob_sanity_fail at index {i}: "
                        f"baseline={baseline_tokens[i]!r} not in miner top-2 "
                        f"or gap too large"
                    ),
                    first_mismatch_index=i,
                    baseline_token_at_mismatch=baseline_tokens[i],
                    miner_token_at_mismatch=miner_tokens[i],
                    miner_logprobs_at_mismatch=lp,
                )

    return CorrectnessVerdict(
        passed=True,
        token_match_rate=rate,
        first_mismatch_index=first_mm_idx,
        baseline_token_at_mismatch=first_mm_baseline,
        miner_token_at_mismatch=first_mm_miner,
        miner_logprobs_at_mismatch=first_mm_lp,
    )


def compute_improvements(
    baseline_ttfts: list[float],
    miner_ttfts: list[float],
    baseline_tps_list: list[float],
    miner_tps_list: list[float],
) -> tuple[float, float, float]:
    """Compute the final score from per-prompt timing measurements.

    Returns ``(score, ttft_improvement, throughput_improvement)``.

    1. Take median TTFT and median throughput across prompts.
    2. Compute relative improvement vs. baseline, floored at 0.
    3. Score = 0.5 * ttft_improvement + 0.5 * throughput_improvement.
    """
    if not baseline_ttfts or not miner_ttfts:
        return (0.0, 0.0, 0.0)
    if not baseline_tps_list or not miner_tps_list:
        return (0.0, 0.0, 0.0)

    med_bl_ttft = statistics.median(baseline_ttfts)
    med_mn_ttft = statistics.median(miner_ttfts)
    med_bl_tps = statistics.median(baseline_tps_list)
    med_mn_tps = statistics.median(miner_tps_list)

    if med_bl_ttft > 0:
        ttft_imp = max(0.0, (med_bl_ttft - med_mn_ttft) / med_bl_ttft)
    else:
        ttft_imp = 0.0

    if med_bl_tps > 0:
        tps_imp = max(0.0, (med_mn_tps - med_bl_tps) / med_bl_tps)
    else:
        tps_imp = 0.0

    score = 0.5 * ttft_imp + 0.5 * tps_imp
    return (score, ttft_imp, tps_imp)

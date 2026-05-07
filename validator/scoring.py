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
    max_gap: float = 0.5,
    top_k: int = 5,
) -> bool:
    """Check that a divergent position is explainable by TP numerical noise.

    Returns True (sane) when:
      1. The baseline's token appears in the miner's top-k.
      2. The logprob gap between the miner's top-1 and the baseline token
         is <= ``max_gap``.

    Greedy-decoding verification (miner's chosen token == its own top-1)
    is checked separately in ``compute_correctness``.
    """
    if not miner_top_logprobs:
        return False
    topk_tokens = [entry.get("token", "") for entry in miner_top_logprobs[:top_k]]
    if baseline_token not in topk_tokens:
        return False
    top1_lp = float(miner_top_logprobs[0].get("logprob", float("-inf")))
    for entry in miner_top_logprobs[:top_k]:
        if entry.get("token") == baseline_token:
            baseline_lp = float(entry.get("logprob", float("-inf")))
            return abs(top1_lp - baseline_lp) <= max_gap
    return False


def compute_correctness(
    baseline_tokens: list[str],
    miner_tokens: list[str],
    miner_top_logprobs: list[list[dict[str, Any]]] | None,
) -> CorrectnessVerdict:
    """First-mismatch correctness gate for TP-safe greedy decoding.

    With tensor parallelism, greedy outputs are non-deterministic at
    positions where two tokens have near-identical probabilities.  Once
    one token flips, the entire rest of the sequence cascades, making
    token-match-rate useless.

    Instead we check only the **first** divergence point:
      1. No divergence at all -> pass.
      2. Divergence exists but logprobs are unavailable -> pass (can't
         disprove correctness without evidence).
      3. Divergence with logprobs -> ``check_logprob_sanity``.  If the
         baseline token is in the miner's top-5 with a small gap, the
         flip is explainable by TP noise -> pass.  Otherwise -> fail.
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
        if bt != mt:
            first_mm_idx = i
            first_mm_baseline = bt
            first_mm_miner = mt
            if miner_top_logprobs and i < len(miner_top_logprobs):
                first_mm_lp = miner_top_logprobs[i]
            break

    if first_mm_idx is None:
        return CorrectnessVerdict(
            passed=True,
            token_match_rate=rate,
        )

    if miner_top_logprobs is None:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=rate,
            reason=(
                f"first_mismatch_fail at index {first_mm_idx}: "
                f"logprobs missing from response"
            ),
            first_mismatch_index=first_mm_idx,
            baseline_token_at_mismatch=first_mm_baseline,
            miner_token_at_mismatch=first_mm_miner,
        )

    if first_mm_lp is None:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=rate,
            reason=(
                f"first_mismatch_fail at index {first_mm_idx}: "
                f"no logprobs available at divergence point"
            ),
            first_mismatch_index=first_mm_idx,
            baseline_token_at_mismatch=first_mm_baseline,
            miner_token_at_mismatch=first_mm_miner,
        )

    miner_top1 = first_mm_lp[0].get("token") if first_mm_lp else None
    if miner_top1 is not None and miner_top1 != first_mm_miner:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=rate,
            reason=(
                f"non_greedy at index {first_mm_idx}: miner chose "
                f"{first_mm_miner!r} but top-1 is {miner_top1!r}"
            ),
            first_mismatch_index=first_mm_idx,
            baseline_token_at_mismatch=first_mm_baseline,
            miner_token_at_mismatch=first_mm_miner,
            miner_logprobs_at_mismatch=first_mm_lp,
        )

    if check_logprob_sanity(first_mm_baseline or "", first_mm_lp):
        return CorrectnessVerdict(
            passed=True,
            token_match_rate=rate,
            first_mismatch_index=first_mm_idx,
            baseline_token_at_mismatch=first_mm_baseline,
            miner_token_at_mismatch=first_mm_miner,
            miner_logprobs_at_mismatch=first_mm_lp,
        )

    return CorrectnessVerdict(
        passed=False,
        token_match_rate=rate,
        reason=(
            f"first_mismatch_fail at index {first_mm_idx}: "
            f"baseline={first_mm_baseline!r} not in miner top-5 "
            f"or logprob gap > 0.5"
        ),
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

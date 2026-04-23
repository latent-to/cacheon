"""Phase 2 — Scoring.

Takes two RunResult objects (baseline + miner) and produces a ScoreResult
with KL divergence, memory reduction, latency improvement, and a final
weighted score gated by quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .harness import RunResult

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.1  # nats — hard reject above this
MEMORY_WEIGHT = 0.6
LATENCY_WEIGHT = 0.4


@dataclass
class ScoreResult:
    kl_divergence: float
    memory_reduction: float
    latency_improvement: float
    score: float
    disqualified: bool
    disqualify_reason: str | None


def _compute_kl(baseline: RunResult, miner: RunResult) -> float:
    """Average per-token KL(baseline || miner) in nats across all prompts.

    If the miner generated fewer tokens than baseline on a given prompt,
    we truncate to the shorter length and log a warning.
    """
    total_kl = 0.0
    total_tokens = 0

    for i, (bl, mn) in enumerate(zip(baseline.all_logits, miner.all_logits)):
        n_bl, n_mn = bl.shape[0], mn.shape[0]
        n = min(n_bl, n_mn)
        if n == 0:
            continue
        if n_bl != n_mn:
            logger.warning(
                "Prompt %d: token count mismatch (baseline=%d, miner=%d), "
                "truncating to %d",
                i, n_bl, n_mn, n,
            )
        bl_trunc = bl[:n].float()
        mn_trunc = mn[:n].float()

        # KL(P || Q) where P=baseline, Q=miner
        # F.kl_div expects (log_input, target) so log_input=log(Q), target=P
        log_q = F.log_softmax(mn_trunc, dim=-1)
        p = F.softmax(bl_trunc, dim=-1)
        kl = F.kl_div(log_q, p, reduction="sum", log_target=False)
        total_kl += kl.item()
        total_tokens += n

    if total_tokens == 0:
        return 0.0
    return total_kl / total_tokens


def _check_logits_valid(miner: RunResult) -> str | None:
    """Return a disqualify reason if miner logits contain NaN or Inf."""
    for i, logits in enumerate(miner.all_logits):
        if torch.isnan(logits).any():
            return f"NaN in miner logits for prompt {i}"
        if torch.isinf(logits).any():
            return f"Inf in miner logits for prompt {i}"
    return None


def score(baseline: RunResult, miner: RunResult) -> ScoreResult:
    """Score a miner's run against the baseline."""

    # NaN/Inf check before any computation
    invalid = _check_logits_valid(miner)
    if invalid is not None:
        return ScoreResult(
            kl_divergence=float("inf"),
            memory_reduction=0.0,
            latency_improvement=0.0,
            score=0.0,
            disqualified=True,
            disqualify_reason=invalid,
        )

    kl = _compute_kl(baseline, miner)

    # KV-cache memory reduction: positive means miner's cache is smaller.
    # Uses policy_memory_bytes (actual KV-cache footprint) rather than
    # peak VRAM which includes transient attention buffers and model weights.
    if baseline.policy_memory_bytes > 0:
        mem_reduction = (
            (baseline.policy_memory_bytes - miner.policy_memory_bytes)
            / baseline.policy_memory_bytes
        )
    else:
        mem_reduction = 0.0

    # Latency improvement: positive means miner was faster
    if baseline.latency_s > 0:
        lat_improvement = (
            (baseline.latency_s - miner.latency_s) / baseline.latency_s
        )
    else:
        lat_improvement = 0.0

    mem_reduction = max(-1.0, min(1.0, mem_reduction))
    lat_improvement = max(-1.0, min(1.0, lat_improvement))

    if kl > QUALITY_THRESHOLD:
        return ScoreResult(
            kl_divergence=kl,
            memory_reduction=mem_reduction,
            latency_improvement=lat_improvement,
            score=0.0,
            disqualified=True,
            disqualify_reason=f"KL divergence {kl:.4f} exceeds threshold {QUALITY_THRESHOLD}",
        )

    final_score = MEMORY_WEIGHT * mem_reduction + LATENCY_WEIGHT * lat_improvement

    return ScoreResult(
        kl_divergence=kl,
        memory_reduction=mem_reduction,
        latency_improvement=lat_improvement,
        score=final_score,
        disqualified=False,
        disqualify_reason=None,
    )

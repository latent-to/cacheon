"""KL between two runs' per-position top-k token distributions.

SGLang returns, per position, a list of ``(logprob, token_id, text|None)`` for
the top-k tokens (``output_top_logprobs`` / ``input_top_logprobs``). We turn each
into a distribution and compute KL(reference || candidate) per position, then
average.

Caveat, stated honestly: top-k truncation means each distribution only carries
the head mass, so this is an *approximation* of the true full-vocab KL. It is
sensitive enough to catch the cheats that matter (calibration collapse, biased
quant, dropped precision) when k is reasonably large (e.g. 20+), but it is not a
substitute for a full-vocab teacher-forced KL when you can afford the logits.
The production path should capture full logits at the reference seam; this MVP
uses top-k because that is what the stock Engine API exposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# A per-position top-k entry as returned by sglang: (logprob, token_id, text|None)
TopK = Sequence[tuple]


def _dist_from_topk(topk: TopK) -> dict[int, float]:
    d: dict[int, float] = {}
    for entry in topk:
        lp = float(entry[0])
        tid = int(entry[1])
        d[tid] = d.get(tid, 0.0) + math.exp(lp)
    return d


def kl_position(ref_topk: TopK, cand_topk: TopK, *, eps: float = 1e-8) -> float:
    """KL(ref || cand) over the union of the two top-k supports."""
    P = _dist_from_topk(ref_topk)
    Q = _dist_from_topk(cand_topk)
    support = set(P) | set(Q)
    if not support:
        return 0.0
    # Renormalize each over the shared support with a floor for missing mass.
    pz = sum(P.get(t, 0.0) + eps for t in support)
    qz = sum(Q.get(t, 0.0) + eps for t in support)
    kl = 0.0
    for t in support:
        p = (P.get(t, 0.0) + eps) / pz
        q = (Q.get(t, 0.0) + eps) / qz
        kl += p * math.log(p / q)
    return max(0.0, kl)


@dataclass
class KLReport:
    num_positions: int
    mean_kl: float
    max_kl: float
    p99_kl: float
    # number of positions where the argmax token differs between ref and cand
    argmax_disagreements: int


def _argmax(topk: TopK) -> Optional[int]:
    best_lp = -math.inf
    best_tid: Optional[int] = None
    for entry in topk:
        lp = float(entry[0])
        if lp > best_lp:
            best_lp = lp
            best_tid = int(entry[1])
    return best_tid


def kl_over_positions(
    ref: Sequence[TopK], cand: Sequence[TopK], *, eps: float = 1e-8
) -> KLReport:
    """Aggregate KL across aligned positions.

    ``ref`` and ``cand`` are per-position top-k lists; they must already be
    aligned (same positions). Positions beyond the shorter list are ignored and
    counted by the caller as divergence.
    """
    n = min(len(ref), len(cand))
    kls: list[float] = []
    disagree = 0
    for i in range(n):
        kls.append(kl_position(ref[i], cand[i], eps=eps))
        if _argmax(ref[i]) != _argmax(cand[i]):
            disagree += 1
    if not kls:
        return KLReport(0, 0.0, 0.0, 0.0, 0)
    kls_sorted = sorted(kls)
    p99 = kls_sorted[min(len(kls_sorted) - 1, int(0.99 * len(kls_sorted)))]
    return KLReport(
        num_positions=n,
        mean_kl=sum(kls) / len(kls),
        max_kl=max(kls),
        p99_kl=p99,
        argmax_disagreements=disagree,
    )


# A single prompt's run, as KL consumes it: (generated token ids, per-position top-k).
PromptRun = tuple[Sequence[int], Sequence[TopK]]


def extract_per_prompt(outputs: Sequence[dict]) -> list[tuple[list[int], list]]:
    """Pull ``(output_ids, per-position top-k)`` out of sglang's generate() outputs.

    Shared by the throughput+KL eval and the benchmark eval so both build the exact
    same structure for ``aligned_kl``.
    """
    per_prompt: list[tuple[list[int], list]] = []
    for o in outputs:
        meta = o.get("meta_info", {})
        output_ids = o.get("output_ids") or meta.get("output_ids") or []
        topk = meta.get("output_top_logprobs") or []
        per_prompt.append(([int(t) for t in output_ids], topk))
    return per_prompt


def aligned_kl(
    baseline: Sequence[PromptRun], candidate: Sequence[PromptRun], *, eps: float = 1e-8
) -> KLReport:
    """KL between two runs, aligned per prompt up to the first token divergence.

    Greedy decoding means the candidate can diverge from the baseline mid-sequence;
    once the generated token at position ``i`` differs, the two runs no longer share
    a context and later positions aren't comparable. So we compare position ``i``
    and then stop at the first mismatch. Position 0 always shares the prompt, so a
    kernel that derails the very first token still gets scored (a large KL) instead
    of silently contributing zero comparable positions.
    """
    ref_positions: list = []
    cand_positions: list = []
    for (b_ids, b_topk), (c_ids, c_topk) in zip(baseline, candidate):
        n = min(len(b_topk), len(c_topk))
        for i in range(n):
            ref_positions.append(b_topk[i])
            cand_positions.append(c_topk[i])
            if i < len(b_ids) and i < len(c_ids) and b_ids[i] != c_ids[i]:
                break
    return kl_over_positions(ref_positions, cand_positions, eps=eps)

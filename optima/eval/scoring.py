"""Noise-robust speedup scoring for a validator that CANNOT lock GPU clocks.

The validator runs on rented pods where ``nvidia-smi -lgc`` is denied and the
profiling counters are blocked, so per-launch throughput carries a large
warmup/thermal/boost component (measured at **±7-17%**, worst-case ~±32% cold).
Crucially that component is a *between-launch systematic offset*, so median-of-K
*within* a single launch cannot remove it: if the baseline launch runs cold and
the candidate launch runs warm, the candidate "wins" on clock state alone. The
old gate compared one cold baseline to one warm candidate against a hand-picked
2% margin that sits an order of magnitude *below* that noise — the exact source
of the project's repeated phantom wins.

This module turns raw per-launch tok/s into a trustworthy verdict using only what
is available without privileged clock control:

* **Bookended / interleaved A/B.** The baseline is measured both *before* and
  *after* the candidate (``B, C, B'``), so the candidate is bracketed and the two
  baseline reads bound the drift that occurred across it. More rounds tighten it.
* **Paired speedup.** Speedup is ``candidate / mean(bracketing baselines)``, so a
  monotonic ramp across the run partly cancels rather than biasing one side.
* **Noise-derived margin.** The bar a speedup must clear is
  ``1 + max(min_margin, k * noise)``, where ``noise`` is the *measured* relative
  spread of the repeated baseline reads — not a constant that ignores the box.
* **Drift rejection (no-decision).** If the bracketing baselines disagree by more
  than ``max_noise``, the box was too unstable this round to trust: the verdict is
  ``confident=False`` and it must NOT crown. A real subnet re-queues such a round;
  it never pays emissions on a measurement it cannot reproduce.

Every function here is pure and unit-tested on CPU with synthetic samples — the
statistics are hardware-independent, so they are validated here and run unchanged
on the pod. This is the piece that makes a sub-10% real win resolvable on a noisy
box, which is the regime every win on a mature model lives in.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


EXTERNAL_QUALITY_GATE_V1 = "controller-one-shot-paired-topk-v1"


@dataclass(frozen=True)
class SpeedupVerdict:
    """The outcome of comparing a candidate launch to bracketing baseline launches."""

    speedup: float  # robust paired estimate: candidate / mean(baseline reads)
    noise: float  # measured relative spread of the baseline reads (the floor)
    required: float  # the bar it had to clear: 1 + max(min_margin, k*noise)
    passed_speedup: bool  # cleared `required` AND the round was trustworthy
    confident: bool  # False -> box too noisy this round; treat as NO-DECISION, never crown
    n_baselines: int
    detail: str = ""


@dataclass(frozen=True)
class ExternalFidelityMetrics:
    """Raw controller-observed candidate/control fidelity measurements.

    These are bounded and serialized by ``QualificationReport``.  They contain no
    pass/fail claim: the consumer replays the arena-registered external gate over
    them.  ``argmax_disagreements`` is stored as a count so its rate cannot be rounded
    into or out of a pass in JSON.
    """

    num_positions: int
    mean_kl: float
    max_kl: float
    p99_kl: float
    argmax_disagreements: int
    mean_coverage_dev: float
    dropped_positions: int

    @property
    def argmax_disagree_rate(self) -> float:
        if self.num_positions <= 0:
            return 0.0
        return self.argmax_disagreements / self.num_positions


@dataclass(frozen=True)
class ExternalQualityBatch:
    """One controller-observed prompt batch, kept separate across phases.

    Candidate-vs-baseline and stock-bookend-vs-baseline measurements are paired
    at the same batch index.  Keeping this unit explicit prevents a correct
    warmup (or another timed batch) from diluting a bad timed batch.
    """

    candidate: ExternalFidelityMetrics
    stock_control: ExternalFidelityMetrics
    token_matches: int
    token_total: int
    stock_token_matches: int
    stock_token_total: int


@dataclass(frozen=True)
class ExternalQualityVerdict:
    """Recomputed result of candidate-vs-stock-control fidelity policy."""

    passed: bool
    kl_limit: float
    argmax_limit: float
    coverage_limit: float
    dropped_limit: int
    minimum_positions: int
    checks: tuple[tuple[str, bool], ...]
    detail: str


def score_output_token_match(
    matched: int,
    total: int,
    *,
    threshold: float,
    stock_matched: int | None = None,
    stock_total: int | None = None,
) -> tuple[bool, str]:
    """Model-consumed output-token gate calibrated by the paired stock control.

    ``threshold`` remains the arena's declared floor.  When a B-vs-B' control is
    supplied, the effective floor is no stricter than the match rate stock
    achieved in this exact bracket.  This introduces no new numerical tolerance:
    it uses only the registered floor and measured stock behavior.
    """
    if (
        type(matched) is not int
        or type(total) is not int
        or matched < 0
        or total <= 0
        or matched > total
        or type(threshold) not in (int, float)
        or not math.isfinite(float(threshold))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("invalid output token-match evidence/policy")
    if (stock_matched is None) != (stock_total is None):
        raise ValueError("stock token-match evidence must be supplied as a pair")
    control_rate: float | None = None
    if stock_matched is not None and stock_total is not None:
        if (
            type(stock_matched) is not int
            or type(stock_total) is not int
            or stock_matched < 0
            or stock_total <= 0
            or stock_matched > stock_total
        ):
            raise ValueError("invalid stock token-match evidence")
        control_rate = stock_matched / stock_total
    rate = matched / total
    effective_threshold = (
        float(threshold)
        if control_rate is None
        else min(float(threshold), control_rate)
    )
    passed = rate >= effective_threshold
    control = (
        ""
        if control_rate is None
        else f" stock={stock_matched}/{stock_total} rate={control_rate:.6f}"
    )
    return passed, (
        f"; output_token_match={matched}/{total} rate={rate:.6f} "
        f"policy_floor={float(threshold):.6f}{control} "
        f"limit={effective_threshold:.6f} pass={int(passed)}"
    )


def score_external_quality_batches(
    batches: tuple[ExternalQualityBatch, ...],
    *,
    gate: str,
    phase: str,
) -> tuple[bool, str]:
    """Grade every batch independently under the external top-k policy."""

    if not batches or phase not in {"warmup", "timed"}:
        raise ValueError("external quality requires a non-empty named phase")
    verdicts = tuple(
        score_external_quality(batch.candidate, batch.stock_control, gate=gate)
        for batch in batches
    )
    detail = f"{phase}_topk[" + " | ".join(
        f"batch{index}:{verdict.detail}"
        for index, verdict in enumerate(verdicts, start=1)
    ) + "]"
    return all(verdict.passed for verdict in verdicts), detail


def score_output_token_match_batches(
    batches: tuple[ExternalQualityBatch, ...],
    *,
    threshold: float,
    phase: str,
) -> tuple[bool, str]:
    """Grade every output-token batch against its paired stock batch."""

    if not batches or phase not in {"warmup", "timed"}:
        raise ValueError("output token quality requires a non-empty named phase")
    verdicts = tuple(
        score_output_token_match(
            batch.token_matches,
            batch.token_total,
            threshold=threshold,
            stock_matched=batch.stock_token_matches,
            stock_total=batch.stock_token_total,
        )
        for batch in batches
    )
    detail = f"; {phase}_output_tokens[" + " | ".join(
        f"batch{index}{description}"
        for index, (_passed, description) in enumerate(verdicts, start=1)
    ) + "]"
    return all(passed for passed, _description in verdicts), detail


# Compatibility spellings for callers written before model-consumed token evidence
# became load-bearing for every product class. Their behavior is intentionally the
# generic output-token policy above.
score_system_token_match = score_output_token_match
score_system_token_match_batches = score_output_token_match_batches


def score_external_quality(
    candidate: ExternalFidelityMetrics,
    stock_control: ExternalFidelityMetrics,
    *,
    gate: str,
) -> ExternalQualityVerdict:
    """Apply the versioned external fidelity envelope to raw measurements.

    ``candidate`` is B-vs-C and ``stock_control`` is B-vs-B'.  The gate name is
    fingerprinted by ``ArenaProfile``; changing any envelope constant requires a new
    name and therefore a new arena bracket.  Scheduler audit receipts are deliberately
    absent here: process-local evidence is diagnostic and cannot mint quality.
    """
    if gate != EXTERNAL_QUALITY_GATE_V1:
        raise ValueError(f"unsupported external quality gate {gate!r}")

    # These are the existing V1 consensus constants, retained verbatim. They are
    # not newly justified by the system-product token work above: changing them
    # requires GPU stock-control calibration and a newly versioned arena/gate.
    kl_limit = max(0.05, stock_control.mean_kl * 1.50 + 0.05)
    argmax_limit = min(1.0, stock_control.argmax_disagree_rate + 0.05)
    coverage_limit = min(1.0, stock_control.mean_coverage_dev + 0.05)
    dropped_limit = stock_control.dropped_positions + max(
        2, int(0.01 * stock_control.num_positions)
    )
    minimum_positions = max(1, int(0.90 * stock_control.num_positions))
    checks = (
        ("candidate_positions", candidate.num_positions > 0),
        ("control_positions", stock_control.num_positions > 0),
        ("mean_kl", candidate.mean_kl <= kl_limit),
        ("argmax", candidate.argmax_disagree_rate <= argmax_limit),
        ("coverage", candidate.mean_coverage_dev <= coverage_limit),
        ("dropped", candidate.dropped_positions <= dropped_limit),
        ("positions", candidate.num_positions >= minimum_positions),
    )
    passed = all(value for _, value in checks)
    check_text = ",".join(f"{name}={int(value)}" for name, value in checks)
    detail = (
        f"controller quality cand/control: mean_kl={candidate.mean_kl:.4g}/"
        f"{stock_control.mean_kl:.4g} limit={kl_limit:.4g}; argmax="
        f"{candidate.argmax_disagree_rate:.3f}/"
        f"{stock_control.argmax_disagree_rate:.3f} limit={argmax_limit:.3f}; "
        f"coverage={candidate.mean_coverage_dev:.3f}/"
        f"{stock_control.mean_coverage_dev:.3f} limit={coverage_limit:.3f}; "
        f"positions={candidate.num_positions}/{stock_control.num_positions}; "
        f"dropped={candidate.dropped_positions}/{stock_control.dropped_positions}; "
        f"checks={check_text}"
    )
    return ExternalQualityVerdict(
        passed=passed,
        kl_limit=kl_limit,
        argmax_limit=argmax_limit,
        coverage_limit=coverage_limit,
        dropped_limit=dropped_limit,
        minimum_positions=minimum_positions,
        checks=checks,
        detail=detail,
    )


def relative_spread(samples: list[float]) -> float:
    """A scale-free measure of run-to-run noise across point estimates.

    For >=3 reads use population stdev / mean (smooth, uses all points). For exactly
    2 reads (the default bookend ``B, B'``) stdev underestimates, so use the range /
    mean — the honest worst-case gap between the two bracketing baselines. Returns
    ``inf`` for <2 reads (noise unmeasurable -> the caller must not claim confidence).
    """
    vals = [s for s in samples if s > 0]
    if len(vals) < 2:
        return float("inf")
    mean = statistics.fmean(vals)
    if mean <= 0:
        return float("inf")
    if len(vals) == 2:
        return (max(vals) - min(vals)) / mean
    return statistics.pstdev(vals) / mean


def score_speedup(
    baseline_reads: list[float],
    candidate_read: float,
    *,
    min_margin: float = 0.005,
    k: float = 2.0,
    max_noise: float = 0.10,
) -> SpeedupVerdict:
    """Decide whether ``candidate_read`` is a *real* speedup over the bracketing baselines.

    ``baseline_reads`` are the tok/s of the bookending baseline launches (>=2 for a
    real verdict: e.g. the ``B`` and ``B'`` around the candidate ``C``). The speedup
    is paired against their mean; the required margin scales with the measured
    baseline noise; and a round whose baselines disagree by more than ``max_noise``
    is flagged ``confident=False`` (no-decision) so noise can never mint a champion.

    * ``min_margin`` — floor on the required improvement even on a perfectly stable box.
      Default 0.5% (2026-07-07, Shiv's call): real kernel wins arrive as STACKS of
      1-2% improvements, so a 2% floor rejects most of the genuine win distribution
      — a campaign-validated +2.2% win measured at exactly the old floor's edge. The
      ``k*noise`` term (not this constant) is what protects against unstable boxes:
      on a drifting box the noise term dominates the bar, and on a quiet/clock-locked
      box (bracket spread ~0.013% measured 2026-06-15) sub-1% wins ARE resolvable.
      Per-arena override belongs in the arena registry when it lands.
    * ``k`` — how many noise-widths above 1.0 the speedup must sit (2.0 ~= a 2-sigma-ish bar).
    * ``max_noise`` — relative baseline spread above which the round is untrustworthy.
    """
    reads = [b for b in baseline_reads if b > 0]
    if not reads or candidate_read <= 0:
        return SpeedupVerdict(0.0, float("inf"), 1.0 + min_margin, False, False,
                              len(reads), "missing/zero throughput sample")
    base = statistics.fmean(reads)
    noise = relative_spread(reads)
    speedup = candidate_read / base
    required = 1.0 + max(min_margin, k * (noise if noise != float("inf") else 0.0))
    confident = len(reads) >= 2 and noise <= max_noise
    passed = confident and speedup >= required
    if not confident:
        if len(reads) < 2:
            detail = "single baseline read -> noise unmeasured; cannot crown (bookend the baseline)"
        else:
            detail = f"baseline drift {noise:.1%} > max_noise {max_noise:.0%}; NO-DECISION (re-queue)"
    elif passed:
        detail = f"speedup {speedup:.3f} >= required {required:.3f} (noise {noise:.1%})"
    else:
        detail = f"speedup {speedup:.3f} < required {required:.3f} (noise {noise:.1%})"
    return SpeedupVerdict(
        speedup=speedup, noise=noise, required=required,
        passed_speedup=passed, confident=confident, n_baselines=len(reads), detail=detail,
    )

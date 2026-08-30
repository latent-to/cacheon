"""Recompute calibrated B/C/B-prime speed evidence from sealed lifecycle rows."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

@dataclass(frozen=True)
class SpeedupVerdict:
    speedup: float  # robust paired estimate: mean(candidate reads) / mean(baseline reads)
    noise: float  # measured relative spread floor: baselines, and candidates when >= 2 reads
    required: float  # the bar it had to clear: 1 + max(min_margin, k*noise)
    passed_speedup: bool  # cleared `required` AND the round was trustworthy
    confident: bool  # False -> box too noisy this round; treat as NO-DECISION, never crown
    n_baselines: int
    detail: str = ""
    n_candidates: int = 1

class RawSpeedEvidenceError(ValueError):
    pass

@dataclass(frozen=True)
class ChargedExecutionRate:
    launch_digest: str
    session_id: str
    conditioning_tokens: int
    timed_tokens: int
    charged_tokens: int
    conditioning_seconds: float
    timed_seconds: float
    charged_seconds: float
    tokens_per_second: float

def _finite_time(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RawSpeedEvidenceError(f"{field} must be finite")
    return float(value)


def _positive_number(value: object, *, field: str) -> float:
    result = _finite_time(value, field=field)
    if result <= 0:
        raise RawSpeedEvidenceError(f"{field} must be positive")
    return result

def marginal_workload_digest(plan: object) -> str:
    from cacheon.eval.oci_outer_session import SessionExecutionPlan
    from cacheon.stack_identity import canonical_digest
    if type(plan) is not SessionExecutionPlan:
        raise RawSpeedEvidenceError("workload plan must be exact typed evidence")
    payload = {
        "conditioning_count": plan.conditioning_count,
        "engine_config_digest": plan.expected_engine_config_digest,
        "expected_prompt_tokens": plan.expected_prompt_tokens,
        "max_new_tokens": plan.max_new_tokens,
        "prompt_batches": plan.prompt_batches,
        "temperature": format(plan.temperature, ".17g"),
        "top_logprobs_num": plan.top_logprobs_num,
        "warmup_count": plan.warmup_count,
    }
    if not plan.batch_max_new_tokens and plan.quality_max_new_tokens is None:
        # Retained one-shape evidence keeps its exact v2 identity.
        return canonical_digest(
            "cacheon.qualification.marginal-workload.v2", payload
        )
    return canonical_digest(
        "cacheon.qualification.marginal-workload.v3",
        {
            **payload,
            "batch_request_geometry": [
                [tokens, prompt_tokens]
                for tokens, prompt_tokens in zip(
                    plan.batch_max_new_tokens,
                    plan.batch_expected_prompt_tokens,
                    strict=True,
                )
            ],
            "quality_max_new_tokens": plan.quality_tokens_per_prompt,
        },
    )


def _projection_digest(selected: str, candidate: str, calibration: str, context: str,
                       workload: str, runtime_policy: str, rates: tuple[ChargedExecutionRate, ...]) -> str:
    from cacheon.stack_identity import canonical_digest
    def row(rate: ChargedExecutionRate) -> list[object]:
        return [
            rate.launch_digest,
            rate.session_id,
            rate.conditioning_tokens,
            rate.timed_tokens,
            rate.charged_tokens,
            *(format(value, ".17g") for value in (
                rate.conditioning_seconds, rate.timed_seconds, rate.charged_seconds
            )),
        ]
    return canonical_digest(
        "cacheon.qualification.marginal-speed-evidence.v1",
        {
            "selected_delta_digest": selected,
            "candidate_launch_digest": candidate,
            "calibration_digest": calibration,
            "calibration_context_digest": context,
            "workload_digest": workload,
            "runtime_resource_policy_digest": runtime_policy,
            "rates": [row(rate) for rate in rates],
        },
    )


def relative_spread(samples: list[float]) -> float:
    vals = [_positive_number(sample, field="baseline throughput") for sample in samples]
    if len(vals) < 2:
        return float("inf")
    mean = statistics.fmean(vals)
    if not math.isfinite(mean) or mean <= 0:
        raise RawSpeedEvidenceError("baseline mean is not finite and positive")
    if len(vals) == 2:
        return (max(vals) - min(vals)) / mean
    return statistics.pstdev(vals) / mean


def score_speedup(
    baseline_reads: list[float],
    candidate_read: float | list[float],
    *,
    min_margin: float = 0.005,
    k: float = 2.0,
    max_noise: float = 0.10,
) -> SpeedupVerdict:
    margin = _finite_time(min_margin, field="min_margin")
    multiplier = _finite_time(k, field="noise multiplier")
    noise_ceiling = _finite_time(max_noise, field="max_noise")
    if not 0 < margin < 1 or multiplier <= 0 or not 0 <= noise_ceiling < 1:
        raise RawSpeedEvidenceError("speed policy is outside its allowed range")
    reads = [
        _positive_number(sample, field="baseline throughput")
        for sample in baseline_reads
    ]
    raw_candidates = (
        list(candidate_read) if isinstance(candidate_read, (list, tuple)) else [candidate_read]
    )
    candidate_reads = [
        _positive_number(sample, field="candidate throughput")
        for sample in raw_candidates
    ]
    if not reads or not candidate_reads:
        return SpeedupVerdict(0.0, float("inf"), 1.0 + min_margin, False, False,
                              len(reads), "missing/zero throughput sample",
                              n_candidates=len(candidate_reads))
    base = statistics.fmean(reads)
    candidate = statistics.fmean(candidate_reads)
    baseline_noise = relative_spread(reads)
    # A single candidate read (the historical B/C/B' shape) leaves the candidate's own
    # spread unmeasured and keeps the verdict identical to the legacy behavior. With
    # repeated candidate reads (B C B' C' B''), the candidate draw is measured too and
    # a noisy candidate is as disqualifying as a noisy baseline: 2026-07-16 forensics
    # measured 7.2% spread between two honest candidate legs, so a single-C verdict at
    # small margins crowns or kills on a per-boot draw.
    candidate_noise = relative_spread(candidate_reads) if len(candidate_reads) >= 2 else 0.0
    noise = max(baseline_noise, candidate_noise) if math.isfinite(baseline_noise) else baseline_noise
    speedup = candidate / base
    required = 1.0 + max(margin, multiplier * (noise if math.isfinite(noise) else 0.0))
    if not math.isfinite(speedup) or not math.isfinite(required):
        raise RawSpeedEvidenceError("derived speed verdict is non-finite")
    confident = len(reads) >= 2 and noise <= noise_ceiling
    passed = confident and speedup >= required
    spread_note = (
        f"noise {noise:.1%}"
        if len(candidate_reads) < 2
        else f"noise {noise:.1%} (baseline {baseline_noise:.1%}, candidate {candidate_noise:.1%})"
    )
    if not confident:
        if len(reads) < 2:
            detail = "single baseline read -> noise unmeasured; cannot crown (bookend the baseline)"
        elif candidate_noise > noise_ceiling >= baseline_noise:
            detail = f"candidate drift {candidate_noise:.1%} > max_noise {noise_ceiling:.0%}; NO-DECISION (re-queue)"
        else:
            detail = f"baseline drift {baseline_noise:.1%} > max_noise {noise_ceiling:.0%}; NO-DECISION (re-queue)"
    elif passed:
        detail = f"speedup {speedup:.3f} >= required {required:.3f} ({spread_note})"
    else:
        detail = f"speedup {speedup:.3f} < required {required:.3f} ({spread_note})"
    return SpeedupVerdict(
        speedup=speedup, noise=noise, required=required,
        passed_speedup=passed, confident=confident, n_baselines=len(reads), detail=detail,
        n_candidates=len(candidate_reads),
    )

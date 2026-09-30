"""The speed verdict record and the replay workload identity it is graded against."""
from __future__ import annotations

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


class CrossoverRuntimeError(RuntimeError):
    """Resident execution or retained measurement violates its sealed plan."""


# Existing continuation records bind this public type name.
CrossoverRuntimeError.__module__ = "cacheon.eval.crossover_runtime"


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
    if plan.measure_phase_latency:
        payload["measure_phase_latency"] = True
    if plan.batch_max_new_tokens:
        payload["batch_request_geometry"] = [
            [tokens, prompt_tokens]
            for tokens, prompt_tokens in zip(
                plan.batch_max_new_tokens, plan.batch_expected_prompt_tokens, strict=True,
            )
        ]
    if plan.replay is None:
        # The batch-cell workloads (v2/v3 digests) left with speed policies 8-15.
        raise RawSpeedEvidenceError("speed workloads are sealed session replays")
    return canonical_digest("cacheon.qualification.agent-workload.v1", {
        **payload, "replay": plan.replay.workload_identity(),
    })


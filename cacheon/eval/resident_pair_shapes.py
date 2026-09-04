"""Shape and bound helpers shared by the resident-pair factory surfaces.

Pure comparisons and validations with no lifecycle state: workload/resident
shape tuples used to decide whether two lane plans describe one reusable
pair, and the small numeric bounds the factory applies to timeouts and
deadlines.  The caller supplies its own error type so these helpers raise
inside the caller's failure domain.
"""

from __future__ import annotations

import math
from typing import Callable

from cacheon.eval.b300_qualification_lanes import B300QualificationLanePolicy
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_resident_session import ResidentSessionPlan


def positive_seconds(
    value: object, field: str, *, error: type[Exception]
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 86_400
    ):
        raise error(f"{field} is invalid")
    return float(value)


def absolute_deadline(
    value: object, clock: Callable[[], float], *, error: type[Exception]
) -> float:
    try:
        now = float(clock())
    except BaseException as exc:
        raise error(f"resident pair host clock failed: {exc}") from None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not math.isfinite(now)
        or float(value) <= now
    ):
        raise error("resident pair requires a future absolute deadline")
    return float(value)


def physical_ids(policy: B300QualificationLanePolicy) -> tuple[str, ...]:
    return tuple(str(value) for value in policy.physical_gpu_ids)


def workload_shape(plan: SessionExecutionPlan) -> tuple[object, ...]:
    return (
        plan.engine_config,
        plan.prompt_batches,
        plan.warmup_count,
        plan.conditioning_count,
        plan.max_new_tokens,
        plan.top_logprobs_num,
        plan.temperature,
        plan.batch_max_new_tokens,
        plan.batch_expected_prompt_tokens,
        plan.quality_max_new_tokens,
        plan.expected_discovery_overlay_identity_digest,
        plan.audit_policy,
    )


def resident_shape(plan: ResidentSessionPlan) -> tuple[object, ...]:
    return (
        plan.expected_engine_config_digest,
        plan.engine_config,
        plan.max_swaps,
        plan.max_batches,
        plan.max_new_tokens,
        plan.top_logprobs_num,
        plan.temperature,
    )

"""The precommitted resident read schedule: expansion, one read, sync, grade.

Version 8 reads B, C and B-prime and grades them once. Version 12 keeps that
decode schedule byte for byte and appends a prefill-only pass of the same
sealed batches on each arm (B_prefill, C_prefill, B_prime_prefill): every
request in the pass generates one token, so the read measures prompt
processing with no decode work to dilute it. The decode verdict still decides
exactly as before. The prefill lane can only admit a candidate that the decode
floor neither admitted nor convicted, and it is credited at the sealed weight
rather than at its raw prefill speedup, because a prefill gain is not a
one-to-one throughput gain for the serving customer (owner ruling 2026-09-08:
one end-to-end score, decode bundles untouched, prefill rewarded on its own
terms).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from cacheon.eval.oci_outer_session import (
    BatchExecutionEvidence,
    OpenedOuterSession,
    SessionExecutionPlan,
)
from cacheon.eval.resident_measurement import (
    CrossoverRuntimeError,
    ResidentReadRate,
    _timed_windows,
)
from cacheon.eval.scoring import SpeedupVerdict
from cacheon.eval.speed_verdict import (
    SpeedStageDecision,
    fail_reason,
    resident_speed_roles,
    speed_grade,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cacheon.eval.crossover_runtime import ResidentSpeedPolicy

# One generated token per request: the prefill of the whole prompt plus a
# single sampling step, which is the phase-pure prefill measurement every
# serving benchmark uses (input length N, output length 1).
PREFILL_READ_BUDGET = 1


def expanded_schedule(
    plan: SessionExecutionPlan, reads: int, *, prefill_reads: int = 0
) -> SessionExecutionPlan:
    """Repeat the complete read ``reads`` times, then ``prefill_reads`` more
    times with every request budgeted to one token.

    Each repeat includes its validator-owned warmup: the model stays loaded and
    only the cheap workload conditioning repeats between arms. A prefill read
    keeps the exact prompts and prompt geometry of the decode read, so the
    sealed prompt identity is shared and only the output budget differs."""

    count = len(plan.prompt_batches)
    if prefill_reads <= 0:
        return replace(
            plan,
            prompt_batches=plan.prompt_batches * reads,
            batch_max_new_tokens=plan.batch_max_new_tokens * reads,
            batch_expected_prompt_tokens=plan.batch_expected_prompt_tokens * reads,
        )
    budgets = plan.batch_max_new_tokens or (plan.max_new_tokens,) * count
    prompts = plan.batch_expected_prompt_tokens or (
        (plan.expected_prompt_tokens,) * count
    )
    return replace(
        plan,
        prompt_batches=plan.prompt_batches * (reads + prefill_reads),
        batch_max_new_tokens=budgets * reads
        + (PREFILL_READ_BUDGET,) * (count * prefill_reads),
        batch_expected_prompt_tokens=prompts * (reads + prefill_reads),
    )


def read_rate(
    role: str,
    lane_digest: str,
    controller: OpenedOuterSession,
    template: SessionExecutionPlan,
    *,
    with_windows: bool = False,
) -> ResidentReadRate:
    """Take one complete read from the resident controller and rate it."""

    first = controller.next_batch_index
    rows = tuple(
        controller.execute_next() for _ in range(len(template.prompt_batches))
    )
    timed = rows[template.warmup_count :]
    conditioning_start = template.warmup_count - template.conditioning_count
    conditioning = rows[conditioning_start : template.warmup_count]
    if (
        not rows
        or any(type(row) is not BatchExecutionEvidence or row.audit_receipts for row in rows)
        or tuple(row.batch_index for row in rows)
        != tuple(range(rows[0].batch_index, rows[-1].batch_index + 1))
        or tuple(
            controller.plan.prompt_batches[row.batch_index] for row in rows
        )
        != template.prompt_batches
        or not timed
        or not conditioning
    ):
        raise CrossoverRuntimeError("resident read batches are incomplete")
    conditioning_seconds = (
        timed[0].request_started_at - conditioning[0].request_started_at
    )
    timed_seconds = (
        timed[-1].response_completed_at - timed[0].request_started_at
    )
    conditioning_tokens = sum(row.token_numerator for row in conditioning)
    timed_tokens = sum(row.token_numerator for row in timed)
    charged_seconds = conditioning_seconds + timed_seconds
    charged_tokens = conditioning_tokens + timed_tokens
    return ResidentReadRate(
        role,
        lane_digest,
        controller.plan.launch_digest,
        controller.session_id,
        first,
        first + len(rows) - 1,
        timed[0].batch_index,
        timed[-1].batch_index,
        conditioning_tokens,
        timed_tokens,
        charged_tokens,
        float(conditioning_seconds),
        float(timed_seconds),
        float(charged_seconds),
        float(charged_tokens / charged_seconds),
        _timed_windows(timed) if with_windows else (),
    )


class ReadSchedule:
    """Cross-lane hand-off of the serialized reads, keyed by role."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.values: dict[str, object] = {}
        self.failure: BaseException | None = None

    def put(self, key: str, value: object = True) -> None:
        with self.condition:
            if key in self.values:
                raise CrossoverRuntimeError(f"resident schedule repeated {key}")
            self.values[key] = value
            self.condition.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self.condition:
            if self.failure is None:
                self.failure = exc
            self.condition.notify_all()

    def get(
        self, key: str, *, deadline: float, clock: Callable[[], float]
    ) -> object:
        with self.condition:
            while key not in self.values:
                if self.failure is not None:
                    raise CrossoverRuntimeError(
                        f"resident peer failed: {self.failure}"
                    ) from self.failure
                remaining = deadline - float(clock())
                if not math.isfinite(remaining) or remaining <= 0:
                    raise CrossoverRuntimeError(
                        "resident speed stage exceeded its deadline"
                    )
                self.condition.wait(timeout=min(0.1, remaining))
            return self.values[key]


@dataclass(frozen=True)
class ScheduleGrade:
    """One grade over the whole precommitted schedule.

    ``verdict`` is the decode verdict and is what the witness retains as its
    headline under every version; ``settled_speedup`` is the exact decimal text
    settlement consumes, which equals the decode speedup unless the prefill
    lane admitted the candidate."""

    verdict: SpeedupVerdict
    decision: SpeedStageDecision
    conditioning_failed: bool
    prefill_verdict: SpeedupVerdict | None
    lane: str | None
    settled_speedup: str


def prefill_policy(policy: "ResidentSpeedPolicy") -> "ResidentSpeedPolicy":
    """The decode policy graded at the sealed prefill margin."""

    return replace(policy, min_margin=policy.prefill_min_margin)


def credited_speedup(policy: "ResidentSpeedPolicy", prefill: SpeedupVerdict) -> float:
    """A prefill-lane admission settles at the sealed fraction of its prefill gain."""

    return 1.0 + policy.prefill_credit_weight * (prefill.speedup - 1.0)


def grade_schedule(
    policy: "ResidentSpeedPolicy", rates: tuple[ResidentReadRate, ...]
) -> ScheduleGrade:
    """Grade one read set, shared by the live stage, the evidence regrade and
    the witness so the three cannot drift apart.

    The decode reads are graded first and alone, exactly as version 8 grades
    them. Under version 12 the prefill reads are graded at the prefill margin,
    and that verdict admits only when the decode grade was a competitive miss
    or a boundary-crossing uncertainty on a valid measurement: a measured
    slowdown or a conditioning regression is a regression whatever the prompt
    pass says, and an invalid decode measurement stays NO_DECISION."""

    roles = resident_speed_roles(policy.version, len(rates))
    if policy.version < 8 or roles is None or tuple(row.role for row in rates) != roles:
        raise CrossoverRuntimeError("resident speed read set is not the precommitted schedule")
    before, candidate, bookend = rates[:3]
    verdict, decision = speed_grade(
        policy, [before, bookend], [candidate], concluding=True
    )
    conditioning = policy.conditioning_regression(before, candidate)
    if conditioning and decision is not SpeedStageDecision.NO_DECISION:
        decision = SpeedStageDecision.FAIL
    lane = "decode" if decision is SpeedStageDecision.PASS else None
    settled = verdict.speedup
    prefill = None
    if policy.version >= 12:
        prefill, admitted = speed_grade(
            prefill_policy(policy), [rates[3], rates[5]], [rates[4]], concluding=True
        )
        if (
            lane is None
            and admitted is SpeedStageDecision.PASS
            and verdict.confident
            and not conditioning
            and fail_reason(verdict) != "candidate_slower"
        ):
            decision, lane = SpeedStageDecision.PASS, "prefill"
            settled = credited_speedup(policy, prefill)
    return ScheduleGrade(
        verdict, decision, conditioning, prefill, lane, format(settled, ".17g")
    )


__all__ = [
    "PREFILL_READ_BUDGET",
    "ReadSchedule",
    "ScheduleGrade",
    "credited_speedup",
    "expanded_schedule",
    "grade_schedule",
    "prefill_policy",
    "read_rate",
]

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

Versions 13-15 allow one complete repeat of a valid borderline round, with both
rounds graded together inside the original deadline and resident sessions.
"""

from __future__ import annotations

import concurrent.futures
import math
import time
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from cacheon.eval.oci_outer_session import (
    BatchExecutionEvidence,
    OpenedOuterSession,
    SessionExecutionPlan,
    SessionExecutionEvidence,
)
from cacheon.eval.resident_measurement import (
    CrossoverRuntimeError,
    ResidentReadRate,
    _timed_windows,
)
from cacheon.eval.scoring import SpeedupVerdict, marginal_workload_digest
from cacheon.eval.oci_backend import EngineExecutionEvidence, OCIEngineExecutor, TrustedArenaModelMountReceipt
from cacheon.eval.speed_verdict import (
    SpeedStageDecision,
    combined_speed_grade,
    schedule_roles,
    fail_reason,
    resident_speed_roles,
    speed_grade,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cacheon.eval.crossover_runtime import ResidentSpeedPolicy, ResidentCrossoverPlan, ResidentCrossoverEvidence

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


def planned_schedule(plan: SessionExecutionPlan, roles: tuple[str, ...]) -> SessionExecutionPlan:
    """Expand the exact per-arm role order, retaining conditioning in every read."""
    if not any("prefill" in role for role in roles):
        return expanded_schedule(plan, len(roles))
    count = len(plan.prompt_batches)
    budgets = plan.batch_max_new_tokens or (plan.max_new_tokens,) * count
    prompts = plan.batch_expected_prompt_tokens or (plan.expected_prompt_tokens,) * count
    return replace(plan, prompt_batches=plan.prompt_batches * len(roles),
                   batch_max_new_tokens=tuple(token for role in roles for token in
                       ((PREFILL_READ_BUDGET,) * count if "prefill" in role else budgets)),
                   batch_expected_prompt_tokens=prompts * len(roles))


def read_rate(
    role: str,
    lane_digest: str,
    controller: OpenedOuterSession,
    template: SessionExecutionPlan,
) -> ResidentReadRate:
    """Take one complete read from the resident controller and rate it."""

    first = controller.next_batch_index
    rows = tuple(
        controller.execute_next() for _ in range(len(template.prompt_batches))
    )
    if (
        not rows
        or any(type(row) is not BatchExecutionEvidence or row.audit_receipts for row in rows)
        or tuple(row.batch_index for row in rows)
        != tuple(range(rows[0].batch_index, rows[-1].batch_index + 1))
        or tuple(
            controller.plan.prompt_batches[row.batch_index] for row in rows
        )
        != template.prompt_batches
    ):
        raise CrossoverRuntimeError("resident read batches are incomplete")
    return _rate_from_batches(role, lane_digest, controller.plan.launch_digest,
                              controller.session_id, first, rows, template)


def _rate_from_batches(
    role: str, lane_digest: str, launch_digest: str, session_id: str,
    first: int, rows: tuple[BatchExecutionEvidence, ...], template: SessionExecutionPlan,
) -> ResidentReadRate:
    """Derive one read from host batch evidence, shared by execution and raw regrade."""
    timed = rows[template.warmup_count:]
    conditioning = rows[template.warmup_count - template.conditioning_count:template.warmup_count]
    if not timed or not conditioning:
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
        launch_digest,
        session_id,
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
        _timed_windows(timed),
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


def repeat_required(policy: "ResidentSpeedPolicy", rates: tuple[ResidentReadRate, ...]) -> bool:
    """Repeat only a valid threshold crossing, never an invalid or decisive run."""
    if policy.version < 13 or tuple(row.role for row in rates) != schedule_roles(policy.version):
        return False
    grade = _grade_rounds(policy, rates)
    return (grade.decision is SpeedStageDecision.NO_DECISION and grade.verdict.confident
            and not grade.conditioning_failed
            and (grade.prefill_verdict is None or grade.prefill_verdict.confident))


def grade_schedule(policy: "ResidentSpeedPolicy", rates: tuple[ResidentReadRate, ...]) -> ScheduleGrade:
    """Regrade the complete sealed schedule, including the bounded-repeat trigger."""
    roles = resident_speed_roles(policy.version, len(rates))
    if policy.version < 8 or roles is None or tuple(row.role for row in rates) != roles:
        raise CrossoverRuntimeError("resident speed read set is not the precommitted schedule")
    count = len(schedule_roles(policy.version))
    if len(rates) == count:
        return _grade_rounds(policy, rates)
    if not repeat_required(policy, rates[:count]):
        raise CrossoverRuntimeError("resident repeat was not authorized by the initial measurement")
    return _grade_rounds(policy, rates[:count], rates[count:])


def _grade_rounds(policy, rates, repeated=None) -> ScheduleGrade:
    def grade_phase(phase_policy, start):
        first = rates[start:start + 3]
        if repeated is not None:
            return combined_speed_grade(phase_policy, first, repeated[start:start + 3])
        return speed_grade(phase_policy, [first[0], first[2]], [first[1]])

    verdict, decision = grade_phase(policy, 0)
    conditioning = policy.conditioning_regression(rates[0], rates[1])
    if repeated is not None:
        conditioning |= policy.conditioning_regression(repeated[0], repeated[1])
    if conditioning and verdict.confident and (policy.version >= 13 or decision is not SpeedStageDecision.NO_DECISION):
        decision = SpeedStageDecision.FAIL
    lane = "decode" if decision is SpeedStageDecision.PASS else None
    settled = verdict.speedup
    prefill = None
    if policy.version in (12, 15):
        prefill, admitted = grade_phase(prefill_policy(policy), 3)
        if (
            lane is None
            and admitted is SpeedStageDecision.PASS
            and verdict.confident
            and not conditioning
            and fail_reason(verdict) != "candidate_slower"
        ):
            decision, lane = SpeedStageDecision.PASS, "prefill"
            settled = credited_speedup(policy, prefill)
        elif (policy.version >= 13 and lane is None
              and verdict.confident and not conditioning
              and fail_reason(verdict) != "candidate_slower"
              and admitted is SpeedStageDecision.NO_DECISION):
            decision = SpeedStageDecision.NO_DECISION
    return ScheduleGrade(
        verdict, decision, conditioning, prefill, lane, format(settled, ".17g")
    )


def run_resident_crossover_speed(
    plan: ResidentCrossoverPlan,
    *,
    baseline_executor: OCIEngineExecutor,
    candidate_executor: OCIEngineExecutor,
    model_mount: TrustedArenaModelMountReceipt,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> ResidentCrossoverEvidence:
    """Run the exact production/testnet speed scheduler for one candidate."""

    from cacheon.eval.crossover_runtime import ResidentCrossoverPlan, ResidentCrossoverEvidence, _lane_digest

    if (
        type(plan) is not ResidentCrossoverPlan
        or type(baseline_executor) is not OCIEngineExecutor
        or type(candidate_executor) is not OCIEngineExecutor
        or baseline_executor is candidate_executor
        or type(model_mount) is not TrustedArenaModelMountReceipt
    ):
        raise CrossoverRuntimeError("resident crossover authorities are not exact")
    started = float(clock())
    thresholds = (deadline, started)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in thresholds
    ):
        raise CrossoverRuntimeError("resident crossover thresholds are invalid")
    stage_deadline = min(float(deadline), started + plan.policy.max_stage_seconds)
    if stage_deadline <= started:
        raise CrossoverRuntimeError("resident speed stage has no wall-clock budget")
    baseline_lane = _lane_digest(baseline_executor, plan.baseline)
    candidate_lane = _lane_digest(candidate_executor, plan.candidate)
    if baseline_lane == candidate_lane:
        raise CrossoverRuntimeError("resident executors reused one lane namespace")
    roles = schedule_roles(plan.policy.version)
    maximum_roles = schedule_roles(plan.policy.version, repeat=plan.policy.version >= 13)
    baseline_plan = planned_schedule(plan.baseline.session_plan,
                                     tuple(role for role in maximum_roles if role.startswith("B")))
    candidate_plan = planned_schedule(plan.candidate.session_plan,
                                      tuple(role for role in maximum_roles if role.startswith("C")))
    schedule = ReadSchedule()

    def driver(prefix, peer, arm, lane):
        def run(controller: OpenedOuterSession) -> SessionExecutionEvidence:
            try:
                schedule.put(prefix + "_ready")
                schedule.get(peer + "_ready", deadline=stage_deadline, clock=clock)
                for round_index in range(2 if plan.policy.version >= 13 else 1):
                    current = roles if round_index == 0 else maximum_roles[len(roles):]
                    for index, role in enumerate(current):
                        if not role.startswith(prefix):
                            continue
                        if index:
                            schedule.get(current[index - 1], deadline=stage_deadline, clock=clock)
                        schedule.put(role, read_rate(role, lane, controller, arm.session_plan))
                    key = f"round_{round_index}"
                    if prefix == "B":
                        taken = roles if round_index == 0 else maximum_roles
                        rates = tuple(schedule.values[role] for role in taken)
                        again = round_index == 0 and repeat_required(plan.policy, rates)
                        if not again:
                            schedule.put("rates", rates)
                            schedule.put("grade", grade_schedule(plan.policy, rates))
                        schedule.put(key, again)
                    # Neither lane tears down or advances until the whole round is graded.
                    if not schedule.get(key, deadline=stage_deadline, clock=clock):
                        break
                return controller.finish(require_all=False)
            except BaseException as exc:
                schedule.fail(exc)
                raise
        return run

    def execute(executor, arm, expanded_plan, driver):
        try:
            return executor.execute_opened(
                arm.launch,
                arm.binding,
                model_mount,
                expanded_plan,
                deadline=stage_deadline,
                driver=driver,
            )
        except BaseException as exc:
            schedule.fail(exc)
            raise

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="cacheon-resident"
    ) as pool:
        futures = (
            pool.submit(
                execute,
                baseline_executor,
                plan.baseline,
                baseline_plan,
                driver("B", "C", plan.baseline, baseline_lane),
            ),
            pool.submit(
                execute,
                candidate_executor,
                plan.candidate,
                candidate_plan,
                driver("C", "B", plan.candidate, candidate_lane),
            ),
        )
        executions: list[EngineExecutionEvidence] = []
        errors: list[BaseException] = []
        for future in futures:
            try:
                executions.append(future.result())
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise schedule.failure or errors[0]
    if len(executions) != 2 or any(
        type(row) is not EngineExecutionEvidence for row in executions
    ):
        raise CrossoverRuntimeError("resident speed returned incomplete evidence")
    grade = schedule.values["grade"]
    if type(grade) is not ScheduleGrade:
        raise CrossoverRuntimeError("resident speed grade is incomplete")
    rates = schedule.values["rates"]
    if any(type(row) is not ResidentReadRate for row in rates):
        raise CrossoverRuntimeError("resident speed rates are incomplete")
    baseline_quiescence = baseline_executor.prove_quiescent()
    candidate_quiescence = candidate_executor.prove_quiescent()
    completed = float(clock())
    evidence = ResidentCrossoverEvidence(
        plan.digest,
        plan.selected_delta_digest,
        plan.policy,
        marginal_workload_digest(plan.baseline.session_plan),
        baseline_lane,
        candidate_lane,
        executions[0],
        executions[1],
        baseline_quiescence,
        candidate_quiescence,
        rates,  # type: ignore[arg-type]
        grade_schedule(plan.policy, rates[:len(roles)]).verdict,
        grade.verdict,
        len(rates) > len(roles),
        grade.decision,
        ("borderline_" if len(rates) > len(roles) else "clear_") + grade.decision.value.lower(),
        started,
        completed,
    )
    evidence.regrade(plan)
    return evidence


__all__ = [
    "PREFILL_READ_BUDGET",
    "ReadSchedule",
    "ScheduleGrade",
    "credited_speedup",
    "expanded_schedule",
    "planned_schedule",
    "repeat_required",
    "grade_schedule",
    "prefill_policy",
    "read_rate",
    "run_resident_crossover_speed",
]

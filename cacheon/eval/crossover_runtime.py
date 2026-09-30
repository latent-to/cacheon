"""Bounded production speed stage with two resident TP lanes.

Stock and candidate load once on disjoint lanes and replay the sealed session
slice in paired windows (speed policies 16 and 17; ``goodput_runtime`` runs the
windows and ``service_capacity`` grades them). Audit and pristine T are later
stages and must not run unless this returns PASS.

The batch-cell B/C/B-prime schedules (policies 8-15) were deleted with the
batch evaluator; their retained products stay readable as stored JSON only.
"""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Callable

from cacheon.eval.resident_speed_policy import ResidentSpeedPolicy
from cacheon.eval.goodput_runtime import GoodputPolicy as GoodputPolicy, GoodputReadSet
from cacheon.eval.reference_protocol import ReferencePromptInput
from cacheon.eval.qualification import SelectionEntropyReceipt
from cacheon.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
from cacheon.eval.oci_backend import (
    EngineExecutionEvidence,
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
)
from cacheon.eval.oci_outer_session import (
    SessionExecutionEvidence,
    SessionExecutionPlan,
)
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.scoring import CrossoverRuntimeError, SpeedupVerdict, marginal_workload_digest
from cacheon.eval.speed_verdict import SpeedStageDecision
from cacheon.stack_identity import canonical_digest, require_sha256_hex


@dataclass(frozen=True)
class ResidentArmPlan:
    launch: EngineLaunchSpec
    binding: TrustedLaunchBinding
    session_plan: SessionExecutionPlan
    executor_namespace_digest: str
    runtime_resource_policy_digest: str
    device_configuration_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.launch) is not EngineLaunchSpec
            or type(self.binding) is not TrustedLaunchBinding
            or type(self.session_plan) is not SessionExecutionPlan
            or self.session_plan.launch_digest != self.launch.digest
            or self.session_plan.expected_engine_config_digest
            != self.launch.engine_config_digest
            or self.session_plan.audit_policy is not None
            or self.session_plan.engine_config.disable_cuda_graph
        ):
            raise CrossoverRuntimeError(
                "resident speed arm must be one exact graph-on, audit-free launch"
            )
        for field in (
            "executor_namespace_digest",
            "runtime_resource_policy_digest",
            "device_configuration_digest",
        ):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None


def _workload(plan: SessionExecutionPlan) -> tuple[object, ...]:
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
        plan.measure_phase_latency,
        None if plan.replay is None else plan.replay.workload_identity(),
    )


@dataclass(frozen=True)
class ResidentCrossoverPlan:
    selected_delta_digest: str
    baseline: ResidentArmPlan
    candidate: ResidentArmPlan
    policy: ResidentSpeedPolicy

    def __post_init__(self) -> None:
        try:
            selected = require_sha256_hex(
                self.selected_delta_digest, field="selected_delta_digest"
            )
        except ValueError as exc:
            raise CrossoverRuntimeError(str(exc)) from None
        object.__setattr__(self, "selected_delta_digest", selected)
        if (
            type(self.baseline) is not ResidentArmPlan
            or type(self.candidate) is not ResidentArmPlan
            or type(self.policy) is not ResidentSpeedPolicy
            or _workload(self.baseline.session_plan)
            != _workload(self.candidate.session_plan)
        ):
            raise CrossoverRuntimeError("resident crossover plan is inconsistent")
        replay = self.baseline.session_plan.replay
        if (replay is None or len(replay.loads) != 1 or replay.contract != self.policy.goodput.contract
                or replay.elapsed_cost != bool(self.policy.goodput.error_rate)):
            raise CrossoverRuntimeError("goodput plan requires one sealed load and its service contract")
        allowed_differences = {
            "stack_digest",
            "tree_digest",
            "native_build_spec_digest",
            "resource_policy_digest",
        }
        common = set(self.baseline.launch.__dataclass_fields__) - {
            "hardware",
            *allowed_differences,
        }
        if any(
            getattr(self.baseline.launch, field)
            != getattr(self.candidate.launch, field)
            for field in common
        ):
            raise CrossoverRuntimeError("resident arms differ outside contribution identity")
        left, right = self.baseline.launch.hardware, self.candidate.launch.hardware
        shape = lambda row: (
            row.visible_gpu_count,
            row.architecture,
            row.topology_class,
            row.tp_size,
            row.ep_size,
            row.dp_size,
        )
        if shape(left) != shape(right) or left.visible_gpu_count != left.tp_size:
            raise CrossoverRuntimeError("resident TP lanes are not equivalent")
        if set(self.baseline.binding.physical_hardware.physical_gpu_ids) & set(
            self.candidate.binding.physical_hardware.physical_gpu_ids
        ):
            raise CrossoverRuntimeError("resident TP lanes overlap physical GPUs")
        if (
            self.baseline.executor_namespace_digest
            == self.candidate.executor_namespace_digest
        ):
            raise CrossoverRuntimeError(
                "resident TP lanes require distinct executor namespaces"
            )

    @property
    def digest(self) -> str:
        def arm(value: ResidentArmPlan) -> dict[str, object]:
            physical = value.binding.physical_hardware
            return {
                "device_policy": physical.device_policy_digest,
                "device_configuration": value.device_configuration_digest,
                "executor_namespace": value.executor_namespace_digest,
                "launch": value.launch.digest,
                "physical_gpu_ids": list(physical.physical_gpu_ids),
                "runtime_resource_policy": value.runtime_resource_policy_digest,
                "session_workload": marginal_workload_digest(value.session_plan),
                "topology": physical.topology_digest,
            }

        return canonical_digest(
            "cacheon.qualification.resident-crossover-plan",
            {
                "baseline": arm(self.baseline),
                # The deleted pair-native lane injected an incumbent bundle
                # here; every two-process plan sealed the empty member, and the
                # literal stays so no sealed plan digest moves.
                "baseline_bundle": {"digest": "", "slots": []},
                "candidate": arm(self.candidate),
                "policy": self.policy.digest,
                "selected_delta": self.selected_delta_digest,
            },
        )

    @property
    def baseline_lane_digest(self) -> str:
        return _expected_lane_digest(self.baseline)

    @property
    def candidate_lane_digest(self) -> str:
        return _expected_lane_digest(self.candidate)


def _expected_lane_digest(arm: ResidentArmPlan) -> str:
    physical = arm.binding.physical_hardware
    return canonical_digest(
        "cacheon.qualification.resident-lane",
        {
            "configuration": arm.device_configuration_digest,
            "namespace": arm.executor_namespace_digest,
            "physical_gpu_ids": list(physical.physical_gpu_ids),
            "policy": physical.device_policy_digest,
            "launch_resource_policy": arm.launch.resource_policy_digest,
            "runtime_policy": arm.runtime_resource_policy_digest,
            "topology": arm.launch.hardware.topology_digest,
        },
    )


def _lane_digest(executor: OCIEngineExecutor, arm: ResidentArmPlan) -> str:
    policy = executor.device_policy
    physical = arm.binding.physical_hardware
    if (
        policy.policy_sha256 != arm.launch.hardware.device_policy_digest
        or physical.device_policy_digest != policy.policy_sha256
        or tuple(map(str, policy.physical_gpu_ids)) != physical.physical_gpu_ids
        or executor.manager.namespace_digest != arm.executor_namespace_digest
        or executor.config.prebuild.policy.resource_policy_digest
        != arm.launch.resource_policy_digest
        or executor.config.runtime.digest != arm.runtime_resource_policy_digest
        or policy.configuration_sha256 != arm.device_configuration_digest
    ):
        raise CrossoverRuntimeError("executor and resident lane binding differ")
    return _expected_lane_digest(arm)



def _execution_digest(value: EngineExecutionEvidence) -> str:
    return canonical_digest(
        "cacheon.qualification.resident-engine-execution",
        {
            "argv": value.runtime_argv_sha256,
            "batches": [
                [
                    row.batch_index,
                    row.request_id,
                    row.nonce,
                    format(row.request_started_at, ".17g"),
                    format(row.response_completed_at, ".17g"),
                    row.token_numerator,
                ] + ([[format(first, ".17g"), format(last, ".17g")]
                      for first, last in row.prompt_latencies]
                     if row.prompt_latencies else []) + ([list(row.input_ids_sha256)] if row.input_ids_sha256 else [])
                for row in value.session.batches
            ],
            "devices": [
                [row.launch_id, row.sequence, list(row.selected_physical_gpu_ids)]
                for row in value.device_receipts
            ],
            "launch": value.launch_digest,
            "native": value.native_publication_digest,
            "resource_policy": value.resource_policy_digest,
            "schema": value.schema,
            "session": value.session.session_id,
        },
    )


def _validate_execution_binding(execution: EngineExecutionEvidence, arm: ResidentArmPlan) -> None:
    """Bind either measurement schedule to the same launched engine and physical lane."""
    session, plan = execution.session, arm.session_plan
    if (
        execution.schema != "cacheon.oci-resident-engine-execution.v1"
        or execution.launch_digest != arm.launch.digest
        or execution.resource_policy_digest != arm.runtime_resource_policy_digest
        or type(session) is not SessionExecutionEvidence
        or session.launch_digest != arm.launch.digest
        or session.preflight != plan.expected_preflight
        or session.warmup_count != plan.warmup_count
        or session.conditioning_count != plan.conditioning_count
    ):
        raise CrossoverRuntimeError("resident execution differs from its sealed arm")
    receipts = execution.device_receipts
    physical = arm.binding.physical_hardware.physical_gpu_ids
    if type(receipts) is not tuple or len(receipts) != 3:
        raise CrossoverRuntimeError("resident execution lacks device receipt coverage")
    if all(value.isdecimal() for value in physical) and any(
        tuple(row.selected_physical_gpu_ids) != tuple(map(int, physical))
        for row in receipts
    ):
        raise CrossoverRuntimeError("resident execution ran on another physical lane")
    if any(
        getattr(row, "policy_sha256", arm.launch.hardware.device_policy_digest)
        != arm.launch.hardware.device_policy_digest
        or getattr(row, "configuration_sha256", arm.device_configuration_digest)
        != arm.device_configuration_digest
        for row in receipts
    ):
        raise CrossoverRuntimeError("resident execution changed device authority")


@dataclass(frozen=True)
class ResidentCrossoverEvidence:
    plan_digest: str
    selected_delta_digest: str
    policy: ResidentSpeedPolicy
    workload_digest: str
    baseline_lane_digest: str
    candidate_lane_digest: str
    baseline_execution: EngineExecutionEvidence
    candidate_execution: EngineExecutionEvidence
    baseline_quiescence: OCIQuiescenceReceipt
    candidate_quiescence: OCIQuiescenceReceipt
    # Batch read rows (policies 8-15). Replay evidence seals the empty tuple, and
    # the field stays so retained continuations decode unchanged.
    rates: tuple[str, ...]
    initial_verdict: SpeedupVerdict
    final_verdict: SpeedupVerdict
    escalated: bool
    decision: SpeedStageDecision
    exit_reason: str
    started_monotonic_s: float
    completed_monotonic_s: float
    goodput: GoodputReadSet | None = dataclass_field(default=None, metadata={"wire_optional": True})
    prompt_pairs: tuple[tuple[str, int, int], ...] = dataclass_field(default=(), metadata={"wire_optional": True})
    reference_inputs: tuple[ReferencePromptInput, ...] = dataclass_field(default=(), metadata={"wire_optional": True})
    quality_entropy: SelectionEntropyReceipt | None = dataclass_field(default=None, metadata={"wire_optional": True})
    prior_executions: tuple[EngineExecutionEvidence, ...] = dataclass_field(default=(), metadata={"wire_optional": True})

    def __post_init__(self) -> None:
        version = (
            self.policy.version if type(self.policy) is ResidentSpeedPolicy else 0
        )
        if ((version == 17 and (len(self.prior_executions) != 2
                               or any(type(v) is not EngineExecutionEvidence for v in self.prior_executions)))
            or (version != 17 and self.prior_executions)):
            raise CrossoverRuntimeError("swapped replay requires both first-orientation executions")
        for field in (
            "plan_digest",
            "selected_delta_digest",
            "workload_digest",
            "baseline_lane_digest",
            "candidate_lane_digest",
        ):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None
        if (
            self.escalated
            or type(self.goodput) is not GoodputReadSet
            or self.rates != ()
            or type(self.policy) is not ResidentSpeedPolicy
            or type(self.baseline_execution) is not EngineExecutionEvidence
            or type(self.candidate_execution) is not EngineExecutionEvidence
            or type(self.baseline_quiescence) is not OCIQuiescenceReceipt
            or type(self.candidate_quiescence) is not OCIQuiescenceReceipt
            or type(self.initial_verdict) is not SpeedupVerdict
            or type(self.final_verdict) is not SpeedupVerdict
            or type(self.escalated) is not bool
            or type(self.decision) is not SpeedStageDecision
            or self.baseline_lane_digest == self.candidate_lane_digest
            or self.exit_reason
            != ("borderline_" if self.escalated else "clear_")
            + self.decision.value.lower()
            or not all(
                math.isfinite(value)
                for value in (
                    self.started_monotonic_s,
                    self.completed_monotonic_s,
                )
            )
            or self.completed_monotonic_s <= self.started_monotonic_s
            or self.completed_monotonic_s - self.started_monotonic_s
            > self.policy.max_stage_seconds
        ):
            raise CrossoverRuntimeError("resident crossover evidence is malformed")

    def regrade(self, plan: ResidentCrossoverPlan) -> SpeedupVerdict:
        """Recompute the grade only from the sealed plan and the retained turn records."""

        final_plan = plan
        if plan.policy.version == 17:
            from cacheon.eval.goodput_runtime import _orientation_plan
            windows = plan.baseline.session_plan.replay.windows
            final_plan = _orientation_plan(plan, windows-windows//2, swapped=True)
        if (
            type(plan) is not ResidentCrossoverPlan
            or self.plan_digest != plan.digest
            or self.selected_delta_digest != plan.selected_delta_digest
            or self.policy != plan.policy
            or self.workload_digest
            != marginal_workload_digest(plan.baseline.session_plan)
            or self.baseline_lane_digest != _expected_lane_digest(final_plan.baseline)
            or self.candidate_lane_digest != _expected_lane_digest(final_plan.candidate)
            or self.baseline_quiescence.namespace_digest
            != final_plan.baseline.executor_namespace_digest
            or self.candidate_quiescence.namespace_digest
            != final_plan.candidate.executor_namespace_digest
            or any(
                (row.container_ids, row.lease_records, row.resource_entries)
                != ((), (), ())
                for row in (
                    self.baseline_quiescence,
                    self.candidate_quiescence,
                )
            )
        ):
            raise CrossoverRuntimeError("resident evidence names another sealed plan")
        from cacheon.eval.goodput_runtime import regrade_goodput_execution
        grade = regrade_goodput_execution(self, plan)
        if (self.initial_verdict != grade.verdict or self.final_verdict != grade.verdict
            or self.decision is not grade.decision):
            raise CrossoverRuntimeError("goodput headline does not regrade")
        return grade.verdict

    @property
    def digest(self) -> str:
        verdict = lambda row: {
            "confident": row.confident,
            "noise": format(row.noise, ".17g"),
            "required": format(row.required, ".17g"),
            "speedup": format(row.speedup, ".17g"),
        }
        return canonical_digest(
            "cacheon.qualification.resident-crossover-speed",
            {
                "baseline_execution": _execution_digest(self.baseline_execution),
                "baseline_lane": self.baseline_lane_digest,
                "baseline_quiescence": self.baseline_quiescence.digest,
                "candidate_execution": _execution_digest(self.candidate_execution),
                "candidate_lane": self.candidate_lane_digest,
                "candidate_quiescence": self.candidate_quiescence.digest,
                "completed": format(self.completed_monotonic_s, ".17g"),
                "decision": self.decision.value,
                "escalated": self.escalated,
                "final": verdict(self.final_verdict),
                "initial": verdict(self.initial_verdict),
                "policy": self.policy.digest,
                "plan": self.plan_digest,
                "rates": [],
                "goodput": self.goodput.to_dict(),
                **({"prior_executions": [_execution_digest(row) for row in self.prior_executions]}
                   if self.prior_executions else {}),
                **({"prompt_pairs": self.prompt_pairs,
                    "reference_inputs": [{"prompt": row.prompt_digest,
                                          "input": hashlib.sha256(row.input_bytes).hexdigest(),
                                          "roles": [[role.output_ids, role.supports] for role in row.roles]}
                                         for row in self.reference_inputs],
                    "quality_entropy": None if self.quality_entropy is None else self.quality_entropy.digest}),
                "selected_delta": self.selected_delta_digest,
                "started": format(self.started_monotonic_s, ".17g"),
                "workload": self.workload_digest,
            },
        )


@dataclass(frozen=True)
class ResidentCandidateView:
    """Compatibility view of the singleton prepared candidate."""

    candidate: object
    execution: EngineExecutionEvidence

    @property
    def arm(self):
        return self.candidate.arm


@dataclass(frozen=True)
class ResidentMarginalLifecycleEvidence:
    """One singleton marginal lifecycle backed by exact resident role spans."""

    prepared: object
    plan: ResidentCrossoverPlan
    crossover: ResidentCrossoverEvidence

    def __post_init__(self) -> None:
        from cacheon.eval.marginal_runtime import PreparedMarginalRuntime

        if (
            type(self.prepared) is not PreparedMarginalRuntime
            or len(self.prepared.candidates) != 1
            or type(self.plan) is not ResidentCrossoverPlan
            or type(self.crossover) is not ResidentCrossoverEvidence
        ):
            raise CrossoverRuntimeError("resident lifecycle is not a singleton authority")
        candidate = self.prepared.candidates[0]
        if (
            candidate.arm.selected_delta_digest != self.plan.selected_delta_digest
            or candidate.launch.digest != self.plan.candidate.launch.digest
            or candidate.session_plan != self.plan.candidate.session_plan
            or candidate.binding.launch_binding != self.plan.candidate.binding
            or self.prepared.baseline_launch.stack_digest
            != self.plan.baseline.launch.stack_digest
            or self.prepared.baseline_launch.tree_digest
            != self.plan.baseline.launch.tree_digest
        ):
            raise CrossoverRuntimeError("resident lifecycle differs from its prepared arm")
        self.crossover.regrade(self.plan)

    @property
    def source(self):
        return self.prepared.source

    @property
    def candidates(self) -> tuple[ResidentCandidateView, ...]:
        return (
            ResidentCandidateView(
                self.prepared.candidates[0], self.crossover.candidate_execution
            ),
        )

    @property
    def final_baseline(self) -> EngineExecutionEvidence:
        """The resident baseline lifetime that served the last paired window."""

        return self.crossover.baseline_execution

    @property
    def timed_session_ids(self) -> frozenset[str]:
        return frozenset(
            {
                self.crossover.baseline_execution.session.session_id,
                self.crossover.candidate_execution.session.session_id,
            }
        )


def run_resident_crossover_speed(
    plan: ResidentCrossoverPlan,
    *,
    baseline_executor: OCIEngineExecutor,
    candidate_executor: OCIEngineExecutor,
    model_mount: TrustedArenaModelMountReceipt,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
    quality_control=None,
) -> ResidentCrossoverEvidence:
    """Run the exact production speed stage for one candidate."""

    from cacheon.eval.goodput_runtime import run_goodput_pair

    if (
        type(plan) is not ResidentCrossoverPlan
        or type(baseline_executor) is not OCIEngineExecutor
        or type(candidate_executor) is not OCIEngineExecutor
        or baseline_executor is candidate_executor
        or type(model_mount) is not TrustedArenaModelMountReceipt
    ):
        raise CrossoverRuntimeError("resident crossover authorities are not exact")
    return run_goodput_pair(
        plan, baseline_executor=baseline_executor, candidate_executor=candidate_executor,
        model_mount=model_mount, deadline=deadline, clock=clock, quality_control=quality_control,
    )


__all__ = [
    "CrossoverRuntimeError",
    "ResidentArmPlan",
    "ResidentCrossoverEvidence",
    "ResidentCrossoverPlan",
    "ResidentMarginalLifecycleEvidence",
    "ResidentSpeedPolicy",
    "SpeedStageDecision",
    "run_resident_crossover_speed",
]

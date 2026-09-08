"""Bounded production speed stage with two resident TP lanes.

Stock and candidate load once on disjoint lanes. GPU work is serialized because
simultaneous TP4 reads were measured to distort both lanes. Audit and pristine T
are later stages and must not run unless this returns PASS.

This module is the substrate for every candidate since the pair-native lane
was deleted on 2026-09-06: CUDA, C++ and PTX bundles whose kernels must be
compiled and linked into the engine that runs them, and hot-swappable bundles
alike. It refused every version-6 policy outright between 2026-08-15
(87944430) and the version-8 change, so those bundles screened clean and then
never received a speed verdict at all.

Its schedule is version 8 (version 9 for a mixed-cell workload): B, C and
B-prime, always, and nothing beyond them. Policies below version 8 are refused
here and their evidence is refused at construction; the adaptive five-read
escalation and the conditional bookend left with the MiniMax-M3 history seal.

B-prime is precommitted rather than earned by a close call. The reason is the
quality gate, not the speed gate: `reference_quality.stock_drift_upper_bound`
harvests its stock-drift control from the second baseline read, and it is the
only consumer of that read -- the candidate-versus-baseline comparison discards
it. Reading it unconditionally also keeps the anti-reroll property: a read
taken regardless of the outcome cannot be a read taken because of it.
"""

from __future__ import annotations

import concurrent.futures
import math
import statistics
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from cacheon.eval.resident_measurement import (
    CrossoverRuntimeError, ResidentReadRate, TimedWindow as TimedWindow, _timed_windows,
)
from cacheon.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
from cacheon.eval.oci_backend import (
    EngineExecutionEvidence,
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
)
from cacheon.eval.oci_outer_session import (
    BatchExecutionEvidence,
    OpenedOuterSession,
    SessionExecutionEvidence,
    SessionExecutionPlan,
)
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.scoring import SpeedupVerdict, marginal_workload_digest
from cacheon.eval.speed_verdict import (
    SpeedStageDecision,
    speed_grade,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex


@dataclass(frozen=True)
class ResidentSpeedPolicy:
    """Authority for adaptive reads and the speed-stage wall-clock SLA."""

    max_stage_seconds: int
    min_margin: float
    noise_multiplier: float
    max_noise: float
    calibration_digest: str
    calibration_context_digest: str
    version: int = 1
    max_qualification_seconds: int = 7_200
    min_windows: int = 0
    max_window_scatter: float = 0.0
    max_conditioning_slowdown: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.version) is not int
            or self.version not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
            or type(self.max_stage_seconds) is not int
            or not 60 <= self.max_stage_seconds <= 7_200
            or type(self.max_qualification_seconds) is not int
            or not self.max_stage_seconds
            <= self.max_qualification_seconds
            <= 14_400
            or type(self.min_windows) is not int
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (
                    self.min_margin,
                    self.noise_multiplier,
                    self.max_noise,
                    self.max_window_scatter,
                    self.max_conditioning_slowdown,
                )
            )
            or not 0 < self.min_margin < 1
            or self.noise_multiplier <= 0
            or not 0 <= self.max_noise < 1
        ):
            raise CrossoverRuntimeError("resident speed policy is unsupported")
        if self.version >= 2 and self.max_noise > 0.02:
            # Version 2 scores timed windows, where the hardened stack has
            # demonstrated <=0.8% honest spread; a looser ceiling would let a
            # broken measurement convict or crown instead of NO_DECISION.
            raise CrossoverRuntimeError(
                "resident speed policy v2 requires max_noise <= 0.02"
            )
        if self.version >= 3:
            # Version 3 grades the median over per-batch timed windows and
            # refuses any read whose own window scatter exceeds the sealed
            # bound: the noise floor becomes measured-per-read evidence
            # instead of an assumption about the box.
            if not 3 <= self.min_windows <= 512:
                raise CrossoverRuntimeError(
                    "resident speed policy v3 requires 3..512 timed windows"
                )
            # Version 3 refuses to grade an unfit read at all, so its bound is
            # necessarily also a ceiling on what may be sealed. Version 4
            # always produces a rate and carries scatter as recorded fitness
            # evidence, so a looser advisory bound is admissible there.
            scatter_ceiling = 0.25 if self.version >= 4 else 0.05
            if not 0 < self.max_window_scatter <= scatter_ceiling:
                raise CrossoverRuntimeError(
                    f"resident speed policy v{min(self.version, 4)} requires a"
                    f" window scatter bound in (0, {scatter_ceiling}]"
                )
            # Conditioning is outside scored windows. The historical v3
            # bound catches gross startup regressions; each timed full request
            # also includes prefill. Streaming phase measurements separate
            # first-token delivery from subsequent generation.
            if not 1.0 < self.max_conditioning_slowdown <= 2.0:
                raise CrossoverRuntimeError(
                    "resident speed policy v3 requires a conditioning slowdown"
                    " bound in (1, 2]"
                )
        elif (
            self.min_windows != 0
            or self.max_window_scatter != 0.0
            or self.max_conditioning_slowdown != 0.0
        ):
            raise CrossoverRuntimeError(
                "window thresholds require resident speed policy v3"
            )
        for field in ("calibration_digest", "calibration_context_digest"):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None

    def conditioning_regression(
        self, baseline_row: object, candidate_row: object
    ) -> bool:
        """Whether the candidate's conditioning span regressed past the bound.

        Conditioning spans carry warm/cold session structure (measured 50%
        cold-vs-warm on 2026-07-25), so callers must pair reads of the same
        warmth position: C against B (both first reads, cold) and C-prime
        against B-prime (both continuations, warm). Never mix positions."""

        if self.version < 3:
            return False
        baseline_tokens = baseline_row.conditioning_tokens  # type: ignore[attr-defined]
        candidate_tokens = candidate_row.conditioning_tokens  # type: ignore[attr-defined]
        if baseline_tokens != candidate_tokens:
            raise CrossoverRuntimeError(
                "conditioning spans are not workload-comparable"
            )
        return (
            candidate_row.conditioning_seconds  # type: ignore[attr-defined]
            > self.max_conditioning_slowdown
            * baseline_row.conditioning_seconds  # type: ignore[attr-defined]
        )

    def read_window_scatter(self, row: object) -> float:
        """Relative MAD-about-the-median of one read's per-window rates.

        Robust by construction: the same tail events (completion churn,
        admission hiccups) that motivate the median statistic must not be
        allowed to inflate its own fitness gate."""

        windows = getattr(row, "windows", ())
        if len(windows) < max(self.min_windows, 3):
            raise CrossoverRuntimeError(
                "resident read lacks its required timed windows"
            )
        rates = [window.tokens / window.seconds for window in windows]
        median = statistics.median(rates)
        if not math.isfinite(median) or median <= 0:
            raise CrossoverRuntimeError("resident read window rates are invalid")
        return statistics.median([abs(rate - median) for rate in rates]) / median

    def scored_tokens_per_second(self, row: object) -> float:
        """Verdict basis for one read. Version 1 grades the charged rate
        (conditioning + timed), which double-counts session cold-start against
        whichever arm read first; version 2 grades the steady-state timed
        window only, with conditioning still bounded by the operational
        budget; version 3 grades the median over per-batch timed windows and
        refuses to produce a number at all when the read's own window scatter
        exceeds the sealed bound, so no consumer anywhere can grade an unfit
        measurement."""

        if self.version in (9, 11):
            # Mixed-cell qualification deliberately contains heterogeneous
            # batch widths and output budgets. A median of per-batch rates
            # would erase the minority cell; total timed tokens over the
            # host-observed makespan gives the sealed mixture one rate.
            self.read_window_scatter(row)
            return row.timed_tokens / row.timed_seconds  # type: ignore[attr-defined]
        if self.version >= 3:
            # The call also enforces the sealed window count, which is a
            # structural evidence requirement under every version.
            scatter = self.read_window_scatter(row)
            if self.version < 4 and scatter > self.max_window_scatter:
                raise CrossoverRuntimeError(
                    "resident read window scatter exceeds the sealed bound"
                )
            return statistics.median(
                window.tokens / window.seconds
                for window in row.windows  # type: ignore[attr-defined]
            )
        if self.version >= 2:
            return row.timed_tokens / row.timed_seconds  # type: ignore[attr-defined]
        return row.tokens_per_second  # type: ignore[attr-defined]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.qualification.resident-speed-policy",
            {
                **self.to_dict(),
                # The sealed identity states the schedule it was measured under.
                # Version 8 reads the bookend unconditionally, so it must not
                # claim the conditional read order that versions 6 and 7 seal.
                "borderline_band": (
                    "invariant_over_reads_taken"
                    if self.version >= 8
                    else "pod_bookend_invariant"
                    if self.version >= 6
                    else "one_min_margin_around_required"
                ),
                "read_order": (
                    ["B", "C", "B_prime"]
                    if self.version >= 8
                    else ["B", "C", "B_prime_if_inconclusive"]
                    if self.version >= 6
                    else ["B", "C", "B_prime", "C_prime", "B_double_prime"]
                ),
                "timing": "serialized_resident_host_time",
            },
        )

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "calibration_context_digest": self.calibration_context_digest,
            "calibration_digest": self.calibration_digest,
            "max_noise": format(self.max_noise, ".17g"),
            "max_qualification_seconds": self.max_qualification_seconds,
            "max_stage_seconds": self.max_stage_seconds,
            "min_margin": format(self.min_margin, ".17g"),
            "noise_multiplier": format(self.noise_multiplier, ".17g"),
            "version": self.version,
        }
        if self.version >= 3:
            # Window thresholds exist only under version 3; earlier sealed
            # policies keep their exact historical bytes and digests.
            row["max_conditioning_slowdown"] = format(
                self.max_conditioning_slowdown, ".17g"
            )
            row["max_window_scatter"] = format(self.max_window_scatter, ".17g")
            row["min_windows"] = self.min_windows
        return row

    @classmethod
    def from_dict(cls, value: object) -> "ResidentSpeedPolicy":
        fields = {
            "calibration_context_digest",
            "calibration_digest",
            "max_noise",
            "max_qualification_seconds",
            "max_stage_seconds",
            "min_margin",
            "noise_multiplier",
            "version",
        }
        if (
            type(value) is not dict
            or type(value.get("version")) is not int
        ):
            raise CrossoverRuntimeError("resident speed policy fields differ")
        window_kwargs: dict[str, object] = {}
        if value["version"] >= 3:
            fields |= {
                "max_conditioning_slowdown",
                "max_window_scatter",
                "min_windows",
            }
        if set(value) != fields:
            raise CrossoverRuntimeError("resident speed policy fields differ")
        try:
            if value["version"] >= 3:
                window_kwargs = {
                    "min_windows": value["min_windows"],
                    "max_window_scatter": float(value["max_window_scatter"]),
                    "max_conditioning_slowdown": float(
                        value["max_conditioning_slowdown"]
                    ),
                }
            result = cls(
                max_stage_seconds=value["max_stage_seconds"],  # type: ignore[arg-type]
                min_margin=float(value["min_margin"]),
                noise_multiplier=float(value["noise_multiplier"]),
                max_noise=float(value["max_noise"]),
                calibration_digest=value["calibration_digest"],  # type: ignore[arg-type]
                calibration_context_digest=value["calibration_context_digest"],  # type: ignore[arg-type]
                version=value["version"],  # type: ignore[arg-type]
                max_qualification_seconds=value["max_qualification_seconds"],  # type: ignore[arg-type]
                **window_kwargs,  # type: ignore[arg-type]
            )
            if result.to_dict() != value:
                raise CrossoverRuntimeError("resident speed policy is noncanonical")
            return result
        except (TypeError, ValueError) as exc:
            raise CrossoverRuntimeError("resident speed policy is malformed") from exc

    @classmethod
    def from_calibration(
        cls,
        *,
        max_stage_seconds: int,
        max_qualification_seconds: int = 7_200,
        calibration: object,
        context: object,
        version: int = 1,
        min_windows: int = 0,
        max_window_scatter: float = 0.0,
        max_conditioning_slowdown: float = 0.0,
    ) -> "ResidentSpeedPolicy":
        from cacheon.eval.calibration import (
            CalibrationContext,
            CalibrationManifest,
            decimal_value,
        )

        if (
            type(calibration) is not CalibrationManifest
            or type(context) is not CalibrationContext
            or not calibration.thresholds_frozen
        ):
            raise CrossoverRuntimeError("resident speed calibration is not frozen")
        try:
            calibration.require_context(context)
        except ValueError as exc:
            raise CrossoverRuntimeError(str(exc)) from None
        return cls(
            max_stage_seconds=max_stage_seconds,
            min_margin=float(decimal_value(calibration.speed.min_margin)),
            noise_multiplier=float(decimal_value(calibration.speed.noise_multiplier)),
            max_noise=float(decimal_value(calibration.speed.max_noise)),
            calibration_digest=calibration.digest,
            calibration_context_digest=context.digest,
            version=version,
            max_qualification_seconds=max_qualification_seconds,
            min_windows=min_windows,
            max_window_scatter=max_window_scatter,
            max_conditioning_slowdown=max_conditioning_slowdown,
        )


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


def _expanded(plan: SessionExecutionPlan, reads: int) -> SessionExecutionPlan:
    # Repeat the complete read, including its validator-owned warmup.  The model
    # remains loaded; only the cheap workload conditioning repeats between arms.
    return replace(
        plan,
        prompt_batches=plan.prompt_batches * reads,
        batch_max_new_tokens=(
            plan.batch_max_new_tokens * reads
            if plan.batch_max_new_tokens
            else ()
        ),
        batch_expected_prompt_tokens=(
            plan.batch_expected_prompt_tokens * reads
            if plan.batch_expected_prompt_tokens
            else ()
        ),
    )


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



def _rate(
    role: str,
    lane_digest: str,
    controller: OpenedOuterSession,
    template: SessionExecutionPlan,
    *,
    with_windows: bool = False,
) -> ResidentReadRate:
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


class _Schedule:
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
                     if row.prompt_latencies else [])
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


def _validate_resident_execution(
    execution: EngineExecutionEvidence,
    arm: ResidentArmPlan,
    *,
    reads: int,
) -> None:
    plan = _expanded(arm.session_plan, reads)
    session = execution.session
    if (
        execution.schema != "cacheon.oci-resident-engine-execution.v1"
        or execution.launch_digest != arm.launch.digest
        or execution.resource_policy_digest != arm.runtime_resource_policy_digest
        or type(session) is not SessionExecutionEvidence
        or session.launch_digest != arm.launch.digest
        or session.preflight != plan.expected_preflight
        or session.warmup_count != plan.warmup_count
        or session.conditioning_count != plan.conditioning_count
        or len(session.batches) != len(plan.prompt_batches)
    ):
        raise CrossoverRuntimeError("resident execution differs from its sealed arm")
    previous = session.ready_completed_at
    request_ids: set[str] = set()
    nonces: set[str] = set()
    for index, (row, prompts) in enumerate(
        zip(session.batches, plan.prompt_batches, strict=True)
    ):
        tokens = len(prompts) * plan.request_geometry(index)[0]
        if (
            type(row) is not BatchExecutionEvidence
            or row.batch_index != index
            or row.request_id in request_ids
            or row.nonce in nonces
            or row.request_started_at < previous
            or row.response_completed_at <= row.request_started_at
            or row.token_numerator != tokens
            or row.evidence.observed_tokens != tokens
            or row.audit_receipts
            or bool(row.prompt_latencies) != plan.measure_phase_latency
        ):
            raise CrossoverRuntimeError("resident execution batch evidence is malformed")
        if row.prompt_latencies:
            _timed_windows((row,))
            expected_prompt_tokens = plan.request_geometry(index)[1]
            if expected_prompt_tokens is not None and any(
                prompt.prompt_tokens != expected_prompt_tokens
                for prompt in row.evidence.prompts
            ):
                raise CrossoverRuntimeError("phase input lengths differ from the sealed cell")
        request_ids.add(row.request_id)
        nonces.add(row.nonce)
        previous = row.response_completed_at
    if session.session_completed_at < previous:
        raise CrossoverRuntimeError("resident execution cleanup predates its last batch")
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


def _recomputed_rate(
    rate: ResidentReadRate,
    execution: EngineExecutionEvidence,
    arm: ResidentArmPlan,
) -> ResidentReadRate:
    plan = arm.session_plan
    start = rate.first_batch_index
    stop = start + len(plan.prompt_batches)
    if stop > len(execution.session.batches):
        raise CrossoverRuntimeError("resident rate exceeds its retained session")
    rows = execution.session.batches[start:stop]
    timed = rows[plan.warmup_count :]
    conditioning_start = plan.warmup_count - plan.conditioning_count
    conditioning = rows[conditioning_start : plan.warmup_count]
    if (
        not timed
        or not conditioning
        or rate.last_batch_index != stop - 1
        or tuple(
            _expanded(plan, stop // len(plan.prompt_batches)).prompt_batches[
                row.batch_index
            ]
            for row in rows
        )
        != plan.prompt_batches
    ):
        raise CrossoverRuntimeError("resident rate does not name one complete read")
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
        rate.role,
        _expected_lane_digest(arm),
        arm.launch.digest,
        execution.session.session_id,
        start,
        stop - 1,
        timed[0].batch_index,
        timed[-1].batch_index,
        conditioning_tokens,
        timed_tokens,
        charged_tokens,
        float(conditioning_seconds),
        float(timed_seconds),
        float(charged_seconds),
        float(charged_tokens / charged_seconds),
        # Recompute exactly what the sealed row claims: a windowed row must
        # rebuild from the same raw spans, a legacy row must stay window-free.
        _timed_windows(timed) if rate.windows else (),
    )


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
    rates: tuple[ResidentReadRate, ...]
    initial_verdict: SpeedupVerdict
    final_verdict: SpeedupVerdict
    escalated: bool
    decision: SpeedStageDecision
    exit_reason: str
    started_monotonic_s: float
    completed_monotonic_s: float

    def __post_init__(self) -> None:
        version = (
            self.policy.version if type(self.policy) is ResidentSpeedPolicy else 0
        )
        if version < 8:
            # Evidence sealed under the adaptive or conditional-bookend
            # schedules is MiniMax-M3 history and is not decodable here.
            raise CrossoverRuntimeError(
                "resident crossover evidence below version 8 is sealed history"
            )
        # Three reads, always. Escalation is unreachable, so evidence that
        # claims it is malformed rather than merely unusual.
        roles = ("B", "C", "B_prime")
        schedule_valid = self.escalated is False
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
            not schedule_valid
            or type(self.policy) is not ResidentSpeedPolicy
            or type(self.baseline_execution) is not EngineExecutionEvidence
            or type(self.candidate_execution) is not EngineExecutionEvidence
            or type(self.baseline_quiescence) is not OCIQuiescenceReceipt
            or type(self.candidate_quiescence) is not OCIQuiescenceReceipt
            or type(self.initial_verdict) is not SpeedupVerdict
            or type(self.final_verdict) is not SpeedupVerdict
            or type(self.escalated) is not bool
            or type(self.decision) is not SpeedStageDecision
            or tuple(row.role for row in self.rates) != roles
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
        if any(
            bool(row.windows) != (self.policy.version >= 3) for row in self.rates
        ):
            raise CrossoverRuntimeError(
                "resident read window retention differs from the policy version"
            )

    def regrade(self, plan: ResidentCrossoverPlan) -> SpeedupVerdict:
        """Recompute the adaptive grade only from the sealed plan and raw spans."""

        if (
            type(plan) is not ResidentCrossoverPlan
            or self.plan_digest != plan.digest
            or self.selected_delta_digest != plan.selected_delta_digest
            or self.policy != plan.policy
            or self.workload_digest
            != marginal_workload_digest(plan.baseline.session_plan)
            or self.baseline_lane_digest != _expected_lane_digest(plan.baseline)
            or self.candidate_lane_digest != _expected_lane_digest(plan.candidate)
            or self.baseline_quiescence.namespace_digest
            != plan.baseline.executor_namespace_digest
            or self.candidate_quiescence.namespace_digest
            != plan.candidate.executor_namespace_digest
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
        baseline_rates = tuple(
            row for row in self.rates if row.role.startswith("B")
        )
        candidate_rates = tuple(
            row for row in self.rates if row.role.startswith("C")
        )
        # __post_init__ has already pinned the exact role tuple for this policy
        # version, so the read counts follow from the roles themselves and do
        # not need a second per-version table to fall out of step with.
        expected_baseline_reads = len(baseline_rates)
        expected_candidate_reads = len(candidate_rates)
        _validate_resident_execution(
            self.baseline_execution,
            plan.baseline,
            reads=expected_baseline_reads,
        )
        _validate_resident_execution(
            self.candidate_execution,
            plan.candidate,
            reads=expected_candidate_reads,
        )
        for rows, execution, arm in (
            (baseline_rates, self.baseline_execution, plan.baseline),
            (candidate_rates, self.candidate_execution, plan.candidate),
        ):
            block = len(arm.session_plan.prompt_batches)
            if tuple(row.first_batch_index for row in rows) != tuple(
                range(0, block * len(rows), block)
            ) or any(_recomputed_rate(row, execution, arm) != row for row in rows):
                raise CrossoverRuntimeError("resident rate spans do not independently regrade")
        # One grade over the whole precommitted schedule. There is no
        # adaptive read-shape assertion to make: three reads are not a
        # response to the result, they are the result's entire evidence.
        final, decision = speed_grade(
            plan.policy,
            [baseline_rates[0], baseline_rates[1]],
            [candidate_rates[0]],
            concluding=True,
        )
        if decision is not SpeedStageDecision.NO_DECISION and plan.policy.conditioning_regression(
            baseline_rates[0], candidate_rates[0]
        ):
            decision = SpeedStageDecision.FAIL
        initial = final
        if (
            self.initial_verdict != initial
            or self.final_verdict != final
            or self.decision is not decision
        ):
            raise CrossoverRuntimeError("resident speed headline does not regrade")
        return final

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
                "rates": [row.to_dict() for row in self.rates],
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
        """The resident baseline lifetime containing B-prime."""

        return self.crossover.baseline_execution

    @property
    def timed_session_ids(self) -> frozenset[str]:
        return frozenset(
            {
                self.crossover.baseline_execution.session.session_id,
                self.crossover.candidate_execution.session.session_id,
            }
        )

    @property
    def role_names(self) -> tuple[str, str, str]:
        return ("B", "C", "B_prime")

    def role_batches(self, role: str) -> tuple[BatchExecutionEvidence, ...]:
        matches = tuple(row for row in self.crossover.rates if row.role == role)
        if len(matches) != 1 or role not in self.role_names:
            raise CrossoverRuntimeError("resident quality role is absent or ambiguous")
        rate = matches[0]
        execution = (
            self.crossover.baseline_execution
            if role.startswith("B")
            else self.crossover.candidate_execution
        )
        return execution.session.batches[
            rate.first_batch_index : rate.last_batch_index + 1
        ]


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
    if plan.policy.version < 8:
        raise CrossoverRuntimeError(
            "two-process crossover requires the precommitted B/C/B-prime policy"
        )
    # Version 8 reads B and B-prime on the baseline lane and exactly one C on
    # the candidate lane.
    baseline_plan = _expanded(plan.baseline.session_plan, 2)
    candidate_plan = _expanded(plan.candidate.session_plan, 1)
    schedule = _Schedule()

    windowed = True

    def baseline_driver(controller: OpenedOuterSession) -> SessionExecutionEvidence:
        try:
            schedule.put("baseline_ready")
            schedule.get("candidate_ready", deadline=stage_deadline, clock=clock)
            before = _rate(
                "B",
                baseline_lane,
                controller,
                plan.baseline.session_plan,
                with_windows=windowed,
            )
            schedule.put("B", before)
            candidate = schedule.get("C", deadline=stage_deadline, clock=clock)
            bookend = _rate(
                "B_prime",
                baseline_lane,
                controller,
                plan.baseline.session_plan,
                with_windows=windowed,
            )
            schedule.put("B_prime", bookend)
            final, disposition = speed_grade(
                plan.policy, [before, bookend], [candidate], concluding=True
            )
            # Conditioning pairs by warmth position: C against B, the cold
            # first reads. A regression there is a clear FAIL -- the
            # candidate's unscored work already blew its sealed bound.
            if disposition is not SpeedStageDecision.NO_DECISION and plan.policy.conditioning_regression(before, candidate):
                disposition = SpeedStageDecision.FAIL
            schedule.put("initial", final)
            schedule.put("escalate", False)
            schedule.put("final", final)
            schedule.put("decision", disposition)
            return controller.finish(require_all=False)
        except BaseException as exc:
            schedule.fail(exc)
            raise

    def candidate_driver(controller: OpenedOuterSession) -> SessionExecutionEvidence:
        try:
            schedule.put("candidate_ready")
            schedule.get("baseline_ready", deadline=stage_deadline, clock=clock)
            schedule.get("B", deadline=stage_deadline, clock=clock)
            schedule.put(
                "C",
                _rate(
                    "C",
                    candidate_lane,
                    controller,
                    plan.candidate.session_plan,
                    with_windows=windowed,
                ),
            )
            # Stay resident and idle until the baseline lane has finished.
            # Tearing this CUDA context down while B-prime is charging would
            # contaminate it.
            schedule.get("decision", deadline=stage_deadline, clock=clock)
            return controller.finish(require_all=False)
        except BaseException as exc:
            schedule.fail(exc)
            raise

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
                baseline_driver,
            ),
            pool.submit(
                execute,
                candidate_executor,
                plan.candidate,
                candidate_plan,
                candidate_driver,
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
    initial = schedule.values["initial"]
    final = schedule.values["final"]
    decision = schedule.values["decision"]
    escalated = schedule.values["escalate"] is True
    if (
        type(initial) is not SpeedupVerdict
        or type(final) is not SpeedupVerdict
        or type(decision) is not SpeedStageDecision
    ):
        raise CrossoverRuntimeError("resident speed grade is incomplete")
    rates = tuple(schedule.values[role] for role in ("B", "C", "B_prime"))
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
        initial,
        final,
        escalated,
        decision,
        ("borderline_" if escalated else "clear_") + decision.value.lower(),
        started,
        completed,
    )
    evidence.regrade(plan)
    return evidence


__all__ = [
    "CrossoverRuntimeError",
    "ResidentArmPlan",
    "ResidentCrossoverEvidence",
    "ResidentCrossoverPlan",
    "ResidentMarginalLifecycleEvidence",
    "ResidentReadRate",
    "ResidentSpeedPolicy",
    "SpeedStageDecision",
    "run_resident_crossover_speed",
]

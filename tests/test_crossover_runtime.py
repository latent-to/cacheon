from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.crossover_runtime import (
    CrossoverRuntimeError,
    ResidentArmPlan,
    ResidentCrossoverPlan,
    ResidentSpeedPolicy,
    SpeedStageDecision,
    TimedWindow,
    run_resident_crossover_speed,
)
from cacheon.eval.engine_launch import PhysicalHardwareBinding
from cacheon.eval.oci_backend import EngineExecutionEvidence, OCIEngineExecutor
from cacheon.eval.oci_outer_session import BatchExecutionEvidence, SessionExecutionEvidence
from cacheon.eval.qualification_runner import (
    QualificationStageExit,
    QualificationRunnerError,
    ResidentSpeedWitness,
    _resident_speed_projection_digest,
)
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.scoring import score_speedup
from cacheon.settlement import ResidentLaneOrientation
from tests.test_oci_backend import _case, _manager


def _right_lane(case):
    gpu = replace(
        case.device_policy.expected_gpus[0],
        physical_id=1,
        uuid="GPU-11111111-1111-1111-1111-111111111111",
        pci_bus_id="00000000:02:00.0",
    )
    policy = replace(case.device_policy, expected_gpus=(gpu,))
    hardware = replace(
        case.launch.hardware,
        device_policy_digest=policy.policy_sha256,
    )
    physical = PhysicalHardwareBinding(
        ("1",),
        hardware.architecture,
        hardware.topology_class,
        hardware.topology_digest,
        hardware.tp_size,
        hardware.ep_size,
        hardware.dp_size,
        hardware.device_policy_digest,
    )
    launch = replace(case.launch, hardware=hardware)
    binding = replace(case.binding, physical_hardware=physical)
    plan = replace(
        case.plan,
        launch_digest=launch.digest,
        expected_preflight=replace(
            case.plan.expected_preflight,
            launch_digest=launch.digest,
        ),
    )
    return policy, launch, binding, plan


class _Controller:
    def __init__(
        self,
        plan,
        *,
        lane: str,
        timed_durations: tuple[float, ...],
        trace: list[str],
        trace_lock: threading.Lock,
        active_count: list[int],
        overlap: list[bool],
        batches_per_read: int = 2,
        conditioning_duration: float = 0.1,
    ) -> None:
        self.plan = plan
        self.lane = lane
        self.timed_durations = timed_durations
        self.trace = trace
        self.trace_lock = trace_lock
        self.active_count = active_count
        self.overlap = overlap
        self.batches_per_read = batches_per_read
        self.conditioning_duration = conditioning_duration
        self.session_id = ("a" if lane == "left" else "b") * 32
        self.rows: list[BatchExecutionEvidence] = []
        self.clock = 10.0
        self.closed = False

    @property
    def next_batch_index(self) -> int:
        return len(self.rows)

    def execute_next(self) -> BatchExecutionEvidence:
        index = len(self.rows)
        if index >= len(self.plan.prompt_batches):
            raise CrossoverRuntimeError("fake session exhausted")
        with self.trace_lock:
            if self.active_count[0]:
                self.overlap[0] = True
            self.active_count[0] += 1
            self.trace.append(f"{self.lane}:{index}")
        try:
            time.sleep(0.001)
            role_index, local_index = divmod(index, self.batches_per_read)
            # Timed batches get a tiny deterministic per-window wobble so a
            # multi-window read exercises a real (non-degenerate) median.
            duration = (
                self.conditioning_duration
                if local_index == 0
                else self.timed_durations[role_index]
                * (1.0, 1.001, 0.999)[(local_index - 1) % 3]
            )
            prompts = self.plan.prompt_batches[index]
            tokens = len(prompts) * self.plan.max_new_tokens
            started = self.clock
            self.clock += duration
            row = BatchExecutionEvidence(
                index,
                f"{index + (1 if self.lane == 'left' else 100):032x}",
                f"{index + (1000 if self.lane == 'left' else 2000):032x}",
                started,
                self.clock,
                tokens,
                SimpleNamespace(observed_tokens=tokens),
            )
            self.rows.append(row)
            return row
        finally:
            with self.trace_lock:
                self.active_count[0] -= 1

    def finish(self, *, require_all: bool = True) -> SessionExecutionEvidence:
        assert not require_all
        if len(self.rows) <= self.plan.warmup_count:
            raise CrossoverRuntimeError("fake session has no timed evidence")
        with self.trace_lock:
            self.trace.append(f"{self.lane}:close")
        self.closed = True
        first_timed = self.rows[self.plan.warmup_count]
        return SessionExecutionEvidence(
            self.session_id,
            self.plan.launch_digest,
            self.plan.expected_preflight,
            9.0,
            tuple(self.rows),
            self.plan.warmup_count,
            self.plan.conditioning_count,
            self.rows[0].request_started_at,
            first_timed.response_completed_at,
            sum(
                row.token_numerator
                for row in self.rows[: self.plan.warmup_count + 1]
            ),
            self.clock + 0.01,
        )

    def abort(self) -> None:
        self.closed = True


def _install_fake_execution(
    executor: OCIEngineExecutor,
    *,
    lane: str,
    durations: tuple[float, ...],
    trace: list[str],
    trace_lock: threading.Lock,
    active_count: list[int],
    overlap: list[bool],
    batches_per_read: int = 2,
    conditioning_duration: float = 0.1,
) -> None:
    def execute_opened(launch, binding, mount, plan, *, deadline, driver):
        del binding, mount, deadline
        controller = _Controller(
            plan,
            lane=lane,
            timed_durations=durations,
            trace=trace,
            trace_lock=trace_lock,
            active_count=active_count,
            overlap=overlap,
            batches_per_read=batches_per_read,
            conditioning_duration=conditioning_duration,
        )
        session = driver(controller)
        physical_ids = executor.device_policy.physical_gpu_ids
        receipts = tuple(
            SimpleNamespace(
                completed_monotonic_s=float(index + 1),
                launch_id=("c" if lane == "left" else "d") * 32,
                selected_physical_gpu_ids=physical_ids,
                sequence=index,
                started_monotonic_s=float(index),
            )
            for index in (1, 2, 3)
        )
        return EngineExecutionEvidence(
            "cacheon.oci-resident-engine-execution.v1",
            launch.digest,
            SimpleNamespace(),
            "1" * 64,
            "2" * 64,
            executor.config.runtime.digest,
            SimpleNamespace(),
            ("3" if lane == "left" else "4") * 64,
            ("5" if lane == "left" else "6") * 64,
            (),
            receipts,  # type: ignore[arg-type]
            session,
        )

    executor.execute_opened = execute_opened  # type: ignore[method-assign]


def _speed(plan, baseline, candidate, mount):
    return run_resident_crossover_speed(
        plan,
        baseline_executor=baseline,
        candidate_executor=candidate,
        model_mount=mount,
        deadline=time.monotonic() + 60,
    )


def _rig(
    tmp_path: Path,
    candidate_durations: tuple[float, ...],
    *,
    distinct_runtime_policies: bool = False,
    policy: ResidentSpeedPolicy | None = None,
    timed_batches: int = 1,
    candidate_conditioning: float = 0.1,
    baseline_durations: tuple[float, ...] = (1.0, 1.0, 1.0),
):
    left_case = _case(tmp_path / "left")
    right_case = _case(tmp_path / "right")
    right_policy, right_launch, right_binding, right_plan = _right_lane(right_case)
    left_config = left_case.config
    right_config = right_case.config
    left_launch = left_case.launch
    left_plan = left_case.plan
    if timed_batches != 1:
        batches = (("warmup",),) + tuple(
            (f"timed{index}",) for index in range(timed_batches)
        )
        left_plan = replace(left_plan, prompt_batches=batches)
        right_plan = replace(right_plan, prompt_batches=batches)
    if distinct_runtime_policies:
        left_runtime = replace(
            left_config.runtime,
            cpuset_cpus="0-7",
            cpuset_mems="0",
        )
        right_runtime = replace(
            right_config.runtime,
            cpuset_cpus="8-15",
            cpuset_mems="1",
        )
        left_config = replace(
            left_config,
            prebuild=replace(
                left_config.prebuild,
                policy=replace(
                    left_config.prebuild.policy,
                    runtime_policy_digest=left_runtime.digest,
                ),
            ),
            runtime=left_runtime,
        )
        right_config = replace(
            right_config,
            prebuild=replace(
                right_config.prebuild,
                policy=replace(
                    right_config.prebuild.policy,
                    runtime_policy_digest=right_runtime.digest,
                ),
            ),
            runtime=right_runtime,
        )
        left_launch = replace(
            left_launch,
            resource_policy_digest=(
                left_config.prebuild.policy.resource_policy_digest
            ),
        )
        left_plan = replace(
            left_plan,
            launch_digest=left_launch.digest,
            expected_preflight=replace(
                left_plan.expected_preflight,
                launch_digest=left_launch.digest,
            ),
        )
        right_launch = replace(
            right_launch,
            resource_policy_digest=(
                right_config.prebuild.policy.resource_policy_digest
            ),
        )
        right_plan = replace(
            right_plan,
            launch_digest=right_launch.digest,
            expected_preflight=replace(
                right_plan.expected_preflight,
                launch_digest=right_launch.digest,
            ),
        )
    baseline_executor = OCIEngineExecutor(
        left_config,
        left_case.device_policy,
        manager=_manager(left_case),
    )
    candidate_executor = OCIEngineExecutor(
        right_config,
        right_policy,
        manager=_manager(right_case),
    )
    baseline = ResidentArmPlan(
        left_launch,
        left_case.binding,
        left_plan,
        baseline_executor.manager.namespace_digest,
        baseline_executor.config.runtime.digest,
        baseline_executor.device_policy.configuration_sha256,
    )
    candidate = ResidentArmPlan(
        right_launch,
        right_binding,
        right_plan,
        candidate_executor.manager.namespace_digest,
        candidate_executor.config.runtime.digest,
        candidate_executor.device_policy.configuration_sha256,
    )
    trace: list[str] = []
    trace_lock = threading.Lock()
    active_count = [0]
    overlap = [False]
    _install_fake_execution(
        baseline_executor,
        lane="left",
        durations=baseline_durations,
        trace=trace,
        trace_lock=trace_lock,
        active_count=active_count,
        overlap=overlap,
        batches_per_read=1 + timed_batches,
    )
    _install_fake_execution(
        candidate_executor,
        lane="right",
        durations=candidate_durations,
        trace=trace,
        trace_lock=trace_lock,
        active_count=active_count,
        overlap=overlap,
        batches_per_read=1 + timed_batches,
        conditioning_duration=candidate_conditioning,
    )
    plan = ResidentCrossoverPlan(
        "7" * 64,
        baseline,
        candidate,
        policy
        if policy is not None
        else ResidentSpeedPolicy(60, 0.005, 2.0, 0.1, "8" * 64, "9" * 64),
    )
    return (
        plan,
        baseline_executor,
        candidate_executor,
        left_case.mount,
        trace,
        overlap,
    )


@pytest.mark.parametrize(
    ("candidate_durations", "expected_decision"),
    (
        ((1.02, 1.02), SpeedStageDecision.FAIL),
        ((0.90, 0.90), SpeedStageDecision.PASS),
    ),
)
def test_clear_result_stops_after_three_serialized_reads(
    tmp_path: Path,
    candidate_durations: tuple[float, ...],
    expected_decision: SpeedStageDecision,
) -> None:
    plan, baseline, candidate, mount, trace, overlap = _rig(
        tmp_path, candidate_durations
    )
    result = _speed(plan, baseline, candidate, mount)

    assert result.decision is expected_decision
    assert not result.escalated
    assert tuple(row.role for row in result.rates) == ("B", "C", "B_prime")
    assert trace[:6] == [
        "left:0",
        "left:1",
        "right:0",
        "right:1",
        "left:2",
        "left:3",
    ]
    assert trace.index("right:close") > trace.index("left:3")
    assert not overlap[0]
    assert len(result.baseline_execution.session.batches) == 4
    assert len(result.candidate_execution.session.batches) == 2
    assert result.regrade(plan) == result.final_verdict
    assert result.digest


def test_borderline_result_adds_only_candidate_and_baseline_repeat(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, trace, overlap = _rig(
        tmp_path, (0.993, 0.993)
    )
    result = _speed(plan, baseline, candidate, mount)

    assert result.escalated
    assert result.decision is SpeedStageDecision.PASS
    assert tuple(row.role for row in result.rates) == (
        "B",
        "C",
        "B_prime",
        "C_prime",
        "B_double_prime",
    )
    assert trace.index("right:2") < trace.index("left:4")
    assert trace.index("right:close") > trace.index("left:5")
    assert not overlap[0]
    assert len(result.baseline_execution.session.batches) == 6
    assert len(result.candidate_execution.session.batches) == 4


def test_plan_rejects_overlapping_physical_lanes(tmp_path: Path) -> None:
    case = _case(tmp_path)
    executor = OCIEngineExecutor(
        case.config,
        case.device_policy,
        manager=_manager(case),
    )
    arm = ResidentArmPlan(
        case.launch,
        case.binding,
        case.plan,
        executor.manager.namespace_digest,
        executor.config.runtime.digest,
        executor.device_policy.configuration_sha256,
    )
    with pytest.raises(CrossoverRuntimeError, match="overlap"):
        ResidentCrossoverPlan(
            "7" * 64,
            arm,
            arm,
            ResidentSpeedPolicy(60, 0.005, 2.0, 0.1, "8" * 64, "9" * 64),
        )


def test_retained_rate_span_is_independently_regraded(tmp_path: Path) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.90, 0.90)
    )
    result = _speed(plan, baseline, candidate, mount)
    first = result.rates[0]
    changed_seconds = first.timed_seconds * 2
    changed_charged_seconds = first.conditioning_seconds + changed_seconds
    tampered = replace(
        result,
        rates=(
            replace(
                first,
                timed_seconds=changed_seconds,
                charged_seconds=changed_charged_seconds,
                tokens_per_second=first.charged_tokens / changed_charged_seconds,
            ),
            *result.rates[1:],
        ),
    )

    with pytest.raises(CrossoverRuntimeError, match="independently regrade"):
        tampered.regrade(plan)


@pytest.mark.parametrize("candidate_durations", ((0.90, 0.90), (1.02, 1.02)))
def test_resident_speed_witness_round_trips_pass_or_fail_raw_stage(
    tmp_path: Path, candidate_durations: tuple[float, ...]
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, candidate_durations
    )
    result = _speed(plan, baseline, candidate, mount)

    witness = ResidentSpeedWitness.from_evidence(result, plan)
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness
    assert witness.policy.version == 3
    assert witness.started_monotonic_s == result.started_monotonic_s
    assert witness.completed_monotonic_s == result.completed_monotonic_s
    assert witness.resident_policy.max_qualification_seconds == 7_200

    if result.decision is not SpeedStageDecision.PASS:
        decision = (
            QualificationDecision.NO_DECISION
            if result.decision is SpeedStageDecision.NO_DECISION
            else QualificationDecision.FAIL
        )
        stage = QualificationStageExit(
            "a" * 64,
            "b" * 64,
            plan.selected_delta_digest,
            "speed",
            decision,
            "speed_noise"
            if decision is QualificationDecision.NO_DECISION
            else "speed_regression",
            witness,
            None,
            None,
            None,
            None,
        )
        assert QualificationStageExit.from_dict(stage.to_dict()) == stage

    tampered = witness.to_dict()
    tampered["rates"][0]["lane_digest"] = plan.candidate_lane_digest
    with pytest.raises(QualificationRunnerError, match="digest"):
        ResidentSpeedWitness.from_dict(tampered)


def test_resident_speed_witness_binds_distinct_numa_lane_policies(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90, 0.90),
        distinct_runtime_policies=True,
    )
    assert baseline.config.runtime.cpuset_cpus == "0-7"
    assert baseline.config.runtime.cpuset_mems == "0"
    assert candidate.config.runtime.cpuset_cpus == "8-15"
    assert candidate.config.runtime.cpuset_mems == "1"
    assert (
        plan.baseline.runtime_resource_policy_digest
        != plan.candidate.runtime_resource_policy_digest
    )
    assert (
        plan.baseline.launch.resource_policy_digest
        != plan.candidate.launch.resource_policy_digest
    )

    result = _speed(plan, baseline, candidate, mount)
    witness = ResidentSpeedWitness.from_evidence(result, plan)

    assert witness.baseline_runtime_resource_policy_digest == (
        plan.baseline.runtime_resource_policy_digest
    )
    assert witness.candidate_runtime_resource_policy_digest == (
        plan.candidate.runtime_resource_policy_digest
    )
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness

    tampered = witness.to_dict()
    tampered["baseline_runtime_resource_policy_digest"] = (
        witness.candidate_runtime_resource_policy_digest
    )
    with pytest.raises(QualificationRunnerError, match="digest"):
        ResidentSpeedWitness.from_dict(tampered)


def test_resident_settlement_control_accepts_exact_lane_policy_swap(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90, 0.90),
        distinct_runtime_policies=True,
    )
    result = _speed(plan, baseline, candidate, mount)
    primary_witness = ResidentSpeedWitness.from_evidence(result, plan)
    reproduction_policy = replace(
        primary_witness.resident_policy,
        calibration_digest="2" * 64,
        calibration_context_digest="3" * 64,
    )
    swapped_rates = tuple(
        replace(
            row,
            lane_digest=(
                primary_witness.candidate_lane_digest
                if row.role.startswith("B")
                else primary_witness.baseline_lane_digest
            ),
        )
        for row in primary_witness.rates
    )
    swapped_fields = {
        "selected_delta_digest": primary_witness.selected_delta_digest,
        "candidate_launch_digest": primary_witness.candidate_launch_digest,
        "calibration_digest": reproduction_policy.calibration_digest,
        "calibration_context_digest": (
            reproduction_policy.calibration_context_digest
        ),
        "workload_digest": primary_witness.workload_digest,
        "baseline_runtime_resource_policy_digest": (
            primary_witness.candidate_runtime_resource_policy_digest
        ),
        "candidate_runtime_resource_policy_digest": (
            primary_witness.baseline_runtime_resource_policy_digest
        ),
        "plan_digest": "d" * 64,
        "baseline_lane_digest": primary_witness.candidate_lane_digest,
        "candidate_lane_digest": primary_witness.baseline_lane_digest,
        "baseline_quiescence_digest": "e" * 64,
        "candidate_quiescence_digest": "f" * 64,
        "raw_crossover_digest": "1" * 64,
        "resident_policy": reproduction_policy,
        "rates": swapped_rates,
        "started_monotonic_s": primary_witness.started_monotonic_s,
        "completed_monotonic_s": primary_witness.completed_monotonic_s,
    }
    reproduction_witness = ResidentSpeedWitness(
        **swapped_fields,
        evidence_digest=_resident_speed_projection_digest(**swapped_fields),
    )

    primary = ResidentLaneOrientation.from_resident_speed_witness(
        primary_witness
    )
    reproduction = ResidentLaneOrientation.from_resident_speed_witness(
        reproduction_witness
    )
    assert reproduction.control_digest == primary.control_digest
    assert reproduction.is_exact_swap_of(primary)


def test_resident_policy_binds_total_qualification_budget() -> None:
    policy = ResidentSpeedPolicy(
        600,
        0.005,
        2.0,
        0.1,
        "8" * 64,
        "9" * 64,
        max_qualification_seconds=1_800,
    )
    assert ResidentSpeedPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(CrossoverRuntimeError, match="unsupported"):
        replace(policy, max_qualification_seconds=599)


def test_speed_verdict_v2_regrades_the_sealed_b300_stage_exit_both_ways() -> None:
    """The 2026-07-24 B300 joined-primary stage-exit — the first production v3
    speed verdict ever issued — pinned both ways. Version-1 charged-basis
    arithmetic must reproduce the shipped FAIL exactly; version-2 timed-basis
    arithmetic must grade the same sealed reads as a clear PASS. The warm/cold
    conditioning split between B (fresh session, cold first read) and B_prime
    (warm continuation of B's session) is accounting structure, not noise, and
    only the charged basis scores it: it inflated baseline noise to 6.3% and
    the required bar to 1.126 against a candidate faster on every timed
    window."""

    fixture = Path(__file__).parent / "fixtures" / "speed_stage_exit_97eb1808.json"
    raw = fixture.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "97eb1808b908d0704bbddc350dacf8f1a051fb72f87c70a1ec9ba1cd61d13152"
    )
    exit_ = QualificationStageExit.from_dict(json.loads(raw))
    assert exit_.stage == "speed"
    assert exit_.decision is QualificationDecision.FAIL
    assert exit_.reason == "speed_regression"

    witness = exit_.speed_witness
    policy_v1 = witness.resident_policy
    assert policy_v1.version == 1
    baselines = [row for row in witness.rates if row.role.startswith("B")]
    candidates = [row for row in witness.rates if row.role.startswith("C")]
    assert [row.role for row in witness.rates] == ["B", "C", "B_prime"]

    shipped = score_speedup(
        [policy_v1.scored_tokens_per_second(row) for row in baselines[:2]],
        [policy_v1.scored_tokens_per_second(row) for row in candidates[:1]],
        min_margin=policy_v1.min_margin,
        k=policy_v1.noise_multiplier,
        max_noise=policy_v1.max_noise,
    )
    assert shipped.confident and not shipped.passed_speedup
    assert shipped.speedup == pytest.approx(0.992288407418, rel=1e-9)
    assert shipped.noise == pytest.approx(0.063093295300, rel=1e-9)
    assert shipped.required == pytest.approx(1.126186590600, rel=1e-9)
    # Clear FAIL (outside the min_margin band around required), so the shipped
    # 3-read shape with no C_prime/B_double_prime extension regrades.
    assert shipped.speedup <= shipped.required - policy_v1.min_margin

    policy_v2 = ResidentSpeedPolicy.from_dict(
        {**policy_v1.to_dict(), "version": 2, "max_noise": "0.02"}
    )
    regraded = score_speedup(
        [policy_v2.scored_tokens_per_second(row) for row in baselines[:2]],
        [policy_v2.scored_tokens_per_second(row) for row in candidates[:1]],
        min_margin=policy_v2.min_margin,
        k=policy_v2.noise_multiplier,
        max_noise=policy_v2.max_noise,
    )
    assert regraded.confident and regraded.passed_speedup
    assert regraded.speedup == pytest.approx(1.031639399357, rel=1e-9)
    assert regraded.noise == pytest.approx(0.006904721491, rel=1e-9)
    assert regraded.required == pytest.approx(1.013809442981, rel=1e-9)
    # Also clear: the verdict would not have needed the repeat-read extension.
    assert regraded.speedup >= regraded.required + policy_v2.min_margin

    # Flipping the version on the sealed policy without tightening the noise
    # ceiling is refused outright — a v2 policy cannot carry the v1 ceiling.
    with pytest.raises(CrossoverRuntimeError, match="max_noise <= 0.02"):
        ResidentSpeedPolicy.from_dict({**policy_v1.to_dict(), "version": 2})


def _policy_v3(**overrides) -> ResidentSpeedPolicy:
    kwargs = {
        "max_stage_seconds": 60,
        "min_margin": 0.005,
        "noise_multiplier": 2.0,
        "max_noise": 0.02,
        "calibration_digest": "8" * 64,
        "calibration_context_digest": "9" * 64,
        "version": 3,
        "min_windows": 3,
        "max_window_scatter": 0.01,
        "max_conditioning_slowdown": 1.25,
    }
    kwargs.update(overrides)
    return ResidentSpeedPolicy(**kwargs)


def test_policy_v3_serialization_is_version_dependent() -> None:
    policy = _policy_v3()
    row = policy.to_dict()
    assert row["min_windows"] == 3 and "max_window_scatter" in row
    assert ResidentSpeedPolicy.from_dict(row) == policy
    legacy = ResidentSpeedPolicy(60, 0.005, 2.0, 0.1, "8" * 64, "9" * 64)
    assert "min_windows" not in legacy.to_dict()
    # Window thresholds cannot ride a pre-v3 policy, and a v3 policy cannot
    # omit them: the field set is version-exact both ways.
    with pytest.raises(CrossoverRuntimeError, match="require resident speed policy v3"):
        replace(legacy, min_windows=3, max_window_scatter=0.01)
    with pytest.raises(CrossoverRuntimeError, match="3..512 timed windows"):
        replace(legacy, version=3, max_noise=0.02)
    with pytest.raises(CrossoverRuntimeError, match="fields differ"):
        ResidentSpeedPolicy.from_dict(
            {key: value for key, value in row.items() if key != "min_windows"}
        )
    with pytest.raises(CrossoverRuntimeError, match="fields differ"):
        ResidentSpeedPolicy.from_dict(
            {**legacy.to_dict(), "min_windows": 3, "max_window_scatter": "0.01"}
        )
    # The conditioning slowdown bound is required and range-checked under v3
    # and forbidden before it.
    with pytest.raises(CrossoverRuntimeError, match="conditioning slowdown"):
        _policy_v3(max_conditioning_slowdown=1.0)
    with pytest.raises(CrossoverRuntimeError, match="conditioning slowdown"):
        _policy_v3(max_conditioning_slowdown=2.5)
    with pytest.raises(CrossoverRuntimeError, match="require resident speed policy v3"):
        replace(legacy, max_conditioning_slowdown=1.25)


# --- version 8: the two-process substrate ---------------------------------
#
# This path serves candidates that cannot be hot-swapped into a resident
# engine: a CUDA, C++ or PTX kernel has to be compiled and linked into the
# engine that runs it, so it needs its own launched process rather than a swap
# into a live one. Between 2026-08-15 (87944430) and this change the path
# refused every version-6 policy outright, so those bundles screened clean and
# never received a speed verdict at all.
#
# Version 8 is B, C, B-prime -- always three, never more. B-prime is
# precommitted rather than earned by a close call because the quality gate
# harvests its stock-drift control from the second baseline read.


def _policy_v8(**overrides) -> ResidentSpeedPolicy:
    return replace(_policy_v3(**overrides), version=8)


@pytest.mark.parametrize(
    ("candidate_duration", "expected_decision"),
    (
        (1.02, SpeedStageDecision.FAIL),
        (0.90, SpeedStageDecision.PASS),
    ),
)
def test_v8_reads_exactly_b_c_and_the_bookend(
    tmp_path: Path,
    candidate_duration: float,
    expected_decision: SpeedStageDecision,
) -> None:
    plan, baseline, candidate, mount, trace, overlap = _rig(
        tmp_path,
        (candidate_duration,),
        policy=_policy_v8(),
        timed_batches=3,
    )
    result = _speed(plan, baseline, candidate, mount)

    assert result.decision is expected_decision
    assert not result.escalated
    assert tuple(row.role for row in result.rates) == ("B", "C", "B_prime")
    # Two baseline reads, one candidate read; four batches per read.
    assert len(result.baseline_execution.session.batches) == 8
    assert len(result.candidate_execution.session.batches) == 4
    assert not overlap[0]
    # The candidate lane stays resident until the baseline lane is finished; a
    # CUDA context torn down alongside a charging baseline read contaminates it.
    assert trace.index("right:close") > trace.index("left:7")
    assert result.regrade(plan) == result.final_verdict
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness
    # The witness must reach the live decision from the sealed reads alone.
    # This is the seam the two-process path was missing: v6_result asserts a
    # conditional read shape and would reject an unconditional bookend.
    assert witness.always_bookend_result()[0].value == expected_decision.value


def test_v8_reads_the_bookend_even_when_the_call_is_not_close(
    tmp_path: Path,
) -> None:
    # 2x is far outside any band a bookend could reverse. It is still read,
    # because the quality gate's stock-drift control comes from it: the read is
    # owed to the next stage, not to this one's uncertainty.
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.5,), policy=_policy_v8(), timed_batches=3
    )
    result = _speed(plan, baseline, candidate, mount)

    assert result.decision is SpeedStageDecision.PASS
    assert tuple(row.role for row in result.rates) == ("B", "C", "B_prime")
    assert result.final_verdict.speedup == pytest.approx(2.0, rel=1e-9)


def test_v8_bookend_can_convict_a_borderline_candidate(tmp_path: Path) -> None:
    # The baseline drifts 2% between B and B-prime, which raises the required
    # margin above what the candidate cleared against B alone. The bookend is
    # what decides -- the whole reason it is read for the speed verdict too.
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.99,),
        policy=_policy_v8(),
        timed_batches=3,
        baseline_durations=(1.0, 1.02, 1.0),
    )
    result = _speed(plan, baseline, candidate, mount)

    assert result.decision is SpeedStageDecision.FAIL
    assert tuple(row.role for row in result.rates) == ("B", "C", "B_prime")
    assert not result.escalated
    assert result.regrade(plan) == result.final_verdict


def test_v8_conditioning_regression_fails_a_fast_candidate(tmp_path: Path) -> None:
    # Decodes 10% faster, prefills at 2x the baseline: a real regression a
    # decode-only gate would crown.
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90,),
        policy=_policy_v8(),
        timed_batches=3,
        candidate_conditioning=0.2,
    )
    result = _speed(plan, baseline, candidate, mount)

    assert result.decision is SpeedStageDecision.FAIL
    assert result.final_verdict.passed_speedup  # the speed number alone passed
    assert result.regrade(plan) == result.final_verdict


def test_v8_settled_speedup_is_not_graded_by_v6_arithmetic(
    tmp_path: Path,
) -> None:
    """Settlement must dispatch on version, not call ``v6_result`` by hand.

    ``v6_result`` does not refuse a v8 witness up front -- its own guard is
    ``version < 6``. It fails deeper, inside ``v6_grade``, on v6's invariant
    that a *clear* decision never carries a third read. v8 precommits B-prime,
    so a clear v8 verdict is exactly the shape v6 calls impossible: settlement
    reaching for ``v6_result`` raises instead of settling. Native bundles are
    the only ones v8 ever grades, so this is the CUDA payout path.
    """

    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90,),
        policy=_policy_v8(),
        timed_batches=3,
    )
    result = _speed(plan, baseline, candidate, mount)
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    assert witness.resident_policy.version == 8

    # The settled number is the v8 one, and it is a real speedup.
    assert witness.accepted_speedup() == witness.always_bookend_result()[1]
    assert float(witness.accepted_speedup()) > 0.0

    # Reaching for v6 by hand does not settle this witness at all.
    with pytest.raises(QualificationRunnerError, match="clear decision added"):
        witness.v6_result()


def test_v8_evidence_cannot_claim_the_retired_five_arm_schedule(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.90,), policy=_policy_v8(), timed_batches=3
    )
    result = _speed(plan, baseline, candidate, mount)
    # C-prime and B-double-prime do not exist under v8. Sealed evidence that
    # claims the escalated schedule is malformed, not merely unusual.
    with pytest.raises(CrossoverRuntimeError, match="evidence is malformed"):
        replace(result, escalated=True, exit_reason="borderline_pass")


@pytest.mark.parametrize("version", (6, 7))
def test_conditional_bookend_policies_cannot_serve_this_substrate(
    tmp_path: Path, version: int
) -> None:
    # v6/v7 read the bookend only when the speed call is close, so a clear PASS
    # under them seals two reads and leaves the quality gate with no stock-drift
    # control to harvest. They belong to the pair-native crossover; refuse them
    # here rather than produce evidence the next stage cannot use.
    plan, baseline, candidate, mount, trace, _overlap = _rig(
        tmp_path,
        (0.9,),
        policy=replace(_policy_v3(), version=version),
        timed_batches=3,
    )
    with pytest.raises(CrossoverRuntimeError, match="conditional-bookend"):
        _speed(plan, baseline, candidate, mount)
    assert trace == []


def test_v3_crossover_scores_window_medians_and_retains_windows(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.90, 0.90), policy=_policy_v3(), timed_batches=3
    )
    result = _speed(plan, baseline, candidate, mount)

    assert result.decision is SpeedStageDecision.PASS
    assert not result.escalated
    for row in result.rates:
        assert [window.batch_index for window in row.windows] == [
            row.first_timed_batch_index,
            row.first_timed_batch_index + 1,
            row.first_timed_batch_index + 2,
        ]
        assert sum(window.tokens for window in row.windows) == row.timed_tokens
    # The scored value is the median over per-window rates: the wobbled
    # windows (d, 1.001d, 0.999d) make that exactly 1/d per read.
    scored_b = plan.policy.scored_tokens_per_second(result.rates[0])
    scored_c = plan.policy.scored_tokens_per_second(result.rates[1])
    assert scored_b == pytest.approx(1.0, rel=1e-12)
    assert scored_c == pytest.approx(1.0 / 0.90, rel=1e-12)
    assert result.final_verdict.speedup == pytest.approx(
        scored_c / scored_b, rel=1e-12
    )
    # Sealed windows regrade from raw spans, and the witness round-trips.
    assert result.regrade(plan) == result.final_verdict
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness


def test_v3_tampered_window_fails_independent_regrade(tmp_path: Path) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.90, 0.90), policy=_policy_v3(), timed_batches=3
    )
    result = _speed(plan, baseline, candidate, mount)
    first = result.rates[0]
    slowed = replace(first.windows[1], seconds=first.windows[1].seconds * 2)
    tampered = replace(
        result,
        rates=(
            replace(first, windows=(first.windows[0], slowed, first.windows[2])),
            *result.rates[1:],
        ),
    )
    with pytest.raises(CrossoverRuntimeError, match="independently regrade"):
        tampered.regrade(plan)


def test_v3_scatter_blowout_refuses_the_stage_not_the_candidate(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.90, 0.90), policy=_policy_v3(max_window_scatter=0.0005),
        timed_batches=3,
    )
    # The rig's +-0.1% window wobble exceeds a 0.05% sealed scatter bound:
    # the read refuses to produce a scored number, so the stage dies as
    # typed infrastructure before any verdict exists.
    with pytest.raises(CrossoverRuntimeError, match="window scatter exceeds"):
        _speed(plan, baseline, candidate, mount)


def test_v3_witness_refuses_window_retention_mismatch(tmp_path: Path) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.90, 0.90), policy=_policy_v3(), timed_batches=3
    )
    result = _speed(plan, baseline, candidate, mount)
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    stripped = witness.to_dict()
    stripped["rates"] = [
        {key: value for key, value in row.items() if key != "windows"}
        for row in stripped["rates"]
    ]
    with pytest.raises(QualificationRunnerError):
        ResidentSpeedWitness.from_dict(stripped)


def test_timed_window_rows_are_exact_and_tiled() -> None:
    with pytest.raises(CrossoverRuntimeError, match="malformed"):
        TimedWindow(1, 0, 1.0)
    with pytest.raises(CrossoverRuntimeError, match="malformed"):
        TimedWindow(1, 10, 0.0)
    window = TimedWindow(4, 10, 0.25)
    assert TimedWindow.from_dict(window.to_dict()) == window
    with pytest.raises(CrossoverRuntimeError, match="noncanonical"):
        TimedWindow.from_dict({**window.to_dict(), "seconds": "0.250"})


def test_v3_conditioning_regression_fails_a_fast_but_prefill_slow_candidate(
    tmp_path: Path,
) -> None:
    # Candidate decodes 10% faster but its conditioning (the only span where
    # prefill cost is host-visible) runs 2x the baseline's: a real kernel
    # regression a decode-only gate would crown. Costs zero extra wall-clock:
    # the graded numbers are already sealed in every read.
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90, 0.90),
        policy=_policy_v3(),
        timed_batches=3,
        candidate_conditioning=0.2,
    )
    result = _speed(plan, baseline, candidate, mount)
    assert result.decision is SpeedStageDecision.FAIL
    assert not result.escalated
    assert result.final_verdict.passed_speedup  # the speed number alone passed
    assert result.regrade(plan) == result.final_verdict
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness


def test_v3_conditioning_within_bound_does_not_disturb_the_verdict(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90, 0.90),
        policy=_policy_v3(),
        timed_batches=3,
        candidate_conditioning=0.12,  # 1.2x, inside the sealed 1.25 bound
    )
    result = _speed(plan, baseline, candidate, mount)
    assert result.decision is SpeedStageDecision.PASS
    assert result.regrade(plan) == result.final_verdict

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval.crossover_runtime import (
    CrossoverRuntimeError,
    ResidentArmPlan,
    ResidentCrossoverPlan,
    ResidentSpeedPolicy,
)
from cacheon.eval.engine_launch import PhysicalHardwareBinding
from cacheon.eval.goodput_runtime import GoodputReadSet
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.oci_session_protocol import AuditReceiptFacts, SlotAuditPolicy
from cacheon.eval.qualification_runner import (
    AuditWitness,
    QualificationStageExit,
    QualificationRunnerError,
    ResidentSpeedWitness,
    _resident_speed_projection_digest,
)
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.settlement import ResidentLaneOrientation
from tests.support.replay import GOODPUT, replay_plan, replay_session
from tests.test_oci_backend import _case, _manager
from tests.test_service_statistics import _read


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


def _rig(tmp_path: Path, *, distinct_runtime_policies: bool = False):
    """Two disjoint resident TP lanes replaying one sealed slice under policy 17."""

    left_case = _case(tmp_path / "left")
    right_case = _case(tmp_path / "right")
    right_policy, right_launch, right_binding, right_plan = _right_lane(right_case)
    replay = replay_plan(tmp_path / "replay")
    left_plan = replay_session(left_case.plan, replay)
    right_plan = replay_session(right_plan, replay)
    left_config = left_case.config
    right_config = right_case.config
    left_launch = left_case.launch
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
    plan = ResidentCrossoverPlan("7" * 64, baseline, candidate, _resident_policy())
    return plan, baseline_executor, candidate_executor


def _witness(plan: ResidentCrossoverPlan, **fields) -> ResidentSpeedWitness:
    """Project ``plan``'s sealed lanes into a replay speed witness; no engine runs.

    The witness codec binds its paired reads only through the evidence digest,
    so any complete paired read set stands in for the retained turn records.
    """

    policy = fields.pop("resident_policy", plan.policy)
    values = {
        "selected_delta_digest": plan.selected_delta_digest,
        "candidate_launch_digest": plan.candidate.launch.digest,
        "calibration_digest": policy.calibration_digest,
        "calibration_context_digest": policy.calibration_context_digest,
        "workload_digest": marginal_workload_digest(plan.baseline.session_plan),
        "baseline_runtime_resource_policy_digest": (
            plan.baseline.runtime_resource_policy_digest
        ),
        "candidate_runtime_resource_policy_digest": (
            plan.candidate.runtime_resource_policy_digest
        ),
        "plan_digest": plan.digest,
        "baseline_lane_digest": plan.baseline_lane_digest,
        "candidate_lane_digest": plan.candidate_lane_digest,
        "baseline_quiescence_digest": "e" * 64,
        "candidate_quiescence_digest": "f" * 64,
        "raw_crossover_digest": "1" * 64,
        "resident_policy": policy,
        "rates": (),
        "started_monotonic_s": 10.0,
        "completed_monotonic_s": 40.0,
        "goodput": GoodputReadSet(
            (_read("incumbent", "A", 1, 1.0),),
            (_read("candidate", "B", 1, 0.9),),
            (("session", 2, 0),),
        ),
        **fields,
    }
    return ResidentSpeedWitness(
        **values, evidence_digest=_resident_speed_projection_digest(**values)
    )


def test_plan_rejects_overlapping_physical_lanes(tmp_path: Path) -> None:
    plan, *_ = _rig(tmp_path)
    with pytest.raises(CrossoverRuntimeError, match="overlap"):
        ResidentCrossoverPlan("7" * 64, plan.baseline, plan.baseline, plan.policy)


@pytest.mark.parametrize(
    ("decision", "reason"),
    (
        (QualificationDecision.FAIL, "speed_regression"),
        (QualificationDecision.NO_DECISION, "speed_noise"),
    ),
)
def test_resident_speed_witness_round_trips_through_its_stage_exit(
    tmp_path: Path, decision: QualificationDecision, reason: str
) -> None:
    plan, *_ = _rig(tmp_path)
    witness = _witness(plan)
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness
    assert witness.policy.version == 3
    assert witness.resident_policy.max_qualification_seconds == 7_200

    stage = QualificationStageExit(
        "a" * 64,
        "b" * 64,
        plan.selected_delta_digest,
        "speed",
        decision,
        reason,
        witness,
        None,
        None,
        None,
        None,
    )
    assert QualificationStageExit.from_dict(stage.to_dict()) == stage

    tampered = witness.to_dict()
    tampered["baseline_lane_digest"] = plan.candidate_lane_digest
    with pytest.raises(QualificationRunnerError, match="digest"):
        ResidentSpeedWitness.from_dict(tampered)


def test_calibration_observation_speed_exit_carries_only_a_later_passing_audit(
    tmp_path: Path,
) -> None:
    from cacheon.audit import gate

    plan, *_ = _rig(tmp_path)
    witness = _witness(plan)
    policy = SlotAuditPolicy("a" * 32, 250_000, 1, ("norm.rmsnorm",), 1)

    def audit(violations: int) -> AuditWitness:
        receipt = AuditReceiptFacts(
            "norm.rmsnorm", 32, violations, 0, 0, 1.0 - violations, 0.995, "allclose", 901, 0, 1
        )
        decision, detail = gate(
            [receipt.to_gate_dict()], min_calls=1, expected_slots=("norm.rmsnorm",),
            expected_member_count=1,
        )
        return AuditWitness(
            plan.selected_delta_digest, "c" * 64, "d" * 64, "1" * 32,
            plan.candidate.runtime_resource_policy_digest, policy, (receipt,),
            QualificationDecision(decision), detail,
        )

    def speed_exit(audit_witness: AuditWitness, started: float = 41.0) -> QualificationStageExit:
        return QualificationStageExit(
            "a" * 64, "b" * 64, plan.selected_delta_digest, "speed",
            QualificationDecision.FAIL, "speed_threshold_not_met", witness,
            audit_witness, started, 50.0, "9" * 64,
        )

    observed = speed_exit(audit(0))
    assert QualificationStageExit.from_dict(observed.to_dict()) == observed
    # A failing audit exits at its own stage, and the audit must follow the speed read.
    for violations, started in ((1, 41.0), (0, 39.0)):
        with pytest.raises(QualificationRunnerError, match="audit stage-exit timing"):
            speed_exit(audit(violations), started)


def test_resident_speed_witness_binds_distinct_numa_lane_policies(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate = _rig(tmp_path, distinct_runtime_policies=True)
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

    witness = _witness(plan)
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
    plan, *_ = _rig(tmp_path, distinct_runtime_policies=True)
    primary_witness = _witness(plan)
    reproduction_witness = _witness(
        plan,
        resident_policy=replace(
            plan.policy,
            calibration_digest="2" * 64,
            calibration_context_digest="3" * 64,
        ),
        baseline_runtime_resource_policy_digest=(
            plan.candidate.runtime_resource_policy_digest
        ),
        candidate_runtime_resource_policy_digest=(
            plan.baseline.runtime_resource_policy_digest
        ),
        plan_digest="d" * 64,
        baseline_lane_digest=plan.candidate_lane_digest,
        candidate_lane_digest=plan.baseline_lane_digest,
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
    policy = _resident_policy(max_stage_seconds=600, max_qualification_seconds=1_800)
    assert ResidentSpeedPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(CrossoverRuntimeError, match="unsupported"):
        replace(policy, max_qualification_seconds=599)


def _resident_policy(**overrides) -> ResidentSpeedPolicy:
    kwargs = {
        "max_stage_seconds": 60,
        "min_margin": 0.0,
        "noise_multiplier": 2.0,
        "max_noise": 0.02,
        "calibration_digest": "8" * 64,
        "calibration_context_digest": "9" * 64,
        "version": 17,
        "goodput": GOODPUT,
    }
    kwargs.update(overrides)
    return ResidentSpeedPolicy(**kwargs)


def test_batch_cell_policy_fields_stay_on_the_wire_at_zero() -> None:
    policy = _resident_policy()
    row = policy.to_dict()
    assert (row["min_windows"], row["max_window_scatter"]) == (0, "0")
    assert row["max_conditioning_slowdown"] == "0"
    with pytest.raises(CrossoverRuntimeError, match="fields differ"):
        ResidentSpeedPolicy.from_dict(
            {key: value for key, value in row.items() if key != "min_windows"}
        )
    for field in ("min_windows", "max_window_scatter", "max_conditioning_slowdown", "prefill_min_margin",
                  "prefill_credit_weight"):
        with pytest.raises(CrossoverRuntimeError, match="frozen calibration"):
            replace(policy, **{field: 1})
    # Owner ruling 2026-10-01: 90% at the final look; 0.01 is what earlier stages sealed.
    assert replace(policy, goodput=replace(GOODPUT, error_rate=0.125)).goodput.error_rate == 0.125
    with pytest.raises(CrossoverRuntimeError, match="frozen calibration"):
        replace(policy, goodput=replace(GOODPUT, error_rate=0.05))


@pytest.mark.parametrize("version", range(1, 16))
def test_retired_policy_versions_cannot_be_constructed_or_decoded(
    version: int,
) -> None:
    # The adaptive five-read schedule (v1-v5), the conditional bookend (v6/v7)
    # and the batch-cell B/C/B-prime schedules (v8-v15) are retired; only the
    # session-replay policies 16 and 17 remain. A sealed policy outside them is
    # refused before any read rather than measured under a schedule this tree
    # no longer runs.
    policy = _resident_policy()
    with pytest.raises(CrossoverRuntimeError, match="unsupported"):
        replace(policy, version=version)
    with pytest.raises(CrossoverRuntimeError, match="unsupported"):
        ResidentSpeedPolicy.from_dict({**policy.to_dict(), "version": version})

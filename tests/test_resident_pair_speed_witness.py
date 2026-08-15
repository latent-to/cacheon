from dataclasses import replace

import pytest

from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_runner import ResidentSpeedWitness
from cacheon.eval.resident_pair_crossover import run_resident_pair_crossover
from cacheon.eval.resident_pair_speed_witness import (
    ResidentPairLiveSpeedWitness,
    ResidentPairSpeedWitnessError,
    project_resident_pair_live_speed_witness,
    project_resident_pair_speed_witness,
)
from cacheon.eval.speed_verdict import v6_decision_limits
from tests.test_resident_pair_crossover import (
    _borderline_policy,
    _setup,
    cleanup_pairs as _cleanup_pairs,
)
from tests.test_standing_controls import _authority


resident_pairs = _cleanup_pairs


def _quiescence(plan, completed):
    return tuple(
        (
            lane.lane_id,
            OCIQuiescenceReceipt(
                "cacheon.oci-quiescence.v1",
                f"executor-{lane.lane_id.lower()}",
                ("a" if lane.lane_id == "A" else "b") * 32,
                lane.executor_namespace_digest,
                1,
                float(completed + 1.0),
                (),
                (),
                (),
            ),
        )
        for lane in plan.pair_binding.lanes
    )


@pytest.mark.parametrize(
    ("baseline_lane", "candidate_lane"), (("A", "B"), ("B", "A"))
)
def test_pair_speed_projects_one_report_witness_for_both_orientations(
    tmp_path, resident_pairs, baseline_lane, candidate_lane
):
    plan, pair, clock, *_ = _setup(
        tmp_path,
        resident_pairs,
        baseline_pair_lane=baseline_lane,
    )
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    receipts = _quiescence(plan, evidence.completed_monotonic_s)

    witness = project_resident_pair_speed_witness(
        evidence, plan=plan, lane_quiescence=receipts
    )

    assert witness.plan_digest == plan.crossover_plan.digest
    assert witness.raw_crossover_digest == evidence.digest
    assert witness.rates == evidence.rates
    assert witness.candidate_launch_digest == plan.crossover_plan.candidate.launch.digest
    by_lane = dict(receipts)
    assert witness.baseline_quiescence_digest == by_lane[baseline_lane].digest
    assert witness.candidate_quiescence_digest == by_lane[candidate_lane].digest


def test_pair_speed_projection_rejects_foreign_or_premature_quiescence(
    tmp_path, resident_pairs
):
    plan, pair, clock, *_ = _setup(tmp_path, resident_pairs)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    receipts = _quiescence(plan, evidence.completed_monotonic_s)
    first_lane, first = receipts[0]

    foreign = replace(first, namespace_digest="f" * 64)
    with pytest.raises(ResidentPairSpeedWitnessError, match="quiescence"):
        project_resident_pair_speed_witness(
            evidence,
            plan=plan,
            lane_quiescence=((first_lane, foreign), receipts[1]),
        )

    premature = replace(
        first, observed_monotonic_s=evidence.completed_monotonic_s - 0.01
    )
    with pytest.raises(ResidentPairSpeedWitnessError, match="precedes"):
        project_resident_pair_speed_witness(
            evidence,
            plan=plan,
            lane_quiescence=((first_lane, premature), receipts[1]),
        )


def test_pair_speed_projection_rejects_noncanonical_lane_order(
    tmp_path, resident_pairs
):
    plan, pair, clock, *_ = _setup(tmp_path, resident_pairs)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    receipts = _quiescence(plan, evidence.completed_monotonic_s)
    with pytest.raises(ResidentPairSpeedWitnessError, match="authorities"):
        project_resident_pair_speed_witness(
            evidence, plan=plan, lane_quiescence=tuple(reversed(receipts))
        )


@pytest.mark.parametrize(
    ("case", "roles", "decision"),
    (
        ("clear_fail", ("B", "C"), QualificationDecision.FAIL),
        ("clear_pass", ("B", "C"), QualificationDecision.PASS),
        ("bookended_fail", ("B", "C", "B_prime"), QualificationDecision.FAIL),
        ("bookended_pass", ("B", "C", "B_prime"), QualificationDecision.PASS),
    ),
)
def test_v6_terminal_witness_round_trips_and_imports(
    tmp_path, resident_pairs, case, roles, decision
):
    plan, pair, clock, _activity, stock, candidate = _setup(
        tmp_path,
        resident_pairs,
        baseline=((1.0,) * 3,) * 2,
        candidate=((1.0,) * 3,),
        policy=_borderline_policy(version=6),
        timed_batches=3,
    )
    context, calibration, policy = _authority(
        6, workload_digest=plan.workload_digest
    )
    fail_below, pass_at = v6_decision_limits(policy)
    product_bar = 1.0 + policy.min_margin
    ambiguous = (product_bar + pass_at) / 2.0
    ratio = {
        "clear_fail": fail_below * 0.99,
        "clear_pass": pass_at * 1.01,
        "bookended_fail": ambiguous,
        "bookended_pass": ambiguous,
    }[case]
    later_ratio = (
        (2.0 + policy.max_noise)
        / (2.0 - policy.max_noise)
        * (1.0 - 1e-9)
        if case == "bookended_fail"
        else 1.0
    )
    stock.sessions[0].durations = (
        (1.0,) * 3,
        (1.0 / later_ratio,) * 3,
    )
    candidate.sessions[0].durations = (1.0 / ratio,)
    plan = replace(plan, crossover_plan=replace(plan.crossover_plan, policy=policy))
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    if decision is QualificationDecision.FAIL:
        witness = project_resident_pair_live_speed_witness(evidence, plan=plan)
        reopened = ResidentPairLiveSpeedWitness.from_dict(witness.to_dict())
    else:
        witness = project_resident_pair_speed_witness(
            evidence,
            plan=plan,
            lane_quiescence=_quiescence(plan, evidence.completed_monotonic_s),
        )
        reopened = ResidentSpeedWitness.from_dict(witness.to_dict())
    assert tuple(row.role for row in witness.rates) == roles
    assert reopened == witness
    assert reopened.regrade(calibration, context)[0] is decision

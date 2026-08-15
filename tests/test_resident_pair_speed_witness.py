from dataclasses import replace

import pytest

from cacheon.eval.crossover_runtime import SpeedStageDecision
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.resident_pair_crossover import run_resident_pair_crossover
from cacheon.eval.resident_pair_speed_witness import (
    ResidentPairSpeedWitnessError,
    project_resident_pair_live_speed_witness,
    project_resident_pair_speed_witness,
)
from tests.test_resident_pair_crossover import (
    _borderline_policy,
    _setup,
    cleanup_pairs as _cleanup_pairs,
)


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


def test_live_pair_witness_grades_v6_like_the_retired_projection(
    tmp_path, resident_pairs
):
    """A terminal speed FAIL on a retained pair regrades under v6.

    Exact production shape that crashed mainnet request ``de4bceb0…`` on
    2026-08-15: the live projection exists only for a stock-restored speed
    failure, ``ResidentSpeedWitness.regrade`` dispatches to
    ``self.v6_result()`` for policy v6+, and the live pair projection lacked
    that method — so every v6 speed FAIL in pair mode crashed the worker.
    """

    plan, pair, clock, *_ = _setup(
        tmp_path,
        resident_pairs,
        baseline=((1.0,) * 3,) * 2,
        candidate=((1.05,) * 3,),
        policy=_borderline_policy(version=6),
        timed_batches=3,
    )
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    live = project_resident_pair_live_speed_witness(evidence, plan=plan)
    retired = project_resident_pair_speed_witness(
        evidence,
        plan=plan,
        lane_quiescence=_quiescence(plan, evidence.completed_monotonic_s),
    )

    assert evidence.decision is SpeedStageDecision.FAIL
    assert live.resident_policy.version >= 6
    assert live.v6_result() == retired.v6_result()
    assert live.v6_result()[0].value == evidence.decision.value

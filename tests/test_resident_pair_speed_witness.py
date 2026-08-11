from dataclasses import replace

import pytest

from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.resident_pair_crossover import run_resident_pair_crossover
from cacheon.eval.resident_pair_speed_witness import (
    ResidentPairLiveSpeedWitness,
    ResidentPairSpeedWitnessError,
    project_resident_pair_live_speed_witness,
    project_resident_pair_speed_witness,
)
from tests.test_resident_pair_crossover import _setup, cleanup_pairs as _cleanup_pairs


resident_pairs = _cleanup_pairs


def test_live_speed_failure_reopens_without_retiring_loaded_pair(
    tmp_path, resident_pairs
):
    plan, pair, clock, _, factory_a, factory_b = _setup(
        tmp_path,
        resident_pairs,
        baseline=(1.0, 1.0),
        candidate=(1.25,),
    )
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    witness = project_resident_pair_live_speed_witness(evidence, plan=plan)
    reopened = ResidentPairLiveSpeedWitness.from_dict(witness.to_dict())

    assert reopened == witness
    assert witness.pair_session_ids == tuple(
        row.session_id for row in plan.pair_binding.lanes
    )
    assert len(witness.stock_return_digests) == len(evidence.request_slices)
    assert all(
        row.ending_bundle_digest is None and not row.ending_slots
        for row in evidence.request_slices
    )
    assert factory_a.sessions[0].finish_calls == 0
    assert factory_b.sessions[0].finish_calls == 0


def test_live_projection_refuses_a_speed_pass(tmp_path, resident_pairs):
    plan, pair, clock, *_ = _setup(tmp_path, resident_pairs)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    with pytest.raises(ResidentPairSpeedWitnessError, match="speed failure"):
        project_resident_pair_live_speed_witness(evidence, plan=plan)


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

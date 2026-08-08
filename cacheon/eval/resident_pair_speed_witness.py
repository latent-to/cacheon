"""Project pair-native speed evidence into the qualification speed witness.

The resident pair launches stock on both physical lanes and activates a candidate
through an authenticated swap.  The existing qualification report still names
the candidate's prepared semantic launch.  This bridge binds both truths without
claiming that the candidate tree created the resident process: raw pair evidence
proves the sessions and swaps, while post-close quiescence proves both physical
executor namespaces retired before downstream audit and pristine-T work.
"""

from __future__ import annotations

from cacheon.eval.crossover_runtime import CrossoverRuntimeError
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification_runner import (
    QualificationRunnerError,
    ResidentSpeedWitness,
    _resident_speed_projection_digest,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverError,
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
)


class ResidentPairSpeedWitnessError(RuntimeError):
    """Pair speed or retirement evidence cannot support a report witness."""


def project_resident_pair_speed_witness(
    evidence: ResidentPairCrossoverEvidence,
    *,
    plan: ResidentPairCrossoverPlan,
    lane_quiescence: tuple[
        tuple[str, OCIQuiescenceReceipt], tuple[str, OCIQuiescenceReceipt]
    ],
) -> ResidentSpeedWitness:
    """Regrade one pair speed product and bind canonical A/B retirement.

    ``lane_quiescence`` is deliberately canonical rather than positional by
    speed role.  Primary and reproduction may swap physical baseline/candidate
    roles, so the plan performs that mapping after both exact namespaces have
    independently proven empty.
    """

    if (
        type(evidence) is not ResidentPairCrossoverEvidence
        or type(plan) is not ResidentPairCrossoverPlan
        or type(lane_quiescence) is not tuple
        or len(lane_quiescence) != 2
        or tuple(row[0] for row in lane_quiescence) != ("A", "B")
        or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[1]) is not OCIQuiescenceReceipt
            for row in lane_quiescence
        )
    ):
        raise ResidentPairSpeedWitnessError(
            "resident pair speed witness authorities are not exact"
        )
    try:
        evidence.regrade(plan)
    except (ResidentPairCrossoverError, CrossoverRuntimeError) as exc:
        raise ResidentPairSpeedWitnessError(
            f"resident pair speed evidence does not regrade: {exc}"
        ) from None

    receipts = {lane_id: receipt for lane_id, receipt in lane_quiescence}
    for binding in plan.pair_binding.lanes:
        receipt = receipts[binding.lane_id]
        if (
            receipt.namespace_digest != binding.executor_namespace_digest
            or receipt.observed_monotonic_s < evidence.completed_monotonic_s
        ):
            raise ResidentPairSpeedWitnessError(
                "resident pair quiescence differs from its lane or precedes speed"
            )

    crossover = plan.crossover_plan
    baseline_receipt = receipts[plan.baseline_pair_lane]
    candidate_receipt = receipts[plan.candidate_pair_lane]
    kwargs = {
        "selected_delta_digest": evidence.selected_delta_digest,
        "candidate_launch_digest": crossover.candidate.launch.digest,
        "calibration_digest": evidence.policy.calibration_digest,
        "calibration_context_digest": evidence.policy.calibration_context_digest,
        "workload_digest": evidence.workload_digest,
        "baseline_runtime_resource_policy_digest": (
            crossover.baseline.runtime_resource_policy_digest
        ),
        "candidate_runtime_resource_policy_digest": (
            crossover.candidate.runtime_resource_policy_digest
        ),
        "plan_digest": crossover.digest,
        "baseline_lane_digest": crossover.baseline_lane_digest,
        "candidate_lane_digest": crossover.candidate_lane_digest,
        "baseline_quiescence_digest": baseline_receipt.digest,
        "candidate_quiescence_digest": candidate_receipt.digest,
        "raw_crossover_digest": evidence.digest,
        "resident_policy": evidence.policy,
        "rates": evidence.rates,
        "started_monotonic_s": float(evidence.started_monotonic_s),
        "completed_monotonic_s": float(evidence.completed_monotonic_s),
    }
    try:
        return ResidentSpeedWitness(
            **kwargs,
            evidence_digest=_resident_speed_projection_digest(**kwargs),
        )
    except (QualificationRunnerError, TypeError, ValueError) as exc:
        raise ResidentPairSpeedWitnessError(
            f"resident pair speed witness projection failed: {exc}"
        ) from None


__all__ = [
    "ResidentPairSpeedWitnessError",
    "project_resident_pair_speed_witness",
]

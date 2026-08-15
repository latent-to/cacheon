"""Project pair-native speed evidence into the qualification speed witness.

The resident pair launches stock on both physical lanes and activates a candidate
through an authenticated swap.  The existing qualification report still names
the candidate's prepared semantic launch.  This bridge binds both truths without
claiming that the candidate tree created the resident process: raw pair evidence
proves the sessions and swaps, while post-close quiescence proves both physical
executor namespaces retired before downstream audit and pristine-T work.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from cacheon.eval.crossover_runtime import CrossoverRuntimeError, SpeedStageDecision
from cacheon.eval.continuation_codec import ContinuationCodec, ContinuationCodecError
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification_runner import (
    QualificationRunnerError,
    ResidentSpeedWitness,
    SpeedEvidencePolicy,
    _resident_speed_projection_digest,
)
from cacheon.eval.speed_verdict import resident_speed_roles
from cacheon.eval.resident_evaluation_pair import ResidentRequestSlice
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverError,
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex


class ResidentPairSpeedWitnessError(RuntimeError):
    """Pair speed or retirement evidence cannot support a report witness."""


_SLICE_CODEC = ContinuationCodec((ResidentRequestSlice,))


def _slice_digest(value: ResidentRequestSlice) -> str:
    try:
        encoded = _SLICE_CODEC.encode(value)
    except ContinuationCodecError as exc:
        raise ResidentPairSpeedWitnessError(str(exc)) from None
    return canonical_digest("cacheon.eval.resident-request-stock-return.v1", encoded)


def _live_digest(**values: object) -> str:
    return canonical_digest(
        "cacheon.qualification.resident-live-speed-witness.v1",
        {
            "baseline_lane": values["baseline_lane_digest"],
            "baseline_runtime_resource_policy": values[
                "baseline_runtime_resource_policy_digest"
            ],
            "calibration": values["calibration_digest"],
            "calibration_context": values["calibration_context_digest"],
            "candidate_lane": values["candidate_lane_digest"],
            "candidate_launch": values["candidate_launch_digest"],
            "candidate_runtime_resource_policy": values[
                "candidate_runtime_resource_policy_digest"
            ],
            "completed_monotonic_s": format(
                values["completed_monotonic_s"], ".17g"
            ),
            "pair_binding": values["pair_binding_digest"],
            "pair_sessions": list(values["pair_session_ids"]),
            "plan": values["plan_digest"],
            "policy": values["resident_policy"].digest,
            "rates": [row.to_dict() for row in values["rates"]],
            "raw_crossover": values["raw_crossover_digest"],
            "selected_delta": values["selected_delta_digest"],
            "started_monotonic_s": format(values["started_monotonic_s"], ".17g"),
            "stock_returns": list(values["stock_return_digests"]),
            "workload": values["workload_digest"],
        },
    )


@dataclass(frozen=True)
class ResidentPairLiveSpeedWitness:
    """Speed-failure witness whose every request slice ended back on stock."""

    selected_delta_digest: str
    candidate_launch_digest: str
    calibration_digest: str
    calibration_context_digest: str
    workload_digest: str
    baseline_runtime_resource_policy_digest: str
    candidate_runtime_resource_policy_digest: str
    plan_digest: str
    baseline_lane_digest: str
    candidate_lane_digest: str
    pair_binding_digest: str
    pair_session_ids: tuple[str, str]
    stock_return_digests: tuple[str, ...]
    raw_crossover_digest: str
    resident_policy: object
    rates: tuple[object, ...]
    started_monotonic_s: float
    completed_monotonic_s: float
    evidence_digest: str

    def __post_init__(self) -> None:
        from cacheon.eval.crossover_runtime import ResidentReadRate, ResidentSpeedPolicy

        for name in (
            "selected_delta_digest",
            "candidate_launch_digest",
            "calibration_digest",
            "calibration_context_digest",
            "workload_digest",
            "baseline_runtime_resource_policy_digest",
            "candidate_runtime_resource_policy_digest",
            "plan_digest",
            "baseline_lane_digest",
            "candidate_lane_digest",
            "pair_binding_digest",
            "raw_crossover_digest",
            "evidence_digest",
        ):
            object.__setattr__(
                self, name, require_sha256_hex(getattr(self, name), field=name)
            )
        if (
            type(self.resident_policy) is not ResidentSpeedPolicy
            or self.calibration_digest != self.resident_policy.calibration_digest
            or self.calibration_context_digest
            != self.resident_policy.calibration_context_digest
            or type(self.rates) is not tuple
            or any(type(row) is not ResidentReadRate for row in self.rates)
            or type(self.stock_return_digests) is not tuple
            or len(self.stock_return_digests) != len(self.rates)
            or type(self.pair_session_ids) is not tuple
            or len(self.pair_session_ids) != 2
            or any(
                not isinstance(row, str)
                or len(row) != 32
                or row == "0" * 32
                or any(char not in "0123456789abcdef" for char in row)
                for row in self.pair_session_ids
            )
            or len(set(self.pair_session_ids)) != 2
            or type(self.started_monotonic_s) is not float
            or type(self.completed_monotonic_s) is not float
            or not math.isfinite(self.started_monotonic_s)
            or not math.isfinite(self.completed_monotonic_s)
            or self.completed_monotonic_s <= self.started_monotonic_s
            or self.completed_monotonic_s - self.started_monotonic_s
            > self.resident_policy.max_stage_seconds
        ):
            raise QualificationRunnerError("live resident speed witness is malformed")
        for row in self.stock_return_digests:
            require_sha256_hex(row, field="stock return digest")
        if tuple(row.role for row in self.rates) != resident_speed_roles(
            self.resident_policy.version, len(self.rates)
        ):
            raise QualificationRunnerError("live resident speed read order differs")
        if any(
            bool(row.windows) != (self.resident_policy.version >= 3)
            for row in self.rates
        ):
            raise QualificationRunnerError(
                "live resident read retention differs from its policy"
            )
        values = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "evidence_digest"
        }
        if self.evidence_digest != _live_digest(**values):
            raise QualificationRunnerError(
                "live resident speed witness digest does not recompute"
            )

    @property
    def policy(self) -> SpeedEvidencePolicy:
        return SpeedEvidencePolicy.resident()

    @property
    def has_repeat(self) -> bool:
        return len(self.rates) == 5

    def regrade(self, calibration, context, *, expected_policy=None):
        # The arithmetic depends only on the common calibrated rate fields.
        return ResidentSpeedWitness.regrade(
            self, calibration, context, expected_policy=expected_policy
        )

    def v6_result(self):
        # ``ResidentSpeedWitness.regrade`` dispatches here for policy v6+;
        # it reads only ``resident_policy`` and ``rates``, which this live
        # projection shares.
        return ResidentSpeedWitness.v6_result(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                field: (
                    format(getattr(self, field), ".17g")
                    if field in {"started_monotonic_s", "completed_monotonic_s"}
                    else list(getattr(self, field))
                    if field in {"pair_session_ids", "stock_return_digests"}
                    else getattr(self, field)
                )
                for field in self.__dataclass_fields__
                if field not in {"resident_policy", "rates"}
            },
            "rates": [row.to_dict() for row in self.rates],
            "resident_policy": self.resident_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ResidentPairLiveSpeedWitness":
        from cacheon.eval.crossover_runtime import ResidentReadRate, ResidentSpeedPolicy

        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise QualificationRunnerError("live resident speed witness is not closed")
        return cls(
            **{
                **value,
                "pair_session_ids": tuple(value["pair_session_ids"]),
                "stock_return_digests": tuple(value["stock_return_digests"]),
                "rates": tuple(ResidentReadRate.from_dict(row) for row in value["rates"]),
                "resident_policy": ResidentSpeedPolicy.from_dict(
                    value["resident_policy"]
                ),
                "started_monotonic_s": float(value["started_monotonic_s"]),
                "completed_monotonic_s": float(value["completed_monotonic_s"]),
            }
        )


def project_resident_pair_live_speed_witness(
    evidence: ResidentPairCrossoverEvidence,
    *,
    plan: ResidentPairCrossoverPlan,
) -> ResidentPairLiveSpeedWitness:
    """Project a terminal speed failure without retiring the loaded pair."""

    if (
        type(evidence) is not ResidentPairCrossoverEvidence
        or type(plan) is not ResidentPairCrossoverPlan
    ):
        raise ResidentPairSpeedWitnessError(
            "live resident speed witness authorities are not exact"
        )
    try:
        evidence.regrade(plan)
    except (ResidentPairCrossoverError, CrossoverRuntimeError) as exc:
        raise ResidentPairSpeedWitnessError(
            f"live resident speed evidence does not regrade: {exc}"
        ) from None
    if any(
        row.ending_bundle_digest is not None or row.ending_slots
        for row in evidence.request_slices
    ) or evidence.decision is not SpeedStageDecision.FAIL:
        raise ResidentPairSpeedWitnessError(
            "live resident terminal is not a stock-restored speed failure"
        )
    crossover = plan.crossover_plan
    values = {
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
        "pair_binding_digest": plan.pair_binding.digest,
        "pair_session_ids": tuple(
            row.session_id for row in plan.pair_binding.lanes
        ),
        "stock_return_digests": tuple(
            _slice_digest(row) for row in evidence.request_slices
        ),
        "raw_crossover_digest": evidence.digest,
        "resident_policy": evidence.policy,
        "rates": evidence.rates,
        "started_monotonic_s": float(evidence.started_monotonic_s),
        "completed_monotonic_s": float(evidence.completed_monotonic_s),
    }
    try:
        return ResidentPairLiveSpeedWitness(
            **values,
            evidence_digest=_live_digest(**values),
        )
    except (QualificationRunnerError, TypeError, ValueError) as exc:
        raise ResidentPairSpeedWitnessError(
            f"live resident speed witness projection failed: {exc}"
        ) from None


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
    "ResidentPairLiveSpeedWitness",
    "ResidentPairSpeedWitnessError",
    "project_resident_pair_live_speed_witness",
    "project_resident_pair_speed_witness",
]

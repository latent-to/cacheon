"""Qualification view of stock-restored or fully retired pair evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.marginal_runtime import PreparedMarginalRuntime
from cacheon.eval.qualification_continuation import (
    ResidentCountQualityCheckpoint,
    ResidentPairRetirementCheckpoint,
)
from cacheon.eval.resident_count_quality import ResidentCountQualityStockAuthority
from cacheon.eval.registered_resident_count_quality import (
    RegisteredResidentCountQualityResult,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
)
from cacheon.eval.crossover_runtime import SpeedStageDecision
from cacheon.stack_identity import canonical_json_bytes, require_sha256_hex


class ResidentPairQualityLifecycleError(RuntimeError):
    """Pair evidence cannot support downstream audit and pristine T."""


def _count_retirement_matches(
    result: RegisteredResidentCountQualityResult,
    checkpoint: ResidentCountQualityCheckpoint,
    stock: ResidentCountQualityStockAuthority,
    retirement: ResidentPairRetirementCheckpoint,
) -> bool:
    return (
        type(result) is RegisteredResidentCountQualityResult
        and type(checkpoint) is ResidentCountQualityCheckpoint
        and type(stock) is ResidentCountQualityStockAuthority
        and type(retirement) is ResidentPairRetirementCheckpoint
        and result.execution_plan_digest == checkpoint.execution_plan_digest
        and result.raw_execution_evidence_digest
        == checkpoint.raw_execution_evidence_semantic_digest
        and result.candidate_observation_digest
        == checkpoint.candidate_observation_semantic_digest
        and result.fixed_stock_authority_digest
        == checkpoint.fixed_stock_authority_digest
        and result.pair_binding_digest == checkpoint.pair_binding_digest
        and result.fixed_stock_authority_digest == stock.digest
        and result.stock_observation_digest == stock.observation_digest
        and result.execution_envelope_digest == stock.envelope_digest
        and result.policy_digest == stock.policy.digest
        and retirement.count_plan_digest == result.execution_plan_digest
        and retirement.count_evidence_digest
        == result.raw_execution_evidence_digest
        and retirement.pair_binding.digest == result.pair_binding_digest
    )


@dataclass(frozen=True)
class ResidentPairQualificationClosure:
    """Self-contained count and retirement authority carried by a final report."""

    count_result: RegisteredResidentCountQualityResult
    count_checkpoint: ResidentCountQualityCheckpoint
    stock_authority: ResidentCountQualityStockAuthority
    retirement: ResidentPairRetirementCheckpoint

    def __post_init__(self) -> None:
        result, checkpoint, stock, retirement = (
            self.count_result, self.count_checkpoint,
            self.stock_authority, self.retirement,
        )
        if not _count_retirement_matches(
            result, checkpoint, stock, retirement
        ) or result.decision != "PASS":
            raise ResidentPairQualityLifecycleError(
                "resident pair qualification closure does not recompute"
            )


@dataclass(frozen=True)
class ResidentPairCandidateView:
    """Candidate identity used by the existing qualification projections."""

    candidate: object

    @property
    def arm(self):
        return self.candidate.arm


@dataclass(frozen=True)
class ResidentPairMarginalLifecycleEvidence:
    """Path-free B/C/B-prime evidence at a durable pair boundary."""

    prepared: PreparedMarginalRuntime
    plan: ResidentPairCrossoverPlan
    crossover: ResidentPairCrossoverEvidence
    retirement: ResidentPairRetirementCheckpoint | None
    count_result: RegisteredResidentCountQualityResult | None
    count_checkpoint: ResidentCountQualityCheckpoint | None
    stock_authority: ResidentCountQualityStockAuthority | None
    quality_read: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.prepared) is not PreparedMarginalRuntime
            or len(self.prepared.candidates) != 1
            or type(self.plan) is not ResidentPairCrossoverPlan
            or type(self.crossover) is not ResidentPairCrossoverEvidence
            or (
                self.retirement is not None
                and type(self.retirement) is not ResidentPairRetirementCheckpoint
            )
            or (
                self.count_result is not None
                and type(self.count_result) is not RegisteredResidentCountQualityResult
            )
            or (self.count_result is None) != (self.count_checkpoint is None)
            or (self.count_result is None) != (self.stock_authority is None)
            or self.quality_read not in (1, 2)
            or type(self.quality_read) is not int
        ):
            raise ResidentPairQualityLifecycleError(
                "resident pair lifecycle is not exactly typed"
            )
        candidate = self.prepared.candidates[0]
        crossover_plan = self.plan.crossover_plan
        if (
            candidate.arm.selected_delta_digest
            != crossover_plan.selected_delta_digest
            or candidate.launch.digest != crossover_plan.candidate.launch.digest
            or candidate.session_plan != crossover_plan.candidate.session_plan
            or candidate.binding.launch_binding != crossover_plan.candidate.binding
            or self.prepared.baseline_launch.stack_digest
            != crossover_plan.baseline.launch.stack_digest
            or self.prepared.baseline_launch.tree_digest
            != crossover_plan.baseline.launch.tree_digest
            or self.crossover.plan_digest != self.plan.digest
            or self.crossover.pair_binding != self.plan.pair_binding
            or (
                self.retirement is not None
                and (
                    self.retirement.pair_binding != self.plan.pair_binding
                    or self.retirement.speed_plan_digest != self.plan.digest
                    or self.retirement.speed_evidence_digest
                    != self.crossover.digest
                )
            )
            or (self.quality_read == 2 and not self.crossover.escalated)
            or (
                self.retirement is not None
                and len(self.retirement.request_history_slice_digests)
                < len(self.crossover.request_slices)
                + (0 if self.count_result is None else 2)
            )
            or (
                self.retirement is None
                and (
                    self.crossover.decision is not SpeedStageDecision.FAIL
                    or self.count_result is not None
                    or any(
                        row.ending_bundle_digest is not None or row.ending_slots
                        for row in self.crossover.request_slices
                    )
                )
            )
        ):
            raise ResidentPairQualityLifecycleError(
                "resident pair lifecycle differs from its prepared authority"
            )
        self.crossover.regrade(self.plan)
        if self.count_result is None:
            if (
                self.crossover.decision is not SpeedStageDecision.FAIL
                or (
                    self.retirement is not None
                    and self.retirement.count_plan_digest is not None
                )
            ):
                raise ResidentPairQualityLifecycleError(
                    "resident count retirement lacks its registered result"
                )
        else:
            result = self.count_result
            checkpoint = self.count_checkpoint
            stock = self.stock_authority
            assert checkpoint is not None and stock is not None
            if (
                self.crossover.decision is not SpeedStageDecision.PASS
                or self.retirement is None
                or result.candidate_bundle_digest
                != self.plan.candidate_bundle_digest
                or not _count_retirement_matches(
                    result, checkpoint, stock, self.retirement
                )
            ):
                raise ResidentPairQualityLifecycleError(
                    "registered count result differs from retired pair evidence"
                )
            if result.decision == "PASS":
                ResidentPairQualificationClosure(
                    result, checkpoint, stock, self.retirement
                )
        if self.retirement is not None:
            for binding, lane in zip(
                self.plan.pair_binding.lanes, self.retirement.lanes, strict=True
            ):
                if (
                    lane.lane_id != binding.lane_id
                    or lane.quiescence.namespace_digest
                    != binding.executor_namespace_digest
                    or lane.quiescence.observed_monotonic_s
                    < self.crossover.completed_monotonic_s
                ):
                    raise ResidentPairQualityLifecycleError(
                        "resident pair quiescence differs from its runtime binding"
                    )

    @property
    def source(self):
        return self.prepared.source

    @property
    def candidates(self) -> tuple[ResidentPairCandidateView, ...]:
        return (ResidentPairCandidateView(self.prepared.candidates[0]),)

    @property
    def candidates_repeat(self) -> tuple[ResidentPairCandidateView, ...]:
        return self.candidates if self.crossover.escalated else ()

    @property
    def role_names(self) -> tuple[str, str, str]:
        return (
            ("B", "C", "B_prime")
            if self.quality_read == 1
            else ("B_prime", "C_prime", "B_double_prime")
        )

    def role_batches(self, role: str):
        roles = (
            ("B", "C", "B_prime", "C_prime", "B_double_prime")
            if self.crossover.escalated
            else ("B", "C", "B_prime")
        )
        if role not in self.role_names:
            raise ResidentPairQualityLifecycleError(
                "resident pair quality role is absent"
            )
        return self.crossover.request_slices[roles.index(role)].new_batches

    def quality_leg(self, candidate_read: int) -> "ResidentPairMarginalLifecycleEvidence":
        return replace(self, quality_read=candidate_read)

    @property
    def timed_session_ids(self) -> frozenset[str]:
        return frozenset(row.session_id for row in self.plan.pair_binding.lanes)

    @property
    def lane_quiescence(self):
        if self.retirement is None:
            raise ResidentPairQualityLifecycleError(
                "live resident pair has no quiescence evidence"
            )
        return tuple((row.lane_id, row.quiescence) for row in self.retirement.lanes)

    @property
    def retirement_cutoff(self) -> float:
        if self.retirement is None:
            raise ResidentPairQualityLifecycleError(
                "live resident pair has no retirement cutoff"
            )
        return max(row.quiescence.observed_monotonic_s for row in self.retirement.lanes)

    @property
    def closure(self) -> ResidentPairQualificationClosure | None:
        if (
            self.retirement is None
            or self.count_result is None
            or self.count_result.decision != "PASS"
        ):
            return None
        return ResidentPairQualificationClosure(
            self.count_result, self.count_checkpoint,
            self.stock_authority, self.retirement,
        )


def reopen_resident_pair_qualification_attempt(
    payload: bytes,
    *,
    authority_digest: str,
    report_digests: tuple[str, ...],
    evidence_inventory: tuple[EvidenceArtifactRef, ...],
):
    """Reopen one self-contained resident attempt after CPU CAS import."""

    from cacheon.eval.qualification_runner import CohortQualificationAttempt

    try:
        authority_digest = require_sha256_hex(
            authority_digest, field="qualification authority digest"
        )
        attempt = CohortQualificationAttempt.from_dict(
            json.loads(payload.decode("utf-8"))
        )
    except (UnicodeError, ValueError, TypeError, RuntimeError) as exc:
        raise ResidentPairQualityLifecycleError(
            f"CPU resident attempt cannot reopen: {exc}"
        ) from None
    closures = tuple(row.resident_pair_closure for row in attempt.reports)
    if (
        canonical_json_bytes(attempt.to_dict()) != payload
        or attempt.authority_digest != authority_digest
        or tuple(row.digest for row in attempt.reports) != report_digests
        or any(row is None for row in closures)
    ):
        raise ResidentPairQualityLifecycleError(
            "CPU resident attempt differs from its qualification product"
        )
    required = {
        reference
        for report, closure in zip(attempt.reports, closures, strict=True)
        for reference in (
            report.raw_quality_artifact,
            closure.count_checkpoint.raw_execution_evidence,
            closure.count_checkpoint.candidate_observation,
            closure.stock_authority.artifact,
            *(() if report.repeat_quality is None else (
                report.repeat_quality.raw_quality_artifact,
            )),
        )
    }
    if not required.issubset(evidence_inventory):
        raise ResidentPairQualityLifecycleError(
            "CPU resident attempt lacks its supporting evidence inventory"
        )
    return attempt


__all__ = [
    "ResidentPairMarginalLifecycleEvidence",
    "ResidentPairQualificationClosure",
    "ResidentPairQualityLifecycleError",
    "reopen_resident_pair_qualification_attempt",
]

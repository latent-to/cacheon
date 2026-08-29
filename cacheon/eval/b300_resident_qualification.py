"""Sole retained-pair prefix for registered B300 qualification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_canonical_json_evidence
from cacheon.eval.b300_qualification_lanes import B300ArenaProviderError
from cacheon.eval.b300_resident_pair_factory import (
    B300CommissionedResidentPairFactory,
    B300ResidentPairFactoryError,
    B300ResidentPairRequestAuthority,
    B300ResidentRequestPair,
)
from cacheon.eval.crossover_runtime import SpeedStageDecision
from cacheon.eval.oci_backend import OCIBackendError, stage_swap_bundle
from cacheon.eval.qualification_continuation import (
    QualificationContinuation,
    QualificationContinuationError,
    ResidentCountQualityCheckpoint,
    ResidentPairRetirementCheckpoint,
)
from cacheon.eval.qualification_runner import (
    CandidateQualificationAuthority,
    CausalQualificationInput,
    qualification_authority_digest,
)
from cacheon.eval.registered_resident_count_quality import (
    B300ResidentCountQualityCapability,
    RegisteredResidentCountQualityAuthority,
    RegisteredResidentCountQualityError,
    RegisteredResidentCountQualityResult,
    evaluate_registered_resident_count_quality,
)
from cacheon.eval.resident_count_continuation import (
    ResidentCountQualityContinuationHold,
    publish_resident_count_quality_continuation,
    reopen_resident_count_quality_continuation,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionError,
    ResidentCountQualityExecutionPlan,
    ResidentCountQualityExecutionResult,
    execute_candidate_count_quality,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverError,
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
    run_resident_pair_crossover,
)
from cacheon.eval.resident_pair_retirement_checkpoint import (
    ResidentPairRetirementHold,
    build_resident_pair_retirement_checkpoint,
    regrade_resident_pair_retirement_checkpoint,
)
from cacheon.eval.resident_execution_evidence import EXECUTION_CODEC
from cacheon.eval.resident_pair_binding import ResidentPairRuntimeBinding

_LOG = logging.getLogger(__name__)


class B300ResidentQualificationError(RuntimeError):
    """The retained-pair qualification authority is malformed."""


class B300ResidentQualificationHold(B300ResidentQualificationError):
    """Durable or runtime state is incomplete and must never be rerun."""

    decision = "HOLD"


@dataclass(frozen=True)
class B300ResidentQualificationPrefix:
    """Reopened speed product, optionally followed by count and retirement."""

    speed_plan: ResidentPairCrossoverPlan
    speed: ResidentPairCrossoverEvidence
    count_plan: ResidentCountQualityExecutionPlan | None
    count: ResidentCountQualityExecutionResult | None
    count_result: RegisteredResidentCountQualityResult | None
    count_checkpoint: ResidentCountQualityCheckpoint | None
    retirement: ResidentPairRetirementCheckpoint | None

    def __post_init__(self) -> None:
        count_rows = (
            self.count_plan, self.count, self.count_result, self.count_checkpoint
        )
        if (
            type(self.speed_plan) is not ResidentPairCrossoverPlan
            or type(self.speed) is not ResidentPairCrossoverEvidence
            or self.speed.plan_digest != self.speed_plan.digest
            or (
                self.retirement is not None
                and (
                    type(self.retirement) is not ResidentPairRetirementCheckpoint
                    or self.speed.pair_binding != self.retirement.pair_binding
                )
            )
            or (
                self.retirement is None
                and self.speed.decision is not SpeedStageDecision.FAIL
            )
            or (all(row is None for row in count_rows))
            != (self.speed.decision is SpeedStageDecision.FAIL)
            or (
                any(row is not None for row in count_rows)
                and (
                    type(self.count_plan) is not ResidentCountQualityExecutionPlan
                    or type(self.count) is not ResidentCountQualityExecutionResult
                    or type(self.count_result)
                    is not RegisteredResidentCountQualityResult
                    or type(self.count_checkpoint) is not ResidentCountQualityCheckpoint
                    or self.count_plan.pair_binding != self.speed.pair_binding
                    or self.count.evidence.execution_plan_digest
                    != self.count_plan.digest
                    or self.count_result.execution_plan_digest
                    != self.count_plan.digest
                )
            )
        ):
            raise B300ResidentQualificationError(
                "resident qualification prefix is not exactly bound"
            )


_HOLD_ERRORS = (
    B300ArenaProviderError,
    B300ResidentPairFactoryError,
    OCIBackendError,
    QualificationContinuationError,
    RegisteredResidentCountQualityError,
    ResidentCountQualityContinuationHold,
    ResidentCountQualityExecutionError,
    ResidentPairCrossoverError,
    ResidentPairRetirementHold,
)


def _speed_plan(
    plan: CausalQualificationInput,
    factory: B300CommissionedResidentPairFactory,
    pair: B300ResidentRequestPair | None,
    retirement: ResidentPairRetirementCheckpoint | None,
    *,
    candidate_bundle_digest: str,
    screen_lane: str,
) -> ResidentPairCrossoverPlan:
    orientation = factory.lane_pair.orientation(screen_lane)
    resolved = pair.binding if pair is not None else retirement.pair_binding  # type: ignore[union-attr]
    if type(resolved) is not ResidentPairRuntimeBinding:
        raise B300ResidentQualificationError(
            "resident speed plan lacks one exact pair binding"
        )
    return ResidentPairCrossoverPlan(
        candidate_bundle_digest,
        plan.resident_speed_plan,
        resolved,
        orientation.resident_baseline.lane_id,
        orientation.candidate.lane_id,
    )


def _speed_plan_from_binding(
    plan: CausalQualificationInput,
    factory: B300CommissionedResidentPairFactory,
    binding: ResidentPairRuntimeBinding,
    *,
    candidate_bundle_digest: str,
    screen_lane: str,
) -> ResidentPairCrossoverPlan:
    class _BindingView:
        def __init__(self, value: ResidentPairRuntimeBinding) -> None:
            self.binding = value

    return _speed_plan(
        plan,
        factory,
        _BindingView(binding),  # type: ignore[arg-type]
        None,
        candidate_bundle_digest=candidate_bundle_digest,
        screen_lane=screen_lane,
    )


def _count_plan(
    capability: B300ResidentCountQualityCapability,
    speed_plan: ResidentPairCrossoverPlan,
) -> ResidentCountQualityExecutionPlan:
    return ResidentCountQualityExecutionPlan(
        speed_plan.candidate_bundle_digest,
        capability.envelope,
        capability.prompt_batches,
        capability.selected_ordinals,
        capability.batch_shape,
        capability.admission,
        speed_plan.pair_binding,
    )


def _registered_count(
    capability: B300ResidentCountQualityCapability,
    candidate: ArenaCandidateBinding,
    plan: ResidentCountQualityExecutionPlan,
    execution: ResidentCountQualityExecutionResult,
    *,
    evidence_root,
) -> RegisteredResidentCountQualityResult:
    authority = RegisteredResidentCountQualityAuthority.register(
        capability.catalog,
        candidate.reservation.target_id,
        plan=plan,
        stock_authority=capability.stock_authority,
        judge=capability.judge,
        policy=capability.stock_authority.policy,
    )
    return evaluate_registered_resident_count_quality(
        evidence_root,
        catalog=capability.catalog,
        target_id=candidate.reservation.target_id,
        authority=authority,
        plan=plan,
        execution=execution,
        stock_authority=capability.stock_authority,
        judge=capability.judge,
    )


def _reopen(
    *,
    factory: B300CommissionedResidentPairFactory,
    capability: B300ResidentCountQualityCapability,
    candidate: ArenaCandidateBinding,
    plan: CausalQualificationInput,
    authority: B300ResidentPairRequestAuthority,
    continuation: QualificationContinuation,
    retirement: ResidentPairRetirementCheckpoint,
    screen_lane: str,
) -> B300ResidentQualificationPrefix:
    speed_plan = _speed_plan(
        plan,
        factory,
        None,
        retirement,
        candidate_bundle_digest=candidate.publication.content_hash,
        screen_lane=screen_lane,
    )
    speed = continuation.load_resident_pair_speed(speed_plan)
    if speed is None:
        raise B300ResidentQualificationHold(
            "resident retirement exists without durable speed evidence"
        )
    count_plan = count = count_result = None
    raw_count = continuation.load_resident_count_quality()
    if retirement.count_plan_digest is not None:
        count_plan = _count_plan(capability, speed_plan)
        count = reopen_resident_count_quality_continuation(
            plan.evidence_root,
            continuation,
            plan=count_plan,
            fixed_stock_authority_digest=capability.stock_authority.digest,
            pair_binding=speed_plan.pair_binding,
            judge=capability.judge,
        )
        if count is None:
            raise B300ResidentQualificationHold(
                "resident retirement exists without durable count evidence"
            )
        count_result = _registered_count(
            capability, candidate, count_plan, count, evidence_root=plan.evidence_root
        )
    elif raw_count is not None:
        raise B300ResidentQualificationHold(
            "resident count evidence exists outside the retirement history"
        )
    regrade_resident_pair_retirement_checkpoint(
        retirement,
        factory=factory,
        authority=authority,
        speed_plan=speed_plan,
        speed=speed,
        count_plan=count_plan,
        count=None if count is None else count.evidence,
    )
    return B300ResidentQualificationPrefix(
        speed_plan, speed, count_plan, count, count_result, raw_count, retirement
    )


def run_b300_resident_qualification_prefix(
    *,
    factory: B300CommissionedResidentPairFactory,
    capability: B300ResidentCountQualityCapability,
    candidate: ArenaCandidateBinding,
    plan: CausalQualificationInput,
    continuation: QualificationContinuation,
    screen_lane: str,
    deadline: float,
) -> B300ResidentQualificationPrefix:
    """Run once or purely reopen graph-on speed, count, and pair retirement."""

    if (
        type(factory) is not B300CommissionedResidentPairFactory
        or type(capability) is not B300ResidentCountQualityCapability
        or type(candidate) is not ArenaCandidateBinding
        or type(plan) is not CausalQualificationInput
        or type(continuation) is not QualificationContinuation
        or len(plan.candidates) != 1
        or type(plan.candidates[0]) is not CandidateQualificationAuthority
        or plan.resident_speed_plan is None
        or factory.model_mount != plan.model_mount
        or candidate.reservation.selected_delta_digest
        != plan.candidates[0].selected_delta_digest
        or capability.catalog.digest
        != plan.candidates[0].profile.reference.catalog_digest
        or continuation.authority_digest != qualification_authority_digest(plan)
        or continuation.source_digest != plan.prepared.source.digest
    ):
        raise B300ResidentQualificationError(
            "resident qualification inputs differ from the sealed plan"
        )
    authority = B300ResidentPairRequestAuthority(
        continuation.request_digest,
        continuation.authority_digest,
        plan.candidates[0].profile.digest,
    )
    try:
        retirement = continuation.load_resident_pair_retirement()
        if retirement is not None:
            return _reopen(
                factory=factory,
                capability=capability,
                candidate=candidate,
                plan=plan,
                authority=authority,
                continuation=continuation,
                retirement=retirement,
                screen_lane=screen_lane,
            )
        durable_speed = continuation.load_resident_pair_speed_raw()
        if durable_speed is not None:
            speed_plan = _speed_plan_from_binding(
                plan,
                factory,
                durable_speed.pair_binding,
                candidate_bundle_digest=candidate.publication.content_hash,
                screen_lane=screen_lane,
            )
            speed = continuation.load_resident_pair_speed(speed_plan)
            if speed is None:
                raise B300ResidentQualificationHold(
                    "durable resident speed disappeared while reopening"
                )
            if (
                speed.decision is not SpeedStageDecision.FAIL
                or continuation.load_resident_count_quality() is not None
            ):
                raise B300ResidentQualificationHold(
                    "durable resident work exists without exact pair retirement"
                )
            return B300ResidentQualificationPrefix(
                speed_plan, speed, None, None, None, None, None
            )
        if continuation.load_resident_count_quality() is not None:
            raise B300ResidentQualificationHold(
                "durable resident work exists without exact pair retirement"
            )
        staged = stage_swap_bundle(
            factory.swap_intake_root,
            candidate.publication.root,
            expected_digest=candidate.publication.content_hash,
        )
        if staged != candidate.publication.content_hash:
            raise B300ResidentQualificationHold(
                "staged resident bundle changed publication identity"
            )
        factory.open_request(authority, deadline=deadline)
        borrowed: B300ResidentRequestPair | None = None
        completed = False
        try:
            borrowed = factory.borrow(authority)
            speed_plan = _speed_plan(
                plan,
                factory,
                borrowed,
                None,
                candidate_bundle_digest=staged,
                screen_lane=screen_lane,
            )
            speed = run_resident_pair_crossover(
                speed_plan,
                pair=borrowed.pair,
                deadline=deadline,
                clock=factory.clock,
            )
            continuation.record_resident_pair_speed(speed)
            speed = continuation.load_resident_pair_speed(speed_plan)
            if speed is None:
                raise B300ResidentQualificationHold(
                    "resident speed disappeared after durable publication"
                )
            if speed.decision is SpeedStageDecision.FAIL:
                factory.release(authority, borrowed.binding)
                completed = True
                return B300ResidentQualificationPrefix(
                    speed_plan, speed, None, None, None, None, None
                )
            count_plan = count = None
            count_plan = _count_plan(capability, speed_plan)
            count = execute_candidate_count_quality(
                count_plan,
                pair=borrowed.pair,
                judge=capability.judge,
                deadline=deadline,
            )
            publish_resident_count_quality_continuation(
                plan.evidence_root,
                continuation,
                count,
                plan=count_plan,
                fixed_stock_authority_digest=capability.stock_authority.digest,
                pair_binding=borrowed.binding,
                judge=capability.judge,
                deadline=deadline,
            )
            count = reopen_resident_count_quality_continuation(
                plan.evidence_root,
                continuation,
                plan=count_plan,
                fixed_stock_authority_digest=capability.stock_authority.digest,
                pair_binding=borrowed.binding,
                judge=capability.judge,
            )
            if count is None:
                raise B300ResidentQualificationHold(
                    "resident count disappeared after durable publication"
                )
            _registered_count(
                capability,
                candidate,
                count_plan,
                count,
                evidence_root=plan.evidence_root,
            )
            retirement = build_resident_pair_retirement_checkpoint(
                factory=factory,
                authority=authority,
                speed_plan=speed_plan,
                speed=speed,
                count_plan=count_plan,
                count=None if count is None else count.evidence,
            )
            continuation.record_resident_pair_retirement(retirement)
            reopened = continuation.load_resident_pair_retirement()
            if reopened is None:
                raise B300ResidentQualificationHold(
                    "resident retirement disappeared after durable publication"
                )
            completed = True
            return _reopen(
                factory=factory,
                capability=capability,
                candidate=candidate,
                plan=plan,
                authority=authority,
                continuation=continuation,
                retirement=reopened,
                screen_lane=screen_lane,
            )
        finally:
            if not completed:
                if borrowed is None:
                    factory.close_request()
                else:
                    factory.retire_and_quiesce(authority, borrowed.binding)
    except B300ResidentQualificationHold:
        raise
    except _HOLD_ERRORS as exc:
        raise B300ResidentQualificationHold(
            "resident qualification prefix is incomplete or unauthenticated"
        ) from exc


EXECUTION_EVIDENCE_DOMAIN = "qualification.execution"
EXECUTION_EVIDENCE_SCHEMA = "cacheon.qualification.execution.v1"


def execution_evidence_refs(
    speed: ResidentPairCrossoverEvidence,
    *,
    evidence_root: str | Path,
    request_digest: str,
    authority_digest: str,
    source_digest: str,
) -> tuple[EvidenceArtifactRef, ...]:
    """Publish what every rank did under each candidate generation; never raises.

    The rows come back across the swap that closes a generation and decide,
    inside the crossover, whether its reads may be graded. They do not decide
    anything here: this writes them to the evidence store as an unsealed
    artifact so the product carries them off the pod, where the miner's report
    renders them. A run that could not write the artifact is still a valid run,
    so the failure is logged and an empty tuple returned.
    """

    try:
        swaps = []
        for row in speed.request_slices:
            for swap in row.new_swaps:
                execution = swap.execution
                # Stock generations close with no loaded rank and nothing to say.
                if not any(rank.loaded or rank.load_error for rank in execution.ranks):
                    continue
                swaps.append(
                    {
                        "executed_ranks": execution.prior_execution_ranks,
                        "expected_ranks": swap.expected_ranks,
                        "generation": execution.prior_generation,
                        "lane_id": row.lane_id,
                        "ranks": [EXECUTION_CODEC.encode(rank) for rank in execution.ranks],
                        "request_id": row.request_id,
                    }
                )
        if not swaps:
            return ()
        return (
            publish_canonical_json_evidence(
                evidence_root,
                {
                    "authority_digest": authority_digest,
                    "bundle_digest": speed.candidate_bundle_digest,
                    "request_digest": request_digest,
                    "schema": EXECUTION_EVIDENCE_SCHEMA,
                    "source_digest": source_digest,
                    "swaps": swaps,
                },
                domain=EXECUTION_EVIDENCE_DOMAIN,
                schema=EXECUTION_EVIDENCE_SCHEMA,
            ),
        )
    except Exception:  # noqa: BLE001 - a report artifact must not fail the run
        _LOG.exception("resident execution evidence could not be published")
        return ()


__all__ = [
    "B300ResidentQualificationError",
    "B300ResidentQualificationHold",
    "B300ResidentQualificationPrefix",
    "EXECUTION_EVIDENCE_DOMAIN",
    "EXECUTION_EVIDENCE_SCHEMA",
    "execution_evidence_refs",
    "run_b300_resident_qualification_prefix",
]

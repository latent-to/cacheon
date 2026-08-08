"""Pre-execution graph gate for one registered B300 qualification plan.

The deployment has already built and sealed the private qualification plan by
the time this gate runs.  This module never constructs another plan and never
executes a resident arm.  It independently reopens the raw graph CAS, then
returns one of three exact in-process products:

* PASS preserves the same plan/factory objects for ordinary qualification;
* FAIL publishes and reopens the graph-only terminal before projecting a
  non-retryable intake batch; and
* HOLD maps only typed evidence states onto the closed remote HOLD algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldReason
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.marginal_runtime import PreparedCandidateRuntime, PreparedMarginalRuntime
from cacheon.eval.qualification import (
    GRAPH_EVIDENCE_DOMAIN,
    GRAPH_EVIDENCE_MEDIA_TYPE,
    GRAPH_EVIDENCE_SCHEMA,
    GraphVerificationGrade,
    QualificationDecision,
    QualificationError,
    reopen_graph_verification,
)
from cacheon.eval.qualification_graph_exit import (
    QualificationGraphExit,
    QualificationGraphExitError,
    QualificationGraphExitHold,
    publish_qualification_graph_exit,
    reopen_qualification_graph_exit,
)
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationPlanFactory,
    QualificationReservation,
)
from cacheon.eval.qualification_runner import (
    CandidateQualificationAuthority,
    CausalQualificationInput,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex
from cacheon.stack_plan import MarginalArmPlan


GRAPH_GATE_DIAGNOSTIC_SCHEMA = "cacheon.eval.b300-qualification-graph-gate-hold.v1"


class B300QualificationGraphGateError(RuntimeError):
    """The prebuilt plan, factory, candidate, or gate result is inconsistent."""


class B300QualificationGraphHoldCode(str, Enum):
    """Closed internal facts used only to derive a path-free diagnostic digest."""

    GRAPH_PROVIDER_UNAVAILABLE = "graph_provider_unavailable"
    RAW_EVIDENCE_UNAVAILABLE = "raw_evidence_unavailable"
    RAW_EVIDENCE_INCOMPLETE = "raw_evidence_incomplete"
    EXIT_PUBLICATION_AMBIGUOUS = "exit_publication_ambiguous"


def _digest(value: object, field_name: str) -> str:
    try:
        return require_sha256_hex(value, field=field_name)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise B300QualificationGraphGateError(str(exc)) from None


def _evidence_key(reference: EvidenceArtifactRef) -> tuple[str, str, str, str, int]:
    return (
        reference.domain,
        reference.sha256,
        reference.media_type,
        reference.schema,
        reference.size,
    )


def _references(value: object) -> tuple[EvidenceArtifactRef, ...]:
    if type(value) is not tuple or any(
        type(row) is not EvidenceArtifactRef for row in value
    ):
        raise B300QualificationGraphGateError(
            "graph gate evidence references are not an exact tuple"
        )
    rows = tuple(value)
    if (
        rows != tuple(sorted(rows, key=_evidence_key))
        or len(set(rows)) != len(rows)
        or len({row.sha256 for row in rows}) != len(rows)
    ):
        raise B300QualificationGraphGateError(
            "graph gate evidence references are duplicate or noncanonical"
        )
    return rows


@dataclass(frozen=True)
class B300QualificationGraphGateHold:
    """A typed graph HOLD which the authenticated adapter may safely capture."""

    reason: RemoteQualificationHoldReason
    diagnostic_digest: str

    def __post_init__(self) -> None:
        if type(self.reason) is not RemoteQualificationHoldReason:
            raise B300QualificationGraphGateError(
                "graph gate HOLD reason is not closed"
            )
        object.__setattr__(
            self,
            "diagnostic_digest",
            _digest(self.diagnostic_digest, "graph gate diagnostic digest"),
        )


@dataclass(frozen=True)
class B300QualificationGraphGatePass:
    """Permission to continue with the same exact prebuilt plan and factory."""

    plan: CausalQualificationInput = field(repr=False, compare=False)
    factory: QualificationPlanFactory = field(repr=False, compare=False)
    supporting_evidence_refs: tuple[EvidenceArtifactRef, ...]

    def __post_init__(self) -> None:
        if (
            type(self.plan) is not CausalQualificationInput
            or type(self.factory) is not QualificationPlanFactory
        ):
            raise B300QualificationGraphGateError(
                "graph PASS changed the exact plan or factory type"
            )
        refs = _references(self.supporting_evidence_refs)
        if (
            len(self.plan.candidates) != 1
            or type(self.plan.candidates[0]) is not CandidateQualificationAuthority
            or refs != (self.plan.candidates[0].graph_artifact_ref,)
        ):
            raise B300QualificationGraphGateError(
                "graph PASS evidence differs from its registered authority"
            )
        object.__setattr__(self, "supporting_evidence_refs", refs)


@dataclass(frozen=True)
class B300QualificationGraphGateFail:
    """One reopened graph-only FAIL and its intake projection."""

    plan: CausalQualificationInput = field(repr=False, compare=False)
    factory: QualificationPlanFactory = field(repr=False, compare=False)
    batch: QualificationIntakeBatch
    graph_exit_ref: EvidenceArtifactRef
    graph_exit: QualificationGraphExit
    supporting_evidence_refs: tuple[EvidenceArtifactRef, ...]

    def __post_init__(self) -> None:
        if (
            type(self.plan) is not CausalQualificationInput
            or type(self.factory) is not QualificationPlanFactory
            or type(self.batch) is not QualificationIntakeBatch
            or type(self.graph_exit_ref) is not EvidenceArtifactRef
            or type(self.graph_exit) is not QualificationGraphExit
        ):
            raise B300QualificationGraphGateError(
                "graph FAIL changed the exact terminal product type"
            )
        refs = _references(self.supporting_evidence_refs)
        expected_refs = tuple(
            sorted(
                (self.graph_exit_ref, self.graph_exit.graph_artifact_ref),
                key=_evidence_key,
            )
        )
        if (
            self.batch.attempt_ref != self.graph_exit_ref
            or len(self.batch.outcomes) != 1
            or self.batch.outcomes[0].decision is not QualificationDecision.FAIL
            or self.batch.outcomes[0].retryable
            or self.batch.outcomes[0].attempt_artifact_sha256
            != self.graph_exit_ref.sha256
            or self.batch.outcomes[0].report_digest != self.graph_exit.digest
            or refs != expected_refs
        ):
            raise B300QualificationGraphGateError(
                "graph FAIL intake projection differs from its reopened exit"
            )
        object.__setattr__(self, "supporting_evidence_refs", refs)


B300QualificationGraphGateResult = (
    B300QualificationGraphGatePass
    | B300QualificationGraphGateFail
    | B300QualificationGraphGateHold
)


def qualification_graph_gate_hold(
    reason: RemoteQualificationHoldReason,
    *,
    authenticated_request_digest: str,
    authority_context_digest: str,
    code: B300QualificationGraphHoldCode,
) -> B300QualificationGraphGateHold:
    """Create a closed diagnostic without retaining exception text or paths."""

    if (
        type(reason) is not RemoteQualificationHoldReason
        or type(code) is not B300QualificationGraphHoldCode
    ):
        raise B300QualificationGraphGateError(
            "graph HOLD diagnostic inputs are not closed"
        )
    request = _digest(authenticated_request_digest, "authenticated request digest")
    authority = _digest(authority_context_digest, "graph authority context digest")
    return B300QualificationGraphGateHold(
        reason,
        canonical_digest(
            GRAPH_GATE_DIAGNOSTIC_SCHEMA,
            {
                "authority_context_digest": authority,
                "code": code.value,
                "reason": reason.value,
                "request_digest": request,
            },
        ),
    )


def _context(
    factory: QualificationPlanFactory,
    plan: CausalQualificationInput,
    evidence_root: Path,
    candidates: tuple[ArenaCandidateBinding, ...],
) -> tuple[
    ArenaCandidateBinding,
    QualificationReservation,
    CandidateQualificationAuthority,
]:
    if (
        type(factory) is not QualificationPlanFactory
        or type(plan) is not CausalQualificationInput
        or not isinstance(evidence_root, Path)
        or evidence_root != plan.evidence_root
        or type(candidates) is not tuple
        or len(candidates) != 1
        or type(candidates[0]) is not ArenaCandidateBinding
    ):
        raise B300QualificationGraphGateError(
            "graph gate requires one exact prebuilt plan, factory, root, and candidate"
        )
    manifest = factory.manifest
    if (
        type(manifest) is not QualificationAuthorityManifest
        or manifest.lane != "registered"
        or len(manifest.reservations) != 1
        or len(plan.candidates) != 1
        or type(plan.candidates[0]) is not CandidateQualificationAuthority
        or type(plan.prepared) is not PreparedMarginalRuntime
        or len(plan.prepared.candidates) != 1
        or type(plan.prepared.candidates[0]) is not PreparedCandidateRuntime
    ):
        raise B300QualificationGraphGateError(
            "graph gate supports one registered qualification authority"
        )
    observed = QualificationAuthorityManifest.seal(
        plan,
        reservations=manifest.reservations,
        selection_secret_reference=manifest.selection_secret_reference,
    )
    candidate = candidates[0]
    reservation = manifest.reservations[0]
    authority = plan.candidates[0]
    prepared = plan.prepared.candidates[0]
    arm = prepared.arm
    binding = authority.graph_requirement.binding
    member_ids = tuple(row.slot_id for row in binding.members)
    raw_type = (
        authority.graph_artifact_ref.domain,
        authority.graph_artifact_ref.media_type,
        authority.graph_artifact_ref.schema,
    )
    exact = (
        observed == manifest,
        candidate.reservation == reservation,
        reservation.selected_delta_digest == authority.selected_delta_digest,
        type(arm) is MarginalArmPlan,
        getattr(arm, "selected_delta_digest", None) == binding.selected_delta_digest,
        getattr(getattr(arm, "transition", None), "target_id", None)
        == binding.target_id
        == reservation.target_id,
        getattr(getattr(arm, "transition", None), "target_spec_digest", None)
        == binding.target_spec_digest,
        getattr(arm, "digest", None) == binding.marginal_arm_digest,
        prepared.launch.digest == binding.candidate_launch_digest,
        getattr(
            getattr(getattr(arm, "transition", None), "replacement", None),
            "digest",
            None,
        )
        == binding.contribution_ref_digest,
        getattr(
            getattr(getattr(arm, "transition", None), "replacement", None),
            "artifact_digest",
            None,
        )
        == candidate.publication.content_hash,
        member_ids == tuple(reservation.target_members),
        authority.graph_evidence_ref.binding == binding,
        authority.graph_evidence_ref.requirement_digest
        == authority.graph_requirement.digest,
        raw_type
        == (
            GRAPH_EVIDENCE_DOMAIN,
            GRAPH_EVIDENCE_MEDIA_TYPE,
            GRAPH_EVIDENCE_SCHEMA,
        ),
    )
    if not all(exact):
        raise B300QualificationGraphGateError(
            "graph gate plan differs from its registered candidate authority"
        )
    return candidate, reservation, authority


def _raw_grade(
    evidence_root: Path,
    authority: CandidateQualificationAuthority,
) -> GraphVerificationGrade | None:
    try:
        return reopen_graph_verification(
            evidence_root,
            authority.graph_artifact_ref,
            authority.graph_requirement,
            authority.graph_evidence_ref,
        )
    except QualificationError:
        return None


def _hold_for_grade(
    grade: GraphVerificationGrade | None,
    *,
    request_digest: str,
    authority_digest: str,
) -> B300QualificationGraphGateHold | None:
    if grade is None:
        return qualification_graph_gate_hold(
            RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE,
            authenticated_request_digest=request_digest,
            authority_context_digest=authority_digest,
            code=B300QualificationGraphHoldCode.RAW_EVIDENCE_UNAVAILABLE,
        )
    if grade.decision is QualificationDecision.NO_DECISION:
        return qualification_graph_gate_hold(
            RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE,
            authenticated_request_digest=request_digest,
            authority_context_digest=authority_digest,
            code=B300QualificationGraphHoldCode.RAW_EVIDENCE_INCOMPLETE,
        )
    return None


def _publication_failure_hold(
    evidence_root: Path,
    authority: CandidateQualificationAuthority,
    *,
    request_digest: str,
    authority_digest: str,
) -> B300QualificationGraphGateHold:
    grade = _raw_grade(evidence_root, authority)
    typed = _hold_for_grade(
        grade,
        request_digest=request_digest,
        authority_digest=authority_digest,
    )
    if typed is not None:
        return typed
    return qualification_graph_gate_hold(
        RemoteQualificationHoldReason.GRAPH_EXIT_PUBLICATION_AMBIGUOUS,
        authenticated_request_digest=request_digest,
        authority_context_digest=authority_digest,
        code=B300QualificationGraphHoldCode.EXIT_PUBLICATION_AMBIGUOUS,
    )


def run_b300_qualification_graph_gate(
    factory: QualificationPlanFactory,
    plan: CausalQualificationInput,
    *,
    evidence_root: Path,
    candidates: tuple[ArenaCandidateBinding, ...],
    authenticated_request_digest: str,
) -> B300QualificationGraphGateResult:
    """Reopen and grade raw graph evidence before any qualification execution."""

    request_digest = _digest(
        authenticated_request_digest, "authenticated request digest"
    )
    candidate, reservation, authority = _context(
        factory,
        plan,
        evidence_root,
        candidates,
    )
    authority_context_digest = factory.manifest.digest
    grade = _raw_grade(evidence_root, authority)
    hold = _hold_for_grade(
        grade,
        request_digest=request_digest,
        authority_digest=authority_context_digest,
    )
    if hold is not None:
        return hold
    assert grade is not None
    raw_reference = authority.graph_artifact_ref
    if grade.decision is QualificationDecision.PASS:
        return B300QualificationGraphGatePass(
            plan,
            factory,
            _references((raw_reference,)),
        )
    if grade.decision is not QualificationDecision.FAIL:
        return qualification_graph_gate_hold(
            RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE,
            authenticated_request_digest=request_digest,
            authority_context_digest=authority_context_digest,
            code=B300QualificationGraphHoldCode.RAW_EVIDENCE_INCOMPLETE,
        )
    kwargs = {
        "expected_plan": plan,
        "expected_authority": authority,
        "expected_reservation": reservation,
        "authenticated_request_digest": request_digest,
        "expected_candidate_binding": candidate,
    }
    try:
        exit_reference = publish_qualification_graph_exit(
            evidence_root,
            **kwargs,
        )
        graph_exit = reopen_qualification_graph_exit(
            evidence_root,
            exit_reference,
            **kwargs,
        )
    except (QualificationGraphExitError, QualificationGraphExitHold):
        return _publication_failure_hold(
            evidence_root,
            authority,
            request_digest=request_digest,
            authority_digest=authority_context_digest,
        )
    outcome = QualificationIntakeOutcome(
        reservation.reservation_digest,
        reservation.selected_delta_digest,
        factory.manifest.digest,
        QualificationDecision.FAIL,
        graph_exit.terminal_reason,
        False,
        attempt_artifact_sha256=exit_reference.sha256,
        report_digest=graph_exit.digest,
    )
    batch = QualificationIntakeBatch(
        factory.manifest.digest,
        (outcome,),
        exit_reference,
    )
    refs = _references(tuple(sorted((raw_reference, exit_reference), key=_evidence_key)))
    return B300QualificationGraphGateFail(
        plan,
        factory,
        batch,
        exit_reference,
        graph_exit,
        refs,
    )


__all__ = [
    "B300QualificationGraphGateError",
    "B300QualificationGraphGateFail",
    "B300QualificationGraphGateHold",
    "B300QualificationGraphGatePass",
    "B300QualificationGraphGateResult",
    "B300QualificationGraphHoldCode",
    "GRAPH_GATE_DIAGNOSTIC_SCHEMA",
    "qualification_graph_gate_hold",
    "run_b300_qualification_graph_gate",
]

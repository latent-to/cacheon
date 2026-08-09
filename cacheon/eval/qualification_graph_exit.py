"""A durable graph-only FAIL for one registered qualification candidate.

PASS continues to later qualification.  Missing, incomplete, or unavailable
graph evidence is a HOLD.  This closed product cannot carry later-stage
witnesses and is accepted only after reopening and regrading the raw graph CAS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef, EvidenceStoreError, publish_evidence, reopen_evidence,
)
from cacheon.eval.qualification import (
    GRAPH_EVIDENCE_DOMAIN, GRAPH_EVIDENCE_MEDIA_TYPE, GRAPH_EVIDENCE_SCHEMA,
    GraphVariantRequirement, GraphVerificationBinding,
    GraphVerificationGrade, GraphVerificationMemberBinding,
    GraphVerificationRequirement, QualificationDecision, QualificationError,
    reopen_graph_verification,
)
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.eval.qualification_runner import (
    CandidateQualificationAuthority, CausalQualificationInput,
    qualification_authority_digest,
)
from cacheon.stack_identity import (
    StackIdentityError, canonical_digest, canonical_json_bytes, require_sha256_hex,
)


QUALIFICATION_GRAPH_EXIT_DOMAIN = "qualification.graph-exit"
QUALIFICATION_GRAPH_EXIT_MEDIA_TYPE = "application/vnd.cacheon.qualification-graph-exit+json"
QUALIFICATION_GRAPH_EXIT_SCHEMA = "cacheon.qualification.graph-exit.v1"
MAX_QUALIFICATION_GRAPH_EXIT_BYTES = 256 << 10


class QualificationGraphExitError(ValueError):
    """The API input or retained terminal record is inconsistent."""


class QualificationGraphExitHold(RuntimeError):
    """Graph evidence is unavailable or inconclusive; never candidate FAIL."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise QualificationGraphExitError(str(exc)) from None


@dataclass(frozen=True)
class QualificationGraphExit:
    """The sole path-free graph terminal artifact."""

    authenticated_request_digest: str
    qualification_authority_digest: str
    source_digest: str
    reservation_digest: str
    selected_delta_digest: str
    candidate_publication_binding_digest: str
    graph_requirement: GraphVerificationRequirement
    graph_requirement_digest: str
    graph_artifact_ref: EvidenceArtifactRef
    graph_evidence_ref_digest: str
    graph_grade: GraphVerificationGrade
    decision: QualificationDecision
    terminal_reason: str

    def __post_init__(self) -> None:
        digest_fields = (
            "authenticated_request_digest", "qualification_authority_digest",
            "source_digest", "reservation_digest", "selected_delta_digest",
            "candidate_publication_binding_digest", "graph_requirement_digest",
            "graph_evidence_ref_digest",
        )
        for name in digest_fields:
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        exact_types = (
            type(self.graph_requirement) is GraphVerificationRequirement,
            type(self.graph_artifact_ref) is EvidenceArtifactRef,
            type(self.graph_grade) is GraphVerificationGrade,
            type(self.decision) is QualificationDecision,
        )
        if not all(exact_types):
            raise QualificationGraphExitError("graph exit contains an unsupported type")
        artifact_type = (
            self.graph_artifact_ref.domain, self.graph_artifact_ref.media_type,
            self.graph_artifact_ref.schema,
        )
        if artifact_type != (
            GRAPH_EVIDENCE_DOMAIN, GRAPH_EVIDENCE_MEDIA_TYPE, GRAPH_EVIDENCE_SCHEMA,
        ):
            raise QualificationGraphExitError("graph exit names another evidence type")
        identities = (
            self.graph_requirement.digest == self.graph_requirement_digest,
            self.graph_requirement.binding.selected_delta_digest
            == self.selected_delta_digest,
            self.graph_grade.requirement_digest == self.graph_requirement_digest,
            self.graph_grade.evidence_ref_digest == self.graph_evidence_ref_digest,
        )
        if not all(identities):
            raise QualificationGraphExitError("graph exit graph identities disagree")
        if (
            self.graph_grade.decision is not QualificationDecision.FAIL
            or self.decision is not self.graph_grade.decision
            or self.terminal_reason != self.graph_grade.reason
        ):
            raise QualificationGraphExitError("graph exit is not derived from graph FAIL")

    def to_dict(self) -> dict[str, object]:
        row = {field.name: getattr(self, field.name) for field in fields(self)}
        row.update(
            decision=self.decision.value,
            graph_artifact_ref=self.graph_artifact_ref.to_dict(),
            graph_grade=self.graph_grade.to_dict(),
            graph_requirement=self.graph_requirement.to_dict(),
        )
        _assert_safe_payload_keys(row)
        return row

    @classmethod
    def from_dict(cls, value: object) -> "QualificationGraphExit":
        expected = frozenset(field.name for field in fields(cls))
        if type(value) is not dict or set(value) != expected:
            raise QualificationGraphExitError("graph exit fields do not match the schema")
        try:
            result = cls(**{
                **value,
                "decision": QualificationDecision(value["decision"]),
                "graph_artifact_ref": EvidenceArtifactRef.from_dict(
                    value["graph_artifact_ref"]
                ),
                "graph_grade": GraphVerificationGrade.from_dict(value["graph_grade"]),
                "graph_requirement": GraphVerificationRequirement.from_dict(
                    value["graph_requirement"]
                ),
            })
        except QualificationGraphExitError:
            raise
        except (TypeError, ValueError) as exc:
            raise QualificationGraphExitError(f"graph exit is malformed: {exc}") from None
        if result.to_dict() != value:
            raise QualificationGraphExitError("graph exit is not canonically represented")
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(QUALIFICATION_GRAPH_EXIT_SCHEMA, self.to_dict())


_SCHEMA_FIELDS = (
    (QualificationGraphExit, "authenticated_request_digest qualification_authority_digest "
     "source_digest reservation_digest selected_delta_digest "
     "candidate_publication_binding_digest graph_requirement graph_requirement_digest "
     "graph_artifact_ref graph_evidence_ref_digest graph_grade decision terminal_reason"),
    (GraphVerificationRequirement,
     "binding variants expected_graph_replays policy_version schema_version"),
    (GraphVerificationBinding, "marginal_arm_digest candidate_launch_digest "
     "contribution_ref_digest selected_delta_digest target_id target_spec_digest "
     "catalog_digest members verification_policy_digest"),
    (GraphVerificationMemberBinding,
     "slot_id target_spec_digest contract_digest verification_profile_id"),
    (GraphVariantRequirement, "slot_id variant_id shape_descriptor_digests "
     "context_applicable applicable_shape_descriptor_digests"),
    (EvidenceArtifactRef, "domain sha256 size media_type schema"),
    (GraphVerificationGrade,
     "decision reason requirement_digest evidence_ref_digest raw_evidence_digest"),
)
_FORBIDDEN_SCHEMA_WORDS = frozenset(
    "speed pair audit quality nll count settlement weight weights path".split()
)


def assert_qualification_graph_exit_schema_safe() -> None:
    """Refuse schema growth into any later-stage or filesystem carrier."""

    for record_type, expected_names in _SCHEMA_FIELDS:
        actual = frozenset(field.name for field in fields(record_type))
        if actual != frozenset(expected_names.split()):
            raise RuntimeError(f"{record_type.__name__} changed without schema review")
        if any(_FORBIDDEN_SCHEMA_WORDS & set(name.lower().split("_")) for name in actual):
            raise RuntimeError(f"{record_type.__name__} contains a forbidden field")


def _assert_safe_payload_keys(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or _FORBIDDEN_SCHEMA_WORDS & set(key.lower().split("_")):
                raise QualificationGraphExitError("graph exit has a forbidden field")
            _assert_safe_payload_keys(item)
    elif type(value) is list:
        for item in value:
            _assert_safe_payload_keys(item)


assert_qualification_graph_exit_schema_safe()


def _expected_context(
    plan: CausalQualificationInput,
    authority: CandidateQualificationAuthority,
    reservation: QualificationReservation,
    request_digest: str,
    candidate: ArenaCandidateBinding,
) -> tuple[str, str, str, str]:
    if type(plan) is not CausalQualificationInput:
        raise QualificationGraphExitError("expected qualification plan is not exact")
    if type(authority) is not CandidateQualificationAuthority:
        raise QualificationGraphExitError("graph exit supports registered authority only")
    if type(reservation) is not QualificationReservation:
        raise QualificationGraphExitError("expected reservation is not exact")
    if type(candidate) is not ArenaCandidateBinding:
        raise QualificationGraphExitError("expected candidate binding is not exact")
    request = _digest(request_digest, "authenticated request digest")
    matches = tuple(
        index for index, row in enumerate(plan.candidates)
        if row.selected_delta_digest == authority.selected_delta_digest
    )
    if (
        len(matches) != 1 or plan.candidates[matches[0]] != authority
        or candidate.reservation != reservation
    ):
        raise QualificationGraphExitError("plan, authority, binding, and reservation differ")
    prepared = plan.prepared.candidates[matches[0]]
    requirement, binding = authority.graph_requirement, authority.graph_requirement.binding
    member_ids = tuple(member.slot_id for member in binding.members)
    exact = (
        reservation.selected_delta_digest == authority.selected_delta_digest,
        reservation.target_id == binding.target_id,
        reservation.target_members == member_ids,
        prepared.arm.selected_delta_digest == binding.selected_delta_digest,
        prepared.arm.transition.target_id == binding.target_id,
        prepared.arm.transition.target_spec_digest == binding.target_spec_digest,
        prepared.arm.digest == binding.marginal_arm_digest,
        prepared.arm.contribution_digest == binding.contribution_ref_digest,
        prepared.launch.digest == binding.candidate_launch_digest,
        authority.graph_evidence_ref.binding == binding,
        authority.graph_evidence_ref.requirement_digest == requirement.digest,
        (authority.graph_artifact_ref.domain, authority.graph_artifact_ref.media_type,
         authority.graph_artifact_ref.schema)
        == (GRAPH_EVIDENCE_DOMAIN, GRAPH_EVIDENCE_MEDIA_TYPE, GRAPH_EVIDENCE_SCHEMA),
    )
    if not all(exact):
        raise QualificationGraphExitError("graph authority differs from candidate context")
    try:
        authority_digest = _digest(
            qualification_authority_digest(plan), "qualification authority digest"
        )
        source = _digest(plan.prepared.source.digest, "qualification source digest")
        candidate_digest = _digest(candidate.digest, "candidate publication binding digest")
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise QualificationGraphExitError(f"graph context cannot be derived: {exc}") from None
    return request, authority_digest, source, candidate_digest


def _terminal_grade(root: object, authority: CandidateQualificationAuthority) -> GraphVerificationGrade:
    try:
        grade = reopen_graph_verification(
            root, authority.graph_artifact_ref, authority.graph_requirement,
            authority.graph_evidence_ref,
        )
    except QualificationError as exc:
        raise QualificationGraphExitHold(f"raw graph evidence cannot reopen: {exc}") from None
    if grade.decision is QualificationDecision.NO_DECISION:
        raise QualificationGraphExitHold(f"raw graph evidence is inconclusive: {grade.reason}")
    if grade.decision is QualificationDecision.PASS:
        raise QualificationGraphExitError("graph PASS continues; it is not terminal")
    if grade.decision is not QualificationDecision.FAIL:
        raise QualificationGraphExitHold("raw graph evidence has no decision")
    return grade


def publish_qualification_graph_exit(
    evidence_root: object, *, expected_plan: CausalQualificationInput,
    expected_authority: CandidateQualificationAuthority,
    expected_reservation: QualificationReservation,
    authenticated_request_digest: str,
    expected_candidate_binding: ArenaCandidateBinding,
) -> EvidenceArtifactRef:
    """Publish only after raw graph bytes independently regrade to FAIL."""

    request, authority_digest, source, candidate = _expected_context(
        expected_plan, expected_authority, expected_reservation,
        authenticated_request_digest, expected_candidate_binding,
    )
    grade = _terminal_grade(evidence_root, expected_authority)
    result = QualificationGraphExit(
        authenticated_request_digest=request,
        qualification_authority_digest=authority_digest,
        source_digest=source,
        reservation_digest=expected_reservation.reservation_digest,
        selected_delta_digest=expected_authority.selected_delta_digest,
        candidate_publication_binding_digest=candidate,
        graph_requirement=expected_authority.graph_requirement,
        graph_requirement_digest=expected_authority.graph_requirement.digest,
        graph_artifact_ref=expected_authority.graph_artifact_ref,
        graph_evidence_ref_digest=expected_authority.graph_evidence_ref.digest,
        graph_grade=grade,
        decision=grade.decision,
        terminal_reason=grade.reason,
    )
    payload = canonical_json_bytes(result.to_dict())
    if len(payload) > MAX_QUALIFICATION_GRAPH_EXIT_BYTES:
        raise QualificationGraphExitError("graph exit exceeds its size bound")
    try:
        return publish_evidence(
            evidence_root, payload, domain=QUALIFICATION_GRAPH_EXIT_DOMAIN,
            media_type=QUALIFICATION_GRAPH_EXIT_MEDIA_TYPE,
            schema=QUALIFICATION_GRAPH_EXIT_SCHEMA,
            max_bytes=MAX_QUALIFICATION_GRAPH_EXIT_BYTES,
        )
    except EvidenceStoreError as exc:
        raise QualificationGraphExitHold(f"graph exit cannot publish: {exc}") from None


def _canonical_payload(payload: bytes) -> dict[str, object]:
    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise QualificationGraphExitError(f"graph exit repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise QualificationGraphExitError(f"graph exit contains nonfinite {value}")

    try:
        decoded = json.loads(payload.decode(), object_pairs_hook=strict_object,
                             parse_constant=reject_constant)
        if type(decoded) is not dict or canonical_json_bytes(decoded) != payload:
            raise QualificationGraphExitError("graph exit bytes are not canonical")
    except QualificationGraphExitError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, StackIdentityError) as exc:
        raise QualificationGraphExitError(f"graph exit JSON is invalid: {exc}") from None
    return decoded


def reopen_qualification_graph_exit(
    evidence_root: object, reference: EvidenceArtifactRef, *,
    expected_plan: CausalQualificationInput,
    expected_authority: CandidateQualificationAuthority,
    expected_reservation: QualificationReservation,
    authenticated_request_digest: str,
    expected_candidate_binding: ArenaCandidateBinding,
) -> QualificationGraphExit:
    """Reopen, rebind, and independently regrade one graph-only terminal."""

    expected = _expected_context(
        expected_plan, expected_authority, expected_reservation,
        authenticated_request_digest, expected_candidate_binding,
    )
    artifact_type = (
        getattr(reference, "domain", None), getattr(reference, "media_type", None),
        getattr(reference, "schema", None),
    )
    if type(reference) is not EvidenceArtifactRef or artifact_type != (
        QUALIFICATION_GRAPH_EXIT_DOMAIN, QUALIFICATION_GRAPH_EXIT_MEDIA_TYPE,
        QUALIFICATION_GRAPH_EXIT_SCHEMA,
    ):
        raise QualificationGraphExitError("graph exit artifact has another domain/schema")
    try:
        payload = reopen_evidence(
            evidence_root, reference, max_bytes=MAX_QUALIFICATION_GRAPH_EXIT_BYTES
        )
    except EvidenceStoreError as exc:
        raise QualificationGraphExitHold(f"graph exit artifact cannot reopen: {exc}") from None
    result = QualificationGraphExit.from_dict(_canonical_payload(payload))
    request, authority_digest, source, candidate = expected
    exact = (
        result.authenticated_request_digest == request,
        result.qualification_authority_digest == authority_digest,
        result.source_digest == source,
        result.reservation_digest == expected_reservation.reservation_digest,
        result.selected_delta_digest == expected_authority.selected_delta_digest,
        result.candidate_publication_binding_digest == candidate,
        result.graph_requirement == expected_authority.graph_requirement,
        result.graph_requirement_digest == expected_authority.graph_requirement.digest,
        result.graph_artifact_ref == expected_authority.graph_artifact_ref,
        result.graph_evidence_ref_digest == expected_authority.graph_evidence_ref.digest,
    )
    if not all(exact):
        raise QualificationGraphExitError("graph exit differs from expected authority")
    grade = _terminal_grade(evidence_root, expected_authority)
    if (
        result.graph_grade != grade or result.decision is not grade.decision
        or result.terminal_reason != grade.reason
    ):
        raise QualificationGraphExitError("graph exit does not independently regrade")
    return result

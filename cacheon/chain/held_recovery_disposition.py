"""Closed operator review for one legacy screen-only qualification HOLD.

This is deliberately separate from an authenticated pre-resident refusal.  A
legacy screen-only adapter did not emit that newer proof.  The only authority
modeled here is an explicit review of already-verified, immutable inputs; it
does not authenticate new evidence and it never authorizes reuse of the old
request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cacheon.chain.evaluation_recovery import (
    EvaluationRecovery,
    EvaluationRecoveryEvent,
    RecoveryEventType,
    RecoveryPhase,
    RecoveryResolution,
    reviewed_legacy_screen_only_release_reason,
)
from cacheon.chain.remote_worker_request_plan import (
    PlannedQualificationObservation,
    QualificationRequestPlan,
)
from cacheon.chain.remote_worker_registration import verify_registration
from cacheon.chain.remote_worker_spool import (
    ADAPTER_RESULT_FIELDS,
    SCHEMA_ADAPTER_RESULT,
    RemoteWorkerError,
    spool_canonical_json,
    strict_json_object,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex, sha256_hex


LEGACY_SCREEN_ONLY_FAILURE = "adapter_request_failed"
LEGACY_SCREEN_ONLY_DECISION = "NO_DECISION"
LEGACY_RESIDENT_MARKER_STATE = "absent"
LEGACY_SCREEN_ONLY_REVIEW_BASIS = (
    "registered_screen_only_adapter_rejected_qualification_before_"
    "qualification_resident_implementation_existed.v1"
)
_DISPOSITION_DOMAIN = "cacheon.reviewed-legacy-screen-only-disposition.v1"
_LEGACY_RESULT_FIELDS = frozenset({"failure_code", "request_id", "state"})


class HeldRecoveryDispositionError(ValueError):
    """A reviewed legacy disposition is malformed or bound elsewhere."""


def _digest(value: str, field_name: str) -> str:
    try:
        return require_sha256_hex(value, field=field_name)
    except (TypeError, ValueError) as exc:
        raise HeldRecoveryDispositionError(str(exc)) from None


def _canonical_object(payload: bytes, name: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        raise HeldRecoveryDispositionError(f"{name} bytes are malformed")
    try:
        value = strict_json_object(payload.decode("utf-8"))
        if payload != spool_canonical_json(value) + b"\n":
            raise HeldRecoveryDispositionError(f"{name} bytes are not canonical")
    except (RemoteWorkerError, UnicodeError) as exc:
        raise HeldRecoveryDispositionError(f"{name} bytes are malformed: {exc}") from None
    return value


def _plan_matches_recovery(
    plan: QualificationRequestPlan, recovery: EvaluationRecovery
) -> bool:
    planned = plan.lease
    current = recovery.lease
    return (
        plan.plan_digest == recovery.plan_digest
        and plan.request_id == recovery.request_id
        and planned.lease_id == current.lease_id
        and planned.generation == current.generation
        and planned.stage == current.stage
        and planned.owner == current.owner
        and planned.members == current.members
        and planned.claimed_block == current.claimed_block
        and planned.initial_expires_block == current.initial_expires_block
        and planned.expires_block <= current.expires_block
    )


def _has_request_ready_origin(events: tuple[EvaluationRecoveryEvent, ...]) -> bool:
    """Require REQUEST_READY, then only lease renewals, immediately before HOLD."""

    index = len(events) - 2
    while index >= 0 and events[index].event_type is RecoveryEventType.RENEWED:
        if (
            events[index].phase is not RecoveryPhase.REQUEST_READY
            or events[index].resolution is not RecoveryResolution.UNRESOLVED
        ):
            return False
        index -= 1
    return (
        index >= 0
        and events[index].event_type is RecoveryEventType.REQUEST_READY
        and events[index].phase is RecoveryPhase.REQUEST_READY
        and events[index].resolution is RecoveryResolution.UNRESOLVED
    )


@dataclass(frozen=True)
class ReviewedLegacyScreenOnlyDisposition:
    """One exact review of a legacy, verified screen-only adapter result.

    ``observation`` is the typed result of the normal request-plan inspector.
    The raw result bytes are retained here only so their canonical digests and
    legacy payload shape can be independently rebound.  The operator-review
    digests name the authority and evidence bundle used to decide that the
    registered adapter had no qualification resident implementation.
    """

    recovery: EvaluationRecovery
    plan: QualificationRequestPlan
    observation: PlannedQualificationObservation
    registration_bytes: bytes = field(repr=False)
    result_envelope_bytes: bytes = field(repr=False)
    adapter_result_bytes: bytes = field(repr=False)
    result_envelope_digest: str
    adapter_result_blob_digest: str
    operator_review_authority_digest: str
    operator_review_evidence_digest: str
    failure_code: str = LEGACY_SCREEN_ONLY_FAILURE
    decision: str = LEGACY_SCREEN_ONLY_DECISION
    resident_marker_state: str = LEGACY_RESIDENT_MARKER_STATE
    review_basis: str = LEGACY_SCREEN_ONLY_REVIEW_BASIS

    def __post_init__(self) -> None:
        if (
            type(self.recovery) is not EvaluationRecovery
            or self.recovery.phase is not RecoveryPhase.HELD
            or self.recovery.resolution is not RecoveryResolution.UNRESOLVED
            or not self.recovery.request_plan
            or type(self.plan) is not QualificationRequestPlan
            or not _plan_matches_recovery(self.plan, self.recovery)
            or type(self.observation) is not PlannedQualificationObservation
            or self.observation.plan_digest != self.plan.plan_digest
            or self.observation.request_id != self.plan.request_id
            or self.observation.state != "result_ready"
            or self.observation.failure_code != LEGACY_SCREEN_ONLY_FAILURE
            or self.observation.response is not None
            or self.observation.refusal is not None
            or self.failure_code != LEGACY_SCREEN_ONLY_FAILURE
            or self.decision != LEGACY_SCREEN_ONLY_DECISION
            or self.resident_marker_state != LEGACY_RESIDENT_MARKER_STATE
            or self.review_basis != LEGACY_SCREEN_ONLY_REVIEW_BASIS
        ):
            raise HeldRecoveryDispositionError(
                "legacy screen-only disposition is not the exact reviewed state"
            )
        _digest(
            self.result_envelope_digest,
            "authenticated result envelope digest",
        )
        _digest(
            self.adapter_result_blob_digest,
            "authenticated adapter result blob digest",
        )
        _digest(
            self.operator_review_authority_digest,
            "operator review authority digest",
        )
        _digest(self.operator_review_evidence_digest, "operator review evidence digest")
        self._verify_registration()
        self._verify_result_bytes()

    @classmethod
    def review(
        cls,
        *,
        recovery: EvaluationRecovery,
        plan: QualificationRequestPlan,
        observation: PlannedQualificationObservation,
        registration: object,
        result_envelope_bytes: bytes,
        adapter_result_bytes: bytes,
        result_envelope_digest: str,
        adapter_result_blob_digest: str,
        operator_review_authority_digest: str,
        operator_review_evidence_digest: str,
    ) -> "ReviewedLegacyScreenOnlyDisposition":
        """Build from the exact verified registration and inspected result."""

        try:
            verified = verify_registration(registration)
            registration_bytes = spool_canonical_json(verified) + b"\n"
        except RemoteWorkerError as exc:
            raise HeldRecoveryDispositionError(
                f"reviewed worker registration is invalid: {exc}"
            ) from None
        return cls(
            recovery=recovery,
            plan=plan,
            observation=observation,
            registration_bytes=registration_bytes,
            result_envelope_bytes=result_envelope_bytes,
            adapter_result_bytes=adapter_result_bytes,
            result_envelope_digest=result_envelope_digest,
            adapter_result_blob_digest=adapter_result_blob_digest,
            operator_review_authority_digest=operator_review_authority_digest,
            operator_review_evidence_digest=operator_review_evidence_digest,
        )

    def _verify_registration(self) -> None:
        registration = _canonical_object(
            self.registration_bytes, "reviewed worker registration"
        )
        try:
            verified = verify_registration(registration)
        except RemoteWorkerError as exc:
            raise HeldRecoveryDispositionError(
                f"reviewed worker registration is invalid: {exc}"
            ) from None
        request = self.plan.remote_request
        if (
            verified["registration_digest"] != self.plan.registration_digest
            or verified["worker_epoch"] != self.plan.worker_epoch
            or verified["transport_identity_digest"]
            != self.plan.transport_identity_digest
            or verified["credential_digest"] != self.plan.credential_digest
            or verified["worker_readiness_digest"]
            != request.worker_readiness_digest
            or verified["ready_receipt_digest"] != request.ready_receipt_digest
            or verified["service_identity"] != request.service_identity
        ):
            raise HeldRecoveryDispositionError(
                "reviewed registration differs from the retained request plan"
            )

    def _verify_result_bytes(self) -> None:
        result = _canonical_object(
            self.result_envelope_bytes, "reviewed adapter result envelope"
        )
        payload = _canonical_object(
            self.adapter_result_bytes, "reviewed legacy adapter result"
        )
        if (
            set(result) != ADAPTER_RESULT_FIELDS
            or result.get("schema") != SCHEMA_ADAPTER_RESULT
            or result.get("state") != "no_decision"
            or result.get("failure_code") != LEGACY_SCREEN_ONLY_FAILURE
            or result.get("request_id") != self.plan.request_id
            or result.get("response_digest") is not None
            or result.get("response_sha256") is not None
            or set(payload) != _LEGACY_RESULT_FIELDS
            or payload.get("state") != "no_decision"
            or payload.get("failure_code") != LEGACY_SCREEN_ONLY_FAILURE
            or payload.get("request_id") != self.plan.request_id
        ):
            raise HeldRecoveryDispositionError(
                "reviewed result is not the exact legacy screen-only NO_DECISION"
            )
        artifacts = result.get("artifacts")
        if type(artifacts) is not list:
            raise HeldRecoveryDispositionError("reviewed result artifacts are malformed")
        matching = [
            row
            for row in artifacts
            if type(row) is dict and row.get("role") == "adapter_result"
        ]
        blob_digest = sha256_hex(self.adapter_result_bytes)
        if (
            self.result_envelope_digest != sha256_hex(self.result_envelope_bytes)
            or self.adapter_result_blob_digest != blob_digest
            or len(matching) != 1
            or set(matching[0]) != {"role", "sha256", "size"}
            or matching[0]["sha256"] != blob_digest
            or matching[0]["size"] != len(self.adapter_result_bytes)
        ):
            raise HeldRecoveryDispositionError(
                "reviewed result does not bind its exact legacy payload blob"
            )

    @property
    def registration_digest(self) -> str:
        return self.plan.registration_digest

    @property
    def adapter_sha256(self) -> str:
        return str(
            _canonical_object(self.registration_bytes, "reviewed worker registration")[
                "adapter_sha256"
            ]
        )

    @property
    def remote_service_sha256(self) -> str:
        return str(
            _canonical_object(self.registration_bytes, "reviewed worker registration")[
                "remote_service_sha256"
            ]
        )

    @property
    def service_identity(self) -> str:
        return self.plan.remote_request.service_identity

    @property
    def held_reason_digest(self) -> str:
        return sha256_hex(self.recovery.reason.encode("utf-8"))

    @property
    def digest(self) -> str:
        return canonical_digest(
            _DISPOSITION_DOMAIN,
            {
                "adapter_result_blob_digest": self.adapter_result_blob_digest,
                "adapter_sha256": self.adapter_sha256,
                "decision": self.decision,
                "failure_code": self.failure_code,
                "held_reason_digest": self.held_reason_digest,
                "lease_generation": self.recovery.lease.generation,
                "lease_id": self.recovery.lease.lease_id,
                "observation_dispatch_state": self.observation.dispatch_state,
                "observation_state": self.observation.state,
                "operator_review_authority_digest": (
                    self.operator_review_authority_digest
                ),
                "operator_review_evidence_digest": self.operator_review_evidence_digest,
                "plan_digest": self.plan.plan_digest,
                "registration_digest": self.registration_digest,
                "remote_service_sha256": self.remote_service_sha256,
                "resident_marker_state": self.resident_marker_state,
                "request_id": self.plan.request_id,
                "result_envelope_digest": self.result_envelope_digest,
                "review_basis": self.review_basis,
                "recovery_id": self.recovery.recovery_id,
                "recovery_revision": self.recovery.revision,
                "service_identity": self.service_identity,
                "worker_epoch": self.plan.worker_epoch,
            },
        )

    @property
    def release_reason(self) -> str:
        return reviewed_legacy_screen_only_release_reason(
            held_reason_digest=self.held_reason_digest,
            disposition_digest=self.digest,
        )

    def require_exact_store_state(
        self,
        recovery: EvaluationRecovery,
        plan: QualificationRequestPlan,
        events: tuple[EvaluationRecoveryEvent, ...],
    ) -> str:
        """Reject every stale or non-request-ready origin before store mutation."""

        if (
            type(recovery) is not EvaluationRecovery
            or recovery != self.recovery
            or type(plan) is not QualificationRequestPlan
            or plan != self.plan
            or len(events) < 2
            or events[-1].event_type is not RecoveryEventType.HELD
            or events[-1].revision != recovery.revision
            or not _has_request_ready_origin(events)
            or events[-1].reason != recovery.reason
        ):
            raise HeldRecoveryDispositionError(
                "reviewed disposition differs from the exact held store state"
            )
        return self.release_reason


__all__ = [
    "HeldRecoveryDispositionError",
    "LEGACY_SCREEN_ONLY_DECISION",
    "LEGACY_SCREEN_ONLY_FAILURE",
    "LEGACY_RESIDENT_MARKER_STATE",
    "LEGACY_SCREEN_ONLY_REVIEW_BASIS",
    "ReviewedLegacyScreenOnlyDisposition",
]

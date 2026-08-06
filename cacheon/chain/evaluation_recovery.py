"""Closed identities and phases for durable qualification-carrier recovery.

These types describe CPU orchestration ownership only.  They deliberately do
not model resident speed, quality, or product-execution phases; those require
their own authenticated evidence contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.stack_identity import canonical_digest, require_sha256_hex


_RECOVERY_DOMAIN = "cacheon.evaluation-recovery.v1"
_EVENT_DOMAIN = "cacheon.evaluation-recovery-event.v1"


class EvaluationRecoveryError(ValueError):
    """A recovery identity, transition, or retained row is malformed."""


class EvaluationRecoveryHoldError(RuntimeError):
    """Durable qualification ownership is ambiguous and must remain held."""


class RecoveryPhase(str, Enum):
    CLAIMED = "claimed"
    PREPARED = "prepared"
    PUBLICATION_COMMITTED = "publication_committed"
    REQUEST_READY = "request_ready"
    RESULT_READY = "result_ready"
    EVIDENCE_IMPORTED = "evidence_imported"
    HELD = "held"


class RecoveryResolution(str, Enum):
    UNRESOLVED = ""
    PRE_RESIDENT_RELEASED = "pre_resident_released"
    COMMITTED = "committed"


class RecoveryAction(str, Enum):
    SAME_REQUEST = "same_request"
    DOWNSTREAM_ONLY = "downstream_only"
    IMPORT_ONLY = "import_only"
    PRE_RESIDENT_RELEASE = "pre_resident_release"
    HOLD = "hold"
    COMPLETE = "complete"


class RecoveryEventType(str, Enum):
    CLAIMED = "claimed"
    PREPARED = "prepared"
    PUBLICATION_COMMITTED = "publication_committed"
    REQUEST_READY = "request_ready"
    RESULT_READY = "result_ready"
    EVIDENCE_IMPORTED = "evidence_imported"
    RENEWED = "renewed"
    HELD = "held"
    PRE_RESIDENT_RELEASED = "pre_resident_released"
    COMMITTED = "committed"


def _require_reason(reason: str, *, required: bool = False) -> str:
    if (
        not isinstance(reason, str)
        or len(reason) > 2_048
        or reason.strip() != reason
        or any(ord(char) < 32 or ord(char) == 127 for char in reason)
        or (required and not reason)
    ):
        raise EvaluationRecoveryError("evaluation recovery reason is malformed")
    return reason


def evaluation_recovery_id(lease: EvaluationLease) -> str:
    if type(lease) is not EvaluationLease or lease.stage != "qualification":
        raise EvaluationRecoveryError(
            "evaluation recovery requires an exact qualification lease"
        )
    return canonical_digest(
        _RECOVERY_DOMAIN,
        {
            "generation": lease.generation,
            "lease_id": lease.lease_id,
            "members": [member.to_dict() for member in lease.members],
            "owner": lease.owner,
        },
    )


def evaluation_recovery_event_id(
    *,
    recovery_id: str,
    lease_id: str,
    revision: int,
    event_type: RecoveryEventType,
    phase: RecoveryPhase,
    resolution: RecoveryResolution,
    finalized_block: int,
    expires_block: int,
    plan_digest: str,
    request_id: str,
    reason: str,
) -> str:
    require_sha256_hex(recovery_id, field="evaluation recovery id")
    require_sha256_hex(lease_id, field="evaluation recovery lease id")
    if (
        type(revision) is not int
        or revision < 0
        or type(event_type) is not RecoveryEventType
        or type(phase) is not RecoveryPhase
        or type(resolution) is not RecoveryResolution
        or type(finalized_block) is not int
        or finalized_block < 0
        or type(expires_block) is not int
        or expires_block <= 0
    ):
        raise EvaluationRecoveryError("evaluation recovery event input is malformed")
    _require_reason(reason)
    if bool(plan_digest) != bool(request_id):
        raise EvaluationRecoveryError("evaluation recovery plan identity is partial")
    if plan_digest:
        require_sha256_hex(plan_digest, field="evaluation recovery plan digest")
        require_sha256_hex(request_id, field="evaluation recovery request id")
    return canonical_digest(
        _EVENT_DOMAIN,
        {
            "event_type": event_type.value,
            "expires_block": expires_block,
            "finalized_block": finalized_block,
            "lease_id": lease_id,
            "phase": phase.value,
            "plan_digest": plan_digest,
            "reason": reason,
            "recovery_id": recovery_id,
            "request_id": request_id,
            "resolution": resolution.value,
            "revision": revision,
        },
    )


@dataclass(frozen=True)
class EvaluationRecovery:
    recovery_id: str
    lease: EvaluationLease
    revision: int
    phase: RecoveryPhase
    resolution: RecoveryResolution
    created_block: int
    updated_block: int
    plan_digest: str = ""
    request_id: str = ""
    request_plan: bytes = b""
    reason: str = ""

    def __post_init__(self) -> None:
        require_sha256_hex(self.recovery_id, field="evaluation recovery id")
        if (
            type(self.lease) is not EvaluationLease
            or self.lease.stage != "qualification"
            or self.recovery_id != evaluation_recovery_id(self.lease)
            or type(self.revision) is not int
            or self.revision < 0
            or type(self.phase) is not RecoveryPhase
            or type(self.resolution) is not RecoveryResolution
            or type(self.created_block) is not int
            or self.created_block < 0
            or type(self.updated_block) is not int
            or self.updated_block < self.created_block
        ):
            raise EvaluationRecoveryError("evaluation recovery is malformed")
        _require_reason(self.reason)
        plan_bound = bool(self.request_plan)
        if (
            type(self.request_plan) is not bytes
            or plan_bound != bool(self.plan_digest)
            or plan_bound != bool(self.request_id)
            or (self.phase is RecoveryPhase.CLAIMED and plan_bound)
            or (
                self.phase
                not in {RecoveryPhase.CLAIMED, RecoveryPhase.HELD}
                and not plan_bound
            )
        ):
            raise EvaluationRecoveryError("evaluation recovery request plan is malformed")
        if plan_bound:
            require_sha256_hex(
                self.plan_digest, field="evaluation recovery plan digest"
            )
            require_sha256_hex(self.request_id, field="evaluation recovery request id")
        if (
            (
                self.phase is RecoveryPhase.HELD
                or self.resolution is RecoveryResolution.PRE_RESIDENT_RELEASED
            )
            != bool(self.reason)
            or (
                self.resolution is not RecoveryResolution.UNRESOLVED
                and self.phase is RecoveryPhase.HELD
            )
        ):
            raise EvaluationRecoveryError("evaluation recovery terminal state conflicts")

    @property
    def action(self) -> RecoveryAction:
        if self.resolution is RecoveryResolution.PRE_RESIDENT_RELEASED:
            return RecoveryAction.PRE_RESIDENT_RELEASE
        if self.resolution is RecoveryResolution.COMMITTED:
            return RecoveryAction.COMPLETE
        if self.phase is RecoveryPhase.HELD:
            return RecoveryAction.HOLD
        if self.phase is RecoveryPhase.CLAIMED:
            return RecoveryAction.PRE_RESIDENT_RELEASE
        if self.phase in {
            RecoveryPhase.PREPARED,
            RecoveryPhase.PUBLICATION_COMMITTED,
            RecoveryPhase.REQUEST_READY,
        }:
            return RecoveryAction.SAME_REQUEST
        return RecoveryAction.IMPORT_ONLY


@dataclass(frozen=True)
class EvaluationRecoveryEvent:
    sequence: int
    event_id: str
    recovery_id: str
    lease_id: str
    revision: int
    event_type: RecoveryEventType
    phase: RecoveryPhase
    resolution: RecoveryResolution
    finalized_block: int
    expires_block: int
    plan_digest: str = ""
    request_id: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence <= 0:
            raise EvaluationRecoveryError("evaluation recovery event is malformed")
        require_sha256_hex(self.event_id, field="evaluation recovery event id")
        expected = evaluation_recovery_event_id(
            recovery_id=self.recovery_id,
            lease_id=self.lease_id,
            revision=self.revision,
            event_type=self.event_type,
            phase=self.phase,
            resolution=self.resolution,
            finalized_block=self.finalized_block,
            expires_block=self.expires_block,
            plan_digest=self.plan_digest,
            request_id=self.request_id,
            reason=self.reason,
        )
        if self.event_id != expected:
            raise EvaluationRecoveryError("evaluation recovery event identity is corrupt")


def valid_evaluation_recovery_event_transition(
    previous: EvaluationRecoveryEvent, event: EvaluationRecoveryEvent
) -> bool:
    """Return whether one immutable recovery event may follow another."""

    common = (
        event.revision == previous.revision + 1
        and event.finalized_block >= previous.finalized_block
        and event.expires_block >= previous.expires_block
    )
    if not common:
        return False
    same_plan = (
        event.plan_digest == previous.plan_digest
        and event.request_id == previous.request_id
    )
    if event.event_type is RecoveryEventType.RENEWED:
        return (
            same_plan
            and event.phase is previous.phase
            and event.resolution is previous.resolution
            and event.reason == previous.reason
            and event.expires_block > previous.expires_block
            and previous.resolution is RecoveryResolution.UNRESOLVED
            and previous.phase is not RecoveryPhase.HELD
        )
    if event.expires_block != previous.expires_block:
        return False
    phase_continuation = (
        previous.resolution is RecoveryResolution.UNRESOLVED
        and event.resolution is RecoveryResolution.UNRESOLVED
        and previous.phase is not RecoveryPhase.HELD
    )
    if event.event_type is RecoveryEventType.PREPARED:
        return (
            phase_continuation
            and previous.phase is RecoveryPhase.CLAIMED
            and not previous.plan_digest
            and event.phase is RecoveryPhase.PREPARED
            and bool(event.plan_digest)
            and bool(event.request_id)
            and not event.reason
        )
    if not same_plan:
        return False
    if event.event_type is RecoveryEventType.PUBLICATION_COMMITTED:
        return (
            phase_continuation
            and previous.phase is RecoveryPhase.PREPARED
            and event.phase is RecoveryPhase.PUBLICATION_COMMITTED
            and not event.reason
        )
    if event.event_type is RecoveryEventType.REQUEST_READY:
        return (
            phase_continuation
            and previous.phase is RecoveryPhase.PUBLICATION_COMMITTED
            and event.phase is RecoveryPhase.REQUEST_READY
            and not event.reason
        )
    if event.event_type is RecoveryEventType.RESULT_READY:
        return (
            phase_continuation
            and previous.phase
            in {RecoveryPhase.PUBLICATION_COMMITTED, RecoveryPhase.REQUEST_READY}
            and event.phase is RecoveryPhase.RESULT_READY
            and not event.reason
        )
    if event.event_type is RecoveryEventType.EVIDENCE_IMPORTED:
        return (
            phase_continuation
            and previous.phase is RecoveryPhase.RESULT_READY
            and event.phase is RecoveryPhase.EVIDENCE_IMPORTED
            and not event.reason
        )
    if event.event_type is RecoveryEventType.HELD:
        return (
            previous.resolution is RecoveryResolution.UNRESOLVED
            and previous.phase is not RecoveryPhase.HELD
            and event.phase is RecoveryPhase.HELD
            and event.resolution is RecoveryResolution.UNRESOLVED
            and bool(event.reason)
        )
    if event.event_type is RecoveryEventType.PRE_RESIDENT_RELEASED:
        return (
            previous.resolution is RecoveryResolution.UNRESOLVED
            and previous.phase in {RecoveryPhase.CLAIMED, RecoveryPhase.PREPARED}
            and event.phase is previous.phase
            and event.resolution is RecoveryResolution.PRE_RESIDENT_RELEASED
            and bool(event.reason)
        )
    if event.event_type is RecoveryEventType.COMMITTED:
        return (
            previous.resolution is RecoveryResolution.UNRESOLVED
            and previous.phase is not RecoveryPhase.HELD
            and event.phase is previous.phase
            and event.resolution is RecoveryResolution.COMMITTED
            and not event.reason
        )
    return False


__all__ = [
    "EvaluationRecovery",
    "EvaluationRecoveryError",
    "EvaluationRecoveryEvent",
    "EvaluationRecoveryHoldError",
    "RecoveryAction",
    "RecoveryEventType",
    "RecoveryPhase",
    "RecoveryResolution",
    "evaluation_recovery_event_id",
    "evaluation_recovery_id",
    "valid_evaluation_recovery_event_transition",
]

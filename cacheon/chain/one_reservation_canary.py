"""Bounded autonomous screen-to-qualification canary for one reservation.

This module is a fence around existing screen and qualification stages.  It is
not another evaluator and it deliberately has no settlement, incentive, weight,
service-launch, or retry authority.  A caller supplies a read-only store
observation boundary, the two existing stage callables, and a durable checkpoint
sink.  The fence journals intent before either callable can mutate durable state.

The store observation is authoritative for FIFO and terminal state.  Stage
receipts are authoritative only for the exact mutation they report.  Neither a
tick count nor a callback returning ``completed`` proves canary success: the
expected reservation must subsequently be observed in a qualification-complete
durable phase.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from cacheon.stack_identity import canonical_digest, require_sha256_hex


CANARY_CHECKPOINT_SCHEMA = "cacheon-one-reservation-canary-checkpoint-v1"
CANARY_RECEIPT_SCHEMA = "cacheon-one-reservation-canary-receipt-v1"

class OneReservationCanaryError(RuntimeError):
    """The canary boundary or one of its exact typed products is malformed."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise OneReservationCanaryError(str(exc)) from None


def _optional_digest(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field)


def _digest_tuple(value: object, field: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise OneReservationCanaryError(f"{field} is not an exact tuple")
    result = tuple(_digest(item, field) for item in value)
    if len(set(result)) != len(result):
        raise OneReservationCanaryError(f"{field} contains duplicate identities")
    return result


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise OneReservationCanaryError(f"{field} must be a positive integer")
    return value


def _finite_time(value: object, field: str) -> float:
    if (
        type(value) is bool
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise OneReservationCanaryError(f"{field} must be a finite monotonic time")
    return float(value)


def _text(value: object, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > 2_048
    ):
        raise OneReservationCanaryError(f"{field} is malformed")
    return value


class CanaryStage(str, Enum):
    SCREEN = "screen"
    QUALIFICATION = "qualification"


class CanaryReservationPhase(str, Enum):
    """Only phases needed to fence a weights-off canary."""

    PUBLISHED = "published"
    SCREENING = "screening"
    PROMOTED = "promoted"
    QUALIFYING = "qualifying"
    REPRODUCTION_PENDING = "reproduction_pending"
    QUALIFIED = "qualified"
    FAILED = "failed"
    EXPIRED = "expired"

    @property
    def qualification_complete(self) -> bool:
        return self in {
            CanaryReservationPhase.REPRODUCTION_PENDING,
            CanaryReservationPhase.QUALIFIED,
        }


class CanaryStageDisposition(str, Enum):
    PROGRESSED = "progressed"
    COMPLETED = "completed"
    HOLD = "hold"
    REQUEUE = "requeue"
    NO_WORK = "no_work"


class CanaryTerminalOutcome(str, Enum):
    COMPLETED = "completed"
    HOLD = "hold"
    REQUEUE = "requeue"
    DEADLINE = "deadline"
    MAX_TICKS = "max_ticks"
    MAX_STAGE_RECEIPTS = "max_stage_receipts"
    WRONG_FIFO_HEAD = "wrong_fifo_head"
    IDENTITY_DRIFT = "identity_drift"
    LEASE_DRIFT = "lease_drift"
    REQUEST_DRIFT = "request_drift"
    SECOND_CLAIM = "second_claim"
    REPEATED_STAGE_RECEIPT = "repeated_stage_receipt"
    REPEATED_EXPENSIVE_STAGE = "repeated_expensive_stage"
    NO_WORK = "no_work"


@dataclass(frozen=True)
class CanaryLeaseObservation:
    """Read-only identity of the one active screen or qualification lease."""

    stage: CanaryStage
    lease_id: str
    reservation_digests: tuple[str, ...]
    request_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.stage) is not CanaryStage:
            raise OneReservationCanaryError("observed lease stage is not exactly typed")
        object.__setattr__(self, "lease_id", _digest(self.lease_id, "lease id"))
        members = _digest_tuple(self.reservation_digests, "lease reservation digest")
        if not members:
            raise OneReservationCanaryError("observed lease has no reservations")
        object.__setattr__(self, "reservation_digests", members)
        object.__setattr__(
            self, "request_id", _optional_digest(self.request_id, "lease request id")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "request_id": self.request_id,
            "reservation_digests": list(self.reservation_digests),
            "stage": self.stage.value,
        }


@dataclass(frozen=True)
class CanaryStoreObservation:
    """One closed, read-only view used before or after a stage mutation."""

    reservation_digest: str
    target_profile_digest: str
    phase: CanaryReservationPhase
    fifo_head_reservation_digest: str | None
    next_qualification_reservation_digests: tuple[str, ...] = ()
    active_lease: CanaryLeaseObservation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reservation_digest",
            _digest(self.reservation_digest, "observed reservation digest"),
        )
        object.__setattr__(
            self,
            "target_profile_digest",
            _digest(self.target_profile_digest, "target profile digest"),
        )
        if type(self.phase) is not CanaryReservationPhase:
            raise OneReservationCanaryError("reservation phase is not exactly typed")
        object.__setattr__(
            self,
            "fifo_head_reservation_digest",
            _optional_digest(
                self.fifo_head_reservation_digest, "FIFO head reservation digest"
            ),
        )
        object.__setattr__(
            self,
            "next_qualification_reservation_digests",
            _digest_tuple(
                self.next_qualification_reservation_digests,
                "qualification preview reservation digest",
            ),
        )
        if self.active_lease is not None and type(self.active_lease) is not CanaryLeaseObservation:
            raise OneReservationCanaryError("active lease is not exactly typed")

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.one-reservation-canary-store-observation.v1",
            self.to_dict(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "active_lease": (
                None if self.active_lease is None else self.active_lease.to_dict()
            ),
            "fifo_head_reservation_digest": self.fifo_head_reservation_digest,
            "next_qualification_reservation_digests": list(
                self.next_qualification_reservation_digests
            ),
            "phase": self.phase.value,
            "reservation_digest": self.reservation_digest,
            "target_profile_digest": self.target_profile_digest,
        }


@dataclass(frozen=True)
class CanaryStageReceipt:
    """Exact result of one invocation of an existing supervisor stage."""

    stage: CanaryStage
    disposition: CanaryStageDisposition
    receipt_digest: str
    reservation_digests: tuple[str, ...]
    lease_id: str | None = None
    request_id: str | None = None
    expensive_stage_receipt_digests: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.stage) is not CanaryStage:
            raise OneReservationCanaryError("stage receipt stage is not exactly typed")
        if type(self.disposition) is not CanaryStageDisposition:
            raise OneReservationCanaryError("stage receipt disposition is not typed")
        object.__setattr__(
            self, "receipt_digest", _digest(self.receipt_digest, "stage receipt digest")
        )
        members = _digest_tuple(
            self.reservation_digests, "stage receipt reservation digest"
        )
        object.__setattr__(self, "reservation_digests", members)
        object.__setattr__(self, "lease_id", _optional_digest(self.lease_id, "lease id"))
        object.__setattr__(
            self, "request_id", _optional_digest(self.request_id, "request id")
        )
        object.__setattr__(
            self,
            "expensive_stage_receipt_digests",
            _digest_tuple(
                self.expensive_stage_receipt_digests,
                "expensive stage receipt digest",
            ),
        )
        object.__setattr__(self, "reason", _text(self.reason, "stage reason", required=False))
        if self.disposition is CanaryStageDisposition.NO_WORK:
            if members or self.lease_id is not None or self.request_id is not None:
                raise OneReservationCanaryError("no-work receipt carries mutation identity")
        elif not members or self.lease_id is None or (
            self.request_id is None
            and not (
                self.stage is CanaryStage.QUALIFICATION
                and self.disposition is CanaryStageDisposition.HOLD
            )
        ):
            raise OneReservationCanaryError(
                "mutating stage receipt lacks reservation, lease, or request identity"
            )
        if self.disposition in {
            CanaryStageDisposition.HOLD,
            CanaryStageDisposition.REQUEUE,
        }:
            if self.reason is None:
                raise OneReservationCanaryError("hold/requeue receipt lacks a reason")
        elif self.reason is not None:
            raise OneReservationCanaryError("non-hold stage receipt carries a reason")


@dataclass(frozen=True)
class CanaryTransition:
    """One observed store transition bound to at most one stage receipt."""

    sequence: int
    before_phase: CanaryReservationPhase
    after_phase: CanaryReservationPhase
    before_observation_digest: str
    after_observation_digest: str
    stage: CanaryStage | None = None
    stage_receipt_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence <= 0:
            raise OneReservationCanaryError("transition sequence is malformed")
        if (
            type(self.before_phase) is not CanaryReservationPhase
            or type(self.after_phase) is not CanaryReservationPhase
        ):
            raise OneReservationCanaryError("transition phase is not exactly typed")
        object.__setattr__(
            self,
            "before_observation_digest",
            _digest(self.before_observation_digest, "before observation digest"),
        )
        object.__setattr__(
            self,
            "after_observation_digest",
            _digest(self.after_observation_digest, "after observation digest"),
        )
        if self.stage is None:
            if self.stage_receipt_digest is not None:
                raise OneReservationCanaryError("transition receipt has no stage")
        elif type(self.stage) is not CanaryStage:
            raise OneReservationCanaryError("transition stage is not exactly typed")
        else:
            object.__setattr__(
                self,
                "stage_receipt_digest",
                _digest(self.stage_receipt_digest, "transition stage receipt digest"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "after_observation_digest": self.after_observation_digest,
            "after_phase": self.after_phase.value,
            "before_observation_digest": self.before_observation_digest,
            "before_phase": self.before_phase.value,
            "sequence": self.sequence,
            "stage": None if self.stage is None else self.stage.value,
            "stage_receipt_digest": self.stage_receipt_digest,
        }


@dataclass(frozen=True)
class CanaryCheckpoint:
    """Durable restart fence; persist it before each stage callback."""

    expected_reservation_digest: str
    target_profile_digest: str
    started_monotonic: float
    deadline_monotonic: float
    max_ticks: int
    max_stage_receipts: int
    ticks_used: int = 0
    screen_claim_started: bool = False
    qualification_started: bool = False
    screen_lease_id: str | None = None
    screen_request_id: str | None = None
    qualification_lease_id: str | None = None
    qualification_request_id: str | None = None
    stage_receipt_digests: tuple[str, ...] = ()
    expensive_stage_receipt_digests: tuple[str, ...] = ()
    transitions: tuple[CanaryTransition, ...] = ()
    terminal_outcome: CanaryTerminalOutcome | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_reservation_digest",
            _digest(self.expected_reservation_digest, "expected reservation digest"),
        )
        object.__setattr__(
            self,
            "target_profile_digest",
            _digest(self.target_profile_digest, "checkpoint target profile digest"),
        )
        started = _finite_time(self.started_monotonic, "checkpoint start")
        deadline = _finite_time(self.deadline_monotonic, "checkpoint deadline")
        if deadline < started:
            raise OneReservationCanaryError("checkpoint deadline precedes its start")
        object.__setattr__(self, "started_monotonic", started)
        object.__setattr__(self, "deadline_monotonic", deadline)
        _positive_int(self.max_ticks, "max ticks")
        _positive_int(self.max_stage_receipts, "max stage receipts")
        if type(self.ticks_used) is not int or not 0 <= self.ticks_used <= self.max_ticks:
            raise OneReservationCanaryError("checkpoint tick count is malformed")
        for field in ("screen_claim_started", "qualification_started"):
            if type(getattr(self, field)) is not bool:
                raise OneReservationCanaryError(f"checkpoint {field} is not boolean")
        for field in (
            "screen_lease_id",
            "screen_request_id",
            "qualification_lease_id",
            "qualification_request_id",
        ):
            object.__setattr__(
                self, field, _optional_digest(getattr(self, field), f"checkpoint {field}")
            )
        stage_receipts = _digest_tuple(
            self.stage_receipt_digests, "checkpoint stage receipt digest"
        )
        expensive_receipts = _digest_tuple(
            self.expensive_stage_receipt_digests,
            "checkpoint expensive stage receipt digest",
        )
        if len(stage_receipts) + len(expensive_receipts) > self.max_stage_receipts:
            raise OneReservationCanaryError("checkpoint exceeds its stage receipt bound")
        object.__setattr__(self, "stage_receipt_digests", stage_receipts)
        object.__setattr__(self, "expensive_stage_receipt_digests", expensive_receipts)
        transitions = tuple(self.transitions)
        if any(type(row) is not CanaryTransition for row in transitions) or tuple(
            row.sequence for row in transitions
        ) != tuple(range(1, len(transitions) + 1)):
            raise OneReservationCanaryError("checkpoint transitions are malformed")
        object.__setattr__(self, "transitions", transitions)
        if (
            self.terminal_outcome is not None
            and type(self.terminal_outcome) is not CanaryTerminalOutcome
        ):
            raise OneReservationCanaryError("checkpoint terminal outcome is not typed")
        object.__setattr__(
            self,
            "terminal_reason",
            _text(self.terminal_reason, "terminal reason", required=False),
        )
        if (self.terminal_outcome is None) != (self.terminal_reason is None):
            raise OneReservationCanaryError(
                "checkpoint terminal outcome and reason must appear together"
            )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.one-reservation-canary-checkpoint.v1", self.to_dict()
        )

    @property
    def receipt_identity_count(self) -> int:
        return len(self.stage_receipt_digests) + len(
            self.expensive_stage_receipt_digests
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "deadline_monotonic_ns": int(round(self.deadline_monotonic * 1_000_000_000)),
            "expected_reservation_digest": self.expected_reservation_digest,
            "expensive_stage_receipt_digests": list(
                self.expensive_stage_receipt_digests
            ),
            "max_stage_receipts": self.max_stage_receipts,
            "max_ticks": self.max_ticks,
            "qualification_lease_id": self.qualification_lease_id,
            "qualification_request_id": self.qualification_request_id,
            "qualification_started": self.qualification_started,
            "schema": CANARY_CHECKPOINT_SCHEMA,
            "screen_claim_started": self.screen_claim_started,
            "screen_lease_id": self.screen_lease_id,
            "screen_request_id": self.screen_request_id,
            "stage_receipt_digests": list(self.stage_receipt_digests),
            "started_monotonic_ns": int(round(self.started_monotonic * 1_000_000_000)),
            "target_profile_digest": self.target_profile_digest,
            "terminal_outcome": (
                None if self.terminal_outcome is None else self.terminal_outcome.value
            ),
            "terminal_reason": self.terminal_reason,
            "ticks_used": self.ticks_used,
            "transitions": [row.to_dict() for row in self.transitions],
        }


@dataclass(frozen=True)
class CanaryReceipt:
    """Terminal machine-readable receipt for a bounded canary attempt."""

    checkpoint: CanaryCheckpoint
    finished_monotonic: float
    final_observation_digest: str
    final_phase: CanaryReservationPhase

    def __post_init__(self) -> None:
        if type(self.checkpoint) is not CanaryCheckpoint:
            raise OneReservationCanaryError("receipt checkpoint is not exactly typed")
        if self.checkpoint.terminal_outcome is None:
            raise OneReservationCanaryError("receipt checkpoint is not terminal")
        finished = _finite_time(self.finished_monotonic, "receipt finish")
        if finished < self.checkpoint.started_monotonic:
            raise OneReservationCanaryError("receipt finish precedes canary start")
        object.__setattr__(self, "finished_monotonic", finished)
        object.__setattr__(
            self,
            "final_observation_digest",
            _digest(self.final_observation_digest, "final observation digest"),
        )
        if type(self.final_phase) is not CanaryReservationPhase:
            raise OneReservationCanaryError("receipt final phase is not exactly typed")

    @property
    def outcome(self) -> CanaryTerminalOutcome:
        assert self.checkpoint.terminal_outcome is not None
        return self.checkpoint.terminal_outcome

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.one-reservation-canary-receipt.v1", self.to_dict()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint.to_dict(),
            "checkpoint_digest": self.checkpoint.digest,
            "final_observation_digest": self.final_observation_digest,
            "final_phase": self.final_phase.value,
            "finished_monotonic_ns": int(round(self.finished_monotonic * 1_000_000_000)),
            "outcome": self.outcome.value,
            "schema": CANARY_RECEIPT_SCHEMA,
        }


StoreObserver = Callable[[], CanaryStoreObservation]
# A stage adapter must compare the exact observation/checkpoint again inside the
# same store transaction that claims work.  Passing both products prevents a
# generic zero-argument FIFO dispatcher from being wired here without the
# required reservation guard.
StageOnce = Callable[
    [CanaryStoreObservation, CanaryCheckpoint], CanaryStageReceipt
]
CheckpointSink = Callable[[CanaryCheckpoint], None]


@dataclass(frozen=True)
class OneReservationCanaryBoundaries:
    """The complete and deliberately narrow authority surface of the fence."""

    observe_store: StoreObserver
    screen_once: StageOnce
    qualification_once: StageOnce
    persist_checkpoint: CheckpointSink

    def __post_init__(self) -> None:
        for field in (
            "observe_store",
            "screen_once",
            "qualification_once",
            "persist_checkpoint",
        ):
            if not callable(getattr(self, field)):
                raise OneReservationCanaryError(f"{field} boundary is not callable")


@dataclass
class OneReservationCanaryController:
    """Drive at most one exact reservation through screen and qualification."""

    boundaries: OneReservationCanaryBoundaries
    expected_fifo_head_reservation_digest: str
    deadline_monotonic: float
    max_ticks: int
    max_stage_receipts: int
    monotonic: Callable[[], float] = time.monotonic
    retained_checkpoint: CanaryCheckpoint | None = None

    def __post_init__(self) -> None:
        if type(self.boundaries) is not OneReservationCanaryBoundaries:
            raise OneReservationCanaryError("canary boundaries are not exactly typed")
        self.expected_fifo_head_reservation_digest = _digest(
            self.expected_fifo_head_reservation_digest,
            "expected FIFO-head reservation digest",
        )
        self.deadline_monotonic = _finite_time(
            self.deadline_monotonic, "canary deadline"
        )
        self.max_ticks = _positive_int(self.max_ticks, "max ticks")
        self.max_stage_receipts = _positive_int(
            self.max_stage_receipts, "max stage receipts"
        )
        if not callable(self.monotonic):
            raise OneReservationCanaryError("monotonic clock is not callable")
        if (
            self.retained_checkpoint is not None
            and type(self.retained_checkpoint) is not CanaryCheckpoint
        ):
            raise OneReservationCanaryError("retained checkpoint is not exactly typed")
        self._last_now: float | None = None

    def _now(self) -> float:
        now = _finite_time(self.monotonic(), "monotonic clock result")
        if self._last_now is not None and now < self._last_now:
            raise OneReservationCanaryError("monotonic clock moved backwards")
        self._last_now = now
        return now

    def _observe(self) -> CanaryStoreObservation:
        observed = self.boundaries.observe_store()
        if type(observed) is not CanaryStoreObservation:
            raise OneReservationCanaryError("store boundary returned an untyped observation")
        return observed

    def _persist(self, checkpoint: CanaryCheckpoint) -> None:
        result = self.boundaries.persist_checkpoint(checkpoint)
        if result is not None:
            raise OneReservationCanaryError("checkpoint sink returned an unexpected value")

    def _terminal(
        self,
        checkpoint: CanaryCheckpoint,
        observation: CanaryStoreObservation,
        outcome: CanaryTerminalOutcome,
        reason: str,
    ) -> CanaryReceipt:
        terminal = replace(
            checkpoint,
            terminal_outcome=outcome,
            terminal_reason=reason,
        )
        self._persist(terminal)
        return CanaryReceipt(
            checkpoint=terminal,
            finished_monotonic=self._now(),
            final_observation_digest=observation.digest,
            final_phase=observation.phase,
        )

    def _identity_outcome(
        self,
        checkpoint: CanaryCheckpoint,
        observation: CanaryStoreObservation,
    ) -> tuple[CanaryTerminalOutcome, str] | None:
        expected = checkpoint.expected_reservation_digest
        if observation.reservation_digest != expected:
            return CanaryTerminalOutcome.IDENTITY_DRIFT, "observed_reservation_changed"
        if observation.target_profile_digest != checkpoint.target_profile_digest:
            return CanaryTerminalOutcome.IDENTITY_DRIFT, "target_profile_changed"
        lease = observation.active_lease
        if lease is not None and lease.reservation_digests != (expected,):
            return CanaryTerminalOutcome.LEASE_DRIFT, "active_lease_members_changed"
        if lease is not None and lease.stage is CanaryStage.SCREEN:
            if (
                checkpoint.screen_lease_id is not None
                and lease.lease_id != checkpoint.screen_lease_id
            ):
                return CanaryTerminalOutcome.LEASE_DRIFT, "screen_lease_changed"
            if (
                checkpoint.screen_request_id is not None
                and lease.request_id != checkpoint.screen_request_id
            ):
                return CanaryTerminalOutcome.REQUEST_DRIFT, "screen_request_changed"
        if lease is not None and lease.stage is CanaryStage.QUALIFICATION:
            if (
                checkpoint.qualification_lease_id is not None
                and lease.lease_id != checkpoint.qualification_lease_id
            ):
                return CanaryTerminalOutcome.LEASE_DRIFT, "qualification_lease_changed"
            if (
                checkpoint.qualification_request_id is not None
                and lease.request_id != checkpoint.qualification_request_id
            ):
                return CanaryTerminalOutcome.REQUEST_DRIFT, "qualification_request_changed"
        return None

    def _receipt_problem(
        self,
        checkpoint: CanaryCheckpoint,
        stage: CanaryStage,
        receipt: CanaryStageReceipt,
    ) -> tuple[CanaryTerminalOutcome, str] | None:
        expected = checkpoint.expected_reservation_digest
        if receipt.stage is not stage:
            return CanaryTerminalOutcome.IDENTITY_DRIFT, "stage_receipt_names_other_stage"
        if (
            receipt.disposition is not CanaryStageDisposition.NO_WORK
            and receipt.reservation_digests != (expected,)
        ):
            return CanaryTerminalOutcome.LEASE_DRIFT, "stage_receipt_members_changed"
        if receipt.receipt_digest in checkpoint.stage_receipt_digests:
            return CanaryTerminalOutcome.REPEATED_STAGE_RECEIPT, "stage_receipt_repeated"
        seen_expensive = set(checkpoint.expensive_stage_receipt_digests)
        if any(item in seen_expensive for item in receipt.expensive_stage_receipt_digests):
            return (
                CanaryTerminalOutcome.REPEATED_EXPENSIVE_STAGE,
                "expensive_stage_receipt_repeated",
            )
        incoming = 1 + len(receipt.expensive_stage_receipt_digests)
        if checkpoint.receipt_identity_count + incoming > checkpoint.max_stage_receipts:
            return CanaryTerminalOutcome.MAX_STAGE_RECEIPTS, "stage_receipt_bound_reached"
        if (
            receipt.disposition is not CanaryStageDisposition.NO_WORK
            and stage is CanaryStage.SCREEN
        ):
            if (
                checkpoint.screen_lease_id is not None
                and receipt.lease_id != checkpoint.screen_lease_id
            ):
                return CanaryTerminalOutcome.LEASE_DRIFT, "second_screen_lease_observed"
            if (
                checkpoint.screen_request_id is not None
                and receipt.request_id != checkpoint.screen_request_id
            ):
                return CanaryTerminalOutcome.REQUEST_DRIFT, "second_screen_request_observed"
        elif receipt.disposition is not CanaryStageDisposition.NO_WORK:
            if (
                checkpoint.qualification_lease_id is not None
                and receipt.lease_id != checkpoint.qualification_lease_id
            ):
                return CanaryTerminalOutcome.LEASE_DRIFT, "qualification_lease_changed"
            if (
                checkpoint.qualification_request_id is not None
                and receipt.request_id != checkpoint.qualification_request_id
            ):
                return CanaryTerminalOutcome.REQUEST_DRIFT, "qualification_request_changed"
        return None

    def _record_stage(
        self,
        checkpoint: CanaryCheckpoint,
        before: CanaryStoreObservation,
        after: CanaryStoreObservation,
        receipt: CanaryStageReceipt,
    ) -> CanaryCheckpoint:
        transition = CanaryTransition(
            sequence=len(checkpoint.transitions) + 1,
            before_phase=before.phase,
            after_phase=after.phase,
            before_observation_digest=before.digest,
            after_observation_digest=after.digest,
            stage=receipt.stage,
            stage_receipt_digest=receipt.receipt_digest,
        )
        values: dict[str, object] = {
            "stage_receipt_digests": checkpoint.stage_receipt_digests
            + (receipt.receipt_digest,),
            "expensive_stage_receipt_digests": (
                checkpoint.expensive_stage_receipt_digests
                + receipt.expensive_stage_receipt_digests
            ),
            "transitions": checkpoint.transitions + (transition,),
        }
        if (
            receipt.stage is CanaryStage.SCREEN
            and receipt.disposition is not CanaryStageDisposition.NO_WORK
        ):
            values["screen_lease_id"] = receipt.lease_id
            values["screen_request_id"] = receipt.request_id
        if (
            receipt.stage is CanaryStage.QUALIFICATION
            and receipt.disposition is not CanaryStageDisposition.NO_WORK
        ):
            values["qualification_lease_id"] = receipt.lease_id
            values["qualification_request_id"] = receipt.request_id
        updated = replace(checkpoint, **values)
        self._persist(updated)
        return updated

    def _invoke(
        self,
        checkpoint: CanaryCheckpoint,
        observation: CanaryStoreObservation,
        stage: CanaryStage,
    ) -> tuple[CanaryCheckpoint, CanaryStoreObservation, CanaryStageReceipt] | CanaryReceipt:
        now = self._now()
        if now >= checkpoint.deadline_monotonic:
            return self._terminal(
                checkpoint, observation, CanaryTerminalOutcome.DEADLINE, "deadline_reached"
            )
        if checkpoint.ticks_used >= checkpoint.max_ticks:
            return self._terminal(
                checkpoint, observation, CanaryTerminalOutcome.MAX_TICKS, "tick_bound_reached"
            )
        if checkpoint.receipt_identity_count >= checkpoint.max_stage_receipts:
            return self._terminal(
                checkpoint,
                observation,
                CanaryTerminalOutcome.MAX_STAGE_RECEIPTS,
                "stage_receipt_bound_reached",
            )

        intent_fields: dict[str, object] = {"ticks_used": checkpoint.ticks_used + 1}
        if stage is CanaryStage.SCREEN:
            intent_fields["screen_claim_started"] = True
        else:
            intent_fields["qualification_started"] = True
            lease = observation.active_lease
            if lease is not None and lease.stage is CanaryStage.QUALIFICATION:
                intent_fields["qualification_lease_id"] = lease.lease_id
                if lease.request_id is not None:
                    intent_fields["qualification_request_id"] = lease.request_id
        intent = replace(checkpoint, **intent_fields)
        self._persist(intent)  # durable intent always precedes the mutating callback

        callback = (
            self.boundaries.screen_once
            if stage is CanaryStage.SCREEN
            else self.boundaries.qualification_once
        )
        receipt = callback(observation, intent)
        if type(receipt) is not CanaryStageReceipt:
            raise OneReservationCanaryError("stage boundary returned an untyped receipt")
        after = self._observe()
        problem = self._receipt_problem(intent, stage, receipt)
        if problem is not None:
            return self._terminal(intent, after, problem[0], problem[1])
        identity = self._identity_outcome(intent, after)
        if identity is not None:
            return self._terminal(intent, after, identity[0], identity[1])
        recorded = self._record_stage(intent, observation, after, receipt)
        return recorded, after, receipt

    def run(self) -> CanaryReceipt:
        """Run synchronously until one explicit terminal outcome is observed."""

        observed = self._observe()
        now = self._now()
        expected = self.expected_fifo_head_reservation_digest
        retained = self.retained_checkpoint
        if retained is None:
            checkpoint = CanaryCheckpoint(
                expected_reservation_digest=expected,
                target_profile_digest=observed.target_profile_digest,
                started_monotonic=now,
                deadline_monotonic=self.deadline_monotonic,
                max_ticks=self.max_ticks,
                max_stage_receipts=self.max_stage_receipts,
            )
            self._persist(checkpoint)
        else:
            checkpoint = retained
            if (
                checkpoint.expected_reservation_digest != expected
                or checkpoint.deadline_monotonic != self.deadline_monotonic
                or checkpoint.max_ticks != self.max_ticks
                or checkpoint.max_stage_receipts != self.max_stage_receipts
            ):
                raise OneReservationCanaryError(
                    "retained checkpoint differs from the requested canary bounds"
                )

        identity = self._identity_outcome(checkpoint, observed)
        if identity is not None:
            return self._terminal(checkpoint, observed, identity[0], identity[1])
        if checkpoint.terminal_outcome is not None:
            # A retained terminal checkpoint cannot be reopened into mutation.
            return CanaryReceipt(
                checkpoint=checkpoint,
                finished_monotonic=now,
                final_observation_digest=observed.digest,
                final_phase=observed.phase,
            )

        for _pass in range(checkpoint.max_ticks - checkpoint.ticks_used + 1):
            now = self._now()
            if now >= checkpoint.deadline_monotonic:
                return self._terminal(
                    checkpoint, observed, CanaryTerminalOutcome.DEADLINE, "deadline_reached"
                )
            identity = self._identity_outcome(checkpoint, observed)
            if identity is not None:
                return self._terminal(checkpoint, observed, identity[0], identity[1])
            if observed.phase.qualification_complete:
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.COMPLETED,
                    "qualification_complete_in_store",
                )
            if observed.phase in {
                CanaryReservationPhase.FAILED,
                CanaryReservationPhase.EXPIRED,
            }:
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.HOLD,
                    f"reservation_terminal_{observed.phase.value}",
                )

            active = observed.active_lease
            if active is not None and active.stage is CanaryStage.QUALIFICATION:
                invoked = self._invoke(
                    checkpoint, observed, CanaryStage.QUALIFICATION
                )
            elif active is not None and active.stage is CanaryStage.SCREEN:
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.SECOND_CLAIM,
                    "screen_lease_already_active",
                )
            elif observed.phase is CanaryReservationPhase.PUBLISHED:
                if checkpoint.screen_claim_started:
                    return self._terminal(
                        checkpoint,
                        observed,
                        CanaryTerminalOutcome.SECOND_CLAIM,
                        "screen_claim_already_started",
                    )
                if observed.fifo_head_reservation_digest != expected:
                    return self._terminal(
                        checkpoint,
                        observed,
                        CanaryTerminalOutcome.WRONG_FIFO_HEAD,
                        "expected_reservation_is_not_fifo_head",
                    )
                invoked = self._invoke(checkpoint, observed, CanaryStage.SCREEN)
            elif observed.phase is CanaryReservationPhase.PROMOTED:
                if checkpoint.qualification_started:
                    return self._terminal(
                        checkpoint,
                        observed,
                        CanaryTerminalOutcome.SECOND_CLAIM,
                        "qualification_claim_already_started_without_active_lease",
                    )
                if observed.next_qualification_reservation_digests != (expected,):
                    return self._terminal(
                        checkpoint,
                        observed,
                        CanaryTerminalOutcome.LEASE_DRIFT,
                        "qualification_preview_is_not_exact_reservation",
                    )
                invoked = self._invoke(
                    checkpoint, observed, CanaryStage.QUALIFICATION
                )
            elif observed.phase is CanaryReservationPhase.SCREENING:
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.SECOND_CLAIM,
                    "screening_state_cannot_be_reclaimed",
                )
            elif observed.phase is CanaryReservationPhase.QUALIFYING:
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.LEASE_DRIFT,
                    "qualifying_state_has_no_active_lease",
                )
            else:  # pragma: no cover - enum exhaustiveness guard
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.IDENTITY_DRIFT,
                    "unsupported_reservation_phase",
                )

            if type(invoked) is CanaryReceipt:
                return invoked
            checkpoint, after, stage_receipt = invoked
            observed = after
            if stage_receipt.disposition is CanaryStageDisposition.HOLD:
                assert stage_receipt.reason is not None
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.HOLD,
                    f"stage_hold:{stage_receipt.reason}",
                )
            if stage_receipt.disposition is CanaryStageDisposition.REQUEUE:
                assert stage_receipt.reason is not None
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.REQUEUE,
                    f"stage_requeue:{stage_receipt.reason}",
                )
            if stage_receipt.disposition is CanaryStageDisposition.NO_WORK:
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.NO_WORK,
                    f"{stage_receipt.stage.value}_returned_no_work",
                )
            if (
                stage_receipt.stage is CanaryStage.SCREEN
                and observed.phase
                not in {
                    CanaryReservationPhase.PROMOTED,
                    CanaryReservationPhase.QUALIFYING,
                    CanaryReservationPhase.REPRODUCTION_PENDING,
                    CanaryReservationPhase.QUALIFIED,
                }
            ):
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.HOLD,
                    "screen_receipt_did_not_promote_reservation",
                )
            if (
                stage_receipt.stage is CanaryStage.QUALIFICATION
                and stage_receipt.disposition is CanaryStageDisposition.COMPLETED
                and not observed.phase.qualification_complete
            ):
                return self._terminal(
                    checkpoint,
                    observed,
                    CanaryTerminalOutcome.HOLD,
                    "qualification_receipt_not_complete_in_store",
                )
        raise OneReservationCanaryError("bounded canary loop exhausted")

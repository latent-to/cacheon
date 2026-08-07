"""Closed production bindings for the one-reservation canary fence.

The runtime binds one expected reservation to the durable coordinator and the
two guarded remote dispatchers.  It contributes no queue mutation of its own:
the store view is read-only and every stage mutation remains inside an existing
guarded dispatcher.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Callable

from cacheon.arena_service import ArenaScreenReceipt, ArenaService
from cacheon.chain.evaluation_coordinator import (
    EvaluationCoordinator,
    EvaluationRun,
    WorkerReadiness,
)
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.evaluation_recovery import EvaluationRecovery
from cacheon.chain.one_reservation_canary import (
    CANARY_CHECKPOINT_SCHEMA,
    CanaryCheckpoint,
    CanaryLeaseObservation,
    CanaryReservationPhase,
    CanaryStage,
    CanaryStageDisposition,
    CanaryStageReceipt,
    CanaryStoreObservation,
    CanaryTerminalOutcome,
    CanaryTransition,
    OneReservationCanaryBoundaries,
    OneReservationCanaryError,
)
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationDispatcher,
    RecoverableQualificationHold,
    RecoverableQualificationRequeue,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    GuardedEvaluationRun,
    RemoteEvaluationDispatcher,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
)
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.qualification_intake import QualificationIntakeBatch
from cacheon.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)
from cacheon.stack_manifest import EvaluationStackManifest


CANARY_CHECKPOINT_JOURNAL_SCHEMA = "cacheon-one-reservation-canary-journal-v1"
_MAX_JOURNAL_BYTES = 8 << 20
_PROFILE_DOMAIN = "cacheon.chain.one-reservation-canary-target-profile.v1"
_SCREEN_RECEIPT_DOMAIN = "cacheon.chain.one-reservation-canary-screen-run.v1"
_QUALIFICATION_RECEIPT_DOMAIN = (
    "cacheon.chain.one-reservation-canary-qualification-run.v1"
)
_NO_WORK_RECEIPT_DOMAIN = "cacheon.chain.one-reservation-canary-no-work.v1"


class OneReservationCanaryRuntimeError(OneReservationCanaryError):
    """The concrete canary authority or its retained state is not exact."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise OneReservationCanaryRuntimeError(str(exc)) from None


def _runtime_error(message: str) -> OneReservationCanaryRuntimeError:
    return OneReservationCanaryRuntimeError(message)


_PHASES = {
    "published": CanaryReservationPhase.PUBLISHED,
    "screening": CanaryReservationPhase.SCREENING,
    "promoted": CanaryReservationPhase.PROMOTED,
    "qualifying": CanaryReservationPhase.QUALIFYING,
    "reproduction_pending": CanaryReservationPhase.REPRODUCTION_PENDING,
    "qualified": CanaryReservationPhase.QUALIFIED,
    "failed": CanaryReservationPhase.FAILED,
    "expired": CanaryReservationPhase.EXPIRED,
}


class OneReservationCanaryRuntime:
    """Exact durable-store and guarded-dispatch authority for one reservation."""

    def __init__(
        self,
        *,
        screen_dispatcher: RemoteEvaluationDispatcher,
        qualification_dispatcher: RecoverableQualificationDispatcher,
        expected_reservation_digest: str,
    ) -> None:
        if type(screen_dispatcher) is not RemoteEvaluationDispatcher:
            raise _runtime_error("screen dispatcher is not exactly typed")
        if type(qualification_dispatcher) is not RecoverableQualificationDispatcher:
            raise _runtime_error("qualification dispatcher is not exactly typed")
        coordinator = screen_dispatcher.coordinator
        if (
            type(coordinator) is not EvaluationCoordinator
            or qualification_dispatcher.coordinator is not coordinator
            or coordinator._store_factory is not RecoverableFinalizedIntakeStore
        ):
            raise _runtime_error("dispatchers do not share one recoverable coordinator")
        if (
            type(coordinator.service) is not ArenaService
            or type(coordinator.readiness) is not WorkerReadiness
        ):
            raise _runtime_error("coordinator service authority is not exact")
        identity = screen_dispatcher.transport_identity
        credential = screen_dispatcher.credential
        if (
            type(identity) is not RemoteWorkerTransportIdentity
            or qualification_dispatcher.transport_identity != identity
            or type(credential) is not RemoteWorkerCredential
            or qualification_dispatcher.credential != credential
        ):
            raise _runtime_error("dispatchers do not share transport authority")
        incumbent = qualification_dispatcher.qualification_incumbent_stack
        if type(incumbent) is not EvaluationStackManifest:
            raise _runtime_error("qualification incumbent is not exactly typed")
        tree_digest = _digest(
            qualification_dispatcher.qualification_incumbent_tree_digest,
            "qualification incumbent tree digest",
        )
        self.expected_reservation_digest = _digest(
            expected_reservation_digest, "expected reservation digest"
        )
        self.screen_dispatcher = screen_dispatcher
        self.qualification_dispatcher = qualification_dispatcher
        self.coordinator = coordinator
        self.scope = coordinator.scope
        self.service = coordinator.service
        self.readiness = coordinator.readiness
        self.transport_identity = identity
        self.credential = credential
        self.qualification_incumbent_stack = incumbent
        self.qualification_incumbent_tree_digest = tree_digest
        self._validate_live_authority()

    def _validate_live_authority(self) -> None:
        coordinator = self.coordinator
        if (
            self.screen_dispatcher.coordinator is not coordinator
            or self.qualification_dispatcher.coordinator is not coordinator
            or coordinator.scope is not self.scope
            or coordinator.service is not self.service
            or coordinator.readiness is not self.readiness
            or coordinator._store_factory is not RecoverableFinalizedIntakeStore
            or self.screen_dispatcher.transport_identity != self.transport_identity
            or self.qualification_dispatcher.transport_identity
            != self.transport_identity
            or self.screen_dispatcher.credential != self.credential
            or self.qualification_dispatcher.credential != self.credential
            or getattr(self.screen_dispatcher.transport, "identity", None)
            != self.transport_identity
            or getattr(self.qualification_dispatcher.transport, "identity", None)
            != self.transport_identity
            or self.qualification_dispatcher.qualification_incumbent_stack
            != self.qualification_incumbent_stack
            or self.qualification_dispatcher.qualification_incumbent_tree_digest
            != self.qualification_incumbent_tree_digest
        ):
            raise _runtime_error("canary runtime authority drifted")
        self.readiness.validate(self.service)
        if (
            self.transport_identity.service_digest != self.service.identity
            or self.transport_identity.worker_readiness_digest != self.readiness.digest
            or self.transport_identity.credential_digest != self.credential.digest
        ):
            raise _runtime_error("canary transport differs from live service authority")

    def _target_profile_digest(self, reservation) -> str:
        fingerprint = reservation.delta_fingerprint
        if type(fingerprint) is not SubmittedDeltaFingerprint:
            raise _runtime_error("reservation has no exact submitted fingerprint")
        if (
            fingerprint.target_id != reservation.target_id
            or fingerprint.members != reservation.target_members
        ):
            raise _runtime_error("reservation submitted fingerprint is stale")
        if (
            reservation.arena_service_digest
            and reservation.arena_service_digest != self.service.identity
        ):
            raise _runtime_error("reservation service identity is stale")
        return canonical_digest(
            _PROFILE_DOMAIN,
            {
                "coordinator_scope_digest": self.coordinator.scope.digest,
                "incumbent_stack_digest": self.qualification_incumbent_stack.digest,
                "incumbent_tree_digest": self.qualification_incumbent_tree_digest,
                "readiness_digest": self.readiness.digest,
                "service_manifest_digest": self.service.manifest.digest,
                "submitted_target": {
                    "members": list(fingerprint.members),
                    "product_kind": fingerprint.product_kind,
                    "target_id": fingerprint.target_id,
                    "target_spec_digest": fingerprint.target_spec_digest,
                },
                "transport_identity_digest": self.transport_identity.digest,
            },
        )

    @staticmethod
    def _active_lease(
        leases: tuple[EvaluationLease, ...],
        recovery: EvaluationRecovery | None,
    ) -> CanaryLeaseObservation | None:
        if type(leases) is not tuple or any(
            type(row) is not EvaluationLease for row in leases
        ):
            raise _runtime_error("active lease view is not exactly typed")
        if len(leases) > 1:
            raise _runtime_error("active lease view is ambiguous")
        if not leases:
            if recovery is not None:
                raise _runtime_error("qualification recovery has no active lease")
            return None
        lease = leases[0]
        if lease.stage == "screen":
            if recovery is not None:
                raise _runtime_error(
                    "screen lease conflicts with qualification recovery"
                )
            return CanaryLeaseObservation(
                CanaryStage.SCREEN,
                lease.lease_id,
                lease.reservation_ids,
            )
        if type(recovery) is not EvaluationRecovery or recovery.lease != lease:
            raise _runtime_error("qualification lease has no exact retained recovery")
        return CanaryLeaseObservation(
            CanaryStage.QUALIFICATION,
            lease.lease_id,
            lease.reservation_ids,
            recovery.request_id or None,
        )

    def observe_store(self) -> CanaryStoreObservation:
        """Open one durable cursor view, copy its exact identities, and close it."""

        self._validate_live_authority()
        store, _point = self.coordinator._open_at_durable_cursor()
        if type(store) is not RecoverableFinalizedIntakeStore:
            store.close()
            raise _runtime_error("coordinator opened a non-recoverable store")
        try:
            reservation = store.get(self.expected_reservation_digest)
            phase = _PHASES.get(reservation.status)
            if phase is None:
                raise _runtime_error("reservation durable status is unsupported")
            screen_preview = store.preview_evaluation_claim(stage="screen")
            if len(screen_preview) > 1:
                raise _runtime_error("screen preview is ambiguous")
            qualification_preview = store.preview_evaluation_claim(
                stage="qualification",
                max_members=self.coordinator.qualification_max_members,
            )
            leases = store.active_evaluation_leases()
            recovery = store.pending_qualification_recovery()
            profile = self._target_profile_digest(reservation)
            active = self._active_lease(leases, recovery)
            return CanaryStoreObservation(
                reservation_digest=reservation.reservation_id,
                target_profile_digest=profile,
                phase=phase,
                fifo_head_reservation_digest=(
                    None if not screen_preview else screen_preview[0]
                ),
                next_qualification_reservation_digests=qualification_preview,
                active_lease=active,
            )
        finally:
            store.close()

    def _validate_stage_input(
        self,
        observation: CanaryStoreObservation,
        checkpoint: CanaryCheckpoint,
        *,
        stage: CanaryStage,
    ) -> None:
        self._validate_live_authority()
        if (
            type(observation) is not CanaryStoreObservation
            or type(checkpoint) is not CanaryCheckpoint
            or observation.reservation_digest != self.expected_reservation_digest
            or checkpoint.expected_reservation_digest
            != self.expected_reservation_digest
            or observation.target_profile_digest != checkpoint.target_profile_digest
            or checkpoint.terminal_outcome is not None
        ):
            raise _runtime_error("stage input differs from the canary identity")
        if stage is CanaryStage.SCREEN:
            if (
                observation.phase is not CanaryReservationPhase.PUBLISHED
                or observation.fifo_head_reservation_digest
                != self.expected_reservation_digest
                or observation.active_lease is not None
                or not checkpoint.screen_claim_started
                or checkpoint.screen_lease_id is not None
                or checkpoint.screen_request_id is not None
            ):
                raise _runtime_error(
                    "screen stage input is not the exact published head"
                )
        elif (
            observation.phase is not CanaryReservationPhase.PROMOTED
            or not checkpoint.qualification_started
        ):
            raise _runtime_error("qualification stage input is not exactly promoted")

    @staticmethod
    def _no_work(
        stage: CanaryStage,
        observation: CanaryStoreObservation,
        checkpoint: CanaryCheckpoint,
    ) -> CanaryStageReceipt:
        return CanaryStageReceipt(
            stage=stage,
            disposition=CanaryStageDisposition.NO_WORK,
            receipt_digest=canonical_digest(
                _NO_WORK_RECEIPT_DOMAIN,
                {
                    "checkpoint_digest": checkpoint.digest,
                    "observation_digest": observation.digest,
                    "stage": stage.value,
                },
            ),
            reservation_digests=(),
        )

    def _guarded_run(
        self,
        guarded: GuardedEvaluationRun,
        *,
        stage: CanaryStage,
        expected_members: tuple[EvaluationLeaseMember, ...],
    ) -> EvaluationRun:
        if (
            type(guarded) is not GuardedEvaluationRun
            or type(guarded.run) is not EvaluationRun
        ):
            raise _runtime_error("guarded stage returned an untyped run")
        run = guarded.run
        expected_stage = stage.value
        if (
            run.disposition != "completed"
            or run.lease.stage != expected_stage
            or run.lease.members != expected_members
        ):
            raise _runtime_error("guarded stage changed its exact lease")
        run.envelope.verify(run.lease, self.readiness, self.service, run.payload)
        return run

    def screen_once(
        self,
        observation: CanaryStoreObservation,
        checkpoint: CanaryCheckpoint,
    ) -> CanaryStageReceipt:
        self._validate_stage_input(observation, checkpoint, stage=CanaryStage.SCREEN)
        members = (
            EvaluationLeaseMember(self.expected_reservation_digest, "published"),
        )
        result = self.screen_dispatcher.dispatch_guarded_screen_once(
            expected_members=members
        )
        if result is None:
            return self._no_work(CanaryStage.SCREEN, observation, checkpoint)
        run = self._guarded_run(
            result, stage=CanaryStage.SCREEN, expected_members=members
        )
        if type(run.payload) is not ArenaScreenReceipt:
            raise _runtime_error("guarded screen returned another payload type")
        receipt_digest = canonical_digest(
            _SCREEN_RECEIPT_DOMAIN,
            {
                "envelope_digest": run.envelope.digest,
                "lease_id": run.lease.lease_id,
                "members": [row.to_dict() for row in run.lease.members],
                "payload_digest": run.payload.digest,
                "request_id": result.request_id,
            },
        )
        return CanaryStageReceipt(
            stage=CanaryStage.SCREEN,
            disposition=CanaryStageDisposition.COMPLETED,
            receipt_digest=receipt_digest,
            reservation_digests=run.lease.reservation_ids,
            lease_id=run.lease.lease_id,
            request_id=result.request_id,
        )

    @staticmethod
    def _qualification_guards(
        observation: CanaryStoreObservation,
        checkpoint: CanaryCheckpoint,
    ) -> tuple[tuple[EvaluationLeaseMember, ...], str | None, str | None]:
        active = observation.active_lease
        if active is None:
            if (
                observation.next_qualification_reservation_digests
                != (checkpoint.expected_reservation_digest,)
                or checkpoint.qualification_lease_id is not None
                or checkpoint.qualification_request_id is not None
            ):
                raise _runtime_error("qualification preview or retained guard drifted")
            members = (
                EvaluationLeaseMember(
                    checkpoint.expected_reservation_digest, "promoted"
                ),
            )
            return members, None, None
        if (
            active.stage is not CanaryStage.QUALIFICATION
            or active.reservation_digests
            != (checkpoint.expected_reservation_digest,)
            or checkpoint.qualification_lease_id != active.lease_id
            or checkpoint.qualification_request_id != active.request_id
        ):
            raise _runtime_error("active qualification guard drifted")
        members = tuple(
            EvaluationLeaseMember(reservation, "promoted")
            for reservation in active.reservation_digests
        )
        return members, active.lease_id, active.request_id

    @staticmethod
    def _expensive_batch_identities(batch: QualificationIntakeBatch) -> tuple[str, ...]:
        identities: set[str] = set()
        if batch.attempt_ref is not None:
            identities.add(batch.attempt_ref.sha256)
        for outcome in batch.outcomes:
            for value in (
                outcome.attempt_artifact_sha256,
                outcome.report_digest,
                outcome.failure_digest,
            ):
                if value is not None:
                    identities.add(value)
        if batch.retry_plan is not None:
            identities.add(batch.retry_plan.failure_digest)
        return tuple(sorted(identities))

    def qualification_once(
        self,
        observation: CanaryStoreObservation,
        checkpoint: CanaryCheckpoint,
    ) -> CanaryStageReceipt:
        self._validate_stage_input(
            observation, checkpoint, stage=CanaryStage.QUALIFICATION
        )
        members, lease_guard, request_guard = self._qualification_guards(
            observation, checkpoint
        )
        result = self.qualification_dispatcher.dispatch_guarded_once(
            expected_members=members,
            expected_lease_id=lease_guard,
            expected_request_id=request_guard,
        )
        if result is None:
            return self._no_work(CanaryStage.QUALIFICATION, observation, checkpoint)
        if type(result) is GuardedEvaluationRun:
            run = self._guarded_run(
                result,
                stage=CanaryStage.QUALIFICATION,
                expected_members=members,
            )
            if type(run.payload) is not QualificationIntakeBatch:
                raise _runtime_error(
                    "guarded qualification returned another payload type"
                )
            expensive = self._expensive_batch_identities(run.payload)
            receipt_digest = canonical_digest(
                _QUALIFICATION_RECEIPT_DOMAIN,
                {
                    "envelope_digest": run.envelope.digest,
                    "expensive_evidence_digests": list(expensive),
                    "lease_id": run.lease.lease_id,
                    "members": [row.to_dict() for row in run.lease.members],
                    "request_id": result.request_id,
                },
            )
            return CanaryStageReceipt(
                stage=CanaryStage.QUALIFICATION,
                disposition=CanaryStageDisposition.COMPLETED,
                receipt_digest=receipt_digest,
                reservation_digests=run.lease.reservation_ids,
                lease_id=run.lease.lease_id,
                request_id=result.request_id,
                expensive_stage_receipt_digests=expensive,
            )
        active = observation.active_lease
        if active is None or active.stage is not CanaryStage.QUALIFICATION:
            raise _runtime_error(
                "qualification disposition has no before lease identity"
            )
        if type(result) is RecoverableQualificationHold:
            request_id = result.request_id or None
            if active.request_id is not None and request_id != active.request_id:
                raise _runtime_error("qualification HOLD request identity drifted")
            disposition = CanaryStageDisposition.HOLD
            reason = result.reason
            outcome_identity: dict[str, object] = {
                "phase": result.phase.value,
                "reason": result.reason,
                "recovery_id": result.recovery_id,
            }
        elif type(result) is RecoverableQualificationRequeue:
            request_id = result.request_id
            if active.request_id is not None and request_id != active.request_id:
                raise _runtime_error("qualification REQUEUE request identity drifted")
            disposition = CanaryStageDisposition.REQUEUE
            reason = result.outcome.failure_code
            outcome_identity = {
                "decision": result.outcome.decision,
                "failure_code": result.outcome.failure_code,
                "recovery_id": result.recovery_id,
            }
        else:
            raise _runtime_error("qualification dispatcher returned an untyped result")
        receipt_digest = canonical_digest(
            _QUALIFICATION_RECEIPT_DOMAIN,
            {
                "disposition": disposition.value,
                "lease_id": active.lease_id,
                "members": [row.to_dict() for row in members],
                "outcome": outcome_identity,
                "request_id": request_id,
            },
        )
        return CanaryStageReceipt(
            stage=CanaryStage.QUALIFICATION,
            disposition=disposition,
            receipt_digest=receipt_digest,
            reservation_digests=active.reservation_digests,
            lease_id=active.lease_id,
            request_id=request_id,
            reason=reason,
        )

    def boundaries(
        self, checkpoint_sink: Callable[[CanaryCheckpoint], None]
    ) -> OneReservationCanaryBoundaries:
        """Return the fence's complete four-call authority surface."""

        if not callable(checkpoint_sink):
            raise _runtime_error("checkpoint sink is not callable")
        return OneReservationCanaryBoundaries(
            observe_store=self.observe_store,
            screen_once=self.screen_once,
            qualification_once=self.qualification_once,
            persist_checkpoint=checkpoint_sink,
        )


def _reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _runtime_error("journal JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_number(_value: str) -> object:
    raise _runtime_error("journal JSON contains a non-integer number")


def _closed_object(
    value: object, fields: frozenset[str], label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise _runtime_error(f"{label} fields are not closed")
    return value


def _enum(enum_type, value: object, label: str):
    if type(value) is not str:
        raise _runtime_error(f"{label} is malformed")
    try:
        return enum_type(value)
    except ValueError:
        raise _runtime_error(f"{label} is unsupported") from None


_TRANSITION_FIELDS = frozenset(
    {
        "after_observation_digest",
        "after_phase",
        "before_observation_digest",
        "before_phase",
        "sequence",
        "stage",
        "stage_receipt_digest",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "deadline_monotonic_ns",
        "expected_reservation_digest",
        "expensive_stage_receipt_digests",
        "max_stage_receipts",
        "max_ticks",
        "qualification_lease_id",
        "qualification_request_id",
        "qualification_started",
        "schema",
        "screen_claim_started",
        "screen_lease_id",
        "screen_request_id",
        "stage_receipt_digests",
        "started_monotonic_ns",
        "target_profile_digest",
        "terminal_outcome",
        "terminal_reason",
        "ticks_used",
        "transitions",
    }
)
_JOURNAL_FIELDS = frozenset({"checkpoint", "checkpoint_digest", "schema"})


def _parse_transition(value: object) -> CanaryTransition:
    row = _closed_object(value, _TRANSITION_FIELDS, "journal transition")
    raw_stage = row["stage"]
    stage = (
        None
        if raw_stage is None
        else _enum(CanaryStage, raw_stage, "transition stage")
    )
    return CanaryTransition(
        sequence=row["sequence"],  # type: ignore[arg-type]
        before_phase=_enum(
            CanaryReservationPhase, row["before_phase"], "transition before phase"
        ),
        after_phase=_enum(
            CanaryReservationPhase, row["after_phase"], "transition after phase"
        ),
        before_observation_digest=row[  # type: ignore[arg-type]
            "before_observation_digest"
        ],
        after_observation_digest=row[  # type: ignore[arg-type]
            "after_observation_digest"
        ],
        stage=stage,
        stage_receipt_digest=row["stage_receipt_digest"],  # type: ignore[arg-type]
    )


def _parse_checkpoint(value: object) -> CanaryCheckpoint:
    row = _closed_object(value, _CHECKPOINT_FIELDS, "journal checkpoint")
    if row["schema"] != CANARY_CHECKPOINT_SCHEMA:
        raise _runtime_error("journal checkpoint schema is unsupported")
    started_ns = row["started_monotonic_ns"]
    deadline_ns = row["deadline_monotonic_ns"]
    transitions = row["transitions"]
    terminal = row["terminal_outcome"]
    if (
        type(started_ns) is not int
        or started_ns < 0
        or type(deadline_ns) is not int
        or deadline_ns < 0
        or type(transitions) is not list
    ):
        raise _runtime_error("journal checkpoint scalar fields are malformed")
    return CanaryCheckpoint(
        expected_reservation_digest=row[  # type: ignore[arg-type]
            "expected_reservation_digest"
        ],
        target_profile_digest=row["target_profile_digest"],  # type: ignore[arg-type]
        started_monotonic=started_ns / 1_000_000_000,
        deadline_monotonic=deadline_ns / 1_000_000_000,
        max_ticks=row["max_ticks"],  # type: ignore[arg-type]
        max_stage_receipts=row["max_stage_receipts"],  # type: ignore[arg-type]
        ticks_used=row["ticks_used"],  # type: ignore[arg-type]
        screen_claim_started=row["screen_claim_started"],  # type: ignore[arg-type]
        qualification_started=row["qualification_started"],  # type: ignore[arg-type]
        screen_lease_id=row["screen_lease_id"],  # type: ignore[arg-type]
        screen_request_id=row["screen_request_id"],  # type: ignore[arg-type]
        qualification_lease_id=row["qualification_lease_id"],  # type: ignore[arg-type]
        qualification_request_id=row[  # type: ignore[arg-type]
            "qualification_request_id"
        ],
        stage_receipt_digests=tuple(
            row["stage_receipt_digests"]  # type: ignore[arg-type]
        ),
        expensive_stage_receipt_digests=tuple(
            row["expensive_stage_receipt_digests"]  # type: ignore[arg-type]
        ),
        transitions=tuple(_parse_transition(item) for item in transitions),
        terminal_outcome=(
            None
            if terminal is None
            else _enum(CanaryTerminalOutcome, terminal, "checkpoint terminal outcome")
        ),
        terminal_reason=row["terminal_reason"],  # type: ignore[arg-type]
    )


class CanaryCheckpointJournal:
    """Single-process durable journal for one exact canary checkpoint."""

    def __init__(self, path: str | Path) -> None:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise _runtime_error("checkpoint journal requires no-follow file opens")
        requested = Path(path)
        if (
            not requested.is_absolute()
            or requested != Path(os.path.normpath(requested))
            or not requested.name
        ):
            raise _runtime_error(
                "checkpoint journal path is not canonical and absolute"
            )
        try:
            parent_before = requested.parent.lstat()
            parent = requested.parent.resolve(strict=True)
            parent_after = parent.stat()
        except OSError as exc:
            raise _runtime_error(
                f"checkpoint journal parent is unavailable: {exc}"
            ) from None
        if (
            parent != requested.parent
            or not stat.S_ISDIR(parent_before.st_mode)
            or (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
        ):
            raise _runtime_error("checkpoint journal parent is not canonical")
        self.path = parent / requested.name
        self._nofollow = nofollow
        self._closed = False
        self._check_existing(self.path, allow_missing=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        flags = os.O_RDWR | os.O_CREAT | nofollow | getattr(os, "O_CLOEXEC", 0)
        try:
            self._lock_fd = os.open(lock_path, flags, 0o600)
            os.fchmod(self._lock_fd, 0o600)
            info = os.fstat(self._lock_fd)
            self._require_owned_regular(info, "checkpoint journal lock")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._check_existing(self.path, allow_missing=True)
            self._check_no_partials()
            os.fsync(self._lock_fd)
        except BlockingIOError:
            if hasattr(self, "_lock_fd"):
                os.close(self._lock_fd)
            raise _runtime_error(
                "checkpoint journal already has an active owner"
            ) from None
        except OSError as exc:
            if hasattr(self, "_lock_fd"):
                os.close(self._lock_fd)
            raise _runtime_error(
                f"checkpoint journal lock cannot open: {exc}"
            ) from None
        except BaseException:
            if hasattr(self, "_lock_fd"):
                os.close(self._lock_fd)
            raise

    @staticmethod
    def _require_owned_regular(info: os.stat_result, label: str) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
        ):
            raise _runtime_error(f"{label} is not owner-controlled mode 0600")

    @classmethod
    def _check_existing(cls, path: Path, *, allow_missing: bool) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            if allow_missing:
                return
            raise _runtime_error("checkpoint journal is absent") from None
        cls._require_owned_regular(info, "checkpoint journal")

    def _check_no_partials(self) -> None:
        prefix = f".{self.path.name}."
        if any(row.name.startswith(prefix) for row in self.path.parent.iterdir()):
            raise _runtime_error("checkpoint journal has a stale partial")

    def _require_open(self) -> None:
        if self._closed:
            raise _runtime_error("checkpoint journal is closed")

    def _read_exact(self, path: Path) -> bytes:
        flags = os.O_RDONLY | self._nofollow | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise _runtime_error(f"checkpoint journal cannot open: {exc}") from None
        try:
            before = os.fstat(descriptor)
            self._require_owned_regular(before, "checkpoint journal file")
            if (
                not 1 <= before.st_size <= _MAX_JOURNAL_BYTES
            ):
                raise _runtime_error("checkpoint journal file shape is unsafe")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    raise _runtime_error("checkpoint journal read was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise _runtime_error("checkpoint journal changed while read")
            return raw
        finally:
            os.close(descriptor)

    def load(self) -> CanaryCheckpoint | None:
        self._require_open()
        self._check_existing(self.path, allow_missing=True)
        if not self.path.exists():
            return None
        raw = self._read_exact(self.path)
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_pairs,
                parse_float=_reject_number,
                parse_constant=_reject_number,
            )
            if canonical_json_bytes(value) != raw:
                raise _runtime_error("checkpoint journal is not canonical JSON")
        except OneReservationCanaryRuntimeError:
            raise
        except (
            UnicodeError,
            json.JSONDecodeError,
            StackIdentityError,
            ValueError,
        ) as exc:
            raise _runtime_error(f"checkpoint journal is corrupt: {exc}") from None
        envelope = _closed_object(value, _JOURNAL_FIELDS, "checkpoint journal")
        if envelope["schema"] != CANARY_CHECKPOINT_JOURNAL_SCHEMA:
            raise _runtime_error("checkpoint journal schema is unsupported")
        checkpoint = _parse_checkpoint(envelope["checkpoint"])
        retained_digest = _digest(
            envelope["checkpoint_digest"], "journal checkpoint digest"
        )
        if retained_digest != checkpoint.digest:
            raise _runtime_error("checkpoint journal digest differs")
        return checkpoint

    def persist(self, checkpoint: CanaryCheckpoint) -> None:
        self._require_open()
        if type(checkpoint) is not CanaryCheckpoint:
            raise _runtime_error("journal checkpoint is not exactly typed")
        self._check_existing(self.path, allow_missing=True)
        self._check_no_partials()
        raw = canonical_json_bytes(
            {
                "checkpoint": checkpoint.to_dict(),
                "checkpoint_digest": checkpoint.digest,
                "schema": CANARY_CHECKPOINT_JOURNAL_SCHEMA,
            }
        )
        if len(raw) > _MAX_JOURNAL_BYTES:
            raise _runtime_error("checkpoint journal exceeds its byte bound")
        descriptor: int | None = None
        temporary: Path | None = None
        temporary_identity: tuple[int, int] | None = None
        replaced = False
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", dir=self.path.parent
            )
            temporary = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            temporary_identity = (created.st_dev, created.st_ino)
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise _runtime_error("checkpoint journal write stalled")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            reopened = self._read_exact(temporary)
            if sha256_hex(reopened) != sha256_hex(raw) or reopened != raw:
                raise _runtime_error("checkpoint journal temporary hash differs")
            self._check_existing(self.path, allow_missing=True)
            os.replace(temporary, self.path)
            replaced = True
            installed = self._read_exact(self.path)
            if sha256_hex(installed) != sha256_hex(raw) or installed != raw:
                raise _runtime_error("checkpoint journal installed hash differs")
            parent_flags = (
                os.O_RDONLY
                | self._nofollow
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
            )
            parent_fd = os.open(self.path.parent, parent_flags)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OneReservationCanaryRuntimeError:
            raise
        except OSError as exc:
            raise _runtime_error(f"checkpoint journal persist failed: {exc}") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None and not replaced:
                try:
                    retained = temporary.lstat()
                    if (retained.st_dev, retained.st_ino) == temporary_identity:
                        temporary.unlink()
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._closed = True

    def __enter__(self) -> "CanaryCheckpointJournal":
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "CANARY_CHECKPOINT_JOURNAL_SCHEMA",
    "CanaryCheckpointJournal",
    "OneReservationCanaryRuntime",
    "OneReservationCanaryRuntimeError",
]

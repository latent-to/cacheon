"""Fence each qualification carrier without replacing its lease or experiment."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.chain.evaluation_recovery_hold import CompletedQualificationHoldMixin
from cacheon.chain.evaluation_recovery import (
    EvaluationRecovery,
    EvaluationRecoveryError,
    EvaluationRecoveryEvent,
    EvaluationRecoveryHoldError,
    RecoveryEventType,
    RecoveryPhase,
    RecoveryResolution,
    evaluation_recovery_event_id,
    evaluation_recovery_id,
    stale_incumbent_release_reason,
    valid_evaluation_recovery_event_transition as _valid_recovery_event_transition,
)
from cacheon.chain.evaluation_recovery_plan import (
    EvaluationRecoveryPlanError,
    decode_recovery_request_plan,
    encode_recovery_request_plan,
)
from cacheon.stack_identity import require_sha256_hex

if TYPE_CHECKING:
    from cacheon.chain.remote_worker_request_plan import QualificationRequestPlan


from cacheon.chain.evaluation_recovery_schema import (
    EvaluationRecoveryStoreError,
    configure_evaluation_recovery_connection,
    ensure_evaluation_recovery_schema,
)


def _intake_error(message: str) -> RuntimeError:
    from cacheon.chain.intake import IntakeError

    return IntakeError(message)


_PHASE_EVENTS = {
    RecoveryPhase.PREPARED: RecoveryEventType.PREPARED,
    RecoveryPhase.PUBLICATION_COMMITTED: RecoveryEventType.PUBLICATION_COMMITTED,
    RecoveryPhase.REQUEST_READY: RecoveryEventType.REQUEST_READY,
    RecoveryPhase.RESULT_READY: RecoveryEventType.RESULT_READY,
    RecoveryPhase.EVIDENCE_IMPORTED: RecoveryEventType.EVIDENCE_IMPORTED,
}


class EvaluationRecoveryStoreMixin(CompletedQualificationHoldMixin):
    @contextmanager
    def _evaluation_recovery_mutation(self, lease_id: str) -> Iterator[None]:
        if self._evaluation_recovery_mutation_authority:
            raise _intake_error("nested evaluation recovery mutation is forbidden")
        self._evaluation_recovery_mutation_authority.add(lease_id)
        try:
            yield
        finally:
            self._evaluation_recovery_mutation_authority.remove(lease_id)

    def _evaluation_recovery(self, row: sqlite3.Row) -> EvaluationRecovery:
        lease_row = self._db.execute(
            "SELECT * FROM evaluation_leases WHERE lease_id=?", (row["lease_id"],)
        ).fetchone()
        if lease_row is None:
            raise EvaluationRecoveryHoldError("evaluation recovery lost its lease")
        try:
            recovery = EvaluationRecovery(
                recovery_id=row["recovery_id"],
                lease=self._evaluation_lease(lease_row),
                revision=row["revision"],
                phase=RecoveryPhase(row["phase"]),
                resolution=RecoveryResolution(row["resolution"]),
                created_block=row["created_block"],
                updated_block=row["updated_block"],
                plan_digest=row["plan_digest"],
                request_id=row["request_id"],
                request_plan=bytes(row["request_plan"]),
                reason=row["reason"],
            )
        except (EvaluationRecoveryError, ValueError) as exc:
            raise EvaluationRecoveryHoldError(
                f"evaluation recovery is corrupt: {exc}"
            ) from None
        if recovery.request_plan:
            try:
                decode_recovery_request_plan(
                    recovery.request_plan,
                    expected_lease=recovery.lease,
                    expected_plan_digest=recovery.plan_digest,
                    expected_request_id=recovery.request_id,
                )
            except EvaluationRecoveryPlanError as exc:
                raise EvaluationRecoveryHoldError(
                    f"evaluation recovery request plan is corrupt: {exc}; HOLD"
                ) from None
        return recovery

    def _evaluation_recovery_event(
        self, row: sqlite3.Row
    ) -> EvaluationRecoveryEvent:
        try:
            return EvaluationRecoveryEvent(
                sequence=row["sequence"],
                event_id=row["event_id"],
                recovery_id=row["recovery_id"],
                lease_id=row["lease_id"],
                revision=row["revision"],
                event_type=RecoveryEventType(row["event_type"]),
                phase=RecoveryPhase(row["phase"]),
                resolution=RecoveryResolution(row["resolution"]),
                finalized_block=row["finalized_block"],
                expires_block=row["expires_block"],
                plan_digest=row["plan_digest"],
                request_id=row["request_id"],
                reason=row["reason"],
            )
        except (EvaluationRecoveryError, ValueError) as exc:
            raise EvaluationRecoveryHoldError(
                f"evaluation recovery event is corrupt: {exc}"
            ) from None

    def _append_evaluation_recovery_event_locked(
        self,
        recovery: EvaluationRecovery,
        event_type: RecoveryEventType,
        *,
        finalized_block: int,
    ) -> EvaluationRecoveryEvent:
        event_id = evaluation_recovery_event_id(
            recovery_id=recovery.recovery_id,
            lease_id=recovery.lease.lease_id,
            revision=recovery.revision,
            event_type=event_type,
            phase=recovery.phase,
            resolution=recovery.resolution,
            finalized_block=finalized_block,
            expires_block=recovery.lease.expires_block,
            plan_digest=recovery.plan_digest,
            request_id=recovery.request_id,
            reason=recovery.reason,
        )
        cursor = self._db.execute(
            "INSERT INTO evaluation_recovery_events(event_id,recovery_id,lease_id,"
            "revision,event_type,phase,resolution,finalized_block,expires_block,"
            "plan_digest,request_id,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                recovery.recovery_id,
                recovery.lease.lease_id,
                recovery.revision,
                event_type.value,
                recovery.phase.value,
                recovery.resolution.value,
                finalized_block,
                recovery.lease.expires_block,
                recovery.plan_digest,
                recovery.request_id,
                recovery.reason,
            ),
        )
        row = self._db.execute(
            "SELECT * FROM evaluation_recovery_events WHERE sequence=?",
            (cursor.lastrowid,),
        ).fetchone()
        if row is None:
            raise _intake_error("evaluation recovery event was not retained")
        return self._evaluation_recovery_event(row)

    def _create_evaluation_recovery_locked(
        self, lease: EvaluationLease
    ) -> EvaluationRecovery:
        recovery = EvaluationRecovery(
            recovery_id=evaluation_recovery_id(lease),
            lease=lease,
            revision=0,
            phase=RecoveryPhase.CLAIMED,
            resolution=RecoveryResolution.UNRESOLVED,
            created_block=lease.claimed_block,
            updated_block=lease.claimed_block,
        )
        self._db.execute(
            "INSERT INTO evaluation_recoveries(recovery_id,lease_id,revision,phase,"
            "resolution,created_block,updated_block,request_plan,plan_digest,request_id,"
            "reason,competition_arena) VALUES(?,?,0,'claimed','',?,?,X'','','','',?)",
            (
                recovery.recovery_id,
                lease.lease_id,
                lease.claimed_block,
                lease.claimed_block,
                self._competition_arena,
            ),
        )
        self._append_evaluation_recovery_event_locked(
            recovery, RecoveryEventType.CLAIMED, finalized_block=lease.claimed_block
        )
        return recovery

    def _reopen_evaluation_recovery_for_lease(
        self, lease: EvaluationLease
    ) -> EvaluationRecovery:
        row = self._db.execute(
            "SELECT * FROM evaluation_recoveries WHERE lease_id=?",
            (lease.lease_id,),
        ).fetchone()
        if row is None:
            raise EvaluationRecoveryHoldError(
                "active qualification lease has no recovery authority; HOLD"
            )
        recovery = self._evaluation_recovery(row)
        if recovery.lease != lease:
            raise _intake_error("evaluation recovery lease object is stale")
        return recovery

    def _active_qualification_recovery(
        self, lease: EvaluationLease
    ) -> EvaluationRecovery:
        if type(lease) is not EvaluationLease or lease.stage != "qualification":
            raise _intake_error("evaluation recovery requires a qualification lease")
        recovery = self._reopen_evaluation_recovery_for_lease(lease)
        if recovery.resolution is not RecoveryResolution.UNRESOLVED:
            raise EvaluationRecoveryHoldError(
                "active qualification lease has a resolved recovery; HOLD"
            )
        self.evaluation_recovery_events(recovery)
        return recovery

    def _require_no_orphan_active_qualification(self) -> None:
        for active in self._active_qualification_rows():
            self._active_qualification_recovery(self._evaluation_lease(active))

    def _generic_lease_operation_allowed(
        self, lease: EvaluationLease, operation: str
    ) -> None:
        if lease.stage != "qualification":
            return
        recovery = self._active_qualification_recovery(lease)
        raise _intake_error(
            f"protected evaluation recovery forbids generic {operation}; "
            f"action={recovery.action.value}"
        )

    def claim_recoverable_qualification(
        self,
        *,
        owner: str,
        current_block: int,
        lease_blocks: int = 30,
        max_members: int | None = None,
        max_active: int = 1,
    ) -> EvaluationRecovery | None:
        """Atomically claim the next cohort and retain its recovery intent."""
        lease = self.claim_evaluation_lease(
            stage="qualification",
            owner=owner,
            current_block=current_block,
            lease_blocks=lease_blocks,
            max_members=max_members,
            max_active=max_active,
        )
        if lease is None:
            return None
        return self._active_qualification_recovery(lease)

    def pending_qualification_recovery(self, *, owner: str | None = None) -> EvaluationRecovery | None:
        """Reopen one worker's exact request, retaining the unambiguous legacy reader."""
        rows = self._active_qualification_rows(owner=owner)
        if not rows:
            return None
        if len(rows) != 1:
            raise EvaluationRecoveryHoldError(
                "multiple active qualification leases exist; HOLD"
            )
        return self._active_qualification_recovery(self._evaluation_lease(rows[0]))

    def reopen_recovery_request_plan(
        self, recovery: EvaluationRecovery
    ) -> "QualificationRequestPlan":
        """Reopen the one canonical request plan retained by this recovery."""
        if type(recovery) is not EvaluationRecovery:
            raise _intake_error("evaluation recovery is not exactly typed")
        current = self._active_qualification_recovery(recovery.lease)
        if current != recovery or not current.request_plan:
            raise EvaluationRecoveryHoldError(
                "evaluation recovery has no exact request plan; HOLD"
            )
        try:
            return decode_recovery_request_plan(
                current.request_plan,
                expected_lease=current.lease,
                expected_plan_digest=current.plan_digest,
                expected_request_id=current.request_id,
            )
        except EvaluationRecoveryPlanError as exc:
            raise EvaluationRecoveryHoldError(
                f"evaluation recovery request plan cannot reopen: {exc}; HOLD"
            ) from None

    def evaluation_recovery_events(
        self, recovery: EvaluationRecovery
    ) -> tuple[EvaluationRecoveryEvent, ...]:
        if type(recovery) is not EvaluationRecovery:
            raise _intake_error("evaluation recovery is not exactly typed")
        current = self._reopen_evaluation_recovery_for_lease(recovery.lease)
        events = tuple(
            self._evaluation_recovery_event(row)
            for row in self._db.execute(
                "SELECT * FROM evaluation_recovery_events WHERE recovery_id=? "
                "ORDER BY revision",
                (recovery.recovery_id,),
            )
        )
        if not events or events[0].event_type is not RecoveryEventType.CLAIMED:
            raise EvaluationRecoveryHoldError(
                "evaluation recovery history has no exact claim; HOLD"
            )
        previous = events[0]
        if (
            previous.revision != 0
            or previous.phase is not RecoveryPhase.CLAIMED
            or previous.resolution is not RecoveryResolution.UNRESOLVED
            or previous.reason
            or previous.plan_digest
            or previous.request_id
            or previous.lease_id != current.lease.lease_id
            or previous.finalized_block != current.created_block
            or previous.expires_block != current.lease.initial_expires_block
        ):
            raise EvaluationRecoveryHoldError(
                "evaluation recovery claim event is inconsistent; HOLD"
            )
        for event in events[1:]:
            if (
                event.lease_id != current.lease.lease_id
                or event.sequence <= previous.sequence
                or not _valid_recovery_event_transition(previous, event)
            ):
                raise EvaluationRecoveryHoldError(
                    "evaluation recovery event stream is inconsistent; HOLD"
                )
            previous = event
        last = events[-1]
        if (
            last.revision != current.revision
            or last.phase is not current.phase
            or last.resolution is not current.resolution
            or last.expires_block != current.lease.expires_block
            or last.finalized_block != current.updated_block
            or last.plan_digest != current.plan_digest
            or last.request_id != current.request_id
            or last.reason != current.reason
        ):
            raise EvaluationRecoveryHoldError(
                "evaluation recovery head differs from immutable history; HOLD"
            )
        return events

    def _transition_evaluation_recovery_locked(
        self,
        recovery: EvaluationRecovery,
        *,
        phase: RecoveryPhase,
        resolution: RecoveryResolution,
        event_type: RecoveryEventType,
        current_block: int,
        reason: str = "",
        lease: EvaluationLease | None = None,
        plan_binding: tuple[bytes, str, str] | None = None,
    ) -> EvaluationRecovery:
        current = self._active_qualification_recovery(recovery.lease)
        if current != recovery:
            raise _intake_error("evaluation recovery object is stale")
        retained_lease = recovery.lease if lease is None else lease
        request_plan, plan_digest, request_id = (
            (recovery.request_plan, recovery.plan_digest, recovery.request_id)
            if plan_binding is None
            else plan_binding
        )
        updated = EvaluationRecovery(
            recovery_id=recovery.recovery_id,
            lease=retained_lease,
            revision=recovery.revision + 1,
            phase=phase,
            resolution=resolution,
            created_block=recovery.created_block,
            updated_block=current_block,
            plan_digest=plan_digest,
            request_id=request_id,
            request_plan=request_plan,
            reason=reason,
        )
        cursor = self._db.execute(
            "UPDATE evaluation_recoveries SET revision=?,phase=?,resolution=?,"
            "updated_block=?,request_plan=?,plan_digest=?,request_id=?,reason=? "
            "WHERE recovery_id=? AND revision=? AND phase=? AND resolution=? "
            "AND updated_block=? AND request_plan=? AND plan_digest=? AND request_id=? "
            "AND reason=?",
            (
                updated.revision,
                updated.phase.value,
                updated.resolution.value,
                updated.updated_block,
                updated.request_plan,
                updated.plan_digest,
                updated.request_id,
                updated.reason,
                recovery.recovery_id,
                recovery.revision,
                recovery.phase.value,
                recovery.resolution.value,
                recovery.updated_block,
                recovery.request_plan,
                recovery.plan_digest,
                recovery.request_id,
                recovery.reason,
            ),
        )
        if cursor.rowcount != 1:
            raise _intake_error("evaluation recovery changed during transition")
        self._append_evaluation_recovery_event_locked(
            updated, event_type, finalized_block=current_block
        )
        return updated

    def _advance_recovery_phase(
        self,
        recovery: EvaluationRecovery,
        *,
        expected: tuple[RecoveryPhase, ...],
        phase: RecoveryPhase,
        current_block: int,
        request_plan: "QualificationRequestPlan | None" = None,
    ) -> EvaluationRecovery:
        self._require_evaluation_clock(current_block)
        if (
            type(recovery) is not EvaluationRecovery
            or recovery.resolution is not RecoveryResolution.UNRESOLVED
            or recovery.phase not in expected
            or current_block >= recovery.lease.expires_block
        ):
            raise _intake_error("evaluation recovery phase transition is forbidden")
        plan_binding = None
        if phase is RecoveryPhase.PREPARED:
            if request_plan is None:
                raise _intake_error("preparing recovery requires an exact request plan")
            try:
                plan_binding = encode_recovery_request_plan(
                    request_plan, expected_lease=recovery.lease
                )
            except EvaluationRecoveryPlanError as exc:
                raise _intake_error(f"qualification request plan is invalid: {exc}")
        elif request_plan is not None:
            raise _intake_error("request plan may only bind the prepared transition")
        with self._transaction():
            with self._evaluation_recovery_mutation(recovery.lease.lease_id):
                return self._transition_evaluation_recovery_locked(
                    recovery,
                    phase=phase,
                    resolution=RecoveryResolution.UNRESOLVED,
                    event_type=_PHASE_EVENTS[phase],
                    current_block=current_block,
                    plan_binding=plan_binding,
                )

    def prepare_qualification_recovery(
        self,
        recovery: EvaluationRecovery,
        request_plan: "QualificationRequestPlan",
        *,
        current_block: int,
    ) -> EvaluationRecovery:
        return self._advance_recovery_phase(
            recovery,
            expected=(RecoveryPhase.CLAIMED,),
            phase=RecoveryPhase.PREPARED,
            current_block=current_block,
            request_plan=request_plan,
        )

    def commit_recovery_publication(
        self, recovery: EvaluationRecovery, *, current_block: int
    ) -> EvaluationRecovery:
        return self._advance_recovery_phase(
            recovery,
            expected=(RecoveryPhase.PREPARED,),
            phase=RecoveryPhase.PUBLICATION_COMMITTED,
            current_block=current_block,
        )

    def observe_recovery_request_ready(
        self, recovery: EvaluationRecovery, *, current_block: int
    ) -> EvaluationRecovery:
        return self._advance_recovery_phase(
            recovery,
            expected=(RecoveryPhase.PUBLICATION_COMMITTED,),
            phase=RecoveryPhase.REQUEST_READY,
            current_block=current_block,
        )

    def record_recovery_result(
        self, recovery: EvaluationRecovery, *, current_block: int
    ) -> EvaluationRecovery:
        return self._advance_recovery_phase(
            recovery,
            expected=(
                RecoveryPhase.PUBLICATION_COMMITTED,
                RecoveryPhase.REQUEST_READY,
            ),
            phase=RecoveryPhase.RESULT_READY,
            current_block=current_block,
        )

    def record_recovery_import(
        self, recovery: EvaluationRecovery, *, current_block: int
    ) -> EvaluationRecovery:
        return self._advance_recovery_phase(
            recovery,
            expected=(RecoveryPhase.RESULT_READY,),
            phase=RecoveryPhase.EVIDENCE_IMPORTED,
            current_block=current_block,
        )

    def renew_recovery_lease(
        self,
        recovery: EvaluationRecovery,
        *,
        current_block: int,
        lease_blocks: int = 30,
    ) -> tuple[EvaluationRecovery, EvaluationLease]:
        if (
            type(recovery) is not EvaluationRecovery
            or recovery.resolution is not RecoveryResolution.UNRESOLVED
            or recovery.phase is RecoveryPhase.HELD
            or type(lease_blocks) is not int
            or lease_blocks <= 0
            or lease_blocks > self.policy.expiry_blocks
        ):
            raise _intake_error("evaluation recovery renewal is malformed")
        self._require_evaluation_clock(current_block)
        with self._transaction():
            current = self._active_qualification_recovery(recovery.lease)
            if current != recovery:
                raise _intake_error("evaluation recovery object is stale")
            expires = current_block + lease_blocks
            if expires <= recovery.lease.expires_block:
                raise _intake_error("evaluation recovery renewal does not extend lease")
            renewed_lease = EvaluationLease(
                recovery.lease.lease_id,
                recovery.lease.generation,
                recovery.lease.stage,
                recovery.lease.owner,
                recovery.lease.members,
                recovery.lease.claimed_block,
                recovery.lease.initial_expires_block,
                expires,
            )
            with self._evaluation_recovery_mutation(recovery.lease.lease_id):
                renewed = self._transition_evaluation_recovery_locked(
                    recovery,
                    phase=recovery.phase,
                    resolution=RecoveryResolution.UNRESOLVED,
                    event_type=RecoveryEventType.RENEWED,
                    current_block=current_block,
                    lease=renewed_lease,
                )
                cursor = self._db.execute(
                    "UPDATE evaluation_leases SET expires_block=? WHERE lease_id=? "
                    "AND state='active' AND expires_block=?",
                    (expires, recovery.lease.lease_id, recovery.lease.expires_block),
                )
                if cursor.rowcount != 1:
                    raise _intake_error("evaluation lease changed during recovery renewal")
        return renewed, renewed_lease

    def hold_recovery(
        self,
        recovery: EvaluationRecovery,
        *,
        current_block: int,
        reason: str,
    ) -> EvaluationRecovery:
        self._require_evaluation_clock(current_block)
        if (
            type(recovery) is not EvaluationRecovery
            or recovery.resolution is not RecoveryResolution.UNRESOLVED
            or recovery.phase is RecoveryPhase.HELD
            or not isinstance(reason, str)
            or not reason
            or reason.strip() != reason
            or len(reason) > 2_048
            or any(ord(char) < 32 or ord(char) == 127 for char in reason)
        ):
            raise _intake_error("evaluation recovery hold is malformed")
        with self._transaction():
            with self._evaluation_recovery_mutation(recovery.lease.lease_id):
                return self._transition_evaluation_recovery_locked(
                    recovery,
                    phase=RecoveryPhase.HELD,
                    resolution=RecoveryResolution.UNRESOLVED,
                    event_type=RecoveryEventType.HELD,
                    current_block=current_block,
                    reason=reason,
                )

    def release_worker_pre_resident_recovery(
        self,
        recovery: EvaluationRecovery,
        *,
        refusal: object,
        current_block: int,
    ) -> EvaluationLease:
        """Requeue one published request after an authenticated, marker-absent
        pre-resident refusal signed by the pod.  Never generic; never a rerun
        of anything that may have entered resident execution."""

        from cacheon.chain.execution_disposition import AuthenticatedPreResidentRefusal

        if (
            type(refusal) is not AuthenticatedPreResidentRefusal
            or type(recovery) is not EvaluationRecovery
            or recovery.phase is not RecoveryPhase.REQUEST_READY
            or refusal.request_id != recovery.request_id
        ):
            raise _intake_error("worker pre-resident recovery release is forbidden")
        return self._release_recovery(
            recovery, current_block=current_block, reason=refusal.release_reason
        )

    def release_worker_infrastructure_recovery(
        self,
        recovery: EvaluationRecovery,
        *,
        failure_code: str,
        current_block: int,
        live_worker_epoch: str = "",
    ) -> EvaluationLease:
        """Requeue one published request the worker terminated with an unproven
        infrastructure result (no authenticated refusal, no completed
        response).  The dead request retires with its recovery and a fresh
        claim mints a fresh request.  Also accepts a recovery already parked
        HELD under the pre-change worker-infrastructure reason or under the
        authority-changed reason (both mean the retained request is durably
        dead), migrating it into the same requeue; a completed-product hold
        joins them only when the caller names the live worker epoch and the
        retained request plan provably binds a different one -- the store
        verifies the mismatch against its own sealed plan, so an orphan of a
        torn-down epoch migrates while a live-epoch hold stays parked.
        Repeats are bounded by the systemic release cap, so an unfixed
        infrastructure fault parks for the operator instead of free-looping."""

        from cacheon.chain.execution_disposition import (
            AUTHORITY_CHANGED_HOLD_REASON,
            COMPLETED_NO_DECISION_HOLD_REASON,
            ORPHANED_CARRIER_HOLD_REASON,
            WORKER_INFRASTRUCTURE_HOLD_REASON,
        )

        parked_held = (
            type(recovery) is EvaluationRecovery
            and recovery.phase is RecoveryPhase.HELD
        )
        completed_orphan = (
            parked_held
            and recovery.reason == COMPLETED_NO_DECISION_HOLD_REASON
            and isinstance(live_worker_epoch, str)
            and re.fullmatch(r"[0-9a-f]{32}", live_worker_epoch) is not None
            and self.reopen_recovery_request_plan(recovery).worker_epoch
            != live_worker_epoch
        )
        held_migration = completed_orphan or (
            parked_held
            and recovery.reason
            in (
                WORKER_INFRASTRUCTURE_HOLD_REASON,
                AUTHORITY_CHANGED_HOLD_REASON,
                ORPHANED_CARRIER_HOLD_REASON,
            )
        )
        if (
            type(recovery) is not EvaluationRecovery
            or not isinstance(failure_code, str)
            or not failure_code
            or failure_code.strip() != failure_code
            or len(failure_code) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in failure_code)
            or not (recovery.phase is RecoveryPhase.REQUEST_READY or held_migration)
        ):
            raise _intake_error("worker infrastructure recovery release is forbidden")
        lease = self._release_recovery(
            recovery,
            current_block=current_block,
            reason=f"systemic:worker_infrastructure:{failure_code}",
            # A resolved recovery may not remain HELD; the migration releases
            # back through the phase it was parked from.
            release_phase=(
                RecoveryPhase.REQUEST_READY if held_migration else None
            ),
            allow_expired=held_migration,
        )
        self._cap_infrastructure_releases(lease)
        return lease

    def release_stale_incumbent_qualification_recovery(
        self,
        recovery: EvaluationRecovery,
        *,
        product: object,
        live_stack: object,
        live_tree_digest: str,
        current_block: int,
    ) -> EvaluationLease:
        """Retire a completed qualification product bound to an old baseline."""

        from cacheon.chain.remote_qualification_evidence import (
            RemoteQualificationProduct,
        )
        from cacheon.stack_manifest import EvaluationStackManifest

        if (
            type(recovery) is not EvaluationRecovery
            or recovery.phase
            not in {
                RecoveryPhase.REQUEST_READY,
                RecoveryPhase.RESULT_READY,
                RecoveryPhase.EVIDENCE_IMPORTED,
            }
            or type(product) is not RemoteQualificationProduct
            or type(live_stack) is not EvaluationStackManifest
        ):
            raise _intake_error("stale-incumbent recovery release is forbidden")
        try:
            live_tree = require_sha256_hex(
                live_tree_digest, field="live qualification tree digest"
            )
        except (TypeError, ValueError) as exc:
            raise _intake_error("stale-incumbent recovery release is forbidden") from exc
        if (
            (
                product.incumbent_stack.digest == live_stack.digest
                and product.incumbent_tree_digest == live_tree
            )
            or product.incumbent_stack.catalog_digest != live_stack.catalog_digest
            or product.incumbent_stack.arena_digest != live_stack.arena_digest
            or product.service_digest != live_stack.arena_digest
        ):
            raise _intake_error("stale-incumbent recovery release is forbidden")
        lease = self._release_recovery(
            recovery,
            current_block=current_block,
            reason=stale_incumbent_release_reason(
                product_digest=product.digest,
                previous_stack_digest=product.incumbent_stack.digest,
                previous_tree_digest=product.incumbent_tree_digest,
                live_stack_digest=live_stack.digest,
                live_tree_digest=live_tree,
            ),
            release_phase=RecoveryPhase.REQUEST_READY,
            allow_expired=True,
        )
        self._cap_infrastructure_releases(lease)
        return lease

    def _release_recovery(
        self,
        recovery: EvaluationRecovery,
        *,
        current_block: int,
        reason: str,
        release_phase: RecoveryPhase | None = None,
        allow_expired: bool = False,
    ) -> EvaluationLease:
        self._require_evaluation_clock(current_block)
        if (
            recovery.resolution is not RecoveryResolution.UNRESOLVED
            or (current_block >= recovery.lease.expires_block and not allow_expired)
        ):
            raise _intake_error("pre-resident recovery release is forbidden")
        with self._transaction():
            self._active_evaluation_lease_row(recovery.lease)
            if any(
                self.get(member.reservation_id).status != member.prior_status
                for member in recovery.lease.members
            ):
                raise _intake_error("recovery lease no longer has its exact queue state")
            with self._evaluation_recovery_mutation(recovery.lease.lease_id):
                resolved = self._transition_evaluation_recovery_locked(
                    recovery,
                    phase=recovery.phase if release_phase is None else release_phase,
                    resolution=RecoveryResolution.PRE_RESIDENT_RELEASED,
                    event_type=RecoveryEventType.PRE_RESIDENT_RELEASED,
                    current_block=current_block,
                    reason=reason,
                )
                cursor = self._db.execute(
                    "UPDATE evaluation_leases SET state='released',completed_block=?,"
                    "reason=?,result_digest='' WHERE lease_id=? AND state='active' "
                    "AND expires_block=?",
                    (
                        current_block,
                        reason,
                        recovery.lease.lease_id,
                        recovery.lease.expires_block,
                    ),
                )
                if cursor.rowcount != 1:
                    raise _intake_error("evaluation lease changed during recovery release")
                members = self._db.execute(
                    "UPDATE evaluation_lease_members SET active=0 WHERE lease_id=? "
                    "AND active=1",
                    (recovery.lease.lease_id,),
                )
                if members.rowcount != len(recovery.lease.members):
                    raise _intake_error("recovery lease members changed during release")
                self._append_evaluation_lease_event(
                    recovery.lease,
                    "released",
                    finalized_block=current_block,
                    reason=reason,
                )
            if resolved.resolution is not RecoveryResolution.PRE_RESIDENT_RELEASED:
                raise _intake_error("recovery release resolution was not retained")
        return recovery.lease

    def _complete_evaluation_recovery_locked(
        self, recovery: EvaluationRecovery, *, current_block: int
    ) -> EvaluationRecovery:
        if recovery.phase is RecoveryPhase.HELD:
            raise _intake_error("held evaluation recovery cannot commit")
        return self._transition_evaluation_recovery_locked(
            recovery,
            phase=recovery.phase,
            resolution=RecoveryResolution.COMMITTED,
            event_type=RecoveryEventType.COMMITTED,
            current_block=current_block,
        )


__all__ = [
    "EvaluationRecoveryStoreError",
    "EvaluationRecoveryStoreMixin",
    "configure_evaluation_recovery_connection",
    "ensure_evaluation_recovery_schema",
]

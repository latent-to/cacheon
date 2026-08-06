"""Additive SQLite authority for restart-safe qualification ownership.

The recovery store fences an exact qualification lease while a durable carrier
is being orchestrated.  It does not interpret resident execution evidence and
never creates a replacement request, lease generation, or experiment.
"""

from __future__ import annotations

import sqlite3
from collections.abc import MutableSet
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from cacheon.chain.evaluation_leases import EvaluationLease
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
    valid_evaluation_recovery_event_transition,
)
from cacheon.chain.evaluation_recovery_plan import (
    EvaluationRecoveryPlanError,
    decode_recovery_request_plan,
    encode_recovery_request_plan,
)

if TYPE_CHECKING:
    from cacheon.chain.remote_worker_request_plan import QualificationRequestPlan


class EvaluationRecoveryStoreError(RuntimeError):
    """The additive recovery schema or a retained authority cannot be opened."""


def configure_evaluation_recovery_connection(
    db: sqlite3.Connection, mutation_authority: MutableSet[str]
) -> None:
    """Install the connection-local capability used by recovery SQL triggers."""

    db.create_function(
        "cacheon_evaluation_recovery_mutation_authorized",
        1,
        lambda lease_id: int(lease_id in mutation_authority),
    )


def ensure_evaluation_recovery_schema(db: sqlite3.Connection) -> None:
    """Create or verify additive recovery schema version 1 and its backstops."""

    try:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_recoveries (
                recovery_id TEXT PRIMARY KEY,
                lease_id TEXT NOT NULL UNIQUE REFERENCES evaluation_leases(lease_id),
                revision INTEGER NOT NULL CHECK(revision>=0),
                phase TEXT NOT NULL CHECK(phase IN (
                    'claimed','prepared','publication_committed','request_ready',
                    'result_ready','evidence_imported','held'
                )),
                resolution TEXT NOT NULL CHECK(resolution IN (
                    '','pre_resident_released','committed'
                )),
                created_block INTEGER NOT NULL CHECK(created_block>=0),
                updated_block INTEGER NOT NULL CHECK(updated_block>=created_block),
                request_plan BLOB NOT NULL DEFAULT X'',
                plan_digest TEXT NOT NULL DEFAULT '',
                request_id TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                CHECK((phase='held' OR resolution='pre_resident_released')=(reason!='')),
                CHECK(NOT (phase='held' AND resolution!='')),
                CHECK(
                    (length(request_plan)=0 AND plan_digest='' AND request_id=''
                     AND phase IN ('claimed','held'))
                    OR
                    (typeof(request_plan)='blob' AND length(request_plan)>0
                     AND length(request_plan)<=4194304
                     AND length(plan_digest)=64 AND length(request_id)=64)
                )
            ) STRICT;
            CREATE UNIQUE INDEX IF NOT EXISTS evaluation_recoveries_one_unresolved
                ON evaluation_recoveries(resolution) WHERE resolution='';
            CREATE TABLE IF NOT EXISTS evaluation_recovery_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                recovery_id TEXT NOT NULL REFERENCES evaluation_recoveries(recovery_id),
                lease_id TEXT NOT NULL REFERENCES evaluation_leases(lease_id),
                revision INTEGER NOT NULL CHECK(revision>=0),
                event_type TEXT NOT NULL CHECK(event_type IN (
                    'claimed','prepared','publication_committed','request_ready',
                    'result_ready','evidence_imported','renewed','held',
                    'pre_resident_released','committed'
                )),
                phase TEXT NOT NULL CHECK(phase IN (
                    'claimed','prepared','publication_committed','request_ready',
                    'result_ready','evidence_imported','held'
                )),
                resolution TEXT NOT NULL CHECK(resolution IN (
                    '','pre_resident_released','committed'
                )),
                finalized_block INTEGER NOT NULL CHECK(finalized_block>=0),
                expires_block INTEGER NOT NULL CHECK(expires_block>0),
                plan_digest TEXT NOT NULL DEFAULT '',
                request_id TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                CHECK(
                    (plan_digest='' AND request_id='' AND phase IN ('claimed','held'))
                    OR (length(plan_digest)=64 AND length(request_id)=64)
                ),
                UNIQUE(recovery_id, revision)
            ) STRICT;
            CREATE TRIGGER IF NOT EXISTS evaluation_recovery_events_reject_update
                BEFORE UPDATE ON evaluation_recovery_events
                BEGIN SELECT RAISE(ABORT,'evaluation recovery events are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS evaluation_recovery_events_reject_delete
                BEFORE DELETE ON evaluation_recovery_events
                BEGIN SELECT RAISE(ABORT,'evaluation recovery events are immutable'); END;

            DROP TRIGGER IF EXISTS evaluation_recoveries_require_insert_authority;
            CREATE TRIGGER evaluation_recoveries_require_insert_authority
                BEFORE INSERT ON evaluation_recoveries
                WHEN cacheon_evaluation_recovery_mutation_authorized(NEW.lease_id)=0
                BEGIN SELECT RAISE(ABORT,'evaluation recovery mutation is unauthorized'); END;
            DROP TRIGGER IF EXISTS evaluation_recoveries_require_update_authority;
            CREATE TRIGGER evaluation_recoveries_require_update_authority
                BEFORE UPDATE ON evaluation_recoveries
                WHEN cacheon_evaluation_recovery_mutation_authorized(OLD.lease_id)=0
                BEGIN SELECT RAISE(ABORT,'evaluation recovery mutation is unauthorized'); END;
            DROP TRIGGER IF EXISTS evaluation_recoveries_reject_delete;
            CREATE TRIGGER evaluation_recoveries_reject_delete
                BEFORE DELETE ON evaluation_recoveries
                BEGIN SELECT RAISE(ABORT,'evaluation recoveries are immutable'); END;
            DROP TRIGGER IF EXISTS evaluation_recovery_events_require_insert_authority;
            CREATE TRIGGER evaluation_recovery_events_require_insert_authority
                BEFORE INSERT ON evaluation_recovery_events
                WHEN cacheon_evaluation_recovery_mutation_authorized(NEW.lease_id)=0
                BEGIN SELECT RAISE(ABORT,'evaluation recovery event is unauthorized'); END;

            -- A qualification lease is always recovery-owned once this schema
            -- creates it.  Existing active rows without a recovery record are
            -- treated as ambiguous HOLD state and receive the same backstop.
            DROP TRIGGER IF EXISTS evaluation_qualification_lease_insert_guard;
            CREATE TRIGGER evaluation_qualification_lease_insert_guard
                BEFORE INSERT ON evaluation_leases
                WHEN NEW.stage='qualification'
                 AND cacheon_evaluation_recovery_mutation_authorized(NEW.lease_id)=0
                BEGIN SELECT RAISE(ABORT,'qualification lease requires recovery authority'); END;
            DROP TRIGGER IF EXISTS evaluation_qualification_lease_update_guard;
            CREATE TRIGGER evaluation_qualification_lease_update_guard
                BEFORE UPDATE ON evaluation_leases
                WHEN OLD.stage='qualification' AND OLD.state='active'
                 AND cacheon_evaluation_recovery_mutation_authorized(OLD.lease_id)=0
                BEGIN SELECT RAISE(ABORT,'protected qualification lease mutation'); END;
            DROP TRIGGER IF EXISTS evaluation_qualification_lease_delete_guard;
            CREATE TRIGGER evaluation_qualification_lease_delete_guard
                BEFORE DELETE ON evaluation_leases
                WHEN OLD.stage='qualification' AND OLD.state='active'
                BEGIN SELECT RAISE(ABORT,'protected qualification lease deletion'); END;
            DROP TRIGGER IF EXISTS evaluation_qualification_member_update_guard;
            CREATE TRIGGER evaluation_qualification_member_update_guard
                BEFORE UPDATE ON evaluation_lease_members
                WHEN OLD.active=1
                 AND EXISTS (
                    SELECT 1 FROM evaluation_leases AS el
                    WHERE el.lease_id=OLD.lease_id AND el.stage='qualification'
                         AND el.state='active'
                 )
                 AND cacheon_evaluation_recovery_mutation_authorized(OLD.lease_id)=0
                BEGIN SELECT RAISE(ABORT,'protected qualification member mutation'); END;
            DROP TRIGGER IF EXISTS evaluation_qualification_member_delete_guard;
            CREATE TRIGGER evaluation_qualification_member_delete_guard
                BEFORE DELETE ON evaluation_lease_members
                WHEN OLD.active=1
                 AND EXISTS (
                    SELECT 1 FROM evaluation_leases AS el
                    WHERE el.lease_id=OLD.lease_id AND el.stage='qualification'
                         AND el.state='active'
                )
                BEGIN SELECT RAISE(ABORT,'protected qualification member deletion'); END;
            DROP TRIGGER IF EXISTS evaluation_qualification_member_insert_guard;
            CREATE TRIGGER evaluation_qualification_member_insert_guard
                BEFORE INSERT ON evaluation_lease_members
                WHEN EXISTS (
                    SELECT 1 FROM evaluation_leases AS el
                    WHERE el.lease_id=NEW.lease_id AND el.stage='qualification'
                         AND el.state='active'
                )
                 AND cacheon_evaluation_recovery_mutation_authorized(NEW.lease_id)=0
                BEGIN SELECT RAISE(ABORT,'protected qualification member insertion'); END;
            """
        )
    except sqlite3.Error as exc:
        raise EvaluationRecoveryStoreError(
            f"evaluation recovery schema creation failed: {exc}"
        ) from None

    required = {
        "evaluation_recoveries": {
            "recovery_id", "lease_id", "revision", "phase", "resolution",
            "created_block", "updated_block", "request_plan", "plan_digest",
            "request_id", "reason",
        },
        "evaluation_recovery_events": {
            "sequence", "event_id", "recovery_id", "lease_id", "revision",
            "event_type", "phase", "resolution", "finalized_block",
            "expires_block", "plan_digest", "request_id", "reason",
        },
    }
    if any(
        not columns.issubset(
            {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
        )
        for table, columns in required.items()
    ):
        raise EvaluationRecoveryStoreError("evaluation recovery schema is incomplete")
    required_triggers = {
        "evaluation_recovery_events_reject_update",
        "evaluation_recovery_events_reject_delete",
        "evaluation_recoveries_require_insert_authority",
        "evaluation_recoveries_require_update_authority",
        "evaluation_recoveries_reject_delete",
        "evaluation_recovery_events_require_insert_authority",
        "evaluation_qualification_lease_insert_guard",
        "evaluation_qualification_lease_update_guard",
        "evaluation_qualification_lease_delete_guard",
        "evaluation_qualification_member_update_guard",
        "evaluation_qualification_member_delete_guard",
        "evaluation_qualification_member_insert_guard",
    }
    retained_triggers = {
        row["name"]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    if not required_triggers.issubset(retained_triggers):
        raise EvaluationRecoveryStoreError(
            "evaluation recovery schema triggers are incomplete"
        )
    schema = db.execute(
        "SELECT value FROM metadata WHERE key='evaluation_recovery_schema'"
    ).fetchone()
    if schema is None:
        db.execute(
            "INSERT INTO metadata(key,value) VALUES('evaluation_recovery_schema','1')"
        )
    elif schema["value"] != "1":
        raise EvaluationRecoveryStoreError(
            "evaluation recovery schema is unsupported"
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


def _valid_recovery_event_transition(
    previous: EvaluationRecoveryEvent, event: EvaluationRecoveryEvent
) -> bool:
    return valid_evaluation_recovery_event_transition(previous, event)


class EvaluationRecoveryStoreMixin:
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
            "reason) VALUES(?,?,0,'claimed','',?,?,X'','','','')",
            (
                recovery.recovery_id,
                lease.lease_id,
                lease.claimed_block,
                lease.claimed_block,
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
        active = self._db.execute(
            "SELECT * FROM evaluation_leases WHERE stage='qualification' "
            "AND state='active' LIMIT 1"
        ).fetchone()
        if active is not None:
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
    ) -> EvaluationRecovery | None:
        """Atomically claim the next cohort and retain its recovery intent."""

        lease = self.claim_evaluation_lease(
            stage="qualification",
            owner=owner,
            current_block=current_block,
            lease_blocks=lease_blocks,
            max_members=max_members,
        )
        if lease is None:
            return None
        return self._active_qualification_recovery(lease)

    def pending_qualification_recovery(self) -> EvaluationRecovery | None:
        rows = tuple(
            self._db.execute(
                "SELECT * FROM evaluation_leases WHERE stage='qualification' "
                "AND state='active'"
            )
        )
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

    def release_pre_resident_recovery(
        self,
        recovery: EvaluationRecovery,
        *,
        current_block: int,
        reason: str,
    ) -> EvaluationLease:
        """Atomically resolve and release only before publication is committed."""

        self._require_evaluation_clock(current_block)
        if (
            type(recovery) is not EvaluationRecovery
            or recovery.resolution is not RecoveryResolution.UNRESOLVED
            or recovery.phase not in {RecoveryPhase.CLAIMED, RecoveryPhase.PREPARED}
            or not isinstance(reason, str)
            or not reason
            or reason.strip() != reason
            or len(reason) > 2_048
            or any(ord(char) < 32 or ord(char) == 127 for char in reason)
            or current_block >= recovery.lease.expires_block
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
                    phase=recovery.phase,
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

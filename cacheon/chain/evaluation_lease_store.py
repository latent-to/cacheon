"""Transactional SQLite authority for asynchronous evaluation leases.

This module is a focused mixin for FinalizedIntakeStore. It owns lease SQL and
CAS behavior while the intake store continues to own the connection, core
reservation lifecycle, finalized cursor, and outer transaction helper.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import MutableSet
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from cacheon.chain.evaluation_leases import (
    EVALUATION_LEASE_EVENTS as _EVALUATION_LEASE_EVENTS,
    EVALUATION_PRIOR_STATUSES as _EVALUATION_PRIOR_STATUSES,
    EVALUATION_STAGES as _EVALUATION_STAGES,
    EvaluationLease,
    EvaluationLeaseError,
    EvaluationLeaseEvent,
    EvaluationLeaseMember,
    decode_members,
    encode_members,
    evaluation_lease_event_id,
    evaluation_lease_id,
    require_evaluation_owner,
)
from cacheon.stack_identity import require_sha256_hex

if TYPE_CHECKING:
    from cacheon.chain.intake import IntakeReservation


_HASH = re.compile(r"[0-9a-f]{64}\Z")


class EvaluationLeaseStoreError(RuntimeError):
    """The additive lease schema or its connection fence cannot be opened."""


def configure_evaluation_lease_connection(
    db: sqlite3.Connection, mutation_authority: MutableSet[str]
) -> None:
    """Install connection-local functions used by the reservation SQL fence."""

    db.create_function(
        "cacheon_evaluation_mutation_authorized",
        1,
        lambda reservation_id: int(reservation_id in mutation_authority),
    )
    db.create_function(
        "cacheon_evaluation_mutation_context_active",
        0,
        lambda: int(bool(mutation_authority)),
    )


def ensure_evaluation_lease_schema(db: sqlite3.Connection) -> None:
    """Create or verify the additive version-1 evaluation lease authority."""

    try:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_leases (
                lease_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL CHECK(generation>0),
                stage TEXT NOT NULL CHECK(stage IN ('screen','qualification')),
                owner TEXT NOT NULL,
                claimed_block INTEGER NOT NULL CHECK(claimed_block>=0),
                initial_expires_block INTEGER NOT NULL CHECK(initial_expires_block>claimed_block),
                expires_block INTEGER NOT NULL CHECK(expires_block>=initial_expires_block),
                state TEXT NOT NULL CHECK(
                    state IN ('active','expired','released','completed')
                ),
                completed_block INTEGER NOT NULL DEFAULT 0 CHECK(completed_block>=0),
                result_digest TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT ''
            ) STRICT;
            CREATE INDEX IF NOT EXISTS evaluation_leases_active_expiry
                ON evaluation_leases(state, expires_block, lease_id);
            CREATE UNIQUE INDEX IF NOT EXISTS evaluation_leases_one_active_qualification
                ON evaluation_leases(stage) WHERE state='active' AND stage='qualification';
            CREATE TABLE IF NOT EXISTS evaluation_lease_members (
                lease_id TEXT NOT NULL REFERENCES evaluation_leases(lease_id),
                position INTEGER NOT NULL CHECK(position>=0),
                reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
                prior_status TEXT NOT NULL CHECK(
                    prior_status IN ('published','reproduction_pending','promoted')
                ),
                active INTEGER NOT NULL CHECK(active IN (0,1)),
                PRIMARY KEY(lease_id, position),
                UNIQUE(lease_id, reservation_id)
            ) STRICT;
            CREATE UNIQUE INDEX IF NOT EXISTS evaluation_lease_members_one_active
                ON evaluation_lease_members(reservation_id) WHERE active=1;
            CREATE INDEX IF NOT EXISTS evaluation_lease_members_reservation
                ON evaluation_lease_members(reservation_id, lease_id);
            CREATE TABLE IF NOT EXISTS evaluation_lease_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                lease_id TEXT NOT NULL REFERENCES evaluation_leases(lease_id),
                generation INTEGER NOT NULL CHECK(generation>0),
                event_index INTEGER NOT NULL CHECK(event_index>=0),
                event_type TEXT NOT NULL CHECK(
                    event_type IN ('claimed','heartbeat','expired','released','completed')
                ),
                stage TEXT NOT NULL CHECK(stage IN ('screen','qualification')),
                owner TEXT NOT NULL,
                members_json TEXT NOT NULL,
                finalized_block INTEGER NOT NULL CHECK(finalized_block>=0),
                expires_block INTEGER NOT NULL CHECK(expires_block>0),
                result_digest TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                UNIQUE(lease_id, event_index)
            ) STRICT;
            CREATE TRIGGER IF NOT EXISTS evaluation_lease_events_reject_update
                BEFORE UPDATE ON evaluation_lease_events
                BEGIN SELECT RAISE(ABORT,'evaluation lease events are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS evaluation_lease_events_reject_delete
                BEFORE DELETE ON evaluation_lease_events
                BEGIN SELECT RAISE(ABORT,'evaluation lease events are immutable'); END;

            -- Recreate the connection-aware fence on every open so an additive
            -- migration cannot retain an older, weaker trigger body.
            DROP TRIGGER IF EXISTS reservations_fence_active_evaluation;
            CREATE TRIGGER reservations_fence_active_evaluation
                BEFORE UPDATE ON reservations
                WHEN (
                    EXISTS (
                        SELECT 1 FROM evaluation_lease_members AS em
                        WHERE em.reservation_id=OLD.reservation_id AND em.active=1
                    ) OR cacheon_evaluation_mutation_context_active()=1
                ) AND cacheon_evaluation_mutation_authorized(OLD.reservation_id)=0
                BEGIN SELECT RAISE(ABORT,'active evaluation lease fences reservation'); END;
            """
        )
    except sqlite3.Error as exc:
        raise EvaluationLeaseStoreError(
            f"evaluation lease schema creation failed: {exc}"
        ) from None

    required = {
        "evaluation_leases": {
            "lease_id", "generation", "stage", "owner", "claimed_block",
            "initial_expires_block", "expires_block", "state", "completed_block",
            "result_digest", "reason",
        },
        "evaluation_lease_members": {
            "lease_id", "position", "reservation_id", "prior_status", "active",
        },
        "evaluation_lease_events": {
            "sequence", "event_id", "lease_id", "generation", "event_index",
            "event_type", "stage", "owner", "members_json", "finalized_block",
            "expires_block", "result_digest", "reason",
        },
    }
    if any(
        not columns.issubset(
            {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
        )
        for table, columns in required.items()
    ):
        raise EvaluationLeaseStoreError("evaluation lease schema is incomplete")
    triggers = {
        row["name"]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    if not {
        "evaluation_lease_events_reject_update",
        "evaluation_lease_events_reject_delete",
        "reservations_fence_active_evaluation",
    }.issubset(triggers):
        raise EvaluationLeaseStoreError("evaluation lease schema triggers are incomplete")

    schema = db.execute(
        "SELECT value FROM metadata WHERE key='evaluation_lease_schema'"
    ).fetchone()
    if schema is None:
        db.execute(
            "INSERT INTO metadata(key,value) VALUES('evaluation_lease_schema','1')"
        )
    elif schema["value"] != "1":
        raise EvaluationLeaseStoreError("evaluation lease schema is unsupported")


def _intake_error(message: str) -> RuntimeError:
    # Imported lazily to keep the mixin reusable without a module cycle while
    # preserving the public IntakeError contract at runtime.
    from cacheon.chain.intake import IntakeError

    return IntakeError(message)


class EvaluationLeaseStoreMixin:
    def _require_evaluation_clock(self, current_block: int) -> None:
        cursor = self._cursor()
        if (
            type(current_block) is not int
            or current_block < 0
            or cursor is None
            or current_block != cursor[0]
        ):
            raise _intake_error(
                "evaluation lease block differs from the durable finalized cursor"
            )

    def _require_evaluation_mutation_authority(self, reservation_id: str) -> None:
        if (
            self._evaluation_mutation_authority
            and reservation_id not in self._evaluation_mutation_authority
        ):
            raise _intake_error(
                "evaluation result context forbids non-member mutation"
            )
        active = self._db.execute(
            "SELECT 1 FROM evaluation_lease_members WHERE reservation_id=? "
            "AND active=1 LIMIT 1",
            (reservation_id,),
        ).fetchone()
        if (
            active is not None
            and reservation_id not in self._evaluation_mutation_authority
        ):
            raise _intake_error("active evaluation lease fences reservation mutation")

    def _evaluation_lease(self, row: sqlite3.Row) -> EvaluationLease:
        try:
            members = tuple(
                EvaluationLeaseMember(member["reservation_id"], member["prior_status"])
                for member in self._db.execute(
                    "SELECT reservation_id,prior_status FROM evaluation_lease_members "
                    "WHERE lease_id=? ORDER BY position",
                    (row["lease_id"],),
                )
            )
            lease = EvaluationLease(
                row["lease_id"],
                row["generation"],
                row["stage"],
                row["owner"],
                members,
                row["claimed_block"],
                row["initial_expires_block"],
                row["expires_block"],
            )
            expected = evaluation_lease_id(
                scope_digest=self.scope.digest,
                generation=lease.generation,
                stage=lease.stage,
                owner=lease.owner,
                members=lease.members,
                claimed_block=lease.claimed_block,
                initial_expires_block=lease.initial_expires_block,
            )
        except EvaluationLeaseError as exc:
            raise _intake_error(f"evaluation lease is corrupt: {exc}") from None
        if lease.lease_id != expected:
            raise _intake_error("evaluation lease identity is corrupt")
        return lease

    def _evaluation_lease_event(self, row: sqlite3.Row) -> EvaluationLeaseEvent:
        try:
            event = EvaluationLeaseEvent(
                row["sequence"],
                row["event_id"],
                row["lease_id"],
                row["generation"],
                row["event_index"],
                row["event_type"],
                row["stage"],
                row["owner"],
                decode_members(row["members_json"]),
                row["finalized_block"],
                row["expires_block"],
                row["result_digest"],
                row["reason"],
            )
            lease_row = self._db.execute(
                "SELECT * FROM evaluation_leases WHERE lease_id=?",
                (event.lease_id,),
            ).fetchone()
            if lease_row is None:
                raise EvaluationLeaseError("event lease is absent")
            retained = self._evaluation_lease(lease_row)
            event_lease = EvaluationLease(
                retained.lease_id,
                event.generation,
                event.stage,
                event.owner,
                event.members,
                retained.claimed_block,
                retained.initial_expires_block,
                event.expires_block,
            )
        except EvaluationLeaseError as exc:
            raise _intake_error(f"evaluation lease event is corrupt: {exc}") from None
        if (
            event_lease.generation != retained.generation
            or event_lease.stage != retained.stage
            or event_lease.owner != retained.owner
            or event_lease.members != retained.members
            or (
                event.event_type in {"claimed", "heartbeat"}
                and (event.reason or event.result_digest)
            )
            or (
                event.event_type == "expired"
                and (
                    event.reason != "evaluation_lease_expired"
                    or event.result_digest
                )
            )
            or (
                event.event_type == "completed"
                and (not event.result_digest or event.reason)
            )
            or (event.event_type == "released" and not event.reason)
            or event.event_id
            != evaluation_lease_event_id(
                event_lease,
                event_index=event.event_index,
                event_type=event.event_type,
                finalized_block=event.finalized_block,
                result_digest=event.result_digest,
                reason=event.reason,
            )
        ):
            raise _intake_error("evaluation lease event identity is corrupt")
        return event

    def _append_evaluation_lease_event(
        self,
        lease: EvaluationLease,
        event_type: str,
        *,
        finalized_block: int,
        result_digest: str = "",
        reason: str = "",
    ) -> EvaluationLeaseEvent:
        if (
            type(lease) is not EvaluationLease
            or event_type not in _EVALUATION_LEASE_EVENTS
            or type(finalized_block) is not int
            or finalized_block < 0
            or (result_digest and _HASH.fullmatch(result_digest) is None)
            or not isinstance(reason, str)
            or len(reason) > 2_048
            or (event_type == "completed" and not result_digest)
            or (event_type not in {"completed", "released"} and bool(result_digest))
            or (event_type == "released" and not reason)
        ):
            raise _intake_error("evaluation lease event input is malformed")
        event_index = self._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_lease_events WHERE lease_id=?",
            (lease.lease_id,),
        ).fetchone()["n"]
        members_json = encode_members(lease.members)
        event_id = evaluation_lease_event_id(
            lease,
            event_index=event_index,
            event_type=event_type,
            finalized_block=finalized_block,
            result_digest=result_digest,
            reason=reason,
        )
        cursor = self._db.execute(
            "INSERT INTO evaluation_lease_events(event_id,lease_id,generation,event_index,"
            "event_type,stage,owner,members_json,finalized_block,expires_block,"
            "result_digest,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                lease.lease_id,
                lease.generation,
                event_index,
                event_type,
                lease.stage,
                lease.owner,
                members_json,
                finalized_block,
                lease.expires_block,
                result_digest,
                reason,
            ),
        )
        retained = self._db.execute(
            "SELECT * FROM evaluation_lease_events WHERE sequence=?",
            (cursor.lastrowid,),
        ).fetchone()
        if retained is None:
            raise _intake_error("evaluation lease event was not retained")
        return self._evaluation_lease_event(retained)

    def _active_evaluation_lease_row(
        self, lease: EvaluationLease
    ) -> sqlite3.Row:
        if type(lease) is not EvaluationLease:
            raise _intake_error("evaluation lease is not exactly typed")
        row = self._db.execute(
            "SELECT * FROM evaluation_leases WHERE lease_id=? AND generation=? "
            "AND stage=? AND owner=? AND claimed_block=? AND initial_expires_block=? "
            "AND expires_block=? "
            "AND state='active'",
            (
                lease.lease_id,
                lease.generation,
                lease.stage,
                lease.owner,
                lease.claimed_block,
                lease.initial_expires_block,
                lease.expires_block,
            ),
        ).fetchone()
        if row is None:
            raise _intake_error("evaluation lease is stale or no longer active")
        # Reconstructing also verifies the durable canonical identity.
        if self._evaluation_lease(row) != lease:
            raise _intake_error("evaluation lease members changed")
        active_members = self._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_lease_members WHERE lease_id=? "
            "AND active=1",
            (lease.lease_id,),
        ).fetchone()["n"]
        if active_members != len(lease.members):
            raise _intake_error("evaluation lease membership is no longer active")
        return row

    def _expire_evaluation_leases(
        self, current_block: int
    ) -> tuple[EvaluationLease, ...]:
        expired: list[EvaluationLease] = []
        rows = tuple(
            self._db.execute(
                "SELECT el.* FROM evaluation_leases AS el JOIN "
                "evaluation_lease_members AS em ON em.lease_id=el.lease_id "
                "AND em.position=0 JOIN reservations AS r USING(reservation_id) "
                "WHERE el.state='active' AND el.expires_block<=? ORDER BY "
                "r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash,"
                "el.lease_id",
                (current_block,),
            )
        )
        for raw in rows:
            lease = self._evaluation_lease(raw)
            if any(
                self.get(member.reservation_id).status != member.prior_status
                for member in lease.members
            ):
                raise _intake_error(
                    "expired evaluation lease no longer has its exact queue state"
                )
            cursor = self._db.execute(
                "UPDATE evaluation_leases SET state='expired',completed_block=?,"
                "reason='evaluation_lease_expired' WHERE lease_id=? AND state='active' "
                "AND expires_block=?",
                (current_block, lease.lease_id, lease.expires_block),
            )
            if cursor.rowcount != 1:
                raise _intake_error("evaluation lease changed while expiring")
            members = self._db.execute(
                "UPDATE evaluation_lease_members SET active=0 WHERE lease_id=? "
                "AND active=1",
                (lease.lease_id,),
            )
            if members.rowcount != len(lease.members):
                raise _intake_error("evaluation lease members changed while expiring")
            self._append_evaluation_lease_event(
                lease,
                "expired",
                finalized_block=current_block,
                reason="evaluation_lease_expired",
            )
            expired.append(lease)
        return tuple(expired)

    def _select_evaluation_rows(
        self, stage: str, bound: int
    ) -> tuple[sqlite3.Row, ...]:
        """Shared, non-mutating ordered selector for preview and atomic claim."""

        if stage == "qualification" and self._db.execute(
            "SELECT 1 FROM evaluation_leases WHERE state='active' "
            "AND stage='qualification' LIMIT 1"
        ).fetchone() is not None:
            # One mutable evaluation-stack authority is shared by every
            # qualification cohort.  Until leases bind an arena/stack
            # generation, concurrent qualification must fail closed globally.
            return ()
        if stage == "screen":
            predicate = "r.status IN ('published','reproduction_pending')"
            priority = "CASE r.status WHEN 'reproduction_pending' THEN 0 ELSE 1 END"
        else:
            predicate = "r.status='promoted'"
            priority = "CASE r.screen_lane WHEN 'reproduction' THEN 0 ELSE 1 END"
        first = self._db.execute(
            "SELECT r.* FROM reservations AS r WHERE "
            f"{predicate} AND NOT EXISTS (SELECT 1 FROM evaluation_lease_members AS em "
            "WHERE em.reservation_id=r.reservation_id AND em.active=1) "
            f"ORDER BY {priority},r.block,r.event_index,r.event_subindex,"
            "r.hotkey,r.content_hash LIMIT 1"
        ).fetchone()
        if first is None:
            return ()
        if stage == "screen" or first["screen_lane"] == "reproduction":
            return (first,)
        if first["retry_group_digest"]:
            selected = tuple(
                self._db.execute(
                    "SELECT r.* FROM reservations AS r WHERE r.status='promoted' "
                    "AND r.retry_group_digest=? AND NOT EXISTS (SELECT 1 FROM "
                    "evaluation_lease_members AS em WHERE "
                    "em.reservation_id=r.reservation_id AND em.active=1) "
                    "ORDER BY r.retry_position",
                    (first["retry_group_digest"],),
                )
            )
            total = self._db.execute(
                "SELECT COUNT(*) AS n FROM reservations WHERE status='promoted' "
                "AND retry_group_digest=?",
                (first["retry_group_digest"],),
            ).fetchone()["n"]
            if len(selected) != total:
                raise _intake_error("qualification retry group is partially leased")
            if len(selected) > bound:
                raise _intake_error("qualification retry group exceeds lease capacity")
            return selected
        return tuple(
            self._db.execute(
                "SELECT r.* FROM reservations AS r WHERE r.status='promoted' "
                "AND r.screen_lane='primary' AND r.retry_group_digest='' "
                "AND NOT EXISTS (SELECT 1 FROM evaluation_lease_members AS em "
                "WHERE em.reservation_id=r.reservation_id AND em.active=1) "
                "ORDER BY r.block,r.event_index,r.event_subindex,r.hotkey,"
                "r.content_hash LIMIT ?",
                (bound,),
            )
        )

    def preview_evaluation_claim(
        self, *, stage: str, max_members: int | None = None
    ) -> tuple[str, ...]:
        """Return the exact ordered member IDs the next claim would bind.

        This is a read-only snapshot.  The later claim repeats the same selector
        under ``BEGIN IMMEDIATE`` and is the authority if the queue changed.
        """

        bound = self.policy.max_cohort if max_members is None else max_members
        if (
            stage not in _EVALUATION_STAGES
            or type(bound) is not int
            or bound <= 0
            or bound > self.policy.max_cohort
            or (stage == "screen" and bound != 1 and max_members is not None)
        ):
            raise _intake_error("evaluation lease preview bounds are malformed")
        if stage == "screen":
            bound = 1
        return tuple(
            row["reservation_id"] for row in self._select_evaluation_rows(stage, bound)
        )

    def claim_evaluation_lease(
        self,
        *,
        stage: str,
        owner: str,
        current_block: int,
        lease_blocks: int = 30,
        max_members: int | None = None,
    ) -> EvaluationLease | None:
        """Atomically claim one oldest eligible screen or qualification cohort.

        Selection is FIFO by finalized ``arrival_key`` within the requested
        stage, except that the existing independent-reproduction contract is
        retained: reproduction work is selected before primary work.  The
        A screen lease is always a singleton.  Qualification preserves the
        existing indivisible retry-group and reproduction semantics; otherwise
        it claims up to ``max_members`` oldest primary rows.  Every member stays
        in its exact prior status and is hidden from legacy queue readers until
        completion, release, or finalized-block expiry.
        """

        try:
            owner = require_evaluation_owner(owner)
        except EvaluationLeaseError as exc:
            raise _intake_error(str(exc)) from None
        bound = self.policy.max_cohort if max_members is None else max_members
        if (
            stage not in _EVALUATION_STAGES
            or type(lease_blocks) is not int
            or lease_blocks <= 0
            or lease_blocks > self.policy.expiry_blocks
            or type(bound) is not int
            or bound <= 0
            or bound > self.policy.max_cohort
            or (stage == "screen" and bound != 1 and max_members is not None)
        ):
            raise _intake_error("evaluation lease claim bounds are malformed")
        if stage == "screen":
            bound = 1
        self._require_evaluation_clock(current_block)
        with self._transaction():
            self._expire_evaluation_leases(current_block)
            self._expire_stale_rows(current_block)
            selected = self._select_evaluation_rows(stage, bound)
            if not selected:
                return None
            reservations = tuple(self._row(row) for row in selected)
            members = tuple(
                EvaluationLeaseMember(row.reservation_id, row.status)
                for row in reservations
            )
            ids = tuple(row.reservation_id for row in reservations)
            marks = ",".join("?" for _ in ids)
            generation = self._db.execute(
                "SELECT COALESCE(MAX(el.generation),0)+1 AS generation FROM "
                "evaluation_leases AS el JOIN evaluation_lease_members AS em "
                f"USING(lease_id) WHERE em.reservation_id IN ({marks})",
                ids,
            ).fetchone()["generation"]
            expires = current_block + lease_blocks
            lease_id = evaluation_lease_id(
                scope_digest=self.scope.digest,
                generation=generation,
                stage=stage,
                owner=owner,
                members=members,
                claimed_block=current_block,
                initial_expires_block=expires,
            )
            self._db.execute(
                "INSERT INTO evaluation_leases(lease_id,generation,stage,owner,"
                "claimed_block,initial_expires_block,expires_block,state) "
                "VALUES(?,?,?,?,?,?,?,'active')",
                (
                    lease_id,
                    generation,
                    stage,
                    owner,
                    current_block,
                    expires,
                    expires,
                ),
            )
            self._db.executemany(
                "INSERT INTO evaluation_lease_members(lease_id,position,"
                "reservation_id,prior_status,active) VALUES(?,?,?,?,1)",
                (
                    (lease_id, position, member.reservation_id, member.prior_status)
                    for position, member in enumerate(members)
                ),
            )
            retained = self._db.execute(
                "SELECT * FROM evaluation_leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            if retained is None:
                raise _intake_error("evaluation lease was not retained")
            lease = self._evaluation_lease(retained)
            self._append_evaluation_lease_event(
                lease, "claimed", finalized_block=current_block
            )
        return lease

    def active_evaluation_leases(self) -> tuple[EvaluationLease, ...]:
        """Return exact active leases in finalized arrival order."""

        return tuple(
            self._evaluation_lease(row)
            for row in self._db.execute(
                "SELECT el.* FROM evaluation_leases AS el JOIN "
                "evaluation_lease_members AS em ON em.lease_id=el.lease_id "
                "AND em.position=0 JOIN reservations AS r USING(reservation_id) "
                "WHERE el.state='active' ORDER BY r.block,r.event_index,"
                "r.event_subindex,r.hotkey,r.content_hash,el.lease_id"
            )
        )

    def evaluation_lease_events(
        self,
        *,
        reservation_id: str | None = None,
        lease_id: str | None = None,
    ) -> tuple[EvaluationLeaseEvent, ...]:
        """Read immutable lease history, optionally for one exact authority."""

        if reservation_id is not None:
            require_sha256_hex(reservation_id, field="evaluation reservation id")
        if lease_id is not None:
            require_sha256_hex(lease_id, field="evaluation lease id")
        clauses: list[str] = []
        parameters: list[str] = []
        if reservation_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM evaluation_lease_members AS em WHERE "
                "em.lease_id=e.lease_id AND em.reservation_id=?)"
            )
            parameters.append(reservation_id)
        if lease_id is not None:
            clauses.append("e.lease_id=?")
            parameters.append(lease_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        events = tuple(
            self._evaluation_lease_event(row)
            for row in self._db.execute(
                "SELECT e.* FROM evaluation_lease_events AS e"
                + where
                + " ORDER BY sequence",
                tuple(parameters),
            )
        )
        by_lease: dict[str, list[EvaluationLeaseEvent]] = {}
        for event in events:
            by_lease.setdefault(event.lease_id, []).append(event)
        terminal = {"expired", "released", "completed"}
        for event_stream in by_lease.values():
            first = event_stream[0]
            lease_row = self._db.execute(
                "SELECT * FROM evaluation_leases WHERE lease_id=?",
                (first.lease_id,),
            ).fetchone()
            if lease_row is None:
                raise _intake_error("evaluation lease event stream lost its lease")
            lease = self._evaluation_lease(lease_row)
            if (
                first.event_index != 0
                or first.event_type != "claimed"
                or first.finalized_block != lease.claimed_block
                or first.expires_block != lease.initial_expires_block
            ):
                raise _intake_error("evaluation lease event stream has no exact claim")
            previous = first
            for index, event in enumerate(event_stream[1:], start=1):
                if (
                    event.event_index != index
                    or previous.event_type in terminal
                    or event.finalized_block < previous.finalized_block
                    or (
                        event.event_type == "heartbeat"
                        and (
                            event.expires_block <= previous.expires_block
                            or event.finalized_block >= previous.expires_block
                        )
                    )
                    or (
                        event.event_type in {"completed", "released"}
                        and event.finalized_block >= event.expires_block
                    )
                    or (
                        event.event_type == "expired"
                        and event.finalized_block < event.expires_block
                    )
                ):
                    raise _intake_error("evaluation lease event stream is inconsistent")
                previous = event
            if any(
                event.event_type in terminal
                for event in event_stream[:-1]
            ):
                raise _intake_error("evaluation lease event stream continued after terminal")
        return events

    def heartbeat_evaluation_lease(
        self,
        lease: EvaluationLease,
        *,
        current_block: int,
        lease_blocks: int = 30,
    ) -> EvaluationLease:
        """CAS-extend an active lease; every older lease object becomes stale."""

        if (
            type(lease) is not EvaluationLease
            or type(lease_blocks) is not int
            or lease_blocks <= 0
            or lease_blocks > self.policy.expiry_blocks
        ):
            raise _intake_error("evaluation lease heartbeat bounds are malformed")
        self._require_evaluation_clock(current_block)
        if self._durably_expire_exact_evaluation_lease_if_due(
            lease, current_block
        ):
            raise _intake_error("evaluation lease expired before heartbeat")
        with self._transaction():
            self._active_evaluation_lease_row(lease)
            expires = current_block + lease_blocks
            if expires <= lease.expires_block:
                raise _intake_error("evaluation heartbeat does not extend the deadline")
            cursor = self._db.execute(
                "UPDATE evaluation_leases SET expires_block=? WHERE lease_id=? "
                "AND state='active' AND expires_block=?",
                (expires, lease.lease_id, lease.expires_block),
            )
            if cursor.rowcount != 1:
                raise _intake_error("evaluation lease changed during heartbeat")
            retained = self._db.execute(
                "SELECT * FROM evaluation_leases WHERE lease_id=?", (lease.lease_id,)
            ).fetchone()
            if retained is None:
                raise _intake_error("evaluation lease disappeared during heartbeat")
            extended = self._evaluation_lease(retained)
            self._append_evaluation_lease_event(
                extended, "heartbeat", finalized_block=current_block
            )
        return extended

    def expire_evaluation_leases(
        self, *, current_block: int
    ) -> tuple[EvaluationLease, ...]:
        """Expire due leases and expose their unchanged queue rows for reclaim."""

        self._require_evaluation_clock(current_block)
        with self._transaction():
            return self._expire_evaluation_leases(current_block)

    def _durably_expire_exact_evaluation_lease_if_due(
        self, lease: EvaluationLease, current_block: int
    ) -> bool:
        """Commit a due exact lease before its caller raises a stale-result error."""

        if current_block < lease.expires_block:
            return False
        with self._transaction():
            self._active_evaluation_lease_row(lease)
            expired = self._expire_evaluation_leases(current_block)
            if lease not in expired:
                raise _intake_error("due evaluation lease was not expired")
        return True

    def release_evaluation_lease(
        self,
        lease: EvaluationLease,
        *,
        current_block: int,
        reason: str,
        result_digest: str = "",
    ) -> EvaluationLease:
        """CAS-release infrastructure work without consuming a candidate attempt."""

        if (
            type(lease) is not EvaluationLease
            or not isinstance(reason, str)
            or not reason
            or reason.strip() != reason
            or len(reason) > 2_048
            or any(ord(char) < 32 or ord(char) == 127 for char in reason)
            or (result_digest and _HASH.fullmatch(result_digest) is None)
        ):
            raise _intake_error("evaluation lease release is malformed")
        self._require_evaluation_clock(current_block)
        if self._durably_expire_exact_evaluation_lease_if_due(
            lease, current_block
        ):
            raise _intake_error("evaluation lease expired before release")
        with self._transaction():
            self._active_evaluation_lease_row(lease)
            if any(
                self.get(member.reservation_id).status != member.prior_status
                for member in lease.members
            ):
                raise _intake_error("evaluation lease no longer has its exact queue state")
            cursor = self._db.execute(
                "UPDATE evaluation_leases SET state='released',completed_block=?,"
                "reason=?,result_digest=? WHERE lease_id=? AND state='active' "
                "AND expires_block=?",
                (
                    current_block,
                    reason,
                    result_digest,
                    lease.lease_id,
                    lease.expires_block,
                ),
            )
            if cursor.rowcount != 1:
                raise _intake_error("evaluation lease changed during release")
            members = self._db.execute(
                "UPDATE evaluation_lease_members SET active=0 WHERE lease_id=? "
                "AND active=1",
                (lease.lease_id,),
            )
            if members.rowcount != len(lease.members):
                raise _intake_error("evaluation lease members changed during release")
            self._append_evaluation_lease_event(
                lease,
                "released",
                finalized_block=current_block,
                reason=reason,
                result_digest=result_digest,
            )
        return lease

    @contextmanager
    def accept_evaluation_result(
        self,
        lease: EvaluationLease,
        *,
        current_block: int,
        result_digest: str,
    ) -> Iterator[tuple[IntakeReservation, ...]]:
        """Atomically apply one retained worker result under an exact live lease.

        The caller performs only the existing durable screen or qualification
        transition or exact qualification batch inside this context.  A crash
        or exception rolls back both
        that transition and lease completion.  Completion requires exactly one
        new stage disposition for every ordered member and no dispositions for
        rows outside the cohort, so acknowledging an empty, partial, or widened
        result is impossible. A stale generation, owner, heartbeat object, or
        deadline is rejected before any reservation mutation.
        """

        if type(lease) is not EvaluationLease:
            raise _intake_error("evaluation result lease is not exactly typed")
        require_sha256_hex(result_digest, field="evaluation result digest")
        self._require_evaluation_clock(current_block)
        if self._durably_expire_exact_evaluation_lease_if_due(
            lease, current_block
        ):
            raise _intake_error("evaluation result arrived after lease expiry")
        with self._transaction():
            self._active_evaluation_lease_row(lease)
            reservations = tuple(
                self.get(member.reservation_id) for member in lease.members
            )
            if any(
                row.status != member.prior_status
                for row, member in zip(reservations, lease.members, strict=True)
            ):
                raise _intake_error("evaluation lease no longer has its exact queue state")
            table = (
                "arena_screen_dispositions"
                if lease.stage == "screen"
                else "qualification_dispositions"
            )
            before_total = self._db.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()["n"]
            before = {
                member.reservation_id: self._db.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE reservation_id=?",
                    (member.reservation_id,),
                ).fetchone()["n"]
                for member in lease.members
            }
            if self._evaluation_mutation_authority:
                raise _intake_error("nested evaluation mutation authority is forbidden")
            authorized = set(lease.reservation_ids)
            self._evaluation_mutation_authority.update(authorized)
            try:
                yield reservations
            finally:
                self._evaluation_mutation_authority.difference_update(authorized)
            after_total = self._db.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()["n"]
            complete = all(
                self._db.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE reservation_id=?",
                    (member.reservation_id,),
                ).fetchone()["n"]
                == before[member.reservation_id] + 1
                and self.get(member.reservation_id).status
                not in {"screening", "qualifying"}
                for member in lease.members
            )
            if not complete or after_total != before_total + len(lease.members):
                raise _intake_error(
                    "evaluation result did not retain the exact cohort dispositions"
                )
            cursor = self._db.execute(
                "UPDATE evaluation_leases SET state='completed',completed_block=?,"
                "result_digest=?,reason='' WHERE lease_id=? AND state='active' "
                "AND expires_block=?",
                (
                    current_block,
                    result_digest,
                    lease.lease_id,
                    lease.expires_block,
                ),
            )
            if cursor.rowcount != 1:
                raise _intake_error("evaluation lease changed during completion")
            members = self._db.execute(
                "UPDATE evaluation_lease_members SET active=0 WHERE lease_id=? "
                "AND active=1",
                (lease.lease_id,),
            )
            if members.rowcount != len(lease.members):
                raise _intake_error("evaluation lease members changed during completion")
            self._append_evaluation_lease_event(
                lease,
                "completed",
                finalized_block=current_block,
                result_digest=result_digest,
            )



__all__ = [
    "EvaluationLeaseStoreError",
    "EvaluationLeaseStoreMixin",
    "configure_evaluation_lease_connection",
    "ensure_evaluation_lease_schema",
]

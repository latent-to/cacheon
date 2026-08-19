"""Connection setup and additive SQLite schema for evaluation leases.

Keeping schema installation separate from lease state transitions makes the
lease store small enough to remain reviewable and gives later additive schemas
one explicit installation boundary.
"""

from __future__ import annotations

import sqlite3
from collections.abc import MutableSet


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


__all__ = [
    "EvaluationLeaseStoreError",
    "configure_evaluation_lease_connection",
    "ensure_evaluation_lease_schema",
]

"""SQLite recovery schema and mutation guards for independent worker requests."""

from __future__ import annotations

import sqlite3
from collections.abc import MutableSet

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
    """Create or verify worker-scoped recovery schema version 3 and its backstops."""

    schema = db.execute(
        "SELECT value FROM metadata WHERE key='evaluation_recovery_schema'"
    ).fetchone()
    if schema is not None and schema["value"] not in {"1", "2", "3"}:
        raise EvaluationRecoveryStoreError("evaluation recovery schema is unsupported")
    try:
        columns = {row["name"] for row in db.execute("PRAGMA table_info(evaluation_recoveries)")}
        if columns and "competition_arena" not in columns:
            db.execute("ALTER TABLE evaluation_recoveries ADD COLUMN competition_arena TEXT NOT NULL DEFAULT ''")
        if schema is not None and schema["value"] in {"1", "2"}:
            db.execute("DROP INDEX IF EXISTS evaluation_recoveries_one_unresolved")
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_recoveries (
                competition_arena TEXT NOT NULL DEFAULT '',
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
                ON evaluation_recoveries(lease_id) WHERE resolution='';
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
    if schema is None:
        db.execute(
            "INSERT INTO metadata(key,value) VALUES('evaluation_recovery_schema','3')"
        )
    elif schema["value"] in {"1", "2"}:
        db.execute("UPDATE metadata SET value='3' WHERE key='evaluation_recovery_schema'")

from __future__ import annotations

import sqlite3

from cacheon.chain.intake import IntakePolicy, IntakeScope
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore


SCOPE = IntakeScope("0x" + "9" * 64, 91)


def _store(tmp_path):
    return RecoverableFinalizedIntakeStore(
        tmp_path / "validator" / "intake.sqlite3",
        IntakePolicy(),
        scope=SCOPE,
    )


def test_recovery_schema_is_additive_and_reopens(tmp_path):
    with _store(tmp_path) as store:
        path = store.path
        assert store._db.execute(
            "SELECT value FROM metadata WHERE key='evaluation_lease_schema'"
        ).fetchone()["value"] == "1"
        assert store._db.execute(
            "SELECT value FROM metadata WHERE key='evaluation_recovery_schema'"
        ).fetchone()["value"] == "1"

    db = sqlite3.connect(path)
    try:
        db.execute("DROP TABLE evaluation_recovery_events")
        db.execute("DROP TABLE evaluation_recoveries")
        db.execute("DELETE FROM metadata WHERE key='evaluation_recovery_schema'")
        db.commit()
    finally:
        db.close()

    with _store(tmp_path) as reopened:
        assert reopened._db.execute(
            "SELECT value FROM metadata WHERE key='evaluation_recovery_schema'"
        ).fetchone()["value"] == "1"
        names = {
            row["name"]
            for row in reopened._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"evaluation_recoveries", "evaluation_recovery_events"} <= names


def test_recovery_schema_rejects_an_unsupported_version(tmp_path):
    with _store(tmp_path) as store:
        path = store.path
    db = sqlite3.connect(path)
    try:
        db.execute(
            "UPDATE metadata SET value='2' WHERE key='evaluation_recovery_schema'"
        )
        db.commit()
    finally:
        db.close()

    try:
        _store(tmp_path)
    except Exception as exc:
        assert "recovery schema is unsupported" in str(exc)
    else:
        raise AssertionError("unsupported recovery schema reopened")

"""Dashboard link from an operator-recorded payment recovery to its live evaluation."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def evaluation_recovery(con: sqlite3.Connection, submission: dict[str, Any]) -> dict[str, Any] | None:
    """Link an operator-recorded payment recovery to its actual live evaluation.

    The original rejection remains historical fact. A corrected submission owns
    its own result, even when the operator accepted it to resolve an earlier fee.
    """
    if (submission.get("reason") or submission.get("invalid_reason")) not in (
        "eval_cost_payment_invalid", "missing_eval_cost_payment",
    ):
        return None
    record = con.execute("SELECT value FROM metadata WHERE key='evaluation_recoveries'").fetchone()
    if not record:
        return None
    target = json.loads(record[0]).get(submission["reservation_id"])
    if not target:
        return None
    row = con.execute(
        "SELECT reservation_id, status, decision, reason FROM reservations "
        "WHERE reservation_id=? AND hotkey=? AND (target_id=? OR ?='')",
        (target, submission["hotkey"], submission["target_id"], submission["target_id"]),
    ).fetchone()
    if not row:
        return None
    recovery = dict(row)
    lease = con.execute(
        "SELECT el.stage FROM evaluation_lease_members m "
        "JOIN evaluation_leases el ON el.lease_id=m.lease_id "
        "WHERE m.reservation_id=? AND el.state='active'",
        (target,),
    ).fetchone()
    recovery["active_stage"] = lease[0] if lease else None
    return recovery


__all__ = ["evaluation_recovery"]

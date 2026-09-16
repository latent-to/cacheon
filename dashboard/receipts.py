"""Dashboard rendering of signed screen receipts and qualification speed reads.

The screen path has recorded per-stage grades and numeric reasons inside the
signed receipt since the 2026-08 hardening, and every graded qualification
leaves a stage-exit artifact in an evidence store. Both already answer "which
check failed" and "what did the lanes measure" — this module only renders
them for the submissions API, so a rejected miner reads the verdict instead
of asking the operator.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def screen_stages(receipt_json: object) -> list[dict[str, Any]] | None:
    """Per-stage rows from one signed screen receipt; ``None`` when unreadable.

    Receipts sealed before the hardening carry no per-stage reasons; their
    rows still render with the stage and grade so the history stays honest.
    """

    if not isinstance(receipt_json, (str, bytes)):
        return None
    try:
        results = json.loads(receipt_json)["results"]
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(results, list):
        return None
    stages: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            return None
        stages.append(
            {
                "stage": row.get("stage"),
                "grade": row.get("grade"),
                "reason": row.get("reason"),
                "elapsed_ms": row.get("elapsed_ms"),
            }
        )
    return stages


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


__all__ = ["evaluation_recovery", "screen_stages"]

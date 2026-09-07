"""Read-only views of the commissioned baseline and completed competitive decisions."""

import sqlite3

from fastapi import HTTPException


def baseline_head(con):
    """Publish the exact reference miners must quote and submit."""

    try:
        row = con.execute("SELECT * FROM finalized_baselines WHERE position=1").fetchone()
    except sqlite3.OperationalError as exc:
        raise HTTPException(503, "finalized baseline is not published") from exc
    if row is None:
        raise HTTPException(503, "finalized baseline is not commissioned")
    return {"baseline_ref": row["stack_digest"], "tree_digest": row["tree_digest"],
            "arena_digest": row["arena_id"], "generation": row["generation"]}


def competition_details(con, reservation_id):
    """Return the stored completion-time classification; never infer stale from today's tip."""

    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='submission_rankings'"
    ).fetchone()
    if not exists:
        return {}
    row = con.execute(
        "SELECT * FROM submission_rankings WHERE reservation_id=?", (reservation_id,)
    ).fetchone()
    if row is None:
        return {}
    return {"stale": bool(row["stale"]), "won": bool(row["won"]),
            "baseline_ref": row["baseline_ref"], "competitor_id": row["competitor_id"],
            "candidate_rate": row["candidate_rate"], "required_ratio": row["required_ratio"],
            "completed_block": row["completed_block"]}

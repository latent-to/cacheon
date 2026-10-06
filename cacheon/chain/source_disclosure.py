"""The retained result clock shared by public source release and baseline admission."""

from __future__ import annotations

import time

DISCLOSURE_DELAY_SECONDS = 8 * 60 * 60


def bundle_visibility(connection, reservation_id, block_time, *, now=None):
    """Return release eligibility from exact result timestamps, never estimates."""
    row = connection.execute(
        "SELECT status FROM reservations WHERE reservation_id=?", (reservation_id,),
    ).fetchone()
    if row is None:
        raise LookupError("reservation not found")
    result_block = 0
    active = connection.execute(
        "SELECT 1 FROM evaluation_leases l JOIN evaluation_lease_members m "
        "ON m.lease_id=l.lease_id WHERE m.reservation_id=? AND l.state='active'",
        (reservation_id,),
    ).fetchone()
    if row[0] in ("qualified", "failed") and not active:
        result_block = connection.execute(
            "SELECT max(block) FROM ("
            "SELECT max(l.completed_block) AS block FROM evaluation_leases l "
            "JOIN evaluation_lease_members m ON m.lease_id=l.lease_id "
            "WHERE m.reservation_id=? AND l.state='completed' "
            "AND l.stage='qualification' UNION ALL "
            "SELECT max(retained_block) FROM settlement_qualifications "
            "WHERE reservation_id=?)", (reservation_id, reservation_id),
        ).fetchone()[0] or 0
    stamp = block_time(result_block) if result_block else {}
    release_at = (stamp["unix"] + DISCLOSURE_DELAY_SECONDS
                  if stamp.get("unix") is not None and stamp.get("estimated") is False
                  else None)
    instant = time.time() if now is None else now
    return {"available": release_at is not None and instant >= release_at,
            "release_at": release_at, "result_block": result_block or None}

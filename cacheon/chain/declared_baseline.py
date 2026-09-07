"""Commissioned HEAD publication and transactional admission of declared baselines."""

from __future__ import annotations

import json

from cacheon.stack_identity import canonical_digest


def create_schema(store) -> None:
    """Extend the intake store without inventing declarations for historical rows."""

    columns = {row[1] for row in store._db.execute("PRAGMA table_info(reservations)")}
    if "baseline_ref" not in columns:
        store._db.execute("ALTER TABLE reservations ADD COLUMN baseline_ref TEXT NOT NULL DEFAULT ''")
    store._db.execute(
        "CREATE TABLE IF NOT EXISTS finalized_baselines ("
        "position INTEGER PRIMARY KEY CHECK(position=1), arena_id TEXT NOT NULL,"
        "generation INTEGER NOT NULL, stack_digest TEXT NOT NULL, tree_digest TEXT NOT NULL,"
        "stack_json TEXT NOT NULL, transition_event_id TEXT NOT NULL)"
    )


def current_baseline(store):
    """Read the commissioned HEAD separately from the settlement stack."""

    row = store._db.execute("SELECT * FROM finalized_baselines WHERE position=1").fetchone()
    return None if row is None else store._evaluation_stack_state_from_row(
        row, context="finalized baseline"
    )


def commission_baseline(store, manifest, tree_digest: str):
    """Publish a commissioned stack without rewriting any queued declaration."""

    from cacheon.chain.intake import EvaluationStackState, IntakeError

    with store._transaction():
        prior = current_baseline(store)
        if prior is not None and prior.manifest.digest == manifest.digest:
            if prior.tree_digest != tree_digest:
                raise IntakeError("commissioned baseline tree changed")
            return prior
        if prior is not None and prior.arena_digest == manifest.arena_digest:
            tip = store.evaluation_stack(manifest.arena_digest)
            if tip.manifest.digest != manifest.digest or tip.tree_digest != tree_digest:
                raise IntakeError("commissioned baseline must advance to the accepted HEAD")
        generation = 0 if prior is None else prior.generation + 1
        event = canonical_digest("cacheon.chain.finalized-baseline.v1", {
            "stack_digest": manifest.digest, "tree_digest": tree_digest,
            "generation": generation,
        })
        state = EvaluationStackState(manifest.arena_digest, generation, manifest, tree_digest, event)
        store._db.execute(
            "INSERT INTO finalized_baselines VALUES(1,?,?,?,?,?,?) "
            "ON CONFLICT(position) DO UPDATE SET arena_id=excluded.arena_id,"
            "generation=excluded.generation,stack_digest=excluded.stack_digest,"
            "tree_digest=excluded.tree_digest,stack_json=excluded.stack_json,"
            "transition_event_id=excluded.transition_event_id",
            (state.arena_digest, generation, manifest.digest, tree_digest,
             json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")), event),
        )
        return state


def admission_reason(store, baseline_ref: str, arrival=None) -> str:
    """Reject missing and stale references before reserving evaluation capacity."""

    if not baseline_ref:
        return "missing_baseline_ref"
    head = current_baseline(store)
    if head is None:
        return "finalized_baseline_unavailable"
    if baseline_ref != head.manifest.digest:
        return "stale_baseline_ref"
    if arrival is not None:
        from cacheon.chain.intake import _ACTIVE
        for row in store.all():
            if row.status not in _ACTIVE or row.arrival.arrival_key <= arrival.arrival_key:
                continue
            segment = store.reservation_baseline_segment(row.reservation_id)
            if segment is not None and segment.generation < head.generation:
                return "baseline_queue_order"
    return ""


def bind_admitted(store, reservation_id: str) -> None:
    """Freeze the admission snapshot inside the transaction that admits the row."""

    from cacheon.chain.baseline_segments import bind_reservation_baseline_segment
    from cacheon.chain.intake import IntakeError

    row = store.get(reservation_id)
    if row.status not in {"reserved", "deferred"}:
        return
    reason = admission_reason(store, row.arrival.baseline_ref)
    if reason:
        raise IntakeError(reason)
    bind_reservation_baseline_segment(
        store, reservation_id, current_baseline(store), reason="miner_declared_baseline"
    )


def expire_deferred_baselines(store) -> None:
    """Recheck capacity-deferred submissions when they can actually enter the queue."""

    for row in tuple(store._db.execute(
        "SELECT reservation_id,baseline_ref FROM reservations WHERE status='deferred'"
    )):
        reason = admission_reason(store, row["baseline_ref"])
        if reason:
            store._db.execute(
                "UPDATE reservations SET status='failed',decision='FAIL',reason=? WHERE reservation_id=?",
                (reason, row["reservation_id"]),
            )

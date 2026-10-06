"""Pin new work to the newest disclosed crown, independently of the loaded engine."""

from __future__ import annotations

from cacheon.chain.intake import EvaluationStackState, FinalizedIntakeStore, IntakeError
from cacheon.chain.source_disclosure import bundle_visibility


def same_baseline(left, right):
    """Compare contribution content across two commissions of the same runtime."""
    return all(getattr(left, key) == getattr(right, key) for key in (
        "runtime_digest", "base_engine_digest", "catalog_digest", "catalog_snapshot", "entries",
    ))


def revealed_baselines(connection, competition_arena, block_time, incumbent=None):
    """Read canonical winning stacks in settlement order with their exact release times."""
    rows = connection.execute(
        "SELECT e.sequence,e.event_id,e.arena_id,e.reservation_id,n.crowned_block,sc.* "
        "FROM settlement_events e JOIN settlement_candidates sc USING(reservation_id) "
        "JOIN settlement_events c ON c.reservation_id=e.reservation_id AND c.event_type='CROWN' "
        "JOIN target_lineage_nodes n ON n.transition_event_id=c.event_id "
        "JOIN reservations r USING(reservation_id) "
        "WHERE e.event_type='STACK_TRANSITION' AND r.competition_arena=? "
        "ORDER BY e.sequence", (competition_arena,),
    ).fetchall()
    history = []
    for row in rows:
        candidate = FinalizedIntakeStore._settlement_candidate(row)
        manifest = candidate.candidate_manifest
        if manifest is None or (incumbent is not None and (
            manifest.runtime_digest != incumbent.runtime_digest
            or manifest.base_engine_digest != incumbent.base_engine_digest
            or manifest.catalog_digest != incumbent.catalog_digest
        )):
            continue
        visibility = bundle_visibility(connection, row["reservation_id"], block_time)
        release = visibility["release_at"]
        crowned = block_time(row["crowned_block"])
        if release is None or crowned.get("estimated") is not False or crowned.get("unix") is None:
            raise IntakeError("winning baseline disclosure timestamp is unavailable")
        release = max(release, crowned["unix"])
        generation = connection.execute(
            "SELECT count(*) FROM settlement_events WHERE arena_id=? "
            "AND event_type='STACK_TRANSITION' AND sequence<=?",
            (row["arena_id"], row["sequence"]),
        ).fetchone()[0]
        history.append((release, row["reservation_id"], EvaluationStackState(
            manifest.arena_digest, generation, manifest,
            candidate.candidate_tree_digest, row["event_id"],
        )))
    return tuple(history)


def latest_revealed(history, timestamp):
    """Select the most advanced disclosed winner, never the largest historical speed ratio."""
    return next((row for row in reversed(history) if row[0] <= timestamp), None)


def admit_revealed_baselines(store, incumbent, tree_digest, block_time, activation_block):
    """Check hash-bound baseline declarations once, using finalized submission time.

    Existing claims and accepted segments survive a later disclosure. The activation
    block preserves pre-rollout work; that work continues on its commissioned baseline.
    """
    from cacheon.chain.baseline_segments import bind_reservation_baseline_segment, commissioned_baseline
    from cacheon.chain.publication import reopen_worker_bundle
    from cacheon.manifest import load_manifest

    with store._transaction():
        rows = store._db.execute(
            "SELECT r.* FROM reservations r LEFT JOIN reservation_baseline_segments b "
            "USING(reservation_id) WHERE r.competition_arena=? AND r.status='published' "
            "AND r.arena_service_digest='' AND r.block>? "
            "AND coalesce(b.binding_reason,'') NOT LIKE 'revealed_baseline%' "
            "AND NOT EXISTS (SELECT 1 FROM evaluation_lease_members m "
            "WHERE m.reservation_id=r.reservation_id AND m.active=1) "
            "AND NOT EXISTS (SELECT 1 FROM settlement_qualifications q "
            "WHERE q.reservation_id=r.reservation_id) "
            "ORDER BY r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash",
            (store._competition_arena, activation_block),
        ).fetchall()
        if not rows:
            return
        history = revealed_baselines(store._db, store._competition_arena, block_time, incumbent)
        for row in rows:
            timestamp = block_time(row["block"])
            if timestamp.get("estimated") is not False or timestamp.get("unix") is None:
                raise IntakeError("submission baseline requires an exact finalized timestamp")
            selected = latest_revealed(history, timestamp["unix"])
            expected = "stock" if selected is None else selected[1]
            publication = reopen_worker_bundle(
                row["publication_root"], row["content_hash"],
                expected_receipt_digest=row["publication_digest"],
            )
            competition = load_manifest(publication.root).competition
            if competition is None or competition.baseline != expected:
                store._expire_before_claim(row["reservation_id"], "baseline_not_latest_revealed")
                continue
            if selected is None:
                # Genesis can only mean stock, never an undisclosed loaded optimization.
                if incumbent.entries:
                    raise IntakeError("stock admission has no commissioned stock baseline")
                state = commissioned_baseline(incumbent, tree_digest)
            else:
                state = selected[2]
            store._db.execute(
                "DELETE FROM reservation_baseline_segments WHERE reservation_id=?",
                (row["reservation_id"],),
            )
            bind_reservation_baseline_segment(
                store, row["reservation_id"], state, reason="revealed_baseline",
            )


def promotion_target(store, incumbent):
    """Return the next accepted segment only after earlier work and leases drain."""
    if store._db.execute(
        "SELECT 1 FROM evaluation_leases WHERE competition_arena=? AND state='active'",
        (store._competition_arena,),
    ).fetchone():
        return None
    head = store._db.execute(
        "SELECT reservation_id,status FROM reservations WHERE competition_arena=? "
        "AND status NOT IN ('qualified','failed','expired') "
        "ORDER BY block,event_index,event_subindex,hotkey,content_hash LIMIT 1",
        (store._competition_arena,),
    ).fetchone()
    if head is None or head["status"] != "published":
        return None
    target = store.reservation_baseline_segment(head["reservation_id"])
    if target is None or same_baseline(target.manifest, incumbent):
        return None
    if store._db.execute(
        "SELECT 1 FROM settlement_candidates sc JOIN reservations r USING(reservation_id) "
        "WHERE r.competition_arena=? AND sc.status IN ('pending','leased') LIMIT 1",
        (store._competition_arena,),
    ).fetchone():
        return None
    return target

"""Baseline-segment authority for the intake queue.

A reservation drains under the evaluation stack its baseline segment names.
This module owns how a segment is bound, backfilled, read as the queue head,
checked as the commission boundary, and rebound when the arena it names has
been retired. It is split out of ``cacheon.chain.intake`` because that file
is at its size ceiling; the store keeps thin delegating methods so every
existing caller is unchanged.

The 2026-09-06 GLM mock mainnet run is the incident behind the rebind rule. A
promoted reservation screened under one deployment was re-queued after the
validator redeployed on new worker bytes. Its segment named the retired arena,
the qualification fence refused to claim across the segment boundary, the
selector could never pick it, and the automatic rotated-cohort re-screen sat
behind that same fence. Left alone the row would have blocked the new arena's
qualification lane for every later submission until it expired. A segment
naming a retired arena is therefore not a commission boundary: the live arena
rebinds it, and the screen-identity rotation rule decides whether its screen
receipt still stands. A same-arena generation boundary (a crown advancing the
durable stack) keeps its halt; those rows drain under the still-resident
commission by design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cacheon.chain.intake import (
    _ACTIVE,
    _EVALUATION_STACK_GENESIS_DOMAIN,
    EvaluationStackState,
    IntakeError,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex

if TYPE_CHECKING:
    from cacheon.chain.intake import FinalizedIntakeStore
    from cacheon.stack_manifest import EvaluationStackManifest

RETIRED_ARENA_REBIND = "retired_arena_rebind"

# Rows that hold no qualification evidence yet. A qualifying row holds a lease
# and a reproduction_pending row holds a PASS half; both keep their segment so
# the boundary stays visible to the operator instead of being rebound.
_REBINDABLE = ("published", "transport_retry", "screening", "promoted")


def bind_reservation_baseline_segment(
    store: "FinalizedIntakeStore",
    reservation_id: str,
    state: EvaluationStackState,
    *,
    reason: str,
) -> None:
    """Bind one reservation, and its promoted retry group, to ``state``.

    A segment that already names ``state``'s arena is kept whatever its
    generation: learning the service after a crown must not rewrite a durable
    baseline. A segment naming another arena is replaced and the new binding
    records the arena it left, because that arena was retired from under the
    row (module docstring).
    """

    if (
        type(state) is not EvaluationStackState
        or not isinstance(reason, str)
        or not reason
        or len(reason) > 128
    ):
        raise IntakeError("reservation baseline binding is malformed")
    reservation = store._db.execute(
        "SELECT retry_group_digest FROM reservations WHERE reservation_id=?",
        (reservation_id,),
    ).fetchone()
    if reservation is None:
        raise IntakeError("reservation baseline binding lost its reservation")
    group = reservation["retry_group_digest"]
    reservation_ids = (reservation_id,)
    if group:
        # The row being bound belongs to its own group even while it is back
        # in the screen queue; the group query alone lists only promoted rows.
        reservation_ids = tuple(dict.fromkeys((
            reservation_id,
            *(
                row["reservation_id"]
                for row in store._db.execute(
                    "SELECT reservation_id FROM reservations "
                    "WHERE status='promoted' AND retry_group_digest=? "
                    "ORDER BY retry_position",
                    (group,),
                )
            ),
        )))
    marks = ",".join("?" for _ in reservation_ids)
    retired = store._db.execute(
        "SELECT arena_id FROM reservation_baseline_segments "
        f"WHERE reservation_id IN ({marks}) AND arena_id<>? "
        "ORDER BY arena_id LIMIT 1",
        (*reservation_ids, state.arena_digest),
    ).fetchone()
    if retired is not None:
        store._db.execute(
            "DELETE FROM reservation_baseline_segments "
            f"WHERE reservation_id IN ({marks}) AND arena_id<>?",
            (*reservation_ids, state.arena_digest),
        )
        reason = f"{reason}:rebound_from_{retired['arena_id'][:16]}"
        if len(reason) > 128:
            raise IntakeError("reservation baseline rebind reason is malformed")
    if group:
        existing = tuple(
            reservation_baseline_segment(store, row_id)
            for row_id in reservation_ids
            if store._db.execute(
                "SELECT 1 FROM reservation_baseline_segments "
                "WHERE reservation_id=?",
                (row_id,),
            ).fetchone()
            is not None
        )
        if existing:
            if any(row != existing[0] for row in existing[1:]):
                raise IntakeError(
                    "qualification retry group spans baseline segments"
                )
            state = existing[0]
    for row_id in reservation_ids:
        store._db.execute(
            "INSERT OR IGNORE INTO reservation_baseline_segments("
            "reservation_id,arena_id,generation,stack_digest,tree_digest,"
            "stack_json,transition_event_id,binding_reason) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                row_id,
                state.arena_digest,
                state.generation,
                state.manifest.digest,
                state.tree_digest,
                store._encoded_stack_manifest(state),
                state.transition_event_id,
                reason,
            ),
        )


def bind_unbound_queue_to_stack(
    store: "FinalizedIntakeStore", state: EvaluationStackState, *, reason: str
) -> None:
    """Bind every unbound active row the stack's arena may answer for."""

    stack_count = store._db.execute(
        "SELECT COUNT(*) AS n FROM evaluation_stacks"
    ).fetchone()["n"]
    active_marks = ",".join("?" for _ in _ACTIVE)
    if stack_count == 1:
        authority = "(r.arena_service_digest=? OR r.arena_service_digest='')"
    else:
        authority = "r.arena_service_digest=?"
    rows = tuple(
        store._db.execute(
            "SELECT r.reservation_id FROM reservations AS r WHERE "
            f"r.status IN ({active_marks}) AND {authority} AND NOT EXISTS ("
            "SELECT 1 FROM reservation_baseline_segments AS b "
            "WHERE b.reservation_id=r.reservation_id)",
            (*_ACTIVE, state.arena_digest),
        )
    )
    for row in rows:
        bind_reservation_baseline_segment(
            store, row["reservation_id"], state, reason=reason
        )


def backfill_reservation_baseline_segments(
    store: "FinalizedIntakeStore",
) -> tuple[str, ...]:
    """Bind pre-upgrade queue rows to the stack active when they arrived.

    Each CROWN snapshots every reservation already present. The first
    target-lineage transition containing a reservation therefore identifies
    the exact incumbent segment that must drain it. Rows arriving after the
    latest transition bind to the current durable stack.
    """

    from cacheon.settlement import SettlementCandidate

    bound: list[str] = []
    active_marks = ",".join("?" for _ in _ACTIVE)
    with store._transaction():
        pending = tuple(
            store._db.execute(
                "SELECT r.reservation_id,r.target_id,r.arena_service_digest "
                "FROM reservations AS r WHERE "
                f"r.status IN ({active_marks}) AND NOT EXISTS (SELECT 1 FROM "
                "reservation_baseline_segments AS b WHERE "
                "b.reservation_id=r.reservation_id) "
                "ORDER BY r.block,r.event_index,r.event_subindex,r.hotkey,"
                "r.content_hash",
                _ACTIVE,
            )
        )
        for reservation in pending:
            transition = store._db.execute(
                "SELECT n.arena_id,se.sequence,se.reservation_id AS winner_id "
                "FROM target_lineage_pretransition_reservations AS p "
                "JOIN target_lineage_nodes AS n "
                "ON n.transition_event_id=p.transition_event_id "
                "JOIN settlement_events AS se "
                "ON se.event_id=n.transition_event_id "
                "WHERE p.reservation_id=? "
                "ORDER BY se.sequence LIMIT 1",
                (reservation["reservation_id"],),
            ).fetchone()
            if transition is None:
                state = store._unambiguous_evaluation_stack(
                    reservation["arena_service_digest"]
                )
                reason = "backfill_current_stack"
            else:
                candidate_row = store._db.execute(
                    "SELECT * FROM settlement_candidates WHERE reservation_id=?",
                    (transition["winner_id"],),
                ).fetchone()
                if candidate_row is None:
                    raise IntakeError(
                        "baseline segment backfill lost its crown candidate"
                    )
                candidate = store._settlement_candidate(candidate_row)
                if (
                    type(candidate) is not SettlementCandidate
                    or candidate.arena_digest != transition["arena_id"]
                ):
                    raise IntakeError(
                        "baseline segment backfill candidate is malformed"
                    )
                previous = store._db.execute(
                    "SELECT event_id FROM settlement_events WHERE arena_id=? "
                    "AND event_type='STACK_TRANSITION' AND sequence<? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (transition["arena_id"], transition["sequence"]),
                ).fetchone()
                generation = store._db.execute(
                    "SELECT COUNT(*) AS n FROM settlement_events "
                    "WHERE arena_id=? AND event_type='STACK_TRANSITION' "
                    "AND sequence<?",
                    (transition["arena_id"], transition["sequence"]),
                ).fetchone()["n"]
                event_id = (
                    previous["event_id"]
                    if previous is not None
                    else canonical_digest(
                        _EVALUATION_STACK_GENESIS_DOMAIN,
                        {
                            "arena_digest": candidate.arena_digest,
                            "stack_digest": candidate.incumbent_manifest.digest,
                            "tree_digest": candidate.incumbent_tree_digest,
                        },
                    )
                )
                state = EvaluationStackState(
                    candidate.arena_digest,
                    generation,
                    candidate.incumbent_manifest,
                    candidate.incumbent_tree_digest,
                    event_id,
                )
                reason = "backfill_pretransition_stack"
            if state is None:
                continue
            bind_reservation_baseline_segment(
                store, reservation["reservation_id"], state, reason=reason
            )
            bound.append(reservation["reservation_id"])
    return tuple(bound)


def reservation_baseline_segment(
    store: "FinalizedIntakeStore", reservation_id: str
) -> EvaluationStackState | None:
    """The stack one reservation must drain under, or None while unbound."""

    require_sha256_hex(reservation_id, field="reservation_id")
    row = store._db.execute(
        "SELECT * FROM reservation_baseline_segments WHERE reservation_id=?",
        (reservation_id,),
    ).fetchone()
    if row is None:
        return None
    return store._evaluation_stack_state_from_row(
        row, context="reservation baseline segment"
    )


def qualification_queue_baseline(
    store: "FinalizedIntakeStore",
) -> EvaluationStackState | None:
    """The segment of the oldest active row, or None while that row is unbound."""

    active_marks = ",".join("?" for _ in _ACTIVE)
    row = store._db.execute(
        "SELECT b.* FROM reservations AS r LEFT JOIN "
        "reservation_baseline_segments AS b USING(reservation_id) "
        f"WHERE r.status IN ({active_marks}) ORDER BY r.block,r.event_index,"
        "r.event_subindex,r.hotkey,r.content_hash LIMIT 1",
        _ACTIVE,
    ).fetchone()
    if row is None or row["arena_id"] is None:
        return None
    return store._evaluation_stack_state_from_row(
        row, context="qualification queue baseline"
    )


def rebind_retired_arena_segments(
    store: "FinalizedIntakeStore", live: EvaluationStackState
) -> tuple[str, ...]:
    """Rebind every evidence-free row whose segment names an arena other than ``live``'s.

    Rebinding happens in one transaction for the whole queue rather than one
    head per pass: the selector never picks a row bound to a retired arena, so
    each such row would otherwise surface as the head in turn.
    """

    marks = ",".join("?" for _ in _REBINDABLE)
    with store._transaction():
        rows = tuple(
            store._db.execute(
                "SELECT b.reservation_id FROM reservation_baseline_segments AS b "
                "JOIN reservations AS r USING(reservation_id) "
                f"WHERE r.status IN ({marks}) AND b.arena_id<>? "
                "ORDER BY r.block,r.event_index,r.event_subindex,r.hotkey,"
                "r.content_hash",
                (*_REBINDABLE, live.arena_digest),
            )
        )
        for row in rows:
            bind_reservation_baseline_segment(
                store, row["reservation_id"], live, reason=RETIRED_ARENA_REBIND
            )
    return tuple(row["reservation_id"] for row in rows)


def commission_boundary(
    store: "FinalizedIntakeStore",
    incumbent: "EvaluationStackManifest",
    *,
    tree_digest: str,
) -> tuple[str, str, str] | None:
    """Install genesis or check the queue head's segment before any claim.

    Returns None when the head drains under ``incumbent``, otherwise the
    commissioned, required, and required-tree digests of the boundary.
    Settlement may advance durable lineage while older reservations remain
    queued; those keep their exact segment and run under the still-resident
    commission, and the boundary halts the evaluator before a lease, request,
    publication, or GPU action. A head bound to a retired arena is rebound to
    the live durable stack first, because no commission can ever drain it.
    """

    try:
        store.evaluation_stack(incumbent.arena_digest)
    except IntakeError as exc:
        if str(exc) != "evaluation stack is not initialized":
            raise
        store.initialize_evaluation_stack(incumbent, tree_digest=tree_digest)
    backfill_reservation_baseline_segments(store)
    required = qualification_queue_baseline(store)
    if required is not None and required.arena_digest != incumbent.arena_digest:
        rebind_retired_arena_segments(
            store, store.evaluation_stack(incumbent.arena_digest)
        )
        required = qualification_queue_baseline(store)
    if required is None or (
        required.manifest.digest == incumbent.digest
        and required.tree_digest == tree_digest
    ):
        return None
    return (incumbent.digest, required.manifest.digest, required.tree_digest)


__all__ = [
    "RETIRED_ARENA_REBIND",
    "backfill_reservation_baseline_segments",
    "bind_reservation_baseline_segment",
    "bind_unbound_queue_to_stack",
    "commission_boundary",
    "qualification_queue_baseline",
    "rebind_retired_arena_segments",
    "reservation_baseline_segment",
]

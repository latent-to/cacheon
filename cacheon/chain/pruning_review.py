"""Admission before a first claim, including the operator's pruning review.

Owner ruling 2026-10-17: routed-expert pruning beyond stock's prune fails, whatever the audit bar
says. Before a bundle's first GPU claim, Claude Opus 5.5 reads its source and decides whether the
served code drops routed experts that stock keeps. The review is an operator process outside this
repository, and its prompt is not published. It reviews published bundles ahead of the queue while
other bundles are evaluated and writes one verdict file per reservation, so the head of the queue
normally finds its verdict waiting. When there is none yet, the claim waits for it.
"""

from __future__ import annotations

import json
from pathlib import Path

REVIEW_SCHEMA = "cacheon-pruning-review-v1"
PRUNING_REASON = "routed_expert_pruning"


class PruningReviewError(RuntimeError):
    """A verdict file exists but does not bind to its reservation."""


def review_verdict(root: Path, reservation) -> str | None:
    """Return ``"pass"`` or ``"fail"`` for this reservation's bundle, or None while it is unreviewed."""

    path = root / f"{reservation.reservation_id}.json"
    try:
        value = json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise PruningReviewError(f"pruning review {path.name} cannot reopen: {exc}") from None
    if (
        type(value) is not dict
        or value.get("schema") != REVIEW_SCHEMA
        or value.get("reservation_id") != reservation.reservation_id
        or value.get("content_hash") != reservation.arrival.content_hash
        or value.get("verdict") not in ("pass", "fail")
    ):
        raise PruningReviewError(f"pruning review {path.name} does not bind to its reservation")
    return value["verdict"]


def admit(store, coordinator, review_root: Path | None) -> bool:
    """Run admission before a first claim; False while the claim must wait for a pruning review.

    The store's admission comes first: an exact copy of a loser inherits its FAIL and closed targets
    are released (duplicate_replay, arena_state). Then every first claim the next lease would bind
    needs its review: a pruner fails here, and a bundle not yet reviewed holds the claim. Only first
    claims are reviewed; a reproduction or a requeued claim entered qualification before.
    """

    store.prepare_qualification_queue(
        service_digest=coordinator.service.identity,
        closed_targets=coordinator.service.manifest.closed_targets,
    )
    if review_root is None:
        return True
    while True:
        preview = store.preview_evaluation_claim(
            stage="qualification",
            max_members=coordinator.qualification_max_members,
            owner=coordinator.owner,
            max_active=coordinator.service.manifest.capacity.max_active_qualifications,
        )
        for reservation_id in preview:
            row = store.get(reservation_id)
            if row.status != "published" or row.arena_service_digest:
                continue
            verdict = review_verdict(review_root, row)
            if verdict is None:
                return False
            if verdict == "fail":
                store.mark_failed(reservation_id, PRUNING_REASON)
                break
        else:
            return True

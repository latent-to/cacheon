"""Qualification request fields shared by the CPU writer and worker readers."""

from __future__ import annotations

from cacheon.chain.remote_qualification_evidence import (
    RemoteEvaluationDispatcherError,
    _SCHEMA_VERSION,
)

LEGACY_QUALIFICATION_FIELDS = frozenset({
    "candidates", "kind", "qualification_policy_digest", "schema_version",
    "screen_lane", "service_digest",
})
INCUMBENT_FIELDS = frozenset({"incumbent_stack_digest", "incumbent_tree_digest"})
QUALIFICATION_FIELDS = LEGACY_QUALIFICATION_FIELDS | INCUMBENT_FIELDS
CHECKPOINT_QUALIFICATION_FIELDS = QUALIFICATION_FIELDS | {"checkpoint_digest"}
BOUND_QUALIFICATION_SHAPES = tuple(
    fields | extra for fields in (QUALIFICATION_FIELDS, CHECKPOINT_QUALIFICATION_FIELDS)
    for extra in (frozenset(), {"payment_bar"})
)


def qualification_request_body(
    coordinator, claim, *, incumbent_stack_digest: str, incumbent_tree_digest: str,
    payment_bar: str = "", retained_body: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write a bound request, or reconstruct an existing request without upgrading it.

    A retained request keeps its original pin and grammar. The caller re-seals
    this reconstruction and compares every byte with the retained authenticated
    request; this function does not authorize a different carrier or execution.
    ``payment_bar`` (``cacheon.chain.evaluation_order.payment_bar``) is written
    only into a new request, so a later crown never changes a retained one.
    """

    body = {
        "candidates": [
            {
                "attempt": candidate.attempt,
                "candidate_digest": candidate.digest,
                "publication": publication.to_dict(),
                "reservation": candidate.reservation.to_dict(),
            }
            for candidate, publication in zip(
                claim.candidates, claim.publications, strict=True
            )
        ],
        "kind": "qualification_work",
        "qualification_policy_digest": coordinator.service.manifest.qualification_policy_digest,
        "schema_version": _SCHEMA_VERSION,
        "screen_lane": claim.screen_lane,
        "service_digest": coordinator.service.identity,
    }
    if retained_body is None:
        body.update(
            incumbent_stack_digest=incumbent_stack_digest,
            incumbent_tree_digest=incumbent_tree_digest,
        )
        if payment_bar:
            body["payment_bar"] = payment_bar
    elif INCUMBENT_FIELDS.issubset(retained_body):
        body.update((field, retained_body[field]) for field in INCUMBENT_FIELDS)
    if retained_body is not None:
        body.update((field, retained_body[field]) for field in ("checkpoint_digest", "payment_bar")
                    if field in retained_body)
    return body


def require_commissioned_incumbent(body, construction) -> None:
    """Reject a missing or different request pin before entering the evaluator."""

    if (
        body.get("incumbent_stack_digest") != construction.incumbent_stack.digest
        or body.get("incumbent_tree_digest") != construction.incumbent_tree_digest
    ):
        raise RemoteEvaluationDispatcherError(
            "qualification request incumbent differs from commissioned stack/tree"
        )

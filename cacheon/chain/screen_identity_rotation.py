"""Recovery for a promoted cohort screened under a retired service identity.

Every worker relaunch mints a fresh worker epoch, and that epoch is part of the
arena service identity that signs a screen receipt.  A reservation promoted by
an earlier epoch therefore carries a receipt the live dispatcher refuses with
``qualification request provenance differs``.

Nothing about such a reservation is wrong; only the identity that screened it
is gone.  The qualification selector is deterministic, so a cohort left in
``promoted`` is re-picked on every pass and every submission behind it stalls
until it expires unevaluated.  Mainnet lost a full queue this way: one row
screened by a retired epoch, and 201 reservations behind it aged out as
NO_DECISION on the validator's own SLA.

The recovery is to release the claim and send the cohort back to the screen
queue, where the live identity issues a fresh receipt.  The release tolerates
an expired lease on purpose: the stall outlives the lease window by
construction, so refusing to clean up after expiry would make the wedge
permanent rather than safe.

This lives outside ``evaluation_recovery_store`` because that module is at its
size limit, and outside the two dispatch paths because both of them need the
identical rule and must not drift apart.
"""

from __future__ import annotations

from cacheon.chain.evaluation_recovery import EvaluationRecovery, RecoveryPhase

ROTATED_SCREEN_RELEASE = "screen_receipt_service_rotated"

# An unbuilt claim can be released directly. A REQUEST_READY claim must first
# retire its authority-bound request through the existing bounded systemic
# transition; later phases may contain execution evidence and remain forbidden.
_RELEASABLE_PHASES = frozenset({RecoveryPhase.CLAIMED, RecoveryPhase.PREPARED})


class ScreenIdentityRotationError(RuntimeError):
    """A rotated screen cohort could not be returned to the screen queue."""


def rotated_reservation_ids(
    reservations: tuple[object, ...],
    receipts: tuple[object, ...],
    service_identity: str,
) -> tuple[str, ...]:
    """Reservations whose standing screen receipt is no longer authoritative."""

    return tuple(
        row.reservation_id  # type: ignore[attr-defined]
        for row, receipt in zip(reservations, receipts, strict=True)
        if receipt is None
        or receipt.service_digest != service_identity  # type: ignore[attr-defined]
    )


def release_rotated_cohort(
    store: object,
    recovery: EvaluationRecovery,
    *,
    current_block: int,
    reservation_ids: tuple[str, ...],
) -> None:
    """Release one rotated claim and requeue its exact cohort for re-screening.

    Ordering is load-bearing: the release verifies that every member still holds
    the status it was leased with, so the cohort may only be demoted afterwards.
    """

    if type(recovery) is not EvaluationRecovery or recovery.phase not in {
        *_RELEASABLE_PHASES,
        RecoveryPhase.REQUEST_READY,
    }:
        raise ScreenIdentityRotationError(
            "rotated screen recovery release requires an unexecuted claim"
        )
    if not reservation_ids:
        raise ScreenIdentityRotationError("rotated screen cohort is empty")
    if recovery.phase is RecoveryPhase.REQUEST_READY:
        if current_block >= recovery.lease.expires_block:
            from cacheon.chain.execution_disposition import (
                AUTHORITY_CHANGED_HOLD_REASON,
            )

            recovery = store.hold_recovery(  # type: ignore[attr-defined]
                recovery,
                current_block=current_block,
                reason=AUTHORITY_CHANGED_HOLD_REASON,
            )
        store.release_worker_infrastructure_recovery(  # type: ignore[attr-defined]
            recovery,
            current_block=current_block,
            failure_code=ROTATED_SCREEN_RELEASE,
        )
    else:
        store._release_recovery(  # type: ignore[attr-defined]
            recovery,
            current_block=current_block,
            reason=ROTATED_SCREEN_RELEASE,
            allow_expired=True,
        )
    for reservation_id in reservation_ids:
        if store.get(reservation_id).status == "held":  # type: ignore[attr-defined]
            continue
        store.demote_promoted_for_rescreen(  # type: ignore[attr-defined]
            reservation_id, reason=ROTATED_SCREEN_RELEASE
        )


__all__ = [
    "ROTATED_SCREEN_RELEASE",
    "ScreenIdentityRotationError",
    "release_rotated_cohort",
    "rotated_reservation_ids",
]

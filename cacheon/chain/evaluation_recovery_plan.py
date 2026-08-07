"""Canonical SQLite codec for one exact remote qualification request plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.stack_identity import require_sha256_hex

if TYPE_CHECKING:
    from cacheon.chain.remote_worker_request_plan import QualificationRequestPlan


MAX_RECOVERY_PLAN_BYTES = 4 * 1024 * 1024


class EvaluationRecoveryPlanError(ValueError):
    """A retained plan is malformed, changed, or bound to another lease."""


def _plan_lease_matches_current(
    planned: EvaluationLease, current: EvaluationLease
) -> bool:
    return (
        planned.lease_id == current.lease_id
        and planned.generation == current.generation
        and planned.stage == current.stage
        and planned.owner == current.owner
        and planned.members == current.members
        and planned.claimed_block == current.claimed_block
        and planned.initial_expires_block == current.initial_expires_block
        and planned.expires_block <= current.expires_block
    )


def encode_recovery_request_plan(
    plan: "QualificationRequestPlan", *, expected_lease: EvaluationLease
) -> tuple[bytes, str, str]:
    """Validate and canonically encode the exact plan selected for a lease."""

    from cacheon.chain.remote_worker_request_plan import QualificationRequestPlan
    from cacheon.chain.remote_worker_spool import RemoteWorkerError, spool_canonical_json

    if type(expected_lease) is not EvaluationLease:
        raise EvaluationRecoveryPlanError("expected recovery lease is not exactly typed")
    if type(plan) is not QualificationRequestPlan or plan.lease != expected_lease:
        raise EvaluationRecoveryPlanError("request plan differs from its recovery lease")
    try:
        payload = spool_canonical_json(plan.to_dict())
    except RemoteWorkerError as exc:
        raise EvaluationRecoveryPlanError(f"request plan cannot encode: {exc}") from None
    if not payload or len(payload) > MAX_RECOVERY_PLAN_BYTES:
        raise EvaluationRecoveryPlanError("request plan exceeds its durable size bound")
    return payload, plan.plan_digest, plan.request_id


def decode_recovery_request_plan(
    payload: bytes,
    *,
    expected_lease: EvaluationLease,
    expected_plan_digest: str,
    expected_request_id: str,
) -> "QualificationRequestPlan":
    """Reopen canonical bytes and reject every identity or authority change."""

    from cacheon.chain.remote_evaluation_dispatcher import (
        RemoteEvaluationDispatcherError,
    )
    from cacheon.chain.remote_worker_request_plan import QualificationRequestPlan
    from cacheon.chain.remote_worker_spool import (
        RemoteWorkerError,
        spool_canonical_json,
        strict_json_object,
    )

    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > MAX_RECOVERY_PLAN_BYTES
        or type(expected_lease) is not EvaluationLease
    ):
        raise EvaluationRecoveryPlanError("retained request plan is malformed")
    try:
        require_sha256_hex(expected_plan_digest, field="retained plan digest")
        require_sha256_hex(expected_request_id, field="retained request id")
        value = strict_json_object(payload.decode("utf-8"))
        plan = QualificationRequestPlan.from_dict(value)
        canonical = spool_canonical_json(plan.to_dict())
    except (
        RemoteEvaluationDispatcherError,
        RemoteWorkerError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise EvaluationRecoveryPlanError(f"retained request plan cannot reopen: {exc}") from None
    if (
        canonical != payload
        or not _plan_lease_matches_current(plan.lease, expected_lease)
        or plan.plan_digest != expected_plan_digest
        or plan.request_id != expected_request_id
    ):
        raise EvaluationRecoveryPlanError("retained request plan identity changed")
    return plan


__all__ = [
    "EvaluationRecoveryPlanError",
    "MAX_RECOVERY_PLAN_BYTES",
    "decode_recovery_request_plan",
    "encode_recovery_request_plan",
]

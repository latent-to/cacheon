"""Pure identities and exact types for durable asynchronous evaluation leases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from cacheon.stack_identity import canonical_digest


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LEASE_DOMAIN = "cacheon.chain.evaluation-lease.v1"
_EVENT_DOMAIN = "cacheon.chain.evaluation-lease-event.v1"

EVALUATION_STAGES = frozenset({"screen", "qualification"})
EVALUATION_PRIOR_STATUSES = frozenset(
    {"published", "reproduction_pending", "promoted"}
)
EVALUATION_LEASE_EVENTS = frozenset(
    {"claimed", "heartbeat", "expired", "released", "completed"}
)


class EvaluationLeaseError(RuntimeError):
    """Evaluation lease identity or bounds are malformed."""


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise EvaluationLeaseError(f"{field} is malformed")
    return value


def require_evaluation_owner(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > 256
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise EvaluationLeaseError("evaluation lease owner is malformed")
    return value


@dataclass(frozen=True)
class EvaluationLeaseMember:
    """One exact finalized queue row bound into a cohort lease."""

    reservation_id: str
    prior_status: str

    def __post_init__(self) -> None:
        _digest(self.reservation_id, "evaluation reservation id")
        if self.prior_status not in EVALUATION_PRIOR_STATUSES:
            raise EvaluationLeaseError("evaluation lease member status is malformed")

    def to_dict(self) -> dict[str, str]:
        return {
            "prior_status": self.prior_status,
            "reservation_id": self.reservation_id,
        }


@dataclass(frozen=True)
class EvaluationLease:
    """One durable finalized-block-bounded screen or qualification cohort."""

    lease_id: str
    generation: int
    stage: str
    owner: str
    members: tuple[EvaluationLeaseMember, ...]
    claimed_block: int
    initial_expires_block: int
    expires_block: int

    def __post_init__(self) -> None:
        _digest(self.lease_id, "evaluation lease id")
        require_evaluation_owner(self.owner)
        members = tuple(self.members)
        if (
            type(self.generation) is not int
            or self.generation <= 0
            or self.stage not in EVALUATION_STAGES
            or not members
            or any(type(row) is not EvaluationLeaseMember for row in members)
            or len({row.reservation_id for row in members}) != len(members)
            or type(self.claimed_block) is not int
            or self.claimed_block < 0
            or type(self.initial_expires_block) is not int
            or self.initial_expires_block <= self.claimed_block
            or type(self.expires_block) is not int
            or self.expires_block < self.initial_expires_block
            or (
                self.stage == "screen"
                and (
                    len(members) != 1
                    or members[0].prior_status
                    not in {"published", "reproduction_pending"}
                )
            )
            or (
                self.stage == "qualification"
                and any(row.prior_status != "promoted" for row in members)
            )
        ):
            raise EvaluationLeaseError("evaluation lease bounds are malformed")
        object.__setattr__(self, "members", members)

    @property
    def reservation_ids(self) -> tuple[str, ...]:
        return tuple(row.reservation_id for row in self.members)


@dataclass(frozen=True)
class EvaluationLeaseEvent:
    """One immutable transition in an evaluation lease's audit history."""

    sequence: int
    event_id: str
    lease_id: str
    generation: int
    event_index: int
    event_type: str
    stage: str
    owner: str
    members: tuple[EvaluationLeaseMember, ...]
    finalized_block: int
    expires_block: int
    result_digest: str
    reason: str

    def __post_init__(self) -> None:
        _digest(self.event_id, "event_id")
        _digest(self.lease_id, "lease_id")
        require_evaluation_owner(self.owner)
        members = tuple(self.members)
        if (
            type(self.sequence) is not int
            or self.sequence <= 0
            or type(self.generation) is not int
            or self.generation <= 0
            or type(self.event_index) is not int
            or self.event_index < 0
            or self.event_type not in EVALUATION_LEASE_EVENTS
            or self.stage not in EVALUATION_STAGES
            or not members
            or any(type(row) is not EvaluationLeaseMember for row in members)
            or len({row.reservation_id for row in members}) != len(members)
            or type(self.finalized_block) is not int
            or self.finalized_block < 0
            or type(self.expires_block) is not int
            or self.expires_block <= 0
            or (self.result_digest and _HASH.fullmatch(self.result_digest) is None)
            or not isinstance(self.reason, str)
            or len(self.reason) > 2_048
        ):
            raise EvaluationLeaseError("evaluation lease event is malformed")
        object.__setattr__(self, "members", members)


def encode_members(members: tuple[EvaluationLeaseMember, ...]) -> str:
    return json.dumps(
        [member.to_dict() for member in members],
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_members(encoded: str) -> tuple[EvaluationLeaseMember, ...]:
    try:
        raw = json.loads(encoded)
        if type(raw) is not list:
            raise TypeError("members are not an array")
        return tuple(
            EvaluationLeaseMember(item["reservation_id"], item["prior_status"])
            for item in raw
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationLeaseError(
            f"evaluation lease members are corrupt: {exc}"
        ) from None


def evaluation_lease_id(
    *,
    scope_digest: str,
    generation: int,
    stage: str,
    owner: str,
    members: tuple[EvaluationLeaseMember, ...],
    claimed_block: int,
    initial_expires_block: int,
) -> str:
    return canonical_digest(
        _LEASE_DOMAIN,
        {
            "claimed_block": claimed_block,
            "generation": generation,
            "initial_expires_block": initial_expires_block,
            "members": [member.to_dict() for member in members],
            "owner": owner,
            "scope_digest": scope_digest,
            "stage": stage,
        },
    )


def evaluation_lease_event_id(
    lease: EvaluationLease,
    *,
    event_index: int,
    event_type: str,
    finalized_block: int,
    result_digest: str,
    reason: str,
) -> str:
    return canonical_digest(
        _EVENT_DOMAIN,
        {
            "event_index": event_index,
            "event_type": event_type,
            "expires_block": lease.expires_block,
            "finalized_block": finalized_block,
            "generation": lease.generation,
            "lease_id": lease.lease_id,
            "members": [member.to_dict() for member in lease.members],
            "owner": lease.owner,
            "reason": reason,
            "result_digest": result_digest,
            "stage": lease.stage,
        },
    )


__all__ = [
    "EVALUATION_LEASE_EVENTS",
    "EVALUATION_PRIOR_STATUSES",
    "EVALUATION_STAGES",
    "EvaluationLease",
    "EvaluationLeaseError",
    "EvaluationLeaseEvent",
    "EvaluationLeaseMember",
    "decode_members",
    "encode_members",
    "evaluation_lease_event_id",
    "evaluation_lease_id",
    "require_evaluation_owner",
]

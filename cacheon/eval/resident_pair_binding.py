"""Path-free authority for one commissioned resident evaluation pair.

The prepared crossover arms and the processes that host them are different
authorities.  These immutable values retain the actual stock-launched session,
allocation, and executor identities without carrying validator paths or live
handles into durable product evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cacheon.eval.resident_evaluation_pair import ResidentLaneIdentity
from cacheon.stack_identity import canonical_digest, require_sha256_hex


class ResidentPairBindingError(RuntimeError):
    """A resident pair runtime binding is malformed or ambiguous."""


_SESSION_RE = re.compile(r"[0-9a-f]{32}")


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (TypeError, ValueError) as exc:
        raise ResidentPairBindingError(str(exc)) from None


@dataclass(frozen=True)
class ResidentPairLaneBinding:
    """One physical pair lane and its actual stock runtime authorities."""

    lane_id: str
    session_id: str
    stock_launch_digest: str
    lane_digest: str
    allocation_digest: str
    executor_namespace_digest: str

    def __post_init__(self) -> None:
        if self.lane_id not in ("A", "B"):
            raise ResidentPairBindingError("resident binding lane must be A or B")
        if (
            type(self.session_id) is not str
            or _SESSION_RE.fullmatch(self.session_id) is None
            or self.session_id == "0" * 32
        ):
            raise ResidentPairBindingError(
                "resident binding session must be nonzero lowercase 32-hex"
            )
        for field in (
            "stock_launch_digest",
            "lane_digest",
            "allocation_digest",
            "executor_namespace_digest",
        ):
            object.__setattr__(
                self, field, _digest(getattr(self, field), field.replace("_", " "))
            )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.resident-pair-lane-binding.v1",
            {
                "allocation": self.allocation_digest,
                "executor_namespace": self.executor_namespace_digest,
                "lane": self.lane_id,
                "lane_authority": self.lane_digest,
                "session": self.session_id,
                "stock_launch": self.stock_launch_digest,
            },
        )


@dataclass(frozen=True)
class ResidentPairRuntimeBinding:
    """Canonical A/B runtime authority for one long-lived service epoch."""

    service_epoch_digest: str
    lanes: tuple[ResidentPairLaneBinding, ResidentPairLaneBinding]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "service_epoch_digest",
            _digest(self.service_epoch_digest, "service epoch digest"),
        )
        if (
            type(self.lanes) is not tuple
            or len(self.lanes) != 2
            or any(type(row) is not ResidentPairLaneBinding for row in self.lanes)
            or tuple(row.lane_id for row in self.lanes) != ("A", "B")
        ):
            raise ResidentPairBindingError(
                "resident runtime binding requires exact canonical A/B lanes"
            )
        for field in (
            "session_id",
            "stock_launch_digest",
            "allocation_digest",
            "executor_namespace_digest",
        ):
            if len({getattr(row, field) for row in self.lanes}) != 2:
                raise ResidentPairBindingError(
                    f"resident A/B lanes share {field.replace('_', ' ')}"
                )

    @property
    def identities(self) -> tuple[ResidentLaneIdentity, ResidentLaneIdentity]:
        lane_a, lane_b = self.lanes
        return (
            ResidentLaneIdentity(lane_a.lane_id, lane_a.session_id),
            ResidentLaneIdentity(lane_b.lane_id, lane_b.session_id),
        )

    def lookup(self, lane_id: str) -> ResidentPairLaneBinding:
        if lane_id not in ("A", "B"):
            raise ResidentPairBindingError(
                "resident binding lookup requires lane A or B"
            )
        return self.lanes[0 if lane_id == "A" else 1]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.resident-pair-runtime-binding.v1",
            {
                "lanes": [row.digest for row in self.lanes],
                "service_epoch": self.service_epoch_digest,
            },
        )


__all__ = [
    "ResidentPairBindingError",
    "ResidentPairLaneBinding",
    "ResidentPairRuntimeBinding",
]

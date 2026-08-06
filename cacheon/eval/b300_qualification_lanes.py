"""Physical two-lane qualification authority for the B300/TP4 deployment.

One sealed lane pair carves the commissioned eight-B300 pod into two disjoint
physical TP4 lanes.  Lane A always carries the primary candidate and lane B the
primary resident baseline; reproduction exchanges exactly those physical roles.
``b300_arena_provider`` composes and re-exports these names, so import paths
are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from cacheon._strict import require_digest
from cacheon.eval.device_state import DeviceStatePolicy
from cacheon.stack_identity import canonical_digest


QUALIFICATION_LANE_SCHEMA = "cacheon.eval.b300-qualification-lane.v1"
QUALIFICATION_LANE_PAIR_SCHEMA = "cacheon.eval.b300-qualification-lane-pair.v1"
QUALIFICATION_ROLE_SWAP_SCHEMA = "cacheon.eval.b300-qualification-role-swap.v1"
B300_GPU_COUNT = 4


class B300ArenaProviderError(RuntimeError):
    """A deployment authority or provider lifecycle is inconsistent."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=B300ArenaProviderError)


@dataclass(frozen=True)
class B300QualificationLanePolicy:
    """One exact physical TP4 lane, independent of its execution role."""

    lane_id: str
    physical_gpu_ids: tuple[int, ...]
    gpu_uuids: tuple[str, ...]
    device_configuration_digest: str
    device_policy_digest: str

    def __post_init__(self) -> None:
        if self.lane_id not in {"A", "B"}:
            raise B300ArenaProviderError("qualification lane id must be A or B")
        physical_ids = self.physical_gpu_ids
        uuids = self.gpu_uuids
        if (
            type(physical_ids) is not tuple
            or len(physical_ids) != B300_GPU_COUNT
            or any(type(row) is not int or row < 0 for row in physical_ids)
            or physical_ids != tuple(sorted(set(physical_ids)))
            or type(uuids) is not tuple
            or len(uuids) != B300_GPU_COUNT
            or len(set(uuids)) != B300_GPU_COUNT
            or any(
                not isinstance(row, str)
                or not row
                or row.strip() != row
                or len(row) > 128
                or any(character in row for character in "\x00\r\n")
                for row in uuids
            )
        ):
            raise B300ArenaProviderError(
                "qualification lane must bind one canonical physical TP4"
            )
        for field in ("device_configuration_digest", "device_policy_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))

    @classmethod
    def from_device_policy(
        cls,
        lane_id: str,
        policy: DeviceStatePolicy,
    ) -> "B300QualificationLanePolicy":
        if type(policy) is not DeviceStatePolicy:
            raise B300ArenaProviderError("qualification lane policy is not exact")
        gpus = policy.expected_gpus
        if len(gpus) != B300_GPU_COUNT or any(
            "B300" not in gpu.name.upper() for gpu in gpus
        ):
            raise B300ArenaProviderError(
                "qualification lane does not bind exactly four B300 devices"
            )
        return cls(
            lane_id,
            tuple(gpu.physical_id for gpu in gpus),
            tuple(gpu.uuid for gpu in gpus),
            policy.configuration_sha256,
            policy.policy_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "device_configuration_digest": self.device_configuration_digest,
            "device_policy_digest": self.device_policy_digest,
            "gpu_uuids": list(self.gpu_uuids),
            "lane_id": self.lane_id,
            "physical_gpu_ids": list(self.physical_gpu_ids),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(QUALIFICATION_LANE_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class B300QualificationLaneOrientation:
    """The exact physical roles selected for one qualification stage."""

    stage: str
    candidate: B300QualificationLanePolicy
    resident_baseline: B300QualificationLanePolicy

    def __post_init__(self) -> None:
        if (
            self.stage not in {"primary", "reproduction"}
            or type(self.candidate) is not B300QualificationLanePolicy
            or type(self.resident_baseline) is not B300QualificationLanePolicy
            or self.candidate.lane_id == self.resident_baseline.lane_id
        ):
            raise B300ArenaProviderError(
                "qualification lane orientation is not an exact stage mapping"
            )


@dataclass(frozen=True)
class B300QualificationLanePair:
    """Stable two-lane authority with one permitted primary/reproduction swap.

    Lane A always carries the primary candidate and lane B the primary resident
    baseline.  Reproduction must exchange those exact physical roles.  The pair
    and its digest do not depend on which of those two stages is executing.
    """

    lane_a: B300QualificationLanePolicy
    lane_b: B300QualificationLanePolicy

    def __post_init__(self) -> None:
        if (
            type(self.lane_a) is not B300QualificationLanePolicy
            or type(self.lane_b) is not B300QualificationLanePolicy
            or self.lane_a.lane_id != "A"
            or self.lane_b.lane_id != "B"
        ):
            raise B300ArenaProviderError(
                "qualification lane pair must contain canonical lanes A and B"
            )
        if (
            set(self.lane_a.physical_gpu_ids).intersection(
                self.lane_b.physical_gpu_ids
            )
            or set(self.lane_a.gpu_uuids).intersection(self.lane_b.gpu_uuids)
            or self.lane_a.device_configuration_digest
            == self.lane_b.device_configuration_digest
            or self.lane_a.device_policy_digest == self.lane_b.device_policy_digest
        ):
            raise B300ArenaProviderError(
                "qualification lane pair is overlapping or not physically distinct"
            )

    def orientation(self, stage: str) -> B300QualificationLaneOrientation:
        if stage == "primary":
            return B300QualificationLaneOrientation(
                stage, self.lane_a, self.lane_b
            )
        if stage == "reproduction":
            return B300QualificationLaneOrientation(
                stage, self.lane_b, self.lane_a
            )
        raise B300ArenaProviderError("qualification stage must be primary or reproduction")

    @property
    def role_swap_digest(self) -> str:
        return canonical_digest(
            QUALIFICATION_ROLE_SWAP_SCHEMA,
            {
                "primary": {
                    "candidate": "A",
                    "resident_baseline": "B",
                },
                "reproduction": {
                    "candidate": "B",
                    "resident_baseline": "A",
                },
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lane_a": self.lane_a.to_dict(),
            "lane_b": self.lane_b.to_dict(),
            "role_swap_digest": self.role_swap_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(QUALIFICATION_LANE_PAIR_SCHEMA, self.to_dict())

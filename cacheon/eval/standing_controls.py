"""Closed controls contract for standing two-lane AR evaluation configs.

A sealed standing-evaluation config carries one lane-independent controls
identity: three SHA-256 identity digests plus the integer bookend-drift bound.
The bound is not free policy — it is derived exactly from the version-3
resident speed policy's ``max_noise`` expressed in parts per million — and the
primary and reproduction stages must bind byte-identical controls. A config
missing a key, carrying an extra key, or holding a bound that disagrees with
the sealed speed policy is rejected here, before any GPU work is scheduled.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from cacheon._strict import require_digest
from cacheon.eval.crossover_runtime import ResidentSpeedPolicy


class StandingControlsError(RuntimeError):
    """A standing-evaluation controls block is incomplete or inconsistent."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=StandingControlsError)


_CONTROL_FIELDS = frozenset(
    {
        "baseline_health_policy_digest",
        "calibration_context_digest",
        "maximum_bookend_drift_ppm",
        "reference_manifest_digest",
    }
)
_MAX_DRIFT_PPM = 1_000_000


def bookend_drift_ppm_from_max_noise(max_noise: float) -> int:
    """Express a v3 speed policy's ``max_noise`` exactly in parts per million."""

    canonical = format(float(max_noise), ".17g")
    scaled = Decimal(canonical) * 1_000_000
    integral = int(scaled)
    if scaled != integral or not 0 < integral <= _MAX_DRIFT_PPM:
        raise StandingControlsError(
            "speed-policy max_noise has no exact parts-per-million expression"
        )
    return integral


@dataclass(frozen=True)
class StandingEvaluationControls:
    """The lane-independent controls identity of one sealed standing config."""

    baseline_health_policy_digest: str
    calibration_context_digest: str
    reference_manifest_digest: str
    maximum_bookend_drift_ppm: int

    def __post_init__(self) -> None:
        for field in (
            "baseline_health_policy_digest",
            "calibration_context_digest",
            "reference_manifest_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        ppm = self.maximum_bookend_drift_ppm
        if type(ppm) is not int or not 0 < ppm <= _MAX_DRIFT_PPM:
            raise StandingControlsError(
                "maximum_bookend_drift_ppm is outside its integer bounds"
            )

    @classmethod
    def from_dict(cls, value: object) -> "StandingEvaluationControls":
        if type(value) is not dict or set(value) != _CONTROL_FIELDS:
            raise StandingControlsError(
                "standing controls fields are not the closed four-key schema"
            )
        return cls(
            baseline_health_policy_digest=value["baseline_health_policy_digest"],
            calibration_context_digest=value["calibration_context_digest"],
            reference_manifest_digest=value["reference_manifest_digest"],
            maximum_bookend_drift_ppm=value["maximum_bookend_drift_ppm"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_health_policy_digest": self.baseline_health_policy_digest,
            "calibration_context_digest": self.calibration_context_digest,
            "maximum_bookend_drift_ppm": self.maximum_bookend_drift_ppm,
            "reference_manifest_digest": self.reference_manifest_digest,
        }

    def require_derived_from(self, policy: ResidentSpeedPolicy) -> None:
        if type(policy) is not ResidentSpeedPolicy or policy.version not in (3, 4):
            raise StandingControlsError(
                "standing controls require the version-3 resident speed policy"
            )
        expected = bookend_drift_ppm_from_max_noise(policy.max_noise)
        if self.maximum_bookend_drift_ppm != expected:
            raise StandingControlsError(
                "maximum_bookend_drift_ppm differs from the sealed speed policy"
            )


def verify_standing_controls(
    primary: object,
    reproduction: object,
    policy: ResidentSpeedPolicy,
) -> StandingEvaluationControls:
    """Reopen one identical, policy-derived controls identity for both stages."""

    primary_controls = StandingEvaluationControls.from_dict(primary)
    reproduction_controls = StandingEvaluationControls.from_dict(reproduction)
    if primary_controls != reproduction_controls:
        raise StandingControlsError(
            "primary and reproduction stages bind different controls"
        )
    primary_controls.require_derived_from(policy)
    return primary_controls


__all__ = [
    "StandingControlsError",
    "StandingEvaluationControls",
    "bookend_drift_ppm_from_max_noise",
    "verify_standing_controls",
]

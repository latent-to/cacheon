"""Frozen qualification speed policy, retaining the historical wire identity.

Only the replay policies remain: version 16 (fixed cutoff, historical evidence)
and version 17 (statistical, every new commission). The batch-cell fields stay
on the wire at zero so sealed replay continuations decode byte for byte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field

from cacheon.eval.goodput_runtime import GoodputPolicy
from cacheon.eval.scoring import CrossoverRuntimeError
from cacheon.stack_identity import canonical_digest, require_sha256_hex

@dataclass(frozen=True)
class ResidentSpeedPolicy:
    """Authority for precommitted reads and the speed-stage wall-clock SLA."""

    max_stage_seconds: int
    min_margin: float
    noise_multiplier: float
    max_noise: float
    calibration_digest: str
    calibration_context_digest: str
    version: int
    max_qualification_seconds: int = 7_200
    min_windows: int = 0
    max_window_scatter: float = 0.0
    max_conditioning_slowdown: float = 0.0
    prefill_min_margin: float = 0.0
    prefill_credit_weight: float = 0.0
    goodput: GoodputPolicy | None = dataclass_field(default=None, metadata={"wire_optional": True})

    def __post_init__(self) -> None:
        if (
            type(self.version) is not int
            or self.version not in (16, 17)
            or type(self.max_stage_seconds) is not int
            or not 60 <= self.max_stage_seconds <= 7_200
            or type(self.max_qualification_seconds) is not int
            or not self.max_stage_seconds
            <= self.max_qualification_seconds
            <= 14_400
            or type(self.min_windows) is not int
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (
                    self.min_margin,
                    self.noise_multiplier,
                    self.max_noise,
                    self.max_window_scatter,
                    self.max_conditioning_slowdown,
                    self.prefill_min_margin,
                    self.prefill_credit_weight,
                )
            )
            or not (self.min_margin == 0 if self.version == 17 else 0 < self.min_margin < 1)
            or self.noise_multiplier <= 0
            or not 0 <= self.max_noise < 1
        ):
            raise CrossoverRuntimeError("resident speed policy is unsupported")
        if self.max_noise > 0.02:
            # Version 2 scores timed windows, where the hardened stack has
            # demonstrated <=0.8% honest spread; a looser ceiling would let a
            # broken measurement convict or crown instead of NO_DECISION.
            raise CrossoverRuntimeError(
                "resident speed policy requires max_noise <= 0.02"
            )
        if (type(self.goodput) is not GoodputPolicy
            or (self.version == 16 and (
                self.goodput.error_rate != 0
                or self.goodput.required != 1 + max(self.min_margin, self.noise_multiplier * self.goodput.null_noise)))
            or (self.version == 17 and (self.goodput.error_rate != 0.01 or self.goodput.required != 1))
            or self.goodput.null_noise > self.max_noise
            or self.goodput.boot_noise > self.max_noise
            # The batch-cell thresholds are wire fields only; replay seals them at zero.
            or any((self.min_windows, self.max_window_scatter, self.max_conditioning_slowdown,
                    self.prefill_min_margin, self.prefill_credit_weight))):
            raise CrossoverRuntimeError("goodput policy differs from its frozen calibration")
        for field in ("calibration_digest", "calibration_context_digest"):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.qualification.goodput-speed-policy.v1", {
            **self.to_dict(), "read_order": ["B", "C"],
            "timing": "paired_fixed_work_host_time", "repeat_limit": 0,
            **({"orientations": 2, "aggregation": "pooled_elapsed_serving_time_equal_log_orientation",
                "eligibility": "alpha_spending_0.05_0.05_0.10_0.80"}
               if self.version == 17 else {}),
        })

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "calibration_context_digest": self.calibration_context_digest,
            "calibration_digest": self.calibration_digest,
            "max_noise": format(self.max_noise, ".17g"),
            "max_qualification_seconds": self.max_qualification_seconds,
            "max_stage_seconds": self.max_stage_seconds,
            "min_margin": format(self.min_margin, ".17g"),
            "noise_multiplier": format(self.noise_multiplier, ".17g"),
            "version": self.version,
            "max_conditioning_slowdown": format(self.max_conditioning_slowdown, ".17g"),
            "max_window_scatter": format(self.max_window_scatter, ".17g"),
            "min_windows": self.min_windows,
            "goodput": self.goodput.to_dict(),
        }
        return row

    @classmethod
    def from_dict(cls, value: object) -> "ResidentSpeedPolicy":
        """Reopen the original decimal-string wire shape and its goodput extension."""
        fields = {
            "calibration_context_digest", "calibration_digest", "max_noise",
            "max_qualification_seconds", "max_stage_seconds", "min_margin",
            "noise_multiplier", "version", "max_conditioning_slowdown",
            "max_window_scatter", "min_windows", "goodput",
        }
        if type(value) is not dict or set(value) != fields:
            raise CrossoverRuntimeError("resident speed policy fields differ")
        decimal_fields = {"max_noise", "min_margin", "noise_multiplier", "max_conditioning_slowdown",
                          "max_window_scatter"}
        try:
            result = cls(**{name: GoodputPolicy.from_dict(item) if name == "goodput" else
                            float(item) if name in decimal_fields else item for name, item in value.items()})
            if result.to_dict() != value:
                raise CrossoverRuntimeError("resident speed policy is noncanonical")
            return result
        except (TypeError, ValueError) as exc:
            raise CrossoverRuntimeError("resident speed policy is malformed") from exc

    @classmethod
    def from_calibration(
        cls,
        *,
        max_stage_seconds: int,
        max_qualification_seconds: int = 7_200,
        calibration: object,
        context: object,
        version: int,
        min_windows: int = 0,
        max_window_scatter: float = 0.0,
        max_conditioning_slowdown: float = 0.0,
        goodput: GoodputPolicy | None = None,
    ) -> "ResidentSpeedPolicy":
        from cacheon.eval.calibration import (
            CalibrationContext,
            CalibrationManifest,
            decimal_value,
        )

        if (
            type(calibration) is not CalibrationManifest
            or type(context) is not CalibrationContext
            or not calibration.thresholds_frozen
        ):
            raise CrossoverRuntimeError("resident speed calibration is not frozen")
        try:
            calibration.require_context(context)
        except ValueError as exc:
            raise CrossoverRuntimeError(str(exc)) from None
        return cls(
            max_stage_seconds=max_stage_seconds,
            min_margin=0.0 if version == 17 else float(decimal_value(calibration.speed.min_margin)),
            noise_multiplier=float(decimal_value(calibration.speed.noise_multiplier)),
            max_noise=float(decimal_value(calibration.speed.max_noise)),
            calibration_digest=calibration.digest,
            calibration_context_digest=context.digest,
            version=version,
            max_qualification_seconds=max_qualification_seconds,
            min_windows=min_windows,
            max_window_scatter=max_window_scatter,
            max_conditioning_slowdown=max_conditioning_slowdown,
            goodput=goodput,
        )

    @classmethod
    def rebound(
        cls, policy: "ResidentSpeedPolicy", *, calibration: object, context: object
    ) -> "ResidentSpeedPolicy":
        """Rebuild ``policy`` from its calibration so sealed evidence regrades
        under the arithmetic that produced it. Every threshold the calibration
        does not own is copied from the policy itself; the caller's equality
        check then refuses any cross-version splice."""

        if type(policy) is not cls:
            raise CrossoverRuntimeError("resident speed policy is not exact")
        return cls.from_calibration(
            max_stage_seconds=policy.max_stage_seconds,
            max_qualification_seconds=policy.max_qualification_seconds,
            calibration=calibration,
            context=context,
            version=policy.version,
            min_windows=policy.min_windows,
            max_window_scatter=policy.max_window_scatter,
            max_conditioning_slowdown=policy.max_conditioning_slowdown,
            goodput=policy.goodput,
        )

# Existing continuation records bind this public type name.
ResidentSpeedPolicy.__module__ = "cacheon.eval.crossover_runtime"

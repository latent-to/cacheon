"""Frozen qualification speed policy, retaining the historical wire identity."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field as dataclass_field

from cacheon.eval.goodput_runtime import GoodputPolicy
from cacheon.eval.resident_measurement import CrossoverRuntimeError
from cacheon.eval.speed_verdict import schedule_roles
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
            or self.version not in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
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
        if self.version in (16, 17):
            if (type(self.goodput) is not GoodputPolicy
                or (self.version == 16 and (
                    self.goodput.error_rate != 0
                    or self.goodput.required != 1 + max(self.min_margin, self.noise_multiplier * self.goodput.null_noise)))
                or (self.version == 17 and (self.goodput.error_rate != 0.01 or self.goodput.required != 1))
                or self.goodput.null_noise > self.max_noise
                or self.goodput.boot_noise > self.max_noise
                or any((self.min_windows, self.max_window_scatter, self.max_conditioning_slowdown))):
                raise CrossoverRuntimeError("goodput policy differs from its frozen calibration")
        elif self.goodput is not None:
            raise CrossoverRuntimeError("goodput authority requires policy version 16 or 17")
        if self.version not in (16, 17) and not 3 <= self.min_windows <= 512:
            raise CrossoverRuntimeError("resident speed policy requires 3..512 timed windows")
        if self.version not in (16, 17) and not 0 < self.max_window_scatter <= 0.25:
            raise CrossoverRuntimeError("resident speed policy requires window scatter in (0, 0.25]")
        # Conditioning is outside scored windows. The historical v3 bound
        # catches gross startup regressions; each timed full request includes prefill.
        if self.version not in (16, 17) and not 1.0 < self.max_conditioning_slowdown <= 2.0:
            raise CrossoverRuntimeError(
                "resident speed policy requires a conditioning slowdown bound in (1, 2]"
            )
        if self.version in (12, 15):
            # The prefill lane admits at a sealed margin and settles at a
            # sealed fraction of the prefill gain; neither is derived from a
            # read, so a noisy box cannot widen its own admission.
            if (
                not 0 < self.prefill_min_margin < 1
                or not 0 < self.prefill_credit_weight <= 1
            ):
                raise CrossoverRuntimeError(
                    "resident speed policy v12 requires a prefill margin in"
                    " (0, 1) and a credit weight in (0, 1]"
                )
        elif self.prefill_min_margin != 0.0 or self.prefill_credit_weight != 0.0:
            raise CrossoverRuntimeError(
                "prefill lane thresholds require resident speed policy v12"
            )
        for field in ("calibration_digest", "calibration_context_digest"):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None

    def conditioning_regression(
        self, baseline_row: object, candidate_row: object
    ) -> bool:
        """Whether the candidate's conditioning span regressed past the bound.

        Conditioning spans carry warm/cold session structure (measured 50%
        cold-vs-warm on 2026-07-25), so callers must pair reads of the same
        warmth position: C against B (both first reads, cold) and C-prime
        against B-prime (both continuations, warm). Never mix positions."""

        baseline_tokens = baseline_row.conditioning_tokens  # type: ignore[attr-defined]
        candidate_tokens = candidate_row.conditioning_tokens  # type: ignore[attr-defined]
        if baseline_tokens != candidate_tokens:
            raise CrossoverRuntimeError(
                "conditioning spans are not workload-comparable"
            )
        return (
            candidate_row.conditioning_seconds  # type: ignore[attr-defined]
            > self.max_conditioning_slowdown
            * baseline_row.conditioning_seconds  # type: ignore[attr-defined]
        )

    def read_window_scatter(self, row: object) -> float:
        """Relative MAD-about-the-median of one read's per-window rates.

        Robust by construction: the same tail events (completion churn,
        admission hiccups) that motivate the median statistic must not be
        allowed to inflate its own fitness gate."""

        windows = getattr(row, "windows", ())
        if len(windows) < max(self.min_windows, 3):
            raise CrossoverRuntimeError(
                "resident read lacks its required timed windows"
            )
        rates = [window.tokens / window.seconds for window in windows]
        median = statistics.median(rates)
        if not math.isfinite(median) or median <= 0:
            raise CrossoverRuntimeError("resident read window rates are invalid")
        return statistics.median([abs(rate - median) for rate in rates]) / median

    def scored_tokens_per_second(self, row: object) -> float:
        """Rate the complete mixed workload, or the single-cell window median.

        The retired charged-rate rule double-counted cold start against the
        first arm. Every supported policy scores timed work and retains scatter.
        """

        self.read_window_scatter(row)
        if self.version in (9, 11, 12, 14, 15):
            # Mixed-cell qualification deliberately contains heterogeneous
            # batch widths and output budgets. A median of per-batch rates
            # would erase the minority cell; total timed tokens over the
            # host-observed makespan gives the sealed mixture one rate.
            return row.timed_tokens / row.timed_seconds  # type: ignore[attr-defined]
        return statistics.median(window.tokens / window.seconds for window in row.windows)

    @property
    def digest(self) -> str:
        if self.goodput is not None:
            return canonical_digest("cacheon.qualification.goodput-speed-policy.v1", {
                **self.to_dict(), "read_order": ["B", "C"],
                "timing": "paired_fixed_work_host_time", "repeat_limit": 0,
                **({"orientations": 2, "aggregation": "pooled_elapsed_serving_time_equal_log_orientation",
                    "eligibility": "alpha_spending_0.05_0.05_0.10_0.80"}
                   if self.version == 17 else {}),
            })
        return canonical_digest(
            "cacheon.qualification.resident-speed-policy",
            {
                **self.to_dict(),
                # The sealed identity states the schedule it was measured under.
                # Version 8 reads the bookend unconditionally, so it must not
                # claim the conditional read order that versions 6 and 7 seal.
                "borderline_band": ("valid_threshold_crossing_one_repeat"
                                    if self.version >= 13 else "invariant_over_reads_taken"),
                "read_order": (
                    list(schedule_roles(self.version, repeat=self.version >= 13))
                ),
                "timing": "serialized_resident_host_time",
                **({"repeat_aggregation": "equal_log_weight_faster_bracket",
                    "repeat_limit": 1, "repeat_exhausted": "speed_threshold_not_met"}
                   if self.version >= 13 else {}),
            },
        )

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
        }
        if self.version in (12, 15):
            row["prefill_credit_weight"] = format(self.prefill_credit_weight, ".17g")
            row["prefill_min_margin"] = format(self.prefill_min_margin, ".17g")
        if self.goodput is not None:
            row["goodput"] = self.goodput.to_dict()
        return row

    @classmethod
    def from_dict(cls, value: object) -> "ResidentSpeedPolicy":
        """Reopen the original decimal-string wire shape and its goodput extension."""
        fields = {
            "calibration_context_digest", "calibration_digest", "max_noise",
            "max_qualification_seconds", "max_stage_seconds", "min_margin",
            "noise_multiplier", "version", "max_conditioning_slowdown",
            "max_window_scatter", "min_windows",
        }
        if type(value) is not dict or type(value.get("version")) is not int:
            raise CrossoverRuntimeError("resident speed policy fields differ")
        if value["version"] in (12, 15):
            fields |= {"prefill_credit_weight", "prefill_min_margin"}
        if value["version"] in (16, 17):
            fields.add("goodput")
        if set(value) != fields:
            raise CrossoverRuntimeError("resident speed policy fields differ")
        decimal_fields = {"max_noise", "min_margin", "noise_multiplier", "max_conditioning_slowdown",
                          "max_window_scatter", "prefill_credit_weight", "prefill_min_margin"}
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
        prefill_min_margin: float = 0.0,
        prefill_credit_weight: float = 0.0,
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
            prefill_min_margin=prefill_min_margin,
            prefill_credit_weight=prefill_credit_weight,
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
            prefill_min_margin=policy.prefill_min_margin,
            prefill_credit_weight=policy.prefill_credit_weight,
            goodput=policy.goodput,
        )

# Existing continuation records bind this public type name.
ResidentSpeedPolicy.__module__ = "cacheon.eval.crossover_runtime"

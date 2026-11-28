"""Commercial API-price effective revenue and tail latency metrics for replay grading.

Every turn's list value uses the validator-sealed ideal prefix-cache overlap, never the
candidate's self-reported cache hits, so evicting prefixes can never inflate billable
input revenue. Per-turn continuous weights discount turns that fall below the sealed API
decode, TTFT, or streaming inter-token latency (ITL) promises so batch-tier degradation,
tail starvation, and oversized-chunk streaming freezes lose revenue directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Default floor of the linear per-turn decode speed ramp as a share of the API speed promise.
DECODE_RAMP_FLOOR = 0.6


def _finite_nonneg(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


@dataclass(frozen=True)
class ApiPriceTable:
    """List prices in USD per 1M tokens for uncached input, ideal cached input, and output."""

    input_per_m: float
    cached_input_per_m: float
    output_per_m: float

    def __post_init__(self) -> None:
        if (
            any(not _finite_nonneg(v) for v in (self.input_per_m, self.cached_input_per_m, self.output_per_m))
            or self.input_per_m <= 0.0
            or self.output_per_m <= 0.0
            or self.cached_input_per_m > self.input_per_m
        ):
            raise ValueError("API price table must have positive input/output and cached <= input")
        object.__setattr__(self, "input_per_m", float(self.input_per_m))
        object.__setattr__(self, "cached_input_per_m", float(self.cached_input_per_m))
        object.__setattr__(self, "output_per_m", float(self.output_per_m))

    def to_dict(self) -> dict[str, str]:
        return {
            "cached_input_per_m": format(self.cached_input_per_m, ".17g"),
            "input_per_m": format(self.input_per_m, ".17g"),
            "output_per_m": format(self.output_per_m, ".17g"),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ApiPriceTable":
        expected = {"input_per_m", "cached_input_per_m", "output_per_m"}
        if type(value) is not dict or set(value) != expected or any(type(v) is not str for v in value.values()):
            raise ValueError("API price table fields differ")
        result = cls(**{k: float(v) for k, v in value.items()})
        if result.to_dict() != value:
            raise ValueError("API price table is not canonical")
        return result


def turn_list_value(record, prices: ApiPriceTable, *, ideal_cached: int | None = None) -> float:
    """Ideal USD list value of one turn using only validator-sealed prefix overlap."""
    cached = record.ideal_cached_tokens if ideal_cached is None else ideal_cached
    cached = 0 if cached is None else min(record.prompt_tokens, max(0, cached))
    uncached = record.prompt_tokens - cached
    return (
        uncached * prices.input_per_m
        + cached * prices.cached_input_per_m
        + record.output_tokens * prices.output_per_m
    ) / 1e6


def turn_quality_weight(
    record, contract, *, decode_ramp_floor: float = DECODE_RAMP_FLOOR
) -> float:
    """Smooth [0, 1] API-promise weight across per-user decode speed, TTFT, and streaming ITL."""
    if record.status != "ok":
        return 0.0
    rate = record.decode_tps
    if rate is None:
        w_dec = 1.0
    else:
        low = decode_ramp_floor * contract.decode_floor_tps
        span = contract.decode_floor_tps - low
        w_dec = 1.0 if span <= 0 else min(1.0, max(0.0, (rate - low) / span))
    t_bound = contract.ttft_bound_s
    w_ttft = min(1.0, max(0.0, (2.0 * t_bound - record.ttft_s) / t_bound))
    itl_bound = getattr(contract, "max_itl_s", 0.0)
    if itl_bound > 0.0 and record.output_tokens > 1:
        if record.max_itl_ns is None:
            w_itl = 0.0
        else:
            gap_s = record.max_itl_ns / 1e9
            w_itl = min(1.0, max(0.0, (3.0 * itl_bound - gap_s) / (2.0 * itl_bound)))
    else:
        w_itl = 1.0
    return w_dec * w_ttft * w_itl


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))]


def decode_p10(reads: Sequence[object]) -> float:
    """10th-percentile per-user decode rate across every multi-token turn of ``reads``."""
    rates = [rate for read in reads for r in read.records if (rate := r.decode_tps) is not None]
    if not rates:
        raise ValueError("reads hold no turn with a per-user decode rate")
    return _percentile(rates, 0.10)


def itl_p99_s(reads: Sequence[object]) -> float | None:
    """99th-percentile host-observed max inter-token gap (seconds), or None if unrecorded."""
    gaps = [
        r.max_itl_ns / 1e9
        for read in reads
        for r in read.records
        if r.output_tokens > 1 and r.max_itl_ns is not None
    ]
    if not gaps:
        return None
    return _percentile(gaps, 0.99)


def mean_quality_weight(reads: Sequence[object], contract) -> float:
    """Value-unweighted mean API-promise weight across all turns of ``reads``."""
    records = [r for read in reads for r in read.records]
    if not records:
        return 0.0
    return math.fsum(turn_quality_weight(r, contract) for r in records) / len(records)


def effective_revenue_summary(
    records: Sequence[object], duration_s: float, *, prices: ApiPriceTable, contract, gpus: int = 1
) -> tuple[float, float]:
    """Return ``(cost_s_per_effective_usd, effective_usd_per_gpu_hr)`` over ``records``.

    ``cost_s_per_effective_usd`` is ``math.inf`` when effective revenue is zero so the caller
    can classify a stalled or batch-degraded candidate as FAIL rather than an infra error.
    """
    if duration_s <= 0 or gpus < 1:
        raise ValueError("effective revenue span must be positive")
    effective_usd = math.fsum(
        turn_list_value(r, prices) * turn_quality_weight(r, contract) for r in records
    )
    if effective_usd <= 0.0:
        return math.inf, 0.0
    usd_per_gpu_hr = effective_usd / (gpus * (duration_s / 3600.0))
    return duration_s / effective_usd, usd_per_gpu_hr


__all__ = [
    "ApiPriceTable",
    "DECODE_RAMP_FLOOR",
    "decode_p10",
    "effective_revenue_summary",
    "itl_p99_s",
    "mean_quality_weight",
    "turn_list_value",
    "turn_quality_weight",
]

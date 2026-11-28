"""Commercial API-price list revenue and tail latency metrics for replay grading.

Every turn's list value uses the validator-sealed ideal prefix-cache overlap, never the
candidate's self-reported cache hits, so evicting prefixes can never inflate billable
input revenue. Policy 24 scores unweighted ideal-cached list revenue per GPU-hour while
enforcing per-user decode p50, tail decode p10, TTFT p95, and host-clocked streaming ITL
p99 as hard per-load gates (measuring continuous quality weighting on 4×H200 stock boots
on 2026-11-28 tripled boot SD from 0.64% to 1.98% and inflated bundle ratios by +16–21%).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


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


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))]


def decode_p10(reads: Sequence[object]) -> float:
    """10th-percentile per-user decode rate across every multi-token turn of ``reads``."""
    rates = [rate for read in reads for r in read.records if (rate := r.decode_tps) is not None]
    if not rates:
        raise ValueError("reads hold no turn with a per-user decode rate")
    return _percentile(rates, 0.10)


def ttft_p95_s(reads: Sequence[object]) -> float:
    """95th-percentile time-to-first-token (seconds) across every turn of ``reads``."""
    ttfts = [r.ttft_s for read in reads for r in read.records]
    if not ttfts:
        raise ValueError("reads hold no turn records")
    return _percentile(ttfts, 0.95)


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


def effective_revenue_summary(
    records: Sequence[object], duration_s: float, *, prices: ApiPriceTable, contract=None, gpus: int = 1
) -> tuple[float, float]:
    """Return ``(cost_s_per_usd, usd_per_gpu_hr)`` over completed turns of ``records``.

    Revenue bills each completed turn at its validator-sealed ideal-cached list price without
    per-turn quality weighting; decode p50, tail decode p10, TTFT p95, and streaming ITL p99
    are enforced as separate hard gates. ``cost_s_per_usd`` is ``math.inf`` when zero turns
    completed cleanly so failed engines classify as FAIL rather than an infra error.
    """
    if duration_s <= 0 or gpus < 1:
        raise ValueError("effective revenue span must be positive")
    list_usd = math.fsum(turn_list_value(r, prices) for r in records if r.status == "ok")
    if list_usd <= 0.0:
        return math.inf, 0.0
    usd_per_gpu_hr = list_usd / (gpus * (duration_s / 3600.0))
    return duration_s / list_usd, usd_per_gpu_hr


__all__ = [
    "ApiPriceTable",
    "decode_p10",
    "effective_revenue_summary",
    "itl_p99_s",
    "ttft_p95_s",
    "turn_list_value",
]

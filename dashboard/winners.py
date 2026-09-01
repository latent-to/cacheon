"""Pure winner-speed calculations used by the read-only dashboard."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any


def settled_speedup(candidate: dict[str, Any]) -> Decimal | None:
    """Return the conservative reproduced gain stored for one CROWN."""

    primary = candidate.get("primary") or {}
    reproduction = candidate.get("reproduction") or {}
    try:
        values = (
            Decimal(str(primary["speedup"])),
            Decimal(str(reproduction["speedup"])),
        )
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return None
    if any(not value.is_finite() or value <= 1 for value in values):
        return None
    return min(values)


def _lane_tokens_per_second(speed: object, role: str) -> Decimal | None:
    if not isinstance(speed, dict):
        return None
    lanes = speed.get("lanes")
    if not isinstance(lanes, list):
        return None
    for lane in lanes:
        if not isinstance(lane, dict) or lane.get("role") != role:
            continue
        try:
            rate = Decimal(str(lane["tokens_per_second"]))
        except (InvalidOperation, KeyError, TypeError, ValueError):
            return None
        if rate.is_finite() and rate > 0:
            return rate
        return None
    return None


def conservative_candidate_tokens_per_second(
    speeds: list[object],
) -> Decimal | None:
    """Use the slower independently passing candidate lane as the tok/s estimate."""

    rates = [
        rate
        for speed in speeds
        if (rate := _lane_tokens_per_second(speed, "C")) is not None
    ]
    return min(rates) if rates else None


def estimated_sglang_tokens_per_second(
    candidate_tokens_per_second: Decimal | None,
    cumulative_speedup: Decimal | None,
) -> Decimal | None:
    if (
        candidate_tokens_per_second is None
        or cumulative_speedup is None
        or cumulative_speedup <= 0
    ):
        return None
    return candidate_tokens_per_second / cumulative_speedup


def cumulative_crown_speedups(
    crown_events: list[dict[str, Any]],
) -> dict[str, Decimal]:
    """Compound accepted marginal gains by target from retained SGLang stock."""

    cumulative_by_target: dict[str, Decimal] = {}
    by_reservation: dict[str, Decimal] = {}
    for event in sorted(crown_events, key=lambda row: int(row["sequence"])):
        reservation_id = event.get("reservation_id")
        target_id = event.get("target_id")
        candidate_raw = event.get("candidate_json")
        if (
            not isinstance(reservation_id, str)
            or not reservation_id
            or not isinstance(target_id, str)
            or not target_id
            or not isinstance(candidate_raw, str)
        ):
            continue
        try:
            candidate = json.loads(candidate_raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(candidate, dict):
            continue
        relative = settled_speedup(candidate)
        if relative is None:
            continue
        cumulative = cumulative_by_target.get(target_id, Decimal(1)) * relative
        cumulative_by_target[target_id] = cumulative
        by_reservation[reservation_id] = cumulative
    return by_reservation


__all__ = [
    "conservative_candidate_tokens_per_second",
    "cumulative_crown_speedups",
    "estimated_sglang_tokens_per_second",
    "settled_speedup",
]

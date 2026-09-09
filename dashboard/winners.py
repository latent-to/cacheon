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


def measured_baseline(speed_reads: list[object], primary: dict[str, Any]) -> dict[str, Any]:
    """Slowest measured B/B-prime rate; identify stock versus an incumbent stack."""
    rates = [rate for speed in speed_reads for role in ("B", "B_prime")
             if (rate := _lane_tokens_per_second(speed, role)) is not None]
    manifest = primary.get("incumbent_manifest")
    kind = ("stock" if not manifest.get("entries") else "incumbent") if isinstance(manifest, dict) else "unknown"
    return {
        "baseline_tokens_per_second": round(float(min(rates)), 1) if rates else None,
        "baseline_kind": kind,
    }


def sglang_comparison(candidate: dict[str, Any], cumulative: Decimal | None) -> dict[str, Any]:
    """Prefer a recorded stock comparison; otherwise use retained crown lineage."""
    passes = [candidate.get("primary") or {}]
    if candidate.get("reproduction"):
        passes.append(candidate["reproduction"])
    direct = all(isinstance(p.get("incumbent_manifest"), dict)
                 and p["incumbent_manifest"].get("entries") == {} for p in passes)
    ratio, basis = cumulative, "Crown lineage"
    if direct:
        try:
            values = [Decimal(str(p["speedup"])) for p in passes]
        except (KeyError, InvalidOperation, ValueError, TypeError):
            values = []
        ratio = min(values) if values and all(v.is_finite() and v > 0 for v in values) else None
        basis = "Measured vs stock"
    return {
        "sglang_speedup": float(ratio) if ratio is not None else None,
        "sglang_improvement_pct": float((ratio - 1) * 100) if ratio is not None else None,
        "sglang_comparison_basis": basis if ratio is not None else "Unavailable",
    }


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


def live_offer_shares(path: object) -> tuple[dict[str, Any] | None, dict[str, Decimal]]:
    """Read the validator's served weight offer as ``(summary, {hotkey: share})``.

    The offer file is what the weight-offer service serves to every follower,
    so its vector is the reward split this validator currently stands behind.
    Returns ``(None, {})`` when the file is absent or malformed; the caller
    reports that absence instead of inventing a vector.
    """

    try:
        with open(path, encoding="utf-8") as fh:
            offer = json.load(fh)["offer"]
        projection = offer["projection"]
        rows = projection["weights_ppm"]
        shares = {
            str(hotkey): Decimal(int(ppm)) / Decimal(1_000_000)
            for hotkey, ppm in rows
        }
        summary = {
            "lane": str(offer.get("lane") or ""),
            "projection_digest": str(offer["projection_digest"]),
            "effective_block": int(projection["effective_block"]),
            "crown_count": int(projection.get("crown_count") or 0),
            "stack_generation": int(projection.get("stack_generation") or 0),
            "validator_hotkey": str(projection.get("validator_hotkey") or ""),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, {}
    return summary, shares


def with_competitive_results(con, passed):
    """Filter A/B-only passes and attach immutable completion-time winner details."""

    from dashboard.baseline_api import competition_details

    winners = []
    for row in passed:
        competition = competition_details(con, row["reservation_id"])
        if competition.get("won", True):
            winners.append(dict(row) | {"ranking": competition})
    return winners


def baseline_relationship(con, target_id, baseline_artifact, result, lineage_tables_available=None):
    """Describe the retained baseline's lineage for submission details."""

    if lineage_tables_available is None:
        tables = {
            row["name"]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('target_lineage_tips','target_lineage_nodes')"
            )
        }
        lineage_tables_available = tables == {
            "target_lineage_tips",
            "target_lineage_nodes",
        }
    if not lineage_tables_available:
        return result
    tip = con.execute(
        "SELECT artifact_digest FROM target_lineage_tips WHERE target_id=?",
        (target_id,),
    ).fetchone()
    if tip is None:
        return result
    tip_artifact = str(tip["artifact_digest"])
    result["current_tip_artifact_digest"] = tip_artifact
    if baseline_artifact == tip_artifact:
        result["relationship"] = "current_tip"
        result["threshold_speedup"] = 1.0
        return result

    nodes: list[dict[str, Any]] = []
    artifact = tip_artifact
    seen: set[str] = set()
    while artifact and artifact not in seen:
        seen.add(artifact)
        node = con.execute(
            "SELECT artifact_digest,parent_artifact_digest,winner_speedup "
            "FROM target_lineage_nodes WHERE target_id=? AND artifact_digest=?",
            (target_id, artifact),
        ).fetchone()
        if node is None:
            break
        nodes.append(dict(node))
        artifact = str(node["parent_artifact_digest"])
    nodes.reverse()
    start = next(
        (
            index for index, node in enumerate(nodes)
            if node["parent_artifact_digest"] == baseline_artifact
        ),
        None,
    )
    if start is None:
        result["relationship"] = "outside_active_lineage"
        return result
    threshold = Decimal(1)
    for node in nodes[start:]:
        threshold *= Decimal(str(node["winner_speedup"]))
    result["relationship"] = "ancestor"
    result["threshold_speedup"] = float(threshold)
    return result

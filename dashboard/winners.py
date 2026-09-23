"""Eligible winner queries and speed calculations for the read-only dashboard."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

from cacheon.chain.evaluation_order import reward_visibility_sql


def qualified_winners(con) -> list[dict[str, Any]]:
    """Expose retained PASSes only after their arrival prefix becomes eligible."""
    return [dict(row) for row in con.execute("""
        SELECT sc.reservation_id, sc.status, sc.reason, sc.candidate_json,
               r.*, r.block AS submission_block,
               max(q.retained_block) AS passed_block
        FROM settlement_candidates sc
        JOIN reservations r ON r.reservation_id = sc.reservation_id
        JOIN settlement_qualifications q ON q.reservation_id = sc.reservation_id
        WHERE r.status='qualified' AND r.decision='PASS'
          AND sc.status!='duplicate_proposal' AND
    """ + reward_visibility_sql(con) + """
        GROUP BY sc.reservation_id
    """)]


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


def prefill_summary(speed_reads: list[object]) -> dict[str, float | None]:
    """Keep the conservative observed prompt gain across retained passing attempts."""
    ratios = [speed["prefill"]["speedup"] for speed in speed_reads
              if isinstance(speed, dict) and speed.get("prefill")]
    return {"prefill_speedup": min(ratios) if ratios else None}


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


def latest_hold(connection: Any, reservation_id: str, fallback: str) -> tuple[str, int | None]:
    """Reason and sequence of the newest HOLD event: the candidates table only says ``held``."""
    row = connection.execute(
        "SELECT event_type,event_json,sequence FROM settlement_events "
        "WHERE reservation_id=? ORDER BY sequence DESC LIMIT 1", (reservation_id,)
    ).fetchone()
    if row is not None and row["event_type"] == "HOLD":
        return json.loads(row["event_json"]).get("reason") or fallback, row["sequence"]
    return fallback, None


def settlement_label(connection: Any, reservation_id: str, status: object, reason: object) -> str:
    """A stale hold is a paid pass, not a fault: 2026-09-22 two paid Qwen passes read as unpaid under "held"."""
    if status != "held":
        return str(status or "")
    held_reason, _ = latest_hold(connection, reservation_id, str(reason or "held"))
    return "paid pass" if held_reason == "stale_incumbent" else "held"


def settlement_hold_notice(connection: Any, reservation_id: str,
                           settlement: dict[str, Any]) -> dict[str, Any] | None:
    """Explain a currently held candidate using its latest retained settlement event."""
    if settlement.get("status") != "held":
        return None
    reason, sequence = latest_hold(connection, reservation_id, settlement.get("reason") or "held")
    if reason == "stale_incumbent":
        title = "Paid pass — not the champion"
        message = ("This submission passed evaluation and is paid like every other pass. "
                   "It was timed against an earlier baseline than the current champion, "
                   "so it was not adopted as the champion itself.")
    else:
        title = "Submission held"
        message = ("Settlement is holding this submission. "
                   + ("No more specific reason was retained." if reason == "held"
                      else "Recorded reason: " + reason.replace("_", " ") + "."))
    return {"title": title, "message": message, "reason": reason,
            "event_sequence": sequence}


def reward_exclusion_notice(hotkey: str, offer_path: object) -> dict[str, Any] | None:
    """Report only operator exclusions referenced by the currently served offer."""
    from pathlib import Path
    from cacheon.stack_identity import canonical_digest

    try:
        rule = json.loads(Path("/root/cacheon-ops/weight-controls/20260908/exclusions.json").read_text())
        projection = json.loads(Path(offer_path).read_text())["offer"]["projection"]
        decision = canonical_digest("cacheon.operator.source-copy-exclusion.v1", rule)
        if decision not in projection["evidence_digests"]:
            return None
        record = next((r for r in rule["records"] if r["hotkey"] == hotkey), None)
        if record is None or dict(projection["weights_ppm"]).get(hotkey, 0) > 0:
            return None
        return {
            "title": "Rewards excluded — operator decision",
            "message": "The current validator weight offer excludes this hotkey following an operator source-copy review.",
            "reason": "operator_source_copy_exclusion",
            "evidence": record.get("evidence") or "",
            "source_reservation": record.get("copied_from_reservation") or "",
            "decision_time": rule.get("created_at") or "",
            "offer_block": projection["effective_block"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None

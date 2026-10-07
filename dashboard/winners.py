"""Eligible winner queries and speed calculations for the read-only dashboard."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from cacheon.chain.evaluation_order import reward_visibility_sql, reward_comparisons


def excluded_claims() -> set[str]:
    """Reservations the operator priced at zero in the producer's rule file; empty when none is named."""
    try:
        return {row["reservation_id"] for row in
                json.loads(Path(os.environ["CACHEON_DASH_EXCLUSIONS"]).read_text()).get("claims", ())}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def qualified_winners(con, *, include_waiting: bool = False) -> list[dict[str, Any]]:
    """Read reward winners, optionally including PASSes awaiting queue resolution."""
    comparisons, excluded = reward_comparisons(con), excluded_claims()
    return [dict(row) | {"reward_eligible": False, "excluded": row["reservation_id"] in excluded}
            | reward_comparison_summary(comparisons.get(row["reservation_id"], {}))
            for row in con.execute("""
        SELECT sc.reservation_id, sc.status, sc.reason, sc.candidate_json,
               r.*, r.block AS submission_block,
               NOT (""" + reward_visibility_sql(con) + """) AS waiting_for_queue,
               max(q.retained_block) AS passed_block
        FROM settlement_candidates sc
        JOIN reservations r ON r.reservation_id = sc.reservation_id
        JOIN settlement_qualifications q ON q.reservation_id = sc.reservation_id
        WHERE r.status='qualified' AND r.decision='PASS'
          AND sc.status!='duplicate_proposal'
        GROUP BY sc.reservation_id
    """) if (include_waiting and row["waiting_for_queue"])
            or comparisons.get(row["reservation_id"], {}).get("reward_eligible")]


def winner_lists(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep pending PASSes out of the finalized winner list and count."""
    waiting = [item for item in items if item["waiting_for_queue"]]
    finalized = [item for item in items if not item["waiting_for_queue"]]
    return {"items": finalized, "pass_total": len(finalized),
            "waiting_items": waiting, "waiting_total": len(waiting)}


def reward_comparison_summary(comparison: dict) -> dict:
    """Expose the queue predecessor and scoring gain without changing baseline metrics."""
    from decimal import ROUND_FLOOR
    from cacheon.economics import WEIGHT_PPM

    if not comparison:
        return {}
    score_ppm = int((comparison["score_speedup"] * WEIGHT_PPM).to_integral_value(rounding=ROUND_FLOOR))
    return {
        "previous_best_reservation_id": comparison["previous_best_reservation_id"],
        "previous_best_speedup": float(comparison["previous_best_speedup"]),
        "relative_improvement_pct": float((comparison["relative_speedup"] - 1) * 100),
        "score_improvement_pct": (score_ppm - WEIGHT_PPM) / (WEIGHT_PPM / 100),
        "reward_eligible": comparison["reward_eligible"],
        "reward_reason": None if comparison["reward_eligible"] else "lost_potential",
        "grandfathered": comparison["grandfathered"],
    }


def reward_status(comparisons: dict, reservation_id: str, offer, shares, excluded=()) -> dict:
    """What a PASS earns: the producer's ordered comparison plus its share of the served offer."""
    if reservation_id not in comparisons:
        return {}
    return reward_comparison_summary(comparisons[reservation_id]) | winner_reward(
        {"reservation_id": reservation_id, "waiting_for_queue": False, "excluded": reservation_id in excluded}, offer, shares)


def submission_reward_comparison(con, reservation_id: str, offer=None, shares=None) -> dict:
    """Use the same ordered comparison as the weight producer for submission details."""
    return reward_status(reward_comparisons(con), reservation_id, offer, shares or {}, excluded_claims())


def result_summary(speed: object) -> dict[str, Any] | None:
    """What a replay row leads with: measured and required gain, passes run, decode and first-token time per arm.

    A batch-cell attempt has no retained grade, and its raw C/B lane ratio is not the gain that was credited.
    """
    if not isinstance(speed, dict) or "grading" not in speed:
        return None
    grade = speed["grading"]

    def mean(role: str, field: str) -> float | None:
        values = [lane[field] for lane in speed["lanes"] if lane["role"] == role and lane[field] is not None]
        return sum(values) / len(values) if values else None
    return {"speedup": speed["speedup"], "required_speedup": grade["required_speedup"],
            "passes": speed["windows"], "pass_limit": speed["window_limit"], "detail": grade["detail"],
            "decode_tps": [mean("B", "decode_tps"), mean("C", "decode_tps")],
            "ttft_s": [mean("B", "mean_ttft_s"), mean("C", "mean_ttft_s")]}


def list_results(con, items: list[dict[str, Any]], roots, offer, shares) -> dict[tuple[str, str], dict[str, Any]]:
    """Give each listed submission its last graded result and what it earns; return the pay bars.

    On 2026-10-01 the list read "PASS / qualified" for a +6.13% pass and a +1.06% pass alike,
    and nothing at all for a FAIL that had measured +0.84% against a required +1.21%.
    """
    from dashboard.forensics import retained_speed

    comparisons, excluded = reward_comparisons(con), excluded_claims()
    attempts = {row["reservation_id"]: row["attempt_ref_json"] for row in con.execute(
        "SELECT reservation_id, attempt_ref_json FROM qualification_dispositions WHERE reservation_id IN ("
        + ",".join("?" * len(items)) + ") ORDER BY attempt_index", [item["reservation_id"] for item in items])}
    for item in items:
        item["result"] = result_summary(retained_speed(attempts.get(item["reservation_id"]), roots, item["target_id"]))
        item["reward"] = reward_status(comparisons, item["reservation_id"], offer, shares, excluded) or None
    return reward_bars(con, comparisons)


def reward_bars(con, comparisons=None) -> dict[tuple[str, str], dict[str, Any]]:
    """Best paid PASS per (arena, baseline stack): what a later PASS on that baseline must beat.

    On 2026-10-01 every GLM row read "must beat 1.048008x": the crown lineage of an operator-excluded
    result. Lineage adoption has not decided pay since previous-paid-winner scoring replaced it, and a
    PASS the operator priced at zero is not shown as the result to beat.
    """
    bars: dict[tuple[str, str], dict[str, Any]] = {}
    comparisons, excluded = reward_comparisons(con) if comparisons is None else comparisons, excluded_claims()
    for row in con.execute("SELECT reservation_id, candidate_json FROM settlement_candidates"):
        comparison = comparisons.get(row["reservation_id"])
        if not comparison or not comparison["reward_eligible"] or row["reservation_id"] in excluded:
            continue
        primary = json.loads(row["candidate_json"])["primary"]
        group = primary.get("arena_digest"), primary.get("incumbent_stack_digest")
        score = float(comparison["previous_best_speedup"] * comparison["relative_speedup"])
        if score > bars.get(group, {"speedup": 0.0})["speedup"]:
            bars[group] = {"speedup": score, "reservation_id": row["reservation_id"]}
    return bars


def baseline_kind(primary: dict[str, Any]) -> dict[str, str]:
    """Name what a winner was timed against: stock SGLang or the crowned incumbent stack."""
    manifest = primary.get("incumbent_manifest")
    if not isinstance(manifest, dict):
        return {"baseline_kind": "unknown"}
    return {"baseline_kind": "stock" if not manifest.get("entries") else "incumbent"}


def live_offer_shares(path: object, *, submission_roots=None, submission_path=None) -> tuple[dict[str, Any] | None, dict[str, Decimal]]:
    """Read served shares keyed by hotkey, or reservation when evidence roots are supplied.

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
    if submission_roots is not None or submission_path is not None:
        from cacheon.eval.evidence_store import EvidenceArtifactRef, reopen_evidence

        shares = {}
        summary["submission_shares_available"] = False
        for root in (None,) if submission_path is not None else submission_roots:
            try:
                if submission_path is not None:
                    report = json.loads(Path(submission_path).read_text())
                    if report["projection_digest"] != summary["projection_digest"]:
                        break
                else:
                    ref = EvidenceArtifactRef.from_dict(projection["allocation_evidence"])
                    report = json.loads(reopen_evidence(root, ref))
                shares = {rid: Decimal(ppm) / Decimal(1_000_000)
                          for rid, ppm in report["submission_weights_ppm"].items()}
            except (OSError, KeyError, TypeError, ValueError):
                continue
            summary["submission_shares_available"] = True
            break
    return summary, shares


def winner_reward(row, offer, shares) -> dict[str, Any]:
    """Attribute only this reservation's share; pending or unavailable amounts stay unknown."""
    share = None
    if row["waiting_for_queue"]:
        status = "waiting_for_queue"
    elif offer is None:
        status = "offer_unavailable"
    elif not offer.get("submission_shares_available"):
        status = "attribution_unavailable"
    else:
        share = float(shares.get(row["reservation_id"], Decimal(0)))
        status = "earning" if share else "excluded" if row.get("excluded") else "not_earning"
    return {"weight_share": share, "reward_claim_status": status}


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
    """A crown hold is a PASS, not a fault; reward records are checked separately.

    On 2026-09-22 two earning Qwen passes read as unpaid under "held"; on 2026-10-01 a PASS
    earning 17% did, held as ``lost_potential`` because it did not exceed the crown record.
    """
    if status != "held":
        return str(status or "")
    held_reason, _ = latest_hold(connection, reservation_id, str(reason or "held"))
    return "passed" if held_reason in ("stale_incumbent", "lost_potential") else "held"


def settlement_hold_notice(connection: Any, reservation_id: str,
                           settlement: dict[str, Any]) -> dict[str, Any] | None:
    """Explain a finalized reward loss or the latest retained settlement hold."""
    lost = settlement.get("reward_eligible") is False
    if settlement.get("status") != "held" and not lost:
        return None
    reason, sequence = latest_hold(connection, reservation_id, settlement.get("reason") or "held")
    if lost or reason == "lost_potential":
        # On 2026-10-01 a PASS earning 17% of the offer was titled "lost comparison": its crown hold, not its pay.
        title = "Potential winner lost comparison" if lost else "Passed evaluation — did not take the crown"
        comparison = ("the best earlier PASS by the reward margin" if lost
                      else "the current champion on the measured baseline. Rewards are decided separately from the crown")
        message = ("This submission passed evaluation, but did not beat " + comparison
                   + ". Its PASS and measurements are retained.")
        sequence = sequence if reason == "lost_potential" else None
        reason = "lost_potential"
    elif reason == "stale_incumbent":
        title = "Passed evaluation — not the champion"
        message = ("This submission passed evaluation. Rewards require beating the best earlier "
                   "PASS by the configured margin, except for grandfathered runtime generations. "
                   "It was timed against an earlier baseline than the current champion, "
                   "so it was not adopted as the champion itself.")
    else:
        title = "Submission held"
        message = ("Settlement is holding this submission. "
                   + ("No more specific reason was retained." if reason == "held"
                      else "Recorded reason: " + reason.replace("_", " ") + "."))
    return {"title": title, "message": message, "reason": reason,
            "event_sequence": sequence}


def reward_exclusion_notice(hotkey: str, offer_path: object, reservation_id: str = "") -> dict[str, Any] | None:
    """Report the operator decision the weight producer applies to this PASS or this hotkey.

    ``CACHEON_DASH_EXCLUSIONS`` names the producer's own rule file. The served offer no longer carries
    a digest this reader can recompute, so a hotkey record shows only while that offer pays it nothing.
    """
    try:
        rule = json.loads(Path(os.environ["CACHEON_DASH_EXCLUSIONS"]).read_text())
        projection = json.loads(Path(offer_path).read_text())["offer"]["projection"]
        claim = next((r for r in rule.get("claims", ()) if r["reservation_id"] == reservation_id), None)
        if claim is not None:
            return {"title": "This PASS is priced at zero — operator decision", "message": claim["reason"],
                    "reason": "operator_claim_exclusion", "evidence": "", "source_reservation": "",
                    "decision_time": rule.get("created_at") or "", "offer_block": projection["effective_block"]}
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

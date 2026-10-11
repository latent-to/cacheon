"""Concise Discord results from the same read-only API the dashboard displays."""

from __future__ import annotations

import json
from urllib.parse import quote, urlencode
from urllib.request import urlopen


def _get(url: str) -> dict:
    with urlopen(url, timeout=15) as response:
        return json.load(response)


def result_message(base: str, arena: dict, row: dict) -> str:
    """Describe qualification separately from speed and reward eligibility."""
    verdict = row.get("decision") or row["status"].upper()
    rid, hotkey = row["reservation_id"], row["hotkey"]
    lines = [f"**Cacheon · {arena['label'][:80]} · {verdict}**",
             f"Bundle `{rid[:12]}` · miner `{hotkey[:8]}…{hotkey[-8:]}`",
             f"Target: {row['target_id'][:80]}"]
    reason = row.get("reason") or row.get("invalid_reason")
    if reason and reason != "qualified":
        lines.append("Reason: " + reason.replace("_", " ").replace("\n", " ")[:180])
    result = row.get("result")
    if result:
        measured, required = result["speedup"], result["required_speedup"]
        lines.append(f"Speed: {(measured - 1) * 100:+.2f}% · needed {(required - 1) * 100:+.2f}%"
                     f" · {result['passes']}/{result['pass_limit']} replay passes")
        for key, label, unit, places in (("decode_tps", "Decode", "tok/s", 1),
                                         ("ttft_s", "First token", "s", 2)):
            values = result.get(key) or []
            if len(values) == 2 and all(v is not None for v in values):
                lines.append(f"{label}: {values[0]:.{places}f} → {values[1]:.{places}f} {unit}")
        audit = result.get("audit")
        if audit:
            lines.append(f"Audit {audit['decision']}: {audit['detail'][:280]}")
    if verdict == "PASS":
        reward = row.get("reward") or {}
        eligible = reward.get("reward_eligible")
        lines.append("Reward: " + ("eligible; on-chain activation is reported separately"
                     if eligible is True else "not currently reward-eligible"
                     if eligible is False else "pending settlement"))
    url = base.rstrip("/") + "/" + quote(arena.get("slug") or arena["key"], safe="")
    lines.append("[Details and logs](" + url + "?" + urlencode({"submission": rid}) + ")")
    return "\n".join(lines)


def check_results(base: str, state: dict, *, notify, persist, get=_get) -> None:
    """Announce new outcomes, checkpointing each delivery across ordinary restarts.

    First observation of an arena seeds its existing results without flooding
    Discord. Unfinished submissions are not seeded: their later result is new.
    A failed send leaves that result unacknowledged for the next poll.
    """
    base = base.rstrip("/")
    known = state.setdefault("bundle_results", {})
    for arena in get(base + "/api/arenas")["items"]:
        items, offset = [], 0
        while True:
            page = get(base + "/api/submissions?" + urlencode({
                "arena": arena["key"], "order": "asc", "limit": 1000, "offset": offset,
            }))
            items.extend(page["items"])
            offset += len(page["items"])
            if not page["items"] or offset >= page["total"]:
                break
        first = arena["key"] not in known
        seen = known.setdefault(arena["key"], {})
        for row in items:
            if (row.get("decision") not in {"PASS", "FAIL", "HOLD", "NO_DECISION"}
                    and row["status"] not in {"qualified", "failed", "expired", "held", "no_decision"}):
                continue
            reward = row.get("reward") or {}
            outcome = [row["status"], row.get("decision"), row.get("reason"),
                       reward.get("reward_eligible")]
            rid = row["reservation_id"]
            if seen.get(rid) == outcome:
                continue
            if not first:
                notify(result_message(base, arena, row))
            seen[rid] = outcome
            if not first:
                persist()
        persist()

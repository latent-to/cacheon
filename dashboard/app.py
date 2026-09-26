#!/usr/bin/env python3
"""Cacheon submissions API + dashboard (netuid 14).

Read-only over the live intake SQLite. Never writes the intake DB; keeps its
own enrichment cache (block timestamps, extrinsic signers, metagraph) in a
separate SQLite file so chain lookups survive restarts.

Run with:  /root/miniconda3/envs/prod/bin/python -m dashboard.app
           (see run.sh / bin/cacheon-dashboard)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response

from dashboard.forensics import (
    DashboardForensicsError,
    submission_forensics,
    submission_qualifications,
)
from dashboard.enrichment import Enrichment
from dashboard.sources import selected, value, install_sources, process_matches, scope_reservations
from dashboard.disclosure import disclose_bundle, install_disclosure_routes
from dashboard.competition import competition_label, submission_baseline, target_summary
from cacheon.chain.baseline_band import qualification_evidence_roots, qualification_speed
from cacheon.chain.eval_cost import PUBLISHED_EVAL_COST_TAO_RAO
from dashboard.receipts import evaluation_recovery, screen_stages
from dashboard.winners import (
    conservative_candidate_tokens_per_second,
    measured_baseline,
    prefill_summary,
    settlement_hold_notice, settlement_label,
    reward_exclusion_notice,
    live_offer_shares, winner_reward,
)

MISSION = Path(os.environ.get(
    "CACHEON_DASH_MISSION", "/data/mainnet14-cacheon-h3-m4i-pre-crown"))
DB_PATH = Path(os.environ.get(
    "CACHEON_DASH_DB", str(MISSION / "state" / "intake.sqlite3")))
AUDIT_PATH = MISSION / "state" / "chain-audit.jsonl"
SPOOL = Path(os.environ.get(
    "CACHEON_DASH_SPOOL", "/root/cacheon-ops/remote-worker/spool"))
HEARTBEAT_PATH = SPOOL / "state" / "heartbeat.json"
REGISTRATION_PATH = Path(
    "/root/cacheon-ops/remote-worker/state/current-registration.json")
LOG_ROOT = Path("/root/cacheon-ops/logs")
# Qualification stage-exit artifacts live in per-generation evidence stores;
# these are every place a graded attempt's artifact may still be retained.
QUAL_EVIDENCE_STATE = Path("/root/cacheon-ops/remote-worker/state")
QUAL_EVIDENCE_EXTRA = (
    Path("/root/cacheon-ops/remote-worker/standing-qual-evidence"),
    MISSION / "evidence",
)

# The weight offer this validator serves to every follower, and the follower
# journal that records what this validator itself submitted and confirmed.
OFFER_PATH = Path(os.environ.get(
    "CACHEON_DASH_OFFER", "/var/lib/cacheon/current_weights.json"))
FOLLOW_JOURNAL = os.environ.get("CACHEON_DASH_FOLLOW_JOURNAL", "")

NETWORK = os.environ.get(
    "CACHEON_DASH_NETWORK", "wss://archive.sub.latent.to")
NETUID = int(os.environ.get("CACHEON_DASH_NETUID", "14"))
ENRICH = os.environ.get("CACHEON_DASH_ENRICH", "1") not in ("0", "false", "")

CACHE_DIR = Path(__file__).resolve().parent / "state"
CACHE_DB = CACHE_DIR / "enrichment.sqlite3"
STATIC_DIR = Path(__file__).resolve().parent / "static"

BLOCK_SECONDS = 12

TAO_APP = "https://www.tao.app"

# The dashboard ignores everything submitted before this reservation
# (first paid-era submission, block 8879318). Set to "" to show all history.
CUTOFF_RESERVATION = os.environ.get(
    "CACHEON_DASH_CUTOFF_RESERVATION",
    "bc181c82233335602cc9d07fb3827a8deb37d6fee97fd37183a1d251f9f94593")

ACTIVE_STATUSES = (
    "deferred", "reserved", "fetching", "transport_retry", "published",
    "screening", "promoted", "qualifying", "reproduction_pending",
)
TERMINAL_STATUSES = ("failed", "expired", "qualified")

# Queue stage order used to compute a submission's pipeline progress.
STAGE_ORDER = {
    "deferred": 0, "reserved": 1, "fetching": 2, "transport_retry": 2,
    "published": 3, "screening": 4, "promoted": 5, "qualifying": 6,
    "reproduction_pending": 7,
    "held": 8, "no_decision": 8,
    "qualified": 9, "failed": 9, "expired": 9,
}

# ------------------------------------------------------------- db access ---

def intake_conn() -> sqlite3.Connection:
    """Open the selected live intake DB read-only with current WAL contents."""
    con = sqlite3.connect(value("DB_PATH", DB_PATH).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return scope_reservations(con)


def rows(con: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in con.execute(sql, args)]


ENRICHER = Enrichment(CACHE_DB, NETWORK, NETUID)

# ------------------------------------------------------------ helpers ------


def links_for_block(block: int) -> dict[str, str]:
    return {"tao_app": f"{TAO_APP}/block/{block}"} if block else {}


def links_for_extrinsic(block: int, idx: int) -> dict[str, str]:
    if not block:
        return {}
    return {
        "tao_app": f"{TAO_APP}/blocks/{block}/extrinsics/{idx}",
        "taostats": f"https://taostats.io/extrinsic/{block}-{idx:04d}",
    }


def links_for_address(addr: str) -> dict[str, str]:
    return {"tao_app": f"{TAO_APP}/portfolio/{addr}"} if addr else {}


def with_time(block: int) -> dict[str, Any]:
    bt = value("ENRICHER", ENRICHER).block_time(block)
    return {"block": block, "time_unix": bt["unix"], "time_estimated": bt["estimated"]}


def emission_symbol() -> str:
    return str((value("ENRICHER", ENRICHER).metagraph or {}).get("emission_symbol") or "")


def current_offer(*, submissions=False) -> tuple[dict[str, Any] | None, dict[str, Decimal]]:
    """The served weight offer with a clock, or ``(None, {})`` when absent."""
    roots = [path.parent / "weight-allocation-evidence" for path in (
        DB_PATH, value("DB_PATH", DB_PATH), *value("PEER_DB_PATHS", ()))] if submissions else None
    summary, shares = live_offer_shares(OFFER_PATH, submission_roots=roots,
        submission_path=os.environ.get("CACHEON_DASH_SUBMISSION_SHARES") if submissions else None)
    if summary is None:
        return None, {}
    try:
        served_unix = int(OFFER_PATH.stat().st_mtime)
    except OSError:
        served_unix = None
    summary["effective"] = with_time(summary["effective_block"])
    summary["served_unix"] = served_unix
    return summary, shares


def share_value(shares: dict[str, Decimal], hotkey: str) -> float | None:
    if selected.get() is not None and not selected.get().weights_included:
        return None
    share = shares.get(hotkey)
    return float(share) if share is not None else None


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def cutoff_block(con: sqlite3.Connection) -> int:
    """Resolve the selected source's retained cutoff without sharing cached state."""
    cutoff = value("CUTOFF_RESERVATION", CUTOFF_RESERVATION)
    row = con.execute("SELECT block FROM reservations WHERE reservation_id=?", (cutoff,)).fetchone()
    return int(row[0]) if row else 0


def finalized_tip_from_audit() -> dict[str, Any]:
    """Last finalized block the intake process observed (freshness signal)."""
    if not value("AUDIT_PATH", AUDIT_PATH).exists():
        return {}
    try:
        with value("AUDIT_PATH", AUDIT_PATH).open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16000))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return {}
    mtime = int(value("AUDIT_PATH", AUDIT_PATH).stat().st_mtime)
    for line in reversed(tail.strip().splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("finalized_block") is not None:
            return {"block": int(row["finalized_block"]), "audit_mtime_unix": mtime}
    return {"audit_mtime_unix": mtime}


def supervisor_status(epoch: str) -> dict[str, Any]:
    if not epoch:
        return {}
    log = value("LOG_ROOT", LOG_ROOT) / f"mainnet-standing-supervisor-{epoch}.log"
    if not log.exists():
        return {}
    try:
        with log.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 8000))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return {}
    last: dict[str, Any] | None = None
    for line in tail.strip().splitlines():
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    if not last:
        return {}
    return {
        "phase": last.get("phase"),
        "last_stage": last.get("last_stage"),
        "last_disposition": last.get("last_disposition"),
        "hold_reason": last.get("hold_reason"),
        "request_id": last.get("request_id"),
        "lease_id": last.get("lease_id"),
        "time_unix": last.get("time_unix"),
    }


def spool_requests() -> list[dict[str, Any]]:
    """Outstanding GPU work requests in the CPU→GPU spool (wall-clock view)."""
    out: list[dict[str, Any]] = []
    outbox = value("SPOOL", SPOOL) / "outbox"
    if not outbox.is_dir():
        return out
    for entry in sorted(outbox.iterdir()):
        if not entry.is_dir():
            continue
        req = load_json(entry / "request.json")
        match = re.match(r"^0?(\d{10})\d*-([0-9a-f]+)$", entry.name)
        queued_unix = int(match.group(1)) if match else None
        out.append({
            "name": entry.name,
            "request_id": (match.group(2) if match else entry.name)[:16],
            "queued_unix": req.get("created_at_unix") or queued_unix,
            "deadline_unix": req.get("deadline_unix"),
            "kind": req.get("kind") or req.get("stage") or "",
        })
    return out


def submission_row(r: dict[str, Any]) -> dict[str, Any]:
    """Shape one reservations row for the API."""
    paid_block = int(r.get("eval_cost_payment_block") or 0)
    paid_idx = int(r.get("eval_cost_payment_extrinsic_index") or 0)
    sub = {
        "reservation_id": r["reservation_id"],
        "status": r["status"],
        "decision": r.get("decision") or "",
        "reason": r.get("reason") or "",
        "invalid_reason": r.get("invalid_reason") or "",
        "hotkey": r["hotkey"],
        "hotkey_links": links_for_address(r["hotkey"]),
        "content_hash": r["content_hash"],
        "target_id": r.get("target_id") or "",
        "target_summary": target_summary(r.get("target_id") or ""),
        "submitted": with_time(int(r["block"])),
        "block_links": links_for_block(int(r["block"])),
        "event_index": r["event_index"],
        "admission_epoch": r["admission_epoch"],
        "competition": competition_label(r["block"], r.get("competition_arena", "")),
        "screen_lane": r.get("screen_lane") or "",
        "screen_status": r.get("screen_status") or "",
        "screen_attempts": r.get("screen_attempts") or 0,
        "transport_attempts": r.get("transport_attempts") or 0,
        "retry_position": r.get("retry_position") or 0,
        "stage_order": STAGE_ORDER.get(str(r["status"]), 0),
        "is_active": r["status"] in ACTIVE_STATUSES,
        "is_terminal": r["status"] in TERMINAL_STATUSES,
        "payment": None,
    }
    if paid_block:
        sub["payment"] = {
            **with_time(paid_block),
            "extrinsic_index": paid_idx,
            "ref": f"{paid_block}-{paid_idx}",
            "links": links_for_extrinsic(paid_block, paid_idx),
        }
    return sub


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


app = FastAPI(
    title="Cacheon submissions API",
    description="Read-only API over the netuid-14 intake database, with "
                "best-effort chain enrichment (payment coldkeys, block times, "
                "metagraph emissions).",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.get("/icon-192.png", include_in_schema=False)
def brand_icon() -> FileResponse:
    return FileResponse(STATIC_DIR / "icon-192.png")


@app.get("/api/health")
def health() -> dict[str, Any]:
    db_ok = True
    try:
        con = intake_conn()
        con.execute("SELECT count(*) FROM reservations").fetchone()
        con.close()
    except sqlite3.Error:
        db_ok = False
    audit = finalized_tip_from_audit()
    heartbeat = load_json(value("HEARTBEAT_PATH", HEARTBEAT_PATH))
    hb_age = None
    if isinstance(heartbeat.get("time_unix"), int):
        hb_age = max(0, int(time.time()) - heartbeat["time_unix"])
    registration = load_json(value("REGISTRATION_PATH", REGISTRATION_PATH))
    epoch = str(registration.get("worker_epoch") or heartbeat.get("worker_epoch") or "")
    tip = dict(value("ENRICHER", ENRICHER).tip)
    lag_blocks = None
    if tip.get("block") and audit.get("block"):
        lag_blocks = int(tip["block"]) - int(audit["block"])
    # Deliberately no filesystem paths, hosts, RPC endpoints, or raw error
    # strings here: this API is public-facing. Details go to server logs.
    return {
        "now_unix": int(time.time()),
        "db": {"ok": db_ok},
        "chain": {
            "ok": value("ENRICHER", ENRICHER).chain_ok, "netuid": NETUID, "tip": tip,
            "metagraph_age_s": (int(time.time() - value("ENRICHER", ENRICHER).metagraph["fetched_at"])
                                if value("ENRICHER", ENRICHER).metagraph.get("fetched_at") else None),
        },
        "intake_finalized": audit,
        "intake_lag_blocks": lag_blocks,
        "processes": [
            {"name": "intake (chain-validate)",
             "up": process_matches("intake", ("chain-validate", "--intake-only", str(value("DB_PATH", DB_PATH))))},
            {"name": "standing supervisor",
             "up": process_matches("supervisor", ("cacheon.chain.standing_cpu_supervisor", str(value("REGISTRATION_PATH", REGISTRATION_PATH).parent)))},
            {"name": "CPU spool relay",
             "up": process_matches("relay", ("cpu-serve", str(value("SPOOL", SPOOL))))},
        ],
        "gpu_heartbeat": {
            "state": ("unknown" if hb_age is None else "stale" if hb_age > 120 else
                      "epoch_mismatch" if heartbeat.get("worker_epoch") != epoch else heartbeat.get("state")),
            "reported_state": heartbeat.get("state"),
            "age_s": hb_age,
            "adapter_alive": bool(heartbeat.get("adapter_alive")),
            "active_request_id": heartbeat.get("active_request_id"),
            "worker_epoch": heartbeat.get("worker_epoch"),
        },
        "worker_epoch": epoch,
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    con = intake_conn()
    cutoff = cutoff_block(con)
    counts = Counter(
        str(r["status"]) for r in con.execute(
            "SELECT status FROM reservations WHERE block >= ?", (cutoff,)))
    for key in (*ACTIVE_STATUSES, *TERMINAL_STATUSES, "held", "no_decision"):
        counts.setdefault(key, 0)

    fail_reasons = rows(con, """
        SELECT reason, count(*) AS n FROM reservations
        WHERE status IN ('failed','expired') AND block >= ?
        GROUP BY reason ORDER BY n DESC LIMIT 12
    """, (cutoff,))
    settlement = {r["status"]: r["n"] for r in rows(con, """
        SELECT sc.status, count(*) AS n
        FROM settlement_candidates sc
        JOIN reservations r ON r.reservation_id = sc.reservation_id
        WHERE r.block >= ? GROUP BY sc.status
    """, (cutoff,))}
    claims = {r["status"]: r["n"] for r in rows(
        con, "SELECT status, count(*) AS n FROM standing_reward_claims GROUP BY status")}
    payments = rows(con, "SELECT payment_block, amount_tao_rao FROM eval_cost_payments")
    hotkey_count = con.execute(
        "SELECT count(DISTINCT hotkey) FROM reservations WHERE block >= ?",
        (cutoff,)).fetchone()[0]
    blocks = [int(r["block"]) for r in con.execute(
        "SELECT block FROM reservations WHERE block >= ?", (cutoff,))]
    active_leases = con.execute(
        "SELECT count(*) FROM evaluation_leases WHERE state='active'").fetchone()[0]
    con.close()

    # Submissions per day (block time estimates are fine for a trend chart).
    per_day: Counter[str] = Counter()
    for block in blocks:
        unix = value("ENRICHER", ENRICHER).block_time(block)["unix"]
        if unix:
            per_day[datetime.fromtimestamp(unix, timezone.utc).strftime("%Y-%m-%d")] += 1

    active_total = sum(counts.get(s, 0) for s in ACTIVE_STATUSES)
    return {
        "counts": dict(counts),
        "totals": {
            "submissions": sum(counts.values()),
            "active": active_total,
            "held": counts.get("held", 0),
            "qualified": counts.get("qualified", 0),
            "failed": counts.get("failed", 0),
            "expired": counts.get("expired", 0),
            "unique_hotkeys": hotkey_count,
            "active_leases": active_leases,
            "payments_count": len(payments),
            "payments_tao": sum(int(p["amount_tao_rao"]) for p in payments) / 1e9,
            "crowned": settlement.get("crowned", 0),
            "settlement_held": settlement.get("held", 0),
            "claims": claims,
        },
        "failure_reasons": fail_reasons,
        "submissions_per_day": [
            {"day": d, "count": n} for d, n in sorted(per_day.items())],
        "tip": dict(value("ENRICHER", ENRICHER).tip),
    }


@app.get("/api/submissions")
def submissions(
    status: str | None = None,
    hotkey: str | None = None,
    q: str | None = None,
    active: bool | None = None,
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    where, args = [], []
    if status:
        placeholders = ",".join("?" * len(status.split(",")))
        where.append(f"status IN ({placeholders})")
        args.extend(status.split(","))
    if hotkey:
        where.append("hotkey = ?")
        args.append(hotkey)
    if active is True:
        where.append(f"status IN ({','.join('?' * len(ACTIVE_STATUSES))})")
        args.extend(ACTIVE_STATUSES)
    elif active is False:
        where.append(f"status NOT IN ({','.join('?' * len(ACTIVE_STATUSES))})")
        args.extend(ACTIVE_STATUSES)
    if q:
        where.append("(reservation_id LIKE ? OR hotkey LIKE ? OR content_hash LIKE ?"
                     " OR target_id LIKE ? OR reason LIKE ?)")
        args.extend([f"%{q}%"] * 5)
    con = intake_conn()
    where.append("block >= ?")
    args.append(cutoff_block(con))
    clause = f"WHERE {' AND '.join(where)}"
    total = con.execute(
        f"SELECT count(*) FROM reservations {clause}", args).fetchone()[0]
    data = rows(con, f"""
        SELECT *
        FROM reservations {clause}
        ORDER BY block {'ASC' if order == 'asc' else 'DESC'}, event_index
        LIMIT ? OFFSET ?
    """, (*args, limit, offset))
    shaped = [submission_row(r) for r in data]
    lineage_tables = con.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name IN "
        "('target_lineage_tips','target_lineage_nodes')"
    ).fetchone()[0] == 2
    for item in shaped:
        item["evaluation_recovery"] = evaluation_recovery(con, item)
        item["baseline"] = submission_baseline(
            con,
            item["reservation_id"],
            item["target_id"],
            lineage_tables_available=lineage_tables,
        )
    con.close()
    return {
        "total": total, "limit": limit, "offset": offset,
        "items": shaped,
    }


@app.get("/api/submissions/{reservation_id}")
def submission_detail(reservation_id: str, response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    con = intake_conn()
    row = con.execute(
        "SELECT * FROM reservations WHERE reservation_id = ? OR reservation_id LIKE ?",
        (reservation_id, f"{reservation_id}%")).fetchone()
    if not row:
        con.close()
        raise HTTPException(404, "reservation not found")
    r = dict(row)
    rid = r["reservation_id"]
    detail = submission_row(r)
    detail["evaluation_recovery"] = evaluation_recovery(con, detail)
    detail["payload_digest"] = r.get("payload_digest") or ""
    detail["publication_digest"] = r.get("publication_digest") or ""
    detail["block_hash"] = r.get("block_hash") or ""

    detail["screen_attempts_history"] = [
        {"attempt": d["attempt_index"], "decision": d["decision"],
         "lane": d["lane"], "stage_count": d["stage_count"],
         "stages": screen_stages(d["receipt_json"])}
        for d in rows(con, """
            SELECT attempt_index, decision, lane, stage_count, receipt_json
            FROM arena_screen_dispositions WHERE reservation_id=? ORDER BY attempt_index
        """, (rid,))]
    evidence_roots = qualification_evidence_roots(
        value("QUAL_EVIDENCE_STATE", QUAL_EVIDENCE_STATE), value("QUAL_EVIDENCE_EXTRA", QUAL_EVIDENCE_EXTRA), con, stage_dir=value("STAGE_ROOT", LOG_ROOT.parent / "stage"))
    try:
        detail["forensics"] = submission_forensics(value("SPOOL", SPOOL), rid, target_id=detail["target_id"])
    except DashboardForensicsError as exc:
        detail["forensics"] = []
        detail["forensics_error"] = str(exc)
    detail["qualification_attempts"] = submission_qualifications(
        con, rid, detail["target_id"], evidence_roots, detail["forensics"])

    cand = con.execute(
        "SELECT status, reason, candidate_json FROM settlement_candidates"
        " WHERE reservation_id=?", (rid,)).fetchone()
    if cand:
        cj = json.loads(cand["candidate_json"] or "{}")
        primary = cj.get("primary") or {}
        repro = cj.get("reproduction") or {}
        detail["settlement"] = {
            "status": cand["status"],
            "reason": cand["reason"],
            "speedup_primary": safe_float(primary.get("speedup")),
            "speedup_reproduction": safe_float(repro.get("speedup")),
            "lane": cj.get("lane"),
            "crowned": with_time(int(cj.get("finalized_block") or 0)),
        }
        from dashboard.winners import submission_reward_comparison
        detail["settlement"].update(submission_reward_comparison(con, rid))
    detail["reward_notice"] = reward_exclusion_notice(r["hotkey"], OFFER_PATH)
    detail["hold_notice"] = settlement_hold_notice(con, rid, detail.get("settlement", {}))
    detail["baseline"] = submission_baseline(con, rid, detail["target_id"])
    measured_attempts = [a for a in detail["qualification_attempts"] if a["decision"] == "PASS"] or detail["qualification_attempts"]
    speed_reads = [a["speed"] for a in measured_attempts if a["speed"]]
    detail["baseline_measurements"] = measured_baseline(speed_reads, {})
    candidate_tps = conservative_candidate_tokens_per_second(speed_reads)
    detail["tokens_per_second"] = float(candidate_tps) if candidate_tps is not None else None

    detail["leases"] = rows(con, """
        SELECT el.lease_id, el.stage, el.state, el.generation, el.claimed_block,
               el.expires_block, el.completed_block, el.reason
        FROM evaluation_lease_members m
        JOIN evaluation_leases el ON el.lease_id = m.lease_id
        WHERE m.reservation_id = ? ORDER BY el.claimed_block DESC
    """, (rid,))
    for lease in detail["leases"]:
        lease["claimed"] = with_time(int(lease["claimed_block"]))
        lease["expires"] = with_time(int(lease["expires_block"]))
    disclose_bundle(con, detail, value("ENRICHER", ENRICHER).block_time)
    con.close()
    return detail


install_disclosure_routes(app, intake_conn, lambda block: value("ENRICHER", ENRICHER).block_time(block),
                          lambda: value("MISSION", MISSION) / "private", lambda: value("SPOOL", SPOOL))


@app.get("/api/queue")
def queue() -> dict[str, Any]:
    con = intake_conn()
    cutoff = cutoff_block(con)
    pending = rows(con, f"""
        SELECT *
        FROM reservations
        WHERE (status IN ({','.join('?' * len(ACTIVE_STATUSES))}) OR status='held')
              AND block >= ?
        ORDER BY block ASC, event_index ASC
    """, (*ACTIVE_STATUSES, cutoff))
    leases = rows(con, """
        SELECT el.lease_id, el.generation, el.stage, el.state, el.owner,
               el.claimed_block, el.initial_expires_block, el.expires_block,
               el.completed_block, el.reason,
               m.reservation_id, r.target_id, r.hotkey, r.status AS reservation_status
        FROM evaluation_leases el
        JOIN evaluation_lease_members m ON m.lease_id = el.lease_id AND m.active = 1
        JOIN reservations r ON r.reservation_id = m.reservation_id
        WHERE el.state = 'active' AND r.block >= ?
        ORDER BY el.claimed_block DESC
    """, (cutoff,))
    recent_leases = rows(con, """
        SELECT lease_id, stage, state, claimed_block, expires_block,
               completed_block, reason
        FROM evaluation_leases WHERE state != 'active'
        ORDER BY claimed_block DESC LIMIT 15
    """)
    con.close()

    now = int(time.time())
    tip = dict(value("ENRICHER", ENRICHER).tip)
    items = []
    for pos, r in enumerate(pending, start=1):
        item = submission_row(r)
        item["queue_position"] = pos
        item["waiting_seconds"] = (
            now - item["submitted"]["time_unix"]
            if item["submitted"]["time_unix"] else None)
        items.append(item)

    active = []
    for lease in leases:
        claimed = with_time(int(lease["claimed_block"]))
        expires = with_time(int(lease["expires_block"]))
        remaining_blocks = (
            int(lease["expires_block"]) - int(tip["block"]) if tip.get("block") else None)
        active.append({
            "lease_id": lease["lease_id"],
            "stage": lease["stage"],
            "generation": lease["generation"],
            "owner": lease["owner"],
            "reservation_id": lease["reservation_id"],
            "reservation_status": lease["reservation_status"],
            "hotkey": lease["hotkey"],
            "target_id": lease["target_id"] or "",
            "target_summary": target_summary(lease["target_id"] or ""),
            "claimed": claimed,
            "expires": expires,
            "running_seconds": now - claimed["time_unix"] if claimed["time_unix"] else None,
            "remaining_blocks": remaining_blocks,
            "remaining_seconds": (
                remaining_blocks * BLOCK_SECONDS if remaining_blocks is not None else None),
        })
    for lease in recent_leases:
        lease["claimed"] = with_time(int(lease["claimed_block"]))

    heartbeat = load_json(value("HEARTBEAT_PATH", HEARTBEAT_PATH))
    hb_age = None
    if isinstance(heartbeat.get("time_unix"), int):
        hb_age = max(0, now - heartbeat["time_unix"])
    registration = load_json(value("REGISTRATION_PATH", REGISTRATION_PATH))
    epoch = str(registration.get("worker_epoch") or heartbeat.get("worker_epoch") or "")

    return {
        "now_unix": now,
        "pending": items,
        "active_leases": active,
        "recent_leases": recent_leases,
        "gpu_requests": spool_requests(),
        "supervisor": supervisor_status(epoch),
        "gpu_heartbeat": {
            "state": ("unknown" if hb_age is None else "stale" if hb_age > 120 else
                      "epoch_mismatch" if heartbeat.get("worker_epoch") != epoch else heartbeat.get("state")),
            "reported_state": heartbeat.get("state"), "age_s": hb_age,
            "adapter_alive": bool(heartbeat.get("adapter_alive")),
            "active_request_id": heartbeat.get("active_request_id"),
        },
        "tip": tip,
    }


@app.get("/api/payments")
def payments() -> dict[str, Any]:
    con = intake_conn()
    pays = rows(con, """
        SELECT p.payment_block, p.payment_extrinsic_index, p.reservation_id,
               p.content_hash, p.hotkey, p.amount_tao_rao,
               r.status AS reservation_status, r.target_id, r.reason,
               r.block AS reservation_block,
               r.eval_cost_payment_block AS applied_block
        FROM eval_cost_payments p
        LEFT JOIN reservations r ON r.reservation_id = p.reservation_id
        ORDER BY p.payment_block DESC
    """)
    credits = rows(con, """
        SELECT credit_id, hotkey, coldkey, amount_tao_rao, note, granted_at,
               reservation_id, spent_block
        FROM eval_cost_credits ORDER BY granted_at DESC
    """)
    con.close()

    owner = (value("ENRICHER", ENRICHER).metagraph or {}).get("owner_coldkey") or ""
    items = []
    for p in pays:
        block = int(p["payment_block"])
        idx = int(p["payment_extrinsic_index"])
        signer = value("ENRICHER", ENRICHER).extrinsic_signer(block, idx)
        applied = int(p["applied_block"] or 0) == block
        items.append({
            "ref": f"{block}-{idx}",
            "payment": {**with_time(block), "extrinsic_index": idx},
            "links": links_for_extrinsic(block, idx),
            "amount_tao": int(p["amount_tao_rao"]) / 1e9,
            "coldkey": signer["signer"],
            "coldkey_links": links_for_address(signer["signer"] or ""),
            "extrinsic_call": signer["call"],
            "hotkey": p["hotkey"],
            "hotkey_links": links_for_address(p["hotkey"]),
            "reservation_id": p["reservation_id"],
            "reservation_status": p["reservation_status"],
            "target_id": p["target_id"] or "",
            "applied": applied,
            "consumed": True,  # a row here means intake admitted & consumed it
            "outcome": p["reservation_status"],
            "outcome_reason": p["reason"] or "",
        })
    for c in credits:
        c["amount_tao"] = int(c["amount_tao_rao"]) / 1e9
        c["spent"] = bool(c["reservation_id"])
    return {
        "eval_cost_tao": PUBLISHED_EVAL_COST_TAO_RAO / 1e9,
        "destination_coldkey": owner,
        "destination_links": links_for_address(owner),
        "items": items,
        "credits": credits,
    }


@app.get("/api/winners")
def winners() -> dict[str, Any]:
    from dashboard.winners import qualified_winners, winner_lists
    con = intake_conn()
    passed = qualified_winners(con, include_waiting=True)
    evidence_roots = qualification_evidence_roots(
        value("QUAL_EVIDENCE_STATE", QUAL_EVIDENCE_STATE), value("QUAL_EVIDENCE_EXTRA", QUAL_EVIDENCE_EXTRA), con, stage_dir=value("STAGE_ROOT", LOG_ROOT.parent / "stage"))
    speeds_by_reservation: dict[str, list[object]] = {}
    if passed:
        marks = ",".join("?" for _ in passed)
        for disposition in rows(con, f"""
            SELECT d.reservation_id, d.attempt_ref_json, r.target_id
            FROM qualification_dispositions d
            JOIN reservations r ON r.reservation_id=d.reservation_id
            WHERE d.decision='PASS' AND d.reservation_id IN ({marks})
            ORDER BY d.reservation_id, d.attempt_index
        """, tuple(row["reservation_id"] for row in passed)):
            speed = qualification_speed(
                disposition["attempt_ref_json"], evidence_roots, disposition["target_id"])
            if speed is None:
                continue
            speeds_by_reservation.setdefault(
                disposition["reservation_id"], []).append(speed)
    labels = {row["reservation_id"]: settlement_label(con, row["reservation_id"], row["status"], row["reason"]) for row in passed}
    con.close()
    offer, shares = current_offer(submissions=True)

    items = []
    for row in passed:
        cj = json.loads(row["candidate_json"] or "{}")
        primary = cj.get("primary") or {}
        repro = cj.get("reproduction") or {}
        target = primary.get("target_id") or cj.get("target_id") or ""
        speeds = tuple(filter(None, (safe_float(p.get("speedup")) for p in (primary, repro))))
        speedup = min(speeds) if speeds else None
        candidate_tps = conservative_candidate_tokens_per_second(
            speeds_by_reservation.get(row["reservation_id"], []))
        hotkey = row["hotkey"]
        hk = value("ENRICHER", ENRICHER).hotkey_info(hotkey)
        passed_block = max(int(row["passed_block"] or 0), int(row["submission_block"]))
        items.append({
            "reservation_id": row["reservation_id"],
            "hotkey": hotkey,
            "hotkey_links": links_for_address(hotkey),
            "target_id": target,
            "target_summary": target_summary(target),
            "speedup": speedup,
            "improvement_pct": (speedup - 1) * 100 if speedup else None,
            **{key: row.get(key) for key in ("previous_best_reservation_id", "previous_best_speedup",
                "relative_improvement_pct", "score_improvement_pct", "reward_eligible", "grandfathered")},
            "waiting_for_queue": bool(row["waiting_for_queue"]),
            "speedup_primary": safe_float(primary.get("speedup")),
            "speedup_reproduction": safe_float(repro.get("speedup")),
            "tokens_per_second": round(float(candidate_tps), 1) if candidate_tps is not None else None,
            "passed": with_time(passed_block),
            "passed_links": links_for_block(passed_block),
            "submitted": with_time(int(row["submission_block"])),
            "competition": competition_label(row["submission_block"], row.get("competition_arena", "")),
            **measured_baseline(speeds_by_reservation.get(row["reservation_id"], []), primary),
            **prefill_summary(speeds_by_reservation.get(row["reservation_id"], [])),
            **winner_reward(row, offer, shares),
            "settlement_status": labels[row["reservation_id"]],
            "hotkey_chain": {
                "registered": hk.get("registered", False),
                "uid": hk.get("uid"),
                "coldkey": hk.get("coldkey"),
                "emission_alpha_per_day": hk.get("emission_alpha_per_day"),
                "incentive": hk.get("incentive"),
                "stake_alpha": hk.get("stake_alpha"),
                "metagraph_block": hk.get("metagraph_block"),
            },
        })
    items.sort(key=lambda x: x["passed"]["block"] or 0, reverse=True)
    return {
        **winner_lists(items),
        "emission_symbol": emission_symbol(),
        "offer": offer,
        "note": (
            "A complete audited PASS earns after clearing the previous best by the configured "
            "margin; grandfathered runtimes retain prior eligibility. Weight share is this submission's "
            "portion of the currently served offer; unavailable breakdowns show a dash. The chain reflects it only after commit-reveal "
            "and stake-weighted consensus across validators, so on-chain emission lags."
            if offer is not None
            else "The served weight offer is unavailable; weight shares cannot be shown."
        ),
    }


@app.get("/api/miners")
def miners() -> dict[str, Any]:
    con = intake_conn()
    cutoff = cutoff_block(con)
    data = rows(con, """
        SELECT hotkey,
               count(*) AS submissions,
               sum(status='qualified') AS qualified,
               sum(status='failed') AS failed,
               sum(status='expired') AS expired,
               sum(status='held') AS held,
               sum(eval_cost_payment_block > 0) AS paid,
               min(block) AS first_block,
               max(block) AS last_block
        FROM reservations WHERE block >= ?
        GROUP BY hotkey ORDER BY submissions DESC
    """, (cutoff,))
    crowned_by_hotkey = {r["hotkey"]: r["n"] for r in rows(con, """
        SELECT r.hotkey, count(*) AS n
        FROM settlement_candidates sc
        JOIN reservations r ON r.reservation_id = sc.reservation_id
        WHERE sc.status='crowned' AND r.block >= ? GROUP BY r.hotkey
    """, (cutoff,))}
    con.close()
    offer, shares = current_offer()
    items = []
    for m in data:
        hk = value("ENRICHER", ENRICHER).hotkey_info(m["hotkey"])
        active = m["submissions"] - (m["qualified"] or 0) - (m["failed"] or 0) \
            - (m["expired"] or 0) - (m["held"] or 0)
        items.append({
            **m,
            "active": active,
            "crowned": crowned_by_hotkey.get(m["hotkey"], 0),
            "weight_share": share_value(shares, m["hotkey"]),
            "hotkey_links": links_for_address(m["hotkey"]),
            "first_seen": with_time(int(m["first_block"])),
            "last_seen": with_time(int(m["last_block"])),
            "registered": hk.get("registered", False),
            "uid": hk.get("uid"),
            "emission_alpha_per_day": hk.get("emission_alpha_per_day"),
        })
    items.sort(key=lambda x: (-(x["weight_share"] or 0), -x["submissions"]))
    return {"items": items, "emission_symbol": emission_symbol(), "offer": offer}


@app.get("/api/events")
def events(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    con = intake_conn()
    cutoff = cutoff_block(con)
    data = rows(con, """
        SELECT e.sequence, e.event_type, e.reservation_id, e.target_id, e.event_json
        FROM settlement_events e
        LEFT JOIN reservations r ON r.reservation_id = e.reservation_id
        WHERE r.reservation_id IS NULL OR r.block >= ?
        ORDER BY e.sequence DESC LIMIT ?
    """, (cutoff, limit))
    con.close()
    items = []
    for e in data:
        ej = json.loads(e["event_json"] or "{}")
        block = int(ej.get("finalized_block") or ej.get("crowned_block") or 0)
        items.append({
            "sequence": e["sequence"],
            "event_type": e["event_type"],
            "reservation_id": e["reservation_id"],
            "target_id": e["target_id"],
            "target_summary": target_summary(e["target_id"]),
            "reason": ej.get("reason") or "",
            "when": with_time(block) if block else None,
        })
    return {"items": items}


@app.get("/api/weights")
def weights(limit: int = Query(30, ge=1, le=500)) -> dict[str, Any]:
    """The served offer's vector and this validator's follower journal."""

    offer, shares = current_offer()
    vector = []
    for hotkey, share in sorted(shares.items(), key=lambda kv: -kv[1]):
        hk = value("ENRICHER", ENRICHER).hotkey_info(hotkey)
        vector.append({
            "hotkey": hotkey,
            "hotkey_links": links_for_address(hotkey),
            "uid": hk.get("uid"),
            "registered": hk.get("registered", False),
            "weight_share": float(share),
            "incentive": hk.get("incentive"),
        })
    items: list[dict[str, Any]] = []
    follower_note = ""
    if not FOLLOW_JOURNAL:
        follower_note = "CACHEON_DASH_FOLLOW_JOURNAL is not set; the follower journal is not shown."
    else:
        try:
            con = sqlite3.connect(f"file:{FOLLOW_JOURNAL}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            data = rows(con, """
                SELECT sequence, status, updated_block, projection_digest, record_json
                FROM followed_weight_publications ORDER BY sequence DESC LIMIT ?
            """, (limit,))
            con.close()
        except sqlite3.Error as exc:
            data = []
            follower_note = f"follower journal unreadable: {exc}"
        for w in data:
            rj = json.loads(w["record_json"] or "{}")
            items.append({
                "sequence": w["sequence"],
                "status": w["status"],
                "projection_digest": w["projection_digest"],
                "updated": with_time(int(w["updated_block"])),
                "submit_block": rj.get("submit_block"),
                "confirmed_block": rj.get("confirmed_block"),
                "reason": rj.get("reason") or "",
            })
    return {
        "offer": offer,
        "vector": vector,
        "items": items,
        "follower_journal": FOLLOW_JOURNAL or None,
        "follower_note": follower_note,
    }


@app.get("/api/hotkey/{hotkey}")
def hotkey(hotkey: str) -> dict[str, Any]:
    info = value("ENRICHER", ENRICHER).hotkey_info(hotkey)
    info["links"] = links_for_address(hotkey)
    return info


@app.exception_handler(sqlite3.Error)
def _sqlite_error(_req: Any, exc: sqlite3.Error) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": f"database: {exc}"})


install_sources(app, globals())

if ENRICH:
    ENRICHER.start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("CACHEON_DASH_HOST", "127.0.0.1"),
        port=int(os.environ.get("CACHEON_DASH_PORT", "8788")),
    )

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
import threading
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
    ForensicsNotFound,
    ForensicsUnavailable,
    forensics_log,
    submission_forensics,
)
from dashboard.receipts import (
    qualification_evidence_roots,
    qualification_speed,
    screen_stages,
)
from dashboard.winners import (
    conservative_candidate_tokens_per_second,
    cumulative_crown_speedups,
    estimated_sglang_tokens_per_second,
)

# ---------------------------------------------------------------- config ---

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

NETWORK = os.environ.get(
    "CACHEON_DASH_NETWORK", "wss://archive.sub.latent.to")
NETUID = int(os.environ.get("CACHEON_DASH_NETUID", "14"))
ENRICH = os.environ.get("CACHEON_DASH_ENRICH", "1") not in ("0", "false", "")

CACHE_DIR = Path(__file__).resolve().parent / "state"
CACHE_DB = CACHE_DIR / "enrichment.sqlite3"
STATIC_DIR = Path(__file__).resolve().parent / "static"

BLOCK_SECONDS = 12
METAGRAPH_TTL = 600            # seconds between metagraph refreshes
TEMPO_BLOCKS = 360             # subnet tempo, for emission/day math

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

# Short human descriptions of the optimization targets ("which op they improved").
TARGET_SUMMARIES = {
    "activation.silu_and_mul": "SiLU-and-multiply activation kernel (SwiGLU MLP gate)",
    "norm.rmsnorm": "RMSNorm normalization kernel",
    "attention.decode": "Attention decode kernel (token generation path)",
    "attention.sdpa": "Scaled dot-product attention (radix/prefill path)",
    "attention.msa_block_score": "MSA block-sparse attention decode scoring kernel",
    "attention.msa_prefill_block_score": "MSA block-sparse attention prefill scoring kernel",
    "collective.all_reduce": "All-reduce collective (multi-GPU tensor sum)",
    "collective.ar_residual_rmsnorm": "Fused all-reduce + residual-add + RMSNorm collective",
    "collective.moe_finalize_ar_rmsnorm": "Fused MoE finalize + all-reduce + RMSNorm collective epilogue",
    "collective.moe_epilogue.v1": "Atomic MoE epilogue (AR-residual-RMSNorm + MoE finalize pair)",
    "moe.fused_experts": "Fused MoE expert GEMM dispatch kernel",
    "moe.fused_experts_reduce": "Fused MoE expert GEMM with built-in reduction collective",
}


def target_summary(target_id: str) -> str:
    if target_id in TARGET_SUMMARIES:
        return TARGET_SUMMARIES[target_id]
    if not target_id:
        return ""
    return target_id.replace("_", " ").replace(".", " › ")


# ------------------------------------------------------------- db access ---

def intake_conn() -> sqlite3.Connection:
    """Open the live intake DB read-only (WAL-aware, falls back to immutable)."""
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
        con.execute("SELECT 1 FROM metadata LIMIT 1")
    except sqlite3.Error:
        con = sqlite3.connect(f"file:{DB_PATH}?immutable=1", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def rows(con: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in con.execute(sql, args)]


# ------------------------------------------------------- enrichment cache ---

class Enrichment:
    """Chain-derived data: block timestamps, extrinsic signers, metagraph.

    All lookups are best-effort; the API serves DB truth even when the chain
    is unreachable. Results persist in a private SQLite cache.
    """

    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._substrate = None
        self._subtensor = None
        self.chain_ok = False
        self.chain_error = ""
        self.tip: dict[str, Any] = {}          # {block, unix_time}
        self.metagraph: dict[str, Any] = {}    # {fetched_at, block, hotkeys:{hk:{...}}, owner_coldkey}
        self._wanted_blocks: set[int] = set()
        self._wanted_extrinsics: set[tuple[int, int]] = set()
        con = self._cache()
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS block_times(
                block INTEGER PRIMARY KEY, unix_time INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS extrinsic_signers(
                block INTEGER NOT NULL, ext_index INTEGER NOT NULL,
                signer TEXT NOT NULL, call TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(block, ext_index));
            CREATE TABLE IF NOT EXISTS kv(
                key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        con.commit()
        con.close()
        mg = self._kv_get("metagraph")
        if mg:
            self.metagraph = mg

    def _cache(self) -> sqlite3.Connection:
        con = sqlite3.connect(CACHE_DB, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _kv_get(self, key: str) -> Any:
        con = self._cache()
        row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        con.close()
        return json.loads(row["value"]) if row else None

    def _kv_set(self, key: str, value: Any) -> None:
        con = self._cache()
        con.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        con.commit()
        con.close()

    # -- public read side (never blocks on the chain) --

    def block_time(self, block: int) -> dict[str, Any]:
        """Return {unix, estimated} for a block; exact if cached, else estimate."""
        if not block:
            return {"unix": None, "estimated": True}
        con = self._cache()
        row = con.execute(
            "SELECT unix_time FROM block_times WHERE block=?", (block,)).fetchone()
        con.close()
        if row:
            return {"unix": int(row["unix_time"]), "estimated": False}
        with self._lock:
            self._wanted_blocks.add(block)
            tip = dict(self.tip)
        if tip.get("block") and tip.get("unix_time"):
            est = int(tip["unix_time"]) - (int(tip["block"]) - block) * BLOCK_SECONDS
            return {"unix": est, "estimated": True}
        return {"unix": None, "estimated": True}

    def extrinsic_signer(self, block: int, ext_index: int) -> dict[str, Any]:
        con = self._cache()
        row = con.execute(
            "SELECT signer, call FROM extrinsic_signers WHERE block=? AND ext_index=?",
            (block, ext_index)).fetchone()
        con.close()
        if row:
            return {"signer": row["signer"], "call": row["call"]}
        with self._lock:
            self._wanted_extrinsics.add((block, ext_index))
        return {"signer": None, "call": None}

    def hotkey_info(self, hotkey: str) -> dict[str, Any]:
        mg = self.metagraph or {}
        info = (mg.get("hotkeys") or {}).get(hotkey)
        if not info:
            return {"registered": False, "metagraph_block": mg.get("block")}
        out = dict(info)
        out["registered"] = True
        out["metagraph_block"] = mg.get("block")
        return out

    # -- background worker --

    def start(self) -> None:
        threading.Thread(target=self._loop, name="enrich", daemon=True).start()

    def _connect(self) -> None:
        from async_substrate_interface.sync_substrate import SubstrateInterface
        self._substrate = SubstrateInterface(url=NETWORK)

    def _loop(self) -> None:
        while True:
            try:
                if self._substrate is None:
                    self._connect()
                self._refresh_tip()
                self._refresh_metagraph()
                self._drain_extrinsics()
                self._drain_blocks()
                self.chain_ok = True
                self.chain_error = ""
            except Exception as exc:  # noqa: BLE001 - worker must survive anything
                self.chain_ok = False
                self.chain_error = f"{type(exc).__name__}: {exc}"[:300]
                self._substrate = None
                time.sleep(10)
            time.sleep(5)

    def _refresh_tip(self) -> None:
        sub = self._substrate
        head = sub.get_chain_finalised_head()
        block = sub.get_block_number(head)
        ts = self._block_timestamp(block_hash=head)
        if block and ts:
            self.tip = {"block": int(block), "unix_time": int(ts)}
            con = self._cache()
            con.execute(
                "INSERT OR REPLACE INTO block_times(block, unix_time) VALUES(?,?)",
                (int(block), int(ts)))
            con.commit()
            con.close()

    def _block_timestamp(self, block_hash: str | None = None,
                         block_number: int | None = None) -> int | None:
        sub = self._substrate
        if block_hash is None and block_number is not None:
            block_hash = sub.get_block_hash(block_number)
        result = sub.query("Timestamp", "Now", block_hash=block_hash)
        value = getattr(result, "value", result)
        return int(value) // 1000 if value else None

    def _fetch_block(self, block: int) -> None:
        """Fetch one block: cache its timestamp and all extrinsic signers."""
        sub = self._substrate
        block_hash = sub.get_block_hash(block)
        data = sub.get_block(block_hash=block_hash)
        ts: int | None = None
        signers: list[tuple[int, int, str, str]] = []
        for idx, ext in enumerate(data.get("extrinsics") or []):
            value = getattr(ext, "value", None) or {}
            call = value.get("call") or {}
            name = f"{call.get('call_module', '')}.{call.get('call_function', '')}"
            if name == "Timestamp.set":
                for arg in call.get("call_args") or []:
                    if arg.get("name") == "now":
                        ts = int(arg.get("value")) // 1000
            address = value.get("address") or ""
            if address:
                signers.append((block, idx, str(address), name))
        con = self._cache()
        if ts:
            con.execute(
                "INSERT OR REPLACE INTO block_times(block, unix_time) VALUES(?,?)",
                (block, ts))
        for row in signers:
            con.execute(
                "INSERT OR REPLACE INTO extrinsic_signers(block, ext_index, signer, call)"
                " VALUES(?,?,?,?)", row)
        con.commit()
        con.close()

    def _drain_extrinsics(self) -> None:
        with self._lock:
            wanted = list(self._wanted_extrinsics)[:20]
        for block, _idx in wanted:
            self._fetch_block(block)
        with self._lock:
            for item in wanted:
                self._wanted_extrinsics.discard(item)

    def _drain_blocks(self) -> None:
        with self._lock:
            wanted = sorted(self._wanted_blocks, reverse=True)[:30]
        for block in wanted:
            self._fetch_block(block)
        with self._lock:
            for item in wanted:
                self._wanted_blocks.discard(item)

    def _refresh_metagraph(self) -> None:
        if self.metagraph and time.time() - self.metagraph.get("fetched_at", 0) < METAGRAPH_TTL:
            return
        import bittensor as bt
        sub = self._subtensor
        if sub is None:
            sub = bt.Subtensor(network=NETWORK)
            self._subtensor = sub
        mg = sub.metagraph(NETUID)
        # Emission is denominated in the subnet's own alpha token, not TAO. The
        # symbol is whatever this netuid registered on chain; the local
        # bittensor unit table can disagree, so never render it from there.
        symbol = (self.metagraph or {}).get("emission_symbol") \
            or str(sub.subnet(NETUID).symbol)
        hotkeys: dict[str, Any] = {}
        for uid in range(len(mg.hotkeys)):
            emission_tempo = float(mg.emission[uid])
            hotkeys[str(mg.hotkeys[uid])] = {
                "uid": uid,
                "coldkey": str(mg.coldkeys[uid]),
                "stake_alpha": float(mg.S[uid]),
                "incentive": float(mg.incentive[uid]),
                "emission_alpha_per_tempo": emission_tempo,
                "emission_alpha_per_day": emission_tempo * (86400 / (TEMPO_BLOCKS * BLOCK_SECONDS)),
                "active": bool(mg.active[uid]),
                "validator_permit": bool(mg.validator_permit[uid]),
            }
        owner = ""
        try:
            owner = str(sub.query_subtensor("SubnetOwner", params=[NETUID]))
        except Exception:  # noqa: BLE001
            pass
        self.metagraph = {
            "fetched_at": int(time.time()),
            "block": int(mg.block),
            "n": len(mg.hotkeys),
            "owner_coldkey": owner,
            "emission_symbol": symbol,
            "hotkeys": hotkeys,
        }
        self._kv_set("metagraph", self.metagraph)


ENRICHER = Enrichment()

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
    bt = ENRICHER.block_time(block)
    return {"block": block, "time_unix": bt["unix"], "time_estimated": bt["estimated"]}


def emission_symbol() -> str:
    return str((ENRICHER.metagraph or {}).get("emission_symbol") or "")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_CUTOFF_CACHE: dict[str, int] = {}


def cutoff_block(con: sqlite3.Connection) -> int:
    """Block of the dashboard cutoff reservation.

    Everything submitted before this block (pre-paid-era history: old crowns,
    the free-lane backlog, retired-arena holds) is ignored by the dashboard.
    Returns 0 (no cutoff) when the reservation is not in the DB.
    """
    if not CUTOFF_RESERVATION:
        return 0
    if "block" not in _CUTOFF_CACHE:
        row = con.execute(
            "SELECT block FROM reservations WHERE reservation_id = ?",
            (CUTOFF_RESERVATION,)).fetchone()
        if row is None:
            return 0
        _CUTOFF_CACHE["block"] = int(row[0])
    return _CUTOFF_CACHE["block"]


def finalized_tip_from_audit() -> dict[str, Any]:
    """Last finalized block the intake process observed (freshness signal)."""
    if not AUDIT_PATH.exists():
        return {}
    try:
        with AUDIT_PATH.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16000))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return {}
    mtime = int(AUDIT_PATH.stat().st_mtime)
    for line in reversed(tail.strip().splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("finalized_block") is not None:
            return {"block": int(row["finalized_block"]), "audit_mtime_unix": mtime}
    return {"audit_mtime_unix": mtime}


def proc_matches(*needles: str) -> bool:
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            raw = Path(f"/proc/{entry}/cmdline").read_bytes()
        except OSError:
            continue
        cmd = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        if all(n in cmd for n in needles):
            return True
    return False


def supervisor_status(epoch: str) -> dict[str, Any]:
    if not epoch:
        return {}
    log = LOG_ROOT / f"mainnet-standing-supervisor-{epoch}.log"
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
    outbox = SPOOL / "outbox"
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


def submission_baseline(
    con: sqlite3.Connection,
    reservation_id: str,
    target_id: str,
    *,
    lineage_tables_available: bool | None = None,
) -> dict[str, Any]:
    """Describe the baseline used and its relationship to the active tip."""

    candidate = con.execute(
        "SELECT candidate_json FROM settlement_candidates "
        "WHERE reservation_id=?",
        (reservation_id,),
    ).fetchone()
    raw: dict[str, Any] = {}
    evaluated = False
    assigned = False
    if candidate is not None:
        doc = json.loads(candidate["candidate_json"] or "{}")
        raw = doc.get("primary") or doc
        evaluated = True
    else:
        qualification = con.execute(
            "SELECT qualification_json FROM settlement_qualifications "
            "WHERE reservation_id=? ORDER BY reproduction_index LIMIT 1",
            (reservation_id,),
        ).fetchone()
        if qualification is not None:
            raw = json.loads(qualification["qualification_json"] or "{}")
            evaluated = True
        elif con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='reservation_baseline_segments'"
        ).fetchone() is not None:
            segment = con.execute(
                "SELECT arena_id,stack_digest,tree_digest,stack_json "
                "FROM reservation_baseline_segments WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if segment is not None:
                manifest = json.loads(segment["stack_json"])
                raw = {
                    "arena_digest": segment["arena_id"],
                    "incumbent_manifest": manifest,
                    "incumbent_stack_digest": segment["stack_digest"],
                    "incumbent_tree_digest": segment["tree_digest"],
                }
                assigned = True

    if not raw:
        return {
            "evaluated": False,
            "assigned": False,
            "relationship": "not_evaluated",
            "artifact_digest": "",
            "current_tip_artifact_digest": "",
            "threshold_speedup": None,
        }

    manifest = raw.get("incumbent_manifest") or {}
    entry = (manifest.get("entries") or {}).get(target_id) or {}
    baseline_artifact = entry.get("artifact_digest") or ""
    result: dict[str, Any] = {
        "evaluated": evaluated,
        "assigned": assigned,
        "relationship": "no_active_tip",
        "artifact_digest": baseline_artifact,
        "stack_digest": raw.get("incumbent_stack_digest") or manifest.get("digest") or "",
        "tree_digest": raw.get("incumbent_tree_digest") or "",
        "arena_digest": raw.get("arena_digest") or manifest.get("arena_digest") or "",
        "current_tip_artifact_digest": "",
        "threshold_speedup": None,
    }
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


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- app ------

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
    heartbeat = load_json(HEARTBEAT_PATH)
    hb_age = None
    if isinstance(heartbeat.get("time_unix"), int):
        hb_age = max(0, int(time.time()) - heartbeat["time_unix"])
    registration = load_json(REGISTRATION_PATH)
    epoch = str(registration.get("worker_epoch") or heartbeat.get("worker_epoch") or "")
    tip = dict(ENRICHER.tip)
    lag_blocks = None
    if tip.get("block") and audit.get("block"):
        lag_blocks = int(tip["block"]) - int(audit["block"])
    # Deliberately no filesystem paths, hosts, RPC endpoints, or raw error
    # strings here: this API is public-facing. Details go to server logs.
    return {
        "now_unix": int(time.time()),
        "db": {"ok": db_ok},
        "chain": {
            "ok": ENRICHER.chain_ok, "netuid": NETUID, "tip": tip,
            "metagraph_age_s": (int(time.time() - ENRICHER.metagraph["fetched_at"])
                                if ENRICHER.metagraph.get("fetched_at") else None),
        },
        "intake_finalized": audit,
        "intake_lag_blocks": lag_blocks,
        "processes": [
            {"name": "intake (chain-validate)",
             "up": proc_matches("chain-validate", "--intake-only")},
            {"name": "standing supervisor",
             "up": proc_matches("standing_cpu_supervisor")},
            {"name": "CPU spool relay",
             "up": proc_matches("remote_worker_service.py", "cpu-serve")},
        ],
        "gpu_heartbeat": {
            "state": heartbeat.get("state"),
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
        unix = ENRICHER.block_time(block)["unix"]
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
        "tip": dict(ENRICHER.tip),
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
        SELECT reservation_id, block, event_index, event_subindex, hotkey,
               content_hash, invalid_reason, admission_epoch, status, target_id,
               transport_attempts, screen_lane, screen_status, screen_attempts,
               retry_position, decision, reason,
               eval_cost_payment_block, eval_cost_payment_extrinsic_index
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
def submission_detail(reservation_id: str) -> dict[str, Any]:
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
    detail["url"] = r.get("url") or ""
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
        QUAL_EVIDENCE_STATE, QUAL_EVIDENCE_EXTRA)
    detail["qualification_attempts"] = [
        {"attempt": d["attempt_index"], "decision": d["decision"], "reason": d["reason"],
         "speed": qualification_speed(d["attempt_ref_json"], evidence_roots)}
        for d in rows(con, """
            SELECT attempt_index, decision, reason, attempt_ref_json
            FROM qualification_dispositions WHERE reservation_id=? ORDER BY attempt_index
        """, (rid,))]

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
    detail["baseline"] = submission_baseline(con, rid, detail["target_id"])

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
    con.close()
    try:
        detail["forensics"] = submission_forensics(SPOOL, rid)
    except DashboardForensicsError as exc:
        detail["forensics"] = []
        detail["forensics_error"] = str(exc)
    return detail


@app.get("/api/submissions/{reservation_id}/forensics/{request_id}.log")
def download_forensics(reservation_id: str, request_id: str) -> Response:
    try:
        log = forensics_log(SPOOL, reservation_id, request_id)
    except ForensicsNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except ForensicsUnavailable as exc:
        raise HTTPException(404, str(exc)) from None
    except DashboardForensicsError as exc:
        raise HTTPException(409, str(exc)) from None
    return Response(
        content=log.payload,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{log.filename}"',
            "ETag": f'"{log.etag}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/queue")
def queue() -> dict[str, Any]:
    con = intake_conn()
    cutoff = cutoff_block(con)
    pending = rows(con, f"""
        SELECT reservation_id, block, event_index, hotkey, content_hash, status,
               target_id, screen_lane, screen_status, screen_attempts,
               transport_attempts, retry_position, decision, reason,
               admission_epoch, invalid_reason, event_subindex,
               eval_cost_payment_block, eval_cost_payment_extrinsic_index
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
    tip = dict(ENRICHER.tip)
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

    heartbeat = load_json(HEARTBEAT_PATH)
    hb_age = None
    if isinstance(heartbeat.get("time_unix"), int):
        hb_age = max(0, now - heartbeat["time_unix"])
    registration = load_json(REGISTRATION_PATH)
    epoch = str(registration.get("worker_epoch") or heartbeat.get("worker_epoch") or "")

    return {
        "now_unix": now,
        "pending": items,
        "active_leases": active,
        "recent_leases": recent_leases,
        "gpu_requests": spool_requests(),
        "supervisor": supervisor_status(epoch),
        "gpu_heartbeat": {
            "state": heartbeat.get("state"), "age_s": hb_age,
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

    owner = (ENRICHER.metagraph or {}).get("owner_coldkey") or ""
    items = []
    for p in pays:
        block = int(p["payment_block"])
        idx = int(p["payment_extrinsic_index"])
        signer = ENRICHER.extrinsic_signer(block, idx)
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
        "eval_cost_tao": 1.0,
        "destination_coldkey": owner,
        "destination_links": links_for_address(owner),
        "items": items,
        "credits": credits,
    }


@app.get("/api/winners")
def winners() -> dict[str, Any]:
    con = intake_conn()
    passed = rows(con, """
        SELECT sc.reservation_id, sc.status, sc.reason, sc.candidate_json,
               r.hotkey, r.block AS submission_block, r.content_hash,
               max(q.retained_block) AS passed_block
        FROM settlement_candidates sc
        JOIN reservations r ON r.reservation_id = sc.reservation_id
        JOIN settlement_qualifications q ON q.reservation_id = sc.reservation_id
        WHERE r.status='qualified' AND r.decision='PASS'
          AND sc.status!='duplicate_proposal'
        GROUP BY sc.reservation_id
    """)
    crown_events = rows(con, """
        SELECT e.sequence, e.reservation_id, e.target_id, sc.candidate_json
        FROM settlement_events e
        JOIN settlement_candidates sc ON sc.reservation_id = e.reservation_id
        WHERE e.event_type = 'CROWN'
        ORDER BY e.sequence
    """)
    evidence_roots = qualification_evidence_roots(
        QUAL_EVIDENCE_STATE, QUAL_EVIDENCE_EXTRA)
    speeds_by_reservation: dict[str, list[object]] = {}
    if passed:
        marks = ",".join("?" for _ in passed)
        for disposition in rows(con, f"""
            SELECT reservation_id, attempt_ref_json FROM qualification_dispositions
            WHERE decision='PASS' AND reservation_id IN ({marks})
            ORDER BY reservation_id, attempt_index
        """, tuple(row["reservation_id"] for row in passed)):
            speed = qualification_speed(
                disposition["attempt_ref_json"], evidence_roots)
            if speed is None:
                continue
            speeds_by_reservation.setdefault(
                disposition["reservation_id"], []).append(speed)
    con.close()
    cumulative_by_reservation = cumulative_crown_speedups(crown_events)

    items = []
    for row in passed:
        cj = json.loads(row["candidate_json"] or "{}")
        primary = cj.get("primary") or {}
        repro = cj.get("reproduction") or {}
        target = primary.get("target_id") or cj.get("target_id") or ""
        speeds = tuple(filter(None, (
            safe_float(primary.get("speedup")),
            safe_float(repro.get("speedup")),
        )))
        speedup = min(speeds) if speeds else None
        cumulative = cumulative_by_reservation.get(row["reservation_id"])
        candidate_tps = conservative_candidate_tokens_per_second(
            speeds_by_reservation.get(row["reservation_id"], []))
        sglang_tps = estimated_sglang_tokens_per_second(
            candidate_tps, cumulative)
        hotkey = row["hotkey"]
        hk = ENRICHER.hotkey_info(hotkey)
        passed_block = max(int(row["passed_block"] or 0), int(row["submission_block"]))
        items.append({
            "reservation_id": row["reservation_id"],
            "hotkey": hotkey,
            "hotkey_links": links_for_address(hotkey),
            "target_id": target,
            "target_summary": target_summary(target),
            "speedup": speedup,
            "improvement_pct": (speedup - 1) * 100 if speedup else None,
            "speedup_primary": safe_float(primary.get("speedup")),
            "speedup_reproduction": safe_float(repro.get("speedup")),
            "cumulative_speedup_over_sglang": (
                float(cumulative) if cumulative is not None else None),
            "cumulative_improvement_pct_over_sglang": (
                float((cumulative - 1) * 100) if cumulative is not None else None),
            "tokens_per_second": (
                round(float(candidate_tps), 1)
                if candidate_tps is not None else None),
            "sglang_tokens_per_second": (
                round(float(sglang_tps), 1) if sglang_tps is not None else None),
            "passed": with_time(passed_block),
            "passed_links": links_for_block(passed_block),
            "submitted": with_time(int(row["submission_block"])),
            "reward_claim_status": (
                "earning"
                if row["status"] == "crowned"
                else (
                    "awaiting_settlement"
                    if row["status"] in {"pending", "leased"}
                    else "stale"
                )
            ),
            "settlement_status": row["status"],
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
        "items": items,
        "emission_symbol": emission_symbol(),
        "pass_total": len(items),
        "note": (
            "Only settled CROWN contributions earn. Pending pairs await settlement; "
            "held measurements are stale and must be recomputed after recommission."
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
    items = []
    for m in data:
        hk = ENRICHER.hotkey_info(m["hotkey"])
        active = m["submissions"] - (m["qualified"] or 0) - (m["failed"] or 0) \
            - (m["expired"] or 0) - (m["held"] or 0)
        items.append({
            **m,
            "active": active,
            "crowned": crowned_by_hotkey.get(m["hotkey"], 0),
            "hotkey_links": links_for_address(m["hotkey"]),
            "first_seen": with_time(int(m["first_block"])),
            "last_seen": with_time(int(m["last_block"])),
            "registered": hk.get("registered", False),
            "uid": hk.get("uid"),
            "emission_alpha_per_day": hk.get("emission_alpha_per_day"),
        })
    return {"items": items, "emission_symbol": emission_symbol()}


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
    con = intake_conn()
    data = rows(con, """
        SELECT sequence, status, updated_block, record_json
        FROM weight_publications ORDER BY sequence DESC LIMIT ?
    """, (limit,))
    con.close()
    items = []
    for w in data:
        rj = json.loads(w["record_json"] or "{}")
        items.append({
            "sequence": w["sequence"],
            "status": w["status"],
            "updated": with_time(int(w["updated_block"])),
            "submit_block": rj.get("submit_block"),
            "confirmed_block": rj.get("confirmed_block"),
            "reason": rj.get("reason") or "",
        })
    return {"items": items}


@app.get("/api/hotkey/{hotkey}")
def hotkey(hotkey: str) -> dict[str, Any]:
    info = ENRICHER.hotkey_info(hotkey)
    info["links"] = links_for_address(hotkey)
    return info


@app.exception_handler(sqlite3.Error)
def _sqlite_error(_req: Any, exc: sqlite3.Error) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": f"database: {exc}"})


if ENRICH:
    ENRICHER.start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("CACHEON_DASH_HOST", "127.0.0.1"),
        port=int(os.environ.get("CACHEON_DASH_PORT", "8788")),
    )

"""Request-scoped dashboard sources; presentation never changes reward authority."""

from contextlib import closing
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlencode

from fastapi import HTTPException
from fastapi.responses import JSONResponse

selected = ContextVar("dashboard_source", default=None)
_PATHS = {"db": "DB_PATH", "mission": "MISSION", "audit": "AUDIT_PATH",
          "spool": "SPOOL", "heartbeat": "HEARTBEAT_PATH", "registration": "REGISTRATION_PATH",
          "logs": "LOG_ROOT", "evidence_state": "QUAL_EVIDENCE_STATE", "stage": "STAGE_ROOT"}


@dataclass
class DashboardSource:
    """One explicit read-only plane and its independent enrichment cache."""

    key: str
    label: str
    model: str
    values: dict
    processes: dict
    weights_included: bool
    checkpoint: dict | None

    def public(self):
        """Expose labels and reward status, never operator filesystem coordinates."""
        return {"key": self.key, "label": self.label, "model": self.model,
                "weights_status": "included in global offer" if self.weights_included
                else "weights off / not yet in served vector"}


def value(name, fallback):
    """Resolve request-local inputs without mutating shared module globals."""
    source = selected.get()
    return source.values[name] if source is not None and name in source.values else fallback


def load_sources(path, network, netuid, enrich):
    """Read explicitly configured planes; no discovery of private roots by model name."""
    from dashboard.enrichment import Enrichment

    raw = json.loads(Path(path).read_text())
    if type(raw) is not dict or set(raw) != {"default", "sources"} or not raw["sources"]:
        raise ValueError("dashboard sources require default and sources")
    intake_paths = {Path(row["paths"]["db"]).resolve() for row in raw["sources"]}
    if any(Path(row["cache"]).resolve() in intake_paths for row in raw["sources"]):
        raise ValueError("enrichment cache cannot be an intake database")
    result, databases, caches, private_roots = {}, set(), set(), []
    for row in raw["sources"]:
        if set(row) != {"key", "label", "model", "paths", "cache", "evidence_roots",
                        "cutoff_reservation", "processes", "weights_included", "checkpoint"}:
            raise ValueError("dashboard source fields do not match")
        key = row["key"]
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", key) or key in result:
            raise ValueError("dashboard source keys must be unique")
        if set(row["paths"]) != set(_PATHS) or type(row["weights_included"]) is not bool:
            raise ValueError("source paths and weight status must be explicit")
        paths = {name: Path(path) for name, path in row["paths"].items()}
        extras, cache = tuple(Path(p) for p in row["evidence_roots"]), Path(row["cache"])
        if any(not p.is_absolute() for p in (*paths.values(), *extras, cache)):
            raise ValueError("dashboard source paths must be absolute")
        db, private = paths["db"].resolve(), (paths["mission"] / "private").resolve()
        if db in databases or cache.resolve() in caches or any(
                private == other or private in other.parents or other in private.parents for other in private_roots):
            raise ValueError("dashboard sources require distinct databases, caches and private roots")
        if set(row["processes"]) != {"intake", "supervisor", "relay"} or any(
                not needles or not all(isinstance(n, str) and n for n in needles)
                or not any(n.startswith("/") for n in needles) for needles in row["processes"].values()):
            raise ValueError("process checks require command arguments and an absolute source path")
        values = {_PATHS[name]: p for name, p in paths.items()}
        values.update(QUAL_EVIDENCE_EXTRA=extras, CUTOFF_RESERVATION=row["cutoff_reservation"],
                      ENRICHER=Enrichment(cache, network, netuid))
        result[key] = DashboardSource(key, row["label"], row["model"], values,
                                      row["processes"], row["weights_included"], row["checkpoint"])
        databases.add(db)
        caches.add(cache.resolve())
        private_roots.append(private)
    if raw["default"] not in result:
        raise ValueError("dashboard default source is absent")
    if enrich:
        for source in result.values():
            source.values["ENRICHER"].start()
    return raw["default"], result


def process_matches(role, default=(), *, proc_root=Path("/proc")):
    """Match argv tokens and source paths, never another plane's module substring."""
    source = selected.get()
    needles = source.processes[role] if source else default
    if not needles:
        return False
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            tokens = [part for arg in argv for part in (arg.split("=", 1) if arg.startswith("--") else [arg])]
            cwd = entry.joinpath("cwd").resolve()
            def matches(needle):
                if not needle.startswith("/"):
                    return needle in tokens
                root = Path(needle)
                return any(root == p or root in p.parents for p in (
                    (Path(token) if token.startswith("/") else cwd / token)
                    for token in tokens if "/" in token))
            if all(matches(needle) for needle in needles):
                return True
        except (OSError, UnicodeError):
            continue
    return False


def check_ownership(source, sources):
    """Refuse ambiguous reservation ownership without hiding unavailable peers."""
    # Both chain listeners retain arrivals. A missing-payment rejection before
    # publication does not claim a submission admitted by the other listener.
    ownership = """SELECT reservation_id FROM reservations WHERE NOT (
        status='failed' AND reason='missing_eval_cost_payment' AND target_id=''
        AND publication_digest='' AND arena_service_digest='' AND screen_status=''
        AND decision='')"""
    with closing(sqlite3.connect(source.values["DB_PATH"].as_uri() + "?mode=ro", uri=True)) as con:
        owned = {row[0] for row in con.execute(ownership)}
    for other in sources.values():
        if other is source:
            continue
        try:
            with closing(sqlite3.connect(other.values["DB_PATH"].as_uri() + "?mode=ro", uri=True)) as con:
                if any(row[0] in owned for row in con.execute(ownership)):
                    raise HTTPException(409, "Reservation ownership conflicts across arenas")
        except sqlite3.Error:
            continue


def qualify_response(payload, source):
    """Keep source-qualified links through detail, recovery and delayed downloads."""
    if isinstance(payload, list):
        return [qualify_response(item, source) for item in payload]
    if not isinstance(payload, dict):
        if isinstance(payload, str) and payload.startswith("/api/submissions/"):
            return payload + ("&" if "?" in payload else "?") + urlencode({"arena": source.key})
        return payload
    result = {key: qualify_response(item, source) for key, item in payload.items()}
    if "reservation_id" in result:
        result["source"] = source.key
    if not source.weights_included:
        if "weight_share" in result:
            result["weight_share"] = None
        if "reward_claim_status" in result:
            result["reward_claim_status"] = "weights_off"
        if "note" in result:
            result["note"] = "Weights off / not yet in served vector."
    return result


def install_sources(app, api):
    """Bind existing routes to per-request sources and expose all-plane health."""
    path = os.environ.get("CACHEON_DASH_SOURCES")
    default, sources = load_sources(path, api["NETWORK"], api["NETUID"], api["ENRICH"]) if path else (None, {})
    app.state.dashboard_sources = sources
    app.state.dashboard_default = default

    @app.get("/api/arenas")
    def arenas():
        items = []
        for source in app.state.dashboard_sources.values():
            token = selected.set(source)
            try:
                items.append({**source.public(), "health": api["health"]()})
            finally:
                selected.reset(token)
        return {"default": app.state.dashboard_default, "items": items}

    @app.get("/api/arena-events")
    def arena_events():
        items, unavailable = [], []
        for source in app.state.dashboard_sources.values():
            token = selected.set(source)
            try:
                check_ownership(source, app.state.dashboard_sources)
                items.extend({**row, "source": source.key} for row in api["events"](limit=200)["items"])
            except sqlite3.Error:
                unavailable.append(source.key)
            finally:
                selected.reset(token)
        items.sort(key=lambda row: ((row.get("when") or {}).get("block", 0), row["source"], row["sequence"]), reverse=True)
        return {"items": items[:200], "unavailable_sources": unavailable}

    @app.middleware("http")
    async def source_request(request, call_next):
        sources = app.state.dashboard_sources
        global_route = request.url.path in {"/api/weights", "/api/arenas", "/api/arena-events", "/api/bundle-encryption-key"}
        if not request.url.path.startswith("/api/") or global_route:
            return await call_next(request)
        key = request.query_params.get("arena", app.state.dashboard_default)
        if key is None and not sources:
            return await call_next(request)
        if key not in sources:
            return JSONResponse({"detail": "Unknown arena"}, status_code=404)
        source = sources[key]
        token = selected.set(source)
        try:
            if request.url.path != "/api/health":
                try:
                    check_ownership(source, sources)
                except HTTPException as exc:
                    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
                except sqlite3.Error:
                    return JSONResponse({"detail": "Arena database unavailable", "source": key}, status_code=503)
            response = await call_next(request)
            if response.headers.get("content-type", "").startswith("application/json"):
                payload = json.loads(b"".join([chunk async for chunk in response.body_iterator]))
                payload = qualify_response(payload, source)
                payload["arena"] = source.public()
                headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
                return JSONResponse(payload, status_code=response.status_code, headers=headers)
            return response
        finally:
            selected.reset(token)

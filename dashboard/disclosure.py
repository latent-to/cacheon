"""Delay dashboard source disclosure until eight hours after evaluation."""

from __future__ import annotations

import json
import time
import tempfile
from contextlib import closing
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import Response

from dashboard.forensics import (
    DashboardForensicsError, ForensicsNotFound, ForensicsUnavailable, forensics_log,
)
from cacheon.chain.source_disclosure import (
    DISCLOSURE_DELAY_SECONDS as DISCLOSURE_DELAY_SECONDS,
    bundle_visibility as source_visibility,
)


def bundle_visibility(connection, reservation_id, block_time):
    """Use retained result blocks, never submission time or an estimated timestamp."""
    try:
        return source_visibility(connection, reservation_id, block_time, now=time.time())
    except LookupError:
        raise HTTPException(404, "reservation not found") from None


def disclose_bundle(connection, detail, block_time):
    """Keep results visible while withholding links and source-bearing diagnostics."""
    visibility = bundle_visibility(connection, detail["reservation_id"], block_time)
    detail["bundle_visibility"] = visibility
    detail["url"] = (f"/api/submissions/{detail['reservation_id']}/bundle.tar.gz"
                     if visibility["available"] else "")
    if visibility["available"]:
        return
    for run in detail.get("forensics", []):
        if isinstance(run.get("worker_log"), dict):
            run["worker_log"].update(download_url=None, explanation=[])
        if isinstance(run.get("qualification_hold"), dict):
            run["qualification_hold"].pop("failure_message", None)


def download_public_log(connection, spool, reservation_id, request_id, block_time):
    """Apply the same disclosure gate to direct raw-log requests."""
    visibility = bundle_visibility(connection, reservation_id, block_time)
    if not visibility["available"]:
        raise HTTPException(403, {
            "message": "Source and raw logs are available eight hours after evaluation.",
            "release_at": visibility["release_at"],
        }, headers={"Cache-Control": "no-store"})
    try:
        log = forensics_log(spool, reservation_id, request_id)
    except (ForensicsNotFound, ForensicsUnavailable) as exc:
        raise HTTPException(404, str(exc)) from None
    except DashboardForensicsError as exc:
        raise HTTPException(409, str(exc)) from None
    return Response(content=log.payload, media_type="text/plain", headers={
        "Content-Disposition": f'attachment; filename="{log.filename}"',
        "ETag": f'"{log.etag}"', "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    })


def install_disclosure_routes(app, connection, block_time, private_root, spool, registration=None):
    """Use the same result clock for logs and the validator's checked bundle bytes."""
    @app.get("/api/baseline")
    def baseline(arena: str = ""):
        from cacheon.chain.baseline_admission import latest_revealed, revealed_baselines
        from cacheon.chain.intake import IntakeError
        from cacheon.stack_manifest import EvaluationStackManifest
        from dashboard.sources import selected

        registered = {} if registration is None else registration()
        actual_arena = registered.get("worker_readiness", {}).get("arena_id")
        if not actual_arena or not registered.get("service_identity"):
            raise HTTPException(503, "Commissioned baseline registration unavailable")
        if selected.get() is None and arena and arena != actual_arena:
            raise HTTPException(404, "Unknown competition arena")
        with closing(connection()) as con:
            legacy = con.execute("SELECT value FROM metadata WHERE key='legacy_arena_id'").fetchone()
            scope = "" if legacy is not None and actual_arena == legacy[0] else actual_arena
            stack = con.execute("SELECT stack_json FROM evaluation_stacks WHERE arena_id=? "
                                "AND competition_arena=?", (registered["service_identity"], scope)).fetchone()
            if stack is None:
                raise HTTPException(503, "Commissioned baseline state unavailable")
            try:
                incumbent = EvaluationStackManifest.from_dict(json.loads(stack[0]))
                winner = latest_revealed(revealed_baselines(con, scope, block_time, incumbent), time.time())
            except IntakeError as exc:
                raise HTTPException(503, str(exc)) from exc
        return {"competition_arena": actual_arena, "baseline": "stock" if winner is None else winner[1],
                "release_at": None if winner is None else winner[0],
                "bundle_url": None if winner is None else
                    f"/api/submissions/{winner[1]}/bundle.tar.gz"}

    @app.get("/api/bundle-encryption-key")
    def encryption_key():
        from cacheon.chain.bundle_privacy import validator_key
        from cacheon.chain.fetch import FetchTransientError

        try:
            return {"algorithm": "x25519-sealedbox-v1", "public_key": bytes(validator_key().public_key).hex()}
        except FetchTransientError:
            raise HTTPException(503, "Bundle encryption is not configured") from None

    @app.get("/api/submissions/{reservation_id}/forensics/{request_id}.log")
    def log(reservation_id: str, request_id: str):
        with closing(connection()) as con:
            return download_public_log(con, spool(), reservation_id, request_id, block_time)

    @app.get("/api/submissions/{reservation_id}/bundle.tar.gz")
    def bundle(reservation_id: str):
        from cacheon.chain.fetch import package_bundle, _validate_private_tree, FetchError

        with closing(connection()) as con:
            visibility = bundle_visibility(con, reservation_id, block_time)
            if not visibility["available"]:
                raise HTTPException(403, visibility, headers={"Cache-Control": "no-store"})
            digest = con.execute("SELECT content_hash FROM reservations WHERE reservation_id=?",
                                 (reservation_id,)).fetchone()[0]
        source = private_root() / digest
        if not source.is_dir():
            raise HTTPException(404, "Checked bundle is not retained")
        with tempfile.TemporaryDirectory(prefix="cacheon-disclosure.") as temporary:
            try:
                _validate_private_tree(source)
                archive, actual = package_bundle(source, Path(temporary) / "bundle.tar.gz")
                if actual != digest:
                    raise FetchError("retained bundle differs from committed hash")
            except (OSError, ValueError, FetchError) as exc:
                raise HTTPException(409, "Retained bundle verification failed") from exc
            return Response(archive.read_bytes(), media_type="application/gzip", headers={
                "Content-Disposition": f'attachment; filename="{digest}.tar.gz"',
                "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

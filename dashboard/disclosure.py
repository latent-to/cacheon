"""Delay dashboard source disclosure until eight hours after evaluation."""

from __future__ import annotations

import time
import tempfile
from contextlib import closing
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import Response

from dashboard.forensics import (
    DashboardForensicsError, ForensicsNotFound, ForensicsUnavailable, forensics_log,
)

DISCLOSURE_DELAY_SECONDS = 8 * 60 * 60


def bundle_visibility(connection, reservation_id, block_time):
    """Use retained result blocks, never submission time or an estimated timestamp."""
    row = connection.execute(
        "SELECT status FROM reservations WHERE reservation_id=?", (reservation_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "reservation not found")
    result_block = 0
    active = connection.execute(
        "SELECT 1 FROM evaluation_leases l JOIN evaluation_lease_members m "
        "ON m.lease_id=l.lease_id WHERE m.reservation_id=? AND l.state='active'",
        (reservation_id,),
    ).fetchone()
    if row[0] in ("qualified", "failed") and not active:
        result_block = connection.execute(
            "SELECT max(block) FROM ("
            "SELECT max(l.completed_block) AS block FROM evaluation_leases l "
            "JOIN evaluation_lease_members m ON m.lease_id=l.lease_id "
            "WHERE m.reservation_id=? AND l.state='completed' "
            "AND l.stage IN ('screen','qualification') UNION ALL "
            "SELECT max(retained_block) FROM settlement_qualifications "
            "WHERE reservation_id=?)", (reservation_id, reservation_id),
        ).fetchone()[0] or 0
    stamp = block_time(result_block) if result_block else {}
    release_at = (stamp["unix"] + DISCLOSURE_DELAY_SECONDS
                  if stamp.get("unix") is not None and stamp.get("estimated") is False
                  else None)
    return {"available": release_at is not None and time.time() >= release_at,
            "release_at": release_at, "result_block": result_block or None}


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


def install_disclosure_routes(app, connection, block_time, private_root, spool):
    """Use the same result clock for logs and the validator's checked bundle bytes."""
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

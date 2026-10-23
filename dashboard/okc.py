"""The Open Kernel Challenge arena: each claim's standing and the code behind it.

scripts/arena/okc_shares.py rewrites the claims file on every refresh; the dashboard only serves it,
so the code behind every paid claim is public.
"""
from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

CLAIMS_PATH = Path(os.environ.get("CACHEON_DASH_OKC_CLAIMS", "/root/cacheon-ops/state/okc/claims.json"))


def _claims() -> dict:
    try:
        return json.loads(CLAIMS_PATH.read_text())
    except FileNotFoundError:
        raise HTTPException(503, "The Open Kernel Challenge claims have not been refreshed yet") from None


def _page(doc: dict) -> str:
    e = html.escape
    share = "".join(f"<li><code>{e(hotkey)}</code> {parts / 10_000:.2f}% of the arena's share</li>"
                    for hotkey, parts in sorted(doc["shares"].items(), key=lambda item: -item[1]))
    boards = "".join(
        f"<h3>{e(board)}</h3><ol>" + "".join(
            f"<li>{e(str(row['team']))}: {row['value']:.4g} (job <code>{e(job)}</code>)</li>"
            for job, row in rows.items()) + "</ol>"
        for board, rows in doc["top"].items() if rows)
    claims = "".join(
        f"<section><h3>{e(claim['board'])} job <code>{e(claim['job'])}</code>: {e(claim['status'])}</h3>"
        f"<p>Hotkey <code>{e(claim['hotkey'])}</code>, block {claim['block']}, reservation "
        f"<code>{e(claim['reservation_id'])}</code></p>"
        + "".join(f"<details><summary>{e(name)}</summary><pre>{e(text)}</pre></details>"
                  for name, text in claim["files"].items()) + "</section>"
        for claim in reversed(doc["claims"]))
    refreshed = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(doc["refreshed"]))
    return (
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        "<title>Open Kernel Challenge arena</title>"
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
        "pre{overflow:auto;background:#f4f4f4;padding:.75rem}code{word-break:break-all}</style>"
        f"<h1>Open Kernel Challenge arena</h1><p>Refreshed {refreshed}. A claim whose job is in its board's "
        "top 3 is paid an equal part of the arena's share; with none, the default hotkey "
        f"<code>{e(doc['default_hotkey'])}</code> is paid. "
        "<a href='https://openkernelchallenge.dev/'>The challenge</a>.</p>"
        f"<h2>Paid now</h2><ul>{share}</ul><h2>Top 3 per board</h2>{boards or '<p>No rows yet.</p>'}"
        f"<h2>Claims</h2>{claims or '<p>No claims yet.</p>'}"
    )


def install_okc(app) -> None:
    """Serve the claims file as JSON and as a page."""
    @app.get("/api/okc")
    def okc_claims():
        return _claims()

    @app.get("/okc", include_in_schema=False)
    def okc_page():
        return HTMLResponse(_page(_claims()))

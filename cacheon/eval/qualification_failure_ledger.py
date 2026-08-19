"""Durable, operator-readable record of why a qualification reached no verdict.

A qualification that cannot produce PASS/FAIL crosses the trust boundary
carrying only ``NO_DECISION`` and a ``failure_digest``.  The digest binds the
failure text but does not carry it, and widening that wire schema is exactly
what broke two consumers on 2026-08-15 — so the text cannot ride along.

The digest is already durable on both sides: it is written into the worker's
acked result tars and result blobs, and it reaches the validator inside the
outcome.  That makes it a join key.  Recording ``digest -> text`` on the worker
turns "why was this bundle removed?" into a lookup, with no wire change and no
new authority.

Before this, the answer existed only as a stderr line in a rotating worker log.
On 2026-08-16 three reservations parked as ``legacy_no_decision`` and the real
cause — a resident speed policy with no path for the bundle's class — was
recoverable only because that log happened to still be on disk.

The destination is supplied by the environment, never hardcoded: a generic
evaluator must not name an operator's paths.  With the variable unset this does
nothing, which keeps the default behaviour exactly as it was.

Every failure here is swallowed.  A diagnostic record must never change, delay,
or fail a verdict.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Absolute path of the JSONL ledger. Unset disables recording entirely.
FAILURE_LEDGER_ENV = "CACHEON_QUALIFICATION_FAILURE_LEDGER"

LEDGER_SCHEMA = "cacheon-qualification-failure-ledger-v1"

_MESSAGE_LIMIT = 4096


def record_qualification_failure(
    *,
    failure_digest: str,
    authority_digest: str,
    exc: BaseException,
    recorded_at: str | None = None,
) -> bool:
    """Append one ``digest -> why`` row. True when a row was written.

    Best effort by contract: any failure to record is discarded, because the
    alternative is letting a diagnostic aid turn a real verdict into a fault.
    """

    destination = os.environ.get(FAILURE_LEDGER_ENV)
    if not destination:
        return False
    try:
        path = Path(destination)
        if not path.is_absolute():
            return False
        row = {
            "authority_digest": authority_digest,
            "exception": type(exc).__name__,
            "failure_digest": failure_digest,
            "message": str(exc)[:_MESSAGE_LIMIT],
            "schema": LEDGER_SCHEMA,
        }
        if recorded_at is not None:
            row["recorded_at"] = recorded_at
        line = json.dumps(
            row, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        # One open-append-close per record: the ledger is written from
        # short-lived adapter processes, and an O_APPEND write of a single
        # short line is atomic enough that concurrent adapters cannot
        # interleave partial rows.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return True
    except Exception:  # noqa: BLE001 - diagnostics must never fail a verdict
        return False

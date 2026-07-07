"""Seam-activation receipts — the anti-phantom-pass gate.

The failure mode this closes (hit for real on 2026-07-07): the candidate engine
comes up WITHOUT the seam (missing ``optima.pth``, bad env, bundle load failure
falling back to baseline) and the eval happily scores stock-vs-stock — identical
logits, KL exactly 0.0, accuracy delta 0.0, verdict PASS. ``seam.activate()``
deliberately never wedges the engine on a bad bundle, so the *engine* can't be
the one to fail; the *eval driver* must demand positive evidence.

Evidence lives where the seam lives — in sglang's spawned scheduler ranks — so it
travels by file: the driver sets ``OPTIMA_SEAM_RECEIPT_DIR`` for the candidate
launch, ranks write receipts there, the driver requires them:

  * ``active``      — bundle loaded + registry enabled in a rank (seam.activate).
  * ``load_failed`` — a rank ATTEMPTED the bundle load and fell back to baseline;
                      lets the driver report "bad bundle" instead of "no bootstrap".
  * ``fired``       — the registry actually SELECTED the miner impl for a slot at
                      least once (registry.lookup); written once per slot per process.

No env var set -> every helper is a silent no-op (verify paths, unit tests, and
baseline launches don't produce receipt litter).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("optima.receipts")

_SAFE_RE = re.compile(r"[^0-9A-Za-z._\-]+")


def _dir() -> str:
    return os.environ.get("OPTIMA_SEAM_RECEIPT_DIR", "").strip()


def write(kind: str, payload: dict, *, tag: str = "") -> None:
    """Write one receipt file; never raises (a receipt must not break an engine)."""
    rdir = _dir()
    if not rdir:
        return
    try:
        Path(rdir).mkdir(parents=True, exist_ok=True)
        suffix = f".{_SAFE_RE.sub('_', tag)}" if tag else ""
        p = Path(rdir) / f"{kind}{suffix}.{os.getpid()}.json"
        p.write_text(json.dumps(payload, sort_keys=True))
    except Exception:  # noqa: BLE001
        logger.exception("optima: receipt write failed (kind=%s)", kind)


def collect(rdir: str | Path, kind: str) -> list[dict]:
    """All receipts of ``kind`` under ``rdir`` (unreadable files skipped)."""
    out: list[dict] = []
    root = Path(rdir)
    if not root.is_dir():
        return out
    for p in sorted(root.glob(f"{kind}*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):  # noqa: PERF203
            continue
    return out


def require(rdir: str | Path, kind: str, *, context: str) -> list[dict]:
    """Return receipts of ``kind`` or raise with a diagnosis — the eval-side gate."""
    got = collect(rdir, kind)
    if got:
        return got
    failed = collect(rdir, "load_failed")
    if failed:
        raise RuntimeError(
            f"{context}: seam rank(s) attempted the bundle load and FELL BACK to baseline "
            f"(load_failed receipts: {failed}). The run would have scored stock-vs-stock; "
            "fix the bundle, do not trust any output from this launch."
        )
    raise RuntimeError(
        f"{context}: no '{kind}' seam receipt was written by any engine rank. The candidate "
        "ran WITHOUT the miner kernel (stock-vs-stock) — likely missing optima.pth bootstrap "
        "in the engine interpreter, OPTIMA env not reaching spawned ranks, or the seamed "
        "module was never imported by this engine config. Refusing to score a phantom."
    )

"""Seam execution receipts — positive accounting evidence for the referee.

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
  * ``fired``       — the registry SELECTED the miner impl for a slot at least once;
                      this is routing evidence only.
  * ``completed``   — a dispatcher successfully produced the model-facing output
                      after invoking the selected implementation; once/slot/process.
  * ``fallback``    — a selected implementation raised and the dispatcher served the
                      trusted baseline instead; once/slot/process and disqualifying.

``completed`` is deliberately stronger than ``fired`` but is still only execution
and accounting evidence.  It is not hostile-code trust proof: a candidate running in
the serving process can introspect or forge process-local state.  Isolation and an
external referee remain the security boundary.

No env var set -> every helper is a silent no-op (verify paths, unit tests, and
baseline launches don't produce receipt litter).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("optima.receipts")

_SAFE_RE = re.compile(r"[^0-9A-Za-z._\-]+")
_IDENTITY_KINDS = frozenset({
    "active", "load_failed", "fired", "completed", "fallback", "audit",
})
_ONCE: set[tuple[int, str, str]] = set()
_ONCE_LOCK = threading.Lock()


def _dir() -> str:
    return os.environ.get("OPTIMA_SEAM_RECEIPT_DIR", "").strip()


def identity() -> dict:
    """Best-effort scheduler-member identity, always including stable ``pid``.

    The live sglang TP workers initialize torch.distributed before model calls, so
    completed/fallback/audit receipts normally carry the global rank and world size.
    Non-distributed unit tests and early bootstrap receipts use ``rank=-1`` and
    ``world_size=-1``; coverage then keys members by PID.  Environment values are a
    fallback for torchrun-like launchers.
    """
    pid = os.getpid()
    rank: Optional[int] = None
    world_size: Optional[int] = None
    try:
        import torch.distributed as dist  # deferred: receipts also run in CPU tooling

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:  # noqa: BLE001 - identity must never break model execution
        pass
    if rank is None:
        try:
            rank = int(os.environ["RANK"])
        except (KeyError, TypeError, ValueError):
            rank = -1
    if world_size is None:
        try:
            world_size = int(os.environ["WORLD_SIZE"])
        except (KeyError, TypeError, ValueError):
            world_size = -1
    return {"pid": pid, "rank": rank, "world_size": world_size}


def write(kind: str, payload: dict, *, tag: str = "") -> None:
    """Write one receipt file; never raises (a receipt must not break an engine)."""
    rdir = _dir()
    if not rdir:
        return
    try:
        body = dict(payload)
        if kind in _IDENTITY_KINDS:
            # Call-site values win only for compatibility with old persisted audit
            # payloads; validator-owned callers should use the detected identity.
            body = {**identity(), **body}
        Path(rdir).mkdir(parents=True, exist_ok=True)
        suffix = f".{_SAFE_RE.sub('_', tag)}" if tag else ""
        p = Path(rdir) / f"{kind}{suffix}.{os.getpid()}.json"
        p.write_text(json.dumps(body, sort_keys=True))
    except Exception:  # noqa: BLE001
        logger.exception("optima: receipt write failed (kind=%s)", kind)


def _write_execution_once(kind: str, slot: str, *, error: BaseException | None = None) -> None:
    """Write a one-time slot execution receipt without adding hot-path file churn."""
    # Preserve write()'s no-env no-op contract without consuming the one-time guard.
    # This matters in long-lived test/dev processes that exercise a dispatcher before
    # arming a later receipted launch.
    if not _dir():
        return
    key = (os.getpid(), kind, slot)
    with _ONCE_LOCK:
        if key in _ONCE:
            return
        _ONCE.add(key)
    payload = {"slot": slot}
    if error is not None:
        # Diagnostic only; never serialize traceback/arguments that may retain tensors.
        payload.update(error_type=type(error).__name__, error=str(error)[:512])
    write(kind, payload, tag=slot)


def completed(slot: str) -> None:
    """Record successful candidate output production once for this slot/process.

    Dispatchers call this only after the selected entry and validator-owned tail work
    have returned successfully.  It proves neither numerical correctness nor hostile
    trust; those remain qualification/audit/isolation responsibilities.
    """
    _write_execution_once("completed", slot)


def fallback(slot: str, error: BaseException) -> None:
    """Record that a selected candidate raised and stock fallback was taken."""
    _write_execution_once("fallback", slot, error=error)


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


def _valid_int(value, *, minimum: int = 0) -> Optional[int]:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= minimum else None


def _member_label(receipt: dict, *, basis: str) -> Optional[str]:
    value = _valid_int(receipt.get(basis))
    return f"{basis}:{value}" if value is not None else None


def _expected_members(observed: list[dict], members: list[dict]) -> tuple[str, list[str]]:
    """Choose a coverage basis and the expected scheduler-member labels.

    Explicit ``members`` (normally ``active`` receipts) use PID, because PID remains
    stable even if an early bootstrap receipt predates process-group initialization.
    Without them, a known distributed world size expands to every rank; the last-resort
    observed-PID mode cannot detect a scheduler process that wrote no receipt at all.
    """
    member_pids = sorted({label for r in members
                          if (label := _member_label(r, basis="pid")) is not None})
    if member_pids:
        return "pid", member_pids

    all_receipts = [*members, *observed]
    world_sizes = [ws for r in all_receipts
                   if (ws := _valid_int(r.get("world_size"), minimum=1)) is not None]
    if world_sizes:
        return "rank", [f"rank:{rank}" for rank in range(max(world_sizes))]

    observed_pids = sorted({label for r in observed
                            if (label := _member_label(r, basis="pid")) is not None})
    return "pid", observed_pids


def coverage_matrix(
    observed: Iterable[dict], *, expected_slots: Iterable[str],
    member_receipts: Iterable[dict] = (), count_field: str | None = None,
    min_count: int = 1,
) -> dict:
    """Build machine-readable per-slot/per-member execution coverage.

    ``member_receipts`` should be the launch's ``active`` receipts.  The expected
    matrix is their member set crossed with ``expected_slots``.  For audit receipts,
    pass ``count_field="n"`` and a per-member minimum.  This helper is intentionally
    generic so qualification can place completed and audit evidence in one report.
    """
    got = list(observed)
    members = list(member_receipts)
    slots = sorted({str(s) for s in expected_slots if str(s)})
    basis, expected_members = _expected_members(got, members)
    expected_pairs = {(slot, member) for slot in slots for member in expected_members}
    counts: dict[tuple[str, str], int] = {}
    malformed: list[dict] = []
    for receipt in got:
        slot = receipt.get("slot")
        member = _member_label(receipt, basis=basis)
        if not isinstance(slot, str) or not slot or member is None:
            malformed.append(receipt)
            continue
        if count_field is None:
            count = 1
        else:
            count = _valid_int(receipt.get(count_field)) or 0
        key = (slot, member)
        counts[key] = counts.get(key, 0) + count
    present = {pair for pair, count in counts.items() if count >= min_count}
    missing = sorted(expected_pairs - present)
    short = sorted((slot, member, counts.get((slot, member), 0))
                   for slot, member in expected_pairs
                   if 0 < counts.get((slot, member), 0) < min_count)
    return {
        "ok": bool(slots) and bool(expected_members) and not missing and not malformed,
        "basis": basis,
        "expected_slots": slots,
        "members": expected_members,
        "expected_pairs": len(expected_pairs),
        "covered_pairs": len(expected_pairs & present),
        "missing": [{"slot": slot, "member": member} for slot, member in missing],
        "short": [{"slot": slot, "member": member, "count": count,
                   "required": min_count} for slot, member, count in short],
        "malformed": malformed,
    }


def completed_gate(
    completed_receipts: Iterable[dict], *, expected_slots: Iterable[str],
    member_receipts: Iterable[dict] = (), fallback_receipts: Iterable[dict] = (),
) -> tuple[bool, str]:
    """Require one successful completion for every expected slot/member pair.

    Any selected-candidate exception is a failure even if a later call completed: the
    scored model execution already consumed stock for at least one candidate call.
    This is an accounting gate, not an isolation or anti-forgery boundary.
    """
    complete = list(completed_receipts)
    members = list(member_receipts)
    fallbacks = list(fallback_receipts)
    detail = coverage_matrix(complete, expected_slots=expected_slots,
                             member_receipts=members)
    fallback_detail = coverage_matrix(
        fallbacks, expected_slots=expected_slots, member_receipts=members)
    relevant_fallbacks = [r for r in fallbacks if r.get("slot") in detail["expected_slots"]]
    ok = detail["ok"] and not relevant_fallbacks
    desc = (f"completed coverage {detail['covered_pairs']}/{detail['expected_pairs']} "
            f"slot/member pairs (basis={detail['basis']})")
    if detail["missing"]:
        desc += f"; missing={detail['missing']}"
    if detail["malformed"]:
        desc += f"; malformed={len(detail['malformed'])}"
    if relevant_fallbacks:
        desc += f"; selected-candidate fallbacks={relevant_fallbacks}"
    # Keep this in the returned description for easy qualification-report debugging.
    if fallback_detail["malformed"] and not relevant_fallbacks:
        desc += f"; malformed_fallbacks={len(fallback_detail['malformed'])}"
    return ok, desc

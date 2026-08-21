"""Answer one miner's question: what happened to my submissions, and why?

``operator_status.reservation_status`` explains exactly one reservation and
refuses a selector that matches more than one, so an operator can never
accidentally explain the wrong submission.  That is the right contract for
incident response and the wrong one for the question miners actually ask, which
is about a hotkey with several submissions over time.

This module answers that question instead.  It reads the same durable rows
through the same read-only snapshot and reuses the same redaction and
attribution helpers, so the two views cannot disagree about a verdict.  What it
adds is the part a miner cannot reconstruct: the typed disposition turned into
a stated cause and a next step.

Nothing here is an authority.  It reports persisted decisions; it never derives
one.  A row that carries no typed decision is reported as having none rather
than being narrated into a failure.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from cacheon.chain.operator_status import (
    OperatorStatusError,
    _attribution,
    _cursor,
    _qualification_dispositions,
    _readonly_connection,
    _reason,
    _screen_dispositions,
)


SCHEMA = "cacheon.miner.submission-report.v1"

# Stated cause and next step per typed disposition.  Keyed by the reason code as
# persisted; prefixed codes (``copy_of:<hash>``) match on the prefix.  A code
# absent here is reported verbatim with no invented explanation.
_GUIDANCE: dict[str, tuple[str, str]] = {
    "qualified": (
        "The bundle beat its baseline and passed qualification.",
        "A first PASS is reproduction_pending: settlement needs an independently "
        "bound second PASS before a crown is awarded.",
    ),
    "candidate_kernel_does_not_compile": (
        "A @triton.jit kernel in the bundle cannot be traced, so it never "
        "compiles and never runs.",
        "Triton is a JIT and compiles on first invocation, so this is not "
        "visible until the kernel is called. Run `cacheon.cli scan` before "
        "submitting; it reports this statically in seconds.",
    ),
    # Retained coarse vocabulary: reports settled before the band split
    # recorded every speed FAIL under this one code.
    "speed_regression": (
        "The bundle was correct, compiled, and graph-safe, and was not faster "
        "than the baseline in the timed bracket.",
        "This is an ordinary competitive result, not a defect. The baseline is a "
        "tuned production stack.",
    ),
    "speed_threshold_not_met": (
        "The bundle was correct, compiled, and graph-safe. Its measured "
        "speedup fell inside the round's noise band: it did not clear the "
        "required threshold, and it was not measurably slower either.",
        "This is an ordinary competitive result, not a regression. The bar "
        "for the round is 1 + max(min_margin, k*noise); a larger win or "
        "lower run-to-run variance clears it.",
    ),
    "candidate_slower": (
        "The bundle was correct, compiled, and graph-safe, and the timed "
        "bracket measured it slower than the baseline beyond the round's "
        "noise band.",
        "The baseline is a tuned production stack. Profile the kernel "
        "against the stock implementation before resubmitting.",
    ),
    "graph_member_not_applicable": (
        "The declared CUDA-graph applicability did not hold for the shapes "
        "actually captured.",
        "CUDA graphs are part of the scored contract. Check the graph metadata "
        "the slot requires against the shapes the target is measured at.",
    ),
    "graph_eager_failed": (
        "The bundle failed in eager execution before graph capture.",
        "Reproduce locally with `cacheon.cli verify` before submitting.",
    ),
    "screen_rejected": (
        "The bundle was rejected at the arena screen, before any timed "
        "measurement.",
        "The screen checks static scan, build, ABI, graph, and abbreviated "
        "serving. Run `cacheon.cli scan` and `cacheon.cli verify` locally.",
    ),
    "copy_of": (
        "The bundle was detected as a copy of an earlier submission or of the "
        "validator's own reference library.",
        "Detection is containment-based over the import closure, so renaming and "
        "reformatting do not defeat it, and it covers cacheon_kernels/.",
    ),
    "duplicate_of": (
        "Byte-identical content was already submitted.",
        "A FAIL verdict replays onto identical bytes. Change the kernel, not the "
        "packaging.",
    ),
    "target_unavailable": (
        "The commissioned arena workload cannot currently measure this "
        "registered target family, so the submission was parked before any "
        "evaluation.",
        "This is NOT a judgement on the bundle and it was not charged: the "
        "cited eval-cost payment stays spendable. Resubmit when the family "
        "reopens; the arena manifest lists closed targets.",
    ),
    "finalized_block_sla_expired": (
        "The submission left the queue on its service-level window without "
        "reaching a verdict.",
        "This is NOT a judgement on the bundle. Nothing about it was measured "
        "and it may be resubmitted.",
    ),
    "remote_qualification_hold": (
        "Evaluation stopped on validator-side infrastructure, not on anything "
        "the bundle did.",
        "This is not attributed to the candidate and is not a FAIL. It may be "
        "resubmitted.",
    ),
}


def _guidance(code: object) -> dict[str, str] | None:
    if not isinstance(code, str) or not code:
        return None
    key = code if code in _GUIDANCE else code.partition(":")[0]
    entry = _GUIDANCE.get(key)
    if entry is None:
        return None
    return {"cause": entry[0], "next_step": entry[1]}


def _compile_defect(publication_root: object) -> str | None:
    """Recompute the static compile finding for a retained bundle.

    This is the closest thing to a traceback that can be handed back without
    re-running anything: the same tracked scan the ``scan`` command applies,
    replayed over the retained tree so the miner gets file, line, and kernel.

    Advisory by construction -- it reports a kernel that cannot be traced, not
    proof that this kernel is the one the verdict rested on.
    """

    if not isinstance(publication_root, str) or not publication_root:
        return None
    from cacheon.sandbox import scan_compilability

    root = Path(publication_root)
    if not root.is_dir():
        return None
    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_text(errors="ignore")
        except OSError:
            continue
        for violation in scan_compilability(source, filename=path.name).violations:
            if "syntax error" not in violation:
                return violation
    return None


def _submission(db: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    reason = _reason(row["reason"])
    record: dict[str, Any] = {
        "content_hash": row["content_hash"],
        "reservation_id": row["reservation_id"],
        "submitted_at_block": row["block"],
        "target_id": row["target_id"] or None,
        "status": row["status"],
        "decision": row["decision"] or None,
        "attribution": _attribution(row),
        "reason": reason,
        "guidance": _guidance(reason.get("code")),
        "screens": _screen_dispositions(db, row["reservation_id"]),
        "qualification_dispositions": _qualification_dispositions(
            db, row["reservation_id"]
        ),
    }
    if reason.get("code") == "candidate_kernel_does_not_compile":
        record["static_finding"] = _compile_defect(row["publication_root"])
    return record


def miner_submissions(
    intake_db: str | Path,
    *,
    hotkey: str,
) -> dict[str, Any]:
    """Return every retained submission for one hotkey, oldest first.

    Unlike the single-reservation operator view this does not refuse a hotkey
    with several rows -- that is the normal case for a real miner and the whole
    point of the report.
    """

    if (
        not isinstance(hotkey, str)
        or not hotkey
        or len(hotkey) > 256
        or hotkey.strip() != hotkey
        or any(ord(character) < 32 for character in hotkey)
    ):
        raise OperatorStatusError("hotkey selector is malformed")

    db = _readonly_connection(intake_db)
    try:
        db.execute("BEGIN")
        rows = db.execute(
            "SELECT * FROM reservations WHERE hotkey=? ORDER BY "
            "block,event_index,event_subindex,content_hash",
            (hotkey,),
        ).fetchall()
        return {
            "schema": SCHEMA,
            "hotkey": hotkey,
            "cursor": _cursor(db),
            "submission_count": len(rows),
            "submissions": [_submission(db, row) for row in rows],
        }
    except OperatorStatusError:
        raise
    except sqlite3.Error as exc:
        raise OperatorStatusError(f"miner submission query failed: {exc}") from None
    finally:
        if db.in_transaction:
            db.execute("ROLLBACK")
        db.close()


def format_miner_submissions(value: dict[str, Any]) -> str:
    """Render the report as text a miner can act on without reading JSON."""

    lines = [
        f"hotkey: {value['hotkey']}",
        f"submissions: {value['submission_count']}",
    ]
    cursor = value.get("cursor")
    if isinstance(cursor, dict):
        lines.append(f"validator has read the chain through block {cursor['block']}")
    for record in value["submissions"]:
        attribution = record["attribution"]
        reason = record["reason"]
        lines.append("")
        lines.append(
            f"[block {record['submitted_at_block']}] {record['content_hash'][:16]} "
            f"target={record['target_id'] or '-'}"
        )
        lines.append(
            f"  outcome: {record['decision'] or record['status']} "
            f"({attribution['class']})  reason: {reason['code'] or '-'}"
        )
        guidance = record.get("guidance")
        if guidance is not None:
            lines.append(f"  what happened: {guidance['cause']}")
            lines.append(f"  what to do:    {guidance['next_step']}")
        finding = record.get("static_finding")
        if finding:
            lines.append(f"  static finding: {finding}")
        for screen in record["screens"]:
            failed = [
                stage["stage"]
                for stage in screen["stages"]
                if stage["grade"] != "pass"
            ]
            if failed:
                lines.append(
                    f"  screen[{screen['attempt_index']}] {screen['decision']}: "
                    f"failed at {', '.join(failed)}"
                )
    return "\n".join(lines)

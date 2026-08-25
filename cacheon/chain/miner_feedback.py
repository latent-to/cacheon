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

import base64
import hashlib
import json
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
    # The three below were persisted to real miners' rows before anything here
    # explained them, so those miners were shown a bare code. Found by listing
    # every reason mainnet has actually produced and diffing against this table,
    # which is what ``tests/test_miner_feedback.py`` now does on every run.
    "missing_eval_cost_payment": (
        "No spendable evaluation-cost payment was matched to this submission, "
        "so it was never queued for evaluation.",
        "This is NOT a judgement on the bundle; nothing about it was measured. "
        "Each evaluation needs its own payment bound to the submission.",
    ),
    "screen_receipt_service_rotated": (
        "The arena service identity changed between the screen and its use, so "
        "the screen receipt no longer described the running service.",
        "This is validator-side and not attributed to the bundle. It is "
        "re-screened against the current service rather than failed.",
    ),
    "screen_promoted": (
        "Every non-crown screen passed and the submission is waiting to enter "
        "qualification.",
        "No action is needed. This is a queue position, not a verdict.",
    ),
}

_GUIDANCE.update(
    {
        "candidate_exception": (
            "A typed rank receipt proves the selected candidate raised.",
            "Read the attached failure product and request worker log.",
        ),
        "candidate_never_executed": (
            "Every expected rank loaded the slot, but none invoked the candidate.",
            "Fix the routing domain or entry binding before resubmitting.",
        ),
        "fetch": (
            "The validator could not fetch or materialize the archive.",
            "Make the URL readable and remove excluded host-metadata members.",
        ),
        "manifest": (
            "The fetched manifest or metadata violated the target contract.",
            "Fix the named field and validate the bundle locally.",
        ),
        "systemic_release_cap": (
            "Repeated validator-side releases reached the configured cap.",
            "The validator must close the repeated fault before another evaluation.",
        ),
    }
)


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


def _execution_artifacts(
    evidence_roots: tuple[Path, ...], publication_digest: object
) -> list[dict[str, Any]]:
    """Every published record of what the ranks did with this bundle.

    The durable row points at the attempt artifact only; the execution rows are
    a separate unsealed artifact the run published beside it, keyed by the
    bundle they describe. The store is content-addressed, so each file is
    authenticated against its own name before it is believed.
    """

    from cacheon.eval.b300_resident_qualification import EXECUTION_EVIDENCE_DOMAIN
    from cacheon.eval.evidence_store import DEFAULT_MAX_EVIDENCE_BYTES

    found: list[dict[str, Any]] = []
    if not isinstance(publication_digest, str) or not publication_digest:
        return found
    for root in evidence_roots:
        domain = Path(root) / EXECUTION_EVIDENCE_DOMAIN
        for path in sorted(domain.glob("??/*")) if domain.is_dir() else ():
            try:
                raw = path.read_bytes()
                if (
                    len(raw) > DEFAULT_MAX_EVIDENCE_BYTES
                    or hashlib.sha256(raw).hexdigest() != path.name
                    or json.loads(raw).get("bundle_digest") != publication_digest
                ):
                    continue
            except (OSError, ValueError, AttributeError):
                continue
            found.append(
                {
                    "reference": {"domain": EXECUTION_EVIDENCE_DOMAIN},
                    "payload_base64": base64.b64encode(raw).decode("ascii"),
                }
            )
    return found


def _attempt_evidence(
    dispositions: list[dict[str, Any]],
    evidence_roots: tuple[Path, ...],
    publication_digest: object,
) -> list[dict[str, Any]]:
    """Reopen each attempt's retained evidence and render what it measured.

    The durable row keeps the verdict and a digest; the bytes behind that digest
    live in a content-addressed store keyed by the worker generation that ran, so
    an attempt's evidence sits in whichever store was live at the time. Reading
    it back is the difference between telling a miner "FAIL, candidate_slower"
    and telling them which shapes passed, how many replays, by how much they
    lost, and whether their kernel was even running when it lost.

    Reports what it could not reopen instead of omitting it: a missing artifact
    is an operator problem the miner should be able to see, not a silent gap.
    """

    from cacheon.eval.evidence_store import (
        EvidenceArtifactRef,
        reopen_evidence_anywhere,
    )
    from cacheon.eval.candidate_failure_product import (
        CANDIDATE_FAILURE_SCHEMA,
        validate_candidate_failure,
    )
    from cacheon.eval.explain import candidate_failure_lines, explain

    execution = _execution_artifacts(evidence_roots, publication_digest)
    reports: list[dict[str, Any]] = []
    for row in dispositions:
        reference = row.get("attempt_ref")
        if not isinstance(reference, dict):
            continue
        try:
            typed = EvidenceArtifactRef(**reference)
            payload = reopen_evidence_anywhere(evidence_roots, typed)
        except Exception:  # noqa: BLE001 - reporting must survive a bad artifact
            payload = None
        entry: dict[str, Any] = {"attempt_index": row.get("attempt_index")}
        if payload is None:
            entry["retained"] = False
            entry["note"] = (
                "the evidence for this attempt was not found in any configured "
                "store; the verdict above stands on the durable record"
            )
        else:
            entry["retained"] = True
            if typed.schema == CANDIDATE_FAILURE_SCHEMA:
                try:
                    failure = validate_candidate_failure(json.loads(payload))
                except (TypeError, ValueError) as exc:
                    entry["explanation"] = [
                        f"candidate failure artifact is unreadable: {exc}"
                    ]
                else:
                    entry["explanation"] = candidate_failure_lines(failure)
            else:
                entry["explanation"] = explain(
                    {
                        "evidence": [
                            {
                                "reference": {"domain": typed.domain},
                                "payload_base64": base64.b64encode(payload).decode("ascii"),
                            },
                            *execution,
                        ]
                    }
                )
        reports.append(entry)
    return reports


def _lease_history(
    db: sqlite3.Connection, reservation_id: str
) -> tuple[list[dict[str, Any]], set[str]]:
    """Read the existing immutable lease/recovery journals for one reservation."""

    names = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'evaluation_%'"
        )
    }
    if not {"evaluation_leases", "evaluation_lease_members"} <= names:
        return [], set()
    histories: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    leases = db.execute(
        "SELECT el.* FROM evaluation_leases AS el JOIN evaluation_lease_members AS em "
        "ON em.lease_id=el.lease_id WHERE em.reservation_id=? ORDER BY el.generation",
        (reservation_id,),
    )
    for lease in leases:
        record = {key: lease[key] for key in lease.keys()}
        record["events"] = (
            [
                {key: event[key] for key in event.keys()}
                for event in db.execute(
                    "SELECT * FROM evaluation_lease_events WHERE lease_id=? ORDER BY sequence",
                    (lease["lease_id"],),
                )
            ]
            if "evaluation_lease_events" in names
            else []
        )
        recoveries = (
            [
                {key: event[key] for key in event.keys()}
                for event in db.execute(
                    "SELECT * FROM evaluation_recovery_events WHERE lease_id=? ORDER BY sequence",
                    (lease["lease_id"],),
                )
            ]
            if "evaluation_recovery_events" in names
            else []
        )
        record["recovery_events"] = recoveries
        request_ids.update(
            event["request_id"] for event in recoveries if event.get("request_id")
        )
        histories.append(record)
    return histories, request_ids


def _submission(
    db: sqlite3.Connection,
    row: sqlite3.Row,
    evidence_roots: tuple[Path, ...] = (),
    spool_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    reason = _reason(row["reason"])
    qualifications = _qualification_dispositions(db, row["reservation_id"])
    for disposition in qualifications:
        detail = disposition.get("reason")
        disposition["guidance"] = _guidance(
            detail.get("code") if isinstance(detail, dict) else None
        )
    leases, request_ids = _lease_history(db, row["reservation_id"])
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
        "qualification_dispositions": qualifications,
        "evaluation_leases": leases,
    }
    if evidence_roots:
        record["attempt_evidence"] = _attempt_evidence(
            record["qualification_dispositions"], evidence_roots, row["publication_digest"]
        )
    if reason.get("code") == "candidate_kernel_does_not_compile":
        record["static_finding"] = _compile_defect(row["publication_root"])
    if spool_roots:
        from cacheon.eval.remote_run_forensics import remote_runs

        record["remote_runs"] = remote_runs(
            spool_roots, row["reservation_id"], request_ids
        )
    return record


def miner_submissions(
    intake_db: str | Path,
    *,
    hotkey: str,
    evidence_roots: tuple[Path, ...] = (),
    spool_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Return every retained submission for one hotkey, oldest first.

    Unlike the single-reservation operator view this does not refuse a hotkey
    with several rows -- that is the normal case for a real miner and the whole
    point of the report.

    ``evidence_roots`` are the content-addressed stores the validator kept; when
    given, each qualification attempt's retained evidence is reopened and
    rendered under ``attempt_evidence``.
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
            "submissions": [
                _submission(db, row, evidence_roots, spool_roots) for row in rows
            ],
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
                stage for stage in screen["stages"] if stage["grade"] != "pass"
            ]
            if failed:
                lines.append(
                    f"  screen[{screen['attempt_index']}] {screen['decision']}: "
                    f"failed at {', '.join(stage['stage'] for stage in failed)}"
                )
            for stage in failed:
                if stage.get("reason"):
                    lines.append(f"    {stage['stage']}: {stage['reason']}")
        for attempt in record.get("attempt_evidence") or ():
            lines.append(f"  attempt[{attempt['attempt_index']}] evidence:")
            if attempt["retained"]:
                lines.extend(f"    {line}" for line in attempt["explanation"])
            else:
                lines.append(f"    {attempt['note']}")
        for lease in record.get("evaluation_leases") or ():
            lines.append(
                f"  lease[{lease['generation']}] {lease['stage']} {lease['state']} "
                f"blocks={lease['claimed_block']}..{lease['expires_block']}"
            )
            for event in (*lease.get("events", ()), *lease.get("recovery_events", ())):
                lines.append(
                    f"    {event['event_type']} @block {event['finalized_block']}"
                    + (f": {event['reason']}" if event.get("reason") else "")
                )
        for run in record.get("remote_runs") or ():
            lines.append(
                f"  remote request {run['request_id'][:16]}: "
                f"{run.get('result_state') or 'no result'}"
                + (f" ({run['failure_code']})" if run.get("failure_code") else "")
            )
            worker_log = run.get("worker_log")
            if worker_log:
                lines.append(
                    f"    retained worker log: {worker_log['size']} bytes, "
                    f"sha256 {worker_log['sha256']}"
                )
                lines.extend(f"    {line}" for line in worker_log["explanation"])
            elif run.get("worker_log_error"):
                lines.append(f"    worker log unreadable: {run['worker_log_error']}")
            elif run.get("events"):
                last = run["events"][-1]
                lines.append(f"    last transport event: {last.get('event')}")
    return "\n".join(lines)

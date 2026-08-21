"""Generate the miner-facing evaluation report from durable evidence.

Reads the intake database and the content-addressed qualification evidence,
re-runs the deployed speed grader over every resolvable stage-exit artifact,
and emits one Markdown table row per evaluation attempt. Every number in the
output comes from the grader executing over verified artifact bytes — nothing
is transcribed by hand. Rows whose evidence cannot be resolved say so; that is
itself a finding, never a blank.

Usage (run where the intake DB lives; PYTHONPATH must carry the cacheon tree):

    python scripts/evaluation_report.py \
        --intake-db /path/to/intake.sqlite3 \
        --evidence-root /path/to/qualification-evidence [--evidence-root ...] \
        --current-arena <sha256> \
        --out docs/results/evaluations.md
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
import statistics
import sys
from pathlib import Path

from cacheon.eval.crossover_runtime import ResidentSpeedPolicy
from cacheon.eval.speed_verdict import speed_grade

STAGE_EXIT_DOMAIN = "qualification.stage-exit"

# Reservation reasons that describe validator/infrastructure state, not a
# judgment about the candidate kernel.
INFRA_REASONS = frozenset(
    {
        "finalized_block_sla_expired",
        "validator_worker_unavailable",
        "qualification_runner",
    }
)

# Dated context for evaluations under superseded arenas. Keyed by target; the
# taint applies only to rows whose arena differs from the current one. Sources:
# docs/reference/state-of-record.md (workload mismatch note) and the deployed
# manifest's closed_targets.
SUPERSEDED_TARGET_NOTES = {
    "attention.msa_prefill_block_score": (
        "superseded arena, disarmed-execution-guard era: no evidence the "
        "candidate kernel was in the measured path, and the advertised "
        "long-prefill surface was not the workload executed; the numbers "
        "describe engine-vs-engine variance, not the kernel"
    ),
    "attention.msa_block_score": (
        "superseded arena, disarmed-execution-guard era; see the prefill "
        "sibling's note"
    ),
}


class _Window:
    def __init__(self, row: dict) -> None:
        self.tokens = float(row["tokens"])
        self.seconds = float(row["seconds"])


class _Read:
    """Adapter giving a witness rate row the attribute shape the deployed
    grader reads. Field values pass through unmodified."""

    def __init__(self, row: dict) -> None:
        self.role = row["role"]
        self.lane_digest = row.get("lane_digest")
        self.windows = tuple(_Window(w) for w in row["windows"])
        for field in (
            "conditioning_tokens",
            "conditioning_seconds",
            "timed_tokens",
            "timed_seconds",
        ):
            setattr(self, field, float(row[field]))


_POLICY_FLOAT_FIELDS = frozenset(
    {
        "min_margin",
        "noise_multiplier",
        "max_noise",
        "max_window_scatter",
        "max_conditioning_slowdown",
    }
)


def _policy_from_block(block: dict) -> ResidentSpeedPolicy:
    names = {field.name for field in dataclasses.fields(ResidentSpeedPolicy)}
    missing = names - set(block)
    if missing:
        raise ValueError(f"policy block lacks {sorted(missing)}")
    # Witness artifacts serialize the float thresholds as decimal strings to
    # preserve exact digits; the policy constructor wants numbers back.
    return ResidentSpeedPolicy(
        **{
            name: float(block[name]) if name in _POLICY_FLOAT_FIELDS else block[name]
            for name in names
        }
    )


def _resolve_artifact(sha: str, roots: list[Path]) -> bytes | None:
    for root in roots:
        for candidate in (
            root / STAGE_EXIT_DOMAIN / sha[:2] / sha,
            root / STAGE_EXIT_DOMAIN / sha,
        ):
            if candidate.is_file():
                raw = candidate.read_bytes()
                if hashlib.sha256(raw).hexdigest() != sha:
                    raise ValueError(f"artifact bytes differ from name: {candidate}")
                return raw
    return None


def _regrade(raw: bytes) -> dict:
    """Run the deployed grader over one stage-exit artifact and report its own
    numbers. Raises on any shape this artifact era does not satisfy."""

    exit_row = json.loads(raw)
    witness = exit_row.get("speed_witness")
    if not witness:
        raise ValueError("stage exit carries no speed measurement")
    policy = _policy_from_block(witness["resident_policy"])
    reads = [_Read(row) for row in witness["rates"]]
    # Arm assignment goes by LANE DIGEST, the witness's own ground truth, with
    # the role vocabulary only as a fallback. In the resident bracket protocol
    # B and B_prime are the BASELINE bookends and C is the CANDIDATE (verified
    # against the FE artifact: role B/B_prime rows carry baseline_lane_digest,
    # role C carries candidate_lane_digest). An earlier revision reversed this
    # and published inverted speedups; deriving from digests prevents a repeat.
    candidate_lane = witness.get("candidate_lane_digest")
    baseline_lane = witness.get("baseline_lane_digest")
    if candidate_lane and baseline_lane and all(r.lane_digest for r in reads):
        baselines = [r for r in reads if r.lane_digest == baseline_lane]
        candidates = [r for r in reads if r.lane_digest == candidate_lane]
    else:
        baselines = [r for r in reads if r.role.startswith("B")]
        candidates = [r for r in reads if r.role.startswith("C")]
    if not baselines or not candidates:
        raise ValueError("witness lacks a baseline or candidate read")
    verdict, decision = speed_grade(policy, baselines, candidates, concluding=True)
    medians = {
        read.role: statistics.median(w.tokens / w.seconds for w in read.windows)
        for read in reads
    }
    return {
        "artifact_decision": exit_row.get("decision"),
        "artifact_reason": exit_row.get("reason"),
        "grader_decision": getattr(decision, "value", str(decision)),
        "grader_speedup": verdict.speedup,
        "grader_required": verdict.required,
        "grader_noise": verdict.noise,
        "grader_detail": verdict.detail,
        "grader_confident": verdict.confident,
        "medians": medians,
        "baseline_roles": tuple(r.role for r in baselines),
        "candidate_roles": tuple(r.role for r in candidates),
        "windows": {read.role: len(read.windows) for read in reads},
        "policy_version": policy.version,
    }


# Individually audited false positives, cited to their evidence. 87982705 and
# 3558c980: per-rank audit receipts read aot_loaded:0 / aot_invoked:0 on all
# four TP ranks — the candidate kernel never executed; the measured edge is the
# documented ~1.6% asymmetric-swap bias (2026-08-16 ledger, "inert-pass
# class"). c9250862: the moe_finalize crown sits inside the v6 inert envelope
# (chainops/GEOMEAN_CROSSCHECK_2026-08-19.md, finding 3).
AUDITED_FALSE_POSITIVES = {
    "87982705": "audited false positive: candidate kernel never loaded (aot_invoked:0 on all ranks); edge = asymmetric-swap bias",
    "3558c980": "audited false positive: inert-pass class, candidate ran stock code on swap bias (2026-08-16 audit)",
    "c9250862": "audited: PASS sits inside the v6 inert envelope (2026-08-19 crown audit); not citable as a kernel win",
}


def _validity(row: dict, current_arena: str | None) -> str:
    override = AUDITED_FALSE_POSITIVES.get((row["reservation"] or "")[:8])
    if override:
        return f"tainted: {override}"
    if row["reason"] in INFRA_REASONS:
        return "infrastructure — not a kernel verdict"
    arena = row["arena"] or ""
    if current_arena and arena and not arena.startswith(current_arena):
        note = SUPERSEDED_TARGET_NOTES.get(row["target"])
        if note:
            return f"tainted: {note}"
        if (row["attempt_decision"] or row["decision"]) in ("PASS", "qualified"):
            return (
                "superseded arena — PASS from the asymmetric-swap era (only the"
                " candidate lane swapped engines; documented ~1.6% swap bias);"
                " not citable as a kernel win"
            )
        return "superseded arena"
    return "current contract"


def collect_rows(db_path: Path, roots: list[Path], current_arena: str | None) -> list[dict]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            select r.reservation_id, r.hotkey, r.target_id, r.block,
                   r.arena_service_digest, r.decision, r.reason,
                   d.attempt_index, d.decision as attempt_decision,
                   d.reason as attempt_reason, d.attempt_ref_json
            from reservations r
            left join qualification_dispositions d
              on d.reservation_id = r.reservation_id
            where r.decision is not null or d.reservation_id is not null
            order by r.block desc, d.attempt_index asc
            """
        ).fetchall()
    finally:
        connection.close()

    rows = []
    for record in cursor:
        row = {
            "reservation": record["reservation_id"],
            "hotkey": record["hotkey"],
            "target": record["target_id"],
            "block": record["block"],
            "arena": record["arena_service_digest"],
            "decision": record["decision"],
            "reason": record["reason"],
            "attempt_index": record["attempt_index"],
            "attempt_decision": record["attempt_decision"],
            "attempt_reason": record["attempt_reason"],
            "regrade": None,
            "regrade_error": None,
        }
        ref_json = record["attempt_ref_json"]
        if ref_json:
            try:
                ref = json.loads(ref_json)
                if ref.get("domain") == STAGE_EXIT_DOMAIN:
                    raw = _resolve_artifact(ref["sha256"], roots)
                    if raw is None:
                        row["regrade_error"] = "evidence artifact not found"
                    else:
                        row["regrade"] = _regrade(raw)
            except Exception as exc:  # degrade per-row, never lose the table
                row["regrade_error"] = f"{type(exc).__name__}: {exc}"
        row["validity"] = _validity(row, current_arena)
        rows.append(row)
    return rows


def render(rows: list[dict], *, db_path: str, current_arena: str | None) -> str:
    lines = [
        "# Evaluation report",
        "",
        "Machine-generated by `scripts/evaluation_report.py`; every rate and",
        "speedup below is the deployed grader executing over the verified",
        "stage-exit artifact. Do not edit by hand — regenerate.",
        "",
        f"- Intake database: `{db_path}`",
        f"- Current arena: `{current_arena or 'unspecified'}`",
        f"- Rows: {len(rows)}",
        "",
        "| Reservation | Hotkey | Target | Decision | Reason | Regrade |"
        " Grader speedup | Required | Candidate/Stock (tok/s) | Validity |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        regrade = row["regrade"]
        if regrade:
            speed = f"{regrade['grader_speedup']:.5f}"
            required = f"{regrade['grader_required']:.5f}"
            med = regrade["medians"]
            stock = min(med[role] for role in regrade["baseline_roles"])
            cand = max(med[role] for role in regrade["candidate_roles"])
            rates = f"{cand:.1f} / {stock:.1f}"
            regrade_says = regrade["grader_decision"]
        else:
            speed = row["regrade_error"] or "no attempt evidence"
            rates = "—"
            required = "—"
            regrade_says = "—"
        lines.append(
            "| {reservation} | {hotkey} | {target} | {decision} | {reason} |"
            " {regrade_says} | {speed} | {required} | {rates} | {validity} |".format(
                regrade_says=regrade_says,
                reservation=(row["reservation"] or "")[:8],
                hotkey=(row["hotkey"] or "")[:8],
                target=row["target"] or "—",
                decision=row["attempt_decision"] or row["decision"] or "—",
                reason=row["attempt_reason"] or row["reason"] or "—",
                speed=speed,
                required=required,
                rates=rates,
                validity=row["validity"],
            )
        )
    lines += [
        "",
        "Legend: *Grader speedup* and *Required* come from the deployed",
        "`speed_grade` re-run; *Candidate/Stock* are per-read window medians",
        "from the same artifact. A tainted row is a validator-caused workload",
        "mismatch, not evidence about the kernel's intended surface.",
        "",
    ]
    return "\n".join(lines)


def render_detail(row: dict) -> str:
    """One reservation, explained the way an operator would answer 'why did
    this miner fail' — recorded verdict, the grader re-run, every number."""

    lines = [
        f"## Reservation {row['reservation']}",
        "",
        f"- Hotkey: `{row['hotkey']}`  |  Target: `{row['target']}`  |  Block: {row['block']}",
        f"- Recorded: **{row['attempt_decision'] or row['decision']}**"
        f" ({row['attempt_reason'] or row['reason']})",
        f"- Validity: {row['validity']}",
    ]
    regrade = row["regrade"]
    if not regrade:
        lines.append(f"- Evidence: {row['regrade_error'] or 'no attempt evidence'}")
        return "\n".join(lines) + "\n"
    lines += [
        f"- Deployed grader re-run: **{regrade['grader_decision']}** — speedup"
        f" {regrade['grader_speedup']:.5f} vs required {regrade['grader_required']:.5f}"
        f" (noise {regrade['grader_noise']:.5f}, policy v{regrade['policy_version']},"
        f" confident={regrade['grader_confident']})",
        f"- Grader detail: {regrade['grader_detail'] or '—'}",
        "- Per-read window medians (tokens/second):",
    ]
    for role, median in sorted(regrade["medians"].items()):
        arm = "candidate" if role in regrade["candidate_roles"] else "baseline"
        lines.append(
            f"    - {role} ({arm}): {median:.3f} over {regrade['windows'][role]} windows"
        )
    if regrade["artifact_decision"] != regrade["grader_decision"]:
        lines.append(
            "- NOTE: the artifact's recorded decision"
            f" ({regrade['artifact_decision']}) differs from the deployed"
            " grader's re-run; the era's verdict used additional gates or a"
            " different verdict layer — investigate before treating either as"
            " final."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-db", required=True, type=Path)
    parser.add_argument("--evidence-root", action="append", default=[], type=Path)
    parser.add_argument("--current-arena", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--reservation",
        default=None,
        help="render a full per-reservation explanation instead of the table",
    )
    args = parser.parse_args(argv)
    rows = collect_rows(args.intake_db, args.evidence_root, args.current_arena)
    if args.reservation:
        matches = [
            row for row in rows if (row["reservation"] or "").startswith(args.reservation)
        ]
        if not matches:
            print(f"no reservation matches {args.reservation!r}")
            return 1
        report = "\n".join(render_detail(row) for row in matches)
    else:
        report = render(rows, db_path=str(args.intake_db), current_arena=args.current_arena)
    if args.out:
        args.out.write_text(report)
        print(f"REPORT_WRITTEN={args.out} ROWS={len(rows)}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

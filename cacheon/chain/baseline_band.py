"""Retained qualification speed measurements, read back for the dashboard.

Every graded qualification half leaves a stage-exit artifact whose speed
witness records what each resident lane measured. This module reopens that
artifact from whichever local evidence store retains it and reports the same
warm-turn reads and verdict the replay scorer consumed, without re-running a
grader. A retained batch-cell witness (speed policies 8-15, whose grader is
retired) is not rendered: its raw lane ratio was never the credited gain.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    reopen_evidence_anywhere,
)

def qualification_evidence_roots(
    state_dir: Path, extra: tuple[Path, ...] = (), connection: Any = None,
    *, stage_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Every local store that may retain a submission's stage-exit artifact.

    Evidence roots rotate per worker generation, so one submission's artifact
    sits in whichever store was live when it graded.
    """

    try:
        rotated = sorted(state_dir.glob("qualification-evidence-*"), reverse=True)
    except OSError:
        rotated = []
    recorded = [] if connection is None else [
        Path(row[0]) for row in connection.execute(
            "SELECT DISTINCT evidence_root FROM settlement_qualifications "
            "WHERE evidence_root != ''")
    ]
    staged = [] if stage_dir is None else sorted(
        stage_dir.glob("*/monday-config/qualification-evidence"), reverse=True)
    return tuple(dict.fromkeys(
        rotated + recorded + staged + [root for root in extra if root.is_dir()]))


def qualification_speed(
    attempt_ref_json: object, roots: tuple[Path, ...], target_id: str = ""
) -> dict[str, Any] | None:
    """Measured lane rates from a graded attempt's stage-exit artifact.

    ``None`` means the reference is absent or no local store retains the
    artifact — normal for rotated stores, never an error to the caller.
    """

    if not attempt_ref_json or not isinstance(attempt_ref_json, (str, bytes)):
        return None
    try:
        reference = EvidenceArtifactRef.from_dict(json.loads(attempt_ref_json))
    except (TypeError, ValueError, EvidenceStoreError):
        return None
    try:
        payload = reopen_evidence_anywhere(roots, reference)
    except EvidenceStoreError:
        return None
    if payload is None:
        return None
    return qualification_speed_from_payload(payload, target_id)


def qualification_speed_from_payload(
    payload: bytes, target_id: str = ""
) -> dict[str, Any] | None:
    """Read a replay attempt's measurements from a local artifact or retained remote result.

    A witness without a replay read set, the retired batch-cell policies, reads
    as ``None``: the verdict row is its record.
    """
    try:
        result = json.loads(payload)
        if "reports" in result:
            reports = [report for report in result["reports"]
                       if not target_id or report.get("target_id") == target_id]
            if len(reports) != 1:
                return None
            result = reports[0]
        witness = result["speed_witness"]
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(witness, dict) or witness.get("goodput") is None:
        return None
    speed = replay_measurements(witness)
    if audit := result.get("audit_witness"):
        speed["audit"] = {key: audit[key] for key in ("decision", "detail")}
    return speed


def replay_measurements(witness: dict[str, Any]) -> dict[str, Any]:
    """Report the same retained warm-turn measurements and verdict the replay scorer consumes."""
    from dataclasses import asdict
    from cacheon.eval.qualification_runner import ResidentSpeedWitness
    from cacheon.eval.service_capacity import attainment, fixed_work_rate

    speed: dict[str, Any] = {"metric": "fixed_work_rate", "lanes": []}
    try:
        retained = ResidentSpeedWitness.from_dict(witness)
        reads, policy = retained.goodput, retained.resident_policy.goodput
        grade = reads.grade(policy)
        lanes = []
        for role, windows in (("B", reads.incumbent), ("C", reads.candidate)):
            for index, read in enumerate(windows):
                work = fixed_work_rate(read, reads.work(read.load))
                # Whole boots: reference boots, then any retained bracket, per arm.
                lanes.append({"role": role, "window": read.window, "load": read.load,
                              "boot": index // (reads.window_limit * len(reads.loads)),
                              "warm_turns": work.turns, "elapsed_s": work.elapsed_s, "turns_per_second": work.rate,
                              "attainment": attainment(read, policy.contract),
                              "goodput_tps": _output_tokens(read) / work.elapsed_s,
                              **_replay_diagnostics(read)})
        speed.update({
            "lanes": lanes, "speedup": float(grade.settled_speedup),
            "policy_version": retained.resident_policy.version,
            "score_basis": "retained_reference_boots",
            "speed_stage_seconds": retained.completed_monotonic_s - retained.started_monotonic_s,
            "workload_digest": retained.workload_digest,
            "loads": list(reads.loads),
            "windows": len(reads.candidate if reads.boots else reads.incumbent) // len(reads.loads),
            "window_limit": reads.window_limit, "reference_boots": reads.reference_boots,
            "candidate_boots": reads.boots,
            "contract": asdict(policy.contract),
            "per_load": _per_load(reads, policy),
            "grading": {"steady_span": policy.steady_span or None,"decision": grade.decision.value, "detail": grade.verdict.detail,
                        "required_speedup": grade.verdict.required,
                        "null_noise": policy.null_noise,
                        "standard_error": grade.verdict.noise,
                        "lower_speedup": float(grade.settled_speedup) / grade.verdict.required,
                        "boot_noise": policy.boot_noise, "error_rate": policy.error_rate,
                        "attainment_tolerance": policy.attainment_tolerance,
                        "attainment_margin": policy.attainment_margin, "futility_margin": policy.futility_margin,
                        "attainment_pooled": bool(policy.attainment_pooled),
                        "payment_bar": float(reads.payment_bar) if reads.payment_bar else None,
                        # The first lifetime's continue-or-stop line: the sealed margin under the bar (stock when
                        # no bar is sealed), the same comparison reference_grade makes before a second boot.
                        "continue_at": ((float(reads.payment_bar) if policy.bar_futility and reads.payment_bar else 1.0)
                                        * (1 - policy.futility_margin) if policy.futility_margin else None)},
        })
    except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        speed["grading_error"] = str(exc)
    return speed


def _per_load(reads, policy) -> list[dict[str, Any]]:
    """Each load's stock and candidate cost as the scorer computes it, their ratio and decode medians.

    A boot's per-window cost under a sealed steady span is seconds per warm turn, not the whole fixed-work span the
    window table shows; on a GLM-5.3-Flash PD twin the two disagreed by up to 1.6 points at one load (2026-11-12)."""
    from cacheon.eval.service_capacity import decode_p50, fixed_work_rate, steady_span_cost

    if not reads.reference_boots or not reads.candidate:
        return []
    stock = reads.incumbent[:reads.reference_boots * reads.window_limit * len(reads.loads)]
    floors = policy.floors(reads.loads)

    def cost(read):
        work = reads.work(read.load)
        return (steady_span_cost(read, work, policy.steady_span) if policy.steady_span
                else fixed_work_rate(read, work).elapsed_s)

    rows = []
    for load in reads.loads:
        base, cand = ([read for read in arm if read.load == load] for arm in (stock, reads.candidate))
        logs = [sum(math.log(cost(read)) for read in arm) / len(arm) for arm in (base, cand)]
        row = {"load": load, "stock_cost": math.exp(logs[0]), "candidate_cost": math.exp(logs[1]),
               "gain": math.exp(logs[0] - logs[1]), "stock_decode_tps": decode_p50(base),
               "candidate_decode_tps": decode_p50(cand), "decode_floor_tps": floors.get(load)}
        for name, arm in (("stock", base), ("candidate", cand)):
            row[f"{name}_goodput_tps"] = sum(map(_output_tokens, arm)) / sum(
                fixed_work_rate(read, reads.work(load)).elapsed_s for read in arm)
            row[f"{name}_service"] = _service_goodput(arm, policy.contract)
        rows.append(row)
    return rows


def _output_tokens(read) -> int:
    """Output tokens served by a window's timed warm turns: goodput is these over the warm wall span."""
    cold = min(row.credit_issued_ns for row in read.records)
    return sum(row.output_tokens for row in read.records if row.credit_issued_ns != cold)


def _service_goodput(reads, contract) -> dict[str, float]:
    """Pool successful turns meeting both service limits over the full observed time of each window."""
    seconds = sum((max(r.request_end_ns for r in read.records) - min(r.credit_issued_ns for r in read.records))
                  / 1e9 for read in reads)
    qualified = [r for read in reads for r in read.records if r.meets(contract)]
    return {"turns_per_second": len(qualified) / seconds,
            "output_tps": sum(r.output_tokens for r in qualified) / seconds,
            "attainment": len(qualified) / sum(len(read.records) for read in reads)}


def _replay_diagnostics(read) -> dict[str, Any]:
    """Summarize retained warm requests without exposing prompts or changing their score."""
    cold = min(row.credit_issued_ns for row in read.records)
    warm = [row for row in read.records if row.credit_issued_ns != cold]
    first = sorted(row.ttft_s for row in warm)
    streamed = [row for row in warm if row.decode_tps is not None]
    return {
        "mean_ttft_s": statistics.fmean(first),
        "p95_ttft_s": first[(95 * len(first) - 1) // 100],
        # Pooled over the arm, not a median: on 2026-10-01 a +3.2% decode gain read -1.5% in one pass's median.
        "decode_tps": (sum(row.output_tokens - 1 for row in streamed) * 1e9
                       / sum(row.request_end_ns - row.first_token_ns for row in streamed) if streamed else None),
        "unsuccessful_turns": sum(row.status != "ok" for row in read.records),
    }


__all__ = [
    "qualification_evidence_roots",
    "qualification_speed",
    "replay_measurements",
]

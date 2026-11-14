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

    speed: dict[str, Any] = {"metric": "warm_turn_latency", "lanes": []}
    try:
        retained = ResidentSpeedWitness.from_dict(witness)
        reads, policy = retained.goodput, retained.resident_policy.goodput
        grade = reads.grade(policy)
        if policy.error_rate:
            speed["metric"] = "fixed_work_rate"
        lanes = []
        for role, windows in (("B", reads.incumbent), ("C", reads.candidate)):
            arm = []
            for index, read in enumerate(windows):
                work = fixed_work_rate(read, reads.work(read.load))
                elapsed = fixed_work_rate(read, reads.work(read.load), wall_time=True)
                # Whole boots: reference boots, then any retained bracket, per arm.
                arm.append({"role": role, "window": read.window, "load": read.load,
                            "physical_lane": read.lane, "boot": index // (reads.window_limit * len(reads.loads)),
                            "warm_turns": work.turns, "request_time_s": work.elapsed_s,
                            "mean_warm_latency_s": work.elapsed_s / work.turns,
                            "elapsed_s": elapsed.elapsed_s, "turns_per_second": elapsed.rate,
                            "attainment": attainment(read, policy.contract),
                            **_replay_diagnostics(read)})
            lanes.extend(arm)
        speed.update({
            "lanes": lanes, "speedup": float(grade.settled_speedup),
            "policy_version": retained.resident_policy.version,
            "score_basis": "retained_reference_boots",
            "speed_stage_seconds": retained.completed_monotonic_s - retained.started_monotonic_s,
            "workload_digest": retained.workload_digest,
            "loads": list(reads.loads),
            "windows": len(reads.candidate if reads.boots else reads.incumbent) // len(reads.loads),
            "window_limit": reads.window_limit, "reference_boots": reads.reference_boots,
            "contract": asdict(policy.contract),
            "per_load": _per_load(reads, policy),
            "grading": {"steady_span": policy.steady_span or None,"decision": grade.decision.value, "detail": grade.verdict.detail,
                        "required_speedup": grade.verdict.required,
                        "null_noise": policy.null_noise,
                        "standard_error": grade.verdict.noise if policy.error_rate else None,
                        "lower_speedup": (float(grade.settled_speedup) / grade.verdict.required
                                          if policy.error_rate else None),
                        "boot_noise": policy.boot_noise if policy.error_rate else None,
                        "error_rate": policy.error_rate or None,
                        "attainment_tolerance": policy.attainment_tolerance,
                        "attainment_margin": policy.attainment_margin, "futility_margin": policy.futility_margin,
                        "attainment_pooled": bool(policy.attainment_pooled)},
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
                else fixed_work_rate(read, work, wall_time=True).elapsed_s)

    rows = []
    for load in reads.loads:
        base, cand = ([read for read in arm if read.load == load] for arm in (stock, reads.candidate))
        logs = [sum(math.log(cost(read)) for read in arm) / len(arm) for arm in (base, cand)]
        rows.append({"load": load, "stock_cost": math.exp(logs[0]), "candidate_cost": math.exp(logs[1]),
                     "gain": math.exp(logs[0] - logs[1]), "stock_decode_tps": decode_p50(base),
                     "candidate_decode_tps": decode_p50(cand), "decode_floor_tps": floors.get(load)})
    return rows


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

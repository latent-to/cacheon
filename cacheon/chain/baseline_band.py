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
    return replay_measurements(witness)


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
        expected = {root: (main, inner) for root, main, inner in reads.expected}
        lanes = []
        for role, windows in (("B", reads.incumbent), ("C", reads.candidate)):
            arm = []
            for read in windows:
                work = fixed_work_rate(read, expected)
                elapsed = fixed_work_rate(read, expected, wall_time=True)
                arm.append({"role": role, "window": read.window, "load": read.load,
                            "physical_lane": read.lane,
                            "warm_turns": work.turns, "request_time_s": work.elapsed_s,
                            "mean_warm_latency_s": work.elapsed_s / work.turns,
                            "elapsed_s": elapsed.elapsed_s, "turns_per_second": elapsed.rate,
                            "attainment": attainment(read, policy.contract),
                            **_replay_diagnostics(read)})
            if retained.resident_policy.version == 16:
                best = min(row["mean_warm_latency_s"] for row in arm)
                for row in arm:
                    row["used_for_score"] = row["mean_warm_latency_s"] == best
            lanes.extend(arm)
        speed.update({
            "lanes": lanes, "speedup": float(grade.settled_speedup),
            "policy_version": retained.resident_policy.version,
            "score_basis": ("fastest_pass_latency" if retained.resident_policy.version == 16
                            else "pooled_elapsed_orientation"),
            "speed_stage_seconds": retained.completed_monotonic_s - retained.started_monotonic_s,
            "workload_digest": retained.workload_digest,
            "load": reads.incumbent[0].load, "windows": len(reads.incumbent), "window_limit": reads.window_limit,
            "contract": asdict(policy.contract),
            "grading": {"decision": grade.decision.value, "detail": grade.verdict.detail,
                        "required_speedup": grade.verdict.required,
                        "null_noise": policy.null_noise,
                        "standard_error": grade.verdict.noise if policy.error_rate else None,
                        "lower_speedup": (float(grade.settled_speedup) / grade.verdict.required
                                          if policy.error_rate else None),
                        "boot_noise": policy.boot_noise if policy.error_rate else None,
                        "error_rate": policy.error_rate or None,
                        "attainment_tolerance": policy.attainment_tolerance,
                        "attainment_margin": policy.attainment_margin, "futility_margin": policy.futility_margin},
        })
    except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        speed["grading_error"] = str(exc)
    return speed


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

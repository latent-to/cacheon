"""Retained qualification speed measurements, read back for the dashboard.

Every graded qualification half leaves a stage-exit artifact whose speed
witness records what each resident lane measured. This module reopens that
artifact from whichever local evidence store retains it and reports the lane
measurements without re-running a grader: replay witnesses report the same
warm-turn reads and verdict the replay scorer consumed, and retained
batch-cell witnesses (speed policies 8-15, whose grader is retired) report
their raw lane rates and per-cell delivery times.
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
    """Read measurements from either a local artifact or retained remote result."""
    try:
        result = json.loads(payload)
        if "reports" in result:
            reports = [report for report in result["reports"]
                       if not target_id or report.get("target_id") == target_id]
            if len(reports) != 1:
                return None
            result = reports[0]
        witness = result["speed_witness"]
        if isinstance(witness, dict) and witness.get("goodput") is not None:
            return _replay_measurements(witness)
        rates = witness["rates"]
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(rates, list):
        return None
    lanes: list[dict[str, Any]] = []
    by_role: dict[str, float] = {}
    for rate in rates:
        try:
            windows = [float(row["seconds"]) for row in rate["windows"]]
            timed_tokens = int(rate["timed_tokens"])
            timed_seconds = float(rate["timed_seconds"])
            conditioning = float(rate["conditioning_seconds"])
            cells = _phase_measurements(rate["windows"])
        except (TypeError, ValueError, KeyError):
            return None
        if not windows or timed_seconds <= 0:
            return None
        average = sum(windows) / len(windows)
        role = rate.get("role")
        prefill = isinstance(role, str) and "prefill" in role
        throughput = timed_tokens / timed_seconds
        lane = {
            "role": role,
            "tokens_per_second": None if prefill else round(throughput, 1),
            # A v12 prompt pass produces exactly one output token per request.
            "prompts_per_second": round(throughput, 6) if prefill else None,
            "timed_seconds": round(timed_seconds, 3),
            "cells": cells,
            "window_seconds": [round(seconds, 3) for seconds in windows],
            "window_scatter": round((max(windows) - min(windows)) / average, 4),
            "conditioning_ratio": round(conditioning / average, 4),
        }
        lanes.append(lane)
        by_role[role] = throughput
    speed: dict[str, Any] = {"lanes": lanes}
    baseline, candidate = by_role.get("B"), by_role.get("C")
    if baseline and candidate:
        speed["speedup"] = round(candidate / baseline, 4)
    if all(by_role.get(role) for role in ("B_prefill", "C_prefill", "B_prime_prefill")):
        policy = witness.get("resident_policy") or {}
        speed["prefill"] = {
            "speedup": by_role["C_prefill"] / max(
                by_role["B_prefill"], by_role["B_prime_prefill"]),
            "min_margin": policy.get("prefill_min_margin"),
        }
    return speed


def _replay_measurements(witness: dict[str, Any]) -> dict[str, Any]:
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
    decode = [row.decode_tps for row in warm if row.decode_tps is not None]
    return {
        "mean_ttft_s": statistics.fmean(first),
        "p95_ttft_s": first[(95 * len(first) - 1) // 100],
        "median_decode_tps": statistics.median(decode) if decode else None,
        "unsuccessful_turns": sum(row.status != "ok" for row in read.records),
    }


def _phase_measurements(windows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Recompute per-cell delivery metrics from retained host times, not cached cell summaries.

    Cells are keyed by input length, output length and concurrency, so unlike
    workloads never blend into one mean.
    """
    if not any(window.get("prompt_latencies") for window in windows):
        return []
    groups: dict[tuple[int, int, int], list[tuple[int, float, list[tuple[float, float]]]]] = {}
    for window in windows:
        latencies = [(float(first), float(last)) for first, last in window.get("prompt_latencies", ())]
        if not latencies:
            raise ValueError("phase read lacks a timed window's token latencies")
        tokens = int(window["tokens"])
        key = int(window["input_tokens"]), tokens // len(latencies), len(latencies)
        groups.setdefault(key, []).append((tokens, float(window["seconds"]), latencies))
    rows: list[dict[str, object]] = []
    for (input_tokens, output_tokens, concurrency), cells in sorted(groups.items()):
        pairs = [pair for _, _, latencies in cells for pair in latencies]
        rows.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "concurrency": concurrency,
            "timed_batches": len(cells),
            "mean_ttft_seconds": format(sum(first for first, _ in pairs) / len(pairs), ".17g"),
            "mean_tpot_seconds": format(
                sum(last - first for first, last in pairs) / (len(pairs) * (output_tokens - 1)), ".17g"),
            "end_to_end_output_tokens_per_second": format(
                sum(tokens for tokens, _, _ in cells) / sum(seconds for _, seconds, _ in cells), ".17g"),
        })
    return rows


__all__ = [
    "qualification_evidence_roots",
    "qualification_speed",
]

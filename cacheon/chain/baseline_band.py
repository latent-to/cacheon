"""Retained lane rates and the arena baseline band behind a re-measurement.

Every graded qualification half leaves a stage-exit artifact whose speed
witness records the tokens per second each resident lane produced. The
verdict itself is a ratio, so a baseline lane that boots into the slow engine
state (roughly ten percent under its normal rate, seen on every arena since
the champion baseline began) inflates the candidate's speedup without any
lane misbehaving. The two-PASS minimum absorbs one slow half; when both halves
draw it, the retained pair credits a gain the kernel never produced.

This module reads those rates back from retained evidence and states, for one
retained PASS pair, whether the half that set its credited speedup read the
baseline lane below the arena's band. The band is the arena's own retained
baseline population, not a tuned constant: the median of every baseline-role
read across retained halves in the same arena, with a fixed tolerance.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    reopen_evidence_anywhere,
)

BASELINE_ROLES = frozenset({"B", "B_prime", "B_double_prime"})
BAND_TOLERANCE = Decimal("0.05")
MIN_BASELINE_READS = 6


class BaselineBandError(ValueError):
    pass


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
        prefill = role in {"B_prefill", "C_prefill", "B_prime_prefill"}
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
    if "evidence_digest" in witness:
        from cacheon.eval.qualification_runner import ResidentSpeedWitness
        from cacheon.eval.resident_schedule import grade_schedule

        try:
            retained = ResidentSpeedWitness.from_dict(witness)
            grade = grade_schedule(retained.resident_policy, retained.rates)
            speed["grading"] = {
                "decision": grade.decision.value,
                "detail": grade.verdict.detail,
                "candidate_vs_before": candidate / baseline,
                "candidate_vs_after": candidate / by_role["B_prime"],
                "required_speedup": grade.verdict.required,
                "min_margin": retained.resident_policy.min_margin,
                "baseline_drift": grade.verdict.noise,
                "max_noise": retained.resident_policy.max_noise,
                "measurement_valid": grade.verdict.confident,
                "conditioning_failed": grade.conditioning_failed,
            }
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            speed["grading_error"] = str(exc)
    return speed


def _phase_measurements(windows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Recompute delivery metrics from retained host times, not cached cell summaries."""
    if not any(window.get("prompt_latencies") for window in windows):
        return []
    from cacheon.eval.resident_measurement import TimedWindow, phase_cells

    return phase_cells(tuple(
        TimedWindow(
            window["batch_index"], window["tokens"], float(window["seconds"]),
            window.get("input_tokens"),
            tuple(tuple(float(t) for t in pair)
                  for pair in window.get("prompt_latencies", ())),
        )
        for window in windows
    ))


@dataclass(frozen=True)
class RetainedHalfRates:
    """One retained qualification half: its settled speedup and lane reads."""

    reservation_id: str
    arena_digest: str
    index: int
    speedup: Decimal
    baseline: tuple[Decimal, ...]


@dataclass(frozen=True)
class RemeasurementEvidence:
    """Why one retained PASS pair does or does not warrant a fresh pair."""

    reservation_id: str
    arena_digest: str
    baseline_reads: int
    baseline_median: Decimal
    floor: Decimal
    credited_index: int
    credited_speedup: Decimal
    credited_baseline: tuple[Decimal, ...]

    @property
    def out_of_band(self) -> bool:
        return any(read < self.floor for read in self.credited_baseline)

    def describe(self) -> str:
        reads = ", ".join(str(read) for read in self.credited_baseline)
        half = "reproduction" if self.credited_index else "primary"
        return (
            f"arena {self.arena_digest[:12]}: {self.baseline_reads} retained baseline "
            f"reads, median {self.baseline_median} tok/s, floor {self.floor} tok/s; "
            f"credited {half} half speedup {self.credited_speedup} read the baseline "
            f"lane at {reads} tok/s -> "
            + ("OUT OF BAND" if self.out_of_band else "inside the band")
        )


def retained_half_rates(store, roots: tuple[Path, ...]) -> tuple[RetainedHalfRates, ...]:
    """Lane reads for every retained PASS half whose artifact a local store holds."""

    halves: list[RetainedHalfRates] = []
    for reservation_id, arena_digest, speedups, refs in store.retained_pass_pairs():
        for index, attempt_ref_json in refs:
            speed = qualification_speed(attempt_ref_json, roots)
            if speed is None:
                continue
            baseline = tuple(
                Decimal(str(lane["tokens_per_second"]))
                for lane in speed["lanes"]
                if lane["role"] in BASELINE_ROLES
            )
            if not baseline:
                continue
            halves.append(
                RetainedHalfRates(
                    reservation_id, arena_digest, index,
                    Decimal(speedups[index]), baseline,
                )
            )
    return tuple(halves)


def baseline_band_verdict(
    halves: tuple[RetainedHalfRates, ...],
    reservation_id: str,
    *,
    tolerance: Decimal = BAND_TOLERANCE,
    min_reads: int = MIN_BASELINE_READS,
) -> RemeasurementEvidence:
    """Judge the half that set a pair's credited speedup against its arena band.

    The credited half is the lower of the two settled speedups, exactly as
    settlement credits it. The band is the median of every baseline-role read
    retained in the same arena, and a pair is out of band only when a baseline
    read of that credited half sits under ``median * (1 - tolerance)``.
    """

    own = sorted(
        (half for half in halves if half.reservation_id == reservation_id),
        key=lambda half: half.index,
    )
    if len(own) not in (1, 2) or {half.index for half in own} != set(range(len(own))):
        raise BaselineBandError(
            "retained evidence does not hold lane rates for its accepted attempts"
        )
    arena_digest = own[0].arena_digest
    reads = [
        read
        for half in halves
        if half.arena_digest == arena_digest
        for read in half.baseline
    ]
    if len(reads) < min_reads:
        raise BaselineBandError(
            f"arena retains {len(reads)} baseline reads; the band needs {min_reads}"
        )
    median = Decimal(str(statistics.median(reads)))
    credited = min(own, key=lambda half: (half.speedup, half.index))
    return RemeasurementEvidence(
        reservation_id=reservation_id,
        arena_digest=arena_digest,
        baseline_reads=len(reads),
        baseline_median=median,
        floor=(median * (Decimal(1) - tolerance)).quantize(Decimal("0.1")),
        credited_index=credited.index,
        credited_speedup=credited.speedup,
        credited_baseline=credited.baseline,
    )


def remeasurement_evidence(
    store, reservation_id: str, roots: tuple[Path, ...]
) -> RemeasurementEvidence:
    """Read retained evidence and judge one pair's credited half."""

    return baseline_band_verdict(retained_half_rates(store, roots), reservation_id)


__all__ = [
    "BAND_TOLERANCE",
    "BaselineBandError",
    "MIN_BASELINE_READS",
    "RemeasurementEvidence",
    "RetainedHalfRates",
    "baseline_band_verdict",
    "qualification_evidence_roots",
    "qualification_speed",
    "remeasurement_evidence",
    "retained_half_rates",
]

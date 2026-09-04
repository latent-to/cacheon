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
    state_dir: Path, extra: tuple[Path, ...] = ()
) -> tuple[Path, ...]:
    """Every local store that may retain a submission's stage-exit artifact.

    Evidence roots rotate per worker generation, so one submission's artifact
    sits in whichever store was live when it graded.
    """

    try:
        rotated = sorted(state_dir.glob("qualification-evidence-*"), reverse=True)
    except OSError:
        rotated = []
    return tuple(rotated) + tuple(root for root in extra if root.is_dir())


def qualification_speed(
    attempt_ref_json: object, roots: tuple[Path, ...]
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
    try:
        rates = json.loads(payload)["speed_witness"]["rates"]
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(rates, list):
        return None
    lanes: list[dict[str, Any]] = []
    by_role: dict[Any, dict[str, Any]] = {}
    for rate in rates:
        try:
            windows = [float(row["seconds"]) for row in rate["windows"]]
            timed_tokens = int(rate["timed_tokens"])
            timed_seconds = float(rate["timed_seconds"])
            conditioning = float(rate["conditioning_seconds"])
        except (TypeError, ValueError, KeyError):
            return None
        if not windows or timed_seconds <= 0:
            return None
        average = sum(windows) / len(windows)
        lane = {
            "role": rate.get("role"),
            "tokens_per_second": round(timed_tokens / timed_seconds, 1),
            "window_seconds": [round(seconds, 3) for seconds in windows],
            "window_scatter": round((max(windows) - min(windows)) / average, 4),
            "conditioning_ratio": round(conditioning / average, 4),
        }
        lanes.append(lane)
        by_role[lane["role"]] = lane
    speed: dict[str, Any] = {"lanes": lanes}
    baseline, candidate = by_role.get("B"), by_role.get("C")
    if baseline and candidate and baseline["tokens_per_second"]:
        speed["speedup"] = round(
            candidate["tokens_per_second"] / baseline["tokens_per_second"], 4
        )
    return speed


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
    if len(own) != 2 or {half.index for half in own} != {0, 1}:
        raise BaselineBandError(
            "retained evidence does not hold lane rates for both halves"
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

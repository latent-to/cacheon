"""Dashboard rendering of signed screen receipts and qualification speed reads.

The screen path has recorded per-stage grades and numeric reasons inside the
signed receipt since the 2026-08 hardening, and every graded qualification
leaves a stage-exit artifact in an evidence store. Both already answer "which
check failed" and "what did the lanes measure" — this module only renders
them for the submissions API, so a rejected miner reads the verdict instead
of asking the operator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    reopen_evidence_anywhere,
)


def screen_stages(receipt_json: object) -> list[dict[str, Any]] | None:
    """Per-stage rows from one signed screen receipt; ``None`` when unreadable.

    Receipts sealed before the hardening carry no per-stage reasons; their
    rows still render with the stage and grade so the history stays honest.
    """

    if not isinstance(receipt_json, (str, bytes)):
        return None
    try:
        results = json.loads(receipt_json)["results"]
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(results, list):
        return None
    stages: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            return None
        stages.append(
            {
                "stage": row.get("stage"),
                "grade": row.get("grade"),
                "reason": row.get("reason"),
                "elapsed_ms": row.get("elapsed_ms"),
            }
        )
    return stages


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


__all__ = [
    "qualification_evidence_roots",
    "qualification_speed",
    "screen_stages",
]

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
from typing import Any


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


__all__ = ["screen_stages"]

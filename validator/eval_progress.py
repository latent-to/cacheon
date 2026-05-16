"""Ephemeral eval progress tracking.

Writes ``eval_progress.json`` to the state directory at each phase
transition during an eval round. The file is overwritten atomically
on every update and deleted between rounds.

All public functions swallow exceptions so a progress-write failure
never interrupts the actual evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROGRESS_FILE = "eval_progress.json"


def _read_progress(state_dir: str | os.PathLike) -> dict[str, Any]:
    path = Path(state_dir) / PROGRESS_FILE
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _write_progress(state_dir: str | os.PathLike, payload: dict[str, Any]) -> None:
    from .state import _atomic_write_json

    payload["updated_at"] = time.time()
    _atomic_write_json(Path(state_dir) / PROGRESS_FILE, payload)


def update_progress(
    state_dir: str | os.PathLike,
    *,
    phase: str,
    round_block: int | None = None,
    detail: str | None = None,
    challengers: list[dict[str, Any]] | None = None,
    gpu: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Write or update the progress file.

    When *phase* is ``"challengers_found"``, a fresh file is created with
    the provided *challengers* list (all set to ``"pending"``).  Otherwise
    the existing file is read-modify-written to preserve accumulated state.
    """
    try:
        now = time.time()
        step: dict[str, Any] = {"ts": now, "phase": phase}
        step.update(extra)

        if phase == "challengers_found":
            entries = [
                {
                    "idx": i,
                    "uid": c.get("uid"),
                    "hotkey": c.get("hotkey"),
                    "image": c.get("image"),
                    "status": "pending",
                }
                for i, c in enumerate(challengers or [])
            ]
            payload: dict[str, Any] = {
                "round_block": round_block,
                "status": "running",
                "phase": phase,
                "detail": detail,
                "current_idx": None,
                "challengers": entries,
                "gpu": None,
                "steps": [step],
                "started_at": now,
            }
        else:
            payload = _read_progress(state_dir)
            if not payload:
                payload = {
                    "round_block": round_block,
                    "status": "running",
                    "phase": phase,
                    "detail": detail,
                    "current_idx": None,
                    "challengers": [],
                    "gpu": None,
                    "steps": [],
                    "started_at": now,
                }
            payload["phase"] = phase
            payload["detail"] = detail
            if round_block is not None:
                payload["round_block"] = round_block
            payload.setdefault("steps", []).append(step)

        if gpu is not None:
            payload["gpu"] = gpu

        _write_progress(state_dir, payload)
    except Exception:
        logger.debug("Failed to update eval progress", exc_info=True)


def update_challenger_status(
    state_dir: str | os.PathLike,
    idx: int,
    *,
    status: str,
    score: float | None = None,
    dq_reason: str | None = None,
    detail: str | None = None,
) -> None:
    """Advance a single challenger's status in the progress file."""
    try:
        payload = _read_progress(state_dir)
        if not payload:
            return

        challengers = payload.get("challengers") or []
        if idx < 0 or idx >= len(challengers):
            return

        challengers[idx]["status"] = status
        if score is not None:
            challengers[idx]["score"] = score
        if dq_reason is not None:
            challengers[idx]["dq_reason"] = dq_reason

        payload["current_idx"] = idx
        payload["phase"] = "challenger_eval"
        payload["detail"] = detail

        step: dict[str, Any] = {
            "ts": time.time(),
            "phase": "challenger_eval",
            "uid": challengers[idx].get("uid"),
            "status": status,
        }
        payload.setdefault("steps", []).append(step)

        _write_progress(state_dir, payload)
    except Exception:
        logger.debug("Failed to update challenger status", exc_info=True)


def clear_progress(state_dir: str | os.PathLike) -> None:
    """Delete the progress file locally and on S3."""
    path = Path(state_dir) / PROGRESS_FILE
    try:
        if path.exists():
            os.unlink(path)
    except OSError:
        logger.debug("Failed to delete local eval progress file", exc_info=True)

    try:
        from .sync import delete_remote_keys

        delete_remote_keys([PROGRESS_FILE])
    except Exception:
        logger.debug("Failed to delete eval progress from S3", exc_info=True)

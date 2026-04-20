"""Phase 5 Part A — Validator state.

Tracks the king, every (hotkey, commit_block) combo we've already evaluated,
and per-miner score history. Persisted to JSON on disk with atomic writes so
a crash mid-write never leaves corrupt state.

Scoring convention: **higher = better**. A policy's score is
`0.6 * memory_reduction + 0.4 * latency_improvement` (see
`inference_engine/scoring.py`). The king is whoever holds the highest score.
Disqualified submissions get score = 0.0 and never take the crown.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1

STATE_FILE_NAME: str = "state.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to `path` atomically (tmp file + os.replace).

    Never leaves a half-written file even on SIGKILL. Does NOT fsync the
    directory — validator state loss on a kernel panic is acceptable
    (we re-eval any unknown challenger on next startup).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _eval_key(hotkey: str, commit_block: int) -> str:
    """Stable dedup key. Miners can technically commit twice — we treat each
    `(hotkey, block)` as its own submission so re-commits trigger re-eval."""
    return f"{hotkey}:{commit_block}"


@dataclass(frozen=True)
class EvaluationRecord:
    """One completed evaluation, keyed by (hotkey, commit_block).

    Immutable on purpose — completed evals are append-only history.
    """
    uid: int
    hotkey: str
    commit_block: int
    model: str
    revision: str
    score: float                # higher = better; 0.0 if disqualified
    kl_divergence: float
    memory_reduction: float
    latency_improvement: float
    disqualified: bool
    disqualify_reason: str | None
    evaluated_at: float         # unix timestamp
    evaluation_block: int       # chain block at eval time

    @property
    def eval_key(self) -> str:
        return _eval_key(self.hotkey, self.commit_block)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRecord:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(frozen=True)
class KingRecord:
    """The reigning champion. Exactly the fields needed to set weights and
    report publicly — no more."""
    uid: int
    hotkey: str
    commit_block: int
    model: str
    revision: str
    score: float
    kl_divergence: float
    memory_reduction: float
    latency_improvement: float
    evaluated_at: float
    evaluation_block: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KingRecord:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    @classmethod
    def from_evaluation(cls, ev: EvaluationRecord) -> KingRecord:
        return cls(
            uid=ev.uid,
            hotkey=ev.hotkey,
            commit_block=ev.commit_block,
            model=ev.model,
            revision=ev.revision,
            score=ev.score,
            kl_divergence=ev.kl_divergence,
            memory_reduction=ev.memory_reduction,
            latency_improvement=ev.latency_improvement,
            evaluated_at=ev.evaluated_at,
            evaluation_block=ev.evaluation_block,
        )


@dataclass
class ValidatorState:
    """The validator's durable state. All fields are JSON-serializable."""

    king: KingRecord | None = None
    evaluations: dict[str, EvaluationRecord] = field(default_factory=dict)
    """Keyed by `"{hotkey}:{commit_block}"`."""

    precheck_failures: dict[str, str] = field(default_factory=dict)
    """Sandbox AST rejections keyed by `"{hotkey}:{commit_block}"` → reason.
    We skip these on future scans without re-evaluating."""

    last_scan_block: int = 0
    """Most recent chain block we successfully scanned (informational)."""

    last_weights_set_block: int = 0
    """Most recent chain block we set weights at (informational)."""

    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # Lookup helpers
    # ------------------------------------------------------------------ #

    def has_evaluation(self, hotkey: str, commit_block: int) -> bool:
        return _eval_key(hotkey, commit_block) in self.evaluations

    def has_precheck_failure(self, hotkey: str, commit_block: int) -> bool:
        return _eval_key(hotkey, commit_block) in self.precheck_failures

    def is_known(self, hotkey: str, commit_block: int) -> bool:
        """Either evaluated or pre-rejected by the sandbox. Either way we've
        already formed an opinion and shouldn't re-run GPU eval."""
        key = _eval_key(hotkey, commit_block)
        return key in self.evaluations or key in self.precheck_failures

    def get_evaluation(
        self, hotkey: str, commit_block: int
    ) -> EvaluationRecord | None:
        return self.evaluations.get(_eval_key(hotkey, commit_block))

    def score_history_for_hotkey(
        self, hotkey: str
    ) -> list[EvaluationRecord]:
        """All evals for a given hotkey, oldest-first by commit_block."""
        matches = [e for e in self.evaluations.values() if e.hotkey == hotkey]
        return sorted(matches, key=lambda e: e.commit_block)

    # ------------------------------------------------------------------ #
    # Mutators — all side-effect-free w.r.t. disk; caller calls save()
    # ------------------------------------------------------------------ #

    def record_precheck_failure(
        self, hotkey: str, commit_block: int, reason: str
    ) -> None:
        self.precheck_failures[_eval_key(hotkey, commit_block)] = reason

    def record_evaluation(self, ev: EvaluationRecord) -> bool:
        """Store an eval. Returns True if this evaluation dethrones the king.

        Dethronement rule: strictly greater score, not disqualified. Ties
        keep the current king (defender's advantage — no pointless churn
        when challenger equals king within float noise)."""
        self.evaluations[ev.eval_key] = ev

        # A precheck failure key should no longer hold if we somehow ran eval
        self.precheck_failures.pop(ev.eval_key, None)

        if ev.disqualified or ev.score <= 0.0:
            return False

        if self.king is None or ev.score > self.king.score:
            self.king = KingRecord.from_evaluation(ev)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "king": self.king.to_dict() if self.king is not None else None,
            "evaluations": {
                k: v.to_dict() for k, v in self.evaluations.items()
            },
            "precheck_failures": dict(self.precheck_failures),
            "last_scan_block": self.last_scan_block,
            "last_weights_set_block": self.last_weights_set_block,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidatorState:
        version = int(data.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"state file schema_version={version} is newer than "
                f"this validator (supports up to {SCHEMA_VERSION}); "
                f"upgrade the validator before proceeding."
            )

        king_data = data.get("king")
        king = KingRecord.from_dict(king_data) if king_data else None

        evaluations = {
            k: EvaluationRecord.from_dict(v)
            for k, v in (data.get("evaluations") or {}).items()
        }

        return cls(
            king=king,
            evaluations=evaluations,
            precheck_failures=dict(data.get("precheck_failures") or {}),
            last_scan_block=int(data.get("last_scan_block", 0) or 0),
            last_weights_set_block=int(
                data.get("last_weights_set_block", 0) or 0
            ),
            schema_version=version,
        )

    # ------------------------------------------------------------------ #
    # Disk I/O
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, state_dir: str | os.PathLike) -> ValidatorState:
        """Load state from `<state_dir>/state.json`, or return a fresh
        state if the file doesn't exist or is unparseable."""
        path = Path(state_dir) / STATE_FILE_NAME
        if not path.exists():
            logger.info("No existing state file at %s — starting fresh.", path)
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Failed to load state from %s: %s — starting fresh.",
                path, exc,
            )
            return cls()
        return cls.from_dict(data)

    def save(self, state_dir: str | os.PathLike) -> None:
        """Atomically write state to `<state_dir>/state.json`."""
        path = Path(state_dir) / STATE_FILE_NAME
        _atomic_write_json(path, self.to_dict())

    # ------------------------------------------------------------------ #
    # Copy helpers (useful in tests)
    # ------------------------------------------------------------------ #

    def clone(self) -> ValidatorState:
        return ValidatorState.from_dict(self.to_dict())


def current_timestamp() -> float:
    return time.time()


def unknown_commits(
    state: ValidatorState,
    commitments: Iterable[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Given `(hotkey, commit_block)` pairs, return those we haven't yet
    evaluated or pre-rejected. Shape-shim for chain code that doesn't
    want to import `ValidatorState` directly."""
    return [
        (hk, blk) for hk, blk in commitments if not state.is_known(hk, blk)
    ]

"""On-disk memory for the validator: who is winning and what we already scored.

`ValidatorState` holds the current king (best-scoring miner), a set of
`(hotkey, commit_block)` pairs that have finished evaluation, per-miner
score history, and reasons for pre-rejects. The loop loads this from
`state.json`, updates it each tick, and saves again.

Writes use a temp file + rename so a crash never leaves a torn JSON
file. `SCHEMA_VERSION` applies to this file only.

Scoring convention: **higher = better**. A miner's score is
`0.5 * max(0, ttft_improvement) + 0.5 * max(0, throughput_improvement)`,
where improvements are relative to the vLLM baseline (median across
prompts). The king is the miner with the highest score. Disqualified
runs store score `0.0` and cannot take the crown.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from . import config as validator_config

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
"""Nothing shipped yet. Start fresh at 1; bump when the first production
state format needs a backward-incompatible change."""

STATE_FILE_NAME: str = "state.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to `path` atomically (tmp file + os.replace).

    Never leaves a half-written file even on SIGKILL. Does NOT fsync the
    directory -- validator state loss on a kernel panic is acceptable
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


def _quarantine_corrupt_state(path: Path) -> None:
    """Move a broken `state.json` aside so it isn't overwritten on save."""
    try:
        quarantined = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
        os.replace(path, quarantined)
        logger.warning(
            "Quarantined corrupt state file to %s for post-mortem.",
            quarantined,
        )
    except OSError as exc:
        logger.warning(
            "Could not quarantine %s (%s); it will be overwritten on next save.",
            path,
            exc,
        )


def _eval_key(hotkey: str, commit_block: int) -> str:
    """Stable dedup key. Miners can technically commit twice -- we treat each
    `(hotkey, block)` as its own submission so re-commits trigger re-eval."""
    return f"{hotkey}:{commit_block}"


@dataclass(frozen=True)
class EvaluationRecord:
    """One completed evaluation, keyed by (hotkey, commit_block).

    Immutable on purpose -- completed evals are append-only history.
    """

    uid: int
    hotkey: str
    commit_block: int
    image: str
    digest: str
    score: float  # higher = better; 0.0 if disqualified
    ttft_improvement: float
    throughput_improvement: float
    token_match_rate: float
    disqualified: bool
    disqualify_reason: str | None
    evaluated_at: float  # unix timestamp
    evaluation_block: int  # chain block at eval time
    per_prompt: list[dict[str, Any]] | None = None

    @property
    def eval_key(self) -> str:
        return _eval_key(self.hotkey, self.commit_block)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("per_prompt") is None:
            d.pop("per_prompt", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRecord:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(frozen=True)
class KingRecord:
    """The reigning champion. Exactly the fields needed to set weights,
    apply defender's-advantage on dethrone attempts, and report publicly.

    `crowned_at_block` is the chain block at which this miner took the
    throne; used to compute the decaying epsilon moat in
    `_effective_dethrone_threshold`.
    """

    uid: int
    hotkey: str
    commit_block: int
    image: str
    digest: str
    score: float
    ttft_improvement: float
    throughput_improvement: float
    token_match_rate: float
    evaluated_at: float
    evaluation_block: int
    crowned_at_block: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KingRecord:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    @classmethod
    def from_evaluation(
        cls,
        ev: EvaluationRecord,
        *,
        crowned_at_block: int,
    ) -> KingRecord:
        return cls(
            uid=ev.uid,
            hotkey=ev.hotkey,
            commit_block=ev.commit_block,
            image=ev.image,
            digest=ev.digest,
            score=ev.score,
            ttft_improvement=ev.ttft_improvement,
            throughput_improvement=ev.throughput_improvement,
            token_match_rate=ev.token_match_rate,
            evaluated_at=ev.evaluated_at,
            evaluation_block=ev.evaluation_block,
            crowned_at_block=crowned_at_block,
        )


# --------------------------------------------------------------------------- #
# Dethrone threshold -- decaying defender's-advantage
# --------------------------------------------------------------------------- #


DUPLICATE_OF_KING_REASON: str = "duplicate_of_king"


@dataclass(frozen=True)
class RecordResult:
    """Outcome of `ValidatorState.record_evaluation`.

    * ``stored`` is the record actually written to state -- it can differ
      from the input when the duplicate-of-king DQ rule fires.
    * ``dethroned`` is True iff this call made ``stored`` the new king.
    * ``dethrone_threshold`` is the score the challenger needed to beat
      (``king.score * (1 + effective_epsilon)``) at ``current_block``.
      ``0.0`` when there was no king to dethrone.
    """

    stored: EvaluationRecord
    dethroned: bool
    dethrone_threshold: float


def _effective_dethrone_threshold(
    king_score: float,
    king_crowned_at_block: int,
    current_block: int,
    *,
    epsilon_initial: float = validator_config.KING_EPSILON_INITIAL,
    decay_blocks: int = validator_config.KING_EPSILON_DECAY_BLOCKS,
) -> float:
    """Score a challenger must strictly exceed to dethrone the king.

    Starts at `king_score * (1 + epsilon_initial)` the block the king is
    crowned and decays linearly to `king_score` over `decay_blocks`.
    """
    if decay_blocks <= 0 or epsilon_initial <= 0.0:
        return king_score
    age = max(0, current_block - king_crowned_at_block)
    decay = max(0.0, 1.0 - age / decay_blocks)
    epsilon = epsilon_initial * decay
    return king_score * (1.0 + epsilon)


@dataclass
class ValidatorState:
    """The validator's durable state. All fields are JSON-serializable."""

    king: KingRecord | None = None
    evaluations: dict[str, EvaluationRecord] = field(default_factory=dict)
    """Keyed by `"{hotkey}:{commit_block}"`."""

    precheck_failures: dict[str, str] = field(default_factory=dict)
    """Pre-check rejections keyed by `"{hotkey}:{commit_block}"` -> reason.
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
        """Either evaluated or pre-rejected. Either way we've already
        formed an opinion and shouldn't re-run eval."""
        key = _eval_key(hotkey, commit_block)
        return key in self.evaluations or key in self.precheck_failures

    def get_evaluation(self, hotkey: str, commit_block: int) -> EvaluationRecord | None:
        return self.evaluations.get(_eval_key(hotkey, commit_block))

    def score_history_for_hotkey(self, hotkey: str) -> list[EvaluationRecord]:
        """All evals for a given hotkey, oldest-first by commit_block."""
        matches = [e for e in self.evaluations.values() if e.hotkey == hotkey]
        return sorted(matches, key=lambda e: e.commit_block)

    # ------------------------------------------------------------------ #
    # Mutators -- all side-effect-free w.r.t. disk; caller calls save()
    # ------------------------------------------------------------------ #

    def record_precheck_failure(
        self, hotkey: str, commit_block: int, reason: str
    ) -> None:
        self.precheck_failures[_eval_key(hotkey, commit_block)] = reason

    def record_evaluation(
        self,
        ev: EvaluationRecord,
        *,
        current_block: int,
    ) -> RecordResult:
        """Store an eval; return the record as actually stored.

        Two-stage dethronement rule:
          1. **Duplicate-of-king DQ.** If `ev.digest` matches the current
             king's digest, the hotkeys differ, and `ev.commit_block` is
             strictly later than the king's, the incoming record is
             rewritten to DQ with reason ``duplicate_of_king`` before
             being stored (score zeroed). Byte-identical Docker images
             can never tie or dethrone -- earliest-block-wins.
          2. **Decaying-epsilon threshold.** A non-DQ'd challenger must
             strictly exceed `_effective_dethrone_threshold(king, block)`
             to take the crown.
        """
        stored = ev
        if (
            self.king is not None
            and ev.digest
            and ev.digest == self.king.digest
            and ev.hotkey != self.king.hotkey
            and ev.commit_block > self.king.commit_block
        ):
            stored = replace(
                ev,
                score=0.0,
                disqualified=True,
                disqualify_reason=DUPLICATE_OF_KING_REASON,
            )
            logger.info(
                "UID %d (%s) DQ'd: %s (matches king digest=%s)",
                ev.uid,
                ev.hotkey[:16],
                DUPLICATE_OF_KING_REASON,
                (ev.digest or "")[:24],
            )

        self.evaluations[stored.eval_key] = stored

        self.precheck_failures.pop(stored.eval_key, None)

        threshold = 0.0
        if self.king is not None:
            threshold = _effective_dethrone_threshold(
                self.king.score,
                self.king.crowned_at_block,
                current_block,
            )

        if (
            stored.disqualified
            or not math.isfinite(stored.score)
            or stored.score <= 0.0
        ):
            return RecordResult(
                stored=stored,
                dethroned=False,
                dethrone_threshold=threshold,
            )

        if self.king is None or stored.score > threshold:
            self.king = KingRecord.from_evaluation(
                stored,
                crowned_at_block=current_block,
            )
            return RecordResult(
                stored=stored,
                dethroned=True,
                dethrone_threshold=threshold,
            )
        return RecordResult(
            stored=stored,
            dethroned=False,
            dethrone_threshold=threshold,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "king": self.king.to_dict() if self.king is not None else None,
            "evaluations": {k: v.to_dict() for k, v in self.evaluations.items()},
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
            last_weights_set_block=int(data.get("last_weights_set_block", 0) or 0),
            schema_version=version,
        )

    # ------------------------------------------------------------------ #
    # Disk I/O
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, state_dir: str | os.PathLike) -> ValidatorState:
        """Load state from `<state_dir>/state.json`, or return a fresh
        state if the file is missing, unreadable, or unparseable.

        Recovery policy:
          * Missing file -> fresh state, info log.
          * Unreadable / malformed JSON / schema drift -> fresh state,
            error log, and the offending file is renamed to
            ``state.json.corrupt.<unix_ts>`` for post-mortem.
          * `schema_version` newer than this validator supports -> hard
            `ValueError` re-raise.
        """
        path = Path(state_dir) / STATE_FILE_NAME
        if not path.exists():
            logger.info("No existing state file at %s -- starting fresh.", path)
            return cls()

        corrupt_reason: str | None = None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except ValueError as exc:
            if isinstance(exc, json.JSONDecodeError):
                corrupt_reason = f"malformed JSON: {exc}"
            elif "newer than" in str(exc):
                raise
            else:
                corrupt_reason = f"schema mismatch: {exc}"
        except OSError as exc:
            corrupt_reason = f"unreadable: {exc}"
        except (TypeError, KeyError) as exc:
            corrupt_reason = f"incompatible shape: {exc!r}"

        logger.error(
            "Failed to load state from %s (%s) -- starting fresh.",
            path,
            corrupt_reason,
        )
        _quarantine_corrupt_state(path)
        return cls()

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


def append_king_history(
    state_dir: str | os.PathLike,
    new_king: EvaluationRecord,
    dethroned_king: KingRecord | None,
    current_block: int,
    dethrone_threshold: float,
) -> None:
    """Append a single JSON line to ``king-history.jsonl`` on dethronement."""
    path = Path(state_dir) / "king-history.jsonl"
    entry = {
        "ts": time.time(),
        "block": current_block,
        "new_king_uid": new_king.uid,
        "new_king_hotkey": new_king.hotkey,
        "new_king_score": new_king.score,
        "new_king_image": new_king.image,
        "new_king_digest": new_king.digest,
        "dethrone_threshold": dethrone_threshold,
    }
    if dethroned_king is not None:
        entry["prev_king_uid"] = dethroned_king.uid
        entry["prev_king_hotkey"] = dethroned_king.hotkey
        entry["prev_king_score"] = dethroned_king.score
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception:
        logger.warning("failed to append king history to %s", path)


def unknown_commits(
    state: ValidatorState,
    commitments: Iterable[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Given `(hotkey, commit_block)` pairs, return those we haven't yet
    evaluated or pre-rejected."""
    return [(hk, blk) for hk, blk in commitments if not state.is_known(hk, blk)]

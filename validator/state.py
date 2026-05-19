"""On-disk memory for the validator: who is winning and what we already scored.

`ValidatorState` holds the current winner (best-scoring miner), a set of
`(hotkey, commit_block)` pairs that have finished evaluation, per-miner
score history, and reasons for pre-rejects. The loop loads this from
`state.json`, updates it each tick, and saves again.

Writes use a temp file + rename so a crash never leaves a torn JSON
file. `SCHEMA_VERSION` applies to this file only.

Scoring convention: **higher = better**. A miner's score is
`0.5 * max(0, ttft_improvement) + 0.5 * max(0, throughput_improvement)`,
where improvements are relative to the vLLM baseline (median across
prompts). The winner is the miner with the highest score. Disqualified
runs store score `0.0` and cannot win.
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

OVERTAKE_EPSILON: float = 0.01
"""Fixed 1% moat: a challenger must strictly exceed
``leader.score * (1 + OVERTAKE_EPSILON)`` to dethrone the leader
within the same round."""

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
class WinnerRecord:
    """The reigning champion. Exactly the fields needed to set weights,
    apply defender's-advantage on overtake attempts, and report publicly.

    `won_at_block` is the chain block at which this miner became the
    winner; used to compute the decaying epsilon moat in
    `_effective_overtake_threshold`.
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
    won_at_block: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WinnerRecord:
        mapped = dict(data)
        if "crowned_at_block" in mapped and "won_at_block" not in mapped:
            mapped["won_at_block"] = mapped.pop("crowned_at_block")
        known = {f: mapped[f] for f in cls.__dataclass_fields__ if f in mapped}
        return cls(**known)

    @classmethod
    def from_evaluation(
        cls,
        ev: EvaluationRecord,
        *,
        won_at_block: int,
    ) -> WinnerRecord:
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
            won_at_block=won_at_block,
        )


# --------------------------------------------------------------------------- #
# Overtake threshold -- decaying defender's-advantage
# --------------------------------------------------------------------------- #


DUPLICATE_OF_WINNER_REASON: str = "duplicate_of_winner"


@dataclass(frozen=True)
class RecordResult:
    """Outcome of `ValidatorState.record_evaluation`.

    * ``stored`` is the record actually written to state -- it can differ
      from the input when the duplicate-of-winner DQ rule fires.
    * ``overtook`` is True iff this call made ``stored`` the new winner.
    * ``overtake_threshold`` is the score the challenger needed to beat
      (``winner.score * (1 + effective_epsilon)``) at ``current_block``.
      ``0.0`` when there was no winner to overtake.
    """

    stored: EvaluationRecord
    overtook: bool
    overtake_threshold: float


def _effective_overtake_threshold(winner_score: float) -> float:
    """Score a challenger must strictly exceed to dethrone the leader.

    Fixed 1% moat (no decay). Fresh leader scores are collected each
    round, so the time-based decay is no longer needed.
    """
    return winner_score * (1.0 + OVERTAKE_EPSILON)


def _pick_runner_up(
    sorted_candidates: list[EvaluationRecord],
    winner_hotkey: str,
    current_block: int,
) -> WinnerRecord | None:
    """From a descending-score list, return the best candidate whose
    hotkey differs from the winner's. Returns None when no such candidate."""
    for rec in sorted_candidates:
        if rec.hotkey != winner_hotkey:
            return WinnerRecord.from_evaluation(rec, won_at_block=current_block)
    return None


@dataclass
class ValidatorState:
    """The validator's durable state. All fields are JSON-serializable."""

    winner: WinnerRecord | None = None
    runner_up_record: WinnerRecord | None = None
    """Persisted runner-up, set by ``rerank_round()`` alongside winner."""

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

    @property
    def runner_up(self) -> EvaluationRecord | WinnerRecord | None:
        """The runner-up. Returns the persisted ``runner_up_record`` when
        set (populated by ``rerank_round``), otherwise falls back to
        scanning ``evaluations`` for the highest-scoring non-winner hotkey.
        """
        if self.runner_up_record is not None:
            return self.runner_up_record
        return self._compute_runner_up()

    def _compute_runner_up(self) -> EvaluationRecord | None:
        """Scan evaluations for the best non-winner hotkey (fallback)."""
        if not self.evaluations:
            return None
        winner_hotkey = self.winner.hotkey if self.winner else None
        best_per_hotkey: dict[str, EvaluationRecord] = {}
        for ev in self.evaluations.values():
            if ev.disqualified or ev.score <= 0.0 or not math.isfinite(ev.score):
                continue
            if ev.hotkey == winner_hotkey:
                continue
            prev = best_per_hotkey.get(ev.hotkey)
            if prev is None or ev.score > prev.score:
                best_per_hotkey[ev.hotkey] = ev
        if not best_per_hotkey:
            return None
        return max(best_per_hotkey.values(), key=lambda e: e.score)

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
        """Store an eval and apply duplicate-of-winner DQ.

        Ranking and throne changes are handled separately by
        ``rerank_round()`` after all participants have been evaluated.
        This method only stores the record (with optional DQ annotation).
        """
        stored = ev
        if (
            self.winner is not None
            and ev.digest
            and ev.digest == self.winner.digest
            and ev.hotkey != self.winner.hotkey
            and ev.commit_block > self.winner.commit_block
        ):
            stored = replace(
                ev,
                score=0.0,
                disqualified=True,
                disqualify_reason=DUPLICATE_OF_WINNER_REASON,
            )
            logger.info(
                "UID %d (%s) DQ'd: %s (matches winner digest=%s)",
                ev.uid,
                ev.hotkey[:16],
                DUPLICATE_OF_WINNER_REASON,
                (ev.digest or "")[:24],
            )

        self.evaluations[stored.eval_key] = stored
        self.precheck_failures.pop(stored.eval_key, None)

        threshold = (
            _effective_overtake_threshold(self.winner.score)
            if self.winner is not None
            else 0.0
        )
        return RecordResult(
            stored=stored,
            overtook=False,
            overtake_threshold=threshold,
        )

    # ------------------------------------------------------------------ #
    # Round-level ranking
    # ------------------------------------------------------------------ #

    @staticmethod
    def rerank_round(
        *,
        leader_record: EvaluationRecord | None,
        ru_record: EvaluationRecord | None,
        challenger_records: list[EvaluationRecord],
        current_block: int,
    ) -> tuple[WinnerRecord | None, WinnerRecord | None]:
        """Pick the new winner and runner-up from all fresh same-round scores.

        The leader keeps the throne if no other participant strictly
        exceeds ``leader.score * (1 + OVERTAKE_EPSILON)``.

        Returns ``(winner, runner_up)`` -- either or both can be ``None``
        when participants are DQ'd or scored <= 0.
        """

        def _eligible(rec: EvaluationRecord | None) -> bool:
            return (
                rec is not None
                and not rec.disqualified
                and math.isfinite(rec.score)
                and rec.score > 0.0
            )

        candidates: list[EvaluationRecord] = []
        if _eligible(leader_record):
            assert leader_record is not None
            candidates.append(leader_record)
        if _eligible(ru_record):
            assert ru_record is not None
            candidates.append(ru_record)
        for rec in challenger_records:
            if _eligible(rec):
                candidates.append(rec)

        if not candidates:
            return None, None

        candidates.sort(key=lambda r: r.score, reverse=True)
        best = candidates[0]

        if best is not leader_record and _eligible(leader_record):
            assert leader_record is not None
            threshold = _effective_overtake_threshold(leader_record.score)
            if best.score <= threshold:
                winner_rec = leader_record
                winner = WinnerRecord.from_evaluation(
                    winner_rec, won_at_block=current_block
                )
                ru = _pick_runner_up(candidates, winner.hotkey, current_block)
                return winner, ru

        winner = WinnerRecord.from_evaluation(best, won_at_block=current_block)
        ru = _pick_runner_up(candidates, winner.hotkey, current_block)
        return winner, ru

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "winner": self.winner.to_dict() if self.winner is not None else None,
            "runner_up": (
                self.runner_up_record.to_dict()
                if self.runner_up_record is not None
                else None
            ),
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

        winner_data = data.get("winner") or data.get("king")
        winner = WinnerRecord.from_dict(winner_data) if winner_data else None

        ru_data = data.get("runner_up")
        runner_up_record = WinnerRecord.from_dict(ru_data) if ru_data else None

        evaluations = {
            k: EvaluationRecord.from_dict(v)
            for k, v in (data.get("evaluations") or {}).items()
        }

        return cls(
            winner=winner,
            runner_up_record=runner_up_record,
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


def append_winner_history(
    state_dir: str | os.PathLike,
    new_winner: EvaluationRecord,
    prev_winner: WinnerRecord | None,
    current_block: int,
    overtake_threshold: float,
) -> None:
    """Append a single JSON line to ``winner-history.jsonl`` on overtake."""
    path = Path(state_dir) / "winner-history.jsonl"
    entry = {
        "ts": time.time(),
        "block": current_block,
        "new_winner_uid": new_winner.uid,
        "new_winner_hotkey": new_winner.hotkey,
        "new_winner_score": new_winner.score,
        "new_winner_image": new_winner.image,
        "new_winner_digest": new_winner.digest,
        "overtake_threshold": overtake_threshold,
    }
    if prev_winner is not None:
        entry["prev_winner_uid"] = prev_winner.uid
        entry["prev_winner_hotkey"] = prev_winner.hotkey
        entry["prev_winner_score"] = prev_winner.score
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception:
        logger.warning("failed to append winner history to %s", path)


def unknown_commits(
    state: ValidatorState,
    commitments: Iterable[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Given `(hotkey, commit_block)` pairs, return those we haven't yet
    evaluated or pre-rejected."""
    return [(hk, blk) for hk, blk in commitments if not state.is_known(hk, blk)]

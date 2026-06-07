"""Commit-reveal + king-of-the-hill scoring — the anti-copy mechanism.

The problem in any open competition where submissions are evaluated in the open:
a lazy miner copies the current leader's submission (it's just code shipped to
the validator) and resubmits it, splitting reward for no work. Two mechanisms
defeat that here:

1. **Commit-reveal.** A miner first posts a *commitment* — a hash of
   ``(content_hash, hotkey, salt)`` — during the commit window, before any bundle
   is revealed. Later, in the reveal window, they post ``(content_hash, salt)``.
   A reveal is only accepted if it matches a commitment that *that hotkey* posted
   earlier. So you cannot reveal a bundle you didn't already commit to — and you
   couldn't have committed to a competitor's bundle you hadn't seen yet. Copying
   at reveal time is therefore impossible; the copier has no matching commitment.
   If two miners independently committed to the *same* content, the earliest
   commitment (lowest sequence) is the original; later identical ones are copies
   and earn nothing.

2. **Improvement-over-best (king of the hill).** A standing *champion* (the best
   validated bundle so far) holds the title and the emission. A challenger only
   takes the title if its score beats the champion's by a margin (which absorbs
   measurement noise). A copy ties the champion — it never clears the margin — so
   it earns zero. The only way to earn is to genuinely beat the best.

This module is pure-Python and persists to a JSON ledger so it can be tested and
reasoned about without a GPU. In a real Bittensor subnet the commitments live
on-chain, the bundles are fetched from a content-addressed store, and ``hotkey``
is the miner's SS58 address; the semantics here are the same.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("optima.ledger")

# Bump when the on-disk ledger format changes in a way older code cannot read.
SCHEMA_VERSION = 1


def make_commitment(content_hash: str, hotkey: str, salt: str) -> str:
    """The value a miner posts in the commit window."""
    return hashlib.sha256(f"{content_hash}:{hotkey}:{salt}".encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: object) -> None:
    """Write JSON durably: serialize to a sibling temp file, then atomically rename
    it over the target. A crash mid-write leaves the previous file intact — never a
    truncated half-file. ``os.replace`` is atomic on a single filesystem."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _only_fields(cls: type, d: dict) -> dict:
    """Keep only keys that name a field of ``cls``. Unknown keys (written by a newer
    schema) are dropped, and missing keys fall back to the dataclass defaults — so a
    record can gain optional fields without breaking older or newer ledger files."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _quarantine(path: Path) -> Path:
    """Move an unreadable ledger aside to ``<name>.corrupt.N`` so a fresh ledger can
    start without silently destroying the damaged file. Returns the new path."""
    for i in range(1, 10_000):
        target = path.with_name(f"{path.name}.corrupt.{i}")
        if not target.exists():
            os.replace(path, target)
            return target
    target = path.with_name(f"{path.name}.corrupt.overflow")
    os.replace(path, target)
    return target


@dataclass
class Commitment:
    hotkey: str
    commitment: str
    round_id: int
    seq: int  # monotonic; commit order = anti-copy priority


@dataclass
class Reveal:
    hotkey: str
    content_hash: str
    salt: str
    round_id: int
    commit_seq: int
    original: bool = True


@dataclass
class Score:
    hotkey: str
    content_hash: str
    round_id: int
    score: float
    kl_mean: float
    passed: bool


@dataclass
class Champion:
    content_hash: str
    hotkey: str
    score: float
    round_id: int


@dataclass
class ChampionChange:
    """One throne change, appended to the ledger's champion history (audit trail)."""
    content_hash: str
    hotkey: str
    score: float
    round_id: int
    from_hotkey: Optional[str] = None


@dataclass(frozen=True)
class EvalRecord:
    """The full, typed result of evaluating one bundle — the canonical eval record:
    the audit row, the dedup key, and what a dashboard reads. ``score`` / ``passed`` /
    ``mean_kl`` mirror the king-of-the-hill ``Score`` atom; the remaining fields carry
    the fidelity + provenance detail that a bare score throws away. The ledger keys
    these by ``(hotkey, bundle_hash)`` so an already-scored submission is never re-run.
    """
    hotkey: str
    bundle_hash: str
    slot: str
    round_id: int
    score: float
    passed: bool
    throughput: float = 0.0
    baseline_throughput: float = 0.0
    mean_kl: float = 0.0
    p99_kl: float = 0.0
    argmax_rate: float = 0.0
    gsm8k_acc: float = -1.0  # -1 = not measured
    dq_reason: str = ""
    uid: int = -1
    block: int = 0           # chain block of the eval (0 until chain integration)
    ts: float = 0.0          # caller-supplied wall-clock; 0 = unset (kept out of logic)
    per_prompt: tuple = ()


def _eval_from_dict(d: dict) -> EvalRecord:
    fields = _only_fields(EvalRecord, d)
    fields["per_prompt"] = tuple(fields.get("per_prompt", ()))
    return EvalRecord(**fields)


@dataclass
class SettleResult:
    champion: Optional[Champion]
    weights: dict[str, float]
    title_changed: bool
    challenger_score: float
    rejected_copies: list[str] = field(default_factory=list)  # hotkeys


class RevealError(ValueError):
    pass


class Ledger:
    def __init__(self) -> None:
        self.commitments: list[Commitment] = []
        self.reveals: list[Reveal] = []
        self.scores: list[Score] = []
        self.evals: dict[str, EvalRecord] = {}
        self.champion: Optional[Champion] = None
        self.champion_history: list[ChampionChange] = []
        self._seq = 0

    # ---- persistence ----

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        led = cls()
        if not p.exists():
            return led
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            moved = _quarantine(p)
            logger.warning("ledger %s unreadable (%s); quarantined to %s, starting fresh",
                           p, exc, moved)
            return led
        ver = data.get("schema_version", 1)
        if ver > SCHEMA_VERSION:
            raise ValueError(
                f"ledger {p} is schema v{ver}, newer than this build supports (v{SCHEMA_VERSION}); "
                "upgrade optima before reading it"
            )
        led.commitments = [Commitment(**_only_fields(Commitment, c)) for c in data.get("commitments", [])]
        led.reveals = [Reveal(**_only_fields(Reveal, r)) for r in data.get("reveals", [])]
        led.scores = [Score(**_only_fields(Score, s)) for s in data.get("scores", [])]
        led.evals = {k: _eval_from_dict(v) for k, v in data.get("evals", {}).items()}
        led.champion_history = [
            ChampionChange(**_only_fields(ChampionChange, h)) for h in data.get("champion_history", [])
        ]
        champ = data.get("champion")
        led.champion = Champion(**_only_fields(Champion, champ)) if champ else None
        led._seq = data.get("seq", len(led.commitments))
        return led

    def save(self, path: str | Path) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "commitments": [asdict(c) for c in self.commitments],
            "reveals": [asdict(r) for r in self.reveals],
            "scores": [asdict(s) for s in self.scores],
            "evals": {k: asdict(v) for k, v in self.evals.items()},
            "champion": asdict(self.champion) if self.champion else None,
            "champion_history": [asdict(h) for h in self.champion_history],
            "seq": self._seq,
        }
        _atomic_write_json(Path(path), data)

    # ---- commit phase ----

    def commit(self, hotkey: str, commitment: str, round_id: int) -> int:
        seq = self._seq
        self._seq += 1
        self.commitments.append(Commitment(hotkey, commitment, round_id, seq))
        return seq

    # ---- reveal phase ----

    def reveal(self, hotkey: str, content_hash: str, salt: str, round_id: int) -> Reveal:
        """Verify a reveal against this hotkey's prior commitments; record it.

        Raises RevealError if no commitment by this hotkey matches. Sets
        ``original`` False if an earlier-committed reveal of the same content
        already exists (a copy / duplicate).
        """
        target = make_commitment(content_hash, hotkey, salt)
        match = min(
            (c for c in self.commitments
             if c.hotkey == hotkey and c.round_id == round_id and c.commitment == target),
            key=lambda c: c.seq,
            default=None,
        )
        if match is None:
            raise RevealError(
                f"no commitment by {hotkey!r} in round {round_id} matches the revealed bundle"
            )

        # Copy detection: earliest commit_seq for this content_hash wins.
        prior = [r for r in self.reveals if r.content_hash == content_hash and r.round_id == round_id]
        original = all(match.seq < r.commit_seq for r in prior) if prior else True
        if prior and original:
            # This reveal predates earlier-recorded ones; demote them.
            for r in prior:
                r.original = False

        rev = Reveal(hotkey, content_hash, salt, round_id, match.seq, original)
        self.reveals.append(rev)
        return rev

    # ---- scoring ----

    def record_score(self, hotkey: str, content_hash: str, round_id: int,
                     score: float, kl_mean: float, passed: bool) -> None:
        self.scores.append(Score(hotkey, content_hash, round_id, score, kl_mean, passed))

    # ---- full eval records (audit trail + dedup; the rich superset of a Score) ----

    @staticmethod
    def _eval_key(hotkey: str, bundle_hash: str) -> str:
        return f"{hotkey}:{bundle_hash}"

    def record_eval(self, rec: EvalRecord) -> None:
        """Store the full eval record, keyed by (hotkey, bundle_hash). Recording the
        same submission again overwrites it (evaluations are deterministic)."""
        self.evals[self._eval_key(rec.hotkey, rec.bundle_hash)] = rec

    def is_known(self, hotkey: str, bundle_hash: str) -> bool:
        """True if this exact submission already has an eval record — skip re-running it."""
        return self._eval_key(hotkey, bundle_hash) in self.evals

    def eval_for(self, hotkey: str, bundle_hash: str) -> Optional[EvalRecord]:
        return self.evals.get(self._eval_key(hotkey, bundle_hash))

    def _is_original(self, hotkey: str, content_hash: str, round_id: int) -> bool:
        for r in self.reveals:
            if r.hotkey == hotkey and r.content_hash == content_hash and r.round_id == round_id:
                return r.original
        return False

    def settle(self, round_id: int, margin: float = 0.02) -> SettleResult:
        """Apply king-of-the-hill: a challenger takes the title only if it beats
        the champion by ``margin``. Emission goes to the champion (winner-take-all
        baseline). Copies and non-improvers earn nothing.
        """
        rejected_copies: list[str] = []
        candidates: list[Score] = []
        for s in self.scores:
            if s.round_id != round_id or not s.passed:
                continue
            if not self._is_original(s.hotkey, s.content_hash, round_id):
                rejected_copies.append(s.hotkey)
                continue
            candidates.append(s)

        challenger = max(candidates, key=lambda s: s.score, default=None)
        challenger_score = challenger.score if challenger else 0.0

        title_changed = False
        threshold = (self.champion.score * (1.0 + margin)) if self.champion else (1.0 + margin)
        if challenger is not None and challenger_score >= threshold:
            from_hotkey = self.champion.hotkey if self.champion else None
            self.champion = Champion(
                content_hash=challenger.content_hash,
                hotkey=challenger.hotkey,
                score=challenger.score,
                round_id=round_id,
            )
            title_changed = True
            self.champion_history.append(ChampionChange(
                content_hash=challenger.content_hash,
                hotkey=challenger.hotkey,
                score=challenger.score,
                round_id=round_id,
                from_hotkey=from_hotkey,
            ))

        weights = {self.champion.hotkey: 1.0} if self.champion else {}
        return SettleResult(
            champion=self.champion,
            weights=weights,
            title_changed=title_changed,
            challenger_score=challenger_score,
            rejected_copies=sorted(set(rejected_copies)),
        )

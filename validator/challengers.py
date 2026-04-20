"""Phase 5 Part A — Challenger selection.

Given the current validator state and the set of on-chain commitments,
pick which `(uid, hotkey, commit_block)` triples still need GPU
evaluation this round.

Rule: a commitment is a *challenger* iff:
  - we haven't already evaluated its `(hotkey, commit_block)` pair
  - we haven't already pre-rejected it via the AST sandbox

The sandbox precheck itself is a hook — the policy source isn't on-chain,
only a `(model, revision)` pointer, so actually fetching the code lives
in a separate layer (Phase 5 Part B). Here we just expose the decision
surface and a conservative default that lets all commitments through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from .chain import CommitmentRecord
from .state import ValidatorState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrecheckResult:
    ok: bool
    reason: str | None = None


PrecheckFn = Callable[[CommitmentRecord], PrecheckResult]
"""A function that fetches the miner's policy source at the pinned
revision and runs the Phase 3 AST sandbox on it, returning a
`PrecheckResult`. In tests and dry runs, use `allow_all_precheck`."""


def allow_all_precheck(_commitment: CommitmentRecord) -> PrecheckResult:
    """Conservative default — no code fetch, no AST check. Every new
    commitment is forwarded to the GPU evaluator. Used in tests and as
    the Phase 5 Part A fallback until the source-fetch layer exists."""
    return PrecheckResult(ok=True)


@dataclass(frozen=True)
class ChallengerSet:
    """Result of one round of challenger selection."""
    challengers: list[CommitmentRecord]
    newly_rejected: list[tuple[CommitmentRecord, str]]
    """Commitments the sandbox rejected this round (caller should record
    them in state so we don't re-check next loop)."""
    already_known: list[CommitmentRecord]
    """Commitments we've already decided on (evaluated or pre-rejected)."""

    def __len__(self) -> int:
        return len(self.challengers)


def select_challengers(
    state: ValidatorState,
    commitments: Iterable[CommitmentRecord],
    *,
    precheck: PrecheckFn = allow_all_precheck,
) -> ChallengerSet:
    """Decide who to send to the GPU pod this round.

    Pure function — does **not** mutate `state`. The caller is expected
    to apply `newly_rejected` via `state.record_precheck_failure(...)`
    and save state.
    """
    challengers: list[CommitmentRecord] = []
    newly_rejected: list[tuple[CommitmentRecord, str]] = []
    already_known: list[CommitmentRecord] = []

    for com in commitments:
        if state.is_known(com.hotkey, com.commit_block):
            already_known.append(com)
            continue

        result = precheck(com)
        if not result.ok:
            reason = result.reason or "sandbox precheck failed"
            logger.info(
                "UID %d (%s) rejected by sandbox precheck: %s",
                com.uid, com.hotkey[:16] + "...", reason,
            )
            newly_rejected.append((com, reason))
            continue

        challengers.append(com)

    return ChallengerSet(
        challengers=challengers,
        newly_rejected=newly_rejected,
        already_known=already_known,
    )

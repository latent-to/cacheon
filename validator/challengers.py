"""Decide which on-chain commitments still need an evaluation run.

Inputs: the saved `ValidatorState` and the current map of miners to
`CommitmentRecord`. Output: the subset that should be sent to ``eval_fn``
this tick.

A commitment counts as a *challenger* when:
  - we have not already recorded a score for its `(hotkey, commit_block)` pair, and
  - we have not already pre-rejected it from the AST sandbox.

The chain only stores a `(model, revision)` pointer—not policy source—so
**fetching** code and running the sandbox is done by the ``precheck``
hook the caller provides. This module only applies the filters above;
``allow_all_precheck`` is a permissive default for tests and early wiring.
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
"""Fetches the miner's policy at the pinned revision and runs the static
AST sandbox, returning `PrecheckResult`. Tests and dry runs can use
`allow_all_precheck` to skip real fetches."""


def allow_all_precheck(_commitment: CommitmentRecord) -> PrecheckResult:
    """Permissive default — no code fetch, no AST check. Every new
    commitment is forwarded to ``eval_fn``. Useful in tests and early
    wiring before a real precheck fetches source and runs the sandbox."""
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

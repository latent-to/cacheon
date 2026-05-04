"""Decide which on-chain commitments still need an evaluation run.

Inputs: the saved `ValidatorState` and the current map of miners to
`CommitmentRecord`. Output: the subset that should be sent to ``eval_fn``
this tick.

A commitment counts as a *challenger* when:
  - we have not already recorded a score for its `(hotkey, commit_block)` pair, and
  - we have not already pre-rejected it via the precheck hook.

The chain stores an ``(image, digest)`` pointer to a Docker image. The
``precheck`` hook the caller provides can validate the image reference
before we spend GPU time on a full eval. This module only applies the
dedup filters above; ``allow_all_precheck`` is a permissive default for
tests and early wiring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from .chain import CommitmentRecord
from .state import ValidatorState

logger = logging.getLogger(__name__)


class PrecheckOutcome(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class PrecheckResult:
    outcome: PrecheckOutcome
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """Back-compat helper — True iff outcome is OK."""
        return self.outcome is PrecheckOutcome.OK


PrecheckFn = Callable[[CommitmentRecord], PrecheckResult]
"""Validates a miner's Docker image reference before spending GPU time
on a full eval, returning `PrecheckResult`. Tests and dry runs can use
`allow_all_precheck` to skip validation."""


def allow_all_precheck(_commitment: CommitmentRecord) -> PrecheckResult:
    """Permissive default -- every new commitment is forwarded to
    ``eval_fn``. Useful in tests and early wiring before a real precheck
    validates Docker image references."""
    return PrecheckResult(outcome=PrecheckOutcome.OK)


@dataclass(frozen=True)
class ChallengerSet:
    """Result of one round of challenger selection."""

    challengers: list[CommitmentRecord]
    """Commitments that passed precheck and should be evaluated."""

    newly_rejected: list[tuple[CommitmentRecord, str]]
    """Commitments the sandbox rejected this round (caller should record
    them in state so we don't re-check next loop)."""

    deferred: list[tuple[CommitmentRecord, str]]
    """Commitments that hit a transient failure this round (caller should
    log and retry next loop without recording in state)."""

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
    deferred: list[tuple[CommitmentRecord, str]] = []
    already_known: list[CommitmentRecord] = []

    for com in commitments:
        if state.is_known(com.hotkey, com.commit_block):
            already_known.append(com)
            continue

        result = precheck(com)
        if result.outcome is PrecheckOutcome.REJECTED:
            reason = result.reason or "sandbox precheck failed"
            logger.info(
                "UID %d (%s) rejected by precheck: %s",
                com.uid,
                com.hotkey[:16] + "...",
                reason,
            )
            newly_rejected.append((com, reason))
            continue

        if result.outcome is PrecheckOutcome.DEFERRED:
            reason = result.reason or "precheck deferred"
            logger.warning(
                "UID %d (%s) deferred by precheck: %s — will retry next tick",
                com.uid,
                com.hotkey[:16] + "...",
                reason,
            )
            deferred.append((com, reason))
            continue

        challengers.append(com)

    return ChallengerSet(
        challengers=challengers,
        newly_rejected=newly_rejected,
        deferred=deferred,
        already_known=already_known,
    )

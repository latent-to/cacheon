"""Replay a terminal verdict onto a byte-identical resubmission.

A candidate is its publication bytes. If the validator has already produced a
terminal verdict for exactly those bytes, under exactly the arena that produced
it, re-running the evaluation buys nothing and costs a full resident
qualification -- roughly a quarter hour of exclusive 8-GPU time per attempt.

Measured on mainnet 2026-08-18 across all 330 lifetime reservations: 23 groups
of byte-identical bundles were evaluated more than once under the same arena
digest. In **zero** of those groups did the decision differ, and in zero did the
reason differ. 24 evaluations could have been skipped with no loss of
information.

What makes a replay valid
-------------------------
Two conditions, both required:

- **identical bytes** -- the same ``content_hash``. Not the same miner, not the
  same manifest: the same publication.
- **identical arena** -- the same ``arena_service_digest``. This is the whole
  safety argument. A speedup is measured against a baseline, on a model, in an
  image, at a topology, under a policy. Change any of those and the old number
  describes a world that no longer exists. Keying on the arena digest means
  elapsed time is irrelevant: five days or fifty, a replay is valid while the
  arena is unchanged and invalid the moment it is not.

Why FAIL replays and PASS does not
----------------------------------
A replayed FAIL costs a miner nothing they did not already earn: the identical
bytes already lost under the identical arena, and any change at all -- one byte
-- produces a new hash and a fresh evaluation.

A PASS is different in kind. It is not a verdict, it is *evidence*: a first PASS
is ``reproduction_pending`` and settlement requires an independently bound PASS
pair. Replaying one would manufacture the second half of that pair from the
first, which is precisely the independence the pair exists to guarantee. This
module therefore refuses to replay a PASS and reports it for an operator.
"""

from __future__ import annotations

from dataclasses import dataclass

REPLAYABLE_DECISIONS = frozenset({"FAIL"})


@dataclass(frozen=True)
class PriorVerdict:
    """A terminal verdict already recorded for some publication."""

    reservation_id: str
    content_hash: str
    arena_service_digest: str
    decision: str
    reason: str


@dataclass(frozen=True)
class ReplayDecision:
    """What to do with a candidate that may duplicate an earlier verdict."""

    replay: bool
    prior_reservation_id: str = ""
    reason: str = ""
    refused: str = ""


def replay_reason(prior: PriorVerdict) -> str:
    """The durable reason stamped on a replayed row.

    Carries the source reservation so the decision is auditable from the row
    alone, and the original reason so the miner sees why, not merely that.
    """

    return f"duplicate_of:{prior.reservation_id[:16]}:{prior.reason}"


def decide_replay(
    *,
    content_hash: str,
    arena_service_digest: str,
    screen_lane: str,
    priors: tuple[PriorVerdict, ...],
) -> ReplayDecision:
    """Decide whether this candidate may inherit an earlier terminal verdict.

    ``screen_lane`` is checked first and deliberately: a reproduction is by
    definition a second, independently bound measurement of a bundle that
    already passed once. Replaying a verdict onto it would collapse the pair
    into a single observation and defeat the entire point of reproduction.
    """

    if screen_lane == "reproduction":
        return ReplayDecision(False, refused="reproduction lane requires an independent measurement")
    if not content_hash or not arena_service_digest:
        return ReplayDecision(False, refused="candidate has no bytes or no arena identity")

    for prior in priors:
        if prior.content_hash != content_hash:
            continue
        if prior.arena_service_digest != arena_service_digest:
            continue
        if prior.decision not in REPLAYABLE_DECISIONS:
            return ReplayDecision(
                False,
                prior_reservation_id=prior.reservation_id,
                refused=f"{prior.decision} is evidence, not a verdict; a pair must be independently bound",
            )
        return ReplayDecision(
            True,
            prior_reservation_id=prior.reservation_id,
            reason=replay_reason(prior),
        )
    return ReplayDecision(False)

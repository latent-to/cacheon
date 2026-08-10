"""How one resident speed stage turns measured reads into a verdict.

Kept apart from the stage runtime so the decision rule is readable on its own
and so the runtime module stops growing. The rule has one job: never answer
"no decision" when the reads already settle the question.

Version 3 refuses to grade a read whose own window scatter exceeds its sealed
bound, so an unstable box can convert a settled result into a non-answer.
Retained mainnet evidence shows that happening -- a candidate recorded a 0.582
speedup and a 1.78x conditioning regression, two independent hard-FAIL signals,
and still terminated NO_DECISION.

Version 4 grades every read and decides by the spread of the reads actually
taken. It assumes no distribution, so an ill-behaved box cannot manufacture
indecision, and because taking more reads only widens an observed spread, the
concluding grade terminates rather than deferring forever.

Version 5 keeps version 4's termination guarantee and adds the owner's
bracket-drift ruling (2026-08-10): flanking baseline brackets that disagree
beyond the sealed noise ceiling do not void the read set. The earliest bracket
was measured adjacent to the candidate arm, so it is the only valid comparison
baseline; the drifted later brackets are excluded and C against B decides.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from cacheon.eval.scoring import SpeedupVerdict, relative_spread, score_speedup

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cacheon.eval.crossover_runtime import ResidentSpeedPolicy


class SpeedStageDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_DECISION = "NO_DECISION"


def _disposition(
    verdict: SpeedupVerdict, margin: float
) -> SpeedStageDecision | None:
    if not verdict.confident:
        return None
    if verdict.speedup <= verdict.required - margin:
        return SpeedStageDecision.FAIL
    if verdict.speedup >= verdict.required + margin:
        return SpeedStageDecision.PASS
    return None


def _final(verdict: SpeedupVerdict) -> SpeedStageDecision:
    if not verdict.confident:
        return SpeedStageDecision.NO_DECISION
    return SpeedStageDecision.PASS if verdict.passed_speedup else SpeedStageDecision.FAIL


def invariant_decision(
    baselines: list[float], candidates: list[float], required: float
) -> SpeedStageDecision | None:
    """The verdict that survives the full spread of the reads actually taken.

    A candidate that loses even against its most favorable baseline read has
    lost under every reading of the drift; one that wins against its least
    favorable read has won under every reading. Only a verdict that flips
    inside the observed spread is genuinely undetermined."""

    if not baselines or not candidates:
        return None
    if max(candidates) / min(baselines) < required:
        return SpeedStageDecision.FAIL
    if min(candidates) / max(baselines) >= required:
        return SpeedStageDecision.PASS
    return None


def speed_grade(
    policy: "ResidentSpeedPolicy",
    baselines: list[object],
    candidates: list[object],
    *,
    concluding: bool,
) -> tuple[SpeedupVerdict, SpeedStageDecision | None]:
    """Grade one read set, shared by the live stage and the independent regrade
    so the two cannot drift apart. ``concluding`` marks the last grade available
    for this stage, after which no further reads will be taken."""

    baseline_rates = [policy.scored_tokens_per_second(row) for row in baselines]
    candidate_rates = [policy.scored_tokens_per_second(row) for row in candidates]
    dropped_brackets = 0
    bracket_drift = 0.0
    if policy.version >= 5 and len(baseline_rates) >= 2:
        bracket_drift = relative_spread(baseline_rates)
        if bracket_drift > policy.max_noise:
            dropped_brackets = len(baseline_rates) - 1
            baseline_rates = baseline_rates[:1]
    verdict = score_speedup(
        baseline_rates,
        candidate_rates,
        min_margin=policy.min_margin,
        k=policy.noise_multiplier,
        max_noise=policy.max_noise,
    )
    if dropped_brackets:
        verdict = replace(
            verdict,
            detail=(
                f"bracket drift {bracket_drift:.1%} > max_noise "
                f"{policy.max_noise:.0%}; {dropped_brackets} later bracket(s) "
                f"excluded -- C against B decides"
            ),
        )
    if policy.version < 4:
        if concluding:
            return verdict, _final(verdict)
        return verdict, _disposition(verdict, policy.min_margin)
    decision = invariant_decision(baseline_rates, candidate_rates, verdict.required)
    if decision is None and concluding:
        # Escalation cannot be relied on to converge: taking more reads only
        # widens the observed spread. The burden of proof sits with the
        # candidate, so an undetermined conclusion is "not proven faster" -- a
        # decision the miner can act on, never a non-answer.
        decision = SpeedStageDecision.FAIL
    return verdict, decision


__all__ = [
    "SpeedStageDecision",
    "invariant_decision",
    "speed_grade",
]

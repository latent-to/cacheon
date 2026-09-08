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

Version 8 precommits the B/C/B-prime schedule and grades it once, concluding.
The pre-version-8 graders (the adaptive five-read escalation and the version-6
conditional bookend) were deleted with the MiniMax-M3 history seal on
2026-09-06; no policy below version 8 can be constructed or graded.

Versions 10 and 11 implement the owner's 2026-09-07 single-run contract.
Both baseline observations must be stable, including matched timed windows;
neither may be discarded. The credited estimate uses the faster baseline.
An invalid or unresolved measurement cannot become a candidate failure.
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


SPEED_FAIL_REASONS = frozenset({"candidate_slower", "speed_threshold_not_met"})


def fail_reason(verdict: SpeedupVerdict, *, conditioning_failed: bool = False) -> str:
    """Name what a speed FAIL actually proved.

    A bar of ``1 + u`` can only call a candidate slower once its speedup falls
    below the mirrored bound ``1 - u``; a conditioning regression is a measured
    slowdown in its own right. Anything else inside the band proved only that
    the bar was not cleared -- an ordinary competitive miss, not a regression.
    """

    if conditioning_failed or verdict.speedup < 2.0 - verdict.required:
        return "candidate_slower"
    return "speed_threshold_not_met"


DECODE_ROLES = ("B", "C", "B_prime")
# Version 12 appends a prefill-only pass of the same sealed batches on each arm
# after the decode schedule has finished, so the decode reads are taken exactly
# as every earlier version took them.
PREFILL_LANE_ROLES = DECODE_ROLES + ("B_prefill", "C_prefill", "B_prime_prefill")


def resident_speed_roles(version: int, count: int) -> tuple[str, ...] | None:
    """The precommitted read roles for one policy version, or None for a count
    that version never reads."""

    roles = PREFILL_LANE_ROLES if version >= 12 else DECODE_ROLES
    return roles if count == len(roles) else None


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
    if policy.version >= 10:
        return _single_run_grade(policy, baselines, candidates, baseline_rates, candidate_rates)
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
    decision = invariant_decision(baseline_rates, candidate_rates, verdict.required)
    if decision is None and concluding:
        # Escalation cannot be relied on to converge: taking more reads only
        # widens the observed spread. The burden of proof sits with the
        # candidate, so an undetermined conclusion is "not proven faster" -- a
        # decision the miner can act on, never a non-answer.
        decision = SpeedStageDecision.FAIL
    return verdict, decision


def _single_run_grade(policy, baselines, candidates, baseline_rates, candidate_rates):
    """Keep all stock observations and separate measurement validity from competition."""

    verdict = score_speedup(
        baseline_rates, candidate_rates, min_margin=policy.min_margin,
        k=policy.noise_multiplier, max_noise=policy.max_noise,
    )
    invalid = ""
    if len(baselines) != 2 or len(candidates) != 1:
        invalid = "single-run qualification requires complete B/C/B-prime evidence"
    elif any(row.first_timed_batch_index <= row.first_batch_index
             or row.conditioning_tokens <= 0 for row in (*baselines, *candidates)):
        invalid = "measurement lacks the declared conditioning before timing"
    elif not verdict.confident:
        invalid = "baseline brackets exceed the sealed drift limit; no baseline was discarded"
    else:
        before, after = (row.windows for row in baselines)
        current = candidates[0].windows
        if len(before) != len(after) or len(before) != len(current):
            invalid = "B/C/B-prime windows do not cover the same workload"
        else:
            for b, bp, c in zip(before, after, current, strict=True):
                if b.tokens != bp.tokens or b.tokens != c.tokens:
                    invalid = "B/C/B-prime token numerators differ"
                    break
                if relative_spread([b.tokens / b.seconds, bp.tokens / bp.seconds]) > policy.max_window_scatter:
                    invalid = "matched baseline windows exceed the sealed stability limit"
                    break
    if invalid:
        return replace(verdict, confident=False, passed_speedup=False, detail=invalid), SpeedStageDecision.NO_DECISION
    lower = candidate_rates[0] / max(baseline_rates)
    upper = candidate_rates[0] / min(baseline_rates)
    if lower >= verdict.required:
        decision = SpeedStageDecision.PASS
        detail = "candidate clears the bound against both stable baseline observations"
    elif upper < 1.0 + policy.min_margin:
        decision = SpeedStageDecision.FAIL
        detail = "candidate does not clear the speed floor against either stable baseline"
    else:
        decision = SpeedStageDecision.NO_DECISION
        detail = "measurement uncertainty crosses the speed decision boundary"
    return replace(
        verdict, speedup=lower, passed_speedup=decision is SpeedStageDecision.PASS,
        detail=detail,
    ), decision


__all__ = [
    "DECODE_ROLES",
    "PREFILL_LANE_ROLES",
    "SPEED_FAIL_REASONS",
    "SpeedStageDecision",
    "fail_reason",
    "invariant_decision",
    "resident_speed_roles",
    "speed_grade",
]

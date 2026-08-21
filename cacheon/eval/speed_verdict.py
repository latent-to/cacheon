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

Version 6 runs B/C, then B-prime only when a legal bookend could reverse it.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from cacheon.eval.scoring import RawSpeedEvidenceError, SpeedupVerdict, relative_spread, score_speedup

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


def resident_speed_roles(version: int, count: int) -> tuple[str, ...] | None:
    roles = {2: ("B", "C"), 3: ("B", "C", "B_prime")}
    if version < 6:
        roles[5] = ("B", "C", "B_prime", "C_prime", "B_double_prime")
        roles.pop(2)
    return roles.get(count)


def v6_decision_limits(
    policy: "ResidentSpeedPolicy",
) -> tuple[float, float]:
    """B/C limits invariant to every sealed or discarded B-prime."""

    margin = policy.min_margin
    multiplier = policy.noise_multiplier
    noise = policy.max_noise
    clear_fail_below = 1.0 + margin
    high_bookend = (2.0 + noise) / (2.0 - noise)
    clear_pass_at = high_bookend * (
        1.0 + max(margin, multiplier * noise)
    )
    return clear_fail_below, clear_pass_at


def v6_grade(
    policy: "ResidentSpeedPolicy", baseline: object, candidate: object,
    baseline_after: object | None = None,
) -> tuple[SpeedupVerdict, SpeedupVerdict, SpeedStageDecision]:
    initial, initial_decision = speed_grade(
        policy, [baseline], [candidate], concluding=False
    )
    if baseline_after is None and initial_decision is None:
        raise RawSpeedEvidenceError("resident v6 evidence omitted required B-prime")
    if baseline_after is None:
        return initial, initial, initial_decision  # type: ignore[return-value]
    if initial_decision is not None:
        raise RawSpeedEvidenceError("resident v6 clear decision added B-prime")
    final, decision = speed_grade(
        policy, [baseline, baseline_after], [candidate], concluding=True
    )
    if decision is None:
        raise RawSpeedEvidenceError("resident v6 evidence has no terminal decision")
    return initial, final, decision


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
    if policy.version >= 6 and not concluding and len(baseline_rates) == len(candidate_rates) == 1:
        fail_ratio, pass_ratio = v6_decision_limits(policy)
        conditioning = policy.conditioning_regression(baselines[0], candidates[0])
        decision = (
            SpeedStageDecision.FAIL
            if conditioning or verdict.speedup < fail_ratio
            else SpeedStageDecision.PASS
            if verdict.speedup >= pass_ratio
            else None
        )
        return replace(
            verdict,
            noise=0.0,
            required=pass_ratio,
            passed_speedup=decision is SpeedStageDecision.PASS,
            confident=decision is not None,
            detail=(
                "conditioning regression in v6 B/C precheck"
                if conditioning
                else f"v6 B/C ratio {verdict.speedup:.3f}; invariant bounds "
                f"[{fail_ratio:.3f}, {pass_ratio:.3f})"
            ),
        ), decision
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
    "SPEED_FAIL_REASONS",
    "SpeedStageDecision",
    "fail_reason",
    "invariant_decision",
    "resident_speed_roles",
    "speed_grade",
    "v6_decision_limits",
    "v6_grade",
]

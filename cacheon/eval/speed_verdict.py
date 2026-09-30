"""The speed-stage decision and the name of what a speed FAIL proved.

The B/C/B-prime graders (policies 8-15) left with the batch-cell evaluator;
replay policies grade in ``service_capacity``. The decision enum keeps this
module path because retained continuations name it.
"""

from __future__ import annotations

from enum import Enum

from cacheon.eval.scoring import SpeedupVerdict


class SpeedStageDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_DECISION = "NO_DECISION"


SPEED_FAIL_REASONS = frozenset({"candidate_slower", "speed_threshold_not_met", "service_contract_not_met"})


def fail_reason(verdict: SpeedupVerdict) -> str:
    """Name what a speed FAIL actually proved.

    A bar of ``1 + u`` can only call a candidate slower once its speedup falls
    below the mirrored bound ``1 - u``. Anything else inside the band proved only
    that the bar was not cleared -- an ordinary competitive miss, not a regression.
    """

    if verdict.speedup < 2.0 - verdict.required:
        return "candidate_slower"
    return "speed_threshold_not_met"


__all__ = [
    "SPEED_FAIL_REASONS",
    "SpeedStageDecision",
    "fail_reason",
]

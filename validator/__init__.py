"""Public types and helpers for the Cacheon validator (CPU / chain side).

Submodules implement state files, chain calls, challenger filtering, and
the main loop. Heavy model work is **not** here: the loop takes an
``eval_fn`` that the application wires up (Docker-based GPU evaluation).
Re-exports below are the stable surface for callers that only need data
shapes and selection logic.
"""

from .state import (
    EvaluationRecord,
    KingRecord,
    ValidatorState,
)
from .eval_schema import (
    ChatMessage,
    EvaluationJob,
    EvaluationResult,
    PerPromptResult,
    Prompt,
)
from .challengers import select_challengers
from .chain import (
    CommitmentRecord,
    build_commitments,
    build_winner_take_all_weights,
    parse_commitment_data,
)

__all__ = [
    "ChatMessage",
    "CommitmentRecord",
    "EvaluationJob",
    "EvaluationRecord",
    "EvaluationResult",
    "KingRecord",
    "PerPromptResult",
    "Prompt",
    "ValidatorState",
    "build_commitments",
    "build_winner_take_all_weights",
    "parse_commitment_data",
    "select_challengers",
]

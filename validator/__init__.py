"""Public types and helpers for the Cacheon validator (CPU / chain side).

Submodules implement state files, chain calls, challenger filtering, and
the main loop. Heavy model work is **not** here: the loop takes an
``eval_fn`` that the application wires up (Docker-based GPU evaluation).
Re-exports below are the stable surface for callers that only need data
shapes and selection logic.
"""

from .state import (
    EvaluationRecord,
    WinnerRecord,
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
    build_competition_weights,
    parse_commitment_data,
)

__all__ = [
    "ChatMessage",
    "CommitmentRecord",
    "EvaluationJob",
    "EvaluationRecord",
    "EvaluationResult",
    "PerPromptResult",
    "Prompt",
    "ValidatorState",
    "WinnerRecord",
    "build_commitments",
    "build_competition_weights",
    "parse_commitment_data",
    "select_challengers",
]

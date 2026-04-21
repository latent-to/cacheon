"""Phase 5 Part A — CPU-side validator package.

Handles the chain-scanning loop, local state persistence, and challenger
selection. The actual GPU-side evaluation (Phase 5 Part B) is injected as
a callable — this package has zero GPU or HuggingFace dependency.
"""

from .state import (
    EvaluationRecord,
    KingRecord,
    ValidatorState,
)
from .challengers import select_challengers
from .chain import (
    CommitmentRecord,
    build_commitments,
    build_winner_take_all_weights,
    parse_commitment_data,
)

__all__ = [
    "CommitmentRecord",
    "EvaluationRecord",
    "KingRecord",
    "ValidatorState",
    "build_commitments",
    "build_winner_take_all_weights",
    "parse_commitment_data",
    "select_challengers",
]

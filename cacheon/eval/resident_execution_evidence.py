"""Generation-scoped proof that a resident candidate really executed.

The closing swap counts rank receipts. ``UNOBSERVED`` means broken evidence;
observed short coverage means the candidate did not run cleanly everywhere;
only exact rank coverage can authorize speed grading or screen promotion.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sentinel for "not observable", distinct from an observed count of zero. A
# generation is non-negative and a rank count is non-negative, so a negative
# value cannot collide with a real observation.
UNOBSERVED = -1


@dataclass(frozen=True)
class ResidentExecutionEvidence:
    """Execution evidence for the generation a swap closed.

    ``prior_generation`` is the scope the counts describe, not the generation
    being swapped in — a swap reports the receipts of the generation it is
    ending, because that scope is final only once nothing more can run under it.
    """

    prior_generation: int
    prior_execution_ranks: int

    def __post_init__(self) -> None:
        for field in ("prior_generation", "prior_execution_ranks"):
            value = getattr(self, field)
            if type(value) is not int or value < UNOBSERVED:
                raise ValueError(f"resident execution evidence {field} is invalid")

    @property
    def observed(self) -> bool:
        return (
            self.prior_generation >= 0
            and self.prior_execution_ranks >= 0
        )

    def proves_execution(self, *, generation: int, expected_ranks: int) -> bool:
        """True only when every rank cleanly executed under exactly ``generation``."""

        return (
            self.observed
            and self.prior_generation == generation
            and expected_ranks >= 1
            and self.prior_execution_ranks == expected_ranks
        )


UNOBSERVED_EVIDENCE = ResidentExecutionEvidence(UNOBSERVED, UNOBSERVED)


def _rank_executed_cleanly(counts: object) -> bool | None:
    """Whether one rank ran the candidate; ``None`` when its report is unusable.

    ``completed`` is the positive evidence. ``fallback`` and ``load_failed`` are
    disqualifying rather than merely absent evidence: they record that the seam
    selected the candidate and then served the trusted baseline instead, which
    is a stock measurement wearing a candidate's name.
    """

    if not isinstance(counts, dict):
        return None
    values = []
    for kind in ("completed", "fallback", "load_failed"):
        value = counts.get(kind)
        if type(value) is not int or value < 0:
            return None
        values.append(value)
    completed, fallback, load_failed = values
    return completed > 0 and fallback == 0 and load_failed == 0


def summarize_rank_acks(
    rows: dict, *, tp_size: int
) -> ResidentExecutionEvidence:
    """Reduce per-rank swap acks to one execution fact; never raises.

    Any rank that cannot be read, disagrees about which generation is closing,
    or reports malformed counts collapses the whole summary to UNOBSERVED. A
    partial reading is not evidence — it would let a rank that silently stopped
    reporting look like a rank that cleanly executed.
    """

    if type(tp_size) is not int or tp_size < 1:
        return UNOBSERVED_EVIDENCE
    generations: set[int] = set()
    executed = 0
    observable = True
    for rank in range(tp_size):
        row = rows.get(rank)
        if not isinstance(row, dict):
            observable = False
            continue
        prior = row.get("prior_generation")
        if type(prior) is not int:
            observable = False
            continue
        generations.add(prior)
        if prior < 0:
            # The lane's first swap closes nothing; there is no scope to count.
            continue
        clean = _rank_executed_cleanly(row.get("prior_receipts"))
        if clean is None:
            observable = False
            continue
        executed += int(clean)
    if len(generations) != 1:
        return UNOBSERVED_EVIDENCE
    generation = generations.pop()
    if generation < 0 or not observable:
        return ResidentExecutionEvidence(max(generation, UNOBSERVED), UNOBSERVED)
    return ResidentExecutionEvidence(generation, executed)


__all__ = [
    "UNOBSERVED",
    "UNOBSERVED_EVIDENCE",
    "ResidentExecutionEvidence",
    "summarize_rank_acks",
]

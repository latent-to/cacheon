"""Generation-scoped resident execution evidence tests."""

from __future__ import annotations

import pytest

from cacheon.eval.resident_execution_evidence import (
    UNOBSERVED,
    UNOBSERVED_EVIDENCE,
    ResidentExecutionEvidence,
    summarize_rank_acks,
)


def _ack(
    prior: int, *, fired: int = 1, completed: int = 1,
    fallback: int = 0, load_failed: int = 0,
):
    return {
        "prior_generation": prior,
        "prior_receipts": {
            "fired": fired,
            "completed": completed,
            "fallback": fallback,
            "load_failed": load_failed,
        },
    }


class TestPredicate:
    def test_a_full_clean_rank_group_proves_execution(self) -> None:
        evidence = ResidentExecutionEvidence(7, 4)
        assert evidence.observed
        assert evidence.proves_execution(generation=7, expected_ranks=4)

    @pytest.mark.parametrize(
        "evidence, generation, ranks",
        [
            (UNOBSERVED_EVIDENCE, 7, 4),  # nothing was seen
            (ResidentExecutionEvidence(7, UNOBSERVED), 7, 4),  # counts unseen
            (ResidentExecutionEvidence(7, 0), 7, 4),  # seen, and nothing ran
            (ResidentExecutionEvidence(7, 3), 7, 4),  # short of the group
            (ResidentExecutionEvidence(6, 4), 7, 4),  # a different generation
            (ResidentExecutionEvidence(7, 4), 7, 0),  # an empty group proves nothing
        ],
    )
    def test_anything_short_of_that_proves_nothing(
        self, evidence, generation, ranks
    ) -> None:
        assert not evidence.proves_execution(
            generation=generation, expected_ranks=ranks
        )

    def test_evidence_rejects_values_below_the_sentinel(self) -> None:
        with pytest.raises(ValueError):
            ResidentExecutionEvidence(-2, 4)
        with pytest.raises(ValueError):
            ResidentExecutionEvidence(7, -2)


class TestSummary:
    def test_every_rank_clean_counts_every_rank(self) -> None:
        rows = {rank: _ack(7) for rank in range(4)}
        assert summarize_rank_acks(rows, tp_size=4) == ResidentExecutionEvidence(7, 4)

    def test_a_rank_that_fell_back_did_not_execute_the_candidate(self) -> None:
        # Falling back means the seam selected the candidate and then served the
        # trusted baseline: a stock measurement wearing a candidate's name.
        rows = {rank: _ack(7) for rank in range(4)}
        rows[2] = _ack(7, completed=1, fallback=1)
        assert summarize_rank_acks(rows, tp_size=4) == ResidentExecutionEvidence(7, 3)

    def test_a_rank_that_failed_to_load_did_not_execute_the_candidate(self) -> None:
        rows = {rank: _ack(7) for rank in range(4)}
        rows[0] = _ack(7, completed=0, load_failed=1)
        assert summarize_rank_acks(rows, tp_size=4) == ResidentExecutionEvidence(7, 3)

    def test_a_rank_that_ran_nothing_is_counted_as_such(self) -> None:
        rows = {rank: _ack(7, completed=0) for rank in range(4)}
        assert summarize_rank_acks(rows, tp_size=4) == ResidentExecutionEvidence(7, 0)

    def test_completion_without_selection_is_not_execution(self) -> None:
        rows = {rank: _ack(7, fired=0) for rank in range(4)}
        assert summarize_rank_acks(rows, tp_size=4) == ResidentExecutionEvidence(7, 0)

    @pytest.mark.parametrize(
        "corrupt",
        [
            None,
            "not-a-row",
            {"prior_generation": "7", "prior_receipts": {}},
            {"prior_generation": 7},
            {"prior_generation": 7, "prior_receipts": "not-a-dict"},
            {"prior_generation": 7, "prior_receipts": {"completed": -1}},
            {"prior_generation": 7, "prior_receipts": {"completed": True}},
        ],
    )
    def test_one_unreadable_rank_makes_the_whole_group_unobserved(
        self, corrupt
    ) -> None:
        # A partial reading is not evidence: a rank that silently stopped
        # reporting would otherwise be indistinguishable from a rank that ran.
        rows = {rank: _ack(7) for rank in range(4)}
        rows[1] = corrupt
        summary = summarize_rank_acks(rows, tp_size=4)
        assert summary.prior_execution_ranks == UNOBSERVED
        assert not summary.observed

    def test_a_missing_rank_makes_the_group_unobserved(self) -> None:
        rows = {rank: _ack(7) for rank in range(3)}
        assert not summarize_rank_acks(rows, tp_size=4).observed

    def test_ranks_disagreeing_about_the_closed_generation_are_unobserved(
        self,
    ) -> None:
        rows = {rank: _ack(7) for rank in range(4)}
        rows[3] = _ack(6)
        assert summarize_rank_acks(rows, tp_size=4) == UNOBSERVED_EVIDENCE

    def test_the_lanes_first_swap_closes_nothing_and_is_not_an_error(self) -> None:
        rows = {rank: {"prior_generation": UNOBSERVED} for rank in range(4)}
        summary = summarize_rank_acks(rows, tp_size=4)
        assert summary == UNOBSERVED_EVIDENCE
        assert not summary.observed

    @pytest.mark.parametrize("tp_size", [0, -1, "4", None])
    def test_an_invalid_rank_group_is_unobserved_rather_than_raising(
        self, tp_size
    ) -> None:
        rows = {rank: _ack(7) for rank in range(4)}
        assert summarize_rank_acks(rows, tp_size=tp_size) == UNOBSERVED_EVIDENCE

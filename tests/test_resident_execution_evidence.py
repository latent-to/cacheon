"""Generation-scoped resident execution evidence tests.

``LANE_A`` is what mainnet actually printed: the four rank rows of one real
candidate generation on lane A (2026-08-23, msa_block_score, TP4), captured
from the retained container log. Every reduction below is exercised against
those bytes first and synthetic variations second.
"""

from __future__ import annotations

import json

import pytest

from cacheon.eval.continuation_codec import ContinuationCodecError
from cacheon.eval.resident_execution_evidence import (
    EXECUTION_CODEC,
    UNOBSERVED,
    UNOBSERVED_EVIDENCE,
    RankExecution,
    ResidentExecutionEvidence,
    SlotExecution,
    summarize_rank_acks,
)

BUNDLE = "/cacheon/swap-intake/74de67064a1da8d1de1788d247b585f5ca83e678d9d69b4f2fa8f7071e170e9c"
SLOT = "attention.msa_block_score"
LANE_A = {
    rank: {
        "active": [
            {"bundle": BUNDLE, "pid": 154 + rank, "rank": rank, "slots": [SLOT], "world_size": 4}
        ],
        "completed": [
            {"calls": 1140, "captured": True, "pid": 154 + rank, "rank": rank, "slot": SLOT, "world_size": 4}
        ],
    }
    for rank in range(4)
}


def _ack(prior: int, rank: int, rows: dict | None = None) -> dict:
    return {"prior_generation": prior, "prior_rows": LANE_A[rank] if rows is None else rows}


def _lane(prior: int = 7, **overrides: dict) -> dict:
    rows = {rank: _ack(prior, rank) for rank in range(4)}
    for rank, row in overrides.items():
        rows[int(rank)] = row
    return rows


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

    def test_rows_cannot_contradict_their_count(self) -> None:
        rows = tuple(RankExecution(rank, True) for rank in range(2))
        with pytest.raises(ValueError):
            ResidentExecutionEvidence(7, 3, rows)  # more clean ranks than rows
        with pytest.raises(ValueError):
            ResidentExecutionEvidence(7, 1, rows[::-1])  # rows out of rank order


class TestRows:
    def test_the_real_lane_a_rows_reduce_to_one_clean_rank_each(self) -> None:
        reduced = RankExecution.from_receipts(2, LANE_A[2])
        assert reduced == RankExecution(2, True, "", (SlotExecution(SLOT, 1140, True),))
        assert reduced.clean(eager_slots=frozenset())

    def test_rows_survive_the_wire_and_the_summary_line(self) -> None:
        reduced = RankExecution.from_receipts(0, LANE_A[0])
        wire = json.loads(json.dumps(EXECUTION_CODEC.encode(reduced)))
        assert EXECUTION_CODEC.decode(wire) == reduced
        with pytest.raises(ContinuationCodecError):
            EXECUTION_CODEC.decode({**wire, "value": {**wire["value"], "extra": 1}})

    def test_a_registered_slot_that_was_never_called_stays_visible(self) -> None:
        rows = {"active": LANE_A[0]["active"]}
        reduced = RankExecution.from_receipts(0, rows)
        assert reduced.slots == (SlotExecution(SLOT, 0, None),)
        assert not reduced.clean(eager_slots=frozenset())

    def test_a_slot_outside_the_captured_graph_is_not_clean(self) -> None:
        """The phantom-pass shape: called during eager warmup, absent from the
        graph every scored window replays."""

        reduced = RankExecution(0, True, "", (SlotExecution(SLOT, 1140, False),))
        assert not reduced.clean(eager_slots=frozenset())
        unrecorded = RankExecution(0, True, "", (SlotExecution(SLOT, 1140, None),))
        assert not unrecorded.clean(eager_slots=frozenset())
        # A prefill seam SGLang serves eagerly is exempt, and only it.
        assert unrecorded.clean(eager_slots=frozenset({SLOT}))

    def test_a_raised_entry_and_a_failed_load_are_carried_and_not_clean(self) -> None:
        rows = {
            **LANE_A[1],
            "failed": [{"slot": SLOT, "error_type": "RuntimeError", "error": "CUDA error\x00x", "pid": 155}],
        }
        reduced = RankExecution.from_receipts(1, rows)
        assert reduced.slots[0].error == "RuntimeError: CUDA error x"
        assert not reduced.clean(eager_slots=frozenset())
        fell_back = RankExecution.from_receipts(
            1, {"load_failed": [{"reason": "ImportError: no module", "pid": 155}]}
        )
        assert fell_back == RankExecution(1, False, "ImportError: no module")
        assert not fell_back.clean(eager_slots=frozenset())

    def test_routing_reasons_are_kept_once_each(self) -> None:
        rows = {
            **LANE_A[0],
            "not_selected": [
                {
                    "slot": SLOT,
                    "reasons": [
                        {"outcome": "out_of_domain", "fields": ["num_tokens"]},
                        {"outcome": "out_of_domain", "fields": ["num_tokens"]},
                        {"outcome": "no_variant", "fields": []},
                    ],
                }
            ],
        }
        reduced = RankExecution.from_receipts(0, rows)
        assert reduced.slots[0].skipped == (
            "out_of_domain on num_tokens",
            "no_variant on unrecorded",
        )


class TestSummary:
    def test_every_rank_clean_counts_every_rank(self) -> None:
        summary = summarize_rank_acks(_lane(), tp_size=4)
        assert summary.prior_execution_ranks == 4
        assert summary.proves_execution(generation=7, expected_ranks=4)
        assert [row.rank for row in summary.ranks] == [0, 1, 2, 3]

    def test_a_rank_that_failed_to_load_did_not_execute_the_candidate(self) -> None:
        rows = _lane(**{"0": _ack(7, 0, {"load_failed": [{"reason": "boom"}]})})
        summary = summarize_rank_acks(rows, tp_size=4)
        assert summary.prior_execution_ranks == 3
        assert summary.ranks[0].load_error == "boom"

    def test_a_rank_outside_the_graph_did_not_execute_the_candidate(self) -> None:
        outside = json.loads(json.dumps(LANE_A[3]))
        outside["completed"][0]["captured"] = False
        rows = _lane(**{"3": _ack(7, 3, outside)})
        assert summarize_rank_acks(rows, tp_size=4).prior_execution_ranks == 3

    def test_a_rank_that_ran_nothing_is_counted_as_such(self) -> None:
        rows = {rank: _ack(7, rank, {}) for rank in range(4)}
        assert summarize_rank_acks(rows, tp_size=4).prior_execution_ranks == 0

    @pytest.mark.parametrize(
        "corrupt",
        [
            None,
            "not-a-row",
            {"prior_generation": "7", "prior_rows": {}},
            {"prior_generation": 7},
            {"prior_generation": 7, "prior_rows": None},  # the rank could not read
            {"prior_generation": 7, "prior_rows": "not-a-dict"},
            {"prior_generation": 7, "prior_rows": {"active": [{"slots": ["bad slot"]}]}},
        ],
    )
    def test_one_unreadable_rank_makes_the_whole_group_unobserved(
        self, corrupt
    ) -> None:
        # A partial reading is not evidence: a rank that silently stopped
        # reporting would otherwise be indistinguishable from a rank that ran.
        rows = _lane(**{"1": corrupt})
        summary = summarize_rank_acks(rows, tp_size=4)
        assert summary.prior_execution_ranks == UNOBSERVED
        assert summary.ranks == ()
        assert not summary.observed

    def test_a_missing_rank_makes_the_group_unobserved(self) -> None:
        rows = {rank: _ack(7, rank) for rank in range(3)}
        assert not summarize_rank_acks(rows, tp_size=4).observed

    def test_ranks_disagreeing_about_the_closed_generation_are_unobserved(
        self,
    ) -> None:
        rows = _lane(**{"3": _ack(6, 3)})
        assert summarize_rank_acks(rows, tp_size=4) == UNOBSERVED_EVIDENCE

    def test_the_lanes_first_swap_closes_nothing_and_is_not_an_error(self) -> None:
        rows = {rank: {"prior_generation": UNOBSERVED, "prior_rows": {}} for rank in range(4)}
        summary = summarize_rank_acks(rows, tp_size=4)
        assert summary == UNOBSERVED_EVIDENCE
        assert not summary.observed

    @pytest.mark.parametrize("tp_size", [0, -1, "4", None])
    def test_an_invalid_rank_group_is_unobserved_rather_than_raising(
        self, tp_size
    ) -> None:
        assert summarize_rank_acks(_lane(), tp_size=tp_size) == UNOBSERVED_EVIDENCE


def test_a_saturated_rank_group_still_fits_one_control_frame() -> None:
    """TP8, four slots, every text and list bound at its cap: the rows must
    ride the swap frame, so their bounds are sized to the frame, not the reverse."""

    from cacheon.eval.oci_session_protocol import MAX_CONTROL_BYTES
    from cacheon.eval.resident_execution_evidence import (
        MAX_EXECUTION_TEXT,
        MAX_SKIPPED_REASONS,
    )

    def slot(index: int) -> SlotExecution:
        return SlotExecution(
            f"slot.{index}", 10**6, True, "E" * MAX_EXECUTION_TEXT,
            tuple(("r" * (MAX_EXECUTION_TEXT - 1)) + str(i) for i in range(MAX_SKIPPED_REASONS)),
        )

    evidence = ResidentExecutionEvidence(
        9, 8,
        tuple(
            RankExecution(rank, True, "L" * MAX_EXECUTION_TEXT, tuple(slot(n) for n in range(4)))
            for rank in range(8)
        ),
    )
    encoded = json.dumps(EXECUTION_CODEC.encode(evidence), separators=(",", ":"))
    assert len(encoded) < MAX_CONTROL_BYTES - 2048  # headroom for the frame's own fields

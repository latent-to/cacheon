from __future__ import annotations

import pytest

from cacheon.eval.oci_outer_session import (
    OuterSessionCandidateError,
    OuterSessionTimeoutError,
)
from cacheon.eval.oci_resident_session import ResidentBatchEvidence, SwapReceipt
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.resident_execution_evidence import ResidentExecutionEvidence
from cacheon.eval.resident_queue import (
    ResidentQueueError,
    ResidentScreenLoop,
    ScreenCandidate,
    ScreenPolicy,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _evidence(tokens: int = 4) -> BatchEvidence:
    return BatchEvidence(
        (
            PromptEvidence(
                tuple(range(tokens)),
                tuple(((-0.5, 0),) for _ in range(tokens)),
                5,
            ),
        )
    )


class FakeSession:
    """Plays back configured throughputs; models generation bookkeeping."""

    def __init__(self, stock_rate: float, candidate_rates: dict[str, float],
                 slots: dict[str, tuple[str, ...]] | None = None,
                 stock_drift_after: int | None = None,
                 stock_rates: list[float] | None = None,
                 execution_ranks: int = 1) -> None:
        self.stock_rate = stock_rate
        self.candidate_rates = candidate_rates
        self.slots = slots or {
            digest: ("moe.fused_experts",) for digest in candidate_rates
        }
        self.stock_drift_after = stock_drift_after
        self.stock_rates = stock_rates
        self.execution_ranks = execution_ranks
        self.generation = 0
        self.active: str | None = None
        self.batch_count = 0
        self.stock_reads = 0
        self.swaps: list[str | None] = []
        self.clock = 0.0
        self.timeouts: list[float | None] = []
        self.candidate_outlives_budget = False

    def swap(self, bundle_digest: str | None) -> SwapReceipt:
        prior_generation, prior_active = self.generation, self.active
        self.generation += 1
        self.active = bundle_digest
        self.swaps.append(bundle_digest)
        self.clock += 30.0
        return SwapReceipt(
            len(self.swaps) - 1,
            self.generation,
            bundle_digest,
            () if bundle_digest is None else self.slots[bundle_digest],
            self.clock - 30.0,
            self.clock,
            ResidentExecutionEvidence(
                prior_generation, self.execution_ranks if prior_active else 0
            ),
            1,
        )

    def execute_batch(self, prompts, *, canary: bool = False, timeout_s=None):
        assert not canary or self.active is None
        self.timeouts.append(timeout_s)
        tokens = 1000
        if self.active is not None and self.candidate_outlives_budget:
            assert timeout_s is not None
            self.clock += timeout_s
            raise OuterSessionTimeoutError("session response read timed out")
        if self.active is None:
            rate = self.stock_rate
            self.stock_reads += 1
            if self.stock_rates is not None and self.stock_reads <= len(self.stock_rates):
                rate = self.stock_rates[self.stock_reads - 1]
            elif (
                self.stock_drift_after is not None
                and self.stock_reads > self.stock_drift_after
            ):
                rate *= 0.90
        else:
            rate = self.candidate_rates[self.active]
        elapsed = tokens / rate
        started = self.clock
        self.clock += elapsed
        row = ResidentBatchEvidence(
            self.batch_count,
            f"{self.batch_count + 5:032x}",
            f"{self.batch_count + 6:032x}".replace("0", "9", 1),
            self.generation,
            () if self.active is None else self.slots[self.active],
            canary,
            started,
            self.clock,
            tokens,
            _evidence(),
        )
        self.batch_count += 1
        return row


def _candidate(digest: str, name: str = "cand") -> ScreenCandidate:
    return ScreenCandidate(name, digest, ("moe.fused_experts",))


class _Screened:
    def __init__(
        self,
        session: FakeSession,
        candidates: list[ScreenCandidate],
        *,
        prompts: tuple[str, ...],
        policy: ScreenPolicy = ScreenPolicy(),
    ) -> None:
        loop = ResidentScreenLoop(session, prompts=prompts, policy=policy)
        self.verdicts = []
        for candidate in candidates:
            result = loop.screen(candidate)
            if result is None:
                break
            self.verdicts.append(result)
            if loop.stopped_reason is not None:
                break
        self.unprocessed_candidate_ids = tuple(
            row.candidate_id for row in candidates[loop.processed :]
        )
        self.stopped_reason = loop.stopped_reason


class TestScreenQueue:
    def test_clear_winner_passes_without_escalation(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0})
        report = _Screened(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert verdict.passed
        assert not verdict.escalated
        assert report.stopped_reason is None
        # swap in, swap out — exactly two swaps for a clear verdict
        assert session.swaps == [DIGEST_A, None]

    def test_clear_loser_fails_without_escalation(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 80.0})
        report = _Screened(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert not verdict.passed
        assert verdict.failure is None
        assert not verdict.escalated

    def test_borderline_escalates_to_five_legs(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 101.0})
        report = _Screened(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert verdict.escalated
        assert len(verdict.candidate_throughputs) == 2
        assert len(verdict.baseline_throughputs) == 3
        # in, out, in, out — four swaps for an escalated verdict
        assert session.swaps == [DIGEST_A, None, DIGEST_A, None]

    def test_queue_reuses_brackets_across_candidates(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0, DIGEST_B: 80.0})
        report = _Screened(
            session,
            [_candidate(DIGEST_A, "a"), _candidate(DIGEST_B, "b")],
            prompts=("p",),
        )
        assert [v.passed for v in report.verdicts] == [True, False]
        # Zero engine reloads: 1 discarded cold stock + 1 opening stock +
        # per candidate (C + closing B)
        assert session.batch_count == 6
        assert report.stopped_reason is None

    def test_slot_mismatch_fails_closed_and_returns_to_stock(self) -> None:
        session = FakeSession(
            100.0, {DIGEST_A: 112.0}, slots={DIGEST_A: ("other.slot",)}
        )
        report = _Screened(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert not verdict.passed
        assert "differ from expected" in (verdict.failure or "")
        assert session.swaps == [DIGEST_A, None]
        assert session.active is None

    def test_candidate_without_rank_execution_fails_before_promotion(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0}, execution_ranks=0)
        [verdict] = _Screened(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        ).verdicts
        assert not verdict.passed
        assert "execution not proven" in (verdict.failure or "")
        closing = verdict.to_dict()["swap_receipts"][1]
        assert closing["prior_execution_ranks"] == 0
        assert closing["expected_ranks"] == 1

    def test_canary_drift_stops_lifetime_and_withdraws_verdict(self) -> None:
        session = FakeSession(
            100.0,
            {DIGEST_A: 112.0, DIGEST_B: 112.0},
            # Discarded cold read + opening + first closing stay clean; drift
            # on the second candidate's closing stock read.
            stock_drift_after=3,
        )
        report = _Screened(
            session,
            [_candidate(DIGEST_A, "a"), _candidate(DIGEST_B, "b")],
            prompts=("p",),
        )
        assert report.stopped_reason is not None
        assert "recycle" in report.stopped_reason
        withdrawn = report.verdicts[-1]
        assert withdrawn.verdict is None
        assert "withdrawn" in (withdrawn.failure or "")
        assert withdrawn.candidate_id in report.unprocessed_candidate_ids

    def test_cold_first_stock_read_is_discarded_before_the_canary_band(self) -> None:
        """A cold opening must not become the canary reference.

        Mainnet 2026-08-25 pinned ``fmean(_stock[:-1])`` on the first batch after
        load (~5 tok/s) while every later stock read sat at ~13.9.  Discard the
        warmup; the band opens on the next read.
        """

        session = FakeSession(
            100.0,
            {DIGEST_A: 112.0},
            stock_rates=[5.0, 100.0, 100.0],
        )
        report = _Screened(session, [_candidate(DIGEST_A)], prompts=("p",))
        [verdict] = report.verdicts
        assert verdict.passed
        assert not verdict.withdrawn
        assert report.stopped_reason is None
        # Discarded cold read never entered the scored baselines.
        assert verdict.baseline_throughputs[0] == pytest.approx(100.0)
        assert session.stock_reads == 3  # discard + opening + closing

    def test_lifetime_budget_stops_queue(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0, DIGEST_B: 112.0})
        report = _Screened(
            session,
            [_candidate(DIGEST_A, "a"), _candidate(DIGEST_B, "b")],
            prompts=("p",),
            policy=ScreenPolicy(max_candidates_per_lifetime=1),
        )
        assert len(report.verdicts) == 1
        assert report.unprocessed_candidate_ids == ("b",)
        assert report.stopped_reason == "lifetime candidate budget exhausted"

    def test_empty_prompts_rejected(self) -> None:
        with pytest.raises(ResidentQueueError, match="prompt plan"):
            ResidentScreenLoop(
                FakeSession(100.0, {DIGEST_A: 110.0}), prompts=(),
            )


class TestCandidateTimeBudget:
    """Only the candidate read is bounded, by the stock read beside it."""

    def test_candidate_read_gets_budget_from_the_latest_stock_read(self) -> None:
        # Stock reads take 1000 tokens / 100 tok/s = 10 s; 10x is below the
        # 300 s floor, so the floor governs; stock and canary reads are unbounded.
        session = FakeSession(100.0, {DIGEST_A: 112.0, DIGEST_B: 112.0})
        loop = ResidentScreenLoop(session, prompts=("p",))
        loop.screen(_candidate(DIGEST_A, "a"))
        # discard, opening, candidate, closing
        assert session.timeouts == [None, None, 300.0, None]
        session.stock_rates = None
        loop.screen(_candidate(DIGEST_B, "b"))
        assert session.timeouts[4:] == [300.0, None]

    def test_budget_scales_with_a_slow_stock_read(self) -> None:
        # 1000 tokens / 20 tok/s = 50 s stock reads -> 10x = 500 s > floor.
        session = FakeSession(20.0, {DIGEST_A: 22.4})
        loop = ResidentScreenLoop(session, prompts=("p",))
        loop.screen(_candidate(DIGEST_A, "a"))
        assert session.timeouts[2] == pytest.approx(500.0)

    def test_candidate_outliving_its_budget_is_a_candidate_failure(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0})
        session.candidate_outlives_budget = True
        loop = ResidentScreenLoop(session, prompts=("p",))
        with pytest.raises(OuterSessionCandidateError) as caught:
            loop.screen(_candidate(DIGEST_A, "a"))
        assert caught.value.candidate_failure.startswith(
            "candidate_timeout: candidate read exceeded 300s (10x the 10.0s stock read, floor 300s)"
        )
        assert caught.value.candidate_failure_type == "CandidateExecutionFailure"
        assert isinstance(caught.value.__cause__, OuterSessionTimeoutError)
        # The stock read before it ran unbounded; nothing else was attempted.
        assert session.timeouts == [None, None, 300.0]
        assert session.swaps == [DIGEST_A]

    def test_stock_read_timeout_stays_infrastructure(self) -> None:
        class StockHangs(FakeSession):
            def execute_batch(self, prompts, *, canary=False, timeout_s=None):
                if self.active is None:
                    raise OuterSessionTimeoutError("session response read timed out")
                return super().execute_batch(prompts, canary=canary, timeout_s=timeout_s)

        loop = ResidentScreenLoop(StockHangs(100.0, {DIGEST_A: 112.0}), prompts=("p",))
        with pytest.raises(OuterSessionTimeoutError):
            loop.screen(_candidate(DIGEST_A, "a"))

    def test_policy_budget_and_validation(self) -> None:
        policy = ScreenPolicy(candidate_time_multiple=4.0, candidate_time_floor_s=60.0)
        assert policy.candidate_time_budget_s(10.0) == 60.0
        assert policy.candidate_time_budget_s(20.0) == 80.0
        for bad in ({"candidate_time_multiple": 1.0}, {"candidate_time_multiple": True},
                    {"candidate_time_floor_s": 0.0}, {"candidate_time_floor_s": 86_400.0}):
            with pytest.raises(ResidentQueueError, match="screen policy candidate_time"):
                ScreenPolicy(**bad)

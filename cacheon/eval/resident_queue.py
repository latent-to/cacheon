"""Queue scheduler for the resident (hot-swap) speed screen.

Runs N candidates through ONE resident engine lifetime with the same-lane
bracket structure proven by the 2026-07-20 pod probes:

    B_0  swap(k1)  C_1  swap(stock)  B_1  swap(k2)  C_2  swap(stock)  B_2 ...

Every stock read doubles as (a) the closing bracket of the previous candidate,
(b) the opening bracket of the next, and (c) a contamination canary — the
engine provably dispatches stock (the swap-out ack registered zero slots), so a
stock read that leaves the lifetime's stock band flags in-process tampering or
state rot and stops the lifetime for a recycle.  The very first batch after a
cold engine load is discarded before that band opens: it is CUDA/runtime
warmup, not a stock measurement (mainnet 2026-08-25: ~5.3 tok/s then ~13.9).

Verdicts reuse :func:`cacheon.eval.scoring.score_speedup` (noise-derived bar,
NO-DECISION on disagreeing brackets).  Borderline candidates escalate to the
five-leg shape (B C B' C' B'') by swapping back in — an escalation costs two
swaps and two reads, never an engine reload.

Trust tier: screen/routing only.  Payment and crown evidence still come from
the isolated per-candidate qualification path.  Non-swappable bundles
(aot_exports device artifacts, dep-patched trees) never enter this queue — the
seam refuses them — and are scheduled as dedicated launches by the caller.

This module is deliberately free of executor imports: it drives the
:class:`~cacheon.eval.oci_resident_session.ResidentOuterSession` API only, so it
tests without GPUs, containers, or engines.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Protocol, Sequence

from cacheon.eval.oci_outer_session import (
    OuterSessionCandidateError,
    OuterSessionTimeoutError,
)
from cacheon.eval.oci_resident_session import (
    ResidentBatchEvidence,
    SwapReceipt,
)
from cacheon.eval.scoring import SpeedupVerdict, score_speedup
from cacheon.stack_identity import require_sha256_hex


_CANDIDATE_ID = re.compile(r"[A-Za-z0-9_.:+-]{1,128}\Z")

# The exact failure recorded when a tripped canary voids the just-closed
# verdict.  Consumers branch on CandidateScreenVerdict.withdrawn, never on
# this string.
WITHDRAWN_FAILURE = "stock canary drifted beyond tolerance; evidence withdrawn"


class ResidentQueueError(ValueError):
    """A queue plan, policy, or session interaction is invalid."""


class ScreenSession(Protocol):
    """The subset of ResidentOuterSession the screen scheduler drives."""

    def swap(self, bundle_digest: str | None) -> SwapReceipt: ...
    def execute_batch(
        self,
        prompts: Sequence[str],
        *,
        canary: bool = False,
        timeout_s: float | None = None,
    ) -> ResidentBatchEvidence: ...


@dataclass(frozen=True)
class ScreenCandidate:
    """One swappable candidate, already staged in the swap intake."""

    candidate_id: str
    bundle_digest: str
    expected_slots: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or _CANDIDATE_ID.fullmatch(self.candidate_id) is None
        ):
            raise ResidentQueueError("screen candidate_id is invalid")
        try:
            require_sha256_hex(self.bundle_digest, field="screen bundle digest")
        except ValueError as exc:
            raise ResidentQueueError(str(exc)) from None
        slots = tuple(self.expected_slots)
        if (
            not slots
            or slots != tuple(sorted(set(slots)))
            or any(not isinstance(slot, str) or not slot for slot in slots)
        ):
            raise ResidentQueueError(
                "screen expected_slots must be nonempty sorted unique names"
            )
        object.__setattr__(self, "expected_slots", slots)


@dataclass(frozen=True)
class ScreenPolicy:
    """Bar, escalation, canary, and recycle policy for one screen pass.

    Defaults are pinned from the 2026-07-21 noise-qualification campaign on
    the production 4xB300 lane (8 interleaved null/bundle swap cycles, 9 stock
    reads, 16 recaptures): stock band 0.30%, worst null-cycle excursion 0.21%,
    zero nulls above a 1.005 bar.  min_margin sits above 2x the worst null
    excursion; canary_tolerance sits at ~4x the stock band so contamination
    trips it but honest drift does not.
    """

    min_margin: float = 0.0075
    noise_multiplier: float = 2.0
    max_noise: float = 0.10
    escalation_band: float = 0.02
    canary_tolerance: float = 0.012
    max_candidates_per_lifetime: int = 8
    # A candidate read outliving this budget is the candidate's speed, not
    # infrastructure: the stock read seconds earlier on the same engine and
    # prompts is the reference.  Sized for first-call JIT plus a slow kernel;
    # the 2026-09-02 mainnet bundle that motivated it needed hours.
    candidate_time_multiple: float = 10.0
    candidate_time_floor_s: float = 300.0

    def candidate_time_budget_s(self, stock_elapsed_s: float) -> float:
        """Wall-time budget for one candidate read beside one stock read."""

        return max(
            float(self.candidate_time_floor_s),
            float(self.candidate_time_multiple) * float(stock_elapsed_s),
        )

    def __post_init__(self) -> None:
        for name, value, low, high in (
            ("min_margin", self.min_margin, 0.0, 1.0),
            ("noise_multiplier", self.noise_multiplier, 0.0, 100.0),
            ("max_noise", self.max_noise, 0.0, 1.0),
            ("escalation_band", self.escalation_band, 0.0, 1.0),
            ("canary_tolerance", self.canary_tolerance, 0.0, 1.0),
            ("candidate_time_multiple", self.candidate_time_multiple, 1.0, 1_000.0),
            ("candidate_time_floor_s", self.candidate_time_floor_s, 0.0, 86_400.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not low < float(value) < high
            ):
                raise ResidentQueueError(f"screen policy {name} is invalid")
        if (
            type(self.max_candidates_per_lifetime) is not int
            or not 1 <= self.max_candidates_per_lifetime <= 1_000
        ):
            raise ResidentQueueError(
                "screen policy max_candidates_per_lifetime is invalid"
            )


@dataclass(frozen=True)
class CandidateScreenVerdict:
    """Routing verdict for one candidate; never payment evidence."""

    candidate_id: str
    bundle_digest: str
    slots: tuple[str, ...]
    baseline_throughputs: tuple[float, ...]
    candidate_throughputs: tuple[float, ...]
    verdict: SpeedupVerdict | None
    escalated: bool
    failure: str | None
    swap_receipts: tuple[SwapReceipt, ...]
    batch_indices: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return (
            self.failure is None
            and self.verdict is not None
            and self.verdict.passed_speedup
        )

    @property
    def withdrawn(self) -> bool:
        """Evidence voided by a tripped canary; re-screen on a fresh lifetime."""
        return self.failure == WITHDRAWN_FAILURE

    @property
    def rejected_dispatch(self) -> bool:
        """The engine registered slots other than the candidate declared."""
        return self.failure is not None and not self.withdrawn

    def to_dict(self) -> dict[str, object]:
        verdict = self.verdict
        return {
            "baseline_throughputs": [
                format(row, ".17g") for row in self.baseline_throughputs
            ],
            "batch_indices": list(self.batch_indices),
            "bundle_digest": self.bundle_digest,
            "candidate_id": self.candidate_id,
            "candidate_throughputs": [
                format(row, ".17g") for row in self.candidate_throughputs
            ],
            "escalated": self.escalated,
            "failure": self.failure,
            "slots": list(self.slots),
            "swap_receipts": [row.to_dict() for row in self.swap_receipts],
            "verdict": None
            if verdict is None
            else {
                "confident": verdict.confident,
                "detail": verdict.detail,
                "n_baselines": verdict.n_baselines,
                "n_candidates": verdict.n_candidates,
                "noise": format(verdict.noise, ".17g"),
                "passed_speedup": verdict.passed_speedup,
                "required": format(verdict.required, ".17g"),
                "speedup": format(verdict.speedup, ".17g"),
            },
        }


def _throughput(row: ResidentBatchEvidence) -> float:
    elapsed = row.elapsed_seconds
    if elapsed <= 0:
        raise ResidentQueueError("screen read clock did not advance")
    return row.token_numerator / elapsed


def _canary_drifted(
    stock_reads: Sequence[float], latest: float, *, tolerance: float
) -> bool:
    if len(stock_reads) < 2:
        return False
    reference = statistics.fmean(stock_reads[:-1])
    if reference <= 0:
        return True
    return abs(latest - reference) / reference > tolerance


def _is_borderline(verdict: SpeedupVerdict, *, band: float) -> bool:
    if not verdict.confident:
        return True
    return abs(verdict.speedup - verdict.required) <= band


class ResidentScreenLoop:
    """Incremental screen: one candidate at a time on one live session.

    The arena provider drives this over arrivals — candidates trickle in
    while the engine stays resident between them, so the loop carries the
    shared bracket (the last stock read), the lifetime's full stock band (the
    canary reference), and the stop condition across calls.

    ``screen`` returns ``None`` when the lifetime cannot accept the candidate
    (budget exhausted or already stopped) — the candidate was NOT touched and
    must be re-screened on a fresh lifetime.  A returned verdict can still be
    terminal for the lifetime: a tripped canary returns the WITHDRAWN verdict
    and sets ``stopped_reason``, so callers check it after every call.
    """

    def __init__(
        self,
        session: ScreenSession,
        *,
        prompts: Sequence[str],
        policy: ScreenPolicy = ScreenPolicy(),
    ) -> None:
        if type(policy) is not ScreenPolicy:
            raise ResidentQueueError("screen policy has the wrong type")
        prompt_plan = tuple(prompts)
        if not prompt_plan:
            raise ResidentQueueError("screen prompt plan is empty")
        self._session = session
        self._prompts = prompt_plan
        self._policy = policy
        self._stock: list[float] = []
        self._baseline_prev: float | None = None
        self._baseline_elapsed_s: float | None = None
        self._processed = 0
        self._stopped: str | None = None
        self._withdrawn_reference: float | None = None

    @property
    def stopped_reason(self) -> str | None:
        return self._stopped

    @property
    def withdrawn_reference(self) -> float | None:
        """Exact pre-drift stock mean used by the tripped canary."""

        return self._withdrawn_reference

    @property
    def stock_throughputs(self) -> tuple[float, ...]:
        return tuple(self._stock)

    @property
    def processed(self) -> int:
        """Candidates with retained verdicts (a withdrawn one does not count)."""
        return self._processed

    def _candidate_read(
        self, session: ScreenSession, prompt_plan: tuple[str, ...]
    ) -> ResidentBatchEvidence:
        """One candidate read, bounded by the latest stock read on this engine.

        The stock read seconds earlier proved the engine on these exact
        prompts, so a candidate read that outlives its budget is the
        candidate's speed and is raised as a terminal candidate failure, never
        as infrastructure (mainnet 2026-09-02: a per-expert Python fallback
        needed hours and surfaced as a 30-minute session timeout).
        """

        stock_elapsed_s = self._baseline_elapsed_s
        if stock_elapsed_s is None or stock_elapsed_s <= 0:
            raise ResidentQueueError("screen has no stock read to budget against")
        policy = self._policy
        budget_s = policy.candidate_time_budget_s(stock_elapsed_s)
        try:
            return session.execute_batch(prompt_plan, timeout_s=budget_s)
        except OuterSessionTimeoutError as exc:
            stated = (
                f"candidate_timeout: candidate read exceeded {budget_s:.0f}s "
                f"({policy.candidate_time_multiple:g}x the {stock_elapsed_s:.1f}s "
                f"stock read, floor {policy.candidate_time_floor_s:.0f}s)"
            )
            raise OuterSessionCandidateError(stated, candidate_failure=stated) from exc

    def screen(self, candidate: ScreenCandidate) -> CandidateScreenVerdict | None:
        if type(candidate) is not ScreenCandidate:
            raise ResidentQueueError("screen candidate is not exactly typed")
        if self._stopped is not None:
            return None
        policy = self._policy
        if self._processed >= policy.max_candidates_per_lifetime:
            self._stopped = "lifetime candidate budget exhausted"
            return None
        session = self._session
        prompt_plan = self._prompts
        if self._baseline_prev is None:
            # Pay the cold first-batch once, then take the real baseline.  The
            # discarded read must not enter `_stock`: otherwise
            # ``fmean(_stock[:-1])`` on the first bracket is the warmup number
            # and every honest closing/recovery read fails the 1.2% canary
            # (mainnet probe 2026-08-26: 5.32 then 13.87x3).
            session.execute_batch(prompt_plan, canary=True)
            opening = session.execute_batch(prompt_plan, canary=True)
            self._baseline_prev = _throughput(opening)
            self._baseline_elapsed_s = opening.elapsed_seconds
            self._stock.append(self._baseline_prev)

        receipts: list[SwapReceipt] = []
        batch_indices: list[int] = []
        failure: str | None = None
        candidate_reads: list[float] = []
        baseline_reads: list[float] = [self._baseline_prev]
        verdict: SpeedupVerdict | None = None
        escalated = False

        swap_in = session.swap(candidate.bundle_digest)
        receipts.append(swap_in)
        slots = swap_in.slots
        if slots != candidate.expected_slots:
            # The engine is live with unexpected dispatch; return to stock
            # before deciding anything else.
            failure = (
                f"registered slots {list(slots)!r} differ from expected "
                f"{list(candidate.expected_slots)!r}"
            )
        else:
            candidate_row = self._candidate_read(session, prompt_plan)
            batch_indices.append(candidate_row.batch_index)
            candidate_reads.append(_throughput(candidate_row))

        swap_out = session.swap(None)
        receipts.append(swap_out)
        if failure is None and not swap_out.execution.proves_execution(
            generation=swap_in.generation,
            expected_ranks=swap_out.expected_ranks,
        ):
            failure = (
                "candidate execution not proven before screen promotion "
                f"({swap_out.execution.prior_execution_ranks}/"
                f"{swap_out.expected_ranks} ranks)"
            )
        closing = session.execute_batch(prompt_plan, canary=True)
        batch_indices.append(closing.batch_index)
        closing_throughput = _throughput(closing)
        self._baseline_elapsed_s = closing.elapsed_seconds
        self._stock.append(closing_throughput)
        baseline_reads.append(closing_throughput)

        if failure is None:
            verdict = score_speedup(
                baseline_reads,
                candidate_reads,
                min_margin=policy.min_margin,
                k=policy.noise_multiplier,
                max_noise=policy.max_noise,
            )
            if _is_borderline(verdict, band=policy.escalation_band):
                escalated = True
                swap_in_2 = session.swap(candidate.bundle_digest)
                receipts.append(swap_in_2)
                if swap_in_2.slots != candidate.expected_slots:
                    failure = "escalation swap registered different slots"
                else:
                    candidate_row_2 = self._candidate_read(session, prompt_plan)
                    batch_indices.append(candidate_row_2.batch_index)
                    candidate_reads.append(_throughput(candidate_row_2))
                swap_out_2 = session.swap(None)
                receipts.append(swap_out_2)
                if failure is None and not swap_out_2.execution.proves_execution(
                    generation=swap_in_2.generation,
                    expected_ranks=swap_out_2.expected_ranks,
                ):
                    failure = "escalation candidate execution not proven"
                closing_2 = session.execute_batch(prompt_plan, canary=True)
                batch_indices.append(closing_2.batch_index)
                closing_throughput = _throughput(closing_2)
                self._baseline_elapsed_s = closing_2.elapsed_seconds
                self._stock.append(closing_throughput)
                baseline_reads.append(closing_throughput)
                if failure is None:
                    verdict = score_speedup(
                        baseline_reads,
                        candidate_reads,
                        min_margin=policy.min_margin,
                        k=policy.noise_multiplier,
                        max_noise=policy.max_noise,
                    )

        result = CandidateScreenVerdict(
            candidate.candidate_id,
            candidate.bundle_digest,
            slots,
            tuple(baseline_reads),
            tuple(candidate_reads),
            verdict,
            escalated,
            failure,
            tuple(receipts),
            tuple(batch_indices),
        )
        self._processed += 1
        self._baseline_prev = closing_throughput

        if _canary_drifted(
            self._stock, closing_throughput, tolerance=policy.canary_tolerance
        ):
            # The drifted read closed THIS candidate's bracket, so its verdict
            # is built on suspect evidence: withdraw it and re-screen the
            # candidate on the fresh lifetime along with the remainder.
            result = CandidateScreenVerdict(
                result.candidate_id,
                result.bundle_digest,
                result.slots,
                result.baseline_throughputs,
                result.candidate_throughputs,
                None,
                result.escalated,
                WITHDRAWN_FAILURE,
                result.swap_receipts,
                result.batch_indices,
            )
            self._processed -= 1
            self._withdrawn_reference = statistics.fmean(self._stock[:-1])
            self._stopped = (
                "stock canary drifted beyond tolerance after "
                f"{candidate.candidate_id}; lifetime requires recycle"
            )
        return result


__all__ = [
    "CandidateScreenVerdict",
    "ResidentQueueError",
    "ResidentScreenLoop",
    "ScreenCandidate",
    "ScreenPolicy",
    "ScreenSession",
    "WITHDRAWN_FAILURE",
]

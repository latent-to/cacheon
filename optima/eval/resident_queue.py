"""Queue scheduler for the resident (hot-swap) speed screen.

Runs N candidates through ONE resident engine lifetime with the same-lane
bracket structure proven by the 2026-07-20 pod probes:

    B_0  swap(k1)  C_1  swap(stock)  B_1  swap(k2)  C_2  swap(stock)  B_2 ...

Every stock read doubles as (a) the closing bracket of the previous candidate,
(b) the opening bracket of the next, and (c) a contamination canary — the
engine provably dispatches stock (the swap-out ack registered zero slots), so a
stock read that leaves the lifetime's stock band flags in-process tampering or
state rot and stops the lifetime for a recycle.

Verdicts reuse :func:`optima.eval.scoring.score_speedup` (noise-derived bar,
NO-DECISION on disagreeing brackets).  Borderline candidates escalate to the
five-leg shape (B C B' C' B'') by swapping back in — an escalation costs two
swaps and two reads, never an engine reload.

Trust tier: screen/routing only.  Payment and crown evidence still come from
the isolated per-candidate qualification path.  Non-swappable bundles
(aot_exports device artifacts, dep-patched trees) never enter this queue — the
seam refuses them — and are scheduled as dedicated launches by the caller.

This module is deliberately free of executor imports: it drives the
:class:`~optima.eval.oci_resident_session.ResidentOuterSession` API only, so it
tests without GPUs, containers, or engines.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Protocol, Sequence

from optima.eval.oci_resident_session import (
    ResidentBatchEvidence,
    SwapReceipt,
)
from optima.eval.scoring import SpeedupVerdict, score_speedup
from optima.stack_identity import require_sha256_hex


_CANDIDATE_ID = re.compile(r"[A-Za-z0-9_.:+-]{1,128}\Z")


class ResidentQueueError(ValueError):
    """A queue plan, policy, or session interaction is invalid."""


class ScreenSession(Protocol):
    """The subset of ResidentOuterSession the screen scheduler drives."""

    def swap(self, bundle_digest: str | None) -> SwapReceipt: ...
    def execute_batch(
        self, prompts: Sequence[str], *, canary: bool = False
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
    """Bar, escalation, canary, and recycle policy for one screen pass."""

    min_margin: float = 0.005
    noise_multiplier: float = 2.0
    max_noise: float = 0.10
    escalation_band: float = 0.02
    canary_tolerance: float = 0.03
    max_candidates_per_lifetime: int = 8

    def __post_init__(self) -> None:
        for name, value, low, high in (
            ("min_margin", self.min_margin, 0.0, 1.0),
            ("noise_multiplier", self.noise_multiplier, 0.0, 100.0),
            ("max_noise", self.max_noise, 0.0, 1.0),
            ("escalation_band", self.escalation_band, 0.0, 1.0),
            ("canary_tolerance", self.canary_tolerance, 0.0, 1.0),
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


@dataclass(frozen=True)
class ScreenReport:
    """The outcome of one resident lifetime's screen pass."""

    verdicts: tuple[CandidateScreenVerdict, ...]
    stock_throughputs: tuple[float, ...]
    unprocessed_candidate_ids: tuple[str, ...]
    stopped_reason: str | None


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


def run_resident_screen(
    session: ScreenSession,
    candidates: Sequence[ScreenCandidate],
    *,
    prompts: Sequence[str],
    policy: ScreenPolicy = ScreenPolicy(),
) -> ScreenReport:
    """Screen every candidate through one live engine; stop early on drift.

    The caller owns the engine lifetime: it opens the resident session, calls
    this, then closes.  When ``stopped_reason`` is set, the remaining
    candidates in ``unprocessed_candidate_ids`` must be re-screened on a fresh
    lifetime (recycle) — their absence here is scheduling state, not a verdict.
    A canary drift additionally WITHDRAWS the just-closed candidate's verdict
    (failure set, verdict ``None``, receipts retained for the record) and lists
    that candidate as unprocessed too: re-screen it on the fresh lifetime.
    """

    rows = tuple(candidates)
    if not rows or any(type(row) is not ScreenCandidate for row in rows):
        raise ResidentQueueError("screen candidates must be typed and nonempty")
    if len({row.candidate_id for row in rows}) != len(rows):
        raise ResidentQueueError("screen candidate ids must be unique")
    if type(policy) is not ScreenPolicy:
        raise ResidentQueueError("screen policy has the wrong type")
    prompt_plan = tuple(prompts)
    if not prompt_plan:
        raise ResidentQueueError("screen prompt plan is empty")

    verdicts: list[CandidateScreenVerdict] = []
    stock_throughputs: list[float] = []
    stopped_reason: str | None = None

    opening = session.execute_batch(prompt_plan, canary=True)
    baseline_prev = _throughput(opening)
    stock_throughputs.append(baseline_prev)

    processed = 0
    for candidate in rows:
        if processed >= policy.max_candidates_per_lifetime:
            stopped_reason = "lifetime candidate budget exhausted"
            break

        receipts: list[SwapReceipt] = []
        batch_indices: list[int] = []
        failure: str | None = None
        candidate_reads: list[float] = []
        baseline_reads: list[float] = [baseline_prev]
        verdict: SpeedupVerdict | None = None
        escalated = False
        slots: tuple[str, ...] = ()

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
            candidate_row = session.execute_batch(prompt_plan)
            batch_indices.append(candidate_row.batch_index)
            candidate_reads.append(_throughput(candidate_row))

        swap_out = session.swap(None)
        receipts.append(swap_out)
        closing = session.execute_batch(prompt_plan, canary=True)
        batch_indices.append(closing.batch_index)
        closing_throughput = _throughput(closing)
        stock_throughputs.append(closing_throughput)
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
                    candidate_row_2 = session.execute_batch(prompt_plan)
                    batch_indices.append(candidate_row_2.batch_index)
                    candidate_reads.append(_throughput(candidate_row_2))
                swap_out_2 = session.swap(None)
                receipts.append(swap_out_2)
                closing_2 = session.execute_batch(prompt_plan, canary=True)
                batch_indices.append(closing_2.batch_index)
                closing_throughput = _throughput(closing_2)
                stock_throughputs.append(closing_throughput)
                baseline_reads.append(closing_throughput)
                if failure is None:
                    verdict = score_speedup(
                        baseline_reads,
                        candidate_reads,
                        min_margin=policy.min_margin,
                        k=policy.noise_multiplier,
                        max_noise=policy.max_noise,
                    )

        verdicts.append(
            CandidateScreenVerdict(
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
        )
        processed += 1
        baseline_prev = closing_throughput

        if _canary_drifted(
            stock_throughputs, closing_throughput, tolerance=policy.canary_tolerance
        ):
            # The drifted read closed THIS candidate's bracket, so its verdict
            # is built on suspect evidence: withdraw it and re-screen the
            # candidate on the fresh lifetime along with the remainder.
            contaminated = verdicts.pop()
            verdicts.append(
                CandidateScreenVerdict(
                    contaminated.candidate_id,
                    contaminated.bundle_digest,
                    contaminated.slots,
                    contaminated.baseline_throughputs,
                    contaminated.candidate_throughputs,
                    None,
                    contaminated.escalated,
                    "stock canary drifted beyond tolerance; evidence withdrawn",
                    contaminated.swap_receipts,
                    contaminated.batch_indices,
                )
            )
            processed -= 1
            stopped_reason = (
                "stock canary drifted beyond tolerance after "
                f"{candidate.candidate_id}; lifetime requires recycle"
            )
            break

    unprocessed = tuple(row.candidate_id for row in rows[processed:])
    if unprocessed and stopped_reason is None:
        stopped_reason = "lifetime candidate budget exhausted"
    return ScreenReport(
        tuple(verdicts),
        tuple(stock_throughputs),
        unprocessed,
        stopped_reason,
    )


__all__ = [
    "CandidateScreenVerdict",
    "ResidentQueueError",
    "ScreenCandidate",
    "ScreenPolicy",
    "ScreenReport",
    "ScreenSession",
    "run_resident_screen",
]

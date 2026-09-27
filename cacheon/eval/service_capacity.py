"""Service rate of one engine on a fixed agent slice, and the paired verdict.

The agent arena scores speed at the service boundary: the fixed-work turn rate
an engine sustains at the sealed operating load, where the sealed contract
(decode floor, queue-inclusive TTFT bound) holds for the incumbent. Candidate
and incumbent replay identical work at that load in paired windows; the
verdict is the rate ratio, gated by attainment non-inferiority so a candidate
cannot buy rate by starving turns. Every number here is recomputed from
per-turn rows stamped by the trusted controller, so a retained run regrades to
the same verdict. The 2026-09-26 interpolated-capacity design (two bracket
loads, log-linear crossing) was withdrawn on 2026-09-27: its 8×12 pairings
carried a 9-15% same-engine spread.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from cacheon.eval.speed_verdict import SpeedStageDecision, invariant_decision

TURN_KINDS = frozenset({"main", "inner"})
TURN_STATUSES = frozenset({"ok", "error", "cancelled"})
READ_KEYS = ("arm", "window", "lane", "load")


class ServiceEvidenceError(ValueError):
    """A per-turn row, read, or bracket cannot be graded as stated."""


@dataclass(frozen=True)
class ServiceContract:
    """The promise a turn must meet: decode speed floor and a queue-inclusive TTFT bound."""

    decode_floor_tps: float
    ttft_bound_s: float
    attainment: float

    def __post_init__(self) -> None:
        for field in ("decode_floor_tps", "ttft_bound_s"):
            if not _finite_positive(getattr(self, field)):
                raise ServiceEvidenceError(f"service contract {field} must be positive")
        if not _finite_positive(self.attainment) or self.attainment > 1.0:
            raise ServiceEvidenceError("service contract attainment must be in (0, 1]")


@dataclass(frozen=True)
class TurnRecord:
    """One attempted turn as the controller observed it.

    ``root_session_id`` is the trace the turn belongs to, for subagent inner
    requests as well as main turns, so fixed work is checked per root. Nanosecond
    stamps are the controller's; the candidate never supplies time.
    """

    root_session_id: str
    kind: str
    ordinal: int
    credit_issued_ns: int
    request_start_ns: int
    first_token_ns: int | None
    request_end_ns: int
    prompt_tokens: int
    output_tokens: int
    status: str
    cached_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.root_session_id or self.kind not in TURN_KINDS or self.status not in TURN_STATUSES:
            raise ServiceEvidenceError("turn record identity is malformed")
        stamps = (self.credit_issued_ns, self.request_start_ns, self.request_end_ns)
        if any(type(v) is not int or v <= 0 for v in stamps) or not (
            self.credit_issued_ns <= self.request_start_ns <= self.request_end_ns
        ):
            raise ServiceEvidenceError("turn record stamps are not monotone")
        if self.first_token_ns is not None and not (
            type(self.first_token_ns) is int
            and self.request_start_ns <= self.first_token_ns <= self.request_end_ns
        ):
            raise ServiceEvidenceError("turn record first-token stamp is outside the request")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ServiceEvidenceError("turn record ordinal must be a non-negative int")
        if any(type(v) is not int or v < 0 for v in (self.prompt_tokens, self.output_tokens)):
            raise ServiceEvidenceError("turn record token counts must be non-negative ints")
        if self.status == "ok" and self.output_tokens == 0:
            raise ServiceEvidenceError("a completed turn delivered no tokens")

    @property
    def ttft_s(self) -> float:
        """Client-observed time to first token, queueing included."""
        end = self.first_token_ns if self.first_token_ns is not None else self.request_end_ns
        return (end - self.credit_issued_ns) / 1e9

    @property
    def decode_tps(self) -> float | None:
        """Per-user decode rate after the first token; undefined for one-token turns."""
        if self.output_tokens < 2 or self.first_token_ns is None:
            return None
        seconds = (self.request_end_ns - self.first_token_ns) / 1e9
        return (self.output_tokens - 1) / seconds if seconds > 0 else None

    def meets(self, contract: ServiceContract) -> bool:
        if self.status != "ok" or self.ttft_s > contract.ttft_bound_s:
            return False
        rate = self.decode_tps
        return rate is None or rate >= contract.decode_floor_tps


@dataclass(frozen=True)
class LoadRead:
    """Every attempted turn of one arm at one sealed load inside one window."""

    arm: str
    window: int
    lane: str
    load: int
    records: tuple[TurnRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or any(type(r) is not TurnRecord for r in records):
            raise ServiceEvidenceError("a load read needs typed turn records")
        if type(self.load) is not int or self.load <= 0 or self.arm not in ("incumbent", "candidate"):
            raise ServiceEvidenceError("load read identity is malformed")
        object.__setattr__(self, "records", records)


@dataclass(frozen=True)
class WorkRate:
    turns: int
    elapsed_s: float
    rate: float
    drain_fraction: float


def completed_work(read: LoadRead) -> dict[str, tuple[int, int]]:
    """Completed (main, inner) turn counts per root session."""
    counts: dict[str, list[int]] = {}
    for row in read.records:
        if row.status != "ok":
            continue
        pair = counts.setdefault(row.root_session_id, [0, 0])
        pair[0 if row.kind == "main" else 1] += 1
    return {root: (main, inner) for root, (main, inner) in counts.items()}


def fixed_work_rate(read: LoadRead, expected: dict[str, tuple[int, int]]) -> WorkRate:
    """Turns per elapsed second for the sealed work, refusing any other work.

    A read that completed a different set of turns (a wrapped root, a missing
    session, a lane that stopped early) is invalid evidence, never a slower or
    faster engine.
    """
    done = completed_work(read)
    if done != expected:
        missing = sorted(set(expected) - set(done))
        extra = sorted(set(done) - set(expected))
        differing = sorted(r for r in set(done) & set(expected) if done[r] != expected[r])
        raise ServiceEvidenceError(
            f"completed work differs from the sealed slice: missing={missing[:3]} "
            f"extra={extra[:3]} differing={differing[:3]}"
        )
    first = min(r.credit_issued_ns for r in read.records)
    last_start = max(r.request_start_ns for r in read.records)
    last = max(r.request_end_ns for r in read.records)
    if last <= first:
        raise ServiceEvidenceError("read elapsed time is not positive")
    turns = sum(main + inner for main, inner in expected.values())
    elapsed = (last - first) / 1e9
    return WorkRate(turns, elapsed, turns / elapsed, (last - last_start) / (last - first))


def attainment(read: LoadRead, contract: ServiceContract) -> float:
    """Fraction of attempted turns meeting the contract; errors and cancels are misses."""
    return sum(1 for r in read.records if r.meets(contract)) / len(read.records)


@dataclass(frozen=True)
class ServiceVerdict:
    decision: SpeedStageDecision
    ratio: float | None  # conservative: min(candidate rate) / max(incumbent rate)
    required: float
    detail: str


def grade(
    candidate: Sequence[LoadRead],
    incumbent: Sequence[LoadRead],
    contract: ServiceContract,
    expected: dict[str, tuple[int, int]],
    *,
    required: float,
    attainment_tolerance: float,
    attainment_margin: float,
) -> ServiceVerdict:
    """The paired fixed-load verdict: rate over identical work, attainment as a non-inferiority gate.

    Each window pairs one candidate read with one incumbent read at the same
    sealed load over the same fixed work, so the fixed-work turn rates compare
    like for like. The candidate must clear ``required`` against its least
    favorable pairing to PASS and lose against its most favorable to FAIL; a
    ratio that flips inside the observed spread is NO_DECISION, and the sealed
    repeat, never a rerun of one arm, is the only escalation.

    Attainment is graded on the paired difference A_candidate - A_incumbent.
    Its one-sided lower bound, the difference less ``attainment_margin`` (the
    calibrated allowance for the difference's own noise), must stay above
    -``attainment_tolerance`` (the fixed product tolerance) in every window;
    otherwise the candidate bought its rate by starving turns below the
    contract and FAILs before rate is considered. Neither threshold comes from
    the candidate, and more noise makes the gate harder, never easier. Work
    that differs from the sealed slice raises: it is invalid evidence, not a
    verdict.
    """
    if not _finite_positive(required) or required < 1.0:
        raise ServiceEvidenceError("required ratio must be at least 1.0")
    for name, value in (("attainment_tolerance", attainment_tolerance), ("attainment_margin", attainment_margin)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) < 1:
            raise ServiceEvidenceError(f"{name} must be in [0, 1)")
    if not candidate or len(candidate) != len(incumbent):
        raise ServiceEvidenceError("paired windows need one read per arm per window")
    rates_c: list[float] = []
    rates_i: list[float] = []
    windows: list[tuple[int, float, float]] = []
    for cand, inc in zip(candidate, incumbent, strict=True):
        if (cand.arm, inc.arm) != ("candidate", "incumbent") or (cand.window, cand.load) != (inc.window, inc.load):
            raise ServiceEvidenceError("a window pairs one candidate and one incumbent read at one load")
        rates_c.append(fixed_work_rate(cand, expected).rate)
        rates_i.append(fixed_work_rate(inc, expected).rate)
        windows.append((cand.window, attainment(cand, contract), attainment(inc, contract)))
    ratio = min(rates_c) / max(rates_i)
    for window, a_cand, a_inc in windows:
        if a_cand - a_inc - attainment_margin < -attainment_tolerance:
            return ServiceVerdict(
                SpeedStageDecision.FAIL, ratio, required,
                f"service_contract_not_met: window {window} attainment {a_cand:.4f} against the incumbent's "
                f"{a_inc:.4f} is below the non-inferiority bound (tolerance {attainment_tolerance:.4g}, "
                f"margin {attainment_margin:.4g})",
            )
    decision = invariant_decision(rates_i, rates_c, required)
    if decision is SpeedStageDecision.PASS:
        detail = "candidate clears the required fixed-work rate ratio in every window pairing"
    elif decision is SpeedStageDecision.FAIL:
        detail = "candidate does not clear the required fixed-work rate ratio in any window pairing"
    else:
        decision, detail = SpeedStageDecision.NO_DECISION, "window spread crosses the required ratio"
    return ServiceVerdict(decision, ratio, required, detail)


def load_reads_jsonl(path: Path) -> list[LoadRead]:
    """Group flat per-turn rows, one JSON object per line and gzip allowed, into load reads.

    This is the retained evidence shape: every row carries its read identity
    (``arm``, ``window``, ``lane``, ``load``) beside the turn fields, so a run
    regrades from the file alone.
    """
    record_fields = tuple(f.name for f in fields(TurnRecord))
    allowed = frozenset(record_fields) | frozenset(READ_KEYS)
    required = allowed - {"cached_tokens"}
    groups: dict[tuple[str, int, str, int], list[TurnRecord]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ServiceEvidenceError(f"row {number} is not JSON") from None
            if type(row) is not dict or not required <= set(row) <= allowed:
                raise ServiceEvidenceError(f"row {number} fields are not the retained shape")
            key = tuple(row[k] for k in READ_KEYS)
            record = TurnRecord(**{k: row[k] for k in record_fields if k in row})
            groups.setdefault(key, []).append(record)  # type: ignore[arg-type]
    if not groups:
        raise ServiceEvidenceError("no turn rows")
    return [LoadRead(*key, tuple(rows)) for key, rows in sorted(groups.items())]


def _finite_positive(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


__all__ = [
    "LoadRead",
    "ServiceContract",
    "ServiceEvidenceError",
    "ServiceVerdict",
    "TurnRecord",
    "WorkRate",
    "attainment",
    "completed_work",
    "fixed_work_rate",
    "grade",
    "load_reads_jsonl",
]

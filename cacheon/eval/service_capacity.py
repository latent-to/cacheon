"""Service capacity of one engine on a fixed agent slice, and the paired verdict.

The agent arena scores capacity at the service boundary: the fixed-work turn
rate an engine sustains at the load where its latency attainment crosses the
sealed contract. Two sealed loads bracket that boundary; the crossing is
interpolated log-linearly in load between them, which keeps the score
continuous in per-user speed instead of stepping between bracket points.
Every number here is recomputed from per-turn rows stamped by the trusted
controller, so a retained run regrades to the same verdict.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, fields
from enum import Enum
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


class CapacityStatus(str, Enum):
    INTERPOLATED = "interpolated"
    CLAMPED = "clamped"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class Capacity:
    """Fixed-work rate at the load where attainment crosses the contract."""

    value: float | None
    load: float | None
    status: CapacityStatus
    low: tuple[float, float]  # (attainment, rate) at the low bracket load
    high: tuple[float, float]  # (attainment, rate) at the high bracket load


def capacity(
    low: LoadRead,
    high: LoadRead,
    contract: ServiceContract,
    expected_by_load: dict[int, dict[str, tuple[int, int]]],
) -> Capacity:
    """Interpolate the service boundary between the two sealed bracket loads.

    Attainment falls and the fixed-work rate rises with load, so the crossing
    is unique when it lies inside the bracket. Above the bracket the score is
    clamped at the high load and the bracket needs rotating; below it the
    engine cannot serve the contract at the arena's floor load.
    """
    if not (low.arm == high.arm and low.window == high.window and low.load < high.load):
        raise ServiceEvidenceError("a bracket is one arm, one window, two increasing loads")
    a_lo = attainment(low, contract)
    a_hi = attainment(high, contract)
    g_lo = fixed_work_rate(low, expected_by_load[low.load]).rate
    g_hi = fixed_work_rate(high, expected_by_load[high.load]).rate
    ends = ((a_lo, g_lo), (a_hi, g_hi))
    if a_lo < contract.attainment:
        return Capacity(None, None, CapacityStatus.INFEASIBLE, *ends)
    if a_hi >= contract.attainment:
        return Capacity(g_hi, float(high.load), CapacityStatus.CLAMPED, *ends)
    fraction = (a_lo - contract.attainment) / (a_lo - a_hi)
    span = math.log(high.load) - math.log(low.load)
    load = math.exp(math.log(low.load) + fraction * span)
    return Capacity(g_lo + fraction * (g_hi - g_lo), load, CapacityStatus.INTERPOLATED, *ends)


@dataclass(frozen=True)
class ServiceVerdict:
    decision: SpeedStageDecision
    ratio: float | None  # conservative: min(candidate) / max(incumbent)
    required: float
    detail: str


def service_verdict(
    candidate: list[Capacity], incumbent: list[Capacity], required: float
) -> ServiceVerdict:
    """The verdict that survives the spread of the paired windows.

    Each window contributes one capacity per arm. The candidate must clear the
    required ratio against its least favorable pairing to PASS and lose against
    its most favorable to FAIL; anything in between is undetermined and the
    sealed repeat, not a rerun of a favorable arm, is the only escalation.
    """
    if not _finite_positive(required) or required < 1.0:
        raise ServiceEvidenceError("required ratio must be at least 1.0")
    if len(candidate) != len(incumbent) or not candidate:
        raise ServiceEvidenceError("paired windows need one capacity per arm per window")
    if any(c.status is CapacityStatus.INFEASIBLE for c in incumbent):
        return ServiceVerdict(
            SpeedStageDecision.NO_DECISION, None, required,
            "incumbent cannot serve the contract at the floor load; bracket miscalibrated",
        )
    if any(c.status is CapacityStatus.INFEASIBLE for c in candidate):
        return ServiceVerdict(
            SpeedStageDecision.FAIL, None, required, "service_infeasible",
        )
    cands = [c.value for c in candidate]
    bases = [b.value for b in incumbent]
    assert all(v is not None for v in (*cands, *bases))
    decision = invariant_decision(bases, cands, required)  # type: ignore[arg-type]
    ratio = min(cands) / max(bases)  # type: ignore[type-var]
    if decision is SpeedStageDecision.PASS:
        detail = "candidate clears the required capacity ratio in every window pairing"
    elif decision is SpeedStageDecision.FAIL:
        detail = "candidate does not clear the required capacity ratio in any window pairing"
    else:
        decision = SpeedStageDecision.NO_DECISION
        detail = "window spread crosses the capacity decision boundary"
    if any(c.status is CapacityStatus.CLAMPED for c in (*candidate, *incumbent)):
        detail += "; a capacity was clamped at the high bracket load: rotate the bracket"
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
    "Capacity",
    "CapacityStatus",
    "LoadRead",
    "ServiceContract",
    "ServiceEvidenceError",
    "ServiceVerdict",
    "TurnRecord",
    "WorkRate",
    "attainment",
    "capacity",
    "completed_work",
    "fixed_work_rate",
    "load_reads_jsonl",
    "service_verdict",
]

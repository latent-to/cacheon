"""Service rate of one engine on a fixed agent slice, and its verdict against a retained stock reference.

Candidate and stock replay identical work at each sealed load. Controller
timestamps supply elapsed serving cost and a relative service-attainment gate;
that gate does not certify an absolute SLA. The 2026-09-26 interpolated-capacity
design (two bracket loads, log-linear crossing) was withdrawn on 2026-09-27: its
8×12 pairings carried a 9-15% same-engine spread.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from collections.abc import Sequence
from dataclasses import dataclass

from cacheon.eval.speed_verdict import SpeedStageDecision

TURN_KINDS = frozenset({"main", "inner"})
TURN_STATUSES = frozenset({"ok", "error", "cancelled"})


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
    """Warm work and the serving cost selected by the sealed measurement policy."""

    turns: int
    elapsed_s: float
    rate: float


def completed_work(read: LoadRead) -> dict[str, tuple[int, int]]:
    """Completed (main, inner) turn counts per root session."""
    counts: dict[str, list[int]] = {}
    for row in read.records:
        if row.status != "ok":
            continue
        pair = counts.setdefault(row.root_session_id, [0, 0])
        pair[0 if row.kind == "main" else 1] += 1
    return {root: (main, inner) for root, (main, inner) in counts.items()}


def fixed_work_rate(read: LoadRead, expected: dict[str, tuple[int, int]], *, wall_time=False) -> WorkRate:
    """Rate over exact warm work; new policies charge the complete elapsed serving span.

    Cold openings populate the declared warm-cache operating point before timing.
    Wall time includes tails and client handoffs between warm turns. With fixed
    work value and GPU allocation its inverse is monotone in profit. Historical
    policies retain summed request latency only for reopening their old scores;
    inverse mean latency under round barriers is not throughput.
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
    cold = min(record.credit_issued_ns for record in read.records)
    warm = [record for record in read.records if record.credit_issued_ns != cold]
    if not warm:
        raise ServiceEvidenceError("read has no timed warm turns")
    latency = ((max(r.request_end_ns for r in warm) - min(r.credit_issued_ns for r in warm)) / 1e9
               if wall_time else sum((r.request_end_ns-r.credit_issued_ns)/1e9 for r in warm))
    if latency <= 0:
        raise ServiceEvidenceError("read has no timed warm turns")
    return WorkRate(len(warm), latency, len(warm) / latency)


def steady_span_cost(read: LoadRead, expected: dict[str, tuple[int, int]], share: float) -> float:
    """Seconds per warm turn until active sessions fall below ``share`` of the load after the last session starts.

    The fixed work must still complete. A closed loop holds the load until the slice runs out of sessions; the drain
    after it runs on the longest few sessions, so their tails set the whole-span cost (four GLM stock lifetimes,
    2026-11-12: per-lifetime geometric-mean spread 1.45% whole span, 0.58% at half the load)."""
    fixed_work_rate(read, expected, wall_time=True)
    cold = min(record.credit_issued_ns for record in read.records)
    warm = [record for record in read.records if record.credit_issued_ns != cold]
    sessions: dict[str, list[int]] = {}
    for record in read.records:
        span = sessions.setdefault(record.root_session_id, [record.credit_issued_ns, record.request_end_ns])
        span[0], span[1] = min(span[0], record.credit_issued_ns), max(span[1], record.request_end_ns)
    last_start = max(start for start, _ in sessions.values())
    end, active = max(record.request_end_ns for record in warm), 0
    for time_ns, step in sorted([(start, 1) for start, _ in sessions.values()]
                                + [(stop, -1) for _, stop in sessions.values()]):
        active += step
        if step < 0 and time_ns > last_start and active < share * read.load:
            end = time_ns
            break
    start = min(record.credit_issued_ns for record in warm)
    turns = sum(record.request_end_ns <= end for record in warm)
    if end <= start or not turns:
        raise ServiceEvidenceError("read has no steady warm turns")
    return (end - start) / 1e9 / turns


def attainment(read: LoadRead, contract: ServiceContract) -> float:
    """Fraction of attempted turns meeting the contract; errors and cancels are misses."""
    return sum(1 for r in read.records if r.meets(contract)) / len(read.records)


def decode_p50(reads: Sequence[LoadRead]) -> float:
    """Median per-user decode rate over every turn of ``reads`` that has one (version 20's service floor)."""
    rates = sorted(rate for read in reads for r in read.records if (rate := r.decode_tps) is not None)
    if not rates:
        raise ServiceEvidenceError("reads hold no turn with a per-user decode rate")
    middle = len(rates) // 2
    return rates[middle] if len(rates) % 2 else (rates[middle - 1] + rates[middle]) / 2


def _work_shapes(read: LoadRead):
    cold = min(r.credit_issued_ns for r in read.records)
    shapes = {(r.root_session_id, r.kind, r.ordinal):
              (r.prompt_tokens, r.output_tokens, r.status) for r in read.records}
    if len(shapes) != len(read.records):
        raise ServiceEvidenceError("replay contains duplicate turn identities")
    warm = {(r.root_session_id, r.kind, r.ordinal): shapes[r.root_session_id, r.kind, r.ordinal]
            for r in read.records if r.credit_issued_ns != cold}
    return shapes, warm


@dataclass(frozen=True)
class ServiceVerdict:
    decision: SpeedStageDecision
    ratio: float
    required: float
    detail: str
    standard_error: float = 0.0
    lower_ratio: float = 0.0


def _boot_costs(reads: Sequence[LoadRead], arm: str, expected: dict[int, dict[str, tuple[int, int]]],
                steady_span: float = 0.0) -> list[dict[int, list[float]]]:
    """Split one arm's reads into engine lifetimes; a lifetime replays each load once, numbering its windows from 1."""
    boots: list[dict[int, list[float]]] = []
    for read in reads:
        lifetime = boots[-1] if boots else {}
        if read.arm != arm or (read.window != 1 and (
                not lifetime or read.load != next(reversed(lifetime)) or read.window != len(lifetime[read.load]) + 1)):
            raise ServiceEvidenceError("reference reads do not form whole engine lifetimes")
        if read.window == 1:
            if not boots or read.load in lifetime:
                boots.append({})
            boots[-1][read.load] = []
        boots[-1][read.load].append(steady_span_cost(read, expected[read.load], steady_span) if steady_span
                                    else fixed_work_rate(read, expected[read.load], wall_time=True).elapsed_s)
    return boots


def _boot_term(lifetime: dict[int, list[float]], window_noise: float, boot_noise: float) -> tuple[float, float]:
    """Mean over the lifetime's loads of each load's log mean cost, and its variance.

    A stock/candidate difference of these terms is the log of the geometric mean of the per-load throughput ratios
    (one load: that load's ratio). Sealed noise bounds the log ratio; each arm carries half: a whole bound per arm
    put the sealed bar above v18's (simulated 2026-10-25). The boot offset is shared by every load of a lifetime,
    so its share is never divided by the load count."""
    terms = []
    for costs in lifetime.values():
        n, total = len(costs), math.fsum(costs)
        variance = window_noise ** 2 / (2 * n)
        if n > 1:
            deleted = [math.log((total - cost) / (n - 1)) for cost in costs]
            center = math.fsum(deleted) / n
            variance = max(variance, (n - 1) / n * math.fsum((v - center) ** 2 for v in deleted))
        terms.append((math.log(total / n), variance))
    count = len(terms)
    return (math.fsum(row[0] for row in terms) / count,
            math.fsum(row[1] for row in terms) / count ** 2 + boot_noise ** 2 / 2)


def _contrast(stock, terms) -> tuple[float, float]:
    """Stock's mean log cost minus the candidate's (the log throughput ratio) and its standard error."""
    return (math.fsum(row[0] for row in stock) / len(stock) - math.fsum(row[0] for row in terms) / len(terms),
            math.sqrt(math.fsum(row[1] for row in stock) / len(stock) ** 2
                      + math.fsum(row[1] for row in terms) / len(terms) ** 2))


def reference_grade(
    candidate: Sequence[LoadRead], reference: Sequence[LoadRead],
    contract: ServiceContract, expected: dict[int, dict[str, tuple[int, int]]], *,
    window_noise: float, boot_noise: float, error_rate: float, boots: int,
    attainment_tolerance: float, attainment_margin: float, futility_margin: float = 0.0,
    sentinel: Sequence[LoadRead] = (), decode_floors: dict[int, float] | None = None, reference_lane: str = "",
    load_margin: float = 0.0, steady_span: float = 0.0,
) -> ServiceVerdict:
    """Grade candidate engine lifetimes against a retained stock reference campaign on the same lane.

    One engine owns the node, so stock is measured once per reference and every candidate is graded
    against those reads; window correlation between concurrent lanes cancelled only 4-8% of window
    variance (149 retained evaluations, 2026-10-24). The reference variance is shared and never
    divides by the candidate's boot count. Looks fall at candidate boot boundaries under v17's
    alpha-spending table: the first boot can only end the stage on the sealed futility margin. A lifetime
    that replays several sealed loads scores the geometric mean of their throughput ratios (``_boot_term``),
    so no single batch size carries the verdict.

    PASS evidence retained before 2026-11-05 carries one fresh stock lifetime read after it. Stock reading
    faster than its reference beyond one-sided 1% made that PASS an infrastructure NO_DECISION, never a
    candidate FAIL; new evidence carries no sentinel.

    ``decode_floors`` (version 20 on) also holds each load's pooled per-user decode p50 to its sealed floor at
    every look: a candidate below it fails, since served tokens bought by slowing every user are not
    revenue at the sealed price. A stock reference below it is a mis-sealed load that ``open_reference``
    refuses before any candidate boot, never a candidate FAIL.
    """
    if (not candidate or len(reference) < 2 or not 0 < error_rate < 0.5 or type(boots) is not int
        or not 2 <= boots <= 4 or any(not math.isfinite(v) or v < 0 for v in (window_noise, boot_noise))
        or window_noise == boot_noise == 0):
        raise ServiceEvidenceError("reference comparison has invalid reads or uncertainty")
    every = (*reference, *sentinel, *candidate)
    # Each load replays its own sealed prefix of the sessions; its cold round, and so its warm membership, is its own.
    shapes: dict[int, object] = {}
    starts = [min(row.request_start_ns for row in read.records) for read in every]
    if (len({read.lane for read in (*sentinel, *candidate)}) != 1
            or any(read.lane != (reference_lane or candidate[0].lane) for read in reference)
            or len(set(starts)) != len(starts)
            or any(shapes.setdefault(read.load, shape) != shape for read in every for shape in (_work_shapes(read),))):
        raise ServiceEvidenceError("reference and candidate reads differ in lane, work or identity")
    lifetimes = [_boot_costs(arm, role, expected, steady_span)
                 for arm, role in ((reference, "incumbent"), (candidate, "candidate"), (sentinel, "incumbent"))]
    if any(list(lifetime) != list(lifetimes[0][0]) for rows in lifetimes for lifetime in rows):
        raise ServiceEvidenceError("reference and candidate lifetimes replay different loads")
    stock, terms, bracket = ([_boot_term(costs, window_noise, boot_noise) for costs in rows] for rows in lifetimes)
    look = len(terms)
    if len(stock) < 2 or look > boots or len(bracket) > 1:
        raise ServiceEvidenceError("reference comparison exceeds its sealed boots")
    reference_mean = math.fsum(row[0] for row in stock) / len(stock)
    reference_variance = math.fsum(row[1] for row in stock) / len(stock) ** 2
    estimate, se = _contrast(stock, terms)
    share = 0.8 if look == boots else 0.1 if look == boots - 1 else 0.05
    z = NormalDist().inv_cdf(1 - error_rate * share)
    ratio, required, lower = math.exp(estimate), math.exp(z * se), math.exp(estimate - z * se)
    # One-sided per-load guard (owner, 2026-11-11): a load fails only when its ratio is confidently more than the
    # sealed margin below stock, from the same reads, noise and z; a flat or noisy load and an honest tradeoff pass.
    regressed = []
    for load in lifetimes[0][0] if load_margin and look >= 2 else ():
        e, s = _contrast(*([_boot_term({load: costs[load]}, window_noise, boot_noise) for costs in rows]
                           for rows in lifetimes[:2]))
        if (upper := math.exp(e + z * s)) < 1 - load_margin:
            regressed.append((load, upper))
    a_cand, a_ref = (sum(r.meets(contract) for read in arm for r in read.records)
                     / sum(len(read.records) for read in arm) for arm in (candidate, reference))
    floors = decode_floors or {}
    below = [(load, rate, floor) for load, floor in sorted(floors.items())
             if (rate := decode_p50([read for read in candidate if read.load == load])) < floor]
    if below:
        load, rate, floor = below[0]
        # One load keeps the detail its historical verdicts were sealed with.
        decision, detail = SpeedStageDecision.FAIL, (
            f"service_floor_not_met: candidate decode p50 {rate:.1f} tok/s"
            f"{f' at {load} sessions' if len(floors) > 1 else ''} is below the sealed floor {floor:.4g}")
    elif look >= 2 and a_cand - a_ref - attainment_margin < -attainment_tolerance:
        decision, detail = SpeedStageDecision.FAIL, (
            f"service_contract_not_met: pooled attainment {a_cand:.4f} against the reference's {a_ref:.4f} "
            f"is below the non-inferiority bound (tolerance {attainment_tolerance:.4g}, margin {attainment_margin:.4g})")
    elif regressed:
        load, upper = regressed[0]
        decision, detail = SpeedStageDecision.FAIL, (
            f"load_regression: load {load} ratio upper bound {upper:.3f} < {1 - load_margin:.4g}")
    elif look < 2 and futility_margin and estimate < math.log1p(-futility_margin):
        decision, detail = SpeedStageDecision.FAIL, "first candidate boot is slower than the sealed futility margin"
    elif look < 2:
        decision, detail = SpeedStageDecision.NO_DECISION, "two candidate boots are required"
    elif lower > 1:
        decision, detail = SpeedStageDecision.PASS, "gain over the stock reference clears the sealed statistical boundary"
    elif look == boots or estimate + z * se < 0:
        decision, detail = SpeedStageDecision.FAIL, "gain over the stock reference is not established within the sealed boots"
    else:
        decision, detail = SpeedStageDecision.NO_DECISION, "reference comparison needs its remaining sealed boots"
    if bracket and decision is not SpeedStageDecision.PASS:
        raise ServiceEvidenceError("a stock sentinel follows only a passing reference comparison")
    if bracket and reference_mean - bracket[0][0] > NormalDist().inv_cdf(0.99) * math.sqrt(
            bracket[0][1] + reference_variance):
        decision, detail = SpeedStageDecision.NO_DECISION, "stock sentinel reads faster than its reference"
    return ServiceVerdict(decision, ratio, required, detail, se, lower)


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
    "reference_grade",
]

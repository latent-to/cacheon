"""Service rate of one engine on a fixed agent slice, and the paired verdict.

Candidate and incumbent replay identical work at a sealed load. Controller
timestamps supply elapsed serving cost and a relative service-attainment gate;
that gate does not certify an absolute SLA. Historical latency scores retain
their original meaning. The 2026-09-26 interpolated-capacity design (two bracket
loads, log-linear crossing) was withdrawn on 2026-09-27: its 8×12 pairings
carried a 9-15% same-engine spread.
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


STOP_MIN_WINDOWS = 2
STOP_Z = 2.0


def continue_windows(ratios: Sequence[float], *, required: float, null_noise: float, max_windows: int) -> bool:
    """Whether the sealed budget and the running mean still call for another paired window.

    A sequential test on the per-window ratios: from the second window on, the
    read stops once the mean sits ``STOP_Z`` sigma clear of ``required`` on
    either side, with sigma the sealed per-window null noise over the square
    root of the windows read. A clear win or a plain copy settles in two
    windows; a marginal candidate runs to the sealed maximum, where the plain
    mean decides. Both lanes call this on the same published rates, and the
    regrade replays it from the retained reads, so the stopping point is part
    of the evidence.
    """
    count = len(ratios)
    if count >= max_windows:
        return False
    if count < STOP_MIN_WINDOWS:
        return True
    mean = sum(ratios) / count
    band = STOP_Z * null_noise / math.sqrt(count)
    return mean - band < required <= mean + band


def attainment(read: LoadRead, contract: ServiceContract) -> float:
    """Fraction of attempted turns meeting the contract; errors and cancels are misses."""
    return sum(1 for r in read.records if r.meets(contract)) / len(read.records)


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


def grade(
    candidate: Sequence[LoadRead],
    incumbent: Sequence[LoadRead],
    contract: ServiceContract,
    expected: dict[str, tuple[int, int]],
    *,
    required: float,
    attainment_tolerance: float,
    attainment_margin: float,
    elapsed_time: bool = False,
) -> ServiceVerdict:
    """Grade paired fixed work with the sealed service-attainment requirement.

    Elapsed-time scoring pools all warm work and its elapsed occupancy across
    all scheduled passes. Both lanes must complete identical token work, including
    identical cold/warm membership. Their allocated device counts are fixed by
    the commissioned lanes, so the throughput ratio is also the ratio of revenue
    capacity for any fixed positive price of that basket.

    V16's fastest summed-latency pass survives only for historical regrading.
    Incident history: on 2026-09-27 identical prompts changed MTP step counts
    in 233/237 turns; latency-ratio scatter was 0.56% across four windows.
    First windows were 3-20% slower across six runs. On 2026-09-28 allocator
    retries (38 incumbent, 3 candidate) inflated the mean-ratio verdict to
    1.0189; fastest-pass latency instead read 0.9853. Neither statistic is
    elapsed-work throughput, and neither calibrates its uncertainty.

    Attainment difference less the sealed noise margin must be at least
    minus the fixed product tolerance in every window. Different work is
    invalid evidence, never a candidate verdict.
    """
    # V16 is retained for historical evidence. Elapsed accounting charges every
    # pass's occupancy: overlapping request durations are not GPU time
    # (2026-09-29: c6790f93 scored +4.80% latency but only +0.63% elapsed work).
    if type(elapsed_time) is not bool:
        raise ServiceEvidenceError("elapsed-time scoring selection is not boolean")
    if not _finite_positive(required) or required < 1.0:
        raise ServiceEvidenceError("required ratio must be at least 1.0")
    for name, value in (("attainment_tolerance", attainment_tolerance), ("attainment_margin", attainment_margin)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) < 1:
            raise ServiceEvidenceError(f"{name} must be in [0, 1)")
    if not candidate or len(candidate) != len(incumbent):
        raise ServiceEvidenceError("paired windows need one read per arm per window")
    work_c: list[WorkRate] = []
    work_i: list[WorkRate] = []
    windows: list[tuple[int, float, float]] = []
    seen = set()
    for cand, inc in zip(candidate, incumbent, strict=True):
        if (cand.arm, inc.arm) != ("candidate", "incumbent") or (cand.window, cand.load) != (inc.window, inc.load):
            raise ServiceEvidenceError("a window pairs one candidate and one incumbent read at one load")
        if elapsed_time:
            identity = cand.lane, inc.lane, cand.window
            if identity in seen:
                raise ServiceEvidenceError("elapsed-work windows contain a duplicate paired read")
            seen.add(identity)
            if _work_shapes(cand) != _work_shapes(inc):
                raise ServiceEvidenceError("paired replay token work or warm membership differs")
        work_c.append(fixed_work_rate(cand, expected, wall_time=elapsed_time))
        work_i.append(fixed_work_rate(inc, expected, wall_time=elapsed_time))
        if elapsed_time and work_c[-1].turns != work_i[-1].turns:
            raise ServiceEvidenceError("paired replay warm work differs")
        windows.append((cand.window, attainment(cand, contract), attainment(inc, contract)))
    if elapsed_time:
        candidate_rate = sum(w.turns for w in work_c) / sum(w.elapsed_s for w in work_c)
        incumbent_rate = sum(w.turns for w in work_i) / sum(w.elapsed_s for w in work_i)
        ratio = candidate_rate / incumbent_rate
    else:
        ratio = max(w.rate for w in work_c) / max(w.rate for w in work_i)
    for window, a_cand, a_inc in windows:
        if a_cand - a_inc - attainment_margin < -attainment_tolerance:
            return ServiceVerdict(
                SpeedStageDecision.FAIL, ratio, required,
                f"service_contract_not_met: window {window} attainment {a_cand:.4f} against the incumbent's "
                f"{a_inc:.4f} is below the non-inferiority bound (tolerance {attainment_tolerance:.4g}, "
                f"margin {attainment_margin:.4g})",
            )
    basis = "pooled elapsed fixed-work" if elapsed_time else "fastest-pass fixed-work"
    if ratio >= required:
        return ServiceVerdict(SpeedStageDecision.PASS, ratio, required,
                              f"{basis} rate ratio over {len(work_c)} paired window(s) clears the required ratio")
    return ServiceVerdict(SpeedStageDecision.FAIL, ratio, required,
                          f"{basis} rate ratio over {len(work_c)} paired window(s) is below the required ratio")


def statistical_grade(
    candidate: Sequence[LoadRead], incumbent: Sequence[LoadRead],
    contract: ServiceContract, expected: dict[str, tuple[int, int]], *,
    window_noise: float, boot_noise: float, error_rate: float, max_windows: int,
    attainment_tolerance: float, attainment_margin: float,
) -> ServiceVerdict:
    """Pool fixed-work costs within each orientation and spend one sealed error budget.

    Noise parameters bound log-ratio variation: window noise averages within a
    boot; paired boot noise does not. Both physical orientations are necessary
    for eligibility. Unequal numbers of windows retain equal orientation weight.
    """
    if (not candidate or len(candidate) != len(incumbent)
        or not 0 < error_rate < 0.5 or type(max_windows) is not int
        or not 2 <= max_windows <= 5 or len(candidate) > max_windows
        or any(not math.isfinite(v) or v < 0 for v in (window_noise, boot_noise))
        or window_noise == boot_noise == 0):
        raise ServiceEvidenceError("statistical comparison has invalid reads or uncertainty")
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for cand, inc in zip(candidate, incumbent, strict=True):
        if ((cand.arm, inc.arm) != ("candidate", "incumbent")
            or (cand.window, cand.load) != (inc.window, inc.load) or cand.lane == inc.lane):
            raise ServiceEvidenceError("statistical reads are not paired on disjoint lanes")
        c = fixed_work_rate(cand, expected, wall_time=True)
        b = fixed_work_rate(inc, expected, wall_time=True)
        if c.turns != b.turns:
            raise ServiceEvidenceError("paired windows completed different timed work")
        groups.setdefault((inc.lane, cand.lane), []).append((b.elapsed_s, c.elapsed_s))
    orientations = tuple(groups)
    if len(groups) > 2 or (len(groups) == 2 and orientations[0] != orientations[1][::-1]):
        raise ServiceEvidenceError("statistical comparison did not swap the same physical lanes")
    estimates, variances = [], []
    for pairs in groups.values():
        b_total, c_total = (math.fsum(row[i] for row in pairs) for i in (0, 1))
        estimates.append(math.log(b_total / c_total))
        n = len(pairs)
        # A whole window is one observation. Requests within a window are not
        # independent replications, and a boot offset never divides by n.
        variance = window_noise ** 2 / n
        if n > 1:
            deleted = [math.log((b_total-b)/(c_total-c)) for b, c in pairs]
            center = math.fsum(deleted) / n
            variance = max(variance, (n-1)/n * math.fsum((v-center)**2 for v in deleted))
        variances.append(variance + boot_noise ** 2)
    count = len(estimates)
    estimate = math.fsum(estimates) / count
    se = math.sqrt(math.fsum(variances)) / count
    # At most four looks (2,3,4,5). Unused early allocations remain unspent.
    look = len(candidate)
    share = 0.8 if look == max_windows else 0.1 if look == max_windows-1 else 0.05
    z = NormalDist().inv_cdf(1-error_rate*share)
    ratio, required, lower = math.exp(estimate), math.exp(z*se), math.exp(estimate-z*se)
    attainment_check = grade(candidate, incumbent, contract, expected, required=1.0,
                            attainment_tolerance=attainment_tolerance,
                            attainment_margin=attainment_margin, elapsed_time=True)
    if count == 2 and attainment_check.detail.startswith("service_contract_not_met"):
        return ServiceVerdict(SpeedStageDecision.FAIL, ratio, required,
                              attainment_check.detail, se, lower)
    if count < 2:
        decision, detail = SpeedStageDecision.NO_DECISION, "both lane orientations are required"
    elif lower > 1:
        decision, detail = SpeedStageDecision.PASS, "positive paired gain clears the sealed statistical boundary"
    elif look == max_windows or estimate+z*se < 0:
        decision, detail = SpeedStageDecision.FAIL, "positive paired gain is not established within the sealed budget"
    else:
        decision, detail = SpeedStageDecision.NO_DECISION, "paired comparison needs its remaining sealed windows"
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
    "grade",
    "statistical_grade",
]

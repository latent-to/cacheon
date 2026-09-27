"""Service capacity scoring (cacheon/eval/service_capacity.py).

The real-record tests replay the retained 2026-09-26 final budget-4 pairs (AIPerf profile exports converted to
turn rows, profiling phase only) and pin the numbers the arena design was decided on: the step rule collapses the
winner's edge to throughput at 24 sessions, interpolation at the service boundary keeps it. The synthetic tests pin
the invariants: fixed work is checked per root, misses stay in the denominator, and an incumbent that cannot serve
the contract is a miscalibrated bracket rather than a zero denominator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.eval.service_capacity import (
    Capacity,
    CapacityStatus,
    LoadRead,
    ServiceContract,
    ServiceEvidenceError,
    TurnRecord,
    attainment,
    capacity,
    completed_work,
    fixed_work_rate,
    load_reads_jsonl,
    service_verdict,
)
from cacheon.eval.speed_verdict import SpeedStageDecision

FIXTURE = Path(__file__).parent / "fixtures" / "service_capacity_final_pairs.jsonl.gz"
CONTRACT = ServiceContract(decode_floor_tps=60.0, ttft_bound_s=5.0, attainment=0.8)


@pytest.fixture(scope="module")
def final_pairs() -> dict[tuple[str, int], LoadRead]:
    return {(read.arm, read.load): read for read in load_reads_jsonl(FIXTURE)}


def _turn(root: str, kind: str, ordinal: int, *, start_s: float, ttft_s: float, out: int,
          decode_tps: float = 100.0, status: str = "ok", queue_s: float = 0.0) -> TurnRecord:
    """One synthetic turn with the given decode rate; times in seconds from an arbitrary origin."""
    credit = int((1e3 + start_s) * 1e9)
    start = credit + int(queue_s * 1e9)
    first = start + int(ttft_s * 1e9)
    end = first + (int((out - 1) / decode_tps * 1e9) if out > 1 else 0)
    return TurnRecord(root, kind, ordinal, credit, start, first if out > 0 else None, end,
                      prompt_tokens=1000, output_tokens=out, status=status)


# ----------------------------------------------------------------- real records

def test_fixed_work_rates_match_the_study(final_pairs):
    expected = {
        ("candidate", 24): (694, 0.7572), ("candidate", 48): (911, 0.9852),
        ("incumbent", 24): (672, 0.7368), ("incumbent", 48): (873, 0.9496),
    }
    for key, (turns, rate) in expected.items():
        read = final_pairs[key]
        work = fixed_work_rate(read, completed_work(read))
        assert work.turns == turns
        assert work.rate == pytest.approx(rate, abs=5e-4)
        assert work.drain_fraction < 0.03


def test_attainment_under_the_locked_contract(final_pairs):
    assert attainment(final_pairs[("candidate", 24)], CONTRACT) == pytest.approx(0.8862, abs=5e-4)
    assert attainment(final_pairs[("candidate", 48)], CONTRACT) == pytest.approx(0.6323, abs=5e-4)
    assert attainment(final_pairs[("incumbent", 24)], CONTRACT) == pytest.approx(0.8170, abs=5e-4)
    assert attainment(final_pairs[("incumbent", 48)], CONTRACT) == pytest.approx(0.4937, abs=5e-4)


def test_interpolated_capacity_keeps_the_interactivity_edge(final_pairs):
    def cap(arm: str) -> Capacity:
        low, high = final_pairs[(arm, 24)], final_pairs[(arm, 48)]
        expected = {24: completed_work(low), 48: completed_work(high)}
        return capacity(low, high, CONTRACT, expected)

    cand, inc = cap("candidate"), cap("incumbent")
    assert cand.status is CapacityStatus.INTERPOLATED and inc.status is CapacityStatus.INTERPOLATED
    assert cand.load == pytest.approx(30.37, abs=0.01)
    assert inc.load == pytest.approx(24.89, abs=0.01)
    assert cand.value == pytest.approx(0.8346, abs=5e-4)
    assert inc.value == pytest.approx(0.7480, abs=5e-4)
    # The step rule would score both arms at their 24-session rate: +2.8%.
    step = final_pairs[("candidate", 24)].records and 0.7572 / 0.7368
    assert step == pytest.approx(1.028, abs=2e-3)
    verdict = service_verdict([cand], [inc], required=1.05)
    assert verdict.decision is SpeedStageDecision.PASS
    assert verdict.ratio == pytest.approx(1.1157, abs=5e-4)


def test_tighter_contract_makes_the_incumbent_infeasible_not_zero(final_pairs):
    tight = ServiceContract(decode_floor_tps=100.0, ttft_bound_s=5.0, attainment=0.5)
    low, high = final_pairs[("incumbent", 24)], final_pairs[("incumbent", 48)]
    inc = capacity(low, high, tight, {24: completed_work(low), 48: completed_work(high)})
    assert inc.status is CapacityStatus.INFEASIBLE and inc.value is None
    cand_low, cand_high = final_pairs[("candidate", 24)], final_pairs[("candidate", 48)]
    cand = capacity(cand_low, cand_high, tight, {24: completed_work(cand_low), 48: completed_work(cand_high)})
    assert cand.status is CapacityStatus.INTERPOLATED
    verdict = service_verdict([cand], [inc], required=1.05)
    assert verdict.decision is SpeedStageDecision.NO_DECISION
    assert "miscalibrated" in verdict.detail


# ------------------------------------------------------------------- synthetic

def test_fixed_work_refuses_wrapped_or_missing_work():
    rows = [_turn("a", "main", i, start_s=i, ttft_s=0.5, out=50) for i in range(3)]
    read = LoadRead("candidate", 1, "lane-1", 1, tuple(rows))
    assert completed_work(read) == {"a": (3, 0)}
    with pytest.raises(ServiceEvidenceError, match="missing"):
        fixed_work_rate(read, {"a": (3, 0), "b": (1, 0)})
    with pytest.raises(ServiceEvidenceError, match="differing"):
        fixed_work_rate(read, {"a": (2, 0)})
    work = fixed_work_rate(read, {"a": (3, 0)})
    assert work.turns == 3 and work.rate == pytest.approx(3 / work.elapsed_s)


def test_misses_and_one_token_turns():
    ok = _turn("a", "main", 0, start_s=0, ttft_s=0.5, out=50, decode_tps=120.0)
    slow = _turn("a", "main", 1, start_s=1, ttft_s=0.5, out=50, decode_tps=30.0)
    queued = _turn("a", "main", 2, start_s=2, ttft_s=0.5, out=50, queue_s=6.0)
    single = _turn("a", "main", 3, start_s=10, ttft_s=0.2, out=1)
    failed = _turn("a", "main", 4, start_s=11, ttft_s=0.2, out=0, status="error")
    read = LoadRead("candidate", 1, "lane-1", 1, (ok, slow, queued, single, failed))
    assert single.decode_tps is None and single.meets(CONTRACT)
    assert queued.ttft_s == pytest.approx(6.5) and not queued.meets(CONTRACT)
    assert attainment(read, CONTRACT) == pytest.approx(2 / 5)
    assert completed_work(read) == {"a": (4, 0)}


def test_inner_requests_count_per_root():
    rows = (
        _turn("root", "main", 0, start_s=0, ttft_s=0.5, out=20),
        _turn("root", "inner", 0, start_s=1, ttft_s=0.3, out=20),
        _turn("root", "inner", 1, start_s=2, ttft_s=0.3, out=20),
    )
    read = LoadRead("incumbent", 2, "lane-2", 1, rows)
    assert completed_work(read) == {"root": (1, 2)}
    assert fixed_work_rate(read, {"root": (1, 2)}).turns == 3


def _bracket(arm: str, low_attain: float, high_attain: float, low_rate: float, high_rate: float):
    """Two synthetic reads whose attainment and rate are set by construction."""
    def read(load: int, attain: float, rate: float) -> LoadRead:
        turns = 20
        met = round(attain * turns)
        rows = [
            _turn(f"s{i}", "main", 0, start_s=i / rate, ttft_s=0.1, out=10, decode_tps=200.0 if i < met else 10.0)
            for i in range(turns)
        ]
        # Stretch the last turn so the elapsed span yields exactly ``rate`` turns per second.
        last = rows[-1]
        end = rows[0].credit_issued_ns + int(turns / rate * 1e9)
        rows[-1] = TurnRecord(last.root_session_id, last.kind, last.ordinal, last.credit_issued_ns,
                              last.request_start_ns, min(last.first_token_ns, end), end,
                              last.prompt_tokens, last.output_tokens, last.status)
        return LoadRead(arm, 1, "lane-1", load, tuple(rows))
    return read(24, low_attain, low_rate), read(48, high_attain, high_rate)


def test_capacity_is_continuous_between_bracket_points():
    low, high = _bracket("candidate", 0.9, 0.5, 0.75, 1.0)
    expected = {24: completed_work(low), 48: completed_work(high)}
    cap = capacity(low, high, CONTRACT, expected)
    assert cap.status is CapacityStatus.INTERPOLATED
    # a = 0.8 sits a quarter of the way from 0.9 to 0.5: log-linear load and linear rate.
    assert cap.load == pytest.approx(24 * (48 / 24) ** 0.25, rel=1e-3)
    assert cap.value == pytest.approx(0.75 + 0.25 * 0.25, rel=1e-2)
    # Higher attainment at the high load moves the crossing right and the score up, smoothly.
    nudged = capacity(*_bracket("candidate", 0.9, 0.6, 0.75, 1.0), CONTRACT, expected)
    assert 0 < nudged.value - cap.value < 0.05


def test_capacity_clamps_above_the_bracket_and_flags_rotation():
    low, high = _bracket("candidate", 0.95, 0.85, 0.75, 1.0)
    expected = {24: completed_work(low), 48: completed_work(high)}
    cap = capacity(low, high, CONTRACT, expected)
    assert cap.status is CapacityStatus.CLAMPED and cap.load == 48 and cap.value == pytest.approx(1.0, rel=1e-2)
    inc_low, inc_high = _bracket("incumbent", 0.9, 0.5, 0.7, 0.9)
    inc = capacity(inc_low, inc_high, CONTRACT, {24: completed_work(inc_low), 48: completed_work(inc_high)})
    verdict = service_verdict([cap], [inc], required=1.05)
    assert verdict.decision is SpeedStageDecision.PASS
    assert "rotate the bracket" in verdict.detail


def test_verdict_spread_rules():
    def cap(value: float) -> Capacity:
        return Capacity(value, 30.0, CapacityStatus.INTERPOLATED, (0.9, value), (0.5, value))

    assert service_verdict([cap(1.10), cap(1.12)], [cap(1.0), cap(1.02)], 1.05).decision is SpeedStageDecision.PASS
    assert service_verdict([cap(1.00), cap(1.03)], [cap(1.0), cap(1.02)], 1.05).decision is SpeedStageDecision.FAIL
    undetermined = service_verdict([cap(1.04), cap(1.09)], [cap(1.0), cap(1.02)], 1.05)
    assert undetermined.decision is SpeedStageDecision.NO_DECISION
    infeasible = Capacity(None, None, CapacityStatus.INFEASIBLE, (0.7, 0.7), (0.4, 0.9))
    assert service_verdict([infeasible], [cap(1.0)], 1.05).detail == "service_infeasible"
    with pytest.raises(ServiceEvidenceError):
        service_verdict([cap(1.1)], [cap(1.0), cap(1.0)], 1.05)
    with pytest.raises(ServiceEvidenceError):
        service_verdict([cap(1.1)], [cap(1.0)], 0.99)


def test_record_and_contract_validation():
    with pytest.raises(ServiceEvidenceError):
        ServiceContract(60.0, 5.0, 1.5)
    with pytest.raises(ServiceEvidenceError):
        TurnRecord("a", "main", 0, 10, 5, None, 20, 10, 5, "ok")  # start before credit
    with pytest.raises(ServiceEvidenceError):
        TurnRecord("a", "main", 0, 1, 2, None, 3, 10, 0, "ok")  # completed with no tokens
    with pytest.raises(ServiceEvidenceError):
        TurnRecord("a", "tool", 0, 1, 2, None, 3, 10, 5, "ok")  # unknown kind
    with pytest.raises(ServiceEvidenceError):
        LoadRead("stock", 1, "lane-1", 24, (_turn("a", "main", 0, start_s=0, ttft_s=0.1, out=5),))
    low = LoadRead("candidate", 1, "lane-1", 48, (_turn("a", "main", 0, start_s=0, ttft_s=0.1, out=5),))
    high = LoadRead("candidate", 1, "lane-1", 24, (_turn("a", "main", 0, start_s=0, ttft_s=0.1, out=5),))
    with pytest.raises(ServiceEvidenceError, match="increasing loads"):
        capacity(low, high, CONTRACT, {})


def test_rows_file_rejects_the_wrong_shape(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"arm": "candidate", "window": 1, "lane": "l", "load": 24, "root_session_id": "a"}\n')
    with pytest.raises(ServiceEvidenceError, match="retained shape"):
        load_reads_jsonl(rows)
    rows.write_text("")
    with pytest.raises(ServiceEvidenceError, match="no turn rows"):
        load_reads_jsonl(rows)

"""Service rate scoring (cacheon/eval/service_capacity.py).

The real-record tests replay the retained 2026-09-26 final budget-4 pairs (AIPerf profile exports converted to
turn rows, profiling phase only) and pin the fixed-work rates and attainments the arena design was decided on;
those duration-bounded reads also show why `grade` refuses work that differs between arms. The synthetic tests pin
the invariants: fixed work is checked per root, misses stay in the denominator, the rate verdict survives the
spread of the paired windows, and the attainment gate is non-inferiority with fixed thresholds.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval.service_capacity import (
    LoadRead,
    ServiceContract,
    ServiceEvidenceError,
    TurnRecord,
    attainment,
    completed_work,
    fixed_work_rate,
    grade,
    load_reads_jsonl,
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


def test_grade_refuses_duration_bounded_reads_and_grades_identical_work(final_pairs):
    cand, inc = final_pairs[("candidate", 24)], final_pairs[("incumbent", 24)]
    # The 09-26 reads were duration-bounded: the arms completed different work, so they are not gradeable.
    with pytest.raises(ServiceEvidenceError, match="completed work differs"):
        grade([cand], [inc], CONTRACT, completed_work(inc), required=1.01, attainment_tolerance=0.02, attainment_margin=0.0)
    # The same read on both arms is the null: ratio exactly 1, attainment difference exactly 0.
    twin = replace(cand, arm="incumbent", lane="lane-2")
    verdict = grade([cand], [twin], CONTRACT, completed_work(cand), required=1.01, attainment_tolerance=0.0, attainment_margin=0.0)
    assert verdict.decision is SpeedStageDecision.FAIL and verdict.ratio == 1.0
    assert verdict.detail.startswith("candidate does not clear")
    assert attainment(cand, CONTRACT) == pytest.approx(0.8862, abs=5e-4)


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


WORK = {f"s{i}": (1, 0) for i in range(20)}


def _read(arm: str, *, rate: float, attain: float, window: int = 1, load: int = 12) -> LoadRead:
    """A synthetic read of the 20 one-turn roots in WORK whose rate and attainment are set by construction."""
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
    return LoadRead(arm, window, "lane-1" if arm == "candidate" else "lane-2", load, tuple(rows))


def test_grade_rate_verdict_survives_the_window_spread():
    inc = [_read("incumbent", rate=1.0, attain=0.9), _read("incumbent", rate=1.02, attain=0.9, window=2)]
    fast = [_read("candidate", rate=1.10, attain=0.9), _read("candidate", rate=1.12, attain=0.9, window=2)]
    slow = [_read("candidate", rate=1.00, attain=0.9), _read("candidate", rate=1.03, attain=0.9, window=2)]
    mixed = [_read("candidate", rate=1.04, attain=0.9), _read("candidate", rate=1.09, attain=0.9, window=2)]
    thresholds = dict(required=1.05, attainment_tolerance=0.05, attainment_margin=0.0)
    passed = grade(fast, inc, CONTRACT, WORK, **thresholds)
    assert passed.decision is SpeedStageDecision.PASS
    assert passed.ratio == pytest.approx(1.10 / 1.02, rel=1e-3) and passed.required == 1.05
    assert grade(slow, inc, CONTRACT, WORK, **thresholds).decision is SpeedStageDecision.FAIL
    undetermined = grade(mixed, inc, CONTRACT, WORK, **thresholds)
    assert undetermined.decision is SpeedStageDecision.NO_DECISION
    assert "spread" in undetermined.detail


def test_grade_attainment_gate_is_non_inferiority_with_fixed_thresholds():
    inc = [_read("incumbent", rate=1.0, attain=0.90)]
    cand = [_read("candidate", rate=1.20, attain=0.80)]  # faster by starving two of twenty turns
    failed = grade(cand, inc, CONTRACT, WORK, required=1.05, attainment_tolerance=0.05, attainment_margin=0.0)
    assert failed.decision is SpeedStageDecision.FAIL
    assert failed.detail.startswith("service_contract_not_met") and failed.ratio == pytest.approx(1.2, rel=1e-3)
    # A product tolerance that admits the difference lets the rate verdict through.
    admitted = grade(cand, inc, CONTRACT, WORK, required=1.05, attainment_tolerance=0.12, attainment_margin=0.0)
    assert admitted.decision is SpeedStageDecision.PASS
    # The calibrated noise margin only makes the gate stricter: the same tolerance now fails.
    stricter = grade(cand, inc, CONTRACT, WORK, required=1.05, attainment_tolerance=0.12, attainment_margin=0.03)
    assert stricter.decision is SpeedStageDecision.FAIL


def test_grade_refuses_unpaired_reads_other_work_and_bad_thresholds():
    inc, cand = [_read("incumbent", rate=1.0, attain=0.9)], [_read("candidate", rate=1.1, attain=0.9)]
    thresholds = dict(required=1.05, attainment_tolerance=0.05, attainment_margin=0.0)
    with pytest.raises(ServiceEvidenceError, match="completed work differs"):
        grade(cand, inc, CONTRACT, {**WORK, "s99": (1, 0)}, **thresholds)
    with pytest.raises(ServiceEvidenceError, match="one read per arm"):
        grade(cand, inc + inc, CONTRACT, WORK, **thresholds)
    with pytest.raises(ServiceEvidenceError, match="pairs one candidate"):
        grade(cand, [replace(inc[0], load=8)], CONTRACT, WORK, **thresholds)
    with pytest.raises(ServiceEvidenceError, match="pairs one candidate"):
        grade(inc, cand, CONTRACT, WORK, **thresholds)
    with pytest.raises(ServiceEvidenceError, match="required ratio"):
        grade(cand, inc, CONTRACT, WORK, required=0.99, attainment_tolerance=0.05, attainment_margin=0.0)
    with pytest.raises(ServiceEvidenceError, match="attainment_margin"):
        grade(cand, inc, CONTRACT, WORK, required=1.05, attainment_tolerance=0.05, attainment_margin=1.0)


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


def test_rows_file_rejects_the_wrong_shape(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"arm": "candidate", "window": 1, "lane": "l", "load": 24, "root_session_id": "a"}\n')
    with pytest.raises(ServiceEvidenceError, match="retained shape"):
        load_reads_jsonl(rows)
    rows.write_text("")
    with pytest.raises(ServiceEvidenceError, match="no turn rows"):
        load_reads_jsonl(rows)

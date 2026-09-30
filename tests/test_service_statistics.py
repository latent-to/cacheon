"""Statistical eligibility preserves cost, lane, and repeated-observation accounting."""

import math
from dataclasses import replace

import pytest

from cacheon.eval.goodput_runtime import GoodputPolicy
from cacheon.eval.service_capacity import (
    LoadRead, ServiceContract, TurnRecord, statistical_grade,
)
from cacheon.eval.scoring import SpeedupVerdict
from cacheon.eval.speed_verdict import SpeedStageDecision, fail_reason


CONTRACT = ServiceContract(0.001, 1000.0, 0.8)
WORK = {"session": (2, 0)}


def test_tokenizer_failure_stays_before_the_gpu_boundary(monkeypatch):
    import sys
    from types import SimpleNamespace
    from cacheon.eval import goodput_runtime

    def unavailable(*args, **kwargs):
        raise OSError("tokenizer input is unavailable")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=unavailable)))
    monkeypatch.setattr(goodput_runtime, "_run_orientation", lambda *a, **k: pytest.fail("opened GPU lane"))
    arm = SimpleNamespace(session_plan=SimpleNamespace(replay=SimpleNamespace(tokenizer_path="/missing")))
    with pytest.raises(OSError, match="tokenizer input is unavailable"):
        goodput_runtime.run_goodput_pair(SimpleNamespace(baseline=arm, candidate=arm),
            baseline_executor=None, candidate_executor=None, model_mount=None,
            deadline=100, clock=lambda: 0, quality_control=None)


def _read(arm, lane, window, seconds):
    cold = TurnRecord("session", "main", 0, 10**9, 10**9, 10**9+1,
                      2*10**9, 100, 2, "ok")
    warm = TurnRecord("session", "main", 1, 3*10**9, 3*10**9, 3*10**9+1,
                      3*10**9+round(seconds*10**9), 100, 2, "ok")
    return LoadRead(arm, window, lane, 24, (cold, warm))


def _grade(costs, *, window_noise=0.0001, boot_noise=0.0001, max_windows=None, futility_margin=0.0):
    incumbent, candidate = [], []
    for window, (lane, b, c) in enumerate(costs, 1):
        incumbent.append(_read("incumbent", lane, window, b))
        candidate.append(_read("candidate", "B" if lane == "A" else "A", window, c))
    return statistical_grade(
        candidate, incumbent, CONTRACT, WORK, window_noise=window_noise,
        boot_noise=boot_noise, error_rate=0.01, max_windows=max_windows or len(costs),
        attainment_tolerance=0.0, attainment_margin=0.0, futility_margin=futility_margin,
    )


def test_crossover_cancels_lane_speed_with_unequal_window_counts():
    costs = [("A", 108, 100/1.003)]*2 + [("B", 100, 108/1.003)]*3
    result = _grade(costs)
    assert result.ratio == pytest.approx(1.003)
    assert result.decision is SpeedStageDecision.PASS
    assert 1 < result.required < result.ratio < 1.01


def test_pooling_does_not_reward_equal_total_cost_or_erase_slow_windows():
    result = _grade([("A", 100, 88), ("A", 100, 112),
                     ("B", 100, 88), ("B", 100, 112)])
    assert result.ratio == 1
    assert result.decision is SpeedStageDecision.FAIL
    slower = _grade([("A", 100, 98), ("A", 100, 140), ("B", 100, 98)])
    assert slower.ratio < 1
    assert slower.decision is SpeedStageDecision.FAIL


def test_serving_cost_charges_concurrent_tails_instead_of_summed_latency():
    from cacheon.eval.service_capacity import fixed_work_rate

    read = _read("candidate", "A", 1, 9)
    short = replace(read.records[-1], ordinal=2, request_end_ns=4*10**9)
    read = replace(read, records=(*read.records, short))
    work = {"session": (3, 0)}
    assert fixed_work_rate(read, work).elapsed_s == 10
    assert fixed_work_rate(read, work, wall_time=True).elapsed_s == 9
    balanced = replace(read, records=(read.records[0],
                        *(replace(row, request_end_ns=8*10**9) for row in read.records[1:])))
    assert fixed_work_rate(balanced, work).elapsed_s == 10
    assert fixed_work_rate(balanced, work, wall_time=True).elapsed_s == 5


def test_one_orientation_cannot_qualify_and_copies_do_not_pass():
    first = _grade([("A", 100, 90), ("A", 100, 90)], max_windows=5)
    assert first.decision is SpeedStageDecision.NO_DECISION
    copy = _grade([("A", 100, 100), ("B", 100, 100)])
    assert copy.decision is SpeedStageDecision.FAIL
    assert copy.lower_ratio < 1


def test_only_a_complete_first_orientation_beyond_the_futility_margin_fails_early():
    slow, near = [("A", 100, 102)] * 2, [("A", 100, 101)] * 2
    assert _grade(slow, max_windows=4, futility_margin=0.015).decision is SpeedStageDecision.FAIL
    assert _grade(slow, max_windows=4).decision is SpeedStageDecision.NO_DECISION
    assert _grade(near, max_windows=4, futility_margin=0.015).decision is SpeedStageDecision.NO_DECISION
    assert _grade(slow[:1], max_windows=4, futility_margin=0.015).decision is SpeedStageDecision.NO_DECISION


def test_a_futility_fail_skips_the_second_boot_and_keeps_orientation_one(monkeypatch):
    import sys
    from types import SimpleNamespace
    from cacheon.eval import crossover_runtime, goodput_runtime

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object())))
    fail = SimpleNamespace(decision=SpeedStageDecision.FAIL, verdict="verdict")
    calls, built = [], []
    monkeypatch.setattr(goodput_runtime, "_run_orientation", lambda plan, **k: calls.append(k) or dict(
        grade=fail, executions=("run-b", "run-c"), workload="w", started=0.0, reads="reads", quality=((), (), None)))
    monkeypatch.setattr(goodput_runtime, "_orientation_plan", lambda plan, windows, swapped=False: plan)
    monkeypatch.setattr(crossover_runtime, "_expected_lane_digest", lambda arm: arm.name)
    monkeypatch.setattr(crossover_runtime, "ResidentCrossoverEvidence", type("Evidence", (), {
        "__init__": lambda self, *a, **k: built.append((a, k)), "regrade": lambda self, plan: None}))
    replay = SimpleNamespace(tokenizer_path="/t", windows=4)
    arm = lambda name: SimpleNamespace(name=name, session_plan=SimpleNamespace(replay=replay))
    plan = SimpleNamespace(baseline=arm("lane-b"), candidate=arm("lane-c"), digest="p", selected_delta_digest="d",
                           policy=SimpleNamespace(goodput=SimpleNamespace(error_rate=0.01)))
    executor = lambda name: SimpleNamespace(prove_quiescent=lambda: name)
    goodput_runtime.run_goodput_pair(plan, baseline_executor=executor("quiet-b"), candidate_executor=executor("quiet-c"),
                                     model_mount=None, deadline=100, clock=lambda: 1.0, quality_control=None)
    assert len(calls) == 1 and "prior" not in calls[0]
    (args, kwargs), = built
    assert args[4:10] == ("lane-b", "lane-c", "run-b", "run-c", "quiet-b", "quiet-c") and args[15] == "clear_fail"
    assert kwargs == {"goodput": "reads"}
    plan.policy.goodput.error_rate = 0.0
    with pytest.raises(ValueError, match="V16 replay is regrade-only"):
        goodput_runtime.run_goodput_pair(plan, baseline_executor=None, candidate_executor=None, model_mount=None,
                                         deadline=100, clock=lambda: 1.0, quality_control=None)


def test_more_windows_do_not_average_away_paired_boot_uncertainty():
    short = _grade([("A", 100, 100/1.003), ("B", 100, 100/1.003)], boot_noise=0.005)
    longer = _grade([("A", 100, 100/1.003)]*2 + [("B", 100, 100/1.003)]*3,
                    boot_noise=0.005)
    assert longer.standard_error >= 0.005/math.sqrt(2)
    assert longer.standard_error > 0.999*short.standard_error
    assert longer.decision is SpeedStageDecision.FAIL


def test_statistical_policy_roundtrips_without_changing_historical_wire_shape():
    old = GoodputPolicy(CONTRACT, 1.0112, 0.0056, 0.05, 0.02)
    assert "error_rate" not in old.to_dict()
    assert GoodputPolicy.from_dict(old.to_dict()) == old
    new = replace(old, required=1.0, error_rate=0.01, boot_noise=0.002)
    assert GoodputPolicy.from_dict(new.to_dict()) == new
    with pytest.raises(ValueError, match="calibration inputs"):
        replace(new, null_noise=0.0, boot_noise=0.0)
    stop = replace(new, futility_margin=0.015)
    assert "futility_margin" not in new.to_dict() and stop.to_dict()["futility_margin"] == format(0.015, ".17g")
    assert GoodputPolicy.from_dict(stop.to_dict()) == stop
    with pytest.raises(ValueError, match="calibration inputs"):
        replace(old, futility_margin=0.015)


@pytest.mark.parametrize("distinct_policies", [False, True])
def test_lane_swap_rebinds_the_actual_launch_and_preflight(tmp_path, distinct_policies):
    from cacheon.eval.goodput_runtime import _orientation_plan
    from cacheon.eval.engine_launch import validate_native_build_spec, validate_runtime_preflight_receipt
    from tests.test_crossover_runtime import _rig

    plan, *_ = _rig(tmp_path, distinct_runtime_policies=distinct_policies)
    swapped = _orientation_plan(plan, 3, swapped=True)
    for source, lane, actual in ((plan.baseline, plan.candidate, swapped.baseline),
                                  (plan.candidate, plan.baseline, swapped.candidate)):
        assert actual.launch.stack_digest == source.launch.stack_digest
        assert actual.binding.physical_hardware == lane.binding.physical_hardware
        assert actual.session_plan.expected_preflight.launch_digest == actual.launch.digest
        assert actual.session_plan.replay.windows == 3
        validate_native_build_spec(actual.launch, actual.binding.native_build_spec)
        validate_runtime_preflight_receipt(actual.launch, actual.binding.runtime_preflight_receipt)
        actual.binding.physical_hardware.validate_against(actual.launch.hardware)


def test_fail_reason_splits_the_band_miss_from_the_measured_slowdown():
    # The retained defect this pins: a 1.003 speedup against a 1.005 bar was
    # reported as a "regression". Inside the band -- above the mirrored bound
    # 1-u for a bar of 1+u -- a FAIL proves only that the bar was not cleared.
    in_band = SpeedupVerdict(1.003, 0.001, 1.005, False, True, 2)
    slower = SpeedupVerdict(0.9, 0.001, 1.005, False, True, 2)
    assert fail_reason(in_band) == "speed_threshold_not_met"
    assert fail_reason(slower) == "candidate_slower"

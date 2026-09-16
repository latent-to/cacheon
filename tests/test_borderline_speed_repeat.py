"""A borderline result gets one complete repeat, with neither round discarded."""
from dataclasses import replace
import math

import pytest

from cacheon.eval.crossover_runtime import CrossoverRuntimeError
from cacheon.eval.qualification_runner import ResidentSpeedWitness
from cacheon.eval.resident_schedule import grade_schedule, repeat_required
from cacheon.eval.speed_verdict import SpeedStageDecision, schedule_roles
from tests.test_crossover_runtime import _resident_policy, _rig, _speed
from tests.test_prefill_lane import _read, _rates, PREFILL


def policy(version=14):
    return _resident_policy(version=version, min_margin=0.01, max_window_scatter=0.05,
                            **(PREFILL if version == 15 else {}))


def round_rates(candidate=101.0, before=100.0, after=100.2):
    return tuple(_read(role, rate) for role, rate in
                 zip(schedule_roles(14), (before, candidate, after), strict=True))


def repeat_rows(rows):
    # SimpleNamespace test observations have no serialized identity; live-path
    # tests below cover the actual raw spans and witness identities.
    return tuple(type(row)(**{**vars(row), "role": row.role + "_repeat"}) for row in rows)


@pytest.mark.parametrize("version", (13, 14))
@pytest.mark.parametrize("second,expected", ((103.0, "PASS"), (101.0, "FAIL"), (99.0, "FAIL")))
def test_two_rounds_conclude_using_both_conservative_ratios(version, second, expected):
    first = round_rates()
    assert repeat_required(policy(version), first)
    grade = grade_schedule(policy(version), first + repeat_rows(round_rates(second)))
    assert grade.decision.value == expected
    assert grade.verdict.speedup == pytest.approx(math.sqrt(101.0 / 100.2 * second / 100.2))
    assert grade.verdict.n_baselines == 4
    assert grade.verdict.n_candidates == 2
    assert grade.verdict.confident


@pytest.mark.parametrize("candidate", (95.0, 100.0, 105.0))
def test_decisive_first_round_cannot_be_rerolled(candidate):
    first = round_rates(candidate)
    assert not repeat_required(policy(), first)
    with pytest.raises(CrossoverRuntimeError, match="not authorized"):
        grade_schedule(policy(), first + repeat_rows(round_rates(110.0)))


def test_invalid_repeat_does_not_convict_or_admit_candidate():
    first = round_rates()
    for candidate in (1.0, 1000.0):
        grade = grade_schedule(policy(), first + repeat_rows(round_rates(candidate, after=110.0)))
        assert grade.decision is SpeedStageDecision.NO_DECISION
        assert not grade.verdict.confident
    assert not repeat_required(policy(), round_rates(after=110.0))


def test_all_baselines_contribute_to_repeat_validity():
    second = round_rates(120.0)
    second[2].windows = tuple(replace(w, seconds=w.seconds * (1.2 if i == 0 else 1))
                              for i, w in enumerate(second[2].windows))
    grade = grade_schedule(policy(), round_rates() + repeat_rows(second))
    assert grade.decision is SpeedStageDecision.NO_DECISION
    assert not grade.verdict.passed_speedup


def test_repeat_rejects_changed_workload_and_a_third_round():
    second = round_rates(103.0)
    for row in second:
        row.windows = tuple(replace(w, tokens=200) for w in row.windows)
    grade = grade_schedule(policy(), round_rates() + repeat_rows(second))
    assert grade.decision is SpeedStageDecision.NO_DECISION
    with pytest.raises(CrossoverRuntimeError, match="precommitted"):
        grade_schedule(policy(), round_rates() + repeat_rows(round_rates()) * 2)


def test_prefill_boundary_gets_one_repeat_without_rescuing_decode_regression():
    first = _rates(100.5, 42.04)
    first[5].timed_seconds *= 0.998
    first[5].windows = tuple(replace(w, seconds=w.seconds * 0.998) for w in first[5].windows)
    assert repeat_required(policy(15), first)
    grade = grade_schedule(policy(15), first + repeat_rows(_rates(100.5, 44.0)))
    assert grade.decision is SpeedStageDecision.PASS
    assert grade.lane == "prefill"
    assert grade.prefill_verdict.n_candidates == 2
    failed = grade_schedule(policy(15), first + repeat_rows(_rates(95.0, 60.0)))
    assert failed.decision is SpeedStageDecision.FAIL
    assert failed.lane is None


@pytest.mark.parametrize("version", (13, 14, 15))
@pytest.mark.parametrize("second,decision", ((0.97, "PASS"), (0.99, "FAIL")))
def test_real_schedule_repeats_serially_and_reopens_raw_evidence(tmp_path, version, second, decision):
    candidate = (0.99, 1.0, second, 1.0) if version == 15 else (0.99, second)
    baseline = (1.0, 0.998, 1.0, 1.0, 1.0, 0.998, 1.0, 1.0) if version == 15 else (1.0, 0.998, 1.0, 0.998)
    plan, left, right, mount, trace, overlap = _rig(
        tmp_path, candidate, policy=policy(version), baseline_durations=baseline)
    result = _speed(plan, left, right, mount)
    assert result.escalated
    assert result.decision.value == decision
    assert tuple(row.role for row in result.rates) == schedule_roles(version, repeat=True)
    assert not overlap[0]
    assert result.regrade(plan) == result.final_verdict
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    reopened = ResidentSpeedWitness.from_dict(witness.to_dict())
    assert reopened == witness
    assert reopened.always_bookend_result()[0].value == decision
    assert trace[-2:] == ["left:close", "right:close"] or trace[-2:] == ["right:close", "left:close"]
    with pytest.raises(CrossoverRuntimeError):
        replace(result, rates=result.rates[:len(schedule_roles(version))], escalated=False).regrade(plan)


@pytest.mark.parametrize("version", (13, 14, 15))
def test_clear_pass_does_not_execute_reserved_repeat_batches(tmp_path, version):
    candidate = (0.8, 1.0) if version == 15 else (0.8,)
    plan, left, right, mount, trace, overlap = _rig(
        tmp_path, candidate, policy=policy(version), baseline_durations=(1.0,) * 4)
    result = _speed(plan, left, right, mount)
    assert result.decision is SpeedStageDecision.PASS
    assert not result.escalated
    assert len(result.rates) == len(schedule_roles(version))
    assert not overlap[0]
    assert result.regrade(plan) == result.final_verdict


def test_failure_during_repeat_propagates_without_returning_first_round(tmp_path, monkeypatch):
    from tests.test_crossover_runtime import _Controller

    original = _Controller.execute_next

    def fail_in_repeat(controller):
        if controller.lane == "right" and controller.next_batch_index == controller.batches_per_read:
            raise CrossoverRuntimeError("repeat execution interrupted")
        return original(controller)

    monkeypatch.setattr(_Controller, "execute_next", fail_in_repeat)
    plan, left, right, mount, trace, overlap = _rig(
        tmp_path, (0.99, 0.97), policy=policy(),
        baseline_durations=(1.0, 0.998, 1.0, 0.998))
    with pytest.raises(CrossoverRuntimeError, match="repeat execution interrupted"):
        _speed(plan, left, right, mount)
    assert "left:8" in trace
    assert not overlap[0]


def test_versioned_repeat_does_not_reinterpret_retained_v11():
    rates = round_rates()
    old = grade_schedule(replace(policy(), version=11), rates)
    assert old.decision is SpeedStageDecision.NO_DECISION
    assert not repeat_required(replace(policy(), version=11), rates)
    assert replace(policy(), version=11).digest != policy().digest


def test_new_policy_repeats_only_when_actual_required_bound_is_crossed():
    # Valid drift raises the calibrated requirement above both comparisons.
    rates = round_rates(candidate=102.0, before=100.0, after=101.0)
    grade = grade_schedule(policy(), rates)
    assert grade.verdict.required > 1.01
    # This candidate just clears the raised upper comparison; still borderline.
    assert repeat_required(policy(), rates)
    rates = round_rates(candidate=101.8, before=100.0, after=101.0)
    assert grade_schedule(policy(), rates).decision is SpeedStageDecision.FAIL
    assert not repeat_required(policy(), rates)


def test_repeat_conditioning_regression_is_not_hidden_by_a_fast_timed_read():
    second = round_rates(110.0)
    second[1].conditioning_seconds = 2.0
    grade = grade_schedule(policy(), round_rates() + repeat_rows(second))
    assert grade.conditioning_failed
    assert grade.decision is SpeedStageDecision.FAIL


def test_invalid_prefill_repeat_remains_inconclusive_when_decode_cannot_admit():
    first = _rates(101.0, 40.0, after=100.2)
    second = _rates(100.0, 80.0)
    second[5].timed_seconds /= 1.1
    second[5].windows = tuple(replace(w, seconds=w.seconds / 1.1) for w in second[5].windows)
    grade = grade_schedule(policy(15), first + repeat_rows(second))
    assert grade.decision is SpeedStageDecision.NO_DECISION
    assert not grade.prefill_verdict.confident


def test_dashboard_displays_combined_grade_and_repeat_prefill_units(tmp_path):
    from cacheon.chain.baseline_band import qualification_speed
    from cacheon.eval.evidence_store import prepare_evidence_root, publish_canonical_json_evidence
    import json

    plan, left, right, mount, _, _ = _rig(
        tmp_path / "run", (0.99, 1.0, 0.97, 1.0), policy=policy(15),
        baseline_durations=(1.0, 0.998, 1.0, 1.0, 1.0, 0.998, 1.0, 1.0))
    result = _speed(plan, left, right, mount)
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    root = prepare_evidence_root(tmp_path / "evidence")
    ref = publish_canonical_json_evidence(root, {"speed_witness": witness.to_dict()},
          domain="qualification.stage-exit", schema="cacheon.qualification.stage-exit.v1")
    report = qualification_speed(json.dumps(ref.to_dict()), (root,))
    assert report["speedup"] == pytest.approx(result.final_verdict.speedup)
    assert report["grading"]["decision"] == "PASS"
    repeated_prefill = [row for row in report["lanes"] if row["role"].endswith("prefill_repeat")]
    assert len(repeated_prefill) == 3
    assert all(row["tokens_per_second"] is None and row["prompts_per_second"] > 0
               for row in repeated_prefill)

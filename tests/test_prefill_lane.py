"""Speed policy v12: the prefill lane can only help a prefill bundle.

The decode reads are graded byte for byte as v11 grades them. The prefill pass
is appended after the decode schedule and consulted only when the decode floor
neither admitted nor convicted the candidate (owner ruling 2026-09-08: one
end-to-end score, decode bundles not influenced at all).
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.crossover_runtime import (
    CrossoverRuntimeError,
    ResidentSpeedPolicy,
    SpeedStageDecision,
)
from cacheon.eval.qualification_runner import (
    QualificationDecision,
    ResidentSpeedWitness,
)
from cacheon.eval.resident_measurement import TimedWindow
from cacheon.eval.resident_schedule import (
    PREFILL_READ_BUDGET,
    credited_speedup,
    expanded_schedule,
    grade_schedule,
    prefill_policy,
)
from cacheon.eval.speed_verdict import (
    DECODE_ROLES,
    PREFILL_LANE_ROLES,
    fail_reason,
    resident_speed_roles,
)
from tests.test_crossover_runtime import _resident_policy, _rig, _speed
from tests.test_oci_backend import _case

PREFILL = {"prefill_min_margin": 0.05, "prefill_credit_weight": 0.5}


def _policy(version: int = 12, **overrides) -> ResidentSpeedPolicy:
    kwargs: dict[str, object] = {
        "version": version,
        "min_margin": 0.01,
        "max_window_scatter": 0.05,
    }
    if version >= 12:
        kwargs.update(PREFILL)
    kwargs.update(overrides)
    return _resident_policy(**kwargs)


def _read(role: str, rate: float, *, conditioning_seconds: float = 1.0):
    """A five-window read whose every window runs at exactly ``rate``."""

    windows = tuple(TimedWindow(index + 1, 100, 100 / rate) for index in range(5))
    return SimpleNamespace(
        role=role,
        windows=windows,
        timed_tokens=500,
        timed_seconds=sum(window.seconds for window in windows),
        first_batch_index=0,
        first_timed_batch_index=1,
        conditioning_tokens=100,
        conditioning_seconds=conditioning_seconds,
    )


def _rates(
    decode: float,
    prefill: float,
    *,
    before: float = 100.0,
    after: float = 100.0,
    conditioning: float = 1.0,
) -> tuple:
    return (
        _read("B", before),
        _read("C", decode, conditioning_seconds=conditioning),
        _read("B_prime", after),
        _read("B_prefill", 40.0),
        _read("C_prefill", prefill),
        _read("B_prime_prefill", 40.0),
    )


def test_a_decode_pass_is_graded_exactly_as_v11_whatever_the_prefill_pass_says():
    rates = _rates(102.0, 20.0)  # the prompt pass is twice as slow
    grade = grade_schedule(_policy(), rates)
    decode = grade_schedule(_policy(11), rates[:3])

    assert decode.decision is SpeedStageDecision.PASS
    assert grade.verdict == decode.verdict
    assert grade.decision is decode.decision
    assert grade.lane == "decode"
    assert grade.settled_speedup == format(decode.verdict.speedup, ".17g")
    assert grade.prefill_verdict is not None
    assert not grade.prefill_verdict.passed_speedup


@pytest.mark.parametrize("decode", (95.0, 98.0))
def test_a_measured_decode_loss_is_never_rescued_by_prefill(decode: float):
    grade = grade_schedule(_policy(), _rates(decode, 60.0))

    assert grade.decision is SpeedStageDecision.FAIL
    assert grade.lane is None
    assert fail_reason(grade.verdict) == "candidate_slower"
    assert grade.prefill_verdict.passed_speedup
    assert grade.settled_speedup == format(grade.verdict.speedup, ".17g")


def test_a_conditioning_regression_is_never_rescued_by_prefill():
    grade = grade_schedule(_policy(), _rates(100.5, 60.0, conditioning=1.5))

    assert grade.conditioning_failed
    assert grade.decision is SpeedStageDecision.FAIL
    assert grade.lane is None
    assert fail_reason(grade.verdict, conditioning_failed=True) == "candidate_slower"


def test_a_competitive_decode_miss_with_a_prefill_win_admits_on_the_prefill_lane():
    rates = _rates(100.5, 50.0)
    decode = grade_schedule(_policy(11), rates[:3])
    grade = grade_schedule(_policy(), rates)

    assert decode.decision is SpeedStageDecision.FAIL
    assert fail_reason(decode.verdict) == "speed_threshold_not_met"
    assert grade.verdict == decode.verdict  # the decode headline is retained
    assert grade.decision is SpeedStageDecision.PASS
    assert grade.lane == "prefill"
    assert grade.prefill_verdict.speedup == pytest.approx(1.25)
    assert float(grade.settled_speedup) == pytest.approx(1.125)
    assert grade.settled_speedup == format(
        credited_speedup(_policy(), grade.prefill_verdict), ".17g"
    )


def test_a_prefill_gain_below_its_own_margin_does_not_admit():
    grade = grade_schedule(_policy(), _rates(100.5, 41.0))  # 1.025 < 1.05

    assert grade.decision is SpeedStageDecision.FAIL
    assert grade.lane is None
    assert not grade.prefill_verdict.passed_speedup
    assert fail_reason(grade.verdict) == "speed_threshold_not_met"


def test_boundary_uncertainty_on_a_valid_decode_measurement_can_be_rescued():
    rates = _rates(101.7, 50.0, before=100.0, after=101.5)
    decode = grade_schedule(_policy(11), rates[:3])
    grade = grade_schedule(_policy(), rates)

    assert decode.decision is SpeedStageDecision.NO_DECISION
    assert decode.verdict.confident
    assert grade.decision is SpeedStageDecision.PASS
    assert grade.lane == "prefill"


def test_an_invalid_decode_measurement_stays_no_decision():
    grade = grade_schedule(_policy(), _rates(120.0, 50.0, before=80.0, after=100.0))

    assert grade.decision is SpeedStageDecision.NO_DECISION
    assert grade.lane is None
    assert not grade.verdict.confident
    assert grade.prefill_verdict.passed_speedup


def test_the_read_set_must_be_the_precommitted_schedule():
    rates = _rates(102.0, 50.0)
    with pytest.raises(CrossoverRuntimeError, match="precommitted"):
        grade_schedule(_policy(), rates[:3])
    with pytest.raises(CrossoverRuntimeError, match="precommitted"):
        grade_schedule(_policy(11), rates)
    with pytest.raises(CrossoverRuntimeError, match="precommitted"):
        grade_schedule(_policy(), rates[:3] + (rates[4], rates[3], rates[5]))
    assert resident_speed_roles(12, 6) == PREFILL_LANE_ROLES
    assert resident_speed_roles(12, 3) is None
    assert resident_speed_roles(11, 3) == DECODE_ROLES
    assert resident_speed_roles(11, 2) is None
    assert PREFILL_LANE_ROLES[:3] == DECODE_ROLES


def test_v12_seals_its_prefill_thresholds_and_earlier_versions_forbid_them():
    policy = _policy()
    row = policy.to_dict()
    assert {"prefill_min_margin", "prefill_credit_weight"} <= set(row)
    assert ResidentSpeedPolicy.from_dict(row) == policy
    assert "prefill_min_margin" not in _policy(11).to_dict()
    assert policy.digest != replace(policy, prefill_credit_weight=0.25).digest
    with pytest.raises(CrossoverRuntimeError, match="fields differ"):
        ResidentSpeedPolicy.from_dict(
            {key: value for key, value in row.items() if key != "prefill_min_margin"}
        )
    with pytest.raises(CrossoverRuntimeError, match="v12 requires"):
        _policy(prefill_min_margin=0.0)
    with pytest.raises(CrossoverRuntimeError, match="v12 requires"):
        _policy(prefill_credit_weight=1.5)
    with pytest.raises(CrossoverRuntimeError, match="require resident speed policy v12"):
        _policy(11, prefill_min_margin=0.05)
    assert _policy(prefill_credit_weight=1.0).prefill_credit_weight == 1.0
    assert prefill_policy(policy) == replace(policy, min_margin=0.05)
    assert credited_speedup(policy, SimpleNamespace(speedup=1.25)) == 1.125


def test_expanded_schedule_appends_one_token_prefill_reads(tmp_path: Path):
    plan = _case(tmp_path).plan
    count = len(plan.prompt_batches)
    decode_only = expanded_schedule(plan, 2)
    assert decode_only.prompt_batches == plan.prompt_batches * 2
    assert decode_only.batch_max_new_tokens == ()
    assert decode_only.batch_expected_prompt_tokens == ()

    expanded = expanded_schedule(plan, 2, prefill_reads=2)
    assert expanded.prompt_batches == plan.prompt_batches * 4
    assert expanded.batch_max_new_tokens == (
        (plan.max_new_tokens,) * (2 * count) + (PREFILL_READ_BUDGET,) * (2 * count)
    )
    assert expanded.request_geometry(0) == (plan.max_new_tokens, plan.expected_prompt_tokens)
    assert expanded.request_geometry(4 * count - 1) == (1, plan.expected_prompt_tokens)
    assert expanded.quality_tokens_per_prompt == plan.quality_tokens_per_prompt

    mixed = replace(
        plan,
        batch_max_new_tokens=tuple(range(2, 2 + count)),
        batch_expected_prompt_tokens=(7,) * count,
    )
    lane = expanded_schedule(mixed, 1, prefill_reads=1)
    assert lane.batch_max_new_tokens == tuple(range(2, 2 + count)) + (1,) * count
    assert lane.batch_expected_prompt_tokens == (7,) * (2 * count)


def _live_policy() -> ResidentSpeedPolicy:
    return _resident_policy(version=12, **PREFILL)


def _live(tmp_path: Path, decode: float, prefill: float):
    plan, baseline, candidate, mount, trace, overlap = _rig(
        tmp_path,
        (decode, prefill),
        policy=_live_policy(),
        timed_batches=3,
        baseline_durations=(1.0, 1.0, 1.0, 1.0),
    )
    return plan, _speed(plan, baseline, candidate, mount), trace, overlap


def test_v12_appends_the_prefill_pass_after_the_decode_schedule(tmp_path: Path):
    plan, result, trace, overlap = _live(tmp_path, 0.90, 0.5)

    assert plan.policy.version == 12
    assert tuple(row.role for row in result.rates) == PREFILL_LANE_ROLES
    # Two decode reads and two prefill reads of four batches on the baseline
    # lane; one of each on the candidate lane.
    assert len(result.baseline_execution.session.batches) == 16
    assert len(result.candidate_execution.session.batches) == 8
    assert not overlap[0]
    assert trace.index("right:4") > trace.index("left:11")
    assert trace.index("left:12") > trace.index("right:7")
    # One generated token per prompt per timed batch in the prompt pass.
    assert all(row.timed_tokens == 3 * PREFILL_READ_BUDGET for row in result.rates[3:])
    assert all(row.windows for row in result.rates)

    assert result.decision is SpeedStageDecision.PASS
    decode_policy = replace(
        plan.policy, version=11, prefill_min_margin=0.0, prefill_credit_weight=0.0
    )
    assert grade_schedule(decode_policy, result.rates[:3]).verdict == result.final_verdict
    grade = grade_schedule(plan.policy, result.rates)
    assert grade.lane == "decode"
    assert result.regrade(plan) == result.final_verdict
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness
    assert witness.accepted_speedup() == format(result.final_verdict.speedup, ".17g")


def test_v12_admits_a_prefill_win_the_decode_floor_did_not_convict(tmp_path: Path):
    plan, result, _trace, _overlap = _live(tmp_path, 1.0, 0.8)

    assert result.decision is SpeedStageDecision.PASS
    assert not result.final_verdict.passed_speedup  # the decode headline stays
    grade = grade_schedule(plan.policy, result.rates)
    assert grade.lane == "prefill"
    assert grade.prefill_verdict.speedup == pytest.approx(1.25)
    assert result.regrade(plan) == result.final_verdict
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    decision, speedup, reason = witness.always_bookend_result()
    assert decision is QualificationDecision.PASS
    assert reason is None
    assert float(speedup) == pytest.approx(1.125)
    assert witness.accepted_speedup() == speedup


def test_v12_keeps_a_measured_decode_loss_a_loss(tmp_path: Path):
    plan, result, _trace, _overlap = _live(tmp_path, 1.10, 0.5)

    assert result.decision is SpeedStageDecision.FAIL
    assert grade_schedule(plan.policy, result.rates).lane is None
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    decision, speedup, reason = witness.always_bookend_result()
    assert decision is QualificationDecision.FAIL
    assert reason == "candidate_slower"
    assert speedup == format(result.final_verdict.speedup, ".17g")

"""Closed standing-controls contract (``cacheon/eval/standing_controls.py``)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from cacheon.eval.calibration import (
    CalibrationContext,
    CalibrationThresholdPolicy,
    MetricCalibration,
    SpeedCalibration,
    derive_calibration_manifest,
)
from cacheon.eval.crossover_runtime import ResidentSpeedPolicy
from cacheon.eval.standing_controls import (
    StandingControlsError,
    StandingEvaluationControls,
    bookend_drift_ppm_from_max_noise,
    verify_standing_controls,
)
from tests.test_calibration import _observations


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authority(
    version: int = 3, *, workload_digest: str | None = None
) -> tuple[CalibrationContext, object, ResidentSpeedPolicy]:
    context = CalibrationContext(*(_h(f"context-{index}") for index in range(10)))
    if workload_digest is not None:
        context = replace(context, workload_digest=workload_digest)
    threshold = CalibrationThresholdPolicy(
        context,
        "teacher-familywise-v1",
        "frozen",
        SpeedCalibration("0.005", "2", "0.01"),
        (
            MetricCalibration("mean_nll", "lower", "0.02", "0.01"),
            MetricCalibration("task_score", "higher", "0.03", "0.02", "0.8"),
        ),
        "2.576",
    )
    calibration = derive_calibration_manifest(threshold, _observations())
    policy = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=60,
        max_qualification_seconds=600,
        calibration=calibration,
        context=context,
        version=version,
        min_windows=3,
        max_window_scatter=0.05,
        max_conditioning_slowdown=1.5,
    )
    return context, calibration, policy


def _policy() -> ResidentSpeedPolicy:
    return _authority()[2]


def _controls(policy: ResidentSpeedPolicy) -> dict[str, object]:
    return {
        "baseline_health_policy_digest": _h("baseline-health"),
        "calibration_context_digest": _h("calibration-context"),
        "maximum_bookend_drift_ppm": bookend_drift_ppm_from_max_noise(
            policy.max_noise
        ),
        "reference_manifest_digest": _h("reference-manifest"),
    }


def test_identical_policy_derived_controls_are_accepted(tmp_path) -> None:
    policy = _policy()
    raw = _controls(policy)
    controls = verify_standing_controls(raw, dict(raw), policy)
    assert type(controls) is StandingEvaluationControls
    assert controls.to_dict() == raw
    assert controls.maximum_bookend_drift_ppm == bookend_drift_ppm_from_max_noise(
        policy.max_noise
    )


def test_controls_schema_is_closed_to_exactly_four_keys() -> None:
    policy = _policy()
    raw = _controls(policy)
    for missing in raw:
        broken = {key: value for key, value in raw.items() if key != missing}
        with pytest.raises(StandingControlsError, match="closed four-key schema"):
            StandingEvaluationControls.from_dict(broken)
    with pytest.raises(StandingControlsError, match="closed four-key schema"):
        StandingEvaluationControls.from_dict({**raw, "extra_control": _h("extra")})
    with pytest.raises(StandingControlsError, match="closed four-key schema"):
        StandingEvaluationControls.from_dict("not-a-mapping")


def test_controls_reject_untyped_digests_and_drift_bounds() -> None:
    policy = _policy()
    raw = _controls(policy)
    with pytest.raises(StandingControlsError):
        StandingEvaluationControls.from_dict(
            {**raw, "reference_manifest_digest": "zz" * 32}
        )
    for bad_ppm in (True, 0, -5, 1_000_001, "5000", 5000.0, None):
        with pytest.raises(StandingControlsError):
            StandingEvaluationControls.from_dict(
                {**raw, "maximum_bookend_drift_ppm": bad_ppm}
            )


def test_drift_bound_must_equal_the_sealed_speed_policy_exactly() -> None:
    policy = _policy()
    raw = _controls(policy)
    drifted = {
        **raw,
        "maximum_bookend_drift_ppm": raw["maximum_bookend_drift_ppm"] + 1,
    }
    with pytest.raises(
        StandingControlsError, match="differs from the sealed speed policy"
    ):
        verify_standing_controls(drifted, dict(drifted), policy)


def test_unsupported_policy_and_inexact_noise_are_rejected() -> None:
    policy = _policy()
    raw = _controls(policy)
    with pytest.raises(
        StandingControlsError, match="supported resident speed policy"
    ):
        verify_standing_controls(raw, dict(raw), object())
    with pytest.raises(
        StandingControlsError, match="no exact parts-per-million expression"
    ):
        bookend_drift_ppm_from_max_noise(1.5e-07)
    with pytest.raises(
        StandingControlsError, match="no exact parts-per-million expression"
    ):
        bookend_drift_ppm_from_max_noise(0.0)


def test_primary_and_reproduction_must_bind_identical_controls() -> None:
    policy = _policy()
    raw = _controls(policy)
    reproduction = {**raw, "reference_manifest_digest": _h("another-reference")}
    with pytest.raises(
        StandingControlsError, match="bind different controls"
    ):
        verify_standing_controls(raw, reproduction, policy)

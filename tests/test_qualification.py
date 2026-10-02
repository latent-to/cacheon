from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval.qualification import (
    QualificationProfile,
    QualificationError,
    ReferenceManifest,
    SelectionCommitment,
    SelectionEntropyReceipt,
    SelectionReceipt,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reference() -> ReferenceManifest:
    return ReferenceManifest(*(_d(f"reference:{index}") for index in range(18)))


def test_qualification_import_is_stdlib_only_and_does_not_import_torch():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cacheon.eval.qualification; assert 'torch' not in sys.modules",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_reference_profile_and_precommitted_selection_round_trip():
    reference = _reference()
    profile = QualificationProfile(
        reference,
        _d("calibration-context"),
        _d("calibration"),
        ("mean_nll", "task_score", "topk_kl"),
        "2",
        10,
        2,
        2,
        _d("support-policy"),
        _d("hidden-task-policy"),
        _d("runtime-resource-policy"),
        True,
        2,
    )
    assert ReferenceManifest.from_dict(reference.to_dict()) == reference
    assert QualificationProfile.from_dict(profile.to_dict()) == profile
    assert set(profile.to_dict()) == set(
        "reference calibration_context_digest calibration_digest "
        "required_quality_metrics nll_tail_threshold tokens_per_prompt topk_width "
        "hidden_tasks_per_prompt support_policy_digest hidden_task_policy_digest "
        "runtime_resource_policy_digest hidden_tasks_required minimum_prompt_count "
        "policy_version schema_version".split()
    )

    prompts = tuple(sorted(_d(f"prompt:{index}") for index in range(8)))
    secret = b"pre-result secret" * 4
    commitment = SelectionCommitment.seal(
        source_plan_digest=_d("cohort"),
        reference_manifest=reference,
        entropy_source_digest=_d("future-block-source"),
        prompt_digests=prompts,
        select_count=3,
        secret=secret,
    )
    entropy = SelectionEntropyReceipt(
        commitment.entropy_source_digest,
        commitment.digest,
        _d("future-block-value"),
        _d("future-block-receipt"),
    )
    receipt = SelectionReceipt.reveal(
        commitment,
        secret=secret,
        entropy=entropy,
        sealed_cohort_trajectory_digest=_d("sealed-trajectories"),
    )
    assert SelectionCommitment.from_dict(commitment.to_dict()) == commitment
    assert SelectionReceipt.from_dict(receipt.to_dict()).reopen(commitment, entropy) == receipt
    assert len(receipt.selected_prompt_digests) == 3
    rebound = SelectionReceipt.reveal(
        commitment,
        secret=secret,
        entropy=entropy,
        sealed_cohort_trajectory_digest=_d("different-sealed-trajectories"),
    )
    assert rebound.selected_prompt_digests == receipt.selected_prompt_digests


def test_selection_rejects_late_substitution_or_forged_result():
    reference = _reference()
    prompts = tuple(sorted(_d(f"prompt:{index}") for index in range(4)))
    commitment = SelectionCommitment.seal(
        source_plan_digest=_d("cohort"),
        reference_manifest=reference,
        entropy_source_digest=_d("entropy-source"),
        prompt_digests=prompts,
        select_count=2,
        secret=b"a" * 32,
    )
    with pytest.raises(QualificationError, match="does not open"):
        entropy = SelectionEntropyReceipt(
            commitment.entropy_source_digest,
            commitment.digest,
            _d("entropy"),
            _d("entropy-receipt"),
        )
        SelectionReceipt.reveal(
            commitment,
            secret=b"b" * 32,
            entropy=entropy,
            sealed_cohort_trajectory_digest=_d("trajectories"),
        )
    receipt = SelectionReceipt.reveal(
        commitment,
        secret=b"a" * 32,
        entropy=entropy,
        sealed_cohort_trajectory_digest=_d("trajectories"),
    )
    wrong = tuple(sorted(set(prompts) - set(receipt.selected_prompt_digests)))
    with pytest.raises(QualificationError, match="does not reproduce"):
        replace(receipt, selected_prompt_digests=wrong).reopen(commitment, entropy)


def test_teacher_nll_only_profile_admits_width_zero_and_refuses_kl_metrics():
    # Option B (2026-07-25): topk_width 0 selects teacher-NLL-only quality.
    # The mode is digest-bound, and a profile cannot simultaneously declare
    # a distribution metric it retains no evidence for.
    reference = _reference()

    def profile(width: int, metrics: tuple[str, ...]) -> QualificationProfile:
        return QualificationProfile(
            reference,
            _d("calibration-context"),
            _d("calibration"),
            metrics,
            "2",
            10,
            width,
            2,
            _d("support-policy"),
            _d("hidden-task-policy"),
            _d("runtime-resource-policy"),
            True,
            2,
        )

    nll_only = profile(0, ("mean_nll", "task_score"))
    assert QualificationProfile.from_dict(nll_only.to_dict()) == nll_only
    assert nll_only.topk_width == 0
    with pytest.raises(QualificationError, match="cannot require distribution metrics"):
        profile(0, ("mean_nll", "task_score", "topk_kl"))


def test_resident_speed_witness_relabel_forgery_is_internally_undetectable():
    """The real witness's internal digest stops a naive delta relabel, but a
    forger who recomputes the projection digest constructs a fully valid
    witness for a delta the arm never ran. Internal validation therefore
    CANNOT catch arm relabeling; the reopen-time authority comparison
    ("speed witness differs from its marginal arm", pinned in
    test_qualification_runner's arm-relabel test) is the only line of
    defense, and this pair of tests keeps both halves honest."""

    from cacheon.eval.qualification_runner import (
        QualificationRunnerError,
        ResidentSpeedWitness,
        _resident_speed_projection_digest,
    )
    from tests.test_dashboard_replay import _witness

    # A real policy-17 replay witness: paired load reads, no batch rates.
    witness = _witness(statistical=True)
    honest = witness.to_dict()
    relabel = _d("relabeled-delta")

    with pytest.raises(QualificationRunnerError, match="does not recompute"):
        ResidentSpeedWitness.from_dict(
            {**honest, "selected_delta_digest": relabel}
        )

    forged_digest = _resident_speed_projection_digest(
        selected_delta_digest=relabel,
        candidate_launch_digest=witness.candidate_launch_digest,
        calibration_digest=witness.calibration_digest,
        calibration_context_digest=witness.calibration_context_digest,
        workload_digest=witness.workload_digest,
        baseline_runtime_resource_policy_digest=(
            witness.baseline_runtime_resource_policy_digest
        ),
        candidate_runtime_resource_policy_digest=(
            witness.candidate_runtime_resource_policy_digest
        ),
        plan_digest=witness.plan_digest,
        baseline_lane_digest=witness.baseline_lane_digest,
        candidate_lane_digest=witness.candidate_lane_digest,
        baseline_quiescence_digest=witness.baseline_quiescence_digest,
        candidate_quiescence_digest=witness.candidate_quiescence_digest,
        raw_crossover_digest=witness.raw_crossover_digest,
        resident_policy=witness.resident_policy,
        rates=witness.rates,
        started_monotonic_s=witness.started_monotonic_s,
        completed_monotonic_s=witness.completed_monotonic_s,
        goodput=witness.goodput,
    )
    forged = ResidentSpeedWitness.from_dict(
        {
            **honest,
            "selected_delta_digest": relabel,
            "evidence_digest": forged_digest,
        }
    )
    assert forged.selected_delta_digest == relabel
    assert forged.goodput == witness.goodput

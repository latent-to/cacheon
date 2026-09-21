from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval.qualification import (
    declared_qualification_entropy_digest,
    QualificationProfile,
    QualificationError,
    ReferenceManifest,
    SelectionCommitment,
    SelectionEntropyReceipt,
    SelectionReceipt,
    candidate_lifecycle_digest,
    cohort_trajectory_digest,
    derived_hidden_task_plan_digest,
    lifecycle_prompt_digests,
    qualification_identity_digest,
    selected_trajectory_digest,
    selected_trajectory_projection_digest,
    validate_quality_binding,
)
from cacheon.eval.evidence_store import publish_evidence
from cacheon.stack_identity import canonical_digest, canonical_json_bytes


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reference() -> ReferenceManifest:
    return ReferenceManifest(*(_d(f"reference:{index}") for index in range(18)))


class _FixtureLane:
    """Deterministic stand-in for one opened engine session, feeding the
    production ``_rate`` derivation real prompt content on a synthetic clock."""

    def __init__(self, plan, session_id: str, duration: float) -> None:
        self.plan = plan
        self.session_id = session_id
        self.duration = duration
        self.clock = 1.0
        self.rows: list[object] = []

    @property
    def next_batch_index(self) -> int:
        return len(self.rows)

    def execute_next(self):
        from cacheon.eval.oci_outer_session import BatchExecutionEvidence
        from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence

        index = len(self.rows)
        prompts = tuple(
            PromptEvidence(
                tuple(range(self.plan.max_new_tokens)),
                tuple(
                    tuple(
                        (0.0 - rank, rank)
                        for rank in range(self.plan.top_logprobs_num)
                    )
                    for _ in range(self.plan.max_new_tokens)
                ),
                self.plan.expected_prompt_tokens or 5,
            )
            for _ in self.plan.prompt_batches[index]
        )
        started = self.clock + 0.01
        self.clock = started + self.duration
        row = BatchExecutionEvidence(
            index,
            f"{index + 17:032x}",
            f"{index + 4097:032x}",
            started,
            self.clock,
            len(prompts) * self.plan.max_new_tokens,
            BatchEvidence(prompts),
        )
        self.rows.append(row)
        return row

    def finish(self):
        from cacheon.eval.oci_outer_session import SessionExecutionEvidence

        rows = tuple(self.rows)
        first_timed = rows[self.plan.warmup_count]
        return SessionExecutionEvidence(
            self.session_id,
            self.plan.launch_digest,
            self.plan.expected_preflight,
            1.0,
            rows,
            self.plan.warmup_count,
            self.plan.conditioning_count,
            rows[0].request_started_at,
            first_timed.response_completed_at,
            sum(row.token_numerator for row in rows[: self.plan.warmup_count + 1]),
            self.clock + 0.01,
        )


def _resident_execution(arm, session, physical_ids, receipt_label: str):
    from types import SimpleNamespace

    from cacheon.eval.oci_backend import EngineExecutionEvidence

    receipts = tuple(
        SimpleNamespace(
            completed_monotonic_s=float(index + 1),
            launch_id=receipt_label * 32,
            selected_physical_gpu_ids=physical_ids,
            sequence=index,
            started_monotonic_s=float(index),
        )
        for index in (1, 2, 3)
    )
    return EngineExecutionEvidence(
        "cacheon.oci-resident-engine-execution.v1",
        arm.launch.digest,
        SimpleNamespace(),
        "1" * 64,
        "2" * 64,
        arm.runtime_resource_policy_digest,
        SimpleNamespace(),
        "3" * 64,
        "4" * 64,
        (),
        receipts,  # type: ignore[arg-type]
        session,
    )


def _lifecycle(tmp_path: Path, *, top_logprobs_num: int = 1):
    """One valid singleton ResidentMarginalLifecycleEvidence with real prompt
    content, assembled through the production rate/grade/regrade math so the
    lifecycle survives its own __post_init__ and every later re-wrap."""

    from cacheon.eval.calibration import CalibrationContext, SpeedCalibration
    from cacheon.eval.crossover_runtime import (
        ResidentArmPlan,
        ResidentCrossoverEvidence,
        ResidentCrossoverPlan,
        ResidentMarginalLifecycleEvidence,
        _expected_lane_digest,
    )
    from cacheon.eval.resident_schedule import expanded_schedule, read_rate
    from cacheon.eval.oci_process import OCIQuiescenceReceipt
    from cacheon.eval.scoring import marginal_workload_digest
    from cacheon.eval.speed_verdict import speed_grade
    from tests.test_calibration import _manifest as calibration_manifest
    from tests.test_crossover_runtime import _resident_policy
    from tests.test_marginal_runtime import (
        _case as runtime_case,
        _local_binding as runtime_local_binding,
        _prepared as prepared_runtime,
    )

    case = runtime_case(tmp_path)
    case.session = replace(
        case.session,
        prompt_batches=(("warmup",), ("timed-1",), ("timed-2",), ("timed-3",)),
        max_new_tokens=10,
        top_logprobs_num=top_logprobs_num,
    )
    prepared = prepared_runtime(case)
    candidate = prepared.candidates[0]
    runtime_policy = _d("runtime-resource-policy")
    baseline_arm = ResidentArmPlan(
        prepared.baseline_launch,
        runtime_local_binding(
            case.baseline_tree,
            case.baseline_binding.launch_binding.native_build_spec,
            case.launch,
            case.preflight,
            physical_id="1",
        ).launch_binding,
        prepared.baseline_session_plan,
        _d("baseline namespace"),
        runtime_policy,
        _d("baseline device configuration"),
    )
    candidate_arm = ResidentArmPlan(
        candidate.launch,
        candidate.binding.launch_binding,
        candidate.session_plan,
        _d("candidate namespace"),
        runtime_policy,
        _d("candidate device configuration"),
    )
    policy = _resident_policy()
    plan = ResidentCrossoverPlan(
        case.arm.selected_delta_digest, baseline_arm, candidate_arm, policy
    )
    baseline_lane_digest = _expected_lane_digest(baseline_arm)
    candidate_lane_digest = _expected_lane_digest(candidate_arm)
    baseline_lane = _FixtureLane(
        expanded_schedule(baseline_arm.session_plan, 2), "b" * 32, 1.0
    )
    candidate_lane = _FixtureLane(
        expanded_schedule(candidate_arm.session_plan, 1), "c" * 32, 0.75
    )
    rate_b = read_rate(
        "B",
        baseline_lane_digest,
        baseline_lane,
        baseline_arm.session_plan,
    )
    rate_c = read_rate(
        "C",
        candidate_lane_digest,
        candidate_lane,
        candidate_arm.session_plan,
    )
    rate_b_prime = read_rate(
        "B_prime",
        baseline_lane_digest,
        baseline_lane,
        baseline_arm.session_plan,
    )
    final, decision = speed_grade(
        policy, [rate_b, rate_b_prime], [rate_c]
    )
    crossover = ResidentCrossoverEvidence(
        plan.digest,
        plan.selected_delta_digest,
        policy,
        marginal_workload_digest(baseline_arm.session_plan),
        baseline_lane_digest,
        candidate_lane_digest,
        _resident_execution(baseline_arm, baseline_lane.finish(), (1,), "e"),
        _resident_execution(candidate_arm, candidate_lane.finish(), (0,), "f"),
        OCIQuiescenceReceipt(
            "cacheon.oci-quiescence.v1", "lane-baseline", "a" * 32,
            baseline_arm.executor_namespace_digest, 1, 5.0, (), (), (),
        ),
        OCIQuiescenceReceipt(
            "cacheon.oci-quiescence.v1", "lane-candidate", "a" * 32,
            candidate_arm.executor_namespace_digest, 2, 6.0, (), (), (),
        ),
        (rate_b, rate_c, rate_b_prime),
        final,
        final,
        False,
        decision,
        "clear_" + decision.value.lower(),
        0.0,
        50.0,
    )
    lifecycle = ResidentMarginalLifecycleEvidence(prepared, plan, crossover)
    context = CalibrationContext(
        _d("reference-manifest"),
        case.launch.arena_digest,
        case.launch.runtime_digest,
        case.launch.base_engine_digest,
        case.launch.model_revision_digest,
        case.launch.model_manifest_digest,
        case.launch.model_content_digest,
        case.launch.hardware.digest,
        marginal_workload_digest(prepared.baseline_session_plan),
        _d("verification-policy"),
    )
    calibration = replace(
        calibration_manifest(), context=context, speed=SpeedCalibration("0.02", "2", "0.1")
    )
    return lifecycle, case.arm.selected_delta_digest, case, calibration, runtime_policy


def _with_candidate_batches(lifecycle, batches):
    """Re-wrap one lifecycle around corrupted candidate batch content. Content
    corruption leaves the timing spans intact, so the production regrade in
    every __post_init__ re-accepts the evidence while its digests move."""

    execution = lifecycle.crossover.candidate_execution
    session = replace(execution.session, batches=tuple(batches))
    crossover = replace(
        lifecycle.crossover,
        candidate_execution=replace(execution, session=session),
    )
    return replace(lifecycle, crossover=crossover)


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


def test_lifecycle_derives_prompt_pool_and_exact_selected_trajectories(tmp_path: Path):
    lifecycle, delta, _case, _calibration, _runtime_policy = _lifecycle(tmp_path)
    prompts = lifecycle_prompt_digests(lifecycle)
    assert len(prompts) == 4
    cohort = cohort_trajectory_digest(lifecycle)
    selected = selected_trajectory_digest(
        lifecycle, selected_delta_digest=delta, selected_prompt_digests=prompts[:2]
    )
    assert len({cohort, selected}) == 2

    session = lifecycle.candidates[0].execution.session
    batches = list(session.batches)
    # Corrupt a selected prompt occurrence. Sorted prompts[:2] excludes some
    # occurrence digests, so both digests must still move on any corruption.
    evidence = batches[0].evidence
    prompt = evidence.prompts[0]
    corrupted = replace(prompt, output_ids=(999,) + prompt.output_ids[1:])
    batches[0] = replace(
        batches[0], evidence=replace(evidence, prompts=(corrupted,))
    )
    changed = _with_candidate_batches(lifecycle, batches)
    assert cohort_trajectory_digest(changed) != cohort
    assert selected_trajectory_digest(
        changed, selected_delta_digest=delta, selected_prompt_digests=prompts[:2]
    ) != selected


def test_trajectory_topk_accepts_ties_without_relabeling_runtime_top_one():
    from cacheon.eval.qualification import _validated_topk_position

    left = ((-0.5, 19), (-0.5, 7), (-1.0, 3))
    right = ((-1.0, 3), (-0.5, 7), (-0.5, 19))
    assert _validated_topk_position(left) == [
        ["-0.5", 19], ["-0.5", 7], ["-1", 3]
    ]
    assert _validated_topk_position(((-0.25, 4), (-1.5, 8))) == [
        ["-0.25", 4], ["-1.5", 8]
    ]
    with pytest.raises(QualificationError, match="order"):
        _validated_topk_position(right)

    with pytest.raises(QualificationError, match="duplicate"):
        _validated_topk_position(((-0.5, 7), (-1.0, 7)))
    with pytest.raises(QualificationError, match="invalid"):
        _validated_topk_position(((float("nan"), 7),))


def test_trajectory_projection_rejects_subset_relabel_and_short_topk(tmp_path: Path):
    lifecycle, delta, _case, _calibration, _runtime_policy = _lifecycle(tmp_path)
    with pytest.raises(QualificationError, match="prompts differ"):
        selected_trajectory_digest(
            lifecycle,
            selected_delta_digest=delta,
            selected_prompt_digests=(_d("not-a-live-prompt"),),
        )
    batches = list(lifecycle.candidates[0].execution.session.batches)
    prompt = batches[0].evidence.prompts[0]
    batches[0] = replace(
        batches[0], evidence=replace(
            batches[0].evidence,
            prompts=(replace(prompt, top_logprobs=prompt.top_logprobs[:-1]),),
        ),
    )
    broken = _with_candidate_batches(lifecycle, batches)
    with pytest.raises(QualificationError, match="coverage"):
        cohort_trajectory_digest(broken)


def test_width_zero_trajectory_digests_seal_absence_and_match_raw_shape(tmp_path: Path):
    # Option B: a width-0 lifecycle retains one empty support row per token.
    # The cohort digest must seal that absence (2026-07-25 calibration
    # failure: _validated_topk_position rejected the empty rows the coverage
    # check had just required), and the projection digest must byte-match the
    # raw-artifact side's one-None-per-token rollout_topk shape
    # (raw_trajectory_projection_digest) or live raw-quality re-verification
    # fails after the measurement has already run.
    from cacheon.eval.reference_quality import retained_support_policy_digest

    lifecycle, delta, _case, _calibration, _runtime_policy = _lifecycle(
        tmp_path, top_logprobs_num=0
    )
    cohort = cohort_trajectory_digest(lifecycle)

    batches = list(lifecycle.candidates[0].execution.session.batches)
    evidence = batches[1].evidence
    prompt = evidence.prompts[0]
    corrupted = replace(prompt, output_ids=(999,) + prompt.output_ids[1:])
    batches[1] = replace(
        batches[1], evidence=replace(evidence, prompts=(corrupted,))
    )
    changed = _with_candidate_batches(lifecycle, batches)
    assert cohort_trajectory_digest(changed) != cohort

    prompts = lifecycle_prompt_digests(lifecycle)
    selected = tuple(sorted(prompts[:2]))
    projection = selected_trajectory_projection_digest(
        lifecycle, selected_delta_digest=delta, selected_prompt_digests=selected
    )
    rollout = {"output_ids": list(range(10)), "rollout_topk": [None] * 10}
    assert projection == canonical_digest(
        "cacheon.qualification.selected-trajectory-projection",
        {
            "support_policy_digest": retained_support_policy_digest(),
            "prompts": [
                {"prompt": digest, "rollouts": [rollout, rollout, rollout]}
                for digest in selected
            ],
        },
    )


def _quality_world(tmp_path, *, top_logprobs_num=1, topk_width=1, metric_names=None):
    """The shared world of the two quality-binding composition tests. Every
    parameter is fixture data (a width, a metric list); the structurally
    different per-token evidence loops stay inside the tests themselves."""

    from types import SimpleNamespace

    from cacheon.eval.calibration import CalibrationContext
    from cacheon.eval.qualification import _selected_prompt_texts, _trajectory_rows
    from cacheon.eval.reference_quality import retained_support_policy_digest

    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(
        tmp_path, top_logprobs_num=top_logprobs_num
    )
    reference = ReferenceManifest(
        *(_d(f"pristine:{index}") for index in range(3)),
        case.launch.runtime_digest, case.launch.base_engine_digest, case.launch.arena_digest,
        lifecycle.candidates[0].arm.candidate.catalog_digest,
        case.launch.controller_distribution_digest, case.launch.worker_distribution_digest,
        case.launch.model_revision_digest, case.launch.model_manifest_digest,
        case.launch.model_content_digest, case.launch.hardware.digest,
        calibration.context.workload_digest, _d("tokenizer"), _d("hidden-corpus"),
        _d("hidden-judge"), _d("entropy-source"),
    )
    calibration = replace(
        calibration,
        context=CalibrationContext(
            reference.measured_digest, reference.arena_digest, reference.runtime_digest,
            reference.base_engine_digest, reference.model_revision_digest,
            reference.model_manifest_digest, reference.model_content_digest,
            reference.logical_hardware_digest, reference.workload_digest,
            _d("verification-policy"),
        ),
        **(
            {}
            if metric_names is None
            else {
                "quality_metrics": tuple(
                    row
                    for row in calibration.quality_metrics
                    if row.name in metric_names
                )
            }
        ),
    )
    profile = QualificationProfile(
        reference, calibration.context.digest, calibration.digest,
        tuple(row.name for row in calibration.quality_metrics), "2", 10, topk_width, 2,
        retained_support_policy_digest(), _d("hidden-task-policy"), runtime_policy, True, 2,
    )
    commitment = SelectionCommitment.seal(
        source_plan_digest=lifecycle.source.digest, reference_manifest=reference,
        entropy_source_digest=declared_qualification_entropy_digest(
            reference.selection_policy_digest
        ),
        prompt_digests=lifecycle_prompt_digests(lifecycle), select_count=2,
        secret=b"s" * 32,
    )
    entropy = SelectionEntropyReceipt(
        commitment.entropy_source_digest, commitment.digest,
        _d("entropy-value"), _d("entropy-authority"),
    )
    selection = SelectionReceipt.reveal(
        commitment, secret=b"s" * 32, entropy=entropy,
        sealed_cohort_trajectory_digest=cohort_trajectory_digest(lifecycle),
    )
    _, trajectory_rows = _trajectory_rows(lifecycle)
    return SimpleNamespace(
        lifecycle=lifecycle, delta=delta, case=case, calibration=calibration,
        runtime_policy=runtime_policy, reference=reference,
        profile=profile, commitment=commitment, entropy=entropy, selection=selection,
        trajectories=dict(trajectory_rows),
        prompt_texts=_selected_prompt_texts(lifecycle),
    )


def _hidden_tasks(reference, profile, prompt_digest):
    from cacheon.eval.reference_quality import RawHiddenTaskResult

    return tuple(sorted(
        (RawHiddenTaskResult(
            canonical_digest("cacheon.qualification.hidden-task", {
                "corpus": reference.hidden_corpus_commitment,
                "judge": reference.hidden_judge_digest,
                "policy": profile.hidden_task_policy_digest,
                "prompt": prompt_digest,
                "index": index,
            }),
            True,
        ) for index in range(profile.hidden_tasks_per_prompt)),
        key=lambda row: row.task_digest,
    ))


def _pristine_t(world, request_prompts, evidence_prompts, *, width):
    from cacheon.eval.oci_backend import PristineReferenceExecutionEvidence
    from cacheon.eval.oci_reference_session import (
        ReferenceExchangeEvidence,
        ReferenceSessionEvidence,
    )
    from cacheon.eval.reference_protocol import (
        ReferenceEvidence,
        ReferenceRequest,
        encode_reference_evidence,
        request_sha256,
    )
    from cacheon.stack_identity import sha256_hex
    from tests.test_oci_reference_session import (
        _config as reference_config,
        _facts as reference_facts,
    )

    reference = world.reference
    request_plan = _d("reference-request-plan")
    request = ReferenceRequest(
        "1" * 32, reference.pristine_launch_digest, request_plan,
        "2" * 32, "3" * 32, 0, 10, width, tuple(request_prompts),
    )
    teacher_evidence = ReferenceEvidence(
        request.session_id, request.launch_digest, request.plan_digest,
        request_sha256(request), request.request_id, request.nonce, 0, 32_000,
        tuple(evidence_prompts),
    )
    reference_request_sha256 = request_sha256(request)
    exchange = ReferenceExchangeEvidence(
        0, request, reference_request_sha256,
        sha256_hex(encode_reference_evidence(teacher_evidence, request)),
        1.0, 2.0, teacher_evidence,
    )
    t_session = ReferenceSessionEvidence(
        "cacheon.pristine-reference-session.v1", request.session_id,
        reference.pristine_launch_digest, reference.digest,
        _d("reference-session-plan"), request_plan,
        reference_facts(reference, reference_config()), 0.5, (exchange,), 3.0,
    )
    baseline = world.lifecycle.crossover.baseline_execution
    reference_execution = PristineReferenceExecutionEvidence(
        "cacheon.oci-pristine-reference-execution.v1",
        reference.pristine_launch_digest,
        baseline.runtime_identity,
        baseline.runtime_preflight_receipt_sha256,
        baseline.arena_model_receipt_digest,
        baseline.resource_policy_digest,
        baseline.prebuild,
        baseline.native_publication_digest,
        baseline.runtime_argv_sha256,
        (),
        (baseline.device_receipts[0], baseline.device_receipts[-1]),
        t_session,
    )
    return t_session, reference_request_sha256, reference_execution


def _quality_binding(world, t_session, reference_request_sha256, *, width):
    from cacheon.eval.reference_quality import ReferenceQualityRawBinding

    lifecycle_digest = candidate_lifecycle_digest(
        world.lifecycle, selected_delta_digest=world.delta
    )
    identity_digest = qualification_identity_digest(
        world.profile,
        selection=world.selection,
        calibration=world.calibration,
        candidate_lifecycle=lifecycle_digest,
        t_session=t_session,
        t_request_sha256=reference_request_sha256,
        selected_delta_digest=world.delta,
    )
    selected = world.selection.selected_prompt_digests
    binding = ReferenceQualityRawBinding(
        identity_digest, world.reference.measured_digest, world.calibration.digest,
        world.selection.digest, lifecycle_digest,
        selected_trajectory_digest(
            world.lifecycle, selected_delta_digest=world.delta,
            selected_prompt_digests=selected,
        ),
        selected_trajectory_projection_digest(
            world.lifecycle, selected_delta_digest=world.delta,
            selected_prompt_digests=selected,
        ), selected,
        t_session.digest, reference_request_sha256, world.profile.support_policy_digest,
        derived_hidden_task_plan_digest(world.profile, selected),
        world.profile.nll_tail_threshold, 10, width, 2,
    )
    return lifecycle_digest, identity_digest, binding


def _validate(world, raw_artifact, reference_execution, reference_request_sha256, *, profile=None):
    return validate_quality_binding(
        profile or world.profile, raw_artifact, world.lifecycle,
        selected_delta_digest=world.delta,
        commitment=world.commitment, entropy=world.entropy, selection=world.selection,
        calibration=world.calibration,
        reference_execution=reference_execution,
        reference_request_sha256=reference_request_sha256,
    )


def test_quality_binding_projects_exact_lifecycle_coverage(tmp_path: Path):
    from cacheon.eval.reference_protocol import (
        ReferencePromptEvidence,
        ReferencePromptInput,
        ReferenceRoleEvidence,
        ReferenceRoleInput,
        ReferenceTokenEvidence,
    )
    from cacheon.eval.reference_quality import (
        RawPromptQualityEvidence,
        RawRolloutEvidence,
        RawTokenEvidence,
        ReferenceQualityRawArtifact,
        distribution_from_f32_logprobs,
        target_nll_from_f32,
    )

    world = _quality_world(tmp_path)
    request_prompts = []
    evidence_prompts = []
    raw_prompts = []
    for prompt_digest in world.selection.selected_prompt_digests:
        frames = [world.trajectories[prompt_digest][index] for index in (0, 1, -1)]
        role_inputs, role_evidence, raw_rollouts = [], [], []
        tasks = _hidden_tasks(world.reference, world.profile, prompt_digest)
        for frame in frames:
            inputs, teacher, tokens = [], [], []
            for position, (output_id, topk) in enumerate(zip(
                frame["output_ids"], frame["top_logprobs"], strict=True
            )):
                ordered = sorted(topk, key=lambda row: row[1])
                support = tuple(row[1] for row in ordered)
                logprobs = tuple(float(row[0]) for row in ordered)
                distribution = distribution_from_f32_logprobs(
                    support, logprobs, true_argmax_token_id=topk[0][1]
                )
                inputs.append(support)
                teacher.append(ReferenceTokenEvidence(-0.25, topk[0][1], logprobs))
                tokens.append(RawTokenEvidence(
                    position, output_id, target_nll_from_f32(-0.25),
                    distribution, distribution,
                ))
            role_inputs.append(ReferenceRoleInput(tuple(frame["output_ids"]), tuple(inputs)))
            role_evidence.append(ReferenceRoleEvidence(tuple(teacher)))
            raw_rollouts.append(RawRolloutEvidence(tuple(tokens), tasks))
        request_prompts.append(ReferencePromptInput(
            prompt_digest, world.prompt_texts[prompt_digest], tuple(role_inputs)
        ))
        evidence_prompts.append(ReferencePromptEvidence(
            prompt_digest, 3, _d("prompt-tokens:" + prompt_digest), tuple(role_evidence)
        ))
        raw_prompts.append(RawPromptQualityEvidence(
            prompt_digest, *raw_rollouts
        ))
    t_session, reference_request_sha256, reference_execution = _pristine_t(
        world, request_prompts, evidence_prompts, width=1
    )
    lifecycle_digest, identity_digest, binding = _quality_binding(
        world, t_session, reference_request_sha256, width=1
    )
    assert identity_digest == canonical_digest(
        "cacheon.qualification.candidate-identity",
        {
            "calibration_digest": world.calibration.digest,
            "candidate_lifecycle_digest": lifecycle_digest,
            "profile_digest": world.profile.digest,
            "selected_delta_digest": world.delta,
            "selection_digest": world.selection.digest,
            "t_session_digest": t_session.digest,
            "t_request_sha256": reference_request_sha256,
        },
    )
    raw_artifact = ReferenceQualityRawArtifact(binding, tuple(raw_prompts))
    assert _validate(
        world, raw_artifact, reference_execution, reference_request_sha256
    ) == raw_artifact
    with pytest.raises(QualificationError, match="frozen workload"):
        _validate(
            world, raw_artifact, reference_execution, reference_request_sha256,
            profile=replace(
                world.profile, hidden_task_policy_digest=_d("other-hidden-policy")
            ),
        )
    with pytest.raises(QualificationError, match="frozen workload"):
        _validate(
            world, raw_artifact, reference_execution, reference_request_sha256,
            profile=replace(world.profile, tokens_per_prompt=1),
        )
    forged = replace(
        raw_artifact.prompts[0].candidate.tokens[0],
        target_nll=target_nll_from_f32(-0.01),
    )
    candidate_rollout = replace(
        raw_artifact.prompts[0].candidate,
        tokens=(forged, *raw_artifact.prompts[0].candidate.tokens[1:]),
    )
    forged_prompt = replace(raw_artifact.prompts[0], candidate=candidate_rollout)
    forged_artifact = replace(
        raw_artifact,
        prompts=(forged_prompt, *raw_artifact.prompts[1:]),
    )
    with pytest.raises(QualificationError, match="differs from pristine T"):
        _validate(world, forged_artifact, reference_execution, reference_request_sha256)


def test_quality_binding_validates_width_zero_nll_only_end_to_end(tmp_path: Path):
    # Option B, full composition at width 0: real lifecycle, real pristine-T
    # exchange, real raw artifact, real validate_quality_binding — the path
    # the 2026-07-25 r3 calibration died on (_validate_teacher_source derived
    # a distribution from empty support).  Everything the live validator
    # recomputes must accept sealed absence end to end.
    from cacheon.eval.reference_protocol import (
        ReferencePromptEvidence,
        ReferencePromptInput,
        ReferenceRoleEvidence,
        ReferenceRoleInput,
        ReferenceTokenEvidence,
    )
    from cacheon.eval.reference_quality import (
        RawPromptQualityEvidence,
        RawRolloutEvidence,
        RawTokenEvidence,
        ReferenceQualityRawArtifact,
        target_nll_from_f32,
    )

    world = _quality_world(
        tmp_path, top_logprobs_num=0, topk_width=0,
        metric_names={"mean_nll", "worst_nll", "task_score"},
    )
    request_prompts = []
    evidence_prompts = []
    raw_prompts = []
    for prompt_digest in world.selection.selected_prompt_digests:
        frames = [world.trajectories[prompt_digest][index] for index in (0, 1, -1)]
        role_inputs, role_evidence, raw_rollouts = [], [], []
        tasks = _hidden_tasks(world.reference, world.profile, prompt_digest)
        for frame in frames:
            inputs, teacher, tokens = [], [], []
            for position, (output_id, topk) in enumerate(zip(
                frame["output_ids"], frame["top_logprobs"], strict=True
            )):
                assert topk == []
                inputs.append(())
                teacher.append(ReferenceTokenEvidence(-0.25, output_id, ()))
                tokens.append(RawTokenEvidence(
                    position, output_id, target_nll_from_f32(-0.25), None, None,
                ))
            role_inputs.append(ReferenceRoleInput(tuple(frame["output_ids"]), tuple(inputs)))
            role_evidence.append(ReferenceRoleEvidence(tuple(teacher)))
            raw_rollouts.append(RawRolloutEvidence(tuple(tokens), tasks))
        request_prompts.append(ReferencePromptInput(
            prompt_digest, world.prompt_texts[prompt_digest], tuple(role_inputs)
        ))
        evidence_prompts.append(ReferencePromptEvidence(
            prompt_digest, 3, _d("prompt-tokens:" + prompt_digest), tuple(role_evidence)
        ))
        raw_prompts.append(RawPromptQualityEvidence(
            prompt_digest, *raw_rollouts
        ))
    t_session, reference_request_sha256, reference_execution = _pristine_t(
        world, request_prompts, evidence_prompts, width=0
    )
    _lifecycle_digest, _identity_digest, binding = _quality_binding(
        world, t_session, reference_request_sha256, width=0
    )
    raw_artifact = ReferenceQualityRawArtifact(binding, tuple(raw_prompts))
    assert _validate(
        world, raw_artifact, reference_execution, reference_request_sha256
    ) == raw_artifact

    # Continue down the exact runner path (qualification_runner :3819-3831):
    # publish the canonical raw bytes, reopen (which derives the NLL-only
    # summaries), and score against the NLL-only calibration.  Every layer
    # must accept sealed absence without touching distribution machinery.
    from cacheon.eval.reference_quality import (
        RAW_QUALITY_DOMAIN,
        RAW_QUALITY_SCHEMA,
        reopen_reference_quality_evidence,
        score_reference_quality,
    )

    evidence_root = tmp_path / "quality-evidence"
    raw_ref = publish_evidence(
        evidence_root,
        canonical_json_bytes(raw_artifact.to_dict()),
        domain=RAW_QUALITY_DOMAIN,
        media_type="application/json",
        schema=RAW_QUALITY_SCHEMA,
    )
    derived = reopen_reference_quality_evidence(
        evidence_root, raw_ref, expected_binding=raw_artifact.binding
    )
    for prompt in derived.prompts:
        for rollout in (prompt.baseline, prompt.candidate, prompt.stock_control):
            assert rollout.rollout_kl is None
    verdict = score_reference_quality(
        derived, calibration=world.calibration, expected_context=world.calibration.context
    )
    assert verdict.decision in {"PASS", "FAIL", "NO_DECISION"}
    assert verdict.calibration_digest == world.calibration.digest


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


def test_resident_speed_witness_relabel_forgery_is_internally_undetectable(tmp_path):
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

    lifecycle, _delta, _case, _calibration, _runtime_policy = _lifecycle(tmp_path)
    witness = ResidentSpeedWitness.from_evidence(lifecycle.crossover, lifecycle.plan)
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
    )
    forged = ResidentSpeedWitness.from_dict(
        {
            **honest,
            "selected_delta_digest": relabel,
            "evidence_digest": forged_digest,
        }
    )
    assert forged.selected_delta_digest == relabel
    assert forged.rates == witness.rates

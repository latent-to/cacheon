from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval.qualification import (
    GRAPH_EVIDENCE_DOMAIN,
    GRAPH_EVIDENCE_MEDIA_TYPE,
    GRAPH_EVIDENCE_SCHEMA,
    declared_qualification_entropy_digest,
    GraphMemberEvidence,
    GraphShapeEvidence,
    GraphVariantEvidence,
    GraphVariantRequirement,
    GraphVerificationBinding,
    GraphVerificationEvidenceRef,
    GraphVerificationGrade,
    GraphVerificationMemberBinding,
    GraphVerificationRawEvidence,
    GraphVerificationRequirement,
    QualificationProfile,
    QualificationDecision,
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
    reopen_graph_verification,
    regrade_graph_verification,
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


def _member(slot: str) -> GraphVerificationMemberBinding:
    return GraphVerificationMemberBinding(
        slot,
        _d(slot + ":target"),
        _d(slot + ":contract"),
        slot + ".verify.v1",
    )


def _requirement(*, atomic: bool = True) -> GraphVerificationRequirement:
    slots = (
        (
            "collective.ar_residual_rmsnorm",
            "collective.moe_finalize_ar_rmsnorm",
        )
        if atomic
        else ("activation.silu_and_mul",)
    )
    members = tuple(_member(slot) for slot in slots)
    binding = GraphVerificationBinding(
        marginal_arm_digest=_d("arm"),
        candidate_launch_digest=_d("launch"),
        contribution_ref_digest=_d("contribution"),
        selected_delta_digest=_d("delta"),
        target_id="collective.moe_epilogue.v1" if atomic else slots[0],
        target_spec_digest=_d("selected-target"),
        catalog_digest=_d("catalog"),
        members=members,
        verification_policy_digest=_d("verification-policy"),
    )
    variants = []
    for slot in slots:
        shapes = tuple(sorted((_d(slot + ":shape-a"), _d(slot + ":shape-b"))))
        variants.append(GraphVariantRequirement(slot, "default", shapes, True, shapes))
    return GraphVerificationRequirement(binding, tuple(variants), 3)


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
        _expanded,
        _expected_lane_digest,
        _rate,
    )
    from cacheon.eval.oci_process import OCIQuiescenceReceipt
    from cacheon.eval.scoring import marginal_workload_digest
    from cacheon.eval.speed_verdict import speed_grade
    from tests.test_calibration import _manifest as calibration_manifest
    from tests.test_crossover_runtime import _policy_v8
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
    policy = _policy_v8()
    plan = ResidentCrossoverPlan(
        case.arm.selected_delta_digest, baseline_arm, candidate_arm, policy
    )
    baseline_lane_digest = _expected_lane_digest(baseline_arm)
    candidate_lane_digest = _expected_lane_digest(candidate_arm)
    baseline_lane = _FixtureLane(
        _expanded(baseline_arm.session_plan, 2), "b" * 32, 1.0
    )
    candidate_lane = _FixtureLane(
        _expanded(candidate_arm.session_plan, 1), "c" * 32, 0.75
    )
    rate_b = _rate(
        "B",
        baseline_lane_digest,
        baseline_lane,
        baseline_arm.session_plan,
        with_windows=True,
    )
    rate_c = _rate(
        "C",
        candidate_lane_digest,
        candidate_lane,
        candidate_arm.session_plan,
        with_windows=True,
    )
    rate_b_prime = _rate(
        "B_prime",
        baseline_lane_digest,
        baseline_lane,
        baseline_arm.session_plan,
        with_windows=True,
    )
    final, decision = speed_grade(
        policy, [rate_b, rate_b_prime], [rate_c], concluding=True
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


def _shape(
    descriptor: str,
    failure: str = "none",
    *,
    replays: int | None = None,
) -> GraphShapeEvidence:
    states = {
        "none": (True, True, True, True, 3),
        "not_applicable": (False, False, False, False, 0),
        "eager": (True, False, True, False, 0),
        "capture": (True, True, True, False, 0),
        "replay": (True, True, True, False, 1),
        "graph_not_required": (True, True, False, False, 0),
    }
    applicable, eager, required, passed, count = states[failure]
    return GraphShapeEvidence(
        descriptor,
        applicable,
        eager,
        required,
        count if replays is None else replays,
        passed,
        failure,
    )


def _raw(
    requirement: GraphVerificationRequirement,
    *,
    failure: tuple[str, str] | None = None,
    replay_override: tuple[str, int] | None = None,
    domain_incomplete: str | None = None,
    not_applicable: str | None = None,
) -> GraphVerificationRawEvidence:
    by_member: dict[str, list[GraphVariantEvidence]] = {}
    for expected in requirement.variants:
        shapes = []
        for descriptor in expected.shape_descriptor_digests:
            kind = "not_applicable" if expected.slot_id == not_applicable else "none"
            if failure == (expected.slot_id, descriptor):
                kind = "capture"
            count = (
                replay_override[1]
                if replay_override is not None
                and replay_override[0] == descriptor
                else None
            )
            shapes.append(_shape(descriptor, kind, replays=count))
        by_member.setdefault(expected.slot_id, []).append(
            GraphVariantEvidence(
                expected.slot_id,
                expected.variant_id,
                expected.slot_id != not_applicable,
                expected.slot_id != domain_incomplete,
                tuple(shapes),
            )
        )
    return GraphVerificationRawEvidence(
        requirement.digest,
        tuple(
            GraphMemberEvidence(slot, tuple(variants))
            for slot, variants in sorted(by_member.items())
        ),
    )


def _ref(
    requirement: GraphVerificationRequirement,
    raw: GraphVerificationRawEvidence,
) -> GraphVerificationEvidenceRef:
    return GraphVerificationEvidenceRef(requirement.binding, requirement.digest, raw.digest)


def _grade(
    requirement: GraphVerificationRequirement,
    raw: GraphVerificationRawEvidence,
) -> GraphVerificationGrade:
    return regrade_graph_verification(requirement, _ref(requirement, raw), raw)


def test_exact_atomic_graph_evidence_round_trips_and_passes_only_the_veto():
    requirement = _requirement()
    raw = _raw(requirement)
    reference = _ref(requirement, raw)
    grade = regrade_graph_verification(requirement, reference, raw)

    assert grade.decision is QualificationDecision.PASS
    assert grade.veto_passed
    assert grade.reason == "graph_verification_pass"
    assert GraphVerificationRequirement.from_dict(requirement.to_dict()) == requirement
    assert GraphVerificationRawEvidence.from_dict(raw.to_dict()) == raw
    assert GraphVerificationEvidenceRef.from_dict(reference.to_dict()) == reference
    assert GraphVerificationGrade.from_dict(grade.to_dict()) == grade
    assert len({requirement.digest, raw.digest, reference.digest, grade.digest}) == 4
    assert "score" not in grade.to_dict() and "crown" not in grade.to_dict()


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


@pytest.mark.parametrize("which", ["both", "ref", "raw"])
def test_absent_or_unrun_graph_evidence_is_no_decision(which):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    reference = _ref(requirement, raw)
    grade = regrade_graph_verification(
        requirement,
        None if which in {"both", "raw"} else reference,
        None if which in {"both", "ref"} else raw,
    )
    assert grade.decision is QualificationDecision.NO_DECISION
    assert grade.reason == "graph_evidence_missing"


def test_atomic_missing_member_is_incomplete_not_a_partial_pass():
    requirement = _requirement()
    complete = _raw(requirement)
    missing = replace(complete, members=complete.members[:1])
    grade = _grade(requirement, missing)
    assert grade.decision is QualificationDecision.NO_DECISION
    assert grade.reason == "graph_member_coverage_incomplete"


def test_missing_variant_or_shape_is_no_decision():
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    member = raw.members[0]
    variant = member.variants[0]

    wrong_variant = replace(variant, variant_id="other")
    variant_gap = replace(raw, members=(replace(member, variants=(wrong_variant,)),))
    assert _grade(requirement, variant_gap).reason == "graph_variant_coverage_incomplete"

    shape_gap = replace(variant, shapes=variant.shapes[:1])
    shape_raw = replace(raw, members=(replace(member, variants=(shape_gap,)),))
    assert _grade(requirement, shape_raw).reason == "graph_shape_coverage_incomplete"


@pytest.mark.parametrize("failure", ["eager", "capture", "replay", "graph_not_required"])
def test_bound_negative_graph_proof_is_fail(failure):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    member, variant = raw.members[0], raw.members[0].variants[0]
    broken = _shape(variant.shapes[0].descriptor_digest, failure)
    raw = replace(
        raw,
        members=(
            replace(
                member,
                variants=(replace(variant, shapes=(broken, variant.shapes[1])),),
            ),
        ),
    )
    grade = _grade(requirement, raw)
    assert grade.decision is QualificationDecision.FAIL
    assert failure in grade.reason


def test_eager_correct_capture_wrong_is_not_rescued_by_other_shapes():
    requirement = _requirement()
    slot = requirement.binding.members[0].slot_id
    descriptor = requirement.variants[0].shape_descriptor_digests[0]
    grade = _grade(requirement, _raw(requirement, failure=(slot, descriptor)))
    assert grade.decision is QualificationDecision.FAIL
    assert grade.reason == "graph_capture_failed"


def test_domain_incomplete_and_entirely_off_domain_members_fail():
    requirement = _requirement()
    slot = requirement.binding.members[0].slot_id
    domain = _grade(requirement, _raw(requirement, domain_incomplete=slot))
    outside = _grade(requirement, _raw(requirement, not_applicable=slot))
    assert (domain.decision, domain.reason) == (
        QualificationDecision.FAIL,
        "graph_domain_coverage_failed",
    )
    assert (outside.decision, outside.reason) == (
        QualificationDecision.FAIL,
        "graph_applicability_failed",
    )


def test_required_shape_cannot_be_relabelled_not_applicable():
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    member, variant = raw.members[0], raw.members[0].variants[0]
    hidden = _shape(variant.shapes[0].descriptor_digest, "not_applicable")
    raw = replace(
        raw,
        members=(
            replace(
                member,
                variants=(replace(variant, shapes=(hidden, variant.shapes[1])),),
            ),
        ),
    )
    grade = _grade(requirement, raw)
    assert (grade.decision, grade.reason) == (
        QualificationDecision.FAIL,
        "graph_applicability_failed",
    )


def test_success_with_wrong_replay_count_is_unrun_policy_no_decision():
    requirement = _requirement(atomic=False)
    descriptor = requirement.variants[0].shape_descriptor_digests[0]
    grade = _grade(requirement, _raw(requirement, replay_override=(descriptor, 2)))
    assert grade.decision is QualificationDecision.NO_DECISION
    assert grade.reason == "graph_replay_count_mismatch"


def test_swapped_launch_or_tampered_raw_evidence_is_no_decision():
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    reference = _ref(requirement, raw)
    swapped_binding = replace(
        requirement.binding, candidate_launch_digest=_d("another-launch")
    )
    swapped = replace(requirement, binding=swapped_binding)
    grade = regrade_graph_verification(swapped, reference, raw)
    assert (grade.decision, grade.reason) == (
        QualificationDecision.NO_DECISION,
        "graph_identity_mismatch",
    )

    tampered_ref = replace(reference, raw_evidence_digest=_d("forged-raw"))
    grade = regrade_graph_verification(requirement, tampered_ref, raw)
    assert (grade.decision, grade.reason) == (
        QualificationDecision.NO_DECISION,
        "graph_evidence_tampered",
    )


def test_strict_canonical_records_reject_duplicates_reorder_floats_and_audit_fields():
    requirement = _requirement()
    raw = _raw(requirement)

    with pytest.raises(QualificationError, match="ordered"):
        replace(requirement.binding, members=tuple(reversed(requirement.binding.members)))
    with pytest.raises(QualificationError, match="duplicates"):
        replace(
            requirement.binding,
            members=(requirement.binding.members[0], requirement.binding.members[0]),
        )
    with pytest.raises(QualificationError, match="ordered"):
        replace(requirement, variants=tuple(reversed(requirement.variants)))
    with pytest.raises(QualificationError, match="integer"):
        GraphVerificationRequirement.from_dict(
            {**requirement.to_dict(), "expected_graph_replays": 3.0}
        )

    untrusted = raw.to_dict()
    untrusted["audit_passed"] = True
    with pytest.raises(QualificationError, match="fields"):
        GraphVerificationRawEvidence.from_dict(untrusted)
    untrusted = raw.members[0].variants[0].shapes[0].to_dict()
    untrusted["fully_verified"] = True
    with pytest.raises(QualificationError, match="fields"):
        GraphShapeEvidence.from_dict(untrusted)


def test_inconsistent_shape_claims_and_reordered_raw_rows_reject_at_parse_time():
    requirement = _requirement()
    raw = _raw(requirement)
    descriptor = requirement.variants[0].shape_descriptor_digests[0]
    with pytest.raises(QualificationError, match="inconsistent"):
        GraphShapeEvidence(descriptor, True, True, True, 3, False, "none")
    with pytest.raises(QualificationError, match="ordered"):
        replace(raw, members=tuple(reversed(raw.members)))


def test_reference_profile_and_precommitted_selection_round_trip():
    reference = _reference()
    requirement = _requirement(atomic=False)
    profile = QualificationProfile(
        reference,
        _d("calibration-context"),
        _d("calibration"),
        requirement.digest,
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
        "reference calibration_context_digest calibration_digest graph_requirement_digest "
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


def _publish_graph_bytes(root: Path, payload: bytes):
    return publish_evidence(
        root,
        payload,
        domain=GRAPH_EVIDENCE_DOMAIN,
        media_type=GRAPH_EVIDENCE_MEDIA_TYPE,
        schema=GRAPH_EVIDENCE_SCHEMA,
    )


def test_controller_store_graph_artifact_reopens_and_regrades(tmp_path: Path):
    requirement = _requirement()
    raw = _raw(requirement)
    evidence_ref = _ref(requirement, raw)
    artifact = _publish_graph_bytes(tmp_path / "evidence", canonical_json_bytes(raw.to_dict()))

    grade = reopen_graph_verification(
        tmp_path / "evidence", artifact, requirement, evidence_ref
    )
    assert grade == regrade_graph_verification(requirement, evidence_ref, raw)
    assert grade.decision is QualificationDecision.PASS


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("domain", "qualification.other"),
        ("media_type", "application/json"),
        ("schema", "cacheon.qualification.other.v1"),
    ),
)
def test_graph_artifact_requires_exact_type_and_metadata(tmp_path: Path, field, value):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    artifact = _publish_graph_bytes(tmp_path / "evidence", canonical_json_bytes(raw.to_dict()))
    with pytest.raises(QualificationError, match="artifact reference"):
        reopen_graph_verification(
            tmp_path / "evidence",
            replace(artifact, **{field: value}),
            requirement,
            _ref(requirement, raw),
        )
    with pytest.raises(QualificationError, match="artifact reference"):
        reopen_graph_verification(
            tmp_path / "evidence", artifact.to_dict(), requirement, _ref(requirement, raw)
        )


def test_graph_artifact_rejects_noncanonical_or_malformed_json(tmp_path: Path):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    canonical = canonical_json_bytes(raw.to_dict())
    duplicate = (
        b'{"members":[],"members":[],"policy_version":"graph-verification-veto.v1",'
        b'"requirement_digest":"' + requirement.digest.encode() +
        b'","schema_version":1}'
    )
    attacks = (
        duplicate,
        b'{"value":NaN}',
        canonical + b"{}",
        b" " + canonical,
        json.dumps(raw.to_dict()).encode(),
        b"\xff",
    )
    for index, payload in enumerate(attacks):
        root = tmp_path / f"evidence-{index}"
        artifact = _publish_graph_bytes(root, payload)
        with pytest.raises(QualificationError, match="JSON"):
            reopen_graph_verification(root, artifact, requirement, _ref(requirement, raw))


def test_graph_artifact_cannot_substitute_parsed_grade_or_other_raw(tmp_path: Path):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    evidence_ref = _ref(requirement, raw)

    grade_bytes = canonical_json_bytes(_grade(requirement, raw).to_dict())
    grade_root = tmp_path / "grade"
    with pytest.raises(QualificationError, match="fields"):
        reopen_graph_verification(
            grade_root,
            _publish_graph_bytes(grade_root, grade_bytes),
            requirement,
            evidence_ref,
        )

    altered = _raw(
        requirement,
        failure=(
            requirement.variants[0].slot_id,
            requirement.variants[0].shape_descriptor_digests[0],
        ),
    )
    altered_root = tmp_path / "altered"
    result = reopen_graph_verification(
        altered_root,
        _publish_graph_bytes(altered_root, canonical_json_bytes(altered.to_dict())),
        requirement,
        evidence_ref,
    )
    assert (result.decision, result.reason) == (
        QualificationDecision.NO_DECISION,
        "graph_evidence_tampered",
    )


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


def test_quality_binding_projects_exact_lifecycle_coverage(tmp_path: Path):
    from cacheon.eval.qualification import _selected_prompt_texts, _trajectory_rows
    from cacheon.eval.calibration import CalibrationContext
    from cacheon.eval.oci_reference_session import (
        ReferenceExchangeEvidence,
        ReferenceSessionEvidence,
    )
    from cacheon.eval.oci_backend import PristineReferenceExecutionEvidence
    from cacheon.eval.reference_protocol import (
        ReferenceEvidence,
        ReferencePromptEvidence,
        ReferencePromptInput,
        ReferenceRequest,
        ReferenceRoleEvidence,
        ReferenceRoleInput,
        ReferenceTokenEvidence,
        encode_reference_evidence,
        request_sha256,
    )
    from cacheon.eval.reference_quality import (
        RawHiddenTaskResult,
        RawPromptQualityEvidence,
        RawRolloutEvidence,
        RawTokenEvidence,
        ReferenceQualityRawArtifact,
        ReferenceQualityRawBinding,
        distribution_from_f32_logprobs,
        retained_support_policy_digest,
        target_nll_from_f32,
    )
    from cacheon.stack_identity import canonical_digest, sha256_hex
    from tests.test_oci_reference_session import (
        _config as reference_config,
        _facts as reference_facts,
    )

    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
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
    arm = lifecycle.candidates[0].arm
    candidate = lifecycle.candidates[0].candidate
    requirement = _requirement()
    requirement = replace(
        requirement,
        binding=replace(
            requirement.binding,
            marginal_arm_digest=arm.digest,
            candidate_launch_digest=candidate.launch.digest,
            contribution_ref_digest=arm.transition.replacement.digest,
            selected_delta_digest=delta,
            target_id=arm.transition.target_id,
            target_spec_digest=arm.transition.target_spec_digest,
            catalog_digest=arm.candidate.catalog_digest,
        ),
    )
    calibration = replace(
        calibration,
        context=CalibrationContext(
            reference.measured_digest,
            reference.arena_digest,
            reference.runtime_digest,
            reference.base_engine_digest,
            reference.model_revision_digest,
            reference.model_manifest_digest,
            reference.model_content_digest,
            reference.logical_hardware_digest,
            reference.workload_digest,
            requirement.binding.verification_policy_digest,
        ),
    )
    profile = QualificationProfile(
        reference, calibration.context.digest, calibration.digest, requirement.digest,
        tuple(row.name for row in calibration.quality_metrics), "2", 10, 1, 2,
        retained_support_policy_digest(), _d("hidden-task-policy"), runtime_policy, True, 2,
    )
    prompts = lifecycle_prompt_digests(lifecycle)
    commitment = SelectionCommitment.seal(
        source_plan_digest=lifecycle.source.digest, reference_manifest=reference,
        entropy_source_digest=declared_qualification_entropy_digest(
            reference.selection_policy_digest
        ),
        prompt_digests=prompts, select_count=2, secret=b"s" * 32,
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
    trajectories = dict(trajectory_rows)
    prompt_texts = _selected_prompt_texts(lifecycle)
    request_prompts = []
    evidence_prompts = []
    raw_prompts = []
    for prompt_digest in selection.selected_prompt_digests:
        frames = [trajectories[prompt_digest][index] for index in (0, 1, -1)]
        role_inputs, role_evidence, raw_rollouts = [], [], []
        tasks = tuple(sorted(
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
            prompt_digest, prompt_texts[prompt_digest], tuple(role_inputs)
        ))
        evidence_prompts.append(ReferencePromptEvidence(
            prompt_digest, 3, _d("prompt-tokens:" + prompt_digest), tuple(role_evidence)
        ))
        raw_prompts.append(RawPromptQualityEvidence(
            prompt_digest, *raw_rollouts
        ))
    config = reference_config()
    request_plan = _d("reference-request-plan")
    request = ReferenceRequest(
        "1" * 32, reference.pristine_launch_digest, request_plan,
        "2" * 32, "3" * 32, 0, 10, 1, tuple(request_prompts),
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
        reference_facts(reference, config), 0.5, (exchange,), 3.0,
    )
    baseline = lifecycle.crossover.baseline_execution
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
    lifecycle_digest = candidate_lifecycle_digest(
        lifecycle, selected_delta_digest=delta
    )
    identity_digest = qualification_identity_digest(
        profile,
        graph_requirement=requirement,
        selection=selection,
        calibration=calibration,
        candidate_lifecycle=lifecycle_digest,
        t_session=t_session,
        t_request_sha256=reference_request_sha256,
        selected_delta_digest=delta,
    )
    assert identity_digest == canonical_digest(
        "cacheon.qualification.candidate-identity",
        {
            "calibration_digest": calibration.digest,
            "candidate_lifecycle_digest": lifecycle_digest,
            "graph_requirement_digest": requirement.digest,
            "profile_digest": profile.digest,
            "selected_delta_digest": delta,
            "selection_digest": selection.digest,
            "t_session_digest": t_session.digest,
            "t_request_sha256": reference_request_sha256,
        },
    )
    binding = ReferenceQualityRawBinding(
        identity_digest, reference.measured_digest, calibration.digest, selection.digest,
        lifecycle_digest, selected_trajectory_digest(
            lifecycle, selected_delta_digest=delta,
            selected_prompt_digests=selection.selected_prompt_digests,
        ),
        selected_trajectory_projection_digest(
            lifecycle, selected_delta_digest=delta,
            selected_prompt_digests=selection.selected_prompt_digests,
        ), selection.selected_prompt_digests,
        t_session.digest, reference_request_sha256, profile.support_policy_digest,
        derived_hidden_task_plan_digest(profile, selection.selected_prompt_digests),
        profile.nll_tail_threshold, 10, 1, 2,
    )
    raw_artifact = ReferenceQualityRawArtifact(binding, tuple(raw_prompts))
    assert validate_quality_binding(
        profile, raw_artifact, lifecycle, selected_delta_digest=delta,
        commitment=commitment, entropy=entropy, selection=selection,
        calibration=calibration, graph_requirement=requirement,
        reference_execution=reference_execution,
        reference_request_sha256=reference_request_sha256,
    ) == raw_artifact
    with pytest.raises(QualificationError, match="frozen workload"):
        validate_quality_binding(
            replace(profile, hidden_task_policy_digest=_d("other-hidden-policy")),
            raw_artifact, lifecycle, selected_delta_digest=delta,
            commitment=commitment, entropy=entropy, selection=selection,
            calibration=calibration, graph_requirement=requirement,
            reference_execution=reference_execution,
            reference_request_sha256=reference_request_sha256,
        )
    with pytest.raises(QualificationError, match="frozen workload"):
        validate_quality_binding(
            replace(profile, tokens_per_prompt=1), raw_artifact, lifecycle,
            selected_delta_digest=delta, commitment=commitment,
            entropy=entropy, selection=selection, calibration=calibration,
            graph_requirement=requirement,
            reference_execution=reference_execution,
            reference_request_sha256=reference_request_sha256,
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
        validate_quality_binding(
            profile, forged_artifact, lifecycle, selected_delta_digest=delta,
            commitment=commitment, entropy=entropy, selection=selection,
            calibration=calibration, graph_requirement=requirement,
            reference_execution=reference_execution,
            reference_request_sha256=reference_request_sha256,
        )


def test_quality_binding_validates_width_zero_nll_only_end_to_end(tmp_path: Path):
    # Option B, full composition at width 0: real lifecycle, real pristine-T
    # exchange, real raw artifact, real validate_quality_binding — the path
    # the 2026-07-25 r3 calibration died on (_validate_teacher_source derived
    # a distribution from empty support).  Everything the live validator
    # recomputes must accept sealed absence end to end.
    from cacheon.eval.qualification import _selected_prompt_texts, _trajectory_rows
    from cacheon.eval.calibration import CalibrationContext
    from cacheon.eval.oci_reference_session import (
        ReferenceExchangeEvidence,
        ReferenceSessionEvidence,
    )
    from cacheon.eval.oci_backend import PristineReferenceExecutionEvidence
    from cacheon.eval.reference_protocol import (
        ReferenceEvidence,
        ReferencePromptEvidence,
        ReferencePromptInput,
        ReferenceRequest,
        ReferenceRoleEvidence,
        ReferenceRoleInput,
        ReferenceTokenEvidence,
        encode_reference_evidence,
        request_sha256,
    )
    from cacheon.eval.reference_quality import (
        RawHiddenTaskResult,
        RawPromptQualityEvidence,
        RawRolloutEvidence,
        RawTokenEvidence,
        ReferenceQualityRawArtifact,
        ReferenceQualityRawBinding,
        retained_support_policy_digest,
        target_nll_from_f32,
    )
    from cacheon.stack_identity import sha256_hex
    from tests.test_oci_reference_session import (
        _config as reference_config,
        _facts as reference_facts,
    )

    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(
        tmp_path, top_logprobs_num=0
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
    arm = lifecycle.candidates[0].arm
    candidate = lifecycle.candidates[0].candidate
    requirement = _requirement()
    requirement = replace(
        requirement,
        binding=replace(
            requirement.binding,
            marginal_arm_digest=arm.digest,
            candidate_launch_digest=candidate.launch.digest,
            contribution_ref_digest=arm.transition.replacement.digest,
            selected_delta_digest=delta,
            target_id=arm.transition.target_id,
            target_spec_digest=arm.transition.target_spec_digest,
            catalog_digest=arm.candidate.catalog_digest,
        ),
    )
    calibration = replace(
        calibration,
        context=CalibrationContext(
            reference.measured_digest, reference.arena_digest, reference.runtime_digest,
            reference.base_engine_digest, reference.model_revision_digest,
            reference.model_manifest_digest, reference.model_content_digest,
            reference.logical_hardware_digest, reference.workload_digest,
            requirement.binding.verification_policy_digest,
        ),
        quality_metrics=tuple(
            row for row in calibration.quality_metrics
            if row.name in {"mean_nll", "worst_nll", "task_score"}
        ),
    )
    profile = QualificationProfile(
        reference, calibration.context.digest, calibration.digest, requirement.digest,
        tuple(row.name for row in calibration.quality_metrics), "2", 10, 0, 2,
        retained_support_policy_digest(), _d("hidden-task-policy"), runtime_policy, True, 2,
    )
    prompts = lifecycle_prompt_digests(lifecycle)
    commitment = SelectionCommitment.seal(
        source_plan_digest=lifecycle.source.digest, reference_manifest=reference,
        entropy_source_digest=declared_qualification_entropy_digest(
            reference.selection_policy_digest
        ),
        prompt_digests=prompts, select_count=2, secret=b"s" * 32,
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
    trajectories = dict(trajectory_rows)
    prompt_texts = _selected_prompt_texts(lifecycle)
    request_prompts = []
    evidence_prompts = []
    raw_prompts = []
    for prompt_digest in selection.selected_prompt_digests:
        frames = [trajectories[prompt_digest][index] for index in (0, 1, -1)]
        role_inputs, role_evidence, raw_rollouts = [], [], []
        tasks = tuple(sorted(
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
            prompt_digest, prompt_texts[prompt_digest], tuple(role_inputs)
        ))
        evidence_prompts.append(ReferencePromptEvidence(
            prompt_digest, 3, _d("prompt-tokens:" + prompt_digest), tuple(role_evidence)
        ))
        raw_prompts.append(RawPromptQualityEvidence(
            prompt_digest, *raw_rollouts
        ))
    config = reference_config()
    request_plan = _d("reference-request-plan")
    request = ReferenceRequest(
        "1" * 32, reference.pristine_launch_digest, request_plan,
        "2" * 32, "3" * 32, 0, 10, 0, tuple(request_prompts),
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
        reference_facts(reference, config), 0.5, (exchange,), 3.0,
    )
    baseline = lifecycle.crossover.baseline_execution
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
    lifecycle_digest = candidate_lifecycle_digest(
        lifecycle, selected_delta_digest=delta
    )
    identity_digest = qualification_identity_digest(
        profile,
        graph_requirement=requirement,
        selection=selection,
        calibration=calibration,
        candidate_lifecycle=lifecycle_digest,
        t_session=t_session,
        t_request_sha256=reference_request_sha256,
        selected_delta_digest=delta,
    )
    binding = ReferenceQualityRawBinding(
        identity_digest, reference.measured_digest, calibration.digest, selection.digest,
        lifecycle_digest, selected_trajectory_digest(
            lifecycle, selected_delta_digest=delta,
            selected_prompt_digests=selection.selected_prompt_digests,
        ),
        selected_trajectory_projection_digest(
            lifecycle, selected_delta_digest=delta,
            selected_prompt_digests=selection.selected_prompt_digests,
        ), selection.selected_prompt_digests,
        t_session.digest, reference_request_sha256, profile.support_policy_digest,
        derived_hidden_task_plan_digest(profile, selection.selected_prompt_digests),
        profile.nll_tail_threshold, 10, 0, 2,
    )
    raw_artifact = ReferenceQualityRawArtifact(binding, tuple(raw_prompts))
    assert validate_quality_binding(
        profile, raw_artifact, lifecycle, selected_delta_digest=delta,
        commitment=commitment, entropy=entropy, selection=selection,
        calibration=calibration, graph_requirement=requirement,
        reference_execution=reference_execution,
        reference_request_sha256=reference_request_sha256,
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
        derived, calibration=calibration, expected_context=calibration.context
    )
    assert verdict.decision in {"PASS", "FAIL", "NO_DECISION"}
    assert verdict.calibration_digest == calibration.digest


def test_teacher_nll_only_profile_admits_width_zero_and_refuses_kl_metrics():
    # Option B (2026-07-25): topk_width 0 selects teacher-NLL-only quality.
    # The mode is digest-bound, and a profile cannot simultaneously declare
    # a distribution metric it retains no evidence for.
    reference = _reference()
    requirement = _requirement(atomic=False)

    def profile(width: int, metrics: tuple[str, ...]) -> QualificationProfile:
        return QualificationProfile(
            reference,
            _d("calibration-context"),
            _d("calibration"),
            requirement.digest,
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

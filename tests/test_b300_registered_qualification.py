"""CPU-only contracts for the ordinary B300 qualification authority."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import cacheon.eval.b300_registered_qualification as registered
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCandidateBinding,
    ArenaQualificationRequest,
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.engine_tree import inspect_contribution
from cacheon.eval.b300_qualification_deployment import B300QualificationCohort
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationDeploymentError,
)
from cacheon.eval.calibration import (
    CalibrationContext,
    CalibrationEvidenceSet,
    CalibrationThresholdPolicy,
    MetricCalibration,
    SpeedCalibration,
    derive_calibration_manifest,
    publish_calibration_evidence,
)
from cacheon.eval.crossover_runtime import ResidentArmPlan, ResidentSpeedPolicy
from cacheon.eval.engine_launch import TrustedLaunchBinding
from cacheon.eval.marginal_runtime import MaterializedArmBinding
from cacheon.eval.oci_backend import expected_runtime_preflight
from cacheon.eval.qualification import (
    GraphVariantRequirement,
    ReferenceManifest,
)
from cacheon.eval.qualification_intake import (
    GraphShapeObservation,
    GraphVariantObservation,
    QualificationReservation,
)
from cacheon.eval.qualification_runner import SpeedStageDisposition
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.target_catalog import (
    MOE_EPILOGUE_ATOMIC_TARGET,
    SINGLETON_TARGET_IDS,
    default_target_catalog,
)
from tests.test_calibration import _observations
from tests.test_marginal_runtime import _case, _local_binding, _native


TARGET = "attention.msa_prefill_block_score"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _EmptyResolver:
    def resolve_proposal(self, artifact_digest: str) -> Path:
        raise KeyError(artifact_digest)

    def resolve_integrated(self, source_tree_digest: str) -> Path:
        raise KeyError(source_tree_digest)


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _candidate_source(root: Path) -> Path:
    kernels = root / "kernels"
    metadata = root / "metadata"
    kernels.mkdir(parents=True)
    metadata.mkdir()
    (kernels / "msa_prefill_block_score.py").write_text(
        "def msa_prefill_block_score(q, index_k, out=None):\n"
        "    return q if out is None else out.copy_(q)\n"
    )
    (metadata / "msa_prefill.json").write_text(
        '{"op":"attention.msa_prefill_block_score"}\n'
    )
    (root / "rebuild.json").write_text('{"steps":[]}\n')
    (root / "manifest.toml").write_text(
        "\n".join(
            (
                'bundle_id = "ordinary-msa-prefill"',
                'abi_version = "cacheon-op-abi-v0"',
                "[competition]",
                f'target = "{TARGET}"',
                'mode = "slot"',
                "[[ops]]",
                f'slot = "{TARGET}"',
                'source = "kernels/msa_prefill_block_score.py"',
                'entry = "msa_prefill_block_score"',
                'dtypes = ["bfloat16", "float16"]',
                'architectures = ["sm100", "sm103"]',
                'metadata = "metadata/msa_prefill.json"',
            )
        )
        + "\n"
    )
    for path in sorted(root.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)
    return root


def _candidate(tmp_path: Path) -> ArenaCandidateBinding:
    source = _candidate_source(tmp_path / "candidate-source")
    publications = _private_directory(tmp_path / "publications")
    publication = publish_worker_bundle(
        source,
        publications,
        content_hash(source),
    )
    catalog = default_target_catalog()
    inspected = inspect_contribution(publication.root, catalog=catalog)
    reservation = QualificationReservation(
        _h("reservation"),
        publication.digest,
        TARGET,
        inspected.selected_delta_digest,
        0,
        "miner-hotkey",
        8_775_104,
        155,
        0,
        catalog.require(TARGET).members,
    )
    return ArenaCandidateBinding(reservation, publication, 1)


def _cohort(candidate: ArenaCandidateBinding, policy_digest: str) -> B300QualificationCohort:
    service = _h("ordinary-b300-service")
    receipt = ArenaScreenReceipt(
        service,
        candidate.digest,
        candidate.screen_attempt,
        tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(stage), 1)
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )
    return B300QualificationCohort(
        ArenaQualificationRequest(
            service,
            policy_digest,
            (candidate,),
            (receipt,),
        ),
        "primary",
    )


def _focused_graph_facts(
    candidate: ArenaCandidateBinding,
    _prepared,
) -> registered.B300FocusedGraphFacts:
    target_id = candidate.reservation.target_id
    descriptors = tuple(
        sorted(_h(f"{target_id}-shape-{index}") for index in range(3))
    )
    requirement = GraphVariantRequirement(
        target_id,
        "default",
        descriptors,
        True,
        descriptors,
    )
    observed = GraphVariantObservation(
        target_id,
        "default",
        True,
        True,
        tuple(
            GraphShapeObservation(
                descriptor,
                True,
                True,
                True,
                3,
                True,
            )
            for descriptor in descriptors
        ),
    )
    return registered.B300FocusedGraphFacts(3, (requirement,), (observed,))


@dataclass
class _Harness:
    factory: registered.B300RegisteredQualificationFactory
    candidate: ArenaCandidateBinding
    cohort: B300QualificationCohort
    policy: registered.B300RegisteredQualificationPolicy
    inputs: registered.B300RegisteredQualificationInputs


def _harness(tmp_path: Path) -> _Harness:
    case = _case(tmp_path / "runtime")
    catalog = case.catalog
    evidence_root = _private_directory(tmp_path / "evidence")
    materialization_root = _private_directory(tmp_path / "materialized")
    verification_policy = _h("ordinary-focused-graph-policy")
    hidden_policy = _h("ordinary-hidden-task-policy")
    reference = ReferenceManifest.from_pristine(
        case.incumbent,
        case.launch,
        case.baseline_binding,
        workload_digest=marginal_workload_digest(case.session),
        tokenizer_digest=_h("tokenizer"),
        hidden_corpus_commitment=_h("hidden-corpus"),
        hidden_judge_digest=_h("hidden-judge"),
        selection_policy_digest=_h("selection-policy"),
    )
    calibration_context = CalibrationContext(
        reference.digest,
        reference.arena_digest,
        reference.runtime_digest,
        reference.base_engine_digest,
        reference.model_revision_digest,
        reference.model_manifest_digest,
        reference.model_content_digest,
        reference.logical_hardware_digest,
        reference.workload_digest,
        verification_policy,
        reference.controller_distribution_digest,
    )
    threshold = CalibrationThresholdPolicy(
        calibration_context,
        "teacher-familywise-v1",
        "frozen",
        SpeedCalibration("0.005", "2", "0.01"),
        (
            MetricCalibration("mean_nll", "lower", "0.02", "0.01"),
            MetricCalibration("task_score", "higher", "0.03", "0.02", "0.8"),
        ),
        "2.576",
    )
    observations = _observations()
    calibration = derive_calibration_manifest(threshold, observations)
    calibration_ref = publish_calibration_evidence(
        evidence_root,
        CalibrationEvidenceSet.create(threshold, observations),
    )
    policy = registered.B300RegisteredQualificationPolicy.seal(
        catalog,
        verification_policy_digest=verification_policy,
        nll_tail_threshold="20",
        tokens_per_prompt=case.session.max_new_tokens,
        topk_width=case.session.top_logprobs_num,
        hidden_tasks_per_prompt=1,
        support_policy_digest=_h("retained-support-policy"),
        hidden_task_policy_digest=hidden_policy,
        hidden_tasks_required=True,
        select_count=2,
        audit_minimum_calls=2,
    )
    baseline_policy = _h("resident-baseline-device-policy")
    baseline_hardware = replace(
        case.launch.hardware,
        device_policy_digest=baseline_policy,
    )
    baseline_launch = replace(
        case.launch,
        hardware=baseline_hardware,
        resource_policy_digest=_h("resident-baseline-launch-policy"),
    )
    baseline_physical = replace(
        case.baseline_binding.launch_binding.physical_hardware,
        physical_gpu_ids=("1",),
        device_policy_digest=baseline_policy,
    )
    baseline_binding = replace(
        case.baseline_binding.launch_binding,
        physical_hardware=baseline_physical,
    )
    baseline_plan = replace(
        case.session,
        launch_digest=baseline_launch.digest,
        expected_preflight=expected_runtime_preflight(baseline_launch, case.preflight),
    )
    baseline_runtime_policy = _h("resident-baseline-runtime-policy")
    resident_baseline = ResidentArmPlan(
        baseline_launch,
        baseline_binding,
        baseline_plan,
        _h("resident-baseline-namespace"),
        baseline_runtime_policy,
        _h("resident-baseline-device-configuration"),
    )
    candidate_runtime_policy = _h("resident-candidate-runtime-policy")
    resident_speed = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=60,
        max_qualification_seconds=600,
        calibration=calibration,
        context=calibration_context,
        version=3,
        min_windows=3,
        max_window_scatter=0.05,
        max_conditioning_slowdown=1.5,
    )

    def bind_candidate(tree) -> TrustedLaunchBinding:
        native = _native(tree.tree_digest, case.preflight)
        return _local_binding(
            tree,
            native,
            case.launch,
            case.preflight,
        ).launch_binding

    inputs = registered.B300RegisteredQualificationInputs(
        catalog=catalog,
        policy=policy,
        expected_context=case.context,
        incumbent_stack=case.incumbent,
        incumbent_binding=case.baseline_binding,
        incumbent_launch=case.launch,
        baseline_session_plan=case.session,
        model_mount=case.mount,
        materialization_root=materialization_root,
        source_resolver_digest=_h("incumbent-source-resolver"),
        source_resolver=_EmptyResolver(),
        candidate_binding_builder_digest=_h("candidate-binding-builder"),
        candidate_binding_builder=bind_candidate,
        graph_facts_builder_digest=_h("ordinary-focused-graph-builder"),
        graph_facts_builder=_focused_graph_facts,
        evidence_root=evidence_root,
        reference_manifest=reference,
        calibration_threshold_policy=threshold,
        calibration_manifest=calibration,
        calibration_context=calibration_context,
        calibration_artifact_ref=calibration_ref,
        pristine_stack=case.incumbent,
        pristine_binding=case.baseline_binding,
        pristine_launch=case.launch,
        pristine_session_plan=case.session,
        resident_baseline_arm=resident_baseline,
        resident_speed_policy=resident_speed,
        candidate_executor_namespace_digest=_h("resident-candidate-namespace"),
        candidate_runtime_resource_policy_digest=candidate_runtime_policy,
        candidate_device_configuration_digest=_h(
            "resident-candidate-device-configuration"
        ),
    )
    factory = registered.build_b300_registered_qualification_factory(inputs)
    candidate = _candidate(tmp_path / "publication")
    return _Harness(
        factory,
        candidate,
        _cohort(candidate, policy.digest),
        policy,
        inputs,
    )


def test_registry_exactly_covers_all_eleven_ordinary_targets_without_fe_identity(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)

    assert registered.ORDINARY_B300_TARGET_IDS == tuple(
        sorted(SINGLETON_TARGET_IDS)
    )
    assert tuple(row.target_id for row in harness.factory.profiles) == (
        registered.ORDINARY_B300_TARGET_IDS
    )
    assert harness.factory.components.profiles == harness.factory.profiles
    assert (
        harness.factory.components.builder_source_digest
        == harness.inputs.builder_source_digest
    )
    assert all("m4" not in row.target_id for row in harness.factory.profiles)
    assert all("campaign" not in row.target_id for row in harness.factory.profiles)
    assert all(
        row.target_spec_digest
        == harness.inputs.catalog.target_spec_digest(row.target_id)
        for row in harness.factory.profiles
    )


def test_concrete_prefill_blockscore_plan_is_registered_resident_v3_and_repeatable(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    secret = b"ordinary prefill blockscore selection"[:32]

    first = harness.factory.plan_builder(harness.cohort, secret)
    second = harness.factory.plan_builder(harness.cohort, secret)

    authority = first.candidates[0]
    graph = authority.graph_requirement
    contract = harness.inputs.catalog.require(TARGET).contract_ref
    assert contract is not None
    assert first == second
    assert first.prepared.candidates[0].arm.transition.target_id == TARGET
    assert (
        first.prepared.candidates[0].arm.selected_delta_digest
        == harness.candidate.reservation.selected_delta_digest
    )
    assert first.pristine_stack.entries == {}
    assert first.pristine_stack is harness.inputs.pristine_stack
    assert first.speed_evidence_policy.version == 3
    assert first.speed_stage_disposition is SpeedStageDisposition.TERMINAL
    assert first.resident_speed_plan is not None
    assert first.resident_audit_plan is not None
    assert first.resident_speed_plan.policy.version == 3
    assert first.resident_speed_plan.selected_delta_digest == (
        harness.candidate.reservation.selected_delta_digest
    )
    assert not (
        set(first.resident_speed_plan.baseline.binding.physical_hardware.physical_gpu_ids)
        & set(first.resident_speed_plan.candidate.binding.physical_hardware.physical_gpu_ids)
    )
    assert first.audit_policies[0].expected_slots == (TARGET,)
    assert graph.binding.target_id == TARGET
    assert graph.binding.target_spec_digest == (
        harness.inputs.catalog.target_spec_digest(TARGET)
    )
    assert graph.binding.members[0].contract_digest == (
        harness.inputs.catalog.contract_digest(TARGET)
    )
    assert graph.binding.members[0].verification_profile_id == (
        contract.verification_profile_id
    )
    assert authority.profile.reference is harness.inputs.reference_manifest
    assert authority.profile.calibration_digest == harness.inputs.calibration_manifest.digest
    assert authority.profile.runtime_resource_policy_digest == (
        harness.inputs.candidate_runtime_resource_policy_digest
    )
    assert first.evidence_root == harness.inputs.evidence_root


def test_registry_rejects_unsupported_and_stale_target_authority(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)

    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="unsupported",
    ):
        harness.factory.profile_for(MOE_EPILOGUE_ATOMIC_TARGET)

    rows = list(harness.policy.target_spec_digests)
    rows[0] = (rows[0][0], _h("stale-target-spec"))
    stale = replace(harness.policy, target_spec_digests=tuple(rows))
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="stale",
    ):
        stale.require_catalog(default_target_catalog())


def test_graph_facts_cannot_relabel_another_registered_target(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)

    def wrong_target(_candidate, _prepared):
        descriptor = _h("wrong-target-shape")
        requirement = GraphVariantRequirement(
            "attention.decode",
            "default",
            (descriptor,),
            True,
            (descriptor,),
        )
        observation = GraphVariantObservation(
            "attention.decode",
            "default",
            True,
            True,
            (
                GraphShapeObservation(
                    descriptor,
                    True,
                    True,
                    True,
                    2,
                    True,
                ),
            ),
        )
        return registered.B300FocusedGraphFacts(
            2,
            (requirement,),
            (observation,),
        )

    inputs = replace(
        harness.inputs,
        graph_facts_builder_digest=_h("wrong-target-graph-builder"),
        graph_facts_builder=wrong_target,
    )
    factory = registered.build_b300_registered_qualification_factory(inputs)
    with pytest.raises(
        B300QualificationDeploymentError,
        match="registered profile authority failed",
    ) as caught:
        factory.plan_builder(harness.cohort, b"x" * 32)
    assert isinstance(caught.value.__cause__, registered.B300RegisteredQualificationError)
    assert "another target" in str(caught.value.__cause__)


def test_blocker_inventory_names_only_missing_commissioning_authorities() -> None:
    assert tuple(row.blocker_id for row in registered.PRODUCTION_AUTHORITY_BLOCKERS) == (
        "ordinary-focused-graph-facts",
        "ordinary-runtime-binding",
        "typed-frozen-reference-calibration",
    )
    assert all(
        row.donor_coordinate.startswith("experiments/minimax_m3/")
        for row in registered.PRODUCTION_AUTHORITY_BLOCKERS
    )


def test_audit_role_pins_minimum_cost_shortest_prompt_selection(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    secret = b"ordinary prefill blockscore selection"[:32]
    value = harness.factory.plan_builder(harness.cohort, secret)

    session = value.prepared.candidates[0].session_plan
    prompts = tuple(
        prompt for batch in session.prompt_batches for prompt in batch
    )
    expected = min(
        prompts,
        key=lambda prompt: (
            len(prompt),
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        ),
    )
    audit = value.resident_audit_plan

    # Declared policy: the audit role is a minimum-cost slot-call integrity
    # check, never a semantic or shape-coverage instrument; that coverage is
    # owned by pristine T. Changing this selection is a reviewed policy
    # decision.
    assert audit.prompt_batches == tuple(
        (expected,) for _ in range(harness.policy.audit_minimum_calls + 1)
    )
    assert all(len(expected) <= len(prompt) for prompt in prompts)
    assert audit.warmup_count == 1
    assert audit.conditioning_count == 1
    assert audit.max_new_tokens == harness.policy.audit_max_new_tokens

    repeat = harness.factory.plan_builder(harness.cohort, secret)
    assert repeat.resident_audit_plan.prompt_batches == audit.prompt_batches

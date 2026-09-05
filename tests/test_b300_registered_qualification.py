"""CPU-only contracts for the registered B300 qualification authority."""

from __future__ import annotations

import hashlib
import inspect
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import cacheon.eval.b300_registered_qualification as registered
import cacheon.eval.b300_registered_qualification_inputs as inputs_module
import cacheon.eval.b300_qualification_deployment as qualification_deployment
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
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
    B300QualificationGraphEvidenceStoreError,
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
from cacheon.eval.oci_backend import expected_runtime_preflight
from cacheon.eval.qualification import GraphVariantRequirement, ReferenceManifest
from cacheon.eval.qualification_intake import (
    GraphShapeObservation,
    GraphVariantObservation,
    QualificationReservation,
)
from cacheon.eval.qualification_runner import SpeedStageDisposition
from tests.support.b300 import (
    GLM53_REGISTERED_TARGET_IDS,
    M3_REGISTERED_TARGET_IDS,
)
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.target_catalog import SINGLETON_TARGET_IDS, default_target_catalog
from tests.test_calibration import _observations
from tests.test_marginal_runtime import FUSED, _case, _local_binding, _native


TARGET = "norm.rmsnorm"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _EmptyResolver:
    def resolve_proposal(self, artifact_digest: str) -> Path:
        raise KeyError(artifact_digest)


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _candidate_source(root: Path) -> Path:
    kernels = root / "kernels"
    metadata = root / "metadata"
    kernels.mkdir(parents=True)
    metadata.mkdir()
    (kernels / "rmsnorm_stub.py").write_text(
        "def rmsnorm_stub(q, index_k, out=None):\n"
        "    return q if out is None else out.copy_(q)\n"
    )
    (metadata / "rmsnorm_stub.json").write_text(
        '{"op":"norm.rmsnorm"}\n'
    )
    (root / "rebuild.json").write_text('{"steps":[]}\n')
    (root / "manifest.toml").write_text(
        "\n".join(
            (
                'bundle_id = "ordinary-rmsnorm-stub"',
                'abi_version = "cacheon-op-abi-v0"',
                "[competition]",
                f'target = "{TARGET}"',
                'mode = "slot"',
                "[[ops]]",
                f'slot = "{TARGET}"',
                'source = "kernels/rmsnorm_stub.py"',
                'entry = "rmsnorm_stub"',
                'dtypes = ["bfloat16", "float16"]',
                'architectures = ["sm100", "sm103"]',
                'metadata = "metadata/rmsnorm_stub.json"',
            )
        )
        + "\n"
    )
    for path in sorted(root.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)
    return root


def _candidate(
    tmp_path: Path,
    source_fixture: Path | None = None,
) -> ArenaCandidateBinding:
    if source_fixture is None:
        source = _candidate_source(tmp_path / "candidate-source")
    else:
        source = tmp_path / "candidate-source"
        shutil.copytree(source_fixture, source)
        for path in sorted(source.rglob("*")):
            path.chmod(0o700 if path.is_dir() else 0o600)
        source.chmod(0o700)
    publications = _private_directory(tmp_path / "publications")
    publication = publish_worker_bundle(
        source,
        publications,
        content_hash(source),
    )
    catalog = default_target_catalog()
    inspected = inspect_contribution(publication.root, catalog=catalog)
    target_id = inspected.target_id
    reservation = QualificationReservation(
        _h("reservation"),
        publication.digest,
        target_id,
        inspected.selected_delta_digest,
        0,
        "miner-hotkey",
        8_775_104,
        155,
        0,
        catalog.require(target_id).members,
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


def _graph_facts_for_members(
    members: tuple[str, ...],
) -> registered.B300FocusedGraphFacts:
    requirements = []
    observations = []
    for member_id in members:
        descriptors = tuple(
            sorted(_h(f"{member_id}-shape-{index}") for index in range(3))
        )
        requirements.append(
            GraphVariantRequirement(
                member_id,
                "default",
                descriptors,
                True,
                descriptors,
            )
        )
        observations.append(
            GraphVariantObservation(
                member_id,
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
        )
    return registered.B300FocusedGraphFacts(
        3, tuple(requirements), tuple(observations)
    )


def _focused_graph_facts(
    candidate: ArenaCandidateBinding,
    _prepared,
    model_profile_key: str,
) -> registered.B300FocusedGraphFacts:
    assert model_profile_key in {"GLM-5.3-NVFP4", "MiniMax-M3"}
    return _graph_facts_for_members(candidate.reservation.target_members)


@dataclass
class _Harness:
    factory: registered.B300RegisteredQualificationFactory
    candidate: ArenaCandidateBinding
    cohort: B300QualificationCohort
    policy: registered.B300RegisteredQualificationPolicy
    inputs: registered.B300RegisteredQualificationInputs


def _harness(
    tmp_path: Path,
    source_fixture: Path | None = None,
    *,
    evidence_root: Path | None = None,
) -> _Harness:
    case = _case(tmp_path / "runtime")
    catalog = case.catalog
    glm = source_fixture == FUSED
    registered_target_ids = (
        GLM53_REGISTERED_TARGET_IDS
        if glm
        else M3_REGISTERED_TARGET_IDS
    )
    model_profile_key = (
        "GLM-5.3-NVFP4" if glm else "MiniMax-M3"
    )
    evidence_root = (
        _private_directory(tmp_path / "evidence")
        if evidence_root is None
        else evidence_root
    )
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
        reference.measured_digest,
        reference.arena_digest,
        reference.runtime_digest,
        reference.base_engine_digest,
        reference.model_revision_digest,
        reference.model_manifest_digest,
        reference.model_content_digest,
        reference.logical_hardware_digest,
        reference.workload_digest,
        verification_policy,
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
        registered_target_ids=registered_target_ids,
        model_profile_key=model_profile_key,
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
        seal=inputs_module._COMMISSION_SEAL,
    )
    factory = registered.build_b300_registered_qualification_factory(inputs)
    candidate = _candidate(tmp_path / "publication", source_fixture)
    return _Harness(
        factory,
        candidate,
        _cohort(candidate, policy.digest),
        policy,
        inputs,
    )


def test_registry_exactly_covers_the_pinned_registered_targets_without_fe_identity(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)

    # The B300 arena's registered set is PINNED arena data: it excludes catalog
    # rows that belong to other arenas (the GLM fat MoE slot) and must not grow
    # when the cross-arena catalog does.
    expected = tuple(
        sorted(
            (
                *(
                        target
                        for target in SINGLETON_TARGET_IDS
                        if target not in {
                            "collective.all_gather_into_tensor",
                            "collective.reduce_scatter_tensor",
                            "linear.dense",
                            "moe.fused_routed_experts",
                        "norm.fused_add_rmsnorm",
                    }
                ),
            )
        )
    )
    snapshot_ids = tuple(
        row["target_id"]
        for row in harness.inputs.catalog.snapshot()["targets"]
    )
    assert M3_REGISTERED_TARGET_IDS == expected
    assert set(M3_REGISTERED_TARGET_IDS) <= set(snapshot_ids)
    assert len(M3_REGISTERED_TARGET_IDS) == 6
    projection = registered.registered_b300_member_contract_projection(
        harness.inputs.catalog, M3_REGISTERED_TARGET_IDS
    )
    assert tuple(row.target_id for row in harness.factory.profiles) == (
        tuple(row.target_id for row in projection)
    )
    assert "moe.fused_routed_experts" not in expected
    assert "norm.rmsnorm" in expected
    assert "collective.dp_attention_exchange.v1" not in expected
    assert "collective.moe_finalize_ar_rmsnorm" not in expected
    glm_projection = registered.registered_b300_member_contract_projection(
        harness.inputs.catalog, GLM53_REGISTERED_TARGET_IDS
    )
    assert tuple(row.target_id for row in glm_projection) == GLM53_REGISTERED_TARGET_IDS
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
    factory_source = inspect.getsource(
        registered.B300RegisteredQualificationFactory
    )
    target_id_source = inspect.getsource(
        qualification_deployment.registered_b300_target_ids
    )
    assert TARGET not in factory_source
    assert "SINGLETON_TARGET_IDS" not in target_id_source


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
    assert first.resident_audit_plan.plan.engine_config.disable_cuda_graph
    assert not first.resident_audit_plan.charged_plan.engine_config.disable_cuda_graph
    assert first.resident_audit_plan.launch.digest != (
        first.prepared.candidates[0].launch.digest
    )
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


@pytest.mark.parametrize("version", (5, 6))
def test_registered_plan_accepts_commissioned_resident_policy(
    tmp_path: Path, version: int
) -> None:
    harness = _harness(tmp_path)
    current = harness.inputs.resident_speed_policy
    commissioned = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=current.max_stage_seconds,
        max_qualification_seconds=current.max_qualification_seconds,
        calibration=harness.inputs.calibration_manifest,
        context=harness.inputs.calibration_context,
        version=version,
        min_windows=current.min_windows,
        max_window_scatter=current.max_window_scatter,
        max_conditioning_slowdown=current.max_conditioning_slowdown,
    )
    inputs = replace(harness.inputs, resident_speed_policy=commissioned)
    value = registered.build_b300_registered_qualification_factory(
        inputs
    ).plan_builder(harness.cohort, b"v" * 32)

    assert value.resident_speed_plan is not None
    assert value.resident_speed_plan.policy.version == version


def test_pair_native_routing_requires_swap_reachability_and_same_target() -> None:
    """v7 measures both arms by injection over stock-booted engines.

    That comparison is exact only at genesis or when the candidate replaces
    the incumbent's own registered target and the incumbent itself is a
    single injectable bundle; everything else boots real trees on v8.
    """

    bundle = registered.SealedIncumbentBundle(
        "norm.rmsnorm",
        _h("crowned"),
        ("norm.rmsnorm",),
    )
    route = registered.resident_pair_native
    assert route(
        swappable=True, genesis=True, incumbent_bundle=None,
        candidate_target_id="any.target",
    )
    assert route(
        swappable=True, genesis=False, incumbent_bundle=bundle,
        candidate_target_id="norm.rmsnorm",
    )
    # Non-swappable candidates always boot.
    assert not route(
        swappable=False, genesis=True, incumbent_bundle=None,
        candidate_target_id="any.target",
    )
    # A different-target candidate composed over stock would omit the
    # incumbent's win from its own arm.
    assert not route(
        swappable=True, genesis=False, incumbent_bundle=bundle,
        candidate_target_id="moe.fused_experts_reduce",
    )
    # An incumbent one swap cannot realize routes everyone to v8.
    assert not route(
        swappable=True, genesis=False, incumbent_bundle=None,
        candidate_target_id="norm.rmsnorm",
    )


def test_sealed_incumbent_bundle_binds_to_the_declared_stack_entry(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    foreign = registered.SealedIncumbentBundle(
        "norm.rmsnorm",
        _h("not-in-the-stack"),
        ("norm.rmsnorm",),
    )
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="differs from the incumbent stack entry",
    ):
        replace(harness.inputs, incumbent_bundle=foreign)

    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="sorted distinct slots",
    ):
        registered.SealedIncumbentBundle(
            "norm.rmsnorm",
            _h("crowned"),
            ("b.slot", "a.slot"),
        )
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="incumbent bundle digest",
    ):
        registered.SealedIncumbentBundle(
            "norm.rmsnorm", "nothex", ("a.slot",)
        )


def test_native_candidate_is_planned_on_the_two_process_schedule(
    tmp_path: Path,
) -> None:
    """A bundle that cannot be hot-swapped gets the always-bookend schedule.

    A CUDA kernel has to be compiled and linked into the engine that runs it,
    so it cannot be swapped into a live resident lane -- it is measured by the
    two-process crossover, whose schedule reads B-prime unconditionally so the
    quality gate has a stock-drift control to harvest. The sealed commission
    version serves everything swappable; this routing is what a native bundle
    needs in order to receive a speed verdict at all.
    """

    native = _candidate_source(tmp_path / "native-source")
    (native / "kernels" / "native.cu").write_text(
        "extern \"C\" __global__ void noop() {}\n"
    )
    manifest = (native / "manifest.toml").read_text()
    (native / "manifest.toml").write_text(
        manifest.rstrip("\n") + "\ncuda_sources = [\"kernels/native.cu\"]\n"
    )
    # Declared CUDA sources require the matching rebuild step; the catalog
    # refuses the pair otherwise.
    (native / "rebuild.json").write_text(
        '{"steps":[{"type":"repo_python","path":"build_cuda_ext.py"}]}\n'
    )
    for path in sorted(native.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    native.chmod(0o700)

    harness = _harness(tmp_path / "native", source_fixture=native)
    sealed = harness.inputs.resident_speed_policy
    value = registered.build_b300_registered_qualification_factory(
        harness.inputs
    ).plan_builder(harness.cohort, b"v" * 32)

    assert value.resident_speed_plan is not None
    planned = value.resident_speed_plan.policy
    assert planned.version == 8
    # Only the schedule differs. Every calibrated threshold is the sealed one,
    # so this is a different read order, not a different bar.
    assert replace(planned, version=sealed.version) == sealed


def test_registry_rejects_unknown_or_stale_authority(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)

    with pytest.raises(registered.B300RegisteredQualificationError, match="unsupported"):
        harness.factory.profile_for("unknown.registered.target")

    rows = list(harness.policy.target_spec_digests)
    rows[0] = (rows[0][0], _h("stale-target-spec"))
    projections = list(harness.policy.target_contract_projection)
    projections[0] = replace(
        projections[0], target_spec_digest=rows[0][1]
    )
    stale = replace(
        harness.policy,
        target_spec_digests=tuple(rows),
        target_contract_projection=tuple(projections),
    )
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="stale",
    ):
        stale.require_catalog(default_target_catalog())

    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="exactly cover",
    ):
        replace(
            harness.policy,
            target_contract_projection=tuple(
                reversed(harness.policy.target_contract_projection)
            ),
        )

def test_graph_facts_cannot_relabel_another_registered_target(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)

    def wrong_target(_candidate, _prepared, _model_profile_key):
        descriptor = _h("wrong-target-shape")
        requirement = GraphVariantRequirement(
            "moe.fused_experts",
            "default",
            (descriptor,),
            True,
            (descriptor,),
        )
        observation = GraphVariantObservation(
            "moe.fused_experts",
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
    assert "another or incomplete member" in str(caught.value.__cause__)


def test_graph_facts_reject_duplicate_or_reordered_variants() -> None:
    members = (
        "collective.all_reduce",
        "collective.ar_residual_rmsnorm",
    )
    facts = _graph_facts_for_members(members)
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="canonical requirement/observation coverage",
    ):
        registered.B300FocusedGraphFacts(
            facts.expected_graph_replays,
            (facts.variants[0], facts.variants[0], *facts.variants[1:]),
            (facts.observations[0], facts.observations[0], *facts.observations[1:]),
        )
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="canonical requirement/observation coverage",
    ):
        registered.B300FocusedGraphFacts(
            facts.expected_graph_replays,
            tuple(reversed(facts.variants)),
            tuple(reversed(facts.observations)),
        )


def test_graph_evidence_hold_survives_registered_profile_resolution(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    hold = B300QualificationGraphEvidenceHold("graph attempt is still armed")

    def unavailable(_candidate, _prepared, _model_profile_key):
        raise hold

    inputs = replace(
        harness.inputs,
        graph_facts_builder_digest=_h("held-graph-builder"),
        graph_facts_builder=unavailable,
    )
    factory = registered.build_b300_registered_qualification_factory(inputs)
    with pytest.raises(B300QualificationGraphEvidenceHold) as caught:
        factory.plan_builder(harness.cohort, b"h" * 32)
    assert caught.value is hold


def test_corrupt_graph_store_state_becomes_typed_hold(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)

    def corrupt(_candidate, _prepared, _model_profile_key):
        raise B300QualificationGraphEvidenceStoreError("corrupt graph bytes")

    inputs = replace(
        harness.inputs,
        graph_facts_builder_digest=_h("corrupt-graph-builder"),
        graph_facts_builder=corrupt,
    )
    factory = registered.build_b300_registered_qualification_factory(inputs)
    with pytest.raises(B300QualificationGraphEvidenceHold, match="unauthenticated"):
        factory.plan_builder(harness.cohort, b"h" * 32)


def test_inputs_are_built_only_by_the_commissioner(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    unsealed = {
        name: getattr(harness.inputs, name)
        for name in harness.inputs.__dataclass_fields__
        if name != "seal"
    }
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="built only by the commissioner",
    ):
        registered.B300RegisteredQualificationInputs(**unsealed)
    with pytest.raises(
        registered.B300RegisteredQualificationError,
        match="built only by the commissioner",
    ):
        registered.B300RegisteredQualificationInputs(**unsealed, seal=object())


def test_blocker_inventory_names_only_missing_commissioning_authorities() -> None:
    assert tuple(row.blocker_id for row in registered.PRODUCTION_AUTHORITY_BLOCKERS) == (
        "registered-focused-graph-facts",
        "registered-runtime-binding",
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
    assert audit.plan.prompt_batches == tuple(
        (expected,) for _ in range(harness.policy.audit_minimum_calls + 1)
    )
    assert all(len(expected) <= len(prompt) for prompt in prompts)
    assert audit.plan.warmup_count == 1
    assert audit.plan.conditioning_count == 1
    assert audit.plan.max_new_tokens == harness.policy.audit_max_new_tokens

    repeat = harness.factory.plan_builder(harness.cohort, secret)
    assert repeat.resident_audit_plan.plan.prompt_batches == audit.plan.prompt_batches

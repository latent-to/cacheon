"""Tracked primary/reproduction qualification commissioning for one pod."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cacheon.eval.b300_screen_deployment as screen_deployment
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.engine_tree import materialize_engine_tree, reopen_materialized_engine_tree
from cacheon.eval.b300_mainnet_worker import B300MainnetWorker
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300RegisteredProfileAuthority,
    compose_b300_qualification_deployment,
)
from cacheon.eval.b300_registered_qualification_inputs import _COMMISSION_SEAL
from cacheon.eval.b300_registered_qualification import (
    REGISTERED_B300_TARGET_IDS,
    B300RegisteredQualificationError,
    B300RegisteredQualificationInputs,
    B300RegisteredQualificationPolicy,
    build_b300_registered_qualification_factory,
)
from cacheon.eval.b300_resident_pair_factory import (
    B300CommissionedResidentPairFactory,
    B300ResidentStockLanePlan,
)
from cacheon.eval.b300_sealed_qualification_commission import (
    B300QualificationCapabilities,
    B300QualificationCommissionError,
    CALIBRATION_PACKAGE_SCHEMA,
    QUALIFICATION_DEADLINE_MAXIMUM_SECONDS,
    QUALIFICATION_EVIDENCE_POLICY_DIGEST,
    QUALIFICATION_STAGES,
    declared_qualification_deadline_digest,
    declared_qualification_entropy_digest,
    parse_sealed_calibration_package,
    sealed_qualification_profile_rows,
)
from cacheon.eval.b300_screen_qualification_bridge import (
    QUALIFICATION_EXECUTOR_ID,
    CommissionedB300QualificationService,
)
from cacheon.eval.b300_remote_worker_adapter import B300RemoteQualificationCommission
from cacheon.eval.calibration import (
    CalibrationContext,
    CalibrationEvidenceSet,
    CalibrationManifest,
    CalibrationThresholdPolicy,
    publish_calibration_evidence,
)
from cacheon.eval.crossover_runtime import ResidentArmPlan, ResidentSpeedPolicy
from cacheon.eval.device_state import DeviceStatePolicy
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    LogicalHardwareSpec,
    PhysicalHardwareBinding,
    TrustedLaunchBinding,
)
from cacheon.eval.marginal_runtime import MaterializedArmBinding
from cacheon.eval.oci_backend import (
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
)
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_resident_session import ResidentSessionPlan
from cacheon.eval.qualification import ReferenceManifest
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.registered_resident_count_quality import (
    B300ResidentCountQualityBuilderContext,
    B300ResidentCountQualityCapability,
    RegisteredResidentCountQualityError,
)
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.stack_manifest import EvaluationStackContext, EvaluationStackManifest
from cacheon.target_catalog import default_target_catalog


_STAGES = QUALIFICATION_STAGES


def _pristine_reference_authority(
    incumbent_launch: EngineLaunchSpec,
    baseline_session_plan: SessionExecutionPlan,
    runtime_preflight: object,
) -> tuple[EngineLaunchSpec, SessionExecutionPlan]:
    """Derive pristine T without the candidate/stock seam selection."""

    pristine_config = replace(
        baseline_session_plan.engine_config,
        seam_bindings=(),
    )
    pristine_launch = replace(
        incumbent_launch,
        engine_config_digest=pristine_config.digest,
    )
    pristine_session_plan = replace(
        baseline_session_plan,
        launch_digest=pristine_launch.digest,
        expected_engine_config_digest=pristine_config.digest,
        engine_config=pristine_config,
        expected_preflight=expected_runtime_preflight(
            pristine_launch, runtime_preflight
        ),
    )
    if (
        pristine_config.seam_bindings
        or pristine_launch.digest == incumbent_launch.digest
    ):
        raise B300QualificationCommissionError(
            "pristine T did not remove the incumbent seam selection"
        )
    return pristine_launch, pristine_session_plan


def _bind_hidden_judge(
    capability: object,
    *,
    binding: HiddenJudgeBinding,
    tokenizer_digest: str,
    prompt_batches: tuple[tuple[str, ...], ...],
    workload_digest: str,
    hidden_tasks_per_prompt: int,
) -> object:
    judge = capability
    binder = getattr(capability, "bind_prompt_plan", None)
    if callable(binder):
        if getattr(capability, "tokenizer_digest", None) != tokenizer_digest:
            raise B300QualificationCommissionError(
                "hidden judge tokenizer differs from the sealed prompt identity"
            )
        try:
            judge = binder(
                prompt_batches=prompt_batches,
                workload_digest=workload_digest,
                hidden_tasks_per_prompt=hidden_tasks_per_prompt,
            )
        except Exception as exc:
            raise B300QualificationCommissionError(
                f"hidden judge prompt-plan binding failed: {exc}"
            ) from None
    if not callable(judge) or getattr(judge, "binding", None) != binding:
        raise B300QualificationCommissionError(
            "bound hidden judge differs from the sealed prompt identity"
        )
    return judge


def _tracked_deadline_provider(
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[object], float]:
    def deadline(_cohort: object) -> float:
        return float(clock()) + float(QUALIFICATION_DEADLINE_MAXIMUM_SECONDS)

    return deadline


def _require_complete_factory_profiles(
    profiles: object,
) -> dict[str, B300RegisteredProfileAuthority]:
    if (
        type(profiles) is not tuple
        or any(type(row) is not B300RegisteredProfileAuthority for row in profiles)
        or tuple(row.target_id for row in profiles) != REGISTERED_B300_TARGET_IDS
    ):
        raise B300QualificationCommissionError(
            "registered qualification factory does not cover the full catalog"
        )
    return {row.target_id: row for row in profiles}


def _private_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _swap_intake_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o711)
    path.chmod(0o711)
    return path


def _resident_plan(
    launch: EngineLaunchSpec,
    binding: TrustedLaunchBinding,
    workload: SessionExecutionPlan,
) -> ResidentSessionPlan:
    return ResidentSessionPlan(
        launch.digest,
        workload.expected_engine_config_digest,
        workload.engine_config,
        expected_runtime_preflight(launch, binding.runtime_preflight_receipt),
        10_000,
        100_000,
        workload.max_new_tokens,
        workload.top_logprobs_num,
        workload.temperature,
    )


def _sealed_calibration(
    inputs: "screen_deployment._CommissionedInputs",
    context: CalibrationContext,
    stage: str,
) -> tuple[
    CalibrationThresholdPolicy,
    CalibrationManifest,
    CalibrationEvidenceSet,
]:
    reference = inputs.authority_refs["calibration_package"]
    try:
        path, value, sha = screen_deployment._stable_json(
            reference["path"], "calibration package"
        )
    except screen_deployment.B300ScreenDeploymentError as exc:
        raise B300QualificationCommissionError(
            f"sealed calibration package is unreadable: {exc}"
        ) from None
    if str(path) != reference["path"] or sha != reference["sha256"]:
        raise B300QualificationCommissionError(
            "sealed calibration package differs from its deployment reference"
        )
    return parse_sealed_calibration_package(value, context, stage)


def _lane_policies(
    inputs: "screen_deployment._CommissionedInputs",
) -> tuple[DeviceStatePolicy, DeviceStatePolicy]:
    by_id = {gpu.physical_id: gpu for gpu in inputs.qualification_gpus}
    policies = []
    for lane in (
        inputs.qualification_lane_pair.lane_a,
        inputs.qualification_lane_pair.lane_b,
    ):
        try:
            gpus = tuple(by_id[physical_id] for physical_id in lane.physical_gpu_ids)
        except KeyError:
            raise B300QualificationCommissionError(
                "sealed qualification lane is absent from READY inventory"
            ) from None
        policy = screen_deployment._device_policy(gpus)
        if (
            policy.policy_sha256 != lane.device_policy_digest
            or policy.configuration_sha256 != lane.device_configuration_digest
        ):
            raise B300QualificationCommissionError(
                "READY device policy differs from the sealed qualification lane"
            )
        policies.append(policy)
    return policies[0], policies[1]


def compose_commissioned_qualifications(
    inputs: "screen_deployment._CommissionedInputs",
    composition: "screen_deployment._Composition",
    readiness: WorkerReadiness,
    capabilities: B300QualificationCapabilities,
    *,
    calibration_loader: Callable[
        [object, CalibrationContext, str],
        tuple[CalibrationThresholdPolicy, CalibrationManifest, CalibrationEvidenceSet],
    ] = _sealed_calibration,
) -> tuple[tuple[B300RemoteQualificationCommission, ...], tuple[OCIEngineExecutor, ...]]:
    if type(capabilities) is not B300QualificationCapabilities:
        raise B300QualificationCommissionError(
            "qualification capabilities are not exactly typed"
        )
    block = inputs.qualification_commission
    if block is None:
        raise B300QualificationCommissionError(
            "sealed authority config declares no qualification commission"
        )
    declared = inputs.declared_qualification
    manifest = composition.manifest
    hidden_binding = HiddenJudgeBinding(
        inputs.prompt_identity["hidden_corpus_commitment"],
        inputs.prompt_identity["hidden_judge_digest"],
        inputs.prompt_identity["hidden_task_policy_digest"],
    )
    if capabilities.hidden_judge.binding != hidden_binding:
        raise B300QualificationCommissionError(
            "hidden judge capability differs from the sealed prompt identity"
        )
    if (
        capabilities.source_resolver_digest != block["source_resolver_digest"]
        or capabilities.graph_facts_builder_digest
        != block["graph_facts_builder_digest"]
        or capabilities.resident_count_quality_builder_digest
        != block["resident_count_quality_builder_digest"]
    ):
        raise B300QualificationCommissionError(
            "capability identities differ from the sealed commission block"
        )
    if inputs.authority.get("authority_role") not in _STAGES:
        raise B300QualificationCommissionError(
            "sealed authority role is not one retained qualification stage"
        )

    catalog = default_target_catalog()
    policy_block = block["policy"]
    session_block = block["session"]
    speed_block = block["resident_speed"]
    try:
        policy = B300RegisteredQualificationPolicy.seal(
            catalog,
            verification_policy_digest=block["verification_policy_digest"],
            nll_tail_threshold=policy_block["nll_tail_threshold"],
            tokens_per_prompt=policy_block["tokens_per_prompt"],
            topk_width=policy_block["topk_width"],
            hidden_tasks_per_prompt=policy_block["hidden_tasks_per_prompt"],
            support_policy_digest=block["support_policy_digest"],
            hidden_task_policy_digest=hidden_binding.hidden_task_policy_digest,
            hidden_tasks_required=policy_block["hidden_tasks_required"],
            select_count=policy_block["select_count"],
            audit_minimum_calls=policy_block["audit_minimum_calls"],
        )
    except B300RegisteredQualificationError as exc:
        raise B300QualificationCommissionError(
            f"sealed qualification policy failed to seal: {exc}"
        ) from None

    lane_a_policy, lane_b_policy = _lane_policies(inputs)
    lane_a_executor = screen_deployment._build_executor(
        inputs.root / "qualification-lane-a",
        inputs.preflight,
        lane_a_policy,
        executor_id=QUALIFICATION_EXECUTOR_ID,
        runtime_seed_root=inputs.runtime_seed_root,
    )
    lane_b_executor = screen_deployment._build_executor(
        inputs.root / "qualification-lane-b",
        inputs.preflight,
        lane_b_policy,
        executor_id=QUALIFICATION_EXECUTOR_ID,
        runtime_seed_root=inputs.runtime_seed_root,
    )
    executors = (lane_a_executor, lane_b_executor)
    try:
        commissions = tuple(
            _compose_locked(
                inputs,
                manifest,
                composition,
                readiness,
                capabilities,
                block,
                policy,
                catalog,
                hidden_binding,
                candidate_executor,
                baseline_executor,
                screen_lane,
                declared,
                session_block,
                speed_block,
                calibration_loader,
            )
            for screen_lane, candidate_executor, baseline_executor in (
                ("primary", lane_a_executor, lane_b_executor),
                ("reproduction", lane_b_executor, lane_a_executor),
            )
        )
    except BaseException:
        for executor in executors:
            executor.manager.close()
        raise
    if len(commissions) != 2:  # pragma: no cover - fixed tuple invariant
        raise AssertionError("commissioned stage set changed")
    return (commissions[0], commissions[1]), executors


def _compose_locked(
    inputs,
    manifest,
    composition,
    readiness,
    capabilities,
    block,
    policy,
    catalog,
    hidden_binding,
    candidate_executor,
    baseline_executor,
    screen_lane,
    declared,
    session_block,
    speed_block,
    calibration_loader,
) -> B300RemoteQualificationCommission:
    snapshot = catalog.snapshot()
    lane_pair = inputs.qualification_lane_pair
    context = EvaluationStackContext(
        runtime_digest=inputs.runtime.runtime_digest,
        base_engine_digest=inputs.runtime.base_engine_digest,
        arena_digest=manifest.digest,
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        target_spec_digests=screen_deployment._catalog_specs(catalog),
    )
    stock = EvaluationStackManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        entries={},
    )
    trees_root = inputs.root / "engine-trees"
    trees_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = trees_root / f"resident-stock-{stock.digest}"
    if destination.exists():
        tree = reopen_materialized_engine_tree(destination)
    else:
        tree = materialize_engine_tree(
            stock,
            context=context,
            catalog=catalog,
            resolver={},
            destination=destination,
        )
    if tree.stack_digest != stock.digest or tree.runtime_manifest is not None:
        raise B300QualificationCommissionError(
            "qualification stock tree differs from the empty commissioned stack"
        )

    target_rows = snapshot.get("targets")
    if not isinstance(target_rows, list):
        raise B300QualificationCommissionError("target catalog snapshot is malformed")
    target_members = tuple(
        sorted(
            {
                member
                for row in target_rows
                if isinstance(row, dict)
                for member in row.get("members", ())
                if isinstance(member, str)
            }
        )
    )
    if not target_members:
        raise B300QualificationCommissionError(
            "qualification target member set is empty"
        )
    engine_config = screen_deployment._engine_config(
        target_members, disable_cuda_graph=False
    )
    baseline_hardware = LogicalHardwareSpec(
        visible_gpu_count=screen_deployment.GPU_COUNT,
        architecture=screen_deployment.ARCHITECTURE,
        topology_class=inputs.runtime.topology_class,
        topology_digest=inputs.topology_digest,
        tp_size=screen_deployment.TP_SIZE,
        ep_size=1,
        dp_size=1,
        device_policy_digest=candidate_executor.device_policy.policy_sha256,
    )
    baseline_physical = PhysicalHardwareBinding(
        physical_gpu_ids=tuple(
            str(gpu.physical_id)
            for gpu in candidate_executor.device_policy.expected_gpus
        ),
        architecture=screen_deployment.ARCHITECTURE,
        topology_class=inputs.runtime.topology_class,
        topology_digest=inputs.topology_digest,
        tp_size=screen_deployment.TP_SIZE,
        ep_size=1,
        dp_size=1,
        device_policy_digest=candidate_executor.device_policy.policy_sha256,
    )
    stock_native = screen_deployment._native_build(
        tree.tree_digest,
        inputs.preflight,
        candidate_executor.config.prebuild.policy,
    )
    incumbent_launch = EngineLaunchSpec(
        runtime_digest=inputs.runtime.runtime_digest,
        base_engine_digest=inputs.runtime.base_engine_digest,
        arena_digest=manifest.digest,
        stack_digest=tree.stack_digest,
        tree_digest=tree.tree_digest,
        image_digest=inputs.preflight.image_digest,
        platform_digest=inputs.preflight.platform_digest,
        controller_distribution_digest=inputs.controller_distribution_digest,
        worker_distribution_digest=inputs.preflight.worker_distribution_digest,
        model_revision_digest=inputs.runtime.model_revision_digest,
        model_manifest_digest=inputs.runtime.model_manifest_digest,
        model_content_digest=inputs.runtime.model_content_digest,
        validator_overlay_digest=inputs.runtime.validator_overlay_digest,
        engine_config_digest=engine_config.digest,
        seccomp_policy_digest=screen_deployment._file_sha256(
            candidate_executor.config.prebuild.seccomp_profile
        ),
        resource_policy_digest=(
            candidate_executor.config.prebuild.policy.resource_policy_digest
        ),
        native_build_spec_digest=stock_native.digest,
        hardware=baseline_hardware,
    )
    trusted_baseline = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=inputs.controller_distribution_digest,
        native_build_spec=stock_native,
        runtime_preflight_receipt=inputs.preflight,
        physical_hardware=baseline_physical,
    )
    incumbent_binding = MaterializedArmBinding(tree, trusted_baseline)
    baseline_session_plan = SessionExecutionPlan(
        launch_digest=incumbent_launch.digest,
        expected_engine_config_digest=engine_config.digest,
        engine_config=engine_config,
        expected_preflight=expected_runtime_preflight(
            incumbent_launch, inputs.preflight
        ),
        prompt_batches=inputs.prompt_batches,
        warmup_count=session_block["warmup_count"],
        conditioning_count=session_block["conditioning_count"],
        max_new_tokens=policy.tokens_per_prompt,
        top_logprobs_num=policy.topk_width,
        temperature=float(session_block["temperature"]),
    )
    pristine_launch, pristine_session_plan = _pristine_reference_authority(
        incumbent_launch,
        baseline_session_plan,
        inputs.preflight,
    )
    workload_digest = marginal_workload_digest(baseline_session_plan)
    hidden_judge = _bind_hidden_judge(
        capabilities.hidden_judge,
        binding=hidden_binding,
        tokenizer_digest=inputs.prompt_identity["tokenizer_digest"],
        prompt_batches=inputs.prompt_batches,
        workload_digest=workload_digest,
        hidden_tasks_per_prompt=policy.hidden_tasks_per_prompt,
    )
    model_mount = TrustedArenaModelMountReceipt.capture(
        inputs.model_root,
        arena_digest=manifest.digest,
        model_revision_digest=inputs.runtime.model_revision_digest,
        model_manifest_digest=inputs.runtime.model_manifest_digest,
        model_content_digest=inputs.runtime.model_content_digest,
    )
    reference = ReferenceManifest.from_pristine(
        stock,
        pristine_launch,
        incumbent_binding,
        workload_digest=workload_digest,
        tokenizer_digest=inputs.prompt_identity["tokenizer_digest"],
        hidden_corpus_commitment=hidden_binding.hidden_corpus_commitment,
        hidden_judge_digest=hidden_binding.hidden_judge_digest,
        selection_policy_digest=inputs.prompt_identity["selection_policy_digest"],
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
        policy.verification_policy_digest,
    )
    threshold, calibration_manifest, calibration_evidence = calibration_loader(
        inputs, calibration_context, screen_lane
    )
    evidence_root = _private_root(inputs.root / "qualification-evidence")
    materialization_root = _private_root(inputs.root / "qualification-candidates")
    calibration_ref = publish_calibration_evidence(
        evidence_root,
        calibration_evidence,
    )
    resident_hardware = replace(
        baseline_hardware,
        device_policy_digest=baseline_executor.device_policy.policy_sha256,
    )
    resident_physical = replace(
        baseline_physical,
        physical_gpu_ids=tuple(
            str(gpu.physical_id)
            for gpu in baseline_executor.device_policy.expected_gpus
        ),
        device_policy_digest=baseline_executor.device_policy.policy_sha256,
    )
    resident_native = screen_deployment._native_build(
        tree.tree_digest,
        inputs.preflight,
        baseline_executor.config.prebuild.policy,
    )
    resident_launch = replace(
        incumbent_launch,
        hardware=resident_hardware,
        native_build_spec_digest=resident_native.digest,
        resource_policy_digest=(
            baseline_executor.config.prebuild.policy.resource_policy_digest
        ),
        seccomp_policy_digest=screen_deployment._file_sha256(
            baseline_executor.config.prebuild.seccomp_profile
        ),
    )
    resident_binding = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=inputs.controller_distribution_digest,
        native_build_spec=resident_native,
        runtime_preflight_receipt=inputs.preflight,
        physical_hardware=resident_physical,
    )
    resident_session_plan = replace(
        baseline_session_plan,
        launch_digest=resident_launch.digest,
        expected_preflight=expected_runtime_preflight(
            resident_launch, inputs.preflight
        ),
    )
    resident_baseline_arm = ResidentArmPlan(
        resident_launch,
        resident_binding,
        resident_session_plan,
        baseline_executor.manager.namespace_digest,
        baseline_executor.config.runtime.digest,
        baseline_executor.device_policy.configuration_sha256,
    )
    resident_speed_policy = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=speed_block["max_stage_seconds"],
        max_qualification_seconds=speed_block["max_qualification_seconds"],
        calibration=calibration_manifest,
        context=calibration_context,
        # v7 swaps the baseline arm too. Under v6 only the candidate lane took
        # a swap, which handed the candidate role a measured advantage on
        # identical work; every v7 gate is `>=`, so it inherits v6 grading
        # unchanged and adds only the symmetric swap.
        version=7,
        min_windows=speed_block["min_windows"],
        max_window_scatter=float(speed_block["max_window_scatter"]),
        max_conditioning_slowdown=float(speed_block["max_conditioning_slowdown"]),
    )

    def bind_candidate(candidate_tree) -> TrustedLaunchBinding:
        candidate_native = screen_deployment._native_build(
            candidate_tree.tree_digest,
            inputs.preflight,
            candidate_executor.config.prebuild.policy,
        )
        return TrustedLaunchBinding(
            materialized_tree_root=candidate_tree.root,
            controller_distribution_digest=inputs.controller_distribution_digest,
            native_build_spec=candidate_native,
            runtime_preflight_receipt=inputs.preflight,
            physical_hardware=baseline_physical,
        )

    orientation = lane_pair.orientation(screen_lane)
    baseline_lane_plan = B300ResidentStockLanePlan(
        orientation.resident_baseline,
        tree,
        resident_launch,
        resident_binding,
        _resident_plan(resident_launch, resident_binding, resident_session_plan),
        resident_session_plan,
        baseline_executor,
    )
    candidate_lane_plan = B300ResidentStockLanePlan(
        orientation.candidate,
        tree,
        incumbent_launch,
        incumbent_binding.launch_binding,
        _resident_plan(
            incumbent_launch,
            incumbent_binding.launch_binding,
            baseline_session_plan,
        ),
        baseline_session_plan,
        candidate_executor,
    )
    resident_pair_factory = B300CommissionedResidentPairFactory(
        service_digest=manifest.digest,
        readiness=readiness,
        lane_pair=lane_pair,
        lane_plans=(baseline_lane_plan, candidate_lane_plan),
        model_mount=model_mount,
        swap_intake_root=_swap_intake_root(
            inputs.root / "resident-intake" / screen_lane
        ),
    )
    try:
        count_context = B300ResidentCountQualityBuilderContext(
            catalog,
            stock,
            incumbent_launch,
            incumbent_binding,
            evidence_root,
            lane_pair,
            engine_config.max_running_requests,
        )
        resident_count_quality = capabilities.resident_count_quality_builder(
            count_context
        )
        if type(resident_count_quality) is not B300ResidentCountQualityCapability:
            raise RegisteredResidentCountQualityError(
                "resident count builder returned another capability type"
            )
        resident_count_quality.validate(count_context)
    except (RegisteredResidentCountQualityError, TypeError, ValueError) as exc:
        raise B300QualificationCommissionError(
            f"resident count capability failed to compose: {exc}"
        ) from None

    try:
        factory_inputs = B300RegisteredQualificationInputs(
            catalog=catalog,
            policy=policy,
            expected_context=context,
            incumbent_stack=stock,
            incumbent_binding=incumbent_binding,
            incumbent_launch=incumbent_launch,
            baseline_session_plan=baseline_session_plan,
            model_mount=model_mount,
            materialization_root=materialization_root,
            source_resolver_digest=capabilities.source_resolver_digest,
            source_resolver=capabilities.source_resolver,
            candidate_binding_builder_digest=block[
                "candidate_binding_builder_digest"
            ],
            candidate_binding_builder=bind_candidate,
            graph_facts_builder_digest=capabilities.graph_facts_builder_digest,
            graph_facts_builder=capabilities.graph_facts_builder,
            evidence_root=evidence_root,
            reference_manifest=reference,
            calibration_threshold_policy=threshold,
            calibration_manifest=calibration_manifest,
            calibration_context=calibration_context,
            calibration_artifact_ref=calibration_ref,
            pristine_stack=stock,
            pristine_binding=incumbent_binding,
            pristine_launch=pristine_launch,
            pristine_session_plan=pristine_session_plan,
            resident_baseline_arm=resident_baseline_arm,
            resident_speed_policy=resident_speed_policy,
            candidate_executor_namespace_digest=(
                candidate_executor.manager.namespace_digest
            ),
            candidate_runtime_resource_policy_digest=(
                candidate_executor.config.runtime.digest
            ),
            candidate_device_configuration_digest=(
                candidate_executor.device_policy.configuration_sha256
            ),
            seal=_COMMISSION_SEAL,
        )
        factory = build_b300_registered_qualification_factory(factory_inputs)
    except B300RegisteredQualificationError as exc:
        raise B300QualificationCommissionError(
            f"registered qualification factory failed to compose: {exc}"
        ) from None

    factory_rows = _require_complete_factory_profiles(factory.profiles)
    profiles = tuple(
        B300RegisteredProfileAuthority(
            target_id,
            spec_digest,
            resolver_digest,
            factory_rows[target_id].resolver,
        )
        for target_id, spec_digest, resolver_digest in (
            sealed_qualification_profile_rows(
                catalog,
                builder_source_digest=block["builder_source_digest"],
            )
        )
    )
    construction = B300QualificationConstructionAuthority(
        catalog=catalog,
        profiles=profiles,
        incumbent_stack=stock,
        incumbent_tree_digest=tree.tree_digest,
        pristine_stack=stock,
        pristine_tree_digest=tree.tree_digest,
        evidence_root=evidence_root,
        evidence_policy_digest=QUALIFICATION_EVIDENCE_POLICY_DIGEST,
        builder_source_digest=block["builder_source_digest"],
        selection_store_digest=block["selection_store_digest"],
        resident_count_quality_builder_digest=(
            block["resident_count_quality_builder_digest"]
        ),
        resident_count_quality=resident_count_quality,
        secret_loader=capabilities.secret_loader,
        plan_builder=factory.plan_builder,
        entropy_provider_digest=declared_qualification_entropy_digest(
            inputs.prompt_identity["selection_policy_digest"]
        ),
        entropy_provider=capabilities.entropy_provider,
        hidden_judge=hidden_judge,
        deadline_policy_digest=declared_qualification_deadline_digest(),
        deadline_provider=_tracked_deadline_provider(),
    )
    if (
        construction.qualification_policy_digest
        != declared.qualification_policy_digest
        or construction.qualification_builder_digest
        != declared.qualification_builder_digest
    ):
        raise B300QualificationCommissionError(
            "composed qualification identity differs from the sealed declaration"
        )
    deployment = compose_b300_qualification_deployment(
        manifest=manifest,
        screen_authorities=composition.authorities,
        construction=construction,
        candidate_executor=candidate_executor,
        resident_baseline_executor=baseline_executor,
        resident_pair_factory=resident_pair_factory,
        screen_lane=screen_lane,
    )
    return B300RemoteQualificationCommission(deployment, construction, readiness)


def build_commissioned_b300_qualification_service(
    registration: dict[str, object],
    ready_receipt: dict[str, object],
    capabilities: B300QualificationCapabilities,
) -> CommissionedB300QualificationService:
    inputs, composition, readiness = (
        screen_deployment.replay_commissioned_screen_composition(
            registration, ready_receipt
        )
    )
    executors: tuple[OCIEngineExecutor, ...] = ()
    worker: B300MainnetWorker | None = None
    try:
        commissions, executors = compose_commissioned_qualifications(
            inputs, composition, readiness, capabilities
        )
        commission, reproduction_commission = commissions
        worker = B300MainnetWorker(
            commission.deployment.manifest,
            commission.deployment.authorities,
            readiness,
        )
        service = CommissionedB300QualificationService(
            worker,
            commission,
            reproduction_commission,
            executors,
            composition,
        )
        composition = None
        return service
    except BaseException:
        try:
            if worker is not None:
                worker.close()
        finally:
            for executor in executors:
                executor.manager.close()
        raise
    finally:
        if composition is not None:
            composition.close()


__all__ = [
    "B300QualificationCapabilities",
    "B300QualificationCommissionError",
    "CALIBRATION_PACKAGE_SCHEMA",
    "CommissionedB300QualificationService",
    "build_commissioned_b300_qualification_service",
    "compose_commissioned_qualifications",
]

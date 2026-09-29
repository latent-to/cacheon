"""Tracked primary/reproduction qualification commissioning for one pod."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cacheon.eval.b300_deployment as b300_deployment
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.eval.b300_arena_definition import (
    data_parallel_size as _data_parallel_size,
    device_policy as _device_policy,
    hardware_bindings as _hardware_bindings,
    scored_cell as _scored_cell,
)
from cacheon.eval.b300_mainnet_worker import B300MainnetWorker
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300RegisteredProfileAuthority,
    compose_b300_qualification_deployment,
)
from cacheon.eval.b300_registered_qualification_inputs import _COMMISSION_SEAL
from cacheon.eval.b300_registered_qualification import (
    B300RegisteredQualificationError,
    B300RegisteredQualificationInputs,
    B300RegisteredQualificationPolicy,
    build_b300_registered_qualification_factory,
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
from cacheon.eval.b300_qualification_declaration import QUALIFICATION_EXECUTOR_ID
from cacheon.eval.b300_qualification_lanes import commissioned_incumbent_arm as commissioned_incumbent_arm
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
    TrustedLaunchBinding,
)
from cacheon.eval.marginal_runtime import MaterializedArmBinding
from cacheon.eval.oci_backend import (
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
)
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.qualification import ReferenceManifest
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.target_catalog import default_target_catalog


_STAGES = QUALIFICATION_STAGES


def _pristine_reference_authority(
    incumbent_launch: EngineLaunchSpec,
    baseline_session_plan: SessionExecutionPlan,
    runtime_preflight: object,
    *,
    pristine_tree,
    pristine_native,
) -> tuple[EngineLaunchSpec, SessionExecutionPlan]:
    """Derive pristine T on the empty stock tree.

    Pristine T stays anchored to the empty stock tree even when the incumbent
    carries crowned entries, so the quality/audit reference never moves with
    the speed baseline; at genesis the two trees coincide and T is the baseline
    launch. Candidate code reaches an engine only through its tree, so the empty
    tree is what keeps T stock.
    """

    pristine_launch = replace(
        incumbent_launch,
        stack_digest=pristine_tree.stack_digest,
        tree_digest=pristine_tree.tree_digest,
        native_build_spec_digest=pristine_native.digest,
    )
    pristine_session_plan = replace(
        baseline_session_plan,
        launch_digest=pristine_launch.digest,
        expected_preflight=expected_runtime_preflight(
            pristine_launch, runtime_preflight
        ),
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
    if hidden_tasks_per_prompt and callable(binder):
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
    registered_target_ids: tuple[str, ...],
) -> dict[str, B300RegisteredProfileAuthority]:
    if (
        type(profiles) is not tuple
        or any(type(row) is not B300RegisteredProfileAuthority for row in profiles)
        or tuple(row.target_id for row in profiles) != registered_target_ids
    ):
        raise B300QualificationCommissionError(
            "registered qualification factory does not cover the full catalog"
        )
    return {row.target_id: row for row in profiles}


def _private_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _sealed_calibration(
    inputs: "b300_deployment._CommissionedInputs",
    context: CalibrationContext,
    stage: str,
) -> tuple[
    CalibrationThresholdPolicy,
    CalibrationManifest,
    CalibrationEvidenceSet,
]:
    reference = inputs.authority_refs["calibration_package"]
    try:
        path, value, sha = b300_deployment._stable_json(
            reference["path"], "calibration package"
        )
    except b300_deployment.B300DeploymentError as exc:
        raise B300QualificationCommissionError(
            f"sealed calibration package is unreadable: {exc}"
        ) from None
    if str(path) != reference["path"] or sha != reference["sha256"]:
        raise B300QualificationCommissionError(
            "sealed calibration package differs from its deployment reference"
        )
    return parse_sealed_calibration_package(value, context, stage)


def _lane_policies(
    inputs: "b300_deployment._CommissionedInputs",
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
        policy = _device_policy(gpus)
        if (
            policy.policy_sha256 != lane.device_policy_digest
            or policy.configuration_sha256 != lane.device_configuration_digest
        ):
            raise B300QualificationCommissionError(
                "READY device policy differs from the sealed qualification lane"
            )
        policies.append(policy)
    return policies[0], policies[1]


@dataclass
class CommissionedB300QualificationService:
    """One worker plus both sealed qualification orientations."""

    worker: B300MainnetWorker
    commission: B300RemoteQualificationCommission
    reproduction_commission: B300RemoteQualificationCommission
    _executors: tuple[OCIEngineExecutor, ...]
    _reproduction_worker: B300MainnetWorker | None = None
    _lock: object = field(default_factory=threading.RLock)
    _closed: bool = False

    def __post_init__(self) -> None:
        commissions = (self.commission, self.reproduction_commission)
        if (
            type(self._executors) is not tuple
            or len(self._executors) != 2
            or any(type(row) is not OCIEngineExecutor for row in self._executors)
            or len({id(row.manager) for row in self._executors}) != 2
            or tuple(row.deployment.screen_lane for row in commissions)
            != ("primary", "reproduction")
            or commissions[0].deployment.manifest != commissions[1].deployment.manifest
            or commissions[0].readiness != commissions[1].readiness
            or type(self.worker) is not B300MainnetWorker
            or self.worker.service.manifest != self.commission.deployment.manifest
            or self.worker.readiness != self.commission.readiness
            or self.worker._remote_qualification_lane != "primary"
        ):
            raise B300QualificationCommissionError(
                "commissioned service does not own both qualification orientations"
            )

    def adapter_for(self, publications, continuation_store, screen_lane: str):
        with self._lock:
            if self._closed:
                raise B300QualificationCommissionError(
                    "commissioned qualification service is closed"
                )
            if screen_lane == "primary":
                commission, worker = self.commission, self.worker
            elif screen_lane == "reproduction":
                commission = self.reproduction_commission
                worker = self._reproduction_worker
                if worker is None:
                    worker = B300MainnetWorker(
                        commission.deployment.manifest,
                        commission.deployment.authorities,
                        commission.readiness,
                    )
                    self._reproduction_worker = worker
            else:
                raise B300QualificationCommissionError(
                    "qualification stage must be primary or reproduction"
                )
            return commission.adapter_for(
                publications,
                continuation_store,
                worker=worker,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        closers = (
            self.worker.close,
            *(
                ()
                if self._reproduction_worker is None
                else (self._reproduction_worker.close,)
            ),
            *(executor.manager.close for executor in self._executors),
        )
        for closer in closers:
            try:
                closer()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure


def compose_commissioned_qualifications(
    inputs: "b300_deployment._CommissionedInputs",
    composition: "b300_deployment._Composition",
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
    if capabilities.source_resolver_digest != block["source_resolver_digest"]:
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
            registered_target_ids=inputs.registered_target_ids,
            model_profile_key=inputs.model_profile_key,
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

    _require_cell_conformance(inputs, policy, session_block, speed_block)

    lane_a_policy, lane_b_policy = _lane_policies(inputs)
    lane_a_executor = b300_deployment._build_executor(
        inputs.root / "qualification-lane-a",
        inputs.preflight,
        lane_a_policy,
        executor_id=QUALIFICATION_EXECUTOR_ID,
        runtime_seed_root=inputs.runtime_seed_root, resources=inputs.authority.get("resources"),
    )
    lane_b_executor = b300_deployment._build_executor(
        inputs.root / "qualification-lane-b",
        inputs.preflight,
        lane_b_policy,
        executor_id=QUALIFICATION_EXECUTOR_ID,
        runtime_seed_root=inputs.runtime_seed_root, resources=inputs.authority.get("resources"),
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


def _require_cell_conformance(inputs, policy, session_block, speed_block) -> None:
    """The declared workload cell and the consumed session are projections of
    one sealed authority; any mismatch is a commissioning error, never a
    runtime surprise.  Batch widths were validated against the cell at parse.
    A min_windows floor above the cell's timed reads is unsatisfiable by
    construction and must die here, not forty minutes into a measured run.
    """

    if "replay" in session_block:
        if policy.topk_width != 0 or float(session_block["temperature"]) != 0:
            raise B300QualificationCommissionError("goodput replay requires greedy decoding and teacher-NLL quality")
        if policy.hidden_tasks_required or policy.hidden_tasks_per_prompt:
            raise B300QualificationCommissionError("replay prompts require a teacher-only profile without numeric hidden tasks")
        return
    quality_cell = _scored_cell(inputs.workload)
    batch_cells = getattr(
        inputs,
        "prompt_batch_cells",
        (quality_cell.cell_id,) * len(inputs.prompt_batches),
    )
    warmup_cells = batch_cells[:session_block["warmup_count"]]
    expected_counts = {
        cell.cell_id: cell.timed_reads
        + warmup_cells.count(cell.cell_id)
        for cell in inputs.workload.cells
    }
    observed_counts = {
        cell.cell_id: batch_cells.count(cell.cell_id)
        for cell in inputs.workload.cells
    }
    if (
        policy.tokens_per_prompt != max(cell.output_tokens for cell in inputs.workload.cells)
        or type(batch_cells) is not tuple
        or len(batch_cells) != len(inputs.prompt_batches)
        or observed_counts != expected_counts
        or set(warmup_cells) != {cell.cell_id for cell in inputs.workload.cells}
        or speed_block["min_windows"]
        > sum(cell.timed_reads for cell in inputs.workload.cells)
    ):
        raise B300QualificationCommissionError(
            "sealed session does not conform to the declared workload cell"
        )


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
    target_members, context, stock, stock_tree = (
        b300_deployment._commissioned_stock_authority(
            inputs,
            manifest,
            catalog,
            snapshot,
            error=B300QualificationCommissionError,
            label="pristine reference",
        )
    )
    replay = None
    goodput = None
    if "replay" in session_block:
        from cacheon.eval.agent_replay import AgentReplayPlan
        from cacheon.eval.goodput_runtime import GoodputPolicy
        settings = session_block["replay"]
        goodput = GoodputPolicy.from_dict(speed_block["goodput"])
        if not goodput.error_rate:
            raise B300QualificationCommissionError(
                "new replay commissions require elapsed-work statistical eligibility; "
                "V16 latency scoring is retained only for historical evidence"
            )
        replay = AgentReplayPlan(
            Path(settings["manifest_path"]), (settings["load"],),
            Path(settings["aiperf_binary"]), Path(settings["tokenizer_path"]),
            inputs.root / "qualification-replays" / screen_lane,
            goodput.contract, "incumbent", 1, screen_lane, windows=settings["windows"],
            elapsed_cost=True,
            max_work_seconds=settings.get("max_work_seconds", 0),
        )
        if replay.slice.digest != settings["slice_digest"]:
            raise B300QualificationCommissionError("replay slice differs from its sealed authority")
        from cacheon.eval.qualification_trajectories import prompt_pool
        from cacheon.eval.reference_protocol import MAX_TOKENS
        maximum = max(prompt_pool(replay).values())
        if policy.tokens_per_prompt != maximum or maximum > MAX_TOKENS:
            raise B300QualificationCommissionError(
                "reference token maximum must match the replay slice and fit the reference protocol"
            )
    incumbent, incumbent_binding, incumbent_arm = commissioned_incumbent_arm(
        inputs, manifest, candidate_executor,
        entries=capabilities.incumbent_entries, resolver=capabilities.source_resolver,
        replay=replay,
    )
    incumbent_tree = incumbent_binding.tree
    incumbent_launch = incumbent_arm.launch
    baseline_session_plan = incumbent_arm.session_plan
    engine_config = baseline_session_plan.engine_config
    dp_size = _data_parallel_size(engine_config)
    baseline_physical = incumbent_arm.binding.physical_hardware
    mixed_cells = len(inputs.workload.cells) > 1
    pristine_native = b300_deployment._native_build(
        stock_tree.tree_digest,
        inputs.preflight,
        candidate_executor.config.prebuild.policy, inputs.runtime.target_architecture,
    )
    trusted_pristine = replace(
        incumbent_arm.binding, materialized_tree_root=stock_tree.root,
        native_build_spec=pristine_native,
    )
    pristine_binding = MaterializedArmBinding(stock_tree, trusted_pristine)
    pristine_launch, pristine_session_plan = _pristine_reference_authority(
        incumbent_launch,
        baseline_session_plan,
        inputs.preflight,
        pristine_tree=stock_tree,
        pristine_native=pristine_native,
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
        pristine_binding,
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
    resident_hardware, resident_physical = _hardware_bindings(
        inputs.runtime, baseline_executor.device_policy, dp_size=dp_size,
    )
    resident_native = b300_deployment._native_build(
        incumbent_tree.tree_digest,
        inputs.preflight,
        baseline_executor.config.prebuild.policy, inputs.runtime.target_architecture,
    )
    resident_launch = replace(
        incumbent_launch,
        hardware=resident_hardware,
        native_build_spec_digest=resident_native.digest,
        resource_policy_digest=(
            baseline_executor.config.prebuild.policy.resource_policy_digest
        ),
        seccomp_policy_digest=b300_deployment._file_sha256(
            baseline_executor.config.prebuild.seccomp_profile
        ),
    )
    resident_binding = TrustedLaunchBinding(
        materialized_tree_root=incumbent_tree.root,
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
    prefill_lane = speed_block.get("prefill_lane")
    if prefill_lane is not None and not mixed_cells:
        # Version 12 scores the mixed-cell makespan; a single-cell workload
        # has no such rule to append the prefill pass to.
        raise B300QualificationCommissionError(
            "the prefill lane requires a mixed-cell workload"
        )
    resident_speed_policy = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=speed_block["max_stage_seconds"],
        max_qualification_seconds=speed_block["max_qualification_seconds"],
        calibration=calibration_manifest,
        context=calibration_context,
        # New commissions seal one bounded borderline repeat: v13 single-cell,
        # v14 mixed-cell, or v15 with the prefill pass. Existing evidence keeps
        # its original version and arithmetic.
        version=17 if goodput is not None else 15 if prefill_lane is not None else 14 if mixed_cells else 13,
        min_windows=speed_block["min_windows"],
        max_window_scatter=float(speed_block["max_window_scatter"]),
        max_conditioning_slowdown=float(speed_block["max_conditioning_slowdown"]),
        prefill_min_margin=(
            float(prefill_lane["min_margin"]) if prefill_lane is not None else 0.0
        ),
        prefill_credit_weight=(
            float(prefill_lane["credit_weight"]) if prefill_lane is not None else 0.0
        ),
        goodput=goodput,
    )

    def bind_candidate(candidate_tree) -> TrustedLaunchBinding:
        candidate_native = b300_deployment._native_build(
            candidate_tree.tree_digest,
            inputs.preflight,
            candidate_executor.config.prebuild.policy, inputs.runtime.target_architecture,
        )
        return TrustedLaunchBinding(
            materialized_tree_root=candidate_tree.root,
            controller_distribution_digest=inputs.controller_distribution_digest,
            native_build_spec=candidate_native,
            runtime_preflight_receipt=inputs.preflight,
            physical_hardware=baseline_physical,
        )

    try:
        factory_inputs = B300RegisteredQualificationInputs(
            catalog=catalog,
            policy=policy,
            expected_context=context,
            incumbent_stack=incumbent,
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
            evidence_root=evidence_root,
            reference_manifest=reference,
            calibration_threshold_policy=threshold,
            calibration_manifest=calibration_manifest,
            calibration_context=calibration_context,
            calibration_artifact_ref=calibration_ref,
            pristine_stack=stock,
            pristine_binding=pristine_binding,
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

    factory_rows = _require_complete_factory_profiles(
        factory.profiles, inputs.registered_target_ids
    )
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
                registered_target_ids=inputs.registered_target_ids,
                builder_source_digest=block["builder_source_digest"],
            )
        )
    )
    construction = B300QualificationConstructionAuthority(
        catalog=catalog,
        registered_target_ids=inputs.registered_target_ids,
        profiles=profiles,
        incumbent_stack=incumbent,
        incumbent_tree_digest=incumbent_tree.tree_digest,
        pristine_stack=stock,
        pristine_tree_digest=stock_tree.tree_digest,
        evidence_root=evidence_root,
        evidence_policy_digest=QUALIFICATION_EVIDENCE_POLICY_DIGEST,
        builder_source_digest=block["builder_source_digest"],
        selection_store_digest=block["selection_store_digest"],
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
        declared=composition.authorities,
        construction=construction,
        candidate_executor=candidate_executor,
        resident_baseline_executor=baseline_executor,
        screen_lane=screen_lane,
    )
    return B300RemoteQualificationCommission(deployment, construction, readiness)


def build_commissioned_b300_qualification_service(
    registration: dict[str, object],
    ready_receipt: dict[str, object],
    capabilities: B300QualificationCapabilities,
    *, commissioned_root: Path | None = None,
) -> CommissionedB300QualificationService:
    inputs, composition, readiness = (
        b300_deployment.replay_commissioned_composition(
            registration, ready_receipt, commissioned_root=commissioned_root
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
        return CommissionedB300QualificationService(
            worker,
            commission,
            reproduction_commission,
            executors,
        )
    except BaseException:
        try:
            if worker is not None:
                worker.close()
        finally:
            for executor in executors:
                executor.manager.close()
        raise


__all__ = [
    "B300QualificationCapabilities",
    "B300QualificationCommissionError",
    "CALIBRATION_PACKAGE_SCHEMA",
    "CommissionedB300QualificationService",
    "build_commissioned_b300_qualification_service",
    "compose_commissioned_qualifications",
    "commissioned_incumbent_arm",
]

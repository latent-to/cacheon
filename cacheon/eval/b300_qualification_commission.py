"""Tracked single-lane qualification commissioning for one pod."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cacheon.eval.b300_deployment as b300_deployment
from cacheon.chain.evaluation_coordinator import WorkerReadiness
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
from cacheon.eval.crossover_runtime import ResidentSpeedPolicy
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


@dataclass
class CommissionedB300QualificationService:
    """One worker, its single qualification lane, and the primary commission.

    Version 19 qualifies each bundle once on the whole lane; there is no
    physical role swap to reproduce, so a reproduction lease fails loudly.
    """

    worker: B300MainnetWorker
    commission: B300RemoteQualificationCommission
    _executor: OCIEngineExecutor
    _lock: object = field(default_factory=threading.RLock)
    _closed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self._executor) is not OCIEngineExecutor
            or self.commission.deployment.screen_lane != "primary"
            or type(self.worker) is not B300MainnetWorker
            or self.worker.service.manifest != self.commission.deployment.manifest
            or self.worker.readiness != self.commission.readiness
        ):
            raise B300QualificationCommissionError(
                "commissioned service does not own its primary qualification lane"
            )

    def adapter_for(self, publications, continuation_store, screen_lane: str):
        with self._lock:
            if self._closed:
                raise B300QualificationCommissionError(
                    "commissioned qualification service is closed"
                )
            if screen_lane != "primary":
                raise B300QualificationCommissionError(
                    "single-lane qualification commissions no reproduction stage"
                )
            return self.commission.adapter_for(
                publications,
                continuation_store,
                worker=self.worker,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        closers = (self.worker.close, self._executor.manager.close)
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
) -> tuple[B300RemoteQualificationCommission, OCIEngineExecutor]:
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
            audit_max_new_tokens=policy_block.get("audit_max_new_tokens", 2),
            audit_control_delta_digests=tuple(
                policy_block.get("audit_control_delta_digests", ())
            ),
        )
    except B300RegisteredQualificationError as exc:
        raise B300QualificationCommissionError(
            f"sealed qualification policy failed to seal: {exc}"
        ) from None

    _require_replay_quality(policy, session_block)

    executor = b300_deployment._build_executor(
        inputs.root / "qualification-lane-a",
        inputs.preflight,
        inputs.device_policy,
        executor_id=QUALIFICATION_EXECUTOR_ID,
        runtime_seed_root=inputs.runtime_seed_root, resources=inputs.authority.get("resources"),
    )
    try:
        commission = _compose_locked(
            inputs,
            manifest,
            composition,
            readiness,
            capabilities,
            block,
            policy,
            catalog,
            hidden_binding,
            executor,
            declared,
            session_block,
            speed_block,
            calibration_loader,
        )
    except BaseException:
        executor.manager.close()
        raise
    return commission, executor


def _require_replay_quality(policy, session_block) -> None:
    """Replay quality is teacher NLL under greedy decoding; a sealed profile
    asking for sampled or numeric hidden-task quality must die at commission,
    not forty minutes into a measured run.
    """

    if policy.topk_width != 0 or float(session_block["temperature"]) != 0:
        raise B300QualificationCommissionError("goodput replay requires greedy decoding and teacher-NLL quality")
    if policy.hidden_tasks_required or policy.hidden_tasks_per_prompt:
        raise B300QualificationCommissionError("replay prompts require a teacher-only profile without numeric hidden tasks")


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
    executor,
    declared,
    session_block,
    speed_block,
    calibration_loader,
) -> B300RemoteQualificationCommission:
    screen_lane = "primary"
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
        closed_loop=bool(goodput.decode_floor),
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
        inputs, manifest, executor,
        entries=capabilities.incumbent_entries, resolver=capabilities.source_resolver,
        replay=replay,
    )
    incumbent_tree = incumbent_binding.tree
    incumbent_launch = incumbent_arm.launch
    baseline_session_plan = incumbent_arm.session_plan
    baseline_physical = incumbent_arm.binding.physical_hardware
    pristine_native = b300_deployment._native_build(
        stock_tree.tree_digest,
        inputs.preflight,
        executor.config.prebuild.policy, inputs.runtime.target_architecture,
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
    resident_speed_policy = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=speed_block["max_stage_seconds"],
        max_qualification_seconds=speed_block["max_qualification_seconds"],
        calibration=calibration_manifest,
        context=calibration_context,
        # New commissions run one engine on the whole lane against a retained
        # stock reference (v19); PD references use ticket-shaped pairs (v21).
        # Historical witnesses retain their original policy and arithmetic.
        version=21 if goodput.decode_floor else 19,
        min_windows=speed_block["min_windows"],
        max_window_scatter=float(speed_block["max_window_scatter"]),
        max_conditioning_slowdown=float(speed_block["max_conditioning_slowdown"]),
        goodput=goodput,
    )

    def bind_candidate(candidate_tree) -> TrustedLaunchBinding:
        candidate_native = b300_deployment._native_build(
            candidate_tree.tree_digest,
            inputs.preflight,
            executor.config.prebuild.policy, inputs.runtime.target_architecture,
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
            resident_speed_policy=resident_speed_policy,
            candidate_executor_namespace_digest=executor.manager.namespace_digest,
            candidate_runtime_resource_policy_digest=executor.config.runtime.digest,
            candidate_device_configuration_digest=(
                executor.device_policy.configuration_sha256
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
        audit_control_delta_digests=policy.audit_control_delta_digests,
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
        executor=executor,
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
    executor: OCIEngineExecutor | None = None
    worker: B300MainnetWorker | None = None
    try:
        commission, executor = compose_commissioned_qualifications(
            inputs, composition, readiness, capabilities
        )
        worker = B300MainnetWorker(
            commission.deployment.manifest,
            commission.deployment.authorities,
            readiness,
        )
        return CommissionedB300QualificationService(worker, commission, executor)
    except BaseException:
        try:
            if worker is not None:
                worker.close()
        finally:
            if executor is not None:
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

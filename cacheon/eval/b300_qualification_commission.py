"""Tracked qualification commissioning for the one existing pod service.

``B300RemoteQualificationCommission`` (the adapter's sealed authority bundle)
previously had no production constructor: the served adapter always started
screen-only and refused qualification as a typed pre-resident failure.  This
module is the tracked construction path.  It replays the same sealed screen
deployment artifacts the commissioned screen worker replays, reads one sealed
``qualification`` commission block from the same authority config, composes
the full :class:`B300QualificationConstructionAuthority` in tracked code, and
returns one commission alongside the same screen worker -- one process, one
resident model lifetime, screen and qualification in the same service.

No request field, environment variable, or file-existence probe selects any
of this.  Everything digest-bearing comes from the sealed authority config,
the sealed prompt identity, the sealed calibration package, and the exact
target catalog.  The only in-process inputs are the validator-private
capabilities (selection secret loader, entropy provider, hidden judge, source
resolver, focused-graph verifier) whose reviewed identities must equal the
digests sealed in the commission block; any drift fails closed before the
adapter accepts a single request.

Identity note: the registered factory's own ``builder_source_digest`` binds
the frozen calibration and reference manifests, which embed the service
digest.  Declaring that self-referential identity inside the manifest would
be circular, so the sealed block carries one *reviewed* builder source
identity and the construction's profile rows are re-bound to it.  The
resolver callables remain exactly the registered factory's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cacheon.eval.b300_screen_deployment as screen_deployment
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.engine_tree import (
    materialize_engine_tree,
    reopen_materialized_engine_tree,
)
from cacheon.eval.b300_mainnet_worker import B300MainnetWorker
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300RegisteredProfileAuthority,
    compose_b300_qualification_deployment,
)
from cacheon.eval.b300_registered_qualification import (
    REGISTERED_B300_TARGET_IDS,
    B300RegisteredQualificationError,
    B300RegisteredQualificationInputs,
    B300RegisteredQualificationPolicy,
    build_b300_registered_qualification_factory,
)
from cacheon.eval.b300_sealed_qualification_commission import (
    QUALIFICATION_DEADLINE_MAXIMUM_SECONDS,
    QUALIFICATION_EVIDENCE_POLICY_DIGEST,
    declared_qualification_deadline_digest,
    declared_qualification_entropy_digest,
    sealed_qualification_profile_rows,
)
from cacheon.eval.b300_remote_worker_adapter import (
    B300RemoteQualificationCommission,
)
from cacheon.eval.calibration import (
    CalibrationContext,
    CalibrationEvidenceSet,
    CalibrationObservation,
    CalibrationThresholdPolicy,
    derive_calibration_manifest,
    publish_calibration_evidence,
)
from cacheon.eval.crossover_runtime import ResidentArmPlan, ResidentSpeedPolicy
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
from cacheon.eval.qualification import ReferenceManifest
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
)
from cacheon.target_catalog import default_target_catalog


CALIBRATION_PACKAGE_SCHEMA = "cacheon-private-b300-calibration-package-v1"

_STAGES = frozenset({"primary", "reproduction"})


class B300QualificationCommissionError(RuntimeError):
    """Sealed qualification commissioning failed closed."""


@dataclass(frozen=True)
class B300QualificationCapabilities:
    """Validator-private callables plus their reviewed sealed identities.

    The callables never enter any digest.  Each reviewed digest must equal the
    matching field of the sealed commission block; the composer refuses any
    substitution.  ``hidden_judge`` must carry the exact sealed
    :class:`HiddenJudgeBinding` from the sealed prompt identity.
    """

    secret_loader: Callable[[str], bytes]
    entropy_provider: object
    hidden_judge: object
    source_resolver: object
    source_resolver_digest: str
    graph_facts_builder: object
    graph_facts_builder_digest: str

    def __post_init__(self) -> None:
        if (
            not callable(self.secret_loader)
            or not callable(self.entropy_provider)
            or not (
                callable(self.hidden_judge)
                or callable(getattr(self.hidden_judge, "bind_prompt_plan", None))
            )
            or not callable(self.graph_facts_builder)
            or not callable(getattr(self.source_resolver, "resolve_proposal", None))
            or not callable(getattr(self.source_resolver, "resolve_integrated", None))
        ):
            raise B300QualificationCommissionError(
                "qualification capabilities are not callable"
            )
        if type(getattr(self.hidden_judge, "binding", None)) is not HiddenJudgeBinding:
            raise B300QualificationCommissionError(
                "hidden judge capability lacks an exact sealed binding"
            )
        for field in ("source_resolver_digest", "graph_facts_builder_digest"):
            value = getattr(self, field)
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise B300QualificationCommissionError(
                    f"capability {field} is not one SHA-256 identity"
                )


def _bind_hidden_judge(
    capability: object,
    *,
    binding: HiddenJudgeBinding,
    tokenizer_digest: str,
    prompt_batches: tuple[tuple[str, ...], ...],
    workload_digest: str,
    hidden_tasks_per_prompt: int,
) -> object:
    """Bind a deferred judge authority to the exact composed prompt plan."""

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


@dataclass
class CommissionedB300QualificationService:
    """One screen worker and one qualification commission from one replay."""

    worker: B300MainnetWorker
    commission: B300RemoteQualificationCommission
    _executors: tuple[OCIEngineExecutor, ...]
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.worker.close()
        finally:
            for executor in self._executors:
                executor.manager.close()


def _tracked_deadline_provider(
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[object], float]:
    """Lease-bounded monotonic deadline; the sealed declared policy."""

    def deadline(_cohort: object) -> float:
        return float(clock()) + float(QUALIFICATION_DEADLINE_MAXIMUM_SECONDS)

    return deadline


def _require_complete_factory_profiles(
    profiles: object,
) -> dict[str, B300RegisteredProfileAuthority]:
    """Reject partial/stale factory registries before any candidate can run."""

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


def _sealed_calibration(
    inputs: "screen_deployment._CommissionedInputs",
) -> tuple[CalibrationThresholdPolicy, tuple[CalibrationObservation, ...]]:
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
    if (
        type(value) is not dict
        or set(value) != {"observations", "schema", "threshold_policy"}
        or value.get("schema") != CALIBRATION_PACKAGE_SCHEMA
        or type(value.get("observations")) is not list
    ):
        raise B300QualificationCommissionError(
            "sealed calibration package is not one closed frozen authority"
        )
    try:
        threshold = CalibrationThresholdPolicy.from_dict(value["threshold_policy"])
        observations = tuple(
            CalibrationObservation.from_dict(row) for row in value["observations"]
        )
    except (TypeError, ValueError) as exc:
        raise B300QualificationCommissionError(
            f"sealed calibration package is invalid: {exc}"
        ) from None
    return threshold, observations


def _lane_gpus(
    inputs: "screen_deployment._CommissionedInputs",
) -> tuple[tuple, tuple]:
    """(baseline lane, candidate lane) — baseline shares the screen lane."""

    selected = {gpu.physical_id for gpu in inputs.gpus}
    complement = tuple(
        sorted(
            (
                gpu
                for gpu in inputs.qualification_gpus
                if gpu.physical_id not in selected
            ),
            key=lambda gpu: gpu.physical_id,
        )
    )
    if len(complement) != len(inputs.gpus):
        raise B300QualificationCommissionError(
            "commissioned lanes do not form one disjoint TP4 pair"
        )
    return inputs.gpus, complement


def compose_commissioned_qualification(
    inputs: "screen_deployment._CommissionedInputs",
    composition: "screen_deployment._Composition",
    readiness: WorkerReadiness,
    capabilities: B300QualificationCapabilities,
) -> tuple[B300RemoteQualificationCommission, tuple[OCIEngineExecutor, ...]]:
    """Compose one exact qualification commission from replayed sealed inputs."""

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
    ):
        raise B300QualificationCommissionError(
            "capability identities differ from the sealed commission block"
        )
    screen_lane = inputs.authority.get("authority_role")
    if screen_lane not in _STAGES:
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

    _baseline_gpus, candidate_gpus = _lane_gpus(inputs)
    baseline_policy = inputs.device_policy
    candidate_policy = screen_deployment._device_policy(candidate_gpus)
    lane_pair = inputs.qualification_lane_pair
    if {baseline_policy.policy_sha256, candidate_policy.policy_sha256} != {
        lane_pair.lane_a.device_policy_digest,
        lane_pair.lane_b.device_policy_digest,
    }:
        raise B300QualificationCommissionError(
            "commissioned lanes differ from the sealed qualification lane pair"
        )
    candidate_executor = screen_deployment._build_executor(
        inputs.root,
        inputs.preflight,
        candidate_policy,
        executor_id="b300-qualification-candidate",
    )
    baseline_executor = screen_deployment._build_executor(
        inputs.root,
        inputs.preflight,
        baseline_policy,
        executor_id="b300-qualification-resident",
    )
    executors = (candidate_executor, baseline_executor)
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
            candidate_executor,
            baseline_executor,
            screen_lane,
            declared,
            session_block,
            speed_block,
        )
    except BaseException:
        for executor in executors:
            executor.manager.close()
        raise
    return commission, executors


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
) -> B300RemoteQualificationCommission:
    # Stock incumbent identity: mainnet cold start has no accepted incumbent
    # contribution, so B and pristine T are the same sealed empty stack.  The
    # engine tree location matches the resident screen factory's stock tree so
    # both lifetimes share one immutable materialization.
    snapshot = catalog.snapshot()
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
        device_policy_digest=baseline_executor.device_policy.policy_sha256,
    )
    baseline_physical = PhysicalHardwareBinding(
        physical_gpu_ids=tuple(
            str(gpu.physical_id)
            for gpu in baseline_executor.device_policy.expected_gpus
        ),
        architecture=screen_deployment.ARCHITECTURE,
        topology_class=inputs.runtime.topology_class,
        topology_digest=inputs.topology_digest,
        tp_size=screen_deployment.TP_SIZE,
        ep_size=1,
        dp_size=1,
        device_policy_digest=baseline_executor.device_policy.policy_sha256,
    )
    native = screen_deployment._native_build(
        tree.tree_digest,
        inputs.preflight,
        baseline_executor.config.prebuild.policy,
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
            baseline_executor.config.prebuild.seccomp_profile
        ),
        resource_policy_digest=(
            baseline_executor.config.prebuild.policy.resource_policy_digest
        ),
        native_build_spec_digest=native.digest,
        hardware=baseline_hardware,
    )
    trusted_baseline = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=inputs.controller_distribution_digest,
        native_build_spec=native,
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
        incumbent_launch,
        incumbent_binding,
        workload_digest=workload_digest,
        tokenizer_digest=inputs.prompt_identity["tokenizer_digest"],
        hidden_corpus_commitment=hidden_binding.hidden_corpus_commitment,
        hidden_judge_digest=hidden_binding.hidden_judge_digest,
        selection_policy_digest=inputs.prompt_identity["selection_policy_digest"],
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
        policy.verification_policy_digest,
        reference.controller_distribution_digest,
    )
    threshold, observations = _sealed_calibration(inputs)
    if threshold.context != calibration_context:
        raise B300QualificationCommissionError(
            "sealed calibration context differs from the commissioned reference"
        )
    evidence_root = _private_root(inputs.root / "qualification-evidence")
    materialization_root = _private_root(inputs.root / "qualification-candidates")
    calibration_manifest = derive_calibration_manifest(threshold, observations)
    calibration_ref = publish_calibration_evidence(
        evidence_root,
        CalibrationEvidenceSet.create(threshold, observations),
    )
    resident_baseline_arm = ResidentArmPlan(
        incumbent_launch,
        trusted_baseline,
        baseline_session_plan,
        baseline_executor.manager.namespace_digest,
        baseline_executor.config.runtime.digest,
        baseline_executor.device_policy.configuration_sha256,
    )
    resident_speed_policy = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=speed_block["max_stage_seconds"],
        max_qualification_seconds=speed_block["max_qualification_seconds"],
        calibration=calibration_manifest,
        context=calibration_context,
        version=3,
        min_windows=speed_block["min_windows"],
        max_window_scatter=float(speed_block["max_window_scatter"]),
        max_conditioning_slowdown=float(speed_block["max_conditioning_slowdown"]),
    )

    candidate_physical = PhysicalHardwareBinding(
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
            physical_hardware=candidate_physical,
        )

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
            pristine_launch=incumbent_launch,
            pristine_session_plan=baseline_session_plan,
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
        )
        factory = build_b300_registered_qualification_factory(factory_inputs)
    except B300RegisteredQualificationError as exc:
        raise B300QualificationCommissionError(
            f"registered qualification factory failed to compose: {exc}"
        ) from None

    # Re-bind the profile rows to the sealed reviewed builder identity.  The
    # factory's own self-derived identity stays internal (it embeds the frozen
    # calibration/reference manifests, which embed the service digest and can
    # therefore never appear inside the manifest's declared digests).
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
        screen_lane=screen_lane,
    )
    return B300RemoteQualificationCommission(deployment, construction, readiness)


def build_commissioned_b300_qualification_service(
    registration: dict[str, object],
    ready_receipt: dict[str, object],
    capabilities: B300QualificationCapabilities,
) -> CommissionedB300QualificationService:
    """Replay the sealed deployment once into screen worker plus commission."""

    inputs, composition, readiness = (
        screen_deployment.replay_commissioned_screen_composition(
            registration, ready_receipt
        )
    )
    executors: tuple[OCIEngineExecutor, ...] = ()
    try:
        commission, executors = compose_commissioned_qualification(
            inputs, composition, readiness, capabilities
        )
        worker = screen_deployment.commissioned_screen_worker_from_composition(
            composition, readiness
        )
        composition = None
        return CommissionedB300QualificationService(worker, commission, executors)
    except BaseException:
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
    "compose_commissioned_qualification",
]

"""Compose the resident execution probe without loading candidate code on the host."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable
from cacheon.arena_service import ArenaServiceManifest
from cacheon.eval.b300_arena_definition import (
    B300ScreenDeploymentError, data_parallel_size as _data_parallel_size,
    engine_config as _engine_config, scored_cell as _scored_cell,
)
from cacheon.eval.b300_arena_provider import B300ResidentScreenFactory, B300ResidentScreenLifetime
from cacheon.eval.engine_launch import EngineLaunchSpec, LogicalHardwareSpec, PhysicalHardwareBinding, TrustedLaunchBinding
from cacheon.eval.oci_backend import OCIEngineExecutor, TrustedArenaModelMountReceipt, expected_runtime_preflight
from cacheon.eval.oci_resident_session import ResidentSessionPlan
from cacheon.eval.resident_queue import ScreenPolicy
from cacheon.eval.resident_screen_lane import ResidentScreenLane, ResidentServingScreenStage, make_backend_lifetime_factory
from cacheon.target_catalog import TargetCatalog

if TYPE_CHECKING:
    from cacheon.eval.b300_screen_deployment import _CommissionedInputs


def _resident_factory(
    inputs: _CommissionedInputs,
    executor: OCIEngineExecutor,
    catalog: TargetCatalog,
    manifest_provider: Callable[[], ArenaServiceManifest],
) -> B300ResidentScreenFactory:
    """Build one stock TP4 engine lifetime shared by queued arrivals."""

    from cacheon.eval.b300_screen_deployment import (
        ARCHITECTURE, GPU_COUNT, TP_SIZE, _commissioned_stock_authority,
        _file_sha256, _native_build,
    )

    if type(executor) is not OCIEngineExecutor or type(catalog) is not TargetCatalog:
        raise B300ScreenDeploymentError("resident factory inputs are not exact")
    if not callable(manifest_provider):
        raise B300ScreenDeploymentError("resident manifest provider is not callable")
    prompts = tuple(prompt for batch in inputs.prompt_batches[:1] for prompt in batch)

    def create() -> B300ResidentScreenLifetime:
        manifest = manifest_provider()
        if (
            type(manifest) is not ArenaServiceManifest
            or manifest.runtime != inputs.runtime
        ):
            raise B300ScreenDeploymentError(
                "resident service manifest differs from commission"
            )
        snapshot = catalog.snapshot()
        target_members, _context, _stock, tree = _commissioned_stock_authority(
            inputs,
            manifest,
            catalog,
            snapshot,
            error=B300ScreenDeploymentError,
            label="resident",
        )

        policy = inputs.device_policy
        dp_size = _data_parallel_size(inputs.engine_template)
        hardware = LogicalHardwareSpec(
            visible_gpu_count=GPU_COUNT,
            architecture=ARCHITECTURE,
            topology_class=inputs.runtime.topology_class,
            topology_digest=inputs.topology_digest,
            tp_size=TP_SIZE,
            ep_size=1,
            dp_size=dp_size,
            device_policy_digest=policy.policy_sha256,
        )
        physical = PhysicalHardwareBinding(
            physical_gpu_ids=tuple(str(gpu.physical_id) for gpu in inputs.gpus),
            architecture=ARCHITECTURE,
            topology_class=inputs.runtime.topology_class,
            topology_digest=inputs.topology_digest,
            tp_size=TP_SIZE,
            ep_size=1,
            dp_size=dp_size,
            device_policy_digest=policy.policy_sha256,
        )
        native = _native_build(
            tree.tree_digest,
            inputs.preflight,
            executor.config.prebuild.policy,
        )
        binding = TrustedLaunchBinding(
            materialized_tree_root=tree.root,
            controller_distribution_digest=inputs.controller_distribution_digest,
            native_build_spec=native,
            runtime_preflight_receipt=inputs.preflight,
            physical_hardware=physical,
        )
        config = _engine_config(
            inputs.engine_template,
            target_members,
            _scored_cell(inputs.workload),
            disable_cuda_graph=False,
        )
        launch = EngineLaunchSpec(
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
            engine_config_digest=config.digest,
            seccomp_policy_digest=_file_sha256(
                executor.config.prebuild.seccomp_profile
            ),
            resource_policy_digest=(
                executor.config.prebuild.policy.resource_policy_digest
            ),
            native_build_spec_digest=native.digest,
            hardware=hardware,
        )
        mount = TrustedArenaModelMountReceipt.capture(
            inputs.model_root,
            arena_digest=manifest.digest,
            model_revision_digest=inputs.runtime.model_revision_digest,
            model_manifest_digest=inputs.runtime.model_manifest_digest,
            model_content_digest=inputs.runtime.model_content_digest,
        )
        plan = ResidentSessionPlan(
            launch_digest=launch.digest,
            expected_engine_config_digest=config.digest,
            engine_config=config,
            expected_preflight=expected_runtime_preflight(
                launch, inputs.preflight
            ),
            max_swaps=10_000,
            max_batches=100_000,
            max_new_tokens=4,
            top_logprobs_num=0,
            temperature=0.0,
        )
        swap_root = inputs.root / "resident-intake"
        # Owner retains write for host-side staging; other/execute lets the
        # non-root OCI --user traverse to a known digest. mode=0o700 made the
        # intake root pass ST_RDONLY preflight while digest lstat failed with
        # EACCES, surfaced as "staged swap bundle is absent or writable"
        # (mainnet FIFO 2026-08-04 after CUDA-graph capture completed).
        swap_root.mkdir(parents=True, exist_ok=True, mode=0o711)
        os.chmod(swap_root, 0o711)
        lifetime = make_backend_lifetime_factory(
            executor,
            launch,
            binding,
            mount,
            plan,
            swap_intake_root=swap_root,
            deadline_provider=lambda: float(executor.manager.clock())
            + 30 * 24 * 60 * 60,
        )
        lane = ResidentScreenLane(
            lifetime,
            prompts=prompts,
            policy=ScreenPolicy(max_candidates_per_lifetime=1_000),
            verdict_timeout_s=3600.0,
            close_timeout_s=1800.0,
        )
        stage = ResidentServingScreenStage(lane, swap_root)
        return B300ResidentScreenLifetime(stage, lane.close)

    return B300ResidentScreenFactory(
        inputs.resident_factory_digest,
        inputs.resident_resource_ids,
        create,
    )

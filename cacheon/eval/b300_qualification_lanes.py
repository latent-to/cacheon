"""Physical qualification lane authority for a commissioned arena.

Version-19 qualification runs one engine at a time on the whole commissioned
allocation: the retained stock reference, the candidate boots, and the stock
bracket all execute on this one lane.  ``b300_arena_provider`` composes and
re-exports these names, so import paths are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from cacheon.arena_service import ArenaRuntimeIdentity
from cacheon._strict import require_digest
from cacheon.eval.device_state import DeviceStatePolicy
from cacheon.eval.device_policy_identity import logical_device_policy_digest
from cacheon.stack_identity import canonical_digest


QUALIFICATION_LANE_SCHEMA = "cacheon.eval.b300-qualification-lane.v1"


class B300ArenaProviderError(RuntimeError):
    """A deployment authority or provider lifecycle is inconsistent."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=B300ArenaProviderError)


@dataclass(frozen=True)
class B300QualificationLanePolicy:
    """The one physical allocation every qualification engine runs on."""

    lane_id: str
    physical_gpu_ids: tuple[int, ...]
    gpu_uuids: tuple[str, ...]
    device_configuration_digest: str
    device_policy_digest: str
    logical_device_policy_digest: str

    def __post_init__(self) -> None:
        if self.lane_id != "A":
            raise B300ArenaProviderError("qualification lane id must be A")
        physical_ids = self.physical_gpu_ids
        uuids = self.gpu_uuids
        if (
            type(physical_ids) is not tuple
            or not physical_ids
            or any(type(row) is not int or row < 0 for row in physical_ids)
            or physical_ids != tuple(sorted(set(physical_ids)))
            or type(uuids) is not tuple
            or len(uuids) != len(physical_ids)
            or len(set(uuids)) != len(physical_ids)
            or any(
                not isinstance(row, str)
                or not row
                or row.strip() != row
                or len(row) > 128
                or any(character in row for character in "\x00\r\n")
                for row in uuids
            )
        ):
            raise B300ArenaProviderError(
                "qualification lane must bind one canonical physical allocation"
            )
        for field in ("device_configuration_digest", "device_policy_digest", "logical_device_policy_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))

    @classmethod
    def from_device_policy(
        cls,
        lane_id: str,
        policy: DeviceStatePolicy,
    ) -> "B300QualificationLanePolicy":
        if type(policy) is not DeviceStatePolicy:
            raise B300ArenaProviderError("qualification lane policy is not exact")
        gpus = policy.expected_gpus
        return cls(
            lane_id,
            tuple(gpu.physical_id for gpu in gpus),
            tuple(gpu.uuid for gpu in gpus),
            policy.configuration_sha256,
            policy.policy_sha256,
            logical_device_policy_digest(policy),
        )

    def validate_runtime(self, runtime: ArenaRuntimeIdentity) -> None:
        """Bind the allocation to the serving geometry before any execution."""
        if (
            type(runtime) is not ArenaRuntimeIdentity
            or runtime.gpu_count != len(self.physical_gpu_ids)
            or runtime.tensor_parallel_size != runtime.gpu_count
        ):
            raise B300ArenaProviderError(
                "qualification lane differs from the commissioned TP runtime"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "device_configuration_digest": self.device_configuration_digest,
            "device_policy_digest": self.device_policy_digest,
            "logical_device_policy_digest": self.logical_device_policy_digest,
            "gpu_uuids": list(self.gpu_uuids),
            "lane_id": self.lane_id,
            "physical_gpu_ids": list(self.physical_gpu_ids),
        }

    def service_policy(self) -> dict[str, object]:
        """Bind the shared competition while each worker retains its physical lane."""
        return {
            "lane_id": self.lane_id,
            "gpu_count": len(self.physical_gpu_ids),
            "device_policy_digest": self.logical_device_policy_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(QUALIFICATION_LANE_SCHEMA, self.to_dict())


def commissioned_incumbent_arm(inputs, manifest, executor, *, entries, resolver, replay):
    """Bind the commissioned incumbent to one executor without grading it.

    Qualification and a single-lane development replay share this construction.
    Calibration remains mandatory in the qualification consumer, after its
    reference context exists; constructing an engine never fabricates it.
    """
    from dataclasses import replace
    import cacheon.eval.b300_deployment as b300_deployment
    from cacheon.eval.b300_arena_definition import (
        data_parallel_size as _data_parallel_size,
        hardware_bindings as _hardware_bindings, scored_cell as _scored_cell,
    )
    from cacheon.eval.b300_sealed_qualification_commission import B300QualificationCommissionError
    from cacheon.eval.crossover_runtime import ResidentArmPlan
    from cacheon.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
    from cacheon.eval.marginal_runtime import MaterializedArmBinding
    from cacheon.eval.oci_backend import expected_runtime_preflight
    from cacheon.eval.oci_outer_session import SessionExecutionPlan
    from cacheon.target_catalog import default_target_catalog

    catalog = default_target_catalog()
    snapshot = catalog.snapshot()
    policy_block = inputs.qualification_commission["policy"]
    session_block = inputs.qualification_commission["session"]
    # The measured baseline is the durable incumbent the capabilities declare;
    # at genesis the declared entries are empty and this reopens the exact
    # stock tree above, so both arms of the branchless pair coincide.
    _, _, incumbent, incumbent_tree = (
        b300_deployment._commissioned_stock_authority(
            inputs,
            manifest,
            catalog,
            snapshot,
            error=B300QualificationCommissionError,
            label="qualification",
            entries=entries,
            resolver=resolver,
        )
    )
    engine_config = replace(inputs.engine_template, disable_cuda_graph=False)
    dp_size = _data_parallel_size(engine_config)
    baseline_hardware, baseline_physical = _hardware_bindings(
        inputs.runtime, executor.device_policy, dp_size=dp_size,
    )
    incumbent_native = b300_deployment._native_build(
        incumbent_tree.tree_digest,
        inputs.preflight,
        executor.config.prebuild.policy, inputs.runtime.target_architecture,
    )
    incumbent_launch = EngineLaunchSpec(
        runtime_digest=inputs.runtime.runtime_digest,
        base_engine_digest=inputs.runtime.base_engine_digest,
        arena_digest=manifest.digest,
        stack_digest=incumbent_tree.stack_digest,
        tree_digest=incumbent_tree.tree_digest,
        image_digest=inputs.preflight.image_digest,
        platform_digest=inputs.preflight.platform_digest,
        controller_distribution_digest=inputs.controller_distribution_digest,
        worker_distribution_digest=inputs.preflight.worker_distribution_digest,
        model_revision_digest=inputs.runtime.model_revision_digest,
        model_manifest_digest=inputs.runtime.model_manifest_digest,
        model_content_digest=inputs.runtime.model_content_digest,
        validator_overlay_digest=inputs.runtime.validator_overlay_digest,
        engine_config_digest=engine_config.digest,
        seccomp_policy_digest=b300_deployment._file_sha256(
            executor.config.prebuild.seccomp_profile
        ),
        resource_policy_digest=(
            executor.config.prebuild.policy.resource_policy_digest
        ),
        native_build_spec_digest=incumbent_native.digest,
        hardware=baseline_hardware,
    )
    trusted_baseline = TrustedLaunchBinding(
        materialized_tree_root=incumbent_tree.root,
        controller_distribution_digest=inputs.controller_distribution_digest,
        native_build_spec=incumbent_native,
        runtime_preflight_receipt=inputs.preflight,
        physical_hardware=baseline_physical,
    )
    incumbent_binding = MaterializedArmBinding(incumbent_tree, trusted_baseline)
    quality_cell = _scored_cell(inputs.workload)
    cells_by_id = {cell.cell_id: cell for cell in inputs.workload.cells}
    # The old128/24-request conditioning cost229k decode tokens before a
    # two-session replay. Warm the engine at its declared replay load.
    prompts = tuple(batch[:max(replay.loads)] for batch in inputs.prompt_batches[:session_block["warmup_count"]])
    batch_cells = inputs.prompt_batch_cells[:len(prompts)]
    warmup_tokens = 16
    baseline_session_plan = SessionExecutionPlan(
        launch_digest=incumbent_launch.digest,
        expected_engine_config_digest=engine_config.digest,
        engine_config=engine_config,
        expected_preflight=expected_runtime_preflight(
            incumbent_launch, inputs.preflight
        ),
        prompt_batches=prompts,
        warmup_count=session_block["warmup_count"],
        conditioning_count=session_block["conditioning_count"],
        max_new_tokens=policy_block["tokens_per_prompt"],
        top_logprobs_num=policy_block["topk_width"],
        temperature=float(session_block["temperature"]),
        expected_prompt_tokens=quality_cell.input_tokens,
        measure_phase_latency=session_block.get("measure_phase_latency", False),
        replay=replay,
        batch_max_new_tokens=tuple(warmup_tokens for _ in batch_cells),
        batch_expected_prompt_tokens=tuple(cells_by_id[cell_id].input_tokens for cell_id in batch_cells),
    )
    return incumbent, incumbent_binding, ResidentArmPlan(
        incumbent_launch, trusted_baseline, baseline_session_plan,
        executor.manager.namespace_digest, executor.config.runtime.digest,
        executor.device_policy.configuration_sha256,
    )

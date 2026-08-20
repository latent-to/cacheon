"""Bind marginal stack plans to the generic isolated engine executor.

This module is deliberately only identity and lifecycle plumbing.  It derives
every candidate launch from one incumbent launch, derives every session plan
from one incumbent workload, executes ``B,C1..Ck,B'`` under one absolute
deadline, and returns raw execution evidence.  It does not score, retry,
qualify, select, crown, or mutate a stack.

``MaterializedArmBinding`` is a trusted host-local value.  Its tree must be the
live result returned by the validator materializer, not a miner-supplied or
deserialized assertion.  Finalized hostile intake and durable worker-tree
publication remain a separate control-plane boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

from cacheon.engine_tree import MaterializedEngineTree
from cacheon.eval.engine_launch import (
    EngineLaunchError,
    EngineLaunchSpec,
    TrustedLaunchBinding,
    resolve_engine_launch,
)
from cacheon.eval.oci_backend import (
    expected_runtime_preflight,
)
from cacheon.eval.oci_outer_session import (
    OuterSessionWorkerError,
    SessionExecutionPlan,
)
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    IntegratedContributionRef,
    ProposalContributionRef,
)
from cacheon.stack_plan import CohortPlan, MarginalArmPlan, StackPlanError
from cacheon.target_catalog import TargetCatalog
from cacheon._strict import require_digest


_TREE_METADATA = "metadata/cacheon_engine_tree.json"


class MarginalRuntimeError(ValueError):
    """A marginal plan cannot be bound to one exact runtime lifecycle."""


ExecutableArm = MarginalArmPlan
RuntimeSource = MarginalArmPlan | CohortPlan


def _digest(value: object, *, field: str) -> str:
    return require_digest(value, field=field, error=MarginalRuntimeError)


class CandidateArmWorkerError(RuntimeError):
    """One valid worker error emitted while an exact C arm was active."""

    def __init__(
        self,
        *,
        candidate_index: int,
        selected_delta_digest: str,
        arm_digest: str,
        launch_digest: str,
        worker_error: OuterSessionWorkerError,
    ) -> None:
        if type(candidate_index) is not int or candidate_index < 0:
            raise MarginalRuntimeError("candidate worker index is malformed")
        if type(worker_error) is not OuterSessionWorkerError:
            raise MarginalRuntimeError("candidate worker failure is not exactly typed")
        self.candidate_index = candidate_index
        self.selected_delta_digest = _digest(
            selected_delta_digest, field="candidate worker selected delta"
        )
        self.arm_digest = _digest(arm_digest, field="candidate worker arm")
        self.launch_digest = _digest(launch_digest, field="candidate worker launch")
        self.worker_error = worker_error
        super().__init__(
            "candidate arm worker failed "
            f"at index {candidate_index} for delta {self.selected_delta_digest}, "
            f"arm {self.arm_digest}, launch {self.launch_digest}: {worker_error}"
        )


def _native_environment(binding: TrustedLaunchBinding) -> dict[str, object]:
    """Return only the arm-invariant native toolchain authority.

    ``tree_digest`` is necessarily arm-specific.  A direct CuTe-AOT candidate also
    upgrades its native build from schema 1 to schema 2: the compile-profile digest
    and the compiler-policy digest that incorporates it are therefore candidate
    build inputs, not evidence that the image/toolchain changed.  Each complete
    ``NativeBuildSpec`` is already self-validating, and ``resolve_engine_launch``
    separately validates a trusted compile profile against the common launch
    hardware.  Keep comparing the immutable image, platform, worker, toolchain,
    patcher, architecture, and dependency policy exactly.
    """

    native = binding.native_build_spec
    return {
        "dependency_policy_digest": native.dependency_policy_digest,
        "image_digest": native.image_digest,
        "patcher_digest": native.patcher_digest,
        "platform_digest": native.platform_digest,
        "target_architecture": native.target_architecture,
        "toolchain_digest": native.toolchain_digest,
        "worker_distribution_digest": native.worker_distribution_digest,
    }


def _expected_contributions(
    stack: EvaluationStackManifest,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_id, ref in sorted(stack.entries.items()):
        if type(ref) is ProposalContributionRef:
            source_kind = "proposal_artifact"
            source_digest = ref.artifact_digest
        elif type(ref) is IntegratedContributionRef:
            source_kind = "integrated_source"
            source_digest = ref.integrated_source_tree_digest
        else:  # pragma: no cover - EvaluationStackManifest is already closed
            raise MarginalRuntimeError("stack contains an unsupported contribution ref")
        rows.append(
            {
                "contribution_ref_digest": ref.digest,
                "namespace": f"cacheon_c_{ref.selected_delta_digest}",
                "selected_delta_digest": ref.selected_delta_digest,
                "selected_payload_digest": ref.selected_payload_digest,
                "source_digest": source_digest,
                "source_kind": source_kind,
                "target_id": target_id,
                "target_spec_digest": ref.target_spec_digest,
            }
        )
    return rows


def _tree_metadata(tree: MaterializedEngineTree) -> dict[str, object]:
    metadata_row = next((row for row in tree.files if row.path == _TREE_METADATA), None)
    if metadata_row is None:
        raise MarginalRuntimeError("materialized tree lacks its metadata inventory")
    path = tree.root / _TREE_METADATA
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MarginalRuntimeError(f"cannot reopen materialized tree metadata: {exc}") from None
    if (
        len(data) != metadata_row.size
        or hashlib.sha256(data).hexdigest() != metadata_row.sha256
    ):
        raise MarginalRuntimeError("materialized tree metadata differs from trusted inventory")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise MarginalRuntimeError(f"materialized tree metadata is invalid: {exc}") from None
    if not isinstance(value, dict):
        raise MarginalRuntimeError("materialized tree metadata must be an object")
    return value


def _require_tree_stack(
    tree: MaterializedEngineTree,
    stack: EvaluationStackManifest,
    *,
    expected_tree_digest: str,
) -> None:
    """Bind a live trusted materializer result to its exact stack inventory."""

    if type(tree) is not MaterializedEngineTree:
        raise MarginalRuntimeError("tree binding is not a MaterializedEngineTree")
    expected_tree = _digest(expected_tree_digest, field="expected tree_digest")
    if tree.stack_digest != stack.digest or tree.tree_digest != expected_tree:
        raise MarginalRuntimeError("materialized tree identity differs from its stack arm")
    metadata = _tree_metadata(tree)
    if metadata.get("stack_digest") != stack.digest:
        raise MarginalRuntimeError("materialized tree metadata names another stack")
    if metadata.get("contributions") != _expected_contributions(stack):
        raise MarginalRuntimeError(
            "materialized tree contribution inventory differs from its stack manifest"
        )
    expected_manifest = "manifest.toml" if stack.entries else None
    if tree.runtime_manifest != expected_manifest or metadata.get(
        "runtime_manifest"
    ) != expected_manifest:
        raise MarginalRuntimeError("materialized tree runtime manifest differs from its stack")


@dataclass(frozen=True)
class MaterializedArmBinding:
    """Trusted in-process materializer output plus its host-local launch binding."""

    tree: MaterializedEngineTree
    launch_binding: TrustedLaunchBinding

    def __post_init__(self) -> None:
        if type(self.tree) is not MaterializedEngineTree:
            raise MarginalRuntimeError("tree must be an exact MaterializedEngineTree")
        if type(self.launch_binding) is not TrustedLaunchBinding:
            raise MarginalRuntimeError("launch_binding must be a TrustedLaunchBinding")


@dataclass(frozen=True)
class PreparedCandidateRuntime:
    """One C arm bound to a mechanically derived launch and common workload."""

    arm: ExecutableArm
    binding: MaterializedArmBinding
    launch: EngineLaunchSpec
    session_plan: SessionExecutionPlan

    def __post_init__(self) -> None:
        if type(self.arm) is not MarginalArmPlan:
            raise MarginalRuntimeError("candidate arm has an unsupported type")
        if type(self.binding) is not MaterializedArmBinding:
            raise MarginalRuntimeError("candidate binding has the wrong type")
        if type(self.launch) is not EngineLaunchSpec:
            raise MarginalRuntimeError("candidate launch has the wrong type")
        if type(self.session_plan) is not SessionExecutionPlan:
            raise MarginalRuntimeError("candidate session plan has the wrong type")
        if (
            self.launch.stack_digest != self.arm.challenger.stack_digest
            or self.launch.tree_digest != self.arm.challenger.tree_digest
            or self.session_plan.launch_digest != self.launch.digest
        ):
            raise MarginalRuntimeError("candidate runtime does not bind its marginal arm")

@dataclass(frozen=True)
class PreparedMarginalRuntime:
    """A completely validated B,C1..Ck,B-prime runtime lifecycle."""

    source: RuntimeSource
    incumbent_binding: MaterializedArmBinding
    baseline_launch: EngineLaunchSpec
    baseline_session_plan: SessionExecutionPlan
    candidates: tuple[PreparedCandidateRuntime, ...]

    def __post_init__(self) -> None:
        if type(self.source) not in {MarginalArmPlan, CohortPlan}:
            raise MarginalRuntimeError("runtime source has an unsupported type")
        if type(self.incumbent_binding) is not MaterializedArmBinding:
            raise MarginalRuntimeError("incumbent binding has the wrong type")
        if type(self.baseline_launch) is not EngineLaunchSpec:
            raise MarginalRuntimeError("baseline launch has the wrong type")
        if type(self.baseline_session_plan) is not SessionExecutionPlan:
            raise MarginalRuntimeError("baseline session plan has the wrong type")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or any(
            type(candidate) is not PreparedCandidateRuntime
            for candidate in self.candidates
        ):
            raise MarginalRuntimeError("prepared runtime requires typed candidates")
        expected = (
            (self.source,)
            if type(self.source) is MarginalArmPlan
            else self.source.execution_arms
        )
        if tuple(candidate.arm for candidate in self.candidates) != expected:
            raise MarginalRuntimeError("candidate order differs from the sealed runtime source")
        if (
            self.baseline_launch.stack_digest
            != self.candidates[0].arm.baseline_before.stack_digest
            or self.baseline_launch.tree_digest
            != self.candidates[0].arm.baseline_before.tree_digest
        ):
            raise MarginalRuntimeError("baseline launch differs from the frozen incumbent")
        if self.baseline_session_plan.launch_digest != self.baseline_launch.digest:
            raise MarginalRuntimeError("baseline session plan names another launch")
        _validate_prepared_runtime(self)

def _require_context(
    launch: EngineLaunchSpec, expected_context: EvaluationStackContext
) -> None:
    if type(expected_context) is not EvaluationStackContext:
        raise MarginalRuntimeError("expected_context has the wrong type")
    observed = (
        launch.runtime_digest,
        launch.base_engine_digest,
        launch.arena_digest,
    )
    expected = (
        expected_context.runtime_digest,
        expected_context.base_engine_digest,
        expected_context.arena_digest,
    )
    if observed != expected:
        raise MarginalRuntimeError("baseline launch differs from the frozen stack context")


def _require_resolved_tree(
    launch: EngineLaunchSpec,
    binding: MaterializedArmBinding,
) -> None:
    try:
        resolved = resolve_engine_launch(launch, binding.launch_binding)
    except (EngineLaunchError, OSError, TypeError, ValueError) as exc:
        raise MarginalRuntimeError(f"engine launch binding failed: {exc}") from None
    try:
        trusted_root = binding.tree.root.resolve(strict=True)
        resolved_root = resolved.materialized_tree.root.resolve(strict=True)
    except OSError as exc:
        raise MarginalRuntimeError(f"materialized tree root is unavailable: {exc}") from None
    if trusted_root != resolved_root or (
        resolved.materialized_tree.stack_digest,
        resolved.materialized_tree.tree_digest,
        resolved.materialized_tree.files,
        resolved.materialized_tree.runtime_manifest,
    ) != (
        binding.tree.stack_digest,
        binding.tree.tree_digest,
        binding.tree.files,
        binding.tree.runtime_manifest,
    ):
        raise MarginalRuntimeError(
            "reopened engine tree differs from the trusted materializer result"
        )


def _resolve_materialized_binding(
    launch: EngineLaunchSpec,
    binding: MaterializedArmBinding,
    stack: EvaluationStackManifest,
    *,
    expected_tree_digest: str,
) -> None:
    _require_tree_stack(
        binding.tree,
        stack,
        expected_tree_digest=expected_tree_digest,
    )
    _require_resolved_tree(launch, binding)


def _require_baseline_session(
    launch: EngineLaunchSpec,
    binding: TrustedLaunchBinding,
    plan: SessionExecutionPlan,
) -> None:
    if type(plan) is not SessionExecutionPlan:
        raise MarginalRuntimeError("baseline session plan has the wrong type")
    receipt = binding.runtime_preflight_receipt
    if type(receipt) is not RuntimePreflightReceipt:
        raise MarginalRuntimeError("baseline binding lacks a typed runtime preflight")
    expected = expected_runtime_preflight(launch, receipt)
    if (
        plan.launch_digest != launch.digest
        or plan.expected_engine_config_digest != launch.engine_config_digest
        or plan.engine_config.digest != launch.engine_config_digest
        or plan.engine_config.tp_size != launch.hardware.tp_size
        or plan.expected_preflight != expected
        or plan.expected_discovery_overlay_identity_digest is not None
    ):
        raise MarginalRuntimeError("baseline session plan differs from its launch")


def _candidate_runtime(
    arm: ExecutableArm,
    *,
    baseline_launch: EngineLaunchSpec,
    baseline_binding: MaterializedArmBinding,
    baseline_session: SessionExecutionPlan,
    candidate_binding: MaterializedArmBinding,
) -> PreparedCandidateRuntime:
    base_local = baseline_binding.launch_binding
    candidate_local = candidate_binding.launch_binding
    if (
        candidate_local.controller_distribution_digest
        != base_local.controller_distribution_digest
        or candidate_local.runtime_preflight_receipt
        != base_local.runtime_preflight_receipt
        or candidate_local.physical_hardware != base_local.physical_hardware
        or _native_environment(candidate_local) != _native_environment(base_local)
    ):
        raise MarginalRuntimeError(
            "candidate changed controller, runtime preflight, or native-build environment"
        )
    candidate_launch = replace(
        baseline_launch,
        stack_digest=arm.challenger.stack_digest,
        tree_digest=arm.challenger.tree_digest,
        native_build_spec_digest=candidate_local.native_build_spec.digest,
    )
    if type(arm) is not MarginalArmPlan:
        raise MarginalRuntimeError("candidate arm has an unsupported type")
    _resolve_materialized_binding(
        candidate_launch,
        candidate_binding,
        arm.candidate,
        expected_tree_digest=arm.challenger.tree_digest,
    )
    receipt = candidate_local.runtime_preflight_receipt
    if type(receipt) is not RuntimePreflightReceipt:
        raise MarginalRuntimeError("candidate lacks a typed runtime preflight")
    candidate_preflight = expected_runtime_preflight(candidate_launch, receipt)
    candidate_session = replace(
        baseline_session,
        launch_digest=candidate_launch.digest,
        expected_preflight=candidate_preflight,
    )
    return PreparedCandidateRuntime(
        arm,
        candidate_binding,
        candidate_launch,
        candidate_session,
    )


def _validate_prepared_runtime(prepared: PreparedMarginalRuntime) -> None:
    first = prepared.candidates[0].arm
    _resolve_materialized_binding(
        prepared.baseline_launch,
        prepared.incumbent_binding,
        first.incumbent,
        expected_tree_digest=first.baseline_before.tree_digest,
    )
    _require_baseline_session(
        prepared.baseline_launch,
        prepared.incumbent_binding.launch_binding,
        prepared.baseline_session_plan,
    )
    for candidate in prepared.candidates:
        expected = _candidate_runtime(
            candidate.arm,
            baseline_launch=prepared.baseline_launch,
            baseline_binding=prepared.incumbent_binding,
            baseline_session=prepared.baseline_session_plan,
            candidate_binding=candidate.binding,
        )
        if candidate != expected:
            raise MarginalRuntimeError(
                "prepared candidate changed the common launch or workload"
            )


def _prepare(
    source: RuntimeSource,
    arms: tuple[ExecutableArm, ...],
    *,
    expected_context: EvaluationStackContext,
    incumbent_launch: EngineLaunchSpec,
    incumbent_binding: MaterializedArmBinding,
    candidate_bindings: Mapping[str, MaterializedArmBinding],
    baseline_session_plan: SessionExecutionPlan,
) -> PreparedMarginalRuntime:
    if type(incumbent_launch) is not EngineLaunchSpec:
        raise MarginalRuntimeError("incumbent_launch has the wrong type")
    if type(incumbent_binding) is not MaterializedArmBinding:
        raise MarginalRuntimeError("incumbent_binding has the wrong type")
    if not isinstance(candidate_bindings, Mapping):
        raise MarginalRuntimeError("candidate_bindings must be a mapping")
    _require_context(incumbent_launch, expected_context)
    first = arms[0]
    if (
        incumbent_launch.stack_digest != first.baseline_before.stack_digest
        or incumbent_launch.tree_digest != first.baseline_before.tree_digest
    ):
        raise MarginalRuntimeError("incumbent launch differs from the frozen baseline arm")
    expected_keys = {arm.selected_delta_digest for arm in arms}
    if set(candidate_bindings) != expected_keys or any(
        not isinstance(key, str) for key in candidate_bindings
    ):
        raise MarginalRuntimeError(
            "candidate bindings must cover every selected delta exactly once"
        )
    candidates = tuple(
        _candidate_runtime(
            arm,
            baseline_launch=incumbent_launch,
            baseline_binding=incumbent_binding,
            baseline_session=baseline_session_plan,
            candidate_binding=candidate_bindings[arm.selected_delta_digest],
        )
        for arm in arms
    )
    return PreparedMarginalRuntime(
        source,
        incumbent_binding,
        incumbent_launch,
        baseline_session_plan,
        candidates,
    )


def prepare_marginal_runtime(
    arm: MarginalArmPlan,
    *,
    catalog: TargetCatalog,
    expected_context: EvaluationStackContext,
    incumbent_launch: EngineLaunchSpec,
    incumbent_binding: MaterializedArmBinding,
    candidate_binding: MaterializedArmBinding,
    baseline_session_plan: SessionExecutionPlan,
) -> PreparedMarginalRuntime:
    """Reopen one marginal arm and bind its B,C,B-prime runtime inputs."""

    if type(arm) is not MarginalArmPlan or type(catalog) is not TargetCatalog:
        raise MarginalRuntimeError("arm or catalog has the wrong type")
    try:
        arm.reopen(catalog=catalog, expected_context=expected_context)
    except (StackPlanError, ValueError, TypeError) as exc:
        raise MarginalRuntimeError(f"marginal arm is stale or invalid: {exc}") from None
    return _prepare(
        arm,
        (arm,),
        expected_context=expected_context,
        incumbent_launch=incumbent_launch,
        incumbent_binding=incumbent_binding,
        candidate_bindings={arm.selected_delta_digest: candidate_binding},
        baseline_session_plan=baseline_session_plan,
    )


__all__ = [
    "CandidateArmWorkerError",
    "MarginalRuntimeError",
    "MaterializedArmBinding",
    "PreparedCandidateRuntime",
    "PreparedMarginalRuntime",
    "prepare_marginal_runtime",
]

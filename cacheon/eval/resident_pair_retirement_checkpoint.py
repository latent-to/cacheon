"""Restart-safe checkpoint for one fully retired commissioned resident pair."""

from __future__ import annotations

from cacheon.eval.b300_resident_pair_factory import (
    B300CommissionedResidentPairFactory,
    B300ResidentPairRequestAuthority,
    B300ResidentPairRequestOwner,
    B300ResidentStockLanePlan,
)
from cacheon.eval.continuation_codec import ContinuationCodec, ContinuationCodecError
from cacheon.eval.crossover_runtime import CrossoverRuntimeError
from cacheon.eval.device_state import DeviceStateReceipt
from cacheon.eval.oci_backend import (
    ResidentEngineExecutionEvidence,
    expected_runtime_preflight,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.oci_resident_session import ResidentSessionEvidence
from cacheon.eval.resident_count_execution_evidence import (
    ResidentCountQualityExecutionEvidence,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionPlan,
)
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationRetirementEvidence,
    ResidentRequestResult,
    ResidentRequestSlice,
)
from cacheon.eval.resident_pair_binding import (
    ResidentPairLaneBinding,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverError,
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
)
from cacheon.eval.qualification_continuation import (
    ResidentPairLaneRetirement,
    ResidentPairRetirementCheckpoint,
)
from cacheon.stack_identity import canonical_digest


_LIFETIME_SCHEMA = "cacheon.oci-resident-queue-execution.v1"


class ResidentPairRetirementHold(RuntimeError):
    """Retirement evidence is absent, foreign, partial, or corrupt."""

    decision = "HOLD"


_SESSION_CODEC = ContinuationCodec((ResidentSessionEvidence,))
_SLICE_CODEC = ContinuationCodec((ResidentRequestSlice,))


def _typed_digest(codec: ContinuationCodec, value: object, domain: str) -> str:
    try:
        return canonical_digest(domain, codec.encode(value))
    except ContinuationCodecError as exc:
        raise ResidentPairRetirementHold(str(exc)) from None


def _slice_digest(value: ResidentRequestSlice) -> str:
    return _typed_digest(_SLICE_CODEC, value, "cacheon.eval.resident-request-slice.v1")


def _request_epoch(
    factory: B300CommissionedResidentPairFactory,
    authority: B300ResidentPairRequestAuthority,
) -> str:
    return canonical_digest(
        "cacheon.eval.b300-resident-request-pair-epoch.v1",
        {"commissioned_epoch": factory.commissioned_epoch_digest, "request": authority.digest},
    )


def _context(
    factory: B300CommissionedResidentPairFactory,
    authority: B300ResidentPairRequestAuthority,
    speed_plan: ResidentPairCrossoverPlan,
    speed: ResidentPairCrossoverEvidence,
    count_plan: ResidentCountQualityExecutionPlan | None,
    count: ResidentCountQualityExecutionEvidence | None,
) -> None:
    if (
        type(factory) is not B300CommissionedResidentPairFactory
        or type(authority) is not B300ResidentPairRequestAuthority
        or type(speed_plan) is not ResidentPairCrossoverPlan
        or type(speed) is not ResidentPairCrossoverEvidence
        or (count_plan is None) != (count is None)
        or speed.pair_binding != speed_plan.pair_binding
        or speed_plan.pair_binding.service_epoch_digest != factory.commissioned_epoch_digest
    ):
        raise ResidentPairRetirementHold("retirement differs from commissioned speed")
    if count is not None and (
        type(count_plan) is not ResidentCountQualityExecutionPlan
        or type(count) is not ResidentCountQualityExecutionEvidence
        or count_plan.pair_binding != speed_plan.pair_binding
        or count.pair_binding != speed_plan.pair_binding
        or count.execution_plan_digest != count_plan.digest
        or count.candidate_bundle_digest != speed_plan.candidate_bundle_digest
    ):
        raise ResidentPairRetirementHold("retirement differs from commissioned count")
    try:
        speed.regrade(speed_plan)
    except (ResidentPairCrossoverError, CrossoverRuntimeError) as exc:
        raise ResidentPairRetirementHold(f"resident pair speed does not regrade: {exc}") from None


def _history(
    retirement: ResidentEvaluationRetirementEvidence,
    speed: ResidentPairCrossoverEvidence,
    count: ResidentCountQualityExecutionEvidence | None,
) -> tuple[ResidentRequestSlice, ...]:
    results = retirement.request_history
    slices = tuple(row.request_slice for row in results)
    speed_count = len(speed.request_slices)
    if (
        type(results) is not tuple
        or len(results) != speed_count + (2 if count is not None else 0)
        or any(type(row) is not ResidentRequestResult or not row.ok for row in results)
        or slices[:speed_count] != speed.request_slices
        or (count is not None and {row.request_id for row in slices[speed_count:]} != {
            row.request_id for row in count.request_slices
        })
        or len({row.request_id for row in slices}) != len(slices)
        or any(
            row.value
            != (row.request_slice.new_batches if index < speed_count else row.request_slice.new_batches[0])
            for index, row in enumerate(results)
        )
    ):
        raise ResidentPairRetirementHold("retirement history differs from speed then count")
    return slices


def _lane(
    plan: B300ResidentStockLanePlan,
    binding: ResidentPairLaneBinding,
    lifetime: ResidentEngineExecutionEvidence,
    proof: OCIQuiescenceReceipt,
    *,
    model_mount_digest: str,
    slices: tuple[ResidentRequestSlice, ...],
) -> ResidentPairLaneRetirement:
    lane_slices = tuple(row for row in slices if row.lane_id == binding.lane_id)
    session = lifetime.session
    pre, post = lifetime.device_receipts
    manager = plan.executor.manager
    if (
        type(lifetime) is not ResidentEngineExecutionEvidence
        or lifetime.schema != _LIFETIME_SCHEMA
        or type(session) is not ResidentSessionEvidence
        or type(pre) is not DeviceStateReceipt
        or type(post) is not DeviceStateReceipt
        or (pre.schema, post.schema) != ("cacheon.device-state-receipt.v1",) * 2
        or (pre.phase, post.phase) != ("pre", "post")
        or pre.launch_id != post.launch_id
        or pre.sequence >= post.sequence
        or pre.selected_physical_gpu_ids != plan.lane_policy.physical_gpu_ids
        or post.selected_physical_gpu_ids != plan.lane_policy.physical_gpu_ids
        or pre.configuration_sha256 != plan.lane_policy.device_configuration_digest
        or post.configuration_sha256 != plan.lane_policy.device_configuration_digest
        or pre.policy_sha256 != plan.lane_policy.device_policy_digest
        or post.policy_sha256 != plan.lane_policy.device_policy_digest
        or lifetime.launch_digest != plan.stock_launch.digest
        or lifetime.runtime_identity
        != runtime_identity_from_preflight(plan.stock_binding.runtime_preflight_receipt)
        or lifetime.runtime_preflight_receipt_sha256
        != plan.stock_binding.runtime_preflight_receipt.sha256
        or lifetime.arena_model_receipt_digest != model_mount_digest
        or lifetime.resource_policy_digest != plan.executor.config.runtime.digest
        or session.session_id != binding.session_id
        or session.launch_digest != binding.stock_launch_digest
        or session.preflight
        != expected_runtime_preflight(plan.stock_launch, plan.stock_binding.runtime_preflight_receipt)
        or session.batches != tuple(batch for row in lane_slices for batch in row.new_batches)
        or session.swaps != tuple(swap for row in lane_slices for swap in row.new_swaps)
        or pre.completed_monotonic_s > session.ready_completed_at
        or session.session_completed_at > post.started_monotonic_s
        or proof.executor_id != manager.executor_id
        or proof.manager_instance_id != manager.manager_instance_id
        or proof.namespace_digest != binding.executor_namespace_digest
        or proof.observed_monotonic_s < post.completed_monotonic_s
    ):
        raise ResidentPairRetirementHold(f"lane {binding.lane_id} retirement is not exact")
    return ResidentPairLaneRetirement(
        binding.lane_id,
        plan.commissioning_digest,
        lifetime.runtime_identity,
        lifetime.runtime_preflight_receipt_sha256,
        lifetime.arena_model_receipt_digest,
        lifetime.resource_policy_digest,
        lifetime.native_publication_digest,
        lifetime.runtime_argv_sha256,
        tuple(lifetime.recovered_lease_ids),
        (pre, post),
        float(session.ready_completed_at),
        float(session.session_completed_at),
        _typed_digest(_SESSION_CODEC, session, "cacheon.eval.resident-session-evidence.v1"),
        proof,
    )


def build_resident_pair_retirement_checkpoint(
    owner: B300ResidentPairRequestOwner,
    *,
    factory: B300CommissionedResidentPairFactory,
    authority: B300ResidentPairRequestAuthority,
    speed_plan: ResidentPairCrossoverPlan,
    speed: ResidentPairCrossoverEvidence,
    count_plan: ResidentCountQualityExecutionPlan | None,
    count: ResidentCountQualityExecutionEvidence | None,
) -> ResidentPairRetirementCheckpoint:
    """Close once and return the only durable, path-free retirement product."""

    _context(factory, authority, speed_plan, speed, count_plan, count)
    if type(owner) is not B300ResidentPairRequestOwner:
        raise ResidentPairRetirementHold("resident pair owner is not exact")
    retirement, proofs = owner.retire_and_quiesce(authority, speed_plan.pair_binding)
    slices = _history(retirement, speed, count)
    lifetimes = (retirement.lane_a.lifetime_evidence, retirement.lane_b.lifetime_evidence)
    checkpoint = ResidentPairRetirementCheckpoint(
        authority.authenticated_request_digest,
        authority.qualification_authority_digest,
        authority.target_profile_digest,
        _request_epoch(factory, authority),
        speed_plan.pair_binding,
        speed_plan.digest,
        speed.digest,
        None if count_plan is None else count_plan.digest,
        None if count is None else count.digest,
        tuple(_slice_digest(row) for row in slices),
        tuple(
            _lane(
                plan,
                binding,
                lifetime,
                proof,
                model_mount_digest=factory.model_mount.digest,
                slices=slices,
            )
            for plan, binding, lifetime, (_lane_id, proof) in zip(
                factory.lane_plans, speed_plan.pair_binding.lanes, lifetimes, proofs, strict=True
            )
        ),
    )
    return regrade_resident_pair_retirement_checkpoint(
        checkpoint,
        factory=factory,
        authority=authority,
        speed_plan=speed_plan,
        speed=speed,
        count_plan=count_plan,
        count=count,
    )


def regrade_resident_pair_retirement_checkpoint(
    checkpoint: ResidentPairRetirementCheckpoint,
    *,
    factory: B300CommissionedResidentPairFactory,
    authority: B300ResidentPairRequestAuthority,
    speed_plan: ResidentPairCrossoverPlan,
    speed: ResidentPairCrossoverEvidence,
    count_plan: ResidentCountQualityExecutionPlan | None,
    count: ResidentCountQualityExecutionEvidence | None,
) -> ResidentPairRetirementCheckpoint:
    """Authenticate a reopened checkpoint without a pair, evaluator, or rerun seam."""

    _context(factory, authority, speed_plan, speed, count_plan, count)
    if type(checkpoint) is not ResidentPairRetirementCheckpoint:
        raise ResidentPairRetirementHold("resident pair retirement checkpoint is not exact")
    speed_digests = tuple(_slice_digest(row) for row in speed.request_slices)
    count_digests = () if count is None else tuple(_slice_digest(row) for row in count.request_slices)
    if (
        checkpoint.authenticated_request_digest != authority.authenticated_request_digest
        or checkpoint.qualification_authority_digest != authority.qualification_authority_digest
        or checkpoint.target_profile_digest != authority.target_profile_digest
        or checkpoint.request_epoch_digest != _request_epoch(factory, authority)
        or checkpoint.pair_binding != speed_plan.pair_binding
        or checkpoint.speed_plan_digest != speed_plan.digest
        or checkpoint.speed_evidence_digest != speed.digest
        or checkpoint.count_plan_digest != (None if count_plan is None else count_plan.digest)
        or checkpoint.count_evidence_digest != (None if count is None else count.digest)
        or checkpoint.request_history_slice_digests[: len(speed_digests)] != speed_digests
        or sorted(checkpoint.request_history_slice_digests[len(speed_digests) :])
        != sorted(count_digests)
    ):
        raise ResidentPairRetirementHold("retirement names another request or history")
    slices = speed.request_slices + (() if count is None else count.request_slices)
    for plan, binding, lane in zip(
        factory.lane_plans, checkpoint.pair_binding.lanes, checkpoint.lanes, strict=True
    ):
        session = ResidentSessionEvidence(
            binding.session_id,
            binding.stock_launch_digest,
            expected_runtime_preflight(plan.stock_launch, plan.stock_binding.runtime_preflight_receipt),
            lane.session_ready_completed_at,
            tuple(batch for row in slices if row.lane_id == binding.lane_id for batch in row.new_batches),
            tuple(swap for row in slices if row.lane_id == binding.lane_id for swap in row.new_swaps),
            lane.session_completed_at,
        )
        pre, post = lane.device_receipts
        if (
            lane.lane_id != binding.lane_id
            or lane.commissioning_digest != plan.commissioning_digest
            or lane.runtime_identity
            != runtime_identity_from_preflight(plan.stock_binding.runtime_preflight_receipt)
            or lane.runtime_preflight_receipt_sha256
            != plan.stock_binding.runtime_preflight_receipt.sha256
            or lane.arena_model_receipt_digest != factory.model_mount.digest
            or lane.resource_policy_digest != plan.executor.config.runtime.digest
            or pre.selected_physical_gpu_ids != plan.lane_policy.physical_gpu_ids
            or post.selected_physical_gpu_ids != plan.lane_policy.physical_gpu_ids
            or pre.configuration_sha256 != plan.lane_policy.device_configuration_digest
            or post.configuration_sha256 != plan.lane_policy.device_configuration_digest
            or pre.policy_sha256 != plan.lane_policy.device_policy_digest
            or post.policy_sha256 != plan.lane_policy.device_policy_digest
            or _typed_digest(_SESSION_CODEC, session, "cacheon.eval.resident-session-evidence.v1")
            != lane.session_digest
            or lane.quiescence.executor_id != plan.executor.manager.executor_id
            or lane.quiescence.namespace_digest != binding.executor_namespace_digest
            or lane.quiescence.observed_monotonic_s < post.completed_monotonic_s
        ):
            raise ResidentPairRetirementHold(f"lane {binding.lane_id} retirement changed")
    return checkpoint


__all__ = [
    "ResidentPairLaneRetirement",
    "ResidentPairRetirementCheckpoint",
    "ResidentPairRetirementHold",
    "build_resident_pair_retirement_checkpoint",
    "regrade_resident_pair_retirement_checkpoint",
]

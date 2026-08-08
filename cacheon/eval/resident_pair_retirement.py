"""Closed, path-free retirement authority for one commissioned resident pair.

The generic pair owner retains live lifetime products (including host paths).
This module projects only the authority needed after teardown and independently
reconstructs that projection from caller-supplied speed, count, lifetime, and
post-close quiescence evidence.  It never grades a candidate: ambiguity is an
infrastructure HOLD.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import TypeAlias

from cacheon.eval.continuation_codec import (
    ContinuationCodec,
    ContinuationCodecError,
)
from cacheon.eval.device_state import (
    DeviceStateReceipt,
    DeviceStateSample,
    GPUProcess,
    GPUTelemetry,
)
from cacheon.eval.native_artifact import NativeArtifactPublication
from cacheon.eval.oci_backend import (
    CandidateFreeRuntimeIdentity,
    ResidentEngineExecutionEvidence,
)
from cacheon.eval.oci_prebuild import OCIPrebuildResult
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.oci_resident_session import (
    ResidentBatchEvidence,
    ResidentSessionEvidence,
    SwapReceipt,
)
from cacheon.eval.oci_session_protocol import RuntimePreflightFacts
from cacheon.eval.resident_count_execution_evidence import (
    ResidentCountExecutionEvidenceError,
    ResidentCountQualityExecutionEvidence,
    require_resident_count_request_slice,
)
from cacheon.eval.resident_count_quality import (
    ResidentCountQualityObservation,
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
    ResidentPairRuntimeBinding,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverError,
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
)
from cacheon.stack_identity import (
    StackIdentityError,
    canonical_digest,
    require_sha256_hex,
)


RESIDENT_LANE_RETIREMENT_SCHEMA = (
    "cacheon.eval.resident-lane-retirement-projection.v1"
)
RESIDENT_PAIR_RETIREMENT_SCHEMA = "cacheon.eval.resident-pair-retirement.v1"
RESIDENT_PAIR_HISTORY_SCHEMA = "cacheon.eval.resident-pair-full-history.v1"
_EXECUTION_SCHEMA = "cacheon.oci-resident-queue-execution.v1"
_DEVICE_SCHEMA = "cacheon.device-state-receipt.v1"
_SIMPLE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


class ResidentPairRetirementError(ValueError):
    """A retirement authority or projection has an invalid closed shape."""


class ResidentPairRetirementHold(ResidentPairRetirementError):
    """Retirement evidence is foreign, incomplete, or ambiguous."""

    decision = "HOLD"


LaneClosure: TypeAlias = tuple[
    ResidentEngineExecutionEvidence, OCIQuiescenceReceipt
]
PairLaneClosures: TypeAlias = tuple[LaneClosure, LaneClosure]


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (StackIdentityError, TypeError, ValueError) as exc:
        raise ResidentPairRetirementError(str(exc)) from None


def _finite(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ResidentPairRetirementError(f"{field} must be finite")
    return float(value)


def _closed_value(value: object) -> object:
    """Return a canonical-JSON value and reject every host Path-like object."""

    if isinstance(value, os.PathLike):
        raise ResidentPairRetirementError(
            "resident retirement projection contains a host path"
        )
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ResidentPairRetirementError(
                "resident retirement projection contains a non-finite clock"
            )
        return format(value, ".17g")
    if isinstance(value, Enum):
        return _closed_value(value.value)
    if is_dataclass(value):
        return {
            field.name: _closed_value(getattr(value, field.name))
            for field in fields(value)
        }
    if type(value) in (tuple, list):
        return [_closed_value(row) for row in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ResidentPairRetirementError(
                "resident retirement projection has a non-string object key"
            )
        return {key: _closed_value(value[key]) for key in sorted(value)}
    raise ResidentPairRetirementError(
        "resident retirement projection contains an open or unsupported value"
    )


def _codec_dict(value: object) -> dict[str, object]:
    """Mechanically encode one path-free dataclass with its exact type hints."""

    try:
        payload = ContinuationCodec((type(value),)).encode(value)
    except ContinuationCodecError as exc:
        raise ResidentPairRetirementError(
            f"resident retirement projection is not closed: {exc}"
        ) from None
    encoded = payload["value"]
    assert type(encoded) is dict
    return encoded


def _receipt_samples(receipt: DeviceStateReceipt) -> None:
    if (
        type(receipt.samples) is not tuple
        or not receipt.samples
        or any(type(sample) is not DeviceStateSample for sample in receipt.samples)
        or type(receipt.consecutive_idle_samples) is not int
        or not 1 <= receipt.consecutive_idle_samples <= len(receipt.samples)
        or any(
            not sample.idle
            for sample in receipt.samples[-receipt.consecutive_idle_samples :]
        )
    ):
        raise ResidentPairRetirementHold(
            "resident post-close device receipt lacks a proven idle tail"
        )
    previous = receipt.started_monotonic_s
    selected = set(receipt.selected_physical_gpu_ids)
    for sample in receipt.samples:
        observed = _finite(sample.monotonic_s, "device sample clock")
        if (
            observed < previous
            or observed > receipt.completed_monotonic_s
            or type(sample.telemetry) is not tuple
            or type(sample.processes) is not tuple
            or type(sample.idle) is not bool
            or type(sample.idle_reason) is not str
            or type(sample.active_envelope_passed) is not bool
            or type(sample.active_envelope_reason) is not str
            or any(type(row) is not GPUTelemetry for row in sample.telemetry)
            or any(type(row) is not GPUProcess for row in sample.processes)
            or len({row.physical_id for row in sample.telemetry})
            != len(sample.telemetry)
            or {row.physical_id for row in sample.telemetry} != selected
            or any(row.physical_id not in selected for row in sample.processes)
        ):
            raise ResidentPairRetirementHold(
                "resident device receipt raw samples are malformed"
            )
        previous = observed


def _validate_device_receipts(
    receipts: tuple[DeviceStateReceipt, DeviceStateReceipt],
    *,
    session: ResidentSessionEvidence,
) -> None:
    if (
        type(receipts) is not tuple
        or len(receipts) != 2
        or any(type(row) is not DeviceStateReceipt for row in receipts)
    ):
        raise ResidentPairRetirementHold(
            "resident lifetime lacks exact pre/post device receipts"
        )
    pre, post = receipts
    for row in receipts:
        _finite(row.started_monotonic_s, "device receipt start")
        _finite(row.completed_monotonic_s, "device receipt completion")
        _digest(row.configuration_sha256, "device configuration")
        _digest(row.policy_sha256, "device policy")
        _receipt_samples(row)
    selected = pre.selected_physical_gpu_ids
    if (
        (pre.schema, post.schema) != (_DEVICE_SCHEMA, _DEVICE_SCHEMA)
        or (pre.phase, post.phase) != ("pre", "post")
        or type(pre.sequence) is not int
        or type(post.sequence) is not int
        or not 0 <= pre.sequence < post.sequence
        or type(pre.launch_id) is not str
        or _SIMPLE_ID.fullmatch(pre.launch_id) is None
        or post.launch_id != pre.launch_id
        or type(selected) is not tuple
        or not selected
        or any(type(value) is not int or value < 0 for value in selected)
        or len(set(selected)) != len(selected)
        or post.selected_physical_gpu_ids != selected
        or post.configuration_sha256 != pre.configuration_sha256
        or post.policy_sha256 != pre.policy_sha256
        or pre.started_monotonic_s > pre.completed_monotonic_s
        or post.started_monotonic_s > post.completed_monotonic_s
        or pre.completed_monotonic_s > session.ready_completed_at
        or session.session_completed_at > post.started_monotonic_s
    ):
        raise ResidentPairRetirementHold(
            "resident device receipts differ from the closed session"
        )


@dataclass(frozen=True)
class ResidentLaneRetirementProjection:
    """Exact path-free authority projected from one closed engine lifetime."""

    lane_id: str
    session_id: str
    stock_launch_digest: str
    lane_authority_digest: str
    allocation_digest: str
    executor_namespace_digest: str
    execution_schema: str
    runtime_identity: CandidateFreeRuntimeIdentity
    runtime_preflight_receipt_sha256: str
    arena_model_receipt_digest: str
    runtime_resource_policy_digest: str
    native_build_spec_digest: str
    native_publication_digest: str
    runtime_argv_sha256: str
    recovered_lease_ids: tuple[str, ...]
    session: ResidentSessionEvidence
    device_receipts: tuple[DeviceStateReceipt, DeviceStateReceipt]
    quiescence: OCIQuiescenceReceipt
    quiescence_digest: str
    close_count: int = 1
    schema: str = RESIDENT_LANE_RETIREMENT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != RESIDENT_LANE_RETIREMENT_SCHEMA
            or self.lane_id not in ("A", "B")
            or type(self.session_id) is not str
            or _HEX32.fullmatch(self.session_id) is None
            or self.session_id == "0" * 32
            or self.execution_schema != _EXECUTION_SCHEMA
            or self.close_count != 1
        ):
            raise ResidentPairRetirementError(
                "resident lane retirement identity is malformed"
            )
        for name in (
            "stock_launch_digest",
            "lane_authority_digest",
            "allocation_digest",
            "executor_namespace_digest",
            "runtime_preflight_receipt_sha256",
            "arena_model_receipt_digest",
            "runtime_resource_policy_digest",
            "native_build_spec_digest",
            "native_publication_digest",
            "runtime_argv_sha256",
            "quiescence_digest",
        ):
            _digest(getattr(self, name), name.replace("_", " "))
        if (
            type(self.runtime_identity) is not CandidateFreeRuntimeIdentity
            or type(self.session) is not ResidentSessionEvidence
            or type(self.quiescence) is not OCIQuiescenceReceipt
            or self.quiescence.digest != self.quiescence_digest
            or self.session.session_id != self.session_id
            or self.session.launch_digest != self.stock_launch_digest
            or type(self.session.preflight) is not RuntimePreflightFacts
            or self.session.preflight.launch_digest != self.stock_launch_digest
            or self.session.preflight.runtime_digest
            != self.runtime_identity.runtime_digest
            or self.quiescence.namespace_digest != self.executor_namespace_digest
            or self.quiescence.schema != "cacheon.oci-quiescence.v1"
            or any(
                (
                    self.quiescence.lease_records,
                    self.quiescence.resource_entries,
                    self.quiescence.container_ids,
                )
            )
        ):
            raise ResidentPairRetirementError(
                "resident lane retirement authorities are inconsistent"
            )
        leases = self.recovered_lease_ids
        if (
            type(leases) is not tuple
            or leases != tuple(sorted(set(leases)))
            or any(
                type(value) is not str or _SIMPLE_ID.fullmatch(value) is None
                for value in leases
            )
        ):
            raise ResidentPairRetirementError(
                "resident recovered lease identities are not canonical"
            )
        _validate_device_receipts(self.device_receipts, session=self.session)
        if (
            self.quiescence.observed_monotonic_s
            < self.device_receipts[1].completed_monotonic_s
        ):
            raise ResidentPairRetirementError(
                "resident quiescence was observed before post-device closure"
            )
        _closed_value(self)

    @property
    def selected_physical_gpu_ids(self) -> tuple[int, ...]:
        return self.device_receipts[0].selected_physical_gpu_ids

    def to_dict(self) -> dict[str, object]:
        return _codec_dict(self)

    @property
    def digest(self) -> str:
        return canonical_digest(self.schema, self.to_dict())


@dataclass(frozen=True)
class ResidentPairRetirementEvidence:
    """Portable proof that the exact speed/count pair is fully retired."""

    pair_binding: ResidentPairRuntimeBinding
    speed_plan_digest: str
    speed_evidence_digest: str
    count_execution_plan_digest: str
    count_execution_evidence_digest: str
    count_observation_digest: str
    request_history_request_ids: tuple[str, ...]
    canonical_request_ids: tuple[str, ...]
    request_history_digest: str
    lane_a: ResidentLaneRetirementProjection
    lane_b: ResidentLaneRetirementProjection
    retirement_cutoff_monotonic_s: float
    schema: str = RESIDENT_PAIR_RETIREMENT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != RESIDENT_PAIR_RETIREMENT_SCHEMA
            or type(self.pair_binding) is not ResidentPairRuntimeBinding
            or type(self.lane_a) is not ResidentLaneRetirementProjection
            or type(self.lane_b) is not ResidentLaneRetirementProjection
        ):
            raise ResidentPairRetirementError(
                "resident pair retirement product is not exact"
            )
        for name in (
            "speed_plan_digest",
            "speed_evidence_digest",
            "count_execution_plan_digest",
            "count_execution_evidence_digest",
            "count_observation_digest",
            "request_history_digest",
        ):
            _digest(getattr(self, name), name.replace("_", " "))
        actual, canonical = (
            self.request_history_request_ids,
            self.canonical_request_ids,
        )
        if (
            type(actual) is not tuple
            or type(canonical) is not tuple
            or len(actual) not in (5, 7)
            or len(canonical) != len(actual)
            or any(type(value) is not str or _HEX32.fullmatch(value) is None for value in actual)
            or len(set(actual)) != len(actual)
            or actual[:-2] != canonical[:-2]
            or set(actual[-2:]) != set(canonical[-2:])
            or canonical[-2:] == canonical[-2:][::-1]
        ):
            raise ResidentPairRetirementError(
                "resident pair retirement request history is not exact"
            )
        lane_a, lane_b = self.pair_binding.lanes
        if (
            self.lane_a.lane_id != "A"
            or self.lane_b.lane_id != "B"
            or self.lane_a.session_id != lane_a.session_id
            or self.lane_b.session_id != lane_b.session_id
            or self.lane_a.stock_launch_digest != lane_a.stock_launch_digest
            or self.lane_b.stock_launch_digest != lane_b.stock_launch_digest
            or self.lane_a.lane_authority_digest != lane_a.lane_digest
            or self.lane_b.lane_authority_digest != lane_b.lane_digest
            or self.lane_a.allocation_digest != lane_a.allocation_digest
            or self.lane_b.allocation_digest != lane_b.allocation_digest
            or self.lane_a.executor_namespace_digest
            != lane_a.executor_namespace_digest
            or self.lane_b.executor_namespace_digest
            != lane_b.executor_namespace_digest
            or set(self.lane_a.selected_physical_gpu_ids)
            & set(self.lane_b.selected_physical_gpu_ids)
        ):
            raise ResidentPairRetirementError(
                "resident A/B retirement differs from the runtime binding"
            )
        cutoff = _finite(
            self.retirement_cutoff_monotonic_s, "resident retirement cutoff"
        )
        expected = max(
            self.lane_a.quiescence.observed_monotonic_s,
            self.lane_b.quiescence.observed_monotonic_s,
        )
        if cutoff != expected:
            raise ResidentPairRetirementError(
                "resident retirement cutoff differs from post-close quiescence"
            )
        _closed_value(self)

    @property
    def session_ids(self) -> tuple[str, str]:
        return self.lane_a.session_id, self.lane_b.session_id

    def to_dict(self) -> dict[str, object]:
        return _codec_dict(self)

    @property
    def digest(self) -> str:
        return canonical_digest(self.schema, self.to_dict())

    def regrade(
        self,
        *,
        binding: ResidentPairRuntimeBinding,
        speed_plan: ResidentPairCrossoverPlan,
        speed_evidence: ResidentPairCrossoverEvidence,
        count_plan: ResidentCountQualityExecutionPlan,
        count_evidence: ResidentCountQualityExecutionEvidence,
        count_observation: ResidentCountQualityObservation,
        retirement: ResidentEvaluationRetirementEvidence,
        lane_closures: PairLaneClosures,
    ) -> "ResidentPairRetirementEvidence":
        return regrade_resident_pair_retirement(
            self,
            binding=binding,
            speed_plan=speed_plan,
            speed_evidence=speed_evidence,
            count_plan=count_plan,
            count_evidence=count_evidence,
            count_observation=count_observation,
            retirement=retirement,
            lane_closures=lane_closures,
        )


def _validate_count(
    plan: ResidentCountQualityExecutionPlan,
    evidence: ResidentCountQualityExecutionEvidence,
    observation: ResidentCountQualityObservation,
) -> None:
    if (
        evidence.execution_plan_digest != plan.digest
        or evidence.candidate_bundle_digest != plan.candidate_bundle_digest
        or evidence.envelope_digest != plan.envelope.digest
        or evidence.pair_binding != plan.pair_binding
        or observation.role != "candidate"
        or observation.envelope != plan.envelope
        or observation.execution_evidence_digest != evidence.digest
    ):
        raise ResidentPairRetirementHold(
            "resident count evidence or observation is foreign"
        )
    counts = (
        plan.admission.lane_a_prompt_count,
        plan.admission.lane_b_prompt_count,
    )
    batches: list[ResidentBatchEvidence] = []
    try:
        for request, lane, count in zip(
            evidence.request_slices, plan.pair_binding.lanes, counts, strict=True
        ):
            batches.append(
                require_resident_count_request_slice(
                    request,
                    lane_binding=lane,
                    candidate_bundle_digest=plan.candidate_bundle_digest,
                    expected_prompt_count=count,
                    max_new_tokens=plan.batch_shape.max_new_tokens,
                )
            )
    except ResidentCountExecutionEvidenceError as exc:
        raise ResidentPairRetirementHold(
            f"resident count execution cannot retire the pair: {exc}"
        ) from None
    output_ids = tuple(
        prompt.output_ids
        for batch in batches
        for prompt in batch.evidence.prompts
    )
    occurrences = plan.selected_occurrences
    if (
        tuple(row.output_ids for row in observation.prompts) != output_ids
        or tuple(row.prompt_digest for row in observation.prompts)
        != tuple(row.prompt_digest for row in occurrences)
        or tuple(row.task_digests for row in observation.prompts)
        != tuple(row.task_digests for row in occurrences)
    ):
        raise ResidentPairRetirementHold(
            "resident count observation was not derived from the retained A/B outputs"
        )


def _history_digest(slices: tuple[ResidentRequestSlice, ...]) -> str:
    codec = ContinuationCodec((ResidentRequestSlice,))
    return canonical_digest(
        RESIDENT_PAIR_HISTORY_SCHEMA,
        [codec.encode(row) for row in slices],
    )


def _validate_history(
    retirement: ResidentEvaluationRetirementEvidence,
    speed: ResidentPairCrossoverEvidence,
    count: ResidentCountQualityExecutionEvidence,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    history = retirement.request_history
    speed_slices = speed.request_slices
    count_slices = count.request_slices
    if (
        type(history) is not tuple
        or len(history) != len(speed_slices) + 2
        or any(type(row) is not ResidentRequestResult for row in history)
        or tuple(row.request_slice for row in history[: len(speed_slices)])
        != speed_slices
    ):
        raise ResidentPairRetirementHold(
            "resident retirement history is not the exact speed prefix plus count"
        )
    for result, request in zip(
        history[: len(speed_slices)], speed_slices, strict=True
    ):
        if (
            not result.ok
            or type(result.value) is not tuple
            or result.value != request.new_batches
        ):
            raise ResidentPairRetirementHold(
                "resident speed history contains a failed or foreign result"
            )
    final = history[-2:]
    by_lane: dict[str, ResidentRequestResult] = {}
    for result in final:
        lane = result.request_slice.lane_id
        if lane in by_lane:
            raise ResidentPairRetirementHold(
                "resident count history duplicates one physical lane"
            )
        by_lane[lane] = result
    if set(by_lane) != {"A", "B"}:
        raise ResidentPairRetirementHold(
            "resident count history is missing one physical lane"
        )
    for request in count_slices:
        result = by_lane[request.lane_id]
        if (
            result.request_slice != request
            or not result.ok
            or result.value != request.new_batches[0]
        ):
            raise ResidentPairRetirementHold(
                "resident count history contains a failed or foreign result"
            )
    all_slices = tuple(row.request_slice for row in history)
    request_ids = tuple(row.request_id for row in all_slices)
    canonical_ids = tuple(row.request_id for row in (*speed_slices, *count_slices))
    if len(set(request_ids)) != len(request_ids):
        raise ResidentPairRetirementHold(
            "resident retirement history duplicates a request identity"
        )
    speed_evaluations = tuple(row.evaluation_id for row in speed_slices)
    count_evaluation = count_slices[0].evaluation_id
    if (
        len(set(speed_evaluations)) != len(speed_evaluations)
        or count_slices[1].evaluation_id != count_evaluation
        or count_evaluation in set(speed_evaluations) | set(request_ids)
        or set(speed_evaluations) & set(request_ids)
    ):
        raise ResidentPairRetirementHold(
            "resident retirement evaluation identities are interleaved or reused"
        )
    return request_ids, canonical_ids, _history_digest(all_slices)


def _validate_lane_stream(
    execution: ResidentEngineExecutionEvidence,
    *,
    lane: ResidentPairLaneBinding,
    speed_slices: tuple[ResidentRequestSlice, ...],
    count_slice: ResidentRequestSlice,
) -> None:
    session = execution.session
    slices = tuple(row for row in speed_slices if row.lane_id == lane.lane_id) + (
        count_slice,
    )
    if (
        type(session) is not ResidentSessionEvidence
        or not slices
        or slices[0].starting_generation != 0
        or session.ready_completed_at > slices[0].host_started_at
    ):
        raise ResidentPairRetirementHold(
            f"resident lane {lane.lane_id} session start is unbound"
        )
    previous_generation = 0
    previous_host = session.ready_completed_at
    batches: list[ResidentBatchEvidence] = []
    swaps: list[SwapReceipt] = []
    for request in slices:
        if (
            request.session_id != lane.session_id
            or request.starting_generation != previous_generation
            or request.host_started_at < previous_host
            or request.host_completed_at < request.host_started_at
            or len(request.new_batches) != request.expected_batch_count
            or len(request.new_swaps) != request.expected_swap_count
        ):
            raise ResidentPairRetirementHold(
                f"resident lane {lane.lane_id} request stream has a gap or overlap"
            )
        batches.extend(request.new_batches)
        swaps.extend(request.new_swaps)
        previous_generation = request.ending_generation
        previous_host = request.host_completed_at
    if (
        tuple(batches) != session.batches
        or tuple(swaps) != session.swaps
        or tuple(row.batch_index for row in batches) != tuple(range(len(batches)))
        or tuple(row.swap_index for row in swaps) != tuple(range(len(swaps)))
        or not swaps
        or swaps[-1].bundle_digest is not None
        or swaps[-1].slots
        or swaps[-1].generation != previous_generation
        or session.session_completed_at < previous_host
    ):
        raise ResidentPairRetirementHold(
            f"resident lane {lane.lane_id} lifetime contains unaccounted work"
        )


def _canonical_closures(
    binding: ResidentPairRuntimeBinding,
    retirement: ResidentEvaluationRetirementEvidence,
    lane_closures: PairLaneClosures,
) -> tuple[LaneClosure, LaneClosure]:
    if (
        type(retirement) is not ResidentEvaluationRetirementEvidence
        or type(lane_closures) is not tuple
        or len(lane_closures) != 2
        or any(type(row) is not tuple or len(row) != 2 for row in lane_closures)
    ):
        raise ResidentPairRetirementError(
            "resident retirement inputs require exactly two lane closures"
        )
    if (
        retirement.lane_a.identity != binding.identities[0]
        or retirement.lane_b.identity != binding.identities[1]
    ):
        raise ResidentPairRetirementHold(
            "resident lifetime retirement changed the bound A/B sessions"
        )
    by_session: dict[str, LaneClosure] = {}
    for execution, quiescence in lane_closures:
        if (
            type(execution) is not ResidentEngineExecutionEvidence
            or type(quiescence) is not OCIQuiescenceReceipt
            or type(execution.session) is not ResidentSessionEvidence
        ):
            raise ResidentPairRetirementError(
                "resident lane closure values are not exact evidence"
            )
        session_id = execution.session.session_id
        if session_id in by_session:
            raise ResidentPairRetirementHold(
                "resident lane closure duplicates one session"
            )
        by_session[session_id] = (execution, quiescence)
    if set(by_session) != {row.session_id for row in binding.lanes}:
        raise ResidentPairRetirementHold(
            "resident lane closure is missing or names a foreign session"
        )
    canonical = tuple(by_session[row.session_id] for row in binding.lanes)
    lifetimes = (
        retirement.lane_a.lifetime_evidence,
        retirement.lane_b.lifetime_evidence,
    )
    if any(
        type(lifetime) is not ResidentEngineExecutionEvidence
        or lifetime != closure[0]
        for lifetime, closure in zip(lifetimes, canonical, strict=True)
    ):
        raise ResidentPairRetirementHold(
            "resident lane closure differs from its exact lifetime product"
        )
    return canonical  # type: ignore[return-value]


def _project_lane(
    lane: ResidentPairLaneBinding,
    execution: ResidentEngineExecutionEvidence,
    quiescence: OCIQuiescenceReceipt,
) -> ResidentLaneRetirementProjection:
    prebuild = execution.prebuild
    publication = getattr(prebuild, "publication", None)
    if (
        execution.schema != _EXECUTION_SCHEMA
        or execution.launch_digest != lane.stock_launch_digest
        or type(execution.runtime_identity) is not CandidateFreeRuntimeIdentity
        or type(prebuild) is not OCIPrebuildResult
        or type(publication) is not NativeArtifactPublication
        or prebuild.launch_digest != execution.launch_digest
        or prebuild.build_spec_digest != publication.build_spec_digest
        or publication.publication_digest != execution.native_publication_digest
        or quiescence.namespace_digest != lane.executor_namespace_digest
    ):
        raise ResidentPairRetirementHold(
            f"resident lane {lane.lane_id} lifetime authority is foreign"
        )
    return ResidentLaneRetirementProjection(
        lane.lane_id,
        lane.session_id,
        lane.stock_launch_digest,
        lane.lane_digest,
        lane.allocation_digest,
        lane.executor_namespace_digest,
        execution.schema,
        execution.runtime_identity,
        execution.runtime_preflight_receipt_sha256,
        execution.arena_model_receipt_digest,
        execution.resource_policy_digest,
        prebuild.build_spec_digest,
        execution.native_publication_digest,
        execution.runtime_argv_sha256,
        execution.recovered_lease_ids,
        execution.session,
        execution.device_receipts,
        quiescence,
        quiescence.digest,
    )


def _derive(
    *,
    binding: ResidentPairRuntimeBinding,
    speed_plan: ResidentPairCrossoverPlan,
    speed_evidence: ResidentPairCrossoverEvidence,
    count_plan: ResidentCountQualityExecutionPlan,
    count_evidence: ResidentCountQualityExecutionEvidence,
    count_observation: ResidentCountQualityObservation,
    retirement: ResidentEvaluationRetirementEvidence,
    lane_closures: PairLaneClosures,
) -> ResidentPairRetirementEvidence:
    if (
        type(binding) is not ResidentPairRuntimeBinding
        or type(speed_plan) is not ResidentPairCrossoverPlan
        or type(speed_evidence) is not ResidentPairCrossoverEvidence
        or type(count_plan) is not ResidentCountQualityExecutionPlan
        or type(count_evidence) is not ResidentCountQualityExecutionEvidence
        or type(count_observation) is not ResidentCountQualityObservation
    ):
        raise ResidentPairRetirementError(
            "resident pair retirement authorities are not exactly typed"
        )
    if (
        speed_plan.pair_binding != binding
        or speed_evidence.pair_binding != binding
        or count_plan.pair_binding != binding
        or count_evidence.pair_binding != binding
        or speed_plan.candidate_bundle_digest
        != count_plan.candidate_bundle_digest
        or speed_evidence.candidate_bundle_digest
        != count_evidence.candidate_bundle_digest
    ):
        raise ResidentPairRetirementHold(
            "resident speed and count authorities name another pair or candidate"
        )
    try:
        speed_evidence.regrade(speed_plan)
        speed_digest = speed_evidence.digest
    except ResidentPairCrossoverError as exc:
        raise ResidentPairRetirementHold(
            f"resident speed evidence cannot retire the pair: {exc}"
        ) from None
    _validate_count(count_plan, count_evidence, count_observation)
    request_ids, canonical_ids, history_digest = _validate_history(
        retirement, speed_evidence, count_evidence
    )
    closures = _canonical_closures(binding, retirement, lane_closures)
    projections: list[ResidentLaneRetirementProjection] = []
    for lane, closure, count_slice in zip(
        binding.lanes, closures, count_evidence.request_slices, strict=True
    ):
        execution, quiescence = closure
        _validate_lane_stream(
            execution,
            lane=lane,
            speed_slices=speed_evidence.request_slices,
            count_slice=count_slice,
        )
        projections.append(_project_lane(lane, execution, quiescence))
    lane_a, lane_b = projections
    return ResidentPairRetirementEvidence(
        binding,
        speed_plan.digest,
        speed_digest,
        count_plan.digest,
        count_evidence.digest,
        count_observation.digest,
        request_ids,
        canonical_ids,
        history_digest,
        lane_a,
        lane_b,
        max(
            lane_a.quiescence.observed_monotonic_s,
            lane_b.quiescence.observed_monotonic_s,
        ),
    )


def build_resident_pair_retirement(
    *,
    binding: ResidentPairRuntimeBinding,
    speed_plan: ResidentPairCrossoverPlan,
    speed_evidence: ResidentPairCrossoverEvidence,
    count_plan: ResidentCountQualityExecutionPlan,
    count_evidence: ResidentCountQualityExecutionEvidence,
    count_observation: ResidentCountQualityObservation,
    retirement: ResidentEvaluationRetirementEvidence,
    lane_closures: PairLaneClosures,
) -> ResidentPairRetirementEvidence:
    """Build one closed product from the exact live pair's terminal evidence."""

    return _derive(
        binding=binding,
        speed_plan=speed_plan,
        speed_evidence=speed_evidence,
        count_plan=count_plan,
        count_evidence=count_evidence,
        count_observation=count_observation,
        retirement=retirement,
        lane_closures=lane_closures,
    )


def regrade_resident_pair_retirement(
    evidence: ResidentPairRetirementEvidence,
    *,
    binding: ResidentPairRuntimeBinding,
    speed_plan: ResidentPairCrossoverPlan,
    speed_evidence: ResidentPairCrossoverEvidence,
    count_plan: ResidentCountQualityExecutionPlan,
    count_evidence: ResidentCountQualityExecutionEvidence,
    count_observation: ResidentCountQualityObservation,
    retirement: ResidentEvaluationRetirementEvidence,
    lane_closures: PairLaneClosures,
) -> ResidentPairRetirementEvidence:
    """Rebuild every claim from external authorities; trust no stored verdict."""

    if type(evidence) is not ResidentPairRetirementEvidence:
        raise ResidentPairRetirementError(
            "resident pair retirement evidence is not exact"
        )
    expected = _derive(
        binding=binding,
        speed_plan=speed_plan,
        speed_evidence=speed_evidence,
        count_plan=count_plan,
        count_evidence=count_evidence,
        count_observation=count_observation,
        retirement=retirement,
        lane_closures=lane_closures,
    )
    if evidence != expected or evidence.digest != expected.digest:
        raise ResidentPairRetirementHold(
            "resident pair retirement projection differs from external evidence"
        )
    return evidence


__all__ = [
    "RESIDENT_LANE_RETIREMENT_SCHEMA",
    "RESIDENT_PAIR_HISTORY_SCHEMA",
    "RESIDENT_PAIR_RETIREMENT_SCHEMA",
    "LaneClosure",
    "PairLaneClosures",
    "ResidentLaneRetirementProjection",
    "ResidentPairRetirementError",
    "ResidentPairRetirementEvidence",
    "ResidentPairRetirementHold",
    "build_resident_pair_retirement",
    "regrade_resident_pair_retirement",
]

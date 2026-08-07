"""Concurrent candidate-only count-quality execution on one resident TP pair."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from cacheon.eval.numeric_answer_judge import (
    NumericAnswerHiddenJudge,
    NumericAnswerJudgeError,
    NumericAnswerPromptOccurrence,
    derive_numeric_answer_prompt_occurrences,
    numeric_answer_prompt_plan_digest,
)
from cacheon.eval.oci_resident_session import ResidentBatchEvidence, ResidentBatchShape
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.resident_count_quality import (
    ResidentCountPromptObservation,
    ResidentCountQualityEnvelope,
    ResidentCountQualityError,
    ResidentCountQualityInfrastructureError,
    ResidentCountQualityObservation,
)
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationPair,
    ResidentEvaluationPairError,
    ResidentLaneRequest,
    ResidentRequestResult,
    ResidentRequestSlice,
)
from cacheon.eval.resident_request_deadline import require_resident_request_deadline
from cacheon.stack_identity import StackIdentityError, canonical_digest, require_sha256_hex


RESIDENT_COUNT_ADMISSION_SCHEMA = "cacheon.eval.resident-count-lane-admission.v1"
RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA = "cacheon.eval.resident-count-execution-plan.v1"


class ResidentCountQualityExecutionError(ResidentCountQualityError):
    """The candidate-only execution plan or returned lane evidence is invalid."""


class ResidentCountQualityExecutionHold(ResidentCountQualityInfrastructureError):
    """Resident execution became ambiguous and must not be rerun automatically."""

    decision = "HOLD"


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (StackIdentityError, TypeError, ValueError) as exc:
        raise ResidentCountQualityExecutionError(str(exc)) from None


def _positive(value: object, field: str, *, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ResidentCountQualityExecutionError(
            f"{field} must be an integer in [1, {maximum}]"
        )
    return value


def resident_batch_shape_digest(shape: ResidentBatchShape) -> str:
    if type(shape) is not ResidentBatchShape:
        raise ResidentCountQualityExecutionError("resident quality batch shape is not exact")
    return canonical_digest(
        "cacheon.eval.resident-count-batch-shape.v1",
        {
            "max_new_tokens": shape.max_new_tokens,
            "temperature": format(shape.temperature, ".17g"),
            "top_logprobs_num": shape.top_logprobs_num,
        },
    )


@dataclass(frozen=True)
class ResidentCountLaneAdmission:
    """Sealed full-admission and physical-allocation identities for two lanes."""

    lane_a_prompt_count: int
    lane_b_prompt_count: int
    engine_max_running_requests: int
    lane_a_allocation_digest: str
    lane_b_allocation_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "lane_a_prompt_count",
            "lane_b_prompt_count",
            "engine_max_running_requests",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), field_name.replace("_", " ")),
            )
        if max(self.lane_a_prompt_count, self.lane_b_prompt_count) > self.engine_max_running_requests:
            raise ResidentCountQualityExecutionError(
                "resident quality prompt half exceeds engine admission capacity"
            )
        for field_name in ("lane_a_allocation_digest", "lane_b_allocation_digest"):
            object.__setattr__(self, field_name, _digest(getattr(self, field_name), field_name))
        if self.lane_a_allocation_digest == self.lane_b_allocation_digest:
            raise ResidentCountQualityExecutionError(
                "resident quality lanes must bind distinct physical allocations"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "engine_max_running_requests": self.engine_max_running_requests,
            "lane_a_allocation_digest": self.lane_a_allocation_digest,
            "lane_a_prompt_count": self.lane_a_prompt_count,
            "lane_b_allocation_digest": self.lane_b_allocation_digest,
            "lane_b_prompt_count": self.lane_b_prompt_count,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(RESIDENT_COUNT_ADMISSION_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class ResidentCountQualityExecutionPlan:
    """Private prompt bytes plus their path-free commissioned execution identity."""

    candidate_bundle_digest: str
    envelope: ResidentCountQualityEnvelope
    prompt_batches: tuple[tuple[str, ...], ...] = field(repr=False)
    selected_ordinals: tuple[int, ...]
    batch_shape: ResidentBatchShape
    admission: ResidentCountLaneAdmission
    selected_occurrences: tuple[NumericAnswerPromptOccurrence, ...] = field(init=False)
    selected_prompts: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_bundle_digest",
            _digest(self.candidate_bundle_digest, "candidate bundle digest"),
        )
        if type(self.envelope) is not ResidentCountQualityEnvelope:
            raise ResidentCountQualityExecutionError("resident quality envelope is not exact")
        if type(self.batch_shape) is not ResidentBatchShape:
            raise ResidentCountQualityExecutionError("resident quality batch shape is not exact")
        if self.batch_shape.top_logprobs_num != 0:
            raise ResidentCountQualityExecutionError(
                "resident count quality requires pure generation without top-logprobs"
            )
        if type(self.admission) is not ResidentCountLaneAdmission:
            raise ResidentCountQualityExecutionError("resident lane admission is not exact")
        try:
            all_occurrences = derive_numeric_answer_prompt_occurrences(
                self.envelope.judge_binding,
                prompt_batches=self.prompt_batches,
                workload_digest=self.envelope.reference.workload_digest,
                hidden_tasks_per_prompt=1,
            )
        except NumericAnswerJudgeError as exc:
            raise ResidentCountQualityExecutionError(
                f"resident quality prompt plan is invalid: {exc}"
            ) from None
        flattened = tuple(prompt for batch in self.prompt_batches for prompt in batch)
        ordinals = tuple(self.selected_ordinals)
        if (
            type(self.selected_ordinals) is not tuple
            or not ordinals
            or ordinals != tuple(sorted(set(ordinals)))
            or any(type(value) is not int or not 0 <= value < len(all_occurrences) for value in ordinals)
        ):
            raise ResidentCountQualityExecutionError(
                "resident quality selected ordinals are not exact, ordered, and unique"
            )
        occurrences = tuple(all_occurrences[index] for index in ordinals)
        prompts = tuple(flattened[index] for index in ordinals)
        if (
            len(occurrences) != self.envelope.expected_prompt_count
            or len(occurrences)
            != self.admission.lane_a_prompt_count + self.admission.lane_b_prompt_count
        ):
            raise ResidentCountQualityExecutionError(
                "resident quality selection differs from envelope or lane coverage"
            )
        try:
            prompt_plan_digest = numeric_answer_prompt_plan_digest(occurrences)
        except NumericAnswerJudgeError as exc:
            raise ResidentCountQualityExecutionError(str(exc)) from None
        if (
            prompt_plan_digest != self.envelope.prompt_plan_digest
            or resident_batch_shape_digest(self.batch_shape)
            != self.envelope.generation_shape_digest
            or self.admission.digest != self.envelope.admission_policy_digest
        ):
            raise ResidentCountQualityExecutionError(
                "resident quality prompt, generation, or admission identity differs from envelope"
            )
        object.__setattr__(self, "selected_ordinals", ordinals)
        object.__setattr__(self, "selected_occurrences", occurrences)
        object.__setattr__(self, "selected_prompts", prompts)

    def to_dict(self) -> dict[str, object]:
        return {
            "admission": self.admission.to_dict(),
            "candidate_bundle_digest": self.candidate_bundle_digest,
            "envelope_digest": self.envelope.digest,
            "generation_shape_digest": resident_batch_shape_digest(self.batch_shape),
            "schema": RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA,
            "selected_ordinals": list(self.selected_ordinals),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA, self.to_dict())


def _batch_identity(row: ResidentBatchEvidence) -> dict[str, object]:
    return {
        "active_slots": list(row.active_slots),
        "batch_index": row.batch_index,
        "canary": row.canary,
        "generation": row.generation,
        "nonce": row.nonce,
        "output_ids": [list(prompt.output_ids) for prompt in row.evidence.prompts],
        "request_id": row.request_id,
        "request_started_at": format(row.request_started_at, ".17g"),
        "response_completed_at": format(row.response_completed_at, ".17g"),
        "token_numerator": row.token_numerator,
    }


def _slice_identity(row: ResidentRequestSlice) -> dict[str, object]:
    return {
        "bundle_digest": row.bundle_digest,
        "ending_bundle_digest": row.ending_bundle_digest,
        "ending_generation": row.ending_generation,
        "ending_slots": list(row.ending_slots),
        "evaluation_id": row.evaluation_id,
        "expected_batch_count": row.expected_batch_count,
        "expected_swap_count": row.expected_swap_count,
        "host_completed_at": format(row.host_completed_at, ".17g"),
        "host_started_at": format(row.host_started_at, ".17g"),
        "lane_id": row.lane_id,
        "new_batches": [_batch_identity(batch) for batch in row.new_batches],
        "new_swaps": [swap.to_dict() for swap in row.new_swaps],
        "request_id": row.request_id,
        "session_id": row.session_id,
        "starting_generation": row.starting_generation,
    }


def _validate_lane_result(
    result: ResidentRequestResult,
    *,
    lane_id: str,
    bundle_digest: str,
    expected_prompts: int,
    shape: ResidentBatchShape,
) -> ResidentBatchEvidence:
    if type(result) is not ResidentRequestResult or not result.ok:
        raise ResidentCountQualityExecutionHold(
            f"resident quality lane {lane_id} did not return exact success"
        )
    request = result.request_slice
    swaps = request.new_swaps if type(request) is ResidentRequestSlice else ()
    if (
        type(request) is not ResidentRequestSlice
        or request.lane_id != lane_id
        or request.bundle_digest != bundle_digest
        or request.expected_batch_count != 1
        or request.expected_swap_count != 2
        or request.ending_bundle_digest is not None
        or request.ending_slots
        or len(request.new_batches) != 1
        or len(swaps) != 2
        or swaps[0].bundle_digest != bundle_digest
        or not swaps[0].slots
        or swaps[1].bundle_digest is not None
        or swaps[1].slots
        or swaps[0].generation != request.starting_generation + 1
        or swaps[1].generation != swaps[0].generation + 1
        or request.ending_generation != swaps[1].generation
    ):
        raise ResidentCountQualityExecutionHold(
            f"resident quality lane {lane_id} slice is incomplete or did not restore stock"
        )
    batch = request.new_batches[0]
    if (
        type(batch) is not ResidentBatchEvidence
        or type(result.value) is not ResidentBatchEvidence
        or result.value != batch
        or type(batch.batch_index) is not int
        or batch.batch_index < 0
        or type(batch.request_id) is not str
        or len(batch.request_id) != 32
        or any(value not in "0123456789abcdef" for value in batch.request_id)
        or type(batch.nonce) is not str
        or len(batch.nonce) != 32
        or any(value not in "0123456789abcdef" for value in batch.nonce)
        or batch.request_id == batch.nonce
        or batch.generation != swaps[0].generation
        or batch.active_slots != swaps[0].slots
        or batch.canary
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (batch.request_started_at, batch.response_completed_at)
        )
        or not (
            request.host_started_at
            <= swaps[0].requested_at
            < swaps[0].completed_at
            <= batch.request_started_at
            < batch.response_completed_at
            <= swaps[1].requested_at
            < swaps[1].completed_at
            <= request.host_completed_at
        )
        or len(batch.evidence.prompts) != expected_prompts
    ):
        raise ResidentCountQualityExecutionHold(
            f"resident quality lane {lane_id} batch evidence is inexact"
        )
    evidence = batch.evidence
    expected_tokens = expected_prompts * shape.max_new_tokens
    if (
        type(evidence) is not BatchEvidence
        or type(evidence.prompts) is not tuple
        or type(batch.token_numerator) is not int
        or batch.token_numerator != expected_tokens
        or evidence.observed_tokens != expected_tokens
        or any(
            type(prompt) is not PromptEvidence
            or type(prompt.output_ids) is not tuple
            or len(prompt.output_ids) != shape.max_new_tokens
            or any(
                type(token) is not int or not 0 <= token <= 2_147_483_647
                for token in prompt.output_ids
            )
            or type(prompt.top_logprobs) is not tuple
            or len(prompt.top_logprobs) != shape.max_new_tokens
            or any(type(position) is not tuple or position for position in prompt.top_logprobs)
            for prompt in evidence.prompts
        )
    ):
        raise ResidentCountQualityExecutionHold(
            f"resident quality lane {lane_id} output evidence is malformed"
        )
    return batch


def execute_candidate_count_quality(
    plan: ResidentCountQualityExecutionPlan,
    *,
    pair: ResidentEvaluationPair,
    judge: NumericAnswerHiddenJudge,
    deadline: float,
) -> ResidentCountQualityObservation:
    """Execute one candidate on both lanes; there is deliberately no stock runner."""

    if type(plan) is not ResidentCountQualityExecutionPlan:
        raise ResidentCountQualityExecutionError("resident quality execution plan is not exact")
    if type(pair) is not ResidentEvaluationPair:
        raise ResidentCountQualityExecutionError("resident evaluation pair is not exact")
    if type(judge) is not NumericAnswerHiddenJudge:
        raise ResidentCountQualityExecutionError("resident numeric judge is not exact")
    request_deadline = require_resident_request_deadline(
        deadline,
        now=time.monotonic(),
        error_type=ResidentCountQualityExecutionError,
    )
    if (
        judge.binding != plan.envelope.judge_binding
        or judge.tokenizer_digest != plan.envelope.reference.tokenizer_digest
    ):
        raise ResidentCountQualityExecutionError(
            "resident numeric judge differs from commissioned quality envelope"
        )
    split = plan.admission.lane_a_prompt_count
    prompts_a, prompts_b = plan.selected_prompts[:split], plan.selected_prompts[split:]

    def operation(prompts: tuple[str, ...]):
        def run(handle):
            handle.swap(plan.candidate_bundle_digest)
            return handle.execute_batch_with_shape(prompts, shape=plan.batch_shape)

        return run

    try:
        result_a, result_b = pair.run_lanes(
            ResidentLaneRequest(plan.candidate_bundle_digest, operation(prompts_a), 1, 2),
            ResidentLaneRequest(plan.candidate_bundle_digest, operation(prompts_b), 1, 2),
            deadline=request_deadline,
        )
    except ResidentEvaluationPairError as exc:
        raise ResidentCountQualityExecutionHold(
            f"resident candidate quality execution is on HOLD: {exc}"
        ) from None
    batch_a = _validate_lane_result(
        result_a,
        lane_id="A",
        bundle_digest=plan.candidate_bundle_digest,
        expected_prompts=len(prompts_a),
        shape=plan.batch_shape,
    )
    batch_b = _validate_lane_result(
        result_b,
        lane_id="B",
        bundle_digest=plan.candidate_bundle_digest,
        expected_prompts=len(prompts_b),
        shape=plan.batch_shape,
    )
    if result_a.request_slice.evaluation_id != result_b.request_slice.evaluation_id:
        raise ResidentCountQualityExecutionHold(
            "resident quality lane slices name different evaluations"
        )
    outputs = tuple(
        prompt.output_ids
        for batch in (batch_a, batch_b)
        for prompt in batch.evidence.prompts
    )
    if len(outputs) != len(plan.selected_occurrences):
        raise ResidentCountQualityExecutionHold(
            "resident quality lane outputs do not cover the selected prompt plan"
        )
    rows: list[ResidentCountPromptObservation] = []
    for ordinal, (occurrence, output_ids) in enumerate(
        zip(plan.selected_occurrences, outputs, strict=True)
    ):
        try:
            receipt = judge(
                prompt_digest=occurrence.prompt_digest,
                output_ids=output_ids,
                task_digests=occurrence.task_digests,
            )
        except NumericAnswerJudgeError as exc:
            raise ResidentCountQualityExecutionHold(
                f"resident quality hidden judging is on HOLD: {exc}"
            ) from None
        rows.append(
            ResidentCountPromptObservation(
                ordinal,
                occurrence.prompt_digest,
                occurrence.task_digests,
                output_ids,
                receipt,
            )
        )
    execution_digest = canonical_digest(
        "cacheon.eval.resident-count-quality-execution-evidence.v1",
        {
            "execution_plan_digest": plan.digest,
            "lane_slices": [
                _slice_identity(result_a.request_slice),
                _slice_identity(result_b.request_slice),
            ],
        },
    )
    return ResidentCountQualityObservation(
        "candidate",
        plan.envelope,
        execution_digest,
        tuple(rows),
    )


__all__ = [
    "RESIDENT_COUNT_ADMISSION_SCHEMA",
    "RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA",
    "ResidentCountLaneAdmission",
    "ResidentCountQualityExecutionError",
    "ResidentCountQualityExecutionHold",
    "ResidentCountQualityExecutionPlan",
    "execute_candidate_count_quality",
    "resident_batch_shape_digest",
]

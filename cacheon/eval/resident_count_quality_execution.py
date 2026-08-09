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
from cacheon.eval.oci_resident_session import (
    ResidentBatchEvidence,
    ResidentBatchShape,
    SwapReceipt,
)
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
    ResidentRequestSlice,
    ResidentEvaluationPairError,
    ResidentLaneRequest,
    ResidentRequestResult,
)
from cacheon.eval.resident_pair_binding import (
    ResidentPairLaneBinding,
    ResidentPairRuntimeBinding,
)
from cacheon.eval.resident_request_deadline import require_resident_request_deadline
from cacheon.stack_identity import StackIdentityError, canonical_digest, require_sha256_hex


RESIDENT_COUNT_ADMISSION_SCHEMA = "cacheon.eval.resident-count-lane-admission.v1"
RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA = "cacheon.eval.resident-count-execution-plan.v2"


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


RESIDENT_COUNT_EXECUTION_EVIDENCE_SCHEMA = (
    "cacheon.eval.resident-count-quality-execution-evidence.v1"
)


class ResidentCountExecutionEvidenceError(ValueError):
    """Raw resident count evidence is malformed, incomplete, or ambiguous."""


def _evidence_digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (StackIdentityError, TypeError, ValueError) as exc:
        raise ResidentCountExecutionEvidenceError(str(exc)) from None


def _hex32(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 32
        and value != "0" * 32
        and all(char in "0123456789abcdef" for char in value)
    )


def _finite(value: object) -> bool:
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _binding_to_dict(binding: ResidentPairRuntimeBinding) -> dict[str, object]:
    return {
        "lanes": [
            {
                "allocation_digest": row.allocation_digest,
                "executor_namespace_digest": row.executor_namespace_digest,
                "lane_digest": row.lane_digest,
                "lane_id": row.lane_id,
                "session_id": row.session_id,
                "stock_launch_digest": row.stock_launch_digest,
            }
            for row in binding.lanes
        ],
        "service_epoch_digest": binding.service_epoch_digest,
    }


def _prompt_to_dict(row: PromptEvidence) -> dict[str, object]:
    return {
        "output_ids": list(row.output_ids),
        "top_logprobs": [
            [[format(score, ".17g"), token] for score, token in position]
            for position in row.top_logprobs
        ],
    }


def _batch_to_dict(row: ResidentBatchEvidence) -> dict[str, object]:
    return {
        "active_slots": list(row.active_slots),
        "batch_index": row.batch_index,
        "canary": row.canary,
        "evidence": {
            "prompts": [_prompt_to_dict(prompt) for prompt in row.evidence.prompts]
        },
        "generation": row.generation,
        "nonce": row.nonce,
        "request_id": row.request_id,
        "request_started_at": format(row.request_started_at, ".17g"),
        "response_completed_at": format(row.response_completed_at, ".17g"),
        "token_numerator": row.token_numerator,
    }


def _slice_to_dict(row: ResidentRequestSlice) -> dict[str, object]:
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
        "new_batches": [_batch_to_dict(batch) for batch in row.new_batches],
        "new_swaps": [swap.to_dict() for swap in row.new_swaps],
        "request_id": row.request_id,
        "session_id": row.session_id,
        "starting_generation": row.starting_generation,
    }


def require_resident_count_request_slice(
    row: ResidentRequestSlice,
    *,
    lane_binding: ResidentPairLaneBinding,
    candidate_bundle_digest: str,
    expected_prompt_count: int | None = None,
    max_new_tokens: int | None = None,
) -> ResidentBatchEvidence:
    """Revalidate one candidate activation, count read, and stock restoration."""

    candidate = _evidence_digest(candidate_bundle_digest, "candidate bundle digest")
    if type(lane_binding) is not ResidentPairLaneBinding:
        raise ResidentCountExecutionEvidenceError("count lane binding is not exact")
    if type(row) is not ResidentRequestSlice:
        raise ResidentCountExecutionEvidenceError("count request slice is not exact")
    if (expected_prompt_count is None) != (max_new_tokens is None):
        raise ResidentCountExecutionEvidenceError("count output bounds are incomplete")
    batches, swaps = row.new_batches, row.new_swaps
    if (
        row.lane_id != lane_binding.lane_id
        or row.session_id != lane_binding.session_id
        or row.bundle_digest != candidate
        or row.expected_batch_count != 1
        or row.expected_swap_count != 2
        or row.ending_bundle_digest is not None
        or row.ending_slots
        or len(batches) != 1
        or len(swaps) != 2
        or any(type(swap) is not SwapReceipt for swap in swaps)
    ):
        raise ResidentCountExecutionEvidenceError(
            f"count lane {lane_binding.lane_id} request is incomplete"
        )
    activation, restoration = swaps
    if (
        activation.bundle_digest != candidate
        or not activation.slots
        or restoration.bundle_digest is not None
        or restoration.slots
        or activation.generation != row.starting_generation + 1
        or restoration.generation != activation.generation + 1
        or restoration.swap_index != activation.swap_index + 1
        or row.ending_generation != restoration.generation
    ):
        raise ResidentCountExecutionEvidenceError(
            f"count lane {lane_binding.lane_id} swap sequence is ambiguous"
        )
    batch = batches[0]
    evidence = batch.evidence if type(batch) is ResidentBatchEvidence else None
    prompts = evidence.prompts if type(evidence) is BatchEvidence else ()
    if (
        type(batch) is not ResidentBatchEvidence
        or type(evidence) is not BatchEvidence
        or type(evidence.prompts) is not tuple
        or not prompts
        or type(batch.batch_index) is not int
        or batch.batch_index < 0
        or not _hex32(batch.request_id)
        or not _hex32(batch.nonce)
        or batch.request_id == batch.nonce
        or batch.generation != activation.generation
        or batch.active_slots != activation.slots
        or type(batch.canary) is not bool
        or batch.canary
        or type(batch.token_numerator) is not int
        or batch.token_numerator < 1
        or batch.token_numerator != evidence.observed_tokens
        or not all(
            _finite(value)
            for value in (
                row.host_started_at,
                activation.requested_at,
                activation.completed_at,
                batch.request_started_at,
                batch.response_completed_at,
                restoration.requested_at,
                restoration.completed_at,
                row.host_completed_at,
            )
        )
        or not (
            row.host_started_at
            <= activation.requested_at
            < activation.completed_at
            <= batch.request_started_at
            < batch.response_completed_at
            <= restoration.requested_at
            < restoration.completed_at
            <= row.host_completed_at
        )
    ):
        raise ResidentCountExecutionEvidenceError(
            f"count lane {lane_binding.lane_id} batch identity is malformed"
        )
    if any(
        type(prompt) is not PromptEvidence
        or type(prompt.output_ids) is not tuple
        or not prompt.output_ids
        or any(
            type(token) is not int or not 0 <= token <= 2_147_483_647
            for token in prompt.output_ids
        )
        or type(prompt.top_logprobs) is not tuple
        or len(prompt.top_logprobs) != len(prompt.output_ids)
        or any(type(position) is not tuple or position for position in prompt.top_logprobs)
        for prompt in prompts
    ):
        raise ResidentCountExecutionEvidenceError(
            f"count lane {lane_binding.lane_id} output identity is malformed"
        )
    if expected_prompt_count is not None and (
        type(expected_prompt_count) is not int
        or expected_prompt_count < 1
        or type(max_new_tokens) is not int
        or max_new_tokens < 1
        or len(prompts) != expected_prompt_count
        or any(len(prompt.output_ids) != max_new_tokens for prompt in prompts)
        or batch.token_numerator != expected_prompt_count * max_new_tokens
    ):
        raise ResidentCountExecutionEvidenceError(
            f"count lane {lane_binding.lane_id} output coverage differs from plan"
        )
    return batch


@dataclass(frozen=True)
class ResidentCountQualityExecutionEvidence:
    """Path-free full A/B request slices bound to one commissioned execution."""

    execution_plan_digest: str
    candidate_bundle_digest: str
    envelope_digest: str
    pair_binding: ResidentPairRuntimeBinding
    request_slices: tuple[ResidentRequestSlice, ResidentRequestSlice]
    schema: str = RESIDENT_COUNT_EXECUTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESIDENT_COUNT_EXECUTION_EVIDENCE_SCHEMA:
            raise ResidentCountExecutionEvidenceError(
                "resident count execution evidence schema is unsupported"
            )
        for name in (
            "execution_plan_digest",
            "candidate_bundle_digest",
            "envelope_digest",
        ):
            object.__setattr__(
                self, name, _evidence_digest(getattr(self, name), name.replace("_", " "))
            )
        if type(self.pair_binding) is not ResidentPairRuntimeBinding:
            raise ResidentCountExecutionEvidenceError(
                "resident count pair binding is not exact"
            )
        if (
            type(self.request_slices) is not tuple
            or len(self.request_slices) != 2
            or any(type(row) is not ResidentRequestSlice for row in self.request_slices)
            or tuple(row.lane_id for row in self.request_slices) != ("A", "B")
        ):
            raise ResidentCountExecutionEvidenceError(
                "resident count evidence requires canonical A/B request slices"
            )
        for row, binding in zip(
            self.request_slices, self.pair_binding.lanes, strict=True
        ):
            require_resident_count_request_slice(
                row,
                lane_binding=binding,
                candidate_bundle_digest=self.candidate_bundle_digest,
            )
        lane_a, lane_b = self.request_slices
        batches = (lane_a.new_batches[0], lane_b.new_batches[0])
        if (
            lane_a.evaluation_id != lane_b.evaluation_id
            or lane_a.request_id == lane_b.request_id
            or batches[0].request_id == batches[1].request_id
            or batches[0].nonce == batches[1].nonce
        ):
            raise ResidentCountExecutionEvidenceError(
                "resident count A/B execution identities are not distinct and shared"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_bundle_digest": self.candidate_bundle_digest,
            "envelope_digest": self.envelope_digest,
            "execution_plan_digest": self.execution_plan_digest,
            "pair_binding": _binding_to_dict(self.pair_binding),
            "request_slices": [_slice_to_dict(row) for row in self.request_slices],
            "schema": self.schema,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.schema, self.to_dict())

    def regrade(
        self,
        plan: ResidentCountQualityExecutionPlan,
        judge: NumericAnswerHiddenJudge,
    ) -> ResidentCountQualityObservation:
        """Recompute the observation through the independent typed regrader."""

        return regrade_candidate_count_quality_execution(
            self, plan=plan, judge=judge
        )



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
    pair_binding: ResidentPairRuntimeBinding
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
        if type(self.pair_binding) is not ResidentPairRuntimeBinding:
            raise ResidentCountQualityExecutionError(
                "resident pair runtime binding is not exact"
            )
        if (
            self.admission.lane_a_allocation_digest,
            self.admission.lane_b_allocation_digest,
        ) != tuple(row.allocation_digest for row in self.pair_binding.lanes):
            raise ResidentCountQualityExecutionError(
                "resident lane admission allocations differ from pair binding"
            )
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
            "pair_binding_digest": self.pair_binding.digest,
            "schema": RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA,
            "selected_ordinals": list(self.selected_ordinals),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class ResidentCountQualityExecutionResult:
    """Raw resident execution evidence and its independent count observation."""

    evidence: ResidentCountQualityExecutionEvidence
    observation: ResidentCountQualityObservation

    def __post_init__(self) -> None:
        if (
            type(self.evidence) is not ResidentCountQualityExecutionEvidence
            or type(self.observation) is not ResidentCountQualityObservation
            or self.observation.role != "candidate"
            or self.observation.envelope.digest != self.evidence.envelope_digest
            or self.observation.execution_evidence_digest != self.evidence.digest
        ):
            raise ResidentCountQualityExecutionError(
                "resident count execution result is not exactly bound"
            )

    @property
    def execution_evidence_digest(self) -> str:
        """Compatibility projection for callers that only compare raw digests."""

        return self.observation.execution_evidence_digest


def _require_plan_and_judge(
    plan: ResidentCountQualityExecutionPlan,
    judge: NumericAnswerHiddenJudge,
) -> None:
    if type(plan) is not ResidentCountQualityExecutionPlan:
        raise ResidentCountQualityExecutionError(
            "resident quality execution plan is not exact"
        )
    if type(judge) is not NumericAnswerHiddenJudge:
        raise ResidentCountQualityExecutionError("resident numeric judge is not exact")
    if (
        judge.binding != plan.envelope.judge_binding
        or judge.tokenizer_digest != plan.envelope.reference.tokenizer_digest
    ):
        raise ResidentCountQualityExecutionError(
            "resident numeric judge differs from commissioned quality envelope"
        )


def regrade_candidate_count_quality_execution(
    evidence: ResidentCountQualityExecutionEvidence,
    *,
    plan: ResidentCountQualityExecutionPlan,
    judge: NumericAnswerHiddenJudge,
) -> ResidentCountQualityObservation:
    """Derive candidate count quality only from one closed raw A/B product."""

    _require_plan_and_judge(plan, judge)
    if type(evidence) is not ResidentCountQualityExecutionEvidence:
        raise ResidentCountQualityExecutionError(
            "resident count execution evidence is not exact"
        )
    if (
        evidence.execution_plan_digest != plan.digest
        or evidence.candidate_bundle_digest != plan.candidate_bundle_digest
        or evidence.envelope_digest != plan.envelope.digest
        or evidence.pair_binding != plan.pair_binding
    ):
        raise ResidentCountQualityExecutionHold(
            "resident count raw evidence differs from its commissioned plan"
        )
    prompt_counts = (
        plan.admission.lane_a_prompt_count,
        plan.admission.lane_b_prompt_count,
    )
    batches = []
    for request, binding, prompt_count in zip(
        evidence.request_slices,
        plan.pair_binding.lanes,
        prompt_counts,
        strict=True,
    ):
        try:
            batch = require_resident_count_request_slice(
                request,
                lane_binding=binding,
                candidate_bundle_digest=plan.candidate_bundle_digest,
                expected_prompt_count=prompt_count,
                max_new_tokens=plan.batch_shape.max_new_tokens,
            )
        except ResidentCountExecutionEvidenceError as exc:
            raise ResidentCountQualityExecutionHold(
                f"resident count raw slice is on HOLD: {exc}"
            ) from None
        batches.append(batch)
    outputs = tuple(
        prompt.output_ids
        for batch in batches
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
            rows.append(
                ResidentCountPromptObservation(
                    ordinal,
                    occurrence.prompt_digest,
                    occurrence.task_digests,
                    output_ids,
                    receipt,
                )
            )
        except (NumericAnswerJudgeError, ResidentCountQualityError) as exc:
            raise ResidentCountQualityExecutionHold(
                f"resident quality hidden judging is on HOLD: {exc}"
            ) from None
    try:
        return ResidentCountQualityObservation(
            "candidate", plan.envelope, evidence.digest, tuple(rows)
        )
    except ResidentCountQualityError as exc:
        raise ResidentCountQualityExecutionHold(
            f"resident quality observation is on HOLD: {exc}"
        ) from None


def execute_candidate_count_quality(
    plan: ResidentCountQualityExecutionPlan,
    *,
    pair: ResidentEvaluationPair,
    judge: NumericAnswerHiddenJudge,
    deadline: float,
) -> ResidentCountQualityExecutionResult:
    """Execute one candidate on both lanes; there is deliberately no stock runner."""

    _require_plan_and_judge(plan, judge)
    if type(pair) is not ResidentEvaluationPair:
        raise ResidentCountQualityExecutionError("resident evaluation pair is not exact")
    request_deadline = require_resident_request_deadline(
        deadline,
        now=time.monotonic(),
        error_type=ResidentCountQualityExecutionError,
    )
    if (
        plan.admission.lane_a_allocation_digest,
        plan.admission.lane_b_allocation_digest,
    ) != tuple(row.allocation_digest for row in plan.pair_binding.lanes):
        raise ResidentCountQualityExecutionError(
            "resident lane admission allocations differ from pair binding"
        )
    try:
        live_identities = pair.identities
    except ResidentEvaluationPairError as exc:
        raise ResidentCountQualityExecutionHold(
            f"resident candidate quality pair is on HOLD: {exc}"
        ) from None
    if live_identities != plan.pair_binding.identities:
        raise ResidentCountQualityExecutionHold(
            "resident candidate quality pair sessions differ from binding"
        )
    split = plan.admission.lane_a_prompt_count
    prompts_a, prompts_b = (
        plan.selected_prompts[:split],
        plan.selected_prompts[split:],
    )

    def operation(prompts: tuple[str, ...]):
        def run(handle):
            handle.swap(plan.candidate_bundle_digest)
            return handle.execute_batch_with_shape(
                prompts, shape=plan.batch_shape
            )

        return run

    try:
        result_a, result_b = pair.run_lanes(
            ResidentLaneRequest(
                plan.candidate_bundle_digest, operation(prompts_a), 1, 2
            ),
            ResidentLaneRequest(
                plan.candidate_bundle_digest, operation(prompts_b), 1, 2
            ),
            deadline=request_deadline,
        )
    except ResidentEvaluationPairError as exc:
        raise ResidentCountQualityExecutionHold(
            f"resident candidate quality execution is on HOLD: {exc}"
        ) from None
    results = (result_a, result_b)
    if any(
        type(result) is not ResidentRequestResult
        or not result.ok
        or len(result.request_slice.new_batches) != 1
        or result.value != result.request_slice.new_batches[0]
        for result in results
    ):
        raise ResidentCountQualityExecutionHold(
            "resident candidate quality results are not exact raw slices"
        )
    try:
        evidence = ResidentCountQualityExecutionEvidence(
            plan.digest,
            plan.candidate_bundle_digest,
            plan.envelope.digest,
            plan.pair_binding,
            tuple(result.request_slice for result in results),
        )
    except ResidentCountExecutionEvidenceError as exc:
        raise ResidentCountQualityExecutionHold(
            f"resident candidate raw evidence is on HOLD: {exc}"
        ) from None
    observation = regrade_candidate_count_quality_execution(
        evidence, plan=plan, judge=judge
    )
    return ResidentCountQualityExecutionResult(evidence, observation)


__all__ = [
    "RESIDENT_COUNT_ADMISSION_SCHEMA",
    "RESIDENT_COUNT_EXECUTION_EVIDENCE_SCHEMA",
    "ResidentCountExecutionEvidenceError",
    "ResidentCountQualityExecutionEvidence",
    "require_resident_count_request_slice",
    "RESIDENT_COUNT_EXECUTION_PLAN_SCHEMA",
    "ResidentCountLaneAdmission",
    "ResidentCountQualityExecutionError",
    "ResidentCountQualityExecutionHold",
    "ResidentCountQualityExecutionPlan",
    "ResidentCountQualityExecutionResult",
    "execute_candidate_count_quality",
    "regrade_candidate_count_quality_execution",
    "resident_batch_shape_digest",
]

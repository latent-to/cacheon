"""Closed raw A/B evidence for one resident count-quality execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cacheon.eval.oci_resident_session import ResidentBatchEvidence, SwapReceipt
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.resident_evaluation_pair import ResidentRequestSlice
from cacheon.eval.resident_pair_binding import (
    ResidentPairLaneBinding,
    ResidentPairRuntimeBinding,
)
from cacheon.stack_identity import StackIdentityError, canonical_digest, require_sha256_hex

if TYPE_CHECKING:
    from cacheon.eval.numeric_answer_judge import NumericAnswerHiddenJudge
    from cacheon.eval.resident_count_quality import ResidentCountQualityObservation
    from cacheon.eval.resident_count_quality_execution import (
        ResidentCountQualityExecutionPlan,
    )


RESIDENT_COUNT_EXECUTION_EVIDENCE_SCHEMA = (
    "cacheon.eval.resident-count-quality-execution-evidence.v1"
)


class ResidentCountExecutionEvidenceError(ValueError):
    """Raw resident count evidence is malformed, incomplete, or ambiguous."""


def _digest(value: object, field: str) -> str:
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

    candidate = _digest(candidate_bundle_digest, "candidate bundle digest")
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
        for field in (
            "execution_plan_digest",
            "candidate_bundle_digest",
            "envelope_digest",
        ):
            object.__setattr__(
                self, field, _digest(getattr(self, field), field.replace("_", " "))
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

        from cacheon.eval.resident_count_quality_execution import (
            regrade_candidate_count_quality_execution,
        )

        return regrade_candidate_count_quality_execution(
            self, plan=plan, judge=judge
        )


__all__ = [
    "RESIDENT_COUNT_EXECUTION_EVIDENCE_SCHEMA",
    "ResidentCountExecutionEvidenceError",
    "ResidentCountQualityExecutionEvidence",
    "require_resident_count_request_slice",
]

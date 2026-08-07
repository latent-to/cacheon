"""Typed fixed-stock count quality over retained resident output evidence.

This module performs no model execution and cannot rerun stock.  It reopens a
closed observation shape, rejudges every retained output token sequence with the
validator-owned hidden judge, and projects only the exact counts into
``count_quality`` arithmetic.  Publication and exactly-once execution belong to
separate authorities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cacheon.eval.count_quality import (
    CountQualityEvidence,
    CountQualityPolicy,
    CountQualityVerdict,
    score_count_quality,
)
from cacheon.eval.numeric_answer_judge import (
    NumericAnswerHiddenJudge,
    NumericAnswerJudgeError,
)
from cacheon.eval.qualification import QualificationError, ReferenceManifest
from cacheon.eval.qualification_runner import (
    HiddenJudgeBinding,
    HiddenJudgeReceipt,
    QualificationRunnerError,
    hidden_judge_output_digest,
)
from cacheon.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


RESIDENT_COUNT_ENVELOPE_SCHEMA = "cacheon.eval.resident-count-quality-envelope.v1"
RESIDENT_COUNT_OBSERVATION_SCHEMA = "cacheon.eval.resident-count-quality-observation.v1"
RESIDENT_COUNT_RESULT_SCHEMA = "cacheon.eval.resident-count-quality-result.v1"
RESIDENT_COUNT_ROLES = frozenset({"stock", "candidate"})
MAX_QUALITY_PROMPTS = 4096
MAX_OUTPUT_TOKENS_PER_PROMPT = 4096


class ResidentCountQualityError(ValueError):
    """The fixed-stock quality authority or retained evidence is malformed."""


class ResidentCountQualityInfrastructureError(ResidentCountQualityError):
    """Trusted reopening or hidden judging failed; this is never candidate FAIL."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (StackIdentityError, TypeError, ValueError) as exc:
        raise ResidentCountQualityError(str(exc)) from None


def _strict(value: object, fields: frozenset[str], field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ResidentCountQualityError(f"{field} fields do not match the closed schema")
    return value


def _integer(value: object, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ResidentCountQualityError(
            f"{field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _canonical_object(payload: bytes) -> dict[str, object]:
    def reject_number(value: str) -> object:
        raise ResidentCountQualityError(
            f"resident count observation contains unsupported number {value!r}"
        )

    def unique(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ResidentCountQualityError(
                    f"resident count observation repeats key {key!r}"
                )
            result[key] = value
        return result

    if type(payload) is not bytes or not payload:
        raise ResidentCountQualityError("resident count observation bytes are empty")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=reject_number,
            parse_constant=reject_number,
            object_pairs_hook=unique,
        )
    except ResidentCountQualityError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResidentCountQualityError(
            f"resident count observation is malformed: {exc}"
        ) from None
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ResidentCountQualityError(
            "resident count observation is not exact canonical JSON"
        )
    return value


def _binding_to_dict(value: HiddenJudgeBinding) -> dict[str, str]:
    return {
        "hidden_corpus_commitment": value.hidden_corpus_commitment,
        "hidden_judge_digest": value.hidden_judge_digest,
        "hidden_task_policy_digest": value.hidden_task_policy_digest,
    }


def _binding_from_dict(value: object) -> HiddenJudgeBinding:
    row = _strict(
        value,
        frozenset(
            {
                "hidden_corpus_commitment",
                "hidden_judge_digest",
                "hidden_task_policy_digest",
            }
        ),
        "hidden judge binding",
    )
    try:
        return HiddenJudgeBinding(**row)  # type: ignore[arg-type]
    except (QualificationRunnerError, TypeError, ValueError) as exc:
        raise ResidentCountQualityError(f"hidden judge binding is invalid: {exc}") from None


def _receipt_to_dict(value: HiddenJudgeReceipt) -> dict[str, object]:
    return {
        "binding_digest": value.binding_digest,
        "output_ids_digest": value.output_ids_digest,
        "passed": list(value.passed),
        "prompt_digest": value.prompt_digest,
        "task_digests": list(value.task_digests),
    }


def _receipt_from_dict(value: object) -> HiddenJudgeReceipt:
    row = _strict(
        value,
        frozenset(
            {
                "binding_digest",
                "output_ids_digest",
                "passed",
                "prompt_digest",
                "task_digests",
            }
        ),
        "hidden judge receipt",
    )
    if type(row["task_digests"]) is not list or type(row["passed"]) is not list:
        raise ResidentCountQualityError("hidden judge receipt arrays are malformed")
    try:
        return HiddenJudgeReceipt(
            row["binding_digest"],  # type: ignore[arg-type]
            row["prompt_digest"],  # type: ignore[arg-type]
            row["output_ids_digest"],  # type: ignore[arg-type]
            tuple(row["task_digests"]),  # type: ignore[arg-type]
            tuple(row["passed"]),  # type: ignore[arg-type]
        )
    except (QualificationRunnerError, TypeError, ValueError) as exc:
        raise ResidentCountQualityError(f"hidden judge receipt is invalid: {exc}") from None


@dataclass(frozen=True)
class ResidentCountQualityEnvelope:
    """Frozen fidelity identity shared by reusable stock and one candidate arm."""

    reference: ReferenceManifest
    judge_binding: HiddenJudgeBinding
    prompt_plan_digest: str
    generation_shape_digest: str
    admission_policy_digest: str
    expected_prompt_count: int

    def __post_init__(self) -> None:
        if type(self.reference) is not ReferenceManifest:
            raise ResidentCountQualityError("quality reference manifest is not exact")
        if type(self.judge_binding) is not HiddenJudgeBinding:
            raise ResidentCountQualityError("quality hidden judge binding is not exact")
        if (
            self.reference.hidden_corpus_commitment
            != self.judge_binding.hidden_corpus_commitment
            or self.reference.hidden_judge_digest
            != self.judge_binding.hidden_judge_digest
        ):
            raise ResidentCountQualityError(
                "quality reference differs from the hidden judge authority"
            )
        for field in (
            "prompt_plan_digest",
            "generation_shape_digest",
            "admission_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self,
            "expected_prompt_count",
            _integer(
                self.expected_prompt_count,
                "expected prompt count",
                minimum=1,
                maximum=MAX_QUALITY_PROMPTS,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "admission_policy_digest": self.admission_policy_digest,
            "expected_prompt_count": self.expected_prompt_count,
            "generation_shape_digest": self.generation_shape_digest,
            "judge_binding": _binding_to_dict(self.judge_binding),
            "prompt_plan_digest": self.prompt_plan_digest,
            "reference": self.reference.to_dict(),
            "schema": RESIDENT_COUNT_ENVELOPE_SCHEMA,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ResidentCountQualityEnvelope":
        row = _strict(
            value,
            frozenset(
                {
                    "admission_policy_digest",
                    "expected_prompt_count",
                    "generation_shape_digest",
                    "judge_binding",
                    "prompt_plan_digest",
                    "reference",
                    "schema",
                }
            ),
            "resident count quality envelope",
        )
        if row["schema"] != RESIDENT_COUNT_ENVELOPE_SCHEMA:
            raise ResidentCountQualityError("resident count envelope schema is unsupported")
        try:
            reference = ReferenceManifest.from_dict(row["reference"])
        except (QualificationError, TypeError, ValueError) as exc:
            raise ResidentCountQualityError(f"quality reference is invalid: {exc}") from None
        return cls(
            reference,
            _binding_from_dict(row["judge_binding"]),
            row["prompt_plan_digest"],  # type: ignore[arg-type]
            row["generation_shape_digest"],  # type: ignore[arg-type]
            row["admission_policy_digest"],  # type: ignore[arg-type]
            row["expected_prompt_count"],  # type: ignore[arg-type]
        )

    @property
    def digest(self) -> str:
        return canonical_digest(RESIDENT_COUNT_ENVELOPE_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class ResidentCountPromptObservation:
    """One retained output and the exact hidden-judge receipt derived from it."""

    ordinal: int
    prompt_digest: str
    task_digests: tuple[str, ...]
    output_ids: tuple[int, ...]
    judge_receipt: HiddenJudgeReceipt

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ordinal",
            _integer(self.ordinal, "prompt ordinal", minimum=0, maximum=MAX_QUALITY_PROMPTS - 1),
        )
        object.__setattr__(self, "prompt_digest", _digest(self.prompt_digest, "prompt digest"))
        tasks = tuple(self.task_digests)
        if (
            type(self.task_digests) is not tuple
            or len(tasks) != 1
            or tasks != tuple(sorted(set(tasks)))
        ):
            raise ResidentCountQualityError(
                "count quality prompt must contain one canonical hidden task"
            )
        object.__setattr__(
            self,
            "task_digests",
            tuple(_digest(row, "hidden task digest") for row in tasks),
        )
        outputs = tuple(self.output_ids)
        if (
            type(self.output_ids) is not tuple
            or len(outputs) > MAX_OUTPUT_TOKENS_PER_PROMPT
            or any(type(token) is not int or token < 0 for token in outputs)
        ):
            raise ResidentCountQualityError("count quality output IDs are malformed")
        object.__setattr__(self, "output_ids", outputs)
        if type(self.judge_receipt) is not HiddenJudgeReceipt:
            raise ResidentCountQualityError("count quality judge receipt is not exact")
        expected_output = hidden_judge_output_digest(self.prompt_digest, outputs)
        if (
            self.judge_receipt.prompt_digest != self.prompt_digest
            or self.judge_receipt.task_digests != self.task_digests
            or self.judge_receipt.output_ids_digest != expected_output
            or len(self.judge_receipt.passed) != 1
        ):
            raise ResidentCountQualityError(
                "count quality judge receipt differs from retained prompt evidence"
            )

    @property
    def correct(self) -> bool:
        return self.judge_receipt.passed == (True,)

    def to_dict(self) -> dict[str, object]:
        return {
            "judge_receipt": _receipt_to_dict(self.judge_receipt),
            "ordinal": self.ordinal,
            "output_ids": list(self.output_ids),
            "prompt_digest": self.prompt_digest,
            "task_digests": list(self.task_digests),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ResidentCountPromptObservation":
        row = _strict(
            value,
            frozenset(
                {
                    "judge_receipt",
                    "ordinal",
                    "output_ids",
                    "prompt_digest",
                    "task_digests",
                }
            ),
            "resident count prompt observation",
        )
        if type(row["task_digests"]) is not list or type(row["output_ids"]) is not list:
            raise ResidentCountQualityError("count quality prompt arrays are malformed")
        return cls(
            row["ordinal"],  # type: ignore[arg-type]
            row["prompt_digest"],  # type: ignore[arg-type]
            tuple(row["task_digests"]),  # type: ignore[arg-type]
            tuple(row["output_ids"]),  # type: ignore[arg-type]
            _receipt_from_dict(row["judge_receipt"]),
        )


@dataclass(frozen=True)
class ResidentCountQualityObservation:
    """Closed retained stock or candidate outputs; count is always derived."""

    role: str
    envelope: ResidentCountQualityEnvelope
    execution_evidence_digest: str
    prompts: tuple[ResidentCountPromptObservation, ...]

    def __post_init__(self) -> None:
        if self.role not in RESIDENT_COUNT_ROLES:
            raise ResidentCountQualityError("resident count quality role is invalid")
        if type(self.envelope) is not ResidentCountQualityEnvelope:
            raise ResidentCountQualityError("resident count quality envelope is not exact")
        object.__setattr__(
            self,
            "execution_evidence_digest",
            _digest(self.execution_evidence_digest, "quality execution evidence"),
        )
        rows = tuple(self.prompts)
        if (
            type(self.prompts) is not tuple
            or len(rows) != self.envelope.expected_prompt_count
            or any(type(row) is not ResidentCountPromptObservation for row in rows)
            or tuple(row.ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.prompt_digest for row in rows}) != len(rows)
        ):
            raise ResidentCountQualityError(
                "resident count prompts are not exact, complete, ordered, and unique"
            )
        if any(
            row.judge_receipt.binding_digest != self.envelope.judge_binding.digest
            for row in rows
        ):
            raise ResidentCountQualityError(
                "resident count prompt receipt differs from envelope judge binding"
            )
        object.__setattr__(self, "prompts", rows)

    @property
    def correct(self) -> int:
        return sum(row.correct for row in self.prompts)

    @property
    def total(self) -> int:
        return len(self.prompts)

    def to_dict(self) -> dict[str, object]:
        return {
            "envelope": self.envelope.to_dict(),
            "execution_evidence_digest": self.execution_evidence_digest,
            "prompts": [row.to_dict() for row in self.prompts],
            "role": self.role,
            "schema": RESIDENT_COUNT_OBSERVATION_SCHEMA,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def digest(self) -> str:
        return canonical_digest(RESIDENT_COUNT_OBSERVATION_SCHEMA, self.to_dict())

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "ResidentCountQualityObservation":
        row = _strict(
            _canonical_object(payload),
            frozenset(
                {
                    "envelope",
                    "execution_evidence_digest",
                    "prompts",
                    "role",
                    "schema",
                }
            ),
            "resident count quality observation",
        )
        if row["schema"] != RESIDENT_COUNT_OBSERVATION_SCHEMA or type(row["prompts"]) is not list:
            raise ResidentCountQualityError("resident count observation schema is unsupported")
        observation = cls(
            row["role"],  # type: ignore[arg-type]
            ResidentCountQualityEnvelope.from_dict(row["envelope"]),
            row["execution_evidence_digest"],  # type: ignore[arg-type]
            tuple(ResidentCountPromptObservation.from_dict(value) for value in row["prompts"]),
        )
        if observation.canonical_bytes != payload:
            raise ResidentCountQualityError(
                "resident count observation changed during typed reopening"
            )
        return observation


@dataclass(frozen=True)
class ResidentCountQualityResult:
    """Rejudged immutable observations and their exact-count launch verdict."""

    stock_observation_digest: str
    candidate_observation_digest: str
    evidence: CountQualityEvidence
    policy: CountQualityPolicy
    verdict: CountQualityVerdict

    def __post_init__(self) -> None:
        for field in ("stock_observation_digest", "candidate_observation_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if (
            type(self.evidence) is not CountQualityEvidence
            or type(self.policy) is not CountQualityPolicy
            or type(self.verdict) is not CountQualityVerdict
        ):
            raise ResidentCountQualityError("resident count result fields are not exact")
        if (
            self.evidence.stock_observation_digest != self.stock_observation_digest
            or self.evidence.candidate_observation_digest
            != self.candidate_observation_digest
            or score_count_quality(self.evidence, self.policy) != self.verdict
        ):
            raise ResidentCountQualityError("resident count result does not recompute")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_observation_digest": self.candidate_observation_digest,
            "evidence": self.evidence.to_dict(),
            "policy": self.policy.to_dict(),
            "schema": RESIDENT_COUNT_RESULT_SCHEMA,
            "stock_observation_digest": self.stock_observation_digest,
            "verdict": self.verdict.to_dict(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(RESIDENT_COUNT_RESULT_SCHEMA, self.to_dict())


def rejudge_resident_count_observation(
    observation: ResidentCountQualityObservation,
    judge: NumericAnswerHiddenJudge,
) -> int:
    """Recompute every retained receipt; do not trust a stored aggregate count."""

    if type(observation) is not ResidentCountQualityObservation:
        raise ResidentCountQualityError("resident count observation is not exact")
    if type(judge) is not NumericAnswerHiddenJudge:
        raise ResidentCountQualityError("resident numeric judge is not exact")
    if (
        judge.binding != observation.envelope.judge_binding
        or judge.tokenizer_digest != observation.envelope.reference.tokenizer_digest
    ):
        raise ResidentCountQualityInfrastructureError(
            "resident numeric judge differs from the frozen quality envelope"
        )
    correct = 0
    for row in observation.prompts:
        try:
            reopened = judge(
                prompt_digest=row.prompt_digest,
                output_ids=row.output_ids,
                task_digests=row.task_digests,
            )
        except NumericAnswerJudgeError as exc:
            raise ResidentCountQualityInfrastructureError(
                f"resident numeric observation could not be rejudged: {exc}"
            ) from None
        if reopened != row.judge_receipt:
            raise ResidentCountQualityInfrastructureError(
                "resident numeric receipt differs during reopening"
            )
        correct += int(reopened.passed == (True,))
    return correct


def compare_resident_count_quality(
    stock: ResidentCountQualityObservation,
    candidate: ResidentCountQualityObservation,
    *,
    judge: NumericAnswerHiddenJudge,
    policy: CountQualityPolicy,
) -> ResidentCountQualityResult:
    """Rejudge stock and candidate, then apply exact-count launch arithmetic."""

    if type(stock) is not ResidentCountQualityObservation or stock.role != "stock":
        raise ResidentCountQualityError("stock quality observation is not exact")
    if type(candidate) is not ResidentCountQualityObservation or candidate.role != "candidate":
        raise ResidentCountQualityError("candidate quality observation is not exact")
    if stock.envelope != candidate.envelope:
        raise ResidentCountQualityInfrastructureError(
            "stock and candidate quality envelopes differ"
        )
    if tuple((row.prompt_digest, row.task_digests) for row in stock.prompts) != tuple(
        (row.prompt_digest, row.task_digests) for row in candidate.prompts
    ):
        raise ResidentCountQualityInfrastructureError(
            "stock and candidate prompt occurrence authority differs"
        )
    if type(policy) is not CountQualityPolicy:
        raise ResidentCountQualityError("count quality policy is not exact")
    stock_correct = rejudge_resident_count_observation(stock, judge)
    candidate_correct = rejudge_resident_count_observation(candidate, judge)
    evidence = CountQualityEvidence(
        stock.digest,
        candidate.digest,
        stock_correct,
        candidate_correct,
        stock.total,
    )
    verdict = score_count_quality(evidence, policy)
    return ResidentCountQualityResult(
        stock.digest,
        candidate.digest,
        evidence,
        policy,
        verdict,
    )


__all__ = [
    "MAX_OUTPUT_TOKENS_PER_PROMPT",
    "MAX_QUALITY_PROMPTS",
    "RESIDENT_COUNT_ENVELOPE_SCHEMA",
    "RESIDENT_COUNT_OBSERVATION_SCHEMA",
    "RESIDENT_COUNT_RESULT_SCHEMA",
    "ResidentCountPromptObservation",
    "ResidentCountQualityEnvelope",
    "ResidentCountQualityError",
    "ResidentCountQualityInfrastructureError",
    "ResidentCountQualityObservation",
    "ResidentCountQualityResult",
    "compare_resident_count_quality",
    "rejudge_resident_count_observation",
]

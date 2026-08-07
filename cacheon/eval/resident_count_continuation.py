"""Durable raw-evidence continuation for resident fixed-stock count quality.

The resident evaluator is deliberately absent from this module.  Publication
starts from an already-completed ``ResidentCountQualityExecutionResult`` and
writes two independent content-addressed products: the raw A/B execution
evidence and the derived candidate observation.  The continuation checkpoint
is written last and binds both artifacts plus the commissioned plan, fixed
stock authority, and physical resident-pair authority.

Reopening never trusts successful decoding as a verdict.  It authenticates
both artifact byte identities, reconstructs the exact raw type, independently
regrades that raw evidence with the caller's sealed plan and hidden judge, and
requires exact equality with the separately reopened observation.  Every
failure after a checkpoint exists is a HOLD, never candidate failure or an
authorization to rerun the model.
"""

from __future__ import annotations

import json
from pathlib import Path

from cacheon.eval.continuation_codec import (
    ContinuationCodec,
    ContinuationCodecError,
)
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    publish_evidence,
    reopen_evidence,
)
from cacheon.eval.numeric_answer_judge import NumericAnswerHiddenJudge
from cacheon.eval.qualification_continuation import (
    QualificationContinuation,
    QualificationContinuationError,
    ResidentCountQualityCheckpoint,
)
from cacheon.eval.resident_count_execution_evidence import (
    ResidentCountExecutionEvidenceError,
    ResidentCountQualityExecutionEvidence,
)
from cacheon.eval.resident_count_quality import (
    ResidentCountQualityError,
    ResidentCountQualityInfrastructureError,
    ResidentCountQualityObservation,
    publish_resident_count_observation,
    reopen_resident_count_observation,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionError,
    ResidentCountQualityExecutionPlan,
    ResidentCountQualityExecutionResult,
    regrade_candidate_count_quality_execution,
)
from cacheon.eval.resident_pair_binding import ResidentPairRuntimeBinding
from cacheon.stack_identity import canonical_json_bytes, require_sha256_hex


RESIDENT_COUNT_EXECUTION_ARTIFACT_DOMAIN = (
    "cacheon.resident-count-quality-execution"
)
RESIDENT_COUNT_EXECUTION_ARTIFACT_SCHEMA = (
    "cacheon.eval.resident-count-quality-execution-artifact.v1"
)
MAX_RESIDENT_COUNT_EXECUTION_ARTIFACT_BYTES = 64 << 20

_RAW_CODEC = ContinuationCodec((ResidentCountQualityExecutionEvidence,))


class ResidentCountQualityContinuationHold(QualificationContinuationError):
    """Durable count continuation is absent, foreign, partial, or corrupted."""

    decision = "HOLD"


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (TypeError, ValueError) as exc:
        raise ResidentCountQualityContinuationHold(str(exc)) from None


def _require_context(
    continuation: QualificationContinuation,
    *,
    plan: ResidentCountQualityExecutionPlan,
    fixed_stock_authority_digest: str,
    pair_binding: ResidentPairRuntimeBinding,
    judge: NumericAnswerHiddenJudge,
) -> str:
    if type(continuation) is not QualificationContinuation:
        raise ResidentCountQualityContinuationHold(
            "resident count continuation scope is not exact"
        )
    if type(plan) is not ResidentCountQualityExecutionPlan:
        raise ResidentCountQualityContinuationHold(
            "resident count continuation plan is not exact"
        )
    if type(pair_binding) is not ResidentPairRuntimeBinding:
        raise ResidentCountQualityContinuationHold(
            "resident count continuation pair binding is not exact"
        )
    if type(judge) is not NumericAnswerHiddenJudge:
        raise ResidentCountQualityContinuationHold(
            "resident count continuation hidden judge is not exact"
        )
    if plan.pair_binding != pair_binding:
        raise ResidentCountQualityContinuationHold(
            "resident count continuation pair differs from commissioned plan"
        )
    return _digest(
        fixed_stock_authority_digest,
        "resident count fixed-stock authority digest",
    )


def _raw_reference(reference: object) -> EvidenceArtifactRef:
    if type(reference) is not EvidenceArtifactRef or (
        reference.domain,
        reference.media_type,
        reference.schema,
    ) != (
        RESIDENT_COUNT_EXECUTION_ARTIFACT_DOMAIN,
        "application/json",
        RESIDENT_COUNT_EXECUTION_ARTIFACT_SCHEMA,
    ):
        raise ResidentCountQualityContinuationHold(
            "resident count raw execution reference is not exact"
        )
    if reference.size > MAX_RESIDENT_COUNT_EXECUTION_ARTIFACT_BYTES:
        raise ResidentCountQualityContinuationHold(
            "resident count raw execution reference exceeds its bound"
        )
    return reference


def _canonical_object(payload: bytes) -> dict[str, object]:
    def reject_number(value: str) -> object:
        raise ResidentCountQualityContinuationHold(
            f"resident count raw execution contains unsupported number {value!r}"
        )

    def unique(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ResidentCountQualityContinuationHold(
                    f"resident count raw execution repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=reject_number,
            parse_constant=reject_number,
            object_pairs_hook=unique,
        )
    except ResidentCountQualityContinuationHold:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ResidentCountQualityContinuationHold(
            f"resident count raw execution is malformed: {exc}"
        ) from None
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ResidentCountQualityContinuationHold(
            "resident count raw execution is not exact canonical JSON"
        )
    return value


def _publish_raw_execution(
    evidence_root: str | Path,
    evidence: ResidentCountQualityExecutionEvidence,
    *,
    deadline: float | None,
) -> EvidenceArtifactRef:
    if type(evidence) is not ResidentCountQualityExecutionEvidence:
        raise ResidentCountQualityContinuationHold(
            "resident count raw execution evidence is not exact"
        )
    try:
        payload = canonical_json_bytes(_RAW_CODEC.encode(evidence))
        return publish_evidence(
            evidence_root,
            payload,
            domain=RESIDENT_COUNT_EXECUTION_ARTIFACT_DOMAIN,
            media_type="application/json",
            schema=RESIDENT_COUNT_EXECUTION_ARTIFACT_SCHEMA,
            max_bytes=MAX_RESIDENT_COUNT_EXECUTION_ARTIFACT_BYTES,
            deadline=deadline,
        )
    except (ContinuationCodecError, EvidenceStoreError, TypeError, ValueError) as exc:
        raise ResidentCountQualityContinuationHold(
            f"resident count raw execution could not be published: {exc}"
        ) from None


def _reopen_raw_execution(
    evidence_root: str | Path,
    reference: EvidenceArtifactRef,
) -> ResidentCountQualityExecutionEvidence:
    exact = _raw_reference(reference)
    try:
        payload = reopen_evidence(
            evidence_root,
            exact,
            max_bytes=MAX_RESIDENT_COUNT_EXECUTION_ARTIFACT_BYTES,
        )
        decoded = _RAW_CODEC.decode(_canonical_object(payload))
    except ResidentCountQualityContinuationHold:
        raise
    except (ContinuationCodecError, EvidenceStoreError, TypeError, ValueError) as exc:
        raise ResidentCountQualityContinuationHold(
            f"resident count raw execution could not be reopened: {exc}"
        ) from None
    if type(decoded) is not ResidentCountQualityExecutionEvidence:
        raise ResidentCountQualityContinuationHold(
            "resident count raw execution reopened another evidence type"
        )
    return decoded


def _require_checkpoint_context(
    checkpoint: ResidentCountQualityCheckpoint,
    *,
    plan: ResidentCountQualityExecutionPlan,
    fixed_stock_authority_digest: str,
    pair_binding: ResidentPairRuntimeBinding,
) -> None:
    if type(checkpoint) is not ResidentCountQualityCheckpoint:
        raise ResidentCountQualityContinuationHold(
            "resident count continuation checkpoint is not exact"
        )
    if (
        checkpoint.execution_plan_digest != plan.digest
        or checkpoint.fixed_stock_authority_digest
        != fixed_stock_authority_digest
        or checkpoint.pair_binding_digest != pair_binding.digest
    ):
        raise ResidentCountQualityContinuationHold(
            "resident count checkpoint differs from caller-supplied authority"
        )


def _reopen_checkpoint(
    evidence_root: str | Path,
    checkpoint: ResidentCountQualityCheckpoint,
    *,
    plan: ResidentCountQualityExecutionPlan,
    fixed_stock_authority_digest: str,
    pair_binding: ResidentPairRuntimeBinding,
    judge: NumericAnswerHiddenJudge,
) -> ResidentCountQualityExecutionResult:
    _require_checkpoint_context(
        checkpoint,
        plan=plan,
        fixed_stock_authority_digest=fixed_stock_authority_digest,
        pair_binding=pair_binding,
    )
    raw = _reopen_raw_execution(evidence_root, checkpoint.raw_execution_evidence)
    if raw.digest != checkpoint.raw_execution_evidence_semantic_digest:
        raise ResidentCountQualityContinuationHold(
            "resident count raw execution semantic digest differs from checkpoint"
        )
    derived = regrade_candidate_count_quality_execution(
        raw,
        plan=plan,
        judge=judge,
    )
    observation = reopen_resident_count_observation(
        evidence_root,
        checkpoint.candidate_observation,
    )
    if (
        observation.digest != checkpoint.candidate_observation_semantic_digest
        or observation.execution_evidence_digest != raw.digest
        or observation != derived
        or observation.digest != derived.digest
    ):
        raise ResidentCountQualityContinuationHold(
            "resident count observation differs from independent raw regrade"
        )
    return ResidentCountQualityExecutionResult(raw, observation)


def publish_resident_count_quality_continuation(
    evidence_root: str | Path,
    continuation: QualificationContinuation,
    execution: ResidentCountQualityExecutionResult,
    *,
    plan: ResidentCountQualityExecutionPlan,
    fixed_stock_authority_digest: str,
    pair_binding: ResidentPairRuntimeBinding,
    judge: NumericAnswerHiddenJudge,
    deadline: float | None = None,
) -> ResidentCountQualityCheckpoint:
    """Publish both products and record the immutable checkpoint last."""

    try:
        stock_digest = _require_context(
            continuation,
            plan=plan,
            fixed_stock_authority_digest=fixed_stock_authority_digest,
            pair_binding=pair_binding,
            judge=judge,
        )
        if type(execution) is not ResidentCountQualityExecutionResult:
            raise ResidentCountQualityContinuationHold(
                "resident count continuation execution result is not exact"
            )

        existing = continuation.load_resident_count_quality()
        if existing is not None:
            reopened = _reopen_checkpoint(
                evidence_root,
                existing,
                plan=plan,
                fixed_stock_authority_digest=stock_digest,
                pair_binding=pair_binding,
                judge=judge,
            )
            if reopened != execution:
                raise ResidentCountQualityContinuationHold(
                    "resident count checkpoint already binds other evidence"
                )
            return existing

        raw_reference = _publish_raw_execution(
            evidence_root,
            execution.evidence,
            deadline=deadline,
        )
        reopened_raw = _reopen_raw_execution(evidence_root, raw_reference)
        derived = regrade_candidate_count_quality_execution(
            reopened_raw,
            plan=plan,
            judge=judge,
        )
        if derived != execution.observation or derived.digest != execution.observation.digest:
            raise ResidentCountQualityContinuationHold(
                "resident count supplied observation differs from independent raw regrade"
            )
        observation_reference = publish_resident_count_observation(
            evidence_root,
            derived,
            deadline=deadline,
        )
        reopened_observation = reopen_resident_count_observation(
            evidence_root,
            observation_reference,
        )
        if reopened_observation != derived or reopened_observation.digest != derived.digest:
            raise ResidentCountQualityContinuationHold(
                "resident count published observation changed during reopening"
            )
        checkpoint = ResidentCountQualityCheckpoint(
            raw_execution_evidence=raw_reference,
            raw_execution_evidence_semantic_digest=reopened_raw.digest,
            candidate_observation=observation_reference,
            candidate_observation_semantic_digest=reopened_observation.digest,
            execution_plan_digest=plan.digest,
            fixed_stock_authority_digest=stock_digest,
            pair_binding_digest=pair_binding.digest,
        )
        continuation.record_resident_count_quality(checkpoint)
        return checkpoint
    except ResidentCountQualityContinuationHold:
        raise
    except (
        QualificationContinuationError,
        ResidentCountExecutionEvidenceError,
        ResidentCountQualityError,
        ResidentCountQualityInfrastructureError,
        ResidentCountQualityExecutionError,
        TypeError,
        ValueError,
    ) as exc:
        raise ResidentCountQualityContinuationHold(
            f"resident count continuation publication is on HOLD: {exc}"
        ) from None


def reopen_resident_count_quality_continuation(
    evidence_root: str | Path,
    continuation: QualificationContinuation,
    *,
    plan: ResidentCountQualityExecutionPlan,
    fixed_stock_authority_digest: str,
    pair_binding: ResidentPairRuntimeBinding,
    judge: NumericAnswerHiddenJudge,
) -> ResidentCountQualityExecutionResult | None:
    """Reopen and regrade a completed checkpoint without any model callable."""

    try:
        stock_digest = _require_context(
            continuation,
            plan=plan,
            fixed_stock_authority_digest=fixed_stock_authority_digest,
            pair_binding=pair_binding,
            judge=judge,
        )
        checkpoint = continuation.load_resident_count_quality()
        if checkpoint is None:
            return None
        return _reopen_checkpoint(
            evidence_root,
            checkpoint,
            plan=plan,
            fixed_stock_authority_digest=stock_digest,
            pair_binding=pair_binding,
            judge=judge,
        )
    except ResidentCountQualityContinuationHold:
        raise
    except (
        QualificationContinuationError,
        ResidentCountExecutionEvidenceError,
        ResidentCountQualityError,
        ResidentCountQualityInfrastructureError,
        ResidentCountQualityExecutionError,
        TypeError,
        ValueError,
    ) as exc:
        raise ResidentCountQualityContinuationHold(
            f"resident count continuation reopening is on HOLD: {exc}"
        ) from None


__all__ = [
    "MAX_RESIDENT_COUNT_EXECUTION_ARTIFACT_BYTES",
    "RESIDENT_COUNT_EXECUTION_ARTIFACT_DOMAIN",
    "RESIDENT_COUNT_EXECUTION_ARTIFACT_SCHEMA",
    "ResidentCountQualityContinuationHold",
    "publish_resident_count_quality_continuation",
    "reopen_resident_count_quality_continuation",
]

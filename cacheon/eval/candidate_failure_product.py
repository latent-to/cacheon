"""Closed CAS product for one candidate-owned qualification failure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
from cacheon.eval.oci_outer_session import OuterSessionCandidateError
from cacheon.stack_identity import canonical_digest, canonical_json_bytes, require_sha256_hex

if TYPE_CHECKING:
    from cacheon.eval.qualification_intake import (
        QualificationAuthorityManifest,
        QualificationIntakeBatch,
    )
    from cacheon.eval.qualification_runner import CausalQualificationInput

CANDIDATE_FAILURE_DOMAIN = "qualification.candidate-failure"
CANDIDATE_FAILURE_SCHEMA = "cacheon.qualification.candidate-failure.v1"
_FIELDS = {
    "authority_manifest_digest", "culprit", "failure",
    "failure_kind", "schema", "source_digest",
}
_CULPRIT_FIELDS = {
    "arm_digest", "launch_digest", "reservation_digest",
    "selected_delta_digest", "target_id",
}


class CandidateFailureProductError(ValueError):
    pass


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CandidateFailureProductError(str(exc)) from None


def validate_candidate_failure(value: object) -> dict:
    if type(value) is not dict or set(value) != _FIELDS:
        raise CandidateFailureProductError("candidate failure fields are not closed")
    culprit = value["culprit"]
    if (
        value["schema"] != CANDIDATE_FAILURE_SCHEMA
        or type(culprit) is not dict or set(culprit) != _CULPRIT_FIELDS
        or value["failure_kind"] not in {"candidate_exception", "candidate_never_executed"}
        or not isinstance(value["failure"], str) or not value["failure"]
        or len(value["failure"]) > 16_384 or "\x00" in value["failure"]
        or not isinstance(culprit["target_id"], str) or not culprit["target_id"]
        or len(culprit["target_id"]) > 256
    ):
        raise CandidateFailureProductError("candidate failure product is malformed")
    for name in ("authority_manifest_digest", "source_digest"):
        _digest(value[name], name)
    for name in ("reservation_digest", "selected_delta_digest", "arm_digest", "launch_digest"):
        _digest(culprit[name], name)
    return value


def candidate_failure_digest(value: object) -> str:
    return canonical_digest(CANDIDATE_FAILURE_SCHEMA, validate_candidate_failure(value))


def publish_candidate_failure(
    root: Path, *, authority_manifest_digest: str, source_digest: str,
    culprit_reservation_digest: str, selected_delta_digest: str, target_id: str,
    arm_digest: str, launch_digest: str, failure_kind: str, failure: str,
) -> tuple[EvidenceArtifactRef, dict]:
    value = validate_candidate_failure({
        "authority_manifest_digest": authority_manifest_digest,
        "culprit": {"arm_digest": arm_digest,
            "launch_digest": launch_digest, "reservation_digest": culprit_reservation_digest,
            "selected_delta_digest": selected_delta_digest, "target_id": target_id},
        "failure": failure[:16_384], "failure_kind": failure_kind,
        "schema": CANDIDATE_FAILURE_SCHEMA, "source_digest": source_digest,
    })
    reference = publish_evidence(root, canonical_json_bytes(value),
        domain=CANDIDATE_FAILURE_DOMAIN, media_type="application/json",
        schema=CANDIDATE_FAILURE_SCHEMA)
    if reopen_candidate_failure(root, reference) != value:
        raise CandidateFailureProductError("candidate failure changed after publication")
    return reference, value


def reopen_candidate_failure(root: Path, reference: EvidenceArtifactRef) -> dict:
    if (type(reference) is not EvidenceArtifactRef
            or (reference.domain, reference.schema, reference.media_type) !=
            (CANDIDATE_FAILURE_DOMAIN, CANDIDATE_FAILURE_SCHEMA, "application/json")):
        raise CandidateFailureProductError("candidate failure reference is unsupported")
    try:
        return validate_candidate_failure(json.loads(reopen_evidence(root, reference)))
    except (TypeError, ValueError) as exc:
        raise CandidateFailureProductError(f"candidate failure cannot reopen: {exc}") from None


def candidate_failure_batch(
    manifest: QualificationAuthorityManifest,
    value: CausalQualificationInput,
    worker_error: OuterSessionCandidateError,
) -> QualificationIntakeBatch:
    """Publish the registered singleton's typed terminal failure."""

    from cacheon.eval.qualification import QualificationDecision
    from cacheon.eval.qualification_intake import (
        QualificationIntakeBatch,
        QualificationIntakeError,
        QualificationIntakeOutcome,
    )

    if (
        not isinstance(worker_error, OuterSessionCandidateError)
        or len(manifest.reservations) != 1
        or len(value.prepared.candidates) != 1
    ):
        raise QualificationIntakeError("candidate failure lacks typed ownership")
    culprit = manifest.reservations[0]
    prepared = value.prepared.candidates[0]
    if prepared.arm.selected_delta_digest != culprit.selected_delta_digest:
        raise QualificationIntakeError("candidate failure differs from intake authority")
    failure_kind = (
        "candidate_never_executed"
        if worker_error.candidate_failure_type == "CandidateNeverExecutedError"
        else "candidate_exception"
    )
    reference, product = publish_candidate_failure(
        value.evidence_root,
        authority_manifest_digest=manifest.digest,
        source_digest=manifest.source_digest,
        culprit_reservation_digest=culprit.reservation_digest,
        selected_delta_digest=culprit.selected_delta_digest,
        target_id=culprit.target_id,
        arm_digest=prepared.arm.digest,
        launch_digest=prepared.launch.digest,
        failure_kind=failure_kind,
        failure=worker_error.candidate_failure,
    )
    report_digest = candidate_failure_digest(product)
    outcome = QualificationIntakeOutcome(
        culprit.reservation_digest,
        culprit.selected_delta_digest,
        manifest.digest,
        QualificationDecision.FAIL,
        failure_kind,
        False,
        attempt_artifact_sha256=reference.sha256,
        report_digest=report_digest,
    )
    return QualificationIntakeBatch(manifest.digest, (outcome,), reference)


__all__ = ["CANDIDATE_FAILURE_SCHEMA", "CandidateFailureProductError",
    "candidate_failure_batch", "candidate_failure_digest", "publish_candidate_failure",
    "reopen_candidate_failure", "validate_candidate_failure"]

"""Sealed remote qualification evidence products and their CPU-side import.

The pod captures exact evidence bytes into a size-bounded, digest-inventoried
product; the CPU rehashes every artifact on import before anything durable is
committed.  ``remote_evaluation_dispatcher`` composes and re-exports these
names, so import paths are unchanged.
"""

from __future__ import annotations

import base64
import binascii

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from cacheon.arena_service import (
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    publish_evidence,
    reopen_evidence,
)
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationRetryPlan,
)
from cacheon.eval.qualification_runner import ATTEMPT_SCHEMA_V4, STAGE_EXIT_SCHEMA_V3
from cacheon.eval.resident_pair_quality_lifecycle import (
    ResidentPairQualityLifecycleError,
    reopen_resident_pair_qualification_product,
)
from cacheon.settlement import SettlementQualification
from cacheon.stack_identity import canonical_digest, require_sha256_hex, sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest


_SCHEMA_VERSION = 2
_MAX_REMOTE_EVIDENCE_ARTIFACTS = 256
_MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES = 16 << 20
_MAX_REMOTE_QUALIFICATION_EVIDENCE_BYTES = 32 << 20


class RemoteEvaluationDispatcherError(RuntimeError):
    """Remote work cannot be authenticated, reopened, released, or committed."""


def _digest(value: object, field_name: str) -> str:
    try:
        return require_sha256_hex(value, field=field_name)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError(str(exc)) from None


def _evidence_key(reference: EvidenceArtifactRef) -> tuple[str, str, str, str]:
    return (
        reference.domain,
        reference.sha256,
        reference.media_type,
        reference.schema,
    )


@dataclass(frozen=True)
class RemoteEvidenceArtifact:
    """One bounded CAS reference and its exact authenticated transport bytes."""

    reference: EvidenceArtifactRef
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not EvidenceArtifactRef:
            raise RemoteEvaluationDispatcherError(
                "remote evidence reference is not exactly typed"
            )
        if type(self.payload) is not bytes:
            raise RemoteEvaluationDispatcherError("remote evidence payload is not exact bytes")
        if (
            len(self.payload) > _MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES
            or len(self.payload) != self.reference.size
            or sha256_hex(self.payload) != self.reference.sha256
        ):
            raise RemoteEvaluationDispatcherError(
                "remote evidence payload differs from its bounded reference"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "payload_base64": base64.b64encode(self.payload).decode("ascii"),
            "reference": self.reference.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RemoteEvidenceArtifact":
        if type(value) is not dict or set(value) != {"payload_base64", "reference"}:
            raise RemoteEvaluationDispatcherError(
                "remote evidence artifact fields are not closed"
            )
        encoded = value["payload_base64"]
        if not isinstance(encoded, str) or len(encoded) > (
            ((_MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES + 2) // 3) * 4
        ):
            raise RemoteEvaluationDispatcherError(
                "remote evidence payload encoding exceeds its bound"
            )
        try:
            payload = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise RemoteEvaluationDispatcherError(
                "remote evidence payload is not canonical base64"
            ) from exc
        if base64.b64encode(payload).decode("ascii") != encoded:
            raise RemoteEvaluationDispatcherError(
                "remote evidence payload is not canonical base64"
            )
        try:
            reference = EvidenceArtifactRef.from_dict(value["reference"])
        except (TypeError, ValueError) as exc:
            raise RemoteEvaluationDispatcherError(
                "remote evidence reference is invalid"
            ) from exc
        return cls(reference, payload)


@dataclass(frozen=True)
class RemoteQualificationProduct:
    """Closed worker product sufficient for CPU-owned import and durable commit."""

    service_digest: str
    worker_readiness_digest: str
    ready_receipt_digest: str
    ready_epoch: int
    screen_lane: str
    authority_manifest: QualificationAuthorityManifest
    incumbent_stack: EvaluationStackManifest
    incumbent_tree_digest: str
    batch: QualificationIntakeBatch
    evidence_inventory: tuple[EvidenceArtifactRef, ...]
    evidence: tuple[RemoteEvidenceArtifact, ...]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "service_digest",
            "worker_readiness_digest",
            "ready_receipt_digest",
            "incumbent_tree_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        inventory = tuple(self.evidence_inventory)
        artifacts = tuple(self.evidence)
        if (
            type(self.ready_epoch) is not int
            or self.ready_epoch < 0
            or self.screen_lane not in {"primary", "reproduction"}
            or type(self.authority_manifest) is not QualificationAuthorityManifest
            or type(self.incumbent_stack) is not EvaluationStackManifest
            or type(self.batch) is not QualificationIntakeBatch
            or type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
            or len(inventory) > _MAX_REMOTE_EVIDENCE_ARTIFACTS
            or any(type(row) is not EvidenceArtifactRef for row in inventory)
            or len(artifacts) != len(inventory)
            or any(type(row) is not RemoteEvidenceArtifact for row in artifacts)
        ):
            raise RemoteEvaluationDispatcherError(
                "remote qualification product authority is malformed"
            )
        ordered_inventory = tuple(sorted(inventory, key=_evidence_key))
        ordered_artifacts = tuple(sorted(artifacts, key=lambda row: _evidence_key(row.reference)))
        if (
            inventory != ordered_inventory
            or artifacts != ordered_artifacts
            or tuple(row.reference for row in artifacts) != inventory
            or len({row.sha256 for row in inventory}) != len(inventory)
            or sum(row.size for row in inventory)
            > _MAX_REMOTE_QUALIFICATION_EVIDENCE_BYTES
        ):
            raise RemoteEvaluationDispatcherError(
                "remote qualification evidence inventory is duplicate, unordered, or oversized"
            )
        manifest = self.authority_manifest
        expected_reservations = tuple(row.reservation_digest for row in manifest.reservations)
        expected_deltas = tuple(row.selected_delta_digest for row in manifest.reservations)
        if (
            self.batch.authority_manifest_digest != manifest.digest
            or tuple(row.reservation_digest for row in self.batch.outcomes)
            != expected_reservations
            or tuple(row.selected_delta_digest for row in self.batch.outcomes)
            != expected_deltas
            or (self.screen_lane == "reproduction" and len(expected_reservations) != 1)
            or self.incumbent_stack.arena_digest != self.service_digest
            or (
                self.batch.attempt_ref is not None
                and self.batch.attempt_ref not in inventory
            )
        ):
            raise RemoteEvaluationDispatcherError(
                "remote qualification product differs from its cohort authority"
            )
        object.__setattr__(self, "evidence_inventory", inventory)
        object.__setattr__(self, "evidence", artifacts)

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_manifest": self.authority_manifest.to_dict(),
            "batch": qualification_batch_to_dict(self.batch),
            "evidence": [row.to_dict() for row in self.evidence],
            "evidence_inventory": [row.to_dict() for row in self.evidence_inventory],
            "incumbent_stack": self.incumbent_stack.to_dict(),
            "incumbent_tree_digest": self.incumbent_tree_digest,
            "ready_epoch": self.ready_epoch,
            "ready_receipt_digest": self.ready_receipt_digest,
            "schema_version": self.schema_version,
            "screen_lane": self.screen_lane,
            "service_digest": self.service_digest,
            "worker_readiness_digest": self.worker_readiness_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.remote-qualification-product.v1", self.to_dict()
        )


def remote_qualification_product_to_dict(
    product: RemoteQualificationProduct,
) -> dict[str, object]:
    if type(product) is not RemoteQualificationProduct:
        raise RemoteEvaluationDispatcherError(
            "remote qualification product is not exactly typed"
        )
    return product.to_dict()


def remote_qualification_product_from_dict(value: object) -> RemoteQualificationProduct:
    """Strictly reopen an untrusted remote qualification product."""

    fields = {
        "authority_manifest",
        "batch",
        "evidence",
        "evidence_inventory",
        "incumbent_stack",
        "incumbent_tree_digest",
        "ready_epoch",
        "ready_receipt_digest",
        "schema_version",
        "screen_lane",
        "service_digest",
        "worker_readiness_digest",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or type(value["evidence"]) is not list
        or type(value["evidence_inventory"]) is not list
    ):
        raise RemoteEvaluationDispatcherError(
            "remote qualification product fields are not closed"
        )
    try:
        return RemoteQualificationProduct(
            service_digest=value["service_digest"],  # type: ignore[arg-type]
            worker_readiness_digest=value["worker_readiness_digest"],  # type: ignore[arg-type]
            ready_receipt_digest=value["ready_receipt_digest"],  # type: ignore[arg-type]
            ready_epoch=value["ready_epoch"],  # type: ignore[arg-type]
            screen_lane=value["screen_lane"],  # type: ignore[arg-type]
            authority_manifest=QualificationAuthorityManifest.from_dict(
                value["authority_manifest"]
            ),
            incumbent_stack=EvaluationStackManifest.from_dict(value["incumbent_stack"]),
            incumbent_tree_digest=value["incumbent_tree_digest"],  # type: ignore[arg-type]
            batch=qualification_batch_from_dict(value["batch"]),
            evidence_inventory=tuple(
                EvidenceArtifactRef.from_dict(row)
                for row in value["evidence_inventory"]  # type: ignore[union-attr]
            ),
            evidence=tuple(
                RemoteEvidenceArtifact.from_dict(row)
                for row in value["evidence"]  # type: ignore[union-attr]
            ),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
    except RemoteEvaluationDispatcherError:
        raise
    except (TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError(
            "remote qualification product is invalid"
        ) from exc


def capture_remote_qualification_product(
    *,
    batch: QualificationIntakeBatch,
    authority_manifest: QualificationAuthorityManifest,
    incumbent_stack: EvaluationStackManifest,
    incumbent_tree_digest: str,
    screen_lane: str,
    service_digest: str,
    readiness: WorkerReadiness,
    evidence_root: str | Path,
    evidence_references: Iterable[EvidenceArtifactRef],
) -> RemoteQualificationProduct:
    """Capture exact pod CAS bytes without carrying the pod path onto the wire."""

    if (
        type(batch) is not QualificationIntakeBatch
        or type(authority_manifest) is not QualificationAuthorityManifest
        or type(incumbent_stack) is not EvaluationStackManifest
        or type(readiness) is not WorkerReadiness
    ):
        raise RemoteEvaluationDispatcherError(
            "remote qualification capture authority is not exactly typed"
        )
    if service_digest != readiness.service_digest:
        raise RemoteEvaluationDispatcherError(
            "remote qualification capture service differs from READY authority"
        )
    try:
        supplied = tuple(evidence_references)
    except TypeError as exc:
        raise RemoteEvaluationDispatcherError(
            "remote qualification evidence references are not iterable"
        ) from exc
    if any(type(row) is not EvidenceArtifactRef for row in supplied):
        raise RemoteEvaluationDispatcherError(
            "remote qualification evidence reference is not exactly typed"
        )
    if len(set(supplied)) != len(supplied):
        raise RemoteEvaluationDispatcherError(
            "remote qualification evidence references are duplicated"
        )
    required = list(supplied)
    if batch.attempt_ref is not None and batch.attempt_ref not in required:
        required.append(batch.attempt_ref)
    inventory = tuple(sorted(required, key=_evidence_key))
    artifacts = []
    total = 0
    try:
        for reference in inventory:
            payload = reopen_evidence(
                evidence_root,
                reference,
                max_bytes=_MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES,
            )
            total += len(payload)
            if total > _MAX_REMOTE_QUALIFICATION_EVIDENCE_BYTES:
                raise RemoteEvaluationDispatcherError(
                    "remote qualification evidence exceeds its aggregate bound"
                )
            artifacts.append(RemoteEvidenceArtifact(reference, payload))
    except EvidenceStoreError as exc:
        raise RemoteEvaluationDispatcherError(
            f"remote qualification evidence cannot be captured: {exc}"
        ) from exc
    return RemoteQualificationProduct(
        service_digest=service_digest,
        worker_readiness_digest=readiness.digest,
        ready_receipt_digest=readiness.ready_receipt_digest,
        ready_epoch=readiness.ready_epoch,
        screen_lane=screen_lane,
        authority_manifest=authority_manifest,
        incumbent_stack=incumbent_stack,
        incumbent_tree_digest=incumbent_tree_digest,
        batch=batch,
        evidence_inventory=inventory,
        evidence=tuple(artifacts),
    )


def import_remote_qualification_evidence(
    product: RemoteQualificationProduct,
    evidence_root: str | Path,
) -> tuple[EvidenceArtifactRef, ...]:
    """Publish authenticated worker bytes into the CPU-owned CAS and reopen all."""

    if type(product) is not RemoteQualificationProduct:
        raise RemoteEvaluationDispatcherError(
            "remote qualification import product is not exactly typed"
        )
    imported = []
    try:
        for artifact in product.evidence:
            reference = artifact.reference
            observed = publish_evidence(
                evidence_root,
                artifact.payload,
                domain=reference.domain,
                media_type=reference.media_type,
                schema=reference.schema,
                max_bytes=_MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES,
            )
            if observed != reference:
                raise RemoteEvaluationDispatcherError(
                    "CPU evidence import changed the worker reference"
                )
            if (
                reopen_evidence(
                    evidence_root,
                    observed,
                    max_bytes=_MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES,
                )
                != artifact.payload
            ):
                raise RemoteEvaluationDispatcherError(
                    "CPU evidence import did not reopen exact bytes"
                )
            imported.append(observed)
    except EvidenceStoreError as exc:
        raise RemoteEvaluationDispatcherError(
            f"CPU evidence import failed closed: {exc}"
        ) from exc
    result = tuple(imported)
    if result != product.evidence_inventory:
        raise RemoteEvaluationDispatcherError(
            "CPU evidence import differs from the authenticated inventory"
        )
    attempt_ref = product.batch.attempt_ref
    if attempt_ref is not None and attempt_ref.schema in {
        ATTEMPT_SCHEMA_V4,
        STAGE_EXIT_SCHEMA_V3,
    }:
        try:
            reopen_resident_pair_qualification_product(
                reopen_evidence(
                    evidence_root, attempt_ref,
                    max_bytes=_MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES,
                ),
                authority_digest=product.authority_manifest.authority_digest,
                report_digests=tuple(
                    row.report_digest for row in product.batch.outcomes
                ),
                evidence_inventory=result,
            )
        except ResidentPairQualityLifecycleError as exc:
            raise RemoteEvaluationDispatcherError(
                str(exc)
            ) from None
    return result


def _screen_receipt_from_dict(value: object) -> ArenaScreenReceipt:
    fields = {"candidate_digest", "decision", "results", "screen_attempt", "service_digest"}
    if type(value) is not dict or set(value) != fields or type(value["results"]) is not list:
        raise RemoteEvaluationDispatcherError("screen response fields are not closed")
    results = []
    for row in value["results"]:
        if type(row) is not dict or set(row) != {
            "elapsed_ms",
            "evidence_digest",
            "grade",
            "stage",
        }:
            raise RemoteEvaluationDispatcherError("screen stage response is malformed")
        try:
            results.append(
                ScreenStageResult(
                    row["stage"],
                    ScreenGrade(row["grade"]),
                    row["evidence_digest"],
                    row["elapsed_ms"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise RemoteEvaluationDispatcherError("screen stage response is invalid") from exc
    try:
        return ArenaScreenReceipt(
            value["service_digest"],  # type: ignore[arg-type]
            value["candidate_digest"],  # type: ignore[arg-type]
            value["screen_attempt"],  # type: ignore[arg-type]
            tuple(results),
            PromotionDecision(value["decision"]),
        )
    except (TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError("screen response is invalid") from exc


def qualification_batch_to_dict(batch: QualificationIntakeBatch) -> dict[str, object]:
    """Return the closed remote representation of an exact qualification batch."""

    if type(batch) is not QualificationIntakeBatch:
        raise RemoteEvaluationDispatcherError("qualification result is not exactly typed")
    retry = batch.retry_plan
    return {
        "attempt_ref": None if batch.attempt_ref is None else batch.attempt_ref.to_dict(),
        "authority_manifest_digest": batch.authority_manifest_digest,
        "outcomes": [
            {
                "attempt_artifact_sha256": row.attempt_artifact_sha256,
                "authority_manifest_digest": row.authority_manifest_digest,
                "decision": row.decision.value,
                "failure_digest": row.failure_digest,
                "reason": row.reason,
                "report_digest": row.report_digest,
                "reservation_digest": row.reservation_digest,
                "retryable": row.retryable,
                "selected_delta_digest": row.selected_delta_digest,
                "settlement_qualification": (
                    None
                    if row.settlement_qualification is None
                    else row.settlement_qualification.to_dict()
                ),
            }
            for row in batch.outcomes
        ],
        "retry_plan": (
            None
            if retry is None
            else {
                "authority_manifest_digest": retry.authority_manifest_digest,
                "failure_digest": retry.failure_digest,
                "reservation_groups": [list(group) for group in retry.reservation_groups],
                "strategy": retry.strategy,
            }
        ),
        "schema_version": _SCHEMA_VERSION,
    }


def qualification_batch_from_dict(value: object) -> QualificationIntakeBatch:
    """Strictly reopen an untrusted remote qualification batch."""

    fields = {
        "attempt_ref",
        "authority_manifest_digest",
        "outcomes",
        "retry_plan",
        "schema_version",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != _SCHEMA_VERSION
        or type(value["outcomes"]) is not list
    ):
        raise RemoteEvaluationDispatcherError("qualification response fields are not closed")
    attempt_value = value["attempt_ref"]
    retry_value = value["retry_plan"]
    try:
        attempt_ref = (
            None
            if attempt_value is None
            else EvidenceArtifactRef.from_dict(attempt_value)
        )
        retry_plan = None
        if retry_value is not None:
            retry_fields = {
                "authority_manifest_digest",
                "failure_digest",
                "reservation_groups",
                "strategy",
            }
            if (
                type(retry_value) is not dict
                or set(retry_value) != retry_fields
                or type(retry_value["reservation_groups"]) is not list
                or any(type(group) is not list for group in retry_value["reservation_groups"])
            ):
                raise RemoteEvaluationDispatcherError("qualification retry plan is malformed")
            retry_plan = QualificationRetryPlan(
                retry_value["authority_manifest_digest"],
                retry_value["strategy"],
                tuple(tuple(group) for group in retry_value["reservation_groups"]),
                retry_value["failure_digest"],
            )
        outcomes = []
        outcome_fields = {
            "attempt_artifact_sha256",
            "authority_manifest_digest",
            "decision",
            "failure_digest",
            "reason",
            "report_digest",
            "reservation_digest",
            "retryable",
            "selected_delta_digest",
            "settlement_qualification",
        }
        for row in value["outcomes"]:
            if type(row) is not dict or set(row) != outcome_fields:
                raise RemoteEvaluationDispatcherError("qualification outcome is malformed")
            settlement_value = row["settlement_qualification"]
            outcomes.append(
                QualificationIntakeOutcome(
                    reservation_digest=row["reservation_digest"],
                    selected_delta_digest=row["selected_delta_digest"],
                    authority_manifest_digest=row["authority_manifest_digest"],
                    decision=QualificationDecision(row["decision"]),
                    reason=row["reason"],
                    retryable=row["retryable"],
                    attempt_artifact_sha256=row["attempt_artifact_sha256"],
                    report_digest=row["report_digest"],
                    failure_digest=row["failure_digest"],
                    settlement_qualification=(
                        None
                        if settlement_value is None
                        else SettlementQualification.from_dict(settlement_value)
                    ),
                )
            )
        return QualificationIntakeBatch(
            value["authority_manifest_digest"],  # type: ignore[arg-type]
            tuple(outcomes),
            attempt_ref,
            retry_plan,
        )
    except RemoteEvaluationDispatcherError:
        raise
    except (TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError("qualification response is invalid") from exc

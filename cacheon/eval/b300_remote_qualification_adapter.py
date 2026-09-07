"""Closed pod adapter for authenticated ordinary-bundle qualification work.

The remote protocol deliberately carries publication *identities*, never pod
paths.  This module is the deployment-owned seam which resolves those
identities to an already materialized immutable publication, reconstructs the
exact typed qualification lease, executes one sealed B300 qualification
deployment, and captures the resulting evidence bytes for CPU import.

Request authentication is intentionally outside this seam.  Callers must run
``verify_remote_request`` before invoking :meth:`B300RemoteQualificationAdapter.run`.
Keeping authentication and execution separate lets the transport own secrets
without giving a submitted bundle any credential, path, command, environment,
or operator-control field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import logging
from pathlib import Path

from cacheon.arena_service import (
    ArenaCandidateBinding,
    PromotionDecision,
)
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.chain.publication import (
    WorkerBundlePublication,
    WorkerBundlePublicationError,
    reopen_worker_bundle,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteEvaluationRequest,
    RemoteQualificationProduct,
    capture_remote_qualification_product,
)
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldProduct,
    RemoteQualificationHoldReason,
    capture_remote_qualification_hold,
)
from cacheon.chain.remote_qualification_evidence import _screen_receipt_from_dict
from cacheon.eval.b300_mainnet_worker import (
    B300MainnetWorker,
    B300MainnetWorkerError,
    B300RemoteQualificationRun,
)
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300QualificationDeployment,
)
from cacheon.eval.b300_qualification_graph_gate import (
    B300QualificationGraphGateHold,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.oci_outer_session import OuterSessionWorkerError
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationReservation,
)
from cacheon.stack_identity import canonical_digest
from cacheon.chain.qualification_request import (
    QUALIFICATION_FIELDS, require_commissioned_incumbent,
)


_LOG = logging.getLogger(__name__)
_SCALAR_TYPES = {type(None), bool, int, float, str, bytes}


class B300RemoteQualificationAdapterError(RuntimeError):
    """Remote qualification differs from fixed pod deployment authority."""


@dataclass(frozen=True)
class B300WorkerBundleResolver:
    """Fixed local mapping from path-free wire identity to one immutable root.

    The tuple is canonicalized by publication receipt digest.  Resolution
    compares the complete public receipt and then independently reopens the
    configured root.  A request therefore cannot introduce a root, select a
    sibling with the same basename, or rely on stale bytes at a once-valid
    location.
    """

    publications: tuple[WorkerBundlePublication, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.publications) if type(self.publications) is tuple else ()
        if (
            not rows
            or any(type(row) is not WorkerBundlePublication for row in rows)
            or rows != tuple(sorted(rows, key=lambda row: row.digest))
            or len({row.digest for row in rows}) != len(rows)
            or len({row.root for row in rows}) != len(rows)
        ):
            raise B300RemoteQualificationAdapterError(
                "worker publication resolver is not one canonical fixed mapping"
            )
        object.__setattr__(self, "publications", rows)

    @property
    def digest(self) -> str:
        """Private deployment identity; the roots never enter the wire product."""

        return canonical_digest(
            "cacheon.eval.b300-worker-bundle-resolver.v1",
            {
                "publications": [
                    {
                        "publication_digest": row.digest,
                        "root": row.root.as_posix(),
                    }
                    for row in self.publications
                ]
            },
        )

    def resolve(self, wire_publication: object) -> WorkerBundlePublication:
        if type(wire_publication) is not dict:
            raise B300RemoteQualificationAdapterError(
                "remote publication identity is not a closed object"
            )
        matches = tuple(
            row for row in self.publications if row.to_dict() == wire_publication
        )
        if len(matches) != 1:
            raise B300RemoteQualificationAdapterError(
                "remote publication identity is absent from the fixed pod mapping"
            )
        configured = matches[0]
        try:
            reopened = reopen_worker_bundle(
                configured.root,
                configured.content_hash,
                expected_publication_digest=configured.publication_digest,
                expected_receipt_digest=configured.digest,
            )
        except WorkerBundlePublicationError as exc:
            raise B300RemoteQualificationAdapterError(
                f"fixed pod publication failed to reopen: {exc}"
            ) from exc
        if (
            reopened.root != configured.root
            or reopened.to_dict() != configured.to_dict()
            or reopened.digest != configured.digest
        ):
            raise B300RemoteQualificationAdapterError(
                "reopened pod publication differs from its fixed mapping"
            )
        return reopened


def _readiness_matches_deployment(
    readiness: WorkerReadiness,
    deployment: B300QualificationDeployment,
) -> bool:
    manifest = deployment.manifest
    runtime = manifest.runtime
    return (
        readiness.service_digest == manifest.digest
        and readiness.arena_id == runtime.arena_id
        and readiness.provider_digest == manifest.provider_digest
        and readiness.runtime_digest == runtime.runtime_digest
        and readiness.worker_distribution_digest
        == runtime.worker_distribution_digest
        and readiness.model_revision_digest == runtime.model_revision_digest
        and readiness.model_manifest_digest == runtime.model_manifest_digest
        and readiness.model_content_digest == runtime.model_content_digest
        and readiness.target_architecture == runtime.target_architecture
        and readiness.topology_class == runtime.topology_class
        and readiness.topology_digest == runtime.topology_digest
        and readiness.gpu_count == runtime.gpu_count
        and readiness.tensor_parallel_size == runtime.tensor_parallel_size
        and readiness.workload_digest == manifest.workload.digest
        and readiness.qualification_policy_digest
        == manifest.qualification_policy_digest
    )


def _qualification_evidence_references(
    batch: QualificationIntakeBatch,
) -> tuple[EvidenceArtifactRef, ...]:
    """Exhaustively find CAS references in a typed batch, without opening paths.

    This intentionally walks every dataclass field, including any nested
    settlement projection.  A future field containing another typed
    ``EvidenceArtifactRef`` is therefore included automatically.  Any new
    opaque/container type fails closed instead of being ignored.
    """

    if type(batch) is not QualificationIntakeBatch:
        raise B300RemoteQualificationAdapterError(
            "qualification evidence batch is not exactly typed"
        )
    found: list[EvidenceArtifactRef] = []
    active: set[int] = set()

    def visit(value: object) -> None:
        if type(value) is EvidenceArtifactRef:
            found.append(value)
            return
        if type(value) in _SCALAR_TYPES:
            return
        if isinstance(value, Enum):
            visit(value.value)
            return
        identity = id(value)
        if identity in active:
            raise B300RemoteQualificationAdapterError(
                "qualification evidence graph contains a cycle"
            )
        if type(value) is tuple or type(value) is list:
            active.add(identity)
            try:
                for row in value:
                    visit(row)
            finally:
                active.remove(identity)
            return
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise B300RemoteQualificationAdapterError(
                    "qualification evidence mapping has a non-string key"
                )
            active.add(identity)
            try:
                for key in sorted(value):
                    visit(value[key])
            finally:
                active.remove(identity)
            return
        if is_dataclass(value) and type(value).__module__.startswith("cacheon."):
            active.add(identity)
            try:
                for row in fields(value):
                    visit(getattr(value, row.name))
            finally:
                active.remove(identity)
            return
        if isinstance(value, Path):
            raise B300RemoteQualificationAdapterError(
                "qualification result exposed a filesystem path"
            )
        raise B300RemoteQualificationAdapterError(
            "qualification evidence graph contains an unknown type: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    visit(batch)
    unique = {row: row for row in found}
    ordered = tuple(
        sorted(
            unique,
            key=lambda row: (
                row.domain,
                row.sha256,
                row.media_type,
                row.schema,
                row.size,
            ),
        )
    )
    if len({row.sha256 for row in ordered}) != len(ordered):
        raise B300RemoteQualificationAdapterError(
            "qualification evidence reused one digest with conflicting metadata"
        )
    return ordered


def _merge_evidence_references(
    batch_references: tuple[EvidenceArtifactRef, ...],
    supporting_references: tuple[EvidenceArtifactRef, ...],
) -> tuple[EvidenceArtifactRef, ...]:
    """Merge worker-retained graph refs with the typed batch inventory."""

    if (
        type(batch_references) is not tuple
        or type(supporting_references) is not tuple
        or any(
            type(row) is not EvidenceArtifactRef
            for row in (*batch_references, *supporting_references)
        )
    ):
        raise B300RemoteQualificationAdapterError(
            "qualification supporting evidence references are not exactly typed"
        )
    unique = {row: row for row in (*batch_references, *supporting_references)}
    ordered = tuple(
        sorted(
            unique,
            key=lambda row: (
                row.domain,
                row.sha256,
                row.media_type,
                row.schema,
                row.size,
            ),
        )
    )
    if len({row.sha256 for row in ordered}) != len(ordered):
        raise B300RemoteQualificationAdapterError(
            "qualification evidence reused one digest with conflicting metadata"
        )
    return ordered


@dataclass(frozen=True)
class B300RemoteQualificationAdapter:
    """One fixed path-free adapter for a sealed B300 qualification deployment."""

    deployment: B300QualificationDeployment
    construction: B300QualificationConstructionAuthority
    readiness: WorkerReadiness
    publications: B300WorkerBundleResolver
    continuation_store: QualificationContinuationStore
    worker: B300MainnetWorker | None = None
    _owns_worker: bool = field(default=False, init=False, repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.deployment) is not B300QualificationDeployment
            or type(self.construction)
            is not B300QualificationConstructionAuthority
            or type(self.readiness) is not WorkerReadiness
            or type(self.publications) is not B300WorkerBundleResolver
            or type(self.continuation_store) is not QualificationContinuationStore
        ):
            raise B300RemoteQualificationAdapterError(
                "remote qualification adapter authority is not exactly typed"
            )
        manifest = self.deployment.manifest
        if (
            self.construction.digest != self.deployment.construction_digest
            or self.construction.incumbent_stack.arena_digest != manifest.digest
            or self.construction.incumbent_stack.runtime_digest
            != manifest.runtime.runtime_digest
            or self.construction.incumbent_stack.base_engine_digest
            != manifest.runtime.base_engine_digest
            or self.construction.qualification_policy_digest
            != manifest.qualification_policy_digest
            or self.deployment.authorities.qualification_stage
            != self.deployment.screen_lane
        ):
            raise B300RemoteQualificationAdapterError(
                "qualification construction differs from the fixed deployment"
            )
        if not _readiness_matches_deployment(self.readiness, self.deployment):
            raise B300RemoteQualificationAdapterError(
                "worker READY authority differs from the qualification deployment"
            )
        worker = self.worker
        owns_worker = worker is None
        if worker is None:
            # Direct construction is a standalone adapter lifetime.  The served
            # runtime always supplies its commissioned owner, so request-local
            # adapters never construct or own a worker.
            worker = B300MainnetWorker(
                manifest,
                self.deployment.authorities,
                self.readiness,
            )
        if (
            type(worker) is not B300MainnetWorker
            or worker.service.manifest != manifest
            or worker.readiness != self.readiness
            or worker._remote_qualification_lane != self.deployment.screen_lane
            or worker.service._provider is not worker._provider
        ):
            if owns_worker:
                worker.close()
            raise B300RemoteQualificationAdapterError(
                "qualification worker differs from the fixed deployment owner"
            )
        try:
            worker._bind_remote_qualification_graph_gate_root(
                self.construction.evidence_root
            )
        except B300MainnetWorkerError as exc:
            if owns_worker:
                worker.close()
            raise B300RemoteQualificationAdapterError(
                "qualification graph gate differs from the fixed deployment"
            ) from exc
        object.__setattr__(self, "worker", worker)
        object.__setattr__(self, "_owns_worker", owns_worker)

    def close(self) -> None:
        """Close a standalone owner; never close an injected epoch owner."""

        if not self._owns_worker or self._closed:
            return
        worker = self.worker
        assert worker is not None
        worker.close()
        object.__setattr__(self, "_closed", True)

    def __enter__(self) -> "B300RemoteQualificationAdapter":
        if self._closed:
            raise B300RemoteQualificationAdapterError(
                "remote qualification adapter is closed"
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.b300-remote-qualification-adapter.v1",
            {
                "construction_digest": self.construction.digest,
                "deployment_manifest_digest": self.deployment.manifest.digest,
                "publication_resolver_digest": self.publications.digest,
                "screen_lane": self.deployment.screen_lane,
                "worker_readiness_digest": self.readiness.digest,
            },
        )

    def run(
        self,
        request: RemoteEvaluationRequest,
    ) -> RemoteQualificationProduct | RemoteQualificationHoldProduct:
        """Execute one already-authenticated closed-v2 qualification request."""

        if self._closed:
            raise B300RemoteQualificationAdapterError(
                "remote qualification adapter is closed"
            )
        if type(request) is not RemoteEvaluationRequest or request.stage != "qualification":
            raise B300RemoteQualificationAdapterError(
                "adapter requires an exact authenticated qualification request"
            )
        body = request.body
        manifest = self.deployment.manifest
        if (
            set(body) != QUALIFICATION_FIELDS
            or body["kind"] != "qualification_work"
            or type(body["candidates"]) is not list
            or not 1 <= len(body["candidates"]) <= manifest.capacity.max_cohort_size
            or len(request.members) != len(body["candidates"])
            or request.worker_readiness_digest != self.readiness.digest
            or request.ready_receipt_digest != self.readiness.ready_receipt_digest
            or request.ready_epoch != self.readiness.ready_epoch
            or request.service_identity != manifest.service_id
            or body["service_digest"] != manifest.digest
            or body["qualification_policy_digest"]
            != manifest.qualification_policy_digest
            or body["qualification_policy_digest"]
            != self.construction.qualification_policy_digest
            or body["screen_lane"] != self.deployment.screen_lane
            or body["screen_lane"]
            != self.deployment.authorities.qualification_stage
        ):
            raise B300RemoteQualificationAdapterError(
                "remote request differs from deployment service, policy, lane, or READY authority"
            )

        require_commissioned_incumbent(body, self.construction)
        candidate_rows = []
        receipt_rows = []
        for row in body["candidates"]:
            reservation = QualificationReservation.from_dict(row["reservation"])
            publication = self.publications.resolve(row["publication"])
            receipt = _screen_receipt_from_dict(row["screen_receipt"])
            try:
                candidate = ArenaCandidateBinding(
                    reservation,
                    publication,
                    receipt.screen_attempt,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise B300RemoteQualificationAdapterError(
                    "remote candidate differs from its fixed pod publication"
                ) from exc
            if (
                row["candidate_digest"] != candidate.digest
                or receipt.candidate_digest != candidate.digest
                or receipt.service_digest != manifest.digest
                or receipt.decision is not PromotionDecision.PROMOTE
            ):
                raise B300RemoteQualificationAdapterError(
                    "remote candidate differs from its promoted receipt"
                )
            candidate_rows.append(candidate)
            receipt_rows.append(receipt)

        candidates = tuple(candidate_rows)
        receipts = tuple(receipt_rows)
        try:
            lease = EvaluationLease(
                request.lease_id,
                request.generation,
                request.stage,
                request.owner,
                request.members,
                request.claimed_block,
                request.initial_expires_block,
                request.initial_expires_block,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise B300RemoteQualificationAdapterError(
                "remote qualification lease or promoted cohort is invalid"
            ) from exc
        if (
            lease.reservation_ids
            != tuple(row.reservation.reservation_digest for row in candidates)
        ):
            raise B300RemoteQualificationAdapterError(
                "remote lease differs from the exact promoted cohort"
            )

        worker = self.worker
        assert worker is not None
        try:
            result = worker.run_remote_qualification(
                lease,
                candidates,
                receipts,
                screen_lane=self.deployment.screen_lane,
                continuation_store=self.continuation_store,
                request_digest=request.digest,
            )
        except OuterSessionWorkerError as exc:
            diagnostic = exc.diagnostic
            _LOG.exception(
                "qualification worker error for request %s: %s: %s",
                request.digest,
                type(exc).__name__,
                exc.message,
            )
            return capture_remote_qualification_hold(
                request,
                reason=RemoteQualificationHoldReason.QUALIFICATION_WORKER_ERROR,
                diagnostic_digest=(
                    ""
                    if diagnostic is None or diagnostic.stream_sha256 is None
                    else diagnostic.stream_sha256
                ),
                failure_type=type(exc).__name__,
                failure_message=exc.message,
            )
        if type(result) is B300QualificationGraphGateHold:
            try:
                hold = capture_remote_qualification_hold(
                    request,
                    reason=result.reason,
                    diagnostic_digest=result.diagnostic_digest,
                    failure_type=result.failure_type,
                    failure_message=result.failure_message,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise B300RemoteQualificationAdapterError(
                    f"qualification graph HOLD could not be captured: {exc}"
                ) from exc
            if type(hold) is not RemoteQualificationHoldProduct:
                raise B300RemoteQualificationAdapterError(
                    "qualification graph HOLD capture returned an untyped product"
                )
            return hold
        if (
            type(result) is not B300RemoteQualificationRun
            or result.run.lease != lease
            or result.screen_lane != self.deployment.screen_lane
            or tuple(result.authority_manifest.reservations)
            != tuple(row.reservation for row in candidates)
        ):
            raise B300RemoteQualificationAdapterError(
                "qualification worker changed the sealed lease or cohort authority"
            )
        result.run.envelope.verify(
            lease,
            self.readiness,
            worker.service,
            result.run.payload,
        )

        batch = result.run.payload
        if type(batch) is not QualificationIntakeBatch:
            raise B300RemoteQualificationAdapterError(
                "qualification worker returned an untyped batch"
            )
        evidence_references = _merge_evidence_references(
            _qualification_evidence_references(batch),
            result.supporting_evidence_refs,
        )
        try:
            product = capture_remote_qualification_product(
                batch=batch,
                authority_manifest=result.authority_manifest,
                incumbent_stack=self.construction.incumbent_stack,
                incumbent_tree_digest=self.construction.incumbent_tree_digest,
                screen_lane=self.deployment.screen_lane,
                service_digest=manifest.digest,
                readiness=self.readiness,
                evidence_root=self.construction.evidence_root,
                evidence_references=evidence_references,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise B300RemoteQualificationAdapterError(
                f"qualification evidence product could not be captured: {exc}"
            ) from exc
        if type(product) is not RemoteQualificationProduct:
            raise B300RemoteQualificationAdapterError(
                "qualification evidence capture returned an untyped product"
            )
        return product


__all__ = [
    "B300RemoteQualificationAdapter",
    "B300RemoteQualificationAdapterError",
    "B300WorkerBundleResolver",
]

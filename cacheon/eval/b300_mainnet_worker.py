"""Closed pod-side execution for one leased B300 evaluation job.

The worker accepts only the exact in-process lease products constructed by the
CPU coordinator.  It does not decode candidate metadata into modules, commands,
arguments, or executables.  Deployment code must supply the already sealed arena
manifest, B300 authorities, and READY receipt before a job can run.

Transport and durable lease mutation remain outside this module.  A fixed,
deployment-owned adapter may deserialize a sealed request into these exact types
and serialize the returned typed envelope, but no field in a candidate bundle or
arena manifest selects that adapter.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaQualificationWork,
    ArenaScreenReceipt,
    ArenaService,
    ArenaServiceManifest,
    PromotionDecision,
)
from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.chain.evaluation_coordinator import (
    SYSTEMIC_QUALIFICATION_REASONS,
    ClaimedQualificationEvaluation,
    ClaimedScreenEvaluation,
    EvaluationCoordinatorError,
    EvaluationResultEnvelope,
    EvaluationRun,
    WorkerReadiness,
)
from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldReason
from cacheon.eval.b300_arena_provider import (
    B300ArenaServiceProvider,
    B300DeploymentAuthorities,
    B300ScreenDeploymentAuthorities,
)
from cacheon.eval.b300_qualification_graph_gate import (
    B300QualificationGraphGateFail,
    B300QualificationGraphGateHold,
    B300QualificationGraphGatePass,
    B300QualificationGraphHoldCode,
    qualification_graph_gate_hold,
    run_b300_qualification_graph_gate,
)
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
    B300QualificationGraphEvidenceStoreError,
)
from cacheon.eval.b300_resident_qualification import (
    B300ResidentQualificationError,
    run_b300_resident_qualification_prefix,
)
from cacheon.eval.resident_pair_quality_lifecycle import (
    ResidentPairMarginalLifecycleEvidence,
    ResidentPairQualityLifecycleError,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import (
    QualificationContinuationError,
    QualificationContinuationStore,
)
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeError,
    QualificationPlanFactory,
    run_qualification_intake,
)
from cacheon.eval.qualification_runner import (
    ATTEMPT_SCHEMA_V3,
    ATTEMPT_SCHEMA_V4,
    reopen_causal_qualification,
)
from cacheon.eval.resident_screen_lane import screen_swappability
from cacheon.manifest import load_manifest
from cacheon.stack_identity import canonical_digest, require_sha256_hex


WORKER_SCHEMA = "cacheon.eval.b300-mainnet-worker.v1"
_LOG = logging.getLogger(__name__)


class B300MainnetWorkerError(RuntimeError):
    """A leased job or sealed worker authority is inconsistent."""


def _resident_evidence_hold(
    request_digest: str,
    authority_digest: str,
    source_digest: str,
) -> B300QualificationGraphGateHold:
    return B300QualificationGraphGateHold(
        RemoteQualificationHoldReason.RESIDENT_EVIDENCE_UNAVAILABLE,
        canonical_digest(
            "cacheon.eval.b300-resident-qualification-hold.v1",
            {
                "authority": authority_digest,
                "request": request_digest,
                "source": source_digest,
            },
        ),
    )


@dataclass(frozen=True)
class B300RemoteQualificationRun:
    """Pod result plus the public authority that produced its evidence.

    ``EvaluationRun`` deliberately carries only the qualification batch.  The
    CPU cannot safely infer the private plan's public authority from that batch,
    so the remote boundary returns the exact manifest alongside it.  Evidence
    bytes and incumbent identities remain deployment-owned product fields; they
    are not filesystem paths on this object.
    """

    run: EvaluationRun
    authority_manifest: QualificationAuthorityManifest
    screen_lane: str
    supporting_evidence_refs: tuple[EvidenceArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        references = (
            tuple(self.supporting_evidence_refs)
            if type(self.supporting_evidence_refs) is tuple
            else ()
        )
        if (
            type(self.run) is not EvaluationRun
            or type(self.authority_manifest) is not QualificationAuthorityManifest
            or type(self.run.payload) is not QualificationIntakeBatch
            or self.run.lease.stage != "qualification"
            or self.run.payload.authority_manifest_digest
            != self.authority_manifest.digest
            or self.screen_lane not in {"primary", "reproduction"}
            or type(self.supporting_evidence_refs) is not tuple
            or any(type(row) is not EvidenceArtifactRef for row in references)
            or references
            != tuple(
                sorted(
                    references,
                    key=lambda row: (
                        row.domain,
                        row.sha256,
                        row.media_type,
                        row.schema,
                        row.size,
                    ),
                )
            )
            or len(set(references)) != len(references)
            or len({row.sha256 for row in references}) != len(references)
        ):
            raise B300MainnetWorkerError(
                "remote qualification result changed its sealed authority"
            )
        object.__setattr__(self, "supporting_evidence_refs", references)


class B300MainnetWorker:
    """Long-lived, single-flight worker over one sealed B300 service epoch."""

    def __init__(
        self,
        manifest: ArenaServiceManifest,
        authorities: B300DeploymentAuthorities | B300ScreenDeploymentAuthorities,
        readiness: WorkerReadiness,
    ) -> None:
        if type(manifest) is not ArenaServiceManifest:
            raise B300MainnetWorkerError("worker manifest is not exactly typed")
        if type(authorities) not in {
            B300DeploymentAuthorities,
            B300ScreenDeploymentAuthorities,
        }:
            raise B300MainnetWorkerError("worker deployment authorities are not exact")
        if type(readiness) is not WorkerReadiness:
            raise B300MainnetWorkerError("worker readiness is not exactly typed")
        provider = B300ArenaServiceProvider(manifest, authorities)
        service = ArenaService(manifest, provider)
        self._validate_readiness(readiness, service)
        self.service = service
        self.readiness = readiness
        self._provider = provider
        self._remote_qualification_lane = (
            authorities.qualification_stage
            if type(authorities) is B300DeploymentAuthorities
            else None
        )
        self._resident_pair_factory = (
            authorities.resident_pair_factory
            if type(authorities) is B300DeploymentAuthorities
            else None
        )
        self._resident_count_quality = (
            authorities.resident_count_quality
            if type(authorities) is B300DeploymentAuthorities
            else None
        )
        self._remote_qualification_graph_root: Path | None = None
        self._closed = False
        self._lock = threading.RLock()
        self.worker_digest = canonical_digest(
            WORKER_SCHEMA,
            {
                "provider_digest": provider.provider_digest,
                "ready_epoch": readiness.ready_epoch,
                "ready_receipt_digest": readiness.ready_receipt_digest,
                "remote_qualification_lane": self._remote_qualification_lane,
                "service_digest": service.identity,
                "worker_readiness_digest": readiness.digest,
            },
        )

    def run(
        self,
        job: ClaimedScreenEvaluation | ClaimedQualificationEvaluation,
    ) -> EvaluationRun:
        """Execute one exact leased job and seal its typed result envelope."""

        if type(job) not in {ClaimedScreenEvaluation, ClaimedQualificationEvaluation}:
            raise B300MainnetWorkerError("leased evaluation job is not exactly typed")
        with self._lock:
            if self._closed:
                raise B300MainnetWorkerError("B300 mainnet worker is closed")
            self._validate_readiness(self.readiness, self.service)
            if type(job) is ClaimedScreenEvaluation:
                payload = self._run_screen(job)
                disposition = "completed"
            else:
                payload = self._run_qualification(job)
                disposition = (
                    "released" if self._systemic(payload) else "completed"
                )
            envelope = EvaluationResultEnvelope.seal(
                job.lease,
                self.readiness,
                self.service,
                payload,
            )
            return EvaluationRun(job.lease, envelope, payload, disposition)

    def run_remote_screen(
        self,
        lease: EvaluationLease,
        candidate: ArenaCandidateBinding,
    ) -> EvaluationRun:
        """Run the path-free authenticated screen DTO used by the pod codec.

        Remote transport does not possess the CPU-only ``IntakeReservation``
        needed to recreate ``ClaimedScreenEvaluation``.  The lease already
        commits the sole qualification reservation, so this narrow entrypoint
        verifies that exact binding and returns the ordinary sealed run.
        """

        if (
            type(lease) is not EvaluationLease
            or type(candidate) is not ArenaCandidateBinding
            or lease.stage != "screen"
            or lease.reservation_ids
            != (candidate.reservation.reservation_digest,)
        ):
            raise B300MainnetWorkerError(
                "remote screen lease differs from the exact candidate"
            )
        with self._lock:
            if self._closed:
                raise B300MainnetWorkerError("B300 mainnet worker is closed")
            self._validate_readiness(self.readiness, self.service)
            payload = self._screen_candidate(candidate)
            envelope = EvaluationResultEnvelope.seal(
                lease,
                self.readiness,
                self.service,
                payload,
            )
            return EvaluationRun(lease, envelope, payload, "completed")

    def run_remote_qualification(
        self,
        lease: EvaluationLease,
        candidates: tuple[ArenaCandidateBinding, ...],
        screen_receipts: tuple[ArenaScreenReceipt, ...],
        *,
        screen_lane: str,
        continuation_store: QualificationContinuationStore,
        request_digest: str,
    ) -> B300RemoteQualificationRun | B300QualificationGraphGateHold:
        """Run one path-free, lane-bound remote qualification cohort.

        The CPU transport sends immutable publications, reservations, promoted
        receipts, and the retained primary/reproduction lane.  A deployment
        composes this worker with the corresponding physical TP4 role
        orientation; this method refuses a missing or differently oriented lane
        before constructing private qualification work.
        """

        if type(continuation_store) is not QualificationContinuationStore:
            raise B300MainnetWorkerError(
                "remote qualification continuation store is not exact"
            )
        try:
            request_digest = require_sha256_hex(
                request_digest, field="authenticated request digest"
            )
        except ValueError as exc:
            raise B300MainnetWorkerError(str(exc)) from None
        candidate_rows = tuple(candidates) if type(candidates) is tuple else ()
        receipt_rows = (
            tuple(screen_receipts) if type(screen_receipts) is tuple else ()
        )
        if (
            type(lease) is not EvaluationLease
            or lease.stage != "qualification"
            or not candidate_rows
            or any(type(row) is not ArenaCandidateBinding for row in candidate_rows)
            or len(candidate_rows) != len(receipt_rows)
            or any(type(row) is not ArenaScreenReceipt for row in receipt_rows)
            or lease.reservation_ids
            != tuple(row.reservation.reservation_digest for row in candidate_rows)
            or tuple(row.candidate_digest for row in receipt_rows)
            != tuple(row.digest for row in candidate_rows)
            or any(
                row.service_digest != self.service.identity
                or row.decision is not PromotionDecision.PROMOTE
                for row in receipt_rows
            )
            or screen_lane not in {"primary", "reproduction"}
            or screen_lane != self._remote_qualification_lane
            or (screen_lane == "reproduction" and len(candidate_rows) != 1)
        ):
            raise B300MainnetWorkerError(
                "remote qualification lease differs from the exact promoted cohort"
            )
        with self._lock:
            if self._closed:
                raise B300MainnetWorkerError("B300 mainnet worker is closed")
            self._validate_readiness(self.readiness, self.service)
            execution = self._execute_qualification(
                candidate_rows,
                receipt_rows,
                continuation_store=continuation_store,
                request_digest=request_digest,
            )
            if type(execution) is B300QualificationGraphGateHold:
                return execution
            payload, authority_manifest, supporting_evidence_refs = execution
            disposition = "released" if self._systemic(payload) else "completed"
            envelope = EvaluationResultEnvelope.seal(
                lease,
                self.readiness,
                self.service,
                payload,
            )
            run = EvaluationRun(lease, envelope, payload, disposition)
            return B300RemoteQualificationRun(
                run,
                authority_manifest,
                screen_lane,
                supporting_evidence_refs,
            )

    def close(self) -> None:
        """Permanently release qualification and screen resident lifetimes."""

        with self._lock:
            if self._closed:
                return
            if self._resident_pair_factory is not None:
                self._resident_pair_factory.close()
            self._provider.close()
            self._closed = True

    def retire_resident_screen(self) -> None:
        """Fence the standing screen lifetime before either TP4 orientation runs."""

        with self._lock:
            if self._closed:
                raise B300MainnetWorkerError("B300 mainnet worker is closed")
            self._validate_readiness(self.readiness, self.service)
            self._provider.retire_resident_screen()

    def _bind_remote_qualification_graph_gate_root(self, root: Path) -> None:
        """Bind the adapter-owned CAS root once for this resident worker epoch."""

        if not isinstance(root, Path) or not root.is_absolute() or root != Path(
            root.as_posix()
        ):
            raise B300MainnetWorkerError(
                "remote qualification graph root is not canonical and absolute"
            )
        with self._lock:
            if self._closed:
                raise B300MainnetWorkerError("B300 mainnet worker is closed")
            current = self._remote_qualification_graph_root
            if current is not None and current != root:
                raise B300MainnetWorkerError(
                    "remote qualification graph root changed within the worker epoch"
                )
            self._remote_qualification_graph_root = root

    def __enter__(self) -> "B300MainnetWorker":
        with self._lock:
            if self._closed:
                raise B300MainnetWorkerError("B300 mainnet worker is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _run_screen(self, job: ClaimedScreenEvaluation) -> ArenaScreenReceipt:
        return self._screen_candidate(job.candidate)

    def _screen_candidate(
        self,
        candidate: ArenaCandidateBinding,
    ) -> ArenaScreenReceipt:
        receipt = self.service.screen(candidate)
        if (
            type(receipt) is not ArenaScreenReceipt
            or receipt.service_digest != self.service.identity
            or receipt.candidate_digest != candidate.digest
            or receipt.screen_attempt != candidate.screen_attempt
        ):
            raise B300MainnetWorkerError(
                "screen result changed the exact leased candidate"
            )
        return receipt

    def _run_qualification(
        self,
        job: ClaimedQualificationEvaluation,
    ) -> QualificationIntakeBatch:
        execution = self._execute_qualification(
            job.candidates,
            job.screen_receipts,
        )
        if type(execution) is B300QualificationGraphGateHold:
            raise B300MainnetWorkerError(
                "local qualification unexpectedly returned a remote graph HOLD"
            )
        batch, _manifest, _references = execution
        return batch

    def _execute_qualification(
        self,
        candidates: tuple[ArenaCandidateBinding, ...],
        screen_receipts: tuple[ArenaScreenReceipt, ...],
        *,
        continuation_store: QualificationContinuationStore | None = None,
        request_digest: str | None = None,
    ) -> (
        tuple[
            QualificationIntakeBatch,
            QualificationAuthorityManifest,
            tuple[EvidenceArtifactRef, ...],
        ]
        | B300QualificationGraphGateHold
    ):
        try:
            work = self.service.plan_qualification(
                candidates,
                screen_receipts,
                state=None,
            )
        except (
            B300QualificationGraphEvidenceHold,
            B300QualificationGraphEvidenceStoreError,
        ):
            if request_digest is None:
                raise
            _LOG.exception(
                "qualification graph evidence unavailable while planning for "
                "request %s; the qualification is held without a candidate decision",
                request_digest,
            )
            return qualification_graph_gate_hold(
                RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE,
                authenticated_request_digest=request_digest,
                authority_context_digest=canonical_digest(
                    "cacheon.eval.b300-qualification-graph-provider-context.v1",
                    {
                        "candidate_digests": [row.digest for row in candidates],
                        "reservation_digests": [
                            row.reservation.reservation_digest for row in candidates
                        ],
                    },
                ),
                code=B300QualificationGraphHoldCode.GRAPH_PROVIDER_UNAVAILABLE,
            )
        self._validate_work(work, candidates)
        supporting_evidence_refs: tuple[EvidenceArtifactRef, ...] = ()
        resident_pair_lifecycle = None
        graph_root = self._remote_qualification_graph_root
        if request_digest is not None:
            if graph_root is None:
                return qualification_graph_gate_hold(
                    RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE,
                    authenticated_request_digest=request_digest,
                    authority_context_digest=work.factory.manifest.digest,
                    code=B300QualificationGraphHoldCode.GRAPH_PROVIDER_UNAVAILABLE,
                )
            try:
                plan = work.factory.build()
            except (
                B300QualificationGraphEvidenceHold,
                B300QualificationGraphEvidenceStoreError,
            ):
                retired = False
                if self._resident_pair_factory is not None:
                    try:
                        retired = (
                            self._resident_pair_factory.retire_released_pair()
                        )
                    except Exception:
                        _LOG.exception(
                            "released resident pair retirement failed after a "
                            "graph evidence hold for request %s",
                            request_digest,
                        )
                if not retired:
                    _LOG.exception(
                        "qualification graph evidence unavailable while building "
                        "the plan for request %s; the qualification is held "
                        "without a candidate decision",
                        request_digest,
                    )
                    return qualification_graph_gate_hold(
                        RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE,
                        authenticated_request_digest=request_digest,
                        authority_context_digest=work.factory.manifest.digest,
                        code=B300QualificationGraphHoldCode.GRAPH_PROVIDER_UNAVAILABLE,
                    )
                _LOG.warning(
                    "graph evidence capture held on busy devices for request %s; "
                    "retired the released resident pair and retrying the plan "
                    "build once",
                    request_digest,
                )
                try:
                    plan = work.factory.build()
                except (
                    B300QualificationGraphEvidenceHold,
                    B300QualificationGraphEvidenceStoreError,
                ):
                    _LOG.exception(
                        "qualification graph evidence still unavailable after "
                        "pair retirement for request %s; the qualification is "
                        "held without a candidate decision",
                        request_digest,
                    )
                    return qualification_graph_gate_hold(
                        RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE,
                        authenticated_request_digest=request_digest,
                        authority_context_digest=work.factory.manifest.digest,
                        code=B300QualificationGraphHoldCode.GRAPH_PROVIDER_UNAVAILABLE,
                    )
                except QualificationIntakeError as exc:
                    raise B300MainnetWorkerError(
                        "remote graph gate could not reopen the prebuilt "
                        "qualification plan"
                    ) from exc
            except QualificationIntakeError as exc:
                raise B300MainnetWorkerError(
                    "remote graph gate could not reopen the prebuilt qualification plan"
                ) from exc
            graph = run_b300_qualification_graph_gate(
                work.factory,
                plan,
                evidence_root=graph_root,
                candidates=candidates,
                authenticated_request_digest=request_digest,
            )
            if type(graph) is B300QualificationGraphGateHold:
                _LOG.error(
                    "qualification graph gate held request %s with reason %s "
                    "diagnostic %s",
                    request_digest,
                    graph.reason,
                    graph.diagnostic_digest,
                )
                return graph
            if (
                graph.plan is not plan
                or graph.factory is not work.factory
                or type(graph)
                not in {
                    B300QualificationGraphGatePass,
                    B300QualificationGraphGateFail,
                }
            ):
                raise B300MainnetWorkerError(
                    "graph gate changed the exact prebuilt qualification plan"
                )
            supporting_evidence_refs = graph.supporting_evidence_refs
            if type(graph) is B300QualificationGraphGateFail:
                self._validate_batch(graph.batch, work, candidates)
                return (
                    graph.batch,
                    work.factory.manifest,
                    supporting_evidence_refs,
                )
            assert continuation_store is not None
            continuation = continuation_store.scope(
                request_digest=request_digest,
                authority_digest=work.factory.manifest.authority_digest,
                source_digest=plan.prepared.source.digest,
            )
            if screen_swappability(load_manifest(candidates[0].publication.root)) is None:
                try:
                    resident_prefix = run_b300_resident_qualification_prefix(
                        factory=self._resident_pair_factory,
                        capability=self._resident_count_quality,
                        candidate=candidates[0],
                        plan=plan,
                        continuation=continuation,
                        screen_lane=self._remote_qualification_lane,
                        deadline=float(work.deadline),
                    )
                    resident_pair_lifecycle = ResidentPairMarginalLifecycleEvidence(
                        plan.prepared,
                        resident_prefix.speed_plan,
                        resident_prefix.speed,
                        resident_prefix.retirement,
                        resident_prefix.count_result,
                        resident_prefix.count_checkpoint,
                        (
                            None if resident_prefix.count_result is None
                            else self._resident_count_quality.stock_authority
                        ),
                    )
                except (
                    B300ResidentQualificationError,
                    ResidentPairQualityLifecycleError,
                ):
                    _LOG.exception(
                        "resident qualification prefix failed for request %s; the "
                        "qualification is held without a candidate decision",
                        request_digest,
                    )
                    return _resident_evidence_hold(
                        request_digest,
                        work.factory.manifest.authority_digest,
                        plan.prepared.source.digest,
                    )
        try:
            batch = run_qualification_intake(
                work.factory,
                executor=work.executor,
                resident_baseline_executor=work.resident_baseline_executor,
                entropy_provider=work.entropy_provider,
                hidden_judge=work.hidden_judge,
                deadline=float(work.deadline),
                continuation_store=continuation_store,
                request_digest=request_digest,
                prebuilt_plan=plan if request_digest is not None else None,
                resident_pair_lifecycle=(
                    resident_pair_lifecycle if request_digest is not None else None
                ),
            )
        except QualificationContinuationError:
            if request_digest is None:
                raise
            _LOG.exception(
                "resident continuation failed for request %s; the qualification "
                "is held without a candidate decision",
                request_digest,
            )
            return _resident_evidence_hold(
                request_digest,
                work.factory.manifest.authority_digest,
                plan.prepared.source.digest,
            )
        if request_digest is not None:
            count_checkpoint = (
                None
                if resident_pair_lifecycle is None
                else resident_pair_lifecycle.count_checkpoint
            )
            if count_checkpoint is not None and resident_pair_lifecycle is not None:
                assert resident_pair_lifecycle.stock_authority is not None
                supporting_evidence_refs += (
                    count_checkpoint.raw_execution_evidence,
                    count_checkpoint.candidate_observation,
                    resident_pair_lifecycle.stock_authority.artifact,
                )
            if (
                batch.attempt_ref is not None
                and batch.attempt_ref.schema in {ATTEMPT_SCHEMA_V3, ATTEMPT_SCHEMA_V4}
            ):
                attempt = (
                    reopen_causal_qualification(
                        plan.evidence_root,
                        batch.attempt_ref,
                        expected=plan,
                        resident_pair_lifecycle=resident_pair_lifecycle,
                    )
                    if resident_pair_lifecycle is not None
                    else reopen_causal_qualification(
                        plan.evidence_root,
                        batch.attempt_ref,
                        expected=plan,
                    )
                )
                supporting_evidence_refs += tuple(
                    reference
                    for report in attempt.reports
                    for reference in (
                        report.raw_quality_artifact,
                        *(() if report.repeat_quality is None else (
                            report.repeat_quality.raw_quality_artifact,
                        )),
                    )
                )
            supporting_evidence_refs = tuple(sorted(
                set(supporting_evidence_refs),
                key=lambda row: (
                    row.domain, row.sha256, row.media_type, row.schema, row.size
                ),
            ))
        self._validate_batch(batch, work, candidates)
        return batch, work.factory.manifest, supporting_evidence_refs

    @staticmethod
    def _validate_work(
        work: ArenaQualificationWork,
        candidates: tuple[ArenaCandidateBinding, ...],
    ) -> None:
        expected = tuple(candidate.reservation for candidate in candidates)
        if (
            type(work) is not ArenaQualificationWork
            or type(work.factory) is not QualificationPlanFactory
            or work.factory.manifest.reservations != expected
            or type(work.executor) is not OCIEngineExecutor
            or type(work.resident_baseline_executor) is not OCIEngineExecutor
            or work.executor is work.resident_baseline_executor
            or not callable(work.entropy_provider)
            or not callable(work.hidden_judge)
        ):
            raise B300MainnetWorkerError(
                "qualification work changed the exact leased authority"
            )

    @staticmethod
    def _validate_batch(
        batch: QualificationIntakeBatch,
        work: ArenaQualificationWork,
        candidates: tuple[ArenaCandidateBinding, ...],
    ) -> None:
        expected = tuple(candidate.reservation for candidate in candidates)
        if (
            type(batch) is not QualificationIntakeBatch
            or batch.authority_manifest_digest != work.factory.manifest.digest
            or tuple(row.reservation_digest for row in batch.outcomes)
            != tuple(row.reservation_digest for row in expected)
            or tuple(row.selected_delta_digest for row in batch.outcomes)
            != tuple(row.selected_delta_digest for row in expected)
        ):
            raise B300MainnetWorkerError(
                "qualification result changed the exact leased cohort"
            )

    @staticmethod
    def _systemic(batch: QualificationIntakeBatch) -> bool:
        return bool(batch.outcomes) and all(
            outcome.decision is QualificationDecision.NO_DECISION
            and outcome.reason in SYSTEMIC_QUALIFICATION_REASONS
            for outcome in batch.outcomes
        )

    @staticmethod
    def _validate_readiness(
        readiness: WorkerReadiness,
        service: ArenaService,
    ) -> None:
        try:
            readiness.validate(service)
        except EvaluationCoordinatorError as exc:
            raise B300MainnetWorkerError(
                "worker readiness differs from the sealed B300 service"
            ) from exc


def run_b300_mainnet_job(
    job: ClaimedScreenEvaluation | ClaimedQualificationEvaluation,
    *,
    manifest: ArenaServiceManifest,
    authorities: B300DeploymentAuthorities | B300ScreenDeploymentAuthorities,
    readiness: WorkerReadiness,
) -> EvaluationRun:
    """One-shot in-process entrypoint for a deployment-owned sealed adapter."""

    with B300MainnetWorker(manifest, authorities, readiness) as worker:
        return worker.run(job)


__all__ = [
    "B300MainnetWorker",
    "B300MainnetWorkerError",
    "B300RemoteQualificationRun",
    "WORKER_SCHEMA",
    "run_b300_mainnet_job",
]

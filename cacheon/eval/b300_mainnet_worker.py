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

import threading
from dataclasses import dataclass

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
from cacheon.eval.b300_arena_provider import (
    B300ArenaServiceProvider,
    B300DeploymentAuthorities,
    B300ScreenDeploymentAuthorities,
)
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationPlanFactory,
    run_qualification_intake,
)
from cacheon.stack_identity import canonical_digest


WORKER_SCHEMA = "cacheon.eval.b300-mainnet-worker.v1"


class B300MainnetWorkerError(RuntimeError):
    """A leased job or sealed worker authority is inconsistent."""


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

    def __post_init__(self) -> None:
        if (
            type(self.run) is not EvaluationRun
            or type(self.authority_manifest) is not QualificationAuthorityManifest
            or type(self.run.payload) is not QualificationIntakeBatch
            or self.run.lease.stage != "qualification"
            or self.run.payload.authority_manifest_digest
            != self.authority_manifest.digest
            or self.screen_lane not in {"primary", "reproduction"}
        ):
            raise B300MainnetWorkerError(
                "remote qualification result changed its sealed authority"
            )


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
    ) -> B300RemoteQualificationRun:
        """Run one path-free, lane-bound remote qualification cohort.

        The CPU transport sends immutable publications, reservations, promoted
        receipts, and the retained primary/reproduction lane.  A deployment
        composes this worker with the corresponding physical TP4 role
        orientation; this method refuses a missing or differently oriented lane
        before constructing private qualification work.
        """

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
            payload, authority_manifest = self._execute_qualification(
                candidate_rows,
                receipt_rows,
            )
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
            )

    def close(self) -> None:
        """Permanently release the resident provider lifetime."""

        with self._lock:
            if self._closed:
                return
            self._provider.close()
            self._closed = True

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
        batch, _manifest = self._execute_qualification(
            job.candidates,
            job.screen_receipts,
        )
        return batch

    def _execute_qualification(
        self,
        candidates: tuple[ArenaCandidateBinding, ...],
        screen_receipts: tuple[ArenaScreenReceipt, ...],
    ) -> tuple[QualificationIntakeBatch, QualificationAuthorityManifest]:
        work = self.service.plan_qualification(
            candidates,
            screen_receipts,
            state=None,
        )
        self._validate_work(work, candidates)
        batch = run_qualification_intake(
            work.factory,
            executor=work.executor,
            resident_baseline_executor=work.resident_baseline_executor,
            entropy_provider=work.entropy_provider,
            hidden_judge=work.hidden_judge,
            deadline=float(work.deadline),
        )
        self._validate_batch(batch, work, candidates)
        return batch, work.factory.manifest

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

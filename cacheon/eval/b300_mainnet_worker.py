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

from cacheon.arena_service import (
    ArenaQualificationWork,
    ArenaScreenReceipt,
    ArenaService,
    ArenaServiceManifest,
)
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
)
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationPlanFactory,
    run_qualification_intake,
)
from cacheon.stack_identity import canonical_digest


WORKER_SCHEMA = "cacheon.eval.b300-mainnet-worker.v1"


class B300MainnetWorkerError(RuntimeError):
    """A leased job or sealed worker authority is inconsistent."""


class B300MainnetWorker:
    """Long-lived, single-flight worker over one sealed B300 service epoch."""

    def __init__(
        self,
        manifest: ArenaServiceManifest,
        authorities: B300DeploymentAuthorities,
        readiness: WorkerReadiness,
    ) -> None:
        if type(manifest) is not ArenaServiceManifest:
            raise B300MainnetWorkerError("worker manifest is not exactly typed")
        if type(authorities) is not B300DeploymentAuthorities:
            raise B300MainnetWorkerError("worker deployment authorities are not exact")
        if type(readiness) is not WorkerReadiness:
            raise B300MainnetWorkerError("worker readiness is not exactly typed")
        provider = B300ArenaServiceProvider(manifest, authorities)
        service = ArenaService(manifest, provider)
        self._validate_readiness(readiness, service)
        self.service = service
        self.readiness = readiness
        self._provider = provider
        self._closed = False
        self._lock = threading.RLock()
        self.worker_digest = canonical_digest(
            WORKER_SCHEMA,
            {
                "provider_digest": provider.provider_digest,
                "ready_epoch": readiness.ready_epoch,
                "ready_receipt_digest": readiness.ready_receipt_digest,
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
        receipt = self.service.screen(job.candidate)
        if (
            type(receipt) is not ArenaScreenReceipt
            or receipt.service_digest != self.service.identity
            or receipt.candidate_digest != job.candidate.digest
            or receipt.screen_attempt != job.candidate.screen_attempt
        ):
            raise B300MainnetWorkerError(
                "screen result changed the exact leased candidate"
            )
        return receipt

    def _run_qualification(
        self,
        job: ClaimedQualificationEvaluation,
    ) -> QualificationIntakeBatch:
        work = self.service.plan_qualification(
            job.candidates,
            job.screen_receipts,
            state=None,
        )
        self._validate_work(work, job)
        batch = run_qualification_intake(
            work.factory,
            executor=work.executor,
            resident_baseline_executor=work.resident_baseline_executor,
            entropy_provider=work.entropy_provider,
            hidden_judge=work.hidden_judge,
            deadline=float(work.deadline),
        )
        self._validate_batch(batch, work, job)
        return batch

    @staticmethod
    def _validate_work(
        work: ArenaQualificationWork,
        job: ClaimedQualificationEvaluation,
    ) -> None:
        expected = tuple(candidate.reservation for candidate in job.candidates)
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
        job: ClaimedQualificationEvaluation,
    ) -> None:
        expected = tuple(candidate.reservation for candidate in job.candidates)
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
    authorities: B300DeploymentAuthorities,
    readiness: WorkerReadiness,
) -> EvaluationRun:
    """One-shot in-process entrypoint for a deployment-owned sealed adapter."""

    with B300MainnetWorker(manifest, authorities, readiness) as worker:
        return worker.run(job)


__all__ = [
    "B300MainnetWorker",
    "B300MainnetWorkerError",
    "WORKER_SCHEMA",
    "run_b300_mainnet_job",
]

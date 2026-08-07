"""CPU-owned durable dispatch for asynchronous screening and qualification.

The coordinator deliberately has no settlement, signing, weight, chain-client, or
worker implementation authority.  It briefly owns :class:`FinalizedIntakeStore`
to claim or CAS-commit an exact lease, and closes that flock-backed controller
before reopening publications or calling an arena provider.  A bounded heartbeat
thread may briefly reopen the store while synchronous worker code is running; it
never shares a store object with that code.

``advance_finalized_cursor`` is the deployment seam for finalized-chain progress.
It must finish any intake advancement itself and return the exact durable
``(block, block_hash)`` cursor.  The coordinator calls it before opening the store
and verifies the returned cursor against SQLite, so network work cannot occur
while the controller lock is held.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, NoReturn

from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaQualificationWork,
    ArenaScreenReceipt,
    ArenaService,
)
from cacheon.chain.evaluation_leases import (
    EvaluationLease,
    EvaluationLeaseMember,
    require_evaluation_owner,
)
from cacheon.chain.intake import (
    FinalizedIntakeStore,
    IntakeError,
    IntakePolicy,
    IntakeReservation,
    IntakeScope,
    is_lock_collision,
)
from cacheon.chain.publication import (
    WorkerBundlePublication,
    reopen_worker_bundle,
)
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    EvidenceStoreError,
    reopen_evidence,
)
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationPlanFactory,
    QualificationReservation,
    run_qualification_intake,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest


_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_READINESS_SCHEMA_VERSION = 1
_RESULT_SCHEMA_VERSION = 1
_TRANSIENT_COORDINATOR_OWNERSHIP_MESSAGE = (
    "evaluation store lock/cursor did not stabilize within retry bounds"
)
_REMOTE_EVIDENCE_ARTIFACT_LIMIT = 16 << 20
_REMOTE_EVIDENCE_ARTIFACTS_LIMIT = 256
_REMOTE_EVIDENCE_TOTAL_LIMIT = 32 << 20

SYSTEMIC_QUALIFICATION_REASONS = frozenset(
    {
        "qualification_plan",
        "raw_speed_evidence",
        "outer_session_process",
        "outer_session_worker",
        "oci_backend",
        "qualification_runner",
    }
)


class EvaluationCoordinatorError(RuntimeError):
    """The CPU coordinator cannot safely claim, run, or commit evaluation work."""


class _TransientCoordinatorOwnership(EvaluationCoordinatorError):
    """A bounded flock/cursor acquisition pass lost only a transient race."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EvaluationCoordinatorError(str(exc)) from None


def _closed_cursor(value: object) -> tuple[int, str]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or type(value[0]) is not int
        or value[0] < 0
        or not isinstance(value[1], str)
        or _BLOCK_HASH.fullmatch(value[1]) is None
    ):
        raise EvaluationCoordinatorError(
            "finalized cursor advancer returned a malformed durable cursor"
        )
    return value


@dataclass(frozen=True)
class WorkerReadiness:
    """Closed READY identity for one worker fleet and arena service epoch."""

    ready_receipt_digest: str
    ready_epoch: int
    service_digest: str
    arena_id: str
    provider_digest: str
    runtime_digest: str
    worker_distribution_digest: str
    model_revision_digest: str
    model_manifest_digest: str
    model_content_digest: str
    target_architecture: str
    topology_class: str
    topology_digest: str
    gpu_count: int
    tensor_parallel_size: int
    workload_digest: str
    qualification_policy_digest: str
    schema_version: int = _READINESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "ready_receipt_digest",
            "service_digest",
            "provider_digest",
            "runtime_digest",
            "worker_distribution_digest",
            "model_revision_digest",
            "model_manifest_digest",
            "model_content_digest",
            "topology_digest",
            "workload_digest",
            "qualification_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if (
            type(self.ready_epoch) is not int
            or self.ready_epoch < 0
            or type(self.schema_version) is not int
            or self.schema_version != _READINESS_SCHEMA_VERSION
            or not isinstance(self.arena_id, str)
            or not self.arena_id
            or not isinstance(self.target_architecture, str)
            or not self.target_architecture
            or not isinstance(self.topology_class, str)
            or not self.topology_class
            or type(self.gpu_count) is not int
            or self.gpu_count <= 0
            or type(self.tensor_parallel_size) is not int
            or not 0 < self.tensor_parallel_size <= self.gpu_count
        ):
            raise EvaluationCoordinatorError("worker readiness is not a closed identity")

    @classmethod
    def for_service(
        cls,
        service: ArenaService,
        *,
        ready_receipt_digest: str,
        ready_epoch: int,
    ) -> "WorkerReadiness":
        if type(service) is not ArenaService:
            raise EvaluationCoordinatorError("worker readiness requires an exact arena service")
        manifest = service.manifest
        runtime = manifest.runtime
        return cls(
            ready_receipt_digest=ready_receipt_digest,
            ready_epoch=ready_epoch,
            service_digest=manifest.digest,
            arena_id=runtime.arena_id,
            provider_digest=manifest.provider_digest,
            runtime_digest=runtime.runtime_digest,
            worker_distribution_digest=runtime.worker_distribution_digest,
            model_revision_digest=runtime.model_revision_digest,
            model_manifest_digest=runtime.model_manifest_digest,
            model_content_digest=runtime.model_content_digest,
            target_architecture=runtime.target_architecture,
            topology_class=runtime.topology_class,
            topology_digest=runtime.topology_digest,
            gpu_count=runtime.gpu_count,
            tensor_parallel_size=runtime.tensor_parallel_size,
            workload_digest=manifest.workload.digest,
            qualification_policy_digest=manifest.qualification_policy_digest,
        )

    def validate(self, service: ArenaService) -> None:
        """Reject any service/worker drift before a durable lease is claimed."""

        expected = WorkerReadiness.for_service(
            service,
            ready_receipt_digest=self.ready_receipt_digest,
            ready_epoch=self.ready_epoch,
        )
        if self != expected:
            raise EvaluationCoordinatorError(
                "worker READY identity differs from the arena service manifest"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.worker-readiness.v1",
            self.to_dict(),
        )


def qualification_batch_payload_digest(batch: QualificationIntakeBatch) -> str:
    """Hash every typed result field needed to make a batch replay-safe."""

    if type(batch) is not QualificationIntakeBatch:
        raise EvaluationCoordinatorError("qualification payload is not exactly typed")
    retry = batch.retry_plan
    retry_value = None
    if retry is not None:
        retry_value = {
            "authority_manifest_digest": retry.authority_manifest_digest,
            "failure_digest": retry.failure_digest,
            "reservation_groups": [list(group) for group in retry.reservation_groups],
            "strategy": retry.strategy,
        }
    outcomes = []
    for outcome in batch.outcomes:
        qualification = outcome.settlement_qualification
        outcomes.append(
            {
                "attempt_artifact_sha256": outcome.attempt_artifact_sha256,
                "authority_manifest_digest": outcome.authority_manifest_digest,
                "decision": outcome.decision.value,
                "failure_digest": outcome.failure_digest,
                "reason": outcome.reason,
                "report_digest": outcome.report_digest,
                "reservation_digest": outcome.reservation_digest,
                "retryable": outcome.retryable,
                "selected_delta_digest": outcome.selected_delta_digest,
                "settlement_qualification": (
                    None if qualification is None else qualification.to_dict()
                ),
            }
        )
    return canonical_digest(
        "cacheon.chain.qualification-result-payload.v1",
        {
            "attempt_ref": (
                None if batch.attempt_ref is None else batch.attempt_ref.to_dict()
            ),
            "authority_manifest_digest": batch.authority_manifest_digest,
            "outcomes": outcomes,
            "retry_plan": retry_value,
        },
    )


def _typed_payload(payload: object) -> tuple[str, str]:
    if type(payload) is ArenaScreenReceipt:
        return "arena_screen_receipt", payload.digest
    if type(payload) is QualificationIntakeBatch:
        return "qualification_intake_batch", qualification_batch_payload_digest(payload)
    raise EvaluationCoordinatorError("evaluation result payload is not exactly typed")


@dataclass(frozen=True)
class EvaluationResultEnvelope:
    """Closed result identity bound to one exact lease and READY epoch."""

    lease_id: str
    generation: int
    stage: str
    owner: str
    members: tuple[EvaluationLeaseMember, ...]
    ready_receipt_digest: str
    ready_epoch: int
    worker_readiness_digest: str
    service_identity: str
    payload_kind: str
    payload_digest: str
    schema_version: int = _RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "lease_id",
            "ready_receipt_digest",
            "worker_readiness_digest",
            "payload_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        try:
            require_evaluation_owner(self.owner)
        except Exception as exc:
            raise EvaluationCoordinatorError(str(exc)) from None
        members = tuple(self.members)
        if (
            type(self.generation) is not int
            or self.generation <= 0
            or self.stage not in {"screen", "qualification"}
            or not members
            or any(type(row) is not EvaluationLeaseMember for row in members)
            or len({row.reservation_id for row in members}) != len(members)
            or type(self.ready_epoch) is not int
            or self.ready_epoch < 0
            or not isinstance(self.service_identity, str)
            or not self.service_identity
            or self.payload_kind
            not in {"arena_screen_receipt", "qualification_intake_batch"}
            or type(self.schema_version) is not int
            or self.schema_version != _RESULT_SCHEMA_VERSION
            or (self.stage == "screen")
            != (self.payload_kind == "arena_screen_receipt")
        ):
            raise EvaluationCoordinatorError("evaluation result envelope is malformed")
        object.__setattr__(self, "members", members)

    @classmethod
    def seal(
        cls,
        lease: EvaluationLease,
        readiness: WorkerReadiness,
        service: ArenaService,
        payload: object,
    ) -> "EvaluationResultEnvelope":
        if type(lease) is not EvaluationLease:
            raise EvaluationCoordinatorError("result lease is not exactly typed")
        if type(readiness) is not WorkerReadiness:
            raise EvaluationCoordinatorError("result readiness is not exactly typed")
        readiness.validate(service)
        payload_kind, payload_digest = _typed_payload(payload)
        return cls(
            lease_id=lease.lease_id,
            generation=lease.generation,
            stage=lease.stage,
            owner=lease.owner,
            members=lease.members,
            ready_receipt_digest=readiness.ready_receipt_digest,
            ready_epoch=readiness.ready_epoch,
            worker_readiness_digest=readiness.digest,
            service_identity=service.manifest.service_id,
            payload_kind=payload_kind,
            payload_digest=payload_digest,
        )

    def verify(
        self,
        lease: EvaluationLease,
        readiness: WorkerReadiness,
        service: ArenaService,
        payload: object,
    ) -> None:
        expected = EvaluationResultEnvelope.seal(lease, readiness, service, payload)
        if self != expected:
            raise EvaluationCoordinatorError(
                "evaluation result envelope differs from the exact live lease"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "lease_id": self.lease_id,
            "members": [member.to_dict() for member in self.members],
            "owner": self.owner,
            "payload_digest": self.payload_digest,
            "payload_kind": self.payload_kind,
            "ready_epoch": self.ready_epoch,
            "ready_receipt_digest": self.ready_receipt_digest,
            "schema_version": self.schema_version,
            "service_identity": self.service_identity,
            "stage": self.stage,
            "worker_readiness_digest": self.worker_readiness_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.evaluation-result-envelope.v1",
            self.to_dict(),
        )


def _qualification_reservations(
    reservations: tuple[IntakeReservation, ...],
    publications: tuple[WorkerBundlePublication, ...],
) -> tuple[QualificationReservation, ...]:
    if len(reservations) != len(publications):
        raise EvaluationCoordinatorError("qualification publication coverage differs")
    result = []
    for position, (reservation, publication) in enumerate(
        zip(reservations, publications, strict=True)
    ):
        fingerprint = reservation.delta_fingerprint
        if (
            fingerprint is None
            or reservation.publication_digest != publication.digest
            or reservation.arrival.content_hash != publication.content_hash
            or reservation.target_id != fingerprint.target_id
            or reservation.target_members != fingerprint.members
        ):
            raise EvaluationCoordinatorError("evaluation intake provenance differs")
        result.append(
            QualificationReservation(
                reservation.reservation_id,
                publication.digest,
                fingerprint.target_id,
                fingerprint.selected_delta_digest,
                position,
                reservation.arrival.hotkey,
                reservation.arrival.block,
                reservation.arrival.event_index,
                reservation.arrival.event_subindex,
                reservation.target_members,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class ClaimedScreenEvaluation:
    lease: EvaluationLease
    reservation: IntakeReservation
    publication: WorkerBundlePublication
    candidate: ArenaCandidateBinding

    def __post_init__(self) -> None:
        if (
            type(self.lease) is not EvaluationLease
            or self.lease.stage != "screen"
            or len(self.lease.members) != 1
            or type(self.reservation) is not IntakeReservation
            or type(self.publication) is not WorkerBundlePublication
            or type(self.candidate) is not ArenaCandidateBinding
            or self.lease.reservation_ids != (self.reservation.reservation_id,)
            or self.candidate.reservation.reservation_digest
            != self.reservation.reservation_id
            or self.candidate.publication != self.publication
            or self.candidate.screen_attempt != self.reservation.screen_attempts + 1
        ):
            raise EvaluationCoordinatorError("claimed screen work is inconsistent")


@dataclass(frozen=True)
class ClaimedQualificationEvaluation:
    lease: EvaluationLease
    reservations: tuple[IntakeReservation, ...]
    publications: tuple[WorkerBundlePublication, ...]
    candidates: tuple[ArenaCandidateBinding, ...]
    screen_receipts: tuple[ArenaScreenReceipt, ...]

    def __post_init__(self) -> None:
        reservations = tuple(self.reservations)
        publications = tuple(self.publications)
        candidates = tuple(self.candidates)
        receipts = tuple(self.screen_receipts)
        lanes = {row.screen_lane for row in reservations}
        if (
            type(self.lease) is not EvaluationLease
            or self.lease.stage != "qualification"
            or not reservations
            or any(type(row) is not IntakeReservation for row in reservations)
            or any(type(row) is not WorkerBundlePublication for row in publications)
            or any(type(row) is not ArenaCandidateBinding for row in candidates)
            or any(type(row) is not ArenaScreenReceipt for row in receipts)
            or not (
                len(reservations)
                == len(publications)
                == len(candidates)
                == len(receipts)
            )
            or self.lease.reservation_ids
            != tuple(row.reservation_id for row in reservations)
            or tuple(row.publication for row in candidates) != publications
            or tuple(row.reservation.reservation_digest for row in candidates)
            != self.lease.reservation_ids
            or tuple(row.candidate_digest for row in receipts)
            != tuple(row.digest for row in candidates)
            or lanes not in ({"primary"}, {"reproduction"})
            or (lanes == {"reproduction"} and len(reservations) != 1)
        ):
            raise EvaluationCoordinatorError("claimed qualification work is inconsistent")
        object.__setattr__(self, "reservations", reservations)
        object.__setattr__(self, "publications", publications)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "screen_receipts", receipts)

    @property
    def screen_lane(self) -> str:
        """Return the exact uniform primary or reproduction intake lane."""

        return self.reservations[0].screen_lane


@dataclass(frozen=True)
class EvaluationRun:
    """One completed or systemically released worker invocation."""

    lease: EvaluationLease
    envelope: EvaluationResultEnvelope
    payload: ArenaScreenReceipt | QualificationIntakeBatch
    disposition: str

    def __post_init__(self) -> None:
        if (
            type(self.lease) is not EvaluationLease
            or type(self.envelope) is not EvaluationResultEnvelope
            or type(self.payload) not in {ArenaScreenReceipt, QualificationIntakeBatch}
            or self.disposition not in {"completed", "released"}
        ):
            raise EvaluationCoordinatorError("evaluation run result is malformed")


class _HeartbeatCancelled(RuntimeError):
    pass


class _LeaseHeartbeat:
    """Serial CAS heartbeat owner for one synchronous provider invocation."""

    def __init__(self, coordinator: "EvaluationCoordinator", lease: EvaluationLease):
        self._coordinator = coordinator
        self._lease = lease
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"cacheon-{lease.stage}-heartbeat-{lease.lease_id[:12]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        interval = self._coordinator.heartbeat_interval_s
        while not self._stop.wait(interval):
            with self._lock:
                lease = self._lease
            try:
                extended = self._coordinator._heartbeat_once(lease, self._stop)
            except _HeartbeatCancelled:
                return
            except _TransientCoordinatorOwnership:
                # The independently running intake controller legitimately owns
                # the flock during a pass.  A single collision must not
                # permanently kill a multi-minute lease heartbeat; retry promptly
                # until ownership becomes available or the remote call ends.
                # Expired/stale/CAS failures still surface below as fatal errors.
                interval = min(
                    self._coordinator.heartbeat_interval_s,
                    max(0.05, self._coordinator.lock_retry_delay_s),
                )
                continue
            except EvaluationCoordinatorError as exc:
                with self._lock:
                    self._error = exc
                return
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                return
            interval = self._coordinator.heartbeat_interval_s
            with self._lock:
                self._lease = extended

    def stop(self) -> tuple[EvaluationLease, BaseException | None]:
        self._stop.set()
        self._thread.join(self._coordinator.heartbeat_join_timeout_s)
        with self._lock:
            lease, error = self._lease, self._error
        if self._thread.is_alive() and error is None:
            error = EvaluationCoordinatorError(
                "evaluation heartbeat did not stop within its bounded join"
            )
        return lease, error


class EvaluationCoordinator:
    """Claim, run, and CAS-commit one CPU-dispatched evaluation at a time."""

    def __init__(
        self,
        *,
        intake_db: str | Path,
        policy: IntakePolicy,
        scope: IntakeScope,
        service: ArenaService,
        readiness: WorkerReadiness,
        owner: str,
        advance_finalized_cursor: Callable[[], tuple[int, str]],
        lease_blocks: int = 30,
        qualification_max_members: int | None = None,
        heartbeat_interval_s: float = 5.0,
        heartbeat_join_timeout_s: float = 5.0,
        lock_attempts: int = 3,
        lock_retry_delay_s: float = 0.05,
        store_factory: Callable[..., FinalizedIntakeStore] = FinalizedIntakeStore,
    ):
        if type(policy) is not IntakePolicy or type(scope) is not IntakeScope:
            raise EvaluationCoordinatorError("coordinator store authority is not exact")
        if type(service) is not ArenaService or type(readiness) is not WorkerReadiness:
            raise EvaluationCoordinatorError("coordinator arena authority is not exact")
        try:
            require_evaluation_owner(owner)
        except Exception as exc:
            raise EvaluationCoordinatorError(str(exc)) from None
        if not callable(advance_finalized_cursor) or not callable(store_factory):
            raise EvaluationCoordinatorError("coordinator callbacks are not callable")
        if (
            type(lease_blocks) is not int
            or not 0 < lease_blocks <= policy.expiry_blocks
            or type(lock_attempts) is not int
            or lock_attempts <= 0
            or isinstance(heartbeat_interval_s, bool)
            or not isinstance(heartbeat_interval_s, (int, float))
            or not math.isfinite(float(heartbeat_interval_s))
            or heartbeat_interval_s <= 0
            or isinstance(heartbeat_join_timeout_s, bool)
            or not isinstance(heartbeat_join_timeout_s, (int, float))
            or not math.isfinite(float(heartbeat_join_timeout_s))
            or heartbeat_join_timeout_s <= 0
            or isinstance(lock_retry_delay_s, bool)
            or not isinstance(lock_retry_delay_s, (int, float))
            or not math.isfinite(float(lock_retry_delay_s))
            or lock_retry_delay_s < 0
        ):
            raise EvaluationCoordinatorError("coordinator timing bounds are malformed")
        maximum = (
            min(policy.max_cohort, service.manifest.capacity.max_cohort_size)
            if qualification_max_members is None
            else qualification_max_members
        )
        if (
            type(maximum) is not int
            or maximum <= 0
            or maximum > policy.max_cohort
            or maximum > service.manifest.capacity.max_cohort_size
        ):
            raise EvaluationCoordinatorError("qualification cohort bound is malformed")
        self.intake_db = Path(intake_db)
        self.policy = policy
        self.scope = scope
        self.service = service
        self.readiness = readiness
        self.owner = owner
        self.advance_finalized_cursor = advance_finalized_cursor
        self.lease_blocks = lease_blocks
        self.qualification_max_members = maximum
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.heartbeat_join_timeout_s = float(heartbeat_join_timeout_s)
        self.lock_attempts = lock_attempts
        self.lock_retry_delay_s = float(lock_retry_delay_s)
        self._store_factory = store_factory

    def _open_at_durable_cursor(
        self,
        cancel: threading.Event | None = None,
    ) -> tuple[FinalizedIntakeStore, tuple[int, str]]:
        """Advance outside flock, then briefly acquire and verify exact cursor."""

        last_error: BaseException | None = None
        for attempt in range(self.lock_attempts):
            point = _closed_cursor(self.advance_finalized_cursor())
            if cancel is not None and cancel.is_set():
                raise _HeartbeatCancelled()
            try:
                store = self._store_factory(
                    self.intake_db,
                    self.policy,
                    scope=self.scope,
                )
            except IntakeError as exc:
                if not is_lock_collision(exc):
                    raise EvaluationCoordinatorError(
                        f"evaluation store cannot open: {exc}"
                    ) from exc
                last_error = exc
            else:
                try:
                    observed = store.finalized_cursor()
                except IntakeError as exc:
                    store.close()
                    raise EvaluationCoordinatorError(
                        f"durable finalized cursor cannot reopen: {exc}"
                    ) from exc
                if observed == point:
                    return store, point
                store.close()
                last_error = EvaluationCoordinatorError(
                    "finalized cursor changed before coordinator ownership"
                )
            if cancel is not None and cancel.is_set():
                raise _HeartbeatCancelled()
            if attempt + 1 < self.lock_attempts and self.lock_retry_delay_s:
                time.sleep(self.lock_retry_delay_s)
        raise _TransientCoordinatorOwnership(
            _TRANSIENT_COORDINATOR_OWNERSHIP_MESSAGE
        ) from last_error

    def _heartbeat_once(
        self,
        lease: EvaluationLease,
        cancel: threading.Event,
    ) -> EvaluationLease:
        store, point = self._open_at_durable_cursor(cancel)
        try:
            if cancel.is_set():
                raise _HeartbeatCancelled()
            if point[0] + self.lease_blocks <= lease.expires_block:
                return lease
            return store.heartbeat_evaluation_lease(
                lease,
                current_block=point[0],
                lease_blocks=self.lease_blocks,
            )
        except IntakeError as exc:
            raise EvaluationCoordinatorError(
                f"evaluation heartbeat failed closed: {exc}"
            ) from exc
        finally:
            store.close()

    def _release(
        self,
        lease: EvaluationLease,
        *,
        reason: str,
        result_digest: str = "",
    ) -> None:
        store, point = self._open_at_durable_cursor()
        try:
            store.release_evaluation_lease(
                lease,
                current_block=point[0],
                reason=reason,
                result_digest=result_digest,
            )
        except IntakeError as exc:
            raise EvaluationCoordinatorError(
                f"evaluation lease could not be safely released: {exc}"
            ) from exc
        finally:
            store.close()

    def _release_after_error(
        self,
        lease: EvaluationLease,
        *,
        reason: str,
        cause: BaseException,
        result_digest: str = "",
    ) -> NoReturn:
        try:
            self._release(lease, reason=reason, result_digest=result_digest)
        except BaseException as release_error:
            raise EvaluationCoordinatorError(
                f"{reason}; lease release also failed closed: {release_error}"
            ) from cause
        raise EvaluationCoordinatorError(reason) from cause

    def claim_screen(self, *, expected_members: tuple[EvaluationLeaseMember, ...] | None = None) -> ClaimedScreenEvaluation | None:
        """Claim the exact oldest screen singleton, then reopen bytes unlocked."""
        self.readiness.validate(self.service)
        store, point = self._open_at_durable_cursor()
        lease: EvaluationLease | None = None
        claim_error: IntakeError | None = None
        try:
            lease = store.claim_evaluation_lease(
                stage="screen",
                owner=self.owner,
                current_block=point[0],
                lease_blocks=self.lease_blocks,
                expected_members=expected_members,
            )
            if lease is None:
                return None
            reservation = store.get(lease.reservation_ids[0])
        except IntakeError as exc:
            if lease is None:
                raise EvaluationCoordinatorError(f"screen claim failed: {exc}") from exc
            claim_error = exc
        finally:
            store.close()
        if claim_error is not None:
            self._release_after_error(
                lease,
                reason="screen_claim_snapshot",
                cause=claim_error,
            )
        try:
            publication = reopen_worker_bundle(
                reservation.publication_root,
                reservation.arrival.content_hash,
                expected_receipt_digest=reservation.publication_digest,
            )
            authority = _qualification_reservations((reservation,), (publication,))[0]
            candidate = ArenaCandidateBinding(
                authority,
                publication,
                reservation.screen_attempts + 1,
            )
            return ClaimedScreenEvaluation(
                lease,
                reservation,
                publication,
                candidate,
            )
        except BaseException as exc:
            self._release_after_error(
                lease,
                reason="screen_claim_materialization",
                cause=exc,
            )

    def claim_qualification(self) -> ClaimedQualificationEvaluation | None:
        """Claim one exact promoted cohort and gather all retained CPU authority."""

        self.readiness.validate(self.service)
        store, point = self._open_at_durable_cursor()
        lease: EvaluationLease | None = None
        claim_error: IntakeError | None = None
        try:
            lease = store.claim_evaluation_lease(
                stage="qualification",
                owner=self.owner,
                current_block=point[0],
                lease_blocks=self.lease_blocks,
                max_members=self.qualification_max_members,
            )
            if lease is None:
                return None
            reservations = tuple(store.get(row) for row in lease.reservation_ids)
            receipts = tuple(
                store.latest_promoted_screen(row.reservation_id)
                for row in reservations
            )
        except IntakeError as exc:
            if lease is None:
                raise EvaluationCoordinatorError(
                    f"qualification claim failed: {exc}"
                ) from exc
            claim_error = exc
        finally:
            store.close()
        if claim_error is not None:
            self._release_after_error(
                lease,
                reason="qualification_claim_snapshot",
                cause=claim_error,
            )
        try:
            publications = tuple(
                reopen_worker_bundle(
                    row.publication_root,
                    row.arrival.content_hash,
                    expected_receipt_digest=row.publication_digest,
                )
                for row in reservations
            )
            authority = _qualification_reservations(reservations, publications)
            candidates = tuple(
                ArenaCandidateBinding(item, publication, row.screen_attempts)
                for row, publication, item in zip(
                    reservations, publications, authority, strict=True
                )
            )
            return ClaimedQualificationEvaluation(
                lease,
                reservations,
                publications,
                candidates,
                receipts,
            )
        except BaseException as exc:
            self._release_after_error(
                lease,
                reason="qualification_claim_materialization",
                cause=exc,
            )

    def commit_screen_result(
        self,
        claim: ClaimedScreenEvaluation,
        receipt: ArenaScreenReceipt,
        envelope: EvaluationResultEnvelope,
    ) -> IntakeReservation:
        """CAS-commit only begin-screen plus the exact typed screen receipt."""

        if type(claim) is not ClaimedScreenEvaluation:
            raise EvaluationCoordinatorError("screen claim is not exactly typed")
        if type(receipt) is not ArenaScreenReceipt:
            raise EvaluationCoordinatorError("screen receipt is not exactly typed")
        envelope.verify(claim.lease, self.readiness, self.service, receipt)
        if (
            receipt.service_digest != self.service.identity
            or receipt.candidate_digest != claim.candidate.digest
            or receipt.screen_attempt != claim.candidate.screen_attempt
        ):
            raise EvaluationCoordinatorError("screen result changed claimed authority")
        store, point = self._open_at_durable_cursor()
        try:
            with store.accept_evaluation_result(
                claim.lease,
                current_block=point[0],
                result_digest=envelope.digest,
            ) as rows:
                if rows != (claim.reservation,):
                    raise EvaluationCoordinatorError(
                        "screen queue row changed before result commit"
                    )
                active = store.begin_screen(
                    claim.reservation.reservation_id,
                    service_digest=self.service.identity,
                )
                if active.screen_attempts != claim.candidate.screen_attempt:
                    raise EvaluationCoordinatorError(
                        "screen attempt changed before result commit"
                    )
                result = store.apply_screen_receipt(
                    active.reservation_id,
                    candidate_digest=claim.candidate.digest,
                    receipt=receipt,
                )
            return result
        except IntakeError as exc:
            raise EvaluationCoordinatorError(
                f"screen result was rejected by the durable lease: {exc}"
            ) from exc
        finally:
            store.close()

    @staticmethod
    def _validate_work(
        work: ArenaQualificationWork,
        claim: ClaimedQualificationEvaluation,
    ) -> None:
        expected = tuple(row.reservation for row in claim.candidates)
        if (
            type(work) is not ArenaQualificationWork
            or type(work.factory) is not QualificationPlanFactory
            or type(work.factory.manifest) is not QualificationAuthorityManifest
            or work.factory.manifest.reservations != expected
        ):
            raise EvaluationCoordinatorError(
                "qualification planner changed the exact finalized cohort"
            )

    @staticmethod
    def _validate_batch(
        batch: QualificationIntakeBatch,
        work: ArenaQualificationWork,
        claim: ClaimedQualificationEvaluation,
    ) -> None:
        expected = tuple(row.reservation for row in claim.candidates)
        if (
            type(batch) is not QualificationIntakeBatch
            or batch.authority_manifest_digest != work.factory.manifest.digest
            or tuple(row.reservation_digest for row in batch.outcomes)
            != tuple(row.reservation_digest for row in expected)
            or tuple(row.selected_delta_digest for row in batch.outcomes)
            != tuple(row.selected_delta_digest for row in expected)
        ):
            raise EvaluationCoordinatorError(
                "qualification result changed the exact finalized cohort"
            )

    @staticmethod
    def _systemic(batch: QualificationIntakeBatch) -> bool:
        return bool(batch.outcomes) and all(
            outcome.decision is QualificationDecision.NO_DECISION
            and outcome.reason in SYSTEMIC_QUALIFICATION_REASONS
            for outcome in batch.outcomes
        )

    @staticmethod
    def _stack_authority(prepared) -> tuple[object, str, Path]:
        arms = tuple(row.arm for row in prepared.prepared.candidates)
        if (
            not arms
            or len({row.baseline_before for row in arms}) != 1
            or any(row.incumbent != arms[0].incumbent for row in arms)
        ):
            raise EvaluationCoordinatorError(
                "qualification planner has no single incumbent stack"
            )
        return (
            arms[0].incumbent,
            arms[0].baseline_before.tree_digest,
            Path(prepared.evidence_root),
        )

    def commit_remote_qualification_result(
        self,
        claim: ClaimedQualificationEvaluation,
        *,
        authority_manifest: QualificationAuthorityManifest,
        incumbent_stack: EvaluationStackManifest,
        incumbent_tree_digest: str,
        batch: QualificationIntakeBatch,
        envelope: EvaluationResultEnvelope,
        evidence_root: str | Path,
        evidence_inventory: tuple[EvidenceArtifactRef, ...],
    ) -> tuple[IntakeReservation, ...]:
        """CAS-commit one imported, path-free worker qualification product.

        Every evidence byte has already crossed the authenticated transport and
        been published into ``evidence_root`` by the CPU.  This method reopens
        the complete inventory itself before acquiring the intake controller,
        then atomically installs the exact public authority and applies the
        batch under the live lease.  No provider plan or pod filesystem path is
        consulted.
        """

        if type(claim) is not ClaimedQualificationEvaluation:
            raise EvaluationCoordinatorError("qualification claim is not exactly typed")
        if (
            type(authority_manifest) is not QualificationAuthorityManifest
            or type(incumbent_stack) is not EvaluationStackManifest
            or type(batch) is not QualificationIntakeBatch
            or type(envelope) is not EvaluationResultEnvelope
        ):
            raise EvaluationCoordinatorError(
                "remote qualification authority is not exactly typed"
            )
        expected = tuple(row.reservation for row in claim.candidates)
        if authority_manifest.reservations != expected:
            raise EvaluationCoordinatorError(
                "remote qualification authority changed the exact finalized cohort"
            )
        if (
            batch.authority_manifest_digest != authority_manifest.digest
            or tuple(row.reservation_digest for row in batch.outcomes)
            != tuple(row.reservation_digest for row in expected)
            or tuple(row.selected_delta_digest for row in batch.outcomes)
            != tuple(row.selected_delta_digest for row in expected)
        ):
            raise EvaluationCoordinatorError(
                "qualification result changed the exact finalized cohort"
            )
        envelope.verify(claim.lease, self.readiness, self.service, batch)
        tree_digest = _digest(incumbent_tree_digest, "incumbent_tree_digest")
        runtime = self.service.manifest.runtime
        if (
            incumbent_stack.runtime_digest != runtime.runtime_digest
            or incumbent_stack.base_engine_digest != runtime.base_engine_digest
            or incumbent_stack.arena_digest != self.service.identity
        ):
            raise EvaluationCoordinatorError(
                "remote qualification incumbent differs from the CPU service"
            )
        root = Path(evidence_root)
        if (
            not root.is_absolute()
            or root != Path(os.path.normpath(root))
        ):
            raise EvaluationCoordinatorError(
                "remote qualification evidence root is not canonical and absolute"
            )
        inventory = tuple(evidence_inventory)
        if (
            type(evidence_inventory) is not tuple
            or len(inventory) > _REMOTE_EVIDENCE_ARTIFACTS_LIMIT
            or any(type(row) is not EvidenceArtifactRef for row in inventory)
            or len(set(inventory)) != len(inventory)
            or len({row.sha256 for row in inventory}) != len(inventory)
            or sum(row.size for row in inventory) > _REMOTE_EVIDENCE_TOTAL_LIMIT
            or (
                batch.attempt_ref is not None
                and batch.attempt_ref not in inventory
            )
        ):
            raise EvaluationCoordinatorError(
                "remote qualification evidence inventory is incomplete or duplicated"
            )
        try:
            for reference in inventory:
                reopen_evidence(
                    root,
                    reference,
                    max_bytes=_REMOTE_EVIDENCE_ARTIFACT_LIMIT,
                )
        except EvidenceStoreError as exc:
            raise EvaluationCoordinatorError(
                f"remote qualification evidence cannot reopen from the CPU CAS: {exc}"
            ) from exc
        store, point = self._open_at_durable_cursor()
        try:
            with store.accept_evaluation_result(
                claim.lease,
                current_block=point[0],
                result_digest=envelope.digest,
            ) as rows:
                if rows != claim.reservations:
                    raise EvaluationCoordinatorError(
                        "qualification cohort changed before result commit"
                    )
                store.initialize_evaluation_stack(
                    incumbent_stack,
                    tree_digest=tree_digest,
                )
                authority_digest = authority_manifest.digest
                authority_value = authority_manifest.to_dict()
                for row in rows:
                    store.mark_qualifying(
                        row.reservation_id,
                        authority_digest,
                        authority_value,
                    )
                result = store.apply_qualification_batch(
                    batch,
                    current_finalized_block=point[0],
                    evidence_root=root,
                )
            return result
        except IntakeError as exc:
            raise EvaluationCoordinatorError(
                f"qualification result was rejected by the durable lease: {exc}"
            ) from exc
        finally:
            store.close()

    def commit_qualification_result(
        self,
        claim: ClaimedQualificationEvaluation,
        work: ArenaQualificationWork,
        prepared,
        batch: QualificationIntakeBatch,
        envelope: EvaluationResultEnvelope,
    ) -> tuple[IntakeReservation, ...]:
        """Atomically install stack authority and apply one exact typed cohort."""

        if type(claim) is not ClaimedQualificationEvaluation:
            raise EvaluationCoordinatorError("qualification claim is not exactly typed")
        self._validate_work(work, claim)
        self._validate_batch(batch, work, claim)
        envelope.verify(claim.lease, self.readiness, self.service, batch)
        incumbent, tree_digest, evidence_root = self._stack_authority(prepared)
        authority_digest = work.factory.manifest.digest
        authority_manifest = work.factory.manifest.to_dict()
        store, point = self._open_at_durable_cursor()
        try:
            with store.accept_evaluation_result(
                claim.lease,
                current_block=point[0],
                result_digest=envelope.digest,
            ) as rows:
                if rows != claim.reservations:
                    raise EvaluationCoordinatorError(
                        "qualification cohort changed before result commit"
                    )
                store.initialize_evaluation_stack(
                    incumbent,
                    tree_digest=tree_digest,
                )
                for row in rows:
                    store.mark_qualifying(
                        row.reservation_id,
                        authority_digest,
                        authority_manifest,
                    )
                result = store.apply_qualification_batch(
                    batch,
                    current_finalized_block=point[0],
                    evidence_root=evidence_root,
                )
            return result
        except IntakeError as exc:
            raise EvaluationCoordinatorError(
                f"qualification result was rejected by the durable lease: {exc}"
            ) from exc
        finally:
            store.close()

    def run_screen_once(self) -> EvaluationRun | None:
        """Run one screen outside controller ownership and commit its envelope."""

        claim = self.claim_screen()
        if claim is None:
            return None
        heartbeat = _LeaseHeartbeat(self, claim.lease)
        heartbeat.start()
        try:
            receipt = self.service.screen(claim.candidate)
        except BaseException as exc:
            lease, heartbeat_error = heartbeat.stop()
            cause = heartbeat_error or exc
            self._release_after_error(
                lease,
                reason="screen_provider_exception",
                cause=cause,
            )
        lease, heartbeat_error = heartbeat.stop()
        claim = replace(claim, lease=lease)
        envelope = EvaluationResultEnvelope.seal(
            lease,
            self.readiness,
            self.service,
            receipt,
        )
        if heartbeat_error is not None:
            self._release_after_error(
                lease,
                reason="screen_heartbeat_failure",
                cause=heartbeat_error,
                result_digest=envelope.digest,
            )
        self.commit_screen_result(claim, receipt, envelope)
        return EvaluationRun(lease, envelope, receipt, "completed")

    def run_qualification_once(self) -> EvaluationRun | None:
        """Plan and run one exact cohort unlocked, then commit or requeue it."""

        claim = self.claim_qualification()
        if claim is None:
            return None
        heartbeat = _LeaseHeartbeat(self, claim.lease)
        heartbeat.start()
        prepared = None
        try:
            # ``state=None`` is intentional: an open controller/store is never
            # delegated to the provider.  All finalized rows and screen receipts
            # needed by the request were snapshotted before the flock was closed.
            work = self.service.plan_qualification(
                claim.candidates,
                claim.screen_receipts,
                state=None,
            )
            self._validate_work(work, claim)
            try:
                prepared = work.factory.build()
            except BaseException:
                # The typed intake runner converts supported plan failures into
                # a manifest-bound ``qualification_plan`` NO_DECISION batch.
                prepared = None
            batch = run_qualification_intake(
                work.factory,
                executor=work.executor,
                resident_baseline_executor=work.resident_baseline_executor,
                entropy_provider=work.entropy_provider,
                hidden_judge=work.hidden_judge,
                deadline=float(work.deadline),
            )
            self._validate_batch(batch, work, claim)
            if prepared is None and not self._systemic(batch):
                prepared = work.factory.build()
        except BaseException as exc:
            lease, heartbeat_error = heartbeat.stop()
            cause = heartbeat_error or exc
            self._release_after_error(
                lease,
                reason="qualification_provider_exception",
                cause=cause,
            )
        lease, heartbeat_error = heartbeat.stop()
        claim = replace(claim, lease=lease)
        envelope = EvaluationResultEnvelope.seal(
            lease,
            self.readiness,
            self.service,
            batch,
        )
        if heartbeat_error is not None:
            self._release_after_error(
                lease,
                reason="qualification_heartbeat_failure",
                cause=heartbeat_error,
                result_digest=envelope.digest,
            )
        if self._systemic(batch):
            reasons = ",".join(sorted({row.reason for row in batch.outcomes}))
            self._release(
                lease,
                reason=f"systemic_qualification:{reasons}",
                result_digest=envelope.digest,
            )
            return EvaluationRun(lease, envelope, batch, "released")
        if prepared is None:
            self._release_after_error(
                lease,
                reason="qualification_stack_authority_missing",
                cause=EvaluationCoordinatorError(
                    "qualification completed without a rebuildable stack authority"
                ),
                result_digest=envelope.digest,
            )
        self.commit_qualification_result(claim, work, prepared, batch, envelope)
        return EvaluationRun(lease, envelope, batch, "completed")


__all__ = [
    "ClaimedQualificationEvaluation",
    "ClaimedScreenEvaluation",
    "EvaluationCoordinator",
    "EvaluationCoordinatorError",
    "EvaluationResultEnvelope",
    "EvaluationRun",
    "SYSTEMIC_QUALIFICATION_REASONS",
    "WorkerReadiness",
    "qualification_batch_payload_digest",
]

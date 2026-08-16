"""Restart-safe CPU dispatch for one exact remote qualification request.

This dispatcher is deliberately separate from the legacy remote dispatcher.
It never calls ``run_qualification()``, never creates a second carrier after a
plan is retained, and never generically releases post-publication work.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.chain.evaluation_coordinator import (
    ClaimedQualificationEvaluation,
    EvaluationCoordinator,
    EvaluationResultEnvelope,
    EvaluationRun,
    _qualification_reservations,
)
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.evaluation_lease_store import (
    EvaluationClaimConflict,
    _closed_expected_members,
)
from cacheon.chain.evaluation_recovery import (
    EvaluationRecovery,
    EvaluationRecoveryHoldError,
    RecoveryPhase,
)
from cacheon.chain.guarded_evaluation_run import GuardedEvaluationRun
from cacheon.chain.execution_disposition import (
    AUTHORITY_CHANGED_HOLD_REASON,
    AuthenticatedPreResidentRefusal,
    COMPLETED_NO_DECISION_HOLD_REASON,
    ExecutionDisposition,
    ExecutionOutcome,
    ORPHANED_CARRIER_HOLD_REASON,
    PRE_RESIDENT_REQUEUE_FAILURES,
    WORKER_INFRASTRUCTURE_HOLD_REASON,
    WORKER_INFRASTRUCTURE_REQUEUE_FAILURE,
    resolve_infrastructure_result,
)
from cacheon.chain.intake import IntakeError
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.screen_identity_rotation import (
    ScreenIdentityRotationError,
    release_rotated_cohort,
    rotated_reservation_ids,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    AuthenticatedRemoteEvaluationResponse,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    _request_body_for_qualification,
    reopen_remote_response,
    seal_remote_request,
)
from cacheon.chain.remote_qualification_evidence import (
    RemoteQualificationProduct,
    import_remote_qualification_evidence,
)
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldProduct,
    durable_remote_qualification_hold_reason,
)
from cacheon.chain.remote_worker_request_plan import (
    PlannedQualificationObservation,
    QualificationPrepublicationProof,
    QualificationRecoveryHold,
    QualificationRequestPlan,
)
from cacheon.chain.publication import reopen_worker_bundle
from cacheon.chain.qualification_hold_requeue import QualificationHoldRequeuePolicy
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import QualificationIntakeBatch
from cacheon.stack_identity import require_sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest


class RecoverableQualificationDispatcherError(RuntimeError):
    """Qualification orchestration cannot safely advance its retained state."""


class RecoverableQualificationTransport(Protocol):
    identity: RemoteWorkerTransportIdentity

    def plan_qualification_request(
        self, request: RemoteEvaluationRequest
    ) -> QualificationRequestPlan: ...

    def materialize_planned_qualification(
        self, plan: QualificationRequestPlan, request: RemoteEvaluationRequest
    ) -> PlannedQualificationObservation: ...

    def inspect_planned_qualification(
        self, plan: QualificationRequestPlan
    ) -> PlannedQualificationObservation: ...

    def prove_planned_qualification_prepublication(
        self, plan: QualificationRequestPlan
    ) -> QualificationPrepublicationProof: ...

    def publish_planned_qualification(
        self, plan: QualificationRequestPlan
    ) -> PlannedQualificationObservation: ...

    def resume_planned_qualification(
        self, plan: QualificationRequestPlan
    ) -> AuthenticatedRemoteEvaluationResponse: ...


@dataclass(frozen=True)
class RecoverableQualificationHold:
    """One durable HOLD; no new lease, carrier, or experiment is permitted."""

    recovery_id: str
    phase: RecoveryPhase
    request_id: str
    reason: str

    def __post_init__(self) -> None:
        require_sha256_hex(self.recovery_id, field="held recovery id")
        if type(self.phase) is not RecoveryPhase or self.phase is not RecoveryPhase.HELD:
            raise RecoverableQualificationDispatcherError(
                "qualification hold phase is malformed"
            )
        if self.request_id:
            require_sha256_hex(self.request_id, field="held request id")
        if (
            not isinstance(self.reason, str)
            or not self.reason
            or self.reason.strip() != self.reason
            or len(self.reason) > 2_048
        ):
            raise RecoverableQualificationDispatcherError(
                "qualification hold reason is malformed"
            )


@dataclass(frozen=True)
class CompletedQualificationHold:
    """Authenticated terminal HOLD that released the qualification lane."""

    recovery_id: str
    request_id: str
    lease: EvaluationLease
    reason: str
    result_digest: str

    def __post_init__(self) -> None:
        require_sha256_hex(self.recovery_id, field="completed HOLD recovery id")
        require_sha256_hex(self.request_id, field="completed HOLD request id")
        require_sha256_hex(self.result_digest, field="completed HOLD result digest")
        if (
            type(self.lease) is not EvaluationLease
            or self.lease.stage != "qualification"
            or not self.reason.startswith("remote_qualification_hold:")
        ):
            raise RecoverableQualificationDispatcherError(
                "completed qualification HOLD is malformed"
            )


@dataclass(frozen=True)
class RecoverableQualificationRequeue:
    """One typed NO_DECISION + REQUEUE from an authenticated pre-resident
    refusal or from an unproven worker infrastructure result."""

    recovery_id: str
    request_id: str
    outcome: ExecutionOutcome

    def __post_init__(self) -> None:
        require_sha256_hex(self.recovery_id, field="requeued recovery id")
        require_sha256_hex(self.request_id, field="requeued request id")
        if (
            type(self.outcome) is not ExecutionOutcome
            or self.outcome.disposition is not ExecutionDisposition.REQUEUE
        ):
            raise RecoverableQualificationDispatcherError(
                "qualification requeue outcome is malformed"
            )


class _PreResidentRefusalObserved(Exception):
    """Internal control flow: an authenticated refusal permits one requeue."""

    def __init__(
        self, refusal: AuthenticatedPreResidentRefusal, outcome: ExecutionOutcome
    ) -> None:
        super().__init__(refusal.failure_code)
        self.refusal = refusal
        self.outcome = outcome


class _InfrastructureResultObserved(Exception):
    """Internal control flow: an unproven worker infrastructure result retires
    its dead request and requeues instead of parking the recovery HELD."""

    def __init__(self, failure_code: str, outcome: ExecutionOutcome) -> None:
        super().__init__(failure_code)
        self.failure_code = failure_code
        self.outcome = outcome


def _infrastructure_requeue_signal(failure_code: str) -> _InfrastructureResultObserved:
    outcome_code = (
        failure_code
        if failure_code in PRE_RESIDENT_REQUEUE_FAILURES
        else WORKER_INFRASTRUCTURE_REQUEUE_FAILURE
    )
    return _InfrastructureResultObserved(
        failure_code,
        ExecutionOutcome(
            ExecutionDisposition.REQUEUE,
            decision="NO_DECISION",
            failure_code=outcome_code,
        ),
    )


class _RecoveryLeaseRenewalDenied(Exception):
    """Internal control flow: the retained recovery can no longer renew."""

    def __init__(self, recovery: EvaluationRecovery, reason: str) -> None:
        super().__init__(reason)
        self.recovery = recovery
        self.reason = reason


@dataclass(frozen=True)
class _RecoveryClaim:
    recovery: EvaluationRecovery
    claim: ClaimedQualificationEvaluation | None


class _RecoveryHeartbeat:
    """Serially renew one recovery while the same published request is awaited."""

    def __init__(
        self,
        dispatcher: "RecoverableQualificationDispatcher",
        recovery: EvaluationRecovery,
    ) -> None:
        self._dispatcher = dispatcher
        self._recovery = recovery
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"cacheon-recovery-heartbeat-{recovery.recovery_id[:12]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        interval = self._dispatcher.coordinator.heartbeat_interval_s
        while not self._stop.wait(interval):
            with self._lock:
                recovery = self._recovery
            try:
                renewed = self._dispatcher._renew_if_due(recovery)
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                return
            with self._lock:
                self._recovery = renewed

    def stop(self) -> tuple[EvaluationRecovery, BaseException | None]:
        self._stop.set()
        self._thread.join(self._dispatcher.coordinator.heartbeat_join_timeout_s)
        with self._lock:
            recovery, error = self._recovery, self._error
        if self._thread.is_alive() and error is None:
            error = RecoverableQualificationDispatcherError(
                "recovery heartbeat did not stop within its bounded join"
            )
        return recovery, error


class RecoverableQualificationDispatcher:
    """Advance one FIFO qualification through an exact durable request."""

    _TRANSPORT_METHODS = (
        "plan_qualification_request",
        "materialize_planned_qualification",
        "inspect_planned_qualification",
        "prove_planned_qualification_prepublication",
        "publish_planned_qualification",
        "resume_planned_qualification",
    )

    def __init__(
        self,
        *,
        coordinator: EvaluationCoordinator,
        transport: RecoverableQualificationTransport,
        credential: RemoteWorkerCredential,
        qualification_evidence_root: str | Path,
        qualification_incumbent_stack: EvaluationStackManifest,
        qualification_incumbent_tree_digest: str,
    ) -> None:
        if (
            type(coordinator) is not EvaluationCoordinator
            or coordinator._store_factory is not RecoverableFinalizedIntakeStore
            or type(credential) is not RemoteWorkerCredential
        ):
            raise RecoverableQualificationDispatcherError(
                "recoverable dispatcher authority is not exact"
            )
        identity = getattr(transport, "identity", None)
        if type(identity) is not RemoteWorkerTransportIdentity or any(
            not callable(getattr(transport, method, None))
            for method in self._TRANSPORT_METHODS
        ):
            raise RecoverableQualificationDispatcherError(
                "recoverable transport is not closed and typed"
            )
        if (
            identity.service_digest != coordinator.service.identity
            or identity.worker_readiness_digest != coordinator.readiness.digest
            or identity.credential_digest != credential.digest
        ):
            raise RecoverableQualificationDispatcherError(
                "recoverable transport differs from CPU authority"
            )
        root = Path(qualification_evidence_root)
        if not root.is_absolute() or root != Path(os.path.normpath(root)):
            raise RecoverableQualificationDispatcherError(
                "qualification evidence root is not canonical and absolute"
            )
        if type(qualification_incumbent_stack) is not EvaluationStackManifest:
            raise RecoverableQualificationDispatcherError(
                "qualification incumbent is not exactly typed"
            )
        try:
            tree_digest = require_sha256_hex(
                qualification_incumbent_tree_digest,
                field="qualification incumbent tree digest",
            )
        except (TypeError, ValueError) as exc:
            raise RecoverableQualificationDispatcherError(str(exc)) from None
        runtime = coordinator.service.manifest.runtime
        if (
            qualification_incumbent_stack.runtime_digest != runtime.runtime_digest
            or qualification_incumbent_stack.base_engine_digest
            != runtime.base_engine_digest
            or qualification_incumbent_stack.arena_digest
            != coordinator.service.identity
        ):
            raise RecoverableQualificationDispatcherError(
                "qualification incumbent differs from the sealed service"
            )
        coordinator.readiness.validate(coordinator.service)
        self.coordinator = coordinator
        self.transport = transport
        self.credential = credential
        self.transport_identity = identity
        self.qualification_evidence_root = root
        self.qualification_incumbent_stack = qualification_incumbent_stack
        self.qualification_incumbent_tree_digest = tree_digest
        self.hold_requeue = QualificationHoldRequeuePolicy()

    def _validate_live_authority(self) -> None:
        self.coordinator.readiness.validate(self.coordinator.service)
        if getattr(self.transport, "identity", None) != self.transport_identity:
            raise RecoverableQualificationDispatcherError(
                "recoverable transport identity drifted"
            )

    def _open_store(
        self,
    ) -> tuple[RecoverableFinalizedIntakeStore, tuple[int, str]]:
        store, point = self.coordinator._open_at_durable_cursor()
        if type(store) is not RecoverableFinalizedIntakeStore:
            store.close()
            raise RecoverableQualificationDispatcherError(
                "coordinator opened a non-recoverable intake store"
            )
        return store, point

    def _current_recovery(
        self, recovery_id: str
    ) -> tuple[RecoverableFinalizedIntakeStore, tuple[int, str], EvaluationRecovery]:
        store, point = self._open_store()
        try:
            current = store.pending_qualification_recovery()
            if current is None or current.recovery_id != recovery_id:
                raise RecoverableQualificationDispatcherError(
                    "active recovery identity changed"
                )
            return store, point, current
        except BaseException:
            store.close()
            raise

    def _claim_or_reopen(
        self,
        *,
        expected_members: tuple[EvaluationLeaseMember, ...] | None = None,
        expected_lease_id: str | None = None,
        expected_request_id: str | None = None,
    ) -> _RecoveryClaim | None:
        store, point = self._open_store()
        try:
            recovery = store.pending_qualification_recovery()
            if recovery is None:
                if expected_lease_id is not None or expected_request_id is not None:
                    raise EvaluationClaimConflict(
                        expected_members,
                        (),
                        expected_lease_id=expected_lease_id,
                        expected_request_id=expected_request_id,
                    )
                recovery = store.claim_recoverable_qualification(
                    owner=self.coordinator.owner,
                    current_block=point[0],
                    lease_blocks=self.coordinator.lease_blocks,
                    max_members=self.coordinator.qualification_max_members,
                    expected_members=expected_members,
                )
            elif (
                (expected_members is not None and recovery.lease.members != expected_members)
                or (
                    expected_lease_id is not None
                    and recovery.lease.lease_id != expected_lease_id
                )
                or (
                    expected_request_id is not None
                    and recovery.request_id != expected_request_id
                )
            ):
                raise EvaluationClaimConflict(
                    expected_members,
                    recovery.lease.members,
                    expected_lease_id=expected_lease_id,
                    observed_lease_id=recovery.lease.lease_id,
                    expected_request_id=expected_request_id,
                    observed_request_id=recovery.request_id,
                )
            if recovery is None:
                return None
            if (
                recovery.phase is RecoveryPhase.HELD
                and not recovery.reason.startswith("remote_qualification_hold:")
                and recovery.reason != COMPLETED_NO_DECISION_HOLD_REASON
            ):
                return _RecoveryClaim(recovery, None)
            reservations = tuple(
                store.get(reservation_id)
                for reservation_id in recovery.lease.reservation_ids
            )
            receipts = tuple(
                store.latest_promoted_screen(row.reservation_id)
                for row in reservations
            )
            rotated = rotated_reservation_ids(
                reservations, receipts, self.coordinator.service.identity
            )
        finally:
            store.close()
        if rotated:
            self._rescreen_rotated(recovery, rotated)
            return None
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
            claim = ClaimedQualificationEvaluation(
                recovery.lease,
                reservations,
                publications,
                candidates,
                receipts,
            )
        except Exception as exc:
            self._hold(recovery, "claim_reopen_failed")
            raise RecoverableQualificationDispatcherError(
                "qualification claim could not be reconstructed"
            ) from exc
        return _RecoveryClaim(recovery, claim)

    def _rescreen_rotated(
        self, recovery: EvaluationRecovery, rotated: tuple[str, ...]
    ) -> None:
        """Return a cohort screened by a retired identity to the screen queue."""

        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            release_rotated_cohort(
                store,
                current,
                current_block=point[0],
                reservation_ids=rotated,
            )
        except (IntakeError, ScreenIdentityRotationError) as exc:
            raise RecoverableQualificationDispatcherError(
                f"rotated screen cohort could not be requeued: {exc}"
            ) from exc
        finally:
            store.close()

    def _hold(
        self, recovery: EvaluationRecovery, reason: str
    ) -> RecoverableQualificationHold:
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            held = (
                current
                if current.phase is RecoveryPhase.HELD
                else store.hold_recovery(
                    current,
                    current_block=point[0],
                    reason=reason,
                )
            )
        finally:
            store.close()
        return RecoverableQualificationHold(
            held.recovery_id,
            held.phase,
            held.request_id,
            held.reason,
        )

    def _commit_remote_hold(
        self,
        recovery: EvaluationRecovery,
        product: RemoteQualificationHoldProduct,
    ) -> CompletedQualificationHold:
        reason = durable_remote_qualification_hold_reason(product)
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            lease = store.commit_remote_qualification_hold(
                current,
                current_block=point[0],
                result_digest=product.digest,
                reason=reason,
                reservation_ids=product.reservation_digests,
                lease_blocks=self.coordinator.lease_blocks,
            )
            self.hold_requeue.after_hold(
                store,
                reservation_ids=lease.reservation_ids,
                reason=reason,
            )
        finally:
            store.close()
        return CompletedQualificationHold(
            recovery.recovery_id,
            recovery.request_id,
            lease,
            reason,
            product.digest,
        )

    def _commit_legacy_no_decision_hold(
        self,
        recovery: EvaluationRecovery,
        product: RemoteQualificationProduct,
    ) -> CompletedQualificationHold:
        """Terminalize a retained legacy non-verdict without rerunning GPU work."""

        if not self._has_no_decision(product.batch):
            raise RecoverableQualificationDispatcherError(
                "legacy qualification product has no non-verdict to migrate"
            )
        reason = "remote_qualification_hold:legacy_no_decision"
        reservation_ids = tuple(
            outcome.reservation_digest for outcome in product.batch.outcomes
        )
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            lease = store.commit_remote_qualification_hold(
                current,
                current_block=point[0],
                result_digest=product.digest,
                reason=reason,
                reservation_ids=reservation_ids,
                lease_blocks=self.coordinator.lease_blocks,
            )
            self.hold_requeue.after_hold(
                store,
                reservation_ids=lease.reservation_ids,
                reason=reason,
            )
        finally:
            store.close()
        return CompletedQualificationHold(
            recovery.recovery_id,
            recovery.request_id,
            lease,
            reason,
            product.digest,
        )

    def reconcile_parked_holds(self) -> tuple[str, ...]:
        """Reopen evaluation holds parked before this lifetime; see the policy."""

        store, _point = self._open_store()
        try:
            return self.hold_requeue.reconcile_parked(store)
        finally:
            store.close()

    def _reopen_held_legacy_product(
        self,
        recovery: EvaluationRecovery,
        claim: ClaimedQualificationEvaluation,
    ) -> RemoteQualificationProduct | None:
        """Reopen only an already-completed legacy response; never republish it."""

        plan = self._reopen_plan(recovery, claim)
        observed = self.transport.inspect_planned_qualification(plan)
        if observed.state != "completed_response":
            return None
        response = self.transport.resume_planned_qualification(plan)
        if type(response) is not AuthenticatedRemoteEvaluationResponse:
            raise RecoverableQualificationDispatcherError(
                "held legacy qualification response changed type"
            )
        product = self._product(plan, response)
        if type(product) is not RemoteQualificationProduct or not self._has_no_decision(
            product.batch
        ):
            raise RecoverableQualificationDispatcherError(
                "held legacy qualification product changed"
            )
        return product

    def _reopen_held_remote_product(
        self,
        recovery: EvaluationRecovery,
        claim: ClaimedQualificationEvaluation,
    ) -> RemoteQualificationHoldProduct:
        plan = self._reopen_plan(recovery, claim)
        observed = self.transport.inspect_planned_qualification(plan)
        if observed.state != "completed_response":
            raise RecoverableQualificationDispatcherError(
                "held remote qualification response is no longer retained"
            )
        response = self.transport.resume_planned_qualification(plan)
        if type(response) is not AuthenticatedRemoteEvaluationResponse:
            raise RecoverableQualificationDispatcherError(
                "held remote qualification response changed type"
            )
        product = self._product(plan, response)
        if (
            type(product) is not RemoteQualificationHoldProduct
            or durable_remote_qualification_hold_reason(product) != recovery.reason
        ):
            raise RecoverableQualificationDispatcherError(
                "held remote qualification product changed"
            )
        return product

    def _requeue(
        self,
        recovery: EvaluationRecovery,
        refusal: AuthenticatedPreResidentRefusal,
        outcome: ExecutionOutcome,
    ) -> RecoverableQualificationRequeue:
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            store.release_worker_pre_resident_recovery(
                current, refusal=refusal, current_block=point[0]
            )
        finally:
            store.close()
        return RecoverableQualificationRequeue(
            recovery.recovery_id, refusal.request_id, outcome
        )

    def _requeue_infrastructure(
        self,
        recovery: EvaluationRecovery,
        signal: _InfrastructureResultObserved,
        *,
        live_worker_epoch: str = "",
    ) -> RecoverableQualificationRequeue:
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            store.release_worker_infrastructure_recovery(
                current,
                failure_code=signal.failure_code,
                current_block=point[0],
                live_worker_epoch=live_worker_epoch,
            )
        finally:
            store.close()
        return RecoverableQualificationRequeue(
            recovery.recovery_id, recovery.request_id, signal.outcome
        )

    def _live_worker_epoch(self) -> str | None:
        """Read the live registered worker epoch when the transport carries one.

        Only the spool-backed transport exposes its verified registration;
        every other transport keeps epoch-orphan migration inert and the
        completed-product hold parks for the operator exactly as before.
        """
        registration = getattr(self.transport, "registration", None)
        if not isinstance(registration, dict):
            return None
        value = registration.get("worker_epoch")
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value):
            return value
        return None

    def _retained_epoch(self, recovery: EvaluationRecovery) -> str:
        store, _point, current = self._current_recovery(recovery.recovery_id)
        try:
            return store.reopen_recovery_request_plan(current).worker_epoch
        finally:
            store.close()

    def _renew_if_due(self, recovery: EvaluationRecovery) -> EvaluationRecovery:
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            if point[0] + self.coordinator.lease_blocks <= current.lease.expires_block:
                return current
            renewed, _lease = store.renew_recovery_lease(
                current,
                current_block=point[0],
                lease_blocks=self.coordinator.lease_blocks,
            )
            return renewed
        finally:
            store.close()

    def _renew_before_transition(
        self, recovery: EvaluationRecovery
    ) -> EvaluationRecovery:
        store, point, current = self._current_recovery(recovery.recovery_id)
        store.close()
        if current.phase is RecoveryPhase.CLAIMED:
            if point[0] >= current.lease.expires_block:
                raise _RecoveryLeaseRenewalDenied(
                    current,
                    "lease_expired_before_request_plan",
                )
            return current
        try:
            return self._renew_if_due(current)
        except IntakeError as exc:
            raise _RecoveryLeaseRenewalDenied(
                current,
                "lease_renewal_not_authorized",
            ) from exc

    def _expected_request(
        self, claim: ClaimedQualificationEvaluation
    ) -> RemoteEvaluationRequest:
        return seal_remote_request(
            claim.lease,
            self.coordinator.readiness,
            self.coordinator.service.manifest.service_id,
            self.transport_identity,
            self.credential,
            _request_body_for_qualification(self.coordinator, claim),
        )

    def _reopen_plan(
        self,
        recovery: EvaluationRecovery,
        claim: ClaimedQualificationEvaluation,
    ) -> QualificationRequestPlan:
        store, _point, current = self._current_recovery(recovery.recovery_id)
        try:
            plan = store.reopen_recovery_request_plan(current)
        finally:
            store.close()
        expected = self._expected_request(replace(claim, lease=current.lease))
        if plan.remote_request.to_dict() != expected.to_dict():
            raise QualificationRecoveryHold(
                "cpu_authority_changed",
                plan.request_id,
                "retained request differs from the current CPU claim",
            )
        return plan

    def _prepare(
        self,
        recovery: EvaluationRecovery,
        claim: ClaimedQualificationEvaluation,
    ) -> EvaluationRecovery:
        request = self._expected_request(claim)
        plan = self.transport.plan_qualification_request(request)
        if type(plan) is not QualificationRequestPlan or plan.remote_request != request:
            raise RecoverableQualificationDispatcherError(
                "transport planned another qualification request"
            )
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            return store.prepare_qualification_recovery(
                current,
                plan,
                current_block=point[0],
            )
        finally:
            store.close()

    def _commit_publication(
        self, recovery: EvaluationRecovery
    ) -> EvaluationRecovery:
        recovery = self._renew_before_transition(recovery)
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            return store.commit_recovery_publication(
                current, current_block=point[0]
            )
        finally:
            store.close()

    def _observe_request_ready(
        self, recovery: EvaluationRecovery
    ) -> EvaluationRecovery:
        recovery = self._renew_before_transition(recovery)
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            return store.observe_recovery_request_ready(
                current, current_block=point[0]
            )
        finally:
            store.close()

    def _record_result(self, recovery: EvaluationRecovery) -> EvaluationRecovery:
        recovery = self._renew_before_transition(recovery)
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            return store.record_recovery_result(current, current_block=point[0])
        finally:
            store.close()

    def _record_import(self, recovery: EvaluationRecovery) -> EvaluationRecovery:
        recovery = self._renew_before_transition(recovery)
        store, point, current = self._current_recovery(recovery.recovery_id)
        try:
            return store.record_recovery_import(current, current_block=point[0])
        finally:
            store.close()

    def _await_response(
        self,
        recovery: EvaluationRecovery,
        plan: QualificationRequestPlan,
    ) -> tuple[EvaluationRecovery, AuthenticatedRemoteEvaluationResponse]:
        observed = self.transport.inspect_planned_qualification(plan)
        if observed.state == "result_ready":
            outcome = resolve_infrastructure_result(
                observed.failure_code, observed.refusal, request_id=plan.request_id
            )
            if (
                outcome.disposition is ExecutionDisposition.REQUEUE
                and recovery.phase is RecoveryPhase.REQUEST_READY
            ):
                assert observed.refusal is not None
                raise _PreResidentRefusalObserved(observed.refusal, outcome)
            if recovery.phase is RecoveryPhase.REQUEST_READY:
                # No authenticated refusal and no completed response: the
                # worker terminated this request on its own infrastructure.
                # Retire the dead request and requeue a fresh attempt instead
                # of parking the recovery HELD forever. A completed response
                # recorded in an earlier phase still holds below — that is a
                # store/spool contradiction, not a retryable failure.
                raise _infrastructure_requeue_signal(
                    observed.failure_code
                    or "worker_returned_no_completed_response"
                )
            raise QualificationRecoveryHold(
                "worker_infrastructure_result",
                plan.request_id,
                observed.failure_code or "worker returned no completed response",
            )
        if observed.state not in {"request_ready", "completed_response"}:
            raise QualificationRecoveryHold(
                "published_request_missing",
                plan.request_id,
                "durable recovery says published but spool does not",
            )
        heartbeat = _RecoveryHeartbeat(self, recovery)
        heartbeat.start()
        try:
            response = self.transport.resume_planned_qualification(plan)
        except Exception as exc:
            latest, heartbeat_error = heartbeat.stop()
            if isinstance(exc, QualificationRecoveryHold):
                raise
            cause = heartbeat_error or exc
            raise RecoverableQualificationDispatcherError(
                "same-request qualification result is not ready"
            ) from cause
        latest, heartbeat_error = heartbeat.stop()
        if type(response) is not AuthenticatedRemoteEvaluationResponse:
            raise RecoverableQualificationDispatcherError(
                "same-request resume returned another response type"
            )
        if heartbeat_error is not None:
            # The authenticated local result is durable. Continue from it; a
            # later CAS failure leaves RESULT_READY/SAME_REQUEST recoverable.
            latest = self._renew_if_due(latest)
        return latest, response

    def _product(
        self,
        plan: QualificationRequestPlan,
        response: AuthenticatedRemoteEvaluationResponse,
    ) -> RemoteQualificationProduct | RemoteQualificationHoldProduct:
        try:
            product = reopen_remote_response(
                plan.remote_request,
                response,
                self.transport_identity,
                self.credential,
            )
        except RemoteEvaluationDispatcherError as exc:
            raise QualificationRecoveryHold(
                "remote_response_invalid", plan.request_id, str(exc)
            ) from None
        if type(product) is RemoteQualificationHoldProduct:
            return product
        if type(product) is not RemoteQualificationProduct:
            raise QualificationRecoveryHold(
                "remote_payload_changed",
                plan.request_id,
                "qualification returned another payload type",
            )
        if (
            product.incumbent_stack != self.qualification_incumbent_stack
            or product.incumbent_tree_digest
            != self.qualification_incumbent_tree_digest
        ):
            raise QualificationRecoveryHold(
                "incumbent_changed",
                plan.request_id,
                "remote qualification changed the CPU-owned incumbent",
            )
        return product

    @staticmethod
    def _has_no_decision(batch: QualificationIntakeBatch) -> bool:
        return any(
            outcome.decision is QualificationDecision.NO_DECISION
            for outcome in batch.outcomes
        )

    def _commit_product(
        self,
        recovery: EvaluationRecovery,
        claim: ClaimedQualificationEvaluation,
        product: RemoteQualificationProduct,
    ) -> EvaluationRun:
        current = self._renew_before_transition(recovery)
        claim = replace(claim, lease=current.lease)
        envelope = EvaluationResultEnvelope.seal(
            current.lease,
            self.coordinator.readiness,
            self.coordinator.service,
            product.batch,
        )
        self.coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=product.authority_manifest,
            incumbent_stack=product.incumbent_stack,
            incumbent_tree_digest=product.incumbent_tree_digest,
            batch=product.batch,
            envelope=envelope,
            evidence_root=self.qualification_evidence_root,
            evidence_inventory=product.evidence_inventory,
        )
        return EvaluationRun(current.lease, envelope, product.batch, "completed")

    def dispatch_once(
        self,
    ) -> (
        EvaluationRun
        | RecoverableQualificationHold
        | CompletedQualificationHold
        | RecoverableQualificationRequeue
        | None
    ):
        result = self._dispatch_once(
            expected_members=None,
            expected_lease_id=None,
            expected_request_id=None,
        )
        return result.run if type(result) is GuardedEvaluationRun else result

    def dispatch_guarded_once(
        self,
        *,
        expected_members: tuple[EvaluationLeaseMember, ...],
        expected_lease_id: str | None = None,
        expected_request_id: str | None = None,
    ) -> (
        GuardedEvaluationRun
        | RecoverableQualificationHold
        | CompletedQualificationHold
        | RecoverableQualificationRequeue
        | None
    ):
        return self._dispatch_once(
            expected_members=expected_members,
            expected_lease_id=expected_lease_id,
            expected_request_id=expected_request_id,
        )

    def _dispatch_once(
        self,
        *,
        expected_members: tuple[EvaluationLeaseMember, ...] | None,
        expected_lease_id: str | None,
        expected_request_id: str | None,
    ) -> (
        GuardedEvaluationRun
        | RecoverableQualificationHold
        | CompletedQualificationHold
        | RecoverableQualificationRequeue
        | None
    ):
        """Run or resume one FIFO item without ever creating replacement work."""

        expected_members = _closed_expected_members(expected_members)
        try:
            if expected_lease_id is not None:
                require_sha256_hex(expected_lease_id, field="expected lease id")
            if expected_request_id is not None and expected_request_id != "":
                require_sha256_hex(expected_request_id, field="expected request id")
        except (TypeError, ValueError) as exc:
            raise RecoverableQualificationDispatcherError(str(exc)) from None
        self._validate_live_authority()
        if (
            expected_members is None
            and expected_lease_id is None
            and expected_request_id is None
            and self.hold_requeue.refuse_fresh_claim()
        ):
            return None
        selected = self._claim_or_reopen(
            expected_members=expected_members,
            expected_lease_id=expected_lease_id,
            expected_request_id=expected_request_id,
        )
        if selected is None:
            return None
        recovery, claim = selected.recovery, selected.claim
        if recovery.phase is RecoveryPhase.HELD:
            if recovery.reason.startswith("remote_qualification_hold:"):
                if claim is None:
                    raise RecoverableQualificationDispatcherError(
                        "remote qualification HOLD lost its exact claim"
                    )
                return self._commit_remote_hold(
                    recovery,
                    self._reopen_held_remote_product(recovery, claim),
                )
            if recovery.reason in (
                WORKER_INFRASTRUCTURE_HOLD_REASON,
                AUTHORITY_CHANGED_HOLD_REASON,
                ORPHANED_CARRIER_HOLD_REASON,
            ):
                # All three reasons mean the retained request is durably dead --
                # parked before infrastructure results became requeue-class,
                # sealed against an authority that no longer verifies, or left
                # with a result whose carrier is gone and can never deliver it.
                # Retire the dead request and requeue it the same bounded way.
                return self._requeue_infrastructure(
                    recovery,
                    _infrastructure_requeue_signal("worker_infrastructure_result"),
                )
            if recovery.reason == COMPLETED_NO_DECISION_HOLD_REASON:
                live_epoch = self._live_worker_epoch()
                if (
                    live_epoch is not None
                    and self._retained_epoch(recovery) != live_epoch
                ):
                    # The completed product binds a torn-down worker epoch:
                    # nothing can ever consume it (2026-08-13 zombie: a result
                    # published seconds before teardown starved dispatch). The
                    # store independently re-verifies the epoch mismatch before
                    # migrating the hold into the bounded requeue; for the live
                    # epoch the product stays parked for the operator.
                    return self._requeue_infrastructure(
                        recovery,
                        _infrastructure_requeue_signal("retained_epoch_retired"),
                        live_worker_epoch=live_epoch,
                    )
                if claim is None:
                    raise RecoverableQualificationDispatcherError(
                        "completed legacy qualification lost its exact claim"
                    )
                legacy = self._reopen_held_legacy_product(recovery, claim)
                if legacy is not None:
                    return self._commit_legacy_no_decision_hold(recovery, legacy)
            return RecoverableQualificationHold(
                recovery.recovery_id,
                recovery.phase,
                recovery.request_id,
                recovery.reason,
            )
        assert claim is not None
        try:
            while True:
                recovery = self._renew_before_transition(recovery)
                claim = replace(claim, lease=recovery.lease)
                if recovery.phase is RecoveryPhase.CLAIMED:
                    recovery = self._prepare(recovery, claim)
                    continue
                plan = self._reopen_plan(recovery, claim)
                if recovery.phase is RecoveryPhase.PREPARED:
                    observed = self.transport.materialize_planned_qualification(
                        plan, plan.remote_request
                    )
                    if observed.state != "carrier_materialized":
                        raise QualificationRecoveryHold(
                            "prepublication_state_changed",
                            plan.request_id,
                            "materialization found post-publication evidence",
                        )
                    proof = self.transport.prove_planned_qualification_prepublication(
                        plan
                    )
                    if (
                        type(proof) is not QualificationPrepublicationProof
                        or proof.plan_digest != plan.plan_digest
                        or proof.request_id != plan.request_id
                        or not proof.carrier_materialized
                    ):
                        raise QualificationRecoveryHold(
                            "prepublication_proof_changed",
                            plan.request_id,
                            "transport did not prove the exact materialized carrier",
                        )
                    recovery = self._commit_publication(recovery)
                    continue
                if recovery.phase is RecoveryPhase.PUBLICATION_COMMITTED:
                    observed = self.transport.publish_planned_qualification(plan)
                    if observed.state not in {
                        "request_ready",
                        "result_ready",
                        "completed_response",
                    }:
                        raise QualificationRecoveryHold(
                            "publication_missing",
                            plan.request_id,
                            "REQUEST_READY was not durably published",
                        )
                    recovery = self._observe_request_ready(recovery)
                    continue
                if recovery.phase is RecoveryPhase.REQUEST_READY:
                    recovery, response = self._await_response(recovery, plan)
                    product = self._product(plan, response)
                    recovery = self._record_result(recovery)
                elif recovery.phase in {
                    RecoveryPhase.RESULT_READY,
                    RecoveryPhase.EVIDENCE_IMPORTED,
                }:
                    _latest, response = self._await_response(recovery, plan)
                    product = self._product(plan, response)
                else:
                    raise RecoverableQualificationDispatcherError(
                        "recovery entered an unsupported phase"
                    )
                if type(product) is RemoteQualificationHoldProduct:
                    return self._commit_remote_hold(recovery, product)
                if self._has_no_decision(product.batch):
                    return self._commit_legacy_no_decision_hold(recovery, product)
                if recovery.phase is RecoveryPhase.RESULT_READY:
                    import_remote_qualification_evidence(
                        product, self.qualification_evidence_root
                    )
                    recovery = self._record_import(recovery)
                    continue
                if recovery.phase is RecoveryPhase.EVIDENCE_IMPORTED:
                    run = GuardedEvaluationRun(
                        recovery.request_id,
                        self._commit_product(recovery, claim, product),
                    )
                    self.hold_requeue.note_terminal()
                    return run
        except _PreResidentRefusalObserved as signal:
            return self._requeue(recovery, signal.refusal, signal.outcome)
        except _InfrastructureResultObserved as signal:
            return self._requeue_infrastructure(recovery, signal)
        except _RecoveryLeaseRenewalDenied as signal:
            return self._hold(
                signal.recovery,
                signal.reason,
            )
        except QualificationRecoveryHold as exc:
            return self._hold(recovery, f"transport_hold:{exc.code}")
        except (EvaluationRecoveryHoldError, IntakeError) as exc:
            raise RecoverableQualificationDispatcherError(
                "durable qualification recovery failed closed"
            ) from exc


__all__ = [
    "GuardedEvaluationRun",
    "CompletedQualificationHold",
    "RecoverableQualificationDispatcher",
    "RecoverableQualificationDispatcherError",
    "RecoverableQualificationHold",
    "RecoverableQualificationRequeue",
    "RecoverableQualificationTransport",
]

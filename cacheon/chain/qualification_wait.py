"""Keep the CPU waiter alive without replacing an already-published GPU request."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cacheon.chain.recoverable_qualification_dispatcher import RecoverableQualificationDispatcher

from cacheon.chain.evaluation_recovery import EvaluationRecovery, RecoveryPhase
from cacheon.chain.execution_disposition import (
    AuthenticatedPreResidentRefusal, ExecutionDisposition, ExecutionOutcome,
    resolve_infrastructure_result,
)
from cacheon.chain.remote_evaluation_dispatcher import AuthenticatedRemoteEvaluationResponse
from cacheon.chain.remote_worker_request_plan import QualificationRecoveryHold, QualificationRequestPlan
from cacheon.chain.ssh_worker_transport import RemoteQualificationWaitTimeout


class QualificationWaitPending(Exception):
    """The retained request still owns its lease after a bounded result wait."""

    def __init__(self, recovery: EvaluationRecovery):
        if type(recovery) is not EvaluationRecovery or recovery.phase is not RecoveryPhase.REQUEST_READY:
            raise ValueError("pending qualification requires a published durable recovery")
        self.recovery = recovery
        super().__init__("awaiting_same_request_result")


class _PreResidentRefusalObserved(Exception):
    """Internal control flow: an authenticated refusal permits one requeue."""

    def __init__(
        self, refusal: AuthenticatedPreResidentRefusal, outcome: ExecutionOutcome
    ) -> None:
        super().__init__(refusal.failure_code)
        self.refusal = refusal
        self.outcome = outcome


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
            from cacheon.chain.recoverable_qualification_dispatcher import RecoverableQualificationDispatcherError
            error = RecoverableQualificationDispatcherError(
                "recovery heartbeat did not stop within its bounded join"
            )
        return recovery, error


def await_response(
    dispatcher: "RecoverableQualificationDispatcher",
    recovery: EvaluationRecovery,
    plan: QualificationRequestPlan,
) -> tuple[EvaluationRecovery, AuthenticatedRemoteEvaluationResponse]:
    """Await only the durable published request, preserving ownership on timeout."""
    from cacheon.chain.recoverable_qualification_dispatcher import RecoverableQualificationDispatcherError

    observed = dispatcher.transport.inspect_planned_qualification(plan)
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
        # September 2026: absent completion is not proof that paid work never ran.
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
    heartbeat = _RecoveryHeartbeat(dispatcher, recovery)
    heartbeat.start()
    try:
        response = dispatcher.transport.resume_planned_qualification(plan)
    except Exception as exc:
        latest, heartbeat_error = heartbeat.stop()
        if isinstance(exc, QualificationRecoveryHold):
            raise
        if (heartbeat_error is None and type(exc) is RemoteQualificationWaitTimeout
                and exc.request_id == plan.request_id and latest.phase is RecoveryPhase.REQUEST_READY):
            latest = dispatcher._renew_if_due(latest)
            observed = dispatcher.transport.inspect_planned_qualification(plan)
            if observed.state not in {"request_ready", "completed_response", "result_ready"}:
                raise QualificationRecoveryHold("published_request_missing", plan.request_id,
                                                "published carrier disappeared during the wait") from exc
            raise QualificationWaitPending(latest) from None
        cause = heartbeat_error or exc
        raise RecoverableQualificationDispatcherError(
            f"same-request qualification result is not ready: {type(cause).__name__}: {cause}"
        ) from cause
    latest, heartbeat_error = heartbeat.stop()
    if type(response) is not AuthenticatedRemoteEvaluationResponse:
        raise RecoverableQualificationDispatcherError(
            "same-request resume returned another response type"
        )
    if heartbeat_error is not None:
        # The authenticated local result is durable. Continue from it; a
        # later CAS failure leaves RESULT_READY/SAME_REQUEST recoverable.
        latest = dispatcher._renew_if_due(latest)
    return latest, response

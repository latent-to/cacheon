"""Contract tests for the recovery-event release fence.

The transition validator's central invariant: once a request is published
(PUBLICATION_COMMITTED and later), release requires either the authenticated
worker pre-resident refusal reason (only from REQUEST_READY) or a reviewed
legacy HELD disposition. A generic operator reason releases only before
publication (CLAIMED / PREPARED). These tests pin the fence directly at the
pure transition function, independent of any store harness.
"""

from __future__ import annotations

from cacheon.chain.evaluation_recovery import (
    EvaluationRecoveryEvent,
    RecoveryEventType,
    RecoveryPhase,
    RecoveryResolution,
    evaluation_recovery_event_id,
    stale_incumbent_release_reason,
    stale_incumbent_release_reason_digests,
    valid_evaluation_recovery_event_transition,
)
from cacheon.stack_identity import sha256_hex


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _event(
    *,
    revision: int,
    event_type: RecoveryEventType,
    phase: RecoveryPhase,
    resolution: RecoveryResolution = RecoveryResolution.UNRESOLVED,
    plan: bool = False,
    reason: str = "",
) -> EvaluationRecoveryEvent:
    plan_digest = _h("plan") if plan else ""
    request_id = _h("request") if plan else ""
    event_id = evaluation_recovery_event_id(
        recovery_id=_h("recovery"),
        lease_id=_h("lease"),
        revision=revision,
        event_type=event_type,
        phase=phase,
        resolution=resolution,
        finalized_block=10,
        expires_block=100,
        plan_digest=plan_digest,
        request_id=request_id,
        reason=reason,
    )
    return EvaluationRecoveryEvent(
        sequence=revision + 1,
        event_id=event_id,
        recovery_id=_h("recovery"),
        lease_id=_h("lease"),
        revision=revision,
        event_type=event_type,
        phase=phase,
        resolution=resolution,
        finalized_block=10,
        expires_block=100,
        plan_digest=plan_digest,
        request_id=request_id,
        reason=reason,
    )


def _release(
    *, revision: int, phase: RecoveryPhase, plan: bool, reason: str
) -> EvaluationRecoveryEvent:
    return _event(
        revision=revision,
        event_type=RecoveryEventType.PRE_RESIDENT_RELEASED,
        phase=phase,
        resolution=RecoveryResolution.PRE_RESIDENT_RELEASED,
        plan=plan,
        reason=reason,
    )


def test_stale_incumbent_release_is_digest_bound_and_post_publication_only():
    reason = stale_incumbent_release_reason(
        product_digest=_h("product"),
        previous_stack_digest=_h("old-stack"),
        previous_tree_digest=_h("old-tree"),
        live_stack_digest=_h("new-stack"),
        live_tree_digest=_h("new-tree"),
    )
    assert stale_incumbent_release_reason_digests(reason) == (
        _h("product"),
        _h("old-stack"),
        _h("old-tree"),
        _h("new-stack"),
        _h("new-tree"),
    )
    assert stale_incumbent_release_reason_digests(reason + ":extra") is None

    released = _release(
        revision=4,
        phase=RecoveryPhase.REQUEST_READY,
        plan=True,
        reason=reason,
    )
    for phase in (
        RecoveryPhase.REQUEST_READY,
        RecoveryPhase.RESULT_READY,
        RecoveryPhase.EVIDENCE_IMPORTED,
    ):
        previous = _event(
            revision=3,
            event_type=RecoveryEventType(phase.value),
            phase=phase,
            plan=True,
        )
        assert valid_evaluation_recovery_event_transition(previous, released)
    claimed = _event(
        revision=3,
        event_type=RecoveryEventType.CLAIMED,
        phase=RecoveryPhase.CLAIMED,
    )
    assert not valid_evaluation_recovery_event_transition(claimed, released)


def test_generic_release_is_refused_after_publication() -> None:
    ready = _event(
        revision=3,
        event_type=RecoveryEventType.REQUEST_READY,
        phase=RecoveryPhase.REQUEST_READY,
        plan=True,
    )
    generic = _release(
        revision=4,
        phase=RecoveryPhase.REQUEST_READY,
        plan=True,
        reason="operator_gave_up",
    )
    assert not valid_evaluation_recovery_event_transition(ready, generic)

    committed = _event(
        revision=2,
        event_type=RecoveryEventType.PUBLICATION_COMMITTED,
        phase=RecoveryPhase.PUBLICATION_COMMITTED,
        plan=True,
    )
    generic_committed = _release(
        revision=3,
        phase=RecoveryPhase.PUBLICATION_COMMITTED,
        plan=True,
        reason="operator_gave_up",
    )
    assert not valid_evaluation_recovery_event_transition(
        committed, generic_committed
    )


def test_worker_refusal_reason_releases_only_from_request_ready() -> None:
    worker_reason = "worker_pre_resident:adapter_start_failed"
    ready = _event(
        revision=3,
        event_type=RecoveryEventType.REQUEST_READY,
        phase=RecoveryPhase.REQUEST_READY,
        plan=True,
    )
    worker = _release(
        revision=4,
        phase=RecoveryPhase.REQUEST_READY,
        plan=True,
        reason=worker_reason,
    )
    assert valid_evaluation_recovery_event_transition(ready, worker)

    committed = _event(
        revision=2,
        event_type=RecoveryEventType.PUBLICATION_COMMITTED,
        phase=RecoveryPhase.PUBLICATION_COMMITTED,
        plan=True,
    )
    worker_committed = _release(
        revision=3,
        phase=RecoveryPhase.PUBLICATION_COMMITTED,
        plan=True,
        reason=worker_reason,
    )
    assert not valid_evaluation_recovery_event_transition(
        committed, worker_committed
    )


def test_generic_release_is_legal_before_publication() -> None:
    claimed = _event(
        revision=1,
        event_type=RecoveryEventType.CLAIMED,
        phase=RecoveryPhase.CLAIMED,
    )
    released = _release(
        revision=2,
        phase=RecoveryPhase.CLAIMED,
        plan=False,
        reason="operator_gave_up",
    )
    assert valid_evaluation_recovery_event_transition(claimed, released)

    prepared = _event(
        revision=2,
        event_type=RecoveryEventType.PREPARED,
        phase=RecoveryPhase.PREPARED,
        plan=True,
    )
    released_prepared = _release(
        revision=3,
        phase=RecoveryPhase.PREPARED,
        plan=True,
        reason="operator_gave_up",
    )
    assert valid_evaluation_recovery_event_transition(prepared, released_prepared)


def test_completed_result_reopens_after_incumbent_changes() -> None:
    held = _event(
        revision=4,
        event_type=RecoveryEventType.HELD,
        phase=RecoveryPhase.HELD,
        plan=True,
        reason="transport_hold:incumbent_changed",
    )
    result = _event(
        revision=5,
        event_type=RecoveryEventType.RESULT_READY,
        phase=RecoveryPhase.RESULT_READY,
        plan=True,
    )
    assert valid_evaluation_recovery_event_transition(held, result)

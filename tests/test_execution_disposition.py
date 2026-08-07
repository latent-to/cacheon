from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_pod_service as pod_service
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.evaluation_recovery import (
    WORKER_PRE_RESIDENT_REASON_PREFIX,
    WORKER_PRE_RESIDENT_RELEASE_REASONS,
    RecoveryPhase,
    RecoveryResolution,
)
from cacheon.chain.execution_disposition import (
    AuthenticatedPreResidentRefusal,
    ExecutionDisposition,
    ExecutionDispositionError,
    ExecutionOutcome,
    PRE_RESIDENT_REQUEUE_FAILURES,
    reopen_pre_resident_refusal,
    resolve_completed_result,
    resolve_infrastructure_result,
    seal_pre_resident_refusal,
    worker_pre_resident_release_reason,
)
from cacheon.chain.intake import IntakeError
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationDispatcher,
    RecoverableQualificationHold,
    RecoverableQualificationRequeue,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteWorkerCredential,
    seal_remote_response,
)
from cacheon.chain.remote_qualification_evidence import (
    capture_remote_qualification_product,
    publish_evidence,
)


def _fixtures():
    path = Path(__file__).with_name("test_remote_worker_request_plan.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_execution_disposition_test_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_CREDENTIAL = RemoteWorkerCredential("disposition-test-key", b"d" * 32)
_REQUEST = {"request_id": "a" * 64, "worker_epoch": "b" * 32}


def test_closed_sets_stay_bound_across_pod_store_and_disposition() -> None:
    assert pod_service.PRE_RESIDENT_FAILURES == PRE_RESIDENT_REQUEUE_FAILURES
    assert WORKER_PRE_RESIDENT_RELEASE_REASONS == {
        WORKER_PRE_RESIDENT_REASON_PREFIX + code
        for code in PRE_RESIDENT_REQUEUE_FAILURES
    }
    for code in PRE_RESIDENT_REQUEUE_FAILURES:
        assert worker_pre_resident_release_reason(code).endswith(code)
    with pytest.raises(ExecutionDispositionError):
        worker_pre_resident_release_reason("adapter_timeout")


def test_refusal_seals_and_reopens_only_for_the_exact_request() -> None:
    refusal = seal_pre_resident_refusal(_REQUEST, "adapter_start_failed", _CREDENTIAL)
    payload = refusal.to_payload()
    reopened = reopen_pre_resident_refusal(
        payload,
        request_id=_REQUEST["request_id"],
        worker_epoch=_REQUEST["worker_epoch"],
        credential=_CREDENTIAL,
    )
    assert reopened == refusal
    assert reopened.release_reason == "worker_pre_resident:adapter_start_failed"

    tampered_code = {**payload, "failure_code": "adapter_request_failed"}
    tampered_tag = {**payload, "auth_tag": "f" * 64}
    legacy = {
        "failure_code": "adapter_start_failed",
        "request_id": _REQUEST["request_id"],
        "state": "no_decision",
    }
    other_credential = RemoteWorkerCredential("disposition-test-key", b"x" * 32)
    for case in (tampered_code, tampered_tag, legacy):
        with pytest.raises(ExecutionDispositionError):
            reopen_pre_resident_refusal(
                case,
                request_id=_REQUEST["request_id"],
                worker_epoch=_REQUEST["worker_epoch"],
                credential=_CREDENTIAL,
            )
    with pytest.raises(ExecutionDispositionError):
        reopen_pre_resident_refusal(
            payload,
            request_id="c" * 64,
            worker_epoch=_REQUEST["worker_epoch"],
            credential=_CREDENTIAL,
        )
    with pytest.raises(ExecutionDispositionError):
        reopen_pre_resident_refusal(
            payload,
            request_id=_REQUEST["request_id"],
            worker_epoch=_REQUEST["worker_epoch"],
            credential=other_credential,
        )
    with pytest.raises(ExecutionDispositionError):
        seal_pre_resident_refusal(_REQUEST, "adapter_timeout", _CREDENTIAL)


def test_outcome_keeps_decision_and_disposition_separate() -> None:
    requeue = ExecutionOutcome(
        ExecutionDisposition.REQUEUE,
        decision="NO_DECISION",
        failure_code="adapter_start_failed",
    )
    assert (requeue.decision, requeue.disposition.value) == ("NO_DECISION", "requeue")
    with pytest.raises(ExecutionDispositionError):
        ExecutionOutcome(
            ExecutionDisposition.REQUEUE,
            decision="FAIL",
            failure_code="adapter_start_failed",
        )
    with pytest.raises(ExecutionDispositionError):
        ExecutionOutcome(
            ExecutionDisposition.HOLD,
            decision="PASS",
            failure_code="adapter_timeout",
            reason="infra is never a miner decision",
        )
    with pytest.raises(ExecutionDispositionError):
        ExecutionOutcome(
            ExecutionDisposition.REQUEUE,
            decision="NO_DECISION",
            failure_code="adapter_timeout",
        )
    with pytest.raises(ExecutionDispositionError):
        ExecutionOutcome(ExecutionDisposition.HOLD, decision="NO_DECISION")


def test_resolvers_requeue_only_authenticated_pre_resident_proof() -> None:
    refusal = seal_pre_resident_refusal(_REQUEST, "adapter_request_failed", _CREDENTIAL)
    requeued = resolve_infrastructure_result(
        "adapter_request_failed", refusal, request_id=_REQUEST["request_id"]
    )
    assert requeued.disposition is ExecutionDisposition.REQUEUE
    assert requeued.decision == "NO_DECISION"

    unproven = resolve_infrastructure_result(
        "adapter_request_failed", None, request_id=_REQUEST["request_id"]
    )
    foreign = resolve_infrastructure_result(
        "adapter_request_failed", refusal, request_id="c" * 64
    )
    unknown = resolve_infrastructure_result(
        "adapter_timeout", None, request_id=_REQUEST["request_id"]
    )
    for held in (unproven, foreign, unknown):
        assert held.disposition is ExecutionDisposition.HOLD
        assert held.decision == "NO_DECISION"

    completed_hold = resolve_completed_result(True)
    assert completed_hold.disposition is ExecutionDisposition.HOLD
    assert completed_hold.disposition is not ExecutionDisposition.REQUEUE
    assert completed_hold.reason == "post_publication_no_decision"
    assert resolve_completed_result(False).disposition is (
        ExecutionDisposition.COMPLETE
    )


def _store(authority):
    return RecoverableFinalizedIntakeStore(
        authority.fixtures._db_path(authority.root),
        authority.fixtures.POLICY,
        scope=authority.fixtures.SCOPE,
    )


def _request_ready_recovery(fixtures, authority, store, plan):
    recovery = store.pending_qualification_recovery()
    assert recovery is not None
    prepared = store.prepare_qualification_recovery(
        recovery, plan, current_block=authority.fixtures.BLOCK
    )
    assert fixtures._materialize(authority, plan).state == "carrier_materialized"
    proof = authority.transport().prove_planned_qualification_prepublication(plan)
    assert proof.carrier_materialized is True
    committed = store.commit_recovery_publication(
        prepared, current_block=authority.fixtures.BLOCK
    )
    assert fixtures._publish(authority, plan).state == "request_ready"
    return store.observe_recovery_request_ready(
        committed, current_block=authority.fixtures.BLOCK
    )


def test_store_worker_release_requires_request_ready_and_exact_proof(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    plan = fixtures._plan(authority)
    refusal = seal_pre_resident_refusal(
        plan.request_dict(), "adapter_start_failed", authority.credential
    )
    foreign = seal_pre_resident_refusal(
        {"request_id": "c" * 64, "worker_epoch": plan.worker_epoch},
        "adapter_start_failed",
        authority.credential,
    )
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        with pytest.raises(IntakeError, match="worker pre-resident"):
            store.release_worker_pre_resident_recovery(
                recovery, refusal=refusal, current_block=authority.fixtures.BLOCK
            )
        with pytest.raises(IntakeError, match="release is forbidden"):
            store.release_pre_resident_recovery(
                recovery,
                current_block=authority.fixtures.BLOCK,
                reason="worker_pre_resident:adapter_start_failed",
            )
        ready = _request_ready_recovery(fixtures, authority, store, plan)
        assert ready.phase is RecoveryPhase.REQUEST_READY
        with pytest.raises(IntakeError, match="worker pre-resident"):
            store.release_worker_pre_resident_recovery(
                ready, refusal=foreign, current_block=authority.fixtures.BLOCK
            )
        released_lease = store.release_worker_pre_resident_recovery(
            ready, refusal=refusal, current_block=authority.fixtures.BLOCK
        )
        assert released_lease.lease_id == ready.lease.lease_id
        assert store.pending_qualification_recovery() is None
        retained = store.get(ready.lease.reservation_ids[0])
        assert retained.status == "promoted"
        rows = tuple(
            store._db.execute(
                "SELECT * FROM evaluation_recovery_events WHERE recovery_id=? "
                "ORDER BY sequence",
                (ready.recovery_id,),
            )
        )
        events = tuple(store._evaluation_recovery_event(row) for row in rows)
    assert events[-1].event_type.value == "pre_resident_released"
    assert events[-1].phase is RecoveryPhase.REQUEST_READY
    assert events[-1].resolution is RecoveryResolution.PRE_RESIDENT_RELEASED
    assert events[-1].reason == "worker_pre_resident:adapter_start_failed"
    from cacheon.chain.evaluation_recovery import (
        valid_evaluation_recovery_event_transition,
    )

    assert valid_evaluation_recovery_event_transition(events[-2], events[-1])
    generic = events[-1]
    forged = dict(
        recovery_id=generic.recovery_id,
        lease_id=generic.lease_id,
        revision=generic.revision,
        event_type=generic.event_type,
        phase=generic.phase,
        resolution=generic.resolution,
        finalized_block=generic.finalized_block,
        expires_block=generic.expires_block,
        plan_digest=generic.plan_digest,
        request_id=generic.request_id,
        reason="operator_release",
    )
    from cacheon.chain.evaluation_recovery import (
        EvaluationRecoveryEvent,
        evaluation_recovery_event_id,
    )

    forged_event = EvaluationRecoveryEvent(
        sequence=generic.sequence,
        event_id=evaluation_recovery_event_id(**forged),
        **forged,
    )
    assert not valid_evaluation_recovery_event_transition(events[-2], forged_event)


class _Transport:
    def __init__(self, authority, fixtures, *, on_publish=None):
        self.authority = authority
        self.fixtures = fixtures
        self.delegate = authority.transport()
        self.identity = self.delegate.identity
        self.on_publish = on_publish
        self.plans = 0
        self.plan = None

    def plan_qualification_request(self, request):
        self.plans += 1
        self.plan = self.delegate.plan_qualification_request(request)
        return self.plan

    def materialize_planned_qualification(self, plan, request):
        return self.delegate.materialize_planned_qualification(plan, request)

    def inspect_planned_qualification(self, plan):
        return self.delegate.inspect_planned_qualification(plan)

    def prove_planned_qualification_prepublication(self, plan):
        return self.delegate.prove_planned_qualification_prepublication(plan)

    def publish_planned_qualification(self, plan):
        observed = self.delegate.publish_planned_qualification(plan)
        if self.on_publish is not None:
            self.on_publish(plan)
            self.on_publish = None
        return observed

    def resume_planned_qualification(self, plan):
        return self.delegate.resume_planned_qualification(plan)


def _dispatcher(authority, transport):
    return RecoverableQualificationDispatcher(
        coordinator=authority.coordinator,
        transport=transport,
        credential=authority.credential,
        qualification_evidence_root=authority.root / "cpu-evidence",
        qualification_incumbent_stack=authority.fixtures._incumbent(authority.service),
        qualification_incumbent_tree_digest=authority.fixtures._h("incumbent-tree"),
    )


def _write_pod_refusal_result(authority, plan, failure_code):
    authority.results.mkdir(parents=True, exist_ok=True)
    result_root = authority.results / plan.request_id
    pod_service.infrastructure_result(
        plan.request_dict(), result_root, failure_code, credential=authority.credential
    )
    spool.atomic_bytes(
        result_root / "RESULT_READY", (plan.request_id + "\n").encode(), mode=0o400
    )


def _write_completed_result_for_plan(authority, fixtures, plan):
    request = plan.remote_request
    pod_evidence = authority.root / f"pod-evidence-{plan.request_id[:12]}"
    reference = publish_evidence(
        pod_evidence,
        b'{"attempt":"requeue-cycle-test"}',
        domain="qualification-attempt",
        media_type="application/json",
        schema="cacheon.qualification.plan-recovery-test.v1",
    )
    manifest = authority.fixtures._authority_for_request(request)
    product = capture_remote_qualification_product(
        batch=authority.fixtures._failed_batch(manifest, reference),
        authority_manifest=manifest,
        incumbent_stack=authority.fixtures._incumbent(authority.service),
        incumbent_tree_digest=authority.fixtures._h("incumbent-tree"),
        screen_lane=request.body["screen_lane"],
        service_digest=authority.service.identity,
        readiness=authority.coordinator.readiness,
        evidence_root=pod_evidence,
        evidence_references=(reference,),
    )
    response = seal_remote_response(
        request, product, authority.identity, authority.credential
    )
    carrier = fixtures._inspect(authority, plan).carrier_path
    assert carrier is not None
    result_root = authority.results / plan.request_id
    result_root.mkdir(parents=True)
    (result_root / "response.json").write_bytes(
        spool.spool_canonical_json(response.to_dict()) + b"\n"
    )
    spool.finalize_adapter_response(
        plan.request_dict(),
        carrier,
        result_root,
        identity=authority.identity,
        credential=authority.credential,
    )
    spool.atomic_bytes(
        result_root / "RESULT_READY", (plan.request_id + "\n").encode(), mode=0o400
    )


@pytest.mark.parametrize(
    "failure_code", ["adapter_request_failed", "adapter_start_failed"]
)
def test_dispatcher_requeues_one_authenticated_pre_resident_refusal(
    tmp_path: Path, failure_code: str
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(
        authority,
        fixtures,
        on_publish=lambda plan: _write_pod_refusal_result(
            authority, plan, failure_code
        ),
    )
    outcome = _dispatcher(authority, transport).dispatch_once()

    assert type(outcome) is RecoverableQualificationRequeue
    assert transport.plan is not None
    assert outcome.request_id == transport.plan.request_id
    assert outcome.outcome.disposition is ExecutionDisposition.REQUEUE
    assert outcome.outcome.decision == "NO_DECISION"
    assert outcome.outcome.failure_code == failure_code
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        row = store._db.execute(
            "SELECT resolution, reason FROM evaluation_recoveries WHERE recovery_id=?",
            (outcome.recovery_id,),
        ).fetchone()
        lease = store._db.execute(
            "SELECT state FROM evaluation_leases WHERE lease_id=?",
            (transport.plan.lease.lease_id,),
        ).fetchone()
        retained = store.get(transport.plan.lease.members[0].reservation_id)
    assert (row["resolution"], row["reason"]) == (
        "pre_resident_released",
        WORKER_PRE_RESIDENT_REASON_PREFIX + failure_code,
    )
    assert lease["state"] == "released"
    assert retained.status == "promoted"


def test_requeue_permits_exactly_one_fresh_request_that_completes(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    first = _Transport(
        authority,
        fixtures,
        on_publish=lambda plan: _write_pod_refusal_result(
            authority, plan, "adapter_start_failed"
        ),
    )
    requeued = _dispatcher(authority, first).dispatch_once()
    assert type(requeued) is RecoverableQualificationRequeue

    second = _Transport(
        authority,
        fixtures,
        on_publish=lambda plan: _write_completed_result_for_plan(
            authority, fixtures, plan
        ),
    )
    result = _dispatcher(authority, second).dispatch_once()
    assert result is not None and not isinstance(
        result, (RecoverableQualificationHold, RecoverableQualificationRequeue)
    )
    assert result.disposition == "completed"
    assert second.plan is not None
    assert second.plan.request_id != requeued.request_id
    assert (first.plans, second.plans) == (1, 1)
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        retained = store.get(result.lease.reservation_ids[0])
        resolutions = [
            row["resolution"]
            for row in store._db.execute(
                "SELECT resolution FROM evaluation_recoveries ORDER BY created_block"
            )
        ]
    assert retained.status == "failed"
    assert sorted(resolutions) == ["committed", "pre_resident_released"]

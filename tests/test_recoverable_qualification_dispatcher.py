from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationDispatcher,
    RecoverableQualificationDispatcherError,
    RecoverableQualificationHold,
)
from cacheon.chain.remote_worker_request_plan import QualificationRequestPlan
from cacheon.chain import remote_worker_spool as spool
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationRetryPlan,
)


def _fixtures():
    path = Path(__file__).with_name("test_remote_worker_request_plan.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_recoverable_dispatch_test_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class _Transport:
    def __init__(
        self,
        authority,
        plan_fixtures,
        *,
        complete_on_publish=False,
        fail_resume=False,
    ):
        self.authority = authority
        self.plan_fixtures = plan_fixtures
        self.delegate = authority.transport()
        self.identity = self.delegate.identity
        self.complete_on_publish = complete_on_publish
        self.fail_resume = fail_resume
        self.plans = 0
        self.materializations = 0
        self.publications = 0
        self.resumes = 0
        self.plan: QualificationRequestPlan | None = None

    def plan_qualification_request(self, request):
        self.plans += 1
        self.plan = self.delegate.plan_qualification_request(request)
        return self.plan

    def materialize_planned_qualification(self, plan, request):
        self.materializations += 1
        return self.delegate.materialize_planned_qualification(plan, request)

    def inspect_planned_qualification(self, plan):
        return self.delegate.inspect_planned_qualification(plan)

    def prove_planned_qualification_prepublication(self, plan):
        return self.delegate.prove_planned_qualification_prepublication(plan)

    def publish_planned_qualification(self, plan):
        self.publications += 1
        observed = self.delegate.publish_planned_qualification(plan)
        if self.complete_on_publish:
            self.plan_fixtures._write_completed_result(self.authority, plan)
            self.complete_on_publish = False
        return observed

    def resume_planned_qualification(self, plan):
        self.resumes += 1
        if self.fail_resume:
            raise TimeoutError("simulated CPU waiter interruption")
        return self.delegate.resume_planned_qualification(plan)


def _dispatcher(authority, transport):
    return RecoverableQualificationDispatcher(
        coordinator=authority.coordinator,
        transport=transport,
        credential=authority.credential,
        qualification_evidence_root=authority.root / "cpu-evidence",
        qualification_incumbent_stack=authority.fixtures._incumbent(
            authority.service
        ),
        qualification_incumbent_tree_digest=authority.fixtures._h(
            "incumbent-tree"
        ),
    )


def _store(authority):
    return RecoverableFinalizedIntakeStore(
        authority.fixtures._db_path(authority.root),
        authority.fixtures.POLICY,
        scope=authority.fixtures.SCOPE,
    )


def test_dispatcher_commits_one_authenticated_result_without_legacy_enqueue(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, complete_on_publish=True)

    result = _dispatcher(authority, transport).dispatch_once()

    assert result is not None and result.disposition == "completed"
    assert result.lease.reservation_ids == authority.claim.lease.reservation_ids
    assert (transport.plans, transport.materializations, transport.publications) == (
        1,
        1,
        1,
    )
    assert transport.resumes >= 1
    assert transport.plan is not None
    carriers = [
        path
        for path in authority.outbox.iterdir()
        if path.is_dir() and path.name.endswith(transport.plan.request_id)
    ]
    assert len(carriers) == 1
    with _store(authority) as store:
        retained = store.get(result.lease.reservation_ids[0])
        assert store.pending_qualification_recovery() is None
        recovery_row = store._db.execute(
            "SELECT * FROM evaluation_recoveries WHERE lease_id=?",
            (result.lease.lease_id,),
        ).fetchone()
        assert recovery_row["resolution"] == "committed"
    assert (retained.status, retained.decision) == ("failed", "FAIL")
    evidence_rows = tuple((authority.root / "cpu-evidence").iterdir())
    assert evidence_rows


def test_waiter_restart_reuses_the_same_plan_request_and_carrier(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    dispatcher = _dispatcher(authority, transport)

    with pytest.raises(
        RecoverableQualificationDispatcherError,
        match="same-request qualification result is not ready",
    ):
        dispatcher.dispatch_once()
    assert transport.plan is not None
    request_id = transport.plan.request_id
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        assert (recovery.phase.value, recovery.request_id) == (
            "request_ready",
            request_id,
        )
        assert store.get(recovery.lease.reservation_ids[0]).status == "promoted"

    fixtures._write_completed_result(authority, transport.plan)
    transport.fail_resume = False
    result = dispatcher.dispatch_once()
    assert result is not None and result.disposition == "completed"
    assert (transport.plans, transport.materializations, transport.publications) == (
        1,
        1,
        1,
    )
    carriers = [
        path
        for path in authority.outbox.iterdir()
        if path.is_dir() and path.name.endswith(request_id)
    ]
    assert len(carriers) == 1


def test_postpublication_worker_failure_is_held_and_never_released(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)

    class InfrastructureResultTransport(_Transport):
        def publish_planned_qualification(self, plan):
            observed = super().publish_planned_qualification(plan)
            authority.results.mkdir(parents=True, exist_ok=True)
            spool.write_local_no_decision(
                authority.results,
                plan.request_dict(),
                "adapter_start_failed",
            )
            return observed

    transport = InfrastructureResultTransport(authority, fixtures)
    outcome = _dispatcher(authority, transport).dispatch_once()
    assert type(outcome) is RecoverableQualificationHold
    assert outcome.reason == "transport_hold:worker_infrastructure_result"
    assert transport.plan is not None
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None and recovery.phase.value == "held"
        assert recovery.request_id == transport.plan.request_id
        retained = store.get(recovery.lease.reservation_ids[0])
        lease = store._db.execute(
            "SELECT state FROM evaluation_leases WHERE lease_id=?",
            (recovery.lease.lease_id,),
        ).fetchone()
    assert retained.status == "promoted"
    assert lease["state"] == "active"


def test_completed_no_decision_product_is_held_without_retry_or_second_plan(
    tmp_path: Path, monkeypatch
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)

    def no_decision_batch(manifest, _attempt_ref):
        failure = authority.fixtures._h("post-speed-no-decision")
        outcome = QualificationIntakeOutcome(
            manifest.reservations[0].reservation_digest,
            manifest.reservations[0].selected_delta_digest,
            manifest.digest,
            QualificationDecision.NO_DECISION,
            "raw_speed_evidence",
            True,
            failure_digest=failure,
        )
        retry = QualificationRetryPlan(
            manifest.digest,
            "requeue",
            ((outcome.reservation_digest,),),
            failure,
        )
        return QualificationIntakeBatch(
            manifest.digest, (outcome,), retry_plan=retry
        )

    monkeypatch.setattr(authority.fixtures, "_failed_batch", no_decision_batch)
    transport = _Transport(authority, fixtures, complete_on_publish=True)
    dispatcher = _dispatcher(authority, transport)

    outcome = dispatcher.dispatch_once()
    assert type(outcome) is RecoverableQualificationHold
    assert outcome.reason == "post_publication_no_decision"
    assert (transport.plans, transport.materializations, transport.publications) == (
        1,
        1,
        1,
    )
    repeated = dispatcher.dispatch_once()
    assert repeated == outcome
    assert (transport.plans, transport.materializations, transport.publications) == (
        1,
        1,
        1,
    )
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None and recovery.phase.value == "held"
        retained = store.get(recovery.lease.reservation_ids[0])
    assert retained.status == "promoted"

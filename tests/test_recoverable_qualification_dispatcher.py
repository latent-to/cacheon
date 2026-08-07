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


def _advance_finalized(authority, block: int) -> None:
    cursor = authority.coordinator.advance_finalized_cursor
    assert hasattr(cursor, "set")
    with _store(authority) as store:
        store.reserve_finalized(
            (),
            finalized_block=block,
            finalized_block_hash=authority.fixtures._block_hash(block),
        )
    cursor.set(block)


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


@pytest.mark.parametrize(
    ("after_expiry", "profile"),
    ((0, "collective-expiry"), (3, "block-expiry")),
)
def test_claimed_restart_at_or_after_expiry_holds_without_creating_request(
    tmp_path: Path,
    after_expiry: int,
    profile: str,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(
        tmp_path / profile,
        profile=profile,
        recoverable=True,
    )
    with _store(authority) as store:
        original = store.pending_qualification_recovery()
        assert original is not None and original.phase.value == "claimed"
        assert original.request_id == ""
    restart_block = original.lease.expires_block + after_expiry
    _advance_finalized(authority, restart_block)
    transport = _Transport(authority, fixtures)
    dispatcher = _dispatcher(authority, transport)

    outcome = dispatcher.dispatch_once()
    repeated = dispatcher.dispatch_once()

    assert type(outcome) is RecoverableQualificationHold
    assert repeated == outcome
    assert (
        outcome.recovery_id,
        outcome.request_id,
        outcome.reason,
    ) == (
        original.recovery_id,
        "",
        "lease_expired_before_request_plan",
    )
    assert (
        transport.plans,
        transport.materializations,
        transport.publications,
        transport.resumes,
    ) == (0, 0, 0, 0)
    with _store(authority) as store:
        held = store.pending_qualification_recovery()
        assert held is not None
        events = store.evaluation_recovery_events(held)
        leases = store.active_evaluation_leases()
    assert (
        held.recovery_id,
        held.lease.lease_id,
        held.lease.generation,
        held.lease.expires_block,
        held.request_id,
        held.phase.value,
        held.reason,
    ) == (
        original.recovery_id,
        original.lease.lease_id,
        original.lease.generation,
        original.lease.expires_block,
        "",
        "held",
        "lease_expired_before_request_plan",
    )
    assert leases == (held.lease,)
    assert [event.event_type.value for event in events] == ["claimed", "held"]
    assert not tuple(authority.outbox.iterdir())


def test_prepared_restart_after_expiry_reuses_request_without_second_plan(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(
        tmp_path,
        profile="prepared-restart",
        recoverable=True,
    )

    class InterruptedMaterialization(_Transport):
        def materialize_planned_qualification(self, plan, request):
            self.materializations += 1
            raise RuntimeError("simulated restart after durable prepare")

    first_transport = InterruptedMaterialization(authority, fixtures)
    with pytest.raises(RuntimeError, match="restart after durable prepare"):
        _dispatcher(authority, first_transport).dispatch_once()
    assert first_transport.plan is not None
    request_id = first_transport.plan.request_id
    with _store(authority) as store:
        prepared = store.pending_qualification_recovery()
        assert prepared is not None and prepared.phase.value == "prepared"
        assert prepared.request_id == request_id
    restart_block = prepared.lease.expires_block + 2
    _advance_finalized(authority, restart_block)

    restarted_transport = _Transport(
        authority,
        fixtures,
        complete_on_publish=True,
    )
    result = _dispatcher(authority, restarted_transport).dispatch_once()

    assert result is not None and result.disposition == "completed"
    assert (
        result.lease.lease_id,
        result.lease.generation,
        result.lease.reservation_ids,
    ) == (
        prepared.lease.lease_id,
        prepared.lease.generation,
        prepared.lease.reservation_ids,
    )
    assert result.lease.expires_block > restart_block
    assert first_transport.plans == 1
    assert restarted_transport.plans == 0
    assert restarted_transport.materializations == 1
    carriers = [
        path
        for path in authority.outbox.iterdir()
        if path.is_dir() and path.name.endswith(request_id)
    ]
    assert len(carriers) == 1
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        assert store.active_evaluation_leases() == ()


def test_publication_restart_at_expiry_renews_before_reusing_request(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(
        tmp_path,
        profile="publication-restart",
        recoverable=True,
    )

    class InterruptedPublication(_Transport):
        def publish_planned_qualification(self, plan):
            self.publications += 1
            raise RuntimeError("simulated restart before request publication")

    first_transport = InterruptedPublication(authority, fixtures)
    with pytest.raises(RuntimeError, match="restart before request publication"):
        _dispatcher(authority, first_transport).dispatch_once()
    assert first_transport.plan is not None
    request_id = first_transport.plan.request_id
    with _store(authority) as store:
        committed = store.pending_qualification_recovery()
        assert committed is not None
        assert (committed.phase.value, committed.request_id) == (
            "publication_committed",
            request_id,
        )
    restart_block = committed.lease.expires_block
    _advance_finalized(authority, restart_block)

    restarted_transport = _Transport(
        authority,
        fixtures,
        complete_on_publish=True,
    )
    result = _dispatcher(authority, restarted_transport).dispatch_once()

    assert result is not None and result.disposition == "completed"
    assert (
        result.lease.lease_id,
        result.lease.generation,
        result.lease.reservation_ids,
    ) == (
        committed.lease.lease_id,
        committed.lease.generation,
        committed.lease.reservation_ids,
    )
    assert result.lease.expires_block > restart_block
    assert (
        first_transport.plans,
        restarted_transport.plans,
        restarted_transport.materializations,
        restarted_transport.publications,
    ) == (1, 0, 0, 1)
    carriers = [
        path
        for path in authority.outbox.iterdir()
        if path.is_dir() and path.name.endswith(request_id)
    ]
    assert len(carriers) == 1
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        assert store.active_evaluation_leases() == ()


def test_expired_prepared_recovery_holds_before_transport_when_renewal_denied(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(
        tmp_path,
        profile="renewal-denied",
        recoverable=True,
    )

    class InterruptedMaterialization(_Transport):
        def materialize_planned_qualification(self, plan, request):
            self.materializations += 1
            raise RuntimeError("simulated restart after durable prepare")

    first_transport = InterruptedMaterialization(authority, fixtures)
    with pytest.raises(RuntimeError, match="restart after durable prepare"):
        _dispatcher(authority, first_transport).dispatch_once()
    assert first_transport.plan is not None
    request_id = first_transport.plan.request_id
    with _store(authority) as store:
        prepared = store.pending_qualification_recovery()
        assert prepared is not None and prepared.phase.value == "prepared"
    _advance_finalized(authority, prepared.lease.expires_block)
    authority.coordinator.lease_blocks = authority.fixtures.POLICY.expiry_blocks + 1
    restarted_transport = _Transport(authority, fixtures)

    outcome = _dispatcher(authority, restarted_transport).dispatch_once()

    assert type(outcome) is RecoverableQualificationHold
    assert (
        outcome.recovery_id,
        outcome.request_id,
        outcome.reason,
    ) == (
        prepared.recovery_id,
        request_id,
        "lease_renewal_not_authorized",
    )
    assert (
        restarted_transport.plans,
        restarted_transport.materializations,
        restarted_transport.publications,
        restarted_transport.resumes,
    ) == (0, 0, 0, 0)
    with _store(authority) as store:
        held = store.pending_qualification_recovery()
        assert held is not None
        events = store.evaluation_recovery_events(held)
        leases = store.active_evaluation_leases()
    assert (
        held.recovery_id,
        held.lease.lease_id,
        held.lease.generation,
        held.request_id,
        held.phase.value,
        held.reason,
    ) == (
        prepared.recovery_id,
        prepared.lease.lease_id,
        prepared.lease.generation,
        request_id,
        "held",
        "lease_renewal_not_authorized",
    )
    assert leases == (held.lease,)
    assert [event.event_type.value for event in events] == [
        "claimed",
        "prepared",
        "held",
    ]

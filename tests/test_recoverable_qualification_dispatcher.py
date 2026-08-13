from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import cacheon.chain.recoverable_qualification_dispatcher as dispatcher_module
from cacheon.chain.execution_disposition import ExecutionDisposition
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.recoverable_qualification_dispatcher import (
    CompletedQualificationHold,
    RecoverableQualificationDispatcher,
    RecoverableQualificationDispatcherError,
    RecoverableQualificationHold,
    RecoverableQualificationRequeue,
)
from cacheon.chain.remote_evaluation_dispatcher import seal_remote_response
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldReason,
    capture_remote_qualification_hold,
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


def _write_hold_result(authority, plan, carrier) -> None:
    product = capture_remote_qualification_hold(
        plan.remote_request,
        reason=RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE,
        diagnostic_digest=authority.fixtures._h("graph-evidence-incomplete"),
    )
    response = seal_remote_response(
        plan.remote_request,
        product,
        authority.identity,
        authority.credential,
    )
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


class _InfrastructureResultTransport(_Transport):
    """Every published request terminates with a worker infrastructure result."""

    def __init__(self, authority, plan_fixtures):
        super().__init__(authority, plan_fixtures)
        self.authority = authority
        self.request_ids: list[str] = []

    def publish_planned_qualification(self, plan):
        observed = super().publish_planned_qualification(plan)
        self.request_ids.append(plan.request_id)
        self.authority.results.mkdir(parents=True, exist_ok=True)
        spool.write_local_no_decision(
            self.authority.results,
            plan.request_dict(),
            "adapter_start_failed",
        )
        return observed


def test_postpublication_worker_failure_retires_request_and_requeues_until_capped(
    tmp_path: Path,
) -> None:
    # Owner ruling 2026-08-10: an unproven worker infrastructure result never
    # parks the recovery HELD. The dead request retires, a fresh claim mints a
    # fresh request, and the systemic release cap bounds the retries so an
    # unfixed fault parks visibly for the operator instead of free-looping.
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _InfrastructureResultTransport(authority, fixtures)
    dispatcher = _dispatcher(authority, transport)

    outcomes = [dispatcher.dispatch_once() for _ in range(3)]
    assert [type(outcome).__name__ for outcome in outcomes] == (
        ["RecoverableQualificationRequeue"] * 3
    )
    assert [outcome.request_id for outcome in outcomes] == transport.request_ids
    assert len(set(transport.request_ids)) == 3, "each retry must mint a fresh request"
    assert all(
        outcome.outcome.failure_code == "adapter_start_failed" for outcome in outcomes
    )

    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        reservation_id = store._db.execute(
            "SELECT reservation_id FROM evaluation_lease_members"
        ).fetchone()["reservation_id"]
        retained = store.get(reservation_id)
        reasons = [
            row["reason"]
            for row in store._db.execute(
                "SELECT reason FROM evaluation_leases WHERE state='released'"
                " ORDER BY completed_block, lease_id"
            )
        ]
    assert reasons == ["systemic:worker_infrastructure:adapter_start_failed"] * 3
    assert retained.status == "held"
    # Holding is not a verdict: the park keeps a blank candidate decision
    # (NO_DECISION retired as a decision category, owner order 2026-08-12).
    assert retained.decision == ""
    assert retained.reason.startswith("systemic_release_cap:")

    # The parked reservation is no longer claimable: the queue moves on.
    assert dispatcher.dispatch_once() is None
    assert transport.plans == 3


def test_parked_worker_infrastructure_hold_migrates_to_requeue(
    tmp_path: Path,
) -> None:
    # A recovery parked HELD under the pre-change reason (written before
    # infrastructure results became requeue-class) releases through the same
    # retire-and-requeue on its next claim.
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    dispatcher = _dispatcher(authority, transport)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        dispatcher.dispatch_once()
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None and recovery.phase.value == "request_ready"
        store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="transport_hold:worker_infrastructure_result",
        )

    outcome = _dispatcher(authority, _Transport(authority, fixtures)).dispatch_once()
    assert type(outcome).__name__ == "RecoverableQualificationRequeue"
    assert outcome.outcome.failure_code == "worker_infrastructure_result"
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        released = store._db.execute(
            "SELECT reason FROM evaluation_leases WHERE state='released'"
        ).fetchone()
    assert released["reason"] == (
        "systemic:worker_infrastructure:worker_infrastructure_result"
    )


def test_worker_infrastructure_release_refuses_other_holds(tmp_path: Path) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        _dispatcher(authority, transport).dispatch_once()
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        held = store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="operator_hold",
        )
        with pytest.raises(Exception, match="forbidden"):
            store.release_worker_infrastructure_recovery(
                held,
                failure_code="adapter_start_failed",
                current_block=authority.fixtures.BLOCK,
            )
        still_held = store.pending_qualification_recovery()
    assert still_held is not None and still_held.phase.value == "held"


def test_authority_changed_held_recovery_migrates_into_bounded_requeue(
    tmp_path: Path,
) -> None:
    """A recovery parked HELD with transport_hold:authority_changed carries a
    retained request that can never dispatch again (it was sealed against an
    authority that no longer verifies).  The dispatcher migrates it through
    the same bounded infrastructure requeue: retire the dead request, release
    the reservation for a fresh claim, count one systemic strike."""

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
    dead_request_id = transport.plan.request_id
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None and recovery.phase.value == "request_ready"
        store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="transport_hold:authority_changed",
        )

    transport.fail_resume = False
    outcome = dispatcher.dispatch_once()
    assert type(outcome) is RecoverableQualificationRequeue
    assert outcome.request_id == dead_request_id
    assert outcome.outcome.disposition is ExecutionDisposition.REQUEUE
    assert outcome.outcome.decision == "NO_DECISION"
    with _store(authority) as store:
        # The dead request retired with its recovery; the reservation is
        # claimable again and one systemic strike was recorded.
        assert store.pending_qualification_recovery() is None
        released = store._db.execute(
            "SELECT l.state, l.reason, m.reservation_id "
            "FROM evaluation_leases AS l "
            "JOIN evaluation_lease_members AS m ON m.lease_id=l.lease_id "
            "WHERE l.reason LIKE 'systemic%'",
        ).fetchall()
        assert len(released) == 1
        assert released[0]["state"] == "released"
        assert released[0]["reason"].startswith("systemic:worker_infrastructure:")
        row = store._db.execute(
            "SELECT status FROM reservations WHERE reservation_id=?",
            (released[0]["reservation_id"],),
        ).fetchone()
        assert row["status"] == "promoted"


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


@pytest.mark.parametrize("profile", ("collective-hold", "block-hold"))
def test_authenticated_remote_hold_records_once_then_restarts_same_ids(
    tmp_path: Path,
    monkeypatch,
    profile: str,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(
        tmp_path,
        profile=profile,
        recoverable=True,
    )

    class HoldTransport(_Transport):
        def publish_planned_qualification(self, plan):
            observed = super().publish_planned_qualification(plan)
            assert observed.carrier_path is not None
            _write_hold_result(authority, plan, observed.carrier_path)
            return observed

    counters = {"batch": 0, "claim": 0, "commit": 0, "import": 0}
    original_claim = RecoverableFinalizedIntakeStore.claim_recoverable_qualification

    def counted_claim(store, **kwargs):
        counters["claim"] += 1
        return original_claim(store, **kwargs)

    def forbidden_batch(_batch):
        counters["batch"] += 1
        raise AssertionError("remote HOLD reached miner batch classification")

    def forbidden_import(*_args, **_kwargs):
        counters["import"] += 1
        raise AssertionError("remote HOLD reached evidence import")

    def forbidden_commit(*_args, **_kwargs):
        counters["commit"] += 1
        raise AssertionError("remote HOLD reached intake commit")

    monkeypatch.setattr(
        RecoverableFinalizedIntakeStore,
        "claim_recoverable_qualification",
        counted_claim,
    )
    monkeypatch.setattr(
        dispatcher_module,
        "import_remote_qualification_evidence",
        forbidden_import,
    )
    monkeypatch.setattr(
        authority.coordinator,
        "commit_remote_qualification_result",
        forbidden_commit,
    )
    transport = HoldTransport(authority, fixtures)
    dispatcher = _dispatcher(authority, transport)
    monkeypatch.setattr(dispatcher, "_has_no_decision", forbidden_batch)
    durable_commit = dispatcher._commit_remote_hold
    interrupted = False

    def interrupt_before_commit(recovery, product):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated restart after durable HOLD result")
        return durable_commit(recovery, product)

    monkeypatch.setattr(dispatcher, "_commit_remote_hold", interrupt_before_commit)
    with pytest.raises(RuntimeError, match="restart after durable HOLD result"):
        dispatcher.dispatch_once()
    assert transport.plan is not None
    request_id = transport.plan.request_id
    with _store(authority) as store:
        result_ready = store.pending_qualification_recovery()
        assert result_ready is not None
        assert (result_ready.phase.value, result_ready.request_id) == (
            "result_ready",
            request_id,
        )

    outcome = dispatcher.dispatch_once()

    # The reopened recovery commits the durable HOLD exactly once from the
    # retained response — no replanning, no republication — then bounded
    # auto-requeue releases the cohort straight back to FIFO instead of
    # parking it for an operator (hold-requeue policy, 2026-08-12).
    assert type(outcome) is CompletedQualificationHold
    assert (outcome.request_id, outcome.reason) == (
        request_id,
        "remote_qualification_hold:graph_evidence_incomplete",
    )
    assert counters == {"batch": 0, "claim": 0, "commit": 0, "import": 0}
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
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        completed = store._db.execute(
            "SELECT result_digest FROM evaluation_leases WHERE lease_id=? "
            "AND state='completed'",
            (outcome.lease.lease_id,),
        ).fetchone()
        retained = store.get(outcome.lease.reservation_ids[0])
    assert completed is not None
    assert completed["result_digest"] == outcome.result_digest
    assert retained.status == "published"
    assert retained.reason == (
        "auto_requeue_attempt_2_of_3:"
        "remote_qualification_hold:graph_evidence_incomplete"
    )
    assert not (authority.root / "cpu-evidence").exists()


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


def test_dispatcher_holds_when_product_incumbent_differs_from_cpu_owned(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, complete_on_publish=True)
    dispatcher = RecoverableQualificationDispatcher(
        coordinator=authority.coordinator,
        transport=transport,
        credential=authority.credential,
        qualification_evidence_root=authority.root / "cpu-evidence",
        qualification_incumbent_stack=authority.fixtures._incumbent(
            authority.service
        ),
        qualification_incumbent_tree_digest=authority.fixtures._h("other-tree"),
    )
    outcome = dispatcher.dispatch_once()
    assert type(outcome) is RecoverableQualificationHold
    assert "incumbent_changed" in outcome.reason


def test_completed_no_decision_hold_stays_parked_while_members_are_active(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        _dispatcher(authority, transport).dispatch_once()
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        held = store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="post_publication_no_decision",
        )
        with pytest.raises(Exception, match="forbidden"):
            store.release_worker_infrastructure_recovery(
                held,
                failure_code="retained_epoch_retired",
                current_block=authority.fixtures.BLOCK,
            )
        assert store.pending_qualification_recovery() is not None


def test_completed_no_decision_hold_stays_parked_for_the_live_epoch(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        _dispatcher(authority, transport).dispatch_once()
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        held = store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="post_publication_no_decision",
        )
        live_epoch = store.reopen_recovery_request_plan(held).worker_epoch
        with pytest.raises(Exception, match="forbidden"):
            store.release_worker_infrastructure_recovery(
                held,
                failure_code="retained_epoch_retired",
                current_block=authority.fixtures.BLOCK,
                live_worker_epoch=live_epoch,
            )
        assert store.pending_qualification_recovery() is not None


def test_completed_no_decision_hold_migrates_when_its_epoch_is_retired(
    tmp_path: Path,
) -> None:
    """A completed-product hold whose sealed request plan binds a worker epoch
    other than the live one is an orphan of a torn-down epoch: it migrates
    through the bounded requeue and its fenced member returns to the queue."""

    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        _dispatcher(authority, transport).dispatch_once()
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        held = store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="post_publication_no_decision",
        )
        plan_epoch = store.reopen_recovery_request_plan(held).worker_epoch
        live_epoch = "0" * 32 if plan_epoch != "0" * 32 else "1" * 32
        store.release_worker_infrastructure_recovery(
            held,
            failure_code="retained_epoch_retired",
            current_block=authority.fixtures.BLOCK,
            live_worker_epoch=live_epoch,
        )
        assert store.pending_qualification_recovery() is None
        released = store._db.execute(
            "SELECT reason FROM evaluation_leases WHERE state='released'"
        ).fetchone()
        member = held.lease.members[0]
        restored = store.get(member.reservation_id)
    assert released["reason"] == (
        "systemic:worker_infrastructure:retained_epoch_retired"
    )
    assert restored.status == member.prior_status


def test_epoch_orphaned_completed_hold_migrates_autonomously(
    tmp_path: Path,
) -> None:
    """The dispatcher itself migrates a completed-product hold once the
    transport's live registered epoch provably differs from the epoch the
    retained request plan binds (2026-08-13 zombie: the same migration
    needed a manual operator release and starved both lanes meanwhile)."""

    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    dispatcher = _dispatcher(authority, transport)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        dispatcher.dispatch_once()
    assert transport.plan is not None
    plan_epoch = transport.plan.worker_epoch
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="post_publication_no_decision",
        )

    live_epoch = "0" * 32 if plan_epoch != "0" * 32 else "1" * 32
    transport.registration = {"worker_epoch": live_epoch}
    outcome = dispatcher.dispatch_once()
    assert type(outcome) is RecoverableQualificationRequeue
    assert outcome.outcome.disposition is ExecutionDisposition.REQUEUE
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is None
        released = store._db.execute(
            "SELECT reason FROM evaluation_leases WHERE state='released'"
            " AND reason LIKE 'systemic%'"
        ).fetchone()
    assert released["reason"] == (
        "systemic:worker_infrastructure:retained_epoch_retired"
    )


def test_completed_hold_for_the_live_epoch_parks_at_the_dispatcher(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = _Transport(authority, fixtures, fail_resume=True)
    dispatcher = _dispatcher(authority, transport)
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        dispatcher.dispatch_once()
    assert transport.plan is not None
    with _store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        store.hold_recovery(
            recovery,
            current_block=authority.fixtures.BLOCK,
            reason="post_publication_no_decision",
        )

    transport.registration = {"worker_epoch": transport.plan.worker_epoch}
    outcome = dispatcher.dispatch_once()
    assert type(outcome) is RecoverableQualificationHold
    assert outcome.reason == "post_publication_no_decision"
    with _store(authority) as store:
        assert store.pending_qualification_recovery() is not None

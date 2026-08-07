from __future__ import annotations

import dataclasses
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain.evaluation_lease_store import EvaluationClaimConflict
from cacheon.chain.evaluation_leases import EvaluationLeaseMember
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.remote_evaluation_dispatcher import GuardedEvaluationRun


@dataclass(frozen=True)
class _Profile:
    label: str
    target: str
    topology: str
    architecture: str = "sm120"


PROFILES = (
    _Profile("alpha", "collective.alpha_norm", "tp4"),
    _Profile("beta", "attention.beta_projection", "tp8"),
)


@pytest.fixture(params=PROFILES, ids=lambda profile: profile.label)
def profile(request) -> _Profile:
    return request.param


def _load_fixture_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _fixture_modules(profile: _Profile):
    remote = _load_fixture_module(
        f"cacheon_guarded_remote_{profile.label}",
        "test_remote_evaluation_dispatcher.py",
    )
    base = remote._manifest()
    runtime = dataclasses.replace(
        base.runtime,
        arena_id=f"guarded-{profile.label}",
        target_architecture=profile.architecture,
        topology_class=profile.topology,
        topology_digest=remote._h(f"topology:{profile.label}"),
    )
    manifest = dataclasses.replace(base, runtime=runtime)
    remote._manifest = lambda: manifest
    request_plan = _load_fixture_module(
        f"cacheon_guarded_request_plan_{profile.label}",
        "test_remote_worker_request_plan.py",
    )
    request_plan._dispatcher_fixtures = lambda: remote
    recoverable = _load_fixture_module(
        f"cacheon_guarded_recoverable_{profile.label}",
        "test_recoverable_qualification_dispatcher.py",
    )
    return remote, request_plan, recoverable


def _recoverable_store(authority) -> RecoverableFinalizedIntakeStore:
    return RecoverableFinalizedIntakeStore(
        authority.fixtures._db_path(authority.root),
        authority.fixtures.POLICY,
        scope=authority.fixtures.SCOPE,
    )


def test_guarded_screen_rejects_preview_race_before_transport(
    tmp_path: Path, profile: _Profile
) -> None:
    remote, _request_plan, _recoverable = _fixture_modules(profile)
    root = tmp_path / profile.label
    first, second = remote._published_rows(root, 2)
    service = ArenaService(remote._manifest(), remote._Provider())
    assert service.manifest.runtime.target_architecture == "sm120"
    cursor = remote._Cursor((remote.BLOCK, remote._block_hash(remote.BLOCK)))
    coordinator = remote._coordinator(root, service, cursor)
    credential = remote.RemoteWorkerCredential("guarded-screen-v1", b"g" * 32)
    transport = remote._Transport(coordinator, credential)

    with remote._store(root) as store:
        assert store.preview_evaluation_claim(stage="screen") == (
            first.reservation_id,
        )
        expected = (EvaluationLeaseMember(first.reservation_id, first.status),)
        store._db.execute(
            "UPDATE reservations SET status='held' WHERE reservation_id=?",
            (first.reservation_id,),
        )
        assert store.preview_evaluation_claim(stage="screen") == (
            second.reservation_id,
        )

    dispatcher = remote._dispatcher(coordinator, transport, credential)
    with pytest.raises(EvaluationClaimConflict) as raised:
        dispatcher.dispatch_guarded_screen_once(
            expected_members=expected
        )
    assert raised.value.expected_members == expected
    assert raised.value.observed_members == (
        EvaluationLeaseMember(second.reservation_id, second.status),
    )
    assert transport.requests == []
    with remote._store(root) as store:
        assert store.active_evaluation_leases() == ()
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_leases"
        ).fetchone()["n"] == 0
        assert store.get(first.reservation_id).status == "held"
        assert store.get(second.reservation_id).status == "published"
    guarded = dispatcher.dispatch_guarded_screen_once(
        expected_members=(
            EvaluationLeaseMember(second.reservation_id, second.status),
        )
    )
    assert guarded is not None and guarded.run.disposition == "completed"
    assert guarded.request_id == transport.requests[0].digest
    assert guarded.run.lease.members == guarded.run.envelope.members
    assert len(transport.requests) == 1


def test_guarded_recoverable_claim_rejects_cohort_widening_atomically(
    tmp_path: Path, profile: _Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, _request_plan, _recoverable = _fixture_modules(profile)
    root = tmp_path / profile.label
    first, second = remote._published_rows(root, 2)
    service = ArenaService(remote._manifest(), remote._Provider())
    cursor = remote._Cursor((remote.BLOCK, remote._block_hash(remote.BLOCK)))
    coordinator = remote._coordinator(
        root,
        service,
        cursor,
        qualification_max_members=2,
        store_factory=RecoverableFinalizedIntakeStore,
    )
    assert coordinator.run_screen_once() is not None
    with RecoverableFinalizedIntakeStore(
        remote._db_path(root), remote.POLICY, scope=remote.SCOPE
    ) as store:
        assert store.preview_evaluation_claim(
            stage="qualification", max_members=2
        ) == (first.reservation_id,)
        expected = (EvaluationLeaseMember(first.reservation_id, "promoted"),)
    assert coordinator.run_screen_once() is not None

    with RecoverableFinalizedIntakeStore(
        remote._db_path(root), remote.POLICY, scope=remote.SCOPE
    ) as store:
        expire = store._expire_stale_rows

        def mark_expiry_side_effect(current_block: int) -> None:
            expire(current_block)
            store._db.execute(
                "INSERT INTO metadata(key,value) VALUES('guarded_claim_probe','1')"
            )

        monkeypatch.setattr(store, "_expire_stale_rows", mark_expiry_side_effect)
        with pytest.raises(EvaluationClaimConflict) as raised:
            store.claim_recoverable_qualification(
                owner=coordinator.owner,
                current_block=remote.BLOCK,
                lease_blocks=coordinator.lease_blocks,
                max_members=2,
                expected_members=expected,
            )
        assert raised.value.observed_members == (
            expected[0],
            EvaluationLeaseMember(second.reservation_id, "promoted"),
        )
        assert store.pending_qualification_recovery() is None
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_leases "
            "WHERE stage='qualification'"
        ).fetchone()["n"] == 0
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_recoveries"
        ).fetchone()["n"] == 0
        assert store._db.execute(
            "SELECT value FROM metadata WHERE key='guarded_claim_probe'"
        ).fetchone() is None
        assert tuple(store.get(row.reservation_id).status for row in (first, second)) == (
            "promoted",
            "promoted",
        )


def test_guarded_active_recovery_conflicts_without_transport_then_resumes_exact_request(
    tmp_path: Path, profile: _Profile
) -> None:
    remote, request_plan, recoverable = _fixture_modules(profile)
    authority = request_plan._authority(
        tmp_path / profile.label,
        profile=profile.target,
        recoverable=True,
    )
    assert authority.service.manifest.runtime.target_architecture == "sm120"
    transport = recoverable._Transport(
        authority, request_plan, fail_resume=True
    )
    dispatcher = recoverable._dispatcher(authority, transport)
    with pytest.raises(
        recoverable.RecoverableQualificationDispatcherError,
        match="same-request qualification result is not ready",
    ):
        dispatcher.dispatch_once()
    assert transport.plan is not None

    with _recoverable_store(authority) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        recovery_events = store.evaluation_recovery_events(recovery)
        screen_claim = next(
            event
            for event in store.evaluation_lease_events()
            if event.stage == "screen" and event.event_type == "claimed"
        )
    assert recovery.request_id == transport.plan.request_id
    assert recovery_events[0].request_id == ""
    before_transport = (
        transport.plans,
        transport.materializations,
        transport.publications,
        transport.resumes,
    )
    stale_guards = (
        {
            "expected_members": screen_claim.members,
            "expected_lease_id": recovery.lease.lease_id,
            "expected_request_id": recovery.request_id,
        },
        {
            "expected_members": recovery.lease.members,
            "expected_lease_id": screen_claim.lease_id,
            "expected_request_id": recovery.request_id,
        },
        {
            "expected_members": recovery.lease.members,
            "expected_lease_id": recovery.lease.lease_id,
            "expected_request_id": recovery_events[0].request_id,
        },
    )
    for guard in stale_guards:
        with pytest.raises(EvaluationClaimConflict) as raised:
            dispatcher.dispatch_guarded_once(**guard)
        assert raised.value.observed_members == recovery.lease.members
        assert (
            transport.plans,
            transport.materializations,
            transport.publications,
            transport.resumes,
        ) == before_transport
        with _recoverable_store(authority) as store:
            assert store.pending_qualification_recovery() == recovery

    request_plan._write_completed_result(authority, transport.plan)
    transport.fail_resume = False
    guarded = dispatcher.dispatch_guarded_once(
        expected_members=recovery.lease.members,
        expected_lease_id=recovery.lease.lease_id,
        expected_request_id=recovery.request_id,
    )
    assert type(guarded) is GuardedEvaluationRun
    assert guarded.request_id == recovery.request_id == transport.plan.request_id
    result = guarded.run
    assert result is not None and result.disposition == "completed"
    assert result.lease.lease_id == recovery.lease.lease_id
    assert result.lease.members == recovery.lease.members
    assert (transport.plans, transport.materializations, transport.publications) == (
        1,
        1,
        1,
    )
    carriers = [
        path
        for path in authority.outbox.iterdir()
        if path.is_dir() and path.name.endswith(recovery.request_id)
    ]
    assert len(carriers) == 1

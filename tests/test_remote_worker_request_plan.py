from __future__ import annotations

import dataclasses
import importlib.util
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain import remote_worker_request_plan as planning
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain import ssh_worker_transport as ssh_transport
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.evaluation_recovery import (
    EvaluationRecoveryHoldError,
    RecoveryAction,
    RecoveryPhase,
)
from cacheon.chain.intake import IntakeError
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    _request_body_for_qualification,
    seal_remote_request,
    seal_remote_response,
)
from cacheon.chain.remote_qualification_evidence import (
    capture_remote_qualification_product,
    publish_evidence,
)
from cacheon.chain.remote_worker_artifact_recovery import publication_archive
from cacheon.chain.remote_worker_registration import verify_registration


def _dispatcher_fixtures():
    path = Path(__file__).with_name("test_remote_evaluation_dispatcher.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_request_plan_test_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@dataclass
class _Authority:
    root: Path
    coordinator: object
    claim: object
    service: object
    credential: RemoteWorkerCredential
    identity: RemoteWorkerTransportIdentity
    registration: dict[str, object]
    registration_path: Path
    request: object
    wire_path: Path
    publication_paths: tuple[Path, ...]
    fixtures: object

    @property
    def inputs(self):
        return (
            ("qualification_payload", self.wire_path),
            *(
                ("candidate_publication", path)
                for path in self.publication_paths
            ),
        )

    @property
    def outbox(self) -> Path:
        return self.root / "spool" / "outbox"

    @property
    def results(self) -> Path:
        return self.root / "spool" / "results"

    def transport(self):
        return ssh_transport.DurableSpoolAuthenticatedWorkerTransport(
            registration_path=self.registration_path,
            spool_root=self.root / "spool",
            credential_path=self.registration["credential_path"],
            qualification_publication_resolver=lambda _request: self.claim.publications,
            response_timeout_seconds=60,
            poll_seconds=1,
        )

    def cleanup(self) -> None:
        self.coordinator._release(self.claim.lease, reason="test_cleanup")


def _published_profile(fixtures, root: Path, profile: str) -> None:
    source = root / f"source-{profile}"
    source.mkdir(parents=True)
    leaf = source / "manifest.toml"
    leaf.write_text(f"bundle_id = '{profile}'\n", encoding="utf-8")
    source.chmod(0o700)
    leaf.chmod(0o600)
    committed = fixtures.content_hash(source)
    publication = fixtures.publish_worker_bundle(
        source, root / "publications", committed
    )
    arrival = fixtures.FinalizedArrival(
        f"miner-{profile}",
        committed,
        f"https://example.invalid/{profile}",
        fixtures.BLOCK,
        fixtures._block_hash(fixtures.BLOCK),
        0,
    )
    with fixtures._store(root) as store:
        row = store.reserve_finalized(
            (arrival,),
            finalized_block=fixtures.BLOCK,
            finalized_block_hash=fixtures._block_hash(fixtures.BLOCK),
        )[0]
        store.mark_fetching(row.reservation_id)
        store.mark_published(
            row.reservation_id,
            delta_fingerprint=fixtures.SubmittedDeltaFingerprint(
                "component",
                f"target.{profile}",
                fixtures._h(f"base:{profile}"),
                (f"slot.{profile}",),
                fixtures._h(f"archive:{profile}"),
                fixtures._h(f"selected:{profile}"),
                fixtures._h(f"exact:{profile}"),
                (fixtures._h(f"source:{profile}"),),
                (fixtures._h(f"binary:{profile}"),),
            ),
            publication_digest=publication.digest,
            publication_root=publication.root,
        )


def _authority(
    root: Path,
    *,
    endpoint: str = "worker-endpoint-a",
    profile: str = "alpha",
    profiles: tuple[str, ...] | None = None,
    recoverable: bool = False,
) -> _Authority:
    fixtures = _dispatcher_fixtures()
    cohort = profiles if profiles is not None else (profile,)
    for name in cohort:
        _published_profile(fixtures, root, name)
    service = ArenaService(fixtures._manifest(), fixtures._Provider())
    cursor = fixtures._Cursor((fixtures.BLOCK, fixtures._block_hash(fixtures.BLOCK)))
    coordinator_options: dict[str, object] = {
        "qualification_max_members": len(cohort)
    }
    if recoverable:
        coordinator_options["store_factory"] = RecoverableFinalizedIntakeStore
    coordinator = fixtures._coordinator(root, service, cursor, **coordinator_options)
    for _name in cohort:
        fixtures._promote_one(coordinator)
    claim = fixtures._claim_qualification(coordinator)
    assert len(claim.publications) == len(cohort)
    credential = RemoteWorkerCredential("qualification-key-v1", b"q" * 32)
    identity = fixtures._transport_identity(
        coordinator, credential, endpoint=endpoint
    )
    secret = root / "credential.secret"
    secret.write_bytes(b"q" * 32)
    secret.chmod(0o400)
    known_hosts = root / "known_hosts"
    known_hosts.write_text("pinned-host-key\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    registration: dict[str, object] = {
        "adapter_sha256": "a" * 64,
        "created_at_unix": int(time.time()),
        "credential_digest": credential.digest,
        "credential_file_sha256": spool.file_sha256(secret),
        "credential_id": credential.credential_id,
        "credential_path": str(secret),
        "known_hosts_path": str(known_hosts),
        "known_hosts_sha256": spool.file_sha256(known_hosts),
        "lane_devices": list(range(coordinator.readiness.gpu_count)),
        "lane_digest": "e" * 64,
        "pod_host": "pod.example",
        "pod_port": 22,
        "pod_user": "root",
        "python_executable": sys.executable,
        "python_executable_sha256": spool.file_sha256(Path(sys.executable).resolve()),
        "ready_receipt_digest": coordinator.readiness.ready_receipt_digest,
        "ready_receipt_file_sha256": "b" * 64,
        "remote_service_sha256": "c" * 64,
        "schema": spool.SCHEMA_REGISTRATION,
        "service_identity": service.manifest.service_id,
        "transport_identity": identity.to_dict(),
        "transport_identity_digest": identity.digest,
        "worker_epoch": "d" * 32,
        "worker_readiness": coordinator.readiness.to_dict(),
        "worker_readiness_digest": coordinator.readiness.digest,
    }
    registration["registration_digest"] = spool.spool_digest(
        spool.DOMAIN_REGISTRATION, registration
    )
    verify_registration(registration)
    registration_path = root / "registration.json"
    spool.atomic_json(registration_path, registration, mode=0o400)
    request = seal_remote_request(
        claim.lease,
        coordinator.readiness,
        service.manifest.service_id,
        identity,
        credential,
        _request_body_for_qualification(coordinator, claim),
    )
    wire_path = root / "qualification-request.json"
    spool.atomic_json(wire_path, request.to_dict(), mode=0o400)
    publication_paths = []
    for index, publication in enumerate(claim.publications):
        publication_path = root / f"candidate-publication-{index}.tar"
        publication_archive(publication, publication_path)
        publication_paths.append(publication_path)
    return _Authority(
        root,
        coordinator,
        claim,
        service,
        credential,
        identity,
        registration,
        registration_path,
        request,
        wire_path,
        tuple(publication_paths),
        fixtures,
    )


def _plan(authority: _Authority) -> planning.QualificationRequestPlan:
    return planning.create_qualification_request_plan(
        authority.registration,
        authority.claim.lease,
        authority.request,
        authority.inputs,
        deadline_seconds=60,
        identity=authority.identity,
        credential=authority.credential,
    )


def _materialize(authority: _Authority, plan):
    return planning.materialize_planned_qualification(
        plan,
        authority.inputs,
        authority.outbox,
        authority.results,
        authority.registration,
        identity=authority.identity,
        credential=authority.credential,
    )


def _publish(authority: _Authority, plan):
    return planning.publish_planned_qualification(
        plan,
        authority.outbox,
        authority.results,
        authority.registration,
        identity=authority.identity,
        credential=authority.credential,
    )


def _inspect(authority: _Authority, plan):
    return planning.inspect_planned_qualification(
        plan,
        authority.outbox,
        authority.results,
        authority.registration,
        identity=authority.identity,
        credential=authority.credential,
    )


def _write_completed_result(authority: _Authority, plan) -> object:
    pod_evidence = authority.root / "pod-evidence"
    reference = publish_evidence(
        pod_evidence,
        b'{"attempt":"plan-recovery-test"}',
        domain="qualification-attempt",
        media_type="application/json",
        schema="cacheon.qualification.plan-recovery-test.v1",
    )
    manifest = authority.fixtures._authority_for_request(authority.request)
    product = capture_remote_qualification_product(
        batch=authority.fixtures._failed_batch(manifest, reference),
        authority_manifest=manifest,
        incumbent_stack=authority.fixtures._incumbent(authority.service),
        incumbent_tree_digest=authority.fixtures._h("incumbent-tree"),
        screen_lane=authority.request.body["screen_lane"],
        service_digest=authority.service.identity,
        readiness=authority.coordinator.readiness,
        evidence_root=pod_evidence,
        evidence_references=(reference,),
    )
    response = seal_remote_response(
        authority.request, product, authority.identity, authority.credential
    )
    carrier = _inspect(authority, plan).carrier_path
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
    return response


def test_reconstructed_transport_uses_one_plan_carrier_ready_and_no_enqueue(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _authority(tmp_path)
    calls = 0

    def forbidden_enqueue(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("recovery invoked legacy enqueue")

    monkeypatch.setattr(ssh_transport, "enqueue_request", forbidden_enqueue)
    first = authority.transport()
    plan = first.plan_qualification_request(authority.request)
    reopened = planning.QualificationRequestPlan.from_dict(plan.to_dict())
    assert reopened == plan
    assert first.inspect_planned_qualification(plan).state == "planned_unpublished"
    assert first.materialize_planned_qualification(plan, authority.request).state == (
        "carrier_materialized"
    )
    proof = first.prove_planned_qualification_prepublication(plan)
    assert proof.state == "carrier_materialized"

    second = authority.transport()
    assert second.publish_planned_qualification(reopened).state == "request_ready"
    expected = _write_completed_result(authority, reopened)
    assert second.resume_planned_qualification(reopened).digest == expected.digest
    carriers = [
        path
        for path in authority.outbox.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert len(carriers) == 1
    assert (carriers[0] / "REQUEST_READY").read_bytes() == (
        plan.request_id + "\n"
    ).encode()
    assert calls == 0
    authority.cleanup()


def test_plan_samples_time_once_and_supports_two_independent_roots(
    tmp_path: Path, monkeypatch
) -> None:
    first = _authority(
        tmp_path / "profile-a", endpoint="endpoint-a", profile="collective-alpha"
    )
    second = _authority(
        tmp_path / "profile-b", endpoint="endpoint-b", profile="block-beta"
    )
    sampled: list[int] = []

    def one_sample() -> int:
        value = 1_800_000_000_123_456_789 + len(sampled)
        sampled.append(value)
        return value

    monkeypatch.setattr(planning.time, "time_ns", one_sample)
    first_plan = _plan(first)
    second_plan = _plan(second)
    assert len(sampled) == 2
    assert first_plan.created_at_unix == sampled[0] // 1_000_000_000
    assert second_plan.created_at_unix == sampled[1] // 1_000_000_000
    assert first_plan.request_id != second_plan.request_id
    assert first_plan.remote_request.body["candidates"][0]["publication"] != (
        second_plan.remote_request.body["candidates"][0]["publication"]
    )
    first.cleanup()
    second.cleanup()


def test_partial_hidden_blob_is_repaired_only_while_prepublication(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _authority(tmp_path)
    plan = _plan(authority)
    original = planning.copy_stable_artifact
    crashed = False

    def crash_during_copy(source, destination, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            destination.write_bytes(b"partial")
            raise OSError("simulated hard interruption")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(planning, "copy_stable_artifact", crash_during_copy)
    with pytest.raises(OSError, match="hard interruption"):
        _materialize(authority, plan)
    hidden = authority.outbox / f".planned-{plan.request_id}"
    assert hidden.is_dir() and not (hidden / "request.json").exists()
    assert _materialize(authority, plan).state == "carrier_materialized"
    assert not hidden.exists()
    assert len([p for p in authority.outbox.iterdir() if p.is_dir()]) == 1
    authority.cleanup()


@pytest.mark.parametrize("mode", ["tampered", "duplicate", "missing"])
def test_tampered_duplicate_and_missing_carriers_hold(
    tmp_path: Path, mode: str
) -> None:
    authority = _authority(tmp_path)
    plan = _plan(authority)
    observation = _materialize(authority, plan)
    assert observation.carrier_path is not None
    if mode == "tampered":
        request_path = observation.carrier_path / "request.json"
        request_path.chmod(0o600)
        request_path.write_text("{}\n", encoding="utf-8")
        action = lambda: _inspect(authority, plan)
    elif mode == "duplicate":
        shutil.copytree(observation.carrier_path, authority.outbox / "duplicate")
        action = lambda: _inspect(authority, plan)
    else:
        shutil.rmtree(observation.carrier_path)
        action = lambda: authority.transport().resume_planned_qualification(plan)
    with pytest.raises(planning.QualificationRecoveryHold):
        action()
    authority.cleanup()


def test_changed_worker_readiness_registration_or_credential_holds(
    tmp_path: Path
) -> None:
    authority = _authority(tmp_path)
    plan = _plan(authority)
    _materialize(authority, plan)

    changed_worker = dict(authority.registration)
    changed_worker["worker_epoch"] = "e" * 32
    changed_worker.pop("registration_digest")
    changed_worker["registration_digest"] = spool.spool_digest(
        spool.DOMAIN_REGISTRATION, changed_worker
    )
    with pytest.raises(planning.QualificationRecoveryHold, match="authority_changed"):
        planning.inspect_planned_qualification(
            plan,
            authority.outbox,
            authority.results,
            changed_worker,
            identity=authority.identity,
            credential=authority.credential,
        )

    readiness = WorkerReadiness(
        **{**authority.coordinator.readiness.to_dict(), "ready_epoch": 8}
    )
    changed_identity = dataclasses.replace(
        authority.identity, worker_readiness_digest=readiness.digest
    )
    changed_readiness = dict(authority.registration)
    changed_readiness.update(
        {
            "transport_identity": changed_identity.to_dict(),
            "transport_identity_digest": changed_identity.digest,
            "worker_readiness": readiness.to_dict(),
            "worker_readiness_digest": readiness.digest,
        }
    )
    changed_readiness.pop("registration_digest")
    changed_readiness["registration_digest"] = spool.spool_digest(
        spool.DOMAIN_REGISTRATION, changed_readiness
    )
    with pytest.raises(planning.QualificationRecoveryHold, match="authority_changed"):
        planning.inspect_planned_qualification(
            plan,
            authority.outbox,
            authority.results,
            changed_readiness,
            identity=changed_identity,
            credential=authority.credential,
        )

    wrong_credential = RemoteWorkerCredential("qualification-key-v1", b"x" * 32)
    with pytest.raises(planning.QualificationRecoveryHold, match="authority_changed"):
        planning.inspect_planned_qualification(
            plan,
            authority.outbox,
            authority.results,
            authority.registration,
            identity=authority.identity,
            credential=wrong_credential,
        )
    authority.cleanup()


def test_same_epoch_registration_refresh_keeps_plan_dispatchable(
    tmp_path: Path,
) -> None:
    """A registration refresh that re-digests the registration without touching
    the worker epoch, transport identity, credential, or readiness (for
    example a service file rotation) must not hold retained plans: every
    binding the sealed request carries still verifies."""

    authority = _authority(tmp_path)
    plan = _plan(authority)
    _materialize(authority, plan)

    refreshed = dict(authority.registration)
    refreshed["remote_service_sha256"] = "f" * 64
    refreshed.pop("registration_digest")
    refreshed["registration_digest"] = spool.spool_digest(
        spool.DOMAIN_REGISTRATION, refreshed
    )
    verify_registration(refreshed)
    assert refreshed["registration_digest"] != plan.registration_digest

    observation = planning.inspect_planned_qualification(
        plan,
        authority.outbox,
        authority.results,
        refreshed,
        identity=authority.identity,
        credential=authority.credential,
    )
    assert observation.state == "carrier_materialized"
    authority.cleanup()


@pytest.mark.parametrize("evidence", ["ready", "dispatch", "result"])
def test_prepublication_proof_rejects_any_point_of_no_return_evidence(
    tmp_path: Path, evidence: str
) -> None:
    authority = _authority(tmp_path)
    plan = _plan(authority)
    observation = _materialize(authority, plan)
    assert observation.carrier_path is not None
    if evidence == "ready":
        _publish(authority, plan)
    elif evidence == "dispatch":
        spool.write_dispatch_state(
            observation.carrier_path,
            plan.request_id,
            "transferred",
            plan.worker_epoch,
        )
    else:
        authority.results.mkdir(parents=True, exist_ok=True)
        spool.write_local_no_decision(
            authority.results, plan.request_dict(), "adapter_start_failed"
        )
    with pytest.raises(planning.QualificationRecoveryHold):
        planning.prove_planned_qualification_prepublication(
            plan,
            authority.outbox,
            authority.results,
            authority.registration,
            identity=authority.identity,
            credential=authority.credential,
        )
    authority.cleanup()


def test_concurrent_publishers_create_one_exact_ready_marker(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    plan = _plan(authority)
    observation = _materialize(authority, plan)
    barrier = threading.Barrier(2)

    def publish():
        barrier.wait()
        return _publish(authority, plan).state

    with ThreadPoolExecutor(max_workers=2) as pool:
        states = tuple(pool.map(lambda _: publish(), range(2)))
    assert states == ("request_ready", "request_ready")
    assert observation.carrier_path is not None
    markers = list(observation.carrier_path.glob("REQUEST_READY"))
    assert len(markers) == 1
    assert markers[0].read_bytes() == (plan.request_id + "\n").encode()
    authority.cleanup()


def test_sqlite_recovery_persists_one_plan_across_every_publication_crash_window(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path, recoverable=True)
    plan = _plan(authority)
    store_options = (
        authority.fixtures._db_path(authority.root),
        authority.fixtures.POLICY,
    )
    store_keywords = {"scope": authority.fixtures.SCOPE}

    with RecoverableFinalizedIntakeStore(
        *store_options, **store_keywords
    ) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        prepared = store.prepare_qualification_recovery(
            recovery, plan, current_block=authority.fixtures.BLOCK
        )
        assert prepared.phase is RecoveryPhase.PREPARED
        assert prepared.action is RecoveryAction.SAME_REQUEST
        assert store.reopen_recovery_request_plan(prepared) == plan
        prepared, _renewed_lease = store.renew_recovery_lease(
            prepared,
            current_block=authority.fixtures.BLOCK,
            lease_blocks=30,
        )
        assert store.reopen_recovery_request_plan(prepared) == plan

    materialized = _materialize(authority, plan)
    assert materialized.state == "carrier_materialized"
    proof = authority.transport().prove_planned_qualification_prepublication(plan)
    assert proof.state == "carrier_materialized"

    with RecoverableFinalizedIntakeStore(
        *store_options, **store_keywords
    ) as store:
        reopened = store.pending_qualification_recovery()
        assert reopened == prepared
        assert store.reopen_recovery_request_plan(reopened) == plan
        publication = store.commit_recovery_publication(
            reopened, current_block=authority.fixtures.BLOCK
        )

    assert authority.transport().publish_planned_qualification(plan).state == (
        "request_ready"
    )
    with RecoverableFinalizedIntakeStore(
        *store_options, **store_keywords
    ) as store:
        publication = store.pending_qualification_recovery()
        assert publication is not None
        ready = store.observe_recovery_request_ready(
            publication, current_block=authority.fixtures.BLOCK
        )
        assert ready.request_id == plan.request_id
        assert store.reopen_recovery_request_plan(ready) == plan

    expected = _write_completed_result(authority, plan)
    observed = authority.transport().resume_planned_qualification(plan)
    assert observed.digest == expected.digest

    with RecoverableFinalizedIntakeStore(
        *store_options, **store_keywords
    ) as store:
        ready = store.pending_qualification_recovery()
        assert ready is not None
        result = store.record_recovery_result(
            ready, current_block=authority.fixtures.BLOCK
        )
        imported = store.record_recovery_import(
            result, current_block=authority.fixtures.BLOCK
        )
        assert imported.action is RecoveryAction.IMPORT_ONLY
        held = store.hold_recovery(
            imported,
            current_block=authority.fixtures.BLOCK,
            reason="test_stops_before_result_commit",
        )
        assert held.action is RecoveryAction.HOLD
        assert [
            event.event_type.value
            for event in store.evaluation_recovery_events(held)
        ] == [
            "claimed",
            "prepared",
            "renewed",
            "publication_committed",
            "request_ready",
            "result_ready",
            "evidence_imported",
            "held",
        ]


def test_sqlite_recovery_rejects_a_plan_bound_to_another_lease(
    tmp_path: Path,
) -> None:
    first = _authority(tmp_path / "first", recoverable=True)
    second = _authority(
        tmp_path / "second", profile="beta", recoverable=True
    )
    foreign_plan = _plan(first)
    with RecoverableFinalizedIntakeStore(
        second.fixtures._db_path(second.root),
        second.fixtures.POLICY,
        scope=second.fixtures.SCOPE,
    ) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        with pytest.raises(IntakeError, match="differs from its recovery lease"):
            store.prepare_qualification_recovery(
                recovery,
                foreign_plan,
                current_block=second.fixtures.BLOCK,
            )


def test_sqlite_recovery_tampered_plan_bytes_hold_without_redispatch(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path, recoverable=True)
    plan = _plan(authority)
    with RecoverableFinalizedIntakeStore(
        authority.fixtures._db_path(authority.root),
        authority.fixtures.POLICY,
        scope=authority.fixtures.SCOPE,
    ) as store:
        recovery = store.pending_qualification_recovery()
        assert recovery is not None
        prepared = store.prepare_qualification_recovery(
            recovery, plan, current_block=authority.fixtures.BLOCK
        )
        with store._evaluation_recovery_mutation(prepared.lease.lease_id):
            store._db.execute(
                "UPDATE evaluation_recoveries SET request_plan=? WHERE recovery_id=?",
                (b"{}", prepared.recovery_id),
            )
        with pytest.raises(EvaluationRecoveryHoldError, match="request plan.*HOLD"):
            store.pending_qualification_recovery()


def test_two_member_cohort_plan_carries_every_publication_in_order(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path, profiles=("alpha", "beta"))
    try:
        assert len(authority.claim.lease.reservation_ids) == 2
        plan = _plan(authority)
        roles = tuple(row.role for row in plan.artifacts)
        assert roles == (
            "qualification_payload",
            "candidate_publication",
            "candidate_publication",
        )
        expected = tuple(
            spool.file_sha256(path) for path in authority.publication_paths
        )
        assert tuple(row.sha256 for row in plan.artifacts[1:]) == expected
        assert _materialize(authority, plan).state == "carrier_materialized"
        published = _publish(authority, plan)
        assert published.state == "request_ready"
        assert published.carrier_path is not None
        blobs = published.carrier_path / "blobs"
        for row in plan.artifacts:
            assert (blobs / row.sha256).is_file()
    finally:
        authority.cleanup()

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tarfile
import time
from pathlib import Path

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain import remote_worker_registration as registration_module
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteWorkerCredential,
    _request_body_for_screen,
    seal_remote_request,
    seal_remote_response,
)
from cacheon.stack_identity import canonical_json_bytes


def _dispatcher_fixtures():
    path = Path(__file__).with_name("test_remote_evaluation_dispatcher.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_remote_dispatcher_test_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _screen_authority(tmp_path: Path):
    fixtures = _dispatcher_fixtures()
    fixtures._published_rows(tmp_path, 1)
    service = ArenaService(fixtures._manifest(), fixtures._Provider())
    cursor = fixtures._Cursor((fixtures.BLOCK, fixtures._block_hash(fixtures.BLOCK)))
    coordinator = fixtures._coordinator(tmp_path, service, cursor)
    claim = coordinator.claim_screen()
    assert claim is not None
    credential = RemoteWorkerCredential("screen-key-v1", b"s" * 32)
    identity = fixtures._transport_identity(coordinator, credential)
    secret = tmp_path / "credential.secret"
    secret.write_bytes(b"s" * 32)
    secret.chmod(0o400)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("pinned-host-key\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    registration = {
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
    registration_module.verify_registration(registration)
    request = seal_remote_request(
        claim.lease,
        coordinator.readiness,
        service.manifest.service_id,
        identity,
        credential,
        _request_body_for_screen(coordinator, claim),
    )
    wire_path = tmp_path / "screen-request.json"
    wire_path.write_bytes(spool.spool_canonical_json(request.to_dict()) + b"\n")
    publication_path = tmp_path / "candidate-publication.tar"
    _publication_tar(claim.publication, publication_path)
    request_id, job_dir = spool.enqueue_request(
        registration,
        _lease_dict(claim.lease),
        (
            ("screen_payload", wire_path),
            ("candidate_publication", publication_path),
        ),
        tmp_path / "outbox",
        deadline_seconds=100,
        identity=identity,
        credential=credential,
    )
    return (
        coordinator,
        claim,
        service,
        credential,
        identity,
        registration,
        request,
        request_id,
        job_dir,
    )


def _lease_dict(lease) -> dict[str, object]:
    return {
        "claimed_block": lease.claimed_block,
        "expires_block": lease.expires_block,
        "generation": lease.generation,
        "initial_expires_block": lease.initial_expires_block,
        "lease_id": lease.lease_id,
        "members": [row.to_dict() for row in lease.members],
        "owner": lease.owner,
        "stage": lease.stage,
    }


def _publication_tar(publication, destination: Path) -> None:
    manifest = (
        spool.spool_canonical_json(
            {
                "publication": publication.to_dict(),
                "schema": "cacheon-remote-worker-publication-v1",
            }
        )
        + b"\n"
    )
    with tarfile.open(destination, "w") as archive:
        archive.addfile(
            spool.tar_info("publication.json", len(manifest)), io.BytesIO(manifest)
        )
        native = (publication.root / spool.NATIVE_ARTIFACT_MANIFEST).read_bytes()
        archive.addfile(
            spool.tar_info(
                f"bundle/{spool.NATIVE_ARTIFACT_MANIFEST}", len(native)
            ),
            io.BytesIO(native),
        )
        for row in publication.files:
            data = publication.root.joinpath(*Path(row.path).parts).read_bytes()
            archive.addfile(
                spool.tar_info(f"bundle/{row.path}", len(data)), io.BytesIO(data)
            )


def test_spool_digest_matches_deployed_semantic_envelope() -> None:
    domain = "cacheon.chain.remote-evaluation-request.v1"
    payload = {"b": ["x", 2], "a": {"nested": True, "n": None}}
    literal = hashlib.sha256(
        canonical_json_bytes(
            {"domain": domain, "payload": payload, "schema_version": 1}
        )
    ).hexdigest()
    assert spool.spool_digest(domain, payload) == literal


def test_registration_typed_identities_reopen_exactly(tmp_path: Path) -> None:
    (
        coordinator,
        claim,
        _service,
        credential,
        identity,
        registration,
        _request,
        _request_id,
        _job_dir,
    ) = _screen_authority(tmp_path)
    try:
        reopened_identity = registration_module.registration_transport_identity(
            registration
        )
        assert reopened_identity == identity
        reopened_credential = registration_module.registration_credential(
            registration, Path(registration["credential_path"])
        )
        assert reopened_credential.digest == credential.digest
        readiness_value, readiness_digest = registration_module.verify_readiness(
            registration["worker_readiness"]
        )
        assert readiness_digest == coordinator.readiness.digest
        assert readiness_value == coordinator.readiness.to_dict()
        assert (
            registration_module.registration_is_current(
                registration, Path("/nonexistent/registration.json")
            )
            is False
        )
        mutated = dict(registration)
        mutated["pod_host"] = "other.example"
        with pytest.raises(spool.RemoteWorkerError, match="digest mismatch"):
            registration_module.verify_registration(mutated)
    finally:
        coordinator._release(claim.lease, reason="test_cleanup")


def test_spool_screen_request_and_response_are_exact_authenticated_authority(
    tmp_path: Path,
) -> None:
    (
        coordinator,
        claim,
        service,
        credential,
        identity,
        registration,
        request,
        request_id,
        job_dir,
    ) = _screen_authority(tmp_path)
    outer = spool.verify_request(
        spool.load_json(job_dir / "request.json"),
        job_dir,
        registration,
        identity=identity,
        credential=credential,
    )
    assert outer["request_id"] == request_id
    assert outer["lease"]["lease_id"] == claim.lease.lease_id

    receipt = service.screen(claim.candidate)
    response = seal_remote_response(request, receipt, identity, credential)
    result_root = tmp_path / "result"
    result_root.mkdir()
    (result_root / "response.json").write_bytes(
        spool.spool_canonical_json(response.to_dict()) + b"\n"
    )
    spool.finalize_adapter_response(
        outer, job_dir, result_root, identity=identity, credential=credential
    )
    result = spool.verify_adapter_result(
        spool.load_json(result_root / "result.json"),
        result_root,
        outer,
        registration,
        request_root=job_dir,
        identity=identity,
        credential=credential,
    )
    assert result["state"] == "completed"
    assert result["response_digest"] == response.digest
    queue = spool.iter_queue(
        tmp_path / "outbox", registration, identity=identity, credential=credential
    )
    assert [row[1]["request_id"] for row in queue] == [request_id]
    coordinator._release(claim.lease, reason="test_cleanup")


def test_spool_rejects_forged_request_hmac(tmp_path: Path) -> None:
    (
        coordinator,
        claim,
        _service,
        credential,
        identity,
        registration,
        _request,
        _request_id,
        job_dir,
    ) = _screen_authority(tmp_path)
    outer = spool.load_json(job_dir / "request.json")
    payload = spool.artifact_for_role(outer, job_dir, "screen_payload")
    value = spool.load_json(payload)
    value["auth_tag"] = "f" * 64
    payload.chmod(0o600)
    payload.write_bytes(spool.spool_canonical_json(value) + b"\n")
    artifact = next(
        row for row in outer["artifacts"] if row["role"] == "screen_payload"
    )
    artifact["sha256"] = spool.file_sha256(payload)
    artifact["size"] = payload.stat().st_size
    renamed = job_dir / "blobs" / artifact["sha256"]
    payload.rename(renamed)
    unsigned = dict(outer)
    unsigned.pop("request_id")
    outer["request_id"] = spool.spool_digest(spool.DOMAIN_REQUEST, unsigned)
    with pytest.raises(spool.RemoteWorkerError, match="HMAC"):
        spool.verify_request(
            outer, job_dir, registration, identity=identity, credential=credential
        )
    coordinator._release(claim.lease, reason="test_cleanup")


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as handle:
        payload = b"bad"
        member = tarfile.TarInfo("../outside")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(spool.RemoteWorkerError, match="unsafe member"):
        spool.safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_wire_bodies_reject_command_surfaces() -> None:
    assert spool.contains_command_surface({"outer": [{"argv": ["x"]}]}) is True
    assert spool.contains_command_surface({"outer": [{"role": "screen"}]}) is False
    assert (
        registration_module.REMOTE_EVALUATION_PROTOCOL_DIGEST
        == REMOTE_EVALUATION_PROTOCOL_DIGEST
    )


def test_verify_lease_enforces_exact_stage_membership() -> None:
    member = {"prior_status": "published", "reservation_id": "1" * 64}
    base = {
        "claimed_block": 10,
        "expires_block": 30,
        "generation": 1,
        "initial_expires_block": 30,
        "lease_id": "a" * 64,
        "members": [member],
        "owner": "operator-a",
        "stage": "screen",
    }
    assert spool.verify_lease(dict(base)) == base
    two_members = dict(base)
    two_members["members"] = [
        member,
        {"prior_status": "published", "reservation_id": "2" * 64},
    ]
    with pytest.raises(spool.RemoteWorkerError, match="lease projection"):
        spool.verify_lease(two_members)
    unpromoted = dict(base)
    unpromoted["stage"] = "qualification"
    with pytest.raises(spool.RemoteWorkerError, match="lease projection"):
        spool.verify_lease(unpromoted)
    promoted = dict(unpromoted)
    promoted["members"] = [
        {"prior_status": "promoted", "reservation_id": "1" * 64}
    ]
    assert spool.verify_lease(promoted) == promoted


def test_heartbeat_roundtrip_binding_and_liveness(tmp_path: Path) -> None:
    registration = {
        "ready_receipt_digest": "a" * 64,
        "worker_epoch": "b" * 32,
        "worker_readiness_digest": "c" * 64,
    }
    heartbeat = spool.heartbeat_payload(
        registration,
        "running",
        None,
        adapter_start_count=1,
        adapter_alive=True,
        consecutive_adapter_failures=0,
    )
    assert spool.verify_heartbeat(heartbeat, registration, 30) == heartbeat
    other = {**registration, "worker_epoch": "e" * 32}
    with pytest.raises(spool.RemoteWorkerError, match="registration binding"):
        spool.verify_heartbeat(heartbeat, other, 30)
    stale = dict(heartbeat)
    unsigned = dict(stale)
    unsigned.pop("heartbeat_digest")
    unsigned["time_unix"] = int(time.time()) - 3600
    stale = {
        **unsigned,
        "heartbeat_digest": spool.spool_digest(spool.DOMAIN_HEARTBEAT, unsigned),
    }
    with pytest.raises(spool.RemoteWorkerError, match="liveness bound"):
        spool.verify_heartbeat(stale, registration, 30)
    unstarted = dict(heartbeat)
    unsigned = dict(unstarted)
    unsigned.pop("heartbeat_digest")
    unsigned["adapter_start_count"] = 0
    unstarted = {
        **unsigned,
        "heartbeat_digest": spool.spool_digest(spool.DOMAIN_HEARTBEAT, unsigned),
    }
    with pytest.raises(spool.RemoteWorkerError, match="unstarted live adapter"):
        spool.verify_heartbeat(unstarted, registration, 30)


def test_result_ready_receipt_binds_request_and_epoch() -> None:
    registration = {"worker_epoch": "b" * 32}
    request = {
        "request_id": "1" * 64,
        "ready_receipt_digest": "a" * 64,
        "worker_epoch": "b" * 32,
        "worker_readiness_digest": "c" * 64,
    }
    unsigned = {
        "archive_sha256": "d" * 64,
        "archive_size": 100,
        "ready_receipt_digest": request["ready_receipt_digest"],
        "request_id": request["request_id"],
        "schema": spool.SCHEMA_RESULT_READY,
        "state": "ready",
        "worker_epoch": request["worker_epoch"],
        "worker_readiness_digest": request["worker_readiness_digest"],
    }
    ready = {
        **unsigned,
        "ready_digest": spool.spool_digest(spool.DOMAIN_RESULT_READY, unsigned),
    }
    assert spool.verify_result_ready(ready, request, registration) == ready
    rebound = dict(unsigned)
    rebound["request_id"] = "2" * 64
    rebound = {
        **rebound,
        "ready_digest": spool.spool_digest(spool.DOMAIN_RESULT_READY, rebound),
    }
    with pytest.raises(spool.RemoteWorkerError, match="changed request binding"):
        spool.verify_result_ready(rebound, request, registration)


def test_local_no_decision_result_is_closed(tmp_path: Path) -> None:
    (
        coordinator,
        claim,
        _service,
        credential,
        identity,
        registration,
        _request,
        request_id,
        job_dir,
    ) = _screen_authority(tmp_path)
    try:
        outer = spool.load_json(job_dir / "request.json")
        results_root = tmp_path / "results"
        results_root.mkdir()
        spool.write_local_no_decision(results_root, outer, "request_deadline_elapsed")
        result = spool.verify_adapter_result(
            spool.load_json(results_root / request_id / "result.json"),
            results_root / request_id,
            outer,
            registration,
            request_root=job_dir,
            identity=identity,
            credential=credential,
        )
        assert result["state"] == "no_decision"
        assert result["failure_code"] == "request_deadline_elapsed"
        with pytest.raises(spool.RemoteWorkerError, match="not registered"):
            spool.write_local_no_decision(results_root, outer, "made_up_code")
    finally:
        coordinator._release(claim.lease, reason="test_cleanup")


def test_make_registration_binds_ready_receipt_and_reopens(tmp_path: Path) -> None:
    fixtures = _dispatcher_fixtures()
    fixtures._published_rows(tmp_path, 1)
    service = ArenaService(fixtures._manifest(), fixtures._Provider())
    cursor = fixtures._Cursor((fixtures.BLOCK, fixtures._block_hash(fixtures.BLOCK)))
    coordinator = fixtures._coordinator(tmp_path, service, cursor)
    unsigned = {
        "base_image": "img",
        "build": {},
        "created_at": "2026-08-06T00:00:00Z",
        "gpu": {"count": 8, "inventory": [{"name": "NVIDIA B300"}] * 8},
        "model": {},
        "provider": {"pod_endpoint": "unknown"},
        "runtime_seed": "seed",
        "schema": "cacheon-lium-worker-ready-v1",
        "source": {},
        "state": "READY_FOR_REGISTRATION",
        "venv": {},
        "worker_epoch": "f" * 32,
        "worker_image": "img",
    }
    receipt_digest = hashlib.sha256(
        b"cacheon.lium-worker-ready.v1\0" + canonical_json_bytes(unsigned)
    ).hexdigest()
    ready = {**unsigned, "receipt_digest": receipt_digest}
    ready_path = tmp_path / "ready-receipt.json"
    ready_path.write_bytes(spool.spool_canonical_json(ready) + b"\n")
    readiness_value = {
        **coordinator.readiness.to_dict(),
        "ready_receipt_digest": "0" * 64,
    }
    readiness_path = tmp_path / "worker-readiness.json"
    readiness_path.write_bytes(spool.spool_canonical_json(readiness_value) + b"\n")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("pinned-host-key\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    remote_service = tmp_path / "remote_worker_service.py"
    remote_service.write_text("service", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.write_text("adapter", encoding="utf-8")
    credential = tmp_path / "credential.secret"
    credential.write_bytes(b"s" * 32)
    output = tmp_path / "registration.json"
    value = registration_module.make_registration(
        ready_receipt=ready_path,
        worker_readiness=readiness_path,
        known_hosts=known_hosts,
        pod_host="pod.example",
        pod_port=22,
        service_identity=service.manifest.service_id,
        remote_service=remote_service,
        adapter=adapter,
        credential=credential,
        credential_id="screen-key-v1",
        output=output,
        python_executable=sys.executable,
        lane_devices=",".join(
            str(device) for device in range(coordinator.readiness.gpu_count)
        ),
        bind_ready_receipt=True,
        transport_id="test-worker-1",
    )
    assert value["worker_epoch"] == "f" * 32
    assert value["worker_readiness"]["ready_receipt_digest"] == receipt_digest
    reopened = registration_module.verify_registration(spool.load_json(output))
    assert reopened == value
    assert registration_module.registration_is_current(value, output) is True

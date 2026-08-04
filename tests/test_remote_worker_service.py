from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteWorkerCredential,
    _request_body_for_screen,
    seal_remote_request,
    seal_remote_response,
)
from chainops import remote_worker_service as worker


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
    cursor = fixtures._Cursor(
        (fixtures.BLOCK, fixtures._block_hash(fixtures.BLOCK))
    )
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
        "credential_file_sha256": worker.file_sha256(secret),
        "credential_id": credential.credential_id,
        "credential_path": str(secret),
        "known_hosts_path": str(known_hosts),
        "known_hosts_sha256": worker.file_sha256(known_hosts),
        "lane_devices": list(range(coordinator.readiness.gpu_count)),
        "lane_digest": "e" * 64,
        "pod_host": "pod.example",
        "pod_port": 22,
        "pod_user": "root",
        "python_executable": sys.executable,
        "python_executable_sha256": worker.file_sha256(
            Path(sys.executable).resolve()
        ),
        "ready_receipt_digest": coordinator.readiness.ready_receipt_digest,
        "ready_receipt_file_sha256": "b" * 64,
        "remote_service_sha256": "c" * 64,
        "schema": worker.SCHEMA_REGISTRATION,
        "service_identity": service.manifest.service_id,
        "transport_identity": identity.to_dict(),
        "transport_identity_digest": identity.digest,
        "worker_epoch": "d" * 32,
        "worker_readiness": coordinator.readiness.to_dict(),
        "worker_readiness_digest": coordinator.readiness.digest,
    }
    registration["registration_digest"] = worker.semantic_digest(
        worker.DOMAIN_REGISTRATION, registration
    )
    worker.verify_registration(registration)
    request = seal_remote_request(
        claim.lease,
        coordinator.readiness,
        service.manifest.service_id,
        identity,
        credential,
        _request_body_for_screen(coordinator, claim),
    )
    wire_path = tmp_path / "screen-request.json"
    wire_path.write_bytes(worker.canonical_json_bytes(request.to_dict()) + b"\n")
    publication_path = tmp_path / "candidate-publication.tar"
    worker._publication_archive(claim.publication, publication_path)
    request_id, job_dir = worker.enqueue_request(
        registration,
        worker.DurableSpoolAuthenticatedWorkerTransport._lease_dict(claim.lease),
        (
            ("screen_payload", wire_path),
            ("candidate_publication", publication_path),
        ),
        tmp_path / "outbox",
        deadline_seconds=100,
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
    outer = worker.verify_request(
        worker.load_json(job_dir / "request.json"), job_dir, registration
    )
    assert outer["request_id"] == request_id
    assert outer["lease"]["lease_id"] == claim.lease.lease_id

    receipt = service.screen(claim.candidate)
    response = seal_remote_response(request, receipt, identity, credential)
    result_root = tmp_path / "result"
    result_root.mkdir()
    (result_root / "response.json").write_bytes(
        worker.canonical_json_bytes(response.to_dict()) + b"\n"
    )
    old_credential = worker.POD_CREDENTIAL
    worker.POD_CREDENTIAL = Path(registration["credential_path"])
    try:
        worker.finalize_adapter_response(
            registration, outer, job_dir, result_root
        )
    finally:
        worker.POD_CREDENTIAL = old_credential
    result = worker.verify_adapter_result(
        worker.load_json(result_root / "result.json"),
        result_root,
        outer,
        registration,
        request_root=job_dir,
    )
    assert result["state"] == "completed"
    assert result["response_digest"] == response.digest
    coordinator._release(claim.lease, reason="test_cleanup")


def test_spool_rejects_forged_request_hmac(tmp_path: Path) -> None:
    *prefix, job_dir = _screen_authority(tmp_path)
    coordinator, claim, *_rest = prefix
    registration = prefix[5]
    outer = worker.load_json(job_dir / "request.json")
    payload = worker._artifact_for_role(outer, job_dir, "screen_payload")
    value = worker.load_json(payload)
    value["auth_tag"] = "f" * 64
    payload.chmod(0o600)
    payload.write_bytes(worker.canonical_json_bytes(value) + b"\n")
    artifact = next(
        row for row in outer["artifacts"] if row["role"] == "screen_payload"
    )
    artifact["sha256"] = worker.file_sha256(payload)
    artifact["size"] = payload.stat().st_size
    renamed = job_dir / "blobs" / artifact["sha256"]
    payload.rename(renamed)
    unsigned = dict(outer)
    unsigned.pop("request_id")
    outer["request_id"] = worker.semantic_digest(worker.DOMAIN_REQUEST, unsigned)
    with pytest.raises(worker.RemoteWorkerError, match="HMAC"):
        worker.verify_request(outer, job_dir, registration)
    coordinator._release(claim.lease, reason="test_cleanup")


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as handle:
        payload = b"bad"
        member = tarfile.TarInfo("../outside")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(worker.RemoteWorkerError, match="unsafe member"):
        worker.safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_local_protocol_digest_matches_typed_dispatcher() -> None:
    assert worker.REMOTE_EVALUATION_PROTOCOL_DIGEST == REMOTE_EVALUATION_PROTOCOL_DIGEST
    parser = worker.build_parser()
    assert "command" not in {action.dest for action in parser._actions}

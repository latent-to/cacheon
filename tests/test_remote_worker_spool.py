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
from cacheon.chain import remote_qualification_hold as remote_hold
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    seal_remote_response,
)
from cacheon.chain.remote_worker_request_plan import _lease_dict
from cacheon.stack_identity import canonical_json_bytes
from cacheon.eval.remote_run_forensics import append_event as append_run_event, journal_path
from tests.test_remote_worker_request_plan import _authority


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


def _qualification_authority(tmp_path: Path):
    authority = _authority(tmp_path)
    request_id, job_dir = spool.enqueue_request(
        authority.registration,
        _lease_dict(authority.claim.lease),
        authority.inputs,
        tmp_path / "outbox",
        deadline_seconds=100,
        identity=authority.identity,
        credential=authority.credential,
    )
    return (
        authority.coordinator,
        authority.claim,
        authority.service,
        authority.credential,
        authority.identity,
        authority.registration,
        authority.request,
        request_id,
        job_dir,
    )


def _hold(request) -> remote_hold.RemoteQualificationHoldProduct:
    return remote_hold.capture_remote_qualification_hold(
        request,
        reason=remote_hold.RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE,
        diagnostic_digest="d" * 64,
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
        _claim,
        _service,
        credential,
        identity,
        registration,
        _request,
        _request_id,
        _job_dir,
    ) = _qualification_authority(tmp_path)
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


def test_spool_qualification_request_and_response_are_exact_authenticated_authority(
    tmp_path: Path,
) -> None:
    (
        _coordinator,
        claim,
        _service,
        credential,
        identity,
        registration,
        request,
        request_id,
        job_dir,
    ) = _qualification_authority(tmp_path)
    outer = spool.verify_request(
        spool.load_json(job_dir / "request.json"),
        job_dir,
        registration,
        identity=identity,
        credential=credential,
    )
    assert outer["request_id"] == request_id
    assert outer["lease"]["lease_id"] == claim.lease.lease_id

    response = seal_remote_response(request, _hold(request), identity, credential)
    result_root = tmp_path / "result"
    result_root.mkdir()
    (result_root / "response.json").write_bytes(
        spool.spool_canonical_json(response.to_dict()) + b"\n"
    )
    append_run_event(
        journal_path(result_root), request_id, "adapter.terminal", "completed"
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


def test_spool_completed_payload_stage_algebra_is_exact_and_closed(
    tmp_path: Path,
) -> None:
    request = _qualification_authority(tmp_path)[6]
    hold = _hold(request)
    assert remote_hold.is_exact_remote_stage_payload(hold, "qualification")
    assert not remote_hold.is_exact_remote_stage_payload(hold, "screen")
    assert not remote_hold.is_exact_remote_stage_payload(request, "qualification")
    assert not remote_hold.is_exact_remote_stage_payload(object(), "qualification")
    assert not remote_hold.is_exact_remote_stage_payload(object(), "screen")
    assert not remote_hold.is_exact_remote_stage_payload(hold, "unknown")


def test_spool_rejects_forged_request_hmac(tmp_path: Path) -> None:
    (
        _coordinator,
        _claim,
        _service,
        credential,
        identity,
        registration,
        _request,
        _request_id,
        job_dir,
    ) = _qualification_authority(tmp_path)
    outer = spool.load_json(job_dir / "request.json")
    payload = spool.artifact_for_role(outer, job_dir, "qualification_payload")
    value = spool.load_json(payload)
    value["auth_tag"] = "f" * 64
    payload.chmod(0o600)
    payload.write_bytes(spool.spool_canonical_json(value) + b"\n")
    artifact = next(
        row for row in outer["artifacts"] if row["role"] == "qualification_payload"
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
    assert spool.contains_command_surface({"outer": [{"role": "qualification"}]}) is False
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
        "stage": "qualification",
    }
    assert spool.verify_lease(dict(base)) == base
    cohort = dict(base)
    cohort["members"] = [
        member,
        {"prior_status": "published", "reservation_id": "2" * 64},
    ]
    assert spool.verify_lease(cohort) == cohort
    retired_stage = dict(base)
    retired_stage["stage"] = "screen"
    with pytest.raises(spool.RemoteWorkerError, match="lease projection"):
        spool.verify_lease(retired_stage)
    promoted = dict(base)
    promoted["members"] = [
        {"prior_status": "promoted", "reservation_id": "1" * 64}
    ]
    with pytest.raises(spool.RemoteWorkerError, match="lease projection"):
        spool.verify_lease(promoted)


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
        _coordinator,
        _claim,
        _service,
        credential,
        identity,
        registration,
        _request,
        request_id,
        job_dir,
    ) = _qualification_authority(tmp_path)
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
        credential_id="qualification-key-v1",
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

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain import remote_worker_pod_service as pod_service
from cacheon.chain import remote_worker_registration as registration_module
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteWorkerCredential,
    _request_body_for_screen,
    seal_remote_request,
    seal_remote_response,
)
from cacheon.chain.execution_disposition import reopen_pre_resident_refusal
from cacheon.chain.remote_worker_execution_marker import (
    RESIDENT_ENTRY_MARKER,
    publish_resident_entry,
)
from cacheon.chain.remote_worker_registration import PodPaths
from cacheon.eval.remote_run_forensics import append_event as append_run_event, journal_path
from cacheon.stack_identity import sha256_hex
from tests import test_remote_evaluation_dispatcher as dispatcher_fixtures


def _digest(label: str) -> str:
    return sha256_hex(label.encode("utf-8"))


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


@dataclass
class RecoveryAuthority:
    credential: Any
    identity: Any
    registration: dict[str, Any]
    outer_request: dict[str, Any]
    request_id: str
    job_root: Path
    paths: PodPaths
    response_bytes: bytes


@pytest.fixture
def authority(tmp_path: Path):
    dispatcher_fixtures._published_rows(tmp_path, 1)
    service = ArenaService(
        dispatcher_fixtures._manifest(), dispatcher_fixtures._Provider()
    )
    cursor = dispatcher_fixtures._Cursor(
        (
            dispatcher_fixtures.BLOCK,
            dispatcher_fixtures._block_hash(dispatcher_fixtures.BLOCK),
        )
    )
    coordinator = dispatcher_fixtures._coordinator(tmp_path, service, cursor)
    claim = coordinator.claim_screen()
    assert claim is not None

    secret_bytes = hashlib.sha256(b"pod-result-recovery-test-secret").digest()
    credential = RemoteWorkerCredential("recovery-test-key", secret_bytes)
    identity = dispatcher_fixtures._transport_identity(coordinator, credential)
    secret = tmp_path / "credential.secret"
    secret.write_bytes(secret_bytes)
    secret.chmod(0o400)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test-host-key\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    lane_devices = list(range(coordinator.readiness.gpu_count))
    registration = {
        "adapter_sha256": _digest("adapter bytes"),
        "created_at_unix": int(time.time()),
        "credential_digest": credential.digest,
        "credential_file_sha256": spool.file_sha256(secret),
        "credential_id": credential.credential_id,
        "credential_path": str(secret),
        "known_hosts_path": str(known_hosts),
        "known_hosts_sha256": spool.file_sha256(known_hosts),
        "lane_devices": lane_devices,
        "lane_digest": sha256_hex(spool.spool_canonical_json(lane_devices)),
        "pod_host": "pod.test.invalid",
        "pod_port": 22,
        "pod_user": "root",
        "python_executable": sys.executable,
        "python_executable_sha256": spool.file_sha256(Path(sys.executable).resolve()),
        "ready_receipt_digest": coordinator.readiness.ready_receipt_digest,
        "ready_receipt_file_sha256": _digest("ready receipt bytes"),
        "remote_service_sha256": _digest("remote service bytes"),
        "schema": spool.SCHEMA_REGISTRATION,
        "service_identity": service.manifest.service_id,
        "transport_identity": identity.to_dict(),
        "transport_identity_digest": identity.digest,
        "worker_epoch": _digest("worker epoch")[:32],
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
    request_id, queued = spool.enqueue_request(
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

    pod_root = tmp_path / "pod"
    paths = PodPaths(
        root=pod_root,
        ready_receipt=pod_root / "ready-receipt.json",
        registration=pod_root / "registration.json",
        service=pod_root / "remote_worker_service.py",
        adapter=pod_root / "adapter",
        credential=secret,
    )
    for name in ("completed", "outgoing", "processing", "results"):
        (pod_root / name).mkdir(parents=True, exist_ok=True)
    job_root = pod_root / "processing" / request_id
    os.replace(queued, job_root)
    outer_request = spool.verify_request(
        spool.load_json(job_root / "request.json"),
        job_root,
        registration,
        identity=identity,
        credential=credential,
    )
    receipt = service.screen(claim.candidate)
    response = seal_remote_response(request, receipt, identity, credential)
    value = RecoveryAuthority(
        credential=credential,
        identity=identity,
        registration=registration,
        outer_request=outer_request,
        request_id=request_id,
        job_root=job_root,
        paths=paths,
        response_bytes=spool.spool_canonical_json(response.to_dict()) + b"\n",
    )
    try:
        yield value
    finally:
        coordinator._release(claim.lease, reason="test_cleanup")


def _temporary(authority: RecoveryAuthority, offset: int = 0) -> Path:
    path = (
        authority.paths.root
        / "results"
        / f".{authority.request_id}.{os.getpid() + offset}"
    )
    path.mkdir()
    return path


def _write_response(authority: RecoveryAuthority, root: Path) -> None:
    append_run_event(
        journal_path(root), authority.request_id, "adapter.terminal", "completed"
    )
    (root / "response.json").write_bytes(authority.response_bytes)


def _finalize(authority: RecoveryAuthority, root: Path) -> None:
    spool.finalize_adapter_response(
        authority.outer_request,
        authority.job_root,
        root,
        identity=authority.identity,
        credential=authority.credential,
    )
    spool.verify_adapter_result(
        spool.load_json(root / "result.json"),
        root,
        authority.outer_request,
        authority.registration,
        request_root=authority.job_root,
        identity=authority.identity,
        credential=authority.credential,
    )


def _events(authority: RecoveryAuthority) -> list[dict[str, Any]]:
    path = authority.paths.root / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _forbid_new_work(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {"adapter": 0, "infrastructure": 0}

    def adapter(*_args, **_kwargs):
        calls["adapter"] += 1
        raise AssertionError("recovery must not execute the adapter")

    def infrastructure(*_args, **_kwargs):
        calls["infrastructure"] += 1
        raise AssertionError("recovery must not manufacture a no-decision")

    monkeypatch.setattr(pod_service, "run_adapter", adapter)
    monkeypatch.setattr(pod_service.PersistentAdapterProcess, "evaluate", adapter)
    monkeypatch.setattr(pod_service, "infrastructure_result", infrastructure)
    return calls


def _recover(authority: RecoveryAuthority) -> None:
    pod_service.recover_interrupted(
        authority.registration,
        authority.paths,
        identity=authority.identity,
        credential=authority.credential,
    )


def _assert_published(authority: RecoveryAuthority) -> None:
    final = authority.paths.root / "results" / authority.request_id
    verified = spool.verify_adapter_result(
        spool.load_json(final / "result.json"),
        final,
        authority.outer_request,
        authority.registration,
        request_root=authority.paths.root / "completed" / authority.request_id,
        identity=authority.identity,
        credential=authority.credential,
    )
    assert verified["state"] == "completed"
    ready_path = authority.paths.root / "outgoing" / f"{authority.request_id}.ready.json"
    ready = spool.verify_result_ready(
        spool.load_json(ready_path), authority.outer_request, authority.registration
    )
    archive = (
        authority.paths.root
        / "outgoing"
        / f"{authority.request_id}.{ready['archive_sha256']}.tar"
    )
    assert archive.is_file()
    assert archive.stat().st_size == ready["archive_size"]
    assert not authority.job_root.exists()
    assert (authority.paths.root / "completed" / authority.request_id).is_dir()


def _assert_hold(authority: RecoveryAuthority, reason: str) -> None:
    assert authority.job_root.is_dir()
    assert not (authority.paths.root / "completed" / authority.request_id).exists()
    assert not (
        authority.paths.root / "outgoing" / f"{authority.request_id}.ready.json"
    ).exists()
    hold = _events(authority)[-1]
    assert set(hold) == {"event", "reason", "request_id", "state", "time_unix"}
    assert hold["event"] == "recovery_hold"
    assert hold["reason"] == reason
    assert hold["request_id"] == authority.request_id
    assert hold["state"] == "processing"
    assert type(hold["time_unix"]) is int


def test_completed_response_finalizes_once_and_recovery_is_idempotent(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = _temporary(authority)
    _write_response(authority, temporary)
    calls = _forbid_new_work(monkeypatch)
    original_finalize = pod_service.finalize_adapter_response
    original_publish = pod_service.publish_result
    finalize_calls = 0

    def counted_finalize(*args, **kwargs):
        nonlocal finalize_calls
        finalize_calls += 1
        return original_finalize(*args, **kwargs)

    def crash_publish(*_args, **_kwargs):
        raise RuntimeError("simulated crash before publication")

    monkeypatch.setattr(pod_service, "finalize_adapter_response", counted_finalize)
    monkeypatch.setattr(pod_service, "publish_result", crash_publish)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _recover(authority)
    assert finalize_calls == 1
    assert not temporary.exists()
    assert authority.job_root.is_dir()

    monkeypatch.setattr(pod_service, "publish_result", original_publish)
    _recover(authority)
    _assert_published(authority)
    assert finalize_calls == 1
    assert calls == {"adapter": 0, "infrastructure": 0}
    events = _events(authority)

    _recover(authority)
    assert finalize_calls == 1
    assert _events(authority) == events
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_completed_result_promotes_without_refinalizing(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = _temporary(authority)
    _write_response(authority, temporary)
    _finalize(authority, temporary)
    result_bytes = (temporary / "result.json").read_bytes()
    calls = _forbid_new_work(monkeypatch)
    results_info = (authority.paths.root / "results").stat()
    real_fsync = os.fsync
    fsynced_results = 0

    def track_fsync(descriptor: int) -> None:
        nonlocal fsynced_results
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) == (
            results_info.st_dev,
            results_info.st_ino,
        ):
            fsynced_results += 1
        real_fsync(descriptor)

    monkeypatch.setattr(pod_service.os, "fsync", track_fsync)
    monkeypatch.setattr(
        pod_service,
        "finalize_adapter_response",
        lambda *_args, **_kwargs: pytest.fail("completed result must not be finalized"),
    )

    _recover(authority)

    final = authority.paths.root / "results" / authority.request_id
    assert (final / "result.json").read_bytes() == result_bytes
    assert not temporary.exists()
    _assert_published(authority)
    assert fsynced_results == 1
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_exact_final_result_reopens_without_adapter_or_finalization(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = authority.paths.root / "results" / authority.request_id
    final.mkdir()
    _write_response(authority, final)
    _finalize(authority, final)
    result_bytes = (final / "result.json").read_bytes()
    calls = _forbid_new_work(monkeypatch)
    monkeypatch.setattr(
        pod_service,
        "finalize_adapter_response",
        lambda *_args, **_kwargs: pytest.fail("final result must only be reopened"),
    )

    _recover(authority)

    assert (final / "result.json").read_bytes() == result_bytes
    _assert_published(authority)
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_symlinked_final_result_json_holds_without_following(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = authority.paths.root / "results" / authority.request_id
    final.mkdir()
    outside = authority.paths.root.parent / "unrelated-final-result.json"
    outside.write_bytes(authority.response_bytes)
    result_json = final / "result.json"
    result_json.symlink_to(outside)
    calls = _forbid_new_work(monkeypatch)

    with pytest.raises(spool.RemoteWorkerError, match="invalid_final_result"):
        _recover(authority)

    assert result_json.is_symlink()
    assert outside.read_bytes() == authority.response_bytes
    _assert_hold(authority, "invalid_final_result")
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_missing_temporary_result_holds_processing(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _forbid_new_work(monkeypatch)
    with pytest.raises(spool.RemoteWorkerError, match="missing_temporary_result"):
        _recover(authority)
    _assert_hold(authority, "missing_temporary_result")
    assert not (authority.paths.root / "results" / authority.request_id).exists()
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_partial_temporary_result_holds_without_deleting_evidence(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = _temporary(authority)
    evidence = temporary / "adapter.log"
    evidence.write_bytes(b"resident work began\n")
    calls = _forbid_new_work(monkeypatch)

    with pytest.raises(spool.RemoteWorkerError, match="partial_temporary_result"):
        _recover(authority)

    assert evidence.read_bytes() == b"resident work began\n"
    _assert_hold(authority, "partial_temporary_result")
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_invalid_authenticated_response_holds_without_mutation_or_deletion(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = _temporary(authority)
    forged = json.loads(authority.response_bytes)
    forged["auth_tag"] = _digest("forged response authentication tag")
    raw = spool.spool_canonical_json(forged) + b"\n"
    response_path = temporary / "response.json"
    response_path.write_bytes(raw)
    calls = _forbid_new_work(monkeypatch)

    with pytest.raises(spool.RemoteWorkerError, match="invalid_temporary_product"):
        _recover(authority)

    assert response_path.read_bytes() == raw
    assert tuple(path.name for path in temporary.iterdir()) == ("response.json",)
    _assert_hold(authority, "invalid_temporary_product")
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_ambiguous_temporary_results_hold_before_inspection(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _temporary(authority)
    second = _temporary(authority, 1)
    _write_response(authority, first)
    marker = second / "partial.marker"
    marker.write_bytes(b"keep both candidates\n")
    calls = _forbid_new_work(monkeypatch)

    with pytest.raises(spool.RemoteWorkerError, match="ambiguous_temporary_results"):
        _recover(authority)

    assert (first / "response.json").read_bytes() == authority.response_bytes
    assert marker.read_bytes() == b"keep both candidates\n"
    _assert_hold(authority, "ambiguous_temporary_results")
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_exactly_named_symlink_is_not_followed_and_holds(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = authority.paths.root.parent / "unrelated-result-directory"
    outside.mkdir()
    sentinel = outside / "response.json"
    sentinel.write_bytes(authority.response_bytes)
    temporary = (
        authority.paths.root
        / "results"
        / f".{authority.request_id}.{os.getpid()}"
    )
    temporary.symlink_to(outside, target_is_directory=True)
    calls = _forbid_new_work(monkeypatch)

    with pytest.raises(spool.RemoteWorkerError, match="invalid_temporary_entry"):
        _recover(authority)

    assert temporary.is_symlink()
    assert sentinel.read_bytes() == authority.response_bytes
    _assert_hold(authority, "invalid_temporary_entry")
    assert calls == {"adapter": 0, "infrastructure": 0}


@pytest.mark.parametrize("json_name", ["result.json", "response.json"])
def test_symlinked_temporary_product_json_is_rejected_without_following(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
    json_name: str,
) -> None:
    temporary = _temporary(authority)
    outside = authority.paths.root.parent / f"unrelated-{json_name}"
    outside.write_bytes(authority.response_bytes)
    product_path = temporary / json_name
    product_path.symlink_to(outside)
    calls = _forbid_new_work(monkeypatch)

    with pytest.raises(spool.RemoteWorkerError, match="invalid_temporary_product"):
        _recover(authority)

    assert product_path.is_symlink()
    assert outside.read_bytes() == authority.response_bytes
    _assert_hold(authority, "invalid_temporary_product")
    assert calls == {"adapter": 0, "infrastructure": 0}


def test_promotion_collision_fails_closed_and_retains_both_products(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = _temporary(authority)
    _write_response(authority, temporary)
    _finalize(authority, temporary)
    result_bytes = (temporary / "result.json").read_bytes()
    final = authority.paths.root / "results" / authority.request_id
    collision_marker = final / "collision.marker"
    calls = _forbid_new_work(monkeypatch)

    def collide(source_fd, source, destination_fd, destination):
        assert source_fd == destination_fd
        assert source == temporary.name
        assert destination == final.name
        final.mkdir()
        collision_marker.write_bytes(b"independent writer arrived\n")
        raise pod_service.NativeArtifactRaceError("simulated promotion collision")

    monkeypatch.setattr(pod_service, "_rename_noreplace", collide)
    with pytest.raises(spool.RemoteWorkerError, match="promotion_collision"):
        _recover(authority)

    assert (temporary / "result.json").read_bytes() == result_bytes
    assert collision_marker.read_bytes() == b"independent writer arrived\n"
    _assert_hold(authority, "promotion_collision")
    assert calls == {"adapter": 0, "infrastructure": 0}


class FailedAdapter:
    def __init__(self, failure: str, marker: str = "absent") -> None:
        self.failure, self.marker = failure, marker
        self.calls = 0
        self.marker_bytes: bytes | None = None

    def evaluate(self, request, _job_root, result_root, *, deadline):
        del deadline
        self.calls += 1
        marker_path = result_root / RESIDENT_ENTRY_MARKER
        if self.marker == "valid":
            publish_resident_entry(result_root, request)
            self.marker_bytes = marker_path.read_bytes()
        elif self.marker == "invalid":
            marker_path.write_bytes(b"invalid resident marker\n")
            marker_path.chmod(0o400)
            self.marker_bytes = marker_path.read_bytes()
        elif self.marker == "symlink":
            target = result_root.parent / f"{result_root.name}.marker-target"
            target.write_bytes(b"outside resident marker\n")
            marker_path.symlink_to(target)
            self.marker_bytes = target.read_bytes()
        return self.failure


def _run_failed_adapter(authority: RecoveryAuthority, process: FailedAdapter) -> Path:
    item = authority
    return pod_service.run_adapter(
        item.registration, item.paths, item.job_root, item.outer_request, 1,
        process,  # type: ignore[arg-type]
        identity=item.identity, credential=item.credential,
    )


def _forbid_infrastructure(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    def forbidden(_request, _root, failure, **_kwargs):
        calls.append(failure)
        raise AssertionError("post-resident failure cannot become infrastructure result")

    monkeypatch.setattr(pod_service, "infrastructure_result", forbidden)
    return calls


@pytest.mark.parametrize("failure", ["adapter_epoch_failed", "adapter_timeout"])
def test_post_resident_failure_publishes_diagnostic_and_preserves_marker(
    authority: RecoveryAuthority,
    failure: str,
) -> None:
    process = FailedAdapter(failure, "valid")
    result = _run_failed_adapter(authority, process)
    row = spool.load_json(result / "result.json")
    assert row["state"] == "no_decision" and row["failure_code"] == failure
    assert (result / RESIDENT_ENTRY_MARKER).read_bytes() == process.marker_bytes
    assert spool.artifact_for_role(row, result, "worker_log").is_file()
    assert process.calls == 1


@pytest.mark.parametrize("marker", ["invalid", "symlink"])
def test_invalid_resident_marker_holds_without_cleanup(
    authority: RecoveryAuthority, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    process = FailedAdapter("adapter_epoch_failed", marker)
    infrastructure = _forbid_infrastructure(monkeypatch)
    with pytest.raises(spool.RemoteWorkerError, match="invalid_resident_entry_marker"):
        _run_failed_adapter(authority, process)
    temporary = authority.paths.root / "results" / f".{authority.request_id}.{os.getpid()}"
    marker_path = temporary / RESIDENT_ENTRY_MARKER
    assert marker_path.read_bytes() == process.marker_bytes
    assert marker_path.is_symlink() is (marker == "symlink")
    _assert_hold(authority, "invalid_resident_entry_marker")
    assert process.calls == 1 and infrastructure == []


@pytest.mark.parametrize("failure", ["adapter_request_failed", "adapter_start_failed"])
def test_typed_pre_resident_failure_without_marker_remains_no_decision(
    authority: RecoveryAuthority,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    process = FailedAdapter(failure)
    original = pod_service.infrastructure_result
    infrastructure: list[str] = []
    def counted(request, root, observed, *, credential=None):
        infrastructure.append(observed)
        original(request, root, observed, credential=credential)

    monkeypatch.setattr(pod_service, "infrastructure_result", counted)
    result = _run_failed_adapter(authority, process)
    row = spool.load_json(result / "result.json")
    assert row["state"] == "no_decision" and row["failure_code"] == failure
    payload = spool.load_json(
        spool.artifact_for_role(row, result, "adapter_result")
    )
    refusal = reopen_pre_resident_refusal(
        payload,
        request_id=authority.request_id,
        worker_epoch=authority.registration["worker_epoch"],
        credential=authority.credential,
    )
    assert refusal.failure_code == failure
    assert process.calls == 1 and infrastructure == [failure]
    assert not (authority.paths.root / "events.jsonl").exists()


def test_request_failure_with_valid_marker_publishes_diagnostic(
    authority: RecoveryAuthority,
) -> None:
    process = FailedAdapter("adapter_request_failed", "valid")
    result = _run_failed_adapter(authority, process)
    row = spool.load_json(result / "result.json")
    assert row["state"] == "no_decision"
    assert row["failure_code"] == "adapter_request_failed"
    assert (result / RESIDENT_ENTRY_MARKER).read_bytes() == process.marker_bytes
    assert spool.artifact_for_role(row, result, "worker_log").is_file()
    assert process.calls == 1

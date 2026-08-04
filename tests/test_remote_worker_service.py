from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import time
import stat
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
from cacheon.chain.evaluation_coordinator import (
    EvaluationResultEnvelope,
    EvaluationRun,
)
from cacheon.chain.publication import reopen_worker_bundle
from chainops import cacheon_b300_evaluation_adapter as adapter
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


def test_publication_transport_reconstructs_reopenable_immutable_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        coordinator,
        claim,
        _service,
        _credential,
        _identity,
        registration,
        _request,
        _request_id,
        job_dir,
    ) = _screen_authority(tmp_path)
    try:
        outer = worker.verify_request(
            worker.load_json(job_dir / "request.json"), job_dir, registration
        )
        archive_path = worker._artifact_for_role(
            outer, job_dir, "candidate_publication"
        )
        with tarfile.open(archive_path, "r:") as archive:
            assert "bundle/.cacheon-native-artifact.json" in archive.getnames()

        publication_root = tmp_path / "pod-publications"
        monkeypatch.setattr(adapter, "PUBLICATION_ROOT", publication_root)
        reconstructed = adapter._safe_publication(
            archive_path, claim.publication.to_dict()
        )

        assert reconstructed.to_dict() == claim.publication.to_dict()
        assert stat.S_IMODE(reconstructed.root.stat().st_mode) == 0o555
        assert stat.S_IMODE(
            (reconstructed.root / ".cacheon-native-artifact.json").stat().st_mode
        ) == 0o444
        for logical in reconstructed.directories:
            assert stat.S_IMODE(
                reconstructed.root.joinpath(*Path(logical).parts).stat().st_mode
            ) == 0o555
        for row in reconstructed.files:
            assert stat.S_IMODE(
                reconstructed.root.joinpath(*Path(row.path).parts).stat().st_mode
            ) == 0o444

        reopened = reopen_worker_bundle(
            reconstructed.root,
            claim.publication.content_hash,
            expected_publication_digest=claim.publication.publication_digest,
            expected_receipt_digest=claim.publication.digest,
        )
        assert reopened.to_dict() == claim.publication.to_dict()
        reused = adapter._safe_publication(
            archive_path, claim.publication.to_dict()
        )
        assert reused.to_dict() == claim.publication.to_dict()
    finally:
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


def test_pod_service_reuses_one_adapter_process_for_two_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.pid = 12345
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

    fake = FakeProcess()
    starts: list[str] = []

    def start(self, *, deadline, request_id):
        del deadline
        starts.append(request_id)
        self.start_count += 1
        self.process = fake
        return True

    def read_control(self, *, deadline, request_id, state):
        del deadline, state
        return {
            "request_id": request_id,
            "schema": worker.SCHEMA_ADAPTER_CONTROL,
            "state": "completed",
        }

    monkeypatch.setattr(worker._PersistentAdapterProcess, "_start", start)
    monkeypatch.setattr(
        worker._PersistentAdapterProcess,
        "_read_control",
        read_control,
    )
    process = worker._PersistentAdapterProcess({}, heartbeat_seconds=5)
    request_ids = ("1" * 64, "2" * 64)
    try:
        for request_id in request_ids:
            assert (
                process.evaluate(
                    {"request_id": request_id},
                    tmp_path / request_id,
                    tmp_path / f".{request_id}.123",
                    deadline=int(time.time()) + 60,
                )
                is None
            )
        frames = tuple(
            json.loads(line)
            for line in fake.stdin.getvalue().splitlines()
        )
    finally:
        process.close()

    assert starts == [request_ids[0]]
    assert process.start_count == 1
    assert tuple(frame["request_id"] for frame in frames) == request_ids
    assert all(
        frame["schema"] == worker.SCHEMA_ADAPTER_COMMAND
        and frame["operation"] == "evaluate"
        for frame in frames
    )


def test_adapter_runtime_does_not_close_worker_between_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        coordinator,
        claim,
        service,
        credential,
        identity,
        registration,
        _request,
        _request_id,
        job_dir,
    ) = _screen_authority(tmp_path)

    class FakeWorker:
        def __init__(self) -> None:
            self.calls = 0
            self.closes = 0

        def run_remote_screen(self, lease, candidate):
            self.calls += 1
            receipt = service.screen(candidate)
            return EvaluationRun(
                lease,
                EvaluationResultEnvelope.seal(
                    lease,
                    coordinator.readiness,
                    service,
                    receipt,
                ),
                receipt,
                "completed",
            )

        def close(self) -> None:
            self.closes += 1

    fake_worker = FakeWorker()
    runtime = object.__new__(adapter._AdapterRuntime)
    runtime.service = worker
    runtime.registration = registration
    runtime.credential = credential
    runtime.identity = identity
    runtime.worker = fake_worker
    runtime.closed = False
    runtime.verify_current = lambda: None
    monkeypatch.setattr(adapter, "PUBLICATION_ROOT", tmp_path / "pod-publications")
    first = tmp_path / "pod-results" / ("." + "1" * 64 + ".123")
    second = tmp_path / "pod-results" / ("." + "2" * 64 + ".123")
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    try:
        adapter._run_with_runtime(job_dir, first, runtime)
        adapter._run_with_runtime(job_dir, second, runtime)
        assert fake_worker.calls == 2
        assert fake_worker.closes == 0
    finally:
        runtime.close()
        coordinator._release(claim.lease, reason="test_cleanup")
    assert fake_worker.closes == 1


def test_adapter_start_count_is_durable_in_events_and_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.pid = 4321
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

    registration = {
        "python_executable": sys.executable,
        "ready_receipt_digest": "a" * 64,
        "worker_epoch": "b" * 32,
        "worker_readiness_digest": "c" * 64,
    }
    events: list[tuple[str, dict[str, object]]] = []
    adapter_path = tmp_path / "adapter"
    adapter_path.write_text("adapter", encoding="utf-8")
    monkeypatch.setattr(worker, "POD_ROOT", tmp_path)
    monkeypatch.setattr(worker, "POD_ADAPTER", adapter_path)
    monkeypatch.setattr(worker, "verify_fixed_adapter", lambda _registration: None)
    monkeypatch.setattr(
        worker,
        "_adapter_environment",
        lambda _registration, _request_id: {},
    )
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(
        worker,
        "append_event",
        lambda _root, event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        worker._PersistentAdapterProcess,
        "_read_control",
        lambda *_args, **_kwargs: {
            "schema": worker.SCHEMA_ADAPTER_CONTROL,
            "state": "ready",
        },
    )

    process = worker._PersistentAdapterProcess(registration, heartbeat_seconds=5)
    try:
        assert process._start(deadline=int(time.time()) + 60, request_id="d" * 64)
        assert process.start_count == 1
        assert process.alive
        process._heartbeat("d" * 64, "evaluating")
        heartbeat = worker.verify_heartbeat(
            worker.load_json(tmp_path / "heartbeat.json"),
            registration,
            30,
        )
    finally:
        process.close()

    assert heartbeat["adapter_alive"] is True
    assert heartbeat["adapter_start_count"] == 1
    assert [event for event, _fields in events] == [
        "adapter_process_started",
        "adapter_process_ready",
    ]
    assert all(fields["adapter_start_count"] == 1 for _event, fields in events)


def test_first_adapter_failure_trips_epoch_and_success_resets() -> None:
    process = worker._PersistentAdapterProcess({}, heartbeat_seconds=5)
    process.record_result(completed=False)
    assert (
        process.consecutive_failures
        == worker.MAX_CONSECUTIVE_ADAPTER_FAILURES
    )
    process.record_result(completed=True)
    assert process.consecutive_failures == 0


def test_dead_adapter_is_never_silently_restarted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeadProcess:
        def poll(self):
            return 2

    process = worker._PersistentAdapterProcess({}, heartbeat_seconds=5)
    process.process = DeadProcess()  # type: ignore[assignment]
    process.start_count = 1
    monkeypatch.setattr(
        process,
        "_start",
        lambda **_kwargs: pytest.fail("dead adapter must not restart"),
    )
    failure = process.evaluate(
        {"request_id": "d" * 64},
        tmp_path / "request",
        tmp_path / "result",
        deadline=int(time.time()) + 60,
    )
    assert failure == "adapter_exit_nonzero"
    assert process.start_count == 1


def test_adapter_runtime_refuses_changed_commission_authority() -> None:
    class FakeService:
        @staticmethod
        def load_json(path):
            if path == adapter.REGISTRATION_PATH:
                return {"identity": "changed"}
            return {"identity": "ready"}

        @staticmethod
        def verify_registration(value):
            return value

        @staticmethod
        def verify_ready_receipt(value):
            return value

    runtime = object.__new__(adapter._AdapterRuntime)
    runtime.service = FakeService()
    runtime.registration = {"identity": "original"}
    runtime.ready = {"identity": "ready"}
    with pytest.raises(adapter.AdapterError, match="authority changed"):
        runtime.verify_current()


def test_adapter_pre_resident_carrier_failure_never_calls_worker(
    tmp_path: Path,
) -> None:
    class FakeService:
        @staticmethod
        def load_json(_path, maximum=None):
            del maximum
            return {}

        @staticmethod
        def verify_request(_value, _root, _registration):
            raise worker.RemoteWorkerError("malformed request carrier")

    class FakeWorker:
        calls = 0

        def run_remote_screen(self, _lease, _candidate):
            self.calls += 1
            raise AssertionError("pre-resident failure must not call worker")

    runtime = object.__new__(adapter._AdapterRuntime)
    runtime.service = FakeService()
    runtime.registration = {}
    runtime.worker = FakeWorker()
    runtime.closed = False
    runtime.verify_current = lambda: None

    with pytest.raises(adapter.AdapterRequestFailed) as captured:
        adapter._run_with_runtime(tmp_path / "request", tmp_path / "result", runtime)

    assert isinstance(captured.value.__cause__, worker.RemoteWorkerError)
    assert runtime.worker.calls == 0


def test_adapter_request_failure_continues_on_same_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_ids = ("1" * 64, "2" * 64)
    paths = {
        b"first\n": (
            request_ids[0],
            tmp_path / request_ids[0],
            tmp_path / f".{request_ids[0]}.1",
        ),
        b"second\n": (
            request_ids[1],
            tmp_path / request_ids[1],
            tmp_path / f".{request_ids[1]}.1",
        ),
    }
    runtime = object()
    seen: list[tuple[object, str]] = []

    monkeypatch.setattr(
        adapter,
        "_validated_command_paths",
        lambda raw: paths[raw],
    )

    def run_with_runtime(request_dir, _result_dir, observed_runtime):
        request_id = request_dir.name
        seen.append((observed_runtime, request_id))
        if request_id == request_ids[0]:
            raise adapter.AdapterRequestFailed("bad carrier")

    monkeypatch.setattr(adapter, "_run_with_runtime", run_with_runtime)
    controls = io.BytesIO()

    assert adapter._serve_runtime(runtime, iter(paths), controls) == 0
    frames = tuple(json.loads(row) for row in controls.getvalue().splitlines())

    assert seen == [(runtime, request_ids[0]), (runtime, request_ids[1])]
    assert frames == (
        {"schema": adapter.ADAPTER_CONTROL_SCHEMA, "state": "ready"},
        {
            "request_id": request_ids[0],
            "schema": adapter.ADAPTER_CONTROL_SCHEMA,
            "state": "request_failed",
        },
        {
            "request_id": request_ids[1],
            "schema": adapter.ADAPTER_CONTROL_SCHEMA,
            "state": "completed",
        },
    )


def test_adapter_epoch_failure_exits_before_next_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_ids = ("3" * 64, "4" * 64)
    paths = {
        b"first\n": (
            request_ids[0],
            tmp_path / request_ids[0],
            tmp_path / f".{request_ids[0]}.1",
        ),
        b"second\n": (
            request_ids[1],
            tmp_path / request_ids[1],
            tmp_path / f".{request_ids[1]}.1",
        ),
    }
    runtime = object()
    seen: list[tuple[object, str]] = []

    monkeypatch.setattr(
        adapter,
        "_validated_command_paths",
        lambda raw: paths[raw],
    )

    def run_with_runtime(request_dir, _result_dir, observed_runtime):
        seen.append((observed_runtime, request_dir.name))
        raise adapter.AdapterEpochFailed("resident lifetime failed")

    monkeypatch.setattr(adapter, "_run_with_runtime", run_with_runtime)
    controls = io.BytesIO()

    assert adapter._serve_runtime(runtime, iter(paths), controls) == 2
    frames = tuple(json.loads(row) for row in controls.getvalue().splitlines())

    assert seen == [(runtime, request_ids[0])]
    assert frames == (
        {"schema": adapter.ADAPTER_CONTROL_SCHEMA, "state": "ready"},
        {
            "request_id": request_ids[0],
            "schema": adapter.ADAPTER_CONTROL_SCHEMA,
            "state": "epoch_failed",
        },
    )

from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_pod_service as pod_service
from cacheon.chain import remote_worker_service as service_cli
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.remote_worker_registration import PodPaths


def _pod_paths(tmp_path: Path) -> PodPaths:
    return PodPaths(
        root=tmp_path,
        ready_receipt=tmp_path / "ready-receipt.json",
        registration=tmp_path / "registration.json",
        service=tmp_path / "remote_worker_service.py",
        adapter=tmp_path / "adapter",
        credential=tmp_path / "credential.secret",
    )


def _spool_fixtures():
    path = Path(__file__).with_name("test_remote_worker_spool.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_remote_spool_test_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class _FakeProcess:
    def __init__(self, pid: int = 12345) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.pid = pid
        self.returncode: int | None = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0


def test_pod_service_reuses_one_adapter_process_for_two_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProcess()
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
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "completed",
        }

    monkeypatch.setattr(pod_service.PersistentAdapterProcess, "_start", start)
    monkeypatch.setattr(
        pod_service.PersistentAdapterProcess, "_read_control", read_control
    )
    process = pod_service.PersistentAdapterProcess(
        {}, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
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
            json.loads(line) for line in fake.stdin.getvalue().splitlines()
        )
    finally:
        process.close()

    assert starts == [request_ids[0]]
    assert process.start_count == 1
    assert tuple(frame["request_id"] for frame in frames) == request_ids
    assert all(
        frame["schema"] == spool.SCHEMA_ADAPTER_COMMAND
        and frame["operation"] == "evaluate"
        for frame in frames
    )


def test_adapter_start_count_is_durable_in_events_and_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = {
        "python_executable": sys.executable,
        "ready_receipt_digest": "a" * 64,
        "worker_epoch": "b" * 32,
        "worker_readiness_digest": "c" * 64,
        "transport_identity_digest": "d" * 64,
        "lane_devices": [0],
    }
    events: list[tuple[str, dict[str, object]]] = []
    paths = _pod_paths(tmp_path)
    paths.adapter.write_text("adapter", encoding="utf-8")
    monkeypatch.setattr(pod_service, "verify_fixed_adapter", lambda _path, _registration: None)
    monkeypatch.setattr(
        pod_service, "adapter_environment", lambda _registration, _paths, _request_id: {}
    )
    monkeypatch.setattr(
        pod_service.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProcess(4321)
    )
    monkeypatch.setattr(
        pod_service,
        "append_event",
        lambda _root, event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        pod_service.PersistentAdapterProcess,
        "_read_control",
        lambda *_args, **_kwargs: {
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "ready",
        },
    )

    process = pod_service.PersistentAdapterProcess(
        registration, paths=paths, heartbeat_seconds=5
    )
    try:
        assert process._start(deadline=int(time.time()) + 60, request_id="d" * 64)
        assert process.start_count == 1
        assert process.alive
        process._heartbeat("d" * 64, "evaluating")
        heartbeat = spool.verify_heartbeat(
            spool.load_json(tmp_path / "heartbeat.json"),
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


def test_first_adapter_failure_trips_epoch_and_success_resets(tmp_path: Path) -> None:
    process = pod_service.PersistentAdapterProcess(
        {}, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
    process.record_result(completed=False)
    assert process.consecutive_failures == spool.MAX_CONSECUTIVE_ADAPTER_FAILURES
    process.record_result(completed=True)
    assert process.consecutive_failures == 0


def test_dead_adapter_is_never_silently_restarted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeadProcess:
        def poll(self):
            return 2

    process = pod_service.PersistentAdapterProcess(
        {}, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
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


def test_publish_result_emits_verifiable_ready_receipt(tmp_path: Path) -> None:
    fixtures = _spool_fixtures()
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
    ) = fixtures._screen_authority(tmp_path)
    try:
        outer = spool.verify_request(
            spool.load_json(job_dir / "request.json"),
            job_dir,
            registration,
            identity=identity,
            credential=credential,
        )
        from cacheon.chain.remote_evaluation_dispatcher import seal_remote_response

        receipt = service.screen(claim.candidate)
        response = seal_remote_response(request, receipt, identity, credential)
        result_root = tmp_path / "pod-result"
        result_root.mkdir()
        (result_root / "response.json").write_bytes(
            spool.spool_canonical_json(response.to_dict()) + b"\n"
        )
        spool.finalize_adapter_response(
            outer, job_dir, result_root, identity=identity, credential=credential
        )
        outgoing = tmp_path / "outgoing"
        pod_service.publish_result(
            registration,
            outer,
            result_root,
            request_root=job_dir,
            outgoing_root=outgoing,
            events_root=tmp_path,
            identity=identity,
            credential=credential,
        )
        ready = spool.load_json(outgoing / f"{request_id}.ready.json")
        verified = spool.verify_result_ready(ready, outer, registration)
        archive = outgoing / f"{request_id}.{verified['archive_sha256']}.tar"
        assert archive.is_file()
        assert archive.stat().st_size == verified["archive_size"]
        pod_service.publish_result(
            registration,
            outer,
            result_root,
            request_root=job_dir,
            outgoing_root=outgoing,
            events_root=tmp_path,
            identity=identity,
            credential=credential,
        )
    finally:
        coordinator._release(claim.lease, reason="test_cleanup")


def test_adapter_control_frames_must_be_canonical() -> None:
    frame = {"schema": spool.SCHEMA_ADAPTER_CONTROL, "state": "ready"}
    raw = spool.spool_canonical_json(frame) + b"\n"
    assert pod_service.decode_adapter_control(raw) == frame
    with pytest.raises(spool.RemoteWorkerError, match="malformed control frame"):
        pod_service.decode_adapter_control(raw[:-1])
    padded = b'{"schema": "' + spool.SCHEMA_ADAPTER_CONTROL.encode() + b'", "state": "ready"}\n'
    with pytest.raises(spool.RemoteWorkerError, match="not canonical"):
        pod_service.decode_adapter_control(padded)


def test_service_parser_is_closed_and_bounds_are_enforced(tmp_path: Path) -> None:
    parser = service_cli.build_parser()
    assert "command" not in {action.dest for action in parser._actions}
    with pytest.raises(SystemExit):
        parser.parse_args(["install-source"])
    args = parser.parse_args(
        [
            "cpu-serve",
            "--registration",
            str(tmp_path / "registration.json"),
            "--current-registration",
            str(tmp_path / "current.json"),
            "--spool-root",
            str(tmp_path / "spool"),
            "--pod-root",
            "/data/pod",
            "--pod-service-path",
            "/data/pod/bin/service.py",
            "--poll-seconds",
            "999",
        ]
    )
    assert args.function is service_cli._cmd_cpu_serve
    assert service_cli.main(
        [
            "cpu-serve",
            "--registration",
            str(tmp_path / "missing.json"),
            "--current-registration",
            str(tmp_path / "current.json"),
            "--spool-root",
            str(tmp_path / "spool"),
            "--pod-root",
            "/data/pod",
            "--pod-service-path",
            "/data/pod/bin/service.py",
            "--poll-seconds",
            "999",
        ]
    ) == 2

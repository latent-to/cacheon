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
from cacheon.eval.remote_run_forensics import append_event as append_run_event, journal_path


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


def test_adapter_failures_reach_cooldown_threshold_and_success_resets(
    tmp_path: Path,
) -> None:
    process = pod_service.PersistentAdapterProcess(
        {}, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
    for expected in range(1, spool.MAX_CONSECUTIVE_ADAPTER_FAILURES + 1):
        assert process.consecutive_failures < spool.MAX_CONSECUTIVE_ADAPTER_FAILURES
        process.record_result(completed=False)
        assert process.consecutive_failures == expected
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


class _ReapedDeadProcess:
    stdin = None
    stdout = None

    def poll(self):
        return 2

    def wait(self, timeout=None):
        del timeout
        return 2


def test_permit_restart_authorizes_exactly_one_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = pod_service.PersistentAdapterProcess(
        {}, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
    process.process = _ReapedDeadProcess()  # type: ignore[assignment]
    process.start_count = 1
    process.consecutive_failures = spool.MAX_CONSECUTIVE_ADAPTER_FAILURES

    process.permit_restart()
    assert process.consecutive_failures == 0
    assert process.process is None
    assert process.restart_permitted

    starts: list[str] = []

    def start(self, *, deadline, request_id):
        del deadline
        starts.append(request_id)
        self.start_count += 1
        return False

    monkeypatch.setattr(pod_service.PersistentAdapterProcess, "_start", start)
    failure = process.evaluate(
        {"request_id": "d" * 64},
        tmp_path / "request",
        tmp_path / "result",
        deadline=int(time.time()) + 60,
    )
    assert failure == "adapter_start_failed"
    assert starts == ["d" * 64]
    assert not process.restart_permitted

    # The permission is consumed by the attempt; a second request must not
    # silently boot another replacement engine before the next cooldown.
    failure = process.evaluate(
        {"request_id": "e" * 64},
        tmp_path / "request",
        tmp_path / "result",
        deadline=int(time.time()) + 60,
    )
    assert failure == "adapter_exit_nonzero"
    assert starts == ["d" * 64]


def test_permit_restart_leaves_live_adapter_resident(tmp_path: Path) -> None:
    process = pod_service.PersistentAdapterProcess(
        {}, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
    fake = _FakeProcess()
    process.process = fake  # type: ignore[assignment]
    process.start_count = 1
    process.consecutive_failures = spool.MAX_CONSECUTIVE_ADAPTER_FAILURES

    process.permit_restart()

    assert process.consecutive_failures == 0
    assert process.process is fake
    assert not process.restart_permitted


def test_adapter_cooldown_parks_resumes_and_doubles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = {"worker_epoch": "b" * 32}
    events: list[tuple[str, dict[str, object]]] = []
    heartbeat_states: list[str] = []
    sleeps: list[float] = []
    now = [1_000.0]

    monkeypatch.setattr(
        pod_service,
        "append_event",
        lambda _root, event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        pod_service, "verify_pod_registration", lambda _paths: registration
    )
    process = pod_service.PersistentAdapterProcess(
        registration, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
    process.start_count = 1
    process.consecutive_failures = spool.MAX_CONSECUTIVE_ADAPTER_FAILURES
    monkeypatch.setattr(
        process,
        "_heartbeat",
        lambda _request_id, state: heartbeat_states.append(state),
    )

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    next_cooldown = pod_service.adapter_cooldown(
        process,
        registration,
        _pod_paths(tmp_path),
        cooldown_seconds=spool.ADAPTER_COOLDOWN_INITIAL_SECONDS,
        poll_seconds=5,
        clock=lambda: now[0],
        sleep=sleep,
    )

    assert next_cooldown == 2 * spool.ADAPTER_COOLDOWN_INITIAL_SECONDS
    assert [event for event, _fields in events] == [
        "adapter_cooldown_started",
        "adapter_cooldown_resumed",
    ]
    assert events[0][1]["cooldown_seconds"] == spool.ADAPTER_COOLDOWN_INITIAL_SECONDS
    assert set(heartbeat_states) == {"adapter_cooldown"}
    assert sum(sleeps) >= spool.ADAPTER_COOLDOWN_INITIAL_SECONDS
    assert process.consecutive_failures == 0
    assert process.restart_permitted

    process.consecutive_failures = spool.MAX_CONSECUTIVE_ADAPTER_FAILURES
    assert (
        pod_service.adapter_cooldown(
            process,
            registration,
            _pod_paths(tmp_path),
            cooldown_seconds=spool.ADAPTER_COOLDOWN_MAX_SECONDS,
            poll_seconds=5,
            clock=lambda: now[0],
            sleep=sleep,
        )
        == spool.ADAPTER_COOLDOWN_MAX_SECONDS
    )


def test_cooldown_heartbeat_is_telemetry_and_failure_cap_holds() -> None:
    registration = {
        "ready_receipt_digest": "a" * 64,
        "worker_epoch": "b" * 32,
        "worker_readiness_digest": "c" * 64,
    }

    def payload(state: str, failures: int) -> dict[str, object]:
        return spool.heartbeat_payload(
            registration,
            state,
            None,
            adapter_start_count=1,
            adapter_alive=False,
            consecutive_adapter_failures=failures,
        )

    verified = spool.verify_heartbeat(
        payload("adapter_cooldown", spool.MAX_CONSECUTIVE_ADAPTER_FAILURES),
        registration,
        30,
    )
    assert verified["state"] == "adapter_cooldown"
    # Cooldown after a single transport-interrupted adapter death is valid
    # telemetry (2026-08-12): no failure-threshold consistency rejection.
    cooled = spool.verify_heartbeat(payload("adapter_cooldown", 1), registration, 30)
    assert cooled["state"] == "adapter_cooldown"
    with pytest.raises(spool.RemoteWorkerError):
        spool.verify_heartbeat(
            payload("idle", spool.MAX_CONSECUTIVE_ADAPTER_FAILURES + 1),
            registration,
            30,
        )


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
        append_run_event(
            journal_path(result_root), request_id, "adapter.terminal", "completed"
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

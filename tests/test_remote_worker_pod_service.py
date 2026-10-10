from __future__ import annotations

import io
import json
import subprocess
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


def test_epoch_failure_retires_adapter_before_another_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProcess()
    process = pod_service.PersistentAdapterProcess(
        {}, paths=_pod_paths(tmp_path), heartbeat_seconds=5
    )
    process.process = fake  # type: ignore[assignment]
    process.start_count = 1
    request_id = "d" * 64
    monkeypatch.setattr(
        process,
        "_read_control",
        lambda **_kwargs: {
            "request_id": request_id,
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "epoch_failed",
        },
    )

    failure = process.evaluate(
        {"request_id": request_id},
        tmp_path / "request",
        tmp_path / "result",
        deadline=int(time.time()) + 60,
    )

    assert failure == "adapter_epoch_failed"
    assert process.process is None
    assert fake.returncode == 0


def test_timeout_reaps_real_adapter_and_cooldown_allows_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = {
        "python_executable": sys.executable, "worker_epoch": "b" * 32,
        "ready_receipt_digest": "a" * 64, "worker_readiness_digest": "c" * 64,
        "transport_identity_digest": "d" * 64, "lane_devices": [0],
    }
    paths = _pod_paths(tmp_path)
    paths.adapter.write_text(
        "import json, sys, time\n"
        "def emit(**fields):\n"
        f"    fields['schema'] = {spool.SCHEMA_ADAPTER_CONTROL!r}\n"
        "    print(json.dumps(fields, sort_keys=True, separators=(',', ':')), flush=True)\n"
        "emit(state='ready')\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        f"    if request['schema'] != {spool.SCHEMA_ADAPTER_COMMAND!r}"
        " or request['operation'] != 'evaluate':\n"
        "        sys.exit(3)\n"
        "    if request['request_id'].startswith('d'):\n"
        "        time.sleep(300)\n"
        "    emit(state='completed', request_id=request['request_id'])\n"
    )
    monkeypatch.setattr(pod_service, "verify_fixed_adapter", lambda *_args: None)
    monkeypatch.setattr(pod_service, "adapter_environment", lambda *_args: {})
    # Exercise actual wait/TERM/reap while shortening only its 20-second grace.
    original_wait = subprocess.Popen.wait
    monkeypatch.setattr(
        subprocess.Popen, "wait",
        lambda self, timeout=None: original_wait(
            self, timeout=min(timeout, 0.1) if timeout is not None else None
        ),
    )
    process = pod_service.PersistentAdapterProcess(
        registration, paths=paths, heartbeat_seconds=1
    )
    children = []
    original_start = process._start

    def start(**kwargs):
        ready = original_start(**kwargs)
        children.append(process.process)
        return ready

    monkeypatch.setattr(process, "_start", start)
    try:
        assert process.evaluate(
            {"request_id": "d" * 64}, tmp_path / "first", tmp_path / "first-result",
            deadline=int(time.time()) + 2,
        ) == "adapter_timeout"
        assert children[0].poll() is not None
        assert process.process is None
        assert process.evaluate(
            {"request_id": "e" * 64}, tmp_path / "next", tmp_path / "next-result",
            deadline=int(time.time()) + 10,
        ) == "adapter_exit_nonzero"
        assert process.start_count == 1
        process.permit_restart()
        assert process.evaluate(
            {"request_id": "e" * 64}, tmp_path / "next", tmp_path / "next-result",
            deadline=int(time.time()) + 10,
        ) is None
        assert process.start_count == 2
        assert children[0].pid != children[1].pid
        # A healthy adapter is reused across requests, and the start count is
        # durable in the event journal and the heartbeat.
        assert process.evaluate(
            {"request_id": "f" * 64}, tmp_path / "third", tmp_path / "third-result",
            deadline=int(time.time()) + 10,
        ) is None
        assert process.start_count == 2 and process.alive
        process._heartbeat("f" * 64, "evaluating")
        heartbeat = spool.verify_heartbeat(
            spool.load_json(tmp_path / "heartbeat.json"), registration, 30
        )
        assert heartbeat["adapter_alive"] is True
        assert heartbeat["adapter_start_count"] == 2
        starts = [
            (row["event"], row["adapter_start_count"])
            for row in map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines())
            if row["event"] in {"adapter_process_started", "adapter_process_ready"}
        ]
        assert starts == [
            ("adapter_process_started", 1), ("adapter_process_ready", 1),
            ("adapter_process_started", 2), ("adapter_process_ready", 2),
        ]
    finally:
        process.close()


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


def test_adapter_control_frames_must_be_canonical() -> None:
    frame = {"schema": spool.SCHEMA_ADAPTER_CONTROL, "state": "ready"}
    raw = spool.spool_canonical_json(frame) + b"\n"
    assert pod_service.decode_adapter_control(raw) == frame
    with pytest.raises(spool.RemoteWorkerError, match="malformed control frame"):
        pod_service.decode_adapter_control(raw[:-1])
    padded = b'{"schema": "' + spool.SCHEMA_ADAPTER_CONTROL.encode() + b'", "state": "ready"}\n'
    with pytest.raises(spool.RemoteWorkerError, match="not canonical"):
        pod_service.decode_adapter_control(padded)


def test_cpu_serve_poll_bound_is_enforced_before_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_cli, "cpu_serve", lambda **_kwargs: pytest.fail("poll bound not enforced")
    )
    assert service_cli.main([
        "cpu-serve", "--registration", str(tmp_path / "registration.json"),
        "--current-registration", str(tmp_path / "current.json"),
        "--spool-root", str(tmp_path / "spool"), "--pod-root", "/data/pod",
        "--pod-service-path", "/data/pod/bin/service.py", "--poll-seconds", "999",
    ]) == 2

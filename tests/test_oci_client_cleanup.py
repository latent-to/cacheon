"""Final cleanup proves both process death and exact-container absence."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import cacheon.eval.oci_process as process_mod
from cacheon.eval.oci_client_cleanup import cleanup_client
from cacheon.eval.oci_process import CommandResult, OCIProcessError, OCIProcessTimeout
from tests.support.oci_process import Commands, FakeProcess, FakeStream, _manager

@pytest.fixture(autouse=True)
def _clear_gpu_reservation_env(monkeypatch):
    monkeypatch.delenv(process_mod.GPU_RESERVATION_ENV, raising=False)


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("finishes", [False, True])
def test_pending_removal_requires_the_final_absence_proof(tmp_path, monkeypatch, attached, finishes):
    class DelayedRemoval(Commands):
        attempts = 0

        def __call__(self, argv, **kwargs):
            if tuple(argv[1:3]) == ("rm", "--force"):
                self.attempts += 1
                if self.attempts == 1 or not finishes:
                    self.rows.append(tuple(argv))
                    return CommandResult(1, b"", b"removal in progress")
            return super().__call__(argv, **kwargs)

    commands = DelayedRemoval()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin, process.stdout, process.stderr = FakeStream(), FakeStream(), None
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(manager, "_terminate_client", lambda p: p.terminate())
    client = manager.spawn_attached(lease, (*lease.run_prefix(manager.docker_binary), "image"))
    commands.present.add("container-1")
    close = client.finalize if attached else lambda: cleanup_client(manager, lease, process)
    if finishes:
        close()
        assert not commands.present
        if attached:
            assert client.closed
    else:
        with pytest.raises(OCIProcessError, match="could not prove both") as raised:
            close()
        assert isinstance(raised.value.__cause__, process_mod._ContainerRemovalPending)
        assert commands.present and not client.closed
    assert commands.attempts == 2 and process.terminated


def test_final_absence_does_not_erase_an_identity_error(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    error = OCIProcessError("refusing removal with wrong lease labels")
    calls = []

    def remove(_lease):
        calls.append(_lease)
        if len(calls) == 1:
            raise error

    monkeypatch.setattr(manager, "force_remove_container", remove)
    monkeypatch.setattr(manager, "_terminate_client", lambda p: p.terminate())
    with pytest.raises(OCIProcessError, match="could not prove both") as raised:
        cleanup_client(manager, lease, process)
    assert raised.value.__cause__ is error
    assert len(calls) == 2 and process.terminated

def test_attached_client_spawn_and_normal_finalize_use_manager_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin = FakeStream()
    process.stdout = FakeStream()
    process.stderr = None
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    events = []
    original_remove = manager.force_remove_container

    def remove(observed_lease):
        events.append("remove")
        return original_remove(observed_lease)

    def terminate(observed_process):
        events.append("terminate")
        observed_process.terminate()
        observed_process.wait(timeout=10)

    monkeypatch.setattr(manager, "force_remove_container", remove)
    monkeypatch.setattr(manager, "_terminate_client", terminate)
    argv = (*lease.run_prefix(manager.docker_binary), "--network=none", "image@sha256:x")

    client = manager.spawn_attached(lease, argv)
    assert client.stdin is process.stdin and client.stdout is process.stdout
    commands.present.add("container-1")
    client.finalize()

    assert events == ["remove", "terminate", "remove"]
    assert client.closed and process.terminated
    assert process.stdin.closed and process.stdout.closed
    assert "container-1" not in commands.present
    # Teardown is idempotent for a defensive finally: abort after finalize cannot
    # revive or remove a different resource.
    client.abort()
    assert events == ["remove", "terminate", "remove"]



def test_attached_abort_rechecks_container_after_client_death(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin = FakeStream()
    process.stdout = FakeStream()
    process.stderr = None
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )

    def terminate_then_late_create(observed_process):
        observed_process.terminate()
        commands.present.add("container-1")

    monkeypatch.setattr(manager, "_terminate_client", terminate_then_late_create)
    client.abort()

    assert client.closed
    assert "container-1" not in commands.present
    assert any(row[1:3] == ("rm", "--force") for row in commands.rows)



def test_attached_finalize_never_signals_an_already_reaped_process_group(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin = FakeStream()
    process.stdout = FakeStream()
    process.stderr = None
    process.returncode = 0
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )
    monkeypatch.setattr(
        process_mod.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("reaped PID was signalled")),
    )

    client.finalize()

    assert client.closed
    assert not process.terminated and not process.killed



def test_timeout_force_removes_container_and_terminates_client(tmp_path: Path, monkeypatch) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    proc = FakeProcess(())

    def communicate(*, input, timeout):
        commands.present.add("container-1")
        raise subprocess.TimeoutExpired("docker", timeout)

    proc.communicate = communicate
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(process_mod.os, "killpg", lambda *_args: (_ for _ in ()).throw(OSError()))

    with pytest.raises(OCIProcessTimeout) as raised:
        manager.run(
            lease,
            (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
            timeout_s=0.1,
        )
    assert proc.terminated
    assert raised.value.diagnostic is not None
    assert raised.value.diagnostic.artifact is not None
    assert manager.reopen_stderr_artifact(
        raised.value.diagnostic.artifact
    ) == raised.value.diagnostic.artifact.artifact_path
    assert any(row[1:3] == ("rm", "--force") for row in commands.rows)



def test_timeout_rechecks_after_client_death_closes_late_create_race(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    proc = FakeProcess(())

    def timeout(*, input, timeout):
        raise subprocess.TimeoutExpired("docker", timeout)

    def terminate_then_late_create(_process):
        commands.present.add("container-1")

    proc.communicate = timeout
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(manager, "_terminate_client", terminate_then_late_create)

    with pytest.raises(OCIProcessTimeout):
        manager.run(
            lease,
            (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
            timeout_s=0.1,
        )
    assert "container-1" not in commands.present
    assert any(row[1:3] == ("rm", "--force") for row in commands.rows)



def test_timeout_still_terminates_client_when_absence_proof_fails(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    proc = FakeProcess(())
    def timeout(*, input, timeout):
        commands.present.add("container-1")
        raise subprocess.TimeoutExpired("docker", timeout)

    proc.communicate = timeout
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(process_mod.os, "killpg", lambda *_args: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(
        manager,
        "force_remove_container",
        lambda _lease: (_ for _ in ()).throw(OCIProcessError("absence unavailable")),
    )
    with pytest.raises(OCIProcessError, match="could not prove"):
        manager.run(
            lease,
            (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
            timeout_s=0.1,
        )
    assert proc.terminated

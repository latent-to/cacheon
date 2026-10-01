"""Docker command and foreground-process doubles shared by lifecycle tests."""

from __future__ import annotations

import json
from pathlib import Path

from cacheon.eval.oci_process import (
    CommandResult, GPU_RESERVATION_LABEL, NAMESPACE_LABEL, OCIProcessManager,
)

CONTAINER_ID = "d" * 64

class Commands:
    def __init__(self) -> None:
        self.rows: list[tuple[str, ...]] = []
        self.present: set[str] = set()
        self.labels: dict[str, tuple[str, str]] = {}
        self.namespace_labels: dict[str, str | None] = {}
        self.default_namespace: str | None = None
        self.gpu_labels: dict[str, str] = {}

    def __call__(self, argv, *, timeout_s, max_output_bytes):
        row = tuple(argv)
        self.rows.append(row)
        if row[1:3] == ("container", "ls"):
            names = [value for value in row if value.startswith("name=^/")]
            labels = [value[6:] for value in row if value.startswith("label=")]
            matching = []
            for name in self.present:
                if names and names[0][7:-1] != name:
                    continue
                executor, lease = self.labels.get(name, ("validator-a", "lease-1"))
                observed = {
                    "cacheon.executor_id": executor,
                    "cacheon.lease_id": lease,
                }
                namespace = self.namespace_labels.get(name, self.default_namespace)
                if namespace is not None:
                    observed[NAMESPACE_LABEL] = namespace
                parsed_labels = tuple(item.split("=", 1) for item in labels if "=" in item)
                if len(parsed_labels) == len(labels) and all(
                    observed.get(key) == value for key, value in parsed_labels
                ):
                    matching.append(name)
            return CommandResult(
                0,
                (CONTAINER_ID + "\n").encode() if matching else b"",
                b"",
            )
        if row[1:3] == ("container", "inspect"):
            name = next(iter(self.present))
            executor, lease = self.labels.get(name, ("validator-a", "lease-1"))
            labels = {
                "cacheon.executor_id": executor,
                "cacheon.lease_id": lease,
            }
            namespace = self.namespace_labels.get(name, self.default_namespace)
            if namespace is not None:
                labels[NAMESPACE_LABEL] = namespace
            if name in self.gpu_labels:
                labels[GPU_RESERVATION_LABEL] = self.gpu_labels[name]
            payload = {
                "Id": CONTAINER_ID,
                "Name": f"/{name}",
                "Labels": labels,
            }
            return CommandResult(0, json.dumps(payload).encode(), b"")
        if row[1:3] == ("rm", "--force"):
            self.present.clear()
        return CommandResult(0, b"", b"")



class FakeProcess:
    next_pid = 4000

    def __init__(self, argv, **kwargs):
        self.argv = tuple(argv)
        self.kwargs = kwargs
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None
        self.input = None
        self.waits = []
        self.terminated = False
        self.killed = False

    def communicate(self, *, input, timeout):
        self.input = input
        self.waits.append(timeout)
        self.returncode = 0

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            self.returncode = -9 if self.killed else -15 if self.terminated else 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def poll(self):
        return self.returncode



class FakeStream:
    next_fd = 80

    def __init__(self) -> None:
        self.closed = False
        self.fd = FakeStream.next_fd
        FakeStream.next_fd += 1

    def fileno(self) -> int:
        return self.fd

    def close(self) -> None:
        self.closed = True



class FakeClock:
    """A monotonic clock that only the manager's own sleeps advance, so bounded waits cost no wall time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def __call__(self) -> float:
        self.now += 0.001
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += float(seconds)
        self.slept += float(seconds)


def _manager(tmp_path: Path, commands: Commands | None = None) -> OCIProcessManager:
    selected = commands or Commands()
    clock = FakeClock()
    manager = OCIProcessManager(
        docker_binary="/usr/bin/docker",
        recovery_root=tmp_path / "recovery",
        executor_id="validator-a",
        runner=selected,
        clock=clock,
        sleep=clock.sleep,
    )
    selected.default_namespace = manager.namespace_digest
    return manager

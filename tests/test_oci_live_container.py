"""Observed containment behavior of the runtime flag vocabulary on a real daemon.

``tests/test_oci_backend.py`` proves the argv the backend constructs; no test
there observes a running container. These tests take the same flag vocabulary
that ``cacheon.eval.oci_backend.build_runtime_argv`` emits (network, IPC,
capability, seccomp, user, and bind-mount settings), the packaged seccomp
profile, and the backend's bind-mount builder, and assert what a real container
runtime does with them: egress fails closed, read-only binds refuse writes, a
representative privileged syscall is refused, and the non-root user is enforced.

Runs wherever a usable container daemon exists (validator hosts, CI Linux
runners); skips cleanly elsewhere. Validated live on a 2xB200 validator host,
2026-08-09. The probe image is digest-pinned; the pin is a multi-architecture
index, so amd64 and arm64 daemons resolve it identically.
"""

from __future__ import annotations

import importlib.resources as resources
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_oci_backend = pytest.importorskip("cacheon.eval.oci_backend")
_bind_mount_arg = getattr(_oci_backend, "build_bind_mount_arg", None)
if _bind_mount_arg is None:
    _bind_mount_arg = _oci_backend._mount

IMAGE = (
    "python@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36"
)

# The GPU- and identity-free subset of build_runtime_argv's flag vocabulary.
_BASE_FLAGS = (
    "--rm",
    "--init",
    "--pull=never",
    "--network=none",
    "--ipc=private",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges=true",
    "--log-driver=none",
    "--workdir=/tmp",
)


def _usable_docker() -> str | None:
    binary = shutil.which("docker")
    if binary is None:
        return None
    try:
        probe = subprocess.run(
            [binary, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return binary if probe.returncode == 0 else None


DOCKER = _usable_docker()

pytestmark = pytest.mark.skipif(
    DOCKER is None, reason="containment tier requires a container daemon"
)


@pytest.fixture(scope="session")
def probe_image() -> str:
    inspect = subprocess.run(
        [DOCKER, "image", "inspect", IMAGE], capture_output=True, timeout=60
    )
    if inspect.returncode != 0:
        pull = subprocess.run(
            [DOCKER, "pull", "--quiet", IMAGE], capture_output=True, timeout=600
        )
        if pull.returncode != 0:
            pytest.skip("pinned probe image is unavailable to this daemon")
    return IMAGE


@pytest.fixture(scope="session")
def seccomp_profile(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw = (
        resources.files("cacheon.eval")
        .joinpath("seccomp_moby_v0_2_1.json")
        .read_bytes()
    )
    parsed = json.loads(raw)
    assert parsed.get("defaultAction") == "SCMP_ACT_ERRNO"
    path = tmp_path_factory.mktemp("containment") / "seccomp.json"
    path.write_bytes(raw)
    return path


def _run(
    args: list[str], *, image: str, probe: str, timeout: float = 120.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [DOCKER, "run", *args, image, "python3", "-c", probe],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_flag_vocabulary_boots_and_enforces_the_nonroot_user(
    probe_image: str, seccomp_profile: Path
) -> None:
    # A daemon that rejected the packaged profile would fail at create, so a
    # successful run is also the "pinned profile loads" proof.
    result = _run(
        [
            *_BASE_FLAGS,
            f"--security-opt=seccomp={seccomp_profile}",
            "--user=65534:65534",
            "--pids-limit=256",
            "--tmpfs=/tmp:rw,nosuid,nodev,exec,size=67108864,"
            "uid=65534,gid=65534,mode=0700",
        ],
        image=probe_image,
        probe="import os; print(f'UID:{os.getuid()}')",
    )
    assert result.returncode == 0, result.stderr
    assert "UID:65534" in result.stdout


def test_network_egress_fails_closed(probe_image: str, seccomp_profile: Path) -> None:
    probe = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=4)\n"
        "    print('EGRESS:OPEN')\n"
        "except OSError as exc:\n"
        "    print(f'EGRESS:REFUSED:{type(exc).__name__}')\n"
    )
    result = _run(
        [
            *_BASE_FLAGS,
            f"--security-opt=seccomp={seccomp_profile}",
            "--user=65534:65534",
        ],
        image=probe_image,
        probe=probe,
    )
    assert result.returncode == 0, result.stderr
    assert "EGRESS:REFUSED" in result.stdout
    assert "EGRESS:OPEN" not in result.stdout


def test_readonly_bind_refuses_writes_and_the_probe_is_valid(
    probe_image: str, tmp_path: Path
) -> None:
    host_dir = tmp_path / "bind"
    host_dir.mkdir()
    identity = f"--user={os.getuid()}:{os.getgid()}"
    write_probe = (
        "import errno\n"
        "try:\n"
        "    with open('/bind-probe/probe', 'w') as handle:\n"
        "        handle.write('x')\n"
        "    print('WRITE:OK')\n"
        "except OSError as exc:\n"
        "    print(f'WRITE:REFUSED:{errno.errorcode.get(exc.errno, exc.errno)}')\n"
    )

    readonly = _run(
        [
            *_BASE_FLAGS,
            identity,
            _bind_mount_arg(host_dir, "/bind-probe", readonly=True),
        ],
        image=probe_image,
        probe=write_probe,
    )
    assert readonly.returncode == 0, readonly.stderr
    assert "WRITE:REFUSED:EROFS" in readonly.stdout

    # The writable variant proves the refusal above came from the read-only
    # bind, not from an unrelated failure of the same probe.
    writable = _run(
        [
            *_BASE_FLAGS,
            identity,
            _bind_mount_arg(host_dir, "/bind-probe", readonly=False),
        ],
        image=probe_image,
        probe=write_probe,
    )
    assert writable.returncode == 0, writable.stderr
    assert "WRITE:OK" in writable.stdout
    assert (host_dir / "probe").is_file()


def test_composite_policy_refuses_module_loading_syscall(
    probe_image: str, seccomp_profile: Path
) -> None:
    # init_module is representative: outside both the pinned profile's
    # allowlist and the retained capability set, so the composite policy must
    # refuse it regardless of which layer answers first.
    probe = (
        "import ctypes, errno, platform\n"
        "nr = {'x86_64': 175, 'aarch64': 105}[platform.machine()]\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "rc = libc.syscall(nr, None, 0, None)\n"
        "code = ctypes.get_errno()\n"
        "state = 'REFUSED' if rc == -1 else 'ALLOWED'\n"
        "print(f'SYSCALL:{state}:{errno.errorcode.get(code, code)}')\n"
    )
    result = _run(
        [
            *_BASE_FLAGS,
            f"--security-opt=seccomp={seccomp_profile}",
            "--user=65534:65534",
        ],
        image=probe_image,
        probe=probe,
    )
    assert result.returncode == 0, result.stderr
    assert "SYSCALL:REFUSED:EPERM" in result.stdout

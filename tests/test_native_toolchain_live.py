"""Opt-in build smoke for the native cutlass.cute.cubin.v1 toolchain.

The native lane's CPU tests exercise its policy and identity code against
hand-forged ELF headers and a monkeypatched compiler; nothing in the suite
proves the toolchain itself works. This tier does, in two containers that
mirror the lane's build/load phase split:

1. A deviceless build container (``CUDA_VISIBLE_DEVICES=""`` exactly as the
   prebuild child requires) compiles a minimal ``@cute.jit`` kernel with the
   validator compiler recipe from ``cacheon.cute_aot`` — GPUArch, OptLevel(3),
   assertions and line info off, KeepCUBIN, ``no_jit_engine=True`` — and
   retains the device CUBIN.
2. A GPU container runs the produced bytes through the production ELF gate
   (``cacheon.cuda_cubin._require_elf_cubin``) and then performs Driver-API
   admission: library load, kernel enumeration, and name resolution, asserting
   the loaded kernel is the one the source declared.

This establishes toolchain compatibility, CUBIN retention, the ELF gate on
genuine compiler output, and device loadability. It does not execute the
prebuild child's sealed specification protocol, any slot's numeric contract,
or qualification of any kind.

The tier never runs by default. Arm it explicitly:

    CACHEON_LIVE_NATIVE_TESTS=1       opt-in master switch (required)
    CACHEON_SERVE_IMAGE=...           worker image with nvidia-cutlass-dsl
    CACHEON_SERVE_REPO=/path          docker-mountable copy of this repo
                                      (default: this checkout)
    CACHEON_SERVE_SCRATCH=/path       docker-mountable scratch directory
                                      (default: a temporary directory)
    CACHEON_SERVE_GPU=0               GPU index for the admission phase
    CACHEON_NATIVE_ARCH=sm_100a       compile architecture; must match the
                                      admission device

With the master switch set, a missing prerequisite is a loud failure, never
a skip. Expect one to two minutes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_MARK = "_cacheon_smoke_kernel"

pytestmark = pytest.mark.skipif(
    os.environ.get("CACHEON_LIVE_NATIVE_TESTS") != "1",
    reason="opt-in native toolchain tier: set CACHEON_LIVE_NATIVE_TESTS=1 to run",
)

# These two mirror the seam tier's helpers; consolidate when the live tiers
# grow a shared fixture home.


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        pytest.fail(
            f"CACHEON_LIVE_NATIVE_TESTS=1 is set but {name} is not; "
            "the live tier fails loudly on misconfiguration"
        )
    return value


def _serve_repo() -> Path:
    configured = os.environ.get("CACHEON_SERVE_REPO", "")
    repo = Path(configured) if configured else _REPO_ROOT
    if not (repo / "cacheon" / "cuda_cubin.py").is_file():
        pytest.fail(f"CACHEON_SERVE_REPO={repo} does not look like a cacheon repo")
    if repo != _REPO_ROOT and content_hash(_REPO_ROOT / "cacheon") != content_hash(
        repo / "cacheon"
    ):
        pytest.fail(
            "CACHEON_SERVE_REPO copy is stale: cacheon/ differs from this "
            "checkout; refresh the mountable copy"
        )
    return repo


_BUILD_SOURCE = """
import cutlass
import cutlass.cute as cute


@cute.kernel
def {mark}():
    pass


@cute.jit
def entry():
    {mark}().launch(grid=(1, 1, 1), block=(32, 1, 1))


options = [
    cute.GPUArch({arch!r}),
    cute.OptLevel(3),
    cute.EnableAssertions(False),
    cute.GenerateLineInfo(False),
    cute.KeepCUBIN(),
]
compiled = cute.compile[tuple(options)].__call__(entry, no_jit_engine=True)
cubin = compiled.__cubin__
if callable(cubin):
    cubin = cubin()
with open("/scratch/smoke.cubin", "wb") as handle:
    handle.write(cubin)
print("BUILD-OK bytes=" + str(len(cubin)))
"""

_ADMIT_SOURCE = """
from cacheon.cuda_cubin import _require_elf_cubin
from cuda.bindings import driver

with open("/scratch/smoke.cubin", "rb") as handle:
    raw = handle.read()
_require_elf_cubin(raw)
print("GATE-OK")


def check(result):
    err, *rest = result
    assert int(err) == 0, err
    return rest[0] if len(rest) == 1 else rest


check(driver.cuInit(0))
device = check(driver.cuDeviceGet(0))
context = check(driver.cuDevicePrimaryCtxRetain(device))
check(driver.cuCtxSetCurrent(context))
library = check(driver.cuLibraryLoadData(raw, [], [], 0, [], [], 0))
count = check(driver.cuLibraryGetKernelCount(library))
kernels = check(driver.cuLibraryEnumerateKernels(count, library))
names = [check(driver.cuKernelGetName(kernel)).decode() for kernel in kernels]
print("ADMIT-OK count=" + str(int(count)) + " names=" + ",".join(names))
"""


def _docker(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def test_cute_toolchain_builds_and_device_admits_a_cubin() -> None:
    if shutil.which("docker") is None:
        pytest.fail("native toolchain tier is armed but docker is not on PATH")
    image = _required_env("CACHEON_SERVE_IMAGE")
    gpu = os.environ.get("CACHEON_SERVE_GPU", "0")
    arch = os.environ.get("CACHEON_NATIVE_ARCH", "sm_100a")
    repo = _serve_repo()
    scratch_env = os.environ.get("CACHEON_SERVE_SCRATCH", "")
    scratch = Path(scratch_env) if scratch_env else Path(tempfile.mkdtemp())
    scratch.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]
    (scratch / "build_kernel.py").write_text(
        _BUILD_SOURCE.format(mark=_KERNEL_MARK, arch=arch)
    )
    (scratch / "admit_kernel.py").write_text(_ADMIT_SOURCE)
    cubin_path = scratch / "smoke.cubin"
    if cubin_path.exists():
        cubin_path.unlink()

    build = _docker(
        [
            "docker", "run", "--rm", "--name", f"cacheon-native-build-{run_id}",
            "-e", "CUDA_VISIBLE_DEVICES=",
            "-v", f"{scratch}:/scratch",
            image, "python3", "/scratch/build_kernel.py",
        ],
        timeout=600,
    )
    assert build.returncode == 0 and "BUILD-OK" in build.stdout, (
        "deviceless cutlass compile failed; the toolchain or the validator "
        f"compile recipe no longer agree:\n{build.stdout}\n{build.stderr[-2000:]}"
    )
    assert cubin_path.is_file() and cubin_path.stat().st_size >= 64

    admit = _docker(
        [
            "docker", "run", "--rm", "--name", f"cacheon-native-admit-{run_id}",
            "--gpus", f"device={gpu}",
            "-v", f"{scratch}:/scratch",
            "-v", f"{repo}:/repo:ro",
            "-e", "PYTHONPATH=/repo",
            image, "python3", "/scratch/admit_kernel.py",
        ],
        timeout=300,
    )
    assert admit.returncode == 0, (
        f"admission phase failed:\n{admit.stdout}\n{admit.stderr[-2000:]}"
    )
    assert "GATE-OK" in admit.stdout, "production ELF gate refused a genuine cubin"
    assert "ADMIT-OK count=1" in admit.stdout, (
        f"driver admission did not resolve exactly one kernel:\n{admit.stdout}"
    )
    assert _KERNEL_MARK in admit.stdout, (
        "the admitted kernel name does not correspond to the compiled source"
    )

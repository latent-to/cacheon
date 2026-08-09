"""End-to-end verdict contract of the real CLI entry point on example bundles.

Unit suites import :mod:`cacheon.cli` in-process; nothing there executes the
documented contributor invocation. These tests run ``python -m cacheon.cli``
as a subprocess — entry-point wiring, bundle loading, device selection, and
process exit codes exercised as one observable contract.

Measured exit-code semantics (validated on CPU and on a 2xB200 validator
host, 2026-08-09): ``0`` verified, ``1`` infrastructure/usage error, ``2``
verdict FAIL. The error/FAIL distinction is load-bearing: an environment
problem must never read as a candidate refusal.

The CPU tier runs everywhere. CUDA tiers activate only where hardware
exists; the two-device tier exercises the distributed collective verify.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

pytestmark = pytest.mark.skipif(
    not EXAMPLES.is_dir(), reason="example bundles require a source checkout"
)

_CUDA = torch.cuda.is_available()
_CUDA2 = _CUDA and torch.cuda.device_count() >= 2

# Pure-torch bundles verify on every device; triton bundles need a CUDA box.
_CUDA_PASS_BUNDLES = ("miner_silu_torch", "miner_silu_triton", "miner_rmsnorm_triton")
_CUDA_FAIL_BUNDLES = ("miner_silu_broken_torch", "miner_rmsnorm_broken")


def _cli(*args: str, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cacheon.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_scan_accepts_the_documented_bundle() -> None:
    result = _cli("scan", "examples/miner_silu_torch")
    assert result.returncode == 0, result.stderr


def test_cpu_verify_accepts_the_documented_bundle() -> None:
    result = _cli(
        "verify", "examples/miner_silu_torch", "--device", "cpu", "--dtype", "float32"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_cpu_verify_fails_the_broken_torch_bundle() -> None:
    result = _cli(
        "verify",
        "examples/miner_silu_broken_torch",
        "--device",
        "cpu",
        "--dtype",
        "float32",
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "FAIL" in result.stdout


@pytest.mark.skipif(not _CUDA, reason="CUDA verify tier requires a GPU")
@pytest.mark.parametrize("bundle", _CUDA_PASS_BUNDLES)
def test_cuda_verify_accepts_bundle_under_graph_capture(bundle: str) -> None:
    result = _cli(
        "verify", f"examples/{bundle}", "--device", "cuda", "--dtype", "bfloat16"
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    # CUDA graphs are part of the scored contract; a verify that silently ran
    # eager-only would pass numerics while proving nothing about capture.
    assert "graph_replays=" in result.stdout


@pytest.mark.skipif(not _CUDA, reason="CUDA verify tier requires a GPU")
@pytest.mark.parametrize("bundle", _CUDA_FAIL_BUNDLES)
def test_cuda_verify_fails_broken_bundle_with_verdict_exit(bundle: str) -> None:
    result = _cli(
        "verify", f"examples/{bundle}", "--device", "cuda", "--dtype", "bfloat16"
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "FAIL" in result.stdout


@pytest.mark.skipif(not _CUDA2, reason="distributed verify requires two CUDA devices")
def test_two_device_collective_verify_accepts_allreduce_bundle() -> None:
    result = _cli(
        "verify",
        "examples/miner_allreduce_torch",
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--world-size",
        "2",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "ok" in result.stdout

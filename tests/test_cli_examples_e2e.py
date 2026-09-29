"""End-to-end verdict contract of the real CLI entry point on example bundles.

Unit suites import :mod:`cacheon.cli` in-process; nothing there executes the
documented contributor invocation. These tests run ``python -m cacheon.cli``
as a subprocess, so entry-point wiring, bundle loading, and process exit codes
are exercised as one observable contract: ``0`` verified, ``2`` verdict FAIL.

``verify`` is a node bundle's static scan plus an import/signature smoke in a
spawned child; its numerical checks need the arena image (``cacheon check``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

pytestmark = pytest.mark.skipif(
    not EXAMPLES.is_dir(), reason="example bundles require a source checkout"
)


def _cli(*args: str, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cacheon.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_scan_accepts_the_documented_bundle() -> None:
    result = _cli("scan", "examples/miner_node_identity")
    assert result.returncode == 0, result.stderr


def test_verify_accepts_the_documented_bundle() -> None:
    result = _cli("verify", "examples/miner_node_identity")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "[INTERFACE OK] model.layers.*.mlp" in result.stdout


def test_verify_fails_a_bundle_whose_entry_cannot_take_the_node_call(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "broken"
    shutil.copytree(EXAMPLES / "miner_node_identity", bundle)
    (bundle / "kernels" / "forward.py").write_text("def forward():\n    return None\n")
    result = _cli("verify", str(bundle))
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "[FAIL]" in result.stderr

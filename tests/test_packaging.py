from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from optima.source_release import REQUIRED_RUNTIME_FILES


def test_wheel_contains_referee_package_data_and_arena_assets(tmp_path):
    """Build from a clean minimal tree so editable-install leakage cannot mask omissions."""

    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(root / "pyproject.toml", source / "pyproject.toml")
    for package in ("optima", "optima_kernels"):
        shutil.copytree(
            root / package,
            source / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(source),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wheels = tuple(wheelhouse.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        files = set(archive.namelist())

    expected = set(REQUIRED_RUNTIME_FILES)
    expected.update(
        {
            "optima/arena_assets/minimax_m3/sglang_patch/flashinfer_trtllm.py",
            "optima/arena_assets/minimax_m3/sglang_patch/modelopt_quant.py",
            "optima/model_provision.py",
        }
    )
    assert expected <= files

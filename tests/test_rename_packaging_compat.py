from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from cacheon import bootstrap


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_installs_one_cacheon_finder_without_legacy_alias() -> None:
    """The rename-era optima.bootstrap alias machinery is retired outright."""

    assert not hasattr(bootstrap, "_retire_legacy_bootstrap")
    assert not hasattr(bootstrap, "_is_legacy_seam_finder")
    assert "optima" not in sys.modules
    assert "optima.bootstrap" not in sys.modules
    finders = [
        finder
        for finder in sys.meta_path
        if isinstance(finder, bootstrap._SeamFinder)
    ]
    assert len(finders) == 1


def test_wheel_build_removes_retired_packages_and_overlays(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for relative in (
        "LICENSE",
        "NOTICE",
        "README.md",
        "pyproject.toml",
        "setup.py",
    ):
        shutil.copy2(ROOT / relative, source / relative)
    shutil.copytree(ROOT / "cacheon", source / "cacheon")
    shutil.copytree(ROOT / "cacheon_kernels", source / "cacheon_kernels")

    stale_packages = (
        source / "build/lib/optima",
        source / "build/lib/optima_kernels",
        source / "build/lib/cacheon/arena_assets/minimax_m3/sglang_patch",
    )
    for package in stale_packages:
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "raise RuntimeError('retired package was installed')\n",
            encoding="utf-8",
        )

    output = source / "dist"
    output.mkdir()
    env = {
        **os.environ,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools import build_meta; build_meta.build_wheel('dist')",
        ],
        cwd=source,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    wheel = next(output.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "cacheon/bootstrap.py" in names
    assert not any(
        name == "optima.py"
        or name.startswith("optima/")
        or name.startswith("optima_kernels/")
        or name.startswith("cacheon/arena_assets/")
        for name in names
    )

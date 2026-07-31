"""Setuptools command customizations for reproducible rename builds."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class _CleanLegacyBuildPy(_build_py):
    """Keep pre-rename packages out of a reused setuptools build directory."""

    def run(self) -> None:
        build_lib = Path(self.build_lib)
        if build_lib.is_dir():
            for legacy_path in build_lib.glob("optima*"):
                if legacy_path.is_dir() and not legacy_path.is_symlink():
                    shutil.rmtree(legacy_path)
                else:
                    legacy_path.unlink()
        super().run()


setup(cmdclass={"build_py": _CleanLegacyBuildPy})

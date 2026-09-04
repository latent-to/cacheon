"""Keep retired payloads out of reused setuptools build directories."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class _CleanLegacyBuildPy(_build_py):
    """Discard pre-rename packages and the retired M3 overlay payload."""

    def run(self) -> None:
        build_lib = Path(self.build_lib)
        if build_lib.is_dir():
            for legacy_path in (*build_lib.glob("optima*"), build_lib / "cacheon/arena_assets"):
                if legacy_path.is_dir() and not legacy_path.is_symlink():
                    shutil.rmtree(legacy_path)
                elif legacy_path.exists() or legacy_path.is_symlink():
                    legacy_path.unlink()
        super().run()


setup(cmdclass={"build_py": _CleanLegacyBuildPy})

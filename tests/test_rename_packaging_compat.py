from __future__ import annotations

import importlib
import importlib.abc
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
import zipfile

import pytest

from cacheon import bootstrap


ROOT = Path(__file__).resolve().parents[1]


def _legacy_optima_finder(
    module_name: str, activations: list[str]
) -> importlib.abc.MetaPathFinder:
    class _SeamFinder(importlib.abc.MetaPathFinder):
        """The delegating and wrapping behavior shipped by optima-harness."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != module_name:
                return None
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is None:
                    continue
                if spec.loader is not None:
                    original_exec = spec.loader.exec_module

                    def exec_module(module):
                        original_exec(module)
                        activations.append("optima")

                    spec.loader.exec_module = exec_module
                return spec
            return None

    # The transition identifies the exact finder class shipped by the old
    # package without importing that package during interpreter startup.
    _SeamFinder.__module__ = "optima.bootstrap"
    return _SeamFinder()


@pytest.mark.parametrize(
    "startup_order", ("optima-then-cacheon", "cacheon-then-optima")
)
def test_cacheon_bootstrap_retires_legacy_finder_in_either_startup_order(
    tmp_path: Path, monkeypatch, startup_order: str
) -> None:
    module_name = f"cacheon_bootstrap_rename_probe_{startup_order.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text("loaded = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bootstrap, "_TARGETS", (module_name,))
    monkeypatch.delitem(sys.modules, "optima", raising=False)
    monkeypatch.delitem(sys.modules, "optima.bootstrap", raising=False)

    activations: list[str] = []
    legacy_finder = _legacy_optima_finder(module_name, activations)
    other_finders = [
        finder
        for finder in sys.meta_path
        if not isinstance(finder, bootstrap._SeamFinder)
        and not bootstrap._is_legacy_seam_finder(finder)
    ]
    monkeypatch.setattr(sys, "meta_path", other_finders)
    monkeypatch.setattr(
        bootstrap, "_run_activate", lambda: activations.append("cacheon")
    )

    state_name = f"cacheon_rename_legacy_state_{startup_order.replace('-', '_')}"
    legacy_state = ModuleType(state_name)
    legacy_state.finder = legacy_finder
    monkeypatch.setitem(sys.modules, state_name, legacy_state)
    legacy_package = tmp_path / "optima"
    legacy_package.mkdir()
    (legacy_package / "__init__.py").write_text("", encoding="utf-8")
    (legacy_package / "bootstrap.py").write_text(
        f"from {state_name} import finder\n"
        "import sys\n"
        "sys.meta_path.insert(0, finder)\n",
        encoding="utf-8",
    )

    if startup_order == "optima-then-cacheon":
        __import__("optima.bootstrap")
        assert any(
            bootstrap._is_legacy_seam_finder(finder) for finder in sys.meta_path
        )

    bootstrap.install()

    # This models the later .pth line in the Cacheon-first order. The alias
    # must return Cacheon's module without executing the old bootstrap body.
    __import__("optima.bootstrap")
    assert sys.modules["optima.bootstrap"] is bootstrap
    assert not any(
        bootstrap._is_legacy_seam_finder(finder) for finder in sys.meta_path
    )

    try:
        imported = importlib.import_module(module_name)
        assert imported.loaded is True
        assert activations == ["cacheon"]
    finally:
        sys.modules.pop(module_name, None)


def test_wheel_build_removes_stale_optima_packages(tmp_path: Path) -> None:
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
    shutil.copytree(ROOT / "LICENSES", source / "LICENSES")
    shutil.copytree(ROOT / "cacheon", source / "cacheon")
    shutil.copytree(ROOT / "cacheon_kernels", source / "cacheon_kernels")

    stale_packages = (
        source / "build/lib/optima",
        source / "build/lib/optima_kernels",
    )
    for package in stale_packages:
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "raise RuntimeError('stale Optima package was installed')\n",
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
        for name in names
    )

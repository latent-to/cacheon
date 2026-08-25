"""Version-independent seam loader via a post-import hook.

SGLang's released versions don't all ship the ``sglang.srt.plugins`` framework,
and ``sglang.Engine`` runs the model in a spawned scheduler process. To install
the Cacheon seam reliably in *every* interpreter in the venv — including that
spawned child — we drop a ``.pth`` file in site-packages containing the single
line ``import cacheon.bootstrap``. Python executes ``.pth`` imports at interpreter
startup (in spawned children too), so this module loads everywhere.

At startup sglang is not yet imported, and importing it here would be heavy and
fragile. So instead we register a meta-path finder that defers patching until
``sglang.srt.layers.activation`` is actually imported, then runs ``seam.activate``
against the freshly-loaded module. ``seam.activate`` is env-driven, so a baseline
process (CACHEON_ACTIVE unset) just installs a pass-through dispatcher.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

# Modules whose import should trigger seam installation — derived from the single seam
# table (cacheon/seams.py), so adding a seam there is the only edit. seams.py is stdlib-only
# (no torch/sglang), safe to import at interpreter startup. seam.activate() installs
# whatever is loaded.
from cacheon.seams import TARGET_MODULES as _TARGETS

_FLASHINFER_ENV = "flashinfer.jit.env"


def _run_activate(_module=None) -> None:
    from cacheon import seam

    seam.activate()


def _redirect_flashinfer_cubins(module) -> None:
    """Apply upstream #3062 semantics to pinned FlashInfer 0.6.12."""

    target = os.environ.get("FLASHINFER_CUBIN_DIR", "")
    if target:
        module.FLASHINFER_CUBIN_DIR = Path(target)


def _wrap_loader(loader, callback=_run_activate):
    orig_exec = loader.exec_module

    def exec_module(module):
        orig_exec(module)
        callback(module)

    loader.exec_module = exec_module
    return loader


class _SeamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS and fullname != _FLASHINFER_ENV:
            return None
        # Resolve SGLang through the standard path machinery instead of walking
        # ``sys.meta_path`` ourselves: PathFinder honors the normal filesystem
        # and zip path hooks used by installed SGLang distributions without
        # re-entering this seam finder.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            callback = (
                _redirect_flashinfer_cubins
                if fullname == _FLASHINFER_ENV
                else _run_activate
            )
            spec.loader = _wrap_loader(spec.loader, callback)
        return spec


def install() -> None:
    if _FLASHINFER_ENV in sys.modules:
        _redirect_flashinfer_cubins(sys.modules[_FLASHINFER_ENV])
    if any(t in sys.modules for t in _TARGETS):
        # Something already imported (e.g. re-entry); patch what's available now.
        _run_activate()
    if not any(isinstance(f, _SeamFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _SeamFinder())


install()

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

# Modules whose import should trigger seam installation — derived from the single seam
# table (cacheon/seams.py), so adding a seam there is the only edit. seams.py is stdlib-only
# (no torch/sglang), safe to import at interpreter startup. seam.activate() installs
# whatever is loaded.
from cacheon.seams import TARGET_MODULES as _TARGETS


_LEGACY_BOOTSTRAP_MODULE = "optima.bootstrap"


# Install the standard-library-only discovery role hook before any SGLang
# import.  The hook is inert unless the isolated worker explicitly arms a
# sealed overlay, and only exact scheduler spawn children receive that overlay.
if os.environ.get("CACHEON_DISCOVERY_OVERLAY_ARMED", "").strip().lower() in {
    "1", "true", "yes", "on",
}:
    from cacheon import discovery_overlay as _discovery_overlay

    _discovery_overlay.install()


def _run_activate() -> None:
    try:
        from cacheon import seam

        seam.activate()
    except Exception:  # noqa: BLE001 - never break the host interpreter
        import logging

        logging.getLogger("cacheon.bootstrap").exception("cacheon: seam activate failed")


def _wrap_loader(loader):
    orig_exec = loader.exec_module

    def exec_module(module):
        orig_exec(module)
        _run_activate()

    loader.exec_module = exec_module
    return loader


class _SeamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        # Resolve SGLang through the standard path machinery instead of walking
        # ``sys.meta_path`` ourselves. During an in-place Optima -> Cacheon
        # upgrade the legacy ``optima.bootstrap`` finder can still be installed;
        # asking it for a spec makes it delegate back to this finder forever.
        # PathFinder still honors the normal filesystem and zip path hooks used
        # by installed SGLang distributions without re-entering either seam
        # finder.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _wrap_loader(spec.loader)
        return spec


def _is_legacy_seam_finder(finder: object) -> bool:
    finder_type = type(finder)
    return (
        finder_type.__module__ == _LEGACY_BOOTSTRAP_MODULE
        and finder_type.__name__ == "_SeamFinder"
    )


def _retire_legacy_bootstrap() -> None:
    """Make Cacheon authoritative when both rename-era bootstraps are installed.

    Site processes ``.pth`` files by filename. The Cacheon release bootstrap
    normally runs before the legacy Optima bootstrap, so reserving the old
    module name prevents its module body from installing a second finder. If
    the legacy module ran first, remove its finder and replace both its module
    cache entry and the parent-package attribute.
    """

    current = sys.modules[__name__]
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not _is_legacy_seam_finder(finder)
    ]
    sys.modules[_LEGACY_BOOTSTRAP_MODULE] = current
    legacy_package = sys.modules.get("optima")
    if legacy_package is not None:
        setattr(legacy_package, "bootstrap", current)


def install() -> None:
    _retire_legacy_bootstrap()
    if any(t in sys.modules for t in _TARGETS):
        # Something already imported (e.g. re-entry); patch what's available now.
        _run_activate()
    if not any(isinstance(f, _SeamFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _SeamFinder())


install()

"""Scheduler-role-only import activation for prebuilt SGLang system overlays.

The trusted timing driver must import stock SGLang.  Merely exporting a
``PYTHONPATH`` overlay would poison that driver and every helper process, so the
validator bootstrap installs a narrow multiprocessing role hook instead:

* ``BaseProcess.start`` recognizes SGLang's exact scheduler target and marks only
  that child in the inherited environment;
* the fresh child interpreter executes Optima's site ``.pth`` before unpickling
  the target, validates the content-addressed read-only overlay, and prepends its
  site root before the first SGLang import;
* detokenizer/tokenizer/controller/driver processes remain on the stock package.

An activation failure installs a blocking import finder.  Site machinery can
swallow exceptions raised by a ``.pth`` import, so raising alone would fail open
to stock; the guard makes the scheduler's subsequent SGLang import fail closed.
"""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.machinery
import logging
import multiprocessing
import multiprocessing.process
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

logger = logging.getLogger("optima.system_overlay")


_ARMED = "OPTIMA_SYSTEM_OVERLAY_ARMED"
_ROLE = "OPTIMA_SYSTEM_PROCESS_ROLE"
_ROLE_PARENT = "OPTIMA_SYSTEM_ROLE_PARENT_PID"
_DRIVER_PID = "OPTIMA_SYSTEM_DRIVER_PID"
_BUNDLE = "OPTIMA_SYSTEM_BUNDLE_PATH"
_TARGET = "OPTIMA_SYSTEM_COMPETITION_TARGET"
_ARENA = "OPTIMA_SYSTEM_ARENA"
_CACHE_ROOT = "OPTIMA_SYSTEM_OVERLAY_ROOT"
_EXPECTED_KEY = "OPTIMA_SYSTEM_EXPECTED_CACHE_KEY"
_ACTIVE_KEY = "OPTIMA_SYSTEM_OVERLAY_ACTIVE"
_FAILED = "OPTIMA_SYSTEM_OVERLAY_FAILED"
_SCHEDULER_ROLE = "scheduler"

_ENV_SPAWN_LOCK = threading.RLock()


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _target_identity(target) -> tuple[str, str]:
    """Return the stable module/qualname under partials and bound methods."""
    while isinstance(target, functools.partial):
        target = target.func
    target = getattr(target, "__func__", target)
    return (
        str(getattr(target, "__module__", "")),
        str(getattr(target, "__qualname__", getattr(target, "__name__", ""))),
    )


def role_for_process_target(target) -> str | None:
    """Classify only validator-known SGLang scheduler entrypoints."""
    module, qualname = _target_identity(target)
    if (
        module == "sglang.srt.managers.scheduler"
        and qualname == "run_scheduler_process"
    ):
        return _SCHEDULER_ROLE
    return None


def install_process_role_hook() -> None:
    """Tag only exact scheduler spawn children; safe to call in every interpreter."""
    base = multiprocessing.process.BaseProcess
    current = base.start
    if getattr(current, "__optima_system_role_hook__", False):
        return
    original = current

    def start(process, *args, **kwargs):
        if not _truthy(os.environ.get(_ARMED)):
            return original(process, *args, **kwargs)
        role = role_for_process_target(getattr(process, "_target", None))
        # Every armed spawn takes the lock: a non-scheduler process starting while
        # the scheduler marker is exported must not inherit the candidate role.
        with _ENV_SPAWN_LOCK:
            if role is None:
                return original(process, *args, **kwargs)
            start_method = getattr(process, "_start_method", None)
            if start_method is None:
                # ``multiprocessing.Process`` delegates to the selected global
                # context and leaves this private attribute unset. SGLang selects
                # spawn before constructing scheduler processes.
                start_method = multiprocessing.get_start_method(allow_none=False)
            if start_method != "spawn":
                raise RuntimeError(
                    "system overlays require a spawn scheduler so Optima bootstrap "
                    "can validate the overlay before the first SGLang import; got "
                    f"{start_method!r}"
                )
            saved_role = os.environ.get(_ROLE)
            saved_parent = os.environ.get(_ROLE_PARENT)
            os.environ[_ROLE] = role
            os.environ[_ROLE_PARENT] = str(os.getpid())
            try:
                return original(process, *args, **kwargs)
            finally:
                if saved_role is None:
                    os.environ.pop(_ROLE, None)
                else:
                    os.environ[_ROLE] = saved_role
                if saved_parent is None:
                    os.environ.pop(_ROLE_PARENT, None)
                else:
                    os.environ[_ROLE_PARENT] = saved_parent

    start.__optima_system_role_hook__ = True
    start.__optima_original_start__ = original
    base.start = start


def _required(env: Mapping[str, str], key: str) -> str:
    value = str(env.get(key, "")).strip()
    if not value:
        raise RuntimeError(f"system overlay bootstrap missing {key}")
    return value


def activate_scheduler_overlay(
    *,
    env: MutableMapping[str, str] | None = None,
    pid: int | None = None,
    parent_pid: int | None = None,
    modules: Mapping[str, object] | None = None,
    sys_path: list[str] | None = None,
    reader: Callable[..., tuple] | None = None,
    read_only_check: Callable[[Path], bool] | None = None,
    defer_to_import: bool = False,
) -> Path | None:
    """Pure-testable core of the scheduler bootstrap activation.

    Returns the prepended overlay site root, or ``None`` in every non-scheduler
    process.  The default path performs full identity/stamp/tree verification and
    requires a live read-only mount.
    """
    environ = os.environ if env is None else env
    if not _truthy(environ.get(_ARMED)):
        return None
    role = str(environ.get(_ROLE, "")).strip()
    if not role:
        return None  # driver, tokenizer/detokenizer, or controller
    if role != _SCHEDULER_ROLE:
        raise RuntimeError(f"unrecognized system overlay process role: {role!r}")

    actual_pid = os.getpid() if pid is None else int(pid)
    actual_parent = os.getppid() if parent_pid is None else int(parent_pid)
    try:
        driver_pid = int(_required(environ, _DRIVER_PID))
        marked_parent = int(_required(environ, _ROLE_PARENT))
    except ValueError as exc:
        raise RuntimeError("system overlay PID role markers are malformed") from exc
    if actual_pid == driver_pid:
        raise RuntimeError("refusing to activate a candidate overlay in the timing driver")
    if actual_parent != marked_parent:
        raise RuntimeError(
            "system overlay scheduler role parent mismatch; refusing inherited/forged role"
        )

    loaded = sys.modules if modules is None else modules
    if any(name == "sglang" or name.startswith("sglang.") for name in loaded):
        raise RuntimeError(
            "system overlay activation occurred after SGLang import; refusing mixed stock/overlay modules"
        )

    bundle = _required(environ, _BUNDLE)
    target = _required(environ, _TARGET)
    arena = _required(environ, _ARENA)
    root = Path(_required(environ, _CACHE_ROOT)).resolve()
    expected_key = _required(environ, _EXPECTED_KEY)

    if reader is None:
        from optima.system_patch import read_validated_system_overlay

        kwargs = {}
        if read_only_check is not None:
            kwargs["read_only_check"] = read_only_check
        identity, _stamp, dest = read_validated_system_overlay(
            bundle,
            competition_target=target,
            arena_name=arena,
            cache_root=root,
            require_read_only=True,
            **kwargs,
        )
    else:
        identity, _stamp, dest = reader(
            bundle_path=bundle,
            competition_target=target,
            arena_name=arena,
            cache_root=root,
            require_read_only=True,
            read_only_check=read_only_check,
        )
    if identity.cache_key != expected_key:
        raise RuntimeError(
            "system overlay cache identity changed between trusted controller and scheduler"
        )
    site = Path(dest).resolve() / "site"
    package = site / "sglang"
    if site.is_symlink() or package.is_symlink() or not package.is_dir():
        raise RuntimeError(f"validated system overlay has no safe SGLang package: {site}")

    if not defer_to_import:
        paths = sys.path if sys_path is None else sys_path
        site_text = str(site)
        if site_text in paths:
            paths.remove(site_text)
        paths.insert(0, site_text)
        importlib.invalidate_caches()
        environ[_ACTIVE_KEY] = expected_key
    return site


class _OverlayReceiptLoader(importlib.abc.Loader):
    """Delegate package execution, then attest its actual loaded origin."""

    def __init__(self, delegate, expected_origin: Path, complete: Callable[[Path], None]):
        self.delegate = delegate
        self.expected_origin = expected_origin
        self.complete = complete

    def create_module(self, spec):
        create = getattr(self.delegate, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        try:
            self.delegate.exec_module(module)
            actual = Path(module.__file__).resolve(strict=True)
            if actual != self.expected_origin:
                raise ImportError("loaded SGLang package escaped the validated overlay")
            self.complete(actual)
        except BaseException as exc:  # noqa: BLE001 - every retry must remain blocked
            reason = f"{type(exc).__name__}: {exc}"[:2048]
            _install_failure_guard(reason)
            raise

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _SchedulerOverlayFinder(importlib.abc.MetaPathFinder):
    """Force the validated package at the first post-spawn SGLang import.

    ``multiprocessing.spawn.prepare`` restores the parent's ``sys.path`` after
    site startup. A path inserted by the ``.pth`` hook can therefore disappear
    before the scheduler target is unpickled. ``sys.meta_path`` survives that
    preparation step, so this finder is the actual import-time authority.
    """

    def __init__(
        self,
        site: Path,
        expected_key: str,
        environ: MutableMapping[str, str] | None = None,
    ):
        self.site = site.resolve()
        self.expected_key = expected_key
        self.environ = os.environ if environ is None else environ

    def _complete(self, module_origin: Path) -> None:
        self.environ[_ACTIVE_KEY] = self.expected_key
        from optima import receipts

        receipts.write(
            "system_active",
            {
                "pid": os.getpid(),
                "target": self.environ.get(_TARGET, ""),
                "arena": self.environ.get(_ARENA, ""),
                "cache_key": self.expected_key,
                "overlay_site": str(self.site),
                "module_origin": str(module_origin),
            },
        )
        logger.info("optima: scheduler imported validated system overlay %s", self.site)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang":
            if fullname.startswith("sglang."):
                raise ImportError(
                    "candidate scheduler attempted an SGLang submodule import "
                    "before the validated package root"
                )
            return None
        package = self.site / "sglang"
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(self.site)])
        origin = getattr(spec, "origin", None) if spec is not None else None
        expected_origin = package / "__init__.py"
        try:
            resolved_origin = Path(origin).resolve(strict=True)
        except (TypeError, OSError, RuntimeError):
            resolved_origin = None
        locations = tuple(
            Path(value).resolve()
            for value in (getattr(spec, "submodule_search_locations", None) or ())
        ) if spec is not None else ()
        if (
            spec is None
            or spec.loader is None
            or resolved_origin != expected_origin.resolve(strict=True)
            or locations != (package.resolve(strict=True),)
        ):
            raise ImportError(
                "validated system overlay did not resolve the exact SGLang package root"
            )
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        site_text = str(self.site)
        if site_text in sys.path:
            sys.path.remove(site_text)
        sys.path.insert(0, site_text)
        inherited = [
            value
            for value in self.environ.get("PYTHONPATH", "").split(os.pathsep)
            if value and value != site_text
        ]
        self.environ["PYTHONPATH"] = os.pathsep.join([site_text, *inherited])
        importlib.invalidate_caches()
        spec.loader = _OverlayReceiptLoader(
            spec.loader,
            resolved_origin,
            self._complete,
        )
        return spec


class _BlockSglangImport(importlib.abc.MetaPathFinder):
    def __init__(self, reason: str):
        self.reason = reason

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang" or fullname.startswith("sglang."):
            raise ImportError(
                "candidate scheduler system-overlay activation failed closed: "
                + self.reason
            )
        return None


def _install_failure_guard(reason: str) -> None:
    if not any(isinstance(finder, _BlockSglangImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BlockSglangImport(reason))
    os.environ[_FAILED] = reason[:2048]


def install() -> None:
    """Install role plumbing and arm import-time activation in a scheduler child."""
    # A DP controller is itself stock, but may spawn scheduler ranks later; every
    # interpreter gets the role hook.
    install_process_role_hook()
    if not _truthy(os.environ.get(_ARMED)) or not os.environ.get(_ROLE):
        return
    try:
        site = activate_scheduler_overlay(defer_to_import=True)
        if site is not None:
            sys.meta_path.insert(
                0,
                _SchedulerOverlayFinder(
                    site, _required(os.environ, _EXPECTED_KEY)
                ),
            )
            logger.info("optima: scheduler armed validated system overlay %s", site)
    except Exception as exc:  # noqa: BLE001 - convert every failure into import refusal
        reason = f"{type(exc).__name__}: {exc}"
        logger.critical("optima: %s", reason, exc_info=True)
        _install_failure_guard(reason)
    finally:
        # These markers identify exactly one spawn child. Leaving them in the
        # scheduler environment would misclassify its later helper subprocesses.
        os.environ.pop(_ROLE, None)
        os.environ.pop(_ROLE_PARENT, None)


def driver_import_is_stock(module_file: str | Path, overlay_dest: str | Path) -> bool:
    """Controller assertion helper used after importing SGLang in the driver."""
    module = Path(module_file).resolve()
    overlay_site = (Path(overlay_dest).resolve() / "site")
    return module != overlay_site and overlay_site not in module.parents


def driver_module_is_stock(module: object, overlay_dest: str | Path) -> bool:
    """Check the imported package path without resolving lazy public attributes."""

    module_file = getattr(module, "__file__", None)
    return isinstance(module_file, (str, Path)) and driver_import_is_stock(
        module_file, overlay_dest
    )

"""Serve an object of the engine from a bundle: a class, a method or a module-level function, by dotted path.

An address under ``sglang`` names what the engine itself calls by that path, such as
``sglang.srt.managers.schedule_policy.SchedulePolicy.calc_priority`` or
``sglang.srt.disaggregation.decode.DecodePreallocQueue``. The bundle supplies ``entry(stock)``: it
receives the stock object and returns its replacement, a callable for a function or a method and a
subclass for a class. ``None`` leaves this process stock, which is how a decode-side object is declined on
a prefill server. The replacement is bound once, when the bundle loads at a scheduler's process entry and
before SGLang builds the scheduler, so every later construction and call reaches it; a name another engine
module imported from the defining module is rebound with it. A replacement is the object's new source, so
a winning one reads as a patch to the file that defines the object.

Nothing here knows what a scheduler method should return, so no call is checked. The engine boundary
judges the result: the speed stage, quality against the teacher, and pristine T. A raise inside a
replacement is receipted as the candidate's and still takes the engine down.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import sys

from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor
from cacheon.integrations.sglang_cache import DECLINED
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.seams import ENGINE_OBJECT as ROOT

_bound: set[str] = set()


def _refuse(address: str, message: str, error: type[Exception] = TypeError):
    failure = error(f"{address}: {message}")
    _receipts.failed(address, failure, phase="prepare")
    raise failure


def _resolve(address: str):
    """The module or class that defines the named object, the name it holds there, and the raw attribute."""

    parts = address.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            owner = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for name in parts[cut:-1]:
            owner = vars(owner).get(name)
            if not inspect.isclass(owner):
                _refuse(address, f"{name!r} is not a class defined on that path", LookupError)
        return owner, parts[-1], vars(owner).get(parts[-1])
    _refuse(address, "names no module of this engine", LookupError)


def _receipted(address: str, function):
    """Run a replacement so its raise is the candidate's. The candidate flag stays clear: a replaced
    scheduler method encloses forward passes, and a flagged call would serve every node inside it stock."""

    @functools.wraps(function)
    def call(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            _receipts.failed(address, exc, entry=function)
            raise
        _receipts.completed(address)
        return result

    return call


def _bind(address: str, impl) -> None:
    if impl.prepare is not None:
        _refuse(address, "an engine entry takes no prepare")
    owner, name, raw = _resolve(address)
    descriptor = type(raw) if isinstance(raw, (staticmethod, classmethod)) else None
    stock = raw.__func__ if descriptor else raw
    if not (inspect.isclass(stock) or inspect.isfunction(stock)):
        _refuse(address, "names neither a class nor a function defined on that path", LookupError)
    if inspect.ismodule(owner) and stock.__module__ != owner.__name__:  # one object, one address
        _refuse(address, f"that object is defined in {stock.__module__}; name it there", LookupError)
    try:
        new = impl.entry(stock)
    except BaseException as exc:
        _receipts.failed(address, exc, phase="prepare", entry=impl.entry)
        raise
    if new is None:
        _receipts.not_selected(address, DECLINED, ())
        return
    if inspect.isclass(stock):
        if not inspect.isclass(new) or new is stock or not issubclass(new, stock):
            _refuse(address, "entry must return a subclass of the stock class")
        for key, value in list(vars(new).items()):
            if inspect.isfunction(value):
                setattr(new, key, _receipted(address, value))
        # Construction is the execution every subclass has; a slot wrapper (``object.__init__``) stays as it is.
        if "__init__" not in vars(new) and inspect.isfunction(new.__init__):
            new.__init__ = _receipted(address, new.__init__)
        new.__name__, new.__qualname__, new.__module__ = stock.__name__, stock.__qualname__, stock.__module__
    elif not callable(new):
        _refuse(address, "entry must return a callable")
    else:
        new = functools.wraps(stock)(_receipted(address, new))
    setattr(owner, name, descriptor(new) if descriptor else new)
    if inspect.ismodule(owner):
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").split(".")[0] == ROOT and module is not owner:
                for key, value in list(vars(module).items()):
                    if value is stock:
                        setattr(module, key, new)


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Bind every engine address the loaded bundle names; a no-op until a bundle is live in this process."""

    for address in sorted(s for s in registry.slots() if s.split(".")[0] == ROOT and s not in _bound):
        impl = registry.select(address, CallDescriptor()).impl
        if impl is not None:
            _bound.add(address)
            _bind(address, impl)

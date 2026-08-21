"""Atomically bind the decode dispatcher to SGLang's source and by-value consumer."""

from __future__ import annotations

from types import ModuleType

from cacheon import audit as _audit
from cacheon.minimax_sparse_decode_dispatch import (
    make_minimax_sparse_decode_dispatcher,
)
from cacheon.registry import REGISTRY, KernelRegistry

_SOURCE = "sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse"
_CONSUMER = "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse"
_FUNCTION = "flash_decode_with_gqa_share_sparse"
_PATCHED = "_cacheon_sparse_decode_patched"
_ORIGINAL = "_cacheon_orig_flash_decode_with_gqa_share_sparse"
_DISPATCHER = "_cacheon_dispatch_flash_decode_with_gqa_share_sparse"
_BACKEND = "sglang.srt.layers.attention.minimax_sparse_backend"
_BACKEND_CLASS = "MiniMaxSparseAttnBackend"
_BACKEND_ORIGINAL = "_cacheon_orig_decode_audit_init"


def _initializing(module: ModuleType) -> bool:
    spec = getattr(module, "__spec__", None)
    return bool(spec is not None and getattr(spec, "_initializing", False))


def _consumer_state(
    consumer: ModuleType | None, *, original: object, dispatcher: object
) -> str:
    if consumer is None:
        return "absent"
    if not hasattr(consumer, _FUNCTION):
        if _initializing(consumer):
            return "initializing"
        raise RuntimeError("MiniMax sparse-decode consumer lacks its stock binding")
    binding = getattr(consumer, _FUNCTION)
    if binding is original:
        return "original"
    if binding is dispatcher:
        return "dispatcher"
    raise RuntimeError("MiniMax sparse-decode consumer binding drifted")


def _installed(source: ModuleType) -> tuple[object, object]:
    try:
        original = getattr(source, _ORIGINAL)
        dispatcher = getattr(source, _DISPATCHER)
    except AttributeError as exc:
        raise RuntimeError("MiniMax sparse-decode seam has partial state") from exc
    if original is dispatcher or getattr(source, _FUNCTION, None) is not dispatcher:
        raise RuntimeError("MiniMax sparse-decode defining binding drifted")
    return original, dispatcher


def _install_audit_mode() -> None:
    import sys

    module = sys.modules.get(_BACKEND)
    cls = getattr(module, _BACKEND_CLASS, None) if module is not None else None
    if cls is None or getattr(cls, _PATCHED, False):
        return
    original = cls.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if _audit.enabled():
            self._use_msa_decode = False
            self._msa_owns_decode = False

    setattr(cls, _BACKEND_ORIGINAL, original)
    cls.__init__ = initialize
    setattr(cls, _PATCHED, True)


def install(registry: KernelRegistry = REGISTRY) -> None:
    import sys

    _install_audit_mode()
    source = sys.modules.get(_SOURCE)
    if source is None:
        return
    consumer = sys.modules.get(_CONSUMER)
    if getattr(source, _PATCHED, False):
        original, dispatcher = _installed(source)
        if _consumer_state(
            consumer, original=original, dispatcher=dispatcher
        ) == "original":
            setattr(consumer, _FUNCTION, dispatcher)
        return
    if hasattr(source, _ORIGINAL) or hasattr(source, _DISPATCHER):
        raise RuntimeError("MiniMax sparse-decode seam has stale patch state")
    original = getattr(source, _FUNCTION, None)
    if original is None:
        return
    if not callable(original):
        raise RuntimeError("MiniMax sparse-decode stock binding is not callable")
    dispatcher = make_minimax_sparse_decode_dispatcher(original, registry=registry)
    state = _consumer_state(
        consumer, original=original, dispatcher=dispatcher
    )
    setattr(source, _ORIGINAL, original)
    setattr(source, _DISPATCHER, dispatcher)
    setattr(source, _FUNCTION, dispatcher)
    setattr(source, _PATCHED, True)
    if state == "original":
        setattr(consumer, _FUNCTION, dispatcher)


def uninstall() -> None:
    import sys

    backend = sys.modules.get(_BACKEND)
    cls = getattr(backend, _BACKEND_CLASS, None) if backend is not None else None
    if cls is not None and getattr(cls, _PATCHED, False):
        cls.__init__ = getattr(cls, _BACKEND_ORIGINAL)
        delattr(cls, _BACKEND_ORIGINAL)
        setattr(cls, _PATCHED, False)
    source = sys.modules.get(_SOURCE)
    if source is None or not getattr(source, _PATCHED, False):
        return
    original, dispatcher = _installed(source)
    consumer = sys.modules.get(_CONSUMER)
    state = _consumer_state(
        consumer, original=original, dispatcher=dispatcher
    )
    if state == "dispatcher":
        setattr(consumer, _FUNCTION, original)
    setattr(source, _FUNCTION, original)
    delattr(source, _ORIGINAL)
    delattr(source, _DISPATCHER)
    setattr(source, _PATCHED, False)


def is_installed() -> bool:
    import sys

    source = sys.modules.get(_SOURCE)
    if source is None or not getattr(source, _PATCHED, False):
        return False
    backend = sys.modules.get(_BACKEND)
    cls = getattr(backend, _BACKEND_CLASS, None) if backend is not None else None
    if cls is not None and not (
        getattr(cls, _PATCHED, False) and hasattr(cls, _BACKEND_ORIGINAL)
    ):
        return False
    try:
        original, dispatcher = _installed(source)
        state = _consumer_state(
            sys.modules.get(_CONSUMER), original=original, dispatcher=dispatcher
        )
    except RuntimeError:
        return False
    return state in {"absent", "initializing", "dispatcher"}


__all__ = ["install", "is_installed", "uninstall"]

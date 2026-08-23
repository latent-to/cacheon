"""MiniMax decode adapter-specific score and audit hooks."""

import sys
from types import ModuleType

from cacheon.integrations import sglang_minimax_sparse_decode as seam


def _stock(*_args, **_kwargs):
    return "stock"


def _decode_source(monkeypatch):
    source = ModuleType(seam._PATCH.source_module)
    setattr(source, seam._PATCH.function, _stock)
    consumer = ModuleType(seam._PATCH.consumer_module)
    setattr(consumer, seam._PATCH.function, _stock)
    monkeypatch.setitem(sys.modules, seam._PATCH.source_module, source)
    monkeypatch.setitem(sys.modules, seam._PATCH.consumer_module, consumer)
    return source, consumer


def test_decode_factory_binds_source_and_consumer(monkeypatch) -> None:
    source, consumer = _decode_source(monkeypatch)
    calls = []

    def make(original, *, registry):
        calls.append((original, registry))
        return lambda *args, **kwargs: original(*args, **kwargs)

    monkeypatch.setattr(seam, "make_minimax_sparse_decode_dispatcher", make)
    registry = object()
    seam.install(registry)
    dispatcher = getattr(source, seam._PATCH.function)
    assert dispatcher is getattr(consumer, seam._PATCH.function)
    assert calls == [(_stock, registry)] and seam.is_installed()
    seam.uninstall()
    assert getattr(source, seam._PATCH.function) is _stock


def test_decode_score_kernel_is_patched_once_and_restored(monkeypatch) -> None:
    _decode_source(monkeypatch)
    score = ModuleType(seam._SCORE_SOURCE)
    stock, made = object(), []
    setattr(score, seam._SCORE_FUNCTION, stock)
    monkeypatch.setitem(sys.modules, seam._SCORE_SOURCE, score)
    monkeypatch.setattr(
        seam, "make_msa_block_score_kernel",
        lambda original, *, registry: made.append((original, registry)) or object(),
    )
    registry = object()
    seam.install(registry)
    assert getattr(score, seam._SCORE_FUNCTION) is not stock and seam.is_installed()
    seam.install(registry)
    assert made == [(stock, registry)]
    seam.uninstall()
    assert getattr(score, seam._SCORE_FUNCTION) is stock


def test_audit_mode_routes_decode_to_triton_and_restores(monkeypatch) -> None:
    _decode_source(monkeypatch)
    state = {"audit": False}

    class Backend:
        def __init__(self):
            self.use_msa = self._use_msa_decode = self._msa_owns_decode = True

    original = Backend.__init__
    backend = ModuleType(seam._BACKEND)
    setattr(backend, seam._BACKEND_CLASS, Backend)
    monkeypatch.setitem(sys.modules, seam._BACKEND, backend)
    monkeypatch.setattr(seam._audit, "enabled", lambda: state["audit"])
    seam.install()
    ordinary = Backend()
    state["audit"] = True
    audited = Backend()
    assert ordinary._use_msa_decode and ordinary._msa_owns_decode
    assert not audited._use_msa_decode and not audited._msa_owns_decode
    seam.uninstall()
    assert Backend.__init__ is original

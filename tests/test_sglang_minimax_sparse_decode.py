"""Import-order and identity checks for the graph-native M3 decode seam."""

from __future__ import annotations

import importlib.machinery
import sys
from types import ModuleType

import pytest

from cacheon.integrations import sglang_minimax_sparse_decode as seam


def _stock(*_args, **_kwargs):
    return "stock"


@pytest.fixture()
def factory(monkeypatch):
    calls = []

    def make(original, *, registry):
        def dispatcher(*args, **kwargs):
            return original(*args, **kwargs)

        calls.append((original, registry))
        return dispatcher

    monkeypatch.setattr(seam, "make_minimax_sparse_decode_dispatcher", make)
    return calls


def _source(monkeypatch):
    source = ModuleType(seam._SOURCE)
    setattr(source, seam._FUNCTION, _stock)
    monkeypatch.setitem(sys.modules, seam._SOURCE, source)
    monkeypatch.delitem(sys.modules, seam._CONSUMER, raising=False)
    return source


def test_loaded_consumer_is_patched_once_and_restored(monkeypatch, factory) -> None:
    source = _source(monkeypatch)
    consumer = ModuleType(seam._CONSUMER)
    setattr(consumer, seam._FUNCTION, _stock)
    monkeypatch.setitem(sys.modules, seam._CONSUMER, consumer)
    registry = object()

    seam.install(registry)
    dispatcher = getattr(source, seam._FUNCTION)

    assert dispatcher is getattr(consumer, seam._FUNCTION)
    assert factory == [(_stock, registry)]
    assert seam.is_installed()
    seam.install(registry)
    assert factory == [(_stock, registry)]

    seam.uninstall()
    assert getattr(source, seam._FUNCTION) is _stock
    assert getattr(consumer, seam._FUNCTION) is _stock
    assert not seam.is_installed()


def test_source_first_patches_future_by_value_import(monkeypatch, factory) -> None:
    source = _source(monkeypatch)
    seam.install()
    dispatcher = getattr(source, seam._FUNCTION)
    consumer = ModuleType(seam._CONSUMER)
    setattr(consumer, seam._FUNCTION, dispatcher)
    monkeypatch.setitem(sys.modules, seam._CONSUMER, consumer)

    assert seam.is_installed()
    seam.uninstall()
    assert getattr(source, seam._FUNCTION) is _stock
    assert getattr(consumer, seam._FUNCTION) is _stock


def test_consumer_import_window_is_accepted(monkeypatch, factory) -> None:
    source = _source(monkeypatch)
    consumer = ModuleType(seam._CONSUMER)
    spec = importlib.machinery.ModuleSpec(seam._CONSUMER, loader=None)
    spec._initializing = True
    consumer.__spec__ = spec
    monkeypatch.setitem(sys.modules, seam._CONSUMER, consumer)

    seam.install()
    setattr(consumer, seam._FUNCTION, getattr(source, seam._FUNCTION))
    spec._initializing = False

    assert seam.is_installed()


def test_foreign_consumer_binding_fails_before_mutation(monkeypatch, factory) -> None:
    source = _source(monkeypatch)
    consumer = ModuleType(seam._CONSUMER)
    foreign = lambda: None
    setattr(consumer, seam._FUNCTION, foreign)
    monkeypatch.setitem(sys.modules, seam._CONSUMER, consumer)

    with pytest.raises(RuntimeError, match="binding drifted"):
        seam.install()

    assert getattr(source, seam._FUNCTION) is _stock
    assert getattr(consumer, seam._FUNCTION) is foreign
    assert not seam.is_installed()


def test_audit_mode_keeps_msa_prefill_but_routes_decode_to_triton(
    monkeypatch, factory
) -> None:
    source = _source(monkeypatch)
    state = {"audit": False}

    class Backend:
        def __init__(self):
            self.use_msa = True
            self._use_msa_decode = True
            self._msa_owns_decode = True

    original = Backend.__init__
    backend = ModuleType(seam._BACKEND)
    setattr(backend, seam._BACKEND_CLASS, Backend)
    monkeypatch.setitem(sys.modules, seam._BACKEND, backend)
    monkeypatch.setattr(seam._audit, "enabled", lambda: state["audit"])

    seam.install()
    ordinary = Backend()
    state["audit"] = True
    audited = Backend()

    assert ordinary.use_msa and ordinary._use_msa_decode
    assert audited.use_msa and not audited._use_msa_decode
    assert not audited._msa_owns_decode
    assert seam.is_installed()
    seam.uninstall()
    assert Backend.__init__ is original
    assert getattr(source, seam._FUNCTION) is _stock

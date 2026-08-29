"""Shared by-value binding contract plus MSA-prefill factory wiring."""

import importlib.machinery
import sys
from types import ModuleType

import pytest

from cacheon.integrations import sglang_msa_prefill as seam


def _stock(*_args, **_kwargs):
    return "stock"


@pytest.fixture()
def factory(monkeypatch):
    calls = []

    def make(original, source, *, registry):
        calls.append((original, source, registry))
        return lambda *args, **kwargs: original(*args, **kwargs)

    monkeypatch.setattr(seam, "make_msa_prefill_dispatcher", make)
    return calls


def _source(monkeypatch):
    source = ModuleType(seam._PATCH.source_module)
    setattr(source, seam._PATCH.function, _stock)
    monkeypatch.setitem(sys.modules, seam._PATCH.source_module, source)
    monkeypatch.delitem(sys.modules, seam._PATCH.consumer_module, raising=False)
    return source


def _consumer(monkeypatch, binding=_stock):
    consumer = ModuleType(seam._PATCH.consumer_module)
    setattr(consumer, seam._PATCH.function, binding)
    monkeypatch.setitem(sys.modules, seam._PATCH.consumer_module, consumer)
    return consumer


def test_loaded_and_future_consumers_patch_once_and_restore(monkeypatch, factory):
    source, consumer, registry = _source(monkeypatch), _consumer(monkeypatch), object()
    seam.install(registry)
    dispatcher = getattr(source, seam._PATCH.function)
    assert dispatcher is getattr(consumer, seam._PATCH.function)
    assert factory == [(_stock, source, registry)] and seam.is_installed()
    seam.install(registry)
    assert len(factory) == 1
    seam.uninstall()
    assert getattr(source, seam._PATCH.function) is _stock
    assert getattr(consumer, seam._PATCH.function) is _stock

    source = _source(monkeypatch)
    seam.install(registry)
    consumer = _consumer(monkeypatch, getattr(source, seam._PATCH.function))
    assert seam.is_installed()
    seam.uninstall()
    assert getattr(consumer, seam._PATCH.function) is _stock


def test_consumer_import_window_is_valid(monkeypatch, factory):
    source = _source(monkeypatch)
    consumer = ModuleType(seam._PATCH.consumer_module)
    spec = importlib.machinery.ModuleSpec(seam._PATCH.consumer_module, loader=None)
    spec._initializing = True
    consumer.__spec__ = spec
    monkeypatch.setitem(sys.modules, seam._PATCH.consumer_module, consumer)
    seam.install()
    setattr(consumer, seam._PATCH.function, getattr(source, seam._PATCH.function))
    spec._initializing = False
    assert seam.is_installed()


def test_foreign_or_missing_consumer_fails_without_clobber(monkeypatch, factory):
    source = _source(monkeypatch)
    foreign = lambda: None
    consumer = _consumer(monkeypatch, foreign)
    with pytest.raises(RuntimeError, match="binding drifted"):
        seam.install()
    assert getattr(source, seam._PATCH.function) is _stock
    assert getattr(consumer, seam._PATCH.function) is foreign

    monkeypatch.delattr(consumer, seam._PATCH.function)
    with pytest.raises(RuntimeError, match="no reachable binding"):
        seam.install()


def test_drift_is_neither_installed_nor_overwritten(monkeypatch, factory):
    source, consumer = _source(monkeypatch), _consumer(monkeypatch)
    seam.install()
    dispatcher = getattr(source, seam._PATCH.function)
    foreign = lambda: None
    setattr(consumer, seam._PATCH.function, foreign)
    assert not seam.is_installed()
    with pytest.raises(RuntimeError, match="binding drifted"):
        seam.uninstall()
    assert getattr(source, seam._PATCH.function) is dispatcher
    assert getattr(consumer, seam._PATCH.function) is foreign

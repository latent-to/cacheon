"""Resident swaps release candidate memory and acknowledge recapture outcomes."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from cacheon.integrations import sglang_resident_swap as swap


class _Layer:
    def __init__(self):
        self._cacheon_moe_prepared_by_impl = {"a": object(), "b": object()}


def _runtime(monkeypatch, tmp_path, *, recapture_error=None):
    layer = _Layer()
    events = []

    class Runner:
        is_draft_worker = False
        tp_rank = 0

        def __init__(self):
            self.model = SimpleNamespace(modules=lambda: (layer,))
            self.decode_cuda_graph_runner = object()

        def init_decode_cuda_graph(self):
            events.append("recapture")
            assert not hasattr(layer, "_cacheon_moe_prepared_by_impl")
            assert self.decode_cuda_graph_runner is None
            if recapture_error is not None:
                raise recapture_error

    runner = Runner()

    class Scheduler:
        def __init__(self):
            self.tp_worker = SimpleNamespace(model_runner=runner)

        def flush_cache(self):
            return True

    model_module = ModuleType(swap._MODULE)
    model_module.ModelRunner = Runner
    scheduler_module = ModuleType(swap._SCHED_MODULE)
    scheduler_module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, swap._MODULE, model_module)
    monkeypatch.setitem(sys.modules, swap._SCHED_MODULE, scheduler_module)
    monkeypatch.setenv("CACHEON_RESIDENT_SWAP", str(tmp_path))
    monkeypatch.setattr(swap, "_applied_generation", -1)
    monkeypatch.setattr(swap.gc, "collect", lambda: events.append("gc"))

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    import cacheon.seam as seam
    import cacheon.receipts as receipts

    monkeypatch.setattr(receipts, "_ROOT", "")
    monkeypatch.setattr(receipts, "_SCOPE", "")
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.setattr(
        seam,
        "swap_resident_bundle",
        lambda bundle: {"bundle": bundle or "", "slots": []},
    )
    (tmp_path / "command.json").write_text(
        json.dumps({"bundle": None, "generation": 1})
    )
    swap.install()
    return Scheduler(), layer, events


def test_swap_evicts_prepared_weights_before_success_ack(monkeypatch, tmp_path):
    scheduler, layer, events = _runtime(monkeypatch, tmp_path)
    assert scheduler.flush_cache() is True
    ack = json.loads((tmp_path / "ack.rank0.json").read_text())
    assert not hasattr(layer, "_cacheon_moe_prepared_by_impl")
    assert events == ["sync", "gc", "empty", "recapture"]
    assert ack["ok"] is True and ack["evicted_prepared_entries"] == 2


def test_recapture_error_is_returned_immediately_in_ack(monkeypatch, tmp_path):
    scheduler, _layer, _events = _runtime(
        monkeypatch, tmp_path, recapture_error=RuntimeError("CUDA out of memory")
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        scheduler.flush_cache()
    ack = json.loads((tmp_path / "ack.rank0.json").read_text())
    assert ack["ok"] is False
    assert ack["error"] == "recapture failed: CUDA out of memory"

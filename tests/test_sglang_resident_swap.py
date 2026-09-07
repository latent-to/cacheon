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


def _runtime(monkeypatch, tmp_path, *, rank=0, recapture_error=None, recapture_hook=None):
    layer = _Layer()
    events = []

    class Runner:
        is_draft_worker = False

        def __init__(self):
            self.model = SimpleNamespace(modules=lambda: (layer,))
            self.decode_cuda_graph_runner = object()

        def init_decode_cuda_graph(self):
            events.append("recapture")
            assert not hasattr(layer, "_cacheon_moe_prepared_by_impl")
            assert self.decode_cuda_graph_runner is None
            if recapture_hook is not None:
                recapture_hook()
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
    parallel = ModuleType("sglang.srt.distributed.parallel_state")
    parallel.get_tensor_model_parallel_rank = lambda: rank
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
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


@pytest.mark.parametrize("rank", range(4))
def test_swap_evicts_prepared_weights_before_success_ack(monkeypatch, tmp_path, rank):
    scheduler, layer, events = _runtime(monkeypatch, tmp_path, rank=rank)
    assert scheduler.flush_cache() is True
    ack = json.loads((tmp_path / f"ack.rank{rank}.json").read_text())
    assert not (tmp_path / "ack.rankunknown.json").exists()
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


def test_swap_without_a_pool_keeps_the_purge_and_says_so(monkeypatch, tmp_path):
    scheduler, _layer, events = _runtime(monkeypatch, tmp_path)
    assert scheduler.flush_cache() is True
    ack = json.loads((tmp_path / "ack.rank0.json").read_text())
    assert ack["graph_pool_carried"] is False
    assert "empty" in events


def test_swap_carries_graph_pool_and_skips_empty_cache(monkeypatch, tmp_path):
    pool = object()

    class Backend:
        def __init__(self):
            self._pool = None

        def capture_session(self, stream):
            return (self._pool, stream)

    backend_module = ModuleType(swap._BACKEND_HOOKS[0][0])
    backend_module.FullCudaGraphBackend = Backend
    monkeypatch.setitem(sys.modules, swap._BACKEND_HOOKS[0][0], backend_module)
    monkeypatch.setattr(swap, "_carried_graph_pool", None)
    rebuilt = Backend()

    scheduler, _layer, events = _runtime(
        monkeypatch,
        tmp_path,
        recapture_hook=lambda: rebuilt.capture_session("stream"),
    )
    runner = scheduler.tp_worker.model_runner
    runner.decode_cuda_graph_runner = SimpleNamespace(
        backend=SimpleNamespace(_pool=pool)
    )
    assert scheduler.flush_cache() is True

    ack = json.loads((tmp_path / "ack.rank0.json").read_text())
    assert ack["graph_pool_carried"] is True
    assert "empty" not in events, "purge between generations forfeits the pool"

    assert rebuilt._pool is pool
    swap.uninstall()
    assert Backend().capture_session("stream") == (None, "stream")

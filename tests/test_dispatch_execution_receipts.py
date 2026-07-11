"""Post-success/fallback accounting across every serving dispatcher family.

These are control-flow receipts, not a hostile-code security boundary.  The tests
pin that every selected implementation reports completion only after its output path
succeeds, and reports a fallback when a selected path raises and stock is served.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import optima.dispatch as dispatch  # noqa: E402
from optima.registry import (  # noqa: E402
    Eligibility,
    KernelImpl,
    KernelRegistry,
    eligibility_from_metadata,
)


@pytest.fixture()
def events(monkeypatch):
    completed: list[str] = []
    fallback: list[tuple[str, str]] = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    monkeypatch.setattr(
        dispatch._receipts, "fallback",
        lambda slot, exc: fallback.append((slot, type(exc).__name__)),
    )
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: False)
    return completed, fallback


def _registry(slot, entry, *, prepare=None, graph_safe=False):
    reg = KernelRegistry()
    reg.register(KernelImpl(
        slot=slot, bundle_id="test", entry=entry, prepare=prepare,
        eligibility=Eligibility(
            dtypes=frozenset({"float32"}), graph_safe=graph_safe),
    ))
    reg.enable()
    return reg


def test_op_dispatchers_receipt_success_and_fallback(events):
    completed, fallback = events
    baseline = object()

    silu = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: baseline,
        registry=_registry("activation.silu_and_mul",
                           lambda x, out: out.copy_(x[..., :x.shape[-1] // 2])),
    )
    assert silu(object(), torch.randn(2, 8)) is not baseline

    class Weight:
        data = torch.ones(8)

    rms_self = SimpleNamespace(variance_epsilon=1e-6, weight=Weight())
    rms = dispatch.make_rmsnorm_dispatcher(
        lambda *_: baseline,
        registry=_registry("norm.rmsnorm", lambda x, weight, out, eps: out.copy_(x)),
    )
    assert rms(rms_self, torch.randn(2, 8)) is not baseline
    assert completed == ["activation.silu_and_mul", "norm.rmsnorm"]

    def boom(*_):
        raise RuntimeError("boom")

    silu_bad = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: baseline, registry=_registry("activation.silu_and_mul", boom))
    rms_bad = dispatch.make_rmsnorm_dispatcher(
        lambda *_: baseline, registry=_registry("norm.rmsnorm", boom))
    assert silu_bad(object(), torch.randn(2, 8)) is baseline
    assert rms_bad(rms_self, torch.randn(2, 8)) is baseline
    assert fallback == [
        ("activation.silu_and_mul", "RuntimeError"),
        ("norm.rmsnorm", "RuntimeError"),
    ]


def _attention_inputs():
    class Pool:
        def __init__(self):
            self.k = torch.zeros(2, 1, 2)
            self.v = torch.zeros(2, 1, 2)

        def set_kv_buffer(self, _layer, locations, k, v):
            self.k[locations] = k
            self.v[locations] = v

        def get_key_buffer(self, _layer_id):
            return self.k

        def get_value_buffer(self, _layer_id):
            return self.v

    pool = Pool()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        token_to_kv_pool=pool,
        out_cache_loc=torch.tensor([1]),
        seq_lens=torch.tensor([2]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[0, 1]])),
        req_pool_indices=torch.tensor([0]),
    )
    layer = SimpleNamespace(
        qk_head_dim=2, v_head_dim=2, tp_q_head_num=1, tp_k_head_num=1,
        scaling=1.0, layer_id=0, is_cross_attention=False, sliding_window_size=-1,
    )
    return layer, batch


def test_attention_dispatcher_receipts(events, monkeypatch):
    completed, fallback = events
    monkeypatch.setenv("OPTIMA_ATTENTION_SEAM", "1")
    baseline = object()
    layer, batch = _attention_inputs()
    args = (layer, torch.ones(1, 2), torch.ones(1, 1, 2),
            torch.ones(1, 1, 2), batch)

    def entry(q, k, v, seq_lens, scale, out):
        out.copy_(q)

    good = dispatch.make_attention_dispatcher(
        lambda *_a, **_k: baseline,
        registry=_registry("attention.decode", entry))
    assert good(*args) is not baseline
    assert completed == ["attention.decode"]

    bad = dispatch.make_attention_dispatcher(
        lambda *_a, **_k: baseline,
        registry=_registry("attention.decode", lambda *_: (_ for _ in ()).throw(
            RuntimeError("boom"))))
    assert bad(*args) is baseline
    assert fallback == [("attention.decode", "RuntimeError")]


def _moe_call(entry):
    slot = "moe.fused_experts"
    x = torch.randn(2, 4)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(data=torch.randn(2, 4, 4)),
        w2_weight=SimpleNamespace(data=torch.randn(2, 4, 2)),
        moe_tp_size=1, moe_ep_size=1, reduce_results=False,
    )
    topk = SimpleNamespace(
        topk_ids=torch.zeros(2, 1, dtype=torch.long),
        topk_weights=torch.ones(2, 1),
    )
    reg = _registry(slot, entry, prepare=lambda *_: object())
    return dispatch.make_moe_dispatcher(lambda *_: "stock", registry=reg), layer, x, topk


def test_moe_dispatcher_receipts(events, monkeypatch):
    completed, fallback = events
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")

    def good_entry(x, ids, weights, prepared, out):
        out.copy_(x)

    good, layer, x, topk = _moe_call(good_entry)
    assert torch.is_tensor(good(layer, x, topk))
    assert completed == ["moe.fused_experts"]

    def boom(*_):
        raise RuntimeError("boom")

    bad, layer, x, topk = _moe_call(boom)
    assert bad(layer, x, topk) == "stock"
    assert fallback == [("moe.fused_experts", "RuntimeError")]


def test_allreduce_dispatcher_receipts(events, monkeypatch):
    completed, fallback = events
    monkeypatch.setenv("OPTIMA_COLLECTIVE_SEAM", "1")
    coordinator = SimpleNamespace(world_size=2, device_group=object())
    x = torch.randn(2, 4)

    def good_entry(inp, out, group):
        out.copy_(inp)

    good = dispatch.make_allreduce_dispatcher(
        lambda *_a, **_k: "stock",
        registry=_registry("collective.all_reduce", good_entry))
    assert torch.is_tensor(good(coordinator, x))
    assert completed == ["collective.all_reduce"]

    def boom(*_):
        raise RuntimeError("boom")

    bad = dispatch.make_allreduce_dispatcher(
        lambda *_a, **_k: "stock",
        registry=_registry("collective.all_reduce", boom))
    assert bad(coordinator, x) == "stock"
    assert fallback == [("collective.all_reduce", "RuntimeError")]


def _fusion_baseline(x, residual, *_args, **_kwargs):
    return "stock", x + residual


def test_shallow_and_deep_fusion_dispatcher_receipts(events, monkeypatch):
    completed, fallback = events
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    monkeypatch.setattr(
        dispatch, "_arfusion_group", lambda _use_attn: SimpleNamespace(size=lambda: 2))
    x, residual, weight = torch.randn(2, 4), torch.randn(2, 4), torch.ones(4)

    def shallow_entry(x, residual, weight, eps, out_norm, out_residual, group):
        out_norm.copy_(x)
        out_residual.copy_(residual)

    shallow = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.ar_residual_rmsnorm", shallow_entry))
    assert torch.is_tensor(shallow(x, residual, weight)[0])
    assert completed == ["collective.ar_residual_rmsnorm"]

    shallow_bad = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.ar_residual_rmsnorm",
                           lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))))
    assert shallow_bad(x, residual, weight)[0] == "stock"
    assert fallback == [("collective.ar_residual_rmsnorm", "RuntimeError")]

    exp = {"T": 2}
    monkeypatch.setattr(dispatch._moe_export, "has_pends", lambda: True)
    monkeypatch.setattr(dispatch._moe_export, "consume", lambda _x: exp)
    monkeypatch.setattr(
        dispatch._moe_export, "export_views",
        lambda _exp, _device: (torch.randn(2, 4), torch.zeros(2, dtype=torch.long),
                               torch.ones(2)),
    )
    monkeypatch.setattr(dispatch._moe_export, "trusted_finalize", lambda _exp, inp: inp)
    monkeypatch.setattr(dispatch._moe_export, "orphaned", lambda _exp: None)

    def deep_entry(gemm, row_map, scales, residual, weight, eps,
                   out_norm, out_residual, group):
        out_norm.copy_(residual)
        out_residual.copy_(residual)

    deep = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.moe_finalize_ar_rmsnorm", deep_entry))
    assert torch.is_tensor(deep(x, residual, weight)[0])
    assert completed[-1] == "collective.moe_finalize_ar_rmsnorm"

    deep_bad = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.moe_finalize_ar_rmsnorm",
                           lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))))
    assert deep_bad(x, residual, weight)[0] == "stock"
    assert fallback[-1] == ("collective.moe_finalize_ar_rmsnorm", "RuntimeError")


class _CudaLikeQ:
    """CPU-backed q with the minimal CUDA-shaped surface used by the MSA wrapper."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.is_cuda = True

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device

    def dim(self):
        return self.tensor.dim()

    def __getitem__(self, item):
        return self.tensor[item]


class _FakeTopKKernel:
    def __getitem__(self, _grid):
        def launch(_score, topk_idx, *_args, **_kwargs):
            topk_idx.zero_()
        return launch


def _msa_args():
    return (
        _CudaLikeQ(torch.ones(1, 1, 2)),
        torch.ones(1, 1, 2), torch.ones(1, 1, 2), None,
        torch.tensor([[0]]), torch.tensor([0]), torch.tensor([0, 1]),
        torch.tensor([1]), torch.tensor([0]), 1, 1, 1, 1, 1,
    )


def _msa_batched_args():
    # Two requests share a [heads,total_q,max_blocks] score slab.  Request 0 has
    # fewer blocks than the batch max, so its 2x2 logical output has row stride 3.
    return (
        _CudaLikeQ(torch.ones(3, 2, 2)),
        torch.ones(5, 1, 2), torch.ones(5, 1, 2), None,
        torch.tensor([[0, 1, 0], [2, 3, 4]]), torch.tensor([0, 1]),
        torch.tensor([0, 2, 3]), torch.tensor([2, 3]), torch.tensor([0, 2]),
        2, 3, 1, 1, 1,
    )


def test_msa_prefill_dispatcher_receipts(events, monkeypatch):
    completed, fallback = events
    monkeypatch.setenv("OPTIMA_MSA_PREFILL_SEAM", "1")
    monkeypatch.setattr(dispatch, "_arch_tag", lambda *_args: "sm103")
    fake_triton = ModuleType("triton")
    fake_triton.set_allocator = lambda _allocator: None
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    module = SimpleNamespace(robust_allocator=object(), _topk_index_kernel=_FakeTopKKernel())
    stock = ("stock", torch.zeros(1, 1, 1, dtype=torch.int32))

    def good_entry(q, k, prefix, scale, block_size, out):
        out.fill_(1.0)

    good = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: stock, module,
        registry=_registry("attention.msa_prefill_block_score", good_entry))
    result = good(*_msa_args(), disable_index_value=True,
                  cu_seqblocks_q=torch.tensor([0, 1]),
                  max_seqblock_q=1, all_seqblock_q=1)
    assert result[0] is None
    assert completed == ["attention.msa_prefill_block_score"]

    def boom(*_):
        raise RuntimeError("boom")

    bad = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: stock, module,
        registry=_registry("attention.msa_prefill_block_score", boom))
    assert bad(*_msa_args(), disable_index_value=True,
               cu_seqblocks_q=torch.tensor([0, 1]),
               max_seqblock_q=1, all_seqblock_q=1) is stock
    assert fallback == [("attention.msa_prefill_block_score", "RuntimeError")]


def test_msa_prefill_dispatcher_uses_typed_strided_score_view(events, monkeypatch):
    completed, _fallback = events
    monkeypatch.setenv("OPTIMA_MSA_PREFILL_SEAM", "1")
    monkeypatch.setattr(dispatch, "_arch_tag", lambda *_args: "sm103")
    fake_triton = ModuleType("triton")
    fake_triton.set_allocator = lambda _allocator: None
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    module = SimpleNamespace(robust_allocator=object(), _topk_index_kernel=_FakeTopKKernel())
    stock = ("stock", torch.zeros(2, 3, 1, dtype=torch.int32))
    observed = []

    def entry(q, k, prefix, scale, block_size, out):
        observed.append((out.dtype, out.is_contiguous(), tuple(out.shape), out.stride()))
        out.fill_(1.0)

    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: stock,
        module,
        registry=_registry("attention.msa_prefill_block_score", entry),
    )
    result = wrapped(
        *_msa_batched_args(),
        disable_index_value=True,
        cu_seqblocks_q=torch.tensor([0, 2, 3]),
        max_seqblock_q=2,
        all_seqblock_q=3,
    )
    assert result[0] is None
    assert completed == ["attention.msa_prefill_block_score"]
    assert observed
    dtype, contiguous, shape, stride = observed[0]
    assert dtype == torch.float32
    assert shape == (2, 2)
    assert not contiguous
    assert stride == (3, 1)


class _RecordingRegistry(KernelRegistry):
    def __init__(self):
        super().__init__()
        self.decisions = []

    def select(self, slot, descriptor, **kwargs):
        self.decisions.append((descriptor, kwargs.get("write_fired_receipt", True)))
        return super().select(slot, descriptor, **kwargs)


def _msa_variant_registry(entries):
    registry = _RecordingRegistry()
    for q_len, entry in entries.items():
        eligibility = eligibility_from_metadata(
            {
                "graph_safe": False,
                "capabilities": {
                    "dtype": "float32",
                    "architecture": "sm103",
                    "head_dim": 2,
                    "block_size": 1,
                    "q_len": q_len,
                    "phase": "prefill",
                    "layout": "row_major",
                    "graph_mode": "eager",
                    "quant": "dense",
                },
            },
            ("float32",),
            ("sm103",),
        )
        registry.register(KernelImpl(
            slot="attention.msa_prefill_block_score",
            bundle_id="test",
            variant=f"q{q_len}",
            entry=entry,
            eligibility=eligibility,
        ))
    registry.enable()
    return registry


def _install_fake_msa_runtime(monkeypatch):
    monkeypatch.setenv("OPTIMA_MSA_PREFILL_SEAM", "1")
    monkeypatch.setattr(dispatch, "_arch_tag", lambda *_args: "sm103")
    monkeypatch.setattr(dispatch, "_runtime_parallel_sizes", lambda: (4, 8))
    fake_triton = ModuleType("triton")
    fake_triton.set_allocator = lambda _allocator: None
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    return SimpleNamespace(
        robust_allocator=object(), _topk_index_kernel=_FakeTopKKernel()
    )


def test_msa_prefill_preflights_each_request_head_and_routes_variants(events, monkeypatch):
    completed, fallback = events
    module = _install_fake_msa_runtime(monkeypatch)
    calls = {1: 0, 2: 0}

    def entry_for(q_len):
        def entry(q, _k, _prefix, _scale, _block_size, out):
            assert q.shape[0] == q_len
            calls[q_len] += 1
            out.fill_(float(q_len))
        return entry

    registry = _msa_variant_registry({1: entry_for(1), 2: entry_for(2)})
    stock_calls = 0

    def stock(*_args, **_kwargs):
        nonlocal stock_calls
        stock_calls += 1
        return "stock"

    wrapped = dispatch.make_msa_prefill_dispatcher(stock, module, registry=registry)
    result = wrapped(
        *_msa_batched_args(),
        disable_index_value=True,
        cu_seqblocks_q=torch.tensor([0, 2, 3]),
        max_seqblock_q=2,
        all_seqblock_q=3,
    )

    assert result[0] is None
    assert stock_calls == 0 and fallback == []
    assert calls == {1: 2, 2: 2}  # two actual per-head calls per request
    assert completed == ["attention.msa_prefill_block_score"]
    preflight = [d for d, fired in registry.decisions if not fired]
    assert len(preflight) == 4
    assert {d["q_len"] for d in preflight} == {1, 2}
    assert {d["kv_len"] for d in preflight} == {2, 3}
    for descriptor in preflight:
        assert descriptor.as_dict().items() >= {
            "dtype": "float32",
            "architecture": "sm103",
            "head_dim": 2,
            "block_size": 1,
            "phase": "prefill",
            "layout": "row_major",
            "graph_mode": "eager",
            "quant": "dense",
            "tp_size": 4,
            "world_size": 8,
            "num_q_heads": 1,
            "num_kv_heads": 1,
        }.items()


def test_msa_prefill_mixed_batch_off_domain_is_wholly_stock(events, monkeypatch):
    completed, fallback = events
    module = _install_fake_msa_runtime(monkeypatch)
    candidate_calls = 0

    def q2_only(*_args):
        nonlocal candidate_calls
        candidate_calls += 1

    registry = _msa_variant_registry({2: q2_only})
    stock_result = object()
    stock_calls = 0

    def stock(*_args, **_kwargs):
        nonlocal stock_calls
        stock_calls += 1
        return stock_result

    wrapped = dispatch.make_msa_prefill_dispatcher(stock, module, registry=registry)
    result = wrapped(
        *_msa_batched_args(),
        disable_index_value=True,
        cu_seqblocks_q=torch.tensor([0, 2, 3]),
        max_seqblock_q=2,
        all_seqblock_q=3,
    )

    assert result is stock_result
    assert stock_calls == 1
    assert candidate_calls == 0
    assert completed == [] and fallback == []

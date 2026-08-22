"""Live V2 MSA prefill dispatch: one paged batch call, no V1 score tail."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import cacheon.dispatch as dispatch  # noqa: E402
from cacheon.registry import (  # noqa: E402
    KernelImpl,
    KernelRegistry,
    eligibility_from_metadata,
)

SLOT = "attention.msa_prefill_block_score"


class _CudaLikeQ:
    """CPU storage with the CUDA-shaped routing surface used by the wrapper."""

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


def _args(*, topk=4):
    return (
        _CudaLikeQ(torch.ones(3, 2, 2)),
        torch.arange(10, dtype=torch.float32).view(5, 1, 2),
        torch.ones(5, 1, 2),
        None,
        torch.tensor([[3, 1, 4], [2, 0, 1]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([2, 3], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        2,
        3,
        1,
        1,
        topk,
        0,
        1,
    )


def _invoke(wrapped, args=None, *, device="cpu", **kwargs):
    return wrapped(
        *(args or _args()),
        disable_index_value=True,
        cu_seqblocks_q=torch.tensor([0, 2, 3], device=device),
        max_seqblock_q=2,
        all_seqblock_q=3,
        **kwargs,
    )


def _install_runtime(monkeypatch):
    monkeypatch.setenv("CACHEON_MSA_PREFILL_SEAM", "1")
    monkeypatch.setattr(dispatch, "_arch_tag", lambda *_: "sm103")
    monkeypatch.setattr(dispatch, "_runtime_parallel_sizes", lambda: (4, 8))
    monkeypatch.setattr(dispatch, "_dynamo_compiling", lambda: False)
    monkeypatch.setattr(dispatch, "_in_cuda_graph", lambda: False)
    return SimpleNamespace()


class _RecordingRegistry(KernelRegistry):
    def __init__(self):
        super().__init__()
        self.decisions = []

    def select(self, slot, descriptor, **kwargs):
        self.decisions.append(descriptor)
        return super().select(slot, descriptor, **kwargs)


def _registry(entry, *, q_len=2):
    registry = _RecordingRegistry()
    eligibility = eligibility_from_metadata(
        {
            "graph_safe": False,
            "capabilities": {
                "dtype": "float32",
                "architecture": "sm103",
                "head_dim": 2,
                "block_size": 1,
                "q_len": q_len,
                "kv_len": 3,
                "batch_size": 2,
                "num_tokens": 3,
                "num_q_heads": 2,
                "num_kv_heads": 1,
                "top_k": 4,
                "q_block_size": 1,
                "init_blocks": 0,
                "local_blocks": 1,
                "phase": "prefill",
                "layout": "paged",
                "graph_mode": "eager",
                "quant": "dense",
            },
        },
        ("float32",),
        ("sm103",),
    )
    registry.register(
        KernelImpl(
            slot=SLOT,
            bundle_id="test",
            entry=entry,
            eligibility=eligibility,
        )
    )
    registry.enable()
    return registry


def _fill_selection(out):
    out.fill_(-1)
    out[..., 0] = 0


def test_msa_v2_routes_one_full_paged_batch(monkeypatch):
    module = _install_runtime(monkeypatch)
    calls = []
    completed = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    args = _args()

    def entry(*call):
        calls.append(call)
        _fill_selection(call[-1])

    stock_calls = 0

    def stock(*_args, **_kwargs):
        nonlocal stock_calls
        stock_calls += 1
        return "stock"

    registry = _registry(entry)
    result = _invoke(
        dispatch.make_msa_prefill_dispatcher(stock, module, registry=registry),
        args,
    )

    assert stock_calls == 0
    assert len(calls) == 1
    assert calls[0][0] is args[0]
    assert calls[0][1] is args[1]
    assert calls[0][2] is args[4]
    assert calls[0][-1] is result[1]
    assert result[0] is None
    assert result[1].shape == (2, 3, 4)
    assert result[1].dtype == torch.int32 and result[1].is_contiguous()
    assert completed == [SLOT]
    assert len(registry.decisions) == 1
    assert registry.decisions[0].as_dict().items() >= {
        "batch_size": 2,
        "num_tokens": 3,
        "q_len": 2,
        "kv_len": 3,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "top_k": 4,
        "q_block_size": 1,
        "init_blocks": 0,
        "local_blocks": 1,
        "layout": "paged",
        "tp_size": 4,
        "world_size": 8,
    }.items()


def test_msa_v2_off_domain_is_wholly_stock(monkeypatch):
    module = _install_runtime(monkeypatch)
    candidate_calls = 0

    def candidate(*_args):
        nonlocal candidate_calls
        candidate_calls += 1

    stock_result = object()
    registry = _registry(candidate, q_len=1)
    assert _invoke(
        dispatch.make_msa_prefill_dispatcher(
            lambda *_a, **_k: stock_result, module, registry=registry
        )
    ) is stock_result
    assert candidate_calls == 0
    assert registry.decisions and registry.decisions[0]["q_len"] == 2


def test_msa_v2_selected_failure_receipts_after_stock_recovery(monkeypatch):
    module = _install_runtime(monkeypatch)
    fallbacks = []
    monkeypatch.setattr(
        dispatch._receipts,
        "fallback",
        lambda slot, exc: fallbacks.append((slot, type(exc).__name__)),
    )

    def boom(*_args):
        raise RuntimeError("candidate path failed")

    stock_result = object()
    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: stock_result, module, registry=_registry(boom)
    )
    assert _invoke(wrapped) is stock_result
    assert fallbacks == [(SLOT, "RuntimeError")]

    fallbacks.clear()
    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("stock failed")),
        module,
        registry=_registry(boom),
    )
    with pytest.raises(ValueError, match="stock failed"):
        _invoke(wrapped)
    assert fallbacks == []


def test_msa_v2_audits_consumed_selection_without_stock_topk_tail(monkeypatch):
    module = _install_runtime(monkeypatch)
    observed = []
    expected = torch.full((2, 3, 4), -1, dtype=torch.int32)
    expected[..., 0] = 0
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: True)
    monkeypatch.setattr(
        dispatch._audit,
        "record",
        lambda slot, actual, reference: observed.append((slot, actual, reference)),
    )

    def entry(*call):
        _fill_selection(call[-1])

    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: (None, expected), module, registry=_registry(entry)
    )
    result = _invoke(wrapped)
    assert len(observed) == 1
    assert observed[0][0] == SLOT
    assert observed[0][1][0] is result[1]
    assert observed[0][2][0] is expected
    assert not hasattr(module, "_topk_index_kernel")


@pytest.mark.parametrize(
    "override",
    [
        {"disable_index_value": False},
        {"score_type": "lse"},
    ],
)
def test_msa_v2_unsupported_live_domain_stays_stock(monkeypatch, override):
    module = _install_runtime(monkeypatch)
    stock_result = object()
    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: stock_result,
        module,
        registry=KernelRegistry(),
    )
    kwargs = {
        "disable_index_value": True,
        "score_type": "max",
        "cu_seqblocks_q": torch.tensor([0, 2, 3]),
        "max_seqblock_q": 2,
        "all_seqblock_q": 3,
        **override,
    }
    assert wrapped(*_args(), **kwargs) is stock_result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_msa_v2_real_cuda_routes_once(monkeypatch, tmp_path):
    module = _install_runtime(monkeypatch)
    args = list(_args())
    args[0] = torch.ones(3, 2, 2, device="cuda")
    args = [value.to("cuda") if torch.is_tensor(value) else value for value in args]
    calls = 0

    def entry(*call):
        nonlocal calls
        calls += 1
        _fill_selection(call[-1])

    receipt_dir = tmp_path / "receipts"
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(receipt_dir))
    monkeypatch.setattr(dispatch._receipts, "_ONCE", set())
    result = _invoke(
        dispatch.make_msa_prefill_dispatcher(
            lambda *_a, **_k: "stock", module, registry=_registry(entry)
        ),
        args,
        device="cuda",
    )
    torch.cuda.synchronize()
    assert calls == 1 and result[1].is_cuda
    assert len(dispatch._receipts.collect(receipt_dir, "completed")) == 1

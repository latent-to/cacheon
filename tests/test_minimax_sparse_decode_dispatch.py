"""Live graph-native MiniMax sparse-decode dispatch and execution evidence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import cacheon.minimax_sparse_decode_dispatch as sparse_dispatch  # noqa: E402
from cacheon.registry import (  # noqa: E402
    KernelImpl,
    KernelRegistry,
    eligibility_from_metadata,
)


def _registry(entry, *, graph_safe=True, capabilities=None) -> KernelRegistry:
    registry = KernelRegistry()
    registry.register(
        KernelImpl(
            slot="attention.decode",
            bundle_id="candidate",
            entry=entry,
            eligibility=eligibility_from_metadata(
                {"graph_safe": graph_safe, "capabilities": capabilities or {}},
                ("float32",),
            ),
        )
    )
    registry.enable()
    return registry


def _args():
    batch, heads, dim, context, block = 2, 4, 8, 16, 4
    q = torch.randn(batch, heads, dim)
    k = torch.randn(batch * context, 1, dim)
    v = torch.randn_like(k)
    req = torch.arange(batch * context, dtype=torch.int32).reshape(batch, context)
    seq = torch.tensor([context, context - 3], dtype=torch.int32)
    rows = torch.arange(batch, dtype=torch.int32)
    selected = torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int32)
    return (q, None, k, v, req, seq, rows, block, selected)


def _candidate(calls):
    def entry(
        q, _k, _v, _req, _seq, _rows, _selected, out, _scale, _block
    ):
        calls.append("candidate")
        out.copy_(q)

    return entry


def test_graph_capture_routes_candidate_and_mints_completion(monkeypatch) -> None:
    calls: list[str] = []
    completed: list[str] = []
    monkeypatch.setattr(sparse_dispatch, "_in_cuda_graph", lambda: True)
    monkeypatch.setattr(sparse_dispatch._receipts, "completed", completed.append)
    args = _args()
    wrapped = sparse_dispatch.make_minimax_sparse_decode_dispatcher(
        lambda *_args, **_kwargs: pytest.fail("graph capture used stock"),
        registry=_registry(_candidate(calls)),
    )

    output = wrapped(*args)

    assert torch.equal(output, args[0])
    assert calls == ["candidate"]
    assert completed == ["attention.decode"]


def test_eager_recapture_warmup_compiles_but_cannot_prove_execution(monkeypatch) -> None:
    calls: list[str] = []
    completed: list[str] = []
    monkeypatch.setattr(sparse_dispatch, "_in_cuda_graph", lambda: False)
    monkeypatch.setattr(sparse_dispatch._receipts, "completed", completed.append)
    args = _args()
    wrapped = sparse_dispatch.make_minimax_sparse_decode_dispatcher(
        lambda *_args, **_kwargs: pytest.fail("warmup used stock"),
        registry=_registry(_candidate(calls)),
    )

    output = wrapped(*args)

    assert torch.equal(output, args[0])
    assert calls == ["candidate"]
    assert completed == []


def test_eager_audit_runs_stock_first_and_mints_completion(monkeypatch) -> None:
    events: list[str] = []
    completed: list[str] = []
    compared = []
    args = _args()

    def stock(*_args, **_kwargs):
        events.append("stock")
        return args[0].clone()

    def candidate(*entry_args):
        events.append("candidate")
        _candidate([])(*entry_args)

    monkeypatch.setattr(sparse_dispatch, "_in_cuda_graph", lambda: False)
    monkeypatch.setattr(sparse_dispatch._audit, "sampled", lambda: True)
    monkeypatch.setattr(sparse_dispatch._audit, "enabled", lambda: True)
    monkeypatch.setattr(
        sparse_dispatch._audit, "record", lambda *record: compared.append(record)
    )
    monkeypatch.setattr(sparse_dispatch._receipts, "completed", completed.append)

    output = sparse_dispatch.make_minimax_sparse_decode_dispatcher(
        stock, registry=_registry(candidate)
    )(*args)

    assert events == ["stock", "candidate"]
    assert torch.equal(output, args[0])
    assert compared[0][0] == "attention.decode"
    assert completed == ["attention.decode"]


def test_non_graph_safe_bundle_warms_but_graph_capture_stays_stock(monkeypatch) -> None:
    calls: list[str] = []
    stock = torch.full((2, 4, 8), 7.0)
    wrapped = sparse_dispatch.make_minimax_sparse_decode_dispatcher(
        lambda *_args, **_kwargs: stock,
        registry=_registry(_candidate(calls), graph_safe=False),
    )
    monkeypatch.setattr(sparse_dispatch, "_in_cuda_graph", lambda: False)
    wrapped(*_args())
    monkeypatch.setattr(sparse_dispatch, "_in_cuda_graph", lambda: True)

    output = wrapped(*_args())

    assert output is stock
    assert calls == ["candidate"]


def test_selected_candidate_failure_never_falls_back_to_stock(monkeypatch) -> None:
    monkeypatch.setattr(sparse_dispatch, "_in_cuda_graph", lambda: True)

    def broken(*_args):
        raise RuntimeError("candidate failed")

    wrapped = sparse_dispatch.make_minimax_sparse_decode_dispatcher(
        lambda *_args, **_kwargs: pytest.fail("selected failure used stock"),
        registry=_registry(broken),
    )
    with pytest.raises(RuntimeError, match="candidate failed"):
        wrapped(*_args())


def test_live_descriptor_routes_a_complete_decode_domain() -> None:
    calls: list[str] = []
    capabilities = {
        "batch_size": 2, "block_size": 4, "head_dim": 8, "kv_len": 16,
        "layout": "paged_nhd", "model": "MiniMax-M3", "num_kv_heads": 1,
        "num_q_heads": 4, "num_tokens": 2, "page_size": 4,
        "phase": "decode", "q_len": 1, "quant": "dense", "top_k": 2,
    }
    wrapped = sparse_dispatch.make_minimax_sparse_decode_dispatcher(
        lambda *_args, **_kwargs: pytest.fail("complete live domain used stock"),
        registry=_registry(_candidate(calls), capabilities=capabilities),
    )
    args = _args()

    output = wrapped(*args)

    assert torch.equal(output, args[0])
    assert calls == ["candidate"]

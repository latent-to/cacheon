"""Dispatch MiniMax-M3 sparse attend after stock scoring and top-k selection."""

from __future__ import annotations

from collections.abc import Callable

import torch

from cacheon import audit as _audit
from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor
from cacheon.dispatch import (
    _allocate_live_outputs,
    _arch_tag,
    _dtype_name,
    _in_cuda_graph,
    _runtime_parallel_sizes,
    _validate_live_outputs,
)
from cacheon.registry import REGISTRY, KernelRegistry


def _descriptor(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    block_size: int,
    in_graph: bool,
) -> CallDescriptor:
    tp_size, world_size = _runtime_parallel_sizes()
    fields: dict[str, object] = {
        "architecture": _arch_tag(q.device.index or 0) if q.is_cuda else None,
        "batch_size": int(q.shape[0]),
        "block_size": int(block_size),
        "dtype": _dtype_name(q.dtype),
        "graph_mode": "cuda_graph" if in_graph else "eager",
        "head_dim": int(q.shape[-1]),
        "last_dim": int(q.shape[-1]),
        "layout": "paged_nhd",
        "kv_len": int(req_to_token.shape[1]),
        "model": "MiniMax-M3",
        "num_kv_heads": int(k_cache.shape[1]),
        "num_q_heads": int(q.shape[1]),
        "num_tokens": int(q.shape[0]),
        "page_size": int(block_size),
        "phase": "decode",
        "q_len": 1,
        "quant": "fp8" if "float8" in str(k_cache.dtype).lower() else "dense",
        "top_k": int(topk_idx.shape[-1]),
    }
    if tp_size is not None:
        fields["tp_size"] = tp_size
    if world_size is not None:
        fields["world_size"] = world_size
    return CallDescriptor({key: value for key, value in fields.items() if value is not None})


def make_minimax_sparse_decode_dispatcher(
    baseline: Callable[..., torch.Tensor],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "attention.decode",
) -> Callable[..., torch.Tensor]:
    """Wrap the graph-native sparse-attend function used by M3 decode."""

    def dispatched(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        block_size,
        topk_idx,
        sm_scale=None,
        use_tma=True,
    ):
        def stock():
            return baseline(
                q, sink, k_cache, v_cache, req_to_token, seq_lens, slot_ids,
                block_size, topk_idx, sm_scale, use_tma,
            )

        supported = (
            torch.is_tensor(q)
            and torch.is_tensor(k_cache)
            and torch.is_tensor(v_cache)
            and torch.is_tensor(req_to_token)
            and torch.is_tensor(seq_lens)
            and torch.is_tensor(slot_ids)
            and torch.is_tensor(topk_idx)
            and sink is None
            and "float8" not in str(k_cache.dtype).lower()
        )
        if not supported:
            return stock()

        in_graph = _in_cuda_graph()
        descriptor = _descriptor(
            q, k_cache, req_to_token, topk_idx,
            block_size=int(block_size), in_graph=in_graph,
        )
        selected = registry.select(slot, descriptor, write_fired_receipt=False).impl
        if selected is None:
            return stock()

        live_inputs = {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "req_to_token": req_to_token,
            "seq_lens": seq_lens,
            "req_pool_indices": slot_ids,
            "topk_idx": topk_idx,
            "sm_scale": float(sm_scale if sm_scale is not None else q.shape[-1] ** -0.5),
            "block_size": int(block_size),
        }
        contract, allocation, tensor_inputs, input_bindings = _allocate_live_outputs(
            slot, live_inputs, like=q
        )
        if len(allocation.outputs) != 1:
            raise RuntimeError("attention.decode must declare exactly one output")
        output = allocation.outputs[0]
        committed = registry.select(slot, descriptor)
        if committed.impl is not selected:
            raise RuntimeError("attention.decode selection changed before invocation")
        # Audit stock first on pristine inputs; never clone the multi-GB cache.
        audited = not in_graph and _audit.sampled()
        expected = stock() if audited else None
        selected.entry(
            q,
            k_cache,
            v_cache,
            req_to_token,
            seq_lens,
            slot_ids,
            topk_idx,
            output,
            live_inputs["sm_scale"],
            int(block_size),
        )
        _validate_live_outputs(
            contract, allocation, tensor_inputs, input_bindings, like=q
        )
        if audited:
            _audit.record(slot, (output,), (expected,))
        # Eager recapture warmup may compile, but cannot prove graph execution.
        if in_graph or _audit.enabled():
            _receipts.completed(slot)
        return output

    return dispatched


__all__ = ["make_minimax_sparse_decode_dispatcher"]

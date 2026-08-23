"""Dispatch MiniMax-M3 sparse attend after stock scoring and top-k selection."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from cacheon import audit as _audit
from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor, msa_decode_score_call_descriptor
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
        selected = registry.select(slot, descriptor).impl
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


class _MsaScoreKernel:
    """Triton-launch-compatible proxy for the pinned decode score producer."""

    def __init__(self, stock: object, registry: KernelRegistry, slot: str) -> None:
        self._stock, self._registry, self._slot = stock, registry, slot

    def __getitem__(self, grid):
        stock_launch = self._stock[grid]

        def launch(
            q, k_cache, req_to_token, score, seq_lens, slot_ids,
            max_slots, batch_size, gqa_group_size, head_dim, block_size, topk,
            sm_scale, init_blocks, local_blocks, *strides, **meta,
        ):
            supported = (
                os.environ.get("CACHEON_MSA_DECODE_SCORE_SEAM") == "1"
                and meta.get("SCORE_TYPE") == "max"
                and all(torch.is_tensor(value) for value in (
                    q, k_cache, req_to_token, score, seq_lens, slot_ids
                ))
                and q.dim() == k_cache.dim() == score.dim() == 3
                and score.dtype == torch.float32
                and int(max_slots) == k_cache.shape[0]
                and int(batch_size) == q.shape[0] == score.shape[1]
                and int(head_dim) == q.shape[-1] == k_cache.shape[-1]
                and int(gqa_group_size) * k_cache.shape[1] == q.shape[1] == score.shape[0]
                and score.shape[2] > int(topk)
            )
            if not supported:
                return stock_launch(
                    q, k_cache, req_to_token, score, seq_lens, slot_ids,
                    max_slots, batch_size, gqa_group_size, head_dim, block_size,
                    topk, sm_scale, init_blocks, local_blocks, *strides, **meta,
                )

            tp_size, world_size = _runtime_parallel_sizes()
            descriptor = msa_decode_score_call_descriptor(
                dtype=_dtype_name(q.dtype),
                architecture=_arch_tag(q.device.index or 0) if q.is_cuda else None,
                graph_mode="cuda_graph" if _in_cuda_graph() else "eager",
                head_dim=int(head_dim), block_size=int(block_size),
                kv_len=int(score.shape[2]) * int(block_size), top_k=int(topk),
                init_blocks=int(init_blocks), local_blocks=int(local_blocks),
                batch_size=int(batch_size), num_q_heads=int(q.shape[1]),
                num_kv_heads=int(k_cache.shape[1]),
                quant="fp8" if "float8" in str(k_cache.dtype).lower() else "dense",
                tp_size=tp_size, world_size=world_size,
            )
            selected = self._registry.select(self._slot, descriptor).impl
            if selected is None:
                return stock_launch(
                    q, k_cache, req_to_token, score, seq_lens, slot_ids,
                    max_slots, batch_size, gqa_group_size, head_dim, block_size,
                    topk, sm_scale, init_blocks, local_blocks, *strides, **meta,
                )
            identity = (score.data_ptr(), score.shape, score.stride(), score.dtype, score.device)
            if self._registry.select(self._slot, descriptor).impl is not selected:
                raise RuntimeError("MSA score selection changed before invocation")
            selected.entry(
                q, k_cache, req_to_token, slot_ids, seq_lens, score, float(sm_scale),
                int(block_size), int(topk), int(init_blocks), int(local_blocks),
            )
            if identity != (
                score.data_ptr(), score.shape, score.stride(), score.dtype, score.device
            ):
                raise RuntimeError("MSA score candidate changed validator output identity")
            if descriptor["graph_mode"] == "cuda_graph" or _audit.enabled():
                _receipts.completed(self._slot)

        return launch


def make_msa_block_score_kernel(
    stock: object, *, registry: KernelRegistry = REGISTRY,
    slot: str = "attention.msa_block_score",
) -> object:
    if not hasattr(stock, "__getitem__"):
        raise TypeError("stock MSA score kernel is not launchable")
    return _MsaScoreKernel(stock, registry, slot)


__all__ = ["make_minimax_sparse_decode_dispatcher", "make_msa_block_score_kernel"]

"""One-call paged MSA prefill score-to-selection dispatch."""

from __future__ import annotations

import os
from typing import Callable

import torch

from cacheon import audit as _audit
from cacheon import receipts as _receipts
from cacheon.capabilities import msa_prefill_call_descriptor
from cacheon.minimax_sparse_prefill_slot import INPUT_NAMES
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.slots import get_slot
from cacheon.tensor_spec import validate_output_spec

def make_dispatcher(
    baseline_fn: Callable[..., object],
    _module: object,
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "attention.msa_prefill_block_score",
    arch_tag: Callable[[int], str | None],
    runtime_parallel_sizes: Callable[[], tuple[int | None, int | None]],
    dynamo_compiling: Callable[[], bool],
    in_cuda_graph: Callable[[], bool],
) -> Callable[..., object]:
    """Replace the stock indexer with one batched, paged selection call."""

    output_slot = get_slot(slot)

    def dispatched(
        q,
        k_cache,
        v_cache,
        sink,
        req_to_token,
        slot_ids,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_seqlen_q,
        max_seqlen_k,
        block_size_q,
        block_size_k,
        topk,
        init_blocks=1,
        local_blocks=2,
        sm_scale=None,
        use_tma=False,
        score_type="max",
        disable_index_value=False,
        cu_seqblocks_q=None,
        max_seqblock_q=None,
        all_seqblock_q=None,
    ):
        def stock():
            return baseline_fn(
                q,
                k_cache,
                v_cache,
                sink,
                req_to_token,
                slot_ids,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                max_seqlen_q,
                max_seqlen_k,
                block_size_q,
                block_size_k,
                topk,
                init_blocks,
                local_blocks,
                sm_scale,
                use_tma,
                score_type,
                disable_index_value,
                cu_seqblocks_q,
                max_seqblock_q,
                all_seqblock_q,
            )

        if dynamo_compiling() or in_cuda_graph():
            return stock()
        if os.environ.get("CACHEON_MSA_PREFILL_SEAM") != "1":
            return stock()

        selected = False
        try:
            if not (
                disable_index_value
                and score_type == "max"
                and sink is None
                and k_cache.shape[1] == 1
                and q.is_cuda
                and q.dim() == 3
            ):
                return stock()
            if (
                cu_seqblocks_q is None
                or max_seqblock_q is None
                or all_seqblock_q is None
            ):
                return stock()

            total_q, num_heads, head_dim = q.shape
            batch_size = cu_seqlens.shape[0] - 1
            scale = float(
                sm_scale if sm_scale is not None else head_dim**-0.5
            ) * 1.4426950409
            tp_size, world_size = runtime_parallel_sizes()
            descriptor = msa_prefill_call_descriptor(
                dtype=str(q.dtype).removeprefix("torch."),
                architecture=arch_tag(q.device.index or 0),
                head_dim=head_dim,
                block_size=block_size_k,
                q_len=max_seqlen_q,
                kv_len=max_seqlen_k,
                top_k=topk,
                q_block_size=block_size_q,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                batch_size=batch_size,
                num_q_heads=num_heads,
                num_kv_heads=1,
                num_tokens=total_q,
                tp_size=tp_size,
                world_size=world_size,
            )
            decision = registry.select(slot, descriptor)
            if not decision.use_candidate:
                return stock()
            selected = True

            inputs = {
                "q": q,
                "index_k_cache": k_cache,
                "req_to_token": req_to_token,
                "slot_ids": slot_ids,
                "cu_seqlens": cu_seqlens,
                "seq_lens": seq_lens,
                "prefix_lens": prefix_lens,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "block_size_q": block_size_q,
                "block_size_k": block_size_k,
                "topk": topk,
                "init_blocks": init_blocks,
                "local_blocks": local_blocks,
                "scale": scale,
                "cu_seqblocks_q": cu_seqblocks_q,
                "max_seqblock_q": max_seqblock_q,
                "all_seqblock_q": all_seqblock_q,
            }
            contract = output_slot.output_contract(inputs)
            spec = contract.outputs[0]
            topk_idx = torch.full(
                spec.shape,
                -1,
                dtype=spec.dtype or torch.int32,
                device=spec.device or q.device,
            )
            validate_output_spec(
                contract,
                [topk_idx],
                fallback_dtype=torch.int32,
                fallback_device=q.device,
                inputs=tuple(v for v in inputs.values() if torch.is_tensor(v)),
            )

            audited = _audit.sampled()
            expected_idx = None
            if audited:
                expected = stock()
                expected_idx = (
                    expected[1]
                    if isinstance(expected, (tuple, list)) and len(expected) > 1
                    else None
                )

            decision.impl.entry(*(inputs[name] for name in INPUT_NAMES), topk_idx)
            if audited:
                if expected_idx is None:
                    _audit.baseline_refused(slot)
                else:
                    _audit.record(slot, (topk_idx,), (expected_idx,))
            _receipts.completed(slot)
            return None, topk_idx
        except Exception as exc:  # noqa: BLE001
            if registry.strict:
                raise
            result = stock()
            if selected:
                _receipts.fallback(slot, exc)
            return result

    return dispatched

"""MiniMax-M3 graph-native sparse attend over validator-selected paged K/V."""

from __future__ import annotations

import math
from typing import Any

import torch


def _inputs(
    *,
    batch: int,
    context: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    top_k: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> dict[str, Any]:
    pages = math.ceil(context / block_size)
    slots_per_request = pages * block_size
    max_requests = batch + 2
    # Shared banks bound verification; every row still has a shuffled page table.
    max_slots = 2 * slots_per_request
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float32
        ).to(dtype)

    req_to_token = torch.zeros(
        max_requests, slots_per_request, dtype=torch.int32, device=device
    )
    logical = torch.arange(slots_per_request, device=device)
    for request in range(max_requests):
        page_order = torch.randperm(pages, generator=generator, device=device)
        req_to_token[request] = (
            (request % 2) * slots_per_request
            + page_order[logical // block_size] * block_size
            + logical % block_size
        ).to(torch.int32)
    # Vary ragged tails inside one graph bucket on every replay.
    seq_lens = torch.randint(
        max(block_size + 1, context - block_size),
        context + 1,
        (batch,),
        generator=generator,
        dtype=torch.int32,
        device=device,
    )
    req_pool_indices = torch.randperm(
        max_requests, generator=generator, device=device
    )[:batch].to(torch.int32)

    topk_idx = torch.full(
        (num_kv_heads, batch, top_k), -1, dtype=torch.int32, device=device
    )
    for head in range(num_kv_heads):
        for request in range(batch):
            # Unsorted selections plus -1 padding exercise the public format.
            width = min(top_k, math.ceil(int(seq_lens[request]) / block_size))
            chosen = torch.randperm(
                math.ceil(int(seq_lens[request]) / block_size),
                generator=generator,
                device=device,
            )[:width]
            topk_idx[head, request, :width] = chosen.to(torch.int32)

    return {
        "q": randn(batch, num_q_heads, head_dim),
        "k_cache": randn(max_slots, num_kv_heads, head_dim),
        "v_cache": randn(max_slots, num_kv_heads, head_dim),
        "req_to_token": req_to_token,
        "seq_lens": seq_lens,
        "req_pool_indices": req_pool_indices,
        "topk_idx": topk_idx,
        "sm_scale": head_dim ** -0.5,
        "block_size": block_size,
    }


def _logical_positions(
    block_ids: torch.Tensor, *, block_size: int, seq_len: int, device
) -> torch.Tensor:
    seen: set[int] = set()
    positions: list[torch.Tensor] = []
    for raw in block_ids.tolist():
        block = int(raw)
        if block < 0 or block in seen:
            continue
        seen.add(block)
        start = block * block_size
        stop = min(seq_len, start + block_size)
        if start < stop:
            positions.append(torch.arange(start, stop, device=device))
    if not positions:
        raise ValueError("sparse-decode selection contains no valid token")
    return torch.cat(positions)


def _reference(inputs: dict[str, Any]) -> torch.Tensor:
    q = inputs["q"]
    k_cache = inputs["k_cache"]
    v_cache = inputs["v_cache"]
    req_to_token = inputs["req_to_token"]
    seq_lens = inputs["seq_lens"]
    req_pool_indices = inputs["req_pool_indices"]
    topk_idx = inputs["topk_idx"]
    block_size = int(inputs["block_size"])
    q_per_kv = q.shape[1] // k_cache.shape[1]
    output = torch.empty_like(q)

    for request in range(q.shape[0]):
        seq_len = int(seq_lens[request])
        request_row = int(req_pool_indices[request])
        for kv_head in range(k_cache.shape[1]):
            logical = _logical_positions(
                topk_idx[kv_head, request],
                block_size=block_size,
                seq_len=seq_len,
                device=q.device,
            )
            physical = req_to_token[request_row, logical].long()
            selected_k = k_cache[physical, kv_head].float()
            selected_v = v_cache[physical, kv_head].float()
            start = kv_head * q_per_kv
            stop = start + q_per_kv
            logits = torch.matmul(
                q[request, start:stop].float(), selected_k.t()
            ) * float(inputs["sm_scale"])
            output[request, start:stop] = torch.matmul(
                torch.softmax(logits, dim=-1), selected_v
            ).to(output.dtype)
    return output


def _invoke(entry, inputs: dict[str, Any], outputs, _prepared) -> None:
    entry(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["req_to_token"],
        inputs["seq_lens"],
        inputs["req_pool_indices"],
        inputs["topk_idx"],
        outputs[0],
        inputs["sm_scale"],
        inputs["block_size"],
    )


def build_slot(SlotSpec, Correctness, tolerances):
    return SlotSpec(
        name="attention.decode",
        entry="attention_decode",
        summary="MiniMax-M3 sparse decode attend over validator-selected paged K/V",
        kind="block",
        make_inputs=_inputs,
        out_shapes=lambda inputs: [tuple(inputs["q"].shape)],
        invoke_reference=lambda inputs: [_reference(inputs)],
        invoke_entry=_invoke,
        graph_dynamic_inputs=(
            "q", "k_cache", "v_cache", "req_to_token", "seq_lens",
            "req_pool_indices", "topk_idx",
        ),
        shapes=tuple(
            {"batch": batch, "context": 9344, "num_q_heads": 16,
             "num_kv_heads": 1, "head_dim": 128, "block_size": 128, "top_k": 16}
            for batch in (1, 32, 128)
        ),
        correctness=Correctness("matched_ratio", min_ratio=0.99),
        tolerances=tolerances,
        kl_threshold=3e-2,
    )


__all__ = ["build_slot"]

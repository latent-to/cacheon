"""Graph-capturable Torch reference bundle for MiniMax-M3 sparse decode attend."""

from __future__ import annotations

import torch


def attention_decode(
    q,
    k_cache,
    v_cache,
    req_to_token,
    seq_lens,
    req_pool_indices,
    topk_idx,
    out,
    sm_scale,
    block_size,
):
    batch, num_q_heads, _head_dim = q.shape
    num_kv_heads, _, top_k = topk_idx.shape
    offsets = torch.arange(block_size, device=q.device, dtype=torch.int64)
    block_ids = topk_idx.to(torch.int64)
    logical = block_ids[..., None] * block_size + offsets
    valid = (block_ids[..., None] >= 0) & (
        logical < seq_lens.view(1, batch, 1, 1)
    )
    logical = logical.clamp(min=0, max=req_to_token.shape[1] - 1)
    request_rows = req_pool_indices.view(1, batch, 1, 1).expand_as(logical).long()
    physical = req_to_token[request_rows, logical].long()
    kv_heads = torch.arange(num_kv_heads, device=q.device).view(
        num_kv_heads, 1, 1, 1
    ).expand_as(physical)
    selected_k = k_cache[physical, kv_heads].reshape(
        num_kv_heads, batch, top_k * block_size, -1
    ).permute(1, 0, 2, 3)
    selected_v = v_cache[physical, kv_heads].reshape(
        num_kv_heads, batch, top_k * block_size, -1
    ).permute(1, 0, 2, 3)
    mask = valid.reshape(num_kv_heads, batch, top_k * block_size).permute(1, 0, 2)
    group = num_q_heads // num_kv_heads
    grouped_q = q.reshape(batch, num_kv_heads, group, -1)
    logits = (
        torch.einsum("bhgd,bhnd->bhgn", grouped_q.float(), selected_k.float())
        * sm_scale
    )
    probabilities = torch.softmax(
        logits.masked_fill(~mask.unsqueeze(2), float("-inf")), dim=-1
    )
    out.copy_(
        torch.einsum("bhgn,bhnd->bhgd", probabilities, selected_v.float())
        .reshape(batch, num_q_heads, -1)
        .to(out.dtype)
    )

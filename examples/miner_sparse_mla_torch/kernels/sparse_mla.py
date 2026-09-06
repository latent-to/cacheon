"""Faithful bounded-memory sparse MLA example, without a performance claim."""

import torch


def sparse_mla(q, kv_cache, indices, seq_lens, out, value_dim, qk_scale, value_scale):
    """Fill BF16 latent output from the active prefix of physical token indices."""
    page_size = kv_cache.shape[1]
    source = kv_cache.view(torch.uint8) if kv_cache.dtype == torch.float8_e4m3fn else kv_cache
    columns = torch.arange(indices.shape[1], device=q.device)
    # Static shape-based chunks are captured normally; lengths and indices stay
    # tensors so new decode values do not become capture-time Python constants.
    for start in range(0, q.shape[0], 4):
        end = min(start + 4, q.shape[0])
        selected = indices[start:end].long()
        valid = (selected >= 0) & (columns < seq_lens[start:end, None])
        safe = torch.where(valid, selected, 0)
        kv = source[safe // page_size, safe % page_size]
        if kv_cache.dtype == torch.float8_e4m3fn:
            kv = kv.view(kv_cache.dtype)
        kv = kv.float().masked_fill(~valid[:, :, None], 0.0)
        logits = torch.bmm(q[start:end].float(), kv.transpose(1, 2)) * qk_scale
        logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
        has_keys = valid.any(-1)[:, None, None]
        logits = torch.where(has_keys, logits, 0.0)
        weights = logits.softmax(-1).masked_fill(~valid[:, None, :], 0.0)
        values = torch.bmm(weights, kv[:, :, :value_dim]) * value_scale
        out[start:end].copy_(values)

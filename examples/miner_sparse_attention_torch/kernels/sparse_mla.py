"""Faithful raw-query sparse MLA example, without a performance claim."""

import torch


def sparse_mla(q, q_rope, positions, cos_sin_cache, is_neox, kv_cache,
               indices, seq_lens, out, value_dim, qk_scale, value_scale):
    """Rotate and quantize Q, then combine the selected engine-owned cache rows."""
    rope = q_rope.float()
    cosine, sine = cos_sin_cache[positions.long()].float().chunk(2, dim=-1)
    cosine, sine = cosine[:, None], sine[:, None]
    if is_neox:
        first, second = rope.chunk(2, dim=-1)
        rotated = torch.cat((first * cosine - second * sine,
                             second * cosine + first * sine), dim=-1)
    else:
        paired = rope.reshape(*rope.shape[:-1], -1, 2)
        first, second = paired[..., 0], paired[..., 1]
        rotated = torch.stack((first * cosine - second * sine,
                               second * cosine + first * sine), dim=-1).flatten(-2)
    q = torch.cat((q.float(), rotated), dim=-1).clamp(-448, 448).to(torch.float8_e4m3fn)
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

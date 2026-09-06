"""Readable merged-selection control; intermediates are private to the candidate."""

import torch


def indexer_select(q, key_pages, key_scales, weights, page_table, row_to_batch,
                   lengths, page_offsets, positions, cos_sin_cache, q_scale_gate,
                   num_init_tokens, num_local_tokens, top_k, out):
    """Score and select one row at a time, preserving graph-dynamic tensor inputs."""
    if q.dtype != torch.float8_e4m3fn:
        dim = cos_sin_cache.shape[-1]
        c, s = cos_sin_cache.index_select(0, positions.long()).unsqueeze(1).chunk(2, -1)
        a, b = q[..., :dim:2].float(), q[..., 1:dim:2].float()
        rope = torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)
        value = torch.cat((rope, q[..., dim:].float()), -1)
        scale = value.abs().amax(-1).clamp_min(1e-4) / 448
        q = (value / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        weights = (weights.float() * q_scale_gate) * scale
    size = key_pages.shape[1]
    columns = torch.arange(page_table.shape[1] * size, device=q.device)
    width = min(top_k, columns.numel())
    out.fill_(-1)
    for row in range(q.shape[0]):
        valid = columns < lengths[row]
        logical = torch.where(valid, columns + page_offsets[row], 0)
        pages = page_table[row_to_batch[row].long(), logical // size].long()
        keys = key_pages.view(torch.uint8)[pages, logical % size].view(key_pages.dtype).float()
        scores = ((q[row].float() @ keys.T).relu() * weights[row, :, None]).sum(0)
        scores *= key_scales[pages, logical % size]
        forced = (columns < num_init_tokens) | (columns >= lengths[row] - num_local_tokens)
        scores = scores.masked_fill(forced, torch.inf).masked_fill(~valid, -torch.inf)
        chosen = torch.argsort(scores, descending=True, stable=True)[:width]
        physical = pages[chosen] * size + logical[chosen] % size
        out[row, :width].copy_(torch.where(scores[chosen] != -torch.inf, physical, -1))

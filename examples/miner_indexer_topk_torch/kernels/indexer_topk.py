"""Faithful top-k and page translation using ordinary Torch operations."""

import torch


def indexer_topk(scores, lengths, row_starts, page_table, row_to_batch,
                 page_offsets, out, page_size, top_k):
    """Fill physical token indices without retaining replay-time metadata."""
    columns = torch.arange(scores.shape[1], device=scores.device)
    out.fill_(-1)
    width = min(top_k, scores.shape[1])
    for start in range(0, scores.shape[0], 64):
        end = min(start + 64, scores.shape[0])
        lo = row_starts[start:end, None]
        mask = (columns >= lo) & (columns < lo + lengths[start:end, None])
        values, selected = scores[start:end].masked_fill(~mask, -torch.inf).topk(width, dim=-1)
        valid = values != -torch.inf
        logical = torch.where(valid, selected - lo + page_offsets[start:end, None], 0)
        physical = page_table[row_to_batch[start:end, None].long(), logical // page_size].long()
        out[start:end, :width].copy_(torch.where(valid, physical * page_size + logical % page_size, -1))

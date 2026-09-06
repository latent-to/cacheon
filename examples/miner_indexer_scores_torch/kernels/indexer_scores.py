"""Faithful score-only example using bounded Torch tiles, with no top-k work."""

import torch


def indexer_scores(q, key_pages, key_scales, weights, starts, ends,
                   page_table, row_to_batch, out):
    """Read current tensors on every replay and overwrite every FP32 score."""
    page_size = key_pages.shape[1]
    key_bytes = key_pages.view(torch.uint8)
    for row in range(0, q.shape[0], 16):
        stop = min(row + 16, q.shape[0])
        for col in range(0, out.shape[1], 256):
            columns = torch.arange(col, min(col + 256, out.shape[1]), device=q.device)
            valid = (columns >= starts[row:stop, None]) & (columns < ends[row:stop, None])
            physical = page_table[row_to_batch[row:stop, None].long(), columns // page_size].long()
            physical = torch.where(valid, physical, 0)
            keys = key_bytes[physical, columns % page_size].view(key_pages.dtype).float()
            dots = torch.einsum("thd,tkd->thk", q[row:stop].float(), keys)
            dots *= key_scales[physical, columns % page_size].unsqueeze(1)
            scores = (dots.relu() * weights[row:stop, :, None]).sum(1)
            out[row:stop, col:col + columns.numel()].copy_(torch.where(valid, scores, 0))

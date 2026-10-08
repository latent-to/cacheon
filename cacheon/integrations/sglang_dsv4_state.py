"""Interpret the DeepSeek-V4 family's paged cache records without changing their stored bytes.

The pool keeps whole pages: a buffer row is one page of token rows followed by
that page's scale rows, padded to the reader's stride, so rows are addressed by
page number rather than by token slot. Sliding-window pages live in their own
pool behind the engine's full-to-window slot map; the compressed pages of a
kv-source layer line up with the full pages they summarise; the indexer keeps a
few shorter FP4 pages per full page; and a ratio-2 source layer holds a
pending-pair ring per request (SGLang 0.5.21, read from source 2026-10-08).
"""

from __future__ import annotations

from functools import partial

import torch

# The sixteen E2M1 values, indexed by their four-bit code.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _nibbles(rows: torch.Tensor) -> torch.Tensor:
    """Unpack two E2M1 codes per byte, the even index in the low nibble."""
    return torch.stack((rows & 0x0F, rows >> 4), -1).reshape(*rows.shape[:-1], -1)


def _fp8_page_values(raw: torch.Tensor, *, page: int, data: int, tile: int, scale: int,
                     rope: int) -> torch.Tensor:
    """A page of ``data``-byte token rows, then ``scale`` exponent bytes per token.

    The quantized head of each row is E4M3 with one UE8M0 exponent per ``tile``
    values; a ``rope`` tail, when present, is kept as BF16.
    """
    n = raw.shape[0]
    if raw.dtype != torch.uint8 or raw.ndim != 2 or raw.shape[1] < page * (data + scale):
        raise RuntimeError("unrecognized paged DeepSeek-V4 cache layout")
    rows = raw[:, : page * data].reshape(n, page, data)
    quantized = data - rope * 2
    values = rows[..., :quantized].view(torch.float8_e4m3fn).float()
    exponents = raw[:, page * data : page * (data + scale)].reshape(n, page, scale)
    scales = torch.exp2(exponents[..., : quantized // tile].float() - 127)
    values = (values.reshape(n, page, quantized // tile, tile) * scales.unsqueeze(-1))
    values = values.reshape(n, page, quantized)
    if not rope:
        return values
    rotary = rows[..., quantized:].contiguous().view(torch.bfloat16).float()
    return torch.cat((values, rotary), -1)


def _fp4_page_values(raw: torch.Tensor, *, page: int, data: int, tile: int,
                     scale: int) -> torch.Tensor:
    """A page of packed E2M1 token rows, then one E4M3 scale per ``tile`` values."""
    n = raw.shape[0]
    if raw.dtype != torch.uint8 or raw.ndim != 2 or raw.shape[1] < page * (data + scale):
        raise RuntimeError("unrecognized paged DeepSeek-V4 FP4 cache layout")
    values = _E2M1.to(raw.device)[_nibbles(raw[:, : page * data].reshape(n, page, data)).long()]
    scales = raw[:, page * data : page * (data + scale)].reshape(n, page, scale)
    scales = scales.view(torch.float8_e4m3fn).float()
    return (values.reshape(n, page, scale, tile) * scales.unsqueeze(-1)).reshape(n, page, data * 2)


def _fp4_index_values(raw: torch.Tensor, *, page: int, head: int) -> torch.Tensor:
    """An index page: packed E2M1 keys, then one UE8M0 exponent per 32 values."""
    n, payload, scale = raw.shape[0], head // 2, head // 32
    if raw.dtype != torch.uint8 or raw.ndim != 2 or raw.shape[1] != page * (payload + scale):
        raise RuntimeError("unrecognized paged DeepSeek-V4 index cache layout")
    values = _E2M1.to(raw.device)[_nibbles(raw[:, : page * payload].reshape(n, page, payload)).long()]
    scales = torch.exp2(raw[:, page * payload :].reshape(n, page, scale).float() - 127)
    return (values.reshape(n, page, scale, 32) * scales.unsqueeze(-1)).reshape(n, page, head)


# The paged FlashMLA layouts a pool declares, by their runtime name.
_LAYOUTS = {
    "v4": partial(_fp8_page_values, data=576, tile=64, scale=8, rope=64),
    "v41": partial(_fp8_page_values, data=512, tile=32, scale=16, rope=0),
    "v41_fp4": partial(_fp4_page_values, data=256, tile=16, scale=32),
}


def _page_format(kv_pool):
    layout = getattr(kv_pool, "kv_layout", None)
    held = _LAYOUTS.get(getattr(layout, "value", layout))
    if held is None or kv_pool.kv_buffer[0].dtype != torch.uint8:
        raise RuntimeError(f"unrecognized DeepSeek-V4 cache layout {layout!r}")
    return partial(held, page=int(kv_pool.page_size))


def dsv4_state_rows(pool, batch, layer: int | None = None) -> list[tuple] | None:
    """Return the paged rows a call may write and their formats, or None for other pools.

    Every layer writes its sliding-window pages. A kv-source layer also writes the
    compressed pages, the index pages and, at ratio 2, the pending-pair ring of
    each request; the other layers of its ratio only read them.
    """
    mapping = getattr(pool, "layer_mapping", None)
    if mapping is None:
        return None
    swa = getattr(pool, "swa_kv_pool", None)
    if swa is None or pool.unified_kv_pool is not None:
        raise RuntimeError("DeepSeek-V4 audit requires the paged sliding-window cache layout")
    full = batch.out_cache_loc.long()
    pages = torch.unique(full // pool.page_size)
    windows = torch.unique(pool.translate_loc_from_full_to_swa(full).long() // swa.page_size)
    slots = batch.req_pool_indices.long()
    window_held = _page_format(swa)
    rows = []
    for index in range(len(mapping)) if layer is None else [layer]:
        item = mapping[index] if 0 <= index < len(mapping) else None
        if item is None:
            raise RuntimeError(f"layer {index} has no DeepSeek-V4 cache in this pool")
        rows.append((swa.kv_buffer[index - pool._stage_start], 0, windows, window_held))
        ratio, local, kv_pool = item
        if kv_pool is None or (ratio in (1, 2) and index not in pool.sources_by_ratio[ratio]):
            continue
        rows.append((kv_pool.kv_buffer[local], 0, pages, _page_format(kv_pool)))
        indexer = pool.index_pools.get(ratio)
        if indexer is not None:
            if not indexer.use_fp4_indexer:
                raise RuntimeError("only the FP4 DeepSeek-V4 index cache is recognized")
            per_page = int(kv_pool.page_size) // int(indexer.page_size)
            index_pages = pages[:, None] * per_page + torch.arange(per_page, device=pages.device)
            held = partial(_fp4_index_values, page=int(indexer.page_size), head=int(indexer.index_head_dim))
            rows.append((indexer.index_k_with_scale_buffer[local], 0, index_pages.reshape(-1), held))
        state = pool.compress_state_pools[index]
        if state is not None:
            if not getattr(state, "request_scoped", False):
                raise RuntimeError("only request-scoped DeepSeek-V4 compressor state is recognized")
            # The allocation also holds a spare ring, sentinel and alignment rows.
            scores = state.kv_score_buffer.kv_score[:pool.num_req_slots * state.ring_size]
            ring = scores.reshape(pool.num_req_slots, -1)
            rows.append((ring, 0, slots, scores.dtype))
    return rows

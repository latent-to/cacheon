"""Interpret the pinned DSA cache records without changing their stored bytes.

GLM's FP8 MLA cache mixes quantized values, FP32 scales and BF16 rotary values;
the indexer keeps another paged key/scale cache. Treating the former as all FP8
and omitting the latter left the wide-node audit incomplete (B300, 2026-09-20).
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch

StateFormat = torch.dtype | Callable[[torch.Tensor], torch.Tensor]


def state_values(raw: torch.Tensor, held: StateFormat) -> torch.Tensor:
    """Read numerical values from a captured record; restoration uses the raw copy."""
    return held(raw) if callable(held) else raw.view(held)


def _mla_values(raw: torch.Tensor, *, latent: int, rope: int) -> torch.Tensor:
    scales_end = latent + latent // 128 * 4
    if raw.dtype != torch.uint8 or raw.shape[-1] != scales_end + rope * 2:
        raise RuntimeError("unrecognized packed DSA MLA cache layout")
    shape = (*raw.shape[:-1], latent // 128, 128)
    keys = raw[..., :latent].view(torch.float8_e4m3fn).float().reshape(shape)
    scales = raw[..., latent:scales_end].contiguous().view(torch.float32)
    keys = (keys * scales.unsqueeze(-1)).reshape(*raw.shape[:-1], latent)
    rotary = raw[..., scales_end:].contiguous().view(torch.bfloat16).float()
    return torch.cat((keys, rotary), dim=-1)


def _index_values(raw: torch.Tensor, *, page: int, head: int) -> torch.Tensor:
    if raw.dtype != torch.uint8 or raw.ndim != 2 or raw.shape[1] != page * (head + 4):
        raise RuntimeError("unrecognized paged DSA index cache layout")
    keys = raw[:, :page * head].contiguous().view(torch.float8_e4m3fn).float()
    scales = raw[:, page * head:].contiguous().view(torch.float32)
    return keys.reshape(raw.shape[0], page, head) * scales.unsqueeze(-1)


def dsa_state_rows(pool, locations: torch.Tensor, layer: int | None = None) -> list[tuple] | None:
    """Return raw DSA rows and their numerical formats, or None for ordinary pools.

    ``layer`` keeps only that decoder layer's two caches. Grading every layer's rows on
    every layer call cost 156 buffers a call and 78 calls a forward: the GLM audit took
    65 minutes against 14 before it (B300, 2026-09-23).
    """
    index_buffers = getattr(pool, "index_k_with_scale_buffer", None)
    if index_buffers is None:
        return None
    if torch.version.hip or pool.page_size != 64 or pool.index_head_dim != 128:
        raise RuntimeError("DSA audit requires the commissioned CUDA cache layout")
    kv_buffers = pool.kv_buffer
    if layer is not None:
        local = layer - pool.start_layer
        if not (0 <= local < len(kv_buffers) and local < len(index_buffers)):
            raise RuntimeError(f"layer {layer} has no DSA cache in this pool")
        kv_buffers, index_buffers = [kv_buffers[local]], [index_buffers[local]]
    index = locations.long()
    rows = []
    for buffer in kv_buffers:
        if getattr(pool, "dsa_kv_cache_store_fp8", False):
            held = partial(_mla_values, latent=pool.kv_lora_rank, rope=pool.qk_rope_head_dim)
        else:
            held = pool.dtype if buffer.dtype == torch.uint8 else buffer.dtype
        rows.append((buffer, 0, index, held))
    # The store addresses tokens, but K and scale occupy separate regions of each
    # page. Preserve whole touched pages, as the engine's own cache offload does.
    pages = torch.unique(index // pool.page_size)
    held = partial(_index_values, page=pool.page_size, head=pool.index_head_dim)
    rows.extend((buffer, 0, pages, held) for buffer in index_buffers if buffer.shape[0])
    return rows

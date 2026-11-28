"""Inkling 2 engine state a node call can change beyond the full-attention cache.

Every Inkling 2 module binds as a node (``sglang_nodes.bind``). Two things such a call
can change sit outside what the generic node audit reads, so each is put back between
stock, its twin and the candidate, and graded after the candidate:

- the sliding-window KV sub-pool, which local attention writes at the full-pool write
  locations translated through the pool's full-to-window mapping;
- the attention-residual state a decoder layer advances in place
  (``Inkling2AttnResState``: the running partial and the closed blocks). Every later
  layer and loop pass reads it, and later passes read the loop messages, so the partial
  and the messages the call was handed are graded too: a call that rewrote them in
  place would pass its own call and poison the next.

Conv state needs no reader of its own: a looped layer writes its visit's conv layer, so
the dispatcher keeps every layer's rows for an Inkling 2 node.
"""

from __future__ import annotations

import sys

import torch


def _first(call: tuple, module: str, name: str):
    kind = getattr(sys.modules.get(module), name, None)
    return next((v for v in call if isinstance(kind, type) and isinstance(v, kind)), None)


def window_rows(pool, loc: torch.Tensor, kv_rows, layer: int | None = None) -> list:
    """The sliding-window sub-pool rows a batch writing at full-pool ``loc`` reaches.

    Only the mapped layout is read here. A per-request ring or request window keeps
    its own addressing and stays as it was before the sub-pool was read at all.
    """

    window = getattr(pool, "swa_kv_pool", None)
    if window is None:
        return []
    if getattr(pool, "swa_uses_full_indices", False):
        index = loc
    elif (getattr(pool, "full_to_swa_index_mapping", None) is None
          or getattr(pool, "request_window", None) is not None
          or getattr(pool, "swa_req_ring_size", None) is not None):
        return []
    else:
        index = pool.translate_loc_from_full_to_swa(loc).long()
    # The mapping's appended sentinel sends a padded location to -1.
    return kv_rows(window, index[index >= 0], layer=layer)


def _visit_layers(module, kwargs: dict) -> tuple[int | bool | None, int | bool | None]:
    """Return ``(kv_layer, conv_layer)`` for one Inkling 2 call, or ``(None, None)`` for whole-model nodes.

    Each element is an int global cache/conv layer ID for this visit, ``False`` when the
    module owns neither KV nor conv writes (e.g. MoE MLP), or ``None`` to audit every layer.
    Auditing all 48 visits on every sampled ``mlp``/``attn`` call copied ~192 buffers per call
    (366 s for a single 16x8K audit batch on H200, 2026-11-27) and diluted ``w-conv2`` 47/48.
    """

    pass_idx = kwargs.get("pass_idx", 0)
    if type(pass_idx) is not int or pass_idx < 0:
        return None, None
    inkling2 = sys.modules.get("sglang.srt.models.inkling2")
    attn_mod = sys.modules.get("sglang.srt.models.inkling_common.inkling2_attn")
    moe_mod = sys.modules.get("sglang.srt.models.inkling_common.inkling2_moe")
    sconv_mod = sys.modules.get("sglang.srt.models.inkling_common.inkling2_sconv")
    sconv_base = sys.modules.get("sglang.srt.models.inkling_common.sconv")
    radix_mod = sys.modules.get("sglang.srt.layers.radix_attention")

    if isinstance(module, getattr(inkling2, "Inkling2DecoderLayer", ())):
        cfg = getattr(module, "config", None)
        lid = getattr(module, "layer_id", None)
        if cfg is not None and type(lid) is int and hasattr(cfg, "cache_layer_id") and hasattr(cfg, "conv_layer_id"):
            return cfg.cache_layer_id(lid, pass_idx), cfg.conv_layer_id(lid, pass_idx)
        return None, None
    if isinstance(module, getattr(attn_mod, "Inkling2Attention", ())):
        conv_ids = getattr(module, "conv_layer_ids", None)
        pass_map = getattr(module, "pass_attn_index", None)
        attns = getattr(module, "attn", None)
        if conv_ids is not None and pass_map is not None and pass_idx < len(conv_ids) and pass_idx < len(pass_map):
            attn = attns[pass_map[pass_idx]] if isinstance(attns, torch.nn.ModuleList) else attns
            kv_lid = getattr(attn, "layer_id", None)
            if type(kv_lid) is int:
                return kv_lid, conv_ids[pass_idx]
        return None, None
    if isinstance(module, getattr(radix_mod, "RadixAttention", ())):
        lid = getattr(module, "layer_id", None)
        if type(lid) is int:
            return lid, False
        return None, False
    if isinstance(module, (getattr(sconv_mod, "ShortConvolution", ()), getattr(sconv_base, "ShortConvolution", ()))):
        lid = getattr(module, "layer_id", None)
        return (False, lid) if type(lid) is int else (False, None)
    if isinstance(module, (
        getattr(moe_mod, "Inkling2MoE", ()),
        getattr(moe_mod, "Inkling2Experts", ()),
        getattr(moe_mod, "Inkling2Gate", ()),
    )):
        return False, False
    return None, None


def carried(call: tuple):
    """The attention-residual state and loop messages a call is handed, as restored later; else None."""

    state = _first(call, "sglang.srt.models.inkling_common.inkling2_attn_res", "Inkling2AttnResState")
    if state is None:
        return None
    bank = _first(call, "sglang.srt.models.inkling_common.inkling2_loop", "LoopMessageBank")
    return state, state.partial, state.blocks, list(state.blocks), [] if bank is None else list(bank.blocks)


def held(saved) -> list[torch.Tensor]:
    """The handed tensors put back by value: the partial, the closed blocks and the messages."""

    return [] if saved is None else [saved[1], *saved[3], *saved[4]]


def put_back(saved) -> None:
    """Hand the state its partial and its own block list again, holding the blocks it held."""

    if saved is not None:
        state, partial, blocks, closed, _ = saved
        blocks[:] = closed
        state.blocks, state.partial = blocks, partial


def left(saved) -> list[torch.Tensor]:
    """What the call leaves: the state's partial and blocks, then the partial and messages it was handed."""

    return [] if saved is None else [saved[0].partial, *saved[0].blocks, saved[1], *saved[4]]

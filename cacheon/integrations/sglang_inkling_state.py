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


def window_rows(pool, loc: torch.Tensor, kv_rows) -> list:
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
    return kv_rows(window, index[index >= 0])


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

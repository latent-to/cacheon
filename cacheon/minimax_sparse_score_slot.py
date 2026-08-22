"""MiniMax-M3 graph-native decode block scores over paged index-K.

Replaces the V1 dense contract, which modelled a different operation than production
runs: V1 took a gathered ``index_k:(B,S,1,D)``, summed every index query head into one
row and emitted one ``(B,nblocks)`` sheet, while the pinned ``_decode_score_kernel``
reads the paged cache through ``req_to_token``/``slot_ids``, emits ``(Hq,B,nblocks)``,
and its consumer takes top-k *per* index head then unions. ``topk(sum(heads))`` is not
``union(topk(head))``, so a candidate verified against V1 could pass and still select
the wrong blocks. The candidate owns score production only; the validator keeps the
slab, the page mapping, every scalar policy, the top-k, the union and the attend.

Two stock details the contract inherits: scores are in log2 units (stock folds
``log2(e)`` into ``sm_scale``) with forced ``1e30`` init and ``1e29`` local blocks; and
columns at or beyond a row's live block count must be ``-inf``. Stock skips those
writes because its consumers clamp their scan, but the validator allocates the slab
uninitialised, so an unwritten column is garbage the correctness gate would score.
Writing the tail is a strict superset of stock and stays a drop-in.
"""

from __future__ import annotations

import math
from typing import Any

import torch

_LOG2E = 1.4426950408889634


def _inputs(
    *,
    batch: int,
    context: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> dict[str, Any]:
    pages = math.ceil(context / block_size)
    slots_per_request = pages * block_size
    max_requests = batch + 2
    # Two shared banks bound verification while every row keeps a shuffled page
    # table, so a candidate that ignores req_to_token reads the wrong tokens.
    max_slots = 2 * slots_per_request
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float32
        ).to(dtype)

    req_to_token = torch.zeros(
        max_requests, slots_per_request, dtype=torch.int32, device=device
    )
    logical = torch.arange(slots_per_request, device=device)
    for request in range(max_requests):
        page_order = torch.randperm(pages, generator=generator, device=device)
        req_to_token[request] = (
            (request % 2) * slots_per_request
            + page_order[logical // block_size] * block_size
            + logical % block_size
        ).to(torch.int32)
    # Ragged tails inside one graph bucket: partial trailing blocks and rows whose
    # live block count falls short of the static column width.
    seq_lens = torch.randint(
        max(block_size + 1, context - 3 * block_size + 1), context + 1, (batch,),
        generator=generator, dtype=torch.int32, device=device,
    )
    seq_lens[0] = context
    slot_ids = torch.randperm(max_requests, generator=generator, device=device)[
        :batch
    ].to(torch.int32)

    return {
        "q": randn(batch, num_q_heads, head_dim),
        "k_cache": randn(max_slots, num_kv_heads, head_dim),
        "req_to_token": req_to_token,
        "slot_ids": slot_ids,
        "seq_lens": seq_lens,
        "sm_scale": head_dim ** -0.5,
        "block_size": block_size,
        "topk": topk,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
    }


def _reference(inputs: dict[str, Any]) -> torch.Tensor:
    q = inputs["q"]
    k_cache = inputs["k_cache"]
    req_to_token = inputs["req_to_token"]
    seq_lens = inputs["seq_lens"]
    slot_ids = inputs["slot_ids"]
    block_size = int(inputs["block_size"])
    init_blocks = int(inputs["init_blocks"])
    local_blocks = int(inputs["local_blocks"])
    scale = float(inputs["sm_scale"]) * _LOG2E

    batch, num_q_heads, _ = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    group = num_q_heads // num_kv_heads
    columns = math.ceil(int(seq_lens.max()) / block_size)
    out = torch.full(
        (num_q_heads, batch, columns), float("-inf"),
        dtype=torch.float32, device=q.device,
    )

    for request in range(batch):
        seq_len = int(seq_lens[request])
        live = math.ceil(seq_len / block_size)
        # Stock wraps both the request row and each slot into range.
        row = int(slot_ids[request]) % max_slots
        physical = (req_to_token[row, :seq_len].long() % max_slots)
        padded = live * block_size
        for head in range(num_q_heads):
            keys = k_cache[physical, head // group].float()
            token = torch.matmul(keys, q[request, head].float()) * scale
            if padded > seq_len:  # partial trailing block
                token = torch.cat([
                    token,
                    token.new_full((padded - seq_len,), float("-inf")),
                ])
            out[head, request, :live] = token.view(live, block_size).amax(dim=-1)
        if init_blocks:
            out[:, request, : min(init_blocks, live)] = 1e30
        if local_blocks:
            out[:, request, max(0, live - local_blocks) : live] = 1e29
    return out


def _invoke(entry, inputs: dict[str, Any], outputs, _prepared) -> None:
    entry(
        inputs["q"],
        inputs["k_cache"],
        inputs["req_to_token"],
        inputs["slot_ids"],
        inputs["seq_lens"],
        outputs[0],
        inputs["sm_scale"],
        inputs["block_size"],
        inputs["topk"],
        inputs["init_blocks"],
        inputs["local_blocks"],
    )


def _columns(inputs: dict[str, Any]) -> int:
    return math.ceil(int(inputs["seq_lens"].max()) / int(inputs["block_size"]))


def build_slot(SlotSpec, Correctness, tolerances, call_abi):
    return SlotSpec(
        name="attention.msa_block_score",
        entry="msa_block_score",
        summary=(
            "MiniMax-M3 decode index scores over paged index-K: "
            "q:(B,Hq,D) k_cache:(max_slots,Hkv,D) req_to_token:(max_reqs,max_kv_len) "
            "slot_ids:(B,) seq_lens:(B,) -> scores:(Hq,B,nblocks) fp32, in log2 units, "
            "block-max of the index QK; forced 1e30 init / 1e29 local blocks; -inf "
            "beyond a row's live block count.  entry(q, k_cache, req_to_token, "
            "slot_ids, seq_lens, out, sm_scale, block_size, topk, init_blocks, "
            "local_blocks).  The validator owns per-head top-k, the head union and "
            "the attend; gated on topk_overlap (the SELECTED set), not score values."
        ),
        kind="block",
        make_inputs=_inputs,
        out_shapes=lambda inputs: [
            (inputs["q"].shape[1], inputs["q"].shape[0], _columns(inputs))
        ],
        invoke_reference=lambda inputs: [_reference(inputs)],
        invoke_entry=_invoke,
        graph_dynamic_inputs=("q", "k_cache", "req_to_token", "slot_ids", "seq_lens"),
        shapes=tuple(
            # M3 decode: 16 index q-heads over one index-k head, D128/block128/topk16.
            # Every shape keeps nblocks > topk, or the selection gate is vacuous.
            {"batch": batch, "context": 9344, "num_q_heads": 16, "num_kv_heads": 1,
             "head_dim": 128, "block_size": 128, "topk": 16,
             "init_blocks": 1, "local_blocks": 2}
            for batch in (1, 8, 32)
        ),
        # A selection output: gate on the top-k block SETS agreeing, not the values.
        correctness=Correctness("topk_overlap", top_k=16, min_overlap=0.875),
        tolerances=tolerances,
        kl_threshold=3e-2,
        call_abi=call_abi,
    )


__all__ = ["build_slot"]

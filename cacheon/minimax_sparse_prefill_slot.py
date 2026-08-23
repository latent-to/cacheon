"""Batched paged score-to-selection contract for MiniMax sparse prefill."""

from __future__ import annotations

import math

import torch

from cacheon.tensor_spec import OutputSpec, TensorSpec


INPUT_NAMES = (
    "q", "index_k_cache", "req_to_token", "slot_ids", "cu_seqlens",
    "seq_lens", "prefix_lens", "max_seqlen_q", "max_seqlen_k",
    "block_size_q", "block_size_k", "topk", "init_blocks", "local_blocks",
    "scale", "cu_seqblocks_q", "max_seqblock_q", "all_seqblock_q",
)


def _inputs(*, q_len: int, prefix_blocks: int, head_dim: int, block_size: int,
            dtype: torch.dtype, device: str, seed: int, batch_size: int = 2,
            num_q_heads: int = 3, topk: int = 8, block_size_q: int | None = None,
            init_blocks: int = 0, local_blocks: int = 1,
            ragged: bool = True, causal_probe: bool = False,
            prefix_len_override: int | None = None) -> dict:
    if min(q_len, head_dim, block_size, batch_size, num_q_heads, topk) < 1:
        raise ValueError("MSA prefill dimensions must be positive")
    block_size_q = block_size if block_size_q is None else int(block_size_q)
    prefix = (prefix_blocks * block_size + 1
              if prefix_len_override is None else int(prefix_len_override))
    if (
        block_size_q < 1 or prefix < 0 or min(init_blocks, local_blocks) < 0
        or init_blocks + local_blocks > topk
    ):
        raise ValueError("invalid MSA block, prefix, or top-k")
    q_lens = [q_len if not ragged else max(1, q_len - b)
              for b in range(batch_size)]
    prefix_lens = [prefix if not ragged else max(0, prefix - b)
                   for b in range(batch_size)]
    seq_lens = [p + n for p, n in zip(prefix_lens, q_lens)]
    max_k = max(seq_lens)
    pages = math.ceil(max_k / block_size)
    slots_per_request = pages * block_size
    generator = torch.Generator(device=device).manual_seed(seed)
    randn = lambda *shape: torch.randn(  # noqa: E731
        *shape, generator=generator, device=device, dtype=torch.float32
    ).to(dtype)
    q = randn(sum(q_lens), num_q_heads, head_dim)
    index_k_cache = randn(batch_size * slots_per_request, 1, head_dim)
    req_to_token = torch.empty(
        batch_size, slots_per_request, dtype=torch.int32, device=device
    )
    logical = torch.arange(slots_per_request, device=device)
    for request in range(batch_size):
        page_order = torch.randperm(pages, generator=generator, device=device)
        req_to_token[request] = (
            request * slots_per_request + page_order[logical // block_size]
            * block_size + logical % block_size
        ).to(torch.int32)
    slot_ids = torch.arange(batch_size - 1, -1, -1, device=device, dtype=torch.int32)
    cu = torch.tensor([0, *q_lens], dtype=torch.int32, device=device).cumsum(0)
    q_blocks = [math.ceil(n / block_size_q) for n in q_lens]
    cu_blocks = torch.tensor([0, *q_blocks], dtype=torch.int32, device=device).cumsum(0)
    if causal_probe:
        for b, (start, length, pre) in enumerate(zip(cu[:-1], q_lens, prefix_lens)):
            sid = int(slot_ids[b])
            for qb in range(math.ceil(length / block_size_q)):
                row = int(start) + qb * block_size_q
                feature = qb % head_dim
                q[row, :, feature] = 1
                future = math.ceil((pre + qb * block_size_q + 1) / block_size) * block_size
                if future < pre + length:
                    index_k_cache[int(req_to_token[sid, future]), 0, feature] = 100
    return {
        "q": q, "index_k_cache": index_k_cache, "req_to_token": req_to_token,
        "slot_ids": slot_ids, "cu_seqlens": cu,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        "max_seqlen_q": max(q_lens), "max_seqlen_k": max_k,
        "block_size_q": block_size_q, "block_size_k": block_size, "topk": topk,
        "init_blocks": init_blocks, "local_blocks": local_blocks,
        "scale": head_dim**-0.5 * 1.4426950409, "cu_seqblocks_q": cu_blocks,
        "max_seqblock_q": max(q_blocks), "all_seqblock_q": sum(q_blocks),
    }


def _reference(inputs: dict) -> torch.Tensor:
    q, cache = inputs["q"], inputs["index_k_cache"]
    bq, bk, topk = (int(inputs[name]) for name in
                     ("block_size_q", "block_size_k", "topk"))
    output = torch.full(
        (q.shape[1], int(inputs["all_seqblock_q"]), topk), -1,
        dtype=torch.int32, device=q.device,
    )
    for b in range(inputs["seq_lens"].numel()):
        qs, block_start = int(inputs["cu_seqlens"][b]), int(inputs["cu_seqblocks_q"][b])
        q_len = int(inputs["cu_seqlens"][b + 1]) - qs
        seq_len, prefix = int(inputs["seq_lens"][b]), int(inputs["prefix_lens"][b])
        sid = int(inputs["slot_ids"][b])
        keys = cache[inputs["req_to_token"][sid, :seq_len].long(), 0].float()
        for qb in range(math.ceil(q_len / bq)):
            visible = min(seq_len, prefix + qb * bq + 1)
            blocks = math.ceil(visible / bk)
            scores = torch.stack([
                (q[qs + qb * bq, head].float()
                 @ keys[k:min(k + bk, visible)].T).amax()
                for head in range(q.shape[1]) for k in range(0, visible, bk)
            ]).view(q.shape[1], blocks) * float(inputs["scale"])
            scores[:, :min(blocks, int(inputs["init_blocks"]))] = torch.inf
            scores[:, max(0, blocks - int(inputs["local_blocks"])):] = torch.finfo(scores.dtype).max
            width = min(topk, blocks)
            output[:, block_start + qb, :width] = scores.topk(width, dim=-1).indices.int()
    return output


def _invoke(entry, inputs: dict, outputs, _prepared) -> None:
    entry(*(inputs[name] for name in INPUT_NAMES), outputs[0])


def build_slot(SlotSpec, Correctness, tolerances, call_abi):
    shape = lambda i: (i["q"].shape[1], int(i["all_seqblock_q"]), int(i["topk"]))  # noqa: E731
    return SlotSpec(
        name="attention.msa_prefill_block_score", entry="msa_prefill_block_score",
        summary="Batched paged MiniMax prefill index score-to-selection",
        kind="block", make_inputs=_inputs, out_shapes=lambda i: [shape(i)],
        output_spec=lambda i: OutputSpec((TensorSpec(
            shape(i), dtype=torch.int32, alignment_bytes=4, name="topk_idx"
        ),)),
        invoke_reference=lambda i: [_reference(i)], invoke_entry=_invoke,
        graph_dynamic_inputs=("q", "index_k_cache"),
        shapes=(
            {"q_len": 7, "prefix_blocks": 5, "head_dim": 16, "block_size": 4,
             "batch_size": 2, "num_q_heads": 3, "topk": 5},
            {"q_len": 3, "prefix_blocks": 1, "head_dim": 8, "block_size": 4,
             "batch_size": 3, "num_q_heads": 2, "topk": 6},
            {"q_len": 17, "prefix_blocks": 7, "head_dim": 32, "block_size": 8,
             "block_size_q": 4, "batch_size": 2, "num_q_heads": 4, "topk": 6,
             "causal_probe": True},
            {"q_len": 8, "prefix_blocks": 20, "head_dim": 128, "block_size": 128,
             "batch_size": 1, "num_q_heads": 1, "topk": 16, "ragged": False},
        ),
        correctness=Correctness("topk_overlap", top_k=8, min_overlap=0.9),
        tolerances=tolerances, kl_threshold=3e-2, call_abi=call_abi,
    )


__all__ = ["INPUT_NAMES", "build_slot"]

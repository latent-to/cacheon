"""Masked index selection and physical-page translation for sparse attention.

Scores may be ragged slices with a row offset. Page tables and row mappings
remain explicit tensors so the same kernel can handle prefill and decode.
"""

from __future__ import annotations

import torch

from cacheon.capabilities import CallDescriptor
from cacheon.tensor_spec import OutputSpec, TensorSpec

SLOT = "attention.indexer_topk"
DYNAMIC_INPUTS = ("scores", "lengths", "row_starts", "page_table", "row_to_batch", "page_offsets")


def output_spec(inputs: dict) -> OutputSpec:
    """The engine consumes physical token indices with trailing -1 padding."""
    return OutputSpec((TensorSpec(
        (inputs["scores"].shape[0], inputs["top_k"]), dtype=torch.int32, name="indices",
    ),))


def call_descriptor(inputs: dict, **context) -> CallDescriptor:
    """Describe both score storage and the page granularity visible to a miner."""
    scores = inputs["scores"]
    return CallDescriptor(
        dtype=str(scores.dtype).removeprefix("torch."), num_tokens=scores.shape[0],
        kv_len=scores.shape[1], last_dim=scores.shape[1], top_k=inputs["top_k"],
        page_size=inputs["page_size"], layout="paged_index_selection", quant="dense",
        **context,
    )


def make_inputs(*, num_tokens: int, kv_len: int, top_k: int, page_size: int,
                dtype: torch.dtype, device: str, seed: int, num_batches: int = 2,
                input_dtype: str | None = None) -> dict:
    """Scramble physical pages and vary causal windows, history and query ownership."""
    g = torch.Generator(device=device).manual_seed(seed)
    pages = (kv_len + page_size - 1) // page_size + 1
    scores = torch.randn(num_tokens, kv_len, device=device, generator=g).to(
        getattr(torch, input_dtype) if input_dtype else dtype)
    table = torch.randperm(num_batches * pages, generator=g, device=device).reshape(num_batches, pages)
    row_to_batch = (torch.arange(num_tokens, device=device) + seed) % num_batches
    starts = (torch.arange(num_tokens, device=device) * 3 + seed) % max(1, kv_len // 4)
    lengths = ((torch.arange(num_tokens, device=device) * 7 + seed) % max(1, kv_len // 2)) + 1
    lengths[0] = 0
    if num_tokens > 1:
        lengths[1] = min(kv_len // 2, top_k + 3)
    return dict(scores=scores, lengths=lengths.to(torch.int32), row_starts=starts.to(torch.int32),
                page_table=table.to(torch.int32), row_to_batch=row_to_batch.to(torch.int32),
                page_offsets=((starts + seed) % page_size).to(torch.int32),
                page_size=page_size, top_k=top_k)


def reference(inputs: dict) -> list[torch.Tensor]:
    """Select in FP64 and translate only valid logical positions into cache slots."""
    scores = inputs["scores"]
    result = torch.full((scores.shape[0], inputs["top_k"]), -1, device=scores.device, dtype=torch.int32)
    starts, lengths, batches, offsets = (
        inputs[k].cpu().tolist() for k in ("row_starts", "lengths", "row_to_batch", "page_offsets")
    )
    for row, (start, length, batch, offset) in enumerate(zip(starts, lengths, batches, offsets)):
        end = min(scores.shape[1], start + max(length, 0))
        if start >= end:
            continue
        values = scores[row, start:end].double()
        indices = torch.argsort(values, descending=True, stable=True)[:inputs["top_k"]]
        indices = indices[values[indices] != -torch.inf]
        logical = indices + offset
        physical = inputs["page_table"][batch, logical // inputs["page_size"]].long()
        result[row, :indices.numel()] = (physical * inputs["page_size"] + logical % inputs["page_size"]).to(torch.int32)
    return [result]


def invoke_entry(entry, inputs: dict, outputs: list, prepared=None) -> None:
    """Keep engine metadata outside the candidate's tensor-only call."""
    entry(*(inputs[k] for k in DYNAMIC_INPUTS), outputs[0], inputs["page_size"], inputs["top_k"])


def slot_spec():
    """Expose selection independently of score GEMMs and sparse attention."""
    from cacheon.slots import Correctness, SlotSpec, Tolerance

    return SlotSpec(
        name=SLOT, entry="indexer_topk", kind="block",
        summary="Masked top-k over score rows and logical-to-physical page translation into int32 output.",
        make_inputs=make_inputs, out_shapes=lambda i: [(i["scores"].shape[0], i["top_k"])],
        output_spec=output_spec, invoke_reference=reference, invoke_entry=invoke_entry,
        graph_dynamic_inputs=DYNAMIC_INPUTS,
        shapes=(dict(num_tokens=7, kv_len=128, top_k=17, page_size=32),
                dict(num_tokens=11, kv_len=256, top_k=23, page_size=64)),
        correctness=Correctness("topk_overlap", min_overlap=0.99),
        tolerances={t: Tolerance(0, 0) for t in (torch.float16, torch.bfloat16, torch.float32)},
    )

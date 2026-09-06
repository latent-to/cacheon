"""Weighted-ReLU MQA scores over explicit key pages and causal row windows.

The query's quantization scale is already folded into weights by SGLang.
Each key has one FP32 scale. Top-k and cache writes remain outside this ABI.
"""

from __future__ import annotations

import torch

from cacheon.capabilities import CallDescriptor
from cacheon.tensor_spec import OutputSpec, TensorSpec

SLOT = "attention.indexer_scores"
DYNAMIC_INPUTS = ("q", "key_pages", "key_scales", "weights", "starts", "ends",
                  "page_table", "row_to_batch")


def output_spec(inputs: dict) -> OutputSpec:
    """Require FP32 scores, including zero in every masked output cell."""
    return OutputSpec((TensorSpec((inputs["q"].shape[0], inputs["kv_len"]),
                                  dtype=torch.float32, name="scores"),))


def call_descriptor(inputs: dict, **context) -> CallDescriptor:
    """Expose score geometry without passing serving objects to candidate code."""
    q = inputs["q"]
    return CallDescriptor(dtype=str(q.dtype).removeprefix("torch."),
                          num_tokens=q.shape[0], num_q_heads=q.shape[1],
                          num_kv_heads=1, head_dim=q.shape[2], last_dim=q.shape[2],
                          kv_len=inputs["kv_len"], page_size=inputs["key_pages"].shape[1],
                          layout="paged_indexer_scores", quant="fp8_e4m3", **context)


def make_inputs(*, num_tokens: int, num_heads: int, head_dim: int, kv_len: int,
                page_size: int, dtype: torch.dtype, device: str, seed: int,
                num_batches: int = 2) -> dict:
    """Vary keys, signed head weights, windows and scrambled physical pages."""
    g = torch.Generator(device=device).manual_seed(seed)
    pages = (kv_len + page_size - 1) // page_size
    physical_pages = num_batches * pages + 1
    rand = lambda *shape: torch.randn(*shape, generator=g, device=device)
    q = rand(num_tokens, num_heads, head_dim).to(torch.float8_e4m3fn)
    keys = rand(physical_pages, page_size, head_dim).to(torch.float8_e4m3fn)
    scales = rand(physical_pages, page_size).abs() + 0.125
    rows = torch.arange(num_tokens, device=device)
    starts = (rows * 3 + seed) % max(1, kv_len // 4)
    ends = starts + (rows * 7 + seed) % max(1, kv_len // 2) + 1
    ends[0] = starts[0]
    return dict(q=q, key_pages=keys, key_scales=scales, weights=rand(num_tokens, num_heads),
                starts=starts.int(), ends=ends.int(),
                page_table=torch.randperm(physical_pages, device=device, generator=g)[:num_batches * pages].reshape(num_batches, pages).int(),
                row_to_batch=((rows + seed) % num_batches).int(), kv_len=kv_len)


def reference(inputs: dict) -> list[torch.Tensor]:
    """Apply declared math in FP64 per row, independently of the example kernel."""
    q, keys, scales, weights, starts, ends, table, batches = (inputs[k] for k in DYNAMIC_INPUTS)
    result = torch.zeros(q.shape[0], inputs["kv_len"], device=q.device, dtype=torch.float32)
    page_size = keys.shape[1]
    metadata = zip(starts.cpu().tolist(), ends.cpu().tolist(), batches.cpu().tolist())
    for row, (start, end, batch) in enumerate(metadata):
        columns = torch.arange(start, end, device=q.device)
        physical = table[batch, columns // page_size].long()
        # Byte gathers also support FP8 source storage on CPU.
        k = keys.view(torch.uint8)[physical, columns % page_size].view(keys.dtype).double()
        k *= scales[physical, columns % page_size].double().unsqueeze(-1)
        per_head = q[row].double() @ k.T
        result[row, start:end] = (per_head.clamp_min(0) * weights[row].double()[:, None]).sum(0).float()
    return [result]


def invoke_entry(entry, inputs: dict, outputs: list, prepared=None) -> None:
    """Fill validator-owned scores using only the declared tensor inputs."""
    entry(*(inputs[k] for k in DYNAMIC_INPUTS), outputs[0])


def slot_spec():
    """Register the score calculation separately from selection and attention."""
    from cacheon.slots import Correctness, SlotSpec, Tolerance

    return SlotSpec(name=SLOT, entry="indexer_scores", kind="block",
                    summary="FP8 MQA dot products, per-key scales, ReLU and weighted head reduction; masked FP32 scores are zero.",
                    make_inputs=make_inputs, output_spec=output_spec,
                    out_shapes=lambda i: [(i["q"].shape[0], i["kv_len"])],
                    invoke_reference=reference, invoke_entry=invoke_entry,
                    graph_dynamic_inputs=DYNAMIC_INPUTS,
                    shapes=(dict(num_tokens=5, num_heads=8, head_dim=128, kv_len=97, page_size=32),
                            dict(num_tokens=7, num_heads=32, head_dim=128, kv_len=193, page_size=64)),
                    correctness=Correctness("allclose"),
                    tolerances={t: Tolerance(1e-3, 1e-3) for t in (torch.float8_e4m3fn, torch.float32)})

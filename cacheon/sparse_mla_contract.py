"""Independent math and tensor ABI for selected-token sparse MLA.

The cache stores latent values followed by positional key components. Indices
name physical tokens, not page IDs or logical sequence positions. The producer
owns causal/index selection and cache updates before this boundary.
"""

from __future__ import annotations

import math

import torch

from cacheon.capabilities import CallDescriptor
from cacheon.tensor_spec import OutputSpec, TensorSpec

SLOT = "attention.sparse_mla"
DYNAMIC_INPUTS = ("q", "kv_cache", "indices", "seq_lens")
_INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn)


def output_spec(inputs: dict) -> OutputSpec:
    """Keep the absorbed latent output BF16, including for FP8 queries."""
    q = inputs["q"]
    return OutputSpec((TensorSpec(
        (q.shape[0], q.shape[1], inputs["value_dim"]),
        dtype=torch.bfloat16, name="sparse_mla",
    ),))


def call_descriptor(
    inputs: dict, *, architecture: str | None, graph_mode: str,
    tp_size: int | None, world_size: int | None,
) -> CallDescriptor:
    """Use identical observable geometry in verification and live selection."""
    q, cache, indices = (inputs[k] for k in ("q", "kv_cache", "indices"))
    return CallDescriptor(
        architecture=architecture, dtype=str(q.dtype).removeprefix("torch."),
        graph_mode=graph_mode, num_tokens=q.shape[0], num_q_heads=q.shape[1],
        num_kv_heads=1, head_dim=q.shape[2], last_dim=q.shape[2],
        output_dim=inputs["value_dim"], page_size=cache.shape[1],
        top_k=indices.shape[1], q_len=1, layout="paged_latent_rope",
        quant="fp8_e4m3" if q.dtype == torch.float8_e4m3fn else "dense",
        tp_size=tp_size, world_size=world_size,
    )


def make_inputs(
    *, num_tokens: int, num_heads: int, value_dim: int, rope_dim: int,
    num_pages: int, page_size: int, top_k: int, dtype: torch.dtype,
    device: str, seed: int, query_chunk: int = 1,
    input_dtype: str | None = None,
) -> dict:
    """Generate scrambled physical cache rows with per-query causal histories.

    Query chunks share one logical history. Selection never includes a future
    logical token; physical addresses deliberately have no causal ordering.
    """
    if min(num_tokens, num_heads, value_dim, rope_dim, num_pages,
           page_size, top_k, query_chunk) < 1:
        raise ValueError("sparse MLA profile dimensions must be positive")
    storage_dtype = getattr(torch, input_dtype) if input_dtype else dtype
    if storage_dtype not in _INPUT_DTYPES:
        raise ValueError("unsupported sparse MLA input dtype")
    capacity = num_pages * page_size
    head_dim = value_dim + rope_dim
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        num_tokens, num_heads, head_dim, generator=generator, device=device,
    ).to(storage_dtype)
    cache = torch.randn(
        num_pages, page_size, head_dim, generator=generator, device=device,
    ).to(storage_dtype)
    # Construct selection metadata on CPU; generation is outside candidate timing
    # and must not synchronize once per query on a GPU.
    metadata_rng = torch.Generator().manual_seed(seed + 97)
    physical = torch.randperm(capacity, generator=metadata_rng)
    priority = torch.randperm(capacity, generator=metadata_rng)
    chunk = min(query_chunk, capacity)
    history = (capacity - chunk) // 2
    shift = seed % max(1, capacity - history - chunk + 1)
    lengths = history + shift + torch.arange(num_tokens) % chunk + 1
    # A separate short request makes length changes observable during replay.
    lengths[-1] = 1 + seed % min(top_k, capacity)
    indices = torch.full((num_tokens, top_k), -1, dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        selected = priority[priority < length][:top_k]
        count = selected.numel()
        indices[row, :count] = physical[selected].to(torch.int32)
        if count > 2:
            indices[row, count // 2] = -1
        # Valid-looking tail addresses must be ignored by the length bound.
        tail = torch.arange(count, top_k, 2)
        indices[row, tail] = physical[(tail + length) % capacity].to(torch.int32)
    return {
        "q": q, "kv_cache": cache, "indices": indices.to(device),
        "seq_lens": lengths.to(device=device, dtype=torch.int32),
        "value_dim": value_dim, "qk_scale": 1 / math.sqrt(head_dim),
        "value_scale": 1.0,
    }


def reference(inputs: dict) -> list[torch.Tensor]:
    """Evaluate declared sparse softmax math in FP32, one query at a time.

    Only selected cache rows are dequantized. Temporary memory scales with
    top-k times head dimension, never with queries times the complete KV pool.
    """
    q, cache, indices, lengths = (inputs[k] for k in DYNAMIC_INPUTS)
    value_dim = inputs["value_dim"]
    result = torch.zeros(
        q.shape[0], q.shape[1], value_dim, device=q.device, dtype=torch.float32,
    )
    capacity = cache.shape[0] * cache.shape[1]
    # Byte gathers keep FP8 verification supported on CPU as well as GPU.
    gather_cache = cache.view(torch.uint8) if cache.dtype == torch.float8_e4m3fn else cache
    for row, length in enumerate(lengths.cpu().tolist()):
        ids = indices[row, :max(0, min(length, indices.shape[1]))].long()
        if bool(((ids < -1) | (ids >= capacity)).any()):
            raise ValueError("sparse MLA selected index outside cache")
        ids = ids[ids >= 0]
        if ids.numel() == 0:
            continue
        selected = gather_cache[ids // cache.shape[1], ids % cache.shape[1]]
        if cache.dtype == torch.float8_e4m3fn:
            selected = selected.view(cache.dtype)
        selected = selected.float()
        scores = (q[row].float() @ selected.T) * inputs["qk_scale"]
        weights = torch.exp(scores - torch.logsumexp(scores, dim=-1, keepdim=True))
        result[row] = (weights @ selected[:, :value_dim]) * inputs["value_scale"]
    return [result.to(torch.bfloat16)]


def invoke_entry(entry, inputs: dict, outputs: list, prepared=None) -> None:
    """Pass only declared tensors/scalars and validator-owned output."""
    entry(
        inputs["q"], inputs["kv_cache"], inputs["indices"], inputs["seq_lens"],
        outputs[0], inputs["value_dim"], inputs["qk_scale"], inputs["value_scale"],
    )


def slot_spec():
    """Register the shared core through the existing catalog."""
    from cacheon.slots import Correctness, SlotSpec, Tolerance

    return SlotSpec(
        name=SLOT, entry="sparse_mla", kind="block",
        summary=(
            "Selected-token sparse MLA: q:(T,H,D), paged latent+RoPE KV:(P,S,D), "
            "physical indices:(T,K), seq_lens:(T) -> BF16 out:(T,H,V). "
            "Owns sparse QK, softmax and latent-value combine; selection, RoPE, "
            "quantization and cache updates stay outside."
        ),
        make_inputs=make_inputs,
        out_shapes=lambda i: [(i["q"].shape[0], i["q"].shape[1], i["value_dim"])],
        output_spec=output_spec, invoke_reference=reference, invoke_entry=invoke_entry,
        graph_dynamic_inputs=DYNAMIC_INPUTS,
        shapes=(
            dict(num_tokens=4, num_heads=2, value_dim=8, rope_dim=4,
                 num_pages=3, page_size=32, top_k=17, query_chunk=4),
            dict(num_tokens=3, num_heads=4, value_dim=12, rope_dim=8,
                 num_pages=2, page_size=64, top_k=23),
        ),
        correctness=Correctness("matched_ratio", min_ratio=0.99),
        tolerances={dtype: Tolerance(2e-2, 2e-2) for dtype in _INPUT_DTYPES},
    )

"""Merged sparse-indexer scoring and physical-index selection for the attention family."""

from __future__ import annotations

import torch

from cacheon.capabilities import CallDescriptor
from cacheon.tensor_spec import OutputSpec, TensorSpec

SLOT = "attention.indexer_select"
DYNAMIC_INPUTS = ("q", "key_pages", "key_scales", "weights", "page_table",
                  "row_to_batch", "lengths", "page_offsets", "positions", "cos_sin_cache")


def output_spec(inputs: dict) -> OutputSpec:
    """Return physical cache indices directly, with unused positions filled by -1."""
    return OutputSpec((TensorSpec((inputs["q"].shape[0], inputs["top_k"]),
                                  dtype=torch.int32, name="indices"),))


def call_descriptor(inputs: dict, **context) -> CallDescriptor:
    """Describe the combined computation without exposing engine metadata objects."""
    q, keys = inputs["q"], inputs["key_pages"]
    return CallDescriptor(dtype=str(q.dtype).removeprefix("torch."), num_tokens=q.shape[0],
                          num_q_heads=q.shape[1], head_dim=q.shape[-1], last_dim=q.shape[-1],
                          page_size=keys.shape[1], kv_len=inputs["page_table"].shape[1] * keys.shape[1],
                          top_k=inputs["top_k"], layout="paged_indexer_select", quant="fp8_e4m3", **context)


def make_inputs(*, num_tokens, num_heads, head_dim, kv_len, page_size, top_k,
                dtype, device, seed, num_batches=2, num_init_tokens=1, num_local_tokens=1) -> dict:
    """Vary cache history, signed gates, empty rows and offsets across shuffled pages."""
    g = torch.Generator(device=device).manual_seed(seed)
    pages = (kv_len + page_size - 1) // page_size + 1
    rand = lambda *s: torch.randn(*s, generator=g, device=device)
    rows = torch.arange(num_tokens, device=device)
    lengths = (rows * 7 + seed) % kv_len + 1
    lengths[0] = 0
    angles = rand(137, head_dim // 4)
    raw = dtype != torch.float8_e4m3fn
    return dict(q=rand(num_tokens, num_heads, head_dim).to(dtype),
                key_pages=rand(num_batches * pages, page_size, head_dim).to(torch.float8_e4m3fn),
                key_scales=rand(num_batches * pages, page_size).abs() + .125,
                weights=rand(num_tokens, num_heads).to(dtype if raw else torch.float32),
                positions=torch.randint(0, 137, (num_tokens,), generator=g, device=device) if raw else None,
                cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1) if raw else None,
                q_scale_gate=.03125 if raw else 1.0,
                page_table=torch.randperm(num_batches * pages, generator=g, device=device).reshape(num_batches, pages).int(),
                row_to_batch=((rows + seed) % num_batches).int(), lengths=lengths.int(),
                page_offsets=((rows * 3 + seed) % page_size).int(), top_k=top_k,
                num_init_tokens=num_init_tokens, num_local_tokens=num_local_tokens)


def reference(inputs: dict) -> list[torch.Tensor]:
    """Compute one FP64 score row at a time, then select and translate its indices."""
    q, keys, scales, weights, table, batches, lengths, offsets = (inputs[n] for n in DYNAMIC_INPUTS[:8])
    if q.dtype != torch.float8_e4m3fn:
        cache = inputs["cos_sin_cache"]
        dim = cache.shape[-1]
        cosine, sine = cache[inputs["positions"].long()].double().unsqueeze(1).chunk(2, -1)
        a, b = q[..., :dim:2].double(), q[..., 1:dim:2].double()
        rotated = torch.stack((a * cosine - b * sine, b * cosine + a * sine), -1).flatten(-2)
        value = torch.cat((rotated, q[..., dim:].double()), -1)
        scale = (value.abs().amax(-1).clamp_min(1e-4) / 448).float()
        q = (value / scale.double().unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        weights = (weights.float() * inputs["q_scale_gate"]) * scale
    out = torch.full((q.shape[0], inputs["top_k"]), -1, dtype=torch.int32, device=q.device)
    size = keys.shape[1]
    for row, (batch, length, offset) in enumerate(zip(batches.tolist(), lengths.tolist(), offsets.tolist())):
        if length == 0:
            continue
        logical = torch.arange(length, device=q.device) + offset
        pages = table[batch, logical // size].long()
        key = keys.view(torch.uint8)[pages, logical % size].view(keys.dtype).double()
        key *= scales[pages, logical % size].double().unsqueeze(-1)
        score = ((q[row].double() @ key.T).clamp_min(0) * weights[row].double()[:, None]).sum(0)
        score[:inputs["num_init_tokens"]] = torch.inf
        if inputs["num_local_tokens"]:
            score[max(0, length - inputs["num_local_tokens"]):] = torch.inf
        chosen = torch.argsort(score, descending=True, stable=True)[:inputs["top_k"]]
        out[row, :chosen.numel()] = (pages[chosen] * size + logical[chosen] % size).int()
    return [out]


def invoke_entry(entry, inputs: dict, outputs: list, prepared=None) -> None:
    """Allow a candidate to fuse scoring and selection without materializing logits."""
    entry(*(inputs[n] for n in DYNAMIC_INPUTS), inputs["q_scale_gate"], inputs["num_init_tokens"],
          inputs["num_local_tokens"], inputs["top_k"], outputs[0])


def slot_spec():
    """Declare one internal member of the atomic attention-family contribution."""
    from cacheon.slots import Correctness, SlotSpec, Tolerance

    return SlotSpec(name=SLOT, entry="indexer_select", kind="block",
                    summary="Optional leading RoPE, query FP8 quantization and head gates; weighted-ReLU scores, forced tokens and physical-index selection.",
                    make_inputs=make_inputs, output_spec=output_spec,
                    out_shapes=lambda i: [(i["q"].shape[0], i["top_k"])],
                    invoke_reference=reference, invoke_entry=invoke_entry, graph_dynamic_inputs=DYNAMIC_INPUTS,
                    shapes=(dict(num_tokens=5, num_heads=8, head_dim=128, kv_len=97, page_size=32, top_k=7),
                            dict(num_tokens=7, num_heads=32, head_dim=128, kv_len=193, page_size=64, top_k=11)),
                    correctness=Correctness("topk_overlap", min_overlap=.99),
                    tolerances={t: Tolerance(0, 0) for t in (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32)})

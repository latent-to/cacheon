"""Post-attention projection, residual, norm and optional NVFP4 tensor contract.

Each attention-DP rank owns different input rows and the same projection weight.
The result contains all ranks' normalized rows, this rank's updated residual,
and optional block-scaled FP4 bytes. Expert execution remains outside this slot.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from cacheon.tensor_spec import OutputSpec, TensorSpec

SLOT = "collective.dp_output_projection_norm"


def make_inputs(*, num_tokens, input_dim, hidden, dtype, device, seed,
                rank=0, world_size=1, quantize=True):
    """Replicate parameters while making activation rows different on every rank."""
    shared = torch.Generator(device=device).manual_seed(seed)
    local = torch.Generator(device=device).manual_seed(seed + 1_000_003 * (rank + 1))
    weight = (torch.randn(hidden, input_dim, generator=shared, device=device)
              / input_dim**0.5).to(dtype)
    gamma = (1 + torch.randn(hidden, generator=shared, device=device) / 8).to(dtype)
    x = torch.randn(num_tokens, input_dim, generator=local, device=device).to(dtype)
    residual = torch.randn(num_tokens, hidden, generator=local, device=device).to(dtype)
    scale = (torch.full((1,), 32., dtype=torch.float32, device=device) if quantize
             else torch.empty(0, dtype=torch.float32, device=device))
    return dict(x=x, residual=residual, weight=weight, gamma=gamma,
                epsilon=1e-5, quant_scale=scale, world_size=world_size)


def output_spec(inputs):
    """Keep replicated normalized rows separate from the rank-local residual."""
    rows, hidden = inputs["residual"].shape
    total = rows * int(inputs["world_size"])
    quantized = inputs["quant_scale"].numel() != 0
    return OutputSpec((
        TensorSpec((total, hidden), name="normalized"),
        TensorSpec((rows, hidden), name="local_residual"),
        TensorSpec((total, hidden // 2) if quantized else (0,), dtype=torch.uint8, name="fp4"),
        TensorSpec((total, hidden // 16) if quantized else (0,), dtype=torch.uint8, name="scales"),
    ))


def quantize_reference(hidden, global_scale):
    """Round linear16-value blocks to E4M3 scales and nearest-even E2M1 values."""
    values = hidden.double().reshape(-1, 16)
    factor = global_scale.double()
    scales = (values.abs().amax(-1) * factor / 6).clamp(max=448).to(torch.float8_e4m3fn)
    multiplier = torch.where(scales.double() == 0, 0., factor / scales.double())
    magnitudes = (values * multiplier[:, None]).abs()
    levels = values.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    distance = (magnitudes[..., None] - levels).abs()
    tied = distance == distance.amin(-1, keepdim=True)
    indices = torch.arange(8, device=hidden.device)
    # An even code wins a midpoint tie, independently of the device conversion.
    priority = indices + (indices % 2) * 8
    codes = torch.where(tied, priority, 32).argmin(-1).to(torch.uint8)
    codes |= torch.signbit(values).to(torch.uint8) * 8
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.reshape(hidden.shape[0], -1), scales.view(torch.uint8).reshape(hidden.shape[0], -1)


def reference(inputs, group, rank, world_size):
    """Derive the full matrix product in FP64, retaining the model's BF16 rounds."""
    x, residual = inputs["x"], inputs["residual"]
    local_projection = (x.double() @ inputs["weight"].double().T).to(x.dtype)
    local_updated = (local_projection.double() + residual.double()).to(x.dtype)
    values = local_updated.double()
    local_normalized = (values * torch.rsqrt(values.square().mean(-1, keepdim=True)
                                            + inputs["epsilon"]) * inputs["gamma"].double()).to(x.dtype)
    if world_size == 1:
        normalized = local_normalized
    else:
        gathered = [torch.empty_like(local_normalized) for _ in range(world_size)]
        dist.all_gather(gathered, local_normalized, group=group)
        normalized = torch.cat(gathered)
    if inputs["quant_scale"].numel():
        packed, scales = quantize_reference(normalized, inputs["quant_scale"])
    else:
        packed = scales = torch.empty(0, dtype=torch.uint8, device=x.device)
    return [normalized, local_updated, packed, scales]


def slot_spec():
    """Expose the same prepared collective ABI to verification and the live adapter."""
    from cacheon.slots import Correctness, SlotSpec, Tolerance

    def invoke(entry, inputs, outputs, group, prepared):
        entry(inputs["x"], inputs["residual"], prepared, *outputs, group)

    return SlotSpec(
        name=SLOT, entry="project_gather_norm", prepare="prepare", kind="collective",
        summary="Replicated-weight attention-DP projection, BF16 residual-add, RMSNorm, row gather and optional static-scale NVFP4 preparation.",
        make_inputs=make_inputs,
        out_shapes=lambda i: [x.shape for x in output_spec(i).outputs], output_spec=output_spec,
        invoke_reference=lambda i: reference(i, None, 0, 1),
        invoke_prepare=lambda fn, i: fn(i["weight"], i["gamma"], i["epsilon"], i["quant_scale"]),
        invoke_entry=lambda fn, i, out, prepared: invoke(fn, i, out, i.get("__group__"), prepared),
        invoke_collective=invoke, collective_reference=reference,
        graph_dynamic_inputs=("x", "residual"),
        shapes=(
            dict(num_tokens=1, input_dim=16384, hidden=6144, quantize=False),
            dict(num_tokens=8, input_dim=16384, hidden=6144, quantize=True),
            dict(num_tokens=32, input_dim=16384, hidden=6144, quantize=True),
        ),
        correctness=Correctness("matched_ratio", min_ratio=0.99),
        tolerances={torch.bfloat16: Tolerance(0.02, 0.02)},
    )

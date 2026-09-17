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
_ATOL = _RTOL = 0.02  # registered BF16 row tolerance; NVFP4 bytes are graded inside the same band
_LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


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


def _codes(magnitudes):
    distance = (magnitudes[..., None] - magnitudes.new_tensor(_LEVELS)).abs()
    tied = distance == distance.amin(-1, keepdim=True)
    indices = torch.arange(8, device=magnitudes.device)
    # An even code wins a midpoint tie, independently of the device conversion.
    return torch.where(tied, indices + (indices % 2) * 8, 32).argmin(-1)


def _block_scales(amax, factor):
    return (amax * factor / 6).clamp(max=448).to(torch.float8_e4m3fn)


def quantize_reference(hidden, global_scale):
    """Round linear16-value blocks to E4M3 scales and nearest-even E2M1 values."""
    values = hidden.double().reshape(-1, 16)
    factor = global_scale.double()
    scales = _block_scales(values.abs().amax(-1), factor)
    multiplier = torch.where(scales.double() == 0, 0., factor / scales.double())
    codes = _codes((values * multiplier[:, None]).abs()).to(torch.uint8)
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


def graded_reference(inputs, outputs, expected):
    """Accept NVFP4 bytes that quantize any row inside the registered BF16 tolerance.

    ``reference`` quantizes its own BF16 rows, so a kernel whose rows differ by one BF16
    step, or that quantizes before rounding to BF16, was graded on byte equality with
    arithmetic it never performed: UID 215's crowned bundle audited at 0.9862-0.9886 against
    the 0.985 bar in six runs (2026-09-16). An accepted byte is graded as itself and any
    other byte as the reference byte, so the registered comparison is unchanged and a wrong
    scale, wrong code or wrong row still fails.
    """
    if not inputs["quant_scale"].numel():
        return list(expected)
    rows, factor = expected[0].double().reshape(-1, 16), inputs["quant_scale"].double()
    band = _ATOL + _RTOL * rows.abs()
    packed, scales = outputs[2].reshape(-1, 8), outputs[3].reshape(-1)
    lowest = _block_scales((rows.abs() - band).clamp(min=0).amax(-1), factor).view(torch.uint8)
    highest = _block_scales((rows.abs() + band).amax(-1), factor).view(torch.uint8)
    scale_ok = (lowest <= scales) & (scales <= highest)  # non-negative E4M3 bytes are ordered
    own = scales.view(torch.float8_e4m3fn).double()
    multiplier = torch.where(scale_ok & (own != 0), factor / own, 0.)
    levels = rows.new_tensor(_LEVELS)

    def level(values):  # under the block scale the bundle itself returned
        scaled = values * multiplier[:, None]
        return torch.copysign(levels[_codes(scaled.abs())], scaled)

    nibbles = torch.stack((packed & 15, packed >> 4), -1).reshape(-1, 16)
    returned = torch.where(nibbles > 7, -1., 1.) * levels[(nibbles & 7).long()]
    nibble_ok = (level(rows - band) <= returned) & (returned <= level(rows + band))
    byte_ok = nibble_ok[:, 0::2] & nibble_ok[:, 1::2] & scale_ok[:, None]
    return [expected[0], expected[1],
            torch.where(byte_ok.reshape(outputs[2].shape), outputs[2], expected[2]),
            torch.where(scale_ok.reshape(outputs[3].shape), outputs[3], expected[3])]


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
        invoke_collective=invoke, collective_reference=reference, graded_reference=graded_reference,
        graph_dynamic_inputs=("x", "residual"),
        shapes=(
            dict(num_tokens=1, input_dim=16384, hidden=6144, quantize=False),
            dict(num_tokens=8, input_dim=16384, hidden=6144, quantize=True),
            dict(num_tokens=32, input_dim=16384, hidden=6144, quantize=True),
        ),
        correctness=Correctness("matched_ratio", min_ratio=0.99),
        tolerances={torch.bfloat16: Tolerance(_ATOL, _RTOL)},
    )

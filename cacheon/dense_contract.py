"""Unquantized GEMM family with canonical weights and validator-owned outputs."""

from __future__ import annotations

import torch
from cacheon.capabilities import CallDescriptor
from cacheon.tensor_spec import OutputSpec, TensorSpec


def make_dense_inputs(
    *,
    num_tokens: int,
    input_dim: int,
    output_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    parallel_role: str = "replicated",
    local_tp_size: int = 1,
    batch_size: int | None = None,
    output_dtype: str | None = None,
) -> dict[str, object]:
    """Add head-batched and FP32-output cases without changing ordinary profiles."""
    generator = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
    if batch_size is None:
        x, weight = rand(num_tokens, input_dim), rand(output_dim, input_dim)
    else:
        x = rand(num_tokens, batch_size, input_dim).transpose(0, 1)
        weight = rand(batch_size, input_dim, output_dim).transpose(1, 2)
    return {
        "x": x,
        "weight": weight,
        "parallel_role": parallel_role,
        "local_tp_size": local_tp_size,
        "output_dtype": output_dtype,
    }


def dense_reference(inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Accumulate each declared matrix product in FP64 before output rounding."""
    x, weight = inputs["x"], inputs["weight"]
    result = (x.double() @ weight.double().T if x.ndim == 2 else
              torch.stack([a.double() @ w.double().T for a, w in zip(x, weight)]))
    return [result.to(output_spec(inputs).outputs[0].dtype or x.dtype)]


def output_spec(inputs: dict) -> OutputSpec:
    """Accept supplied strided buffers and retain FP32 results where requested."""
    dtype = inputs.get("output_dtype")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return OutputSpec((TensorSpec(tuple(inputs["x"].shape[:-1]) + (inputs["weight"].shape[-2],),
                                  dtype=dtype, stride_policy="strided", stride_padding=3),))


def call_descriptor(inputs: dict, **context) -> CallDescriptor:
    """Keep legacy dense routing and distinguish batched or FP32-output regimes."""
    x, weight = inputs["x"], inputs["weight"]
    out_dtype = output_spec(inputs).outputs[0].dtype or x.dtype
    layout = ("batched_weight_out_in_strided" if x.ndim == 3 else
              "weight_out_in_fp32_output" if out_dtype != x.dtype else
              "weight_out_in_row_major" if x.is_contiguous() and weight.is_contiguous() else
              "weight_out_in_strided")
    return CallDescriptor(dtype=str(x.dtype).removeprefix("torch."),
                          batch_size=x.shape[0] if x.ndim == 3 else None,
                          input_dim=x.shape[-1], last_dim=x.shape[-1], num_tokens=x.shape[-2],
                          output_dim=weight.shape[-2], layout=layout, quant="dense", **context)

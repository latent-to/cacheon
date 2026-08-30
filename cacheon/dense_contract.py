"""Validator-owned reference math for an unquantized local dense GEMM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_dense_inputs(
    *,
    num_tokens: int,
    input_dim: int,
    output_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        num_tokens,
        input_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    weight = torch.randn(
        output_dim,
        input_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    return {"x": x, "weight": weight}


def dense_reference(inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    x, weight = inputs["x"], inputs["weight"]
    return [F.linear(x.float(), weight.float()).to(x.dtype)]

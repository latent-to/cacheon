"""Validator-owned math for the fused residual-add + RMSNorm slot."""

from __future__ import annotations

import torch


def make_fused_add_rmsnorm_inputs(
    *,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float32
        ).to(dtype)

    return {
        "x": sample(num_tokens, hidden),
        "residual": sample(num_tokens, hidden),
        "weight": sample(hidden),
        "eps": 1e-6,
    }


def fused_add_rmsnorm_reference(inputs: dict[str, object]) -> list[torch.Tensor]:
    """Match SGLang's fused kernel: round the residual add, then normalize."""

    x = inputs["x"]
    residual = inputs["residual"]
    weight = inputs["weight"]
    eps = float(inputs["eps"])
    assert isinstance(x, torch.Tensor)
    assert isinstance(residual, torch.Tensor)
    assert isinstance(weight, torch.Tensor)
    new_residual = x + residual
    fp32 = new_residual.float()
    variance = fp32.square().mean(dim=-1, keepdim=True)
    norm = fp32 * torch.rsqrt(variance + eps) * weight.float()
    return [norm.to(x.dtype), new_residual]

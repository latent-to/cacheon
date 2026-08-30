"""Faithful fused residual-add plus RMSNorm control."""

import torch


def fused_add_rmsnorm(x, residual, weight, eps, out_norm, out_residual):
    out_residual.copy_(x + residual)
    fp32 = out_residual.float()
    variance = fp32.square().mean(dim=-1, keepdim=True)
    out_norm.copy_(
        (fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
    )

"""Example Triton RMSNorm kernel for the norm.rmsnorm slot.

Contract (validator-owned, see cacheon/slots.py):
    rmsnorm(x, weight, out, eps)
      x:      (..., H)  input  (bf16/fp16)
      weight: (H,)      scale  (same dtype)
      out:    (..., H)  output (same dtype) — the validator already allocated it
      eps:    float
    writes:  out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight

The validator owns the residual add (fused add+norm); the miner only computes the
pure normalization. Host surface is minimal: reshape to 2D, one kernel launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = x * rstd * w
    tl.store(out_ptr + row * row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    x2 = x.reshape(-1, x.shape[-1])
    o2 = out.reshape(-1, out.shape[-1])
    rows, n = x2.shape
    BLOCK = triton.next_power_of_2(n)
    _rmsnorm_kernel[(rows,)](x2, weight, o2, float(eps), n, x2.stride(0), BLOCK=BLOCK)

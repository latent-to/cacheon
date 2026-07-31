"""Example Triton kernel for the activation.silu_and_mul slot.

Contract (validator-owned, see cacheon/slots.py):
    silu_and_mul(x, out)
      x  : (..., 2*d)  input  (bf16/fp16)
      out: (..., d)    output (same dtype) — the validator already allocated it
    writes:  out = silu(x[..., :d]) * x[..., d:]

The miner owns ONLY the device kernel and a thin launch. The validator allocated
``out`` and will time/score the call. Note the host surface here is intentionally
minimal: reshape to 2D, compute a grid, launch. A hardened ABI would push even
the grid formula into manifest data so this launch shrinks further.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    d,
    stride_xr,
    stride_xc,
    stride_or,
    stride_oc,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < d

    gate = tl.load(x_ptr + row * stride_xr + col * stride_xc, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_xr + (col + d) * stride_xc, mask=mask, other=0.0).to(tl.float32)

    silu = gate * tl.sigmoid(gate)
    res = silu * up

    tl.store(out_ptr + row * stride_or + col * stride_oc, res.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    x2 = x.reshape(-1, x.shape[-1])
    o2 = out.reshape(-1, out.shape[-1])
    rows, two_d = x2.shape
    d = o2.shape[1]
    assert two_d == 2 * d, f"expected x last dim == 2*out last dim, got {two_d} vs {d}"

    BLOCK = 1024
    grid = (rows, triton.cdiv(d, BLOCK))
    _silu_and_mul_kernel[grid](
        x2,
        o2,
        d,
        x2.stride(0),
        x2.stride(1),
        o2.stride(0),
        o2.stride(1),
        BLOCK=BLOCK,
    )

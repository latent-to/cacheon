"""CPU tests for cacheon_kernels: the NVFP4 codec/layout primitives (round-trips).

No GPU, no cutlass, no sglang — exercises the portable library spine on the laptop.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from cacheon_kernels import codec  # noqa: E402


# ---- codec: layout transforms round-trip EXACTLY ----------------------------

def test_interleave_w13_halves_roundtrips():
    w = torch.randn(4, 256, 8)  # (E, 2I=256, H); I=128, group=64 -> ng=2
    inter = codec.interleave_w13_halves(w, group=64)
    assert inter.shape == w.shape
    assert not torch.equal(inter, w)  # it actually reorders
    back = codec.deinterleave_w13_halves(inter, group=64)
    assert torch.equal(back, w)


def test_interleave_places_up_before_gate():
    # gate = first half, up = second half; after interleave, the first `group` rows are up's.
    E, I, group = 1, 64, 64
    gate = torch.zeros(E, I, 1)
    up = torch.ones(E, I, 1)
    w = torch.cat([gate, up], dim=1)  # [gate(0) | up(1)]
    inter = codec.interleave_w13_halves(w, group=group)
    assert torch.equal(inter[:, :group], up)   # up first
    assert torch.equal(inter[:, group:], gate)  # gate second


def test_blockscale_swizzle_roundtrips_padding():
    scale = torch.randn(2, 130, 5).to(torch.float8_e4m3fn)
    swizzled = codec.swizzle_blockscale(scale)
    assert swizzled.shape == (2, 256, 8)
    assert torch.equal(codec.unswizzle_blockscale(swizzled, rows=130, cols=5).float(),
                       scale.float())


# ---- codec: NVFP4 quant round-trips within representational error ------------

def test_nvfp4_modelopt_byte_golden():
    x = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                       -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]])
    packed, scales = codec.quantize_nvfp4(x)
    golden = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0xA9, 0xCB, 0xED, 0x0F]],
                          dtype=torch.uint8)
    assert torch.equal(packed, golden) and torch.equal(scales, torch.ones_like(scales))
    outer = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    assert torch.equal(codec.dequantize_nvfp4(golden, outer.float(), global_scale=0.25),
                       x * 0.5)


def test_nvfp4_roundtrip_faithful():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 64, generator=g)
    codes, scales = codec.quantize_nvfp4(x, block=16, global_scale=1.0)
    deq = codec.dequantize_nvfp4(codes, scales, block=16, global_scale=1.0)
    cos = F.cosine_similarity(deq.flatten(), x.flatten(), dim=0)
    assert cos > 0.99  # NVFP4 representational floor on smooth data
    assert codes.dtype == torch.uint8 and codes.shape[-1] == x.shape[-1] // 2

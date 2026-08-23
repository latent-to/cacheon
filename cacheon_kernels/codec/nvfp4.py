"""Pure-Torch packed NVFP4 reference and reversible W13 interleave."""

from __future__ import annotations

import torch

# e2m1 positive magnitudes (1 sign + 2 exp + 1 mantissa).
_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_MAX = 6.0
NVFP4_BLOCK = 16


def _e2m1_nibbles(x: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_E2M1_POS, device=x.device, dtype=torch.float32)
    idx = (x.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return idx.to(torch.uint8) | ((x < 0).to(torch.uint8) << 3)


def quantize_nvfp4(
    x: torch.Tensor, *, block: int = NVFP4_BLOCK, global_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ModelOpt-compatible packed e2m1 bytes and one scale per block."""
    *lead, n = x.shape
    if n % block != 0:
        raise ValueError(f"last dim {n} is not a multiple of block {block}")
    xb = (x.float() / global_scale).reshape(*lead, n // block, block)
    block_scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / NVFP4_MAX
    nibbles = _e2m1_nibbles(xb / block_scale).reshape(*lead, n)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return packed, block_scale.squeeze(-1)


def dequantize_nvfp4(
    packed: torch.Tensor, block_scales: torch.Tensor, *, block: int = NVFP4_BLOCK,
    global_scale: float = 1.0,
) -> torch.Tensor:
    """Dequantize packed e2m1 bytes with linear block and outer scales."""
    if packed.dtype != torch.uint8:
        raise ValueError("packed NVFP4 weights must use uint8 storage")
    *lead, half_n = packed.shape
    n = half_n * 2
    lut = torch.tensor(
        (*_E2M1_POS, *(-value for value in _E2M1_POS)),
        device=packed.device,
        dtype=torch.float32,
    )
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    cb = lut[nibbles.long()].reshape(*lead, n // block, block)
    x = cb * block_scales.unsqueeze(-1) * global_scale
    return x.reshape(*lead, n)


def interleave_w13_halves(w: torch.Tensor, *, group: int = 64) -> torch.Tensor:
    """``w:(E, 2I, ...)`` laid out ``[gate(0:I) | up(I:2I)]`` along dim 1 ->
    ``[up, gate]`` interleaved in ``group``-row chunks (the donor megakernel's subtile
    order; M3 ships chunked ``[gate|up]``). Pure reshape — exactly invertible by
    :func:`deinterleave_w13_halves`. ``I`` must be a multiple of ``group``."""
    E, N = w.shape[0], w.shape[1]
    if N % 2 != 0 or (N // 2) % group != 0:
        raise ValueError(f"w13 dim1={N} must be even with half a multiple of group {group}")
    I = N // 2
    rest = w.shape[2:]
    ng = I // group
    g = w[:, :I].reshape(E, ng, group, *rest)
    u = w[:, I:].reshape(E, ng, group, *rest)
    return torch.stack([u, g], dim=2).reshape(E, N, *rest)


def deinterleave_w13_halves(w: torch.Tensor, *, group: int = 64) -> torch.Tensor:
    """Inverse of :func:`interleave_w13_halves` -> ``[gate | up]``."""
    E, N = w.shape[0], w.shape[1]
    I = N // 2
    rest = w.shape[2:]
    ng = I // group
    inter = w.reshape(E, ng, 2, group, *rest)
    u = inter[:, :, 0].reshape(E, I, *rest)
    g = inter[:, :, 1].reshape(E, I, *rest)
    return torch.cat([g, u], dim=1)


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """Match SGLang's padded 128-row/4-column ModelOpt scale swizzle."""
    squeeze = scale.ndim == 2
    if squeeze:
        scale = scale.unsqueeze(0)
    if scale.ndim != 3:
        raise ValueError("NVFP4 block scales must be rank two or three")
    batch, rows, cols = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    padded = scale.new_zeros(batch, padded_rows, padded_cols)
    padded[:, :rows, :cols] = scale
    swizzled = padded.reshape(
        batch, padded_rows // 128, 4, 32, padded_cols // 4, 4
    ).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(
        batch, padded_rows, padded_cols
    )
    return swizzled[0] if squeeze else swizzled


def unswizzle_blockscale(scale: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    """Invert :func:`swizzle_blockscale` and remove its deterministic padding."""
    squeeze = scale.ndim == 2
    if squeeze:
        scale = scale.unsqueeze(0)
    batch, padded_rows, padded_cols = scale.shape
    linear = scale.reshape(
        batch, padded_rows // 128, padded_cols // 4, 32, 4, 4
    ).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(
        batch, padded_rows, padded_cols
    )[:, :rows, :cols]
    return linear[0] if squeeze else linear

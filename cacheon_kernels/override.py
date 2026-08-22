"""Compose a small miner epilogue into a validator-owned base kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class EpiloguePoint:
    """A typed hole in a base kernel that a miner epilogue fills."""

    key: str  # "<slot>/<override_point>", e.g. "moe.fused_experts/gemm1_epilogue"
    base_kernel: str  # the validator-owned base in cacheon_kernels (e.g. "nvfp4_moe_megakernel")
    summary: str


# THE registry. Add an override-point here (epilogue -> codec -> prologue, per the roadmap).
OVERRIDE_POINTS: dict[str, EpiloguePoint] = {
    "moe.fused_experts/gemm1_epilogue": EpiloguePoint(
        key="moe.fused_experts/gemm1_epilogue",
        base_kernel="nvfp4_moe_megakernel",
        summary=(
            "GEMM1 epilogue of the fused NVFP4 MoE megakernel: a per-element activation "
            "epilogue(tCompute, gate, up, alpha, *act_params) applied to the GEMM1 "
            "accumulator (gate/up subtiles) before the fused NVFP4 requant. The swigluoai win."
        ),
    ),
}


def point_for(slot: str, override_point: str) -> EpiloguePoint:
    """Resolve (slot, override_point) to its EpiloguePoint, or raise a clear error."""
    key = f"{slot}/{override_point}"
    try:
        return OVERRIDE_POINTS[key]
    except KeyError:
        known = ", ".join(sorted(OVERRIDE_POINTS)) or "(none)"
        raise KeyError(f"unknown override-point {key!r}; known: {known}") from None


def _dense_moe(x, topk_ids, topk_weights, prepared, out, *, activation: Callable) -> torch.Tensor:
    """CPU oracle path with a pluggable ``activation(gate, up)``."""
    w13, w2, I = prepared["w13"], prepared["w2"], prepared["inter"]
    M, H = x.shape
    acc = torch.zeros(M, H, dtype=torch.float32, device=x.device)
    x32 = x.float()
    for k in range(topk_ids.shape[1]):
        e = topk_ids[:, k].long()
        wk = topk_weights[:, k].float()
        fc1 = torch.einsum("mh,mih->mi", x32, w13[e].float())  # (M, 2I)
        gate, up = fc1[:, :I], fc1[:, I:]
        act = activation(gate, up).float()
        acc += wk[:, None] * torch.einsum("mi,mhi->mh", act, w2[e].float())
    out.copy_(acc.to(out.dtype))
    return out


def compose(
    slot: str,
    override_point: str,
    *,
    epilogue_torch: Callable,
    epilogue_device: Optional[Callable] = None,
) -> Callable:
    """Return the standard fused-experts callable for one registered point."""
    point = point_for(slot, override_point)  # validates the override-point exists

    def fused_experts(x, topk_ids, topk_weights, prepared, out):
        fmt = prepared.get("fmt") if isinstance(prepared, dict) else None
        if fmt == "dense" or (
            fmt == "nvfp4" and (epilogue_device is None or not x.is_cuda)
        ):
            return _dense_moe(x, topk_ids, topk_weights, prepared, out, activation=epilogue_torch)
        assert point.base_kernel == "nvfp4_moe_megakernel"
        raise NotImplementedError(
            "nvfp4_moe_megakernel has no validator-owned GPU provider"
        )

    fused_experts.__cacheon_override__ = point.key  # provenance (attribution)
    return fused_experts


def default_prepare(*args):
    """Validator-owned dense or NVFP4 preparation for an override."""
    if len(args) == 2 and args[0] == "nvfp4_layer":
        from cacheon.moe_nvfp4_contract import dequantize_prepare_args

        w13, w2 = dequantize_prepare_args(tuple(args))
        return {"fmt": "nvfp4", "w13": w13, "w2": w2, "inter": w13.shape[1] // 2}
    if len(args) == 2:
        w13, w2 = args
        return {"fmt": "dense", "w13": w13.contiguous(), "w2": w2.contiguous(), "inter": w13.shape[1] // 2}
    raise ValueError("invalid override prepare contract")


def build_override(slot: str, override_point: str, entry_name: str, loader: Callable):
    """Build the device epilogue plus mandatory Torch reference."""
    epilogue_torch = loader(entry_name + "_ref")
    if epilogue_torch is None:
        raise ValueError(
            f"override {entry_name!r} must ship a torch reference {entry_name + '_ref'!r} "
            "(the EFC PyTorchEvaluation phase / fidelity oracle)"
        )
    epilogue_device = loader(entry_name)  # GPU-only; may be None on CPU
    entry = compose(slot, override_point, epilogue_torch=epilogue_torch, epilogue_device=epilogue_device)
    return entry, default_prepare

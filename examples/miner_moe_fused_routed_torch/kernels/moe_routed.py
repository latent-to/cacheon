"""CPU/GPU dry-run kernel for the moe.fused_routed_experts FAT slot.

The fat contract hands the miner router LOGITS: routing + experts + combine are
one implementation. Selection follows the engine's biased-sigmoid gate — topk on
``sigmoid(logits) + correction_bias`` — while the combine weights are the
UNBIASED sigmoid scores gathered at the selection, renormalized (+1e-20) and
scaled by ``routed_scaling``. ``prepare`` runs once at load and owns the weight
layout plus the static routing config; a real submission would do packed FP4/FP8
layout there and fuse the whole block into one kernel here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare(w13, w2, topk, routed_scaling):
    """Runs ONCE at load: decode weights, lay them out, seal routing config."""
    if isinstance(w13, str):
        if w13 != "nvfp4_layer":
            raise ValueError(f"unknown routed-MoE prepare tag: {w13!r}")
        from cacheon.moe_nvfp4_contract import dequantize_prepare_args

        w13, w2 = dequantize_prepare_args((w13, w2))
    I = w13.shape[1] // 2
    w13_up_gate = torch.cat([w13[:, I:], w13[:, :I]], dim=1).contiguous()
    return {
        "w13": w13_up_gate,   # (E, 2I, H), order [up; gate]
        "w2": w2.contiguous(),
        "inter": I,
        "topk": int(topk),
        "routed_scaling": float(routed_scaling),
    }


def fused_routed_experts(x, router_logits, correction_bias, prepared, out):
    """Runs per step: route, run the selected experts, combine into `out`."""
    w13 = prepared["w13"]
    w2 = prepared["w2"]
    I = prepared["inter"]
    topk = prepared["topk"]
    scaling = prepared["routed_scaling"]

    scores = torch.sigmoid(router_logits.float())
    choice = scores + correction_bias.float().unsqueeze(0)
    topk_ids = torch.topk(choice, k=topk, dim=-1).indices
    weights = scores.gather(1, topk_ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * scaling

    M, H = x.shape
    x32 = x.float()
    acc = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    for k in range(topk):
        e = topk_ids[:, k]
        wk = weights[:, k]
        w13_e = w13[e].float()                          # (M, 2I, H)  [up; gate]
        w2_e = w2[e].float()                            # (M, H, I)
        fc1 = torch.einsum("mh,mih->mi", x32, w13_e)    # (M, 2I)
        up, gate = fc1[:, :I], fc1[:, I:]               # up FIRST (prepared is [up; gate])
        act = F.silu(gate) * up
        acc += wk[:, None] * torch.einsum("mi,mhi->mh", act, w2_e)
    out.copy_(acc.to(out.dtype))

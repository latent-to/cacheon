"""CPU/GPU dry-run kernel for the moe.fused_experts (prepare+forward) slot.

A *prepare+forward* slot: a quant/layout-sensitive kernel is a PAIR. ``prepare`` runs
ONCE at load to lay out the expert weights the way ``forward``'s kernel wants;
``forward`` runs per step. Here ``prepare`` reorders the fused w13 from ``[gate; up]``
to ``[up; gate]`` (mirroring the GPT-OSS sm120 W13 repack), and ``forward`` consumes
that order. A real Blackwell submission would do MXFP4 weight + block-scale layout in
``prepare`` and a fused CUTLASS MoE (`flashinfer.cutlass_fused_moe`) in ``forward``;
this pure-torch version proves the (prepare, forward) contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare(w13, w2):
    """Runs ONCE at load. Reorder fused gate-up weights [gate; up] -> [up; gate]."""
    I = w13.shape[1] // 2
    w13_up_gate = torch.cat([w13[:, I:], w13[:, :I]], dim=1).contiguous()
    return {"w13": w13_up_gate, "w2": w2.contiguous(), "inter": I}


def fused_experts(x, topk_ids, topk_weights, prepared, out):
    """Runs per step. `prepared` is whatever `prepare` returned (validator-held)."""
    w13 = prepared["w13"]   # (E, 2I, H), order [up; gate]
    w2 = prepared["w2"]     # (E, H, I)
    I = prepared["inter"]
    M, H = x.shape
    K = topk_ids.shape[1]
    x32 = x.float()
    acc = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    for k in range(K):
        e = topk_ids[:, k].long()
        wk = topk_weights[:, k].float()
        w13_e = w13[e].float()                          # (M, 2I, H)  [up; gate]
        w2_e = w2[e].float()                            # (M, H, I)
        fc1 = torch.einsum("mh,mih->mi", x32, w13_e)    # (M, 2I)
        up, gate = fc1[:, :I], fc1[:, I:]               # up FIRST (prepared is [up; gate])
        act = F.silu(gate) * up
        acc += wk[:, None] * torch.einsum("mi,mhi->mh", act, w2_e)
    out.copy_(acc.to(out.dtype))

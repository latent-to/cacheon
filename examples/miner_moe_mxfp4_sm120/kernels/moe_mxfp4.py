"""MXFP4 fused-MoE slot bundle.

The real end-to-end win is expressed by ``rebuild.json`` plus candidate engine
kwargs: the rebuild plan applies the SM120 FlashInfer/SGLang source patch before
SGLang imports, and the candidate launches with ``moe_runner_backend=
flashinfer_mxfp4``. These pure-torch callables keep the bundle verifiable under
the generic prepare/forward slot contract; the live GPT-OSS path is the framework
backend swap, not this synthetic forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare(w13, w2):
    """Verifier-only prepare: mirror the generic MoE example's [gate;up]->[up;gate]."""
    inter = w13.shape[1] // 2
    return {
        "w13": torch.cat([w13[:, inter:], w13[:, :inter]], dim=1).contiguous(),
        "w2": w2.contiguous(),
        "inter": inter,
    }


def fused_experts_mxfp4(x, topk_ids, topk_weights, prepared, out):
    """Verifier-only forward for the slot ABI."""
    w13 = prepared["w13"]
    w2 = prepared["w2"]
    inter = prepared["inter"]
    tokens, hidden = x.shape
    topk = topk_ids.shape[1]
    acc = torch.zeros(tokens, hidden, device=x.device, dtype=torch.float32)
    x32 = x.float()
    for k in range(topk):
        expert = topk_ids[:, k].long()
        weight = topk_weights[:, k].float()
        w13_e = w13[expert].float()
        w2_e = w2[expert].float()
        fc1 = torch.einsum("mh,mih->mi", x32, w13_e)
        up, gate = fc1[:, :inter], fc1[:, inter:]
        act = F.silu(gate) * up
        acc += weight[:, None] * torch.einsum("mi,mhi->mh", act, w2_e)
    out.copy_(acc.to(out.dtype))

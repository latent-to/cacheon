"""CPU/GPU dry-run kernel for the attention.decode BLOCK slot (pure torch).

Contract: ``attention_decode(q, k, v, seq_lens, sm_scale, out)`` writes paged-decode
attention into the validator-allocated ``out``:

    q:(B,Hq,D)  k,v:(B,S,Hkv,D)  seq_lens:(B,)  ->  out:(B,Hq,D)   (GQA/MQA)

Each request's single query attends to its first ``seq_lens[i]`` cached k/v; padding
(positions >= seq_lens[i]) is masked out. We compute via ``scaled_dot_product_attention``
in the *input dtype* (so a faithful kernel sits at the backend's precision, not the
fp32-vs-bf16 gap) — a real submission would be a fused / flash / paged decode kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_decode(q, k, v, seq_lens, sm_scale, out):
    B, Hq, D = q.shape
    S, Hkv = k.shape[1], k.shape[2]
    g = Hq // Hkv
    qh = q.unsqueeze(2)                                  # (B,Hq,1,D)
    kh = k.repeat_interleave(g, dim=2).permute(0, 2, 1, 3)  # (B,Hq,S,D)
    vh = v.repeat_interleave(g, dim=2).permute(0, 2, 1, 3)  # (B,Hq,S,Dv)
    sidx = torch.arange(S, device=q.device).view(1, 1, 1, S)
    attend = sidx < seq_lens.view(B, 1, 1, 1)           # True = valid (B,1,1,S)
    o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attend, scale=sm_scale)  # (B,Hq,1,Dv)
    out.copy_(o.squeeze(2).to(out.dtype))

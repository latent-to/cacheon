"""CPU dry-run kernel for the attention.sdpa BLOCK slot (pure torch).

Contract: ``attention(q, k, v, out, sm_scale, causal)`` writes scaled-dot-product
attention into the validator-allocated ``out``:

    q:(T,Hq,D)  k,v:(S,Hkv,D)  ->  out:(T,Hq,D)     (GQA/MQA: Hq % Hkv == 0)

A real submission would be a fused / flash / fp8 kernel; this mirrors the reference
so the manifest -> scan -> load -> op-correctness path can be tested without a GPU.
It is the *block* analogue of the silu/rmsnorm CPU dry-run kernels: several fused
ops (QK^T, softmax, (.)V) behind one tensor-in/tensor-out contract.
"""

from __future__ import annotations

import torch


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
              sm_scale: float, causal: bool = True) -> None:
    T, Hq, D = q.shape
    S, Hkv, Dv = v.shape
    g = Hq // Hkv
    q32 = q.float()
    k32 = k.float().repeat_interleave(g, dim=1)  # (S,Hq,D)
    v32 = v.float().repeat_interleave(g, dim=1)  # (S,Hq,Dv)
    scores = torch.matmul(q32.permute(1, 0, 2), k32.permute(1, 2, 0)) * sm_scale  # (Hq,T,S)
    if causal:
        offset = S - T
        ti = torch.arange(T, device=q.device).view(T, 1)
        si = torch.arange(S, device=q.device).view(1, S)
        scores = scores.masked_fill((si > ti + offset).view(1, T, S), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, v32.permute(1, 0, 2)).permute(1, 0, 2)  # (T,Hq,Dv)
    out.copy_(o.to(out.dtype))

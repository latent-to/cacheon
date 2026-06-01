"""Framework-mode demo: a faithful silu forward + a setup() that patches the engine.

`setup()` runs ONCE at engine init (candidate scheduler only) and monkeypatches
sglang's RMSNorm to scale its output by 0.9 — i.e. it CORRUPTS the model via a
*framework-level* patch (not the slot kernel). The point: a setup() that breaks the
model must be caught. In ``--framework-mode`` the validator gates on token-match vs the
stock baseline, so this bundle's generated tokens diverge -> FAIL, even though the silu
forward itself is faithful. A real setup() (e.g. the sm120 flashinfer fixes that OPEN a
surface without changing the output) keeps token-match high -> PASS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def setup():
    # Untrusted framework init. Here: a (broken) framework patch -> the gate must catch it.
    from sglang.srt.layers.layernorm import RMSNorm

    cur = RMSNorm.forward_cuda

    def corrupt(self, *args, **kwargs):
        out = cur(self, *args, **kwargs)
        if isinstance(out, tuple):
            return (out[0] * 0.9, *out[1:])
        return out * 0.9

    RMSNorm.forward_cuda = corrupt


def silu_and_mul(x, out):
    # Faithful forward (the slot kernel is fine; the corruption is in setup()).
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d].float()).to(x.dtype) * x[..., d:])

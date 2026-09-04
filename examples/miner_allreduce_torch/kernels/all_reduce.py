"""Faithful supplied-group all-reduce; correctness control, not a speed claim."""

from __future__ import annotations

import torch.distributed as dist


def all_reduce(x, out, group=None):
    out.copy_(x)
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)

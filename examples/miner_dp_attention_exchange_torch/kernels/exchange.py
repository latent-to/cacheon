"""Faithful controls for the caller-owned DP-attention exchanges."""

import torch.distributed as dist


def all_gather_into_tensor(x, out, group=None):
    dist.all_gather_into_tensor(out, x, group=group)


def reduce_scatter_tensor(x, out, group=None):
    dist.reduce_scatter_tensor(out, x, group=group)

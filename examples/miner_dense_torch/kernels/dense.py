"""Faithful GEMM-family control; correctness example, not a speed claim."""

import torch


def prepare(weight):
    """Retain canonical [N,K] or [B,N,K] weights without another layout."""
    return weight


def dense(x, weight, out):
    """Honor the supplied output's dtype and strides for every family member."""
    if out.dtype != x.dtype or weight.dtype != x.dtype:
        x, weight = x.to(out.dtype), weight.to(out.dtype)
    if x.ndim == 2:
        torch.mm(x, weight.t(), out=out)
    else:
        torch.bmm(x, weight.transpose(-1, -2), out=out)

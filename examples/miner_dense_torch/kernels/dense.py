"""Faithful BF16 dense baseline; correctness example, not a speed claim."""

import torch


def prepare(weight):
    return weight


def dense(x, weight, out):
    torch.mm(x, weight.t(), out=out)

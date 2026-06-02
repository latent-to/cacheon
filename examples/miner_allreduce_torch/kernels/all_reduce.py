"""Faithful TP all-reduce for the collective.all_reduce slot (the comms-waist baseline).

Contract: ``all_reduce(x, out, group)`` fills the validator-allocated ``out`` with the
sum of ``x`` across the ranks in ``group``. This baseline just calls torch's all-reduce;
a real miner submits a custom **low-latency** (one-shot / NVLS / symmetric-memory) or
**compute-overlapped** reduce here, gated vs the trusted fp32 cross-rank sum.

The miner is handed the process group (the wider capability of a collective slot) but
still only writes a validator-owned buffer, mid-network and upstream of the sampler.
"""

from __future__ import annotations

import torch.distributed as dist


def all_reduce(x, out, group=None):
    out.copy_(x)
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)

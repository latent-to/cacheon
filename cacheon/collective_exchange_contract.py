"""Independent references for equal-size gather/scatter tensor exchanges."""

from __future__ import annotations

import torch
import torch.distributed as dist


def _rank_tensor(
    rows: int,
    hidden: int,
    *,
    dtype: torch.dtype,
    device: str,
    seed: int,
    rank: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(
        seed + 1_000_003 * rank
    )
    return torch.randn(
        rows,
        hidden,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(dtype)


def make_all_gather_inputs(
    *,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, object]:
    return {
        "x": _rank_tensor(
            num_tokens,
            hidden,
            dtype=dtype,
            device=device,
            seed=seed,
            rank=rank,
        ),
        "world_size": world_size,
    }


def make_reduce_scatter_inputs(
    *,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, object]:
    return {
        "x": _rank_tensor(
            num_tokens * world_size,
            hidden,
            dtype=dtype,
            device=device,
            seed=seed,
            rank=rank,
        ),
        "world_size": world_size,
    }


def all_gather_reference(inputs, group, rank: int, world_size: int):
    del rank
    source = inputs["x"].float()
    gathered = [torch.empty_like(source) for _ in range(world_size)]
    dist.all_gather(gathered, source, group=group)
    return [torch.cat(gathered, dim=0)]


def reduce_scatter_reference(inputs, group, rank: int, world_size: int):
    summed = inputs["x"].float().clone()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=group)
    return [summed.chunk(world_size, dim=0)[rank].contiguous()]

#!/usr/bin/env python3
"""Bounded NCU target for FlashMLA FP8 sparse kvcache decode."""

from __future__ import annotations

import argparse
import json
import time

import torch
from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--query-dim", type=int, default=576)
    parser.add_argument("--value-dim", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--bytes-per-token", type=int, default=656)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(123)
    device = torch.device("cuda")
    q = torch.randn(
        args.batch,
        args.q_len,
        args.heads,
        args.query_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # FP8 MLA cache is stored as bytes and viewed as float8_e4m3fn by the op:
    # [nope_fp8 bytes] + [scale bytes] + [rope bf16 bytes].
    kv_raw = torch.zeros(
        args.num_blocks,
        args.page_size,
        1,
        args.bytes_per_token,
        dtype=torch.uint8,
        device=device,
    )
    kv_cache = kv_raw.view(torch.float8_e4m3fn)

    block_tables = torch.arange(
        args.num_blocks, dtype=torch.int32, device=device
    ).repeat(args.batch, 1)
    seq_lens = torch.full(
        (args.batch,),
        args.num_blocks * args.page_size,
        dtype=torch.int32,
        device=device,
    )
    base = torch.arange(args.topk, dtype=torch.int32, device=device)
    offsets = torch.arange(args.batch, dtype=torch.int32, device=device).view(-1, 1, 1)
    indices = (base.view(1, 1, -1) + offsets * 17) % (
        args.num_blocks * args.page_size
    )
    indices = indices.contiguous()
    sched_meta = FlashMLASchedMeta()
    sm_scale = args.query_dim**-0.5

    def launch():
        return flash_mla_with_kvcache(
            q=q,
            k_cache=kv_cache,
            block_table=block_tables,
            cache_seqlens=seq_lens,
            head_dim_v=args.value_dim,
            tile_scheduler_metadata=sched_meta,
            softmax_scale=sm_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )

    for _ in range(max(args.warmup, 0)):
        out, lse = launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()
    start.record()
    out, lse = launch()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = start.elapsed_time(end)
    result = {
        "target": "flashmla_kvcache_fp8_sparse_decode",
        "shape": {
            "batch": args.batch,
            "q_len": args.q_len,
            "heads": args.heads,
            "query_dim": args.query_dim,
            "value_dim": args.value_dim,
            "page_size": args.page_size,
            "num_blocks": args.num_blocks,
            "topk": args.topk,
            "bytes_per_token": args.bytes_per_token,
        },
        "gpu": torch.cuda.get_device_name(),
        "capability": "".join(map(str, torch.cuda.get_device_capability())),
        "elapsed_ms": elapsed_ms,
        "checksum": float(
            torch.nan_to_num(out.float()).abs().mean().item()
            + torch.nan_to_num(lse.float()).abs().mean().item()
        ),
        "wall_time_unix": time.time(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

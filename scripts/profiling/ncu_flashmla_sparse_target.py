#!/usr/bin/env python3
"""Bounded NCU target for SGLang sparse FlashMLA decode/prefill kernel."""

from __future__ import annotations

import argparse
import json
import time

import torch
from sgl_kernel.flash_mla import flash_mla_sparse_fwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--kv-tokens", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--dv", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--kv-dtype",
        choices=("bf16", "fp8"),
        default="fp8",
        help="Try fp8 first because the serving NSYS kernel is fp8 sparse MLA.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(123)
    device = torch.device("cuda")
    q = torch.randn(
        args.tokens,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    if args.kv_dtype == "fp8":
        kv = torch.randn(
            args.kv_tokens,
            1,
            args.head_dim,
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
    else:
        kv = torch.randn(
            args.kv_tokens,
            1,
            args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

    base = torch.arange(args.topk, dtype=torch.int32, device=device)
    offsets = torch.arange(args.tokens, dtype=torch.int32, device=device).view(-1, 1)
    indices = (base.view(1, -1) + offsets * 17) % args.kv_tokens
    indices = indices.view(args.tokens, 1, args.topk).contiguous()
    sm_scale = args.head_dim**-0.5

    def launch():
        return flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            d_v=args.dv,
        )

    for _ in range(max(args.warmup, 0)):
        out, max_logits, lse = launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()
    start.record()
    out, max_logits, lse = launch()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = start.elapsed_time(end)
    result = {
        "target": "flashmla_sparse_fwd",
        "shape": {
            "tokens": args.tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "kv_tokens": args.kv_tokens,
            "topk": args.topk,
            "dv": args.dv,
            "kv_dtype": args.kv_dtype,
        },
        "gpu": torch.cuda.get_device_name(),
        "capability": "".join(map(str, torch.cuda.get_device_capability())),
        "elapsed_ms": elapsed_ms,
        "checksum": float(
            out.float().abs().mean().item()
            + max_logits.float().abs().mean().item()
            + lse.float().abs().mean().item()
        ),
        "wall_time_unix": time.time(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

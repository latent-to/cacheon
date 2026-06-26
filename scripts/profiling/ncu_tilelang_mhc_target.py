#!/usr/bin/env python3
"""Bounded NCU target for DeepSeek-V4 TileLang MHC pre/post kernels."""

from __future__ import annotations

import argparse
import json
import time

import torch

from sglang.srt.layers.mhc import (
    mhc_post_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(123)
    device = torch.device("cuda")
    hc = args.hc_mult
    hidden = args.hidden
    tokens = args.tokens
    hc_mult3 = hc * (2 + hc)

    gemm_out_mul = torch.randn(
        args.n_splits, tokens, hc_mult3, dtype=torch.float32, device=device
    )
    gemm_out_sqrsum = torch.rand(
        args.n_splits, tokens, dtype=torch.float32, device=device
    )
    hc_scale = torch.ones(3, dtype=torch.float32, device=device)
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device=device)
    residual = torch.randn(tokens, hc, hidden, dtype=torch.bfloat16, device=device)
    post_mix = torch.empty(tokens, hc, dtype=torch.float32, device=device)
    comb_mix = torch.empty(tokens, hc * hc, dtype=torch.float32, device=device)
    layer_input = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
    norm_weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)

    post_out = torch.empty_like(residual)
    post_x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.rand(tokens, hc, dtype=torch.float32, device=device)
    comb_res_mix = torch.rand(tokens, hc, hc, dtype=torch.float32, device=device)

    def run_pre() -> None:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            hidden,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            1e-6,
            args.n_splits,
            hc,
            hc_mult3,
        )

    def run_post() -> None:
        mhc_post_tilelang(
            comb_res_mix,
            residual,
            post_layer_mix,
            post_x,
            post_out,
            hc,
            hidden,
        )

    for _ in range(max(args.warmup, 0)):
        run_pre()
        run_post()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()
    start.record()
    run_pre()
    run_post()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = start.elapsed_time(end)
    result = {
        "target": "tilelang_mhc_pre_post",
        "shape": {
            "tokens": tokens,
            "hidden": hidden,
            "hc_mult": hc,
            "n_splits": args.n_splits,
        },
        "gpu": torch.cuda.get_device_name(),
        "capability": "".join(map(str, torch.cuda.get_device_capability())),
        "elapsed_ms": elapsed_ms,
        "checksum": float(
            layer_input.float().abs().mean().item()
            + post_out.float().abs().mean().item()
        ),
        "wall_time_unix": time.time(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

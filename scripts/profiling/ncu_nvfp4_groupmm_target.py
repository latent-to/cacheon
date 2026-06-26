#!/usr/bin/env python3
"""Bounded NCU target for SGLang Blackwell NVFP4 grouped MoE GEMM.

This intentionally profiles one kernel call, not the HTTP server. The shapes
default to DeepSeek-V4-Flash FP4 MoE dimensions:
  W13 grouped GEMM: total routed tokens x hidden(4096) -> 2*intermediate(4096)
  257 experts = 256 routed experts + one fused shared expert
  total routed tokens = batch(128) * top_k(7)
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from sglang.jit_kernel.benchmark.bench_nvfp4_blockwise_moe import _prepare_case
from sglang.jit_kernel.nvfp4 import cutlass_fp4_group_mm


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=positive_int, default=896)
    parser.add_argument("--n", type=positive_int, default=4096)
    parser.add_argument("--k", type=positive_int, default=4096)
    parser.add_argument("--num-experts", type=positive_int, default=257)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(f"NVFP4 requires sm100+, got sm{major}{minor}")

    torch.manual_seed(123)
    case = _prepare_case(
        args.total_tokens,
        args.n,
        args.k,
        args.num_experts,
        torch.bfloat16,
    )

    def launch() -> torch.Tensor:
        return cutlass_fp4_group_mm(
            case["a_fp4"],
            case["b_fp4"],
            case["a_blockscale"],
            case["b_blockscale"],
            case["alphas"],
            case["dtype"],
            case["params"],
        )

    for _ in range(max(args.warmup, 0)):
        out = launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.cudart().cudaProfilerStart()
    start.record()
    out = launch()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = start.elapsed_time(end)
    checksum = float(out.float().abs().mean().item())
    flops = 2 * args.total_tokens * args.n * args.k
    result = {
        "target": "sglang_nvfp4_cutlass_groupmm",
        "shape": {
            "total_tokens": args.total_tokens,
            "n": args.n,
            "k": args.k,
            "num_experts": args.num_experts,
        },
        "gpu": torch.cuda.get_device_name(),
        "capability": f"sm{major}{minor}",
        "elapsed_ms": elapsed_ms,
        "estimated_tflops": flops / (elapsed_ms * 1e-3) / 1e12,
        "checksum_abs_mean": checksum,
        "wall_time_unix": time.time(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Small CUPTI/Kineto probe for CUDA profiling availability."""

import sys

import torch
from torch.profiler import ProfilerActivity, profile


def _event_device_us(event) -> float:
    total = 0.0
    for name in ("device_time_total", "cuda_time_total", "self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, 0.0)
        if value:
            total += float(value)
    return total


def main() -> int:
    if not torch.cuda.is_available():
        print("KINETO_NO_CUDA")
        return 2

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    x = torch.randn((4096, 4096), device=device, dtype=torch.float16)
    w = torch.randn((4096, 4096), device=device, dtype=torch.float16)

    for _ in range(3):
        torch.matmul(x, w)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.matmul(x, w)
        torch.cuda.synchronize()

    device_us = sum(_event_device_us(evt) for evt in prof.key_averages())
    if device_us > 0:
        print(f"KINETO_OK device_time_us={device_us:.3f}")
        return 0

    print("KINETO_EMPTY device_time_us=0")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

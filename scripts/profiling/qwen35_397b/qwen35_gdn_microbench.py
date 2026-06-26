#!/usr/bin/env python3
"""Standalone Qwen3.5 GDN microbench for Nsight Compute.

This avoids profiling the full TP=2 SGLang serving process with NCU.  It imports
the same GDN kernel entry points used by SGLang/FLA, allocates the expected
per-rank tensors, warms them up, and brackets the measured launches with
cudaProfilerStart/Stop so ncu can use --profile-from-start off.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


PREFILL_CANDIDATES = (
    ("sglang.srt.layers.attention.fla.chunk", "chunk_gated_delta_rule"),
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("fla.ops.gated_delta_rule.chunk", "chunk_gated_delta_rule"),
)

DECODE_CANDIDATES = (
    (
        "sglang.srt.layers.attention.fla.fused_recurrent",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ),
    (
        "sglang.srt.layers.attention.fla.fused_recurrent",
        "fused_recurrent_gated_delta_rule",
    ),
    ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    ("fla.ops.gated_delta_rule.fused_recurrent", "fused_recurrent_gated_delta_rule"),
)


@dataclass
class ResolvedFunction:
    module: str
    name: str
    file: str | None
    signature: str


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise SystemExit(f"unsupported dtype {name!r}; choose one of {sorted(table)}") from exc


def resolve_function(candidates: tuple[tuple[str, str], ...]) -> tuple[Callable, ResolvedFunction]:
    errors: list[str] = []
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{module_name}: import failed: {exc}")
            continue
        fn = getattr(module, attr, None)
        if fn is None:
            errors.append(f"{module_name}: missing {attr}")
            continue
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "<signature unavailable>"
        return fn, ResolvedFunction(
            module=module_name,
            name=attr,
            file=getattr(module, "__file__", None),
            signature=sig,
        )
    raise SystemExit("could not resolve GDN function:\n" + "\n".join(errors))


def import_report() -> dict[str, object]:
    report: dict[str, object] = {"python": sys.version.split()[0], "candidates": {}}
    for label, candidates in {
        "prefill": PREFILL_CANDIDATES,
        "decode": DECODE_CANDIDATES,
    }.items():
        entries = []
        for module_name, attr in candidates:
            item: dict[str, object] = {"module": module_name, "attr": attr}
            try:
                module = importlib.import_module(module_name)
                fn = getattr(module, attr, None)
                item["module_file"] = getattr(module, "__file__", None)
                item["available"] = fn is not None
                if fn is not None:
                    try:
                        item["signature"] = str(inspect.signature(fn))
                    except Exception:
                        item["signature"] = "<signature unavailable>"
            except Exception as exc:
                item["available"] = False
                item["error"] = str(exc)
            entries.append(item)
        report["candidates"][label] = entries
    return report


def make_prefill_tensors(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, torch.Tensor | None]:
    B, T, H, HV, K, V = args.batch, args.seq_len, args.qk_heads, args.v_heads, args.head_dim, args.value_dim
    device = torch.device("cuda")
    if args.layout == "varlen":
        total_t = B * T
        q_shape = (1, total_t, H, K)
        v_shape = (1, total_t, HV, V)
        gate_shape = (1, total_t, HV)
        cu = torch.arange(0, total_t + 1, T, device=device, dtype=torch.long)
        state_n = B
    else:
        q_shape = (B, T, H, K)
        v_shape = (B, T, HV, V)
        gate_shape = (B, T, HV)
        cu = None
        state_n = B

    q = torch.randn(q_shape, device=device, dtype=dtype)
    k = F.normalize(torch.randn(q_shape, device=device, dtype=dtype), p=2, dim=-1)
    v = torch.randn(v_shape, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(gate_shape, device=device, dtype=dtype))
    beta = torch.rand(gate_shape, device=device, dtype=dtype).sigmoid()
    if args.initial_state == "none":
        initial_state = None
        initial_state_indices = None
    elif args.initial_state == "zero":
        initial_state = torch.zeros((state_n, HV, V, K), device=device, dtype=dtype)
        initial_state_indices = torch.arange(state_n, device=device, dtype=torch.int32)
    else:
        initial_state = torch.randn((state_n, HV, V, K), device=device, dtype=dtype)
        initial_state_indices = torch.arange(state_n, device=device, dtype=torch.int32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "initial_state_indices": initial_state_indices,
        "cu_seqlens": cu,
    }


def make_decode_tensors(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, torch.Tensor | None]:
    B, T, H, HV, K, V = args.batch, args.decode_tokens, args.qk_heads, args.v_heads, args.head_dim, args.value_dim
    device = torch.device("cuda")
    q = torch.randn((B, T, H, K), device=device, dtype=dtype)
    k = F.normalize(torch.randn((B, T, H, K), device=device, dtype=dtype), p=2, dim=-1)
    v = torch.randn((B, T, HV, V), device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn((B, T, HV), device=device, dtype=dtype))
    beta = torch.rand((B, T, HV), device=device, dtype=dtype).sigmoid()
    if args.recurrent_state_layout == "hvk":
        state_shape = (args.state_slots, HV, V, K)
    else:
        state_shape = (args.state_slots, HV, K, V)
    initial_state = torch.randn(state_shape, device=device, dtype=dtype)
    ssm_state_indices = torch.arange(B, device=device, dtype=torch.int32) if args.packed_decode else None
    num_accepted_tokens = torch.full((B,), T, device=device, dtype=torch.int32) if args.packed_decode else None

    q_flat = q[:, 0].reshape(B, H * K)
    k_flat = k[:, 0].reshape(B, H * K)
    v_flat = v[:, 0].reshape(B, HV * V)
    mixed_qkv = torch.cat((q_flat, k_flat, v_flat), dim=-1).contiguous()
    # The packed SGLang decode entrypoint receives post-linear gate tensors.
    a = torch.randn((B, HV), device=device, dtype=dtype).mul_(0.1).contiguous()
    b = torch.randn((B, HV), device=device, dtype=dtype).mul_(0.1).contiguous()
    A_log = torch.randn((HV,), device=device, dtype=dtype).mul_(0.1).contiguous()
    dt_bias = torch.randn((HV,), device=device, dtype=dtype).mul_(0.1).contiguous()
    out = torch.empty((B, 1, HV, V), device=device, dtype=dtype).contiguous()
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "out": out,
        "initial_state": initial_state,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
    }


def tensor_shapes(tensors: dict[str, torch.Tensor | None]) -> dict[str, list[int] | None]:
    return {k: (list(v.shape) if isinstance(v, torch.Tensor) else None) for k, v in tensors.items()}


def call_prefill(fn: Callable, tensors: dict[str, torch.Tensor | None], args: argparse.Namespace):
    kwargs = {
        "scale": args.scale,
        "initial_state": tensors["initial_state"],
        "initial_state_indices": tensors["initial_state_indices"],
        "cu_seqlens": tensors["cu_seqlens"],
        "head_first": False,
        "use_qk_l2norm_in_kernel": args.use_qk_l2norm_in_kernel,
    }
    try:
        return fn(tensors["q"], tensors["k"], tensors["v"], tensors["g"], tensors["beta"], **kwargs)
    except TypeError:
        legacy_kwargs = {
            "scale": args.scale,
            "initial_state": tensors["initial_state"],
            "output_final_state": args.output_final_state,
            "use_qk_l2norm_in_kernel": args.use_qk_l2norm_in_kernel,
            "cu_seqlens": tensors["cu_seqlens"],
            "transpose_state_layout": args.transpose_state_layout,
        }
        return fn(tensors["q"], tensors["k"], tensors["v"], tensors["g"], tensors["beta"], **legacy_kwargs)


def call_decode(
    fn: Callable,
    tensors: dict[str, torch.Tensor | None],
    args: argparse.Namespace,
    *,
    packed_decode_entrypoint: bool,
):
    if packed_decode_entrypoint:
        return fn(
            tensors["mixed_qkv"],
            tensors["a"],
            tensors["b"],
            tensors["A_log"],
            tensors["dt_bias"],
            args.scale,
            tensors["initial_state"],
            tensors["out"],
            tensors["ssm_state_indices"],
            args.use_qk_l2norm_in_kernel,
        )

    kwargs = {
        "scale": args.scale,
        "initial_state": tensors["initial_state"],
        "inplace_final_state": True,
        "ssm_state_indices": tensors["ssm_state_indices"],
        "num_accepted_tokens": tensors["num_accepted_tokens"],
        "use_qk_l2norm_in_kernel": args.use_qk_l2norm_in_kernel,
    }
    try:
        return fn(tensors["q"], tensors["k"], tensors["v"], tensors["g"], tensors["beta"], **kwargs)
    except TypeError:
        trimmed = {
            "scale": args.scale,
            "initial_state": tensors["initial_state"],
            "inplace_final_state": True,
            "use_qk_l2norm_in_kernel": args.use_qk_l2norm_in_kernel,
        }
        return fn(tensors["q"], tensors["k"], tensors["v"], tensors["g"], tensors["beta"], **trimmed)


def run_loop(call: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    last = None
    for _ in range(iters):
        last = call()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    torch.cuda.cudart().cudaProfilerStop()
    if isinstance(last, tuple) and isinstance(last[0], torch.Tensor):
        checksum = float(last[0].float().mean().detach().cpu())
    elif isinstance(last, torch.Tensor):
        checksum = float(last.float().mean().detach().cpu())
    else:
        checksum = 0.0
    print(f"MICROBENCH_CHECKSUM {checksum:.8e}", flush=True)
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prefill", "decode", "imports"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--qk-heads", type=int, default=8)
    parser.add_argument("--v-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--state-slots", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layout", choices=("varlen", "equal"), default="varlen")
    parser.add_argument("--initial-state", choices=("none", "zero", "random"), default="zero")
    parser.add_argument("--recurrent-state-layout", choices=("hvk", "hkv"), default="hvk")
    parser.add_argument("--packed-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--output-final-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-qk-l2norm-in-kernel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--transpose-state-layout", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.phase == "imports":
        print(json.dumps(import_report(), indent=2, sort_keys=True))
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    dtype = dtype_from_name(args.dtype)
    args.scale = args.scale if args.scale is not None else args.head_dim ** -0.5
    args.state_slots = args.state_slots or args.batch

    if args.phase == "prefill":
        fn, resolved = resolve_function(PREFILL_CANDIDATES)
        tensors = make_prefill_tensors(args, dtype)
        call = lambda: call_prefill(fn, tensors, args)
        work_tokens = args.batch * args.seq_len
    else:
        fn, resolved = resolve_function(DECODE_CANDIDATES)
        tensors = make_decode_tensors(args, dtype)
        packed_decode_entrypoint = resolved.name.endswith("packed_decode")
        call = lambda: call_decode(fn, tensors, args, packed_decode_entrypoint=packed_decode_entrypoint)
        work_tokens = args.batch * args.decode_tokens

    metadata = {
        "phase": args.phase,
        "resolved_function": asdict(resolved),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "args": vars(args),
        "tensor_shapes": tensor_shapes(tensors),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        },
    }
    print("MICROBENCH_METADATA " + json.dumps(metadata, sort_keys=True), flush=True)

    # Force JIT/autotune failures to show up before profiler start.
    call()
    torch.cuda.synchronize()
    elapsed = run_loop(call, args.warmup, args.iters)
    rate = work_tokens * args.iters / elapsed if elapsed > 0 else math.inf
    print(
        "MICROBENCH_RESULT "
        + json.dumps(
            {
                "elapsed_s": elapsed,
                "work_tokens": work_tokens * args.iters,
                "tokens_per_s": rate,
                "profiled_iters": args.iters,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

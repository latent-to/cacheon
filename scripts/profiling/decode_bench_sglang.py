#!/usr/bin/env python3
"""Controlled offline SGLang decode load for nsys/ncu profiling."""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List


def _make_input_ids(model_path: str, batch_size: int, input_len: int) -> List[List[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    seed = (
        "We need a deterministic synthetic profiling prompt with enough ordinary "
        "language tokens to exercise DeepSeek V4 Flash decode without early stop. "
        "The answer should continue in a concise technical style. "
    )
    base = tokenizer.encode(seed, add_special_tokens=False)
    if not base:
        base = [0]

    requests: List[List[int]] = []
    repeats = math.ceil((input_len + len(base)) / len(base))
    long_ids = (base * repeats)[:input_len]
    for i in range(batch_size):
        shift = i % len(base)
        ids = (long_ids[shift:] + long_ids[:shift])[:input_len]
        requests.append(ids)
    return requests


def _make_prompts(model_path: str, batch_size: int, input_len: int) -> List[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer.batch_decode(_make_input_ids(model_path, batch_size, input_len), skip_special_tokens=True)


def _generate(engine: Any, input_ids: List[List[int]], prompts: List[str], sampling_params: Dict[str, Any]):
    try:
        return engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    except TypeError:
        return engine.generate(prompts, sampling_params)


def _range_push(name: str) -> None:
    try:
        import torch

        torch.cuda.nvtx.range_push(name)
    except Exception:
        pass


def _range_pop() -> None:
    try:
        import torch

        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _cuda_profiler_start() -> None:
    try:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        pass


def _cuda_profiler_stop() -> None:
    try:
        import torch

        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        pass


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sgl-project/DeepSeek-V4-Flash-FP8")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--skip-server-warmup", action="store_true")
    parser.add_argument("--enable-layerwise-nvtx-marker", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--moe-runner-backend", default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--nsa-decode-backend", default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--tag", default="decode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import sglang as sgl

    engine_kwargs: Dict[str, Any] = {
        "model_path": args.model,
        "tp_size": args.tp_size,
        "trust_remote_code": args.trust_remote_code,
        "mem_fraction_static": args.mem_fraction_static,
        "disable_cuda_graph": args.disable_cuda_graph,
        "disable_radix_cache": args.disable_radix_cache,
        "skip_server_warmup": args.skip_server_warmup,
        "enable_layerwise_nvtx_marker": args.enable_layerwise_nvtx_marker,
        "log_level": args.log_level,
    }
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.moe_runner_backend:
        engine_kwargs["moe_runner_backend"] = args.moe_runner_backend
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
    if args.nsa_decode_backend:
        engine_kwargs["nsa_decode_backend"] = args.nsa_decode_backend

    print("ENGINE_KWARGS", json.dumps(engine_kwargs, sort_keys=True), flush=True)
    input_ids = _make_input_ids(args.model, args.batch_size, args.input_len)
    prompts = _make_prompts(args.model, args.batch_size, args.input_len)
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.output_len,
        "min_new_tokens": args.output_len,
        "ignore_eos": True,
        "skip_special_tokens": True,
    }

    engine = sgl.Engine(**engine_kwargs)
    try:
        for i in range(args.warmup_steps):
            print(f"WARMUP step={i + 1}/{args.warmup_steps}", flush=True)
            _generate(engine, input_ids, prompts, sampling_params)

        durations = []
        total_tokens = args.batch_size * args.output_len
        _sync()
        if args.cuda_profiler_range:
            _cuda_profiler_start()
        for i in range(args.timed_steps):
            label = f"{args.tag}:bs{args.batch_size}:in{args.input_len}:out{args.output_len}:step{i + 1}"
            print(f"TIMED_BEGIN {label}", flush=True)
            _range_push(label)
            start = time.perf_counter()
            outputs = _generate(engine, input_ids, prompts, sampling_params)
            _sync()
            elapsed = time.perf_counter() - start
            _range_pop()
            durations.append(elapsed)
            first_text = outputs[0].get("text", "") if outputs else ""
            print(
                "TIMED_END "
                f"{label} seconds={elapsed:.6f} output_tokens={total_tokens} "
                f"tok_per_s={total_tokens / elapsed:.3f} sample_chars={len(first_text)}",
                flush=True,
            )
        if args.cuda_profiler_range:
            _sync()
            _cuda_profiler_stop()

        median = _median(durations)
        print(
            "SUMMARY "
            f"batch_size={args.batch_size} input_len={args.input_len} output_len={args.output_len} "
            f"timed_steps={args.timed_steps} median_seconds={median:.6f} "
            f"median_tok_per_s={total_tokens / median:.3f}",
            flush=True,
        )
    finally:
        try:
            engine.shutdown()
        except Exception as exc:
            print(f"SHUTDOWN_ERROR {type(exc).__name__}: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

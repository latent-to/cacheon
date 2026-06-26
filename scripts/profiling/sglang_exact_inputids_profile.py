#!/usr/bin/env python3
"""Exact-length SGLang profiling client using native input_ids payloads.

This avoids tokenizer-dependent random text length drift in bench_serving.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class RequestResult:
    index: int
    ok: bool
    latency_s: float
    ttft_s: float | None
    error: str | None = None


def make_input_ids(index: int, input_len: int, vocab_span: int) -> list[int]:
    # Keep IDs comfortably away from special-token ranges while staying below
    # the normal DeepSeek vocab size. Different offsets prevent identical prompts.
    base = 1000 + (index * 997) % max(1, vocab_span)
    return [1000 + ((base + j) % vocab_span) for j in range(input_len)]


async def post_profile(
    session: aiohttp.ClientSession,
    base_url: str,
    mode: str,
    activities: list[str],
    num_steps: int | None,
) -> None:
    url = f"{base_url}/{mode}_profile"
    body: dict[str, Any] = {}
    if mode == "start":
        body["activities"] = activities
        if num_steps is not None:
            body["num_steps"] = str(num_steps)
    async with session.post(url, json=body) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"{mode}_profile failed: HTTP {resp.status}: {text}")


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    index: int,
    input_len: int,
    output_len: int,
    vocab_span: int,
    first_token_queue: asyncio.Queue[int],
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "input_ids": make_input_ids(index, input_len, vocab_span),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
    }
    start = time.perf_counter()
    ttft: float | None = None

    async with semaphore:
        try:
            async with session.post(f"{base_url}/generate", json=payload) as resp:
                if resp.status != 200:
                    return RequestResult(
                        index=index,
                        ok=False,
                        latency_s=time.perf_counter() - start,
                        ttft_s=None,
                        error=f"HTTP {resp.status}: {await resp.text()}",
                    )
                async for raw_chunk in resp.content:
                    chunk = raw_chunk.strip()
                    if not chunk:
                        continue
                    text = chunk.decode("utf-8", errors="replace")
                    if text.startswith("data: "):
                        text = text[len("data: ") :]
                    if text == "[DONE]":
                        continue
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if ttft is None and data.get("text"):
                        ttft = time.perf_counter() - start
                        await first_token_queue.put(index)
            return RequestResult(
                index=index,
                ok=True,
                latency_s=time.perf_counter() - start,
                ttft_s=ttft,
            )
        except Exception as exc:  # noqa: BLE001
            return RequestResult(
                index=index,
                ok=False,
                latency_s=time.perf_counter() - start,
                ttft_s=ttft,
                error=repr(exc),
            )


async def wait_ready(session: aiohttp.ClientSession, base_url: str, timeout_s: int) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{base_url}/v1/models") as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(1.0)
    raise TimeoutError(f"Server did not become ready within {timeout_s}s: {base_url}")


async def main_async(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}"
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency + 8, 32))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await wait_ready(session, base_url, args.ready_timeout_s)

        first_token_queue: asyncio.Queue[int] = asyncio.Queue()
        semaphore = asyncio.Semaphore(args.max_concurrency)

        if args.profile_mode == "before_dispatch":
            await post_profile(
                session,
                base_url,
                "start",
                args.profile_activities,
                args.profile_steps,
            )
            profile_started_at = time.perf_counter()
        else:
            profile_started_at = None

        start = time.perf_counter()
        tasks = [
            asyncio.create_task(
                run_request(
                    session,
                    base_url,
                    i,
                    args.input_len,
                    args.output_len,
                    args.vocab_span,
                    first_token_queue,
                    semaphore,
                )
            )
            for i in range(args.num_prompts)
        ]

        if args.profile_mode == "after_all_ttft":
            seen: set[int] = set()
            while len(seen) < args.num_prompts:
                seen.add(await first_token_queue.get())
            if args.profile_delay_after_all_ttft_s:
                await asyncio.sleep(args.profile_delay_after_all_ttft_s)
            await post_profile(
                session,
                base_url,
                "start",
                args.profile_activities,
                args.profile_steps,
            )
            profile_started_at = time.perf_counter()

        results = await asyncio.gather(*tasks)
        end = time.perf_counter()

        if args.profile_mode == "manual_stop":
            await post_profile(session, base_url, "stop", [], None)

    ok_results = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    latencies = [r.latency_s for r in ok_results]
    ttfts = [r.ttft_s for r in ok_results if r.ttft_s is not None]
    duration = end - start
    total_input = args.num_prompts * args.input_len
    total_output = len(ok_results) * args.output_len

    summary = {
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "total_input_tokens_exact": total_input,
        "total_output_tokens_requested": total_output,
        "completed": len(ok_results),
        "failed": len(errors),
        "duration_s": duration,
        "input_throughput_tok_s": total_input / duration if duration else None,
        "output_throughput_tok_s": total_output / duration if duration else None,
        "mean_latency_ms": statistics.mean(latencies) * 1000 if latencies else None,
        "median_latency_ms": statistics.median(latencies) * 1000 if latencies else None,
        "mean_ttft_ms": statistics.mean(ttfts) * 1000 if ttfts else None,
        "median_ttft_ms": statistics.median(ttfts) * 1000 if ttfts else None,
        "profile_mode": args.profile_mode,
        "profile_steps": args.profile_steps,
        "profile_delay_after_all_ttft_s": args.profile_delay_after_all_ttft_s,
        "profile_started_offset_s": (
            profile_started_at - start if profile_started_at is not None else None
        ),
        "errors": [r.__dict__ for r in errors[:8]],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--ready-timeout-s", type=int, default=60)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=16384)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--vocab-span", type=int, default=100000)
    parser.add_argument(
        "--profile-mode",
        choices=["none", "before_dispatch", "after_all_ttft", "manual_stop"],
        default="none",
    )
    parser.add_argument("--profile-steps", type=int, default=None)
    parser.add_argument("--profile-delay-after-all-ttft-s", type=float, default=0.0)
    parser.add_argument(
        "--profile-activities",
        nargs="+",
        default=["CUDA_PROFILER"],
    )
    parser.add_argument("--output-file")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()

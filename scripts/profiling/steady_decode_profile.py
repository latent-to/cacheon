#!/usr/bin/env python3
"""Start long streaming requests, capture a bounded steady-decode profile, then cancel.

This avoids waiting for all 128 x 16k completions just to collect a representative
Nsight Systems / Nsight Compute capture window.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class WorkerResult:
    index: int
    chunks: int
    bytes_read: int
    elapsed_s: float
    error: str = ""


def make_prompt(words: int) -> str:
    return ("Synthetic steady decode profiling prompt. " + "x " * words).strip()


def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = base_url.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server not ready: {url}")


def post_profile(
    base_url: str,
    action: str,
    activities: list[str],
    num_steps: int | None,
    timeout_s: float,
) -> None:
    endpoint = base_url.rstrip("/") + f"/{action}_profile"
    body: dict[str, Any] = {
        "activities": activities,
        "num_steps": num_steps,
        "profile_by_stage": False,
        "profile_stages": None,
        "output_dir": None,
        "profile_prefix": None,
    }
    response = requests.post(endpoint, json=body if action == "start" else None, timeout=timeout_s)
    if response.status_code >= 400:
        raise RuntimeError(f"{action}_profile failed: HTTP {response.status_code}: {response.text[:500]}")


def stream_one(
    index: int,
    base_url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    stop_event: threading.Event,
    timeout_s: float,
) -> WorkerResult:
    endpoint = base_url.rstrip("/") + "/v1/completions"
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    started = time.perf_counter()
    chunks = 0
    bytes_read = 0
    try:
        with requests.post(endpoint, json=body, stream=True, timeout=timeout_s) as response:
            if response.status_code != 200:
                text = response.text[:500]
                return WorkerResult(index, chunks, bytes_read, time.perf_counter() - started, f"HTTP {response.status_code}: {text}")
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks += 1
                    bytes_read += len(chunk)
                if stop_event.is_set():
                    break
    except Exception as exc:
        return WorkerResult(index, chunks, bytes_read, time.perf_counter() - started, repr(exc))
    return WorkerResult(index, chunks, bytes_read, time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/root/models/DeepSeek-V4-Flash-FP4")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--prompt-words", type=int, default=1200)
    parser.add_argument("--output-tokens", type=int, default=16384)
    parser.add_argument("--ready-timeout-s", type=float, default=600)
    parser.add_argument("--settle-s", type=float, default=90)
    parser.add_argument("--capture-s", type=float, default=30)
    parser.add_argument("--request-timeout-s", type=float, default=7200)
    parser.add_argument("--profile-timeout-s", type=float, default=300)
    parser.add_argument("--activities", default="CUDA_PROFILER")
    parser.add_argument("--profile-num-steps", type=int, default=None)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    wait_ready(args.base_url, args.ready_timeout_s)
    prompt = make_prompt(args.prompt_words)
    stop_event = threading.Event()
    activities = [item.strip() for item in args.activities.split(",") if item.strip()]
    started = time.perf_counter()
    results: list[WorkerResult] = []

    print(
        "STEADY_DECODE_BEGIN "
        f"requests={args.num_requests} output_tokens={args.output_tokens} "
        f"settle_s={args.settle_s} capture_s={args.capture_s}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                stream_one,
                index,
                args.base_url,
                args.model,
                prompt,
                args.output_tokens,
                stop_event,
                args.request_timeout_s,
            )
            for index in range(args.num_requests)
        ]
        time.sleep(args.settle_s)
        print("PROFILE_START", flush=True)
        post_profile(args.base_url, "start", activities, args.profile_num_steps, args.profile_timeout_s)
        time.sleep(args.capture_s)
        print("PROFILE_STOP", flush=True)
        stop_event.set()
        time.sleep(2)
        post_profile(args.base_url, "stop", activities, None, args.profile_timeout_s)

        for future in concurrent.futures.as_completed(futures, timeout=120):
            result = future.result()
            results.append(result)
            print(
                "REQUEST_DONE "
                f"idx={result.index} chunks={result.chunks} bytes={result.bytes_read} "
                f"elapsed_s={result.elapsed_s:.3f} error={result.error}",
                flush=True,
            )

    summary = {
        "elapsed_s": time.perf_counter() - started,
        "results": len(results),
        "errors": sum(1 for item in results if item.error),
        "chunks": sum(item.chunks for item in results),
        "bytes_read": sum(item.bytes_read for item in results),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": summary,
                    "results": [item.__dict__ for item in sorted(results, key=lambda row: row.index)],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

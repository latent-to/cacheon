#!/usr/bin/env python3
"""Drive SGLang's OpenAI completions API with fixed output lengths."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class Result:
    index: int
    url: str
    ok: bool
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str = ""


def make_prompt(target_words: int) -> str:
    return ("Profile the following synthetic workload. " + "x " * target_words).strip()


def post_one(index: int, url: str, model: str, prompt: str, output_tokens: int, timeout_s: float) -> Result:
    endpoint = url.rstrip("/") + "/v1/completions"
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    try:
        response = requests.post(endpoint, json=body, timeout=timeout_s)
        elapsed = time.perf_counter() - start
        if response.status_code != 200:
            return Result(index, url, False, elapsed, 0, 0, f"HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        usage = data.get("usage") or {}
        return Result(
            index=index,
            url=url,
            ok=True,
            elapsed_s=elapsed,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
    except Exception as exc:
        return Result(index, url, False, time.perf_counter() - start, 0, 0, repr(exc))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", action="append", required=True, help="SGLang base URL, e.g. http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--prompt-words", type=int, default=1200)
    parser.add_argument("--output-tokens", type=int, default=16384)
    parser.add_argument("--timeout-s", type=float, default=7200)
    parser.add_argument("--require-exact-output", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    prompt = make_prompt(args.prompt_words)
    urls = args.base_url
    started = time.perf_counter()
    results: list[Result] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(post_one, index, urls[index % len(urls)], args.model, prompt, args.output_tokens, args.timeout_s)
            for index in range(args.num_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "ok" if result.ok else "fail"
            print(
                f"{status} idx={result.index} url={result.url} "
                f"elapsed_s={result.elapsed_s:.3f} prompt_tokens={result.prompt_tokens} "
                f"completion_tokens={result.completion_tokens} {result.error}"
            )

    elapsed = time.perf_counter() - started
    ok_results = [result for result in results if result.ok]
    bad_results = [result for result in results if not result.ok]
    exact_failures = [
        result
        for result in ok_results
        if args.require_exact_output and result.completion_tokens != args.output_tokens
    ]
    output_tokens = sum(result.completion_tokens for result in ok_results)
    input_tokens = sum(result.prompt_tokens for result in ok_results)
    latencies = [result.elapsed_s for result in ok_results]

    summary: dict[str, Any] = {
        "ok": len(ok_results),
        "failed": len(bad_results),
        "exact_output_failures": len(exact_failures),
        "elapsed_s": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tok_s": input_tokens / elapsed if elapsed else 0.0,
        "output_tok_s": output_tokens / elapsed if elapsed else 0.0,
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p90_s": percentile(latencies, 90),
        "latency_p99_s": percentile(latencies, 99),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": summary,
                    "results": [result.__dict__ for result in sorted(results, key=lambda item: item.index)],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    return 1 if bad_results or exact_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

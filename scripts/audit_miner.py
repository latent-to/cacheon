#!/usr/bin/env python3
"""Audit a miner's Docker container for token-stuffing / garbage-injection gaming.

Usage:
    # With a running container (already started):
    python scripts/audit_miner.py --url http://localhost:8000

    # Auto-pull and run a container (needs GPU):
    python scripts/audit_miner.py --image 20031108/qwen-ttft-inference:v5 --gpus all

    # Compare against local vLLM baseline:
    python scripts/audit_miner.py --url http://localhost:8000 --baseline-url http://localhost:8001

Checks performed:
  1. Per-token timing burst detection (slow-then-fast pattern)
  2. Repetition ratio in output tail
  3. Output length vs baseline divergence
  4. Token match rate against baseline (if baseline URL provided)
  5. Logprob plausibility (entropy, distribution shape)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.request import Request, urlopen


PROMPTS = [
    "Summarize the following passage in 5 concise bullet points:\n\n"
    "The history of computing is a story of exponential progress. From the "
    "earliest mechanical calculators of the 17th century to modern quantum "
    "processors, each generation of machines has built upon the foundations "
    "laid by its predecessors. Charles Babbage conceived the Analytical Engine "
    "in 1837, a general-purpose mechanical computer that contained the basic "
    "elements of a modern computer: an arithmetic logic unit, control flow "
    "through conditional branching and loops, and integrated memory. Ada "
    "Lovelace recognized its potential beyond pure calculation, writing what "
    "is considered the first algorithm intended to be carried out by such a "
    "machine. The 20th century saw the realization of these ideas in "
    "electronic form, beginning with ENIAC in 1945 and progressing through "
    "transistors, integrated circuits, and microprocessors. Moore's Law, "
    "formulated in 1965, predicted the doubling of transistors on a chip "
    "approximately every two years, a trend that held remarkably steady for "
    "over five decades. Today we stand at a new inflection point, where "
    "classical scaling reaches physical limits and new paradigms such as "
    "neuromorphic computing, photonic processors, and quantum computing "
    "promise to extend the trajectory of computational capability.",
    "List the main named entities in the passage, grouped by person, place, "
    "and organization:\n\n"
    "In the spring of 1969, a group of researchers at ARPA, led by Larry "
    "Roberts at the Pentagon in Washington D.C., began laying the groundwork "
    "for what would become the internet. The first message was sent from "
    "UCLA to Stanford Research Institute on October 29, 1969. Vint Cerf and "
    "Bob Kahn later developed TCP/IP at Stanford, while Tim Berners-Lee at "
    "CERN in Geneva created the World Wide Web in 1989. The National Science "
    "Foundation funded NSFNET which connected five supercomputing centers. "
    "Marc Andreessen at the University of Illinois developed Mosaic, the "
    "first popular web browser, which later became Netscape Navigator. "
    "Meanwhile, companies like IBM, Microsoft, and Cisco built the "
    "infrastructure that carried this academic experiment into every home "
    "and office around the world.",
    "Analyze the writing style of this passage, focusing on tone, pacing, "
    "and point of view:\n\n"
    "The old house stood at the end of Maple Street, its paint peeling and "
    "shutters hanging at odd angles. Nobody went there anymore, not since "
    "Mrs. Henderson had passed away three winters ago. The garden, once a "
    "riot of color that drew admiring glances from every passerby, had gone "
    "to seed. Wild roses tangled with blackberry canes along the fence, and "
    "the lawn had surrendered to dandelions and creeping charlie. Inside, "
    "dust motes danced in shafts of light that filtered through grimy "
    "windows. The grandfather clock in the hallway had stopped at 3:47, "
    "marking the exact moment the last breath left her body, or so the "
    "neighbors liked to say. They told stories about that house now, the way "
    "people do when they need to fill silence with meaning. Some said they "
    "heard piano music on quiet evenings. Others swore they saw a light in "
    "the upstairs window. But Jimmy Dalton, who delivered the mail, said it "
    "was just the streetlamp reflecting off old glass. He was probably right.",
]

MODEL = "Qwen2.5-72B-Instruct"
MAX_TOKENS = 256
TEMPERATURE = 0


@dataclass
class TokenTiming:
    token: str
    arrival_ms: float  # ms since first token
    inter_token_ms: float  # ms since previous token


@dataclass
class PromptAuditResult:
    prompt_index: int
    output_text: str
    tokens: list[str] = field(default_factory=list)
    token_timings: list[TokenTiming] = field(default_factory=list)
    ttft_ms: float = 0.0
    total_tokens: int = 0
    tps: float = 0.0
    repetition_ratio: float = 0.0
    tail_repetition_ratio: float = 0.0
    timing_burst_detected: bool = False
    burst_ratio: float = 0.0  # ratio of fast tail vs slow head
    logprob_entropy: float = 0.0
    error: str | None = None


def send_streaming_prompt(
    url: str, prompt: str, prompt_index: int
) -> PromptAuditResult:
    """Send a prompt and collect per-token timing data."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        "logprobs": True,
        "top_logprobs": 5,
    }

    req = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urlopen(req, timeout=120)
    except Exception as exc:
        return PromptAuditResult(
            prompt_index=prompt_index,
            output_text="",
            error=f"request_failed: {exc}",
        )

    tokens: list[str] = []
    token_timings: list[TokenTiming] = []
    output_parts: list[str] = []
    all_logprobs: list[list[dict]] = []
    t_start = time.monotonic()
    t_first: float | None = None
    t_prev: float | None = None

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "")

        lp_data = choice.get("logprobs") or {}
        lp_content = lp_data.get("content") or []
        for entry in lp_content:
            if isinstance(entry, dict):
                tokens.append(entry.get("token", ""))
                all_logprobs.append(entry.get("top_logprobs", []))

        if content:
            now = time.monotonic()
            if t_first is None:
                t_first = now
                t_prev = now
            inter = (now - t_prev) * 1000
            arrival = (now - t_first) * 1000
            token_timings.append(
                TokenTiming(token=content, arrival_ms=arrival, inter_token_ms=inter)
            )
            t_prev = now
            output_parts.append(content)

    if t_first is None:
        return PromptAuditResult(
            prompt_index=prompt_index,
            output_text="",
            error="no_tokens_received",
        )

    t_last = time.monotonic()
    elapsed = t_last - t_first
    n_tokens = len(tokens) or len(token_timings)
    tps = n_tokens / elapsed if elapsed > 0 and n_tokens > 1 else 0.0
    ttft_ms = (t_first - t_start) * 1000

    rep_ratio = _compute_repetition_ratio(tokens)
    tail_rep = _compute_tail_repetition(tokens)
    burst_detected, burst_ratio = _detect_timing_burst(token_timings)
    lp_entropy = _compute_logprob_entropy(all_logprobs)

    return PromptAuditResult(
        prompt_index=prompt_index,
        output_text="".join(output_parts),
        tokens=tokens,
        token_timings=token_timings,
        ttft_ms=ttft_ms,
        total_tokens=n_tokens,
        tps=tps,
        repetition_ratio=rep_ratio,
        tail_repetition_ratio=tail_rep,
        timing_burst_detected=burst_detected,
        burst_ratio=burst_ratio,
        logprob_entropy=lp_entropy,
    )


def _compute_repetition_ratio(tokens: list[str]) -> float:
    """Ratio of repeated bigrams to total bigrams."""
    if len(tokens) < 4:
        return 0.0
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    counts = Counter(bigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(bigrams)


def _compute_tail_repetition(tokens: list[str], tail_frac: float = 0.5) -> float:
    """Repetition ratio in the last tail_frac of the output.

    Gaming attacks produce correct head then garbage tail.
    """
    if len(tokens) < 10:
        return 0.0
    split = len(tokens) - int(len(tokens) * tail_frac)
    tail = tokens[split:]
    if len(tail) < 4:
        return 0.0
    bigrams = [(tail[i], tail[i + 1]) for i in range(len(tail) - 1)]
    counts = Counter(bigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(bigrams)


def _detect_timing_burst(
    timings: list[TokenTiming], percentile_split: float = 0.4
) -> tuple[bool, float]:
    """Detect if late tokens arrive much faster than early tokens.

    A real model produces tokens at roughly uniform speed.
    A gaming server generates real tokens slowly, then dumps garbage fast.

    Returns (burst_detected, speed_ratio) where speed_ratio > 3 is suspicious.
    """
    if len(timings) < 10:
        return False, 1.0

    inter_times = [t.inter_token_ms for t in timings[1:]]  # skip first (TTFT boundary)
    if not inter_times:
        return False, 1.0

    n = len(inter_times)
    split = int(n * percentile_split)
    if split < 3 or (n - split) < 3:
        return False, 1.0

    head_median = statistics.median(inter_times[:split])
    tail_median = statistics.median(inter_times[split:])

    if tail_median <= 0:
        return True, float("inf")
    if head_median <= 0:
        return False, 1.0

    ratio = head_median / tail_median
    return ratio > 3.0, ratio


def _compute_logprob_entropy(all_logprobs: list[list[dict]]) -> float:
    """Average entropy of top-5 logprob distributions.

    Real model outputs have moderate entropy. Fabricated logprobs
    tend to be either too uniform or too peaked.
    """
    if not all_logprobs:
        return 0.0

    entropies = []
    for top_lps in all_logprobs:
        if not top_lps:
            continue
        logprobs = [entry.get("logprob", -10.0) for entry in top_lps]
        probs = [math.exp(lp) for lp in logprobs]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        entropies.append(entropy)

    return statistics.mean(entropies) if entropies else 0.0


def compare_with_baseline(
    miner_url: str, baseline_url: str, prompts: list[str]
) -> list[dict]:
    """Run same prompts against baseline and miner, compare outputs."""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{len(prompts)}: sending to baseline...")
        bl = send_streaming_prompt(baseline_url, prompt, i)
        print(f"  Prompt {i+1}/{len(prompts)}: sending to miner...")
        mn = send_streaming_prompt(miner_url, prompt, i)

        if bl.error or mn.error:
            results.append({
                "prompt_index": i,
                "error": bl.error or mn.error,
            })
            continue

        bl_tokens = bl.tokens
        mn_tokens = mn.tokens

        total = max(len(bl_tokens), len(mn_tokens))
        matches = sum(b == m for b, m in zip(bl_tokens, mn_tokens))
        match_rate = matches / total if total > 0 else 1.0

        results.append({
            "prompt_index": i,
            "baseline_tokens": len(bl_tokens),
            "miner_tokens": len(mn_tokens),
            "token_match_rate": match_rate,
            "baseline_tps": bl.tps,
            "miner_tps": mn.tps,
            "tps_ratio": mn.tps / bl.tps if bl.tps > 0 else 0.0,
            "length_ratio": len(mn_tokens) / len(bl_tokens) if bl_tokens else 0.0,
            "miner_tail_repetition": mn.tail_repetition_ratio,
            "miner_burst_detected": mn.timing_burst_detected,
        })

    return results


def wait_for_health(url: str, timeout: int = 600, interval: int = 5) -> bool:
    """Poll /health until 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = Request(f"{url}/health")
            resp = urlopen(req, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def start_container(image: str, gpus: str, port: int = 8000) -> str | None:
    """Pull and start a miner container. Returns container ID."""
    print(f"Pulling {image}...")
    subprocess.run(["docker", "pull", image], check=True)

    cmd = [
        "docker", "run", "-d",
        "--gpus", gpus,
        "--shm-size", "16g",
        "-p", f"{port}:8000",
        "-v", "/models:/models:ro",
        image,
    ]
    print(f"Starting container: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    container_id = result.stdout.strip()
    print(f"Container started: {container_id[:12]}")
    return container_id


def stop_container(container_id: str) -> None:
    """Stop and remove a container."""
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


def print_audit_report(results: list[PromptAuditResult], comparison: list[dict] | None = None) -> None:
    """Print a human-readable audit summary."""
    print("\n" + "=" * 70)
    print("MINER AUDIT REPORT")
    print("=" * 70)

    valid = [r for r in results if not r.error]
    if not valid:
        print("ERROR: No valid results. All prompts failed.")
        return

    print(f"\nPrompts sent: {len(results)}")
    print(f"Successful:   {len(valid)}")
    print(f"Failed:       {len(results) - len(valid)}")

    print("\n--- TIMING ---")
    tps_values = [r.tps for r in valid]
    ttft_values = [r.ttft_ms for r in valid]
    print(f"  Median TPS:  {statistics.median(tps_values):.1f}")
    print(f"  Mean TPS:    {statistics.mean(tps_values):.1f}")
    print(f"  Max TPS:     {max(tps_values):.1f}")
    print(f"  Median TTFT: {statistics.median(ttft_values):.0f} ms")

    print("\n--- TOKEN COUNTS ---")
    token_counts = [r.total_tokens for r in valid]
    print(f"  Median output length: {statistics.median(token_counts):.0f} tokens")
    print(f"  Min/Max:              {min(token_counts)} / {max(token_counts)}")
    always_max = sum(1 for c in token_counts if c >= MAX_TOKENS - 1)
    print(f"  Always hits max_tokens ({MAX_TOKENS}): {always_max}/{len(valid)}")

    print("\n--- GAMING SIGNALS ---")

    burst_count = sum(1 for r in valid if r.timing_burst_detected)
    burst_ratios = [r.burst_ratio for r in valid]
    print(f"  Timing burst detected: {burst_count}/{len(valid)} prompts")
    print(f"  Median burst ratio:    {statistics.median(burst_ratios):.2f}x "
          f"(>3x = suspicious)")

    rep_values = [r.repetition_ratio for r in valid]
    tail_rep = [r.tail_repetition_ratio for r in valid]
    print(f"  Median repetition:     {statistics.median(rep_values):.3f}")
    print(f"  Median tail repetition:{statistics.median(tail_rep):.3f} "
          f"(>0.3 = suspicious)")

    lp_ent = [r.logprob_entropy for r in valid]
    print(f"  Median logprob entropy:{statistics.median(lp_ent):.3f} "
          f"(normal: 0.5-2.0)")

    if comparison:
        valid_cmp = [c for c in comparison if "error" not in c]
        if valid_cmp:
            print("\n--- BASELINE COMPARISON ---")
            match_rates = [c["token_match_rate"] for c in valid_cmp]
            tps_ratios = [c["tps_ratio"] for c in valid_cmp]
            len_ratios = [c["length_ratio"] for c in valid_cmp]
            print(f"  Median token match rate: {statistics.median(match_rates):.3f} "
                  f"(<0.5 = suspicious)")
            print(f"  Median TPS ratio:        {statistics.median(tps_ratios):.2f}x "
                  f"vs baseline")
            print(f"  Median length ratio:     {statistics.median(len_ratios):.2f}x "
                  f"(>1.3 = suspicious)")

    # Verdict
    print("\n--- VERDICT ---")
    flags = []
    if burst_count > len(valid) // 2:
        flags.append("TIMING BURST: tokens arrive in slow-then-fast pattern")
    if statistics.median(tail_rep) > 0.3:
        flags.append("TAIL REPETITION: output tail is highly repetitive (garbage)")
    if always_max > len(valid) * 0.8:
        flags.append("LENGTH STUFFING: always generates max_tokens")
    if statistics.median(burst_ratios) > 5.0:
        flags.append(f"EXTREME BURST: head/tail speed ratio {statistics.median(burst_ratios):.1f}x")
    if comparison:
        valid_cmp = [c for c in comparison if "error" not in c]
        if valid_cmp:
            if statistics.median([c["token_match_rate"] for c in valid_cmp]) < 0.3:
                flags.append("LOW MATCH RATE: output diverges heavily from baseline")
            if statistics.median([c["tps_ratio"] for c in valid_cmp]) > 4.0:
                flags.append("IMPLAUSIBLE SPEEDUP: >4x TPS vs vLLM baseline")

    if flags:
        print("  SUSPICIOUS - potential gaming detected:")
        for f in flags:
            print(f"    * {f}")
    else:
        print("  CLEAN - no obvious gaming signals detected")

    # Raw per-prompt data
    print("\n--- PER-PROMPT DETAIL ---")
    print(f"  {'#':<3} {'TPS':<8} {'TTFT':<8} {'Tokens':<8} "
          f"{'Burst':<8} {'TailRep':<8} {'Head':<30}")
    for r in valid:
        head = r.output_text[:30].replace("\n", " ")
        print(f"  {r.prompt_index:<3} {r.tps:<8.1f} {r.ttft_ms:<8.0f} "
              f"{r.total_tokens:<8} "
              f"{'YES' if r.timing_burst_detected else 'no':<8} "
              f"{r.tail_repetition_ratio:<8.3f} {head}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a miner container for token-stuffing gaming"
    )
    parser.add_argument(
        "--url",
        help="URL of already-running miner (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--image",
        help="Docker image to pull and run (e.g. 20031108/qwen-ttft-inference:v5)",
    )
    parser.add_argument(
        "--gpus", default="all",
        help="GPU spec for docker run (default: all)",
    )
    parser.add_argument(
        "--baseline-url",
        help="URL of baseline vLLM server for comparison",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Host port to map container to (default: 8000)",
    )
    parser.add_argument(
        "--health-timeout", type=int, default=600,
        help="Seconds to wait for container /health (default: 600)",
    )
    args = parser.parse_args()

    if not args.url and not args.image:
        parser.error("Provide either --url (running server) or --image (to start)")

    container_id = None
    url = args.url

    try:
        if args.image and not args.url:
            container_id = start_container(args.image, args.gpus, args.port)
            url = f"http://localhost:{args.port}"
            print(f"Waiting for /health at {url} (timeout {args.health_timeout}s)...")
            if not wait_for_health(url, timeout=args.health_timeout):
                print("ERROR: Container never became healthy")
                sys.exit(1)
            print("Container is healthy.")

        print(f"\nRunning audit against {url}")
        print(f"Sending {len(PROMPTS)} prompts (temp=0, max_tokens={MAX_TOKENS})...\n")

        results: list[PromptAuditResult] = []
        for i, prompt in enumerate(PROMPTS):
            print(f"  Prompt {i+1}/{len(PROMPTS)}...")
            r = send_streaming_prompt(url, prompt, i)
            results.append(r)
            if r.error:
                print(f"    ERROR: {r.error}")
            else:
                print(f"    {r.total_tokens} tokens, {r.tps:.1f} TPS, "
                      f"burst={'YES' if r.timing_burst_detected else 'no'}")

        comparison = None
        if args.baseline_url:
            print(f"\nRunning baseline comparison against {args.baseline_url}...")
            comparison = compare_with_baseline(url, args.baseline_url, PROMPTS)

        print_audit_report(results, comparison)

    finally:
        if container_id:
            print(f"\nStopping container {container_id[:12]}...")
            stop_container(container_id)


if __name__ == "__main__":
    main()

"""Docker container lifecycle, HTTP client, and evaluation orchestration.

Shells out to the ``docker`` CLI for container management and uses
stdlib ``urllib`` / ``http`` for talking to miner and baseline servers.
No ``docker`` Python SDK, no ``requests`` -- keeps the dependency surface
at zero beyond the system Python.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from .baseline import (
    BaselineCache,
    BaselinePromptResult,
    derive_cache_key,
    load_cached_baseline,
    save_baseline_cache,
)
from .chain import CommitmentRecord
from .eval_schema import PerPromptResult, Prompt
from .scoring import CorrectnessVerdict, compute_correctness, compute_improvements
from .state import EvaluationRecord

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal data type
# --------------------------------------------------------------------------- #


@dataclass
class RawPromptResult:
    """Pre-scoring data from one prompt against one server."""

    prompt_index: int
    output_text: str
    tokens: list[str]
    top_logprobs: list[list[dict[str, Any]]] | None
    ttft_s: float
    throughput_tps: float
    output_tokens: int
    error: str | None = None


# --------------------------------------------------------------------------- #
# Docker lifecycle
# --------------------------------------------------------------------------- #


INTERNAL_NETWORK = "cacheon-internal"


def _ensure_network(name: str, *, internal: bool = False) -> None:
    """Create a Docker network if it doesn't already exist."""
    result = subprocess.run(
        ["docker", "network", "inspect", name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    cmd = ["docker", "network", "create"]
    if internal:
        cmd.append("--internal")
    cmd.append(name)
    logger.info("Creating Docker network: %s (internal=%s)", name, internal)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create Docker network {name}: {result.stderr.strip()}"
        )


def ensure_eval_network() -> None:
    """Create the internal eval network (no internet, no egress)."""
    _ensure_network(INTERNAL_NETWORK, internal=True)


def pull_image(image: str, digest: str, timeout_s: float = 300) -> None:
    """Pull a Docker image by digest. Raises on failure."""
    ref = f"{image}@{digest}"
    logger.info("Pulling image %s", ref)
    result = subprocess.run(
        ["docker", "pull", ref],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker pull failed (rc={result.returncode}): {result.stderr.strip()}"
        )


def start_container(
    image: str,
    digest: str,
    *,
    model_volume: str,
    container_port: int = 8000,
    memory: str = "200g",
    cpus: int = 32,
    shm_size: str = "16g",
    cmd_args: list[str] | None = None,
    container_name: str | None = None,
) -> tuple[str, str]:
    """Start an isolated container and return ``(container_id, base_url)``.

    The container is placed on the internal Docker network.  The
    validator (also on the same network) reaches it via its container
    IP; no host port publishing is needed.

    ``cmd_args`` are appended after the image reference and become the
    container CMD (e.g. ``["--model", "/models"]`` for vLLM).  Miner
    images define their own entrypoint so this is typically only used
    for the baseline.
    """
    ensure_eval_network()
    ref = f"{image}@{digest}"

    if container_name:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=15,
        )

    cmd = [
        "docker",
        "run",
        "-d",
        "--init",
        "--network",
        INTERNAL_NETWORK,
        "-v",
        f"{model_volume}:/models:ro",
        "--shm-size",
        shm_size,
        "--pids-limit",
        "4096",
        "--memory",
        memory,
        "--cpus",
        str(cpus),
        "--gpus",
        "all",
    ]
    if container_name:
        cmd.extend(["--name", container_name])
    cmd.append(ref)
    if cmd_args:
        cmd.extend(cmd_args)
    logger.info("Starting container: image=%s", ref)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker run failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    container_id = result.stdout.strip()
    try:
        ip = _get_container_ip(container_id)
    except Exception:
        stop_and_remove(container_id)
        reset_gpu_state()
        raise
    base_url = f"http://{ip}:{container_port}"
    logger.info("🐳 Container started: %s url=%s", container_id[:12], base_url)
    return container_id, base_url


def _get_container_ip(container_id: str) -> str:
    """Return the container's IP address on the internal eval network."""
    template = (
        '{{index .NetworkSettings.Networks "' + INTERNAL_NETWORK + '" "IPAddress"}}'
    )
    result = subprocess.run(
        ["docker", "inspect", "-f", template, container_id],
        capture_output=True,
        text=True,
        timeout=10,
    )
    ip = result.stdout.strip()
    if result.returncode != 0 or not ip:
        raise RuntimeError(
            f"Could not get IP for container {container_id[:12]} "
            f"on network {INTERNAL_NETWORK}: {result.stderr.strip()}"
        )
    return ip


def capture_container_logs(
    container_name_or_id: str,
    state_dir: str | Path,
    label: str,
) -> None:
    """Save ``docker logs`` output to ``state_dir/container_logs/{label}.log``.

    Best-effort: never raises. Called before ``stop_and_remove`` so the
    container still exists.
    """
    try:
        result = subprocess.run(
            ["docker", "logs", container_name_or_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        if not output.strip():
            return
        log_dir = Path(state_dir) / "container_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{label}.log"
        log_path.write_text(output, encoding="utf-8")
        logger.info("Container logs saved: %s (%d chars)", log_path, len(output))
    except Exception as exc:
        logger.warning("Failed to capture container logs for %s: %s", label, exc)


def stop_and_remove(container_id: str) -> None:
    """Stop and remove a container. Best-effort, never raises."""
    for action in ("stop", "rm"):
        try:
            subprocess.run(
                ["docker", action, container_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("docker %s %s failed: %s", action, container_id[:12], exc)


def reset_gpu_state() -> None:
    """Attempt to reset GPU state between evaluations. Best-effort."""
    try:
        subprocess.run(
            ["nvidia-smi", "--gpu-reset"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        logger.debug("nvidia-smi --gpu-reset failed (non-fatal): %s", exc)


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #


def wait_for_health(
    base_url: str,
    timeout_s: float = 600,
    poll_interval_s: float = 5,
) -> None:
    """Poll GET /health until 200. Raises TimeoutError on expiry."""
    url = f"{base_url}/health"
    deadline = time.monotonic() + timeout_s
    last_err: str = ""
    while time.monotonic() < deadline:
        try:
            resp = urlopen(url, timeout=5)
            if resp.status == 200:
                logger.info("✅ Container healthy at %s", base_url)
                return
            last_err = f"status={resp.status}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"/health at {base_url} not ready after {timeout_s}s: {last_err}"
    )


def send_prompt(
    base_url: str,
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0,
    stream: bool = True,
    logprobs: bool = False,
    top_logprobs: int = 5,
    timeout_s: float = 120,
    prompt_index: int = 0,
) -> RawPromptResult:
    """Send a chat completion request and parse the response.

    Single-pass design: ``stream=True, logprobs=True`` measures TTFT and
    throughput while simultaneously collecting tokens + logprobs for
    correctness checking. The ``stream=False`` path is kept for testing
    but is not used in production scoring.
    """
    url = f"{base_url}/v1/chat/completions"
    body: dict[str, Any] = {
        "model": "Qwen2.5-72B-Instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs

    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    t_start = time.monotonic()
    try:
        resp = urlopen(req, timeout=timeout_s)
    except Exception as exc:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            error=f"request_failed: {exc}",
        )

    if stream:
        return _parse_sse_response(resp, t_start, prompt_index)
    else:
        return _parse_json_response(resp, t_start, prompt_index)


def _parse_sse_response(
    resp: Any, t_start: float, prompt_index: int
) -> RawPromptResult:
    """Parse a streaming SSE response, extracting speed metrics and
    correctness data in a single pass.

    Returns a ``RawPromptResult`` with TTFT, throughput, output tokens,
    and (when the server includes them) per-token logprobs.

    Token source: when logprobs are present in the SSE chunks, tokens
    are taken from ``logprobs.content[].token`` so each token and its
    logprobs stay paired by construction.  ``delta.content`` is used
    for timing and display text only.
    """
    tokens: list[str] = []
    output_parts: list[str] = []
    all_top_logprobs: list[list[dict[str, Any]]] = []
    t_first: float | None = None
    t_last: float = t_start
    saw_logprobs = False

    try:
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
            if lp_content:
                saw_logprobs = True
                for entry in lp_content:
                    if isinstance(entry, dict):
                        tokens.append(entry.get("token", ""))
                        all_top_logprobs.append(entry.get("top_logprobs", []))

            if content:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                output_parts.append(content)
                if not saw_logprobs:
                    tokens.append(content)
    except Exception as exc:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="".join(output_parts),
            tokens=tokens,
            top_logprobs=all_top_logprobs if saw_logprobs else None,
            ttft_s=(t_first - t_start) if t_first else 0.0,
            throughput_tps=0.0,
            output_tokens=len(tokens),
            error=f"stream_error: {exc}",
        )

    if t_first is None:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            error="no_tokens_in_stream",
        )

    ttft = t_first - t_start
    elapsed = t_last - t_first
    n_tokens = len(tokens)
    tps = (n_tokens / elapsed) if elapsed > 0 and n_tokens > 1 else 0.0

    logprobs_out = all_top_logprobs if saw_logprobs else None

    if logprobs_out is not None and len(logprobs_out) != n_tokens:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="".join(output_parts),
            tokens=tokens,
            top_logprobs=logprobs_out,
            ttft_s=ttft,
            throughput_tps=tps,
            output_tokens=n_tokens,
            error=(
                f"logprob_token_mismatch: {len(logprobs_out)} logprob "
                f"entries vs {n_tokens} tokens"
            ),
        )

    return RawPromptResult(
        prompt_index=prompt_index,
        output_text="".join(output_parts),
        tokens=tokens,
        top_logprobs=logprobs_out,
        ttft_s=ttft,
        throughput_tps=tps,
        output_tokens=n_tokens,
    )


def _parse_json_response(
    resp: Any, t_start: float, prompt_index: int
) -> RawPromptResult:
    """Parse a non-streaming JSON response for correctness checking."""
    t_received = time.monotonic()
    try:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            error=f"json_parse_failed: {exc}",
        )

    choices = body.get("choices", [])
    if not choices:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=t_received - t_start,
            throughput_tps=0.0,
            output_tokens=0,
            error="no_choices_in_response",
        )

    choice = choices[0]
    message = choice.get("message", {})
    output_text = message.get("content", "")

    try:
        lp_data = choice.get("logprobs") or {}
        lp_content = lp_data.get("content") or []
    except AttributeError:
        lp_data = {}
        lp_content = []

    tokens: list[str] = []
    all_top_logprobs: list[list[dict[str, Any]]] = []
    for entry in lp_content:
        if not isinstance(entry, dict):
            continue
        tokens.append(entry.get("token", ""))
        all_top_logprobs.append(entry.get("top_logprobs", []))

    ttft = t_received - t_start
    n_tokens = len(tokens)

    return RawPromptResult(
        prompt_index=prompt_index,
        output_text=output_text,
        tokens=tokens,
        top_logprobs=all_top_logprobs if lp_content else None,
        ttft_s=ttft,
        throughput_tps=0.0,
        output_tokens=n_tokens,
    )


# --------------------------------------------------------------------------- #
# Orchestration helpers
# --------------------------------------------------------------------------- #


def _run_prompts_on_server(
    base_url: str,
    prompts: list[Prompt],
    *,
    stream: bool,
    logprobs: bool,
    per_prompt_timeout_s: int,
    n_warmup: int = 0,
) -> list[RawPromptResult]:
    """Send prompts to a running server, optionally discarding warmup results."""
    results: list[RawPromptResult] = []
    for i, prompt in enumerate(prompts):
        msgs = [{"role": m.role, "content": m.content} for m in prompt.messages]
        r = send_prompt(
            base_url,
            msgs,
            max_tokens=prompt.max_tokens,
            stream=stream,
            logprobs=logprobs,
            timeout_s=per_prompt_timeout_s,
            prompt_index=i,
        )
        if i < n_warmup:
            logger.debug("Warmup prompt %d/%d discarded", i + 1, n_warmup)
            continue
        results.append(r)
    return results


def _detect_gpu_count() -> int:
    """Count GPUs via nvidia-smi. Returns 0 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 0


def _max_model_len(gpu_count: int) -> int:
    """Choose vLLM max_model_len based on available KV cache (GPU count).

    More GPUs = more memory after TP-sharding the 72B model = room for
    longer KV caches.  Values are conservative.
    """
    if gpu_count >= 8:
        return 131_072
    if gpu_count >= 4:
        return 65_536
    return 32_768


def _baseline_cmd_args(gpu_count: int) -> list[str]:
    """Build the vLLM server command args for the baseline container."""
    mml = _max_model_len(gpu_count)
    args = [
        "--model",
        "/models",
        "--served-model-name",
        "Qwen2.5-72B-Instruct",
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(mml),
    ]
    if gpu_count > 1:
        args.extend(["--tensor-parallel-size", str(gpu_count)])
    return args


def run_baseline_if_needed(
    prompts: list[Prompt],
    *,
    baseline_image: str,
    baseline_digest: str,
    model_volume: str,
    gpu_count: int,
    cache_dir: Path,
    block_hash: str,
    state_dir: str | Path = "",
    startup_timeout_s: int = 600,
    per_prompt_timeout_s: int = 120,
    n_warmup: int = 2,
) -> BaselineCache:
    """Load cached baseline or run the vLLM baseline container, measure, and cache."""
    cache_key = derive_cache_key(block_hash, baseline_digest)
    cached = load_cached_baseline(cache_dir, cache_key)
    if cached is not None:
        logger.info(
            "Baseline cache hit for key=%s (%d prompts)", cache_key, len(cached.results)
        )
        return cached

    logger.info("Baseline cache miss for key=%s -- running baseline", cache_key)

    baseline_args = _baseline_cmd_args(gpu_count)
    logger.info("Baseline cmd args: %s", baseline_args)

    container_name = "cacheon-baseline"
    cid: str | None = None
    pull_image(baseline_image, baseline_digest)
    try:
        cid, base_url = start_container(
            baseline_image,
            baseline_digest,
            model_volume=model_volume,
            cmd_args=baseline_args,
            container_name=container_name,
        )
        wait_for_health(base_url, timeout_s=startup_timeout_s)

        results = _run_prompts_on_server(
            base_url,
            prompts,
            stream=True,
            logprobs=True,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
        )
    finally:
        if state_dir:
            log_label = f"baseline_{cache_key}"
            capture_container_logs(cid or container_name, state_dir, log_label)
        stop_and_remove(cid or container_name)
        reset_gpu_state()

    errors = [r for r in results if r.error]
    if errors:
        err_msg = "; ".join(f"prompt {r.prompt_index}: {r.error}" for r in errors)
        raise RuntimeError(f"Baseline had prompt errors (not caching): {err_msg}")

    baseline_results: list[BaselinePromptResult] = []
    for r in results:
        baseline_results.append(
            BaselinePromptResult(
                tokens=r.tokens,
                top_logprobs=r.top_logprobs or [],
                ttft_s=r.ttft_s,
                throughput_tps=r.throughput_tps,
                output_tokens=r.output_tokens,
            )
        )

    cache = BaselineCache(cache_key=cache_key, results=baseline_results)
    save_baseline_cache(cache_dir, cache_key, cache)
    logger.info(
        "✅ Baseline cached: key=%s, %d prompts", cache_key, len(baseline_results)
    )
    return cache


def evaluate_challenger(
    com: CommitmentRecord,
    prompts: list[Prompt],
    baseline: BaselineCache,
    *,
    model_volume: str,
    startup_timeout_s: int,
    per_prompt_timeout_s: int,
    n_warmup: int,
    current_block: int,
    state_dir: str | Path = "",
) -> EvaluationRecord:
    """Full lifecycle for one challenger. Returns an EvaluationRecord."""
    container_name = f"cacheon-uid{com.uid}-{com.hotkey[:8]}"
    cid: str | None = None
    results: list[RawPromptResult] = []
    eval_error: Exception | None = None

    try:
        pull_image(com.image, com.digest)
        cid, base_url = start_container(
            com.image,
            com.digest,
            model_volume=model_volume,
            container_name=container_name,
        )
        wait_for_health(base_url, timeout_s=startup_timeout_s)

        results = _run_prompts_on_server(
            base_url,
            prompts,
            stream=True,
            logprobs=True,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
        )
    except Exception as exc:
        logger.error("❌ Challenger UID %d failed: %s", com.uid, exc)
        eval_error = exc
    finally:
        if state_dir:
            log_label = f"uid{com.uid}_{com.hotkey[:8]}_{current_block}"
            capture_container_logs(cid or container_name, state_dir, log_label)
        stop_and_remove(cid or container_name)
        reset_gpu_state()

    if eval_error is not None:
        return _dq_record(com, current_block, str(eval_error))

    errors = [r for r in results if r.error]
    if errors:
        err_msg = "; ".join(f"prompt {r.prompt_index}: {r.error}" for r in errors)
        logger.warning("Challenger UID %d had prompt errors: %s", com.uid, err_msg)
        return _dq_record(com, current_block, f"prompt_errors: {err_msg}")

    per_prompt: list[PerPromptResult] = []
    all_verdicts: list[CorrectnessVerdict] = []
    miner_ttfts: list[float] = []
    miner_tps_list: list[float] = []
    baseline_ttfts: list[float] = []
    baseline_tps_list: list[float] = []

    n_scored = min(len(results), len(baseline.results))
    for i in range(n_scored):
        bl = baseline.results[i]
        r = results[i]

        verdict = compute_correctness(bl.tokens, r.tokens, r.top_logprobs)
        all_verdicts.append(verdict)

        if not verdict.passed:
            logger.warning(
                "Challenger UID %d correctness fail at prompt %d: %s "
                "(baseline=%r, miner=%r, logprobs=%s)",
                com.uid,
                i,
                verdict.reason,
                verdict.baseline_token_at_mismatch,
                verdict.miner_token_at_mismatch,
                verdict.miner_logprobs_at_mismatch,
            )

        miner_ttfts.append(r.ttft_s)
        miner_tps_list.append(r.throughput_tps)
        baseline_ttfts.append(bl.ttft_s)
        baseline_tps_list.append(bl.throughput_tps)

        per_prompt.append(
            PerPromptResult(
                ttft_s=r.ttft_s,
                throughput_tps=r.throughput_tps,
                output_tokens=r.output_tokens,
                token_match_rate=verdict.token_match_rate,
            )
        )

    agg_match_rate = (
        sum(v.token_match_rate for v in all_verdicts) / len(all_verdicts)
        if all_verdicts
        else 0.0
    )

    per_prompt_dicts = [pp.to_dict() for pp in per_prompt] if per_prompt else None

    any_failed = any(not v.passed for v in all_verdicts)
    if any_failed:
        reasons = [
            f"prompt {i}: {v.reason}"
            for i, v in enumerate(all_verdicts)
            if not v.passed
        ]
        return EvaluationRecord(
            uid=com.uid,
            hotkey=com.hotkey,
            commit_block=com.commit_block,
            image=com.image,
            digest=com.digest,
            score=0.0,
            ttft_improvement=0.0,
            throughput_improvement=0.0,
            token_match_rate=agg_match_rate,
            disqualified=True,
            disqualify_reason="correctness_fail: " + "; ".join(reasons),
            evaluated_at=time.time(),
            evaluation_block=current_block,
            per_prompt=per_prompt_dicts,
        )

    score, ttft_imp, tps_imp = compute_improvements(
        baseline_ttfts,
        miner_ttfts,
        baseline_tps_list,
        miner_tps_list,
    )

    logger.info(
        "Challenger UID %d scored: score=%.4f ttft_imp=%.4f tps_imp=%.4f "
        "match_rate=%.4f (%d prompts)",
        com.uid,
        score,
        ttft_imp,
        tps_imp,
        agg_match_rate,
        len(per_prompt),
    )
    for pp in per_prompt:
        logger.debug(
            "  prompt: ttft=%.4fs tps=%.1f tokens=%d match=%.4f",
            pp.ttft_s,
            pp.throughput_tps,
            pp.output_tokens,
            pp.token_match_rate,
        )

    return EvaluationRecord(
        uid=com.uid,
        hotkey=com.hotkey,
        commit_block=com.commit_block,
        image=com.image,
        digest=com.digest,
        score=score,
        ttft_improvement=ttft_imp,
        throughput_improvement=tps_imp,
        token_match_rate=agg_match_rate,
        disqualified=False,
        disqualify_reason=None,
        evaluated_at=time.time(),
        evaluation_block=current_block,
        per_prompt=per_prompt_dicts,
    )


def _dq_record(
    com: CommitmentRecord, current_block: int, reason: str
) -> EvaluationRecord:
    return EvaluationRecord(
        uid=com.uid,
        hotkey=com.hotkey,
        commit_block=com.commit_block,
        image=com.image,
        digest=com.digest,
        score=0.0,
        ttft_improvement=0.0,
        throughput_improvement=0.0,
        token_match_rate=0.0,
        disqualified=True,
        disqualify_reason=reason,
        evaluated_at=time.time(),
        evaluation_block=current_block,
    )


# --------------------------------------------------------------------------- #
# EvalFn factory
# --------------------------------------------------------------------------- #


def make_eval_fn(
    *,
    model_volume: str,
    baseline_cache_dir: str,
    baseline_image: str,
    baseline_digest: str,
    gpu_count: int = 0,
    state_dir: str = "",
    startup_timeout_s: int = 600,
    per_prompt_timeout_s: int = 120,
    n_warmup: int = 2,
) -> Callable:
    """Return an ``EvalFn`` compatible with ``validator.loop``.

    For each challenger, runs the full Docker lifecycle sequentially.
    Baseline is run once (or loaded from cache) per block hash.

    If ``gpu_count`` is positive it is used directly; otherwise
    ``nvidia-smi`` auto-detection is attempted on first eval.
    """
    cache_dir = Path(baseline_cache_dir)
    resolved_gpu_count = gpu_count

    def eval_fn(
        challengers: list[CommitmentRecord],
        *,
        current_block: int,
        block_hash: str | None,
    ) -> list[EvaluationRecord]:
        nonlocal resolved_gpu_count
        if not block_hash:
            logger.error("No block_hash available -- cannot derive prompt seed")
            return [_dq_record(c, current_block, "no_block_hash") for c in challengers]

        if resolved_gpu_count <= 0:
            resolved_gpu_count = _detect_gpu_count()
            if resolved_gpu_count <= 0:
                logger.error("Could not detect GPU count via nvidia-smi")
                return [
                    _dq_record(c, current_block, "no_gpu_count") for c in challengers
                ]
            logger.info("Auto-detected %d GPU(s)", resolved_gpu_count)

        from .prompts import sample_prompts

        mml = _max_model_len(resolved_gpu_count)
        prompts = sample_prompts(block_hash, n=10, max_context_tokens=mml)

        baseline = run_baseline_if_needed(
            prompts,
            baseline_image=baseline_image,
            baseline_digest=baseline_digest,
            model_volume=model_volume,
            gpu_count=resolved_gpu_count,
            cache_dir=cache_dir,
            block_hash=block_hash,
            state_dir=state_dir,
            startup_timeout_s=startup_timeout_s,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
        )

        results: list[EvaluationRecord] = []
        for com in challengers:
            logger.info(
                "⚔️  Evaluating challenger UID %d (%s) image=%s",
                com.uid,
                com.hotkey[:16],
                com.image,
            )
            record = evaluate_challenger(
                com,
                prompts,
                baseline,
                model_volume=model_volume,
                startup_timeout_s=startup_timeout_s,
                per_prompt_timeout_s=per_prompt_timeout_s,
                n_warmup=n_warmup,
                current_block=current_block,
                state_dir=state_dir,
            )
            results.append(record)

        return results

    return eval_fn

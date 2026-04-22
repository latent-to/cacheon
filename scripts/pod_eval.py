"""GPU-side evaluator: run one `EvaluationJob`, write one `EvaluationResult`.

Reads a job (baseline settings + a list of challengers) from JSON, runs
the baseline through the inference harness once (or loads it from cache),
runs each challenger's `policy.py`, scores each against the baseline, and
writes the per-challenger results back to JSON. The CPU-side validator
then merges those results into its persistent state.

This script is intentionally a **CLI + orchestration layer**. All heavy
lifting (model load, patch, run, score) lives in `inference_engine/`.
Torch + transformers are imported lazily inside `run_job` so argparse /
schema helpers remain usable in tests without a GPU.

Usage:

    python -m scripts.pod_eval \\
        --job /path/to/job.json \\
        --results-out /path/to/results.json \\
        [--device cuda] [--dtype float16]

Exit codes:
    0  — results.json written (per-challenger DQs are still "success")
    1  — fatal error (unreadable job, model load failure, etc.)

Per-challenger exceptions are caught and turned into DQ'd results so one
bad policy can't take down the whole tick.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure the repo root is on sys.path when run as `python scripts/pod_eval.py`
# (editable installs / `-m scripts.pod_eval` already handle this).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validator.eval_schema import (  # noqa: E402
    BaselineMetrics,
    ChallengerJob,
    ChallengerResult,
    EvaluationJob,
    EvaluationResult,
    SCHEMA_VERSION,
    hash_policy_file,
    read_job,
    write_results,
)

logger = logging.getLogger("cacheon.pod_eval")


# --------------------------------------------------------------------------- #
# Policy loading
# --------------------------------------------------------------------------- #


def _load_policy_class(policy_path: str):
    """Import `policy.py` from disk and return the single `KVCachePolicy`
    subclass defined in it. Raises on zero or multiple matches."""
    from inference_engine.policy import KVCachePolicy

    module_name = f"_miner_policy_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {policy_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    candidates = [
        obj for name, obj in vars(mod).items()
        if (
            isinstance(obj, type)
            and issubclass(obj, KVCachePolicy)
            and obj is not KVCachePolicy
        )
    ]
    if not candidates:
        raise RuntimeError("no KVCachePolicy subclass found in policy.py")
    if len(candidates) > 1:
        names = sorted(c.__name__ for c in candidates)
        raise RuntimeError(
            f"multiple KVCachePolicy subclasses found: {names}"
        )
    return candidates[0]


# --------------------------------------------------------------------------- #
# Baseline cache
# --------------------------------------------------------------------------- #
#
# The on-disk cache stores baseline tensors next to a `manifest` describing
# the inputs that produced them (model, prompt set, generation length,
# dtype, format version). On load we recompute the manifest from the
# current job and only reuse the cached tensors when every field matches.
# Mismatches (e.g. operator changed `max_new_tokens` or `dtype`) silently
# turn into a cache miss + recompute rather than a stale baseline.


BASELINE_CACHE_VERSION: int = 1
"""Bump when the saved baseline payload layout changes in a way that
older readers can't safely interpret."""


def _baseline_cache_path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"baseline-{key}.pt"


def _hash_prompts(prompts: list[str]) -> str:
    """Stable digest over the exact prompt list. Independent of any RNG
    seed: catches the case where the same `block_hash` produces a
    different prompt set after a `sample_prompts` change."""
    h = hashlib.sha256()
    for p in prompts:
        b = p.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def _build_baseline_manifest(
    *,
    model_name: str,
    n_prompts: int,
    max_new_tokens: int,
    dtype_name: str,
    prompts: list[str],
) -> dict:
    """Snapshot of every input that influences cached baseline bytes."""
    return {
        "cache_version": BASELINE_CACHE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "n_prompts": int(n_prompts),
        "max_new_tokens": int(max_new_tokens),
        "dtype_name": dtype_name,
        "prompt_hash": _hash_prompts(prompts),
    }


def _manifest_mismatch_reason(saved: dict, expected: dict) -> str | None:
    """`None` when the saved payload is reusable; otherwise a one-line
    human-readable reason for the mismatch (used in logs)."""
    if not isinstance(saved, dict):
        return "saved payload is missing a manifest"
    saved_manifest = saved.get("manifest")
    if not isinstance(saved_manifest, dict):
        return "saved payload is missing a manifest"
    for field in (
        "cache_version",
        "schema_version",
        "model_name",
        "n_prompts",
        "max_new_tokens",
        "dtype_name",
        "prompt_hash",
    ):
        if saved_manifest.get(field) != expected.get(field):
            return (
                f"{field} changed "
                f"(cached={saved_manifest.get(field)!r}, "
                f"expected={expected.get(field)!r})"
            )
    return None


def _try_load_baseline(cache_dir: str, key: str, *, expected_manifest: dict):
    """Return the cached payload dict iff its manifest matches *expected_manifest*.

    Misses (file absent), unreadable files, and manifest mismatches all
    return `None` so the caller recomputes. Mismatch reason is logged.
    """
    import torch

    path = _baseline_cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        cached = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning(
            "baseline cache at %s unreadable (%s) — recomputing", path, exc,
        )
        return None

    reason = _manifest_mismatch_reason(cached, expected_manifest)
    if reason is not None:
        logger.warning(
            "baseline cache at %s rejected: %s — recomputing",
            path, reason,
        )
        return None
    return cached


def _save_baseline(
    cache_dir: str,
    key: str,
    baseline,
    *,
    manifest: dict,
) -> None:
    import torch

    path = _baseline_cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "manifest": dict(manifest),
            "all_logits": [lg.detach().cpu() for lg in baseline.all_logits],
            "output_ids": baseline.output_ids,
            "output_texts": baseline.output_texts,
            "latency_s": baseline.latency_s,
            "peak_memory_bytes": baseline.peak_memory_bytes,
            "policy_memory_bytes": baseline.policy_memory_bytes,
        },
        tmp,
    )
    os.replace(tmp, path)
    logger.info("baseline cached at %s", path)


def _baseline_from_cached(cached: dict):
    """Rehydrate a `RunResult` from cached dict."""
    from inference_engine.harness import RunResult

    return RunResult(
        output_texts=list(cached["output_texts"]),
        output_ids=[list(ids) for ids in cached["output_ids"]],
        all_logits=list(cached["all_logits"]),
        latency_s=float(cached["latency_s"]),
        peak_memory_bytes=int(cached["peak_memory_bytes"]),
        policy_memory_bytes=int(cached["policy_memory_bytes"]),
    )


# --------------------------------------------------------------------------- #
# Challenger evaluation
# --------------------------------------------------------------------------- #


def _dq_result(
    challenger: ChallengerJob,
    reason: str,
) -> ChallengerResult:
    return ChallengerResult(
        uid=challenger.uid,
        hotkey=challenger.hotkey,
        commit_block=challenger.commit_block,
        repo=challenger.repo,
        revision=challenger.revision,
        score=0.0,
        kl_divergence=float("inf"),
        memory_reduction=0.0,
        latency_improvement=0.0,
        disqualified=True,
        disqualify_reason=reason,
        source_hash=challenger.source_hash,
    )


def _evaluate_challenger(
    harness,
    challenger: ChallengerJob,
    prompts: list[str],
    baseline,
    max_new_tokens: int,
) -> ChallengerResult:
    """Run one challenger; catch any exception into a DQ result so
    other challengers in the same job still complete.

    Also verifies the file at `challenger.policy_path` matches
    `challenger.source_hash` (if non-empty). A mismatch means the CPU
    side reviewed different bytes than the pod is about to execute — DQ
    out rather than score anything.
    """
    from inference_engine import scoring

    if challenger.source_hash:
        try:
            actual = hash_policy_file(challenger.policy_path)
        except OSError as exc:
            logger.exception(
                "challenger uid=%d: cannot read policy.py for hash verification",
                challenger.uid,
            )
            return _dq_result(
                challenger, f"policy.py unreadable: {exc}"[:500],
            )
        if actual != challenger.source_hash:
            logger.error(
                "challenger uid=%d source_hash mismatch "
                "(expected %s…, got %s…)",
                challenger.uid,
                challenger.source_hash[:12], actual[:12],
            )
            return _dq_result(challenger, "source_hash_mismatch")

    try:
        policy_cls = _load_policy_class(challenger.policy_path)
        policy = policy_cls()
        miner_run = harness.run(policy, prompts, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.exception(
            "challenger uid=%d failed during run: %s", challenger.uid, exc
        )
        return _dq_result(challenger, f"policy run failed: {exc}"[:500])

    try:
        sr = scoring.score(baseline, miner_run)
    except Exception as exc:
        logger.exception(
            "challenger uid=%d failed during scoring: %s", challenger.uid, exc
        )
        return _dq_result(challenger, f"scoring failed: {exc}"[:500])

    return ChallengerResult(
        uid=challenger.uid,
        hotkey=challenger.hotkey,
        commit_block=challenger.commit_block,
        repo=challenger.repo,
        revision=challenger.revision,
        score=float(sr.score),
        kl_divergence=float(sr.kl_divergence),
        memory_reduction=float(sr.memory_reduction),
        latency_improvement=float(sr.latency_improvement),
        disqualified=bool(sr.disqualified),
        disqualify_reason=sr.disqualify_reason,
        source_hash=challenger.source_hash,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_job(
    job: EvaluationJob,
    *,
    device: str = "cuda",
    dtype_name: str = "float16",
) -> EvaluationResult:
    """Materialize the job on the pod and score each challenger.

    One model load, one baseline run (or cached), N challenger runs.
    """
    import torch

    from inference_engine.harness import Harness
    from inference_engine.passthrough import PassthroughPolicy
    from inference_engine.prompts import sample_prompts

    dtype = getattr(torch, dtype_name)

    if job.block_hash is None:
        raise RuntimeError(
            "block_hash is required to seed prompts deterministically; "
            "CPU side must populate job.block_hash"
        )

    prompts = sample_prompts(job.block_hash, n=job.n_prompts)
    logger.info("sampled %d prompt(s)", len(prompts))

    harness = Harness(model_name=job.model_name, device=device, dtype=dtype)

    expected_manifest = _build_baseline_manifest(
        model_name=job.model_name,
        n_prompts=job.n_prompts,
        max_new_tokens=job.max_new_tokens,
        dtype_name=dtype_name,
        prompts=prompts,
    )
    cached = _try_load_baseline(
        job.baseline_cache_dir,
        job.baseline_cache_key,
        expected_manifest=expected_manifest,
    )
    if cached is not None:
        logger.info(
            "baseline cache HIT (key=%s) — skipping baseline run",
            job.baseline_cache_key,
        )
        baseline = _baseline_from_cached(cached)
        baseline_cached = True
    else:
        logger.info(
            "baseline cache MISS (key=%s) — running passthrough baseline",
            job.baseline_cache_key,
        )
        baseline = harness.run(
            PassthroughPolicy(), prompts, max_new_tokens=job.max_new_tokens,
        )
        _save_baseline(
            job.baseline_cache_dir,
            job.baseline_cache_key,
            baseline,
            manifest=expected_manifest,
        )
        baseline_cached = False

    baseline_metrics = BaselineMetrics(
        latency_s=float(baseline.latency_s),
        peak_memory_bytes=int(baseline.peak_memory_bytes),
        cached=baseline_cached,
    )

    challenger_results: list[ChallengerResult] = []
    for i, challenger in enumerate(job.challengers, start=1):
        logger.info(
            "[%d/%d] evaluating challenger uid=%d (hotkey=%s…)",
            i, len(job.challengers), challenger.uid, challenger.hotkey[:16],
        )
        t0 = time.time()
        r = _evaluate_challenger(
            harness, challenger, prompts, baseline, job.max_new_tokens,
        )
        logger.info(
            "[%d/%d] uid=%d done in %.1fs (score=%.4f, dq=%s)",
            i, len(job.challengers), challenger.uid,
            time.time() - t0, r.score, r.disqualified,
        )
        challenger_results.append(r)

    return EvaluationResult(
        schema_version=SCHEMA_VERSION,
        job_id=job.job_id,
        current_block=job.current_block,
        block_hash=job.block_hash,
        baseline=baseline_metrics,
        challenger_results=challenger_results,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pod_eval",
        description=(
            "Run one EvaluationJob on the GPU pod and write an "
            "EvaluationResult back to disk."
        ),
    )
    p.add_argument("--job", required=True, help="path to job.json (input)")
    p.add_argument(
        "--results-out", required=True, help="path to results.json (output)",
    )
    p.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    p.add_argument(
        "--dtype", default="float16",
        help="torch dtype name, e.g. float16, bfloat16 (default: float16)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG log level",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

    try:
        job = read_job(args.job)
    except Exception as exc:
        logger.error("failed to read job %s: %s", args.job, exc)
        return 1

    logger.info(
        "job_id=%s block=%d challengers=%d model=%s",
        job.job_id, job.current_block, len(job.challengers), job.model_name,
    )

    try:
        result = run_job(job, device=args.device, dtype_name=args.dtype)
    except Exception as exc:
        logger.exception("fatal error running job: %s", exc)
        return 1

    try:
        write_results(result, args.results_out)
    except Exception as exc:
        logger.error("failed to write results to %s: %s", args.results_out, exc)
        return 1

    logger.info("results written to %s", args.results_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

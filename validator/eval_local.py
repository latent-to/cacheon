"""Glue from the validator loop to GPU evaluation via a child process.

The main loop (`validator.loop`) calls an ``eval_fn`` with new on-chain
challengers; it must return `EvaluationRecord` rows to merge into
state. This module implements that hook by:

  1. Building an `EvaluationJob` (see `eval_schema`) with paths to each
     miner's `policy.py` on disk (the caller supplies how those paths
     are resolved—e.g. after a future download step).
  2. Running `python -m scripts.pod_eval` as a subprocess with
     ``--job`` / ``--results-out`` (the same JSON files are the API if
     the script runs on another host later).
  3. Reading `results.json` and turning rows into `EvaluationRecord`.

Evaluation runs out-of-process so the long-lived validator does not
keep a multi-gigabyte model loaded between ticks, and a crash inside
torch/transformers does not tear down the loop.

`job_runner` is injectable so tests can fake the subprocess and assert
on the written job only.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .chain import CommitmentRecord
from .eval_schema import (
    ChallengerJob,
    EvaluationJob,
    EvaluationResult,
    SCHEMA_VERSION,
    JOB_FILE_NAME,
    RESULTS_FILE_NAME,
    hash_policy_file,
    read_results,
    write_job,
)
from .state import EvaluationRecord, current_timestamp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Policy source resolution — injected from the caller
# --------------------------------------------------------------------------- #


PolicySourceFn = Callable[[CommitmentRecord], Path]
"""Resolve a commitment → local path to its `policy.py`.

PR1: tests pass a trivial lambda that points at a fixture file.
PR2 will replace this with `validator/policy_fetch.py` (HF/GitHub fetch
+ AST precheck).
"""


# --------------------------------------------------------------------------- #
# Job runner — injected for testability
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobRunContext:
    job_path: Path
    results_path: Path
    timeout_s: float


JobRunner = Callable[[JobRunContext], None]
"""Run one job; by contract the runner writes `results.json` to
`ctx.results_path`. Raises on failure (non-zero exit, timeout,
unreadable results)."""


def _default_job_runner(
    ctx: JobRunContext,
    *,
    pod_eval_cmd: Sequence[str],
    extra_args: Sequence[str],
    env: dict[str, str] | None,
) -> None:
    """Default runner — spawns `python -m scripts.pod_eval`."""
    cmd = [
        *pod_eval_cmd,
        "--job", str(ctx.job_path),
        "--results-out", str(ctx.results_path),
        *extra_args,
    ]
    logger.info("launching pod_eval: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=ctx.timeout_s,
            env=env or os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"pod_eval exceeded {ctx.timeout_s:.0f}s timeout"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"pod_eval exited {exc.returncode}"
        ) from exc

    if not ctx.results_path.exists():
        raise RuntimeError(
            f"pod_eval completed but did not write {ctx.results_path}"
        )


# --------------------------------------------------------------------------- #
# EvalFn factory
# --------------------------------------------------------------------------- #


def _job_id(current_block: int) -> str:
    return f"block-{current_block}-{uuid.uuid4().hex[:8]}"


def _baseline_cache_key(model_name: str, block_hash: str | None) -> str:
    """Cache key keyed on (model, block_hash) — same hash ⇒ same prompts
    ⇒ reusable baseline. `removeprefix` over `lstrip` so "0x000abc" and
    "0xabc" don't collapse to the same key."""
    raw = (block_hash or "").removeprefix("0x")
    tag = raw[:16] or "nohash"
    safe = model_name.replace("/", "_")
    return f"{safe}-{tag}"


def _result_to_record(
    result: EvaluationResult,
    *,
    evaluation_block: int,
) -> list[EvaluationRecord]:
    now = current_timestamp()
    records: list[EvaluationRecord] = []
    for r in result.challenger_results:
        records.append(EvaluationRecord(
            uid=r.uid,
            hotkey=r.hotkey,
            commit_block=r.commit_block,
            repo=r.repo,
            revision=r.revision,
            score=r.score,
            kl_divergence=r.kl_divergence,
            memory_reduction=r.memory_reduction,
            latency_improvement=r.latency_improvement,
            disqualified=r.disqualified,
            disqualify_reason=r.disqualify_reason,
            evaluated_at=now,
            evaluation_block=evaluation_block,
            source_hash=r.source_hash,
        ))
    return records


def make_local_eval_fn(
    *,
    policy_source_fn: PolicySourceFn,
    work_dir: str | os.PathLike,
    baseline_cache_dir: str | os.PathLike,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 256,
    n_prompts: int = 10,
    device: str = "cuda",
    dtype_name: str = "float16",
    timeout_s: float = 10 * 60,
    pod_eval_cmd: Sequence[str] = (sys.executable, "-m", "scripts.pod_eval"),
    job_runner: JobRunner | None = None,
):
    """Build an `EvalFn` that runs `pod_eval.py` locally.

    Args:
        policy_source_fn: resolves `CommitmentRecord` → local `policy.py`.
        work_dir: per-tick scratch dir; `job.json` + `results.json`
            land in `<work_dir>/<job_id>/`.
        baseline_cache_dir: persistent baseline artifact store
            (survives across ticks).
        timeout_s: hard wall-clock for the subprocess.
        pod_eval_cmd: argv prefix for the runner. Override in tests.
        job_runner: replace the default subprocess runner in tests.

    Returns:
        A callable matching `validator.loop.EvalFn`.
    """
    work_dir = Path(work_dir).resolve()
    baseline_cache_dir = Path(baseline_cache_dir).resolve()

    def _run(ctx: JobRunContext) -> None:
        if job_runner is not None:
            job_runner(ctx)
        else:
            _default_job_runner(
                ctx,
                pod_eval_cmd=pod_eval_cmd,
                extra_args=("--device", device, "--dtype", dtype_name),
                env=None,
            )

    def eval_fn(
        challengers: list[CommitmentRecord],
        *,
        current_block: int,
        block_hash: str | None,
    ) -> list[EvaluationRecord]:
        if not challengers:
            return []

        job_id = _job_id(current_block)
        tick_dir = work_dir / job_id
        tick_dir.mkdir(parents=True, exist_ok=True)
        job_path = tick_dir / JOB_FILE_NAME
        results_path = tick_dir / RESULTS_FILE_NAME

        challenger_jobs: list[ChallengerJob] = []
        for com in challengers:
            path = Path(policy_source_fn(com))
            if not path.exists():
                raise FileNotFoundError(
                    f"policy.py for UID {com.uid} not found at {path}"
                )
            source_hash = hash_policy_file(path)
            challenger_jobs.append(ChallengerJob(
                uid=com.uid,
                hotkey=com.hotkey,
                commit_block=com.commit_block,
                repo=com.repo,
                revision=com.revision,
                policy_path=str(path),
                source_hash=source_hash,
            ))

        job = EvaluationJob(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            current_block=current_block,
            block_hash=block_hash,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            n_prompts=n_prompts,
            baseline_cache_dir=str(baseline_cache_dir),
            baseline_cache_key=_baseline_cache_key(model_name, block_hash),
            challengers=challenger_jobs,
        )
        write_job(job, job_path)
        logger.info(
            "prepared job %s with %d challenger(s) at %s",
            job_id, len(challenger_jobs), job_path,
        )

        started = time.time()
        _run(JobRunContext(
            job_path=job_path,
            results_path=results_path,
            timeout_s=timeout_s,
        ))
        elapsed = time.time() - started
        logger.info("pod_eval finished in %.1fs", elapsed)

        result = read_results(results_path)
        if result.job_id != job.job_id:
            logger.warning(
                "results.job_id=%s does not match job.job_id=%s — pod "
                "may have stale state; accepting anyway",
                result.job_id, job.job_id,
            )

        returned_uids = {r.uid for r in result.challenger_results}
        expected_uids = {c.uid for c in challenger_jobs}
        missing = expected_uids - returned_uids
        if missing:
            logger.warning(
                "pod results missing challenger(s) uid=%s — treated as "
                "silently-failed, not recorded this tick",
                sorted(missing),
            )

        return _result_to_record(result, evaluation_block=current_block)

    return eval_fn

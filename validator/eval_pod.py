"""Glue from the validator loop to GPU evaluation.

The main loop (`validator.loop`) calls an ``eval_fn`` with new on-chain
challengers; it must return `EvaluationRecord` rows to merge into
state.  Two transport backends exist:

  * **Local** — ``make_local_eval_fn`` runs ``pod_eval.py`` as a local
    subprocess on the same machine. Used in tests and CI.
  * **Remote** — ``make_remote_eval_fn`` runs ``pod_eval.py`` on the GPU
    pod over SSH, transferring ``job.json`` + ``policy.py`` files via
    SFTP. Used in production.

Both share the same job-building and result-parsing logic.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
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
"""Resolve a commitment → local path to its ``policy.py``."""


def make_cache_policy_source_fn(cache_dir: str | os.PathLike) -> PolicySourceFn:
    """Build a ``PolicySourceFn`` that reads from the fetch cache.

    The cache layout is ``<cache_dir>/<sanitized_repo>/<revision>/policy.py``,
    matching ``validator/policy_fetch.fetch_policy_source``.
    """
    from .policy_fetch import sanitize_repo

    cache_dir = Path(cache_dir).resolve()

    def resolve(com: CommitmentRecord) -> Path:
        safe = sanitize_repo(com.repo)
        return cache_dir / safe / com.revision / "policy.py"

    return resolve


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _job_id(current_block: int) -> str:
    return f"block-{current_block}-{uuid.uuid4().hex[:8]}"


def _baseline_cache_key(model_name: str, block_hash: str | None) -> str:
    """Cache key keyed on (model, block_hash) — same hash => same prompts
    => reusable baseline.  ``removeprefix`` over ``lstrip`` so "0x000abc"
    and "0xabc" don't collapse to the same key."""
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


def _build_challenger_jobs(
    challengers: list[CommitmentRecord],
    policy_source_fn: PolicySourceFn,
) -> list[tuple[ChallengerJob, Path]]:
    """Resolve local policy paths and hash them.

    Returns ``(ChallengerJob, local_path)`` pairs.  The local_path is
    needed by the remote runner for SFTP upload; the local runner uses
    the path already set in `ChallengerJob.policy_path`.
    """
    jobs: list[tuple[ChallengerJob, Path]] = []
    for com in challengers:
        local_path = Path(policy_source_fn(com))
        if not local_path.exists():
            raise FileNotFoundError(
                f"policy.py for UID {com.uid} not found at {local_path}"
            )
        source_hash = hash_policy_file(local_path)
        jobs.append((
            ChallengerJob(
                uid=com.uid,
                hotkey=com.hotkey,
                commit_block=com.commit_block,
                repo=com.repo,
                revision=com.revision,
                policy_path=str(local_path),
                source_hash=source_hash,
            ),
            local_path,
        ))
    return jobs


def _parse_and_validate_results(
    results_path: Path,
    expected_job_id: str,
    expected_uids: set[int],
    *,
    evaluation_block: int,
) -> list[EvaluationRecord]:
    result = read_results(results_path)
    if result.job_id != expected_job_id:
        logger.warning(
            "results.job_id=%s does not match expected job_id=%s — "
            "pod may have stale state; accepting anyway",
            result.job_id, expected_job_id,
        )

    returned_uids = {r.uid for r in result.challenger_results}
    missing = expected_uids - returned_uids
    if missing:
        logger.warning(
            "pod results missing challenger(s) uid=%s — treated as "
            "silently-failed, not recorded this tick",
            sorted(missing),
        )

    return _result_to_record(result, evaluation_block=evaluation_block)


# --------------------------------------------------------------------------- #
# Local runner — subprocess on the same machine (tests / CI)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobRunContext:
    job_path: Path
    results_path: Path
    timeout_s: float


JobRunner = Callable[[JobRunContext], None]
"""Run one job; by contract the runner writes ``results.json`` to
``ctx.results_path``.  Raises on failure."""


def _default_job_runner(
    ctx: JobRunContext,
    *,
    pod_eval_cmd: Sequence[str],
    extra_args: Sequence[str],
    env: dict[str, str] | None,
) -> None:
    """Default local runner — spawns ``python -m scripts.pod_eval``."""
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
    """Build an ``EvalFn`` that runs ``pod_eval.py`` as a local subprocess.

    Useful for tests, CI, and local development.  Production uses
    ``make_remote_eval_fn`` instead.
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

        pairs = _build_challenger_jobs(challengers, policy_source_fn)
        challenger_jobs = [cj for cj, _ in pairs]

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
        logger.info("pod_eval finished in %.1fs", time.time() - started)

        return _parse_and_validate_results(
            results_path, job.job_id,
            {c.uid for c in challenger_jobs},
            evaluation_block=current_block,
        )

    return eval_fn


# --------------------------------------------------------------------------- #
# Remote runner — SSH/SFTP to a GPU pod (production)
# --------------------------------------------------------------------------- #

_POD_STAGING_ROOT = "/tmp/cacheon-eval"
_POD_BASELINE_CACHE = "/tmp/cacheon-eval-baseline"


_POLL_INTERVAL_S: float = 30.0


PollCallback = Callable[[float, int, str], None]
"""``on_poll(elapsed_s, pid, tail_text)`` — called each poll iteration."""


def make_remote_eval_fn(
    *,
    policy_source_fn: PolicySourceFn,
    transport,
    pod_work_dir: str = "/workspace/cacheon",
    baseline_cache_dir: str = _POD_BASELINE_CACHE,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 256,
    n_prompts: int = 10,
    device: str = "cuda",
    dtype_name: str = "float16",
    timeout_s: float = 20 * 60,
    poll_interval: float = _POLL_INTERVAL_S,
    work_dir: str | os.PathLike | None = None,
    on_poll: PollCallback | None = None,
):
    """Build an ``EvalFn`` that runs ``pod_eval.py`` on a remote GPU pod.

    Uses a detached (nohup) execution pattern to survive SSH proxy
    timeouts: the command is launched in the background, then short-lived
    SSH calls poll for the results file until it appears.

    Args:
        policy_source_fn: resolves ``CommitmentRecord`` -> local
            ``policy.py`` (already fetched + prechecked on the CPU).
        transport: a connected ``PodTransport`` (or compatible mock).
        pod_work_dir: repo checkout on the pod (``cd`` target).
        baseline_cache_dir: path *on the pod* for baseline caching.
        work_dir: local scratch directory for temporary job files.
            Defaults to a system temp dir.
        timeout_s: hard wall-clock before giving up on poll.
        poll_interval: seconds between poll checks.
        on_poll: optional callback invoked each poll iteration with
            ``(elapsed_s, pid, tail_text)``.
    """
    if work_dir is not None:
        _work_dir = Path(work_dir).resolve()
    else:
        _work_dir = Path(tempfile.gettempdir()) / "cacheon-eval-local"

    def eval_fn(
        challengers: list[CommitmentRecord],
        *,
        current_block: int,
        block_hash: str | None,
    ) -> list[EvaluationRecord]:
        if not challengers:
            return []

        job_id = _job_id(current_block)
        remote_dir = f"{_POD_STAGING_ROOT}/{job_id}"
        remote_job = f"{remote_dir}/{JOB_FILE_NAME}"
        remote_results = f"{remote_dir}/{RESULTS_FILE_NAME}"

        local_tick = _work_dir / job_id
        local_tick.mkdir(parents=True, exist_ok=True)
        local_job_path = local_tick / JOB_FILE_NAME
        local_results_path = local_tick / RESULTS_FILE_NAME

        # 1. Resolve policies and hash them (local)
        pairs = _build_challenger_jobs(challengers, policy_source_fn)

        # 2. Rewrite policy_path to remote staging paths
        remote_challenger_jobs: list[ChallengerJob] = []
        for cj, _local in pairs:
            remote_policy = f"{remote_dir}/policy_{cj.uid}.py"
            remote_challenger_jobs.append(ChallengerJob(
                uid=cj.uid,
                hotkey=cj.hotkey,
                commit_block=cj.commit_block,
                repo=cj.repo,
                revision=cj.revision,
                policy_path=remote_policy,
                source_hash=cj.source_hash,
            ))

        job = EvaluationJob(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            current_block=current_block,
            block_hash=block_hash,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            n_prompts=n_prompts,
            baseline_cache_dir=baseline_cache_dir,
            baseline_cache_key=_baseline_cache_key(model_name, block_hash),
            challengers=remote_challenger_jobs,
        )
        write_job(job, local_job_path)
        logger.info(
            "prepared remote job %s with %d challenger(s)",
            job_id, len(remote_challenger_jobs),
        )

        # 3. Create remote staging dir
        out, err, rc = transport.exec(f"mkdir -p {remote_dir}")
        if rc != 0:
            raise RuntimeError(
                f"failed to create staging dir on pod: {err.strip()}"
            )

        # 4. Upload job.json
        transport.upload(local_job_path, remote_job)

        # 5. Upload each policy.py
        for (cj_local, local_path), cj_remote in zip(pairs, remote_challenger_jobs):
            transport.upload(local_path, cj_remote.policy_path)
            logger.debug(
                "uploaded policy uid=%d → %s", cj_local.uid, cj_remote.policy_path,
            )

        # 6. Detached exec pod_eval.py (nohup to survive proxy timeouts)
        venv_python = f"{pod_work_dir}/../venv/bin/python3"
        remote_stdout_log = f"{remote_dir}/stdout.log"
        cmd = (
            f"cd {pod_work_dir} && "
            f"{venv_python} scripts/pod_eval.py "
            f"--job {remote_job} "
            f"--results-out {remote_results} "
            f"--device {device} --dtype {dtype_name} "
            f"> {remote_stdout_log} 2>&1"
        )
        logger.info("detached exec: %s", cmd)
        started = time.time()
        pid = transport.exec_detached(cmd)
        logger.info("pod_eval launched as PID %d", pid)

        # 7. Poll until results.json appears or timeout
        while True:
            elapsed = time.time() - started
            if elapsed > timeout_s:
                raise RuntimeError(
                    f"pod_eval timed out after {elapsed:.0f}s (PID {pid})"
                )
            time.sleep(poll_interval)
            if transport.poll_file(remote_results):
                break
            still_running = transport.is_pid_running(pid)
            tail_text = transport.tail(remote_stdout_log, n=5)
            if not still_running:
                if transport.poll_file(remote_results):
                    break
                full_tail = transport.tail(remote_stdout_log, n=20)
                raise RuntimeError(
                    f"pod_eval (PID {pid}) exited without producing "
                    f"results.json.\nLast output:\n{full_tail}"
                )
            if on_poll is not None:
                on_poll(elapsed, pid, tail_text)
            else:
                logger.debug(
                    "poll: PID %d still running (%.0fs elapsed)",
                    pid, elapsed,
                )

        elapsed = time.time() - started
        logger.info("pod_eval completed in %.1fs", elapsed)

        # 8. Download results.json
        transport.download(remote_results, local_results_path)

        # 9. Cleanup remote staging dir (best-effort)
        try:
            transport.exec(f"rm -rf {remote_dir}")
        except Exception:
            logger.warning("failed to clean up remote dir %s", remote_dir)

        # 10. Parse results
        return _parse_and_validate_results(
            local_results_path, job.job_id,
            {c.uid for c in remote_challenger_jobs},
            evaluation_block=current_block,
        )

    return eval_fn

"""Shared helpers for E2E tests.

Both e2e_pod.py (GPU-side, in-process) and e2e_cpu.py (CPU-side,
subprocess) use this module to:

  1. Read example_policies.json descriptors.
  2. Fetch each policy from HuggingFace.
  3. Run AST precheck / sandbox.
  4. Build an EvaluationJob ready for pod_eval.

This keeps the two scripts thin: one calls pod_eval.run_job() directly,
the other writes job.json and spawns the subprocess — but the setup
logic is identical.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from validator.chain import CommitmentRecord
from validator.eval_schema import (
    ChallengerJob,
    EvaluationJob,
    SCHEMA_VERSION,
    hash_policy_file,
)
from validator.policy_fetch import FetchOutcome, fetch_policy_source
from validator.precheck import make_fetch_precheck

logger = logging.getLogger("e2e_common")

DEFAULT_DESCRIPTORS_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "example_policies.json"
)

DEFAULT_BLOCK_HASH = "0x" + "aa" * 32
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def configure_logging(*, verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "huggingface_hub", "urllib3", "requests", "paramiko"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── per-policy fetch + precheck ──────────────────────────────────────


def _fetch_and_precheck(
    descriptor: dict,
    cache_dir: Path,
    hf_token: str | None,
) -> dict:
    """Fetch one policy from HF, run sandbox precheck.

    Returns a dict with ``name``, ``status`` ("ok" | reason), and
    ``policy_path`` (str or None).
    """
    name = descriptor["name"]
    repo = descriptor["repo"]
    revision = descriptor["revision"]

    logger.info("[%s] Fetching %s @ %s …", name, repo, revision[:8])
    fetch_result = fetch_policy_source(
        repo, revision,
        cache_dir=cache_dir,
        max_bytes=1_048_576,
        etag_timeout_s=30.0,
        hf_token=hf_token,
    )
    if fetch_result.outcome is not FetchOutcome.OK:
        reason = f"fetch_{fetch_result.outcome.value}: {fetch_result.reason}"
        logger.warning("[%s] %s", name, reason)
        return {"name": name, "status": reason, "policy_path": None}

    com = CommitmentRecord(
        uid=0, hotkey="e2e", commit_block=0,
        repo=repo, revision=revision, raw="{}",
    )
    precheck = make_fetch_precheck(lambda _r, _rev: fetch_result)
    precheck_result = precheck(com)
    if precheck_result.outcome.value != "ok":
        reason = f"sandbox_{precheck_result.outcome.value}: {precheck_result.reason}"
        logger.warning("[%s] %s", name, reason)
        return {"name": name, "status": reason, "policy_path": None}

    logger.info("[%s] OK → %s", name, fetch_result.path)
    return {
        "name": name,
        "status": "ok",
        "policy_path": str(fetch_result.path),
        "repo": repo,
        "revision": revision,
    }


# ── job builder ──────────────────────────────────────────────────────


def build_e2e_job(
    descriptors_path: Path = DEFAULT_DESCRIPTORS_PATH,
    *,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
    block_hash: str = DEFAULT_BLOCK_HASH,
    n_prompts: int = 3,
    max_new_tokens: int = 256,
    model_name: str = DEFAULT_MODEL_NAME,
    baseline_cache_dir: str | None = None,
) -> tuple[EvaluationJob, list[dict]]:
    """Fetch policies, precheck, build an EvaluationJob.

    Returns ``(job, reports)`` where *reports* is a per-policy list of
    dicts with ``name``, ``status``, and (when ok) ``policy_path``.
    Policies that fail fetch or sandbox are excluded from the job but
    still appear in *reports* so the caller can print them.
    """
    if not descriptors_path.exists():
        raise FileNotFoundError(
            f"Descriptors file not found: {descriptors_path}\n"
            "Run tests/e2e/e2e_seed_hf.py first to generate it."
        )

    descriptors = json.loads(descriptors_path.read_text())
    if not descriptors:
        raise ValueError("No policies in descriptors file.")

    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "cacheon-e2e-policy-cache"

    if baseline_cache_dir is None:
        baseline_cache_dir = str(
            Path(tempfile.gettempdir()) / "cacheon-e2e-baseline-cache"
        )

    reports: list[dict] = []
    challenger_jobs: list[ChallengerJob] = []

    for desc in descriptors:
        report = _fetch_and_precheck(desc, cache_dir, hf_token)
        reports.append(report)
        if report["status"] != "ok":
            continue
        source_hash = hash_policy_file(report["policy_path"])
        challenger_jobs.append(ChallengerJob(
            uid=len(challenger_jobs),
            hotkey=f"e2e-{report['name']}",
            commit_block=0,
            repo=report["repo"],
            revision=report["revision"],
            policy_path=report["policy_path"],
            source_hash=source_hash,
        ))

    job = EvaluationJob(
        schema_version=SCHEMA_VERSION,
        job_id=f"e2e-{uuid.uuid4().hex[:8]}",
        current_block=0,
        block_hash=block_hash,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        n_prompts=n_prompts,
        baseline_cache_dir=baseline_cache_dir,
        baseline_cache_key=f"e2e-{block_hash[:18]}",
        challengers=challenger_jobs,
    )

    return job, reports


# ── result printing ──────────────────────────────────────────────────


def print_reports(reports: list[dict]) -> None:
    """Print fetch/precheck status for each policy."""
    for r in reports:
        logger.info("  %-15s %s", r["name"], r["status"])


def print_results(result_or_records) -> None:
    """Print a human-readable table.

    Accepts either an ``EvaluationResult`` (has ``.challenger_results``)
    or a flat list of record-like objects (``EvaluationRecord``, etc.)
    — both expose ``.uid``, ``.hotkey``, ``.kl_divergence``, etc.
    """
    if hasattr(result_or_records, "challenger_results"):
        records = result_or_records.challenger_results
    else:
        records = result_or_records

    print()
    header = (
        f"{'uid':<5} {'hotkey':<20} {'kl':>8} {'mem':>6} "
        f"{'lat':>6} {'score':>7} {'dq'}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r.uid:<5} {r.hotkey:<20} "
            f"{r.kl_divergence:>8.4f} {r.memory_reduction:>6.2f} "
            f"{r.latency_improvement:>6.2f} {r.score:>7.4f} "
            f"{r.disqualify_reason or 'no'}"
        )

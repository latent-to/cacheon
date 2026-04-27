#!/usr/bin/env python3
"""E2E test — CPU-side, full SSH/SFTP pipeline to the GPU pod.

Exercises the complete production path without chain interaction:
  1. Fetch fixture policies from HuggingFace (CPU-side).
  2. AST sandbox precheck (CPU-side).
  3. Build an EvaluationJob.
  4. SFTP job.json + policy.py files to the GPU pod.
  5. SSH exec pod_eval.py on the pod.
  6. SFTP results.json back.
  7. Parse and print results.

Usage:
    export HF_TOKEN=hf_...
    python tests/e2e/e2e_cpu.py \\
        --gpu-pod-ssh-host ssh.deployments.targon.com \\
        --gpu-pod-ssh-user wrk-b6ptrqbmfkoj \\
        --n-prompts 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.e2e.e2e_common import (
    DEFAULT_DESCRIPTORS_PATH,
    build_e2e_job,
    configure_logging,
    print_reports,
    print_results,
)
from validator.eval_schema import (
    JOB_FILE_NAME,
    RESULTS_FILE_NAME,
    read_results,
    write_job,
)
from validator.pod_transport import PodTransport

logger = logging.getLogger("e2e_cpu")

_POD_STAGING_ROOT = "/tmp/cacheon-e2e"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="E2E eval — CPU-side, SSH/SFTP pipeline to GPU pod.",
    )
    p.add_argument("--policies", type=Path, default=DEFAULT_DESCRIPTORS_PATH)
    p.add_argument("--gpu-pod-ssh-host", required=True,
                   help="SSH hostname of the GPU pod.")
    p.add_argument("--gpu-pod-ssh-user", required=True,
                   help="SSH username on the GPU pod.")
    p.add_argument("--gpu-pod-ssh-port", type=int, default=22)
    p.add_argument("--gpu-pod-work-dir", default="/workspace/cacheon",
                   help="Repo checkout on the GPU pod.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--timeout", type=int, default=1200,
                   help="SSH exec timeout in seconds.")
    p.add_argument("--json", action="store_true", help="Output NDJSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configure_logging(verbose=args.verbose)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("⚠️  HF_TOKEN not set — private/gated repos will fail")

    # 1. Fetch + precheck (CPU-side)
    logger.info("── 1/7  fetch + precheck ──────────────────────────────────")
    baseline_cache_dir = "/tmp/cacheon-e2e-baseline-cache"
    job, reports = build_e2e_job(
        args.policies,
        hf_token=hf_token,
        n_prompts=args.n_prompts,
        max_new_tokens=args.max_new_tokens,
        baseline_cache_dir=baseline_cache_dir,
    )
    print_reports(reports)

    if not job.challengers:
        logger.error("❌  no challengers passed fetch + precheck — nothing to run")
        return 1

    logger.info("✅  job %s  (%d challengers)", job.job_id, len(job.challengers))

    # 2. Connect
    logger.info("── 2/7  SSH connect ───────────────────────────────────────")
    transport = PodTransport(
        host=args.gpu_pod_ssh_host,
        user=args.gpu_pod_ssh_user,
        port=args.gpu_pod_ssh_port,
    )
    transport.connect()
    logger.info(
        "✅  connected  %s@%s:%d",
        args.gpu_pod_ssh_user, args.gpu_pod_ssh_host, args.gpu_pod_ssh_port,
    )

    try:
        remote_dir = f"{_POD_STAGING_ROOT}/{job.job_id}"
        remote_job = f"{remote_dir}/{JOB_FILE_NAME}"
        remote_results = f"{remote_dir}/{RESULTS_FILE_NAME}"

        # 3. Staging dir
        logger.info("── 3/7  create staging dir ────────────────────────────")
        out, err, rc = transport.exec(f"mkdir -p {remote_dir}")
        if rc != 0:
            logger.error("❌  mkdir failed: %s", err.strip())
            return 2
        logger.info("✅  %s", remote_dir)

        # 4. Upload policies + job.json
        logger.info("── 4/7  SFTP upload ───────────────────────────────────")
        local_tmp = Path(tempfile.mkdtemp(prefix="cacheon-e2e-cpu-"))
        local_job_path = local_tmp / JOB_FILE_NAME

        from validator.eval_schema import ChallengerJob, EvaluationJob, SCHEMA_VERSION

        remote_challengers = []
        for cj in job.challengers:
            local_policy = Path(cj.policy_path)
            remote_policy = f"{remote_dir}/policy_{cj.uid}.py"
            transport.upload(local_policy, remote_policy)
            logger.info("   ↑  policy_%-2d  (%s)", cj.uid, cj.hotkey)
            remote_challengers.append(ChallengerJob(
                uid=cj.uid,
                hotkey=cj.hotkey,
                commit_block=cj.commit_block,
                repo=cj.repo,
                revision=cj.revision,
                policy_path=remote_policy,
                source_hash=cj.source_hash,
            ))

        remote_job_obj = EvaluationJob(
            schema_version=job.schema_version,
            job_id=job.job_id,
            current_block=job.current_block,
            block_hash=job.block_hash,
            model_name=job.model_name,
            max_new_tokens=job.max_new_tokens,
            n_prompts=job.n_prompts,
            baseline_cache_dir=baseline_cache_dir,
            baseline_cache_key=job.baseline_cache_key,
            challengers=remote_challengers,
        )
        write_job(remote_job_obj, local_job_path)
        transport.upload(local_job_path, remote_job)
        logger.info("   ↑  job.json")

        # 5. Run pod_eval.py over SSH
        logger.info("── 5/7  SSH exec pod_eval.py ──────────────────────────")
        venv_python = f"{args.gpu_pod_work_dir}/../venv/bin/python3"
        cmd = (
            f"cd {args.gpu_pod_work_dir} && "
            f"{venv_python} scripts/pod_eval.py "
            f"--job {remote_job} "
            f"--results-out {remote_results} "
            f"--device {args.device} --dtype {args.dtype}"
        )
        logger.info("   $ %s", cmd)
        started = time.time()
        out, err, rc = transport.exec(cmd, timeout=float(args.timeout))
        elapsed = time.time() - started

        if out.strip():
            for line in out.strip().split("\n")[-20:]:
                logger.info("   │  %s", line)
        if err.strip():
            for line in err.strip().split("\n")[-20:]:
                logger.warning("   │  %s", line)

        if rc != 0:
            logger.error("❌  pod_eval exited %d  (%.1fs)", rc, elapsed)
            return 3
        logger.info("✅  pod_eval done  (%.1fs)", elapsed)

        # 6. Download results
        logger.info("── 6/7  SFTP download ─────────────────────────────────")
        local_results = local_tmp / RESULTS_FILE_NAME
        transport.download(remote_results, local_results)
        logger.info("   ↓  results.json")

        # 7. Print
        logger.info("── 7/7  results ───────────────────────────────────────")
        result = read_results(local_results)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_results(result)

        n_scored = sum(1 for r in result.challenger_results if not r.disqualified)
        n_dq = sum(1 for r in result.challenger_results if r.disqualified)
        logger.info("✅  %d scored  %d DQ'd", n_scored, n_dq)

        # cleanup (best-effort)
        try:
            transport.exec(f"rm -rf {remote_dir}")
        except Exception:
            logger.warning("⚠️  cleanup failed for %s", remote_dir)

    finally:
        transport.close()
        logger.info("🔌  SSH closed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

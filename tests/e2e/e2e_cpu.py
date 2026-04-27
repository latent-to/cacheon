#!/usr/bin/env python3
"""E2E test — CPU-side, full SSH/SFTP pipeline to the GPU pod.

Exercises the complete production path without chain interaction:
  1. Fetch fixture policies from HuggingFace (CPU-side).
  2. AST sandbox precheck (CPU-side).
  3. Build CommitmentRecords + connect SSH.
  4. Delegate to ``make_remote_eval_fn`` (upload, exec, poll, download).
  5. Print results.

Uses the *same* transport code as production

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
from validator.chain import CommitmentRecord
from validator.eval_pod import make_remote_eval_fn
from validator.pod_transport import PodTransport

logger = logging.getLogger("e2e_cpu")


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
                   help="Poll timeout in seconds.")
    p.add_argument("--json", action="store_true", help="Output NDJSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configure_logging(verbose=args.verbose)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("⚠️  HF_TOKEN not set — private/gated repos will fail")

    # 1. Fetch + precheck (CPU-side)
    logger.info("── 1/4  fetch + precheck ──────────────────────────────────")
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

    # 2. Build CommitmentRecords + policy_source_fn from the E2E job
    policy_map: dict[str, Path] = {}
    commitments: list[CommitmentRecord] = []
    for cj in job.challengers:
        key = f"{cj.hotkey}:{cj.commit_block}"
        policy_map[key] = Path(cj.policy_path)
        commitments.append(CommitmentRecord(
            uid=cj.uid,
            hotkey=cj.hotkey,
            commit_block=cj.commit_block,
            repo=cj.repo,
            revision=cj.revision,
            raw="{}",
        ))

    def policy_source_fn(com: CommitmentRecord) -> Path:
        return policy_map[f"{com.hotkey}:{com.commit_block}"]

    # 3. Connect + run via make_remote_eval_fn
    logger.info("── 2/4  SSH connect ───────────────────────────────────────")
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

    last_tail = [""]

    def _on_poll(elapsed: float, pid: int, tail_text: str) -> None:
        stripped = tail_text.strip()
        if stripped and stripped != last_tail[0]:
            for line in stripped.splitlines():
                logger.info("   │  %s", line)
            last_tail[0] = stripped
        logger.info("   ⏳  polling… %.0fs elapsed, PID %d alive", elapsed, pid)

    logger.info("── 3/4  remote eval (detached + poll) ────────────────────")
    try:
        eval_fn = make_remote_eval_fn(
            policy_source_fn=policy_source_fn,
            transport=transport,
            pod_work_dir=args.gpu_pod_work_dir,
            baseline_cache_dir=baseline_cache_dir,
            model_name=job.model_name,
            max_new_tokens=job.max_new_tokens,
            n_prompts=job.n_prompts,
            device=args.device,
            dtype_name=args.dtype,
            timeout_s=float(args.timeout),
            on_poll=_on_poll,
        )
        records = eval_fn(
            commitments,
            current_block=job.current_block,
            block_hash=job.block_hash,
        )
    finally:
        transport.close()
        logger.info("🔌  SSH closed")

    # 4. Print results
    logger.info("── 4/4  results ───────────────────────────────────────────")
    if args.json:
        for r in records:
            print(json.dumps({
                "uid": r.uid, "hotkey": r.hotkey,
                "score": r.score, "kl": r.kl_divergence,
                "mem": r.memory_reduction, "lat": r.latency_improvement,
                "dq": r.disqualified, "reason": r.disqualify_reason,
            }))
    else:
        print_results(records)

    n_scored = sum(1 for r in records if not r.disqualified)
    n_dq = sum(1 for r in records if r.disqualified)
    logger.info("✅  %d scored  %d DQ'd", n_scored, n_dq)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

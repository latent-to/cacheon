#!/usr/bin/env python3
"""E2E test — CPU-side, subprocess boundary.

Exercises the full production contract: build job.json, spawn
``python -m scripts.pod_eval``, read results.json. On a split
deployment, override --pod-eval-cmd to SSH into the GPU pod.

Usage (on GPU pod or locally with GPU):
    export HF_TOKEN=hf_...
    python tests/e2e/e2e_cpu.py --device cuda --n-prompts 3

Usage (from CPU host, SSH to GPU pod):
    python tests/e2e/e2e_cpu.py \\
        --pod-eval-cmd "ssh gpuhost python -m scripts.pod_eval" \\
        --n-prompts 3
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validator.eval_schema import (
    read_results,
    write_job,
)

from tests.e2e.e2e_common import (
    DEFAULT_DESCRIPTORS_PATH,
    build_e2e_job,
    configure_logging,
    print_reports,
    print_results,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="E2E eval — CPU-side, spawns pod_eval as a subprocess.",
    )
    p.add_argument("--policies", type=Path, default=DEFAULT_DESCRIPTORS_PATH)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=20 * 60, help="Subprocess timeout (seconds)")
    p.add_argument(
        "--pod-eval-cmd", default=None,
        help="Override pod_eval command (e.g. 'ssh gpuhost python -m scripts.pod_eval'). "
             "Default: python -m scripts.pod_eval",
    )
    p.add_argument("--json", action="store_true", help="Output NDJSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configure_logging(verbose=args.verbose)

    logger = logging.getLogger("e2e_cpu")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set — private/gated repos will fail")

    # Build the job (fetch + precheck)
    job, reports = build_e2e_job(
        args.policies,
        hf_token=hf_token,
        n_prompts=args.n_prompts,
        max_new_tokens=args.max_new_tokens,
    )
    print_reports(reports)

    if not job.challengers:
        logger.error("No challengers passed fetch + precheck. Nothing to run.")
        return 1

    # Write job.json
    work_dir = Path(tempfile.mkdtemp(prefix="cacheon-e2e-cpu-"))
    job_path = work_dir / "job.json"
    results_path = work_dir / "results.json"
    write_job(job, job_path)
    logger.info("Wrote job.json → %s", job_path)

    # Spawn pod_eval — same subprocess boundary as production
    if args.pod_eval_cmd:
        cmd = args.pod_eval_cmd.split()
    else:
        cmd = [sys.executable, "-m", "scripts.pod_eval"]

    full_cmd = [
        *cmd,
        "--job", str(job_path),
        "--results-out", str(results_path),
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    logger.info("Launching: %s", " ".join(full_cmd))

    try:
        subprocess.run(
            full_cmd,
            check=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("pod_eval timed out after %.0fs", args.timeout)
        return 1
    except subprocess.CalledProcessError as exc:
        logger.error("pod_eval exited %d", exc.returncode)
        return 1

    if not results_path.exists():
        logger.error("pod_eval did not write results.json at %s", results_path)
        return 1

    # Read and display results
    result = read_results(results_path)
    logger.info("Read results.json — job_id=%s", result.job_id)

    if result.job_id != job.job_id:
        logger.warning(
            "results.job_id=%s != job.job_id=%s (stale cache?)",
            result.job_id, job.job_id,
        )

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_results(result)

    n_scored = sum(1 for r in result.challenger_results if not r.disqualified)
    n_dq = sum(1 for r in result.challenger_results if r.disqualified)
    logger.info("Done: %d scored, %d DQ'd", n_scored, n_dq)
    logger.info("Artifacts: %s", work_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E2E test — GPU-side, in-process.

Runs directly on the GPU pod. Calls pod_eval.run_job() in-process so
every code path that production uses (baseline caching, per-challenger
try/except → DQ, scoring) is exercised without a subprocess boundary.

Usage (on GPU pod, after e2e_seed_hf.py):
    export HF_TOKEN=hf_...
    python tests/e2e/e2e_pod.py --device cuda --n-prompts 3
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="E2E eval — GPU-side, in-process via pod_eval.run_job().",
    )
    p.add_argument("--policies", type=Path, default=DEFAULT_DESCRIPTORS_PATH)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--json", action="store_true", help="Output NDJSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configure_logging(verbose=args.verbose)

    import logging
    logger = logging.getLogger("e2e_pod")

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

    # Run pod_eval in-process — identical to production
    from scripts.pod_eval import run_job

    logger.info(
        "Running pod_eval.run_job() with %d challenger(s) …",
        len(job.challengers),
    )
    result = run_job(job, device=args.device)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_results(result)

    # Summary
    n_scored = sum(1 for r in result.challenger_results if not r.disqualified)
    n_dq = sum(1 for r in result.challenger_results if r.disqualified)
    logger.info("Done: %d scored, %d DQ'd", n_scored, n_dq)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

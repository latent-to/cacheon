#!/usr/bin/env python3
"""Ad-hoc E2E test: download policies from HF, run harness, score.

NOT FOR CI — requires GPU + HF token + ~14 GB model download.
Uses the actual validator modules (policy_fetch, precheck) so it tests
the real pipeline without touching the chain.

Usage (after running e2e_seed_hf.py):
    export HF_TOKEN=hf_...
    python scripts/e2e_eval.py --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_engine.harness import Harness
from inference_engine.passthrough import PassthroughPolicy
from inference_engine.prompts import sample_prompts
from inference_engine.scoring import score
from validator.chain import CommitmentRecord
from validator.policy_fetch import FetchOutcome, fetch_policy_source
from validator.precheck import make_fetch_precheck

logger = logging.getLogger("e2e_eval")

DEFAULT_DESCRIPTORS_PATH = REPO_ROOT / "scripts" / "example_policies.json"


def _configure_logging(*, verbose: bool = False) -> None:
    """Bump noisy third-party loggers to WARNING so e2e_eval / inference_engine INFO shines through."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "huggingface_hub", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_policy_class(path: Path) -> type:
    """Dynamically load a policy.py and return its KVCachePolicy subclass."""
    spec = importlib.util.spec_from_file_location("policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from inference_engine.policy import KVCachePolicy

    candidates = [
        getattr(module, name)
        for name in dir(module)
        if isinstance(getattr(module, name), type)
        and issubclass(getattr(module, name), KVCachePolicy)
        and getattr(module, name) is not KVCachePolicy
    ]
    if not candidates:
        raise ValueError(f"No KVCachePolicy subclass found in {path}")
    return candidates[0]


def _run_one(
    descriptor: dict[str, Any],
    harness: Harness,
    prompts: list[str],
    cache_dir: Path,
    hf_token: str | None,
) -> dict[str, Any]:
    name = descriptor["name"]
    repo = descriptor["repo"]
    revision = descriptor["revision"]

    # 1. Fetch (tests policy_fetch.py)
    logger.info("[%s] Fetching %s @ %s ...", name, repo, revision[:8])
    fetch_result = fetch_policy_source(
        repo,
        revision,
        cache_dir=cache_dir,
        max_bytes=1_048_576,
        etag_timeout_s=30.0,
        hf_token=hf_token,
    )
    if fetch_result.outcome is not FetchOutcome.OK:
        logger.warning("[%s] Fetch %s: %s", name, fetch_result.outcome.value, fetch_result.reason)
        return {
            "name": name,
            "fetch": fetch_result.outcome.value,
            "fetch_reason": fetch_result.reason,
            "sandbox": "skipped",
            "kl": None,
            "mem_red": None,
            "lat_imp": None,
            "score": None,
            "dq": f"fetch_{fetch_result.outcome.value}",
        }

    # 2. Precheck (tests precheck.py + sandbox.py)
    com = CommitmentRecord(
        uid=0,
        hotkey="e2e",
        commit_block=0,
        repo=repo,
        revision=revision,
        raw="{}",
    )
    precheck = make_fetch_precheck(lambda _r, _rev: fetch_result)
    precheck_result = precheck(com)
    if precheck_result.outcome.value != "ok":
        logger.warning(
            "[%s] Sandbox %s: %s",
            name,
            precheck_result.outcome.value,
            precheck_result.reason,
        )
        return {
            "name": name,
            "fetch": "ok",
            "sandbox": precheck_result.outcome.value,
            "sandbox_reason": precheck_result.reason,
            "kl": None,
            "mem_red": None,
            "lat_imp": None,
            "score": None,
            "dq": f"sandbox_{precheck_result.outcome.value}",
        }

    # 3. Load policy dynamically
    PolicyClass = _load_policy_class(fetch_result.path)
    logger.info("[%s] Loaded policy class: %s", name, PolicyClass.__name__)

    # 4. Run harness
    logger.info("[%s] Running harness ...", name)
    baseline = harness.run(PassthroughPolicy(), prompts)
    miner = harness.run(PolicyClass(), prompts)

    # 5. Score
    result = score(baseline, miner)
    logger.info(
        "[%s] KL=%.4f mem=%.2f lat=%.2f score=%.4f dq=%s",
        name,
        result.kl_divergence,
        result.memory_reduction,
        result.latency_improvement,
        result.score,
        result.disqualify_reason or "no",
    )

    return {
        "name": name,
        "fetch": "ok",
        "sandbox": "ok",
        "kl": result.kl_divergence,
        "mem_red": result.memory_reduction,
        "lat_imp": result.latency_improvement,
        "score": result.score,
        "dq": result.disqualify_reason or "no",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ad-hoc E2E eval of miner policies against the harness.",
    )
    p.add_argument(
        "--policies",
        type=Path,
        default=DEFAULT_DESCRIPTORS_PATH,
        help="JSON file with policy descriptors (default: scripts/example_policies.json)",
    )
    p.add_argument("--device", default="cuda", help="torch device (cuda/cpu)")
    p.add_argument("--n-prompts", type=int, default=3, help="Number of prompts")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--json", action="store_true", help="Output NDJSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set — private/gated repos will fail")

    if not args.policies.exists():
        logger.error("Descriptors file not found: %s", args.policies)
        logger.error("Run scripts/e2e_seed_hf.py first to generate it.")
        return 1

    descriptors = json.loads(args.policies.read_text())
    if not descriptors:
        logger.error("No policies in descriptors file.")
        return 1

    cache_dir = Path(tempfile.gettempdir()) / "cacheon-e2e-policy-cache"

    logger.info("Loading model Qwen/Qwen2.5-7B-Instruct on %s ...", args.device)
    harness = Harness(device=args.device)

    logger.info("Sampling %d prompts ...", args.n_prompts)
    prompts = sample_prompts(block_hash="0x" + "aa" * 32, n=args.n_prompts)

    results: list[dict[str, Any]] = []
    for desc in descriptors:
        result = _run_one(desc, harness, prompts, cache_dir, hf_token)
        results.append(result)

    # Print
    if args.json:
        for r in results:
            print(json.dumps(r))
    else:
        print()
        header = f"{'policy':<15} {'fetch':<8} {'sandbox':<8} {'kl':>8} {'mem':>6} {'lat':>6} {'score':>7} {'dq'}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['name']:<15} {r['fetch']:<8} {r['sandbox']:<8} "
                f"{r['kl'] or 0:>8.4f} {r['mem_red'] or 0:>6.2f} {r['lat_imp'] or 0:>6.2f} "
                f"{r['score'] or 0:>7.4f} {r['dq']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

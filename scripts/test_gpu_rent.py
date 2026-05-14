#!/usr/bin/env python3
"""Quick GPU rent -> SSH -> setup -> teardown test.

Skips chain, S3, and the full validator loop. Run directly on the CPU host:

    cd ~/cacheon
    source venv-cacheon/bin/activate   # or your venv
    python scripts/test_gpu_rent.py

Set LIUM_API_KEY and/or TARGON_API_KEY in the environment (or validator/.env).
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("test_gpu_rent")

# Load .env if present
env_path = os.path.join(os.path.dirname(__file__), "..", "validator", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from validator.providers import search_all_providers

SETUP_BRANCH = "feat/unified-validator-gpu-orchestrator"
SETUP_SCRIPT_URL = f"https://raw.githubusercontent.com/latent-to/cacheon/{SETUP_BRANCH}/validator/setup-gpu.sh"


def main():
    lium_key = os.environ.get("LIUM_API_KEY", "")
    targon_key = os.environ.get("TARGON_API_KEY", "")

    providers = []
    if lium_key:
        from validator.providers.lium_provider import LiumProvider

        providers.append(LiumProvider(lium_key))
    if targon_key:
        from validator.providers.targon_provider import TargonProvider

        providers.append(TargonProvider(targon_key))

    if not providers:
        logger.error("No API keys set (LIUM_API_KEY / TARGON_API_KEY)")
        sys.exit(1)

    prefer = os.environ.get("PREFER_PROVIDER", "").lower()
    max_price = int(os.environ.get("CACHEON_MAX_HOURLY_PRICE", "5000"))

    logger.info("Searching for GPUs (max $%.2f/hr)...", max_price / 100)
    best = search_all_providers(providers, max_hourly_price_cents=max_price)
    if best is None:
        logger.error("No eligible GPU found")
        sys.exit(1)

    if prefer and best.provider != prefer:
        logger.info("Skipping %s (PREFER_PROVIDER=%s)", best.provider, prefer)
        filtered = [p for p in providers if p.name == prefer]
        best = search_all_providers(filtered, max_hourly_price_cents=max_price)
        if best is None:
            logger.error("No eligible GPU from %s", prefer)
            sys.exit(1)

    provider = next(p for p in providers if p.name == best.provider)
    logger.info(
        "Best: %s %s (%dx %s) $%.2f/hr",
        best.provider,
        best.instance_id,
        best.num_gpus,
        best.gpu_type,
        best.hourly_price_cents / 100,
    )

    handle = None
    try:
        logger.info("=== RENTING ===")
        handle = provider.rent(best)
        logger.info("Pod ID: %s", handle.pod_id)

        logger.info("=== WAITING FOR READY ===")
        handle = provider.wait_ready(handle, timeout_s=600)
        logger.info("Pod ready!")

        logger.info("=== TEST: whoami + nvidia-smi ===")
        result = provider.exec(
            handle,
            "whoami && hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        )
        print(result.get("stdout", ""))
        if not result.get("success"):
            logger.error("Basic exec failed: %s", result.get("stderr", ""))
            return

        logger.info("=== RUNNING SETUP ===")
        hf_token = os.environ.get("HF_TOKEN", "")
        env_parts = [
            f'export CACHEON_BRANCH="{SETUP_BRANCH}"',
        ]
        if hf_token:
            env_parts.append(f'export HF_TOKEN="{hf_token}"')
        env_prefix = " && ".join(env_parts)

        setup_cmd = f'{env_prefix} && curl -fsSL "{SETUP_SCRIPT_URL}" | bash'
        logger.info("Running: %s", setup_cmd[:120] + "...")
        t0 = time.monotonic()
        result = provider.exec(handle, setup_cmd)
        elapsed = time.monotonic() - t0
        logger.info(
            "Setup finished in %.0fs (exit=%s)", elapsed, result.get("exit_code")
        )

        stdout = result.get("stdout", "")
        if stdout:
            for line in stdout.splitlines()[-30:]:
                logger.info("  [setup] %s", line)
        if not result.get("success"):
            logger.error("stderr: %s", result.get("stderr", "")[:2000])
            return

        logger.info("=== SETUP OK ===")

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception:
        logger.exception("Failed")
    finally:
        if handle is not None:
            logger.info("=== TEARING DOWN ===")
            provider.teardown(handle)
            logger.info("Done")


if __name__ == "__main__":
    main()

"""Automated GPU pod lifecycle: search, rent, setup, eval, teardown.

Called by ``cpu_validator.run_tick()`` when ``CACHEON_AUTO_RENT=1`` and
there are new challengers. The GPU pod is ephemeral: rent it, run one
eval cycle, and tear it down.

The eval itself still runs on the remote pod via ``gpu_eval.py`` inside
Docker Compose. S3 remains the handoff mechanism: the remote pod
uploads ``state.json`` after eval, and the CPU process downloads it
after teardown.
"""

from __future__ import annotations

import logging
import time

from . import config as validator_config
from .eval_schema import EvalJob
from .providers import GpuInstance, GpuProvider, PodHandle, search_all_providers

logger = logging.getLogger(__name__)

SETUP_SCRIPT_URL = (
    "https://raw.githubusercontent.com/latent-to/cacheon/main/validator/setup.sh"
)


def _build_providers() -> list[GpuProvider]:
    """Instantiate providers that have API keys configured."""
    providers: list[GpuProvider] = []

    if validator_config.LIUM_API_KEY:
        from .providers.lium_provider import LiumProvider

        providers.append(LiumProvider(validator_config.LIUM_API_KEY))

    if validator_config.TARGON_API_KEY:
        from .providers.targon_provider import TargonProvider

        providers.append(TargonProvider(validator_config.TARGON_API_KEY))

    return providers


def _find_provider_for_instance(
    providers: list[GpuProvider], instance: GpuInstance
) -> GpuProvider | None:
    for p in providers:
        if p.name == instance.provider:
            return p
    return None


def _remote_setup(provider: GpuProvider, handle: PodHandle) -> bool:
    """Run setup.sh on the remote pod. Returns True on success."""
    logger.info("Running setup.sh on remote pod %s", handle.pod_id)

    hf_token = validator_config.HF_TOKEN
    hf_export = f"export HF_TOKEN={hf_token} && " if hf_token else ""

    setup_cmd = (
        f"{hf_export}"
        f"apt-get update -qq && apt-get install -y -qq curl git > /dev/null 2>&1 && "
        f'curl -fsSL "{SETUP_SCRIPT_URL}" | bash'
    )

    result = provider.exec(handle, setup_cmd)
    if not result.get("success", False):
        logger.error(
            "setup.sh failed on pod %s (exit=%s): %s",
            handle.pod_id,
            result.get("exit_code"),
            result.get("stderr", "")[:500],
        )
        return False

    logger.info("setup.sh completed on pod %s", handle.pod_id)
    return True


def _remote_configure_env(
    provider: GpuProvider,
    handle: PodHandle,
) -> bool:
    """Write .env on the remote pod with S3 credentials and GPU config."""
    env_lines = [
        f"HIPPIUS_ACCESS_KEY={validator_config.HIPPIUS_ACCESS_KEY}",
        f"HIPPIUS_SECRET_KEY={validator_config.HIPPIUS_SECRET_KEY}",
        f"CACHEON_S3_BUCKET={validator_config.S3_BUCKET}",
        f"CACHEON_S3_PREFIX={validator_config.S3_PREFIX}",
        f"CACHEON_GPU_COUNT={handle.gpu_count}",
        "CACHEON_MODEL_VOLUME=/workspace/models/Qwen2.5-72B-Instruct",
        "CACHEON_BASELINE_IMAGE=vllm/vllm-openai:latest",
    ]

    env_content = "\\n".join(env_lines)
    cmd = f'printf "{env_content}\\n" > ~/cacheon/validator/.env'
    result = provider.exec(handle, cmd)
    if not result.get("success", False):
        logger.error("Failed to write .env on pod %s", handle.pod_id)
        return False

    logger.info(".env configured on pod %s", handle.pod_id)
    return True


def _remote_run_eval(
    provider: GpuProvider,
    handle: PodHandle,
    timeout_min: int,
) -> bool:
    """Run docker compose eval on the remote pod. Returns True on success."""
    logger.info(
        "Starting GPU eval on pod %s (timeout=%d min)", handle.pod_id, timeout_min
    )

    cmd = (
        "cd ~/cacheon && "
        "set -a && . validator/.env && set +a && "
        "docker compose -f validator/docker-compose.yml up --build 2>&1"
    )

    result = provider.exec(handle, cmd)
    exit_code = result.get("exit_code", -1)

    if exit_code == 0:
        logger.info("GPU eval completed successfully on pod %s", handle.pod_id)
        return True

    logger.error(
        "GPU eval failed on pod %s (exit=%d): %s",
        handle.pod_id,
        exit_code,
        result.get("stderr", "")[:500],
    )
    return False


def run_gpu_eval(state_dir: str, eval_job: EvalJob) -> bool:
    """Search for a GPU pod, rent it, run eval, tear it down.

    Returns True if the eval completed successfully and results are on S3.
    """
    providers = _build_providers()
    if not providers:
        logger.error(
            "No GPU provider API keys configured (LIUM_API_KEY / TARGON_API_KEY)"
        )
        return False

    max_price = validator_config.MAX_HOURLY_PRICE
    timeout_min = validator_config.GPU_TIMEOUT_MIN

    # Search
    best = search_all_providers(providers, max_hourly_price_cents=max_price)
    if best is None:
        logger.warning(
            "No GPU instances available matching tier requirements (max $%.2f/hr)",
            max_price / 100,
        )
        return False

    provider = _find_provider_for_instance(providers, best)
    if provider is None:
        logger.error(
            "Could not find provider %s for instance %s",
            best.provider,
            best.instance_id,
        )
        return False

    logger.info(
        "Best GPU match: %s %s (%dx %s) $%.2f/hr",
        best.provider,
        best.instance_id,
        best.num_gpus,
        best.gpu_type,
        best.hourly_price_cents / 100,
    )

    handle: PodHandle | None = None
    try:
        # Rent
        logger.info("Renting pod from %s...", provider.name)
        handle = provider.rent(best)
        logger.info("Pod rented: %s (id=%s)", provider.name, handle.pod_id)

        # Wait ready
        handle = provider.wait_ready(handle, timeout_s=600)
        logger.info("Pod %s is ready", handle.pod_id)

        # Setup
        if not _remote_setup(provider, handle):
            return False

        # Configure .env
        if not _remote_configure_env(provider, handle):
            return False

        # Run eval
        return _remote_run_eval(provider, handle, timeout_min)

    except TimeoutError as exc:
        logger.error("GPU orchestration timed out: %s", exc)
        return False
    except Exception as exc:
        logger.exception("GPU orchestration failed: %s", exc)
        return False
    finally:
        if handle is not None:
            logger.info("Tearing down pod %s...", handle.pod_id)
            provider.teardown(handle)

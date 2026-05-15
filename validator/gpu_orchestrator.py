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

SETUP_BRANCH = "feat/unified-validator-gpu-orchestrator"
SETUP_SCRIPT_URL = f"https://raw.githubusercontent.com/latent-to/cacheon/{SETUP_BRANCH}/validator/setup-gpu.sh"


def _build_providers() -> list[GpuProvider]:
    """Instantiate providers that have API keys configured.

    When ``CACHEON_PREFERRED_PROVIDER`` is set (e.g. 'targon' or 'lium'),
    only that provider is instantiated even if both keys exist.
    """
    pref = validator_config.PREFERRED_PROVIDER.lower().strip()
    providers: list[GpuProvider] = []

    if validator_config.LIUM_API_KEY and pref in ("", "lium"):
        from .providers.lium_provider import LiumProvider

        providers.append(LiumProvider(validator_config.LIUM_API_KEY))

    if validator_config.TARGON_API_KEY and pref in ("", "targon"):
        from .providers.targon_provider import TargonProvider

        providers.append(TargonProvider(validator_config.TARGON_API_KEY))

    if pref and not providers:
        logger.warning(
            "CACHEON_PREFERRED_PROVIDER=%s but no matching API key is configured",
            pref,
        )

    return providers


def _find_provider_for_instance(
    providers: list[GpuProvider], instance: GpuInstance
) -> GpuProvider | None:
    for p in providers:
        if p.name == instance.provider:
            return p
    return None


def _dq_escape(v: str) -> str:
    """Escape a value for safe use inside double-quoted shell strings."""
    return (
        v.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )


def _build_env_exports(handle: PodHandle) -> str:
    """Build 'export K=V && ...' prefix for remote shell commands."""
    env: dict[str, str] = {
        "CACHEON_BRANCH": SETUP_BRANCH,
        "HIPPIUS_ACCESS_KEY": validator_config.HIPPIUS_ACCESS_KEY,
        "HIPPIUS_SECRET_KEY": validator_config.HIPPIUS_SECRET_KEY,
        "CACHEON_S3_BUCKET": validator_config.S3_BUCKET,
        "CACHEON_S3_PREFIX": validator_config.S3_PREFIX,
        "CACHEON_GPU_COUNT": str(handle.gpu_count),
        "CACHEON_MODEL_VOLUME": "/workspace/models/Qwen2.5-72B-Instruct",
        "CACHEON_BASELINE_IMAGE": "vllm/vllm-openai:latest",
    }
    hf_token = validator_config.HF_TOKEN
    if hf_token:
        env["HF_TOKEN"] = hf_token

    return " && ".join(f'export {k}="{_dq_escape(v)}"' for k, v in env.items())


def _log_tail(label: str, result: dict, n: int = 30) -> None:
    """Log the last *n* lines of stdout from a remote exec result."""
    stdout = result.get("stdout", "")
    if stdout:
        for line in stdout.splitlines()[-n:]:
            logger.info("  [%s] %s", label, line)


def _extract_chunk_text(chunk: dict[str, str]) -> str:
    """Pull text from a stream_exec chunk regardless of provider format."""
    return chunk.get("data", "") or chunk.get("stdout", "") or chunk.get("stderr", "")


_HEARTBEAT_INTERVAL = 30  # seconds


def _remote_setup(provider: GpuProvider, handle: PodHandle) -> bool:
    """Run setup-gpu.sh on the remote pod, streaming progress to logs."""
    logger.info("⏳ Running setup.sh on remote pod %s", handle.pod_id)

    env_exports = _build_env_exports(handle)
    setup_cmd = f'{env_exports} && curl -fsSL "{SETUP_SCRIPT_URL}" | bash'

    t0 = time.monotonic()
    last_heartbeat = t0
    full_output: list[str] = []

    for chunk in provider.stream_exec(handle, setup_cmd):
        text = _extract_chunk_text(chunk)
        if not text:
            continue
        full_output.append(text)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("==="):
                logger.info("  [setup] %s", stripped)
            elif stripped.startswith("ERROR"):
                logger.warning("  [setup] %s", stripped)
        now = time.monotonic()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
            logger.info("  [setup] still running (%.0fs elapsed)...", now - t0)
            last_heartbeat = now

    elapsed = time.monotonic() - t0
    output = "".join(full_output)

    if "Setup complete" not in output:
        logger.error(
            "setup.sh did not reach 'Setup complete' on pod %s (%.0fs elapsed)",
            handle.pod_id,
            elapsed,
        )
        for line in output.splitlines()[-30:]:
            logger.info("  [setup] %s", line)
        return False

    logger.info("🛠 setup.sh completed on pod %s (%.0fs)", handle.pod_id, elapsed)
    return True


def _remote_run_eval(
    provider: GpuProvider,
    handle: PodHandle,
    timeout_min: int,
) -> bool:
    """cd into the cloned repo and run docker compose -f gpu-compose.yml up --build."""
    logger.info(
        "⏳ Starting GPU eval on pod %s (timeout=%d min)", handle.pod_id, timeout_min
    )

    env_exports = _build_env_exports(handle)
    timeout_s = timeout_min * 60
    cmd = (
        f"{env_exports} && "
        f"cd ~/cacheon/validator && "
        f"timeout --signal=KILL {timeout_s} "
        f"docker compose -f gpu-compose.yml up --build 2>&1"
    )

    result = provider.exec(handle, cmd)
    _log_tail("eval", result, n=50)

    exit_code = result.get("exit_code", -1)

    if exit_code == 137:
        logger.error(
            "GPU eval KILLED by timeout (%d min) on pod %s",
            timeout_min,
            handle.pod_id,
        )
        return False

    if exit_code == 0:
        logger.info("🎉 GPU eval completed successfully on pod %s", handle.pod_id)
        return True

    stderr = result.get("stderr", "")
    logger.error(
        "❌ GPU eval failed on pod %s (exit=%d)\nstderr (last 2000 chars):\n%s",
        handle.pod_id,
        exit_code,
        stderr[-2000:],
    )
    return False


def run_gpu_eval(state_dir: str, eval_job: EvalJob) -> bool:
    """Search for a GPU pod, rent it, run eval, tear it down.

    Returns True if the eval completed successfully and results are on S3.
    """
    providers = _build_providers()
    if not providers:
        logger.error(
            "❌ No GPU provider API keys configured (LIUM_API_KEY / TARGON_API_KEY)"
        )
        return False

    max_price = validator_config.MAX_HOURLY_PRICE
    timeout_min = validator_config.GPU_TIMEOUT_MIN

    # Search
    best = search_all_providers(providers, max_hourly_price_cents=max_price)
    if best is None:
        logger.warning(
            "⚠️ No GPU instances available matching tier requirements (max $%.2f/hr)",
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
        "🎯 Best GPU match: %s %s (%dx %s) $%.2f/hr",
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
        logger.info("☑️ Pod rented: %s (id=%s)", provider.name, handle.pod_id)

        # Wait ready
        handle = provider.wait_ready(handle, timeout_s=120)
        logger.info("☑️ Pod %s is ready", handle.pod_id)

        # Step 1: curl setup-gpu.sh | sudo -E bash
        if not _remote_setup(provider, handle):
            return False

        # Step 2: cd ~/cacheon/validator && docker compose -f gpu-compose.yml up --build
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

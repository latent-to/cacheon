"""Lium GPU cloud provider using the lium-sdk."""

from __future__ import annotations

import logging
from typing import Any, Generator

from . import GpuInstance, GpuProvider, PodHandle

logger = logging.getLogger(__name__)

VOLUME_NAME = "cacheon-sn14-volume"

_VRAM_GB: dict[str, int] = {
    "H200": 141,
    "H100": 80,
    "A100": 80,
    "B200": 180,
}


class LiumProvider:
    """GpuProvider backed by ``lium-sdk``."""

    name = "lium"

    def __init__(self, api_key: str) -> None:
        from lium.sdk import Lium, Config

        self._client = Lium(config=Config(api_key=api_key))
        self._volume_id: str | None = None

    def search(self) -> list[GpuInstance]:
        executors = self._client.ls(gpu_count=None)
        out: list[GpuInstance] = []
        for ex in executors:
            gpu_type = ex.gpu_type or ""
            vram = _VRAM_GB.get(gpu_type.upper(), 0)
            if not vram:
                continue
            if not ex.docker_in_docker:
                continue

            out.append(
                GpuInstance(
                    provider="lium",
                    instance_id=ex.id,
                    description=ex.machine_name,
                    hourly_price_cents=int(round(ex.price_per_hour * 100)),
                    num_gpus=ex.gpu_count,
                    gpu_type=gpu_type,
                    vram_per_gpu_gb=vram,
                    total_vram_gb=ex.gpu_count * vram,
                    storage_gb=0,
                    memory_gb=0,
                    vcpus=0,
                    docker_in_docker=ex.docker_in_docker,
                    raw=ex,
                )
            )
        return out

    def _ensure_volume(self) -> str | None:
        """Find or create the persistent model volume. Returns volume ID."""
        if self._volume_id is not None:
            return self._volume_id

        try:
            for vol in self._client.volumes():
                if vol.name == VOLUME_NAME:
                    self._volume_id = vol.id
                    logger.info("Reusing Lium volume %s (id=%s)", VOLUME_NAME, vol.id)
                    return self._volume_id

            vol = self._client.volume_create(
                VOLUME_NAME,
                description="Cacheon SN14 model weights, dataset cache, and Docker layers",
            )
            self._volume_id = vol.id
            logger.info("Created Lium volume %s (id=%s)", VOLUME_NAME, vol.id)
            return self._volume_id
        except Exception as exc:
            logger.warning("Could not ensure Lium volume: %s (continuing without)", exc)
            return None

    def rent(self, instance: GpuInstance) -> PodHandle:
        volume_id = self._ensure_volume()
        pod_data = self._client.up(
            executor_id=instance.instance_id,
            name="cacheon-eval",
            volume_id=volume_id,
        )
        pod_id = pod_data.get("id") if isinstance(pod_data, dict) else str(pod_data)
        return PodHandle(
            provider="lium",
            pod_id=pod_id,
            gpu_count=instance.num_gpus,
            hourly_price_cents=instance.hourly_price_cents,
            raw=pod_data,
        )

    def wait_ready(self, handle: PodHandle, timeout_s: int = 600) -> PodHandle:
        pod = self._client.wait_ready(handle.pod_id, timeout=timeout_s)
        if pod is None:
            raise TimeoutError(f"Lium pod {handle.pod_id} not ready after {timeout_s}s")
        handle.raw = pod
        return handle

    def exec(self, handle: PodHandle, command: str) -> dict[str, Any]:
        pod = handle.raw
        return self._client.exec(pod, command=command)

    def stream_exec(
        self, handle: PodHandle, command: str
    ) -> Generator[dict[str, str], None, None]:
        pod = handle.raw
        yield from self._client.stream_exec(pod, command=command)

    def teardown(self, handle: PodHandle) -> None:
        try:
            pod = handle.raw
            self._client.down(pod)
            logger.info("Lium pod %s terminated", handle.pod_id)
        except Exception as exc:
            logger.error("Lium teardown failed for %s: %s", handle.pod_id, exc)

    def balance(self) -> float:
        return self._client.balance()

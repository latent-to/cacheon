"""Lium GPU cloud provider using the lium-sdk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator

from . import GpuInstance, GpuProvider, PodHandle

logger = logging.getLogger(__name__)

VOLUME_NAME = "cacheon-sn14-volume"
SSH_KEY_NAME = "cacheon-cpu-validator"
SSH_KEY_PATHS = [
    Path.home() / ".ssh" / "id_ed25519.pub",
    Path.home() / ".ssh" / "id_rsa.pub",
]

_VRAM_GB: dict[str, int] = {
    "B300": 288,
    "B200": 180,
    "H200": 141,
    "H100": 80,
    "A100": 80,
}


def _lookup_vram(gpu_type_raw: str) -> tuple[str, int]:
    """Return (canonical_type, vram_gb). Handles variants like 'H200 NVL'."""
    t = gpu_type_raw.upper()
    for canon, vram in _VRAM_GB.items():
        if canon in t:
            return canon, vram
    return "", 0


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
            gpu_type_raw = ex.gpu_type or ""
            canon, vram = _lookup_vram(gpu_type_raw)
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
                    gpu_type=canon,
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

    def _ensure_ssh_key(self) -> str | None:
        """Ensure our SSH public key is registered with Lium. Returns the key string."""
        pub_key = None
        for path in SSH_KEY_PATHS:
            if path.exists():
                pub_key = path.read_text().strip()
                break

        if not pub_key:
            logger.warning(
                "No SSH public key found at %s",
                " or ".join(str(p) for p in SSH_KEY_PATHS),
            )
            return None

        existing = self._client.list_ssh_keys()
        for key in existing:
            if key.public_key.strip() == pub_key:
                logger.info("SSH key already registered with Lium (id=%s)", key.id)
                return pub_key

        self._client.register_ssh_key(name=SSH_KEY_NAME, public_key=pub_key)
        logger.info("Registered SSH key %s with Lium", SSH_KEY_NAME)
        return pub_key

    def rent(self, instance: GpuInstance) -> PodHandle:
        volume_id = self._ensure_volume()
        ssh_pub = self._ensure_ssh_key()

        up_kwargs: dict[str, Any] = {
            "executor_id": instance.instance_id,
            "name": "cacheon-eval",
            "volume_id": volume_id,
        }
        if ssh_pub:
            up_kwargs["ssh_keys"] = [ssh_pub]

        pod_data = self._client.up(**up_kwargs)
        pod_id = pod_data.get("id") if isinstance(pod_data, dict) else str(pod_data)
        return PodHandle(
            provider="lium",
            pod_id=pod_id,
            gpu_count=instance.num_gpus,
            hourly_price_cents=instance.hourly_price_cents,
            raw=pod_data,
        )

    def wait_ready(self, handle: PodHandle, timeout_s: int = 600) -> PodHandle:
        logger.info(
            "⏳ Waiting for Lium pod %s to be ready (timeout=%ds)...",
            handle.pod_id,
            timeout_s,
        )
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

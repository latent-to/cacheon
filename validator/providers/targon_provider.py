"""Targon GPU cloud provider using targon-sdk for rentals."""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

import requests

from . import GpuInstance, GpuProvider, PodHandle

logger = logging.getLogger(__name__)

_VRAM_GB: dict[str, int] = {
    "H200": 141,
    "H100": 80,
    "A100": 80,
    "B200": 180,
}


def _normalize_gpu_type(raw: str) -> str:
    for prefix in ("NVIDIA-GeForce-", "NVIDIA-"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


class TargonProvider:
    """GpuProvider backed by ``targon-sdk`` for rentals and raw API for search."""

    name = "targon"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def search(self) -> list[GpuInstance]:
        resp = requests.get(
            "https://api.targon.com/tha/v2/inventory",
            headers={"Authorization": f"Bearer {self._api_key}"},
            params={"type": "rental", "gpu": "true"},
            timeout=30,
        )
        resp.raise_for_status()

        out: list[GpuInstance] = []
        for item in resp.json():
            spec = item.get("spec", {})
            gpu_type = _normalize_gpu_type(spec.get("gpu_type", ""))
            gpu_count = spec.get("gpu_count", 0)
            available = item.get("available", 0)

            if available <= 0:
                continue

            vram = _VRAM_GB.get(gpu_type, 0)
            if not vram:
                continue

            storage_mb = spec.get("storage", 0)
            storage_gb = storage_mb // 1024 if storage_mb else 0
            price_dollars = item.get("cost_per_hour", 0)
            price_cents = int(round(price_dollars * 100))
            memory_mb = spec.get("memory", 0)

            out.append(
                GpuInstance(
                    provider="targon",
                    instance_id=item.get("name", ""),
                    description=item.get("display_name", ""),
                    hourly_price_cents=price_cents,
                    num_gpus=gpu_count,
                    gpu_type=gpu_type,
                    vram_per_gpu_gb=vram,
                    total_vram_gb=gpu_count * vram,
                    storage_gb=storage_gb,
                    memory_gb=memory_mb // 1024 if memory_mb else 0,
                    vcpus=spec.get("vcpu", 0),
                    docker_in_docker=True,
                    raw=item,
                )
            )
        return out

    def rent(self, instance: GpuInstance) -> PodHandle:
        from targon import Targon

        client = Targon(api_key=self._api_key)

        workload = client.workloads.create(
            name="cacheon-eval",
            image="ghcr.io/manifold-inc/ubuntu-systemd-docker:v3",
            resource_name=instance.instance_id,
            type="RENTAL",
        )
        workload_id = workload.id if hasattr(workload, "id") else str(workload)
        return PodHandle(
            provider="targon",
            pod_id=workload_id,
            gpu_count=instance.num_gpus,
            hourly_price_cents=instance.hourly_price_cents,
            raw={"workload": workload, "client": client},
        )

    def wait_ready(self, handle: PodHandle, timeout_s: int = 600) -> PodHandle:
        client = handle.raw["client"]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            workload = client.workloads.get(handle.pod_id)
            state = getattr(workload, "state", None) or ""
            if str(state).upper() == "RUNNING":
                ssh_info = getattr(workload, "ssh", None)
                if ssh_info:
                    handle.raw["ssh"] = ssh_info
                    handle.raw["workload"] = workload
                    return handle
            time.sleep(10)
        raise TimeoutError(
            f"Targon workload {handle.pod_id} not ready after {timeout_s}s"
        )

    def exec(self, handle: PodHandle, command: str) -> dict[str, Any]:
        import paramiko

        ssh_info = handle.raw.get("ssh", {})
        host = ssh_info.get("host", "")
        port = ssh_info.get("port", 22)
        user = ssh_info.get("user", "root")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(host, port=port, username=user, timeout=30)
            stdin, stdout, stderr = client.exec_command(command, timeout=3600)
            exit_code = stdout.channel.recv_exit_status()
            return {
                "stdout": stdout.read().decode("utf-8", errors="replace"),
                "stderr": stderr.read().decode("utf-8", errors="replace"),
                "exit_code": exit_code,
                "success": exit_code == 0,
            }
        finally:
            client.close()

    def stream_exec(
        self, handle: PodHandle, command: str
    ) -> Generator[dict[str, str], None, None]:
        result = self.exec(handle, command)
        if result["stdout"]:
            yield {"type": "stdout", "data": result["stdout"]}
        if result["stderr"]:
            yield {"type": "stderr", "data": result["stderr"]}

    def teardown(self, handle: PodHandle) -> None:
        try:
            client = handle.raw.get("client")
            if client:
                client.workloads.delete(handle.pod_id)
                logger.info("Targon workload %s deleted", handle.pod_id)
        except Exception as exc:
            logger.error("Targon teardown failed for %s: %s", handle.pod_id, exc)

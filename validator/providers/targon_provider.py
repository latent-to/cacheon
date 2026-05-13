"""Targon GPU cloud provider using the v2 REST API.

Supports persistent volumes (cacheon-sn14-volume) so model weights survive
across eval cycles.  Uses a register-then-deploy workload pattern with an
auto-generated SSH keypair for paramiko exec.
"""

from __future__ import annotations


import logging
import time
from typing import Any, Generator
from urllib.parse import urlparse

import paramiko
import requests

from . import GpuInstance, GpuProvider, PodHandle

logger = logging.getLogger(__name__)

VOLUME_NAME = "cacheon-sn14-volume"
VOLUME_SIZE_MB = 500 * 1024  # 500 GB
API_BASE = "https://api.targon.com/tha/v2"

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


def _parse_ssh_url(url: str) -> tuple[str, int]:
    """Extract (host, port) from a DIRECT-routed Targon URL.

    Handles formats like:
      tcp://1.2.3.4:22222
      ssh://1.2.3.4:22
      https://wrk-abc.caas.targon.com
      1.2.3.4:22222
    """
    if "://" not in url:
        url = f"tcp://{url}"
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 22
    return host, port


class TargonProvider:
    """GpuProvider backed by the Targon v2 REST API."""

    name = "targon"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._volume_ids: dict[str, str] = {}
        self._ssh_key_uid: str | None = None
        self._ssh_private_key: paramiko.RSAKey | None = None

    # -- HTTP helpers ------------------------------------------------------

    def _get(self, path: str, **kwargs: Any) -> Any:
        resp = requests.get(
            f"{API_BASE}{path}", headers=self._headers, timeout=30, **kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict | None = None) -> Any:
        resp = requests.post(
            f"{API_BASE}{path}", headers=self._headers, json=json, timeout=30
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def _put(self, path: str, json: dict | None = None) -> Any:
        resp = requests.put(
            f"{API_BASE}{path}", headers=self._headers, json=json, timeout=30
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def _delete(self, path: str) -> None:
        resp = requests.delete(f"{API_BASE}{path}", headers=self._headers, timeout=30)
        resp.raise_for_status()

    # -- SSH key management ------------------------------------------------

    def _ensure_ssh_key(self) -> str:
        """Register a temporary RSA keypair with Targon and return the key UID."""
        if self._ssh_key_uid:
            return self._ssh_key_uid

        self._ssh_private_key = paramiko.RSAKey.generate(4096)
        pub_key = f"ssh-rsa {self._ssh_private_key.get_base64()} cacheon-eval"

        existing = self._get("/ssh-keys")
        for key in existing.get("items", []):
            if key.get("name") == "cacheon-eval":
                try:
                    self._delete(f"/ssh-keys/{key['uid']}")
                except Exception:
                    pass

        resp = self._post(
            "/ssh-keys",
            json={
                "name": "cacheon-eval",
                "ssh_key": pub_key,
            },
        )
        self._ssh_key_uid = resp["uid"]
        logger.info("Registered Targon SSH key: %s", self._ssh_key_uid)
        return self._ssh_key_uid

    # -- Volume management -------------------------------------------------

    def _ensure_volume(self, resource_name: str) -> str | None:
        """Find or create the cacheon-sn14-volume volume for *resource_name*."""
        if resource_name in self._volume_ids:
            return self._volume_ids[resource_name]

        try:
            volumes = self._get("/volumes")
            for vol in volumes.get("items", []):
                if (
                    vol.get("name") == VOLUME_NAME
                    and vol.get("resource_name") == resource_name
                ):
                    uid = vol["uid"]
                    self._volume_ids[resource_name] = uid
                    logger.info(
                        "Reusing Targon volume %s on %s (uid=%s)",
                        VOLUME_NAME,
                        resource_name,
                        uid,
                    )
                    return uid

            resp = self._post(
                "/volumes",
                json={
                    "name": VOLUME_NAME,
                    "size_in_mb": VOLUME_SIZE_MB,
                    "resource_name": resource_name,
                },
            )
            uid = resp["uid"]
            self._volume_ids[resource_name] = uid
            logger.info(
                "Created Targon volume %s on %s (uid=%s)",
                VOLUME_NAME,
                resource_name,
                uid,
            )

            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                state = self._get(f"/volumes/{uid}/state")
                if state.get("status") == "READY":
                    return uid
                time.sleep(5)
            logger.warning(
                "Volume %s still provisioning after 120s, attaching anyway", uid
            )
            return uid
        except Exception as exc:
            logger.warning(
                "Could not ensure Targon volume: %s (continuing without)", exc
            )
            return None

    # -- GpuProvider interface ---------------------------------------------

    def search(self) -> list[GpuInstance]:
        resp = requests.get(
            f"{API_BASE}/inventory",
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
        resource_name = instance.instance_id
        ssh_key_uid = self._ensure_ssh_key()
        volume_uid = self._ensure_volume(resource_name)

        body: dict[str, Any] = {
            "name": "cacheon-eval",
            "image": "ghcr.io/manifold-inc/ubuntu-systemd-docker:v3",
            "resource_name": resource_name,
            "type": "RENTAL",
            "ports": [
                {"port": 22, "protocol": "TCP", "routing": "DIRECT"},
            ],
            "ssh_keys": [ssh_key_uid],
        }
        if volume_uid:
            body["volumes"] = [{"uid": volume_uid, "mount_path": "/workspace"}]

        workload = self._post("/workloads", json=body)
        workload_uid = workload["uid"]

        self._post(f"/workloads/{workload_uid}/deploy")
        logger.info(
            "Targon workload %s created and deploying (resource=%s, volume=%s)",
            workload_uid,
            resource_name,
            volume_uid or "none",
        )

        return PodHandle(
            provider="targon",
            pod_id=workload_uid,
            gpu_count=instance.num_gpus,
            hourly_price_cents=instance.hourly_price_cents,
            raw={"resource_name": resource_name},
        )

    def wait_ready(self, handle: PodHandle, timeout_s: int = 600) -> PodHandle:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self._get(f"/workloads/{handle.pod_id}/state")
            status = state.get("status", "")

            if status == "RUNNING":
                for url_entry in state.get("urls", []):
                    if url_entry.get("port") == 22:
                        host, port = _parse_ssh_url(url_entry.get("url", ""))
                        handle.raw["ssh"] = {
                            "host": host,
                            "port": port,
                            "user": "root",
                        }
                        break

                if "ssh" not in handle.raw:
                    wl = self._get(f"/workloads/{handle.pod_id}")
                    for url_entry in wl.get("state", {}).get("urls", []):
                        if url_entry.get("port") == 22:
                            host, port = _parse_ssh_url(url_entry["url"])
                            handle.raw["ssh"] = {
                                "host": host,
                                "port": port,
                                "user": "root",
                            }
                            break

                if "ssh" not in handle.raw:
                    logger.warning(
                        "Workload RUNNING but no SSH URL for port 22; "
                        "SSH exec will fail"
                    )
                return handle

            if status in ("FAILED", "ERROR", "TERMINATED"):
                raise RuntimeError(
                    f"Targon workload {handle.pod_id} entered {status}: "
                    f"{state.get('message', '')}"
                )
            time.sleep(10)

        raise TimeoutError(
            f"Targon workload {handle.pod_id} not ready after {timeout_s}s"
        )

    def exec(self, handle: PodHandle, command: str) -> dict[str, Any]:
        ssh_info = handle.raw.get("ssh", {})
        host = ssh_info.get("host", "")
        port = ssh_info.get("port", 22)
        user = ssh_info.get("user", "root")

        if not host:
            raise RuntimeError(f"No SSH host for workload {handle.pod_id}")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            connect_kwargs: dict[str, Any] = {
                "hostname": host,
                "port": port,
                "username": user,
                "timeout": 30,
            }
            if self._ssh_private_key:
                connect_kwargs["pkey"] = self._ssh_private_key
            client.connect(**connect_kwargs)
            _stdin, stdout, stderr = client.exec_command(command, timeout=3600)
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
            self._delete(f"/workloads/{handle.pod_id}")
            logger.info("Targon workload %s deleted", handle.pod_id)
        except Exception as exc:
            logger.error("Targon teardown failed for %s: %s", handle.pod_id, exc)

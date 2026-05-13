"""Targon GPU cloud provider using the v2 REST API.

Uses a register-then-deploy workload pattern with an auto-generated SSH
keypair for paramiko exec. Attaches a pre-created persistent volume
(cacheon-sn14-volume, >= 500 GB) when available so model weights survive
across eval cycles.
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
VOLUME_MIN_MB = 500 * 1024  # 500 GB
API_BASE = "https://api.targon.com/tha/v2"

_VRAM_GB: dict[str, int] = {
    "B300": 288,
    "B200": 180,
    "H200": 141,
    "H100": 80,
    "A100": 80,
}


def _normalize_gpu_type(raw: str) -> str:
    for prefix in ("NVIDIA-GeForce-", "NVIDIA-"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def _lookup_vram(gpu_type_raw: str) -> tuple[str, int]:
    """Return (canonical_type, vram_gb). Handles variants like 'H200-NVL'."""
    t = gpu_type_raw.upper()
    for canon, vram in _VRAM_GB.items():
        if canon in t:
            return canon, vram
    return "", 0


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
        self._ssh_key_uid: str | None = None
        self._ssh_private_key: paramiko.RSAKey | None = None
        self._volume_uid: str | None = None

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

    # -- Volume lookup -----------------------------------------------------

    def _find_volume(self) -> str | None:
        """Look up a pre-created volume named cacheon-sn14-volume (>= 500 GB)."""
        if self._volume_uid is not None:
            return self._volume_uid

        try:
            data = self._get("/volumes")
            for vol in data.get("items", []):
                if vol.get("name") == VOLUME_NAME:
                    size_mb = vol.get("size", 0)
                    status = vol.get("state", {}).get("status", "")
                    if size_mb >= VOLUME_MIN_MB and status.upper() in (
                        "READY",
                        "RUNNING",
                    ):
                        self._volume_uid = vol["uid"]
                        logger.info(
                            "Found Targon volume %s (%d GB, uid=%s)",
                            VOLUME_NAME,
                            size_mb // 1024,
                            self._volume_uid,
                        )
                        return self._volume_uid
                    logger.warning(
                        "Targon volume %s exists but not usable (size=%dMB, status=%s)",
                        VOLUME_NAME,
                        size_mb,
                        status,
                    )
        except Exception as exc:
            logger.warning("Could not look up Targon volumes: %s", exc)

        logger.info(
            "No ready %s volume found on Targon; using ephemeral disk",
            VOLUME_NAME,
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
            gpu_type_raw = _normalize_gpu_type(spec.get("gpu_type", ""))
            gpu_count = spec.get("gpu_count", 0)
            available = item.get("available", 0)

            if available <= 0:
                continue

            canon, vram = _lookup_vram(gpu_type_raw)
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
                    gpu_type=canon,
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
        volume_uid = self._find_volume()

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
            volume_uid or "ephemeral",
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
        poll_count = 0
        while time.monotonic() < deadline:
            state = self._get(f"/workloads/{handle.pod_id}/state")
            status = state.get("status", "")
            elapsed = int(time.monotonic() - (deadline - timeout_s))
            poll_count += 1
            if poll_count % 3 == 1:
                logger.info(
                    "⏳ Waiting for workload %s: status=%s (%ds elapsed)",
                    handle.pod_id,
                    status,
                    elapsed,
                )

            if status.upper() == "RUNNING":
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

            if status.upper() in ("FAILED", "ERROR", "TERMINATED"):
                raise RuntimeError(
                    f"Targon workload {handle.pod_id} entered {status}: "
                    f"{state.get('message', '')}"
                )
            time.sleep(10)

        raise TimeoutError(
            f"Targon workload {handle.pod_id} not ready after {timeout_s}s"
        )

    def _ssh_connect(self, handle: PodHandle) -> paramiko.SSHClient:
        """Open an SSH connection, retrying up to 6 times (90s total) for auth propagation."""
        ssh_info = handle.raw.get("ssh", {})
        host = ssh_info.get("host", "")
        port = ssh_info.get("port", 22)
        user = ssh_info.get("user", "root")

        if not host:
            raise RuntimeError(f"No SSH host for workload {handle.pod_id}")

        max_retries = 6
        for attempt in range(1, max_retries + 1):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            connect_kwargs: dict[str, Any] = {
                "hostname": host,
                "port": port,
                "username": user,
                "timeout": 30,
            }
            if self._ssh_private_key:
                connect_kwargs["pkey"] = self._ssh_private_key
            try:
                client.connect(**connect_kwargs)
                if attempt > 1:
                    logger.info("SSH connected on attempt %d", attempt)
                return client
            except paramiko.AuthenticationException:
                client.close()
                if attempt == max_retries:
                    raise
                wait = 15
                logger.info(
                    "SSH auth failed (attempt %d/%d), retrying in %ds...",
                    attempt,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
            except Exception:
                client.close()
                raise
        raise RuntimeError("unreachable")

    def exec(self, handle: PodHandle, command: str) -> dict[str, Any]:
        client = self._ssh_connect(handle)
        try:
            _stdin, stdout, stderr = client.exec_command(command, timeout=10800)
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

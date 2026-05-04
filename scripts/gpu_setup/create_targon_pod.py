#!/usr/bin/env python3
"""Create and deploy a Targon GPU rental workload with volume + SSH keys."""

import os
import sys
import time

import requests

from shared import SSH_KEYS, load_env_dict

WORKLOAD_NAME = "test-cacheon"  # TODO: replace this with your workload name
IMAGE = "ghcr.io/manifold-inc/ubuntu-systemd-docker:v3"
VOLUME_UID = "vol-0lhmsprbolfa"  # TODO: replace this with your volume UID
VOLUME_MOUNT = "/workspace"

# Targon has no inventory fallback API; pick exactly one tier below.
TARGON_GPU = "H200"  # TODO: replace with one of "H100", "H200", "B200"

TARGON_RESOURCE_BY_GPU = {
    "H100": "h100-small",
    "H200": "h200-small",
    "B200": "b200-small",
}

BASE = "https://api.targon.com/tha/v2"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ['TARGON_API_KEY']}",
        "Content-Type": "application/json",
    }


def _req(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{BASE}{path}"
    resp = requests.request(method, url, headers=_headers(), **kwargs)
    if not resp.ok:
        print(f"  {method} {path} -> {resp.status_code}", file=sys.stderr)
        print(f"    {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp


def ensure_ssh_keys() -> list[str]:
    """Register any missing SSH keys and return all UIDs."""
    resp = _req("GET", "/ssh-keys?limit=100")
    existing = {k["name"]: k["uid"] for k in resp.json().get("items", [])}

    uids: list[str] = []
    for name, pubkey in SSH_KEYS.items():
        if name in existing:
            uid = existing[name]
            print(f"  SSH key '{name}' already exists -> {uid}")
        else:
            r = _req("POST", "/ssh-keys", json={"name": name, "ssh_key": pubkey})
            uid = r.json()["uid"]
            print(f"  SSH key '{name}' created -> {uid}")
        uids.append(uid)
    return uids


def create_workload(ssh_key_uids: list[str], pod_envs: list[dict]) -> str:
    """Create a rental workload for the configured GPU tier."""
    resource = TARGON_RESOURCE_BY_GPU.get(TARGON_GPU)
    if not resource:
        allowed = ", ".join(sorted(TARGON_RESOURCE_BY_GPU))
        print(
            f"  ERROR: TARGON_GPU must be one of [{allowed}], got {TARGON_GPU!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    payload = {
        "name": WORKLOAD_NAME,
        "image": IMAGE,
        "resource_name": resource,
        "type": "RENTAL",
        "volumes": [{"uid": VOLUME_UID, "mount_path": VOLUME_MOUNT}],
        "ssh_keys": ssh_key_uids,
        "envs": pod_envs,
    }
    print(f"\nCreating workload '{WORKLOAD_NAME}' ({TARGON_GPU} -> {resource})...")
    print(f"  Injecting {len(pod_envs)} env vars from .env")
    resp = _req("POST", "/workloads", json=payload)
    data = resp.json()
    uid = data["uid"]
    print(f"  Workload registered -> {uid}")
    print(f"  Cost: ${data.get('cost_per_hour', '?')}/hr")
    return uid


def deploy_workload(workload_uid: str) -> None:
    print(f"\nDeploying {workload_uid}...")
    resp = _req("POST", f"/workloads/{workload_uid}/deploy")
    state = resp.json().get("state", {})
    print(f"  Status: {state.get('status', 'UNKNOWN')}")


def wait_for_running(workload_uid: str, interval: int = 15) -> None:
    print("\nWaiting for workload to reach RUNNING...")
    while True:
        resp = _req("GET", f"/workloads/{workload_uid}/state")
        data = resp.json()
        status = data.get("status", "UNKNOWN").upper()
        print(f"  status={status}")
        if status == "RUNNING":
            print(f"\n  Workload is RUNNING.")
            print(f"  ssh {workload_uid}@ssh.deployments.targon.com")
            return
        if status in ("FAILED", "TERMINATED", "ERROR"):
            print(f"\n  Workload entered terminal state: {status}", file=sys.stderr)
            sys.exit(1)
        time.sleep(interval)


def main() -> None:
    if not os.getenv("TARGON_API_KEY"):
        print("ERROR: TARGON_API_KEY not found in environment", file=sys.stderr)
        sys.exit(1)

    print("=== Targon GPU Pod Creator ===\n")

    print("1) Ensuring SSH keys...")
    ssh_uids = ensure_ssh_keys()

    print("\n2) Loading .env for pod injection...")
    env_dict = load_env_dict()
    pod_envs = [{"name": k, "value": v} for k, v in env_dict.items()]
    print(f"  {len(pod_envs)} env vars loaded (values hidden)")

    print("\n3) Creating rental workload...")
    wid = create_workload(ssh_uids, pod_envs)

    print("\n4) Deploying...")
    deploy_workload(wid)

    print("\n5) Polling status...")
    wait_for_running(wid)

    print("\nDone.")


if __name__ == "__main__":
    main()

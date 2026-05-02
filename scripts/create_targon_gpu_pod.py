#!/usr/bin/env python3
"""Create and deploy a Targon GPU rental workload with volume + SSH keys."""

import os
import sys
import time
from pathlib import Path

import requests

TARGON_API_KEY = os.getenv("TARGON_API_KEY")

WORKLOAD_NAME = "test-cacheon"
RESOURCE_NAME = "h100-small"
IMAGE = "ghcr.io/manifold-inc/ubuntu-systemd-docker:v3"
VOLUME_UID = "vol-0lhmsprbolfa"
VOLUME_MOUNT = "/workspace"

SSH_KEYS = {
    "xavier-latent": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ80HSdgwG98gpXqe/bwR+1NLIlmZJMNmHro7H7X04UC xllgms@gmail.com",
    "clement-latent": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMkpQe0zLHSW+heTGX5UV00HbuCA7CUXC9lowjE/aTwR",
    "cacheon-cpu-validator": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHle3erHaBhA/yRREWFyYk7m0cTHuCdcnsHEsOWzOrWe cacheon-cpu-validator",
}

# ── Read .env and build pod env vars ────────────────────────────────────
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_pod_envs() -> list[dict]:
    """Parse .env file and return as Targon envs array."""
    if not ENV_FILE.exists():
        print(
            f"  WARNING: {ENV_FILE} not found, no env vars will be injected.",
            file=sys.stderr,
        )
        return []
    envs = []
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        envs.append({"name": key.strip(), "value": value.strip()})
    return envs


# ── API plumbing ────────────────────────────────────────────────────────
BASE = "https://api.targon.com/tha/v2"
HEADERS = {
    "Authorization": f"Bearer {TARGON_API_KEY}",
    "Content-Type": "application/json",
}


def _req(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{BASE}{path}"
    resp = requests.request(method, url, headers=HEADERS, **kwargs)
    if not resp.ok:
        print(f"  ✗ {method} {path} → {resp.status_code}", file=sys.stderr)
        print(f"    {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp


# ── Step 1: ensure SSH keys exist on Targon ────────────────────────────
def ensure_ssh_keys() -> list[str]:
    """Register any missing SSH keys and return all UIDs."""
    resp = _req("GET", "/ssh-keys?limit=100")
    existing = {k["name"]: k["uid"] for k in resp.json().get("items", [])}

    uids: list[str] = []
    for name, pubkey in SSH_KEYS.items():
        if name in existing:
            uid = existing[name]
            print(f"  SSH key '{name}' already exists → {uid}")
        else:
            r = _req("POST", "/ssh-keys", json={"name": name, "ssh_key": pubkey})
            uid = r.json()["uid"]
            print(f"  SSH key '{name}' created → {uid}")
        uids.append(uid)
    return uids


# ── Step 2: create the workload ────────────────────────────────────────
def create_workload(ssh_key_uids: list[str], pod_envs: list[dict]) -> str:
    payload = {
        "name": WORKLOAD_NAME,
        "image": IMAGE,
        "resource_name": RESOURCE_NAME,
        "type": "RENTAL",
        "volumes": [{"uid": VOLUME_UID, "mount_path": VOLUME_MOUNT}],
        "ssh_keys": ssh_key_uids,
        "envs": pod_envs,
    }
    print(f"\nCreating workload '{WORKLOAD_NAME}' ({RESOURCE_NAME})...")
    print(f"  Injecting {len(pod_envs)} env vars from .env")
    resp = _req("POST", "/workloads", json=payload)
    data = resp.json()
    uid = data["uid"]
    print(f"  Workload registered → {uid}")
    print(f"  Cost: ${data.get('cost_per_hour', '?')}/hr")
    return uid


# ── Step 3: deploy ──────────────────────────────────────────────────────
def deploy_workload(workload_uid: str) -> None:
    print(f"\nDeploying {workload_uid}...")
    resp = _req("POST", f"/workloads/{workload_uid}/deploy")
    state = resp.json().get("state", {})
    print(f"  Status: {state.get('status', 'UNKNOWN')}")


# ── Step 4: poll until running ───────────────────────────────────────────
def wait_for_running(workload_uid: str, interval: int = 15) -> None:
    print("\nWaiting for workload to reach RUNNING...")
    while True:
        resp = _req("GET", f"/workloads/{workload_uid}/state")
        data = resp.json()
        status = data.get("status", "UNKNOWN").upper()
        urls = data.get("urls", [])
        print(f"  status={status}")
        if status == "RUNNING":
            print("\n✓ Workload is RUNNING.\n")
            print(f"  ssh {workload_uid}@ssh.deployments.targon.com")
            return
        if status in ("FAILED", "TERMINATED", "ERROR"):
            print(f"\n  Workload entered terminal state: {status}", file=sys.stderr)
            sys.exit(1)
        time.sleep(interval)


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    if not TARGON_API_KEY:
        print("ERROR: TARGON_API_KEY not found in environment", file=sys.stderr)
        sys.exit(1)

    print("=== Targon GPU Pod Creator ===\n")

    print("1) Ensuring SSH keys...")
    ssh_uids = ensure_ssh_keys()

    print(f"\n2) Loading .env for pod injection...")
    pod_envs = load_pod_envs()

    print(f"\n3) Creating rental workload...")
    wid = create_workload(ssh_uids, pod_envs)

    print(f"\n4) Deploying...")
    deploy_workload(wid)

    print(f"\n5) Polling status...")
    wait_for_running(wid)

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create and deploy a Lium GPU pod with volume + SSH keys + env vars from .env."""

import os
import sys
from pathlib import Path

from lium.sdk import Lium

POD_NAME = "cacheon-gpu"
GPU_TYPE = "H100"
GPU_COUNT = 1
VOLUME_NAME = "SN14"
VOLUME_MOUNT = "/workspace"

DOCKER_IMAGE = "daturaai/pytorch"
DOCKER_IMAGE_TAG = "2.7.0-py3.12-cuda12.8.0-devel-ubuntu24.04"
TEMPLATE_NAME = "cacheon-pytorch-h100"

SSH_KEYS = {
    "xavier-latent": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ80HSdgwG98gpXqe/bwR+1NLIlmZJMNmHro7H7X04UC xllgms@gmail.com",
    "clement-latent": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMkpQe0zLHSW+heTGX5UV00HbuCA7CUXC9lowjE/aTwR",
    "cacheon-cpu-validator": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHle3erHaBhA/yRREWFyYk7m0cTHuCdcnsHEsOWzOrWe cacheon-cpu-validator",
}

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_dict() -> dict[str, str]:
    """Parse .env and return as key-value dict (values not printed)."""
    if not ENV_FILE.exists():
        print(
            f"  WARNING: {ENV_FILE} not found, no env vars will be injected.",
            file=sys.stderr,
        )
        return {}
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def ensure_ssh_keys(lium: Lium) -> list[str]:
    """Register missing SSH keys, return all public key strings."""
    existing = {k.name: k for k in lium.list_ssh_keys()}
    pubkeys: list[str] = []
    for name, pubkey in SSH_KEYS.items():
        if name in existing:
            print(f"  SSH key '{name}' already registered")
        else:
            lium.register_ssh_key(name=name, public_key=pubkey)
            print(f"  SSH key '{name}' registered")
        pubkeys.append(pubkey)
    return pubkeys


def find_volume(lium: Lium) -> str:
    """Find volume by VOLUME_NAME, return its ID."""
    for v in lium.volumes():
        if v.name == VOLUME_NAME:
            print(f"  Volume '{VOLUME_NAME}' found: {v.id} ({v.current_size_gb:.1f} GB)")
            return v.id
    print(f"  ERROR: Volume '{VOLUME_NAME}' not found", file=sys.stderr)
    available = [v.name for v in lium.volumes()]
    if available:
        print(f"  Available volumes: {', '.join(available)}", file=sys.stderr)
    sys.exit(1)


def find_or_create_template(lium: Lium, env_dict: dict[str, str]) -> str:
    """Reuse an existing template by name, or create a new one with env vars baked in."""
    for t in lium.templates(filter=TEMPLATE_NAME, only_my=True):
        if t.name == TEMPLATE_NAME:
            print(f"  Reusing template: {t.name} ({t.id})")
            return t.id

    print(f"  Creating template '{TEMPLATE_NAME}'...")
    tmpl = lium.create_template(
        name=TEMPLATE_NAME,
        docker_image=DOCKER_IMAGE,
        docker_image_tag=DOCKER_IMAGE_TAG,
        ports=[22, 8000],
        environment=env_dict,
        volumes=[VOLUME_MOUNT],
        is_private=True,
    )
    print(f"  Waiting for template verification...")
    ready = lium.wait_template_ready(tmpl.id, timeout=300)
    if not ready:
        print("  ERROR: Template verification timed out", file=sys.stderr)
        sys.exit(1)
    print(f"  Template ready: {tmpl.id}")
    return tmpl.id


EXCLUDED_GPU_MODELS = {"NVIDIA H100 PCIe"}


def find_executor(lium: Lium) -> str:
    """Find cheapest available H100 SXM executor (excludes PCIe variants)."""
    executors = lium.ls(gpu_type=GPU_TYPE, gpu_count=GPU_COUNT)
    executors = [e for e in executors if e.gpu_model not in EXCLUDED_GPU_MODELS]
    if not executors:
        print(
            f"  ERROR: No {GPU_TYPE} x{GPU_COUNT} (non-PCIe) executors available right now",
            file=sys.stderr,
        )
        sys.exit(1)
    best = min(executors, key=lambda e: e.price_per_hour)
    loc = best.location.get("country", "?")
    print(f"  Found {len(executors)} executor(s), picking cheapest:")
    print(f"    {best.gpu_model} | ${best.price_per_hour:.2f}/hr | {loc} | {best.id}")
    return best.id


def main() -> None:
    if not os.getenv("LIUM_API_KEY"):
        print("ERROR: LIUM_API_KEY not found in environment", file=sys.stderr)
        print("  Fix: export LIUM_API_KEY=...", file=sys.stderr)
        sys.exit(1)

    print("=== Lium GPU Pod Creator ===\n")

    lium = Lium()

    print("1) Ensuring SSH keys...")
    pubkeys = ensure_ssh_keys(lium)

    print("\n2) Loading .env for pod injection...")
    env_dict = load_env_dict()
    print(f"  {len(env_dict)} env vars loaded (values hidden)")

    print("\n3) Finding volume...")
    volume_id = find_volume(lium)

    print("\n4) Setting up template...")
    template_id = find_or_create_template(lium, env_dict)

    print("\n5) Finding executor...")
    executor_id = find_executor(lium)

    print(f"\n6) Creating pod '{POD_NAME}'...")
    pod = lium.up(
        executor_id=executor_id,
        name=POD_NAME,
        template_id=template_id,
        volume_id=volume_id,
        ssh_keys=pubkeys,
    )
    pod_id = pod.get("id") or pod.get("pod_id", "unknown")
    print(f"  Pod created: {pod_id}")

    print("\n7) Waiting for pod to be ready (up to 10 min)...")
    ready = lium.wait_ready(pod_id, timeout=600, poll_interval=15)

    if ready:
        print(f"\n=== Pod is RUNNING ===")
        print(f"  Pod ID : {ready.id}")
        print(f"  Name   : {ready.name}")
        print(f"  Status : {ready.status}")
        if ready.ssh_cmd:
            print(f"  SSH    : {ready.ssh_cmd}")
    else:
        print("\n  Pod did not reach RUNNING within timeout.", file=sys.stderr)
        print("  Check the Lium dashboard for status.", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()

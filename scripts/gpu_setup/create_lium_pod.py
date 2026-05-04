#!/usr/bin/env python3
"""Create and deploy a Lium GPU pod with volume + SSH keys + env vars from .env."""

import os
import sys

from lium.sdk import Lium

from shared import GPU_COUNT, GPU_PREFERENCE, SSH_KEYS, load_env_dict

POD_NAME = "cacheon-gpu"  # TODO: replace this with your pod name
VOLUME_NAME = "SN14"  # TODO: replace this with your volume name
VOLUME_MOUNT = "/workspace"

DOCKER_IMAGE = "daturaai/pytorch"
DOCKER_IMAGE_TAG = "2.7.0-py3.12-cuda12.8.0-devel-ubuntu24.04"
TEMPLATE_NAME = "cacheon-pytorch-gpu"


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
            print(
                f"  Volume '{VOLUME_NAME}' found: {v.id} ({v.current_size_gb:.1f} GB)"
            )
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


def find_executor(lium: Lium) -> str:
    """Try GPU types in preference order, return cheapest executor for the first available type."""
    for pref in GPU_PREFERENCE:
        gpu_type = pref["type"]
        exclude = pref["exclude"]
        executors = lium.ls(gpu_type=gpu_type, gpu_count=GPU_COUNT)
        executors = [e for e in executors if e.gpu_model not in exclude]
        if not executors:
            print(f"  No {gpu_type} x{GPU_COUNT} executors available, trying next...")
            continue
        best = min(executors, key=lambda e: e.price_per_hour)
        loc = best.location.get("country", "?")
        print(f"  Found {len(executors)} {gpu_type} executor(s), picking cheapest:")
        print(
            f"    {best.gpu_model} | ${best.price_per_hour:.2f}/hr | {loc} | {best.id}"
        )
        return best.id
    tried = ", ".join(p["type"] for p in GPU_PREFERENCE)
    print(f"  ERROR: No executors available for any of [{tried}]", file=sys.stderr)
    sys.exit(1)


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

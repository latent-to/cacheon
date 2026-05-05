"""Shared constants and helpers for GPU pod provisioning scripts."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = REPO_ROOT / ".env"

GPU_COUNT = int(os.environ.get("CACHEON_GPU_COUNT", "8"))

GPU_PREFERENCE = [
    {"type": "H100", "exclude": {"NVIDIA H100 PCIe"}},
    {"type": "H200", "exclude": set()},
    {"type": "B200", "exclude": set()},
]

SSH_KEYS = {
    "xavier-latent": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ80HSdgwG98gpXqe/bwR+1NLIlmZJMNmHro7H7X04UC xllgms@gmail.com",
    "clement-latent": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMkpQe0zLHSW+heTGX5UV00HbuCA7CUXC9lowjE/aTwR",
    "cacheon-cpu-validator": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHle3erHaBhA/yRREWFyYk7m0cTHuCdcnsHEsOWzOrWe cacheon-cpu-validator",
}


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

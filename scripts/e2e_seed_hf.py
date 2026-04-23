#!/usr/bin/env python3
"""One-time script: upload E2E test policies to HuggingFace.

Run this once, capture the printed JSON, then you may delete this script.
The JSON is written to `scripts/example_policies.json` for convenience.

Usage:
    export HF_TOKEN=hf_...
    python scripts/e2e_seed_hf.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = REPO_ROOT / "scripts" / "example_policies"
OUTPUT_PATH = REPO_ROOT / "scripts" / "example_policies.json"
NAMESPACE = "xavierlyu"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var required", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    descriptors: list[dict] = []

    for policy_file in sorted(POLICIES_DIR.glob("*.py")):
        if policy_file.name == "__init__.py":
            continue
        name = policy_file.stem
        repo_id = f"{NAMESPACE}/cacheon-e2e-{name}"

        print(f"Creating repo {repo_id} ...", end=" ")
        try:
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue
        print("OK")

        print(f"  Uploading {policy_file.name} ...", end=" ")
        try:
            upload_file(
                path_or_fileobj=str(policy_file),
                path_in_repo="policy.py",
                repo_id=repo_id,
                token=token,
            )
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue
        print("OK")

        print(f"  Getting commit SHA ...", end=" ")
        commits = api.list_repo_commits(repo_id)
        latest = commits[0]
        revision = latest.commit_id
        print(f"{revision}")

        descriptors.append(
            {
                "name": name,
                "repo": repo_id,
                "revision": revision,
            }
        )

    print(f"\nDescriptors JSON:")
    payload = json.dumps(descriptors, indent=2)
    print(payload)

    OUTPUT_PATH.write_text(payload + "\n")
    print(f"\nSaved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

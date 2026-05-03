#!/usr/bin/env python3
"""Upload a policy to Hugging Face and commit it on-chain in one shot.

Usage:
    export HF_TOKEN=hf_...
    python miner/commit.py \\
        --policy-file policy.py \\
        --repo you/my-kv-policy \\
        --wallet-name my-miner --wallet-hotkey default \\
        --network test --netuid 460
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Upload policy.py to HF and commit on-chain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--policy-file", required=True, help="Path to your policy.py file.")
    p.add_argument(
        "--repo",
        required=True,
        help="HF repo id (e.g. you/my-kv-policy). "
        "Created automatically if it doesn't exist.",
    )
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--wallet-hotkey", default="default")
    p.add_argument("--network", default="test", help="Bittensor network: finney | test")
    p.add_argument(
        "--netuid", type=int, default=460, help="Subnet UID (460=testnet, 14=mainnet)"
    )
    args = p.parse_args(argv)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("error: HF_TOKEN env var required", file=sys.stderr)
        return 1

    if not os.path.isfile(args.policy_file):
        print(f"error: {args.policy_file} not found", file=sys.stderr)
        return 1

    # --- Step 1: Upload to Hugging Face ---
    from huggingface_hub import HfApi, upload_file

    api = HfApi(token=hf_token)
    print(f"Creating repo {args.repo} (if needed)...", end=" ")
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    print("OK")

    print(f"Uploading {args.policy_file} → {args.repo}/policy.py ...", end=" ")
    commit_info = upload_file(
        path_or_fileobj=args.policy_file,
        path_in_repo="policy.py",
        repo_id=args.repo,
        token=hf_token,
    )
    print("OK")

    revision = commit_info.oid
    print(f"Revision: {revision}")

    # --- Step 2: Commit on-chain ---
    import bittensor as bt

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)

    commit_data = json.dumps({"repo": args.repo, "revision": revision})
    print(f"Committing to netuid={args.netuid}: {commit_data}")

    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=args.netuid,
        data=commit_data,
        blocks_until_reveal=1,
    )
    print("Done. Validator will pick this up within ~6 minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Commit a Docker image reference on-chain for Cacheon evaluation.

Usage:
    python miner/commit.py \
        --image docker.io/myuser/cacheon-miner:v1 \
        --digest sha256:abc123... \
        --wallet-name my-miner --wallet-hotkey default \
        --network test --netuid 460
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Commit a Docker image reference on-chain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image",
        required=True,
        help="Docker image reference (e.g. docker.io/myuser/cacheon-miner:v1)",
    )
    p.add_argument(
        "--digest",
        required=True,
        help="Image manifest digest (e.g. sha256:abc123...)",
    )
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--wallet-hotkey", default="default")
    p.add_argument("--network", default="test", help="Bittensor network: finney | test")
    p.add_argument(
        "--netuid", type=int, default=460, help="Subnet UID (460=testnet, 14=mainnet)"
    )
    args = p.parse_args(argv)

    if not re.match(r"^sha256:[0-9a-f]{64}$", args.digest):
        print(
            f"error: digest must be sha256:<64 hex chars>, got: {args.digest}",
            file=sys.stderr,
        )
        return 1

    import bittensor as bt

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)

    commit_data = json.dumps({"image": args.image, "digest": args.digest})
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

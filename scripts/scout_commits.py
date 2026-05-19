"""Read every revealed image commit on SN14 and print one row per miner.

Run from venv with bittensor installed:
    source .scout-venv/bin/activate
    python scripts/scout_commits.py

Uses the validator's own `fetch_revealed_commitments` which falls back to a
raw substrate query when the bittensor SDK chokes on hex decoding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root on sys.path so we can import validator.chain
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bittensor as bt  # noqa: E402

from validator.chain import fetch_revealed_commitments  # noqa: E402


NETUID = 14
NETWORK = "finney"


def main() -> int:
    st = bt.Subtensor(network=NETWORK)
    mg = st.metagraph(netuid=NETUID, lite=False)
    revealed = fetch_revealed_commitments(st, NETUID)

    hotkey_to_uid = {str(hk): uid for uid, hk in enumerate(mg.hotkeys)}

    rows = []
    for hotkey_str, reveals in revealed.items():
        if not reveals:
            continue
        block, raw = max(reveals, key=lambda p: p[0])
        try:
            obj = json.loads(raw)
            image = obj.get("image", "?")
            digest = obj.get("digest", "?")
        except (json.JSONDecodeError, TypeError):
            image = f"<unparsed:{raw[:40]}>"
            digest = "?"
        uid = hotkey_to_uid.get(hotkey_str, -1)
        try:
            incentive = float(mg.incentive[uid]) if uid >= 0 else 0.0
            emission = float(mg.emission[uid]) if uid >= 0 else 0.0
        except Exception:
            incentive = 0.0
            emission = 0.0
        rows.append(
            {
                "uid": uid,
                "hotkey": hotkey_str,
                "block": int(block),
                "incentive": incentive,
                "emission": emission,
                "image": image,
                "digest": digest,
            }
        )

    # Sort by incentive descending — winner first, runner-up next, then everyone else
    rows.sort(key=lambda r: (-r["incentive"], -r["emission"], r["uid"]))

    print(f"{'UID':>4} | {'Incentive':>9} | {'Emission α':>10} | {'Block':>8} | Image  |  Digest")
    print("-" * 130)
    for r in rows:
        img = r["image"] if len(r["image"]) <= 50 else r["image"][:47] + "..."
        dig = r["digest"][:24] + "..." if len(r["digest"]) > 24 else r["digest"]
        print(
            f"{r['uid']:>4} | {r['incentive']:>9.6f} | {r['emission']:>10.3f} | "
            f"{r['block']:>8} | {img}  |  {dig}"
        )
    print(f"\nTotal commitments: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

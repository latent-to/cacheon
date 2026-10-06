"""Read-only reveal monitoring in a separate process; Discord alerts never gate signing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import time
from urllib.request import Request, urlopen

from cacheon import chain
from cacheon.chain.weight_share import CurrentWeightOffer


def send_discord(webhook: str, message: str) -> None:
    request = Request(webhook, data=json.dumps({
        "content": message[:2000], "allowed_mentions": {"parse": []},
    }).encode(), headers={"Content-Type": "application/json", "User-Agent": "Cacheon/1.0"})
    with urlopen(request, timeout=15) as response:
        response.read()


def check_reveals(subtensor, journal: Path, state: dict, *, deadline_blocks: int, notify) -> dict:
    """Read commit receipts and weights; write only the caller's monitor state."""
    state = json.loads(json.dumps(state))
    pending = state.setdefault("pending", [])
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as db:
        db.execute("PRAGMA query_only=ON")
        cursor = state.get("cursor")
        if cursor is None:
            # Start with the latest accepted commit, not an alert flood over history.
            latest = db.execute("SELECT max(sequence) FROM followed_weight_publications "
                                "WHERE status='confirmed'").fetchone()[0]
            cursor = max(0, (latest or 1) - 1)
        rows = db.execute(
            "SELECT sequence,offer_json,record_json FROM followed_weight_publications "
            "WHERE sequence>? AND status='confirmed' ORDER BY sequence LIMIT 1000", (cursor,)).fetchall()
    for sequence, offer_json, record_json in rows:
        record = json.loads(record_json)
        offer = CurrentWeightOffer.from_dict(json.loads(offer_json))
        if record["reason"] == "block_inclusion":
            pending.append({"sequence": sequence, "projection": offer.projection.to_dict(),
                            "block": record["submit_block"]})
        state["cursor"] = sequence
    if not pending:
        return state
    from cacheon.chain.weight_projection import WeightProjection

    latest = WeightProjection.from_dict(pending[-1]["projection"])
    live = chain.fetch_metagraph(subtensor, latest.netuid)
    observed = chain.read_validator_weight_snapshot(
        subtensor, latest.netuid, latest.validator_hotkey, metagraph_view=live)
    matched = 0
    for item in pending:
        projection = WeightProjection.from_dict(item["projection"])
        if (projection.netuid != latest.netuid or projection.validator_hotkey != latest.validator_hotkey):
            raise ValueError("monitor journal contains multiple signers or subnets")
        # Chain weights use uint16 precision and omit zero-rounded dust recipients.
        expected = projection.weights
        if (observed.last_update_block >= item["block"]
                and all(math.isclose(observed.weights.get(k, 0), expected.get(k, 0),
                                     rel_tol=2e-5, abs_tol=2e-5)
                        for k in set(expected) | set(observed.weights))):
            matched = item["sequence"]
    if matched:
        # A later revealed allocation supersedes earlier commits. They must not
        # raise a false alarm just because polling never saw their intermediate row.
        pending[:] = [item for item in pending if item["sequence"] > matched]
        state["last_revealed_sequence"] = matched
        if state.get("alerted"):
            notify(f"Cacheon SN{latest.netuid}: weight reveal recovered for validator "
                   f"{latest.validator_hotkey}; observed at block {live.block}. Publication continued throughout.")
            state["alerted"] = False
    overdue = [item for item in pending if live.block >= item["block"] + deadline_blocks]
    if overdue and not state.get("alerted"):
        notify(f"Cacheon SN{latest.netuid}: committed weights have not been observed correctly revealed "
               f"within {deadline_blocks} blocks. Validator {latest.validator_hotkey}; "
               f"oldest commit block {overdue[0]['block']}; current block {live.block}; "
               f"last weight update {observed.last_update_block}. Publication is NOT paused.")
        state["alerted"] = True
    state["checked_block"] = live.block
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-db", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--webhook-file", type=Path, required=True)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--deadline-blocks", type=int, default=600)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.deadline_blocks <= 0 or args.interval <= 0:
        parser.error("deadline and interval must be positive")
    if args.state.resolve() in {args.journal_db.resolve(), args.webhook_file.resolve()}:
        parser.error("monitor state must be separate from journal and webhook")
    webhook = args.webhook_file.read_text().strip()
    if not webhook.startswith("https://discord.com/api/webhooks/"):
        parser.error("webhook file must contain a Discord HTTPS webhook URL")
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    client = None
    failures = 0
    while True:
        try:
            if client is None:
                client = chain.connect(args.network)
            state = check_reveals(client, args.journal_db, state,
                                  deadline_blocks=args.deadline_blocks,
                                  notify=lambda message: send_discord(webhook, message))
            args.state.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.state.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
            temporary.replace(args.state)
            print(f"weight-reveal-monitor block={state.get('checked_block')} "
                  f"pending={len(state.get('pending', []))} alerted={bool(state.get('alerted'))}", flush=True)
            failures = 0
        except Exception as exc:
            # Exception text from HTTP libraries can contain the secret webhook URL.
            print(f"weight-reveal-monitor check failed: {type(exc).__name__}", flush=True)
            failures += 1
            if failures == 3:
                try:
                    send_discord(webhook, "Cacheon weight-reveal monitor cannot complete its checks "
                                 f"({type(exc).__name__}). Publication is NOT paused.")
                except Exception:
                    failures = 2  # retry alert delivery on the next check
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
                client = None
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

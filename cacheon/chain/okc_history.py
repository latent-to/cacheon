"""Index finalized native username commitments in their original execution order."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from cacheon import chain
from cacheon.chain.eval_cost import _call_args_map
from cacheon.chain.eval_cost_payment import _block_extrinsics, _decode_extrinsic, _normalize_event
from cacheon.chain.okc_username import _raw_username, validate_username


def _registrations_at(st, netuid: int, block: int, block_hash: str) -> list[dict]:
    records = st.substrate.get_events(block_hash=block_hash)
    writes, successes, failures = [], set(), set()
    for event_index, record in enumerate(records):
        row = _normalize_event(record)
        module, name, attributes, extrinsic = chain._event_envelope(row)
        if extrinsic is None and row.get("phase") == "ApplyExtrinsic":
            extrinsic = row.get("extrinsic_idx", row.get("extrinsic_index"))
        if module == "System" and name in ("ExtrinsicSuccess", "ExtrinsicFailed"):
            (successes if name == "ExtrinsicSuccess" else failures).add(extrinsic)
        if module != "Commitments" or name not in ("Commitment", "TimelockCommitment"):
            continue
        if not isinstance(attributes, dict):
            raise RuntimeError(f"malformed commitment event at {block}/{event_index}")
        event_netuid = chain._attribute(attributes, ("netuid", "net_uid"), field="netuid")
        if type(event_netuid) is not int:
            raise RuntimeError(f"malformed commitment netuid at {block}/{event_index}")
        if event_netuid != netuid:
            continue
        hotkey = chain._attribute(attributes, ("who", "hotkey", "account"), field="hotkey")
        if not isinstance(hotkey, str) or not hotkey:
            raise RuntimeError(f"malformed commitment hotkey at {block}/{event_index}")
        writes.append((event_index, hotkey, name, extrinsic))
    counts = Counter(row[1] for row in writes)
    extrinsics, registrations = None, []
    for event_index, hotkey, name, extrinsic in writes:
        if name != "Commitment":
            continue
        metadata = None
        if counts[hotkey] == 1:
            metadata = st.get_commitment_metadata(netuid=netuid, hotkey_ss58=hotkey, block=block)
        # Final storage loses earlier same-block writes and may have been purged.
        if not isinstance(metadata, dict) or type(metadata.get("block")) is not int or metadata["block"] != block:
            if extrinsics is None:
                extrinsics = _block_extrinsics(st.substrate.get_block(block_hash=block_hash))
            if (type(extrinsic) is not int or not 0 <= extrinsic < len(extrinsics)
                    or extrinsic not in successes or extrinsic in failures):
                raise RuntimeError(f"ambiguous commitment history at {block}/{event_index}: missing successful extrinsic")
            signer, module, function, args = _decode_extrinsic(extrinsics[extrinsic])
            args = _call_args_map(args)
            if (signer != hotkey or module != "Commitments" or function != "set_commitment"
                    or args is None or args.get("netuid") != netuid or not isinstance(args.get("info"), dict)):
                raise RuntimeError(f"ambiguous commitment history at {block}/{event_index}: expected direct set_commitment")
            metadata = {"info": args["info"]}
        try:
            username = validate_username(hotkey, _raw_username(metadata))
        except ValueError:
            continue
        registrations.append({"hotkey": hotkey, "username": username, "block": block,
                              "event_index": event_index, "block_hash": block_hash})
    return registrations


def sync_registrations(st, netuid: int, path: Path, *, max_blocks: int = 5000,
                       start_block: int = 1) -> tuple[int, dict[str, dict]]:
    """Return oldest one-to-one bindings only after catching the pinned finalized head.

    Bounded scans persist complete blocks before raising ``history incomplete``.
    The persisted round boundary determines the first included block; bindings
    never release when storage changes.
    """
    if type(netuid) is not int or netuid < 0 or type(max_blocks) is not int or max_blocks <= 0:
        raise ValueError("netuid must be nonnegative and max_blocks must be positive integers")
    if type(start_block) is not int or start_block <= 0:
        raise ValueError("start_block must be a positive integer")
    finalized, finalized_hash = chain.read_finalized_head(st)
    if start_block > finalized + 1:
        raise ValueError("start_block cannot exceed the finalized head plus one")
    genesis = chain._block_hash(st, 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS history_state (id INTEGER PRIMARY KEY CHECK(id=1), "
                       "genesis TEXT NOT NULL, netuid INTEGER NOT NULL, start_block INTEGER NOT NULL, "
                       "block INTEGER NOT NULL, block_hash TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS owners (hotkey TEXT COLLATE BINARY PRIMARY KEY, "
                       "username TEXT COLLATE BINARY NOT NULL UNIQUE, block INTEGER NOT NULL, "
                       "event_index INTEGER NOT NULL, block_hash TEXT NOT NULL)")
            state = db.execute("SELECT * FROM history_state WHERE id=1").fetchone()
            if state is None:
                initial_hash = chain._block_hash(st, start_block - 1)
                db.execute("INSERT INTO history_state VALUES (1, ?, ?, ?, ?, ?)",
                           (genesis, netuid, start_block, start_block - 1, initial_hash))
            else:
                if "start_block" not in state.keys() or state["start_block"] != start_block:
                    raise RuntimeError("OKC username history has a different or missing round start block")
                if state["genesis"] != genesis or state["netuid"] != netuid:
                    raise RuntimeError("OKC username history belongs to a different network or netuid")
                if state["block"] > finalized or chain._block_hash(st, state["block"]) != state["block_hash"]:
                    raise RuntimeError("OKC username history cursor is not on the finalized chain")
        for _ in range(max_blocks):
            with db:
                db.execute("BEGIN IMMEDIATE")
                cursor = db.execute("SELECT block FROM history_state WHERE id=1").fetchone()[0]
                if cursor >= finalized:
                    break
                block = cursor + 1
                block_hash = chain._block_hash(st, block)
                if block == finalized and block_hash != finalized_hash:
                    raise RuntimeError("OKC username history finalized hash changed")
                for registration in _registrations_at(st, netuid, block, block_hash):
                    db.execute("INSERT OR IGNORE INTO owners VALUES (:hotkey, :username, :block, :event_index, :block_hash)",
                               registration)
                db.execute("UPDATE history_state SET block=?, block_hash=? WHERE id=1", (block, block_hash))
        with db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("SELECT block FROM history_state WHERE id=1").fetchone()[0]
            if cursor != finalized:
                raise RuntimeError(f"OKC username history incomplete: indexed {cursor}, finalized target {finalized}")
            owners = {row["hotkey"]: {key: row[key] for key in ("username", "block", "event_index", "block_hash")}
                      for row in db.execute("SELECT * FROM owners ORDER BY block, event_index")}
        return finalized, owners
    finally:
        db.close()

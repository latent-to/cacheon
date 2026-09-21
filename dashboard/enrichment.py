"""Independent per-source caches for the existing chain enrichment worker."""

import json
import sqlite3
import threading
import time
from typing import Any

BLOCK_SECONDS = 12
METAGRAPH_TTL = 600
TEMPO_BLOCKS = 360


class Enrichment:
    """Chain-derived data: block timestamps, extrinsic signers, metagraph.

    All lookups are best-effort; the API serves DB truth even when the chain
    is unreachable. Results persist in a private SQLite cache.
    """

    def __init__(self, cache_db, network, netuid) -> None:
        self.cache_db, self.network, self.netuid = cache_db, network, netuid
        cache_db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._substrate = None
        self._subtensor = None
        self.chain_ok = False
        self.chain_error = ""
        self.tip: dict[str, Any] = {}          # {block, unix_time}
        self.metagraph: dict[str, Any] = {}    # {fetched_at, block, hotkeys:{hk:{...}}, owner_coldkey}
        self._wanted_blocks: set[int] = set()
        self._wanted_extrinsics: set[tuple[int, int]] = set()
        con = self._cache()
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS block_times(
                block INTEGER PRIMARY KEY, unix_time INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS extrinsic_signers(
                block INTEGER NOT NULL, ext_index INTEGER NOT NULL,
                signer TEXT NOT NULL, call TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(block, ext_index));
            CREATE TABLE IF NOT EXISTS kv(
                key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        con.commit()
        con.close()
        mg = self._kv_get("metagraph")
        if mg:
            self.metagraph = mg

    def _cache(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.cache_db, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _kv_get(self, key: str) -> Any:
        con = self._cache()
        row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        con.close()
        return json.loads(row["value"]) if row else None

    def _kv_set(self, key: str, value: Any) -> None:
        con = self._cache()
        con.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        con.commit()
        con.close()

    # -- public read side (never blocks on the chain) --

    def block_time(self, block: int) -> dict[str, Any]:
        """Return {unix, estimated} for a block; exact if cached, else estimate."""
        if not block:
            return {"unix": None, "estimated": True}
        con = self._cache()
        row = con.execute(
            "SELECT unix_time FROM block_times WHERE block=?", (block,)).fetchone()
        con.close()
        if row:
            return {"unix": int(row["unix_time"]), "estimated": False}
        with self._lock:
            self._wanted_blocks.add(block)
            tip = dict(self.tip)
        if tip.get("block") and tip.get("unix_time"):
            est = int(tip["unix_time"]) - (int(tip["block"]) - block) * BLOCK_SECONDS
            return {"unix": est, "estimated": True}
        return {"unix": None, "estimated": True}

    def extrinsic_signer(self, block: int, ext_index: int) -> dict[str, Any]:
        con = self._cache()
        row = con.execute(
            "SELECT signer, call FROM extrinsic_signers WHERE block=? AND ext_index=?",
            (block, ext_index)).fetchone()
        con.close()
        if row:
            return {"signer": row["signer"], "call": row["call"]}
        with self._lock:
            self._wanted_extrinsics.add((block, ext_index))
        return {"signer": None, "call": None}

    def hotkey_info(self, hotkey: str) -> dict[str, Any]:
        mg = self.metagraph or {}
        info = (mg.get("hotkeys") or {}).get(hotkey)
        if not info:
            return {"registered": False, "metagraph_block": mg.get("block")}
        out = dict(info)
        out["registered"] = True
        out["metagraph_block"] = mg.get("block")
        return out

    # -- background worker --

    def start(self) -> None:
        threading.Thread(target=self._loop, name="enrich", daemon=True).start()

    def _connect(self) -> None:
        from async_substrate_interface.sync_substrate import SubstrateInterface
        self._substrate = SubstrateInterface(url=self.network)

    def _loop(self) -> None:
        while True:
            try:
                if self._substrate is None:
                    self._connect()
                self._refresh_tip()
                self._refresh_metagraph()
                self._drain_extrinsics()
                self._drain_blocks()
                self.chain_ok = True
                self.chain_error = ""
            except Exception as exc:  # noqa: BLE001 - worker must survive anything
                self.chain_ok = False
                self.chain_error = f"{type(exc).__name__}: {exc}"[:300]
                self._substrate = None
                time.sleep(10)
            time.sleep(5)

    def _refresh_tip(self) -> None:
        sub = self._substrate
        head = sub.get_chain_finalised_head()
        block = sub.get_block_number(head)
        ts = self._block_timestamp(block_hash=head)
        if block and ts:
            self.tip = {"block": int(block), "unix_time": int(ts)}
            con = self._cache()
            con.execute(
                "INSERT OR REPLACE INTO block_times(block, unix_time) VALUES(?,?)",
                (int(block), int(ts)))
            con.commit()
            con.close()

    def _block_timestamp(self, block_hash: str | None = None,
                         block_number: int | None = None) -> int | None:
        sub = self._substrate
        if block_hash is None and block_number is not None:
            block_hash = sub.get_block_hash(block_number)
        result = sub.query("Timestamp", "Now", block_hash=block_hash)
        value = getattr(result, "value", result)
        return int(value) // 1000 if value else None

    def _fetch_block(self, block: int) -> None:
        """Fetch one block: cache its timestamp and all extrinsic signers."""
        sub = self._substrate
        block_hash = sub.get_block_hash(block)
        data = sub.get_block(block_hash=block_hash)
        ts: int | None = None
        signers: list[tuple[int, int, str, str]] = []
        for idx, ext in enumerate(data.get("extrinsics") or []):
            value = getattr(ext, "value", None) or {}
            call = value.get("call") or {}
            name = f"{call.get('call_module', '')}.{call.get('call_function', '')}"
            if name == "Timestamp.set":
                for arg in call.get("call_args") or []:
                    if arg.get("name") == "now":
                        ts = int(arg.get("value")) // 1000
            address = value.get("address") or ""
            if address:
                signers.append((block, idx, str(address), name))
        con = self._cache()
        if ts:
            con.execute(
                "INSERT OR REPLACE INTO block_times(block, unix_time) VALUES(?,?)",
                (block, ts))
        for row in signers:
            con.execute(
                "INSERT OR REPLACE INTO extrinsic_signers(block, ext_index, signer, call)"
                " VALUES(?,?,?,?)", row)
        con.commit()
        con.close()

    def _drain_extrinsics(self) -> None:
        with self._lock:
            wanted = list(self._wanted_extrinsics)[:20]
        for block, _idx in wanted:
            self._fetch_block(block)
        with self._lock:
            for item in wanted:
                self._wanted_extrinsics.discard(item)

    def _drain_blocks(self) -> None:
        with self._lock:
            wanted = sorted(self._wanted_blocks, reverse=True)[:30]
        for block in wanted:
            self._fetch_block(block)
        with self._lock:
            for item in wanted:
                self._wanted_blocks.discard(item)

    def _refresh_metagraph(self) -> None:
        if self.metagraph and time.time() - self.metagraph.get("fetched_at", 0) < METAGRAPH_TTL:
            return
        import bittensor as bt
        sub = self._subtensor
        if sub is None:
            sub = bt.Subtensor(network=self.network)
            self._subtensor = sub
        mg = sub.metagraph(self.netuid)
        # Emission is denominated in the subnet's own alpha token, not TAO. The
        # symbol is whatever this netuid registered on chain; the local
        # bittensor unit table can disagree, so never render it from there.
        symbol = (self.metagraph or {}).get("emission_symbol") \
            or str(sub.subnet(self.netuid).symbol)
        hotkeys: dict[str, Any] = {}
        for uid in range(len(mg.hotkeys)):
            emission_tempo = float(mg.emission[uid])
            hotkeys[str(mg.hotkeys[uid])] = {
                "uid": uid,
                "coldkey": str(mg.coldkeys[uid]),
                "stake_alpha": float(mg.S[uid]),
                "incentive": float(mg.incentive[uid]),
                "emission_alpha_per_tempo": emission_tempo,
                "emission_alpha_per_day": emission_tempo * (86400 / (TEMPO_BLOCKS * BLOCK_SECONDS)),
                "active": bool(mg.active[uid]),
                "validator_permit": bool(mg.validator_permit[uid]),
            }
        owner = ""
        try:
            owner = str(sub.query_subtensor("SubnetOwner", params=[self.netuid]))
        except Exception:  # noqa: BLE001
            pass
        self.metagraph = {
            "fetched_at": int(time.time()),
            "block": int(mg.block),
            "n": len(mg.hotkeys),
            "owner_coldkey": owner,
            "emission_symbol": symbol,
            "hotkeys": hotkeys,
        }
        self._kv_set("metagraph", self.metagraph)

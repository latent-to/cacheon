"""The simulated Bittensor chain: finalized reveals, events, metagraph, and weights.

It answers the SDK calls ``cacheon.chain`` makes (storage history, canonical
events, finalized head, metagraph, weight readback, signing) from in-memory
state and records what was submitted. In-process suites construct it directly;
end-to-end subprocesses reach it through ``tests/support/simulated_chain``,
whose long-lived client reads the JSON file named by ``--network`` on every call.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import types
from pathlib import Path

from cacheon import chain


def scale_reveal_frame(data: str) -> bytes:
    payload = data.encode("utf-8")
    size = len(payload)
    if size < 64:
        prefix = bytes([size << 2])
    elif size < 16_384:
        prefix = ((size << 2) | 1).to_bytes(2, "little")
    else:
        raise ValueError("test reveal exceeds the bounded production format")
    return prefix + payload


def scale_reveal_hex(data: str) -> str:
    return "0x" + scale_reveal_frame(data).hex()


def reveal_event(hotkey: str, data: str, *, netuid: int = 1, extrinsic: int = 0):
    return {
        "phase": {"ApplyExtrinsic": extrinsic},
        "event": {
            "module_id": "Commitments",
            "event_id": "CommitmentRevealed",
            "attributes": {
                "netuid": netuid,
                "who": hotkey,
            },
        },
    }


class SimulatedMetagraph:
    def __init__(self, hotkeys, permits=None, last_updates=None, *, block=0, owner_coldkey=""):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.coldkeys = [f"{hotkey}-coldkey" for hotkey in hotkeys]  # the miner wallets' coldkey names
        self.validator_permit = list(permits) if permits is not None else [True] * len(hotkeys)
        self.last_update = list(last_updates) if last_updates is not None else [0] * len(hotkeys)
        self.block = block
        self.owner_coldkey = owner_coldkey


class SimulatedSubtensor:
    def __init__(self, *, hotkeys, revealed=None, block=100,
                 registered=None, events=None, weight_rows=None, last_updates=None,
                 weights_version=29, owner="", extrinsics=None, state_path=None,
                 commits=None, rate_limit=0, reveal_delay=0, lost_submissions=0,
                 registrations=(), finalize=None, unpermitted=()):
        self._hotkeys = list(hotkeys)
        self._unpermitted = set(unpermitted)  # hotkeys without validator_permit
        self._registrations = [list(row) for row in registrations]  # [from_block, hotkeys by uid] changes
        # Finality moving while a call is in flight: after the named call ("weights" or "set_weights")
        # the finalized head becomes the given block, once.
        self._finalize = dict(finalize or {})
        self._weights_version = weights_version
        self._revealed = dict(revealed or {})  # hotkey -> ((block, data), ...)
        self._block = block
        self._registered = set(hotkeys if registered is None else registered)
        self._events = events
        self._weight_rows = list(weight_rows or [])
        self._last_updates = last_updates
        self._owner = owner
        self._extrinsics = {int(key): list(rows) for key, rows in (extrinsics or {}).items()}
        # Commit-reveal weights: [hotkey, commit_block, uids, uint16 values, reveal_block] rows; a row
        # is the hotkey's sparse vector from its reveal block on, and timelocked until then.
        self._commits = [list(row) for row in commits or ()]
        self._rate_limit = rate_limit
        self._reveal_delay = reveal_delay
        self._lost_submissions = lost_submissions  # the next N signed submissions never reach the chain
        self._state_path = state_path  # set when reopened from a file: mutations persist there
        self.substrate = types.SimpleNamespace(
            get_events=self._get_events,
            get_chain_finalised_head=lambda: self.get_block_hash(self._block),
            get_block_number=lambda block_hash: int(block_hash[2:], 16),
            get_block=lambda *, block_hash: {"extrinsics": [
                row["extrinsic"] for row in self._extrinsics.get(int(block_hash[2:], 16), ())]},
            compose_call=lambda *, call_module, call_function, call_params: {
                "call_module": call_module, "call_function": call_function, "call_args": {
                    name: "0x" + value.hex() if isinstance(value, bytes) else value
                    for name, value in call_params.items()}},
        )
        self.set_weights_calls: list[dict] = []
        self.metagraph_calls: list[int | None] = []
        self.weight_calls: list[int | None] = []
        self.set_reveal_commitment_calls: list[tuple] = []

    def metagraph(self, netuid=None, block=None):
        self.metagraph_calls.append(block)
        block = self._block if block is None else block
        revealed, hotkeys = self._revealed_weights(block), self._members(block)
        return SimulatedMetagraph(
            hotkeys,
            permits=[hotkey not in self._unpermitted for hotkey in hotkeys],
            last_updates=self._last_updates if self._last_updates is not None else [
                revealed.get(hotkey, (0,))[0] for hotkey in hotkeys],
            block=block,
            owner_coldkey=self._owner,
        )

    def _members(self, block):
        """Hotkeys by UID at ``block``: the latest registration change at or before it."""

        return [hotkeys for start, hotkeys in [[0, self._hotkeys], *self._registrations] if start <= block][-1]

    def _finalized_after(self, call):
        if call in self._finalize:
            self._block = self._finalize.pop(call)
            self._persist(block=self._block, finalize=self._finalize)

    def _revealed_weights(self, block):
        """Each hotkey's latest revealed vector at ``block``: hotkey -> (reveal_block, uids, values)."""

        latest = {}
        for hotkey, _, uids, values, reveal_block in self._commits:
            if reveal_block <= block and reveal_block >= latest.get(hotkey, (-1,))[0]:
                latest[hotkey] = (reveal_block, uids, values)
        return latest

    def sign_and_send_extrinsic(self, *, call, wallet, wait_for_inclusion, wait_for_finalization):
        """Include one coldkey-signed call at the head; a transfer emits its Balances event."""

        rows = self._extrinsics.setdefault(self._block, [])
        index, payer = len(rows), wallet.coldkey.ss58_address
        inner = call["call_args"]["calls"] if call["call_function"] == "batch_all" else [call]
        events = [{"extrinsic_idx": index, "event": {
            "module_id": "Balances", "event_id": "Transfer",
            "attributes": [payer, row["call_args"]["dest"], row["call_args"]["value"]]}}
            for row in inner if row["call_function"] == "transfer_keep_alive"]
        rows.append({"extrinsic": {"address": payer, "call": call}, "events": events})
        self._persist()
        return types.SimpleNamespace(success=True, message="", extrinsic_receipt=types.SimpleNamespace(
            block_number=self._block, extrinsic_idx=index))

    def _persist(self, **fields):
        if self._state_path is None:
            return
        state = json.loads(Path(self._state_path).read_text())
        state["revealed"] = {hotkey: [list(row) for row in rows] for hotkey, rows in self._revealed.items()}
        state["extrinsics"] = {str(block): rows for block, rows in self._extrinsics.items()}
        state["commits"], state["lost_submissions"] = self._commits, self._lost_submissions
        _write(Path(self._state_path), {**state, **fields})

    def weights(self, netuid=None, block=None):
        self.weight_calls.append(block)
        block = self._block if block is None else block
        revealed, hotkeys = self._revealed_weights(block), self._members(block)
        rows = list(self._weight_rows) + [
            (hotkeys.index(hotkey), list(zip(uids, values)))
            for hotkey, (_, uids, values) in revealed.items() if hotkey in hotkeys]
        self._finalized_after("weights")
        return rows

    def get_timelocked_weight_commits(self, netuid, block=None):
        block = self._block if block is None else block
        return [(hotkey, commit_block, "", reveal_block)
                for hotkey, commit_block, _, _, reveal_block in self._commits if reveal_block > block]

    def get_current_block(self):
        return self._block

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def get_finalized_block_number(self):
        return self._block

    def query_map(self, module, name, params=None, block=None):
        assert (module, name, params) == ("Commitments", "RevealedCommitments", [1])
        return [
            (
                hotkey,
                [
                    (scale_reveal_hex(data), reveal_block)
                    for reveal_block, data in history
                    if block is None or reveal_block <= block
                ][-chain.CHAIN_REVEAL_HISTORY_CAP :],
            )
            for hotkey, history in self._revealed.items()
        ]

    def _get_events(self, *, block_hash):
        block = int(block_hash[2:], 16)
        if self._events is not None:
            return list(self._events.get(block, ()))
        rows = []
        for hotkey, history in self._revealed.items():
            for reveal_block, data in history:
                if reveal_block == block:
                    rows.append(reveal_event(hotkey, data))
        return rows + [event for row in self._extrinsics.get(block, ()) for event in row["events"]]

    def set_reveal_commitment(self, *, wallet, netuid, data, blocks_until_reveal):
        self.set_reveal_commitment_calls.append((data, blocks_until_reveal))
        if self._state_path is not None:  # the timelock opens blocks_until_reveal after the head
            hotkey = wallet.hotkey.ss58_address
            self._revealed[hotkey] = (*self._revealed.get(hotkey, ()), (self._block + blocks_until_reveal, data))
            self._persist()
        return True

    def set_weights(self, *, wallet, netuid, uids, weights, version_key,
                    wait_for_inclusion, wait_for_finalization):
        self.set_weights_calls.append({
            "uids": uids,
            "weights": weights,
            "version_key": version_key,
            "wait_for_inclusion": wait_for_inclusion,
            "wait_for_finalization": wait_for_finalization,
        })
        if self._state_path is None:
            return True
        if self._lost_submissions:
            self._lost_submissions -= 1
            self._persist()
            raise ConnectionError("the signed submission never reached the chain")
        top = max(weights)  # the SDK's uint16 max-upscale
        reveal_block = self._block + self._reveal_delay
        self._commits.append([wallet.hotkey.ss58_address, self._block, list(uids),
                              [round(value / top * 65_535) for value in weights], reveal_block])
        self._persist()
        self._finalized_after("set_weights")
        return types.SimpleNamespace(success=True, message="", data={"reveal_round": reveal_block})

    def get_hyperparameter(self, param_name, netuid=None, block=None):
        return {"WeightsVersionKey": self._weights_version, "WeightsSetRateLimit": self._rate_limit}[param_name]

    def is_hotkey_registered(self, *, hotkey_ss58, netuid):
        return hotkey_ss58 in self._registered


def _write(path: Path, state: dict) -> None:
    """Replace the chain file atomically: other processes read it on every call."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state))
    temporary.replace(path)


def write_state(path: Path, *, hotkeys, revealed, block: int, validator: str, owner: str = "") -> Path:
    """Persist the chain a subprocess reopens: ``revealed`` is hotkey -> ((block, data), ...)."""

    _write(path, {
        "block": block, "hotkeys": list(hotkeys), "validator": validator, "owner": owner, "extrinsics": {},
        "revealed": {hotkey: [list(row) for row in rows] for hotkey, rows in revealed.items()},
        "commits": [], "rate_limit": 0, "reveal_delay": 0, "lost_submissions": 0, "registrations": [],
        "finalize": {}, "unpermitted": [],
    })
    return path


def advance(path: Path, block: int) -> None:
    """Finalize the chain up to ``block``."""

    amend(path, block=block)


def amend(path: Path, **fields) -> dict:
    """Rewrite named chain fields (owner, rate limit, reveal delay, commits, ...); returns the new state."""

    state = json.loads(path.read_text())
    assert set(fields) <= set(state), set(fields) - set(state)
    state.update(fields)
    _write(path, state)
    return state


def reopen_state(path: str | Path) -> SimulatedSubtensor:
    state = json.loads(Path(path).read_text())
    return SimulatedSubtensor(
        hotkeys=state["hotkeys"], block=state["block"], owner=state["owner"],
        revealed={key: tuple(map(tuple, rows)) for key, rows in state["revealed"].items()},
        extrinsics=state["extrinsics"], state_path=Path(path), commits=state["commits"],
        rate_limit=state["rate_limit"], reveal_delay=state["reveal_delay"],
        lost_submissions=state["lost_submissions"], registrations=state["registrations"],
        finalize=state["finalize"], unpermitted=state["unpermitted"],
    )


class LiveEndpoint:
    """A long-lived SDK client: every call reads the chain file as it is now, as a node serves new blocks."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __getattr__(self, name):
        return getattr(reopen_state(self._path), name)


class SimulatedKeypair:
    """An sr25519 stand-in: a hotkey's signature is a SHA-256 MAC under its address."""

    def __init__(self, ss58_address: str) -> None:
        self.ss58_address = ss58_address

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self.ss58_address.encode(), bytes(data), hashlib.sha256).digest()

    def verify(self, data: bytes, signature: bytes) -> bool:
        return hmac.compare_digest(self.sign(data), bytes(signature))


def state_wallet(path: str | Path | None, hotkey: str | None = None):
    """With a state path, the validator's registered signer; without one, the named hotkey's keys."""

    if path is None:
        return types.SimpleNamespace(hotkey=SimulatedKeypair(hotkey),
                                     coldkey=types.SimpleNamespace(ss58_address=f"{hotkey}-coldkey"))
    return types.SimpleNamespace(hotkey=SimulatedKeypair(json.loads(Path(path).read_text())["validator"]))

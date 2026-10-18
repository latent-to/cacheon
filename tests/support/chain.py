"""The simulated Bittensor chain: finalized reveals, events, metagraph, and weights.

It answers the SDK calls ``cacheon.chain`` makes (storage history, canonical
events, finalized head, metagraph, weight readback, signing) from in-memory
state and records what was submitted. In-process suites construct it directly;
end-to-end subprocesses reach it through ``tests/support/simulated_chain``,
which reopens the same state from a JSON file named by ``--network``.
"""

from __future__ import annotations

import json
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
    def __init__(self, hotkeys, permits=None, last_updates=None, *, block=0):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.validator_permit = list(permits) if permits is not None else [True] * len(hotkeys)
        self.last_update = list(last_updates) if last_updates is not None else [0] * len(hotkeys)
        self.block = block


class SimulatedSubtensor:
    def __init__(self, *, hotkeys, revealed=None, block=100,
                 registered=None, events=None, weight_rows=None, last_updates=None,
                 weights_version=29):
        self._hotkeys = list(hotkeys)
        self._weights_version = weights_version
        self._revealed = dict(revealed or {})  # hotkey -> ((block, data), ...)
        self._block = block
        self._registered = set(hotkeys if registered is None else registered)
        self._events = events
        self._weight_rows = list(weight_rows or [])
        self._last_updates = last_updates
        self.substrate = types.SimpleNamespace(
            get_events=self._get_events,
            get_chain_finalised_head=lambda: self.get_block_hash(self._block),
            get_block_number=lambda block_hash: int(block_hash[2:], 16),
        )
        self.set_weights_calls: list[dict] = []
        self.metagraph_calls: list[int | None] = []
        self.weight_calls: list[int | None] = []
        self.set_reveal_commitment_calls: list[tuple] = []

    def metagraph(self, netuid=None, block=None):
        self.metagraph_calls.append(block)
        return SimulatedMetagraph(
            self._hotkeys,
            last_updates=self._last_updates,
            block=self._block if block is None else block,
        )

    def weights(self, netuid=None, block=None):
        self.weight_calls.append(block)
        return list(self._weight_rows)

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
        return rows

    def set_reveal_commitment(self, *, wallet, netuid, data, blocks_until_reveal):
        self.set_reveal_commitment_calls.append((data, blocks_until_reveal))
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
        return True

    def get_hyperparameter(self, param_name, netuid=None, block=None):
        assert param_name == "WeightsVersionKey"
        return self._weights_version

    def is_hotkey_registered(self, *, hotkey_ss58, netuid):
        return hotkey_ss58 in self._registered


def write_state(path: Path, *, hotkeys, revealed, block: int, validator: str) -> Path:
    """Persist the chain a subprocess reopens: ``revealed`` is hotkey -> ((block, data), ...)."""

    path.write_text(json.dumps({
        "block": block, "hotkeys": list(hotkeys), "validator": validator,
        "revealed": {hotkey: [list(row) for row in rows] for hotkey, rows in revealed.items()},
    }))
    return path


def reopen_state(path: str | Path) -> SimulatedSubtensor:
    state = json.loads(Path(path).read_text())
    return SimulatedSubtensor(
        hotkeys=state["hotkeys"], block=state["block"],
        revealed={key: tuple(map(tuple, rows)) for key, rows in state["revealed"].items()},
    )


def state_wallet(path: str | Path):
    """The validator's public wallet: the signer the simulated chain registered."""

    ss58 = json.loads(Path(path).read_text())["validator"]
    return types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address=ss58))

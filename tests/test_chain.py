"""Chain I/O logic — pure helpers + RPC wrappers against a mock subtensor (no network)."""

from __future__ import annotations

import sys
import types

import pytest

from optima import chain
from optima.arenas import ARENAS, MINIMAX_M3_B300_TP4_DECODE_V1 as ARENA
from optima.chain.validator_loop import (
    WEIGHT_STATUS_HELD,
    WeightSafetyError,
    _atomic_write_weights_state,
    _exclusive_ledger_pass,
    _global_arena_set_sha256,
    _load_weights_state,
    _weight_publication_state,
    _weight_state_path,
)
from optima.cli import main
from optima.commit_reveal import Ledger, make_chain_scope


# --- a minimal stand-in for bittensor's subtensor (records what was called) ---

class _MockMetagraph:
    def __init__(self, hotkeys, permits=None):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.last_update = [0] * len(hotkeys)
        self.validator_permit = list(permits) if permits is not None else [True] * len(hotkeys)


class _MockSubtensor:
    def __init__(self, *, hotkeys, commitments=None, revealed=None, block=100,
                 registered=None, weight_rows=None):
        self._hotkeys = list(hotkeys)
        self._commitments = dict(commitments or {})
        self._revealed = dict(revealed or {})  # hotkey -> ((block, data), ...)
        self._weight_rows = list(weight_rows or [])
        self._block = block
        self._registered = set(hotkeys if registered is None else registered)
        self.set_weights_calls: list[dict] = []
        self.set_commitment_calls: list[str] = []
        self.set_reveal_commitment_calls: list[tuple] = []

    def metagraph(self, netuid=None):
        return _MockMetagraph(self._hotkeys)

    def get_current_block(self):
        return self._block

    def get_block_hash(self, block):
        return f"0xhash{block}"

    def get_all_commitments(self, netuid=None):
        return dict(self._commitments)

    def get_all_revealed_commitments(self, netuid=None, block=None):
        return {
            hotkey: tuple(
                entry for entry in history
                if block is None or entry[0] <= block
            )[-10:]
            for hotkey, history in self._revealed.items()
        }

    def weights(self, netuid=None):
        return list(self._weight_rows)

    def set_reveal_commitment(self, *, wallet, netuid, data, blocks_until_reveal):
        self.set_reveal_commitment_calls.append((data, blocks_until_reveal))
        return True

    def set_weights(self, *, wallet, netuid, uids, weights, version_key,
                    wait_for_inclusion, wait_for_finalization):
        self.set_weights_calls.append({"uids": uids, "weights": weights, "version_key": version_key})
        return True

    def is_hotkey_registered(self, *, hotkey_ss58, netuid):
        return hotkey_ss58 in self._registered

    def set_commitment(self, *, wallet, netuid, data):
        self.set_commitment_calls.append(data)
        return True


def _wallet(ss58: str):
    return types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address=ss58))


def _mock_bittensor(monkeypatch):
    fake = types.ModuleType("bittensor")
    fake.Wallet = lambda name, hotkey: _wallet("val")
    monkeypatch.setitem(sys.modules, "bittensor", fake)


# ---- pure helpers ----

def test_normalize():
    assert chain.normalize({"a": 2, "b": 2}) == {"a": 0.5, "b": 0.5}
    assert chain.normalize({"a": 0, "b": -1}) == {}
    assert chain.normalize({}) == {}


def test_weights_map_to_uids():
    mg = chain.MetagraphView(1, 1, "h", uids=[0, 1, 2], hotkeys=["a", "b", "c"],
                             validator_permit=[True] * 3)
    assert chain.weights_to_uid_vector({"b": 1.0}, mg) == ([1], [1.0])


def test_manual_set_weights_uses_global_registered_arena_projection(
    tmp_path, monkeypatch, capsys
):
    calls = {"global": 0, "single": 0, "set": 0}

    class FakeLedger:
        def bind_chain_scope(self, scope):
            assert scope.startswith("genesis-netuid-v1:sha256:")

        def bind_validator_hotkey(self, hotkey):
            assert hotkey == "val"

        def current_weights(self, *args, **kwargs):
            calls["single"] += 1
            raise AssertionError("one-arena projection must never be used")

        def current_weights_across_arenas(
            self, arenas, *, host_attestation_verifier, validator_hotkey
        ):
            calls["global"] += 1
            assert tuple(arena.name for arena in arenas) == tuple(sorted(ARENAS))
            assert callable(host_attestation_verifier)
            assert validator_hotkey == "val"
            return {"miner1": 1.0}

    class FakeSubtensor:
        @staticmethod
        def get_block_hash(block):
            assert block == 0
            return "0x" + "a" * 64

    _mock_bittensor(monkeypatch)
    monkeypatch.setattr("optima.commit_reveal.Ledger.load", lambda _path: FakeLedger())
    monkeypatch.setattr(chain, "connect", lambda _network: FakeSubtensor())
    monkeypatch.setattr(
        chain, "read_validator_weights", lambda *args, **kwargs: {}
    )

    def dry_run_set(_subtensor, _wallet, _netuid, weights, *, dry_run):
        calls["set"] += 1
        assert weights == {"miner1": 1.0} and dry_run is True
        return {"submitted": False, "dry_run": True, "uids": [1], "weights": [1.0]}

    monkeypatch.setattr(chain, "set_weights", dry_run_set)
    ledger_path = tmp_path / "ledger.json"
    assert main([
        "set-weights",
        "--ledger", str(ledger_path),
        "--arena", ARENA.name,
        "--netuid", "1",
        "--network", "mock",
        "--wallet", "default",
        "--hotkey", "default",
        "--dry-run",
    ]) == 0
    assert calls == {"global": 1, "single": 0, "set": 1}
    assert "would set uids=[1]" in capsys.readouterr().out


def test_manual_set_weights_respects_whole_pass_lock(
    tmp_path, monkeypatch, capsys
):
    _mock_bittensor(monkeypatch)
    monkeypatch.setattr(chain, "connect", lambda _network: object())
    ledger_path = tmp_path / "ledger.json"
    argv = [
        "set-weights",
        "--ledger", str(ledger_path),
        "--arena", ARENA.name,
        "--netuid", "1",
        "--network", "mock",
        "--dry-run",
    ]
    with _exclusive_ledger_pass(ledger_path):
        assert main(argv) == 2
    assert "another validator process owns the whole pass" in capsys.readouterr().out


def test_weight_publication_state_fails_closed_on_corruption_or_link(tmp_path):
    path = tmp_path / "weights.json"
    _atomic_write_weights_state(path, {"schema": "test", "status": "pending"})
    assert _load_weights_state(path)["status"] == "pending"

    path.write_text('{"status":"pending","status":"confirmed"}')
    path.chmod(0o600)
    with pytest.raises(WeightSafetyError, match="duplicate JSON key"):
        _load_weights_state(path)

    path.unlink()
    target = tmp_path / "target.json"
    target.write_text("{}")
    target.chmod(0o600)
    path.symlink_to(target)
    with pytest.raises(WeightSafetyError, match="cannot open.*safely"):
        _load_weights_state(path)


def test_operator_weight_hold_release_is_scoped_and_durably_archived(
    tmp_path, monkeypatch, capsys
):
    genesis_hash = "0x" + "a" * 64
    scope = make_chain_scope(
        genesis_hash=genesis_hash,
        netuid=1,
        scheme=ARENA.settlement.chain_scope_scheme,
    )
    ledger_path = tmp_path / "ledger.json"
    ledger = Ledger()
    ledger.bind_chain_scope(scope)
    ledger.bind_validator_hotkey("val")
    ledger.save(ledger_path)
    registered = tuple(ARENAS[name] for name in sorted(ARENAS))
    state_path = _weight_state_path(ledger_path, scope)
    _atomic_write_weights_state(
        state_path,
        _weight_publication_state(
            chain_scope=scope,
            arena_set_sha256=_global_arena_set_sha256(registered),
            emission_policy=ARENA.settlement.emission_policy,
            expected_weights={"miner": 1.0},
            status=WEIGHT_STATUS_HELD,
            submit_block=10,
            retry_after_block=20,
        ),
    )

    class FakeSubtensor:
        @staticmethod
        def get_block_hash(block):
            assert block == 0
            return genesis_hash

        @staticmethod
        def get_current_block():
            return 100

    _mock_bittensor(monkeypatch)
    monkeypatch.setattr(chain, "connect", lambda _network: FakeSubtensor())
    assert main([
        "set-weights",
        "--ledger", str(ledger_path),
        "--arena", ARENA.name,
        "--netuid", "1",
        "--network", "mock",
        "--release-publication-hold",
        "--release-reason", "confirmed no extrinsic landed after operator audit",
    ]) == 0
    output = capsys.readouterr().out
    assert "released held weight publication" in output
    assert not state_path.exists()
    archives = list(tmp_path.glob(state_path.name + ".released.*"))
    assert len(archives) == 1
    archived = _load_weights_state(archives[0])
    assert archived["status"] == WEIGHT_STATUS_HELD
    assert archived["operator_release"] == {
        "block": 100,
        "reason": "confirmed no extrinsic landed after operator audit",
    }


def test_weights_fail_closed_when_a_positive_champion_deregistered():
    mg = chain.MetagraphView(1, 1, "h", uids=[5, 6], hotkeys=["a", "b"],
                             validator_permit=[True, True])
    with pytest.raises(chain.ChainWeightStateError, match="absent"):
        chain.weights_to_uid_vector({"a": 1.0, "ghost": 3.0}, mg)


def test_uid_of():
    mg = chain.MetagraphView(1, 1, "h", uids=[0, 1], hotkeys=["a", "b"])
    assert mg.uid_of("b") == 1 and mg.uid_of("ghost") is None


def test_read_validator_weights_uses_authoritative_sparse_sdk_row():
    # Exact bt 10.3.2 shape observed live on netuid 307. Its default lite
    # metagraph exposes W as shape (0,), so reconciliation must use this sparse
    # storage API rather than indexing the empty dense matrix.
    metagraph = _MockMetagraph(
        ["unused-0", "unused-1", "unused-2", "validator", "miner-a", "miner-b"]
    )
    metagraph.W = []
    metagraph.last_update[3] = 777
    metagraph_calls = 0

    def read_metagraph(netuid=None):
        nonlocal metagraph_calls
        metagraph_calls += 1
        return metagraph

    subtensor = types.SimpleNamespace(
        metagraph=read_metagraph,
        weights=lambda netuid=None: [(3, [(4, 65_535), (5, 32_768)])],
    )
    assert chain.read_validator_weights(
        subtensor, 1, "validator"
    ) == {
        "miner-a": 65_535 / (65_535 + 32_768),
        "miner-b": 32_768 / (65_535 + 32_768),
    }
    snapshot = chain.read_validator_weight_snapshot(subtensor, 1, "validator")
    assert snapshot.weights == chain.read_validator_weights(
        subtensor, 1, "validator"
    )
    assert snapshot.last_update_block == 777
    assert chain.read_validator_weights(subtensor, 1, "missing") == {}
    # Each helper call owns one coherent metagraph snapshot. It must never splice
    # UID/hotkey/last-update fields from repeated RPCs that may advance independently.
    assert metagraph_calls == 4


def test_read_validator_weights_fails_closed_without_sparse_weight_api():
    metagraph = _MockMetagraph(["validator", "miner"])
    subtensor = types.SimpleNamespace(metagraph=lambda netuid=None: metagraph)
    with pytest.raises(chain.ChainWeightStateError, match="cannot fetch.*weights"):
        chain.read_validator_weights(subtensor, 1, "validator")


@pytest.mark.parametrize("last_update", [[], [True, 0], [-1, 0], [0.5, 0]])
def test_read_validator_weights_rejects_invalid_last_update(last_update):
    metagraph = _MockMetagraph(["validator", "miner"])
    metagraph.last_update = last_update
    subtensor = types.SimpleNamespace(
        metagraph=lambda netuid=None: metagraph,
        weights=lambda netuid=None: [],
    )
    with pytest.raises(
        chain.ChainWeightStateError,
        match="last-update|UID/hotkey/last-update widths differ",
    ):
        chain.read_validator_weight_snapshot(subtensor, 1, "validator")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(0, []), (0, [])], "duplicate source"),
        ([(7, [])], "source UID absent"),
        ([(0, [(1, 1), (1, 2)])], "duplicate target"),
        ([(0, [(7, 1)])], "target UID absent"),
        ([(0, [(1, -1)])], "invalid uint16"),
        ([(0, [(1, 65_536)])], "invalid uint16"),
        ([(0, [(1, 0.5)])], "invalid uint16"),
        ([(0, [(1, True)])], "invalid uint16"),
    ],
)
def test_read_validator_weights_rejects_ambiguous_or_invalid_sparse_rows(
    rows, message
):
    subtensor = _MockSubtensor(
        hotkeys=["validator", "miner"],
        weight_rows=rows,
    )
    with pytest.raises(chain.ChainWeightStateError, match=message):
        chain.read_validator_weights(subtensor, 1, "validator")


def test_read_validator_weights_missing_or_empty_source_row_means_no_active_weights():
    missing = _MockSubtensor(
        hotkeys=["validator", "miner"],
        weight_rows=[(1, [(0, 65_535)])],
    )
    empty = _MockSubtensor(
        hotkeys=["validator", "miner"],
        weight_rows=[(0, [])],
    )
    assert chain.read_validator_weights(missing, 1, "validator") == {}
    assert chain.read_validator_weights(empty, 1, "validator") == {}


# ---- RPC wrappers (mock subtensor) ----

def test_fetch_metagraph():
    st = _MockSubtensor(hotkeys=["a", "b"], block=42)
    mg = chain.fetch_metagraph(st, netuid=1)
    assert mg.uids == [0, 1] and mg.hotkeys == ["a", "b"]
    assert mg.block == 42 and mg.block_hash == "0xhash42"


def test_read_commitments():
    st = _MockSubtensor(hotkeys=["a", "b"], commitments={"a": "hashA", "b": "hashB"}, block=7)
    cs = chain.read_commitments(st, netuid=1)
    assert cs["a"].data == "hashA" and cs["a"].block == 7 and cs["b"].hotkey == "b"


def test_set_weights_dry_run_does_not_submit():
    st = _MockSubtensor(hotkeys=["a", "b"])
    res = chain.set_weights(st, wallet=None, netuid=1, weights_by_hotkey={"b": 1.0}, dry_run=True)
    assert res["submitted"] is False and res["dry_run"] is True
    assert res["uids"] == [1] and res["weights"] == [1.0]
    assert st.set_weights_calls == []  # nothing went to chain


def test_set_weights_submits_with_version_key():
    st = _MockSubtensor(hotkeys=["a", "b"])
    res = chain.set_weights(st, wallet=object(), netuid=1, weights_by_hotkey={"b": 1.0})
    assert res["submitted"] is True
    assert st.set_weights_calls == [
        {"uids": [1], "weights": [1.0], "version_key": chain.WEIGHTS_VERSION_KEY}
    ]


def test_set_weights_chain_side_failure_reported():
    # An included extrinsic can still fail on-chain (rate limit / permit / CR
    # window) — submitted must reflect the chain's verdict, not the SDK call
    # returning (measured on 307: a rate-limited CR commit was silently inert).
    class _Failed:
        success = False
        message = "CommittingWeightsTooFast"

    st = _MockSubtensor(hotkeys=["a", "b"])
    st.set_weights = lambda **kw: _Failed()
    res = chain.set_weights(st, wallet=object(), netuid=1, weights_by_hotkey={"b": 1.0})
    assert res["submitted"] is False
    assert "TooFast" in res["message"]


def test_set_weights_deregistered_champion_does_not_submit():
    st = _MockSubtensor(hotkeys=["a", "b"])
    with pytest.raises(chain.ChainWeightStateError, match="absent"):
        chain.set_weights(
            st, wallet=object(), netuid=1, weights_by_hotkey={"ghost": 1.0}
        )
    assert st.set_weights_calls == []


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -0.1, True, "1"])
def test_weight_projection_rejects_invalid_values(weight):
    metagraph = chain.MetagraphView(
        netuid=1,
        block=1,
        block_hash="0x1",
        uids=[0],
        hotkeys=["miner"],
    )
    with pytest.raises(chain.ChainWeightStateError, match="invalid"):
        chain.weights_to_uid_vector({"miner": weight}, metagraph)


def test_post_commitment_dry_run_and_submit():
    st = _MockSubtensor(hotkeys=["a"])
    assert chain.post_commitment(st, None, 1, "thehash", dry_run=True)["submitted"] is False
    assert st.set_commitment_calls == []
    assert chain.post_commitment(st, object(), 1, "thehash")["submitted"] is True
    assert st.set_commitment_calls == ["thehash"]


def test_read_revealed_commitments_takes_latest_per_hotkey():
    st = _MockSubtensor(hotkeys=["a", "b"], revealed={
        "a": ((5, "old"), (9, "new")),
        "b": ((7, "only"),),
        "c": (),  # a hotkey with an empty history is skipped
    })
    out = chain.read_revealed_commitments(st, netuid=1)
    assert out["a"].data == "new" and out["a"].block == 9
    assert out["b"].data == "only" and out["b"].block == 7
    assert "c" not in out


def test_read_reveal_history_preserves_every_row_in_global_order():
    st = _MockSubtensor(hotkeys=["alice", "bob"], revealed={
        "alice": ((20, "alice-y"), (10, "alice-x")),
        "bob": ((15, "bob-x"),),
    })
    history = chain.read_reveal_history(st, netuid=1)
    assert [(row.block, row.hotkey, row.data) for row in history] == [
        (10, "alice", "alice-x"),
        (15, "bob", "bob-x"),
        (20, "alice", "alice-y"),
    ]


def test_saturated_chain_reveal_history_paginates_to_genesis():
    st = _MockSubtensor(
        hotkeys=["alice"],
        revealed={
            "alice": tuple(
                (block, f"payload-{block}") for block in range(1, 13)
            )
        },
    )
    history = chain.read_reveal_history(st, netuid=1)
    assert [row.block for row in history] == list(range(1, 13))


def test_historical_reveal_pagination_does_not_skip_quieter_hotkeys():
    alice = tuple((block, f"alice-{block}") for block in range(1, 23))
    bob = tuple((block, f"bob-{block}") for block in range(2, 30, 2))
    st = _MockSubtensor(
        hotkeys=["alice", "bob"],
        revealed={"alice": alice, "bob": bob},
    )
    history = chain.read_reveal_history(st, netuid=1)
    observed = {(row.block, row.hotkey, row.data) for row in history}
    expected = {
        (block, hotkey, data)
        for hotkey, values in (("alice", alice), ("bob", bob))
        for block, data in values
    }
    assert observed == expected


def test_historical_reveal_pagination_preserves_shared_boundary_block_rows():
    # The head page contains nine newer rows plus only the latter of two rows at
    # block 10. Querying block 9 would silently lose the former; querying block 10
    # first removes the newer rows and exposes both boundary entries.
    alice = (
        (1, "alice-old"),
        (10, "alice-10-a"),
        (10, "alice-10-b"),
        *((block, f"alice-{block}") for block in range(20, 29)),
    )
    st = _MockSubtensor(hotkeys=["alice"], revealed={"alice": alice})

    history = chain.read_reveal_history(st, netuid=1)

    assert {(row.block, row.data) for row in history} == set(alice)


def test_ten_same_hotkey_reveals_at_one_block_fail_closed_as_ambiguous():
    st = _MockSubtensor(
        hotkeys=["alice"],
        revealed={
            "alice": tuple((10, f"same-block-{index}") for index in range(11))
        },
    )

    with pytest.raises(chain.ChainRevealHistoryError, match="same-hotkey reveals"):
        chain.read_reveal_history(st, netuid=1)


def test_saturated_chain_reveal_history_fails_when_archive_state_is_unavailable():
    class HeadOnly(_MockSubtensor):
        def get_all_revealed_commitments(self, netuid=None, block=None):
            if block is not None:
                raise RuntimeError("historical state pruned")
            return super().get_all_revealed_commitments(netuid=netuid)

    st = HeadOnly(
        hotkeys=["alice"],
        revealed={
            "alice": tuple(
                (block, f"payload-{block}") for block in range(1, 11)
            )
        },
    )
    with pytest.raises(chain.ChainRevealHistoryError, match="historical reveal state"):
        chain.read_reveal_history(st, netuid=1)


def test_post_reveal_commitment_dry_run_and_submit():
    st = _MockSubtensor(hotkeys=["a"])
    res = chain.post_reveal_commitment(st, None, 1, "payload", dry_run=True)
    assert res["submitted"] is False and st.set_reveal_commitment_calls == []
    res = chain.post_reveal_commitment(st, object(), 1, "payload", blocks_until_reveal=10)
    assert res["submitted"] is True
    assert st.set_reveal_commitment_calls == [("payload", 10)]


def test_ledger_current_weights_is_the_policy_seam():
    from optima.commit_reveal import Champion, Ledger

    led = Ledger()
    assert led.current_weights() == {}
    led.champion = Champion("h1", "hkA", 1.05, 0)
    assert led.current_weights(per_slot=False) == {"hkA": 1.0}
    assert led.current_weights() == {"hkA": 1.0}  # no per-slot state -> falls back
    led.champions = {
        "slot.x": Champion("h1", "hkA", 1.05, 0),
        "slot.y": Champion("h2", "hkB", 1.10, 0),
        "slot.z": Champion("h3", "hkA", 1.02, 0),
    }
    w = led.current_weights()
    assert w["hkA"] == pytest.approx(2 / 3) and w["hkB"] == pytest.approx(1 / 3)


def test_preflight_registered_with_permit():
    st = _MockSubtensor(hotkeys=["valX"], registered=["valX"])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is True       # registered
    assert checks[1].ok is True       # permit (mock defaults to permitted)


def test_preflight_unregistered_short_circuits():
    st = _MockSubtensor(hotkeys=["other"], registered=[])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is False and len(checks) == 1  # no permit check when unregistered

"""Global username history remains chronological across claims, overwrites and restarts."""

import copy
import sqlite3
from types import SimpleNamespace

import pytest

from cacheon.chain.okc_history import sync_registrations

A = "5CzNs6w9tmimaUwddbqWjndz7LTuQMgtDnp5bND4AuDeVsjK"
B = "5FxbrVDh7brAup3eneqGtyaUsCjXvgm6s6t5ygaikpRmQhq7"
SHARED = "Cacheon_28ef5_ac494"


def info(name):
    raw = name.encode()
    return {"fields": [{f"Raw{len(raw)}": "0x" + raw.hex()}]}


class Chain:
    def __init__(self, head=3, network=1):
        self.head, self.network = head, network
        self.events, self.extrinsics, self.metadata = {}, {}, {}
        self.visited, self.metadata_calls, self.block_calls = [], [], []
        self.fail_events = self.fail_metadata = None
        self.substrate = SimpleNamespace(get_events=self.get_events, get_block=self.get_block)

    def get_finalized_block_number(self):
        return self.head

    def get_block_hash(self, block):
        return "0x" + f"{self.network * 1000 + block:064x}"

    def get_events(self, *, block_hash):
        block = int(block_hash[2:], 16) - self.network * 1000
        self.visited.append(block)
        if block == self.fail_events:
            raise OSError("archive events unavailable")
        return self.events.get(block, [])

    def get_block(self, *, block_hash):
        block = int(block_hash[2:], 16) - self.network * 1000
        self.block_calls.append(block)
        return {"extrinsics": self.extrinsics.get(block, [])}

    def get_commitment_metadata(self, *, netuid, hotkey_ss58, block):
        self.metadata_calls.append((netuid, hotkey_ss58, block))
        if (block, hotkey_ss58) == self.fail_metadata:
            raise OSError("archive metadata unavailable")
        return self.metadata.get((netuid, hotkey_ss58, block), "")

    def add(self, block, hotkey, name, *, netuid=14, encrypted=False):
        calls, events = self.extrinsics.setdefault(block, []), self.events.setdefault(block, [])
        index = len(calls)
        data = {"fields": [{"TimelockEncrypted": {}}]} if encrypted else info(name)
        calls.append({"address": hotkey, "call": {"call_module": "Commitments", "call_function": "set_commitment",
            "call_args": [{"name": "netuid", "value": netuid}, {"name": "info", "value": data}]}})
        events.append({"phase": "ApplyExtrinsic", "extrinsic_idx": index, "event": {
            "module_id": "Commitments", "event_id": "TimelockCommitment" if encrypted else "Commitment",
            "attributes": {"netuid": netuid, "who": hotkey}}})
        events.append({"phase": "ApplyExtrinsic", "extrinsic_idx": index, "event": {
            "module_id": "System", "event_id": "ExtrinsicSuccess", "attributes": {}}})
        self.metadata[netuid, hotkey, block] = {"block": block, "info": copy.deepcopy(data)}


def stored(path):
    with sqlite3.connect(path) as db:
        return (db.execute("SELECT block FROM history_state").fetchone()[0],
                dict(db.execute("SELECT hotkey, username FROM owners")))


def test_earliest_unclaimed_registration_wins_both_uniqueness_constraints(tmp_path):
    st, path = Chain(head=4), tmp_path / "history.db"
    st.add(1, A, SHARED)
    st.add(2, B, SHARED)
    st.add(3, A, "Cacheon_28ef5_later")
    st.add(4, B, "Cacheon_ac494_available")
    finalized, owners = sync_registrations(st, 14, path)
    assert finalized == 4 and owners[A]["username"] == SHARED and owners[A]["block"] == 1
    assert owners[B]["username"] == "Cacheon_ac494_available"
    assert st.visited == [1, 2, 3, 4] and st.block_calls == []
    st.visited.clear()
    assert sync_registrations(st, 14, path)[1] == owners and st.visited == []
    st.metadata.clear()  # Later overwrites or purges cannot release indexed bindings.
    st.head = 5
    st.add(5, A, SHARED)
    assert sync_registrations(st, 14, path)[1] == owners


def test_same_block_order_uses_successful_calls_not_last_storage(tmp_path):
    st = Chain(head=1)
    st.add(1, A, SHARED)
    st.add(1, B, SHARED)
    st.add(1, A, "Cacheon_28ef5_later")
    st.add(1, A, "ignored", encrypted=True)
    _, owners = sync_registrations(st, 14, tmp_path / "history.db")
    assert list(owners) == [A] and owners[A]["username"] == SHARED
    assert owners[A]["event_index"] == 0 and st.block_calls == [1]
    assert st.metadata_calls == [(14, B, 1)]


def test_resume_commits_bounded_progress_without_returning_partial_owners(tmp_path):
    st, path = Chain(), tmp_path / "history.db"
    st.add(1, A, SHARED)
    with pytest.raises(RuntimeError, match="history incomplete"):
        sync_registrations(st, 14, path, max_blocks=1)
    assert stored(path) == (1, {A: SHARED})
    assert sync_registrations(st, 14, path, max_blocks=2)[0] == 3
    assert st.visited == [1, 2, 3]


@pytest.mark.parametrize("failure", ["events", "metadata"])
def test_rpc_error_rolls_back_whole_block_and_resumes(tmp_path, failure):
    st, path = Chain(head=2), tmp_path / "history.db"
    st.add(2, A, "Cacheon_28ef5")
    st.add(2, B, "Cacheon_ac494")
    if failure == "events":
        st.fail_events = 2
    else:
        st.fail_metadata = (2, B)
    with pytest.raises(OSError, match="archive"):
        sync_registrations(st, 14, path)
    assert stored(path) == (1, {})
    st.fail_events = st.fail_metadata = None
    assert len(sync_registrations(st, 14, path)[1]) == 2


def test_invalid_raw_and_wrong_marker_do_not_reserve_names(tmp_path):
    st = Chain(head=3)
    st.add(1, A, "cacheon_28ef5")
    st.add(2, A, "Cacheon_28ef5")
    st.metadata[14, A, 2]["info"] = {"fields": [{"Raw1": "0xff"}]}
    st.add(3, A, "Cacheon_28ef5_valid")
    assert sync_registrations(st, 14, tmp_path / "history.db")[1][A]["block"] == 3


def test_binary_case_sensitive_names_and_second_netuid(tmp_path):
    st = Chain(head=2)
    st.add(1, A, SHARED + "_Alpha", netuid=7)
    st.add(2, B, SHARED + "_alpha", netuid=7)
    assert len(sync_registrations(st, 7, tmp_path / "seven.db")[1]) == 2
    assert sync_registrations(st, 14, tmp_path / "fourteen.db")[1] == {}
    other = Chain(head=1, network=2)
    other.add(1, B, "Cacheon_ac494", netuid=7)
    assert sync_registrations(other, 7, tmp_path / "other.db")[1][B]["block"] == 1


@pytest.mark.parametrize("change", ["network", "netuid", "cursor_hash"])
def test_history_identity_cannot_be_reused(tmp_path, change):
    st, path = Chain(head=1), tmp_path / "history.db"
    sync_registrations(st, 14, path)
    netuid = 14
    if change == "network":
        st.network = 2
    elif change == "netuid":
        netuid = 7
    else:
        with sqlite3.connect(path) as db:
            db.execute("UPDATE history_state SET block_hash='wrong'")
    with pytest.raises(RuntimeError, match="different network|finalized chain"):
        sync_registrations(st, netuid, path)


@pytest.mark.parametrize("ambiguous", ["wrapper", "failed"])
def test_ambiguous_multiple_writes_stop_cursor(tmp_path, ambiguous):
    st, path = Chain(head=1), tmp_path / "history.db"
    st.add(1, A, "Cacheon_28ef5_first")
    st.add(1, A, "Cacheon_28ef5_second")
    if ambiguous == "wrapper":
        st.extrinsics[1][0]["call"]["call_module"] = "Utility"
    else:
        st.events[1][1]["event"]["event_id"] = "ExtrinsicFailed"
    with pytest.raises(RuntimeError, match="ambiguous commitment history"):
        sync_registrations(st, 14, path)
    assert stored(path) == (0, {})


def test_single_wrapped_write_and_same_block_purge_are_recoverable(tmp_path):
    st = Chain(head=2)
    st.add(1, A, "Cacheon_28ef5")
    st.extrinsics[1][0]["call"]["call_module"] = "Utility"
    st.add(2, B, "Cacheon_ac494")
    del st.metadata[14, B, 2]
    assert len(sync_registrations(st, 14, tmp_path / "history.db")[1]) == 2
    assert st.block_calls == [2]


def test_existing_apply_extrinsic_envelope_is_supported(tmp_path):
    st = Chain(head=1)
    st.add(1, A, SHARED)
    st.add(1, A, "Cacheon_28ef5_later")
    for row in st.events[1]:
        row["phase"] = {"ApplyExtrinsic": row.pop("extrinsic_idx")}
    assert sync_registrations(st, 14, tmp_path / "history.db")[1][A]["username"] == SHARED


def test_scan_uses_one_pinned_finalized_head(tmp_path):
    st, path = Chain(head=2), tmp_path / "history.db"
    st.add(1, A, "Cacheon_28ef5")
    st.add(3, B, "Cacheon_ac494")
    original = st.substrate.get_events

    def advance(**kwargs):
        st.head = 3
        return original(**kwargs)

    st.substrate.get_events = advance
    finalized, owners = sync_registrations(st, 14, path)
    assert finalized == 2 and list(owners) == [A]
    assert sync_registrations(st, 14, path)[0] == 3


def test_fresh_round_starts_empty_at_next_block_then_indexes_only_new_registrations(tmp_path):
    st, path = Chain(head=3), tmp_path / "round.db"
    st.add(1, A, SHARED)
    assert sync_registrations(st, 14, path, start_block=4) == (3, {})
    assert stored(path) == (3, {}) and st.visited == []
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT start_block, block_hash FROM history_state").fetchone() == (4, st.get_block_hash(3))
    st.head = 5
    st.add(4, B, SHARED)
    st.add(5, A, "Cacheon_28ef5_new_round")
    _, owners = sync_registrations(st, 14, path, start_block=4)
    assert owners[B]["username"] == SHARED and owners[B]["block"] == 4
    assert owners[A]["block"] == 5 and st.visited == [4, 5]
    with pytest.raises(RuntimeError, match="round start block"):
        sync_registrations(st, 14, path, start_block=5)
    assert sync_registrations(st, 14, path, start_block=4)[1] == owners


@pytest.mark.parametrize("start", [0, -1, True, 6])
def test_invalid_round_boundary_is_rejected_before_creating_database(tmp_path, start):
    path = tmp_path / "round.db"
    with pytest.raises(ValueError, match="start_block"):
        sync_registrations(Chain(head=3), 14, path, start_block=start)
    assert not path.exists()

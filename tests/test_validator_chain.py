"""Unit tests for validator.chain — pure helpers, no bittensor required."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from validator.chain import (
    CommitmentRecord,
    NotRegisteredError,
    build_commitments,
    build_winner_take_all_weights,
    parse_commitment_data,
    preflight_check,
    unique_hotkeys,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeMetagraph:
    hotkeys: list[str]


class TestParseCommitmentData:
    def test_valid_commitment(self):
        raw = json.dumps({"model": "hf/repo", "revision": "abc123"})
        assert parse_commitment_data(raw) == ("hf/repo", "abc123")

    def test_strips_whitespace(self):
        raw = json.dumps({"model": "  hf/repo  ", "revision": "  abc  "})
        assert parse_commitment_data(raw) == ("hf/repo", "abc")

    def test_extra_fields_ignored(self):
        raw = json.dumps({"model": "hf/repo", "revision": "abc", "note": "hi"})
        assert parse_commitment_data(raw) == ("hf/repo", "abc")

    def test_missing_model(self):
        raw = json.dumps({"revision": "abc"})
        assert parse_commitment_data(raw) is None

    def test_missing_revision(self):
        raw = json.dumps({"model": "hf/repo"})
        assert parse_commitment_data(raw) is None

    def test_empty_model(self):
        raw = json.dumps({"model": "", "revision": "abc"})
        assert parse_commitment_data(raw) is None

    def test_non_json(self):
        assert parse_commitment_data("not json at all") is None

    def test_json_but_not_object(self):
        assert parse_commitment_data(json.dumps([1, 2, 3])) is None

    def test_empty_string(self):
        assert parse_commitment_data("") is None

    def test_none_like_input(self):
        assert parse_commitment_data(None) is None  # type: ignore[arg-type]

    def test_non_string_model(self):
        raw = json.dumps({"model": 123, "revision": "abc"})
        assert parse_commitment_data(raw) is None


class TestBuildCommitments:
    def test_single_commitment(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1", "hk2"])
        raw = json.dumps({"model": "hf/repo", "revision": "abc"})
        revealed = {"hk1": [(100, raw)]}
        out = build_commitments(mg, revealed)
        assert set(out.keys()) == {1}
        rec = out[1]
        assert rec.uid == 1
        assert rec.hotkey == "hk1"
        assert rec.commit_block == 100
        assert rec.model == "hf/repo"
        assert rec.revision == "abc"
        assert rec.raw == raw

    def test_picks_latest_block_when_multiple_reveals(self):
        mg = FakeMetagraph(hotkeys=["hk0"])
        raw_old = json.dumps({"model": "old/m", "revision": "old"})
        raw_new = json.dumps({"model": "new/m", "revision": "new"})
        revealed = {"hk0": [(50, raw_old), (200, raw_new), (100, raw_old)]}
        out = build_commitments(mg, revealed)
        assert out[0].commit_block == 200
        assert out[0].model == "new/m"
        assert out[0].revision == "new"

    def test_skips_invalid_commitments(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1"])
        revealed = {
            "hk0": [(10, "garbage not json")],
            "hk1": [(20, json.dumps({"model": "hf/repo", "revision": "rev"}))],
        }
        out = build_commitments(mg, revealed)
        assert set(out.keys()) == {1}

    def test_hotkey_with_no_commitments_skipped(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1"])
        revealed = {"hk1": []}
        out = build_commitments(mg, revealed)
        assert out == {}

    def test_hotkey_not_in_revealed_skipped(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1"])
        revealed = {
            "hk_ghost": [(10, json.dumps({"model": "m", "revision": "r"}))]
        }
        out = build_commitments(mg, revealed)
        assert out == {}

    def test_uid_ordering_matches_metagraph(self):
        mg = FakeMetagraph(hotkeys=[f"hk{i}" for i in range(5)])
        revealed = {
            f"hk{i}": [(100 + i, json.dumps({"model": f"m{i}", "revision": "r"}))]
            for i in range(5)
        }
        out = build_commitments(mg, revealed)
        for uid, rec in out.items():
            assert rec.uid == uid
            assert rec.hotkey == f"hk{uid}"

    def test_empty_metagraph(self):
        out = build_commitments(FakeMetagraph(hotkeys=[]), {})
        assert out == {}

    def test_as_eval_key(self):
        rec = CommitmentRecord(
            uid=1, hotkey="hk1", commit_block=100,
            model="m", revision="r", raw="{}",
        )
        assert rec.as_eval_key() == ("hk1", 100)


class TestBuildWinnerTakeAllWeights:
    def test_basic(self):
        w = build_winner_take_all_weights(5, 2)
        assert w == [0.0, 0.0, 1.0, 0.0, 0.0]

    def test_winner_at_edge(self):
        w = build_winner_take_all_weights(3, 0)
        assert w == [1.0, 0.0, 0.0]

    def test_winner_uid_beyond_n_uids(self):
        # Defensive: if somehow winner_uid > n_uids-1, grow the vector
        w = build_winner_take_all_weights(3, 5)
        assert len(w) == 6
        assert w[5] == 1.0
        assert sum(w) == 1.0

    def test_negative_winner_rejected(self):
        with pytest.raises(ValueError):
            build_winner_take_all_weights(5, -1)


class TestUniqueHotkeys:
    def test_empty(self):
        assert unique_hotkeys([]) == set()

    def test_deduplicates(self):
        recs = [
            CommitmentRecord(uid=0, hotkey="hk1", commit_block=1,
                             model="m", revision="r", raw=""),
            CommitmentRecord(uid=1, hotkey="hk2", commit_block=1,
                             model="m", revision="r", raw=""),
            CommitmentRecord(uid=2, hotkey="hk1", commit_block=2,
                             model="m", revision="r", raw=""),
        ]
        assert unique_hotkeys(recs) == {"hk1", "hk2"}


class _FakeHotkey:
    def __init__(self, ss58: str) -> None:
        self.ss58_address = ss58


class _FakeWallet:
    def __init__(self, ss58: str) -> None:
        self.hotkey = _FakeHotkey(ss58)


class _FakePreflightMetagraph:
    def __init__(
        self, hotkeys: list[str], permits: list[bool], stakes: list[float],
    ) -> None:
        self.hotkeys = hotkeys
        self.validator_permit = permits
        self.S = stakes


class _FakePreflightSubtensor:
    def __init__(
        self,
        *,
        registered_hotkeys: set[str],
        hotkeys: list[str],
        permits: list[bool] | None = None,
        stakes: list[float] | None = None,
    ) -> None:
        self._registered = registered_hotkeys
        self._hotkeys = hotkeys
        self._permits = permits if permits is not None else [False] * len(hotkeys)
        self._stakes = stakes if stakes is not None else [0.0] * len(hotkeys)

    def is_hotkey_registered(self, *, netuid: int, hotkey_ss58: str) -> bool:
        return hotkey_ss58 in self._registered

    def metagraph(self, _netuid: int):
        return _FakePreflightMetagraph(
            self._hotkeys, self._permits, self._stakes,
        )


class TestPreflightCheck:
    def test_unregistered_hotkey_raises(self):
        st = _FakePreflightSubtensor(
            registered_hotkeys=set(), hotkeys=["hk_other"],
        )
        wallet = _FakeWallet(ss58="hk_us")
        with pytest.raises(NotRegisteredError) as excinfo:
            preflight_check(st, wallet, netuid=14)
        assert "hk_us" in str(excinfo.value)
        assert "14" in str(excinfo.value)

    def test_registered_with_permit(self):
        st = _FakePreflightSubtensor(
            registered_hotkeys={"hk_us"},
            hotkeys=["hk_other", "hk_us"],
            permits=[False, True],
            stakes=[10.0, 500.0],
        )
        result = preflight_check(st, _FakeWallet("hk_us"), netuid=14)
        assert result.uid == 1
        assert result.has_validator_permit is True
        assert result.stake == 500.0

    def test_registered_without_permit_warns_but_ok(self, caplog):
        st = _FakePreflightSubtensor(
            registered_hotkeys={"hk_us"},
            hotkeys=["hk_us"],
            permits=[False],
            stakes=[0.0],
        )
        with caplog.at_level("WARNING"):
            result = preflight_check(st, _FakeWallet("hk_us"), netuid=14)
        assert result.uid == 0
        assert result.has_validator_permit is False
        assert any("permit" in rec.message.lower() for rec in caplog.records)

    def test_tolerates_missing_validator_permit_attr(self):
        """Older bittensor / mock metagraphs may not expose validator_permit;
        preflight should still succeed with has_validator_permit=False."""
        class SparseMetagraph:
            hotkeys = ["hk_us"]

        class SparseSubtensor:
            def is_hotkey_registered(self, *, netuid, hotkey_ss58):
                return hotkey_ss58 == "hk_us"

            def metagraph(self, _netuid):
                return SparseMetagraph()

        result = preflight_check(SparseSubtensor(), _FakeWallet("hk_us"), netuid=14)
        assert result.uid == 0
        assert result.has_validator_permit is False
        assert result.stake == 0.0

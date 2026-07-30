"""Subnet-owner burn target selection and resolution."""

from __future__ import annotations

import pytest

from optima import chain
from optima.chain import (
    ChainWeightStateError,
    ChainWeightStateRetryableError,
    select_subnet_owner_burn_uid,
)


def test_select_prefers_owner_hotkey_among_coldkey_matches() -> None:
    uid, hotkey, candidates = select_subnet_owner_burn_uid(
        uids=[0, 3, 7, 9],
        hotkeys=["a", "b", "owner-hk", "c"],
        coldkeys=["other", "owner-ck", "owner-ck", "other"],
        owner_coldkey="owner-ck",
        owner_hotkey="owner-hk",
    )
    assert (uid, hotkey) == (7, "owner-hk")
    assert candidates == (3, 7)


def test_select_falls_back_to_lowest_uid_when_owner_hotkey_absent() -> None:
    uid, hotkey, candidates = select_subnet_owner_burn_uid(
        uids=[5, 2, 9],
        hotkeys=["hi", "lo", "mid"],
        coldkeys=["owner-ck", "owner-ck", "other"],
        owner_coldkey="owner-ck",
        owner_hotkey="not-registered",
    )
    assert (uid, hotkey) == (2, "lo")
    assert candidates == (2, 5)


def test_select_uses_lowest_uid_when_owner_hotkey_unset() -> None:
    uid, hotkey, candidates = select_subnet_owner_burn_uid(
        uids=[4, 1],
        hotkeys=["x", "y"],
        coldkeys=["owner-ck", "owner-ck"],
        owner_coldkey="owner-ck",
        owner_hotkey="",
    )
    assert (uid, hotkey) == (1, "y")
    assert candidates == (1, 4)


def test_select_fails_when_no_coldkey_matches() -> None:
    with pytest.raises(ChainWeightStateError, match="no metagraph neuron"):
        select_subnet_owner_burn_uid(
            uids=[0, 1],
            hotkeys=["a", "b"],
            coldkeys=["x", "y"],
            owner_coldkey="owner-ck",
            owner_hotkey="a",
        )


def test_select_fails_on_missing_owner_coldkey() -> None:
    with pytest.raises(ChainWeightStateError, match="subnet owner coldkey"):
        select_subnet_owner_burn_uid(
            uids=[0],
            hotkeys=["a"],
            coldkeys=["a"],
            owner_coldkey="",
            owner_hotkey="a",
        )


def test_resolve_subnet_owner_burn_target_from_metagraph(monkeypatch) -> None:
    class _Mg:
        block = 11
        uids = [0, 3, 7]
        hotkeys = ["a", "b", "owner-hk"]
        coldkeys = ["x", "owner-ck", "owner-ck"]
        validator_permit = [True, False, False]
        last_update = [1, 2, 3]
        owner_coldkey = "owner-ck"
        owner_hotkey = "owner-hk"

    class _Sub:
        def metagraph(self, *, netuid, block):
            assert netuid == 307
            assert block == 11
            return _Mg()

    monkeypatch.setattr(
        chain,
        "_weight_finalized_point",
        lambda _sub, block=None: (11, "0x" + "ab" * 32),
    )
    monkeypatch.setattr(
        chain,
        "_block_hash",
        lambda _sub, block: "0x" + "ab" * 32,
    )
    target = chain.resolve_subnet_owner_burn_target(_Sub(), 307)
    assert target.uid == 7
    assert target.hotkey == "owner-hk"
    assert target.candidate_uids == (3, 7)
    assert target.block == 11
    assert target.metagraph.block == 11
    assert target.metagraph.hotkeys == ["a", "b", "owner-hk"]
    assert target.metagraph.uid_of("owner-hk") == 7


def test_resolve_refuses_missing_owner_coldkey_without_unpinned_fallback(
    monkeypatch,
) -> None:
    class _Mg:
        block = 11
        uids = [0]
        hotkeys = ["a"]
        coldkeys = ["a"]
        validator_permit = [False]
        last_update = [1]
        owner_coldkey = ""
        owner_hotkey = ""

        def subnet(self, netuid):  # pragma: no cover - must not be consulted
            raise AssertionError("unpinned subnet() fallback must not run")

    class _Sub:
        def metagraph(self, *, netuid, block):
            return _Mg()

        def subnet(self, netuid):  # pragma: no cover
            raise AssertionError("unpinned subnet() fallback must not run")

    monkeypatch.setattr(
        chain,
        "_weight_finalized_point",
        lambda _sub, block=None: (11, "0x" + "ab" * 32),
    )
    with pytest.raises(ChainWeightStateError, match="subnet owner coldkey"):
        chain.resolve_subnet_owner_burn_target(_Sub(), 307)


def test_resolve_marks_transport_faults_retryable(monkeypatch) -> None:
    class _Sub:
        def metagraph(self, *, netuid, block):
            raise ConnectionError("rpc down")

    monkeypatch.setattr(
        chain,
        "_weight_finalized_point",
        lambda _sub, block=None: (11, "0x" + "ab" * 32),
    )
    with pytest.raises(ChainWeightStateRetryableError, match="cannot resolve"):
        chain.resolve_subnet_owner_burn_target(_Sub(), 307)


def test_metagraph_coldkeys_refuse_non_string(monkeypatch) -> None:
    class _Mg:
        block = 11
        uids = [0]
        hotkeys = ["a"]
        coldkeys = [None]
        validator_permit = [False]
        last_update = [1]
        owner_coldkey = "ck"
        owner_hotkey = "a"

    class _Sub:
        def metagraph(self, *, netuid, block):
            return _Mg()

    monkeypatch.setattr(
        chain,
        "_weight_finalized_point",
        lambda _sub, block=None: (11, "0x" + "ab" * 32),
    )
    with pytest.raises(ChainWeightStateError, match="invalid or empty coldkey"):
        chain.resolve_subnet_owner_burn_target(_Sub(), 307)

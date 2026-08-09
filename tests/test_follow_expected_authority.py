"""follow-weights --expected-authority empty → subnet-owner pin."""

from __future__ import annotations

from argparse import Namespace

import cacheon.cli as cli
from cacheon import chain


def test_follow_weights_pins_subnet_owner_when_expected_authority_empty(
    tmp_path, monkeypatch, capsys
) -> None:
    view = chain.MetagraphView(
        netuid=14,
        block=11,
        block_hash="0x" + "ab" * 32,
        uids=[0, 29],
        hotkeys=["follower", "owner-hk"],
        validator_permit=[True, True],
        last_update=[1, 1],
    )
    target = chain.SubnetOwnerBurnTarget(
        uid=29,
        hotkey="owner-hk",
        owner_coldkey="owner-ck",
        owner_hotkey="owner-hk",
        candidate_uids=(29,),
        block=11,
        block_hash=view.block_hash,
        metagraph=view,
    )

    class _Subtensor:
        def get_block_hash(self, block: int) -> str:
            return "0x" + "00" * 32

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "resolve_subnet_owner_burn_target",
        lambda *_a, **_k: target,
    )

    seen: dict[str, object] = {}

    def _fetch(url, *, signer, netuid, expected_authority=None, **_kwargs):
        seen["expected_authority"] = expected_authority
        seen["signer"] = signer.ss58_address
        # Abort before journal/publish — authority pin is what we are testing.
        raise SystemExit("stop-after-pin")

    monkeypatch.setattr(
        "cacheon.chain.weight_share.fetch_current_weights", _fetch
    )

    class _Hotkey:
        ss58_address = "follower"

        def sign(self, data: bytes) -> bytes:
            return b"\x00" * 64

    class _Wallet:
        def __init__(self, name, hotkey, path=None):
            self.hotkey = _Hotkey()

    monkeypatch.setattr(cli, "_wallet_from_args", lambda _args: _Wallet("x", "y"))

    journal = tmp_path / "private" / "follow.sqlite3"
    journal.parent.mkdir(mode=0o700, parents=True)
    args = Namespace(
        url="http://127.0.0.1:8765",
        journal_db=str(journal),
        netuid=14,
        network="finney",
        wallet="burn",
        hotkey="owner",
        wallet_path="",
        refresh_blocks=100,
        expected_authority="",
        max_skew_seconds=60,
        dry_run=True,
        watch=False,
        interval=60.0,
    )
    try:
        cli.cmd_follow_weights(args)
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert str(exc) == "stop-after-pin"
    assert seen["expected_authority"] == "owner-hk"
    out = capsys.readouterr().out
    assert "pinning subnet-owner hotkey owner-hk" in out

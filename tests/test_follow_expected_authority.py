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


def test_follow_weights_watch_reuses_one_chain_client_and_redials_after_a_failure(
    monkeypatch,
) -> None:
    """The watch loop must not leak a websocket per pass.

    Mainnet 2026-08-19: one fresh ``Subtensor`` per pass left an orphaned
    keepalive thread every interval, and one archive-node blip failed 96 of
    them at once. One client per session; drop and re-dial only after a pass
    fails.
    """

    class _Client:
        instances: list["_Client"] = []

        def __init__(self) -> None:
            self.closed = False
            _Client.instances.append(self)

        def close(self) -> None:
            self.closed = True

    dialed: list[dict[str, object]] = []

    def _connect(network: str, **options: object) -> _Client:
        dialed.append({"network": network, **options})
        return _Client()

    passes: list[object] = []
    outcomes = iter([0, RuntimeError("node blip"), 0, 2])

    def _once(args, subtensor=None):
        passes.append(subtensor)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(chain, "connect", _connect)
    monkeypatch.setattr(cli, "_cmd_follow_weights_once", _once)
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda _s: None)

    args = Namespace(network="wss://node.example.invalid", watch=True, dry_run=False, interval=1.0)
    assert cli.cmd_follow_weights(args) == 2

    # Two clients total: the first served passes 1-2 and was closed after the
    # failure; the second served passes 3-4 and was closed on the way out.
    assert len(dialed) == 2
    assert dialed[0]["retry_forever"] is True
    first, second = _Client.instances
    assert passes == [first, first, second, second]
    assert first.closed and second.closed

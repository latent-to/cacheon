"""``follow-weights --watch`` bounds its chain client: re-dialed every few passes, closed on exit.

The ``set-weights`` modes this file used to drive run end to end in
``tests/test_validator_weights.py``.
"""

from __future__ import annotations

import argparse
import types

import pytest

import cacheon.cli as cli


def test_follow_weights_watch_recycles_the_chain_client(monkeypatch):
    class FakeClient:
        closed = 0

        def close(self):
            FakeClient.closed += 1

    dials = []
    monkeypatch.setattr(
        cli, "_connect_chain_from_args", lambda args: dials.append(1) or FakeClient()
    )
    monkeypatch.setattr(cli, "_FOLLOW_WEIGHTS_CLIENT_PASSES", 3)
    statuses = iter((0, 0, 0, 0, 0, 0))
    seen = []

    def run_once(args, subtensor=None):
        seen.append(subtensor)
        if len(seen) == 2:
            raise ConnectionError("temporary chain disconnect")
        try:
            return next(statuses)
        except StopIteration:
            raise KeyboardInterrupt

    sleeps = []
    monkeypatch.setattr(cli, "_cmd_follow_weights_once", run_once)
    monkeypatch.setattr("time.sleep", sleeps.append)
    args = argparse.Namespace(watch=True, interval=5.0, dry_run=False)

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_follow_weights(args)

    # dial, fail (re-dial), passes 2-3 (recycle), passes 4-6 (recycle), dial
    assert len(dials) == 4
    assert FakeClient.closed == 3
    assert seen[0] is seen[1] and seen[2] is not seen[1]
    assert seen[2] is seen[3] and seen[4] is not seen[3]
    assert seen[4] is seen[5] is seen[6] and seen[7] is not seen[6]
    assert sleeps == [5.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0]


def test_follow_weights_watch_closes_the_client_on_a_terminal_status(monkeypatch):
    closed = []
    monkeypatch.setattr(
        cli, "_connect_chain_from_args", lambda args: types.SimpleNamespace(close=lambda: closed.append(1))
    )
    statuses = iter((3, 2))
    monkeypatch.setattr(cli, "_cmd_follow_weights_once", lambda args, subtensor=None: next(statuses))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    assert cli.cmd_follow_weights(argparse.Namespace(watch=True, interval=1.0, dry_run=False)) == 2
    assert closed == [1]

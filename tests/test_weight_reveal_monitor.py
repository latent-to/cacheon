"""Reveal alerts are independent, tolerate chain rounding, and survive monitor restarts."""

import json
import sqlite3

import pytest

from cacheon.chain.weight_reveal_monitor import check_reveals, send_discord
from cacheon.chain.weight_share import CurrentWeightOffer
from tests.test_weight_publication import Chain, _projection


def receipt(path, *, sequence=1, block=100, weights=(("alice", 1_000_000),), netuid=14):
    projection = _projection(block=block, weights=weights, netuid=netuid)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS followed_weight_publications "
                   "(sequence INTEGER, offer_json TEXT, record_json TEXT, status TEXT)")
        db.execute("INSERT INTO followed_weight_publications VALUES(?,?,?,'confirmed')", (
            sequence, json.dumps(CurrentWeightOffer.from_legacy_projection(projection).to_dict()),
            json.dumps({"submit_block": block, "reason": "block_inclusion"})))
    return projection


@pytest.mark.parametrize("netuid", [14, 307])
def test_missing_reveal_alerts_once_then_reports_recovery_without_writing_journal(tmp_path, netuid):
    journal = tmp_path / "signer.sqlite3"
    projection = receipt(journal, netuid=netuid)
    original = journal.read_bytes()
    live, messages = Chain(), []
    state = check_reveals(live, journal, {}, deadline_blocks=20, notify=messages.append)
    assert len(messages) == 1 and "committed for new hotkey alice" in messages[0]
    assert len(state["pending"]) == 1
    live.block = 120
    state = check_reveals(live, journal, state, deadline_blocks=20, notify=messages.append)
    assert len(messages) == 2 and "NOT paused" in messages[1]
    state = json.loads(json.dumps(state))  # persisted restart
    state = check_reveals(live, journal, state, deadline_blocks=20, notify=messages.append)
    assert len(messages) == 2
    live.install(projection.weights, update=101)
    state = check_reveals(live, journal, state, deadline_blocks=20, notify=messages.append)
    assert not state["pending"] and not state["alerted"]
    assert "recovered" in messages[-2] and "now active for hotkey alice" in messages[-1]
    assert len(messages) == 4
    assert journal.read_bytes() == original and live.submit_calls == 0


def test_chain_drops_zero_rounded_dust_without_false_alert(tmp_path):
    journal = tmp_path / "signer.sqlite3"
    receipt(journal, weights=(("alice", 999_999), ("bob", 1)))
    live = Chain(block=130)
    live.install({"alice": 1}, update=101)
    messages = []
    state = check_reveals(live, journal, {}, deadline_blocks=20, notify=messages.append)
    assert not state["pending"] and len(messages) == 1
    assert "committed for new hotkey bob" in messages[0]
    assert "now active" not in messages[0]


def test_later_reveal_supersedes_intermediate_commits(tmp_path):
    journal = tmp_path / "signer.sqlite3"
    receipt(journal)
    live = Chain()
    state = check_reveals(live, journal, {}, deadline_blocks=20, notify=lambda _: None)
    receipt(journal, sequence=2, block=101, weights=(("bob", 1_000_000),))
    live.block = 130
    live.install({"bob": 1}, update=102)
    messages = []
    state = check_reveals(live, journal, state, deadline_blocks=20, notify=messages.append)
    assert not state["pending"] and state["last_revealed_sequence"] == 2
    assert len(messages) == 2 and all("hotkey bob" in message for message in messages)


def test_failed_alert_delivery_can_retry_without_losing_pending_receipt(tmp_path):
    journal = tmp_path / "signer.sqlite3"
    receipt(journal)
    state = {}

    def unavailable(_):
        raise OSError("Discord unavailable")

    with pytest.raises(OSError):
        check_reveals(Chain(block=130), journal, state, deadline_blocks=20, notify=unavailable)
    assert state == {}
    messages = []
    check_reveals(Chain(block=130), journal, state, deadline_blocks=20, notify=messages.append)
    assert len(messages) == 2 and "NOT paused" in messages[0]


def test_hotkey_info_retries_delivery_and_survives_restart_without_reannouncing(tmp_path):
    journal = tmp_path / "signer.sqlite3"
    projection = receipt(journal, weights=(("alice", 500_000), ("bob", 500_000)))
    live = Chain()
    live.install({"alice": 1}, update=90)

    def offline(_):
        raise OSError("Discord offline")

    state = check_reveals(live, journal, {}, deadline_blocks=20, notify=offline)
    assert len(state["info_messages"]) == 1
    live.block = 101
    live.install(projection.weights, update=101)
    messages = []

    def deliver_commit_only(message):
        if "now active" in message:
            raise OSError("Discord offline again")
        messages.append(message)

    state = check_reveals(live, journal, state, deadline_blocks=20, notify=deliver_commit_only)
    assert len(messages) == 1 and "committed for new hotkey bob" in messages[0]
    assert not state["pending"] and len(state["info_messages"]) == 1
    state = json.loads(json.dumps(state))
    state = check_reveals(live, journal, state, deadline_blocks=20, notify=messages.append)
    assert len(messages) == 2 and "now active for hotkey bob" in messages[1]
    assert not state["info_messages"]
    receipt(journal, sequence=2, block=102, weights=(("alice", 250_000), ("bob", 750_000)))
    live.block = 102
    live.install({"alice": .25, "bob": .75}, update=102)
    state = check_reveals(live, journal, state, deadline_blocks=20, notify=messages.append)
    assert len(messages) == 2 and not state["pending"]


def test_initial_active_recipients_do_not_generate_new_hotkey_info(tmp_path):
    journal = tmp_path / "signer.sqlite3"
    projection = receipt(journal)
    live = Chain(block=101)
    live.install(projection.weights, update=101)
    messages = []
    state = check_reveals(live, journal, {}, deadline_blocks=20, notify=messages.append)
    assert not messages and state["known_hotkeys"] == ["alice"]


def test_discord_alert_disables_mentions_and_has_bounded_timeout(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    sent = []

    def post(request, timeout):
        sent.append((json.loads(request.data), timeout))
        return nullcontext(SimpleNamespace(read=lambda: b""))

    monkeypatch.setattr("cacheon.chain.weight_reveal_monitor.urlopen", post)
    send_discord("https://discord.com/api/webhooks/test", "reveal delayed")
    assert sent == [({"content": "reveal delayed", "allowed_mentions": {"parse": []}}, 15)]

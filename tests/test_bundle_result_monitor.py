"""Result delivery survives restart/outage and uses arena-scoped dashboard links."""

import json
from urllib.parse import parse_qs, urlparse

import pytest

from cacheon.chain.bundle_result_monitor import check_results, result_message


ARENAS = [{"key": name, "slug": name + "-model", "label": name.title()}
          for name in ("first", "second")]


def submission(rid, *, status="qualified", decision="PASS", reason="qualified"):
    return {"reservation_id": rid * 64, "hotkey": "miner-" + rid * 32,
            "target_id": "forward_pass", "status": status, "decision": decision,
            "reason": reason, "reward": {"reward_eligible": decision == "PASS"},
            "result": {"speedup": 1.2799, "required_speedup": 1.0428, "passes": 3,
                       "pass_limit": 4, "decode_tps": [158.3, 197.3], "ttft_s": [3.46, 2.69]}}


def api(rows):
    def get(url):
        if url.endswith("/api/arenas"):
            return {"items": ARENAS}
        query = parse_qs(urlparse(url).query)
        items = rows[query["arena"][0]]
        offset = int(query["offset"][0])
        return {"total": len(items), "items": items[offset:offset + 2]}
    return get


def test_all_arenas_paginate_seed_history_and_deliver_new_results_once():
    rows = {a["key"]: [submission("a"), submission("b", status="qualifying", decision="", reason="")]
            for a in ARENAS}
    state, messages, saved = {}, [], []
    persist = lambda: saved.append(json.loads(json.dumps(state)))
    check_results("https://dash.example", state, notify=messages.append, persist=persist, get=api(rows))
    assert messages == []
    rows["first"][1] = submission("b", status="failed", decision="FAIL", reason="slot_audit_failed")
    rows["first"].append(submission("c", status="held", decision="HOLD", reason="worker_unavailable"))
    rows["second"].append(submission("d"))
    check_results("https://dash.example", state, notify=messages.append, persist=persist, get=api(rows))
    assert len(messages) == 3 and "FAIL" in messages[0] and "HOLD" in messages[1]
    assert "/first-model?submission=" in messages[0] and "/second-model?submission=" in messages[2]
    state = saved[-1]
    check_results("https://dash.example", state, notify=messages.append, persist=persist, get=api(rows))
    assert len(messages) == 3


def test_delivery_failure_retries_only_the_unacknowledged_result():
    state, messages, saved = {"bundle_results": {a["key"]: {} for a in ARENAS}}, [], []
    rows = {"first": [submission("a"), submission("b")], "second": []}

    def flaky(message):
        if messages:
            raise OSError("Discord unavailable")
        messages.append(message)

    with pytest.raises(OSError):
        check_results("https://dash.example", state, notify=flaky,
                      persist=lambda: saved.append(json.loads(json.dumps(state))), get=api(rows))
    state = saved[-1]
    check_results("https://dash.example", state, notify=messages.append, persist=lambda: None, get=api(rows))
    assert len(messages) == 2 and "`aaaaaaaaaaaa`" in messages[0] and "`bbbbbbbbbbbb`" in messages[1]


def test_message_explains_audit_failure_without_calling_speed_a_qualification():
    row = submission("f", status="failed", decision="FAIL", reason="slot_audit_failed")
    row["result"]["audit"] = {"decision": "FAIL", "detail": "872 compared units, 2 violations; worst 0.7148"}
    message = result_message("https://dash.example/", ARENAS[0], row)
    assert "FAIL" in message and "+27.99%" in message and "needed +4.28%" in message
    assert "3/4 replay passes" in message and "872 compared units" in message
    assert "158.3 → 197.3 tok/s" in message and "3.46 → 2.69 s" in message
    assert "Reward:" not in message and len(message) < 1200
    row["reason"] = "unexpected output\n" * 500
    message = result_message("https://dash.example", ARENAS[0], row)
    assert len(message) < 1200 and message.endswith(row["reservation_id"] + ")")


def test_main_checks_results_and_saves_them_even_when_chain_is_unavailable(tmp_path, monkeypatch):
    from cacheon.chain import weight_reveal_monitor as monitor

    state_file, webhook = tmp_path / "state.json", tmp_path / "webhook"
    webhook.write_text("https://discord.com/api/webhooks/test")
    monkeypatch.setattr("sys.argv", ["monitor", "--state", str(state_file), "--webhook-file", str(webhook),
                                    "--journal-db", str(tmp_path / "journal"),
                                    "--dashboard-url", "https://dash.example"])

    def checked(base, state, *, notify, persist):
        assert base == "https://dash.example"
        state["bundle_results"] = {"first": {"delivered": ["FAIL"]}}
        persist()

    monkeypatch.setattr(monitor, "check_results", checked)
    monkeypatch.setattr(monitor.chain, "connect", lambda _: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        monitor.main()
    assert json.loads(state_file.read_text())["bundle_results"]["first"]["delivered"] == ["FAIL"]

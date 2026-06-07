"""Ledger durability, eval records, dedup, and champion history — pure, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optima.commit_reveal import EvalRecord, Ledger, SCHEMA_VERSION, make_commitment


def _eval(hotkey: str = "alice", bundle_hash: str = "h1", **kw) -> EvalRecord:
    base = dict(hotkey=hotkey, bundle_hash=bundle_hash, slot="norm.rmsnorm",
                round_id=0, score=1.1, passed=True, mean_kl=1e-4)
    base.update(kw)
    return EvalRecord(**base)


def _submit(led: Ledger, hotkey: str, content_hash: str, round_id: int,
            score: float, salt: str = "s") -> None:
    """Full commit -> reveal (original) -> score for one hotkey."""
    led.commit(hotkey, make_commitment(content_hash, hotkey, salt), round_id)
    led.reveal(hotkey, content_hash, salt, round_id)
    led.record_score(hotkey, content_hash, round_id, score, 1e-4, True)


# ---- atomic write + round-trip ----

def test_save_leaves_no_temp_file(tmp_path: Path):
    p = tmp_path / "ledger.json"
    led = Ledger()
    led.commit("alice", "c1", 0)
    led.save(p)
    assert p.exists()
    assert not list(tmp_path.glob("ledger.json.tmp.*"))  # temp renamed away


def test_save_load_roundtrip(tmp_path: Path):
    p = tmp_path / "ledger.json"
    led = Ledger()
    led.commit("alice", "c1", 0)
    led.record_eval(_eval())
    led.save(p)

    back = Ledger.load(p)
    assert len(back.commitments) == 1
    assert back.is_known("alice", "h1")
    rec = back.eval_for("alice", "h1")
    assert rec.score == 1.1 and rec.per_prompt == ()


def test_schema_version_is_written(tmp_path: Path):
    p = tmp_path / "ledger.json"
    Ledger().save(p)
    assert json.loads(p.read_text())["schema_version"] == SCHEMA_VERSION


def test_future_schema_is_refused(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}))
    with pytest.raises(ValueError):
        Ledger.load(p)


# ---- corruption is quarantined, never fatal ----

def test_corrupt_ledger_is_quarantined(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text("{ this is not valid json")
    led = Ledger.load(p)  # must not raise
    assert led.champion is None and led.commitments == []
    assert (tmp_path / "ledger.json.corrupt.1").exists()
    assert not p.exists()


def test_missing_ledger_loads_empty(tmp_path: Path):
    led = Ledger.load(tmp_path / "nope.json")
    assert led.commitments == [] and led.evals == {}


# ---- eval records + dedup ----

def test_is_known_dedup():
    led = Ledger()
    assert not led.is_known("alice", "h1")
    led.record_eval(_eval())
    assert led.is_known("alice", "h1")
    assert not led.is_known("alice", "h2")
    assert not led.is_known("bob", "h1")


def test_record_eval_overwrites_same_submission():
    led = Ledger()
    led.record_eval(_eval(score=1.0))
    led.record_eval(_eval(score=2.0))
    assert led.eval_for("alice", "h1").score == 2.0
    assert len(led.evals) == 1


# ---- forward/back-compat: tolerate unknown + missing optional fields ----

def test_load_ignores_unknown_fields(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "commitments": [{"hotkey": "a", "commitment": "c", "round_id": 0, "seq": 0,
                         "future_field": 123}],
        "reveals": [], "scores": [],
        "evals": {"a:h1": {"hotkey": "a", "bundle_hash": "h1", "slot": "s",
                           "round_id": 0, "score": 1.0, "passed": True,
                           "unknown_metric": 9.9}},
        "champion": None, "champion_history": [], "seq": 1,
    }))
    led = Ledger.load(p)  # must not raise on the unknown keys
    assert led.commitments[0].hotkey == "a"
    assert led.eval_for("a", "h1").score == 1.0


def test_load_defaults_missing_optional_fields(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "commitments": [], "reveals": [], "scores": [],
        "evals": {"a:h1": {"hotkey": "a", "bundle_hash": "h1", "slot": "s",
                           "round_id": 0, "score": 1.0, "passed": True}},
        "champion": None, "champion_history": [], "seq": 0,
    }))
    rec = Ledger.load(p).eval_for("a", "h1")
    assert rec.mean_kl == 0.0 and rec.gsm8k_acc == -1.0 and rec.block == 0


# ---- champion history (audit trail of throne changes) ----

def test_champion_history_tracks_throne_changes():
    led = Ledger()
    _submit(led, "alice", "ha", 0, score=1.10)
    assert led.settle(0).title_changed
    assert [h.hotkey for h in led.champion_history] == ["alice"]
    assert led.champion_history[0].from_hotkey is None

    _submit(led, "bob", "hb", 1, score=2.00)  # clears alice * 1.02
    assert led.settle(1).title_changed
    assert [h.hotkey for h in led.champion_history] == ["alice", "bob"]
    assert led.champion_history[1].from_hotkey == "alice"

    _submit(led, "carol", "hc", 2, score=2.00)  # ties bob, below margin -> no change
    assert not led.settle(2).title_changed
    assert len(led.champion_history) == 2


def test_champion_history_persists(tmp_path: Path):
    p = tmp_path / "ledger.json"
    led = Ledger()
    _submit(led, "alice", "ha", 0, score=1.10)
    led.settle(0)
    led.save(p)
    back = Ledger.load(p)
    assert len(back.champion_history) == 1
    assert back.champion_history[0].hotkey == "alice"

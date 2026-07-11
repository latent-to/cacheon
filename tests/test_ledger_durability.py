"""Ledger durability, eval records, and dedup — pure, no GPU."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from optima.commit_reveal import (
    EvalRecord,
    Ledger,
    LedgerAttestationError,
    PendingSettlementError,
    RETRY_KIND_INFRASTRUCTURE,
    RETRY_KIND_NO_DECISION,
    RETRY_STATE_AUTOMATIC,
    RETRY_STATE_HELD,
    SCHEMA_VERSION,
)


CHAIN_SCOPE = "genesis-netuid-v1:sha256:" + "a" * 64


def _eval(hotkey: str = "alice", bundle_hash: str = "h1", **kw) -> EvalRecord:
    base = dict(hotkey=hotkey, bundle_hash=bundle_hash, slot="norm.rmsnorm",
                round_id=0, score=1.1, passed=True, mean_kl=1e-4)
    base.update(kw)
    return EvalRecord(**base)


# ---- atomic write + round-trip ----

def test_save_leaves_no_temp_file(tmp_path: Path):
    p = tmp_path / "ledger.json"
    led = Ledger()
    led.commit("alice", "c" * 64, 0)
    led.save(p)
    assert p.exists()
    assert not list(tmp_path.glob("ledger.json.tmp.*"))  # temp renamed away


def test_save_load_roundtrip(tmp_path: Path):
    p = tmp_path / "ledger.json"
    led = Ledger()
    led.commit("alice", "c" * 64, 0)
    led.record_eval(_eval())
    led.save(p)

    back = Ledger.load(p)
    assert len(back.commitments) == 1
    assert back.is_known("alice", "h1")
    rec = back.eval_for("alice", "h1")
    assert rec.score == 1.1 and rec.mean_kl == 1e-4
    assert rec.target == "norm.rmsnorm"
    assert rec.mode == "slot"
    assert rec.member_slots == ("norm.rmsnorm",)


def test_save_fsyncs_file_and_publication_directory(tmp_path, monkeypatch):
    calls = []
    real_fsync = os.fsync

    def observed_fsync(fd):
        calls.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", observed_fsync)
    Ledger().save(tmp_path / "ledger.json")

    assert len(calls) == 2
    assert stat.S_ISREG(calls[0])
    assert stat.S_ISDIR(calls[1])


def test_schema_version_is_written(tmp_path: Path):
    p = tmp_path / "ledger.json"
    Ledger().save(p)
    assert json.loads(p.read_text())["schema_version"] == SCHEMA_VERSION


def test_future_schema_is_refused(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}))
    with pytest.raises(ValueError):
        Ledger.load(p)


@pytest.mark.parametrize("version", [True, 0, "10"])
def test_invalid_schema_version_is_refused(tmp_path: Path, version):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"schema_version": version}))
    with pytest.raises(LedgerAttestationError, match="invalid schema_version"):
        Ledger.load(path)


# ---- corruption fails closed, never starts a fresh authority ----

def test_corrupt_existing_ledger_fails_closed_without_renaming(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text("{ this is not valid json")
    with pytest.raises(LedgerAttestationError, match="refusing to start fresh"):
        Ledger.load(p)
    assert p.exists()
    assert not list(tmp_path.glob("ledger.json.corrupt.*"))


def test_ledger_read_rejects_links_duplicate_keys_and_nonfinite_json(tmp_path: Path):
    real = tmp_path / "real.json"
    Ledger().save(real)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(real)
    with pytest.raises(LedgerAttestationError, match="refusing to start fresh"):
        Ledger.load(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(real, hardlink)
    with pytest.raises(LedgerAttestationError, match="bounded owner-controlled"):
        Ledger.load(real)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":10,"schema_version":10}')
    with pytest.raises(LedgerAttestationError, match="duplicate"):
        Ledger.load(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}')
    with pytest.raises(LedgerAttestationError, match="invalid ledger JSON constant"):
        Ledger.load(nonfinite)

    writable = tmp_path / "writable.json"
    Ledger().save(writable)
    writable.chmod(0o666)
    with pytest.raises(LedgerAttestationError, match="owner-controlled"):
        Ledger.load(writable)


def test_ledger_save_refuses_nonfinite_records(tmp_path: Path):
    ledger = Ledger()
    ledger.record_eval(_eval(score=float("nan")))
    with pytest.raises(ValueError, match="JSON"):
        ledger.save(tmp_path / "ledger.json")


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
        "commitments": [{"hotkey": "a", "commitment": "c" * 64, "round_id": 0, "seq": 0,
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


def test_unscoped_legacy_commitments_cannot_adopt_a_new_chain(tmp_path: Path):
    path = tmp_path / "legacy.json"
    led = Ledger()
    led.commit("alice", "c" * 64, 0)
    led.save(path)
    loaded = Ledger.load(path)
    with pytest.raises(LedgerAttestationError, match="without a chain scope"):
        loaded.bind_chain_scope(CHAIN_SCOPE)


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
    assert rec.mean_kl == 0.0 and rec.gsm8k_acc == -1.0
    assert rec.target == "s" and rec.mode == "slot"
    assert rec.member_slots == ("s",)


def test_legacy_arena_winner_without_host_evidence_requires_requalification():
    led = Ledger()
    bracket = "arena@fingerprint"
    led.record_eval(_eval(
        bundle_hash="b" * 64,
        arena_bracket=bracket,
        passed=True,
        score=1.05,
        host_attestation_sha256="",
    ))
    assert not led.is_known(
        "alice", "b" * 64, arena_bracket=bracket, require_authoritative=True
    )

    led.record_eval(_eval(
        bundle_hash="c" * 64,
        arena_bracket=bracket,
        passed=False,
        score=0.0,
        host_attestation_sha256="",
    ))
    # Arena-shaped terminal rows without an exact chain/validator binding are not
    # authoritative dedup state, even when they represent a failure.
    assert not led.is_known("alice", "c" * 64, arena_bracket=bracket)


@pytest.mark.parametrize("legacy_schema", [6, 8])
def test_pre_v9_arena_eval_is_requalified_because_authority_was_unknown(
    tmp_path: Path, legacy_schema: int,
):
    bracket = "arena@fingerprint"
    path = tmp_path / f"v{legacy_schema}.json"
    path.write_text(json.dumps({
        "schema_version": legacy_schema,
        "commitments": [],
        "reveals": [],
        "scores": [],
        "evals": {
            "cached-key": {
                "hotkey": "alice",
                "bundle_hash": "d" * 64,
                "slot": "norm.rmsnorm",
                "round_id": 0,
                "score": 0.0,
                "passed": False,
                "arena_bracket": bracket,
                "chain_scope": CHAIN_SCOPE,
            }
        },
        "retries": {},
        "champion": None,
        "champions": {},
        "arena_champions": {},
        "chain_scope": CHAIN_SCOPE,
        "seq": 0,
    }))
    loaded = Ledger.load(path)
    record = loaded.eval_for("alice", "d" * 64, arena_bracket=bracket)
    assert record is not None and record.development_only
    assert not loaded.is_known(
        "alice", "d" * 64, arena_bracket=bracket,
        require_authoritative=True,
    )


def test_pre_v9_pending_settlement_fails_loudly_instead_of_migrating(
    tmp_path: Path,
):
    path = tmp_path / "v8-pending.json"
    path.write_text(json.dumps({
        "schema_version": 8,
        # The legacy row's contents are deliberately irrelevant: merely having
        # unresolved v8 work must stop before a loader can normalize, discard,
        # or accidentally broaden its old evidence digest.
        "pending_settlements": {"legacy-disposition": {}},
    }))

    with pytest.raises(
        PendingSettlementError,
        match="pre-v9 pending settlement cannot be migrated safely",
    ):
        Ledger.load(path)

    assert path.exists()
    assert not list(tmp_path.glob("v8-pending.json.corrupt.*"))


def test_non_boolean_development_marker_is_never_authoritative():
    bracket = "arena@fingerprint"
    led = Ledger()
    led.record_eval(_eval(
        bundle_hash="e" * 64,
        arena_bracket=bracket,
        development_only=0,
    ))
    assert not led.is_known(
        "alice", "e" * 64, arena_bracket=bracket,
        require_authoritative=True,
    )


def test_legacy_retry_rows_default_to_automatic_no_decision(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({
        "schema_version": 3,
        "commitments": [],
        "reveals": [],
        "scores": [],
        "evals": {},
        "retries": {
            "cached-key-is-not-authority": {
                "hotkey": "alice",
                "bundle_hash": "b" * 64,
                "arena_bracket": "arena@fingerprint",
                "chain_scope": CHAIN_SCOPE,
                "attempts": 7,
                "next_block": 123,
                "last_reason": "legacy unclassified retry",
            },
        },
        "arena_champions": {},
        "chain_scope": CHAIN_SCOPE,
        "seq": 0,
    }))

    led = Ledger.load(p)
    retry = led.retry_for(
        "alice", "b" * 64, arena_bracket="arena@fingerprint"
    )
    assert retry is not None
    assert retry.kind == RETRY_KIND_NO_DECISION
    assert retry.state == RETRY_STATE_AUTOMATIC
    assert retry.attempts == retry.no_decision_attempts == 7
    assert retry.infrastructure_attempts == 0
    assert not retry.lease_id and retry.lease_block == 0

    held = led.begin_retry_attempt(
        hotkey="alice",
        bundle_hash="b" * 64,
        arena_bracket="arena@fingerprint",
        current_block=retry.next_block,
        reason="post-migration eligibility check",
        max_automatic_infrastructure_attempts=3,
        max_automatic_no_decision_attempts=4,
        max_total_attempts=6,
    )
    assert held.state == RETRY_STATE_HELD
    assert held.attempts == 7  # migration/eligibility must not mint attempt eight


def test_held_infrastructure_retry_roundtrips_and_is_not_dedup(tmp_path: Path):
    p = tmp_path / "ledger.json"
    led = Ledger()
    led.bind_chain_scope(CHAIN_SCOPE)
    retry = None
    for block in (10, 20, 40):
        retry = led.record_retry(
            hotkey="alice",
            bundle_hash="c" * 64,
            arena_bracket="arena@fingerprint",
            kind=RETRY_KIND_INFRASTRUCTURE,
            current_block=block,
            reason="host runtime unavailable",
            base_backoff_blocks=10,
            max_backoff_blocks=100,
            max_automatic_infrastructure_attempts=3,
            max_automatic_no_decision_attempts=4,
            max_total_attempts=6,
        )
    assert retry is not None and retry.state == RETRY_STATE_HELD
    led.save(p)

    loaded = Ledger.load(p)
    held = loaded.retry_for(
        "alice", "c" * 64, arena_bracket="arena@fingerprint"
    )
    assert held is not None
    assert held.kind == RETRY_KIND_INFRASTRUCTURE
    assert held.state == RETRY_STATE_HELD
    assert not loaded.is_known(
        "alice", "c" * 64, arena_bracket="arena@fingerprint"
    )


def test_old_score_rows_normalize_slot_to_singleton_target(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "commitments": [],
        "reveals": [],
        "scores": [{
            "hotkey": "a",
            "content_hash": "h1",
            "round_id": 0,
            "score": 1.1,
            "kl_mean": 0.0,
            "passed": True,
            "slot": "norm.rmsnorm",
        }],
        "evals": {},
        "champion": None,
        "champions": {},
        "seq": 0,
    }))

    score = Ledger.load(p).scores[0]
    assert score.slot == "norm.rmsnorm"
    assert score.target == "norm.rmsnorm"
    assert score.mode == "slot"
    assert score.member_slots == ("norm.rmsnorm",)


def test_atomic_target_identity_roundtrips_without_member_slot_alias(tmp_path: Path):
    p = tmp_path / "ledger.json"
    led = Ledger()
    led.record_score(
        "a",
        "h1",
        0,
        1.1,
        kl_mean=0.0,
        passed=True,
        target="collective.moe_epilogue.v1",
        mode="atomic",
        member_slots=(
            "collective.ar_residual_rmsnorm",
            "collective.moe_finalize_ar_rmsnorm",
        ),
    )
    led.save(p)

    score = Ledger.load(p).scores[0]
    assert score.slot == ""
    assert score.target == "collective.moe_epilogue.v1"
    assert score.member_slots == (
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
    )

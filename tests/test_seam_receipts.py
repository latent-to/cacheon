"""Seam-activation receipts — the anti-phantom-pass gate (optima/receipts.py).

Pins the failure mode hit for real on 2026-07-07: a candidate engine that comes up
WITHOUT the seam (missing .pth bootstrap / bundle load fallback) produced
bit-identical logits, KL exactly 0.0, and a PASS verdict. The eval driver must
demand positive evidence from the ranks, and the diagnosis must distinguish
"no bootstrap at all" from "bundle load fell back to baseline".
"""

from __future__ import annotations

import os

import pytest

from optima import receipts
from optima.registry import Eligibility, KernelImpl, KernelRegistry


@pytest.fixture()
def receipt_dir(tmp_path, monkeypatch):
    rdir = tmp_path / "receipts"
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(rdir))
    return rdir


def test_no_env_is_a_silent_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("OPTIMA_SEAM_RECEIPT_DIR", raising=False)
    receipts.write("active", {"bundle": "x"})  # must not raise, must not create files
    receipts.completed("norm.rmsnorm")
    assert list(tmp_path.iterdir()) == []


def test_no_env_does_not_consume_completed_once_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.delenv("OPTIMA_SEAM_RECEIPT_DIR", raising=False)
    receipts.completed("norm.rmsnorm")
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(tmp_path))
    receipts.completed("norm.rmsnorm")
    assert len(receipts.collect(tmp_path, "completed")) == 1


def test_write_and_collect_roundtrip(receipt_dir):
    receipts.write("active", {"bundle": "b", "slots": ["s"]})
    receipts.write("fired", {"slot": "collective.ar_residual_rmsnorm"},
                   tag="collective.ar_residual_rmsnorm")
    active = receipts.collect(receipt_dir, "active")
    assert active[0]["bundle"] == "b" and active[0]["slots"] == ["s"]
    assert active[0]["pid"] == os.getpid()
    fired = receipts.collect(receipt_dir, "fired")
    assert fired[0]["slot"] == "collective.ar_residual_rmsnorm"
    assert {"pid", "rank", "world_size"} <= fired[0].keys()
    # tag is sanitized into the filename; pid keeps concurrent ranks from colliding
    names = [p.name for p in receipt_dir.iterdir()]
    assert any(n.startswith("fired.collective.ar_residual_rmsnorm") for n in names)
    assert all(str(os.getpid()) in n for n in names)


def test_require_passes_with_receipt(receipt_dir):
    receipts.write("active", {"bundle": "b"})
    got = receipts.require(receipt_dir, "active", context="test")
    assert got and got[0]["bundle"] == "b"


def test_require_diagnoses_missing_bootstrap(receipt_dir):
    receipt_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="WITHOUT the miner kernel"):
        receipts.require(receipt_dir, "active", context="test")


def test_require_diagnoses_bundle_fallback(receipt_dir):
    receipts.write("load_failed", {"bundle": "b", "reason": "exception during load"})
    with pytest.raises(RuntimeError, match="FELL BACK to baseline"):
        receipts.require(receipt_dir, "active", context="test")


def test_registry_lookup_writes_fired_once(receipt_dir, monkeypatch):
    # The fired guard is process-global (one receipt per slot per process); isolate it
    # so earlier suite tests that exercised lookup() can't mask the write.
    monkeypatch.setattr("optima.registry._FIRED_SLOTS", set())
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="activation.silu_and_mul", bundle_id="t",
                            entry=lambda *a: None, eligibility=Eligibility()))
    reg.enable()
    for _ in range(3):  # repeated lookups -> exactly one receipt (per-process guard)
        assert reg.lookup("activation.silu_and_mul", dtype_name="bfloat16",
                          last_dim=128, arch=None) is not None
    fired = receipts.collect(receipt_dir, "fired")
    assert len(fired) == 1 and fired[0]["slot"] == "activation.silu_and_mul"


def test_registry_miss_writes_nothing(receipt_dir, monkeypatch):
    monkeypatch.setattr("optima.registry._FIRED_SLOTS", set())
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="norm.rmsnorm", bundle_id="t", entry=lambda *a: None,
                            eligibility=Eligibility(dtypes=frozenset({"float16"}))))
    reg.enable()
    # Ineligible (dtype mismatch) -> no selection -> no fired receipt.
    assert reg.lookup("norm.rmsnorm", dtype_name="bfloat16", last_dim=128, arch=None) is None
    assert receipts.collect(receipt_dir, "fired") == []


def test_completed_and_fallback_are_once_per_slot_process(receipt_dir, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    for _ in range(3):
        receipts.completed("norm.rmsnorm")
        receipts.fallback("norm.rmsnorm", RuntimeError("candidate exploded"))
    completed = receipts.collect(receipt_dir, "completed")
    fallback = receipts.collect(receipt_dir, "fallback")
    assert len(completed) == 1 and completed[0]["slot"] == "norm.rmsnorm"
    assert len(fallback) == 1 and fallback[0]["error_type"] == "RuntimeError"
    for item in (*completed, *fallback):
        assert item["pid"] == os.getpid()
        assert {"rank", "world_size"} <= item.keys()


def test_completed_gate_requires_every_slot_on_every_active_member():
    active = [
        {"pid": 10, "rank": 0, "world_size": 2},
        {"pid": 11, "rank": 1, "world_size": 2},
    ]
    completed = [
        {"slot": slot, "pid": pid, "rank": rank, "world_size": 2}
        for pid, rank in ((10, 0), (11, 1))
        for slot in ("slot.a", "slot.b")
    ]
    ok, desc = receipts.completed_gate(
        completed, expected_slots=("slot.a", "slot.b"), member_receipts=active)
    assert ok and "4/4" in desc

    ok, desc = receipts.completed_gate(
        completed[:-1], expected_slots=("slot.a", "slot.b"), member_receipts=active)
    assert not ok and "slot.b" in desc and "pid:11" in desc


def test_completed_gate_fails_on_any_selected_candidate_fallback():
    active = [{"pid": 10, "rank": 0, "world_size": 1}]
    completed = [{"slot": "slot.a", "pid": 10, "rank": 0, "world_size": 1}]
    fallback = [{"slot": "slot.a", "pid": 10, "rank": 0, "world_size": 1,
                 "error_type": "RuntimeError"}]
    ok, desc = receipts.completed_gate(
        completed, expected_slots=("slot.a",), member_receipts=active,
        fallback_receipts=fallback)
    assert not ok and "fallbacks" in desc


def test_coverage_without_active_receipts_expands_known_world_size():
    detail = receipts.coverage_matrix(
        [{"slot": "slot.a", "pid": 11, "rank": 1, "world_size": 2}],
        expected_slots=("slot.a",),
    )
    assert not detail["ok"]
    assert detail["basis"] == "rank"
    assert detail["missing"] == [{"slot": "slot.a", "member": "rank:0"}]

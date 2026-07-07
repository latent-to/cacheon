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
    assert list(tmp_path.iterdir()) == []


def test_write_and_collect_roundtrip(receipt_dir):
    receipts.write("active", {"bundle": "b", "slots": ["s"]})
    receipts.write("fired", {"slot": "collective.ar_residual_rmsnorm"},
                   tag="collective.ar_residual_rmsnorm")
    assert receipts.collect(receipt_dir, "active") == [{"bundle": "b", "slots": ["s"]}]
    fired = receipts.collect(receipt_dir, "fired")
    assert fired == [{"slot": "collective.ar_residual_rmsnorm"}]
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
    assert fired == [{"slot": "activation.silu_and_mul"}]


def test_registry_miss_writes_nothing(receipt_dir, monkeypatch):
    monkeypatch.setattr("optima.registry._FIRED_SLOTS", set())
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="norm.rmsnorm", bundle_id="t", entry=lambda *a: None,
                            eligibility=Eligibility(dtypes=frozenset({"float16"}))))
    reg.enable()
    # Ineligible (dtype mismatch) -> no selection -> no fired receipt.
    assert reg.lookup("norm.rmsnorm", dtype_name="bfloat16", last_dim=128, arch=None) is None
    assert receipts.collect(receipt_dir, "fired") == []

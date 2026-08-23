"""The environment an engine session hands its ranks, asserted directly.

The kernel trace previously shipped with a commit message stating the audit
launch armed it, while nothing in production set the variable anywhere. Nothing
could contradict that, because the dict was built inline behind an sglang
import. These assertions are the thing that would have.
"""

from __future__ import annotations

from dataclasses import dataclass

from cacheon.eval.engine_worker import build_session_environment


@dataclass(frozen=True)
class _AuditPolicy:
    sample_rate_ppm: int = 25_000
    validator_seed: str = "ff01"


def _env(**overrides):
    row = {
        "active": True,
        "bundle_path": "/candidate",
        "framework_mode": False,
        "receipt_dir": "/receipts",
        "audit_policy": None,
        "install_seams": True,
        "gate_environment": {},
    }
    row.update(overrides)
    return build_session_environment(**row)


def test_the_audit_arm_arms_the_kernel_trace():
    env = _env(audit_policy=_AuditPolicy())

    assert env["CACHEON_KERNEL_TRACE"] == "1"
    assert env["CACHEON_SLOT_AUDIT"] != ""


def test_a_timed_arm_leaves_the_kernel_trace_disarmed():
    """The whole point: measuring must not pay for describing.

    Arming replaces every registry entry with a wrapper for the process
    lifetime, so a timed candidate arm would carry that cost inside the
    measurement it exists to produce.
    """

    for arm in ({"active": True}, {"active": False, "bundle_path": ""}):
        env = _env(**arm)
        assert env["CACHEON_KERNEL_TRACE"] == ""
        assert env["CACHEON_SLOT_AUDIT"] == ""
        assert env["CACHEON_SLOT_AUDIT_SEED"] == ""


def test_the_trace_and_the_audit_policy_arm_and_disarm_together():
    """Neither may arm without the other; that is what binds it to the role."""

    audited = _env(audit_policy=_AuditPolicy())
    timed = _env()

    assert bool(audited["CACHEON_KERNEL_TRACE"]) is bool(audited["CACHEON_SLOT_AUDIT"])
    assert bool(timed["CACHEON_KERNEL_TRACE"]) is bool(timed["CACHEON_SLOT_AUDIT"])


def test_a_stock_arm_carries_no_bundle_and_no_trace():
    env = _env(active=False, bundle_path="/candidate")

    assert (env["CACHEON_ACTIVE"], env["CACHEON_BUNDLE_PATH"]) == ("0", "")
    assert env["CACHEON_KERNEL_TRACE"] == ""


def test_seam_gate_environment_cannot_be_dropped_or_shadowed():
    env = _env(gate_environment={"CACHEON_SEAM_GATE_X": "1"})

    assert env["CACHEON_SEAM_GATE_X"] == "1"
    assert env["SGLANG_PLUGINS"] == "cacheon"
    assert _env(install_seams=False)["SGLANG_PLUGINS"] == ""


def test_every_value_is_a_string_because_it_becomes_process_environment():
    env = _env(audit_policy=_AuditPolicy())

    assert all(isinstance(key, str) for key in env)
    assert all(isinstance(value, str) for value in env.values())

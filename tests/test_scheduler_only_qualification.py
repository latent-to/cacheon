"""Scheduler-only proposals retain the numerical audit of their actual assembled parent."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.b300_registered_qualification import B300RegisteredQualificationError, _audit_members


def test_pure_scheduler_audits_its_parent_and_mixed_submissions_audit_their_own_slots():
    examples = Path(__file__).resolve().parents[1] / "examples"
    policy = "sglang.srt.managers.schedule_policy.SchedulePolicy.calc_priority"
    parent = SimpleNamespace(root=examples / "miner_node_identity")
    assert _audit_members((policy,), parent) == ("model.layers.*.mlp",)
    assert _audit_members((policy, "tree_cache"), parent) == ("tree_cache",)
    with pytest.raises(B300RegisteredQualificationError, match="no model or cache slot"):
        _audit_members((policy,), SimpleNamespace(root=examples / "miner_prefill_scheduler"))

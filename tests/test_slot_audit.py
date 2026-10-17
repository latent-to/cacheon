"""In-engine slot audit (cacheon/audit.py) — unit + dispatcher-wiring tests.

The audit is the fidelity gate that replaced rollout-KL as primary on
launch-nondeterministic arenas (2026-07-07): sampled dispatcher calls re-run the
captured stock baseline on pre-call clones and compare under the slot's verify
tolerances; per-rank rolling receipts feed the eval driver's gate.
"""


import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from cacheon import audit, receipts
from cacheon.integrations import sglang_dsv4_prune as prune
from cacheon.integrations import sglang_nodes as nodes
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry

SLOT = "norm.rmsnorm"


@pytest.fixture(autouse=True)
def _fresh_audit(monkeypatch):
    monkeypatch.setattr(audit, "_state", {"rate": None, "rng": None})
    monkeypatch.setattr(audit, "_stats", {})
    monkeypatch.delenv("CACHEON_SLOT_AUDIT", raising=False)
    monkeypatch.delenv("CACHEON_SLOT_AUDIT_SEED", raising=False)
    monkeypatch.delenv("CACHEON_SEAM_RECEIPT_DIR", raising=False)


def _arm(monkeypatch, rate="1.0", seed="7"):
    monkeypatch.setenv("CACHEON_SLOT_AUDIT", rate)
    monkeypatch.setenv("CACHEON_SLOT_AUDIT_SEED", seed)


# ---- sampling ------------------------------------------------------------------


def test_disabled_without_env():
    assert not any(audit.sampled() for _ in range(20))


def test_rate_one_always_samples(monkeypatch):
    _arm(monkeypatch)
    assert all(audit.sampled() for _ in range(20))


def test_bad_rate_is_disabled(monkeypatch):
    monkeypatch.setenv("CACHEON_SLOT_AUDIT", "not-a-number")
    assert not any(audit.sampled() for _ in range(20))


def test_seeded_sampling_is_reproducible(monkeypatch):
    # Collective baselines REQUIRE rank-identical decisions: same seed -> same stream.
    _arm(monkeypatch, rate="0.5", seed="123")
    a = [audit.sampled() for _ in range(50)]
    audit._state.update(rate=None, rng=None)  # simulate a second rank, same env
    b = [audit.sampled() for _ in range(50)]
    assert a == b


# ---- recorded units and the stock reference --------------------------------------


def test_a_node_address_is_graded_by_its_adapter_and_recorded_as_units(monkeypatch):
    # The node adapter grades against stock and records the units it graded; a result
    # it could not compare is counted apart from a violation.
    _arm(monkeypatch)
    from cacheon.eval.oci_session_protocol import AuditReceiptFacts, SlotAuditControl
    from cacheon.integrations.sglang_nodes import _MODE

    audit.compare_error("model.layers.*.mlp")
    audit.record_fraction("model.layers.*.mlp", 1.0, 0.9, _MODE)
    audit.record_fraction("model.layers.*.mlp", 0.5, 0.9, _MODE)
    stats = audit._stats["model.layers.*.mlp"]
    assert (stats["n"], stats["violations"], stats["compare_errors"]) == (2, 1, 1)
    assert (stats["worst_frac"], stats["min_ratio"]) == (0.5, 0.9)
    # The adapter's own mode name and a starred slot were each refused by the wire
    # protocol, which would have turned every node verdict into an infrastructure error.
    facts = AuditReceiptFacts(
        "model.layers.*.mlp", stats["n"], stats["violations"], 0, stats["compare_errors"],
        stats["worst_frac"], stats["min_ratio"], stats["mode"], 1, 0, 1,
    )
    assert facts.mode == "matched_ratio"
    assert SlotAuditControl(125_000, 32, ("logits_processor", "model.layers.*.mlp"), 1)


def test_a_baseline_error_is_a_refusal_not_a_crash_or_compare_error(monkeypatch):
    _arm(monkeypatch)

    def boom():
        raise RuntimeError("baseline exploded")

    assert audit.capture_reference(SLOT, boom) is None
    assert audit._stats[SLOT]["compare_errors"] == 0
    assert audit._stats[SLOT]["baseline_refused"] == 1
    assert audit._stats[SLOT]["n"] == 0


def test_the_stock_reference_is_a_copy_whether_stock_returns_a_tensor_or_a_tuple():
    x = torch.randn(4, 8)
    for returned, count in ((x, 1), ((x, x), 2)):
        kept = audit.capture_reference(SLOT, lambda: returned)
        assert len(kept) == count and all(torch.equal(row, x) for row in kept)
        assert all(row.data_ptr() != x.data_ptr() for row in kept)


# ---- receipts ------------------------------------------------------------------


def test_rolling_receipt_overwrites(monkeypatch, tmp_path):
    _arm(monkeypatch)
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(tmp_path))
    audit.record_fraction(SLOT, 1.0, 0.9, "matched_ratio")
    audit.record_fraction(SLOT, 1.0, 0.9, "matched_ratio")
    files = list(tmp_path.glob("audit*.json"))
    assert len(files) == 1  # rolling: same kind+tag+pid file, overwritten
    got = receipts.collect(tmp_path, "audit")
    assert got[0]["n"] == 2 and got[0]["violations"] == 0


# ---- eval-driver gate ----------------------------------------------------------


def test_gate_no_receipts_is_not_a_kernel_verdict():
    decision, desc = audit.gate([], min_calls=32)
    assert decision == "NO_DECISION" and "no audit receipts" in desc


def test_gate_gross_violation_fails():
    decision, desc = audit.gate(
        [{"n": 100, "violations": 1, "worst_frac": 0.2, "min_ratio": 0.985}],
        min_calls=32,
    )
    assert decision == "FAIL" and "1 violations" in desc and "grossly" in desc


def test_gate_violation_without_a_recorded_bar_fails_closed():
    decision, _ = audit.gate(
        [{"n": 100, "violations": 1, "worst_frac": 0.98}], min_calls=32
    )
    assert decision == "FAIL"


def test_gate_insufficient_coverage_is_not_a_kernel_verdict():
    decision, desc = audit.gate([{"n": 5, "violations": 0}], min_calls=32)
    assert decision == "NO_DECISION" and "insufficient coverage" in desc


def test_gate_wrong_kernel_on_thin_coverage_still_fails():
    decision, _ = audit.gate(
        [{"n": 5, "violations": 5, "worst_frac": 0.02, "min_ratio": 0.985}],
        min_calls=32,
    )
    assert decision == "FAIL"


def test_gate_compare_errors_fail_closed():
    decision, _ = audit.gate(
        [{"n": 100, "violations": 0, "compare_errors": 2}], min_calls=32
    )
    assert decision == "FAIL"


def test_gate_clean_passes_and_sums_ranks():
    decision, desc = audit.gate(
        [{"n": 20, "violations": 0}, {"n": 20, "violations": 0}], min_calls=32)
    assert decision == "PASS" and "40 audited calls" in desc


def _member(rank, **facts):
    return {
        "slot": "collective.dp_output_projection_norm", "pid": 100 + rank,
        "rank": rank, "world_size": 4, "compare_errors": 0, "min_ratio": 0.985,
        **facts,
    }


def _grade_members(**facts):
    return audit.gate(
        [_member(rank, **facts) for rank in range(4)],
        min_calls=32,
        expected_slots=("collective.dp_output_projection_norm",),
        expected_member_count=4,
    )[0]


def test_gate_replays_the_september_2026_mainnet_audits():
    # Per-rank facts as retained; sources in the 2026-09-18 failure census.
    # 09-16, our own bundle: the audit workload never reached the adapter.
    assert _grade_members(
        n=0, violations=0, worst_frac=1.0, baseline_refused=78
    ) == "NO_DECISION"
    # 09-18 babcbb40: stock replay raised on every call, zero violations. Those
    # calls are baseline refusals since this change, not compare errors.
    assert _grade_members(
        n=0, violations=0, worst_frac=1.0, baseline_refused=1875
    ) == "NO_DECISION"
    # 09-17 abb914bc under byte-exact FP4 grading: 24 of 1,872 calls per rank
    # under the bar at worst 0.9642. It passes only because a0690e91 fixed the
    # grading (0 violations), not because the gate is loose.
    assert _grade_members(n=1872, violations=24, worst_frac=0.9642) == "FAIL"
    assert _grade_members(n=1872, violations=0, worst_frac=1.0) == "PASS"
    # Crowned honest run, and honest siblings, all a hair over the bar.
    for worst in (0.9862196445, 0.9873, 0.98674, 0.9886):
        assert _grade_members(n=1872, violations=0, worst_frac=worst) == "PASS"
    # One near-miss call in 1,872 fails: the owner withdrew the 09-18 allowance 10-15.
    assert _grade_members(n=1872, violations=1, worst_frac=0.9791) == "FAIL"
    # Wrong kernels: 09-03 83df0417 and the four 09-13..15 norm rows.
    assert _grade_members(n=1197, violations=914, worst_frac=0.1748) == "FAIL"
    assert _grade_members(n=1878, violations=604, worst_frac=0.0177) == "FAIL"
    # 09-18 padding defect: a single gross call fails however rare it is.
    assert _grade_members(n=18000, violations=75, worst_frac=0.749966) == "FAIL"


def test_gate_requires_exact_slot_by_rank_cartesian_coverage():
    rows = [
        {
            "slot": slot,
            "pid": 100 + rank,
            "rank": rank,
            "world_size": 2,
            "n": 32,
            "violations": 0,
            "compare_errors": 0,
            "worst_frac": 1.0,
        }
        for slot in ("norm.rmsnorm", "activation.silu_and_mul")
        for rank in range(2)
    ]
    slots = ("activation.silu_and_mul", "norm.rmsnorm")
    decision, _ = audit.gate(
        rows,
        min_calls=32,
        expected_slots=slots,
        expected_member_count=2,
    )
    assert decision == "PASS"

    decision, desc = audit.gate(
        rows[:-1],
        min_calls=32,
        expected_slots=slots,
        expected_member_count=2,
    )
    assert decision == "NO_DECISION" and "incomplete" in desc


def test_gate_requires_minimum_calls_on_every_slot_rank_receipt():
    rows = [
        {
            "slot": "norm.rmsnorm",
            "pid": 100 + rank,
            "rank": rank,
            "world_size": 2,
            "n": 31 if rank else 100,
            "violations": 0,
            "compare_errors": 0,
            "worst_frac": 1.0,
        }
        for rank in range(2)
    ]
    decision, desc = audit.gate(
        rows,
        min_calls=32,
        expected_slots=("norm.rmsnorm",),
        expected_member_count=2,
    )
    assert decision == "NO_DECISION" and "per-slot/member coverage" in desc


# ---- routed-expert pruning beyond stock's (owner ruling 2026-10-17) -------------


def test_pruning_beyond_stock_fails_under_its_own_reason_inside_the_bar():
    from cacheon.audit_gate import infrastructure_failure, reason

    # 2118ab5b's audit, had it pruned inside the bar: no violation, one finding per rank.
    rows = [_member(rank, n=220, violations=0, worst_frac=0.9, pruned=1) for rank in range(4)]
    covered = {"min_calls": 32, "expected_slots": ("collective.dp_output_projection_norm",),
               "expected_member_count": 4}
    decision, desc = audit.gate(rows, **covered)
    assert decision == "FAIL" and "pruned beyond stock's prune: 4 findings" in desc
    assert reason(decision, rows) == "routed_expert_pruning"
    # Pruning is a kernel verdict, not unavailable evidence.
    assert infrastructure_failure(rows, **covered) is None
    clean = [{**row, "pruned": 0} for row in rows]
    assert audit.gate(clean, **covered)[0] == "PASS" and reason("PASS", clean) is None
    under = [{**row, "violations": 1, "worst_frac": 0.7368} for row in clean]
    assert reason(audit.gate(under, **covered)[0], under) == "slot_audit_failed"
    assert reason("NO_DECISION", rows) == "audit_not_covered"


def test_a_pruning_finding_is_on_the_wire_only_when_there_is_one():
    from cacheon.eval.continuation_codec import ContinuationCodec
    from cacheon.eval.oci_session_protocol import AuditReceiptFacts
    from cacheon.eval.qualification_runner import _record_dict

    honest = AuditReceiptFacts("model.layers.*.mlp", 4, 0, 0, 0, 1.0, 0.75, "matched_ratio", 1, 0, 2)
    pruned = AuditReceiptFacts.from_receipt_dict({**honest.to_dict(), "pruned": 3})
    codec = ContinuationCodec((AuditReceiptFacts,))
    # Retained receipts and witnesses keep their bytes; a finding adds one key.
    for encode in (lambda row: row.to_dict(), _record_dict, lambda row: codec.encode(row)["value"]):
        assert "pruned" not in encode(honest) and encode(pruned)["pruned"] == 3
    assert codec.decode(codec.encode(pruned)) == pruned
    assert AuditReceiptFacts.from_dict(honest.to_dict()) == honest


class _MoE(nn.Module):
    """A stock MoE block whose prune reads the fork's settings at call time, like E5."""

    is_hash = is_nextn = False

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("w", torch.rand(64, 6, generator=generator) + 0.05)
        self.register_buffer("experts", torch.randn(6, 16, 16, generator=generator))

    def routed(self, x, budget, flipped=0):
        w = self.w / self.w.sum(1, keepdim=True)
        order = w.argsort(1)  # lightest first: a pick goes while the running drop fits
        through = torch.zeros_like(w).scatter_(1, order, w.gather(1, order).cumsum(1))
        kept = w * (through > budget)
        # An exact router's one-ulp tie flip: the lightest kept pick also goes on a few rows.
        lightest = kept[:flipped].masked_fill(kept[:flipped] == 0, 2.0).argmin(1)
        kept[torch.arange(flipped), lightest] = 0.0
        return torch.einsum("rk,rd,kde->re", kept, x, self.experts)

    def forward(self, x):
        e5 = sys.modules[prune.E5]
        return self.routed(x, e5.PRUNE_BUDGET + 0.05 * (e5.PRUNE_MAX_COUNT - 1))


@pytest.fixture
def pruning_node(monkeypatch, tmp_path):
    source = tmp_path / "deepseek_v4_prune.py"
    source.write_text("PRUNE_BUDGET = 0.4\nPRUNE_MAX_COUNT = 1\nPRUNE_MAX_ROWS = 256\n")
    e5 = ModuleType(prune.E5)
    e5.__file__, e5._on = str(source), [True]
    e5.PRUNE_BUDGET, e5.PRUNE_MAX_COUNT, e5.PRUNE_MAX_ROWS = 0.4, 1, 256
    monkeypatch.setitem(sys.modules, prune.E5, e5)
    monkeypatch.setattr(audit, "sampled", lambda: True)
    monkeypatch.setattr(nodes._receipts, "completed", lambda slot: None)
    monkeypatch.setattr(nodes, "_WINDOW", 1)
    for name in ("_noise", "_pooled"):
        monkeypatch.setattr(nodes, name, {})
    monkeypatch.setattr(prune, "_counts", {})
    monkeypatch.setattr(prune, "_ROWS", 8)

    def audit_four_calls(entry):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Module()])
        model.layers[0].mlp = _MoE()
        registry = KernelRegistry()
        registry.register(KernelImpl(
            slot="layers.*.mlp", bundle_id="test", entry=entry,
            eligibility=Eligibility(dtypes=frozenset({"float32"}))))
        registry.enable()
        nodes.bind(SimpleNamespace(model=model), registry)
        for _ in range(4):  # two audited calls at each step
            model.layers[0].mlp(torch.randn(64, 16))
        return audit._stats["layers.*.mlp"]

    return e5, audit_four_calls


def test_a_moe_node_pruning_beyond_stock_is_found_and_honest_ones_are_not(pruning_node):
    e5, audit_four_calls = pruning_node
    honest = {
        "stock": lambda module, x: module.forward(x),
        "exact": lambda module, x: module.forward(x) * (1 + 1e-4),
        "tie flips": lambda module, x: module.routed(x, 0.4, flipped=2),
        "less": lambda module, x: module.routed(x, 0.0),  # the bar grades it, not this check
    }
    for name, entry in honest.items():
        stats = audit_four_calls(entry)
        assert stats["n"] == 4 and "pruned" not in stats, name
        audit._stats.clear()
    for budget in (0.5, 0.7, 0.9):
        stats = audit_four_calls(lambda module, x, budget=budget: module.routed(x, budget))
        assert stats["pruned"] >= 1, budget
        audit._stats.clear()
    # The settings are put back after each heavier reference.
    assert (e5.PRUNE_BUDGET, e5.PRUNE_MAX_COUNT) == (0.4, 1)
    # Raising stock's own prune makes stock agree with the bundle; it is found anyway.
    e5.PRUNE_BUDGET = 0.9
    assert audit_four_calls(lambda module, x: module.forward(x))["pruned"] == 4
    e5.PRUNE_BUDGET = 0.4
    # A stack of several pruned blocks is never graded against a heavier reference.
    stack = nn.Module()
    stack.a, stack.b = _MoE(), _MoE()
    assert prune.single_pruned_moe(stack) is False
    e5.__file__ = str(e5.__file__) + ".missing"
    with pytest.raises(OSError):
        prune.raised_setting()


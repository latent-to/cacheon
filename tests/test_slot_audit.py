"""In-engine slot audit (cacheon/audit.py) — unit + dispatcher-wiring tests.

The audit is the fidelity gate that replaced rollout-KL as primary on
launch-nondeterministic arenas (2026-07-07): sampled dispatcher calls re-run the
captured stock baseline on pre-call clones and compare under the slot's verify
tolerances; per-rank rolling receipts feed the eval driver's gate.
"""

from types import SimpleNamespace

import pytest
import torch

from cacheon import audit, receipts
from cacheon.dispatch import make_rmsnorm_dispatcher
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
    assert not audit.enabled()
    assert not audit.sampled()


def test_rate_one_always_samples(monkeypatch):
    _arm(monkeypatch)
    assert audit.enabled()
    assert all(audit.sampled() for _ in range(20))


def test_bad_rate_is_disabled(monkeypatch):
    monkeypatch.setenv("CACHEON_SLOT_AUDIT", "not-a-number")
    assert not audit.enabled()


def test_seeded_sampling_is_reproducible(monkeypatch):
    # Collective baselines REQUIRE rank-identical decisions: same seed -> same stream.
    _arm(monkeypatch, rate="0.5", seed="123")
    a = [audit.sampled() for _ in range(50)]
    audit._state.update(rate=None, rng=None)  # simulate a second rank, same env
    b = [audit.sampled() for _ in range(50)]
    assert a == b


# ---- record / run --------------------------------------------------------------


def test_record_faithful_no_violation(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(8, 64)
    audit.record(SLOT, (x,), (x.clone(),))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 0 and s["worst_frac"] == 1.0


@pytest.mark.parametrize("slot", [SLOT, "collective.dp_output_projection_norm"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_empty_optional_outputs_preserve_nonempty_audit(slot, corrupt):
    expected = torch.ones(4, 8)
    actual = expected + 10 if corrupt else expected.clone()
    absent = torch.empty(0, dtype=torch.uint8)
    audit.record(slot, (actual, absent), (expected, absent.clone()))
    stats = audit._stats[slot]
    assert stats["n"] == 1
    assert stats["compare_errors"] == 0
    assert stats["violations"] == int(corrupt)
    assert stats["worst_frac"] == (0.0 if corrupt else 1.0)


def test_record_garbage_is_violation(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(8, 64)
    audit.record(SLOT, (x + 10.0,), (x,))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 1 and s["worst_frac"] < 0.5


def test_record_ulp_noise_passes(monkeypatch):
    # A few elements at the tolerance edge must NOT fail an otherwise-faithful kernel
    # (the outlier-channel single-ULP class measured on the v6 stockcheck).
    _arm(monkeypatch)
    x = torch.randn(100, 64)
    y = x.clone()
    y[0, 0] += 100.0  # one wild element out of 6400 -> frac 0.99984 >= 0.995
    audit.record(SLOT, (y,), (x,))
    s = audit._stats[SLOT]
    assert s["violations"] == 0 and s["worst_frac"] < 1.0


def test_record_none_expected_counts_refused(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(4, 8)
    audit.record(SLOT, (x,), (None,))
    s = audit._stats[SLOT]
    assert s["baseline_refused"] == 1 and s["n"] == 0 and s["violations"] == 0


def test_a_node_address_is_graded_by_its_adapter_and_recorded_as_units(monkeypatch):
    # A node address has no SlotSpec, so the declared-tolerance path cannot compare
    # it; the node adapter grades against stock and records the units it graded.
    _arm(monkeypatch)
    x = torch.randn(4, 8)
    audit.record("model.layers.*.mlp", (x,), (x,))
    audit.record_fraction("model.layers.*.mlp", 1.0, 0.9, "stock_twin")
    audit.record_fraction("model.layers.*.mlp", 0.5, 0.9, "stock_twin")
    stats = audit._stats["model.layers.*.mlp"]
    assert (stats["n"], stats["violations"], stats["compare_errors"]) == (2, 1, 1)
    assert (stats["worst_frac"], stats["min_ratio"], stats["mode"]) == (0.5, 0.9, "stock_twin")


def test_run_baseline_error_is_a_refusal_not_a_crash_or_compare_error(monkeypatch):
    _arm(monkeypatch)

    def boom():
        raise RuntimeError("baseline exploded")

    audit.run(SLOT, (torch.randn(4, 8),), boom)
    assert audit._stats[SLOT]["compare_errors"] == 0
    assert audit._stats[SLOT]["baseline_refused"] == 1
    assert audit._stats[SLOT]["n"] == 0


def test_run_unwraps_single_tensor_and_tuple(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(4, 8)
    audit.run(SLOT, (x,), lambda: x.clone())          # bare tensor baseline
    audit.run(SLOT, (x, x), lambda: (x.clone(), x.clone()))  # tuple baseline
    s = audit._stats[SLOT]
    assert s["n"] == 2 and s["violations"] == 0


# ---- receipts ------------------------------------------------------------------


def test_rolling_receipt_overwrites(monkeypatch, tmp_path):
    _arm(monkeypatch)
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(tmp_path))
    x = torch.randn(4, 8)
    audit.record(SLOT, (x,), (x.clone(),))
    audit.record(SLOT, (x,), (x.clone(),))
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
    # 09-17 abb914bc under byte-exact FP4 grading: 24 of 1,872 calls per rank at
    # worst 0.9642 is a near-miss rate above budget. It passes only because
    # a0690e91 fixed the grading (0 violations), not because the gate is loose.
    assert _grade_members(n=1872, violations=24, worst_frac=0.9642) == "FAIL"
    assert _grade_members(n=1872, violations=0, worst_frac=1.0) == "PASS"
    # Crowned honest run, and honest siblings, all a hair over the bar.
    for worst in (0.9862196445, 0.9873, 0.98674, 0.9886):
        assert _grade_members(n=1872, violations=0, worst_frac=worst) == "PASS"
    # One near-miss call in 1,872 is no longer terminal (owner ruling 09-18).
    assert _grade_members(n=1872, violations=1, worst_frac=0.9791) == "PASS"
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


# ---- dispatcher wiring (rmsnorm: the pure-op case) -------------------------------


def _rmsnorm_ref(x, weight, eps):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _module():
    return SimpleNamespace(variance_epsilon=1e-6,
                           weight=SimpleNamespace(data=torch.ones(64)))


def _baseline_forward(self, x, residual=None, post_residual_addition=None):
    if residual is None:
        return _rmsnorm_ref(x, self.weight.data, self.variance_epsilon)
    added = x + residual
    return _rmsnorm_ref(added, self.weight.data, self.variance_epsilon), added


def _reg(entry):
    reg = KernelRegistry()
    reg.register(KernelImpl(slot=SLOT, bundle_id="t", entry=entry,
                            eligibility=Eligibility(dtypes=frozenset({"float32"}))))
    reg.enable()
    return reg


def test_rmsnorm_dispatcher_faithful_audits_clean(monkeypatch):
    _arm(monkeypatch)

    def entry(x, weight, out, eps):
        out.copy_(_rmsnorm_ref(x, weight, eps))

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    d(_module(), torch.randn(8, 64))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 0


def test_rmsnorm_dispatcher_garbage_audited_as_violation(monkeypatch):
    _arm(monkeypatch)

    def entry(x, weight, out, eps):
        out.zero_()  # wrong function

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    d(_module(), torch.randn(8, 64))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 1


def test_rmsnorm_dispatcher_fused_path_audits_both_outputs(monkeypatch):
    _arm(monkeypatch)

    def entry(x, weight, out, eps):
        out.copy_(_rmsnorm_ref(x, weight, eps))

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    x, res = torch.randn(8, 64), torch.randn(8, 64)
    out, new_res = d(_module(), x, res)
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 0
    assert torch.equal(new_res, x + res)


def test_rmsnorm_dispatcher_no_audit_without_env():
    calls = {"n": 0}

    def entry(x, weight, out, eps):
        calls["n"] += 1
        out.copy_(_rmsnorm_ref(x, weight, eps))

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    d(_module(), torch.randn(8, 64))
    assert calls["n"] == 1 and SLOT not in audit._stats


# ---- topk_overlap slots (the generic selection-audit mode; no registered slot
# ---- currently uses it, so the tests register a synthetic one) ------------------

TOPK_SLOT = "test.selection"


def _sel(rows):
    return torch.tensor(rows, dtype=torch.int32).unsqueeze(0)  # (H=1, rows, k)


def _arm_topk(monkeypatch):
    from cacheon.slots import SLOTS, SILU_AND_MUL, Correctness
    from dataclasses import replace as _dc_replace

    _arm(monkeypatch)
    spec = _dc_replace(
        SILU_AND_MUL,
        name=TOPK_SLOT,
        correctness=Correctness("topk_overlap", top_k=8, min_overlap=0.9),
    )
    monkeypatch.setitem(SLOTS, TOPK_SLOT, spec)


def test_topk_identical_selection_no_violation(monkeypatch):
    _arm_topk(monkeypatch)
    idx = _sel([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
    audit.record(TOPK_SLOT, (idx,), (idx.clone(),))
    s = audit._stats[TOPK_SLOT]
    assert s["n"] == 1 and s["violations"] == 0 and s["worst_frac"] == 1.0
    assert s["mode"] == "topk_overlap" and s["min_ratio"] == 0.9


def test_topk_disjoint_row_is_violation(monkeypatch):
    # One fully-wrong row of four -> mean overlap 0.75 < the slot's 0.9 floor.
    _arm_topk(monkeypatch)
    base = [[i * 8 + j for j in range(8)] for i in range(4)]
    actual = [row[:] for row in base]
    actual[0] = [100 + j for j in range(8)]
    audit.record(TOPK_SLOT, (_sel(actual),), (_sel(base),))
    s = audit._stats[TOPK_SLOT]
    assert s["n"] == 1 and s["violations"] == 1
    assert abs(s["worst_frac"] - 0.75) < 1e-6


def test_topk_candidate_may_not_replace_padding_with_extra_blocks(monkeypatch):
    _arm_topk(monkeypatch)
    expected = _sel([[0, 1, 2, 3, -1, -1, -1, -1]])
    actual = _sel([[0, 1, 2, 3, 9, 10, 11, 12]])
    audit.record(TOPK_SLOT, (actual,), (expected,))
    s = audit._stats[TOPK_SLOT]
    assert s["n"] == 1 and s["violations"] == 1 and s["worst_frac"] == 0.0


def test_topk_all_invalid_expected_counts_refused(monkeypatch):
    _arm_topk(monkeypatch)
    empty = _sel([[-1] * 8])
    audit.record(TOPK_SLOT, (empty.clone(),), (empty,))
    s = audit._stats[TOPK_SLOT]
    assert s["n"] == 0 and s["violations"] == 0 and s["baseline_refused"] == 1

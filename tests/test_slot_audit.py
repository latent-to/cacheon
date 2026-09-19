"""In-engine slot audit (cacheon/audit.py) — unit + dispatcher-wiring tests.

The audit is the fidelity gate that replaced rollout-KL as primary on
launch-nondeterministic arenas (2026-07-07): sampled dispatcher calls re-run the
captured stock baseline on pre-call clones and compare under the slot's verify
tolerances; per-rank rolling receipts feed the eval driver's gate.
"""


import pytest
import torch

from cacheon import audit, receipts

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


# ---- selected-index grading, the grader `cacheon verify` still uses ---------------


def _sel(rows):
    return torch.tensor(rows, dtype=torch.int32).unsqueeze(0)  # (H=1, rows, k)


def test_selection_overlap_grades_rows_and_refuses_rewritten_padding():
    from cacheon.selection_overlap import NOTHING_SELECTED, selection_overlap

    base = [[i * 8 + j for j in range(8)] for i in range(4)]
    assert selection_overlap(_sel(base), _sel(base)) == (1.0, "")
    # One fully-wrong row of four.
    wrong = [[100 + j for j in range(8)], *base[1:]]
    assert selection_overlap(_sel(wrong), _sel(base)) == (0.75, "")
    # A candidate may not replace the reference's padding with extra blocks.
    padded = _sel([[0, 1, 2, 3, -1, -1, -1, -1]])
    score, reason = selection_overlap(_sel([[0, 1, 2, 3, 9, 10, 11, 12]]), padded)
    assert score == 0.0 and "padding" in reason
    empty = _sel([[-1] * 8])
    assert selection_overlap(empty.clone(), empty) == (0.0, NOTHING_SELECTED)

"""CPU tests for attention.msa_block_score (V2, paged) + the topk_overlap mode.

Proves the subnet can ingest a SELECTION-output win: a kernel emits block scores and
the gate is whether the top-k block SETS agree, not the score values -- so a
value-perturbing but selection-preserving kernel (an fp8 index-K) passes while a
wrong-selection kernel fails.

The V2 contract reads the paged cache the pinned runtime actually passes. A candidate
that ignores ``req_to_token`` or ``slot_ids`` reads the wrong tokens, which the
shuffled page table in ``make_inputs`` turns into a wrong selection rather than a
silent pass -- the hole the V1 gathered contract left open.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cacheon.minimax_sparse_score_slot import _reference  # noqa: E402
from cacheon.slots import Correctness, get_slot  # noqa: E402
from cacheon.verify import _compare, verify_entry  # noqa: E402

SLOT = get_slot("attention.msa_block_score")


def _scores(q, k_cache, req_to_token, slot_ids, seq_lens, sm_scale, block_size,
            topk, init_blocks, local_blocks):
    return _reference({
        "q": q, "k_cache": k_cache, "req_to_token": req_to_token,
        "slot_ids": slot_ids, "seq_lens": seq_lens, "sm_scale": sm_scale,
        "block_size": block_size, "topk": topk,
        "init_blocks": init_blocks, "local_blocks": local_blocks,
    })


def _faithful(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    out.copy_(_scores(q, k_cache, req_to_token, slot_ids, seq_lens, *scalars))


def _monotone_perturb(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    # Values shifted/scaled (like fp8 index-K) but MONOTONICALLY -> same selection.
    out.copy_(_scores(q, k_cache, req_to_token, slot_ids, seq_lens, *scalars) * 1.01
              + 0.001)


def _wrong_selection(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    # Negate -> the top-k becomes the bottom-k -> selection disagrees.
    out.copy_(-_scores(q, k_cache, req_to_token, slot_ids, seq_lens, *scalars))


def _ignores_page_table(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    # Reads the cache in slot order instead of following req_to_token: the exact
    # mistake a candidate written against the old gathered contract would make.
    identity = torch.arange(
        req_to_token.shape[1], device=q.device, dtype=req_to_token.dtype
    ).expand_as(req_to_token).contiguous()
    out.copy_(_scores(q, k_cache, identity, slot_ids, seq_lens, *scalars))


# ---- the slot is in the catalog with the paged contract ---------------------

def test_msa_slot_registered_paged():
    assert SLOT.kind == "block"
    assert SLOT.correctness.mode == "topk_overlap"
    assert SLOT.correctness.top_k == 16
    assert SLOT.graph_dynamic_inputs == (
        "q", "k_cache", "req_to_token", "slot_ids", "seq_lens",
    )


def test_scores_are_per_index_head_not_one_summed_sheet():
    # V1 summed every index q-head into a single (B,nblocks) sheet, which models
    # topk(sum(heads)); production takes top-k per head and unions.
    inputs = SLOT.make_inputs(**SLOT.shapes[1], dtype=torch.float32, device="cpu", seed=0)
    out = SLOT.out_shapes(inputs)[0]
    assert out == (inputs["q"].shape[1], inputs["q"].shape[0], 73)
    scores = _reference(inputs)
    assert not torch.allclose(scores[0], scores[1])


def test_forced_and_out_of_context_columns_follow_stock():
    inputs = SLOT.make_inputs(**SLOT.shapes[0], dtype=torch.float32, device="cpu", seed=3)
    scores = _reference(inputs)
    live = -(-int(inputs["seq_lens"][0]) // int(inputs["block_size"]))
    assert torch.all(scores[:, 0, : inputs["init_blocks"]] == 1e30)
    assert torch.all(scores[:, 0, live - inputs["local_blocks"] : live] == 1e29)
    assert torch.all(torch.isinf(scores[:, 0, live:]) & (scores[:, 0, live:] < 0))


# ---- the topk_overlap metric, unit ------------------------------------------

def test_topk_overlap_metric_unit():
    c = Correctness("topk_overlap", top_k=2, min_overlap=0.875)
    a = torch.tensor([[5.0, 1.0, 4.0, 0.0], [0.0, 9.0, 8.0, 1.0]])  # top-2: {0,2}, {1,2}
    same = torch.tensor([[50.0, 2.0, 40.0, 1.0], [1.0, 90.0, 80.0, 2.0]])
    ok, *_, score, _, metric = _compare(a, same, atol=0, rtol=0, correctness=c)
    assert ok and score == 1.0 and metric == "overlap"
    ok2, *_, score2, _, _ = _compare(-a, same, atol=0, rtol=0, correctness=c)
    assert not ok2 and score2 == 0.0


def test_topk_overlap_tolerates_masked_inf():
    # -inf columns must NOT trip the finite guard (the metric runs before it), and
    # the V2 contract writes -inf beyond every row's live block count.
    c = Correctness("topk_overlap", top_k=2, min_overlap=0.875)
    a = torch.tensor([[5.0, 1.0, 4.0, float("-inf")]])
    e = torch.tensor([[50.0, 2.0, 40.0, float("-inf")]])
    ok, *_rest = _compare(a, e, atol=0, rtol=0, correctness=c)
    assert ok


# ---- end-to-end through verify_entry ----------------------------------------

def test_msa_faithful_kernel_verifies():
    res = verify_entry(SLOT, _faithful, dtype=torch.float32, device="cpu", seed=0,
                       jitter_seed=7)
    assert res.passed, res.shape_results
    assert all(r.metric == "overlap" for r in res.shape_results)


def test_msa_monotone_perturbation_verifies():
    res = verify_entry(SLOT, _monotone_perturb, dtype=torch.float32, device="cpu", seed=0)
    assert res.passed, res.shape_results


def test_msa_wrong_selection_fails():
    res = verify_entry(SLOT, _wrong_selection, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


def test_msa_kernel_that_ignores_the_page_table_fails():
    res = verify_entry(SLOT, _ignores_page_table, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


def test_msa_gate_is_never_vacuous():
    # top-k of exactly k live blocks selects everything, so any output scores 1.0.
    for shape in SLOT.shapes:
        inputs = SLOT.make_inputs(**shape, dtype=torch.float32, device="cpu", seed=0)
        block_size = int(inputs["block_size"])
        live = int(min(-(-int(n) // block_size) for n in inputs["seq_lens"]))
        assert live > SLOT.correctness.top_k, f"vacuous shape: {shape} -> {live} blocks"

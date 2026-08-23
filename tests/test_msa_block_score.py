"""Paged, per-index-head contract for attention.msa_block_score."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cacheon.slots import (  # noqa: E402
    Correctness,
    _msa_block_score_reference,
    get_slot,
)
from cacheon.verify import _compare, verify_entry  # noqa: E402

SLOT = get_slot("attention.msa_block_score")


def _scores(q, k_cache, req_to_token, slot_ids, seq_lens, sm_scale, block_size,
            topk, init_blocks, local_blocks):
    return _msa_block_score_reference({
        "q": q, "k_cache": k_cache, "req_to_token": req_to_token,
        "slot_ids": slot_ids, "seq_lens": seq_lens, "sm_scale": sm_scale,
        "block_size": block_size, "topk": topk,
        "init_blocks": init_blocks, "local_blocks": local_blocks,
    })


def _faithful(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    out.copy_(_scores(q, k_cache, req_to_token, slot_ids, seq_lens, *scalars))


def _wrong(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    out.copy_(-_scores(q, k_cache, req_to_token, slot_ids, seq_lens, *scalars))


def _ignores_pages(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    identity = torch.arange(
        req_to_token.shape[1], dtype=req_to_token.dtype, device=q.device
    ).expand_as(req_to_token).contiguous()
    out.copy_(_scores(q, k_cache, identity, slot_ids, seq_lens, *scalars))


def _sums_heads(q, k_cache, req_to_token, slot_ids, seq_lens, out, *scalars):
    scores = _scores(q, k_cache, req_to_token, slot_ids, seq_lens, *scalars)
    out.copy_(scores.sum(0, keepdim=True).expand_as(scores))


def test_registered_contract_is_paged_and_per_head():
    assert SLOT.kind == "block" and SLOT.correctness.top_k == 16
    assert SLOT.graph_dynamic_inputs == (
        "q", "k_cache", "req_to_token", "slot_ids", "seq_lens",
    )
    inputs = SLOT.make_inputs(
        **SLOT.shapes[1], dtype=torch.float32, device="cpu", seed=0
    )
    scores = _msa_block_score_reference(inputs)
    assert scores.shape == (1, 8, 20)


def test_forced_blocks_and_ragged_tail_match_stock():
    inputs = SLOT.make_inputs(
        **SLOT.shapes[1], dtype=torch.float32, device="cpu", seed=3
    )
    scores = _msa_block_score_reference(inputs)
    row = int(inputs["seq_lens"].argmin())
    live = (int(inputs["seq_lens"][row]) + 127) // 128
    assert torch.all(scores[:, row, live - 1:live] == 1e29)
    assert torch.all(torch.isneginf(scores[:, row, live:]))


def test_topk_overlap_is_selection_not_value_equality():
    policy = Correctness("topk_overlap", top_k=2, min_overlap=0.875)
    expected = torch.tensor([[5.0, 1.0, 4.0, float("-inf")]])
    monotone = torch.tensor([[50.0, 2.0, 40.0, float("-inf")]])
    ok, *_, overlap, _, metric = _compare(
        expected, monotone, atol=0, rtol=0, correctness=policy
    )
    assert ok and overlap == 1.0 and metric == "overlap"
    assert not _compare(-expected, monotone, atol=0, rtol=0, correctness=policy)[0]


@pytest.mark.parametrize("entry,passes", [
    (_faithful, True), (_wrong, False), (_ignores_pages, False),
])
def test_semantic_controls(entry, passes):
    result = verify_entry(SLOT, entry, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed is passes, result.shape_results


def test_summed_heads_fails_when_a_profile_has_multiple_local_heads():
    shape = dict(SLOT.shapes[0], num_q_heads=4)
    result = verify_entry(
        SLOT, _sums_heads, dtype=torch.float32, device="cpu", seed=0, shapes=[shape]
    )
    assert not result.passed


def test_gate_is_never_vacuous():
    for shape in SLOT.shapes:
        inputs = SLOT.make_inputs(**shape, dtype=torch.float32, device="cpu", seed=0)
        live = min(
            (int(length) + inputs["block_size"] - 1) // inputs["block_size"]
            for length in inputs["seq_lens"]
        )
        assert live > SLOT.correctness.top_k

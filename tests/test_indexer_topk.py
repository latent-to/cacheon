"""Selection, causal windows and compact-page dispatch across prefill and decode."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cacheon.indexer_topk_contract import DYNAMIC_INPUTS, SLOT, invoke_entry, reference
from cacheon.integrations import sglang_indexer_topk as seam
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry
from cacheon.sandbox import load_entry
from cacheon.slots import get_slot
from cacheon.verify import _compare, verify_entry
from support.graph_backend import FakeGraphBackend

ENTRY = load_entry(str(Path(__file__).resolve().parents[1] /
                       "examples/miner_indexer_topk_torch/kernels/indexer_topk.py"), "indexer_topk")


def test_causal_chunk_history_permutation_padding_and_priority_tokens():
    inputs = dict(
        scores=torch.tensor([[900., 5., 2., float("inf"), 7., 900.],
                             [3., 8., 5., 4., 900., 900.], [9.] * 6]),
        lengths=torch.tensor([4, 4, 0], dtype=torch.int32),
        row_starts=torch.tensor([1, 0, 0], dtype=torch.int32),
        page_table=torch.tensor([[9, 3, 7], [4, 8, 2]], dtype=torch.int32),
        row_to_batch=torch.tensor([1, 0, 1], dtype=torch.int32),
        page_offsets=torch.tensor([1, 0, 0], dtype=torch.int32), page_size=2, top_k=5,
    )
    expected = torch.tensor([[17, 4, 9, 16, -1], [19, 6, 7, 18, -1], [-1] * 5], dtype=torch.int32)
    assert torch.equal(reference(inputs)[0], expected)
    out = torch.full_like(expected, 999)
    invoke_entry(ENTRY, inputs, [out])
    assert torch.equal(out, expected)
    inputs["scores"][:, 5] = 100000.
    invoke_entry(ENTRY, inputs, [out])
    assert torch.equal(out, expected)


def test_ordinary_verifier_and_graph_orchestration():
    slot = get_slot(SLOT)
    result = verify_entry(slot, ENTRY, dtype=torch.float32, device="cpu", seed=7,
                          graph_replays=3, _graph_backend=FakeGraphBackend())
    assert result.passed
    assert all(row.graph_replays == 3 for row in result.shape_results)

    def wrong(*args):
        args[6].zero_()

    assert not verify_entry(slot, wrong, dtype=torch.float32, device="cpu", seed=7).passed


@pytest.mark.parametrize("name", DYNAMIC_INPUTS)
def test_each_dynamic_tensor_must_change_on_replay(name):
    backend = FakeGraphBackend()
    captured = None

    def stale(*args):
        nonlocal captured
        args = list(args)
        index = DYNAMIC_INPUTS.index(name)
        if backend.phase == "capture":
            captured = args[index].clone()
        if backend.phase == "replay":
            args[index] = captured
        ENTRY(*args)

    assert not verify_entry(get_slot(SLOT), stale, dtype=torch.float32, device="cpu",
                            seed=7, graph_replays=3, _graph_backend=backend).passed


def test_integer_selection_comparator_preserves_set_semantics_without_quadratic_storage():
    expected = torch.tensor([[5, 9, 3, -1], [-1, -1, -1, -1]])
    actual = torch.tensor([[3, 5, 9, -1], [-1, -1, -1, -1]])
    compare = lambda a: _compare(a, expected, atol=0, rtol=0, correctness=get_slot(SLOT).correctness)[0]
    assert compare(actual)
    actual[0, 1] = 3
    assert not compare(actual)
    actual[0] = torch.tensor([3, 5, 9, 99])
    assert not compare(actual)


@pytest.mark.parametrize("prefill", [False, True])
def test_pinned_call_routes_tensor_abi_and_failure_never_retries_stock(monkeypatch, prefill):
    def mapping(attn_metadata, row_starts, cu_seqlens_q_topk, batch_idx_list, device, num_rows):
        if row_starts is None:
            return None, None
        batches = torch.repeat_interleave(torch.arange(cu_seqlens_q_topk.numel() - 1),
                                         cu_seqlens_q_topk.diff(), output_size=num_rows).int()
        return batches, row_starts - attn_metadata.cu_seqlens_k[:-1][batches]

    module = SimpleNamespace(TopkTransformMethod=SimpleNamespace(PAGED="paged"),
                             envs=SimpleNamespace(SGLANG_DSA_FUSE_TOPK=SimpleNamespace(get=lambda: True)),
                             _build_flashinfer_paged_args=mapping)
    table = torch.tensor([[8, 3, 4], [6, 2, 9]], dtype=torch.int32)
    metadata = SimpleNamespace(real_page_table=table, page_size=2,
                               cu_seqlens_k=torch.tensor([0, 4, 8], dtype=torch.int32))
    calls = []

    def stock(self, logits, lengths, topk, topk_transform_method, attn_metadata,
              cu_seqlens_q_topk=None, topk_indices_offset=None, row_starts=None,
              batch_idx_list=None, force_unfused_topk=False):
        calls.append("stock")
        return torch.full((logits.shape[0], topk), -9, dtype=torch.int32)

    scores = torch.arange(24, dtype=torch.float32).reshape(3, 8) if prefill else torch.arange(16, dtype=torch.float32).reshape(2, 8)
    lengths = torch.tensor([2, 3, 2] if prefill else [4, 3], dtype=torch.int32)
    kw = dict(row_starts=torch.tensor([0, 0, 4], dtype=torch.int32),
              cu_seqlens_q_topk=torch.tensor([0, 2, 3], dtype=torch.int32)) if prefill else {}
    registry = KernelRegistry()
    registry.register(KernelImpl(slot=SLOT, bundle_id="test-topk", entry=ENTRY,
                                 eligibility=Eligibility(dtypes=frozenset({"float32"}))))
    registry.enable()
    monkeypatch.setenv("CACHEON_INDEXER_TOPK_SEAM", "1")
    monkeypatch.setattr(seam, "_dynamo_compiling", lambda: False)
    monkeypatch.setattr(seam, "_flashinfer_tuning", lambda: False)
    monkeypatch.setattr(seam, "_runtime_parallel_sizes", lambda: (4, 4))
    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: False))
    monkeypatch.setattr(seam, "_receipts", SimpleNamespace(is_invoking=lambda: False, invoke=lambda s, e, *a: e(*a), completed=calls.append))
    dispatch = seam._make_dispatch(stock, registry, module)
    actual = dispatch(None, scores, lengths, 3, "paged", metadata, **kw)
    expected = [[17, 16, -1], [6, 17, 16], [13, 12, -1]] if prefill else [[7, 6, 17], [4, 13, 12]]
    assert actual.tolist() == expected
    assert calls == [SLOT]

    def broken(*args):
        raise RuntimeError("candidate exploded")

    registry = KernelRegistry()
    registry.register(KernelImpl(slot=SLOT, bundle_id="broken", entry=broken))
    registry.enable()
    with pytest.raises(RuntimeError, match="candidate exploded"):
        seam._make_dispatch(stock, registry, module)(None, scores, lengths, 3, "paged", metadata, **kw)
    assert calls == [SLOT]

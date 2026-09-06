"""Independent weighted-ReLU math and freshness of every score input."""

from pathlib import Path

import pytest
import torch

from cacheon.indexer_scores_contract import DYNAMIC_INPUTS, invoke_entry, reference, slot_spec
from cacheon.sandbox import load_entry
from cacheon.verify import verify_entry
from support.graph_backend import FakeGraphBackend

ENTRY = load_entry(str(Path(__file__).resolve().parents[1] /
                       "examples/miner_indexer_scores_torch/kernels/indexer_scores.py"), "indexer_scores")


def test_declared_relu_per_head_scaling_signed_weights_and_window():
    inputs = dict(q=torch.tensor([[[1., -1.], [-1., 1.]], [[2., 0.], [0., 2.]]]).to(torch.float8_e4m3fn),
                  key_pages=torch.tensor([[[1., 0.], [0., 1.]], [[2., 1.], [1., 3.]]]).to(torch.float8_e4m3fn),
                  key_scales=torch.tensor([[2., 3.], [4., 0.5]]),
                  weights=torch.tensor([[2., -3.], [-1., 2.]]),
                  starts=torch.tensor([1, 0], dtype=torch.int32), ends=torch.tensor([3, 0], dtype=torch.int32),
                  page_table=torch.tensor([[1, 0]], dtype=torch.int32),
                  row_to_batch=torch.zeros(2, dtype=torch.int32), kv_len=4)
    expected = torch.tensor([[0., -3., 4., 0.], [0., 0., 0., 0.]])
    assert torch.equal(reference(inputs)[0], expected)
    out = torch.full_like(expected, torch.nan)
    invoke_entry(ENTRY, inputs, [out])
    assert torch.equal(out, expected)


def test_verifier_two_geometries_and_graph_orchestration():
    result = verify_entry(slot_spec(), ENTRY, dtype=torch.float8_e4m3fn, device="cpu", seed=7,
                          graph_replays=3, _graph_backend=FakeGraphBackend())
    assert result.passed, result
    assert len(result.shape_results) == 2
    assert all(row.graph_replays == 3 for row in result.shape_results)


@pytest.mark.parametrize("name", DYNAMIC_INPUTS)
def test_replay_rejects_frozen_request_tensor(name):
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

    assert not verify_entry(slot_spec(), stale, dtype=torch.float8_e4m3fn, device="cpu",
                            seed=7, graph_replays=3, _graph_backend=backend).passed

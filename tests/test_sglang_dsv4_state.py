"""The DeepSeek-V4 paged cache rows the node audit restores and grades."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cacheon.integrations import sglang_nodes as nodes
from cacheon.integrations.sglang_dsa_state import state_values
from cacheon.integrations.sglang_dsv4_state import dsv4_state_rows
from tests.test_node_adapter import _registry, _served_model, audited  # noqa: F401


def _v4_page(tokens, nope, exponent):
    """One V4 page row: ``tokens`` rows of 448 E4M3 nope bytes and 64 BF16 rope values, then the scales."""
    data = torch.cat((torch.full((tokens, 448), nope).to(torch.float8_e4m3fn).view(torch.uint8),
                      torch.arange(64).to(torch.bfloat16).expand(tokens, 64).contiguous().view(torch.uint8)), -1)
    return torch.cat((data.reshape(-1), torch.full((tokens * 8,), exponent, dtype=torch.uint8)))


def _pool(ring_size=2):
    """A DeepSeek-V4.1 pool as 0.5.21 lays it out: whole pages of four tokens per row.

    Layer 0 keeps only its window; layer 1 is a ratio-2 kv-source layer with
    compressed pages, two FP4 index pages per full page and a pending-pair ring.
    """
    swa = SimpleNamespace(kv_buffer=[torch.zeros(3, 4 * 584, dtype=torch.uint8) for _ in range(2)],
                          page_size=4, kv_layout="v4")
    kv2 = SimpleNamespace(kv_buffer=[torch.zeros(3, 2 * 584, dtype=torch.uint8)], page_size=2, kv_layout="v4")
    indexer = SimpleNamespace(index_k_with_scale_buffer=[torch.zeros(6, 68, dtype=torch.uint8)], page_size=1,
                              index_head_dim=128, use_fp4_indexer=True)
    # Four request rings plus the runtime's spare ring, sentinel and ratio-2 padding.
    state_rows = ((4 * ring_size + ring_size + 2) // 2) * 2
    state = SimpleNamespace(request_scoped=True, ring_size=ring_size,
                            kv_score_buffer=SimpleNamespace(kv_score=torch.zeros(state_rows, 3)))
    swa.kv_buffer[1][2] = _v4_page(4, 2.0, 128)
    kv2.kv_buffer[0][1] = _v4_page(2, 3.0, 127)
    indexer.index_k_with_scale_buffer[0][2:4, :64] = 0x21
    indexer.index_k_with_scale_buffer[0][2:4, 64:] = 128
    return SimpleNamespace(
        layer_mapping=[(0, 0, None), (2, 0, kv2)], sources_by_ratio={2: [1]}, index_pools={2: indexer},
        compress_state_pools=[None, state], swa_kv_pool=swa, unified_kv_pool=None, _stage_start=0, _stage_end=2,
        page_size=4, num_req_slots=4, translate_loc_from_full_to_swa=lambda slots: slots + 4,
    )


def _batch():
    return SimpleNamespace(out_cache_loc=torch.tensor([4, 5, 6, 7]), req_pool_indices=torch.tensor([1]))


def _uniform_pool():
    """TRT-LLM's V4-tagged pools contain 512 unscaled FP8 values per token."""
    pool = _pool()
    layout = Enum("KVLayout", {"V4": "v4"}).V4
    for kv in (pool.swa_kv_pool, pool.layer_mapping[1][2]):
        kv.kv_layout = layout
        kv.kv_buffer = [torch.zeros(3, kv.page_size * 512, dtype=torch.float8_e4m3fn)
                        for _ in kv.kv_buffer]
    return pool


def test_uniform_fp8_reads_every_token_including_rotary_values_without_scales():
    pool = _uniform_pool()
    window, compressed, _, _ = dsv4_state_rows(pool, _batch(), 1)
    for row, page, target in ((window, 4, 2), (compressed, 2, 1)):
        expected = torch.tensor([0.0, -0.5, 2.0, -4.0]).repeat(page * 128).reshape(1, page, 512)
        expected[:, :, 448:] = torch.arange(page).reshape(1, page, 1) + 8
        row[0][target].copy_(expected.reshape(-1).to(torch.float8_e4m3fn).view(torch.uint8))
        raw = row[0].index_select(0, row[2])
        assert row[2].tolist() == [target]
        assert torch.equal(state_values(raw, row[3]), expected)
        with pytest.raises(RuntimeError, match="uniform-FP8"):
            state_values(raw[:, :-1], row[3])
        with pytest.raises(RuntimeError, match="uniform-FP8"):
            state_values(raw.view(torch.float8_e4m3fn), row[3])


@pytest.mark.parametrize("ring_size", [2, 8])
def test_pages_are_addressed_by_page_and_graded_as_the_numbers_they_hold(ring_size):
    pool = _pool(ring_size)
    window, compressed, index, ring = dsv4_state_rows(pool, _batch(), 1)
    assert window[2].tolist() == [2] and compressed[2].tolist() == [1] and index[2].tolist() == [2, 3]
    values = state_values(window[0].index_select(0, window[2]), window[3])
    assert torch.equal(values[..., :448], torch.full((1, 4, 448), 4.0))
    assert torch.equal(values[..., 448:], torch.arange(64).float().expand(1, 4, 64))
    assert torch.equal(state_values(compressed[0].index_select(0, compressed[2]), compressed[3])[..., :448],
                       torch.full((1, 2, 448), 3.0))
    keys = state_values(index[0].index_select(0, index[2]), index[3])
    assert keys.shape == (2, 1, 128) and keys[0, 0, :4].tolist() == [1.0, 2.0, 1.0, 2.0]
    assert ring[0].shape == (4, ring_size * 3) and ring[2].tolist() == [1] and ring[3] == torch.float32
    assert len(dsv4_state_rows(pool, _batch(), 0)) == 1  # a window-only layer
    assert len(dsv4_state_rows(pool, _batch())) == 5  # the whole stack


@pytest.mark.parametrize("start", [0, 40])
def test_whole_node_reads_only_the_layers_owned_by_its_pool(start):
    pool = _pool()
    # Target and draft/PP pools share model-wide arrays; unowned entries are None.
    pool.layer_mapping = [None] * start + pool.layer_mapping + [None] * 3
    pool.compress_state_pools = [None] * start + pool.compress_state_pools + [None] * 3
    pool.sources_by_ratio = {2: [start + 1]}
    pool._stage_start, pool._stage_end = start, start + 2
    rows = dsv4_state_rows(pool, _batch())
    assert len(rows) == 5
    assert rows[0][0] is pool.swa_kv_pool.kv_buffer[0]
    assert rows[1][0] is pool.swa_kv_pool.kv_buffer[1]
    assert len(dsv4_state_rows(pool, _batch(), start + 1)) == 4


@pytest.mark.parametrize("fault, cause", [
    ("unified", "paged sliding-window"),
    ("fp8 index", "FP4"),
    ("shared ring", "request-scoped"),
    ("layout", "unrecognized DeepSeek-V4 cache layout"),
    ("fp8 layout", "unrecognized DeepSeek-V4 cache layout"),
    ("stage", "no DeepSeek-V4 cache"),
])
def test_unrecognized_layouts_raise_rather_than_pass_unchecked(fault, cause):
    pool, layer = _pool(), 1
    if fault == "unified":
        pool.unified_kv_pool = SimpleNamespace()
    elif fault == "fp8 index":
        pool.index_pools[2].use_fp4_indexer = False
    elif fault == "shared ring":
        pool.compress_state_pools[1].request_scoped = False
    elif fault == "layout":
        pool.layer_mapping[1][2].kv_layout = "v5"
    elif fault == "fp8 layout":
        pool = _uniform_pool()
        pool.swa_kv_pool.kv_layout = "v41"
    else:
        layer = 2
    with pytest.raises(RuntimeError, match=cause):
        dsv4_state_rows(pool, _batch(), layer)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("corrupt", [None, "window", "compressed"])
def test_pages_are_restored_before_the_candidate_and_wrong_pages_fail(audited, corrupt, uniform):  # noqa: F811
    runner, batch = _served_model()
    runner.token_to_kv_pool = pool = _uniform_pool() if uniform else _pool()
    batch.out_cache_loc, batch.req_pool_indices = torch.tensor([4, 5, 6, 7]), torch.tensor([1])
    index = pool.index_pools[2].index_k_with_scale_buffer[0]
    window = pool.swa_kv_pool.kv_buffer[1].view(torch.uint8)
    compressed = pool.layer_mapping[1][2].kv_buffer[0].view(torch.uint8)

    class SourceLayer(nn.Module):
        layer_id = 1

        def forward(self, x, batch):
            index[2:4, :64] += 1
            window[2].fill_(0x40)
            compressed[1].fill_(0x40)
            pool.compress_state_pools[1].kv_score_buffer.kv_score[2:4] += 1.0
            return x * 2

    seen = []
    before = [v.clone() for v in (index, window, compressed)]

    def candidate(module, x, batch):
        seen.extend(v.clone() for v in (index, window, compressed))
        result = module.forward(x, batch)
        if corrupt == "window":
            window[2].zero_()
        elif corrupt == "compressed":
            compressed[1].zero_()
        return result

    runner.model.source = SourceLayer()
    nodes.bind(runner, _registry("source", candidate))
    assert torch.equal(runner.model.source(torch.ones(2, 4), batch), torch.full((2, 4), 2.0))
    assert all(torch.equal(a, b) for a, b in zip(seen, before, strict=True))
    assert audited["source"]["violations"] == int(corrupt is not None)


@pytest.mark.parametrize("corrupt", [False, True])
def test_engram_history_is_restored_and_graded(audited, corrupt):  # noqa: F811
    runner, batch = _served_model()
    batch.req_pool_indices = torch.tensor([1])

    class Stack(nn.Module):
        """An Engram lookup whose answer depends on the prior request history."""

        def __init__(self):
            super().__init__()
            self.engram_hasher = SimpleNamespace(history=torch.zeros(3, 3, dtype=torch.int32))

        def forward(self, x, batch):
            history = self.engram_hasher.history
            history[batch.req_pool_indices] += 1
            return x + history[batch.req_pool_indices].float()

    seen = []

    def candidate(module, x, batch):
        seen.append(module.engram_hasher.history.clone())
        result = module.forward(x, batch)
        if corrupt:
            module.engram_hasher.history[batch.req_pool_indices] += 10000
        return result

    runner.model.model = Stack()
    nodes.bind(runner, _registry("model", candidate))
    assert torch.equal(runner.model.model(torch.zeros(1, 3), batch), torch.ones(1, 3))
    assert torch.equal(seen[0], torch.zeros(3, 3, dtype=torch.int32))
    assert runner.model.model.engram_hasher.history.tolist() == [
        [0, 0, 0], [10001 if corrupt else 1] * 3, [0, 0, 0],
    ]
    assert audited["model"]["violations"] == int(corrupt)

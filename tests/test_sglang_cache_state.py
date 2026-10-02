"""Prefix-state audit detects corruption that leaves full-attention KV untouched."""

from types import SimpleNamespace as NS

import pytest
import torch

from cacheon import audit
from cacheon.integrations.sglang_cache import _Guard
from tests.test_sglang_cache_seam import Key, PAGE, Req, _ctx


@pytest.fixture(params=[False, True], ids=["hybrid", "recurrent-only"])
def state(monkeypatch, request):
    monkeypatch.setitem(audit._state, "rate", 1.0)
    ctx = _ctx(eagle=True, is_hybrid_ssm=True)
    params = ctx.params
    if request.param:
        params.token_to_kv_pool_allocator.pool = NS(k_buffer=[], v_buffer=[], layer_num=0)
    params.sliding_window_size, params.enable_mamba_extra_buffer = None, False
    temporal, conv = torch.zeros(16, 4), torch.zeros(16, 2)
    pool = NS(replayssm_write_pos=torch.zeros(16, dtype=torch.int64))
    pool._iter_transfer_state_entries = lambda: iter((
        ("temporal", temporal, 0, 2), ("conv", conv, 0, 2),
    ))
    params.req_to_token_pool.mamba_pool = pool
    # Virtual checkpoint IDs need not equal the pool's physical rows.
    params.req_to_token_pool.translate_mamba_indices = lambda indices: indices + 4
    guard = _Guard(ctx)
    assert guard.bigram == 0
    cache = NS(token_to_kv_pool_allocator=guard.allocator, req_to_token_pool=guard.requests)

    def record(tokens, slot=1, *, lag=0, extra=False, finished=True):
        req = Req(f"source-{slot}", tokens, slot)
        req.kv.mamba_pool_idx = torch.tensor(slot)
        req.kv.swa_evicted_seqlen = 0
        req.kv.mamba_cow_src_index = None
        params.enable_mamba_extra_buffer = extra
        length = len(tokens) - lag
        # Independent model: recurrent state describes the committed token prefix.
        physical = slot + 4
        if extra:
            req.kv.mamba_last_track_seqlen = length
            req.kv.mamba_ping_pong_track_buffer = torch.tensor([slot, slot + 1])
            guard.requests.get_mamba_ping_pong_keep_idx = lambda request: 1
            physical += 1
        else:
            pool.replayssm_write_pos[slot] = lag
        temporal[physical].fill_(sum(tokens[:length]))
        conv[physical] = torch.tensor([tokens[length - 1], length])
        row = torch.arange(PAGE, PAGE + len(tokens))
        guard.requests.req_to_token[slot, :len(row)] = row.int()
        guard.handoff(cache, req, tokens, finished=finished)
        return req, row

    def match(tokens, row, source=1, *, host=False):
        req = Req("query", tokens + [99], 0)
        req.kv.mamba_pool_idx = torch.tensor(4)
        req.kv.mamba_cow_src_index = torch.tensor(source)
        req.kv.swa_evicted_seqlen = 0
        req.prefix_indices = row
        result = NS(device_indices=row[:0] if host else row, host_hit_length=len(row) if host else 0,
                    swa_host_hit_length=0, mamba_host_hit_length=len(row) if host else 0)
        params = NS(key=Key(tokens + [99]), req=req, cow_mamba=True)
        guard.matched(cache, params, result, NS)
        return req

    return NS(guard=guard, cache=cache, temporal=temporal, conv=conv, pool=pool,
              record=record, match=match, params=params)


@pytest.mark.parametrize("lag,extra", [(0, False), (4, False), (4, True)])
def test_recurrent_checkpoint_uses_its_committed_prefix_and_physical_slot(state, lag, extra):
    tokens = list(range(12))
    _, row = state.record(tokens, lag=lag, extra=extra)
    state.match(tokens[:12 - lag], row[:12 - lag], source=2 if extra else 1)
    state.guard.poll(block=True)


def test_wrong_recurrent_checkpoint_fails_even_when_full_kv_is_valid(state):
    tokens = list(range(8))
    _, row = state.record(tokens)
    state.record(list(range(20, 28)), slot=2)
    state.match(tokens, row, source=2)
    with pytest.raises(RuntimeError, match="recurrent state"):
        state.guard.poll(block=True)


def test_recurrent_host_load_checks_destination_instead_of_a_stale_cow_source(state):
    tokens = list(range(8))
    _, row = state.record(tokens)
    state.match(tokens, row, source=2, host=True)
    state.temporal[8] = state.temporal[5]
    state.conv[8] = state.conv[5]
    state.guard.hybrid.ready(state.cache)
    assert not state.guard.hybrid.pending


def test_unchecked_host_state_cannot_be_recorded_as_fresh_model_work(state):
    tokens = list(range(8))
    _, row = state.record(tokens)
    req = state.match(tokens, row, host=True)
    with pytest.raises(RuntimeError, match="before its state audit completed"):
        state.guard.handoff(state.cache, req, tokens)


def test_corrupt_restored_state_cannot_hide_behind_a_valid_device_source(state):
    tokens = list(range(8))
    _, row = state.record(tokens)
    state.match(tokens, row, source=1, host=True)
    with pytest.raises(RuntimeError, match="recurrent state"):
        state.guard.hybrid.ready(state.cache)


def test_cache_must_preserve_unfinished_recurrent_state(state):
    req, row = state.record(list(range(8)), finished=False)
    state.temporal[5].zero_()
    req.prefix_indices, req.kv.cache_protected_len = row, len(row)
    state.guard.settle(req, row)
    with pytest.raises(RuntimeError, match="recurrent state"):
        state.guard.poll(block=True)


@pytest.mark.parametrize("state_buffer", ["kv", "index"])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("evicted,hit", [(True, 16), (False, 16), (False, 8)])
def test_swa_checks_live_windows_including_earlier_branches(monkeypatch, corrupt, evicted, hit, state_buffer):
    monkeypatch.setitem(audit._state, "rate", 1.0)
    ctx = _ctx(is_hybrid_swa=True)
    p = ctx.params
    p.sliding_window_size, p.enable_mamba_extra_buffer = 8, False
    full = p.token_to_kv_pool_allocator.pool
    swa = NS(k_buffer=[torch.zeros(64, 4)], v_buffer=[torch.zeros(64, 2)], page_size=PAGE,
             index_k_with_scale_buffer=[torch.zeros(64 // PAGE, PAGE, dtype=torch.uint8)])
    mapping = torch.full((256,), -1, dtype=torch.int64)
    begin = 8 if evicted else 0
    mapping[PAGE + begin:PAGE + 16] = torch.arange(begin, 16)
    p.token_to_kv_pool_allocator.pool = NS(
        full_kv_pool=full, swa_kv_pool=swa,
        translate_loc_from_full_to_swa=lambda slots: mapping[slots],
    )
    guard = _Guard(ctx)
    cache = NS(token_to_kv_pool_allocator=guard.allocator, req_to_token_pool=guard.requests)
    tokens, row = list(range(16)), torch.arange(PAGE, PAGE + 16)
    req = Req("source", tokens, 0)
    req.kv.swa_evicted_seqlen = begin
    guard.requests.req_to_token[0, :16] = row.int()
    for i in range(begin, 16):
        swa.k_buffer[0][i] = sum(tokens[:i + 1])
        swa.v_buffer[0][i] = tokens[i]
        swa.index_k_with_scale_buffer[0][i // PAGE, i % PAGE] = tokens[i]
    guard.handoff(cache, req, tokens, finished=True)
    if corrupt:
        if state_buffer == "kv":
            swa.k_buffer[0][hit - 3].add_(7)
        else:
            swa.index_k_with_scale_buffer[0][(hit - 3) // PAGE, (hit - 3) % PAGE] += 7
    tokens, row = tokens[:hit], row[:hit]
    query = Req("query", tokens + [99], 1)
    query.kv.swa_evicted_seqlen = 0
    params = NS(key=Key(tokens + [99]), req=query, cow_mamba=False)
    result = NS(device_indices=row, host_hit_length=0, swa_host_hit_length=0, mamba_host_hit_length=0)
    guard.matched(cache, params, result, NS)
    if corrupt:
        with pytest.raises(RuntimeError, match="sliding-window"):
            guard.poll(block=True)
    else:
        guard.poll(block=True)


@pytest.mark.parametrize("corrupt", [None, "quantized", "scale", "conv"])
def test_encoded_checkpoint_uses_declared_rounding_and_separate_slot_ids(monkeypatch, corrupt):
    monkeypatch.setitem(audit._state, "rate", 1.0)
    ctx = _ctx(eagle=True, is_hybrid_ssm=True)
    params = ctx.params
    params.sliding_window_size, params.enable_mamba_extra_buffer = None, False
    temporal, conv = torch.zeros(16, 1, 2, 2), torch.zeros(16, 2)
    active = NS(replayssm_write_pos=None)
    active._iter_transfer_state_entries = lambda: iter((
        ("temporal", temporal, 0, 2), ("conv", conv, 0, 2),
    ))
    encoded = NS(temporal=NS(qdata=torch.zeros(1, 8, 1, 2, 2, dtype=torch.int8),
                             scale=torch.zeros(1, 8, 1, 1, 2)),
                 conv=[torch.zeros(1, 8, 2)])
    params.req_to_token_pool.mamba_pool = active
    params.req_to_token_pool.mamba_ckpt_pool = encoded
    params.req_to_token_pool.translate_mamba_indices = lambda ids: ids + 4
    guard = _Guard(ctx)
    cache = NS(token_to_kv_pool_allocator=guard.allocator, req_to_token_pool=guard.requests)
    tokens = list(range(8))
    source = Req("source", tokens, 0)
    source.kv.mamba_pool_idx = torch.tensor(1)
    temporal[5] = torch.tensor([[[1., -1.], [.005, -.005]]])
    conv[5] = torch.tensor([7., 8.])
    row = torch.arange(PAGE, PAGE + len(tokens))
    guard.requests.req_to_token[0, :len(row)] = row.int()
    guard.handoff(cache, source, tokens, finished=True)
    # Known codec result: the small entries round to one quantization step.
    encoded.temporal.qdata[0, 3] = torch.tensor([[[127, -127], [1, -1]]], dtype=torch.int8)
    encoded.temporal.scale[0, 3] = 1 / 127
    encoded.conv[0][0, 3] = conv[5]
    if corrupt == "quantized":
        encoded.temporal.qdata[0, 3].zero_()
    elif corrupt == "scale":
        encoded.temporal.scale[0, 3].mul_(2)
    elif corrupt == "conv":
        encoded.conv[0][0, 3].zero_()
    query = Req("query", tokens + [99], 1)
    query.kv.mamba_cow_src_index = torch.tensor(3)
    query.kv.mamba_pool_idx = torch.tensor(2)
    match = NS(device_indices=row, host_hit_length=0, swa_host_hit_length=0, mamba_host_hit_length=0)
    guard.matched(cache, NS(key=Key(tokens + [99]), req=query, cow_mamba=True), match, NS)
    if corrupt:
        with pytest.raises(RuntimeError, match="recurrent state"):
            guard.poll(block=True)
    else:
        guard.poll(block=True)

"""Prefix-state audit detects corruption that leaves full-attention KV untouched."""

from types import SimpleNamespace as NS

import pytest
import torch

from cacheon import audit
from cacheon.integrations.sglang_cache import _Guard
from tests.test_sglang_cache_seam import Key, PAGE, Req, _ctx


@pytest.mark.parametrize("state_buffer", ["kv", "index"])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("evicted,hit", [(True, 16), (False, 16), (False, 8)])
def test_swa_checks_live_windows_including_earlier_branches(monkeypatch, corrupt, evicted, hit, state_buffer):
    monkeypatch.setitem(audit._state, "rate", 1.0)
    ctx = _ctx(is_hybrid_swa=True)
    p = ctx.params
    p.sliding_window_size = 8
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
    params = NS(key=Key(tokens + [99]), req=query)
    result = NS(device_indices=row, host_hit_length=0, swa_host_hit_length=0)
    guard.matched(cache, params, result, NS)
    if corrupt:
        with pytest.raises(RuntimeError, match="sliding-window"):
            guard.poll(block=True)
    else:
        guard.poll(block=True)


def test_recurrent_state_is_refused_rather_than_served_unchecked(monkeypatch):
    """No commissioned arena caches recurrent checkpoints; the pool is refused before any handoff."""
    monkeypatch.setitem(audit._state, "rate", 1.0)
    ctx = _ctx(eagle=True, is_hybrid_ssm=True)
    ctx.params.req_to_token_pool.mamba_pool = NS()
    with pytest.raises(RuntimeError, match="state validation is unavailable"):
        _Guard(ctx)

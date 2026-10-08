"""Prefix-state audit detects corruption that leaves full-attention KV untouched."""

import sys
from types import SimpleNamespace as NS

import pytest
import torch

from cacheon import audit
from cacheon.integrations.sglang_cache import _Guard
from cacheon.integrations.sglang_cache_state import _COMPONENTS
from tests.test_sglang_cache_seam import Key, PAGE, Req, _ctx


@pytest.fixture(autouse=True)
def components(monkeypatch):
    """The runtime module naming the sliding-window component, as 0.5.21 loads it."""
    monkeypatch.setitem(audit._state, "rate", 1.0)
    monkeypatch.setitem(sys.modules, _COMPONENTS, NS(ComponentType=NS(SWA="swa")))


def _window_engine(swa, begin=0):
    """A hybrid engine whose window pool maps the first full page's slots from ``begin``."""
    ctx = _ctx(is_hybrid_swa=True)
    p = ctx.params
    p.sliding_window_size = 8
    full = p.token_to_kv_pool_allocator.pool
    mapping = torch.full((256,), -1, dtype=torch.int64)
    mapping[PAGE + begin:PAGE + 16] = torch.arange(begin, 16)
    p.token_to_kv_pool_allocator.pool = NS(
        full_kv_pool=full, swa_kv_pool=swa,
        translate_loc_from_full_to_swa=lambda slots: mapping[slots],
    )
    return ctx


def _request(rid, tokens, row, evicted=0):
    req = Req(rid, tokens, row)
    req.kv.get_evicted_seqlen = lambda component: evicted
    return req


def _cache(guard):
    return NS(token_to_kv_pool_allocator=guard.allocator, req_to_token_pool=guard.requests)


def _query(guard, cache, tokens, row):
    params = NS(key=Key(tokens + [99]), req=_request("query", tokens + [99], 1))
    guard.matched(cache, params, NS(device_indices=row, host_hit_length=0, swa_host_hit_length=0), NS)


@pytest.mark.parametrize("state_buffer", ["kv", "index"])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("evicted,hit", [(True, 16), (False, 16), (False, 8)])
def test_swa_checks_live_windows_including_earlier_branches(corrupt, evicted, hit, state_buffer):
    swa = NS(size=64, k_buffer=[torch.zeros(64, 4)], v_buffer=[torch.zeros(64, 2)], page_size=PAGE,
             index_k_with_scale_buffer=[torch.zeros(64 // PAGE, PAGE, dtype=torch.uint8)])
    begin = 8 if evicted else 0
    guard = _Guard(_window_engine(swa, begin))
    assert guard.hybrid.paged == {("swa", "index_k_with_scale_buffer", 0)}
    cache = _cache(guard)
    tokens, row = list(range(16)), torch.arange(PAGE, PAGE + 16)
    req = _request("source", tokens, 0, begin)
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
    _query(guard, cache, tokens[:hit], row[:hit])
    if corrupt:
        with pytest.raises(RuntimeError, match="sliding-window"):
            guard.poll(block=True)
    else:
        guard.poll(block=True)


@pytest.mark.parametrize("corrupt", [False, True])
def test_a_window_pool_that_stores_whole_pages_is_checked_by_page(corrupt):
    swa = NS(size=64, page_size=PAGE, kv_buffer=[torch.zeros(64 // PAGE + 1, PAGE * 2)])
    guard = _Guard(_window_engine(swa))
    assert guard.hybrid.paged == {("swa", "kv_buffer", 0)}
    cache = _cache(guard)
    tokens, row = list(range(16)), torch.arange(PAGE, PAGE + 16)
    guard.requests.req_to_token[0, :16] = row.int()
    for page in range(16 // PAGE):
        swa.kv_buffer[0][page] = float(sum(tokens[:(page + 1) * PAGE]))
    guard.handoff(cache, _request("source", tokens, 0), tokens, finished=True)
    if corrupt:
        swa.kv_buffer[0][3, 1] += 7  # the last page of the live window, as the pool stores it
    _query(guard, cache, tokens, row)
    if corrupt:
        with pytest.raises(RuntimeError, match="sliding-window"):
            guard.poll(block=True)
    else:
        guard.poll(block=True)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("ring_size", [2, 8])
def test_request_held_compressor_state_must_survive_a_cache_handoff(corrupt, ring_size):
    swa = NS(size=64, page_size=PAGE, kv_buffer=[torch.zeros(64 // PAGE + 1, PAGE * 2)])
    ctx = _window_engine(swa)
    scores = torch.zeros(((8 * ring_size + ring_size + 2) // 2) * 2, 3)
    ctx.params.token_to_kv_pool_allocator.pool.num_req_slots = 8
    ctx.params.token_to_kv_pool_allocator.pool.compress_state_pools = [
        NS(request_scoped=True, ring_size=ring_size, kv_score_buffer=NS(kv_score=scores)), None,
    ]
    guard = _Guard(ctx)
    assert guard.hybrid.live and ("request", "compress_state_pools", 0) in guard.hybrid.buffers
    cache = _cache(guard)
    tokens, row = list(range(16)), torch.arange(PAGE, PAGE + 16)
    req = _request("source", tokens, 0)
    guard.requests.req_to_token[0, :16] = row.int()
    scores[:2] = 5.0
    handed = guard.handoff(cache, req, tokens)
    if corrupt:
        scores[1] += 1.0
    req.prefix_indices, req.kv.cache_protected_len = row.clone(), 16
    guard.settle(req, handed)
    if corrupt:
        with pytest.raises(RuntimeError, match="request-held state"):
            guard.poll(block=True)
    else:
        guard.poll(block=True)


def test_recurrent_state_is_refused_rather_than_served_unchecked():
    """No commissioned arena caches recurrent checkpoints; the pool is refused before any handoff."""
    ctx = _ctx(eagle=True, is_hybrid_ssm=True)
    ctx.params.req_to_token_pool.mamba_pool = NS()
    with pytest.raises(RuntimeError, match="state validation is unavailable"):
        _Guard(ctx)

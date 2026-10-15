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


@pytest.mark.parametrize("corrupt", [False, True])
def test_lazy_prefix_state_is_checked_when_the_runtime_resolves_its_slots(corrupt):
    from tests.test_sglang_cache_seam import LazyMatchResult

    swa = NS(size=64, page_size=PAGE, kv_buffer=[torch.zeros(64 // PAGE + 1, PAGE * 2)])
    guard = _Guard(_window_engine(swa))
    cache, tokens, row = _cache(guard), list(range(16)), torch.arange(PAGE, PAGE + 16)
    req = _request("source", tokens, 0)
    guard.requests.req_to_token[0, :16] = row.int()
    guard.handoff(cache, req, tokens, finished=True)
    query = _request("lazy", tokens + [99], 1)
    query.prefix_len = len(row)
    params = NS(key=Key(tokens + [99]), req=query)
    guard.matched(cache, params, LazyMatchResult(len(row), None), LazyMatchResult)
    assert query.rid in guard.hybrid.pending
    guard.hybrid.ready(cache)
    assert query.rid in guard.hybrid.pending
    if corrupt:
        swa.kv_buffer[0][2].add_(1)
        with pytest.raises(RuntimeError, match="sliding-window"):
            guard.hybrid.prefix(query, row, cache)
    else:
        guard.hybrid.prefix(query, row, cache)
        assert query.rid not in guard.hybrid.pending


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
@pytest.mark.parametrize("window", [2, 6, 8])
def test_a_window_pool_that_stores_whole_pages_is_checked_by_page(corrupt, window):
    swa = NS(size=64, page_size=PAGE, kv_buffer=[torch.zeros(64 // PAGE + 1, PAGE * 2)])
    ctx = _window_engine(swa)
    ctx.params.sliding_window_size = window
    guard = _Guard(ctx)
    assert guard.hybrid.paged == {("swa", "kv_buffer", 0)}
    cache = _cache(guard)
    tokens, row = list(range(16)), torch.arange(PAGE, PAGE + 16)
    guard.requests.req_to_token[0, :16] = row.int()
    for page in range(16 // PAGE):
        swa.kv_buffer[0][page] = float(sum(tokens[:(page + 1) * PAGE]))
    guard.handoff(cache, _request("source", tokens, 0), tokens, finished=True)
    if corrupt:
        swa.kv_buffer[0][(len(tokens) - window) // PAGE, 1] += 7
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
    """Unrecognized recurrent layouts remain refused before any handoff."""
    ctx = _ctx(eagle=True, is_hybrid_ssm=True)
    ctx.params.req_to_token_pool.mamba_pool = NS()
    with pytest.raises(RuntimeError, match="state validation is unavailable"):
        _Guard(ctx)


def _recurrent_engine(extra=True):
    ctx = _ctx(is_hybrid_ssm=True)
    ctx.params.sliding_window_size = None
    ctx.params.enable_mamba_extra_buffer = extra
    requests = ctx.params.req_to_token_pool
    requests.mamba_pool = NS(mamba_cache=NS(
        temporal=torch.randn(2, 16, 3, 4), conv=[torch.randn(2, 16, 3, 2)]))
    # Exercise virtual-to-physical translation, not only identity slot maps.
    requests.translate_mamba_indices = lambda slots: (slots + 1) % 16
    requests.get_mamba_ping_pong_keep_idx = lambda req: req.kv.mamba_last_track_idx
    return ctx


def _recurrent_request(rid, tokens, row):
    req = _request(rid, tokens, row)
    req.kv.mamba_pool_idx = torch.tensor(3)
    req.kv.mamba_last_track_seqlen = len(tokens)
    req.kv.mamba_last_track_idx = 1
    req.kv.mamba_ping_pong_track_buffer = torch.tensor([4, 5])
    req.kv.mamba_cow_src_index = None
    req.kv.mamba_needs_clear = False
    return req


def _recurrent_match(guard, cache, tokens, slot, *, host=False, salt=None, cow=True):
    req = _recurrent_request("query", tokens + [99], 1)
    req.cache_salt = salt
    req.kv.mamba_cow_src_index = None if host else torch.tensor([slot])
    req.kv.mamba_pool_idx = torch.tensor(slot)
    row = torch.arange(PAGE, PAGE + len(tokens))
    req.prefix_indices = row
    params = NS(key=Key(tokens + [99]), req=req, cow_mamba=cow)
    result = NS(device_indices=row, host_hit_length=0, swa_host_hit_length=0,
                mamba_host_hit_length=int(host))
    guard.matched(cache, params, result, NS)
    return req


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("change", [None, "conv", "temporal", "prefix", "namespace", "reset"])
def test_recurrent_hits_bind_all_state_kinds_to_prefix_and_flush(extra, host, change):
    guard = _Guard(_recurrent_engine(extra))
    cache = _cache(guard)
    tokens = list(range(16))
    req = _recurrent_request("source", tokens, 0)
    guard.requests.req_to_token[0, :16] = torch.arange(PAGE, PAGE + 16).int()
    guard.handoff(cache, req, tokens, finished=True)
    state = guard.requests.mamba_pool.mamba_cache
    source = 6 if extra else 4  # translated tracked/live slot
    # Legitimate movement (including a host round trip) preserves content.
    for buffer in [state.temporal, *state.conv]:
        buffer[:, 10].copy_(buffer[:, source])
    if change in ("conv", "temporal"):
        buffer = state.temporal if change == "temporal" else state.conv[0]
        buffer[0, 10].add_(1)
    elif change == "prefix":
        tokens[0] += 1
    elif change == "reset":
        guard.reset()
    _recurrent_match(guard, cache, tokens, 9, host=host,
                     salt="different" if change == "namespace" else None)
    if change:
        with pytest.raises(RuntimeError, match="recurrent"):
            guard.hybrid.ready(cache) if host else guard.poll(block=True)
    else:
        if host:
            guard.hybrid.ready(cache)
        guard.poll(block=True)


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_unfinished_request_keeps_live_recurrent_state_across_donation(extra, corrupt):
    guard = _Guard(_recurrent_engine(extra))
    cache = _cache(guard)
    tokens = list(range(16))
    req = _recurrent_request("source", tokens, 0)
    row = torch.arange(PAGE, PAGE + 16)
    guard.requests.req_to_token[0, :16] = row.int()
    handed = guard.handoff(cache, req, tokens)
    # The cache can donate/replace a checkpoint buffer; it cannot alter live state.
    req.kv.mamba_ping_pong_track_buffer[1] = 8
    req.kv.mamba_last_track_seqlen = None
    if corrupt:
        guard.requests.mamba_pool.mamba_cache.temporal[0, 4].add_(1)
    req.prefix_indices, req.kv.cache_protected_len = row.clone(), 16
    guard.settle(req, handed)
    if corrupt:
        with pytest.raises(RuntimeError, match="recurrent"):
            guard.poll(block=True)
    else:
        guard.poll(block=True)


def test_checkpoint_length_uses_tracked_boundary_and_rejects_uncomputed_tail():
    guard = _Guard(_recurrent_engine())
    cache = _cache(guard)
    tokens = list(range(16))
    req = _recurrent_request("source", tokens, 0)
    guard.requests.req_to_token[0, :16] = torch.arange(PAGE, PAGE + 16).int()
    req.kv.mamba_last_track_seqlen = 12
    guard.handoff(cache, req, tokens, finished=True)
    _recurrent_match(guard, cache, tokens[:12], 5)
    guard.poll(block=True)
    _recurrent_match(guard, cache, tokens, 5)
    with pytest.raises(RuntimeError, match="recurrent"):
        guard.poll(block=True)


def test_internal_match_does_not_restore_an_older_checkpoint_over_live_state():
    """UnifiedRadixCache rematches a request with cow_mamba=False after insertion."""
    guard = _Guard(_recurrent_engine())
    cache = _cache(guard)
    tokens = list(range(16))
    req = _recurrent_request("source", tokens, 0)
    guard.requests.req_to_token[0, :16] = torch.arange(PAGE, PAGE + 16).int()
    req.kv.mamba_last_track_seqlen = 12
    guard.handoff(cache, req, tokens, finished=True)
    # Live slot 3 contains the newer 16-token state, not tracked slot 5's prefix.
    _recurrent_match(guard, cache, tokens[:12], 3, cow=False)
    guard.poll(block=True)
    _recurrent_match(guard, cache, tokens[:12], 5)
    guard.poll(block=True)


@pytest.mark.parametrize("feature", ["mamba_ckpt_pool", "enable_linear_replayssm", "enable_linear_replayssm_spec", "intermediate_ssm"])
def test_other_recurrent_representations_still_require_adapter_support(feature):
    ctx = _recurrent_engine()
    target = ctx.params.req_to_token_pool
    if feature != "mamba_ckpt_pool":
        target = target.mamba_pool
    if feature == "intermediate_ssm":
        target = target.mamba_cache
    setattr(target, feature, True)
    with pytest.raises(RuntimeError, match="state validation is unavailable"):
        _Guard(ctx)

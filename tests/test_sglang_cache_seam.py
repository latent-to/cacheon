"""The prefix-cache seam: a bundle's cache is built where stock's is, on the engine's pools.

What a cache can fake is a hit: a claimed prefix the KV slots do not hold skips its
prefill and returns garbage fast. So besides binding, these tests pin that every
claim is checked against the validator's own record of what each slot holds, and
refused when it is wrong, stitched, stale, too long or from another namespace.
SGLang is not imported; the fakes carry the 0.5.20 shapes the seam reads.
"""

from __future__ import annotations

import abc
import sys
import types
from array import array
from typing import Any, NamedTuple

import pytest
import torch

from cacheon import receipts, seam, seams
from cacheon.engine_tree import EngineTreeError, inspect_contribution
from cacheon.integrations import sglang_cache, sglang_nodes
from cacheon.integrations.sglang_cache import ADDRESS
from cacheon.kernel_trace import arm
from cacheon.registry import REGISTRY, KernelImpl, KernelRegistry
from cacheon.target_catalog import FORWARD_PASS_ROOTS, default_target_catalog
from tests.test_target_catalog import _bundle

_REGISTRY = "sglang.srt.mem_cache.registry"
_BASE = "sglang.srt.mem_cache.base_prefix_cache"
_CACHE_SOURCE = (
    "from sglang.srt.mem_cache.radix_cache import RadixCache as Stock\n\n\n"
    "class Local(Stock):\n    pass\n\n\nclass entry_0(Local):\n    pass\n"
)


class MatchResult(NamedTuple):
    """The fields of SGLang 0.5.20's match result that the seam reads."""

    device_indices: torch.Tensor
    last_device_node: Any = None
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0


class BasePrefixCache(abc.ABC):
    """SGLang 0.5.20's abstract cache, reduced to the methods the seam wraps."""

    @abc.abstractmethod
    def match_prefix(self, params): ...

    @abc.abstractmethod
    def cache_finished_req(self, req, is_insert=True, **kwargs): ...

    @abc.abstractmethod
    def cache_unfinished_req(self, req, **kwargs): ...


class Allocator:
    """The engine's free list: slots freed are the first handed out again."""

    size, page_size, device = 32, 1, torch.device("cpu")

    def __init__(self):
        self.clear()

    def clear(self):
        self.free_slots = list(range(1, self.size + 1))

    def alloc(self, need):
        out, self.free_slots = self.free_slots[:need], self.free_slots[need:]
        return torch.tensor(out, dtype=torch.int64)

    alloc_extend = alloc_decode = alloc

    def free(self, slots):
        self.free_slots[:0] = slots


class Key:
    """A radix key: raw token ids, capped like the scheduler's at ``limit``."""

    def __init__(self, tokens, limit=None, extra_key=None, cache_salt=None):
        self.tokens, self.limit = array("q", tokens), limit
        self.extra_key, self.cache_salt = extra_key, cache_salt

    def raw_token_ids(self):
        return self.tokens[: self.limit]


class Req:
    """The request fields the seam reads: its tokens, its KV row and its namespace."""

    def __init__(self, tokens, row):
        self.origin_input_ids, self.output_ids = array("q", tokens), array("q")
        self.kv = types.SimpleNamespace(req_pool_idx=row)
        self.extra_key = self.cache_salt = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)

    def get_fill_ids(self):
        return self.origin_input_ids + self.output_ids


class Exact(BasePrefixCache):
    """An honest toy cache matching every row it was handed by longest common prefix."""

    def __init__(self, params):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.rows: list[tuple[tuple, torch.Tensor]] = []
        self.claim, self.fields = None, {}  # a test scripts a claim through these

    def match_prefix(self, params):
        if self.claim is not None:
            return MatchResult(torch.tensor(self.claim, dtype=torch.int64), **self.fields)
        ids, best = tuple(params.key.raw_token_ids()), torch.empty(0, dtype=torch.int64)
        for tokens, slots in self.rows:
            n = next((i for i, (a, b) in enumerate(zip(ids, tokens)) if a != b),
                     min(len(ids), len(tokens)))
            best = slots[:n] if n > len(best) else best
        return MatchResult(best)

    def _keep(self, req, tokens):
        row = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, : len(tokens)]
        self.rows.append((tuple(tokens), row.to(torch.int64)))
        return self.rows[-1][1]

    def cache_finished_req(self, req, is_insert=True, **kwargs):
        self._keep(req, (req.origin_input_ids + req.output_ids)[: kwargs["kv_len_to_handle"]])

    def cache_unfinished_req(self, req, **kwargs):
        req.prefix_indices = self._keep(req, req.get_fill_ids()).clone()


class OwnPool(Exact):
    """Builds an allocator of its own instead of keeping the engine's."""

    def __init__(self, params):
        super().__init__(params)
        self.token_to_kv_pool_allocator = Allocator()


class Misfiled(Exact):
    """Leaves an honest prefix on the request but points the row the forward reads elsewhere."""

    def cache_unfinished_req(self, req, **kwargs):
        super().cache_unfinished_req(req, **kwargs)
        self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, 0] = 5


class Borrowing(Exact):
    """Leaves another request's slots as the prefix of a chunked request."""

    def cache_unfinished_req(self, req, **kwargs):
        super().cache_unfinished_req(req, **kwargs)
        req.prefix_indices = torch.tensor([5, 6], dtype=torch.int64)


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    rdir = tmp_path / "receipts"
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(rdir))
    monkeypatch.setattr(receipts, "_ONCE", set())
    base = types.ModuleType(_BASE)
    base.BasePrefixCache, base.MatchResult = BasePrefixCache, MatchResult
    module = types.ModuleType(_REGISTRY)
    module.default_radix_cache_factory = lambda ctx: "stock"
    # As in 0.5.20: create_tree_cache reads the factory from its module per call.
    module.create_tree_cache = lambda ctx: module.default_radix_cache_factory(ctx)
    monkeypatch.setitem(sys.modules, _BASE, base)
    monkeypatch.setitem(sys.modules, _REGISTRY, module)
    yield types.SimpleNamespace(receipts=rdir, module=module)
    REGISTRY.clear()
    REGISTRY.disable()


def _ctx(eagle=False, **flags):
    pool = types.SimpleNamespace(req_to_token=torch.zeros((4, 16), dtype=torch.int32))
    params = types.SimpleNamespace(
        token_to_kv_pool_allocator=Allocator(), req_to_token_pool=pool, is_eagle=eagle
    )
    shape = dict(is_hybrid_swa=False, is_hybrid_ssm=False, enable_hierarchical_cache=False)
    return types.SimpleNamespace(params=params, **{**shape, **flags})


def _build(engine, entry=Exact, ctx=None):
    REGISTRY.register(KernelImpl(slot=ADDRESS, bundle_id="cache-test", entry=entry))
    REGISTRY.enable()
    sglang_cache.install(REGISTRY)
    ctx = ctx or _ctx()
    return engine.module.create_tree_cache(ctx), ctx.params


def _run(params, tokens, row):
    """Place a request's KV as the scheduler does, in fresh slots."""
    slots = params.token_to_kv_pool_allocator.alloc(len(tokens))
    params.req_to_token_pool.req_to_token[row, : len(tokens)] = slots.to(torch.int32)
    return Req(tokens, row)


def _served(engine, eagle=False):
    """Finished requests [5, 6, 7, 8], [1, 2, 3] and [0, 1, 2] in slots 1-4, 5-7 and 8-10."""
    cache, params = _build(engine, ctx=_ctx(eagle))
    for row, tokens in enumerate(([5, 6, 7, 8], [1, 2, 3], [0, 1, 2])):
        cache.cache_finished_req(_run(params, tokens, row), kv_len_to_handle=len(tokens))
    return cache, params


def _match(cache, tokens, **key):
    return cache.match_prefix(types.SimpleNamespace(key=Key(tokens, **key)))


def test_one_address_is_the_seam_row_the_catalog_root_and_no_node(monkeypatch):
    row = next(a for a in seams.SEAM_ADAPTERS if a.integration == "sglang_cache")
    assert (row.target_module, row.chokepoint) == (_REGISTRY, "default_radix_cache_factory")
    assert _REGISTRY in seams.TARGET_MODULES and ADDRESS in FORWARD_PASS_ROOTS
    registry = KernelRegistry()
    registry.register(KernelImpl(slot=ADDRESS, bundle_id="cache-test", entry=Exact))
    assert sglang_nodes.bind(types.SimpleNamespace(model=torch.nn.Module()), registry) == []
    # An audit launch traces entries; a class is built once and must stay a class.
    monkeypatch.setenv("CACHEON_KERNEL_TRACE", "1")
    arm(registry)
    assert registry.variants(ADDRESS)[0].entry is Exact


def test_the_engine_chain_builds_the_registered_cache_on_the_engine_pools(engine):
    stock = engine.module.default_radix_cache_factory
    seam._install_adapters(False)  # the seam table's own install loop
    assert engine.module.default_radix_cache_factory._cacheon_stock_factory is stock
    assert engine.module.create_tree_cache(_ctx()) == "stock"  # the reference arm
    REGISTRY.register(KernelImpl(slot="model.layers.*.mlp", bundle_id="nodes", entry=print))
    REGISTRY.enable()
    assert engine.module.create_tree_cache(_ctx()) == "stock"  # a bundle naming no cache
    cache, params = _build(engine)  # installs again: still one wrapper, around stock
    assert engine.module.default_radix_cache_factory._cacheon_stock_factory is stock
    assert isinstance(cache, Exact) and type(cache).__name__ == "Exact"
    assert cache.token_to_kv_pool_allocator is params.token_to_kv_pool_allocator


@pytest.mark.parametrize("slot, source, cause", [
    (ADDRESS, None, "is not a class defined in its source"),
    (ADDRESS, "class entry_0(dict):\n    pass\n", "does not derive from a class of sglang"),
    ("tree_cache.lru", _CACHE_SOURCE, "has no sub-addresses"),
])
def test_intake_refuses_a_cache_entry_it_cannot_see_is_a_cache_class(tmp_path, slot, source, cause):
    root = _bundle(tmp_path, rows=({"slot": slot},))
    if source is not None:
        (root / "kernels" / "k0.py").write_text(source)
    with pytest.raises(EngineTreeError, match=cause):
        inspect_contribution(root, catalog=default_target_catalog())


def test_intake_admits_a_cache_class_to_the_forward_pass(tmp_path):
    root = _bundle(tmp_path, rows=({"slot": ADDRESS},))
    (root / "kernels" / "k0.py").write_text(_CACHE_SOURCE)
    assert inspect_contribution(root, catalog=default_target_catalog()).target_id == "forward_pass"


@pytest.mark.parametrize("entry, cause", [
    (dict, "is not a BasePrefixCache subclass"),
    (type("Partial", (BasePrefixCache,), {"match_prefix": lambda self, params: None}), "abstract"),
    (OwnPool, "must hold the engine's KV allocator and request pool"),
])
def test_the_scheduler_refuses_what_intake_could_not_see_as_the_candidate_s(engine, entry, cause):
    with pytest.raises(TypeError, match=cause):
        _build(engine, entry)
    (row,) = receipts.collect(engine.receipts, "failed")
    assert (row["slot"], row["phase"], row["error_type"]) == (ADDRESS, "prepare", "TypeError")


def test_a_hybrid_or_hierarchical_engine_is_refused_as_the_arena_s_not_the_candidate_s(engine):
    with pytest.raises(RuntimeError, match="full-attention device caches only"):
        _build(engine, ctx=_ctx(enable_hierarchical_cache=True))
    assert receipts.collect(engine.receipts, "failed") == []


def test_an_honest_hit_passes_and_the_scheduler_keeps_the_copy_that_was_checked(engine):
    cache, _ = _served(engine)
    hit = _match(cache, [5, 6, 7, 9])
    cache.rows[0][1][:] = 0
    assert hit.device_indices.tolist() == [1, 2, 3]
    assert receipts.collect(engine.receipts, "completed")[0]["slot"] == ADDRESS


@pytest.mark.parametrize("prompt, claim, key, fields, cause", [
    ([5, 6, 7], [5, 6, 7], {}, {}, "position 0"),  # slots of another prefix
    ([5, 6, 3], [1, 2, 7], {}, {}, "position 2"),  # slot 7 holds a 3, but after [1, 2]
    ([5, 6, 7], [1, 2, 99], {}, {}, "position 2"),  # no such slot
    ([1, 2], [9, 10], {}, {}, "position 0"),  # the 1, 2 of [0, 1, 2]: a leading 0 counts
    ([5, 6, 7], [1, 2, 3], {"cache_salt": "tenant-b"}, {}, "position 0"),
    ([5, 6, 7, 8], [1, 2, 3, 4], {"limit": 3}, {}, "claims 4 cached tokens of a 3-token"),
    ([5, 6, 7], [], {}, {"host_hit_length": 3}, "host-tier hit"),
])
def test_a_false_claim_dies_at_the_claim(engine, prompt, claim, key, fields, cause):
    cache, _ = _served(engine)
    cache.claim, cache.fields = claim, fields
    with pytest.raises(RuntimeError, match=cause):
        _match(cache, prompt, **key)
    assert receipts.collect(engine.receipts, "failed")[0]["phase"] == "entry"


@pytest.mark.parametrize("take_back", ["reallocate", "flush"])
def test_a_slot_handed_out_again_or_flushed_no_longer_holds_its_prefix(engine, take_back):
    cache, params = _served(engine)
    cache.claim = [1, 2, 3]
    assert _match(cache, [5, 6, 7]).device_indices.tolist() == [1, 2, 3]
    if take_back == "flush":
        params.token_to_kv_pool_allocator.clear()
    else:
        params.token_to_kv_pool_allocator.free([1, 2, 3, 4])
        _run(params, [9, 9], row=3)  # new KV in slots 1 and 2 that nobody recorded yet
    with pytest.raises(RuntimeError, match="position 0"):
        _match(cache, [5, 6, 7])


def test_under_eagle_a_slot_also_holds_the_token_after_it(engine):
    cache, _ = _served(engine, eagle=True)
    cache.claim = [1, 2]
    assert _match(cache, [5, 6, 7, 0]).device_indices.tolist() == [1, 2]
    cache.claim = [1, 2, 3]  # slot 3's draft KV read the 8 that followed it, not a 0
    with pytest.raises(RuntimeError, match="position 2"):
        _match(cache, [5, 6, 7, 0])


@pytest.mark.parametrize("entry, eagle, cause", [
    (Exact, False, None),
    (Exact, True, None),  # its own last slot waits for the next token to be recorded
    (Misfiled, False, "position 0"),
    (Borrowing, False, "position 0"),
])
def test_the_prefix_a_chunked_insert_leaves_on_the_request_is_checked(engine, entry, eagle, cause):
    cache, params = _build(engine, entry, ctx=_ctx(eagle))
    req = _run(params, [5, 6, 7], row=0)
    if cause is None:
        cache.cache_unfinished_req(req, chunked=True)
        assert req.prefix_indices.tolist() == [1, 2, 3]
        return
    with pytest.raises(RuntimeError, match=cause):
        cache.cache_unfinished_req(req, chunked=True)


def test_a_row_pointed_at_another_prefix_s_kv_is_refused_at_the_next_claim(engine):
    cache, params = _served(engine)
    stray = _run(params, [9, 9, 9], row=3)
    params.req_to_token_pool.req_to_token[3, 0] = 5  # slot 5 holds the [1] of [1, 2, 3]
    cache.cache_finished_req(stray, kv_len_to_handle=3)
    with pytest.raises(RuntimeError, match="another prefix's KV"):
        _match(cache, [5, 6, 7])


def test_a_cache_that_trades_the_engine_pools_away_is_refused_at_its_next_claim(engine):
    cache, _ = _served(engine)
    cache.token_to_kv_pool_allocator = Allocator()
    with pytest.raises(TypeError, match="must hold the engine's KV allocator"):
        _match(cache, [5, 6, 7])

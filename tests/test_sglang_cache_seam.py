"""The prefix-cache seam: the stock chain builds a bundle's cache, and what it serves is checked by content.

What a cache can fake is a hit: a served prefix whose KV slots do not hold what the
engine computed skips its prefill and returns garbage fast. These tests pin the
binding (the chain gives the candidate what it gives stock) and the check: every
served page must hold bytes the engine computed for that prefix, wherever the bytes
travelled in between. SGLang is not imported; the stand-ins carry the 0.5.20 shapes
the seam reads, and a toy model writes KV that is a pure function of each
position's token prefix, so honest caching reproduces it exactly.
"""

from __future__ import annotations

import sys
import types
from array import array
from typing import Any, NamedTuple

import pytest
import torch

from cacheon import audit, receipts, seam, seams
from cacheon.engine_tree import EngineTreeError, inspect_contribution
from cacheon.integrations import sglang_cache, sglang_nodes
from cacheon.integrations.sglang_cache import ADDRESS
from cacheon.kernel_trace import arm
from cacheon.registry import REGISTRY, KernelImpl, KernelRegistry
from cacheon.target_catalog import FORWARD_PASS_ROOTS, default_target_catalog
from tests.test_target_catalog import _bundle

_REGISTRY = "sglang.srt.mem_cache.registry"
_BASE = "sglang.srt.mem_cache.base_prefix_cache"
_HOME = "sglang.srt.mem_cache.unified_radix_cache"
_CACHE_SOURCE = (
    "from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as Stock\n\n\n"
    "class Local(Stock):\n    pass\n\n\nclass entry_0(Local):\n    pass\n"
)
PAGE, SLOTS = 4, 256


class MatchResult(NamedTuple):
    """The fields of SGLang 0.5.20's match result that the seam and the stand-in read."""

    device_indices: torch.Tensor
    last_device_node: Any = None
    host_hit_length: int = 0


class Key:
    """A radix key: raw token ids, capped like the scheduler's at ``limit``."""

    is_bigram = False

    def __init__(self, tokens, limit=None):
        self.tokens, self.limit = array("q", tokens), limit

    def raw_token_ids(self):
        return self.tokens[: self.limit]

    def __len__(self):
        return len(self.raw_token_ids())


class UnifiedRadixCache:
    """SGLang 0.5.20's stock cache, reduced: page-aligned prefixes of what requests
    handed in, served longest first, a later duplicate pointed at the first's slots."""

    def __init__(self, params):
        assert params.tree_components is not None, "only the stock chain sets the components"
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.bigram = int(params.is_eagle)  # EAGLE keys pair each token with the next
        self.reset()

    def reset(self):
        self.pages: dict[tuple, torch.Tensor] = {}

    def match_prefix(self, params):
        ids, best = tuple(params.key.raw_token_ids()), torch.empty(0, dtype=torch.int64)
        for m in range(PAGE, (len(ids) - self.bigram) // PAGE * PAGE + 1, PAGE):
            if ids[:m] not in self.pages:
                break
            best = self.pages[ids[:m]]
        return MatchResult(best)

    def _insert(self, req, n):
        tokens = tuple((req.origin_input_ids + req.output_ids)[:n])
        row = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :n].long()
        slots = row[:0]
        for m in range(PAGE, (n - self.bigram) // PAGE * PAGE + 1, PAGE):
            slots = self.pages.setdefault(tokens[:m], torch.cat((slots, row[m - PAGE : m])))
        return slots, row

    def cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle, **kwargs):
        if is_insert:
            self._insert(req, kv_len_to_handle)

    def cache_unfinished_req(self, req, chunked=False, **kwargs):
        slots, row = self._insert(req, len(req.get_fill_ids()))
        self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, : len(slots)] = slots.int()
        req.prefix_indices = torch.cat((slots, row[len(slots) :]))
        req.kv.cache_protected_len = len(slots)


class Allocator:
    """The engine's free list over one KV pool: slots freed are the first handed out again."""

    size, page_size, device = SLOTS, PAGE, "cpu"

    def __init__(self, pool):
        self.pool, self.free_slots = pool, list(range(PAGE, SLOTS))

    def get_kvcache(self):
        return self.pool

    def alloc(self, need):
        out, self.free_slots = self.free_slots[:need], self.free_slots[need:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, slots):
        self.free_slots[:0] = slots.tolist()


class Req:
    """The request fields the seam and the stand-in read."""

    def __init__(self, rid, tokens, row):
        self.rid, self.origin_input_ids, self.output_ids = rid, array("q", tokens), array("q")
        self.kv = types.SimpleNamespace(req_pool_idx=row, cache_protected_len=0)
        self.extra_key = self.cache_salt = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.fill = len(tokens)

    def get_fill_ids(self):
        return (self.origin_input_ids + self.output_ids)[: self.fill]


class Engine:
    """The scheduler's side of each handoff, and a model whose KV is a function of the prefix."""

    def __init__(self, cache, eagle=False):
        self.cache, self.params = cache, cache.guard_params
        self.requests, self.allocator = self.params.req_to_token_pool, self.params.token_to_kv_pool_allocator
        self.eagle, self.rows = eagle, 0

    def write(self, slots, tokens, positions):
        for slot, i in zip(slots.tolist(), positions):
            prefix = hash(tuple(tokens[: i + 1 + self.eagle])) & (1 << 62) - 1
            generator = torch.Generator().manual_seed(prefix)
            for buffer in (*self.allocator.pool.k_buffer, *self.allocator.pool.v_buffer):
                buffer[slot] = torch.randn(buffer.shape[1:], generator=generator)

    def admit(self, tokens, salt=None):
        req = Req(f"r{self.rows}", tokens, self.rows % 8)
        req.cache_salt, self.rows = salt, self.rows + 1
        req.prefix_indices = self.cache.match_prefix(
            types.SimpleNamespace(key=Key(tokens, limit=len(tokens) - 1))).device_indices
        req.kv.cache_protected_len = n = len(req.prefix_indices)
        self.requests.req_to_token[req.kv.req_pool_idx, :n] = req.prefix_indices.int()
        return req

    def compute(self, req, start, end):
        tokens = list(req.origin_input_ids + req.output_ids)
        slots = self.allocator.alloc(end - start)
        self.requests.req_to_token[req.kv.req_pool_idx, start:end] = slots.int()
        self.write(slots, tokens, range(start, end))
        req.fill = end

    def serve(self, tokens, chunk=None, decode=2, salt=None):
        req = self.admit(tokens, salt)
        start = len(req.prefix_indices)
        while chunk and len(tokens) - start > chunk:
            self.compute(req, start, start + chunk)
            self.cache.cache_unfinished_req(req, chunked=True)
            start = len(req.prefix_indices)
        self.compute(req, start, len(tokens))
        self.cache.cache_unfinished_req(req)
        for step in range(decode):
            req.output_ids.append(9000 + step)
            self.compute(req, req.fill, req.fill + 1)
        self.cache.cache_finished_req(req, kv_len_to_handle=req.fill)
        return req


@pytest.fixture()
def sglang(tmp_path, monkeypatch):
    """Stand-ins for the three SGLang modules the seam touches, the chain as 0.5.20 runs it."""

    rdir = tmp_path / "receipts"
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(rdir))
    monkeypatch.setattr(receipts, "_ONCE", set())
    base, home, module = (types.ModuleType(name) for name in (_BASE, _HOME, _REGISTRY))
    base.MatchResult, home.UnifiedRadixCache = MatchResult, UnifiedRadixCache

    def default_radix_cache_factory(ctx):
        if ctx.disable_radix_cache:
            return types.SimpleNamespace()  # stands for ChunkCache
        ctx.params.tree_components = ("full",)
        return home.UnifiedRadixCache(ctx.params)  # the class is looked up per call

    module.default_radix_cache_factory = default_radix_cache_factory
    module.create_tree_cache = lambda ctx: module.default_radix_cache_factory(ctx)
    for name, stand_in in ((_BASE, base), (_HOME, home), (_REGISTRY, module)):
        monkeypatch.setitem(sys.modules, name, stand_in)
    yield types.SimpleNamespace(receipts=rdir, module=module, home=home)
    REGISTRY.clear()
    REGISTRY.disable()


def _ctx(eagle=False, **flags):
    pool = types.SimpleNamespace(
        k_buffer=[torch.zeros(SLOTS, 2, 4) for _ in range(2)],
        v_buffer=[torch.zeros(SLOTS, 2, 4, dtype=torch.bfloat16) for _ in range(2)],
    )
    params = types.SimpleNamespace(
        token_to_kv_pool_allocator=Allocator(pool), page_size=PAGE, is_eagle=eagle,
        req_to_token_pool=types.SimpleNamespace(req_to_token=torch.zeros((8, 64), dtype=torch.int32)),
        mtp_draft_device_pools=(), tree_components=None,
    )
    shape = dict(is_hybrid_swa=False, is_hybrid_ssm=False, disable_radix_cache=False, tp_worker=None)
    return types.SimpleNamespace(params=params, **{**shape, **flags})


def _engine(sglang, entry=None, eagle=False, **flags):
    REGISTRY.register(KernelImpl(slot=ADDRESS, bundle_id="cache-test",
                                 entry=entry or type("Miner", (UnifiedRadixCache,), {})))
    REGISTRY.enable()
    sglang_cache.install(REGISTRY)
    ctx = _ctx(eagle, **flags)
    cache = sglang.module.create_tree_cache(ctx)
    cache.guard_params = ctx.params
    return Engine(cache, eagle)


def _refusal(sglang, cause):
    (row,) = receipts.collect(sglang.receipts, "failed")
    assert row["slot"] == ADDRESS and cause in row["error"]


A, B, C = list(range(1, 23)), list(range(40, 51)), list(range(60, 79))


def test_one_address_is_the_seam_row_the_catalog_root_and_no_node(monkeypatch):
    row = next(a for a in seams.SEAM_ADAPTERS if a.integration == "sglang_cache")
    assert (row.target_module, row.chokepoint) == (_REGISTRY, "default_radix_cache_factory")
    assert _REGISTRY in seams.TARGET_MODULES and ADDRESS in FORWARD_PASS_ROOTS
    registry = KernelRegistry()
    registry.register(KernelImpl(slot=ADDRESS, bundle_id="cache-test", entry=UnifiedRadixCache))
    assert sglang_nodes.bind(types.SimpleNamespace(model=torch.nn.Module()), registry) == []
    # An audit launch traces entries; a class is built once and must stay a class.
    monkeypatch.setenv("CACHEON_KERNEL_TRACE", "1")
    arm(registry)
    assert registry.variants(ADDRESS)[0].entry is UnifiedRadixCache


def test_the_stock_chain_builds_the_candidate_as_it_builds_stock(sglang):
    stock = sglang.module.default_radix_cache_factory
    seam._install_adapters(False)  # the seam table's own install loop
    assert sglang.module.default_radix_cache_factory._cacheon_stock_factory is stock
    assert type(sglang.module.create_tree_cache(_ctx())) is UnifiedRadixCache  # the reference arm
    REGISTRY.register(KernelImpl(slot="model.layers.*.mlp", bundle_id="nodes", entry=print))
    REGISTRY.enable()
    assert type(sglang.module.create_tree_cache(_ctx())) is UnifiedRadixCache  # no cache named

    class Miner(UnifiedRadixCache):
        pass

    engine = _engine(sglang, Miner)  # installs again: still one wrapper, around stock
    assert sglang.module.default_radix_cache_factory._cacheon_stock_factory is stock
    assert type(engine.cache).__mro__[1] is Miner and type(engine.cache).__name__ == "Miner"
    assert engine.cache.token_to_kv_pool_allocator is engine.allocator
    assert sglang.home.UnifiedRadixCache is UnifiedRadixCache  # the swap ends with the call


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


class OwnPool(UnifiedRadixCache):
    """Builds an allocator of its own instead of keeping the engine's."""

    def __init__(self, params):
        super().__init__(params)
        self.token_to_kv_pool_allocator = Allocator(params.token_to_kv_pool_allocator.pool)


@pytest.mark.parametrize("entry, cause", [
    (dict, "is not a UnifiedRadixCache subclass"),
    (OwnPool, "must hold the engine's KV allocator and request pool"),
])
def test_the_scheduler_refuses_what_intake_could_not_see_as_the_candidate_s(sglang, entry, cause):
    with pytest.raises(TypeError, match=cause):
        _engine(sglang, entry)
    (row,) = receipts.collect(sglang.receipts, "failed")
    assert (row["slot"], row["phase"], row["error_type"]) == (ADDRESS, "prepare", "TypeError")


@pytest.mark.parametrize("flags, cause", [
    ({"is_hybrid_ssm": True}, "full-attention models only"),
    ({"disable_radix_cache": True}, "this engine builds SimpleNamespace"),
])
def test_an_engine_that_builds_no_unified_cache_is_the_arena_s_failure(sglang, flags, cause):
    with pytest.raises(RuntimeError, match=cause):
        _engine(sglang, **flags)
    assert receipts.collect(sglang.receipts, "failed") == []


@pytest.mark.parametrize("eagle", [False, True])
def test_an_honest_cache_passes_reuse_chunks_duplicates_and_a_flush(sglang, eagle):
    engine = _engine(sglang, eagle=eagle)
    follow = A + list(engine.serve(A).output_ids) + B
    hit = engine.cache.match_prefix(types.SimpleNamespace(key=Key(follow, len(follow) - 1)))
    assert len(hit.device_indices) == 24 - 4 * eagle  # the first request's prompt and outputs
    engine.serve(follow, chunk=6)
    twins = [engine.admit(C), engine.admit(C)]  # both prefilled before either is handed over
    for req in twins:
        engine.compute(req, 0, len(C))
    for req in twins:
        engine.cache.cache_unfinished_req(req)
    assert torch.equal(twins[1].prefix_indices[:16], twins[0].prefix_indices[:16])
    for req in twins:
        engine.cache.cache_finished_req(req, kv_len_to_handle=req.fill)
    engine.serve(C + B)
    engine.cache.reset()
    engine.serve(A)
    assert receipts.collect(sglang.receipts, "failed") == []
    assert receipts.collect(sglang.receipts, "completed")[0]["slot"] == ADDRESS


class Tiered(UnifiedRadixCache):
    """A RAM tier of the bundle's own: copies finished pages to host memory and serves a
    miss by copying them into fresh slots, or, when ``copy`` is off, by not copying."""

    copy = True

    def cache_finished_req(self, req, is_insert=True, **kwargs):
        n = len(req.origin_input_ids) // PAGE * PAGE
        slots = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :n].long()
        pool = self.token_to_kv_pool_allocator.pool
        self.host = (tuple((req.origin_input_ids + req.output_ids)[:n]),
                     [b[slots].clone() for b in (*pool.k_buffer, *pool.v_buffer)])
        super().cache_finished_req(req, is_insert, **kwargs)

    def match_prefix(self, params):
        result = super().match_prefix(params)
        tokens, rows = getattr(self, "host", ((), ()))
        n = min(len(tokens), len(params.key) // PAGE * PAGE)
        if len(result.device_indices) or not n or tuple(params.key.raw_token_ids()[:n]) != tokens[:n]:
            return result
        slots = self.token_to_kv_pool_allocator.alloc(n)
        pool = self.token_to_kv_pool_allocator.pool
        for buffer, row in zip((*pool.k_buffer, *pool.v_buffer), rows):
            buffer[slots] = row[:n] if self.copy else buffer[slots]
        return MatchResult(slots)


@pytest.mark.parametrize("copy", [True, False])
def test_a_ram_tier_passes_exactly_when_it_brings_the_bytes_back(sglang, copy):
    engine = _engine(sglang, type("Tier", (Tiered,), {"copy": copy}))
    engine.serve(A)
    engine.cache.pages.clear()  # the tree evicted A; only the tier's host copy is left
    if copy:
        engine.serve(A + B)
        return
    with pytest.raises(RuntimeError, match="does not hold the KV the engine computed"):
        engine.serve(A + B)
    _refusal(sglang, "served a page")


class Liar(UnifiedRadixCache):
    """Serves the slots of the first prompt it saw to any other prompt."""

    def match_prefix(self, params):
        result = super().match_prefix(params)
        first = next(iter(self.pages.values()), None) if not len(result.device_indices) else None
        return result if first is None else MatchResult(first)


# The stand-in tree ignores salts, so it serves A's pages across namespaces as they are.
@pytest.mark.parametrize("second, salt", [(C, None), (A, "tenant-b")])
def test_a_served_page_the_engine_did_not_compute_for_that_prefix_is_refused(sglang, second, salt):
    engine = _engine(sglang, Liar)
    engine.serve(A)
    with pytest.raises(RuntimeError, match="does not hold the KV the engine computed"):
        engine.serve(second, salt=salt)
    _refusal(sglang, "served a page")


def test_a_flush_forgets_every_page(sglang):
    engine = _engine(sglang)
    engine.serve(A)
    pages = dict(engine.cache.pages)
    engine.cache.reset()
    engine.cache.pages.update(pages)  # the cache keeps slots the engine let go of
    with pytest.raises(RuntimeError, match="does not hold the KV the engine computed"):
        engine.serve(A + B)


class Mover(UnifiedRadixCache):
    """Points a request's own unfinished page at another slot, in the row and the prefix."""

    def cache_unfinished_req(self, req, chunked=False, **kwargs):
        super().cache_unfinished_req(req, chunked, **kwargs)
        self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, len(req.prefix_indices) - 1] = 3
        req.prefix_indices[-1] = 3


class Misfiled(UnifiedRadixCache):
    """Leaves an honest prefix on the request but points the row the next pass reads elsewhere."""

    def cache_unfinished_req(self, req, chunked=False, **kwargs):
        super().cache_unfinished_req(req, chunked, **kwargs)
        self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, 0] = 3


@pytest.mark.parametrize("entry, cause", [
    (Mover, "moved KV slots a request computed itself"),
    (Misfiled, "left a request row that disagrees"),
])
def test_a_cache_that_moves_what_the_request_holds_is_refused(sglang, entry, cause):
    engine = _engine(sglang, entry)
    with pytest.raises(RuntimeError, match=cause):
        engine.serve(A[:21])


class Greedy(UnifiedRadixCache):
    """Claims the whole key, and one token more."""

    def match_prefix(self, params):
        return MatchResult(torch.arange(len(params.key) + 1, dtype=torch.int64))


class Listed(UnifiedRadixCache):
    """Answers a match with a list of slots."""

    def match_prefix(self, params):
        return [4, 5]


@pytest.mark.parametrize("entry, cause", [
    (Greedy, "claimed 22 cached tokens of a 21-token key"),
    (Listed, "returned a list, not a MatchResult"),
])
def test_a_claim_longer_than_its_key_or_not_a_match_is_refused_at_the_match(sglang, entry, cause):
    with pytest.raises(RuntimeError, match=cause):
        _engine(sglang, entry).admit(A)
    _refusal(sglang, cause.split(" ")[0])


def test_an_audited_request_waits_for_its_verdict_and_is_receipted(sglang, monkeypatch):
    monkeypatch.setitem(audit._state, "rate", 1.0)
    monkeypatch.setitem(audit._state, "rng", __import__("random").Random(0))
    monkeypatch.setattr(audit, "_stats", {})
    engine = _engine(sglang)
    engine.serve(A)
    engine.serve(A + B)
    (row,) = receipts.collect(sglang.receipts, "audit")
    assert (row["slot"], row["n"], row["violations"], row["mode"]) == (ADDRESS, 2, 0, "kv_content")


def test_a_cache_that_trades_the_engine_pools_away_is_refused_at_its_next_handoff(sglang):
    engine = _engine(sglang)
    engine.cache.token_to_kv_pool_allocator = Allocator(engine.allocator.pool)
    with pytest.raises(TypeError, match="must hold the engine's KV allocator"):
        engine.serve(A)

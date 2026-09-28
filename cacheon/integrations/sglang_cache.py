"""Serve the scheduler's prefix cache from a bundle and check the KV behind every prefix it serves.

The address ``tree_cache`` names the object SGLang's scheduler keeps under that
attribute: the cache that matches a request's leading tokens to KV slots already
written, takes finished and chunked requests in, locks, evicts and, with the RAM
tier on, moves KV between the device and host memory. A bundle naming the address
supplies ``entry(cache)``. SGLang first constructs its real cache with its own
components, pools and host tier, then the entry returns a subclass of its runtime
type. The validator binds those methods onto the existing object, preserving
the references its components and transfer workers already hold. The
contract is the runtime cache object's interface, not a particular cache class
or model. No engine flag differs between the arms.

What a cache alone can fake is a hit: a served prefix whose slots do not hold what
the engine computed for it skips that prefill and returns wrong tokens fast. The
check is on content, not on where the bytes travelled, so a RAM tier, stock's or
a bundle's own, passes whenever it brings back the exact bytes. Each time a
request is handed to the cache, before the cache sees it, the validator hashes
every complete page the request's own forward passes computed and records the
pair (digest of the prefix through that page, hash of the page's bytes). The
digest covers the request's namespace, its tokens and, under EAGLE, the token
after the page, which the draft KV reads. At the same handoff it hashes a sample
of the pages the request read from the cache and requires each pair to have been
recorded. The request's own slots must stay where they were between handoffs,
and a flush forgets every pair.

The check runs on the scheduler's stream behind the forward pass that wrote the
bytes and nothing on the host waits for it: each handoff reads the verdict an
earlier one published, and a flush or an audited request waits for it. A refusal
stops the engine and is receipted as the candidate's. Residual: a page is
checked after the forward pass that read it, so bytes a cache moves into a served
slot after that pass read it are not told apart from bytes placed before.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import os
import sys
from array import array
from dataclasses import dataclass
from typing import Any, NoReturn

from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor
from cacheon.registry import REGISTRY, KernelRegistry

ADDRESS = "tree_cache"
_MODULE = "sglang.srt.mem_cache.registry"
_STOCK = "_cacheon_stock_factory"
# (prime, base) pairs below 2**31, so every product of two residues fits in int64.
_MODULI = ((2147483647, 48271), (2147483629, 69621))
_LAYERS = 4  # layers hashed per KV buffer kind, drawn once per engine
_SAMPLE = 64  # served pages checked per handoff
_CHUNK = 64  # pages hashed per pass, which bounds the temporaries
_CELLS = 1 << 24  # one-byte cells of the pair table, two per pair
_AUDIT_MODE = "kv_content"
# The methods the scheduler hands requests through; the check runs inside them.
_HANDOFFS = frozenset({"match_prefix", "cache_unfinished_req", "cache_finished_req", "reset"})
# Verdict bits: the device raises them, a later handoff reads them.
_FAKE, _MOVED, _ROW = 1, 2, 4
_VERDICTS = {
    _FAKE: "served a page that does not hold the KV the engine computed for its prefix",
    _MOVED: "moved KV slots a request computed itself",
    _ROW: "left a request row that disagrees with the request's prefix",
}


def admit(tree: ast.Module, slot: str, entry: str, *, error: type[Exception]) -> None:
    """Check the factory declaration without importing the miner into intake."""

    if slot != ADDRESS:
        raise error(f"{slot!r}: the prefix cache {ADDRESS!r} has no sub-addresses")
    factory = next((node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == entry), None)
    if factory is None:
        raise error(f"{ADDRESS} entry {entry!r} must be a function accepting the runtime cache")
    args = factory.args
    positional = len(args.posonlyargs) + len(args.args)
    if (positional - len(args.defaults) > 1 or (positional < 1 and args.vararg is None)
            or any(default is None for default in args.kw_defaults)):
        raise error(f"{ADDRESS} entry {entry!r} must be callable with only the runtime cache")


def _refuse(message: str, *, error: type[Exception] = RuntimeError,
            phase: str = "entry", candidate: bool = True) -> NoReturn:
    """Receipt the refusal as the candidate's, then take the engine down with it."""

    failure = error(f"{ADDRESS}: {message}")
    if candidate:
        _receipts.failed(ADDRESS, failure, phase=phase)
    raise failure


def _ids(tokens):
    """A token row as a CPU int64 tensor, sharing the array's memory when it can."""

    import torch

    if isinstance(tokens, array) and tokens.typecode == "q" and len(tokens):
        return torch.frombuffer(tokens, dtype=torch.int64)
    return torch.tensor(list(tokens), dtype=torch.int64)


def _powers(base: int, prime: int, count: int):
    """``base**i % prime`` for ``i < count``, doubling so the loop runs log times."""

    import torch

    out = torch.ones(1, dtype=torch.int64)
    while len(out) < count:
        out = torch.cat((out, out * pow(base, len(out), prime) % prime))
    return out[:count]


@dataclass
class _Held:
    """What the validator knows about one request between its handoffs."""

    own: int  # positions from here on hold KV the request's forward passes computed
    recorded: int = 0  # positions before this are recorded or served
    until: int = 0  # the request's KV length at its last handoff
    kept: Any = None  # the request's own slots for [own, until) at its last handoff


class _Guard:
    """The validator's record of which prefix each KV page it saw computed holds."""

    def __init__(self, ctx, *, candidate: bool = True) -> None:
        import torch

        from cacheon.integrations.sglang_dsa_state import KV_BUFFERS

        params = ctx.params
        self.torch, self.candidate = torch, candidate
        self.allocator = params.token_to_kv_pool_allocator
        self.requests = params.req_to_token_pool
        self.page = int(params.page_size)
        self.bigram = int(bool(params.is_eagle))
        self.device = self.requests.req_to_token.device  # with its index, unlike a flag string
        cuda = self.device.type == "cuda"
        # The forward pass writes KV on its own stream; the check reads behind it.
        runner = getattr(ctx.tp_worker, "model_runner", None)
        self.forward = getattr(runner, "forward_stream", None) if cuda else None
        if cuda and self.forward is None:
            raise RuntimeError(f"{ADDRESS}: the scheduler's model runner has no forward stream")
        seed = int.from_bytes(os.urandom(8), "little") >> 1
        draws = torch.Generator().manual_seed(seed)
        self.draws = torch.Generator(device=self.device).manual_seed(seed // 3)
        self.layers = self._layers(KV_BUFFERS, params, draws)
        width = len(self._words(torch.arange(self.page, device=self.device)[None])[0])
        self.weights = torch.randint(1, _MODULI[0][0], (width,), generator=draws).to(self.device)
        self.mix = torch.randint(1, _MODULI[1][0], (len(_MODULI), 2), generator=draws).tolist()
        count = self.requests.req_to_token.shape[1] + 2
        self.powers = [_powers(b, p, count).to(self.device) for p, b in _MODULI]
        self.inverses = [_powers(pow(b, -1, p), p, count).to(self.device) for p, b in _MODULI]
        self.table = torch.zeros(_CELLS, dtype=torch.bool, device=self.device)
        self.bad = torch.zeros((), dtype=torch.int32, device=self.device)
        self.flag = torch.zeros((), dtype=torch.int32, pin_memory=cuda)
        self.event = torch.cuda.Event() if cuda else None
        self.namespaces: dict[tuple, int] = {}
        self.held: dict[object, _Held] = {}

    def refuse(self, message, **kwargs) -> NoReturn:
        _refuse(message, candidate=self.candidate, **kwargs)

    def invoke(self, function, *args, **kwargs):
        """Keep stock failures outside candidate attribution while applying the same checks."""
        if self.candidate:
            return _receipts.invoke(ADDRESS, function, *args, kwargs=kwargs)
        return function(*args, **kwargs)

    def _layers(self, names: tuple[str, ...], params, draws) -> list[tuple[Any, bool]]:
        """Draw the layers hashed, per KV buffer kind of the target and draft pools.

        Token-addressed buffers hold one row per slot; the DSA indexer's hold one
        row per page. A pool with no recognized buffer is the arena's configuration.
        """

        torch = self.torch
        pools = (self.allocator.get_kvcache(), *params.mtp_draft_device_pools)
        kinds = [(getattr(pool, name, None), False) for pool in pools for name in names]
        kinds += [(getattr(pool, "index_k_with_scale_buffer", None), True) for pool in pools]
        chosen = []
        for layers, by_page in kinds:
            layers = [b for b in (layers or ()) if torch.is_tensor(b) and b.shape[0]]
            picks = torch.randperm(len(layers), generator=draws)[:_LAYERS].tolist()
            chosen += [(layers[i], by_page) for i in sorted(picks)]
        if not any(not by_page for _, by_page in chosen):
            raise RuntimeError(f"{ADDRESS}: no KV buffer recognized on {type(pools[0]).__name__}")
        return chosen

    def _words(self, slots):
        """The hashed bytes of each page of ``slots`` (pages x page size), as int32 rows."""

        torch = self.torch
        parts = []
        for buffer, by_page in self.layers:
            index = slots[:, 0] // self.page if by_page else slots.reshape(-1)
            rows = buffer.index_select(0, index).reshape(len(slots), -1)
            rows = rows if rows.dtype == torch.uint8 else rows.view(torch.uint8)
            parts.append(rows.view(torch.int32) if rows.shape[1] % 4 == 0 else rows.to(torch.int32))
        return torch.cat(parts, dim=1)

    def _content(self, slots):
        prime = _MODULI[0][0]
        words = self._words(slots).to(self.torch.int64)
        return words.mul_(self.weights).remainder_(prime).sum(dim=1).remainder_(prime)

    def _digests(self, ids, namespace, lengths):
        """Digest of the namespace and the first ``lengths[i]`` tokens, for every ``i``."""

        torch = self.torch
        seed = self.namespaces.setdefault(namespace, len(self.namespaces))
        # The copy into ``host`` is what gets checked: later edits to ``ids`` do not reach it.
        host = torch.empty(len(ids) + 1, dtype=torch.int64, pin_memory=self.event is not None)
        host[0] = seed
        host[1:] = _ids(ids)
        # Shifted by one so no element is zero: a leading zero leaves a polynomial
        # digest unchanged, and [0, 0, a] would collide with [0, a].
        tokens = host.to(self.device, non_blocking=True) + 1
        out = torch.zeros_like(lengths)
        for (prime, _), powers, inverses in zip(_MODULI, self.powers, self.inverses):
            # sum(t_j * base**(k - j)) for a prefix k: scale by the inverse powers,
            # accumulate, scale back.
            sums = torch.cumsum(tokens % prime * inverses[: len(tokens)] % prime, 0) % prime
            out = out * (1 << 31) + sums[lengths] * powers[lengths] % prime
        return out

    def _cells(self, digests, content) -> list:
        return [
            (digests % prime * a % prime + content * b % prime) % prime % _CELLS
            for (prime, _), (a, b) in zip(_MODULI, self.mix)
        ]

    def _flag(self, bit: int, mismatch) -> None:
        self.bad.bitwise_or_(mismatch.any().to(self.torch.int32) * bit)

    def publish(self) -> None:
        """Copy the verdict to the host behind the checks queued so far."""

        if self.event is not None:
            self.flag.copy_(self.bad, non_blocking=True)
            self.event.record()

    def poll(self, block: bool) -> None:
        """Refuse on a published verdict; without ``block`` only on one already copied."""

        if self.event is None:
            code = int(self.bad)
        elif block or self.event.query():
            self.event.synchronize()
            code = int(self.flag)
        else:
            return
        if code:
            self.refuse("the cache " + "; ".join(t for bit, t in _VERDICTS.items() if code & bit))

    def owned(self, cache, phase: str = "entry") -> None:
        """Refuse a cache that no longer holds the engine's pools."""

        if getattr(cache, "token_to_kv_pool_allocator", None) is not self.allocator or (
            getattr(cache, "req_to_token_pool", None) is not self.requests
        ):
            self.refuse("the cache must hold the engine's KV allocator and request pool",
                    error=TypeError, phase=phase)

    def matched(self, cache, key, result, kind: type) -> None:
        """Refuse a match that is malformed or claims more tokens than its key allows."""

        self.owned(cache)
        if not isinstance(result, kind) or not isinstance(result.device_indices, self.torch.Tensor):
            self.refuse(f"match_prefix returned a {type(result).__name__}, not a MatchResult of slots")
        limit = len(key) - (0 if getattr(key, "is_bigram", False) else self.bigram)
        claimed = len(result.device_indices) + result.host_hit_length
        if claimed > limit:
            self.refuse(f"it claimed {claimed} cached tokens of a {limit}-token key")

    def handoff(self, cache, req, ids):
        """Check what a request read from the cache and record what its forward passes computed.

        Runs before the cache sees the request, so nothing the cache does at this
        handoff changes what is checked. Returns the request's row as handed over,
        or None when the request holds none.
        """

        torch = self.torch
        self.owned(cache)
        self.poll(block=False)
        if req.kv.req_pool_idx is None:
            return None
        n = len(ids)
        held = self.held.get(req.rid)
        if held is None:
            served = len(req.prefix_indices)
            if served % self.page:
                self.refuse(f"it served a prefix of {served} tokens, which ends inside a page")
            held = self.held[req.rid] = _Held(own=min(served, n))
        if self.forward is not None:
            torch.cuda.current_stream().wait_stream(self.forward)
        row = self.requests.req_to_token[req.kv.req_pool_idx, :n].to(torch.int64)
        if held.kept is not None:
            # A finished request may hand over fewer tokens than it last held.
            last = min(held.until, n)
            self._flag(_MOVED, row[held.own : last] != held.kept[: max(0, last - held.own)])
        served = min(held.own, n - self.bigram) // self.page
        start, end = max(held.own, held.recorded) // self.page, (n - self.bigram) // self.page
        if served or end > start:
            check = (
                torch.arange(served, device=self.device) if served <= _SAMPLE
                else torch.randint(served, (_SAMPLE,), device=self.device, generator=self.draws)
            )
            pages = torch.cat((check, torch.arange(start, max(start, end), device=self.device)))
            offsets = torch.arange(self.page, device=self.device)
            slots = row[(pages * self.page)[:, None] + offsets]
            content = torch.cat([self._content(part) for part in slots.split(_CHUNK)])
            lengths = (pages + 1) * self.page + self.bigram
            cells = self._cells(self._digests(ids, (req.extra_key, req.cache_salt or None), lengths),
                                content)
            k = len(check)
            self._flag(_FAKE, ~(self.table[cells[0][:k]] & self.table[cells[1][:k]]))
            for cell in cells:
                self.table[cell[k:]] = True
        held.recorded = max(held.recorded, end * self.page)
        self.publish()
        return row

    def settle(self, req, row) -> None:
        """Hold the cache to the prefix it left on an unfinished request, which the next pass reads.

        It may point the part it now protects at slots of its own; the rest must be
        the request's own slots, unmoved, and the row must agree with the prefix.
        """

        if row is None:
            return
        torch = self.torch
        n, held = len(row), self.held[req.rid]
        prefix, protected = req.prefix_indices, req.kv.cache_protected_len
        if not isinstance(prefix, torch.Tensor) or prefix.device != self.device or len(prefix) != n:
            self.refuse(f"it left a prefix that is not {n} slots on the device on a request with {n} tokens")
        if not 0 <= protected <= n or protected % self.page:
            self.refuse(f"it protected {protected} of a request's {n} tokens, not whole pages of them")
        after = self.requests.req_to_token[req.kv.req_pool_idx, :n].to(torch.int64)
        self._flag(_ROW, after != prefix)
        self._flag(_MOVED, after[protected:] != row[protected:])
        held.own = max(held.own, protected)
        held.until, held.kept = n, after[held.own:].clone()
        self.publish()

    def finished(self, req) -> None:
        """Forget the request; on an audited one, wait for its verdict and receipt it."""

        from cacheon import audit

        self.held.pop(req.rid, None)
        if self.candidate:
            _receipts.completed(ADDRESS)
        if self.candidate and audit.sampled():
            self.poll(block=True)
            audit.record_fraction(ADDRESS, 1.0, 1.0, _AUDIT_MODE)

    def reset(self) -> None:
        """Read every pending verdict, then forget every pair: a flush invalidates the pools."""

        self.publish()
        self.poll(block=True)
        self.table.zero_()
        self.held.clear()


class _Pinned:
    """A handoff method no instance attribute can shadow, since the check runs inside it."""

    def __init__(self, function, guard) -> None:
        self.function, self.guard = function, guard

    def __get__(self, obj, owner=None):
        return self.function if obj is None else self.function.__get__(obj, owner)

    def __set__(self, obj, value) -> NoReturn:
        self.guard.refuse(f"it replaced {self.function.__name__} on the instance, which skips the check")


def _receipted(function, guard):
    """Run a method the bundle defines so that its raise is receipted as the candidate's."""

    @functools.wraps(function)
    def call(self, *args, **kwargs):
        return guard.invoke(function, self, *args, **kwargs)

    return call


def _guarded(cls: type, base: type, guard: _Guard) -> type:
    """Subclass the candidate so every request it is handed passes through the guard.

    The scheduler also calls the cache for eviction, locks and host loads, so every
    method the bundle defines runs receipted: a raise there is the candidate's.
    """

    from sglang.srt.mem_cache.base_prefix_cache import MatchResult

    def pinned(function):
        return _Pinned(function, guard)

    class Sealed(type(cls)):
        """Keeps a handoff from being swapped on the class after it is built."""

        def __setattr__(klass, name, value):
            if name in _HANDOFFS:
                guard.refuse(f"it replaced {name} on its class, which skips the check")
            super().__setattr__(name, value)

        def __delattr__(klass, name):
            if name in _HANDOFFS:
                guard.refuse(f"it replaced {name} on its class, which skips the check")
            super().__delattr__(name)

    class Guarded(cls, metaclass=Sealed):
        """The candidate's class, with the validator at each handoff."""

        @pinned
        def match_prefix(self, params):
            result = guard.invoke(super().match_prefix, params)
            guard.matched(self, params.key, result, MatchResult)
            return result

        @pinned
        def cache_unfinished_req(self, req, *args, **kwargs):
            row = guard.handoff(self, req, req.get_fill_ids())
            guard.invoke(super().cache_unfinished_req, req, *args, **kwargs)
            guard.settle(req, row)

        @pinned
        def cache_finished_req(self, req, *args, **kwargs):
            tokens = req.origin_input_ids + req.output_ids
            guard.handoff(self, req, tokens[: kwargs["kv_len_to_handle"]])
            guard.invoke(super().cache_finished_req, req, *args, **kwargs)
            guard.finished(req)

        @pinned
        def reset(self):
            guard.reset()
            return guard.invoke(super().reset)

    for klass in reversed(cls.__mro__[: cls.__mro__.index(base)]):  # the most derived wins
        for name, value in vars(klass).items():
            if inspect.isfunction(value) and not name.startswith("__") and name not in _HANDOFFS:
                setattr(Guarded, name, _receipted(value, guard))
    Guarded.__name__, Guarded.__qualname__ = cls.__name__, cls.__qualname__
    Guarded.__module__ = cls.__module__
    return Guarded


def _bind(impl, ctx, stock):
    """Pass the runtime-built object to the candidate and guard its handoffs."""

    if ctx.is_hybrid_swa or ctx.is_hybrid_ssm:
        if impl is None:
            return stock(ctx)
        # Missing validator coverage is infrastructure, not a failed miner cache.
        raise RuntimeError(f"{ADDRESS}: state validation is unavailable for sliding-window or recurrent caches")
    base = importlib.import_module("sglang.srt.mem_cache.base_prefix_cache").BasePrefixCache
    if impl is not None and inspect.isclass(impl.entry):
        _refuse("entry must accept the runtime cache, not its construction parameters",
                error=TypeError, phase="prepare")
    if impl is not None and impl.prepare is not None:
        _refuse("a cache entry takes no prepare", error=TypeError, phase="prepare")
    cache = stock(ctx)
    if not isinstance(cache, base):
        if impl is None:
            return cache
        _refuse(f"this engine builds {type(cache).__name__}, not a BasePrefixCache",
                phase="prepare")
    guard = _Guard(ctx, candidate=impl is not None)
    if impl is not None:
        cls = _receipts.invoke(ADDRESS, impl.entry, cache, phase="prepare")
        if not inspect.isclass(cls) or not issubclass(cls, type(cache)):
            _refuse("entry must return a subclass of the runtime cache's type",
                    error=TypeError, phase="prepare")
        if inspect.isabstract(cls):
            _refuse(f"entry leaves {sorted(cls.__abstractmethods__)} abstract",
                    error=TypeError, phase="prepare")
    else:
        cls = type(cache)
    guard.owned(cache, phase="prepare")
    guarded = _guarded(cls, base, guard)
    try:
        cache.__class__ = guarded
    except TypeError as exc:
        guard.refuse(f"cache object cannot bind its guarded methods: {exc}", phase="prepare")
    for name in _HANDOFFS:
        if name in vars(cache):
            guard.refuse(f"it replaced {name} on the instance, which skips the check", phase="prepare")
    return cache


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Bind the cache after the selected backend and runtime wrappers have initialized it."""

    module = sys.modules.get(_MODULE)
    stock = getattr(module, "create_tree_cache", None)
    if stock is None or hasattr(stock, _STOCK):
        return

    @functools.wraps(stock)
    def create_tree_cache(ctx):
        impl = registry.select(ADDRESS, CallDescriptor()).impl
        return _bind(impl, ctx, stock)

    setattr(create_tree_cache, _STOCK, stock)
    module.create_tree_cache = create_tree_cache

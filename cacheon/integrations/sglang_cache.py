"""Serve the scheduler's prefix cache from a bundle and check every prefix it claims.

The address ``tree_cache`` names the object SGLang's scheduler keeps under that
attribute: the ``BasePrefixCache`` that matches a request's leading tokens to KV
slots already written, takes finished and chunked requests in, locks and evicts.
A bundle naming the address supplies a ``BasePrefixCache`` subclass as its entry.
The engine builds it where it builds its own, through the built-in selection
chain ``sglang.srt.mem_cache.registry.default_radix_cache_factory``, so
``create_tree_cache`` applies its own checks and wrappers to the candidate exactly
as to stock, and no engine flag differs between the arms. The class is
constructed as ``entry(params)`` from the engine's ``CacheInitParams`` and must
keep the KV allocator and request-to-token pool it was handed: the scheduler
evicts and allocates through the cache's allocator attribute, so a cache holding
any other pool would take KV ownership from the validator.

What a cache alone can fake is a hit. A claimed prefix the KV slots do not hold
lets the scheduler skip prefilling it, and garbage returns fast. Every claim, the
candidate's internal ones included, is therefore checked against a ledger the
candidate never writes. When a request's row is handed to the cache, the ledger
records for each of its KV slots a digest of the whole prefix that slot was
computed from: the request's namespace, the tokens up to and including its own,
and under EAGLE the next token too, which the draft KV depends on. A record ends
when the allocator hands its slot out again or the engine flushes its pools. A
claim passes only if every claimed slot's live record digests the request's own
prefix at that position. Whole-prefix digests rather than per-slot tokens,
because two requests that computed the same prefix hold equivalent KV in
different slots, and slots stitched from different contexts must not pass.
"""

from __future__ import annotations

import ast
import functools
import inspect
import sys
from array import array
from typing import NoReturn

from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor
from cacheon.registry import REGISTRY, KernelRegistry

ADDRESS = "tree_cache"
_MODULE = "sglang.srt.mem_cache.registry"
_PACKAGE = "sglang.srt.mem_cache"
_STOCK = "_cacheon_stock_factory"
# (prime, base) pairs below 2**31, so every product of two residues fits in int64
# and the two residues pack into one 62-bit digest.
_MODULI = ((2147483647, 48271), (2147483629, 69621))
_ALLOCATIONS = ("alloc", "alloc_extend", "alloc_decode")


def admit(tree: ast.Module, slot: str, entry: str, *, error: type[Exception]) -> None:
    """Refuse an entry its source does not define as a class derived from SGLang's cache package.

    Intake runs this on the parsed source of every op under the cache's address, so
    no bundle code executes. The address has no children. Only that file is read:
    the entry class, or a class of that file it derives from, must name a top-level
    import from ``sglang.srt.mem_cache`` among its bases. The subclass relation
    itself is checked again in the scheduler, where the class exists.
    """

    if slot != ADDRESS:
        raise error(f"{slot!r}: the prefix cache {ADDRESS!r} has no sub-addresses")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and not node.level and _in_package(node.module or ""):
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.asname for a in node.names if a.asname and _in_package(a.name))

    def derives(name: str, seen: frozenset[str]) -> bool:
        node = classes.get(name)
        return node is not None and name not in seen and any(
            text.startswith(_PACKAGE + ".") or text.split(".")[0] in imported
            or derives(text, seen | {name})
            for text in map(ast.unparse, node.bases)
        )

    if entry not in classes:
        raise error(f"{ADDRESS} entry {entry!r} is not a class defined in its source")
    if not derives(entry, frozenset()):
        raise error(f"{ADDRESS} entry {entry!r} does not derive from a class of {_PACKAGE}")


def _in_package(module: str) -> bool:
    return module == _PACKAGE or module.startswith(_PACKAGE + ".")


def _refuse(message: str, *, error: type[Exception] = RuntimeError,
            phase: str = "entry") -> NoReturn:
    """Receipt the refusal as the candidate's, then take the engine down with it."""

    failure = error(f"{ADDRESS}: {message}")
    _receipts.failed(ADDRESS, failure, phase=phase)
    raise failure


def _ids(tokens):
    """A private int64 copy of a token row, so the candidate cannot edit what is checked."""

    import torch

    if isinstance(tokens, array) and tokens.typecode == "q" and len(tokens):
        return torch.frombuffer(tokens, dtype=torch.int64).clone()
    return torch.tensor(list(tokens), dtype=torch.int64)


class _Ledger:
    """Which prefix each KV slot holds, kept by the validator beside the candidate cache."""

    def __init__(self, params) -> None:
        import torch

        self.torch = torch
        self.allocator = params.token_to_kv_pool_allocator
        self.requests = params.req_to_token_pool
        self.bigram = int(bool(params.is_eagle))
        size = self.allocator.size + self.allocator.page_size
        device = self.allocator.device
        # A slot's digest counts while ``written`` equals ``life``. Handing the slot
        # out for new KV advances its ``life``; a flush advances every slot's. These
        # 16 bytes a slot sit on the device beside the pool, on the candidate arm only.
        self.digest = torch.zeros(size, dtype=torch.int64, device=device)
        self.written = torch.full((size,), -1, dtype=torch.int32, device=device)
        self.life = torch.zeros(size, dtype=torch.int32, device=device)
        # Set when a row handed to the cache pointed at another prefix's KV. It is
        # read with the next claim, the only way such a row could be put to use.
        self.crossed = torch.zeros((), dtype=torch.bool, device=device)
        self.powers = [torch.ones(1, dtype=torch.int64, device=device) for _ in _MODULI]
        self.inverses = [torch.ones(1, dtype=torch.int64, device=device) for _ in _MODULI]
        self.namespaces: dict[tuple, int] = {}
        for name in (*_ALLOCATIONS, "clear"):
            setattr(self.allocator, name, self._bumping(name, getattr(self.allocator, name)))

    def _bumping(self, name: str, method):
        @functools.wraps(method)
        def bumped(*args, **kwargs):
            out = method(*args, **kwargs)
            if name == "clear":
                self.life += 1
            elif out is not None:
                self.life[out] += 1
            return out

        return bumped

    def keys(self, ids, extra_key, cache_salt):
        """The digest of the prefix each KV slot of a row of ``ids`` holds."""

        torch = self.torch
        seed = self.namespaces.setdefault((extra_key, cache_salt or None), len(self.namespaces))
        # Shifted by one so no element is zero: a leading zero leaves a polynomial
        # digest unchanged, and [0, 0, a] would collide with [0, a].
        tokens = torch.cat((torch.tensor([seed]), ids)).to(self.life.device) + 1
        n = len(tokens)
        for i, (prime, base) in enumerate(_MODULI):
            while len(self.powers[i]) < n:
                m = len(self.powers[i])
                self.powers[i] = torch.cat((self.powers[i], self.powers[i] * pow(base, m, prime) % prime))
                self.inverses[i] = torch.cat(
                    (self.inverses[i], self.inverses[i] * pow(base, -m, prime) % prime)
                )
        digest = torch.zeros(n, dtype=torch.int64, device=tokens.device)
        for (prime, _), powers, inverses in zip(_MODULI, self.powers, self.inverses):
            # sum(t_j * base**(k - j)) for every prefix k at once: scale by the
            # inverse powers, accumulate, scale back.
            total = torch.cumsum(tokens % prime * inverses[:n] % prime, 0) % prime
            digest = digest * (1 << 31) + total * powers[:n] % prime
        # Entry k digests the namespace and ids[:k]. Slot i holds prefix i + 1, or
        # i + 2 under EAGLE, whose draft KV at i also reads token i + 1.
        return digest[1 + self.bigram:]

    def record(self, req, ids):
        """Record what the request's row holds before the candidate sees it; return the row."""

        row = self.requests.req_to_token[req.kv.req_pool_idx, : len(ids)].to(
            device=self.life.device, dtype=self.torch.int64, copy=True
        )
        keys = self.keys(ids, req.extra_key, req.cache_salt)
        slots = row[: len(keys)]
        live = self.written[slots] == self.life[slots]
        self.crossed |= (live & (self.digest[slots] != keys)).any()
        self.digest[slots] = keys
        self.written[slots] = self.life[slots]
        return row

    def owned(self, cache, phase: str = "entry") -> None:
        """Refuse a cache that no longer holds the engine's pools."""

        if getattr(cache, "token_to_kv_pool_allocator", None) is not self.allocator or (
            getattr(cache, "req_to_token_pool", None) is not self.requests
        ):
            _refuse("the cache must hold the engine's KV allocator and request pool",
                    error=TypeError, phase=phase)

    def checked(self, cache, ids, namespace: tuple, claimed, req=None, row=None):
        """The claim the scheduler will use, once each claimed slot holds the request's prefix.

        With ``row``, the request's row as handed to the cache, its own slots hold
        its own prefix (a last EAGLE slot's next token is not known yet), and the
        row the next forward reads must agree with the claim.
        """

        torch = self.torch
        self.owned(cache)
        if not isinstance(claimed, torch.Tensor):
            _refuse(f"claims a {type(claimed).__name__}, not a tensor of KV slots")
        n = len(claimed)
        if not n:
            return claimed
        limit = len(row) if row is not None else len(ids) - self.bigram
        if n > limit:
            _refuse(f"claims {n} cached tokens of a {limit}-token request")
        keys = self.keys(ids[: n + self.bigram], *namespace)
        slots = claimed.to(device=self.life.device, dtype=torch.int64, copy=True)
        safe = slots.clamp(0, len(self.life) - 1)[: len(keys)]
        held = torch.zeros(n, dtype=torch.bool, device=slots.device)
        held[: len(keys)] = (
            (slots[: len(keys)] == safe) & (self.written[safe] == self.life[safe])
            & (self.digest[safe] == keys)
        )
        if row is not None:
            held |= slots == row[:n]
            held &= self.requests.req_to_token[req.kv.req_pool_idx, :n] == slots
        bad = torch.cat((self.crossed.view(1), ~held)).nonzero()
        if len(bad):
            at = int(bad[0]) - 1
            _refuse(
                "a request row handed to the cache pointed at another prefix's KV" if at < 0
                else f"the slot claimed at position {at} does not hold the request's prefix"
            )
        # Hand the scheduler the tensor that was checked, not one the candidate keeps.
        return slots


def _guarded(cls: type, ledger: _Ledger) -> type:
    """Subclass the candidate so every claim it makes, internal ones included, is checked."""

    from sglang.srt.mem_cache.base_prefix_cache import MatchResult

    class Guarded(cls):
        """The candidate's class, handing the scheduler only claims that were checked."""

        def match_prefix(self, params):
            key = params.key
            ids = _ids(key.raw_token_ids())
            namespace = (key.extra_key, key.cache_salt)
            result = _receipts.invoke(ADDRESS, super().match_prefix, params)
            if not isinstance(result, MatchResult):
                _refuse(f"match_prefix returned a {type(result).__name__}, not a MatchResult")
            if result.host_hit_length or result.swa_host_hit_length or result.mamba_host_hit_length:
                _refuse("a host-tier hit has no validator record")
            claimed = ledger.checked(self, ids, namespace, result.device_indices)
            _receipts.completed(ADDRESS)
            return result._replace(device_indices=claimed)

        def cache_finished_req(self, req, is_insert=True, **kwargs):
            tokens = (req.origin_input_ids + req.output_ids)[: kwargs["kv_len_to_handle"]]
            ledger.record(req, _ids(tokens))
            return _receipts.invoke(ADDRESS, super().cache_finished_req, req, is_insert, kwargs=kwargs)

        def cache_unfinished_req(self, req, **kwargs):
            ids = _ids(req.get_fill_ids())
            row = ledger.record(req, ids)
            result = _receipts.invoke(ADDRESS, super().cache_unfinished_req, req, kwargs=kwargs)
            req.prefix_indices = ledger.checked(
                self, ids, (req.extra_key, req.cache_salt), req.prefix_indices, req=req, row=row
            )
            return result

    Guarded.__name__, Guarded.__qualname__ = cls.__name__, cls.__qualname__
    Guarded.__module__ = cls.__module__
    return Guarded


def _bind(impl, ctx):
    """Build the candidate cache on the engine's pools, guarded, or raise the reason it cannot be."""

    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

    if ctx.is_hybrid_swa or ctx.is_hybrid_ssm or ctx.enable_hierarchical_cache:
        # Their second pools and host tier hold state the ledger does not see. The
        # arena chose them, so this is not receipted as the candidate's failure.
        raise RuntimeError(f"{ADDRESS}: the seam serves full-attention device caches only")
    cls = impl.entry
    if not isinstance(cls, type) or not issubclass(cls, BasePrefixCache):
        _refuse(f"entry {cls!r} is not a BasePrefixCache subclass", error=TypeError, phase="prepare")
    if impl.prepare is not None:
        _refuse("a cache entry takes no prepare", error=TypeError, phase="prepare")
    if inspect.isabstract(cls):
        _refuse(f"entry {cls.__name__} leaves {sorted(cls.__abstractmethods__)} abstract",
                error=TypeError, phase="prepare")
    ledger = _Ledger(ctx.params)
    cache = _receipts.invoke(ADDRESS, _guarded(cls, ledger), ctx.params, phase="prepare")
    ledger.owned(cache, phase="prepare")
    return cache


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Wrap the built-in cache choice once its module is imported."""

    module = sys.modules.get(_MODULE)
    stock = getattr(module, "default_radix_cache_factory", None)
    if stock is None or hasattr(stock, _STOCK):
        return

    @functools.wraps(stock)
    def default_radix_cache_factory(ctx):
        impl = registry.select(ADDRESS, CallDescriptor()).impl
        return stock(ctx) if impl is None else _bind(impl, ctx)

    setattr(default_radix_cache_factory, _STOCK, stock)
    module.default_radix_cache_factory = default_radix_cache_factory

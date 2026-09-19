"""Serve any named node of the served model through one audited call body.

A slot name that ``cacheon.slots`` does not define is a node address: a dotted name
from ``named_modules()`` of the served model, where ``*`` stands for exactly one
segment (``model.layers.*.mlp``). The candidate is a drop-in for that node's stock
``forward``: ``entry(prepared, *args, **kwargs)`` receives the stock arguments and
returns what stock returns. ``prepare(module)`` runs once per bound node; a bundle
without one receives the module itself. The same body therefore serves one
activation, a fused MoE block, a decoder layer or the whole decoder stack, and a
bundle that lists several addresses replaces several nodes at once.

Truth is the stock node in the running engine, not hand-written reference math. On
an audited eager call stock answers first; its outputs and the engine-state rows
this batch may write are kept; the arguments and the state are put back; the
candidate runs on the same call and its outputs and state rows are graded.
Without the restore every call of a node that carries recurrent state false-fails,
and a 1.5x-wrong stack left greedy tokens unchanged, so the end-to-end gate cannot
replace this check (H100 Qwen runs, 2026-09-19).

Binding happens once, right after ``ModelRunner.load_model``: late enough that the
weights exist, early enough that the breakable prefill runner and the decode graph
runner both capture the bound forward at every width, whole stack included (same
runs). While a candidate runs, nested nodes serve stock, so a wide candidate may
call the stock children it does not replace.
"""

from __future__ import annotations

import re
import sys
from typing import Callable

import torch

from cacheon import audit as _audit
from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor
from cacheon.dispatch import _arch_tag, _dtype_name, _dynamo_compiling, _in_cuda_graph
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.slots import SLOTS

_RUNNER = "sglang.srt.model_executor.model_runner"
_STOCK_LOAD = "_cacheon_stock_load_model"
# The pinned engine's per-token cache buffers: the MHA pair or the MLA latent.
_KV_BUFFERS = ("k_buffer", "v_buffer", "kv_buffer")


def node_pattern(slot: str) -> re.Pattern[str]:
    """Compile a node address; ``*`` matches one whole dotted segment."""

    return re.compile(
        r"\.".join("[^.]+" if part == "*" else re.escape(part) for part in slot.split("."))
        + r"\Z"
    )


def _tensors(value: object, found: list[torch.Tensor], *, fields: bool = False) -> list[torch.Tensor]:
    """Flatten the tensors of a call value; ``fields`` also walks object attributes.

    A node's result may be a record rather than a tuple (the logits processor returns
    one), and an unwalked record would leave the output ungraded. Arguments are not
    walked that way: the engine's batch object carries dozens of tensors the node
    does not own.
    """

    if torch.is_tensor(value):
        found.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _tensors(item, found, fields=fields)
    elif isinstance(value, dict):
        for item in value.values():
            _tensors(item, found, fields=fields)
    elif fields and hasattr(value, "__dict__"):
        for item in vars(value).values():
            _tensors(item, found, fields=fields)
    return found


def _state_rows(runner, call: tuple) -> list[tuple[torch.Tensor, int, torch.Tensor]]:
    """Engine-state rows this call may write, as ``(buffer, dim, index)``.

    Only a call that carries the engine's batch can reach the cache pools: cache
    rows at ``out_cache_loc`` and, on hybrid models, the recurrent-state rows of
    the batch's requests. An unrecognized cache layout raises rather than leaving
    written state unchecked.
    """

    batch = next(
        (v for v in call if hasattr(v, "out_cache_loc") and hasattr(v, "req_pool_indices")),
        None,
    )
    if batch is None:
        return []
    rows: list[tuple[torch.Tensor, int, torch.Tensor]] = []
    if batch.out_cache_loc is not None:
        pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
        buffers = [
            buffer
            for name in _KV_BUFFERS
            for buffer in (getattr(pool, name, None) or ())
            if torch.is_tensor(buffer)
        ]
        if not buffers:
            raise RuntimeError(f"no cache buffer recognized on {type(pool).__name__}")
        rows.extend((buffer, 0, batch.out_cache_loc.long()) for buffer in buffers)
    requests = runner.req_to_token_pool
    recurrent = getattr(requests, "mamba_pool", None)
    if recurrent is not None:
        index = requests.get_mamba_indices(batch.req_pool_indices).long()
        cache = recurrent.mamba_cache
        rows.extend((buffer, 1, index) for buffer in (*cache.conv, cache.temporal))
    return rows


def _stock_answer(slot: str, stock: Callable, runner, args: tuple, kwargs: dict):
    """Run stock on the live call, keep its outputs and state rows, then put the call back.

    What stock leaves in its own arguments is not part of the answer. The fused norm
    overwrites its arguments and returns them, so an honest norm that returns fresh
    tensors failed 1,521 of 1,560 calls while that was graded, and a whole decoder
    layer leaves normed intermediates in its dead input (H100 Qwen runs, 2026-09-19).
    Values reach the caller through the result and the engine state; both are graded.
    """

    handed = _tensors((args, kwargs), [])
    rows = _state_rows(runner, (*args, *kwargs.values()))
    before = [t.clone() for t in handed]
    before += [buffer.index_select(dim, index) for buffer, dim, index in rows]
    outputs = _audit.capture_reference(
        slot, lambda: _tensors(stock(*args, **kwargs), [], fields=True)
    )
    written = [buffer.index_select(dim, index) for buffer, dim, index in rows]
    # Write back only what stock changed: an untouched input may be an expanded
    # or otherwise unwritable view.
    for tensor, saved in zip(handed, before):
        if not torch.equal(tensor, saved):
            tensor.copy_(saved)
    for (buffer, dim, index), saved, after in zip(rows, before[len(handed):], written):
        if not torch.equal(saved, after):
            buffer.index_copy_(dim, index, saved)
    return None if outputs is None else [*outputs, *written]


def _descriptor(module, args: tuple, kwargs: dict, in_graph: bool) -> CallDescriptor | None:
    handed = _tensors((args, kwargs), [])
    if not handed or not handed[0].dim():
        return None
    # Rotary positions can lead the call as (3, M); the activations carry M.
    floating = next((t for t in handed if t.is_floating_point()), handed[0])
    weight = next(module.parameters(), floating)
    return CallDescriptor.from_legacy(
        dtype_name=_dtype_name(weight.dtype),
        last_dim=int(floating.shape[-1]),
        arch=_arch_tag(floating.device.index or 0) if floating.is_cuda else None,
        num_tokens=int(floating.shape[0]),
    ).with_updates(graph_mode="cuda_graph" if in_graph else "eager")


def make_node_dispatcher(
    slot: str,
    module,
    stock: Callable[..., object],
    runner,
    *,
    registry: KernelRegistry = REGISTRY,
) -> Callable[..., object]:
    """Build the replacement ``forward`` for one bound node."""

    prepared: dict[int, object] = {}

    def dispatched(*args, **kwargs):
        if _dynamo_compiling() or _receipts.is_invoking() or not registry.active:
            return stock(*args, **kwargs)
        in_graph = _in_cuda_graph()
        descriptor = _descriptor(module, args, kwargs, in_graph)
        impl = registry.select(slot, descriptor).impl if descriptor is not None else None
        if impl is None:
            return stock(*args, **kwargs)
        if id(impl) not in prepared:
            prepared[id(impl)] = (
                _receipts.invoke(slot, impl.prepare, module, phase="prepare")
                if impl.prepare is not None
                else module
            )
        expected = (
            _stock_answer(slot, stock, runner, args, kwargs)
            if not in_graph and _audit.sampled()
            else None
        )
        result = _receipts.invoke(
            slot, impl.entry, prepared[id(impl)], *args, kwargs=kwargs
        )
        if expected is not None:
            actual = _tensors(result, [], fields=True)
            actual += [b.index_select(d, i) for b, d, i in _state_rows(runner, (*args, *kwargs.values()))]
            if len(actual) != len(expected):
                error = RuntimeError(
                    f"candidate for {slot} returned a different tensor structure than stock"
                )
                _receipts.failed(slot, error, entry=impl.entry)
                raise error
            _audit.record(slot, actual, expected, scaled=True)
        _receipts.completed(slot)
        return result

    return dispatched


def bind(runner, registry: KernelRegistry = REGISTRY) -> list[str]:
    """Bind every registered node slot to the served model; return the bound names."""

    named = dict(runner.model.named_modules())
    bound: dict[str, str] = {}
    for slot in sorted(s for s in registry.slots() if s not in SLOTS):
        pattern = node_pattern(slot)
        names = [name for name in named if pattern.match(name)]
        error = None
        if not names:
            error = f"node slot {slot!r} names no module of the served model"
        for name in names:
            # A claimed node inside another claimed node would run as the candidate
            # while the outer node's stock reference is taken, and never otherwise.
            clash = next(
                (o for o in bound if o == name or o.startswith(name + ".") or name.startswith(o + ".")),
                None,
            )
            if clash is not None:
                error = (
                    f"node {name!r} of {slot!r} overlaps node {clash!r} of {bound[clash]!r}"
                )
                break
            bound[name] = slot
        if error is not None:
            failure = RuntimeError(error)
            _receipts.failed(slot, failure, phase="prepare")
            raise failure
    for name, slot in bound.items():
        module = named[name]
        module.forward = make_node_dispatcher(
            slot, module, module.forward, runner, registry=registry
        )
    return sorted(bound)


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Hook ``ModelRunner.load_model`` once its module is imported."""

    module = sys.modules.get(_RUNNER)
    runner = getattr(module, "ModelRunner", None)
    if runner is None or _STOCK_LOAD in runner.__dict__:
        return
    load = runner.load_model

    def load_model(self, *args, **kwargs):
        result = load(self, *args, **kwargs)
        bind(self, registry)
        return result

    setattr(runner, _STOCK_LOAD, load)
    runner.load_model = load_model


def is_installed() -> bool:
    runner = getattr(sys.modules.get(_RUNNER), "ModelRunner", None)
    return runner is not None and _STOCK_LOAD in runner.__dict__


def uninstall() -> None:
    runner = getattr(sys.modules.get(_RUNNER), "ModelRunner", None)
    if runner is not None and _STOCK_LOAD in runner.__dict__:
        runner.load_model = runner.__dict__[_STOCK_LOAD]
        delattr(runner, _STOCK_LOAD)

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

The tolerance is measured, not declared, because honest rounding grows with the
width of the node: a stock node whose fused ops run on SGLang's native reference
paths sat 0.4% from stock at a block and 4-11% at the whole stack, and one fixed
elementwise bar failed that honest twin on 295 of 1,560 layer calls and 39 of 39
stack calls (same runs). So the twin also answers every audited call, and a row of
the candidate passes when its relative error is within a small multiple of the
twin's. Rows are graded, not tensors, because a routing flip moves a whole token
and nothing else; rows pool across calls so a one-token call is not a verdict.

Binding happens once, right after ``ModelRunner.load_model``: late enough that the
weights exist, early enough that the breakable prefill runner and the decode graph
runner both capture the bound forward at every width, whole stack included (same
runs). While a candidate runs, nested nodes serve stock, so a wide candidate may
call the stock children it does not replace.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import deque
from contextlib import contextmanager
from typing import Callable, NamedTuple

import torch

from cacheon import audit as _audit
from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor
from cacheon.dispatch import (
    _arch_tag, _dtype_name, _dynamo_compiling, _flashinfer_tuning, _in_cuda_graph,
)
from cacheon.integrations.sglang_dsa_state import StateFormat, dsa_state_rows, state_values
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.slots import SLOTS

_RUNNER = "sglang.srt.model_executor.model_runner"
_STOCK_LOAD = "_cacheon_stock_load_model"
# The pinned engine's per-token cache buffers: the MHA pair or the MLA latent.
_KV_BUFFERS = ("k_buffer", "v_buffer", "kv_buffer")
# A row passes within max(_FLOOR, _TWIN_FACTOR x the twin's 90th-percentile row
# error), never above _CEILING. No honest single kernel reached a quarter of the
# floor; the widest honest node needed 0.33 (three times 11% at the whole stack) and
# the wrong controls sat at 0.5 (H100 Qwen runs, 2026-09-19). The ceiling is what a
# candidate that managed to make the twin noisy could buy at most.
_FLOOR = 0.02
_TWIN_FACTOR = 3.0
_CEILING = 0.4
# Share of rows that must pass, graded once this many rows have pooled. Honest noise
# is heavy-tailed (the twin graded against itself kept 84% of rows in the worst of
# 2,280 decoder-layer windows and 90% at the whole stack) while a wrong answer keeps
# none, so the bar sits well below honest.
_ROW_BAR = 0.75
_WINDOW = 256
# Elements of engine state read, compared or put back at a time. A hybrid model
# keeps 63 MB of recurrent state per request: at 48 requests one copy is 2.8 GiB,
# and holding the before, stock, twin and candidate copies at once ran an 80 GiB
# H100 out of memory with 5 GiB spare (Qwen cell-width audit, 2026-09-20).
# At MTP width 48, retaining the 2.8 GiB temporal answer plus ReplaySSM rings
# left only 117 MiB for a 240 MiB comparison (H100, 2026-09-22). Answers now
# wait on the host too; a piece is at most this budget or one indivisible row.
_PIECE = 4 << 20
# The wire vocabulary (oci_session_protocol.AuditReceiptFacts) is closed: a share of
# rows within tolerance against a bar is its matched_ratio.
_MODE = "matched_ratio"
# Per bound node and graded-tensor position: recent twin noise, and [passed, seen]
# rows. Per node, not per address: layers under one `*` differ severalfold in noise.
_noise: dict[tuple[int, int], deque] = {}
_pooled: dict[tuple[int, int], list[int]] = {}


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


def _state_rows(runner, call: tuple) -> list[tuple[torch.Tensor, int, torch.Tensor, StateFormat | _Low]]:
    """Engine-state rows this call may write, as ``(buffer, dim, index, graded dtype)``.

    Only a call that carries the engine's batch can reach the cache pools: cache
    rows at ``out_cache_loc`` and, on hybrid models, the recurrent-state rows of
    the batch's requests. An unrecognized cache layout raises rather than leaving
    written state unchecked.

    SGLang keeps an FP8 cache in ``uint8`` storage with the real type on the pool.
    Graded as bytes, a near-zero value whose sign flips reads as a jump of 128 (7% of
    a typical row, above the floor); the rows are graded as the numbers they hold
    (H100 Qwen FP8-KV runs, 2026-09-20: 39,200 of 43,760 graded windows were bytes).
    """

    batch = next(
        (v for v in call if hasattr(v, "out_cache_loc") and hasattr(v, "req_pool_indices")),
        None,
    )
    if batch is None:
        return []
    rows: list[tuple[torch.Tensor, int, torch.Tensor, StateFormat]] = []
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
        held = getattr(pool, "dtype", None)
        fp8 = held is not None and held.is_floating_point and held.itemsize == 1
        dsa = dsa_state_rows(pool, batch.out_cache_loc)
        rows.extend(dsa if dsa is not None else [
            (
                buffer,
                0,
                batch.out_cache_loc.long(),
                held if fp8 and buffer.dtype == torch.uint8 else buffer.dtype,
            )
            for buffer in buffers
        ])
    requests = runner.req_to_token_pool
    recurrent = getattr(requests, "mamba_pool", None)
    if recurrent is not None:
        index = requests.get_mamba_indices(batch.req_pool_indices).long()
        cache = recurrent.mamba_cache
        rows.extend(
            (buffer, 1, index, buffer.dtype) for buffer in (*cache.conv, cache.temporal)
        )
        if cache.replayssm_g is not None:
            rows.extend(_replay_rows(runner, recurrent, batch))
    return rows


class _Low(NamedTuple):
    """A BF16 ring's rounding residual, graded with its high part as the one number they hold."""

    high: torch.Tensor


def _replay_rows(runner, pool, batch) -> list[tuple]:
    """What a speculative-verify call writes besides ``conv`` and ``temporal`` under GDN ReplaySSM.

    Verify leaves ``temporal`` alone and appends this step's drafts to rings keyed by
    request slot (``req_pool_indices``, not the batch's Mamba slots); the i-th request's
    per-draft conv windows go to verify scratch row i. Acceptance, outside the node,
    advances the ring cursors and scatters the accepted window into ``conv``; the
    21,400-call Qwen MTP audit restored and graded none of these writes (2026-09-22).

    ``rawv``/``rawk`` hold what BF16 rounding dropped from ``d``/``k``. Honest
    rounding moves the residual by its whole size, so each is graded summed with its
    high part. The windows are restored through the pool's physical buffers, because
    the per-draft view overlaps itself.
    """
    if pool.replayssm_cache_base is None:
        raise RuntimeError("only the speculative-verify GDN ReplaySSM state layout is recognized")
    if not batch.forward_mode.is_target_verify():
        return []
    cache = pool.mamba_cache
    slots = batch.req_pool_indices.long()
    scratch = runner.attn_backend.linear_attn_backend.verify_intermediate_state_indices
    scratch = scratch[: slots.numel()].long()
    rows = [(ring, 1, slots, ring.dtype)
            for ring in (cache.replayssm_d, cache.replayssm_k, cache.replayssm_g)]
    rows.extend((low, 1, slots, _Low(high)) for low, high in (
        (cache.replayssm_rawv, cache.replayssm_d), (cache.replayssm_rawk, cache.replayssm_k),
    ) if low is not None)
    rows.extend((window, 1, scratch, window.dtype) for window in pool._intermediate_conv_window_phys)
    return rows


def _values(buffer: torch.Tensor, dim: int, index: torch.Tensor, held) -> torch.Tensor:
    """The numbers a state row holds at ``index``."""

    raw = buffer.index_select(dim, index)
    if isinstance(held, _Low):
        return held.high.index_select(dim, index).float() + raw.float()
    return state_values(raw, held)


def _pieces(buffer: torch.Tensor, dim: int, index: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    """Split selected rows into budgeted runs, keeping an individual row intact."""

    step = max(1, _PIECE // max(1, buffer.numel() // max(1, buffer.shape[dim])))
    return [(start, index[start : start + step]) for start in range(0, index.numel(), step)]


def _host_piece(tensor: torch.Tensor) -> torch.Tensor:
    """Keep a separate pinned copy so restoring it never slices a large host tensor."""
    return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu",
                       pin_memory=tensor.is_cuda).copy_(tensor)


def _dsa_choice_position(module, count: int) -> int | None:
    """Identify the unordered token-index result in the pinned DSA node signatures."""
    indexer = sys.modules.get("sglang.srt.layers.attention.dsa.dsa_indexer")
    if isinstance(module, getattr(indexer, "Indexer", ())):
        return 0
    model = sys.modules.get("sglang.srt.models.deepseek_v2")
    if (isinstance(module, getattr(model, "DeepseekV2AttentionMLA", ()))
            and module.use_dsa and count == 2):
        return 1
    if (isinstance(module, getattr(model, "DeepseekV2DecoderLayer", ()))
            and module.self_attn.use_dsa and count == 3):
        return 2
    return None


def _errors(outputs: list, rows: list, expected: list, *, module=None) -> list[torch.Tensor]:
    """Row errors of a result and of the live engine state against stock's answer."""

    if len(outputs) + len(rows) != len(expected):
        raise ValueError("the result does not have stock's tensor structure")
    unordered = _dsa_choice_position(module, len(outputs))
    found = [_row_errors(a, e, dim, unordered=position == unordered)
             for position, ((a, dim), (e, _)) in enumerate(zip(outputs, expected))]
    for (buffer, dim, index, held), (saved, _) in zip(rows, expected[len(outputs):]):
        pieces = _pieces(buffer, dim, index)
        if len(pieces) != len(saved):
            raise ValueError("the engine state does not have stock's row structure")
        parts = [
            _row_errors(
                _values(buffer, dim, piece, held), e.to(buffer.device), dim,
            )
            for (_, piece), e in zip(pieces, saved)
        ]
        found.append(torch.cat(parts) if parts else torch.empty(0))
    return found


def _references(slot: str, module, stock: Callable, runner, args: tuple, kwargs: dict):
    """Stock's ``(tensor, row dim)`` answer and the honest twin's row errors against it.

    Both run on the live call, and the call is put back after each. What stock leaves
    in its own arguments is not part of the answer. The fused norm overwrites its
    arguments and returns them, so an honest norm that returns fresh tensors failed
    1,521 of 1,560 calls while that was graded, and a whole decoder layer leaves
    normed intermediates in its dead input (H100 Qwen runs, 2026-09-19). Values reach
    the caller through the result and the engine state; both are graded.

    Stock's state answers wait on the host in separate pinned pieces. The copy
    the call is put back from also waits there when it is more than one piece;
    the twin's and the candidate's rows are compared a piece at a time.
    """

    handed = _tensors((args, kwargs), [])
    rows = _state_rows(runner, (*args, *kwargs.values()))
    before = [t.clone() for t in handed]
    state = []
    for buffer, dim, index, _ in rows:
        pieces = _pieces(buffer, dim, index)
        # More than one piece waits on the host, each its own pinned tensor: slices of
        # one host copy cost 4.4 s of CPU per call against 0.2 s, forty calls a decode
        # step (H100, 2026-09-20).
        state.append([
            part if len(pieces) < 2
            else _host_piece(part)
            for part in (buffer.index_select(dim, piece) for _, piece in pieces)
        ])

    def run(keep):
        outputs = _audit.capture_reference(
            slot, lambda: _tensors(stock(*args, **kwargs), [], fields=True)
        )
        kept = None if outputs is None else keep([(t, 0) for t in outputs])
        # Write back only what stock changed: an untouched input may be an expanded
        # or otherwise unwritable view.
        for tensor, saved in zip(handed, before):
            if not torch.equal(tensor, saved):
                tensor.copy_(saved)
        for (buffer, dim, index, _), saved in zip(rows, state):
            for (_, piece), part in zip(_pieces(buffer, dim, index), saved):
                buffer.index_copy_(dim, piece, part.to(buffer.device))
        return kept

    expected = run(lambda outputs: outputs + [
        ([_host_piece(_values(buffer, dim, piece, held))
          for _, piece in _pieces(buffer, dim, index)], dim)
        for buffer, dim, index, held in rows
    ])
    with _native(module):
        twin = run(lambda outputs: expected and _errors(outputs, rows, expected, module=module))
    return expected, twin


def _seal(module) -> list[tuple]:
    """The methods stock and its twin run through, held so identity can be re-checked.

    ``prepare`` and ``entry`` receive the live module. A candidate that rebinds a
    ``forward`` inside its node makes stock agree with it, and one that rebinds a
    ``forward_native`` makes the twin noisy and the tolerance wide. Weights are not
    sealed: a rewritten weight changes the served model itself, which the
    end-to-end quality gate grades against the pristine reference.
    """

    return [(m, _methods(m)) for m in module.modules()]


def _methods(m) -> dict:
    """Every callable, and every empty attribute, on a module and its classes below ``nn.Module``."""

    found = {n: v for n, v in vars(m).items() if callable(v) or v is None}
    for cls in type(m).__mro__:
        if cls is torch.nn.Module:
            break
        found.update(((cls, n), v) for n, v in vars(cls).items() if callable(v))
    return found


def _rebound(sealed: list[tuple]) -> str | None:
    """Name the first sealed callable that changed, or None."""

    # Equality compares bound methods by function and instance, so a method the
    # engine re-reads and stores again is not a change; a rebound one is.
    for m, methods in sealed:
        now = _methods(m)
        own = [v for k, v in methods.items() if not isinstance(k, str)]
        for key in methods.keys() | now.keys():
            old, new = methods.get(key), now.get(key)
            if old == new:
                continue
            # SGLang's fused ops hold ``_forward_method = None`` until their first
            # call and then cache one of their own methods there; the first seal
            # refused every honest bundle for it (H100 Qwen run, 2026-09-19, "TopK:
            # _forward_method"). An empty attribute may be filled with the module's
            # own sealed method and with nothing else.
            if (
                key in methods and old is None
                and getattr(new, "__self__", None) is m and new.__func__ in own
            ):
                continue
            name = key if isinstance(key, str) else f"{key[0].__name__}.{key[1]}"
            return f"{type(m).__name__}: {name}"
    return None


def _native_forward(module):
    """Keep the unfused residual-add rounding boundary in the native norm twin.

    SGLang's RMSNorm native path still fuses the add in FP32. An independent
    PyTorch rounded-add norm and the qualified norm both false-failed through
    GLM's quantized layers when that was the only twin (B300, 2026-09-20).
    """
    native = module.forward_native
    norm = getattr(sys.modules.get("sglang.srt.layers.layernorm"), "RMSNorm", ())
    if not isinstance(module, norm):
        return native

    def unfused(x, residual=None, post_residual_addition=None, quant_linear=None):
        if (residual is not None and residual.dtype == x.dtype
                and not module.fp32_residual and module.override_orig_dtype is None
                and post_residual_addition is None and quant_linear is None):
            summed = x + residual
            return native(summed), summed
        return native(x, residual, post_residual_addition, quant_linear)

    return unfused


@contextmanager
def _native(module):
    """Use supported native paths while the DSA indexer retains hardware dispatch.

    The pinned Indexer explicitly has no native implementation. Its children and
    surrounding ops still take native paths; forcing its stub refused 6,048
    references in the first full GLM layer-bundle audit (B300, 2026-09-20).
    """

    indexer = getattr(sys.modules.get("sglang.srt.layers.attention.dsa.dsa_indexer"), "Indexer", ())
    sites = [m for m in module.modules()
             if "_forward_method" in vars(m) and not isinstance(m, indexer)]
    saved = [m._forward_method for m in sites]
    for m in sites:
        m._forward_method = _native_forward(m)
    try:
        yield
    finally:
        for m, method in zip(sites, saved):
            m._forward_method = method


def _row_errors(actual: torch.Tensor, expected: torch.Tensor, dim: int,
                *, unordered: bool = False) -> torch.Tensor:
    if actual.shape != expected.shape:
        raise ValueError(f"shape {tuple(actual.shape)} is not stock's {tuple(expected.shape)}")
    a = actual.detach().reshape(-1, 1) if actual.dim() < 2 else actual.detach()
    e = expected.detach().reshape(-1, 1) if expected.dim() < 2 else expected.detach()
    a, e = a.movedim(dim, 0).flatten(1), e.movedim(dim, 0).flatten(1)
    if not expected.is_floating_point():
        if unordered:
            # DSA's selector emits an unordered index set. Ordered grading rejected
            # stock at all 19 index-producing GLM layers (B300, 2026-09-20). Match
            # occurrences so duplicate IDs and padding cannot manufacture overlap.
            a, e = a.sort(dim=1).values.contiguous(), e.sort(dim=1).values.contiguous()
            occurrence = (torch.arange(e.shape[1], device=e.device)
                          - torch.searchsorted(e, e, right=False))
            available = (torch.searchsorted(a, e, right=True)
                         - torch.searchsorted(a, e, right=False))
            return (occurrence >= available).sum(dim=1) / max(1, e.shape[1])
        # Ids and flags are choices: the error is the share that differ, in stock's order,
        # which the other outputs line up with. As magnitudes, a router that only picked
        # a quarter of the experts kept 96.6% of its rows inside the floor (2026-09-20).
        return (a != e).sum(dim=1) / max(1, e.shape[1])
    a, e = a.float(), e.float()
    errors = (a - e).norm(dim=1) / e.norm(dim=1).clamp_min(1e-12)
    # Rows stock itself left non-finite are nobody's answer: the 2026-09-18 retained
    # bundle matched every real token and failed on NaNs in an idle rank's padding.
    return errors[torch.isfinite(e).all(dim=1)]


def _grade(slot: str, node: int, actual: list, twin: list) -> tuple[float, int] | None:
    """Pool each graded tensor's passing rows; record a unit when a window fills."""

    filled = []
    for position, (errors, honest) in enumerate(zip(actual, twin)):
        if not honest.numel():
            continue
        key = (node, position)
        noise = _noise.setdefault(key, deque(maxlen=64))
        noise.append(torch.quantile(honest, 0.9).item())
        # History stabilizes one-row calls, but cannot discard this call's reference
        # noise: quiet decode history falsely rejected GLM's 4096-row prefill calls.
        measured = _TWIN_FACTOR * max(noise[-1], sorted(noise)[int(0.9 * (len(noise) - 1))])
        tolerance = min(_CEILING, max(_FLOOR, measured))
        pooled = _pooled.setdefault(key, [0, 0])
        pooled[0] += int((errors <= tolerance).sum().item())
        pooled[1] += errors.numel()
        if pooled[1] >= _WINDOW:
            filled.append((pooled[0] / pooled[1], position))
            pooled[:] = [0, 0]
    if filled:
        worst = min(filled)
        _audit.record_fraction(slot, worst[0], _ROW_BAR, _MODE)
        return worst


def _descriptor(module, args: tuple, kwargs: dict, in_graph: bool) -> CallDescriptor | None:
    handed = _tensors((args, kwargs), [])
    if not handed or not handed[0].dim():
        return None
    # Rotary positions can lead the call as (3, M); the activations carry M.
    floating = next((t for t in handed if t.is_floating_point()), None)
    dtype_source = floating if floating is not None else next(module.parameters(), handed[0])
    floating = floating if floating is not None else handed[0]
    return CallDescriptor.from_legacy(
        dtype_name=_dtype_name(dtype_source.dtype),
        last_dim=int(floating.shape[-1]),
        arch=_arch_tag(floating.device.index or 0) if floating.is_cuda else None,
    ).with_updates(graph_mode="cuda_graph" if in_graph else "eager")


def make_node_dispatcher(
    slot: str,
    module,
    stock: Callable[..., object],
    runner,
    sealed: list[tuple],
    *,
    registry: KernelRegistry = REGISTRY,
    node_name: str | None = None,
) -> Callable[..., object]:
    """Build the replacement ``forward`` for one bound node.

    ``sealed`` is filled by ``bind`` once every node is bound and before any
    candidate code has run.
    """

    prepared: dict[int, object] = {}
    reported = False

    def dispatched(*args, **kwargs):
        nonlocal reported
        if (
            _dynamo_compiling() or _flashinfer_tuning()
            or _receipts.is_invoking() or not registry.active
        ):
            return stock(*args, **kwargs)
        in_graph = _in_cuda_graph()
        # Drawn, and both references run, before any rank-local branch: selection
        # must not affect the seeded audit draws, and a node with a collective hangs unless
        # every rank runs it the same number of times with the same seeded draws.
        expected = twin = None
        if not in_graph and _audit.sampled():
            changed = _rebound(sealed)
            if changed is not None:
                failure = RuntimeError(
                    f"a method inside node {slot!r} was rebound after binding ({changed})"
                )
                _receipts.failed(slot, failure, phase="entry")
                raise failure
            expected, twin = _references(slot, module, stock, runner, args, kwargs)
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
        result = _receipts.invoke(
            slot, impl.entry, prepared[id(impl)], *args, kwargs=kwargs
        )
        if expected is not None and twin is not None:
            try:
                actual = _errors(
                    [(t, 0) for t in _tensors(result, [], fields=True)],
                    _state_rows(runner, (*args, *kwargs.values())), expected, module=module,
                )
                grade = _grade(slot, id(module), actual, twin)
                if grade is not None and grade[0] < _ROW_BAR and not reported:
                    logging.getLogger(__name__).error(
                        "node audit mismatch: node=%s tensor_position=%d passing_fraction=%.6f",
                        node_name or slot, grade[1], grade[0],
                    )
                    reported = True
            except ValueError:
                _audit.compare_error(slot)
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
    seals: dict[str, list[tuple]] = {name: [] for name in bound}
    for name, slot in bound.items():
        module = named[name]
        module.forward = make_node_dispatcher(
            slot, module, module.forward, runner, seals[name], registry=registry, node_name=name
        )
    for name in bound:
        seals[name].extend(_seal(named[name]))
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
        # Node addresses belong to the target model, not its speculative drafter.
        if not self.is_draft_worker:
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

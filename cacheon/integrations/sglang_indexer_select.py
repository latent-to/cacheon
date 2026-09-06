"""Replace complete paged/ragged selection regions after engine-owned index-cache writes."""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
from functools import wraps
from types import FunctionType

import torch

from cacheon.capabilities import CapabilityMismatch
from cacheon.dispatch import (
    _allocate_live_outputs, _arch_tag, _audit, _dynamo_compiling, _flashinfer_tuning,
    _in_cuda_graph, _receipts, _runtime_parallel_sizes, _validate_live_outputs,
)
from cacheon.indexer_select_contract import SLOT, call_descriptor, invoke_entry, output_bounds
from cacheon.registry import REGISTRY, KernelRegistry

_MODULE = "sglang.srt.layers.attention.dsa.dsa_indexer"
_FUNCTIONS = ("_get_topk_paged", "_get_topk_ragged")
_ORIGINAL = "_cacheon_original_indexer_select"
_PREPARE = "_fused_q_prepare_and_store"


def _bind_prepare(baseline, module, registry=REGISTRY):
    """Carry raw Q metadata in the existing return pair; keep both K-store branches intact."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(baseline)))
    name = "fused_q_indexer_rope_first_quant"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]
    if len(calls) != 2:
        raise RuntimeError(f"SGLang indexer Q producer changed: expected two calls, got {len(calls)}")
    original = getattr(module, name)

    def defer(q, weights, gate, cache, positions):
        if (os.environ.get("CACHEON_INDEXER_SELECT_SEAM") != "1" or not registry.active
            or not registry.variants(SLOT) or _receipts.is_invoking()
            or _dynamo_compiling() or _flashinfer_tuning()):
            return original(q, weights, gate, cache, positions)
        return q, (weights, positions, cache, gate)

    namespace = dict(baseline.__globals__, **{name: defer})
    bound = FunctionType(baseline.__code__, namespace, baseline.__name__, baseline.__defaults__, baseline.__closure__)
    bound.__kwdefaults__ = baseline.__kwdefaults__
    return wraps(baseline)(bound)


def _declined(registry, field, reason, expected):
    """Receipt a refusal made before selection, so a registered candidate that never ran is explained."""
    if registry.active and registry.variants(SLOT):
        _receipts.not_selected(SLOT, "seam_declined", (CapabilityMismatch(field, reason, expected),))


def _inputs(call, module, registry, *, ragged):
    """Reuse the producer's PAGED mapping and expose zero-copy key/scale views."""
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod, _build_flashinfer_paged_args

    owner, metadata, q = call["self"], call["metadata"], call["q_fp8"]
    weights, positions, cache, gate = call["weights"], None, None, 1.0
    raw_query = isinstance(weights, tuple)
    if raw_query:
        weights, positions, cache, gate = weights
    if metadata.topk_transform_method != TopkTransformMethod.PAGED:
        return _declined(registry, "layout", "unsupported", "paged_indexer_select")
    if metadata.force_unfused_topk:
        # Unfused top-k hands the consumer row-local positions; this ABI emits physical slots.
        return _declined(registry, "top_k", "unfused", "fused_physical_selection")
    if call["forward_batch"].attn_cp_metadata is not None:
        return _declined(registry, "context_parallel", "unsupported", "none")
    if q.dtype not in ((torch.bfloat16, torch.float16) if raw_query else (torch.float8_e4m3fn,)):
        return _declined(registry, "dtype", "outside_domain", "raw bfloat16/float16 or prepared float8_e4m3fn")
    pool = module.get_token_to_kv_pool()
    size, dim = pool.page_size, q.shape[-1]
    raw = owner._get_index_k_read_buffer(pool, call["layer_id"])
    if raw.ndim != 2 or raw.dtype != torch.uint8 or raw.shape[1] != size * (dim + 4):
        return _declined(registry, "index_cache_layout", "unsupported", "packed key/scale pages")
    starts = metadata.get_indexer_kvcache_range()[0] if ragged else None
    count = starts.numel() if ragged else sum(metadata.get_dsa_extend_len_cpu())
    batches, offsets = _build_flashinfer_paged_args(
        attn_metadata=metadata.attn_metadata, row_starts=starts,
        cu_seqlens_q_topk=metadata.attn_metadata.cu_seqlens_q, batch_idx_list=None,
        device=q.device, num_rows=count)
    lengths = metadata.get_seqlens_expanded()[:count]
    return dict(q=q[:count], key_pages=raw[:, :size * dim].view(torch.float8_e4m3fn).view(-1, size, dim),
                key_scales=raw[:, size * dim:].view(torch.float32),
                weights=weights[:count] if raw_query else weights[:count].squeeze(-1),
                positions=positions[:count] if raw_query else None, cos_sin_cache=cache, q_scale_gate=gate,
                page_table=metadata.get_page_table_64(), lengths=lengths,
                row_to_batch=torch.arange(count, dtype=torch.int32, device=q.device) if batches is None else batches,
                page_offsets=torch.zeros_like(lengths) if offsets is None else offsets,
                num_init_tokens=owner.num_init_tokens, num_local_tokens=owner.num_local_tokens, top_k=owner.index_topk)


def _make_dispatch(baseline, registry, module, *, ragged):
    """Replace only the selection region; preserve padding and the caller's output storage."""
    signature = inspect.signature(baseline)

    @wraps(baseline)
    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments

        def stock():
            actual = dict(call)
            if isinstance(actual["weights"], tuple):
                weights, positions, cache, gate = actual["weights"]
                actual["q_fp8"], actual["weights"] = module.fused_q_indexer_rope_first_quant(
                    actual["q_fp8"], weights, gate, cache, positions)
            return baseline(**actual)

        if (os.environ.get("CACHEON_INDEXER_SELECT_SEAM") != "1" or _receipts.is_invoking()
            or _dynamo_compiling() or _flashinfer_tuning()):
            return stock()
        inputs = _inputs(call, module, registry, ragged=ragged)
        if inputs is None:
            return stock()
        q = inputs["q"]
        tp, world = _runtime_parallel_sizes()
        descriptor = call_descriptor(inputs,
            architecture=_arch_tag(q.device.index or 0) if q.is_cuda else None,
            graph_mode="cuda_graph" if _in_cuda_graph() else "eager", tp_size=tp, world_size=world)
        impl = registry.select(SLOT, descriptor).impl
        if impl is None:
            return stock()
        spec, allocation, tensors, bindings = _allocate_live_outputs(SLOT, inputs, like=q)
        expected = stock()[:q.shape[0]].clone() if _audit.sampled() else None
        with torch.inference_mode():
            invoke_entry(lambda *a: _receipts.invoke(SLOT, impl.entry, *a), inputs, allocation.outputs)
        _validate_live_outputs(spec, allocation, tensors, bindings, like=q)
        # A hostile index must never reach the vendor kernel as a cache address;
        # a clamped entry only counts as a wrong selection in the audit.
        result = allocation.outputs[0].clamp_(*output_bounds(inputs))
        if expected is not None:
            _audit.run(SLOT, (result,), lambda: expected)
        destination = call.get("topk_result")
        if destination is not None or q.shape[0] < call["q_fp8"].shape[0]:
            if destination is None:
                destination = torch.full((call["q_fp8"].shape[0], inputs["top_k"]), -1,
                                         dtype=torch.int32, device=q.device)
            destination[:q.shape[0]].copy_(result)
            result = destination
        _receipts.completed(SLOT)
        return result
    return dispatch


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch the actual Indexer class without replacing forward/store orchestration."""
    module = sys.modules.get(_MODULE)
    if module is None or hasattr(module.Indexer, _ORIGINAL):
        return
    originals = {name: getattr(module.Indexer, name) for name in (*_FUNCTIONS, _PREPARE)}
    replacements = {_PREPARE: _bind_prepare(originals[_PREPARE], module, registry)}
    replacements.update({name: _make_dispatch(originals[name], registry, module, ragged=name == _FUNCTIONS[1]) for name in _FUNCTIONS})
    for name, replacement in replacements.items():
        setattr(module.Indexer, name, replacement)
    setattr(module.Indexer, _ORIGINAL, originals)


def uninstall() -> None:
    """Restore both exact method objects on the pinned class."""
    module = sys.modules.get(_MODULE)
    if module is not None and hasattr(module.Indexer, _ORIGINAL):
        for name, baseline in getattr(module.Indexer, _ORIGINAL).items():
            setattr(module.Indexer, name, baseline)
        delattr(module.Indexer, _ORIGINAL)


def is_installed() -> bool:
    """Report whether this interpreter has the merged region binding."""
    module = sys.modules.get(_MODULE)
    return module is not None and hasattr(module.Indexer, _ORIGINAL)

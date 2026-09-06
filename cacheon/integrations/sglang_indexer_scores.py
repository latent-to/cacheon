"""Route SGLang's DeepGEMM ragged/paged MQA logits through one score ABI."""

from __future__ import annotations

import os
import sys
from functools import wraps

import torch

from cacheon.dispatch import (
    _allocate_live_outputs, _arch_tag, _audit, _dynamo_compiling,
    _flashinfer_tuning, _in_cuda_graph, _receipts, _runtime_parallel_sizes,
    _validate_live_outputs,
)
from cacheon.indexer_scores_contract import SLOT, call_descriptor, invoke_entry
from cacheon.registry import REGISTRY, KernelRegistry

_MODULE = "deep_gemm"
_FUNCTIONS = ("fp8_mqa_logits", "fp8_paged_mqa_logits")
_ORIGINAL = "_cacheon_original_indexer_scores"


def _ragged_inputs(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                   clean_logits=True, max_seqlen_k=0) -> dict | None:
    """Treat concatenated prefill keys as one page; preserve global row windows."""
    if clean_logits or max_seqlen_k or q.ndim != 3 or len(kv) != 2:
        return None
    keys, scales = kv
    if (keys.ndim != 2 or keys.shape[1] != q.shape[-1]
        or scales.shape != (keys.shape[0],) or keys.stride(-1) != 1
        or cu_seq_len_k_start.shape != (q.shape[0],)
        or cu_seq_len_k_end.shape != (q.shape[0],)):
        return None
    return dict(q=q, key_pages=keys.unsqueeze(0), key_scales=scales.unsqueeze(0),
                weights=weights, starts=cu_seq_len_k_start, ends=cu_seq_len_k_end,
                page_table=torch.zeros((1, 1), device=q.device, dtype=torch.int32),
                row_to_batch=torch.zeros(q.shape[0], device=q.device, dtype=torch.int32),
                kv_len=keys.shape[0])


def _paged_inputs(q, kv_cache, weights, context_lens, block_table,
                  schedule_meta, max_context_len, clean_logits=False,
                  indices=None) -> dict | None:
    """View each page as FP8 keys followed by FP32 scales, never 132-byte AoS."""
    if (clean_logits or indices is not None or q.ndim != 4
        or kv_cache.ndim != 4 or kv_cache.dtype != torch.uint8
        or kv_cache.shape[2:] != (1, q.shape[-1] + 4)
        or kv_cache.stride(1) != q.shape[-1] + 4 or kv_cache.stride(3) != 1
        or kv_cache.stride(0) % 4 or context_lens.shape != q.shape[:2]
        or block_table.ndim != 2 or block_table.shape[0] != q.shape[0]
        or max_context_len <= 0 or max_context_len > block_table.shape[1] * kv_cache.shape[1]):
        return None
    batch, next_n, heads, dim = q.shape
    pages, page_size = kv_cache.shape[:2]
    raw = kv_cache.view(pages, -1)
    key_bytes = page_size * dim
    keys = raw[:, :key_bytes].view(torch.float8_e4m3fn).view(pages, page_size, dim)
    scales = raw[:, key_bytes:].view(torch.float32)
    ends = context_lens.reshape(-1)
    return dict(q=q.reshape(batch * next_n, heads, dim), key_pages=keys,
                key_scales=scales, weights=weights, starts=torch.zeros_like(ends), ends=ends,
                page_table=block_table,
                row_to_batch=torch.arange(batch, device=q.device, dtype=torch.int32).repeat_interleave(next_n),
                kv_len=max_context_len)


def _make_dispatch(baseline, registry: KernelRegistry, *, paged: bool):
    normalize = _paged_inputs if paged else _ragged_inputs

    @wraps(baseline)
    def dispatch(*args, **kwargs):
        if (os.environ.get("CACHEON_INDEXER_SCORES_SEAM") != "1"
            or _receipts.is_invoking()
            or _dynamo_compiling() or _flashinfer_tuning()):
            return baseline(*args, **kwargs)
        inputs = normalize(*args, **kwargs)
        if inputs is None:
            return baseline(*args, **kwargs)
        q = inputs["q"]
        if (q.dtype != torch.float8_e4m3fn or inputs["key_pages"].dtype != q.dtype
            or inputs["key_scales"].dtype != torch.float32
            or inputs["weights"].dtype != torch.float32
            or inputs["weights"].shape != q.shape[:2]
            or any(inputs[k].dtype != torch.int32 for k in ("starts", "ends", "page_table", "row_to_batch"))
            or any(v.device != q.device for v in inputs.values() if torch.is_tensor(v))):
            return baseline(*args, **kwargs)
        tp_size, world_size = _runtime_parallel_sizes()
        descriptor = call_descriptor(inputs,
            architecture=_arch_tag(q.device.index or 0) if q.is_cuda else None,
            graph_mode="cuda_graph" if _in_cuda_graph() else "eager",
            tp_size=tp_size, world_size=world_size)
        impl = registry.select(SLOT, descriptor).impl
        if impl is None:
            return baseline(*args, **kwargs)
        spec, allocation, tensors, bindings = _allocate_live_outputs(SLOT, inputs, like=q)
        expected = baseline(*args, **kwargs).clone() if _audit.sampled() else None
        with torch.inference_mode():
            invoke_entry(lambda *a: _receipts.invoke(SLOT, impl.entry, *a), inputs, allocation.outputs)
        _validate_live_outputs(spec, allocation, tensors, bindings, like=q)
        result = allocation.outputs[0]
        if expected is not None:
            # clean_logits=False leaves these cells undefined in DeepGEMM;
            # DSA top-k masks exactly these row windows before reading scores.
            columns = torch.arange(result.shape[1], device=q.device)
            expected.masked_fill_((columns < inputs["starts"][:, None]) |
                                  (columns >= inputs["ends"][:, None]), 0)
            _audit.run(SLOT, (result,), lambda: expected)
        _receipts.completed(SLOT)
        return result

    return dispatch


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch both module attributes that DSAIndexer resolves at each call."""
    module = sys.modules.get(_MODULE)
    if module is None or hasattr(module, _ORIGINAL):
        return
    originals = {name: getattr(module, name) for name in _FUNCTIONS}
    for name, baseline in originals.items():
        setattr(module, name, _make_dispatch(baseline, registry, paged=name == _FUNCTIONS[1]))
    setattr(module, _ORIGINAL, originals)


def uninstall() -> None:
    """Restore both original callables without altering the registry."""
    module = sys.modules.get(_MODULE)
    if module is not None and hasattr(module, _ORIGINAL):
        for name, baseline in getattr(module, _ORIGINAL).items():
            setattr(module, name, baseline)
        delattr(module, _ORIGINAL)


def is_installed() -> bool:
    """Report whether this interpreter owns the score replacement."""
    module = sys.modules.get(_MODULE)
    return module is not None and hasattr(module, _ORIGINAL)

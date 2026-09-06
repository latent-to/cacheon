"""Bind sparse MLA at SGLang 0.5.18's shared FlashInfer decode call.

Both TRTLLM DSA phases resolve this module-level symbol per call. The boundary
is after index transformation, RoPE/quantization and KV writes; it does not
replace those producers or expose their engine objects to candidate code.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from functools import wraps

import torch

from cacheon.dispatch import (
    _allocate_live_outputs, _arch_tag, _audit, _dynamo_compiling,
    _flashinfer_tuning, _in_cuda_graph, _validate_live_outputs,
    _receipts, _runtime_parallel_sizes,
)
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.sparse_mla_contract import (
    SLOT, _INPUT_DTYPES, call_descriptor, invoke_entry,
)

_MODULE = "flashinfer.decode"
_FUNCTION = "trtllm_batch_decode_with_kv_cache_mla"
_ORIGINAL = "_cacheon_original_sparse_mla"


def _inputs(call: dict) -> dict | None:
    """Select only the pinned one-query-per-row sparse-MLA call domain."""
    if (
        call["backend"] != "trtllm-gen"
        or call["sparse_mla_top_k"] <= 0
        or call["out"] is not None
        or call["sinks"] is not None
        or call["skip_softmax_threshold_scale_factor"] is not None
        or call["lse"] is not None or call["return_lse"]
        or not call["is_var_seq"] or not call["uses_shared_paged_kv_idx"]
        or call["cum_seq_lens_q"] is not None or call["max_q_len"] is not None
        or call["sparse_mla_top_k_lens"] is not None
        or call["enable_dcp"] or call["cp_world"] != 1 or call["cp_rank"] != 0
        or call["causal_seqlens_kv_global"] is not None
    ):
        return None
    q, cache, indices, lengths = (
        call[k] for k in ("query", "kv_cache", "block_tables", "seq_lens")
    )
    if not all(torch.is_tensor(t) for t in (q, cache, indices, lengths)):
        return None
    if (
        q.ndim != 4 or q.shape[1] != 1 or q.dtype not in _INPUT_DTYPES
        or cache.ndim not in (3, 4)
        or (cache.ndim == 4 and cache.shape[1] != 1)
        or cache.dtype != q.dtype or cache.shape[-2] not in (32, 64)
        or cache.shape[-1] != q.shape[-1]
        or indices.shape != (q.shape[0], 1, call["sparse_mla_top_k"])
        or indices.dtype != torch.int32
        or lengths.shape != (q.shape[0],) or lengths.dtype != torch.int32
        or call["kv_lora_rank"] < 1 or call["qk_rope_head_dim"] < 1
        or q.shape[-1] != call["kv_lora_rank"] + call["qk_rope_head_dim"]
        or not all(t.device == q.device and t.is_contiguous()
                   for t in (q, cache, indices, lengths))
    ):
        return None
    scales = (call["bmm1_scale"], call["bmm2_scale"])
    if any(type(s) not in (int, float) or not math.isfinite(s) for s in scales):
        return None
    return dict(
        q=q[:, 0], kv_cache=cache[:, 0] if cache.ndim == 4 else cache,
        indices=indices[:, 0], seq_lens=lengths, value_dim=call["kv_lora_rank"],
        qk_scale=float(scales[0]), value_scale=float(scales[1]),
    )


def _make_dispatch(baseline, registry: KernelRegistry):
    signature = inspect.signature(baseline)

    @wraps(baseline)
    def dispatch(*args, **kwargs):
        if (
            os.environ.get("CACHEON_SPARSE_MLA_SEAM") != "1"
            or _dynamo_compiling() or _flashinfer_tuning()
        ):
            return baseline(*args, **kwargs)
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        inputs = _inputs(call.arguments)
        if inputs is None:
            return baseline(*args, **kwargs)
        q = inputs["q"]
        tp_size, world_size = _runtime_parallel_sizes()
        descriptor = call_descriptor(
            inputs, architecture=_arch_tag(q.device.index or 0) if q.is_cuda else None,
            graph_mode="cuda_graph" if _in_cuda_graph() else "eager",
            tp_size=tp_size, world_size=world_size,
        )
        impl = registry.select(SLOT, descriptor).impl
        if impl is None:
            return baseline(*args, **kwargs)
        spec, allocation, tensors, bindings = _allocate_live_outputs(SLOT, inputs, like=q)
        # Snapshot only the stock output before untrusted execution. Copying the
        # whole KV pool for a sampled audit would scale with arena capacity.
        expected = baseline(*args, **kwargs).clone() if _audit.sampled() else None
        with torch.inference_mode():
            invoke_entry(
                lambda *a: _receipts.invoke(SLOT, impl.entry, *a),
                inputs, allocation.outputs,
            )
        _validate_live_outputs(spec, allocation, tensors, bindings, like=q)
        result = allocation.outputs[0].unsqueeze(1)
        if expected is not None:
            _audit.run(SLOT, (result,), lambda: expected)
        _receipts.completed(SLOT)
        return result

    return dispatch


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch the exact symbol consumed by SGLang's local FlashInfer import."""
    module = sys.modules.get(_MODULE)
    if module is None or hasattr(module, _ORIGINAL):
        return
    baseline = getattr(module, _FUNCTION)
    setattr(module, _FUNCTION, _make_dispatch(baseline, registry))
    setattr(module, _ORIGINAL, baseline)


def uninstall() -> None:
    """Restore the original library callable without changing registry state."""
    module = sys.modules.get(_MODULE)
    if module is not None and hasattr(module, _ORIGINAL):
        setattr(module, _FUNCTION, getattr(module, _ORIGINAL))
        delattr(module, _ORIGINAL)


def is_installed() -> bool:
    """Report whether this interpreter owns the pinned call replacement."""
    module = sys.modules.get(_MODULE)
    return module is not None and hasattr(module, _ORIGINAL)

"""Replace DSA top-k through its shared paged prefill/decode tensor boundary."""

from __future__ import annotations

import inspect
import os
import sys
from functools import wraps

import torch

from cacheon.dispatch import (
    _allocate_live_outputs, _arch_tag, _audit, _dynamo_compiling,
    _flashinfer_tuning, _in_cuda_graph, _receipts, _runtime_parallel_sizes,
    _validate_live_outputs,
)
from cacheon.indexer_topk_contract import SLOT, call_descriptor, invoke_entry
from cacheon.registry import REGISTRY, KernelRegistry

_MODULE = "sglang.srt.layers.attention.dsa.dsa_topk_backend"
_ORIGINAL = "_cacheon_original_topk_transform"


def _inputs(call: dict, module) -> dict | None:
    """Reuse the pinned producer's mapping without expanding its KV page table."""
    if (call["topk_transform_method"] != module.TopkTransformMethod.PAGED
        or call["force_unfused_topk"] or not module.envs.SGLANG_DSA_FUSE_TOPK.get()):
        return None
    scores, lengths, metadata = call["logits"], call["lengths"], call["attn_metadata"]
    table = metadata.real_page_table
    if (scores.ndim != 2 or scores.stride(-1) != 1 or scores.dtype != torch.float32
        or lengths.shape != (scores.shape[0],) or lengths.dtype != torch.int32
        or table.ndim != 2 or table.dtype != torch.int32 or call["topk"] <= 0):
        return None
    batches, offsets = module._build_flashinfer_paged_args(
        attn_metadata=metadata, row_starts=call["row_starts"],
        cu_seqlens_q_topk=call["cu_seqlens_q_topk"], batch_idx_list=call["batch_idx_list"],
        device=scores.device, num_rows=scores.shape[0],
    )
    starts = call["row_starts"]
    return dict(scores=scores, lengths=lengths, page_table=table,
                row_starts=torch.zeros_like(lengths) if starts is None else starts,
                row_to_batch=torch.arange(scores.shape[0], device=scores.device, dtype=torch.int32)
                if batches is None else batches,
                page_offsets=torch.zeros_like(lengths) if offsets is None else offsets,
                page_size=metadata.page_size, top_k=call["topk"])


def _make_dispatch(baseline, registry: KernelRegistry, module):
    signature = inspect.signature(baseline)

    @wraps(baseline)
    def dispatch(*args, **kwargs):
        if (os.environ.get("CACHEON_INDEXER_TOPK_SEAM") != "1"
            or _dynamo_compiling() or _flashinfer_tuning()):
            return baseline(*args, **kwargs)
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        inputs = _inputs(call.arguments, module)
        if inputs is None:
            return baseline(*args, **kwargs)
        scores = inputs["scores"]
        tp_size, world_size = _runtime_parallel_sizes()
        descriptor = call_descriptor(inputs,
            architecture=_arch_tag(scores.device.index or 0) if scores.is_cuda else None,
            graph_mode="cuda_graph" if _in_cuda_graph() else "eager", tp_size=tp_size, world_size=world_size)
        impl = registry.select(SLOT, descriptor).impl
        if impl is None:
            return baseline(*args, **kwargs)
        spec, allocation, tensors, bindings = _allocate_live_outputs(SLOT, inputs, like=scores)
        expected = baseline(*args, **kwargs).clone() if _audit.sampled() else None
        with torch.inference_mode():
            invoke_entry(lambda *a: _receipts.invoke(SLOT, impl.entry, *a), inputs, allocation.outputs)
        _validate_live_outputs(spec, allocation, tensors, bindings, like=scores)
        result = allocation.outputs[0]
        if expected is not None:
            _audit.run(SLOT, (result,), lambda: expected)
        _receipts.completed(SLOT)
        return result

    return dispatch


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Install once on the enum method used by DSAIndexerMetadata."""
    module = sys.modules.get(_MODULE)
    if module is None or hasattr(module, _ORIGINAL):
        return
    baseline = module.DSATopKBackend.topk_transform
    module.DSATopKBackend.topk_transform = _make_dispatch(baseline, registry, module)
    setattr(module, _ORIGINAL, baseline)


def uninstall() -> None:
    """Restore the exact original method."""
    module = sys.modules.get(_MODULE)
    if module is not None and hasattr(module, _ORIGINAL):
        module.DSATopKBackend.topk_transform = getattr(module, _ORIGINAL)
        delattr(module, _ORIGINAL)


def is_installed() -> bool:
    """Report whether the module-level installation marker exists."""
    module = sys.modules.get(_MODULE)
    return module is not None and hasattr(module, _ORIGINAL)

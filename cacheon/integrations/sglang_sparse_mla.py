"""Expose query preparation and sparse MLA while retaining engine-owned K writes.

The pinned method is changed at three expressions. Its cache, page-table, CP and
workspace orchestration stays in SGLang; unmatched source fails installation.
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
from functools import partial, wraps

import torch

from cacheon.dispatch import (
    _allocate_live_outputs, _arch_tag, _audit, _in_cuda_graph, _receipts,
    _runtime_parallel_sizes, _validate_live_outputs,
)
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.sparse_mla_contract import SLOT, call_descriptor, invoke_entry

_MODULE = "sglang.srt.layers.attention.dsa_backend"
_CLASS = "DeepseekSparseAttnBackend"
_FUNCTION = "_forward_trtllm"
_ORIGINAL = "_cacheon_original_sparse_mla"


def _prepare(q, q_rope, k, k_rope, positions, cache, neox, value_dim, rope_dim,
             *, backend, layer, forward_batch, registry, baseline, module):
    """Select before splitting Q work out of the engine's joint Q/K producer."""
    args = (q, q_rope, k, k_rope, positions, cache, neox, value_dim, rope_dim)
    if (os.environ.get("CACHEON_SPARSE_MLA_SEAM") != "1" or _receipts.is_invoking()
        or q.dtype not in (torch.bfloat16, torch.float16)
        or module.dsa_use_prefill_cp(forward_batch)
        or module.envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None):
        return (*baseline(*args), None)
    kv = backend.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
        -1, backend.real_page_size, backend.kv_cache_dim)
    inputs = dict(q=q, q_rope=q_rope, kv_cache=kv, value_dim=value_dim,
                  top_k=backend.dsa_index_topk)
    tp, world = _runtime_parallel_sizes()
    impl = registry.select(SLOT, call_descriptor(inputs,
        architecture=_arch_tag(q.device.index or 0) if q.is_cuda else None,
        graph_mode="cuda_graph" if _in_cuda_graph() else "eager", tp_size=tp, world_size=world)).impl
    if impl is None:
        return (*baseline(*args), None)
    # Zero query heads make the same vendor operation K-only. Exact-image tests
    # compare K bytes and changing-position replays against its joint Q/K call.
    import flashinfer.rope

    _, k_rotated, _, k_quantized = flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope[:, :0], k_rope=k_rope, q_nope=q[:, :0], k_nope=k,
        cos_sin_cache=cache, pos_ids=positions, is_neox=neox,
        quantize_dtype=torch.float8_e4m3fn)
    return q, k_quantized, k_rotated, (impl, args, baseline)


def _attend(*, implementation, q_rope, positions, cos_sin_cache, is_neox, **call):
    """Invoke the selected family member after SGLang has written the prepared K."""
    import flashinfer.decode

    baseline = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
    if implementation is None:
        return baseline(**call)
    impl, prepare_args, prepare_baseline = implementation
    q, cache = call["query"][:, 0], call["kv_cache"][:, 0]
    inputs = dict(q=q, q_rope=q_rope, positions=positions, cos_sin_cache=cos_sin_cache,
                  is_neox=is_neox, kv_cache=cache, indices=call["block_tables"][:, 0],
                  seq_lens=call["seq_lens"], value_dim=call["kv_lora_rank"],
                  qk_scale=call["bmm1_scale"], value_scale=call.get("bmm2_scale", 1.0))
    spec, allocation, tensors, bindings = _allocate_live_outputs(SLOT, inputs, like=q)
    expected = None
    if _audit.sampled():
        query, _, _ = prepare_baseline(*prepare_args)
        expected = baseline(**dict(call, query=query[:, None])).clone()
    with torch.inference_mode():
        invoke_entry(lambda *a: _receipts.invoke(SLOT, impl.entry, *a), inputs, allocation.outputs)
    _validate_live_outputs(spec, allocation, tensors, bindings, like=q)
    result = allocation.outputs[0][:, None]
    if expected is not None:
        _audit.run(SLOT, (result,), lambda: expected)
    _receipts.completed(SLOT)
    return result


def _bind_method(baseline, registry, module):
    """Retain the pinned producer body and change only its Q preparation and use."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(baseline)))
    function = tree.body[0]
    expected = ("self", "q", "k", "v", "layer", "forward_batch", "seq_lens", "save_kv_cache",
                "q_rope", "k_rope", "topk_indices", "cos_sin_cache", "is_neox", "llama_4_scaling", "is_prefill")
    if (tuple(inspect.signature(baseline).parameters) != expected
        or tuple(a.arg for a in function.args.args) != expected
        or function.args.vararg is not None or function.args.kwarg is not None):
        raise RuntimeError("SGLang sparse MLA producer signature changed")
    changes = [0, 0, 0]
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "mla_quantize_and_rope_for_fp8":
                if not isinstance(node.targets[0], ast.Tuple) or len(node.targets[0].elts) != 3:
                    raise RuntimeError("SGLang MLA preparation assignment changed")
                node.targets[0].elts.append(ast.Name(id="_cacheon_impl", ctx=ast.Store()))
                call.func.id = "_cacheon_prepare_query"
                call.keywords.extend(ast.keyword(arg=k, value=ast.Name(id=v, ctx=ast.Load()))
                                     for k, v in (("backend", "self"), ("layer", "layer"), ("forward_batch", "forward_batch")))
                changes[0] += 1
            elif (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                  and call.func.value.id == "q" and call.func.attr == "view" and len(call.args) == 3
                  and ast.unparse(call.args[-1]) == "layer.head_dim"):
                call.args[-1] = ast.parse("q.shape[-1]", mode="eval").body
                changes[1] += 1
            elif ast.unparse(call.func) == "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla":
                call.func = ast.Name(id="_cacheon_attend_query", ctx=ast.Load())
                call.keywords.extend(ast.keyword(arg=k, value=ast.Name(id=v, ctx=ast.Load())) for k, v in (
                    ("implementation", "_cacheon_impl"), ("q_rope", "q_rope"), ("positions", "rope_positions"),
                    ("cos_sin_cache", "cos_sin_cache"), ("is_neox", "is_neox")))
                changes[2] += 1
    if changes != [1, 1, 1]:
        raise RuntimeError(f"SGLang sparse MLA producer structure changed: {changes}")
    function.body[1:1] = ast.parse("_cacheon_impl = None\nrope_positions = None").body
    namespace = dict(baseline.__globals__,
        _cacheon_prepare_query=partial(_prepare, registry=registry,
            baseline=module.mla_quantize_and_rope_for_fp8, module=module),
        _cacheon_attend_query=_attend)
    exec(compile(ast.fix_missing_locations(tree), baseline.__code__.co_filename, "exec"), namespace)
    return wraps(baseline)(namespace[baseline.__name__])


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Bind the shared prefill/decode method once its defining module is loaded."""
    module = sys.modules.get(_MODULE)
    cls = getattr(module, _CLASS, None)
    if cls is not None and not hasattr(cls, _ORIGINAL):
        baseline = getattr(cls, _FUNCTION)
        setattr(cls, _FUNCTION, _bind_method(baseline, registry, module))
        setattr(cls, _ORIGINAL, baseline)


def uninstall() -> None:
    """Restore the original method without altering vendor APIs or cache ownership."""
    cls = getattr(sys.modules.get(_MODULE), _CLASS, None)
    if cls is not None and hasattr(cls, _ORIGINAL):
        setattr(cls, _FUNCTION, getattr(cls, _ORIGINAL))
        delattr(cls, _ORIGINAL)


def is_installed() -> bool:
    """Report whether the pinned producer is bound in this interpreter."""
    return hasattr(getattr(sys.modules.get(_MODULE), _CLASS, None), _ORIGINAL)

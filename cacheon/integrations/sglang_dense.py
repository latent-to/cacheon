"""Bind unquantized GEMMs, FP32 projections and absorbed BMM to one family."""

from __future__ import annotations

import os
import sys
from functools import wraps

import torch

from cacheon.dense_contract import call_descriptor
from cacheon.dispatch import _arch_tag, _audit, _in_cuda_graph, _receipts, _runtime_parallel_sizes
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.tensor_spec import tensor_bindings, validate_tensor_bindings

_MODULE = "sglang.srt.layers.quantization.unquant"
_PATCH_FLAG = "_cacheon_dense_patched"
_SLOT = "linear.dense"
_PROJECTIONS = (("sglang.srt.models.deepseek_v2", "MoEGate", "forward", "weight"),
                ("sglang.srt.layers.attention.dsa.dsa_indexer", "Indexer",
                 "_weights_proj_bf16_in_fp32_out", "weights_proj"))
_BMM_MODULE = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla"
_ORIGINAL = "_cacheon_dense_original"


def _parallel_role(layer: object) -> str:
    if hasattr(layer, "gather_output"):
        return "column"
    return "row" if hasattr(layer, "input_is_parallel") else "replicated"


@torch.inference_mode()
def _prepared(layer: object, impl: object, weight: torch.Tensor):
    """Reuse preparation while the canonical source storage and version agree."""
    cache = getattr(layer, "_cacheon_dense_prepared_by_impl", None)
    if cache is None:
        cache = layer._cacheon_dense_prepared_by_impl = {}
    binding = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()), weight.dtype,
               None if weight.is_inference() else weight._version)
    key = (impl.bundle_id, impl.variant, id(impl.prepare))
    if key not in cache or cache[key][0] != binding:
        if impl.prepare is None:
            raise RuntimeError("selected linear.dense candidate has no prepare")
        cache[key] = (binding, _receipts.invoke(_SLOT, impl.prepare, weight, phase="prepare"))
    return cache[key][1]


def _dispatch(layer, x, weight, registry, stock, *, output=None, output_dtype=None, reference=None):
    """Select once and preserve caller-owned strided output and original failures."""
    if (os.environ.get("CACHEON_DENSE_SEAM") != "1" or _receipts.is_invoking()
        or not torch.is_tensor(x) or not torch.is_tensor(weight)
        or x.ndim not in (2, 3) or weight.ndim != x.ndim
        or x.shape[-1] != weight.shape[-1]
        or (x.ndim == 3 and x.shape[0] != weight.shape[0])
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or x.device != weight.device):
        return stock()
    dtype = output_dtype or x.dtype
    shape = (*x.shape[:-1], weight.shape[-2])
    if output is not None and (output.shape != shape or output.dtype != dtype or output.device != x.device):
        return stock()
    _, world = _runtime_parallel_sizes()
    descriptor = call_descriptor(dict(x=x, weight=weight, output_dtype=dtype),
        architecture=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        graph_mode="cuda_graph" if _in_cuda_graph() else "eager",
        parallel_role=_parallel_role(layer), tp_size=int(getattr(layer, "tp_size", 1)), world_size=world)
    impl = registry.select(_SLOT, descriptor).impl
    if impl is None:
        return stock()
    # Stock apply_into/BMM returns the same view that the candidate will overwrite.
    expected = (reference or stock)().clone() if _audit.sampled() else None
    out = torch.empty(shape, dtype=dtype, device=x.device) if output is None else output
    tensors = (x, weight, out)
    bindings = tensor_bindings(tensors)
    with torch.inference_mode():
        _receipts.invoke(_SLOT, impl.entry, x, _prepared(layer, impl, weight), out)
    validate_tensor_bindings(tensors, bindings, kind="dense input/output")
    if expected is not None:
        _audit.run(_SLOT, (out,), lambda: expected)
    _receipts.completed(_SLOT)
    return out


def _make_apply(baseline, registry: KernelRegistry, *, stock_apply=None):
    into = stock_apply is not None
    @wraps(baseline)
    def apply(self, layer, x, *args, **kwargs):
        output = args[0] if into and args else kwargs.get("output") if into else None
        tail = args[1:] if into else args
        bias = tail[0] if tail else kwargs.get("bias")
        stock = lambda: baseline(self, layer, x, *args, **kwargs)
        if bias is not None or (into and output is None):
            return stock()
        # apply_into may call the now-patched self.apply; audit must use its saved original.
        reference = (lambda: stock_apply(self, layer, x, None)) if into else None
        return _dispatch(layer, x, getattr(layer, "weight", None), registry, stock,
                         output=output, reference=reference)
    return apply


def _make_projection(baseline, registry, module, weight_attr):
    @wraps(baseline)
    def project(self, x, *args, **kwargs):
        stock = lambda: baseline(self, x, *args, **kwargs)
        # The deterministic router branch returns input dtype, outside this FP32 consumer.
        if (not module._is_cuda or (weight_attr == "weight" and
            module.get_exec().deterministic.enable_deterministic_inference)):
            return stock()
        layer = getattr(self, "weights_proj", None) if weight_attr == "weights_proj" else self
        return _dispatch(layer, x, getattr(layer, "weight", None), registry, stock, output_dtype=torch.float32)
    return project


def _make_bmm(baseline, registry):
    @wraps(baseline)
    def bmm(input, mat2, out_dtype=None, *, out=None):
        def stock():
            kw = dict(out=out)
            if out_dtype is not None:
                kw["out_dtype"] = out_dtype
            return baseline(input, mat2, **kw)
        if (input.ndim != 3 or mat2.ndim != 3 or input.dtype != mat2.dtype
            or out_dtype not in (None, torch.float32)
            or (out_dtype is not None and (not input.is_cuda or input.dtype not in (torch.bfloat16, torch.float16)))):
            return stock()
        return _dispatch(mat2, input, mat2.transpose(-1, -2), registry, stock,
                         output=out, output_dtype=out_dtype)
    return bmm


class _TorchView:
    """Override only this SGLang consumer's BMM; miner imports receive ordinary Torch."""

    def __init__(self, original, bmm):
        self._original, self.bmm = original, bmm

    def __getattr__(self, name):
        return getattr(self._original, name)


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Install whichever family consumers have completed import in this interpreter."""
    cls = getattr(sys.modules.get(_MODULE), "UnquantizedLinearMethod", None)
    if cls is not None and not getattr(cls, _PATCH_FLAG, False):
        cls._cacheon_orig_apply, cls._cacheon_orig_apply_into = cls.apply, cls.apply_into
        cls.apply = _make_apply(cls.apply, registry)
        cls.apply_into = _make_apply(cls.apply_into, registry, stock_apply=cls._cacheon_orig_apply)
        setattr(cls, _PATCH_FLAG, True)
    for name, class_name, method, weight_attr in _PROJECTIONS:
        module = sys.modules.get(name)
        cls = getattr(module, class_name, None)
        if cls is not None and not hasattr(cls, _ORIGINAL):
            baseline = getattr(cls, method)
            setattr(cls, _ORIGINAL, baseline)
            setattr(cls, method, _make_projection(baseline, registry, module, weight_attr))
    module = sys.modules.get(_BMM_MODULE)
    if module is not None and not hasattr(module, _ORIGINAL):
        original = module.torch
        module.torch = _TorchView(original, _make_bmm(original.bmm, registry))
        setattr(module, _ORIGINAL, original)


def uninstall() -> None:
    """Restore all original callables and module references idempotently."""
    cls = getattr(sys.modules.get(_MODULE), "UnquantizedLinearMethod", None)
    if cls is not None and getattr(cls, _PATCH_FLAG, False):
        cls.apply, cls.apply_into = cls._cacheon_orig_apply, cls._cacheon_orig_apply_into
        delattr(cls, "_cacheon_orig_apply")
        delattr(cls, "_cacheon_orig_apply_into")
        setattr(cls, _PATCH_FLAG, False)
    for name, class_name, method, _ in _PROJECTIONS:
        cls = getattr(sys.modules.get(name), class_name, None)
        if cls is not None and hasattr(cls, _ORIGINAL):
            setattr(cls, method, getattr(cls, _ORIGINAL))
            delattr(cls, _ORIGINAL)
    module = sys.modules.get(_BMM_MODULE)
    if module is not None and hasattr(module, _ORIGINAL):
        module.torch = getattr(module, _ORIGINAL)
        delattr(module, _ORIGINAL)


def is_installed() -> bool:
    """Report any installed consumer of the GEMM family."""
    cls = getattr(sys.modules.get(_MODULE), "UnquantizedLinearMethod", None)
    return bool(getattr(cls, _PATCH_FLAG, False) or hasattr(sys.modules.get(_BMM_MODULE), _ORIGINAL)
                or any(hasattr(getattr(sys.modules.get(n), c, None), _ORIGINAL) for n, c, _, _ in _PROJECTIONS))

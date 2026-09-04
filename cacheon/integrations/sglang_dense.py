"""Bind the dense-GEMM slot to SGLang's unquantized linear method."""

from __future__ import annotations

import os
import logging
from functools import wraps

import torch

from cacheon.capabilities import CallDescriptor
from cacheon.dispatch import (
    _arch_tag,
    _audit,
    _dynamo_compiling,
    _in_cuda_graph,
    _receipts,
    _runtime_parallel_sizes,
)
from cacheon.registry import REGISTRY, KernelRegistry

_MODULE = "sglang.srt.layers.quantization.unquant"
_PATCH_FLAG = "_cacheon_dense_patched"
_SLOT = "linear.dense"
_LOGGED_CALLS: set[tuple[object, ...]] = set()
logger = logging.getLogger("cacheon.dense")


def _active() -> bool:
    return os.environ.get("CACHEON_DENSE_SEAM") == "1"


def _parallel_role(layer: object) -> str:
    if hasattr(layer, "gather_output"):
        return "column"
    if hasattr(layer, "input_is_parallel"):
        return "row"
    return "replicated"


def _descriptor(layer: object, x: torch.Tensor, weight: torch.Tensor) -> CallDescriptor:
    _, world_size = _runtime_parallel_sizes()
    return CallDescriptor(
        architecture=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        dtype=str(x.dtype).removeprefix("torch."),
        graph_mode="cuda_graph" if _in_cuda_graph() else "eager",
        input_dim=int(weight.shape[1]),
        last_dim=int(x.shape[-1]),
        layout="weight_out_in_row_major",
        num_tokens=int(x.shape[0]),
        output_dim=int(weight.shape[0]),
        parallel_role=_parallel_role(layer),
        quant="dense",
        tp_size=int(getattr(layer, "tp_size", 1)),
        world_size=world_size,
    )


def _eligible(layer: object, x: torch.Tensor, bias: object) -> torch.Tensor | None:
    weight = getattr(getattr(layer, "weight", None), "data", None)
    if not (
        _active()
        and torch.is_tensor(weight)
        and x.dim() == 2
        and weight.dim() == 2
        and x.shape[1] == weight.shape[1]
        and x.dtype == weight.dtype
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias is None
    ):
        return None
    return weight


@torch.inference_mode()
def _prepared(layer: object, impl: object, weight: torch.Tensor):
    cache = getattr(layer, "_cacheon_dense_prepared_by_impl", None)
    if cache is None:
        cache = {}
        layer._cacheon_dense_prepared_by_impl = cache
    key = (impl.bundle_id, impl.variant, id(impl.prepare))
    if key not in cache:
        if impl.prepare is None:
            raise RuntimeError("selected linear.dense candidate has no prepare")
        cache[key] = _receipts.invoke(
            _SLOT, impl.prepare, weight, phase="prepare"
        )
    return cache[key]


def _select(layer: object, x: torch.Tensor, bias: object, registry: KernelRegistry):
    weight = _eligible(layer, x, bias)
    if weight is None:
        return None
    impl = registry.select(_SLOT, _descriptor(layer, x, weight)).impl
    if impl is None:
        return None
    signature = (
        int(x.shape[-1]), tuple(x.stride()), tuple(weight.shape),
        tuple(weight.stride()), str(x.dtype), _parallel_role(layer),
        int(getattr(layer, "tp_size", 1)),
    )
    if signature not in _LOGGED_CALLS:
        _LOGGED_CALLS.add(signature)
        logger.warning(
            "cacheon: linear.dense seam ACTIVE call=%r",
            (tuple(x.shape), *signature[1:]),
        )
    return impl, weight, _prepared(layer, impl, weight)


def _make_apply(baseline, registry: KernelRegistry):
    @wraps(baseline)
    def apply(self, layer, x, bias=None):
        if _dynamo_compiling():
            return baseline(self, layer, x, bias)
        selected = _select(layer, x, bias, registry)
        if selected is None:
            return baseline(self, layer, x, bias)
        impl, weight, prepared = selected
        out = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
        )
        audit = _audit.sampled()
        audit_x = x.clone() if audit else None
        with torch.inference_mode():
            _receipts.invoke(_SLOT, impl.entry, x, prepared, out)
        if audit:
            _audit.run(
                _SLOT,
                (out,),
                lambda: baseline(self, layer, audit_x, None),
            )
        _receipts.completed(_SLOT)
        return out

    return apply


def _make_apply_into(baseline, stock_apply, registry: KernelRegistry):
    @wraps(baseline)
    def apply_into(self, layer, x, output, bias=None):
        if _dynamo_compiling():
            return baseline(self, layer, x, output, bias)
        expected = (x.shape[0], layer.weight.shape[0])
        if (
            tuple(output.shape) != expected
            or output.dtype != x.dtype
            or output.device != x.device
            or not output.is_contiguous()
        ):
            return baseline(self, layer, x, output, bias)
        selected = _select(layer, x, bias, registry)
        if selected is None:
            return baseline(self, layer, x, output, bias)
        impl, _, prepared = selected
        audit = _audit.sampled()
        audit_x = x.clone() if audit else None
        with torch.inference_mode():
            _receipts.invoke(_SLOT, impl.entry, x, prepared, output)
        if audit:
            _audit.run(
                _SLOT,
                (output,),
                lambda: stock_apply(self, layer, audit_x, None),
            )
        _receipts.completed(_SLOT)
        return output

    return apply_into


def install(registry: KernelRegistry = REGISTRY) -> None:
    import sys

    module = sys.modules.get(_MODULE)
    cls = getattr(module, "UnquantizedLinearMethod", None) if module else None
    if cls is None or getattr(cls, _PATCH_FLAG, False):
        return
    original_apply = cls.apply
    original_apply_into = cls.apply_into
    cls.apply = _make_apply(original_apply, registry)
    cls.apply_into = _make_apply_into(original_apply_into, original_apply, registry)
    cls._cacheon_orig_apply = original_apply
    cls._cacheon_orig_apply_into = original_apply_into
    setattr(cls, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    module = sys.modules.get(_MODULE)
    cls = getattr(module, "UnquantizedLinearMethod", None) if module else None
    if cls is None or not getattr(cls, _PATCH_FLAG, False):
        return
    cls.apply = cls._cacheon_orig_apply
    cls.apply_into = cls._cacheon_orig_apply_into
    delattr(cls, "_cacheon_orig_apply")
    delattr(cls, "_cacheon_orig_apply_into")
    setattr(cls, _PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    module = sys.modules.get(_MODULE)
    cls = getattr(module, "UnquantizedLinearMethod", None) if module else None
    return bool(cls is not None and getattr(cls, _PATCH_FLAG, False))

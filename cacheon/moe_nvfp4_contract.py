"""Validator-owned ModelOpt NVFP4 weight contract for the plain MoE slot."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch

from cacheon.capabilities import CallDescriptor
from cacheon_kernels import codec

NVFP4_PREPARE_TAG = "nvfp4_layer"
NVFP4_GATE_UP_LAYOUT = "gate_up"
NVFP4_INTERLEAVED_LAYOUT = "up_gate_interleaved_64+sf_swizzled_128x4"
_NVFP4_TENSORS = (
    "w13_weight", "w2_weight", "w13_blockscale_swizzled",
    "w2_blockscale_swizzled", "g1_alphas", "g2_alphas",
    "w13_input_scale_quant", "w2_input_scale_quant",
)


def call_descriptor(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    architecture: str | None,
    graph_mode: str,
    quant: str,
    num_experts: int,
    intermediate_dim: int,
    tp_size: int | None = None,
    world_size: int | None = None,
) -> CallDescriptor:
    fields: dict[str, object] = {
        "architecture": architecture,
        "dtype": str(x.dtype).removeprefix("torch."),
        "graph_mode": graph_mode,
        "hidden_dim": int(x.shape[-1]),
        "intermediate_dim": int(intermediate_dim),
        "last_dim": int(x.shape[-1]),
        "layout": "routed_moe",
        "num_experts": int(num_experts),
        "num_tokens": int(x.shape[0]),
        "quant": quant,
        "top_k": int(topk_ids.shape[-1]),
    }
    if tp_size is not None:
        fields["tp_size"] = tp_size
    if world_size is not None:
        fields["world_size"] = world_size
    return CallDescriptor({key: value for key, value in fields.items() if value is not None})


def _value(source: object, name: str) -> Any:
    value = source[name] if isinstance(source, Mapping) else getattr(source, name)
    return getattr(value, "data", value)


def _context(source: object, name: str, default: object) -> object:
    key = f"__moe_{name}__"
    if isinstance(source, Mapping):
        return source.get(key, default)
    attr = f"moe_{name}" if name in {"tp_size", "ep_size", "ep_rank"} else name
    return getattr(source, attr, default)


def supports_layer(layer: object) -> bool:
    try:
        values = tuple(_value(layer, name) for name in _NVFP4_TENSORS)
        return (
            values[0].dtype == values[1].dtype == torch.uint8
            and values[2].dtype == values[3].dtype == torch.float8_e4m3fn
            and all(torch.is_tensor(value) for value in values)
            and int(_value(layer, "intermediate_size_per_partition")) > 0
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _layer_view(
    source: object,
    *,
    layout: str,
    scale_names: tuple[str, str] = ("w13_weight_scale", "w2_weight_scale"),
) -> SimpleNamespace:
    w13, w2 = _value(source, "w13_weight"), _value(source, "w2_weight")
    w13_sf, w2_sf = (_value(source, name) for name in scale_names)
    experts = int(w13.shape[0])
    g1 = _value(source, "g1_alphas")
    g2 = _value(source, "g2_alphas")
    a1 = _value(source, "w13_input_scale_quant")
    a2 = _value(source, "w2_input_scale_quant")
    intermediate = int(_value(source, "intermediate_size_per_partition"))
    top_k = int(source["topk_ids"].shape[-1]) if isinstance(source, Mapping) else 0
    tp_size = int(_context(source, "tp_size", 1))
    fused_shared = int(_context(source, "num_fused_shared_experts", 0))
    view = SimpleNamespace(
        w13_weight=w13, w2_weight=w2,
        w13_weight_scale=w13_sf, w2_weight_scale=w2_sf,
        w13_blockscale_swizzled=w13_sf, w2_blockscale_swizzled=w2_sf,
        g1_alphas=g1, g2_alphas=g2,
        w13_input_scale_quant=a1, w2_input_scale_quant=a2,
        fc1_input_dequant=a1.float().reciprocal(), fc1_dequant=g1,
        fc2_quant=a2, fc2_dequant=g2,
        intermediate_size_per_partition=intermediate,
        num_local_experts=experts, num_experts=experts,
        hidden_size=int(w13.shape[-1]) * 2,
        moe_ep_size=int(_context(source, "ep_size", 1)),
        moe_ep_rank=int(_context(source, "ep_rank", 0)),
        moe_tp_size=tp_size,
        reduce_results=bool(_context(source, "reduce_results", False)),
        num_fused_shared_experts=fused_shared,
        cacheon_group_size=16, cacheon_w13_layout=layout,
        moe_runner_config=SimpleNamespace(
            is_gated=True, num_experts=experts, top_k=top_k,
            hidden_size=int(w13.shape[-1]) * 2,
            intermediate_size_per_partition=intermediate,
            # A MODEL fact, carried by the arena profile through the verification
            # inputs (__moe_activation__). The swigluoai default preserves the
            # live-layer path on M3, where sglang's own runner config is authoritative.
            activation=str(_context(source, "activation", "swigluoai")),
            num_fused_shared_experts=fused_shared,
        ),
    )
    return view


def prepare_args_from_layer(layer: object) -> tuple[object, ...]:
    """Map a live layer to dense args or the explicit NVFP4 tensor contract."""
    w13 = _value(layer, "w13_weight")
    quantized = w13.dtype == torch.uint8 and (
        getattr(layer, "w13_weight_scale", None) is not None
        or getattr(layer, "g1_alphas", None) is not None
    )
    if not quantized:
        return w13, _value(layer, "w2_weight")
    if not supports_layer(layer):
        raise ValueError("NVFP4 MoE layer is outside the canonical weight contract")
    view = _layer_view(
        layer,
        layout=(
            NVFP4_INTERLEAVED_LAYOUT
            if getattr(layer, "w13_blockscale_mma", None) is not None
            else NVFP4_GATE_UP_LAYOUT
        ),
        scale_names=("w13_blockscale_swizzled", "w2_blockscale_swizzled"),
    )
    return NVFP4_PREPARE_TAG, view


def prepare_args_from_inputs(inputs: Mapping[str, object]) -> tuple[object, ...]:
    if inputs.get("__moe_quant__") != "nvfp4":
        return _value(inputs, "w13"), _value(inputs, "w2")
    return NVFP4_PREPARE_TAG, _layer_view(
        inputs, layout=str(inputs["w13_layout"])
    )


def _quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scales = codec.quantize_nvfp4(weight)
    scales = scales.to(torch.float8_e4m3fn)
    dequant = codec.dequantize_nvfp4(packed, scales.float()).to(weight.dtype)
    return packed, scales, dequant


def verification_inputs(dense: Mapping[str, object]) -> dict[str, object]:
    """Quantize validator weights and retain only their dequantized oracle view."""
    w13_q, w13_sf, w13_ref = _quantize(_value(dense, "w13"))
    w2_q, w2_sf, w2_ref = _quantize(_value(dense, "w2"))
    experts = int(w13_q.shape[0])
    ones = torch.ones(experts, dtype=torch.float32, device=w13_q.device)
    return {
        **dense,
        "w13": w13_ref,
        "w2": w2_ref,
        "w13_weight": w13_q,
        "w2_weight": w2_q,
        "w13_weight_scale": codec.swizzle_blockscale(w13_sf),
        "w2_weight_scale": codec.swizzle_blockscale(w2_sf),
        "g1_alphas": ones,
        "g2_alphas": ones.clone(),
        "w13_input_scale_quant": ones.clone(),
        "w2_input_scale_quant": ones.clone(),
        "intermediate_size_per_partition": int(w2_ref.shape[-1]),
        "w13_layout": NVFP4_GATE_UP_LAYOUT,
        "__moe_quant__": "nvfp4",
    }


def dequantize_prepare_args(args: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference decode of the public prepare tuple, used by faithful controls."""
    if len(args) != 2 or args[0] != NVFP4_PREPARE_TAG:
        raise ValueError("invalid NVFP4 MoE prepare contract")
    view = args[1]
    fields = (
        "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
        "g1_alphas", "g2_alphas", "w13_input_scale_quant",
        "w2_input_scale_quant",
    )
    w13_q, w2_q, w13_sf, w2_sf, g1, g2, a1, a2 = (
        getattr(view, name, None) for name in fields
    )
    tensors = (w13_q, w2_q, w13_sf, w2_sf, g1, g2, a1, a2)
    if not all(torch.is_tensor(value) for value in tensors):
        raise ValueError("NVFP4 MoE prepare tensors are malformed")
    w13_sf = codec.unswizzle_blockscale(
        w13_sf, rows=w13_q.shape[1], cols=w13_q.shape[2] * 2 // 16
    )
    w2_sf = codec.unswizzle_blockscale(
        w2_sf, rows=w2_q.shape[1], cols=w2_q.shape[2] * 2 // 16
    )
    layout = getattr(view, "cacheon_w13_layout", "")
    if layout == NVFP4_INTERLEAVED_LAYOUT:
        w13_q = codec.deinterleave_w13_halves(w13_q, group=64)
        w13_sf = codec.deinterleave_w13_halves(w13_sf, group=64)
    elif layout != "gate_up":
        raise ValueError("unknown NVFP4 W13 layout")
    w13 = codec.dequantize_nvfp4(w13_q, w13_sf.float())
    w2 = codec.dequantize_nvfp4(w2_q, w2_sf.float())
    w13 *= (g1.float() * a1.float()).reshape(-1, 1, 1)
    w2 *= (g2.float() * a2.float()).reshape(-1, 1, 1)
    return w13, w2

"""Install both MoE slots at the pinned FusedMoE waist.

MiniMax reduces its fused shared+routed partial outside that waist, immediately or
at the next norm; trusted completion markers suppress exactly that stock reduction.
"""

from __future__ import annotations

from functools import wraps

from cacheon.dispatch import (
    _MOE_OUTER_REDUCE_ATTR,
    _MOE_OUTER_REDUCE_TOKEN,
    _MOE_REDUCED_ATTR,
    _consume_moe_reduced,
    make_moe_dispatcher,
)
from cacheon.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_cacheon_moe_patched"
_MODULE = "sglang.srt.layers.moe.fused_moe_triton.layer"
_MODEL_MODULE = "sglang.srt.models.minimax_m3"
_MODEL_PATCH_FLAG = "_cacheon_moe_reduce_patched"
_MISSING = object()


def _install_minimax_reduce() -> None:
    """Bind the reduce-owning slot to MiniMax's actual outer TP-reduce waist."""
    import sys

    mod = sys.modules.get(_MODEL_MODULE)
    model = getattr(mod, "MiniMaxM3MoE", None) if mod is not None else None
    decoder = getattr(mod, "MiniMaxM3DecoderLayer", None) if mod is not None else None
    if model is None or decoder is None or getattr(model, _MODEL_PATCH_FLAG, False):
        return
    original_forward = getattr(model, "forward_normal", None)
    original_decoder = getattr(decoder, "forward", None)
    original_reduce = getattr(mod, "tensor_model_parallel_all_reduce", None)
    if not all(map(callable, (original_forward, original_decoder, original_reduce))):
        return

    @wraps(original_forward)
    def forward_normal(
        self,
        hidden_states,
        should_allreduce_fusion=False,
        use_reduce_scatter=False,
    ):
        experts = getattr(self, "experts", None)
        prior = getattr(experts, _MOE_OUTER_REDUCE_ATTR, _MISSING)
        fused_shared = getattr(experts, "num_fused_shared_experts", 0)
        eligible = (
            experts is not None
            and hidden_states.shape[0] > 0
            and getattr(self, "tp_size", 1) > 1
            and getattr(self, "shared_experts", _MISSING) is None
            and fused_shared > 0
            and fused_shared == getattr(self, "n_shared_experts", None)
            and not getattr(experts, "reduce_results", True)
            and not use_reduce_scatter
        )
        if experts is not None:
            setattr(
                experts,
                _MOE_OUTER_REDUCE_ATTR,
                (
                    _MOE_OUTER_REDUCE_TOKEN,
                    "deferred" if should_allreduce_fusion else "immediate",
                )
                if eligible
                else None,
            )
        try:
            return original_forward(
                self,
                hidden_states,
                should_allreduce_fusion,
                use_reduce_scatter,
            )
        finally:
            if experts is not None:
                if prior is _MISSING:
                    delattr(experts, _MOE_OUTER_REDUCE_ATTR)
                else:
                    setattr(experts, _MOE_OUTER_REDUCE_ATTR, prior)

    @wraps(original_reduce)
    def outer_reduce(output):
        return (
            output
            if _consume_moe_reduced(output, "immediate")
            else original_reduce(output)
        )

    @wraps(original_decoder)
    def decoder_forward(self, *args, **kwargs):
        result = original_decoder(self, *args, **kwargs)
        hidden_states = result[0]
        if hasattr(hidden_states, _MOE_REDUCED_ATTR):
            if getattr(hidden_states, "_sglang_needs_allreduce_fusion", None) is not True:
                raise RuntimeError("deferred reduce-owning MoE output lost its stock marker")
            delattr(hidden_states, "_sglang_needs_allreduce_fusion")
            if not _consume_moe_reduced(hidden_states, "deferred"):
                raise RuntimeError("deferred reduce-owning MoE output was not consumed")
        return result

    model._cacheon_orig_forward_normal = original_forward
    decoder._cacheon_orig_forward = original_decoder
    mod._cacheon_orig_tensor_model_parallel_all_reduce = original_reduce
    model.forward_normal = forward_normal
    decoder.forward = decoder_forward
    mod.tensor_model_parallel_all_reduce = outer_reduce
    setattr(model, _MODEL_PATCH_FLAG, True)


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch whichever exact MoE consumers have finished importing."""
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is not None and hasattr(FusedMoE, "forward_impl"):
        if not getattr(FusedMoE, _PATCH_FLAG, False):
            orig_impl = FusedMoE.forward_impl
            FusedMoE.forward_impl = make_moe_dispatcher(orig_impl, registry=registry)
            FusedMoE._cacheon_orig_forward_impl = orig_impl  # type: ignore[attr-defined]
            setattr(FusedMoE, _PATCH_FLAG, True)
    _install_minimax_reduce()


def uninstall() -> None:
    import sys

    model_mod = sys.modules.get(_MODEL_MODULE)
    model = getattr(model_mod, "MiniMaxM3MoE", None) if model_mod else None
    if model is not None and getattr(model, _MODEL_PATCH_FLAG, False):
        decoder = model_mod.MiniMaxM3DecoderLayer
        model.forward_normal = model._cacheon_orig_forward_normal
        decoder.forward = decoder._cacheon_orig_forward
        model_mod.tensor_model_parallel_all_reduce = (
            model_mod._cacheon_orig_tensor_model_parallel_all_reduce
        )
        delattr(model, "_cacheon_orig_forward_normal")
        delattr(decoder, "_cacheon_orig_forward")
        delattr(model_mod, "_cacheon_orig_tensor_model_parallel_all_reduce")
        setattr(model, _MODEL_PATCH_FLAG, False)

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is not None and getattr(FusedMoE, _PATCH_FLAG, False):
        FusedMoE.forward_impl = FusedMoE._cacheon_orig_forward_impl
        delattr(FusedMoE, "_cacheon_orig_forward_impl")
        setattr(FusedMoE, _PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is None:
        return False
    installed = bool(getattr(FusedMoE, _PATCH_FLAG, False))
    model_mod = sys.modules.get(_MODEL_MODULE)
    model = getattr(model_mod, "MiniMaxM3MoE", None) if model_mod else None
    return installed and (
        model is None or bool(getattr(model, _MODEL_PATCH_FLAG, False))
    )

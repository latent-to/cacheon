"""Install both MoE slots at the pinned FusedMoE waist."""

from __future__ import annotations

from cacheon.dispatch import (
    make_moe_deferred_dispatcher,
    make_moe_deferred_finalize_dispatcher,
    make_moe_dispatcher,
)
from cacheon.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_cacheon_moe_patched"
_DEFERRED_PATCH_FLAG = "_cacheon_moe_deferred_patched"
_MODULE = "sglang.srt.layers.moe.fused_moe_triton.layer"
_FINALIZER_MODULE = "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm"
_FINALIZER_PATCH_FLAG = "_cacheon_moe_deferred_finalize_patched"
_FINALIZER_FUNC = "finalize_flashinfer_trtllm_deferred_output"


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
        if (
            hasattr(FusedMoE, "forward_deferred_finalize")
            and not getattr(FusedMoE, _DEFERRED_PATCH_FLAG, False)
        ):
            orig_deferred = FusedMoE.forward_deferred_finalize
            FusedMoE.forward_deferred_finalize = make_moe_deferred_dispatcher(
                orig_deferred, registry=registry
            )
            FusedMoE._cacheon_orig_forward_deferred_finalize = orig_deferred
            setattr(FusedMoE, _DEFERRED_PATCH_FLAG, True)

    finalizer_mod = sys.modules.get(_FINALIZER_MODULE)
    finalizer = (
        getattr(finalizer_mod, _FINALIZER_FUNC, None)
        if finalizer_mod is not None
        else None
    )
    if (
        callable(finalizer)
        and not getattr(finalizer_mod, _FINALIZER_PATCH_FLAG, False)
    ):
        finalizer_mod._cacheon_orig_moe_deferred_finalize = finalizer
        setattr(
            finalizer_mod,
            _FINALIZER_FUNC,
            make_moe_deferred_finalize_dispatcher(finalizer),
        )
        setattr(finalizer_mod, _FINALIZER_PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is not None:
        if getattr(FusedMoE, _PATCH_FLAG, False):
            FusedMoE.forward_impl = FusedMoE._cacheon_orig_forward_impl
            delattr(FusedMoE, "_cacheon_orig_forward_impl")
            setattr(FusedMoE, _PATCH_FLAG, False)
        if getattr(FusedMoE, _DEFERRED_PATCH_FLAG, False):
            FusedMoE.forward_deferred_finalize = (
                FusedMoE._cacheon_orig_forward_deferred_finalize
            )
            delattr(FusedMoE, "_cacheon_orig_forward_deferred_finalize")
            setattr(FusedMoE, _DEFERRED_PATCH_FLAG, False)

    finalizer_mod = sys.modules.get(_FINALIZER_MODULE)
    if finalizer_mod is not None and getattr(
        finalizer_mod, _FINALIZER_PATCH_FLAG, False
    ):
        setattr(
            finalizer_mod,
            _FINALIZER_FUNC,
            finalizer_mod._cacheon_orig_moe_deferred_finalize,
        )
        delattr(finalizer_mod, "_cacheon_orig_moe_deferred_finalize")
        setattr(finalizer_mod, _FINALIZER_PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is None:
        return False
    installed = bool(getattr(FusedMoE, _PATCH_FLAG, False))
    if hasattr(FusedMoE, "forward_deferred_finalize"):
        installed = installed and bool(
            getattr(FusedMoE, _DEFERRED_PATCH_FLAG, False)
        )
    finalizer_mod = sys.modules.get(_FINALIZER_MODULE)
    if finalizer_mod is not None and callable(
        getattr(finalizer_mod, _FINALIZER_FUNC, None)
    ):
        installed = installed and bool(
            getattr(finalizer_mod, _FINALIZER_PATCH_FLAG, False)
        )
    return installed

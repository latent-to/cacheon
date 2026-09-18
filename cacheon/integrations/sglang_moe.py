"""Install both MoE slots at the pinned FusedMoE waist."""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from cacheon import receipts as _receipts

from cacheon.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_cacheon_moe_patched"
_DEFERRED_PATCH_FLAG = "_cacheon_moe_deferred_patched"
_MODULE = "sglang.srt.layers.moe.fused_moe_triton.layer"
_FINALIZER_MODULE = "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm"
_FINALIZER_PATCH_FLAG = "_cacheon_moe_deferred_finalize_patched"
_FINALIZER_FUNC = "finalize_flashinfer_trtllm_deferred_output"


def _record_moe_audit(slot, out, expected):
    """Audit the real rows of the DP projection consumed by this MoE call.

    The 2026-09-18 retained bundle matched every real token but failed on NaNs
    in an idle rank's padding. Reuse the producer's row domain and leave buffers intact.
    """
    from cacheon import audit
    from cacheon.integrations.sglang_dp_output import _gathered_audit_rows, _scope

    actual = (out,) if torch.is_tensor(out) else tuple(out)
    state = _scope.get()
    if state is not None and state.quantized is not None:
        from sglang.srt.distributed import get_tp_group

        group = get_tp_group()
        counts = state.batch.original_global_num_tokens_cpu
        if counts is None or any(e is None for e in expected):
            return audit.baseline_refused(slot)
        rows = state.quantized[0].shape[0] // group.world_size
        actual = _gathered_audit_rows(actual, counts, rows, group.world_size)
        expected = _gathered_audit_rows(expected, counts, rows, group.world_size)
        if not actual[0].numel():
            return audit.baseline_refused(slot)
    audit.record(slot, actual, expected)


def _moe_prepared(self, impl, slot, extra_prepare_args=()):
    """Run ``prepare`` once per implementation on this layer's expert weights.

    A layer may route different shapes to different variants.  A single layer-wide
    prepared object would hand variant B the layout produced by variant A, so the
    cache identity includes the slot, bundle, variant, and callable.  The slot's
    ``prepare_from_layer`` (validator-owned) maps the live sglang layer to the prepare
    call shape — weights + biases + layout flags — so the miner owns only the
    transform. ``extra_prepare_args`` appends validator-owned static routing config
    (the fat slot's top_k + routed_scaling, read from the live TopKConfig)."""
    cache = getattr(self, "_cacheon_moe_prepared_by_impl", None)
    if cache is None:
        cache = {}
        self._cacheon_moe_prepared_by_impl = cache
    key = (slot, impl.bundle_id, impl.variant, id(impl.prepare))
    if key not in cache:
        from cacheon.slots import get_slot

        spec = get_slot(slot)
        if spec.prepare_from_layer is not None:
            args = spec.prepare_from_layer(self)
        else:
            args = (self.w13_weight.data, self.w2_weight.data)
        started = time.perf_counter()
        marker = {"slot": slot, "pid": os.getpid(), "started_unix": time.time()}
        print("CACHEON-PREPARE: " + json.dumps(dict(marker, state="started")),
              file=sys.stderr, flush=True)
        state = "failed"
        try:
            cache[key] = _receipts.invoke(
                slot, impl.prepare, *args, *extra_prepare_args, phase="prepare"
            )
            state = "completed"
        finally:
            print("CACHEON-PREPARE: " + json.dumps(dict(
                marker, state=state, elapsed_seconds=time.perf_counter() - started)),
                file=sys.stderr, flush=True)
    return cache[key]


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch whichever exact MoE consumers have finished importing."""
    from cacheon.dispatch import (
        make_moe_deferred_dispatcher,
        make_moe_deferred_finalize_dispatcher,
        make_moe_dispatcher,
    )

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

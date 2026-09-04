"""The validator-owned dispatcher.

This is the only place a miner kernel is actually invoked during serving. It is
written so that the *validator* owns everything risky around the call:

It owns output allocation, eligibility, routing, and the auditable candidate
call. Miner code receives validator-allocated tensors and fills the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from cacheon import audit as _audit
from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor, collective_call_descriptor
from cacheon.moe_nvfp4_contract import (
    call_descriptor as moe_call_descriptor,
    routed_call_descriptor as routed_moe_call_descriptor,
    supports_layer as supports_nvfp4_moe_layer,
)
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.slots import get_slot
from cacheon.tensor_spec import (
    allocate_output_spec,
    tensor_bindings,
    validate_output_allocation,
    validate_tensor_bindings,
)

logger = logging.getLogger("cacheon.dispatch")
_MOE_LOGGED_ACTIVE = False
_MOE_OUTER_REDUCE_ATTR = "_cacheon_outer_reduce_owner"
_MOE_OUTER_REDUCE_TOKEN = object()
_MOE_REDUCED_ATTR = "_cacheon_already_reduced"
_MOE_REDUCED_TOKEN = object()
_MOE_REDUCE_MODES = frozenset({"immediate", "deferred"})


def _moe_reduce_owner(layer: object) -> str | None:
    state = getattr(layer, _MOE_OUTER_REDUCE_ATTR, None)
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and state[0] is _MOE_OUTER_REDUCE_TOKEN
        and state[1] in _MOE_REDUCE_MODES
    ):
        return state[1]
    return None


def _clear_moe_reduced(output: object) -> None:
    if torch.is_tensor(output) and hasattr(output, _MOE_REDUCED_ATTR):
        delattr(output, _MOE_REDUCED_ATTR)


def _consume_moe_reduced(output: object, mode: str) -> bool:
    state = getattr(output, _MOE_REDUCED_ATTR, None) if torch.is_tensor(output) else None
    if state is None:
        return False
    if (
        not isinstance(state, tuple)
        or len(state) != 3
        or state[0] is not _MOE_REDUCED_TOKEN
        or state[1] != mode
        or state[2] is not _tp_device_group()
    ):
        raise RuntimeError("invalid reduce-owning MoE completion context")
    delattr(output, _MOE_REDUCED_ATTR)
    _log_once_active("moe.fused_experts_reduce")
    _receipts.completed("moe.fused_experts_reduce")
    return True


def _arch_tag(device_index: int = 0) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


def _runtime_parallel_sizes() -> tuple[Optional[int], Optional[int]]:
    """Return validator-observed ``(tp_size, world_size)`` when initialized.

    The MSA function itself carries no model-runner object, so these values come
    only from sglang's already-initialized parallel-state authority.  Import or
    initialization failure means unknown (descriptor fields omitted), never a
    guessed value from environment variables.
    """

    tp_size: Optional[int] = None
    world_size: Optional[int] = None
    try:
        from sglang.srt.distributed import parallel_state as ps

        try:
            tp_size = int(ps.get_tensor_model_parallel_world_size())
        except Exception:  # noqa: BLE001 - parallel state may not be initialized
            pass
        try:
            world_size = int(ps.get_world_size())
        except Exception:  # noqa: BLE001 - parallel state may not be initialized
            pass
    except Exception:  # noqa: BLE001 - CPU/unit environments need no sglang import
        pass
    return tp_size, world_size


def _process_group_size(group: object) -> Optional[int]:
    """Return the size of the actual device group, never a caller's claim.

    Sglang exposes a real ``torch.distributed.ProcessGroup`` at every collective
    seam.  Capability routing must describe that authority: trusting a coordinator
    or layer's cached ``world_size``/``moe_tp_size`` alone could select a kernel for
    a topology different from the group it is handed.
    """

    if group is None:
        return None
    try:
        size = int(group.size())
    except Exception:  # noqa: BLE001 - backend/version-specific ProcessGroup surface
        try:
            import torch.distributed as dist

            size = int(dist.get_world_size(group=group))
        except Exception:  # noqa: BLE001 - unknown group means stock, never a guess
            return None
    return size if size > 0 else None


def _allreduce_group_role(coordinator: object, group: object) -> Optional[str]:
    """Classify only model-parallel coordinators the all-reduce slot promises.

    ``GroupCoordinator.all_reduce`` is a class-wide chokepoint also used by DP, EP,
    PP, and control groups. Calling their size ``tp_size`` would be false capability
    authority. Accept the pinned runtime's full/attention/MoE TP coordinators; every
    other role stays stock until the contract has a role-aware topology vocabulary.
    """

    try:
        from sglang.srt.distributed import parallel_state as ps
    except Exception:  # noqa: BLE001 - no trusted role authority
        return None
    for role, getter_name in (
        ("tp", "get_tp_group"),
        ("attn_tp", "get_attn_tp_group"),
        ("moe_tp", "get_moe_tp_group"),
    ):
        try:
            candidate = getattr(ps, getter_name)()
        except Exception:  # noqa: BLE001 - uninitialized/unsupported role
            continue
        if candidate is coordinator or getattr(candidate, "device_group", None) is group:
            return role
    return None


def _elementwise_descriptor(x: torch.Tensor, *, last_dim: int) -> CallDescriptor:
    """Describe an elementwise op call the way the registry expects it.

    ``num_tokens`` is everything but the last dimension. The legacy router used
    to omit it, which silently ignored a declared token window: a kernel scoped
    to decode-sized batches was routed on prefill-sized ones anyway.
    """

    return CallDescriptor.from_legacy(
        dtype_name=_dtype_name(x.dtype),
        last_dim=last_dim,
        arch=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        num_tokens=int(x.numel() // last_dim) if last_dim else 0,
    )


def _collective_call_descriptor(
    x: torch.Tensor,
    *,
    group_size: int,
    quant: str = "dense",
    **fields: object,
):
    """Canonical facts shared by every live Sglang collective binding.

    Model identity and request phase are intentionally absent: these call sites do
    not own a stable canonical value for either.  A future arena manifest can add
    that trusted context; guessing from a local model path would make validators
    disagree.  Missing constrained fields therefore remain safely out of domain.
    """

    return collective_call_descriptor(
        dtype=_dtype_name(x.dtype),
        architecture=(
            _arch_tag(x.device.index or 0) if x.is_cuda else None
        ),
        graph_mode="cuda_graph" if _in_cuda_graph() else "eager",
        quant=quant,
        world_size=group_size,
        dimensions={
            "num_tokens": int(x.shape[0]),
            "hidden_dim": int(x.shape[-1]),
            "last_dim": int(x.shape[-1]),
            **fields,
        },
    )


def _allocate_live_outputs(slot_name: str, inputs: dict, *, like: torch.Tensor):
    """Allocate a slot's live outputs from the same ABI used by verification.

    Return the resolved contract and allocation together so any validator-owned
    workspace stays alive until the candidate call and post-call validation finish.
    """

    contract = get_slot(slot_name).output_contract(inputs)
    tensor_inputs = tuple(value for value in inputs.values() if torch.is_tensor(value))
    input_bindings = tensor_bindings(tensor_inputs)
    allocation = allocate_output_spec(
        contract,
        fallback_dtype=like.dtype,
        fallback_device=like.device,
        inputs=tensor_inputs,
    )
    return contract, allocation, tensor_inputs, input_bindings


def _validate_live_outputs(
    contract,
    allocation,
    tensor_inputs,
    input_bindings,
    *,
    like: torch.Tensor,
) -> None:
    """Re-check the ABI after miner Python had an opportunity to rebind buffers."""

    validate_tensor_bindings(tensor_inputs, input_bindings, kind="candidate input")
    validate_output_allocation(
        contract,
        allocation,
        fallback_dtype=like.dtype,
        fallback_device=like.device,
        inputs=tensor_inputs,
    )


def make_silu_and_mul_dispatcher(
    baseline_forward: Callable[[object, torch.Tensor], torch.Tensor],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "activation.silu_and_mul",
) -> Callable[[object, torch.Tensor], torch.Tensor]:
    """Build a replacement for ``SiluAndMul.forward_*``.

    ``baseline_forward`` is the captured original (used for fallback). The
    returned function has the same ``(self, x)`` signature.
    """

    def dispatched(self: object, x: torch.Tensor) -> torch.Tensor:
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return baseline_forward(self, x)
        last_dim = x.shape[-1]
        impl = registry.select(
            slot, _elementwise_descriptor(x, last_dim=last_dim)
        ).impl
        if impl is None:
            return baseline_forward(self, x)

        d = last_dim // 2
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        aud = _audit.sampled()
        a_x = x.clone() if aud else None  # pre-call clone: the kernel may scribble on x
        _receipts.invoke(slot, impl.entry, x, out)
        if aud:
            _audit.run(slot, (out,), lambda: baseline_forward(self, a_x))
        _receipts.completed(slot)
        return out

    return dispatched


def make_rmsnorm_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "norm.rmsnorm",
    fused_slot: str = "norm.fused_add_rmsnorm",
) -> Callable[..., object]:
    """Build a replacement for ``RMSNorm.forward_cuda`` / ``forward_native``.

    sglang's RMSNorm has two modes: plain (``residual is None`` -> return normed)
    and fused add+norm (``residual`` given -> return ``(normed, x+residual)``).
    The validator owns the residual add; the miner kernel only ever computes the
    pure rmsnorm: ``entry(x, weight, out, eps)``. Unusual paths fall back to the
    trusted baseline.
    """

    def stock(self, x, residual, post_residual_addition, quant_linear):
        if quant_linear is None:
            return baseline_forward(self, x, residual, post_residual_addition)
        return baseline_forward(
            self, x, residual, post_residual_addition, quant_linear=quant_linear
        )

    def dispatched(
        self, x, residual=None, post_residual_addition=None, quant_linear=None
    ):
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return stock(self, x, residual, post_residual_addition, quant_linear)
        # Rare / semantic-override paths -> trusted baseline (keeps the contract simple
        # & safe): fp32 residual, a variance computed over a prefix subset of the hidden
        # dim (variance_size_override), or HF cast-before-multiply semantics
        # (cast_x_before_out_mul) are all NOT the pure rmsnorm the slot contract states.
        if (quant_linear is not None or post_residual_addition is not None
                or getattr(self, "fp32_residual", False)
                or getattr(self, "variance_size_override", None) is not None
                or getattr(self, "cast_x_before_out_mul", False)):
            return stock(self, x, residual, post_residual_addition, quant_linear)

        descriptor = _elementwise_descriptor(x, last_dim=x.shape[-1])
        fused_impl = (
            registry.select(fused_slot, descriptor).impl
            if residual is not None
            else None
        )
        impl = fused_impl or registry.select(slot, descriptor).impl
        if impl is None:
            return stock(self, x, residual, post_residual_addition, quant_linear)

        eps = float(self.variance_epsilon)
        weight = self.weight.data
        aud = _audit.sampled()
        if residual is None:
            a_x = x.clone() if aud else None
            out = torch.empty_like(x)
            _receipts.invoke(slot, impl.entry, x, weight, out, eps)
            if aud:
                _audit.run(slot, (out,), lambda: baseline_forward(self, a_x, None, None))
            _receipts.completed(slot)
            return out
        if fused_impl is not None:
            a_x, a_res = (x.clone(), residual.clone()) if aud else (None, None)
            out = torch.empty_like(x)
            new_residual = torch.empty_like(residual)
            _receipts.invoke(
                fused_slot,
                fused_impl.entry,
                x,
                residual,
                weight,
                eps,
                out,
                new_residual,
            )
            if aud:
                _audit.run(
                    fused_slot,
                    (out, new_residual),
                    lambda: baseline_forward(self, a_x, a_res, None),
                )
            _receipts.completed(fused_slot)
            return out, new_residual
        a_x, a_res = (x.clone(), residual.clone()) if aud else (None, None)
        new_residual = x + residual  # validator owns the add
        out = torch.empty_like(new_residual)
        _receipts.invoke(slot, impl.entry, new_residual, weight, out, eps)
        if aud:
            _audit.run(slot, (out, new_residual),
                       lambda: baseline_forward(self, a_x, a_res, None))
        _receipts.completed(slot)
        return out, new_residual

    return dispatched


def make_moe_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slots: tuple[str, ...] = ("moe.fused_experts_reduce", "moe.fused_experts"),
) -> Callable[..., object]:
    """Wrap the backend-neutral ``FusedMoE.forward_impl`` chokepoint.

    The validator owns routing, weights and output allocation. EP/DP and
    unsupported domains remain stock. The plain slot replays the trusted TP
    reduce; the reduce-owning slot receives the real process group.
    """

    def stock(self, hidden_states, topk_output, pre_quant_input):
        if pre_quant_input is None:
            return baseline_forward(self, hidden_states, topk_output)
        return baseline_forward(
            self, hidden_states, topk_output, pre_quant_input=pre_quant_input
        )

    def dispatched(self, hidden_states, topk_output, pre_quant_input=None):
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return stock(self, hidden_states, topk_output, pre_quant_input)
        active_moe = registry.active and any(
            registry.variants(slot) for slot in (*slots, _ROUTED_MOE_SLOT)
        )
        if _moe_seam_active() and active_moe:
            if not (_moe_supported(self) and hidden_states.dim() == 2):
                pass
            else:
                in_graph = _in_cuda_graph()
                quant_fmt = _moe_quant_format(self)
                routed = _standard_topk(topk_output)
                if routed is None:
                    # BYPASSED routing (e.g. flashinfer_trtllm hands router LOGITS
                    # only): the fat routed-MoE slot is that boundary.
                    result = _try_routed_moe(
                        self,
                        hidden_states,
                        topk_output,
                        registry,
                        in_graph=in_graph,
                        quant_fmt=quant_fmt,
                        stock_reference=lambda audit_x: stock(
                            self, audit_x, topk_output, pre_quant_input
                        ),
                    )
                    if result is not None:
                        return result
                else:
                    x = hidden_states
                    for slot in slots:
                        if not registry.variants(slot):
                            continue
                        reduce_slot = slot.endswith(".fused_experts_reduce")
                        group = None
                        descriptor = None
                        if reduce_slot:
                            topk_ids, topk_weights = routed
                            if quant_fmt not in {None, "nvfp4"} or (
                                quant_fmt == "nvfp4"
                                and not supports_nvfp4_moe_layer(self)
                            ):
                                continue
                            if not (
                                x.is_contiguous()
                                and torch.is_tensor(topk_ids)
                                and torch.is_tensor(topk_weights)
                                and topk_ids.dim() == 2
                                and topk_weights.dim() == 2
                                and tuple(topk_ids.shape) == tuple(topk_weights.shape)
                                and topk_ids.shape[0] == x.shape[0]
                                and topk_ids.dtype == torch.int32
                                and topk_weights.dtype == torch.float32
                                and topk_ids.is_contiguous()
                                and topk_weights.is_contiguous()
                                and topk_ids.device == x.device
                                and topk_weights.device == x.device
                            ):
                                continue
                            if not (
                                getattr(self, "reduce_results", False)
                                or _moe_reduce_owner(self) is not None
                            ):
                                continue
                            group = _tp_device_group()
                            group_size = _process_group_size(group)
                            if group_size is None or group_size <= 1:
                                continue
                            dimensions = {
                                "ep_size": int(getattr(self, "moe_ep_size", 1)),
                                "top_k": int(topk_ids.shape[-1]),
                            }
                            num_experts = getattr(self, "num_local_experts", None)
                            if (
                                isinstance(num_experts, int)
                                and not isinstance(num_experts, bool)
                                and num_experts >= 0
                            ):
                                dimensions["num_experts"] = num_experts
                            intermediate = getattr(
                                self, "intermediate_size_per_partition", None
                            )
                            if (
                                isinstance(intermediate, int)
                                and not isinstance(intermediate, bool)
                                and intermediate >= 0
                            ):
                                dimensions["intermediate_dim"] = intermediate
                            descriptor = _collective_call_descriptor(
                                x,
                                group_size=group_size,
                                quant=quant_fmt or "dense",
                                **dimensions,
                            )
                            decision = registry.select(slot, descriptor)
                            impl = decision.impl
                        else:
                            if (
                                quant_fmt == "nvfp4"
                                and not supports_nvfp4_moe_layer(self)
                            ):
                                continue
                            topk_ids, _topk_weights = routed
                            tp_size, world_size = _runtime_parallel_sizes()
                            w13 = self.w13_weight.data
                            w2 = self.w2_weight.data
                            descriptor = moe_call_descriptor(
                                x,
                                topk_ids,
                                architecture=(
                                    _arch_tag(x.device.index or 0)
                                    if x.is_cuda else None
                                ),
                                graph_mode="cuda_graph" if in_graph else "eager",
                                quant=quant_fmt or "dense",
                                num_experts=int(w13.shape[0]),
                                intermediate_dim=int(
                                    getattr(
                                        self,
                                        "intermediate_size_per_partition",
                                        w2.shape[-1] * (2 if quant_fmt else 1),
                                    )
                                ),
                                tp_size=tp_size,
                                world_size=world_size,
                            )
                            impl = registry.select(slot, descriptor).impl
                        if impl is None:
                            continue
                        if impl.prepare is None:
                            raise RuntimeError(
                                f"selected MoE candidate for {slot} has no prepare"
                            )
                        if reduce_slot:
                            # Commit the routing receipt only after every non-miner
                            # preflight gate passed. Registry state is immutable in a
                            # live engine, so identity must remain exact.
                            committed = registry.select(slot, descriptor)
                            if committed.impl is not impl:
                                raise RuntimeError(
                                    "collective selection changed between "
                                    "preflight and commit"
                                )
                        else:
                            committed = registry.select(slot, descriptor)
                            if committed.impl is not impl:
                                raise RuntimeError(
                                    "MoE selection changed between preflight and commit"
                                )
                        # Audit: baseline forward_impl on a pre-call clone (its TP
                        # reduce is collective — rank-seeded sampling keeps lockstep).
                        # Both sides are post-reduce here (the kernel path replays the
                        # validator reduce for plain fused_experts), so comparable.
                        aud = not in_graph and _audit.sampled()
                        a_x = x.clone() if aud else None
                        out = _run_moe_kernel(
                            self,
                            x,
                            routed,
                            impl,
                            slot,
                            group=group,
                        )
                        if aud:
                            def stock_reference():
                                stock_result = stock(
                                    self, a_x, topk_output, pre_quant_input
                                )
                                if reduce_slot and _moe_reduce_owner(self) is not None:
                                    from sglang.srt.distributed.communication_op import (
                                        tensor_model_parallel_all_reduce,
                                    )

                                    stock_result = tensor_model_parallel_all_reduce(
                                        stock_result
                                    )
                                return stock_result

                            _audit.run(
                                slot,
                                (out,) if torch.is_tensor(out) else tuple(out),
                                stock_reference,
                            )
                        if not (reduce_slot and _moe_reduce_owner(self) is not None):
                            _log_once_active(slot)
                            _receipts.completed(slot)
                        return out
        return stock(self, hidden_states, topk_output, pre_quant_input)

    return dispatched


@dataclass(frozen=True)
class _DeferredRoutedMoEOutput:
    """Candidate output carried across SGLang's stock shared-expert join."""

    routed: torch.Tensor


def make_moe_deferred_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
) -> Callable[..., object]:
    """Wrap the FlashInfer TRT-LLM deferred routed-expert path.

    SGLang bypasses ``FusedMoE.forward_impl`` when separate shared experts can
    fuse with the routed-expert finalize. A selected fat-slot candidate already
    returns the scaled, finalized routed output, so carry it to the stock join;
    the companion finalizer wrapper adds the separately computed shared output.
    """

    def dispatched(self, hidden_states, topk_output):
        if _dynamo_compiling():
            return baseline_forward(self, hidden_states, topk_output)
        if (
            _moe_seam_active()
            and registry.active
            and registry.variants(_ROUTED_MOE_SLOT)
            and _moe_supported(self)
            and hidden_states.dim() == 2
        ):
            result = _try_routed_moe(
                self,
                hidden_states,
                topk_output,
                registry,
                in_graph=_in_cuda_graph(),
                quant_fmt=_moe_quant_format(self),
                stock_reference=None,
                defer_completion=True,
            )
            if result is not None:
                return _DeferredRoutedMoEOutput(result)
        return baseline_forward(self, hidden_states, topk_output)

    return dispatched


def make_moe_deferred_finalize_dispatcher(
    baseline_finalize: Callable[..., object],
) -> Callable[..., object]:
    """Join a selected routed candidate with SGLang's separate shared expert."""

    def dispatched(deferred_output, shared_output):
        if not isinstance(deferred_output, _DeferredRoutedMoEOutput):
            return baseline_finalize(deferred_output, shared_output)
        routed = deferred_output.routed
        if not (
            torch.is_tensor(shared_output)
            and tuple(shared_output.shape) == tuple(routed.shape)
            and shared_output.dtype == routed.dtype
            and shared_output.device == routed.device
        ):
            raise RuntimeError(
                "deferred MoE shared output does not match the selected routed output"
            )
        routed.add_(shared_output)
        _log_once_active(_ROUTED_MOE_SLOT)
        _receipts.completed(_ROUTED_MOE_SLOT)
        return routed

    return dispatched


def _log_once_active(slot: str) -> None:
    global _MOE_LOGGED_ACTIVE
    if not _MOE_LOGGED_ACTIVE:
        _MOE_LOGGED_ACTIVE = True
        logger.warning("cacheon: MoE seam ACTIVE — experts routed through miner kernel (slot=%s)", slot)
def _moe_seam_active() -> bool:
    import os

    return os.environ.get("CACHEON_MOE_SEAM") == "1"


def _moe_supported(self) -> bool:
    """Admit local-expert/TP layers; EP/DP boundaries remain stock."""
    if getattr(self, "moe_ep_size", 1) != 1:
        return False
    if _moe_data_parallel_world_size() != 1:
        return False
    if not (hasattr(self, "w13_weight") and hasattr(self, "w2_weight")):
        return False
    return True


def _moe_quant_format(self) -> Optional[str]:
    """Classify dense, FP8, or packed-uint8 NVFP4 expert weights."""
    if not (hasattr(self, "w13_weight_scale") or hasattr(self, "w2_weight_scale")):
        return None
    w = getattr(getattr(self, "w13_weight", None), "data", None)
    dt = getattr(w, "dtype", None)
    if dt in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return "fp8"
    return "nvfp4" if dt == torch.uint8 else None


def _dynamo_compiling() -> bool:
    """Keep registry machinery out of Dynamo-traced regions."""
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - older torch without torch.compiler
        return False


def _in_cuda_graph() -> bool:
    """Probe pinned, legacy, then direct CUDA capture authorities."""

    detectors = []
    try:
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            is_in_tc_piecewise_cuda_graph,
        )

        detectors.append(is_in_tc_piecewise_cuda_graph)
    except Exception:  # noqa: BLE001 - older pin or CPU-only unit environment
        pass
    try:
        from sglang.srt.compilation.piecewise_context_manager import (
            is_in_piecewise_cuda_graph,
        )

        detectors.append(is_in_piecewise_cuda_graph)
    except Exception:  # noqa: BLE001 - current pin removed this legacy module
        pass
    for detector in detectors:
        try:
            if bool(detector()):
                return True
        except Exception:  # noqa: BLE001 - continue to independent CUDA authority
            pass
    try:
        return bool(
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:  # noqa: BLE001 - CPU/initialization failure means eager
        return False


# Capture detection has exactly one authority and it is the one above. Receipts
# record whether a candidate was invoked inside a capture — the fact that decides
# whether the scored replays run its code or stock — and get it from here rather
# than growing a second, quietly divergent detector.
_receipts.set_graph_probe(_in_cuda_graph)


def _standard_topk(topk_output):
    """Return ``(topk_ids, topk_weights)`` iff routing is already materialized (the
    STANDARD format), else None. BypassedTopKOutput / TritonKernelTopKOutput don't carry
    explicit topk tensors -> fall back (conservative; no implicit re-routing here)."""
    topk_ids = getattr(topk_output, "topk_ids", None)
    topk_weights = getattr(topk_output, "topk_weights", None)
    if topk_ids is None or topk_weights is None:
        return None
    return topk_ids, topk_weights


_ROUTED_MOE_SLOT = "moe.fused_routed_experts"


def _bypassed_routing_contract(topk_output):
    """Return ``(router_logits, correction_bias, top_k, routed_scaling)`` iff the
    BYPASSED routing config matches the fat slot's contract EXACTLY, else None.

    The slot's routing head is fixed: sigmoid scoring, selection on scores+bias,
    weights from the unbiased scores renormalized then scaled in the routed output,
    no expert grouping, no fused shared experts, no custom routing. An engine config
    outside that domain stays stock — the dispatcher never approximates routing
    semantics."""
    router_logits = getattr(topk_output, "router_logits", None)
    cfg = getattr(topk_output, "topk_config", None)
    if cfg is None or not torch.is_tensor(router_logits):
        return None
    bias = getattr(cfg, "correction_bias", None)
    top_k = getattr(cfg, "top_k", None)
    scaling = getattr(cfg, "routed_scaling_factor", None)
    if not torch.is_tensor(bias):
        return None
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return None
    if not isinstance(scaling, (int, float)) or isinstance(scaling, bool):
        return None
    if getattr(cfg, "scoring_func", None) != "sigmoid":
        return None
    if not getattr(cfg, "renormalize", False):
        return None
    if getattr(cfg, "apply_routed_scaling_factor_on_output", None) is not True:
        return None
    if getattr(cfg, "custom_routing_function", None) is not None:
        return None
    if int(getattr(cfg, "num_fused_shared_experts", 0) or 0) != 0:
        return None
    if getattr(cfg, "use_grouped_topk", False):
        if getattr(cfg, "num_expert_group", None) not in (None, 0, 1):
            return None
        if getattr(cfg, "topk_group", None) not in (None, 0, 1):
            return None
    return router_logits, bias, int(top_k), float(scaling)


def _try_routed_moe(
    self,
    x,
    topk_output,
    registry,
    *,
    in_graph,
    quant_fmt,
    stock_reference,
    defer_completion: bool = False,
):
    """Dispatch a BYPASSED-routing batch into the fat routed-MoE slot.

    Returns the combined output, or None when no registered implementation (or an
    out-of-contract routing config) applies — the caller then serves stock."""
    fat = _bypassed_routing_contract(topk_output)
    if fat is None:
        return None
    if quant_fmt == "nvfp4" and not supports_nvfp4_moe_layer(self):
        return None
    router_logits, correction_bias, top_k, routed_scaling = fat
    tp_size, world_size = _runtime_parallel_sizes()
    w13 = self.w13_weight.data
    w2 = self.w2_weight.data
    descriptor = routed_moe_call_descriptor(
        x,
        top_k=top_k,
        architecture=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        graph_mode="cuda_graph" if in_graph else "eager",
        quant=quant_fmt or "dense",
        num_experts=int(w13.shape[0]),
        intermediate_dim=int(
            getattr(
                self,
                "intermediate_size_per_partition",
                w2.shape[-1] * (2 if quant_fmt else 1),
            )
        ),
        tp_size=tp_size,
        world_size=world_size,
    )
    impl = registry.select(_ROUTED_MOE_SLOT, descriptor).impl
    if impl is None:
        return None
    if impl.prepare is None:
        raise RuntimeError(
            f"selected MoE candidate for {_ROUTED_MOE_SLOT} has no prepare"
        )
    committed = registry.select(_ROUTED_MOE_SLOT, descriptor)
    if committed.impl is not impl:
        raise RuntimeError("MoE selection changed between preflight and commit")
    aud = not defer_completion and not in_graph and _audit.sampled()
    a_x = x.clone() if aud else None
    out = _run_routed_moe_kernel(
        self,
        x,
        router_logits,
        correction_bias,
        impl,
        top_k=top_k,
        routed_scaling=routed_scaling,
    )
    if aud:
        assert stock_reference is not None
        _audit.run(
            _ROUTED_MOE_SLOT,
            (out,) if torch.is_tensor(out) else tuple(out),
            lambda: stock_reference(a_x),
        )
    if not defer_completion:
        _log_once_active(_ROUTED_MOE_SLOT)
        _receipts.completed(_ROUTED_MOE_SLOT)
    return out


def _run_routed_moe_kernel(
    self, x, router_logits, correction_bias, impl, *, top_k, routed_scaling
):
    """Run the fat routed-experts contract into validator-allocated output."""
    live_inputs = {
        "x": x,
        "router_logits": router_logits,
        "correction_bias": correction_bias,
    }
    contract, allocation, tensor_inputs, input_bindings = _allocate_live_outputs(
        _ROUTED_MOE_SLOT, live_inputs, like=x
    )
    if len(allocation.outputs) != 1:
        raise RuntimeError(f"{_ROUTED_MOE_SLOT} must declare exactly one live output")
    out = allocation.outputs[0]
    prepared = _moe_prepared(
        self, impl, _ROUTED_MOE_SLOT,
        extra_prepare_args=(top_k, routed_scaling),
    )
    _clear_moe_reduced(out)
    try:
        _receipts.invoke(
            _ROUTED_MOE_SLOT, impl.entry,
            x, router_logits, correction_bias, prepared, out,
        )
    finally:
        _clear_moe_reduced(out)
    _validate_live_outputs(
        contract, allocation, tensor_inputs, input_bindings, like=x
    )
    if getattr(self, "reduce_results", False) and getattr(self, "moe_tp_size", 1) > 1:
        from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(out)
    return out


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
        cache[key] = _receipts.invoke(
            slot, impl.prepare, *args, *extra_prepare_args, phase="prepare"
        )
    return cache[key]


@torch.inference_mode()
def _run_moe_kernel(
    self,
    x,
    routed,
    impl,
    slot,
    *,
    group=None,
):
    """Run the local-expert or reduce-owning expert contract into validator output."""
    topk_ids, topk_weights = routed
    live_inputs = {
        "x": x,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }
    contract, allocation, tensor_inputs, input_bindings = _allocate_live_outputs(
        slot, live_inputs, like=x
    )
    if len(allocation.outputs) != 1:
        raise RuntimeError(f"{slot} must declare exactly one live output")
    out = allocation.outputs[0]

    if slot.endswith(".fused_experts_reduce"):
        # The kernel does experts AND the cross-rank reduce. Hand it the TP group; do not
        # replay a second reduce. The caller resolved and verified this actual group
        # before committing selection.
        if group is None:
            raise RuntimeError("reduce-owning MoE kernel requires a live TP group")
        # ``prepare`` is miner code and receives the live layer's weight tensors.
        # The caller has already made this route non-recoverable before entering
        # any rank-local fallible prelude.
        prepared = _moe_prepared(self, impl, slot)
        _clear_moe_reduced(out)
        try:
            _receipts.invoke(slot, impl.entry, x, topk_ids, topk_weights, prepared, out, group)
        finally:
            _clear_moe_reduced(out)
        _validate_live_outputs(
            contract, allocation, tensor_inputs, input_bindings, like=x
        )
        owner = _moe_reduce_owner(self)
        if owner is not None:
            setattr(out, _MOE_REDUCED_ATTR, (_MOE_REDUCED_TOKEN, owner, group))
        return out

    prepared = _moe_prepared(self, impl, slot)
    _clear_moe_reduced(out)
    try:
        _receipts.invoke(slot, impl.entry, x, topk_ids, topk_weights, prepared, out)
    finally:
        _clear_moe_reduced(out)
    _validate_live_outputs(
        contract, allocation, tensor_inputs, input_bindings, like=x
    )
    if getattr(self, "reduce_results", False) and getattr(self, "moe_tp_size", 1) > 1:
        # Sum this rank's partial expert output across the TP group. A missing
        # collective is terminal after candidate selection.
        from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(out)
    return out


def _tp_device_group():
    """The exact group reduced by the replaced ``FusedMoE.forward_impl`` tail.

    That stock boundary calls ``tensor_model_parallel_all_reduce``, whose pinned
    Sglang implementation owns ``get_tp_group()``. This can differ from the layer's
    internal MoE-TP group under MoE-DP; the slot must follow the model-consumed stock
    product rather than a similarly named cached layer field."""
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = getattr(get_tp_group(), "device_group", None)
    except Exception as exc:  # noqa: BLE001 - preserve pinned-runtime cause
        raise RuntimeError("selected MoE target cannot resolve its TP group") from exc
    if group is None:
        raise RuntimeError("selected MoE target resolved no TP device group")
    return group


def _moe_data_parallel_world_size() -> int:
    """Return SGLang's authoritative MoE-DP size or fail the active target.

    ``FusedMoE.forward_impl`` surrounds its local expert core with dispatcher
    dispatch/combine when MoE data parallelism is active.  Cacheon's current
    ``(M,H)->(M,H)`` expert contracts replace that whole method but do not model
    those operations, so a layer attribute or environment hint cannot authorize
    the route. Missing or drifted pinned-runtime authority is terminal; serving
    stock would turn an active candidate into a phantom comparison.
    """

    try:
        from sglang.srt.distributed.parallel_state import (
            get_moe_data_parallel_world_size,
        )

        size = get_moe_data_parallel_world_size()
    except Exception as exc:  # noqa: BLE001 - preserve pinned-runtime cause
        raise RuntimeError(
            "selected MoE target cannot resolve MoE-DP topology"
        ) from exc
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise RuntimeError("selected MoE target received invalid MoE-DP topology")
    return size


_COLLECTIVE_LOGGED_ACTIVE = False


def make_allreduce_dispatcher(
    baseline_all_reduce: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.all_reduce",
) -> Callable[..., object]:
    """Build a replacement for ``GroupCoordinator.all_reduce`` — the single chokepoint
    every tensor-parallel reduce funnels through (``sglang.srt.distributed.parallel_state``).

    The TP all-reduce is the largest single category of decode time at scale (~32–43%),
    and it is latency-bound — so the lever is a lower-latency reduce or a compute-comm
    overlap, both expressible here. The validator owns the output buffer, the process
    group, and the call site; the miner only fills ``out`` with the cross-rank sum. The
    reduce is mid-network (upstream of the sampler) → no output to substitute.

    SCOPE (MVP, mirrors the other seams): only the multi-rank (``world_size > 1``)
    default SUM all-reduce of a 2D tensor, opt-in via ``CACHEON_COLLECTIVE_SEAM=1``.
    Extra args/kwargs or single-rank → trusted baseline. A registered candidate runs
    under CUDA-graph capture exactly as it runs eager; capture is the scoring config,
    so a kernel that cannot be captured fails loudly here. The miner gets
    the process group (``self.device_group``) — a wider capability than op/block slots —
    so this slot is verified DISTRIBUTED (cacheon.verify_collective) and the end-to-end
    gate is mandatory (docs/architecture/slot-contract.md).
    """

    def dispatched(self, input_, *args, **kwargs):
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return baseline_all_reduce(self, input_, *args, **kwargs)
        if _collective_seam_active() and not args and not kwargs:
            if (
                torch.is_tensor(input_)
                and input_.dim() == 2
                and input_.is_floating_point()
                and input_.is_contiguous()
            ):
                group = getattr(self, "device_group", None)
                group_size = _process_group_size(group)
                group_role = _allreduce_group_role(self, group)
                claimed_size = getattr(self, "world_size", None)
                if (
                    group_size is not None
                    and group_size > 1
                    and group_role is not None
                    and (
                        claimed_size is None
                        or (
                            isinstance(claimed_size, int)
                            and not isinstance(claimed_size, bool)
                            and claimed_size == group_size
                        )
                    )
                ):
                    descriptor = _collective_call_descriptor(
                        input_, group_size=group_size
                    )
                    impl = registry.select(
                        slot, descriptor
                    ).impl
                else:
                    impl = None
                if impl is not None:
                    # Audited baseline is COLLECTIVE (see arfusion note): rank-seeded
                    # sampling + lockstep dispatch make the extra reduce safe.
                    aud = not _in_cuda_graph() and _audit.sampled()
                    a_in = input_.clone() if aud else None
                    live_inputs = {"x": input_}
                    contract, allocation, tensor_inputs, input_bindings = (
                        _allocate_live_outputs(
                            slot, live_inputs, like=input_
                        )
                    )
                    if len(allocation.outputs) != 1:
                        raise RuntimeError(
                            f"{slot} must declare exactly one live output"
                        )
                    out = allocation.outputs[0]
                    committed = registry.select(slot, descriptor)
                    if committed.impl is not impl:
                        raise RuntimeError(
                            "collective selection changed between preflight and commit"
                        )
                    _receipts.invoke(slot, impl.entry, input_, out, group)  # miner fills out with sum-over-ranks
                    _validate_live_outputs(
                        contract,
                        allocation,
                        tensor_inputs,
                        input_bindings,
                        like=input_,
                    )
                    if aud:
                        _audit.run(slot, (out,), lambda: baseline_all_reduce(self, a_in))
                    _log_collective_active()
                    _receipts.completed(slot)
                    return out
        return baseline_all_reduce(self, input_, *args, **kwargs)

    return dispatched


def _collective_seam_active() -> bool:
    import os

    return os.environ.get("CACHEON_COLLECTIVE_SEAM") == "1"


def _log_collective_active() -> None:
    global _COLLECTIVE_LOGGED_ACTIVE
    if not _COLLECTIVE_LOGGED_ACTIVE:
        _COLLECTIVE_LOGGED_ACTIVE = True
        logger.warning("cacheon: collective.all_reduce seam ACTIVE — TP reduce routed through miner kernel")
# ---------------------------------------------------------------------------
# collective.ar_residual_rmsnorm — the fused AR+residual+RMSNorm epilogue waist
# ---------------------------------------------------------------------------

_ARFUSION_LOGGED_ACTIVE = False


def _flashinfer_tuning() -> bool:
    """Keep candidate collectives out of FlashInfer tactic profiling."""

    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:  # noqa: BLE001 - an absent tuner is ordinary stock behavior
        return False


def make_arfusion_dispatcher(
    baseline_fn: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.ar_residual_rmsnorm",
) -> Callable[..., object]:
    """Build a replacement for the MODULE-LEVEL function
    ``sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm``
    — sglang's own fused-epilogue waist. With ``--enable-flashinfer-allreduce-fusion``
    (an arena server flag) every participating layer epilogue funnels through this one
    function: the layer defers its TP all-reduce, and the next norm call performs
    AR + residual-add + RMSNorm fused. The call site resolves the symbol per call via a
    function-local import, so rebinding the module attribute reroutes every caller
    (the mechanism the 2026-07-02 M3 fused-epilogue campaign validated in production).

    The validator owns the call site, BOTH output buffers (norm_out, new_residual), and
    the process group; the miner owns the reduce transport + the fused add/norm math.
    Mid-network, upstream of the sampler — nothing to substitute. Stock signature and
    the ``Tuple[Tensor, Tensor]`` return are preserved exactly; any deviation from the
    plain path (extra semantics via kwargs, missing residual, non-2D input) falls back.

    SCOPE: 2D input with a residual, multi-rank group, opt-in via
    ``CACHEON_ARFUSION_SEAM=1``. Token-count dispatch windows (a kernel measured to win
    only at decode-sized T) are declared via eligibility ``max_num_tokens`` — oversized
    calls (prefill) route to the trusted baseline rather than trusting the kernel to
    decline.
    """

    def dispatched(input_tensor, residual, weight, eps=1e-6, max_token_num=2048,
                   use_oneshot=None, trigger_completion_at_end=False, fp32_acc=False,
                   use_attn_tp_group=True):
        # FIRST, before any Python machinery: inside a Dynamo trace this constant-folds
        # to True and the compiled piece bakes pure stock (see _dynamo_compiling — the
        # piecewise-prefill trace of this exact call site hard-errored otherwise).
        if _dynamo_compiling():
            return baseline_fn(input_tensor, residual, weight, eps, max_token_num,
                               use_oneshot, trigger_completion_at_end, fp32_acc,
                               use_attn_tp_group)
        if _arfusion_seam_active():
            # FlashInfer profiles stock tactics under this same epilogue call site.
            # Miner collectives must be invisible to that lifecycle: decide before
            # deep consume, capability preflight, or the receipted commit boundary.
            if _flashinfer_tuning():
                return baseline_fn(
                    input_tensor, residual, weight, eps, max_token_num, use_oneshot,
                    trigger_completion_at_end, fp32_acc, use_attn_tp_group
                )
            # Contiguity guard = STOCK PARITY: the stock function refuses
            # non-contiguous input/residual/weight (real call sites pass views —
            # upstream guards for it, flashinfer_comm_fusion.py). A raw-pointer
            # kernel fed a strided view reads the wrong layout silently; verify
            # can't see it (it always builds contiguous tensors), only the
            # engine's own call mix does.
            if (
                torch.is_tensor(input_tensor)
                and input_tensor.dim() == 2
                and input_tensor.is_floating_point()
                and torch.is_tensor(residual)
                and tuple(residual.shape) == tuple(input_tensor.shape)
                and residual.dtype == input_tensor.dtype
                and residual.device == input_tensor.device
                and torch.is_tensor(weight)
                and weight.dim() == 1
                and weight.shape[0] == input_tensor.shape[-1]
                and weight.dtype == input_tensor.dtype
                and weight.device == input_tensor.device
                and input_tensor.is_contiguous()
                and residual.is_contiguous()
                and weight.is_contiguous()
            ):
                group = _arfusion_group(use_attn_tp_group)
                group_size = _process_group_size(group)
                impl = None
                if (
                    group_size is not None
                    and group_size > 1
                    and _arfusion_group_role(use_attn_tp_group) == "tp"
                ):
                    descriptor = _collective_call_descriptor(
                        input_tensor, group_size=group_size
                    )
                    impl = registry.select(
                        slot, descriptor
                    ).impl
                if impl is not None:
                    # The audited baseline is COLLECTIVE: safe only because the
                    # sampling RNG is rank-identically seeded (audit.py) and all
                    # ranks reach this dispatcher in lockstep; never under capture.
                    aud = not _in_cuda_graph() and _audit.sampled()
                    if aud:
                        a_x, a_res = input_tensor.clone(), residual.clone()
                    live_inputs = {
                        "x": input_tensor,
                        "residual": residual,
                        "weight": weight,
                        "eps": float(eps),
                    }
                    contract, allocation, tensor_inputs, input_bindings = (
                        _allocate_live_outputs(
                            slot, live_inputs, like=input_tensor
                        )
                    )
                    if len(allocation.outputs) != 2:
                        raise RuntimeError(
                            f"{slot} must declare exactly two live outputs"
                        )
                    out_norm, out_residual = allocation.outputs
                    committed = registry.select(slot, descriptor)
                    if committed.impl is not impl:
                        raise RuntimeError(
                            "collective selection changed between preflight and commit"
                        )
                    _receipts.invoke(slot, impl.entry, input_tensor, residual, weight, float(eps),
                            out_norm, out_residual, group)
                    _validate_live_outputs(
                        contract,
                        allocation,
                        tensor_inputs,
                        input_bindings,
                        like=input_tensor,
                    )
                    if aud:
                        _audit.run(slot, (out_norm, out_residual),
                                   lambda: baseline_fn(a_x, a_res, weight, eps,
                                                       max_token_num, use_oneshot,
                                                       trigger_completion_at_end,
                                                       fp32_acc, use_attn_tp_group))
                    _log_arfusion_active()
                    _receipts.completed(slot)
                    return out_norm, out_residual
        return baseline_fn(input_tensor, residual, weight, eps, max_token_num,
                           use_oneshot, trigger_completion_at_end, fp32_acc,
                           use_attn_tp_group)

    return dispatched


def _arfusion_seam_active() -> bool:
    import os

    return os.environ.get("CACHEON_ARFUSION_SEAM") == "1"


# (The 2026-07-07 one-off "stockcheck" diagnostic that lived here was productized
# into cacheon/audit.py — the in-engine audit is the same mechanism, generic across
# dispatchers, receipted, and gated by the eval driver.)


def _arfusion_group(use_attn_tp_group: bool):
    """The torch ProcessGroup the stock call would reduce over. ``use_attn_tp_group``
    mirrors the stock argument (attention-TP vs full-TP chain); under plain TP the two
    coincide. Resolution failure -> None -> baseline (never guess a group)."""
    try:
        from sglang.srt.distributed import parallel_state as ps

        if use_attn_tp_group:
            coord = ps.get_attn_tp_group()
        elif int(ps.get_moe_expert_parallel_world_size()) > 1:
            coord = ps.get_moe_ep_group()
        else:
            coord = ps.get_moe_tp_group()
        return getattr(coord, "device_group", None)
    except Exception:  # noqa: BLE001 - a different group is never a safe fallback
        return None


def _arfusion_group_role(use_attn_tp_group: bool) -> Optional[str]:
    """Role of the exact stock fusion communicator, without guessing aliases."""

    if use_attn_tp_group:
        return "tp"
    try:
        from sglang.srt.distributed import parallel_state as ps

        return (
            "ep"
            if int(ps.get_moe_expert_parallel_world_size()) > 1
            else "tp"
        )
    except Exception:  # noqa: BLE001
        return None


def _log_arfusion_active() -> None:
    global _ARFUSION_LOGGED_ACTIVE
    if not _ARFUSION_LOGGED_ACTIVE:
        _ARFUSION_LOGGED_ACTIVE = True
        logger.warning(
            "cacheon: collective.ar_residual_rmsnorm seam ACTIVE — fused AR+norm epilogue routed through miner kernel")
def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, str(dtype).replace("torch.", ""))

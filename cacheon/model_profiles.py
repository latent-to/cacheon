"""Validator-owned model facts applied to the generic slot catalog."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from cacheon.capabilities import CallDescriptor
from cacheon.moe_nvfp4_contract import (
    call_descriptor as _moe_call_descriptor,
    prepare_args_from_inputs as _moe_prepare_args_from_inputs,
    routed_call_descriptor as _routed_moe_call_descriptor,
    verification_inputs as _moe_nvfp4_verification_inputs,
)
from cacheon.slots import (
    Activation,
    Correctness,
    SlotProfile,
    SlotSpec,
    _moe_reference,
    _routed_moe_reference,
    get_slot,
)

_MOE_SLOTS = ("moe.fused_experts", "moe.fused_experts_reduce")
_ROUTED_MOE_SLOTS = ("moe.fused_routed_experts",)


def specialize_slot(slot: SlotSpec, profile: SlotProfile) -> SlotSpec:
    """Retarget one generic slot with validator-owned model facts."""
    repl: dict = {}
    if profile.quant == "nvfp4" and profile.shapes is None:
        raise ValueError(
            f"profile for {slot.name!r} sets quant={profile.quant!r} but no shapes; "
            "a quantized profile must carry the arena's per-rank verification shapes"
        )
    if slot.name in _ROUTED_MOE_SLOTS:
        routed_act = profile.activation

        def _routed_ref(i, _act=routed_act):
            return [_routed_moe_reference(i, _act)]

        repl["invoke_reference"] = _routed_ref
        if profile.quant == "nvfp4":
            make_dense_routed = slot.make_inputs
            routed_fused = profile.num_fused_shared_experts

            def _routed_quant_inputs(**kwargs):
                dense = make_dense_routed(**kwargs)
                dense.update(
                    __moe_tp_size__=int(kwargs.get("world_size", 4)),
                    __moe_ep_size__=1,
                    __moe_ep_rank__=0,
                    __moe_reduce_results__=False,
                    __moe_num_fused_shared_experts__=routed_fused,
                    __moe_activation__=routed_act.kind,
                )
                return _moe_nvfp4_verification_inputs(dense)

            repl["make_inputs"] = _routed_quant_inputs
            repl["invoke_prepare"] = lambda prepare_fn, i: prepare_fn(
                *_moe_prepare_args_from_inputs(i), i["topk"], i["routed_scaling"]
            )
            repl["call_abi"] = None
    if slot.name in _MOE_SLOTS:
        act = profile.activation

        def _ref(i, _act=act):
            return [
                _moe_reference(
                    i["x"], i["w13"], i["w2"], i["topk_ids"],
                    i["topk_weights"], _act,
                )
            ]

        repl["invoke_reference"] = _ref
        if slot.collective_partial is not None:
            def _partial(i, prepared, _act=act):
                return _moe_reference(
                    i["x"], i["w13"], i["w2"], i["topk_ids"],
                    i["topk_weights"], _act,
                ).float()

            repl["collective_partial"] = _partial
    if profile.correctness is not None:
        repl["correctness"] = profile.correctness
    if slot.name in _MOE_SLOTS and profile.quant == "nvfp4":
        make_dense_inputs = slot.make_inputs
        fused = profile.num_fused_shared_experts
        scale = profile.routed_weight_scale
        act_kind = profile.activation.kind

        def _quant_inputs(**kwargs):
            dense = make_dense_inputs(**kwargs)
            tokens, top_k = dense["topk_ids"].shape
            experts = dense["w13"].shape[0]
            routed_k = top_k - fused
            generator = torch.Generator(device=kwargs["device"]).manual_seed(
                int(kwargs["seed"]) + 17_171
            )
            routed = torch.rand(
                tokens, experts - fused,
                generator=generator, device=kwargs["device"],
            ).topk(routed_k, dim=-1).indices.to(torch.int32)
            scores = torch.rand(
                tokens, routed_k, generator=generator, device=kwargs["device"]
            )
            routed_weights = scale * scores / scores.sum(-1, keepdim=True)
            if fused:
                shared_ids = torch.arange(
                    experts - fused, experts,
                    device=kwargs["device"], dtype=torch.int32,
                ).expand(tokens, fused)
                dense["topk_ids"] = torch.cat((routed, shared_ids), dim=-1)
                dense["topk_weights"] = torch.cat(
                    (routed_weights, torch.ones_like(scores[:, :fused])), dim=-1
                )
            else:
                dense["topk_ids"] = routed
                dense["topk_weights"] = routed_weights
            dense.update(
                __moe_tp_size__=int(kwargs.get("world_size", 4)),
                __moe_ep_size__=1,
                __moe_ep_rank__=0,
                __moe_reduce_results__=False,
                __moe_num_fused_shared_experts__=fused,
                __moe_activation__=act_kind,
            )
            return _moe_nvfp4_verification_inputs(dense)

        repl["make_inputs"] = _quant_inputs
        repl["invoke_prepare"] = lambda prepare_fn, i: prepare_fn(
            *_moe_prepare_args_from_inputs(i)
        )
        repl["call_abi"] = None
    if profile.shapes is not None:
        repl["shapes"] = profile.shapes
    return replace(slot, **repl) if repl else slot


_M3_MOE_PROFILE = SlotProfile(
    activation=Activation("swigluoai", alpha=1.702, limit=7.0),
    correctness=Correctness("cosine", min_cosine=0.985),
)
_M3_MOE_NVFP4_PROFILE = replace(
    _M3_MOE_PROFILE,
    quant="nvfp4",
    shapes=(
        {"num_tokens": 1, "num_experts": 129, "hidden": 6144, "inter": 768, "topk": 5},
        {"num_tokens": 8, "num_experts": 129, "hidden": 6144, "inter": 768, "topk": 5},
        {"num_tokens": 32, "num_experts": 129, "hidden": 6144, "inter": 768, "topk": 5},
    ),
    num_fused_shared_experts=1,
    routed_weight_scale=2.0,
)

_GLM53_MOE_NVFP4_PROFILE = SlotProfile(
    # Full GLM-5.3 at TP4: 256 routed experts, top-8, hidden 6144,
    # moe_intermediate 2048 -> 512/rank; shared experts remain separate.
    correctness=Correctness("cosine", min_cosine=0.985, max_rel_norm_err=0.05),
    quant="nvfp4",
    shapes=(
        {"num_tokens": 1, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
        {"num_tokens": 8, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
        {"num_tokens": 24, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
        {"num_tokens": 32, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
        {"num_tokens": 128, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
    ),
    num_fused_shared_experts=0,
    routed_weight_scale=2.5,
)

_GLM53_ROUTED_MOE_PROFILE = replace(
    _GLM53_MOE_NVFP4_PROFILE,
    shapes=tuple(
        {**shape, "routed_scaling": 2.5}
        for shape in _GLM53_MOE_NVFP4_PROFILE.shapes
    ),
)


_GLM53_DENSE_PROFILE = SlotProfile(
    shapes=tuple(
        {"num_tokens": m, "input_dim": k, "output_dim": n,
         "parallel_role": role, "local_tp_size": tp}
        for tokens, tp, matrices in (
            ((32, 4096), 1, ((6144, 2624, "replicated"),)),
            ((6, 32, 4096), 1, ((2048, 16384, "column"), (16384, 6144, "row"),
                                (2048, 4096, "replicated"), (6144, 160, "replicated"))),
            ((24, 128, 16384), 4, ((6144, 6144, "column"), (3072, 6144, "row"),
                                    (6144, 1024, "column"), (512, 6144, "row"))),
        )
        for m in tokens for k, n, role in matrices
    ),
)

_GLM53_NORM_PROFILE = SlotProfile(shapes=tuple(
    {"num_tokens": tokens, "hidden": 6144}
    for tokens in (6, 24, 32, 128, 4096, 16384)
))
_GLM53_ALL_REDUCE_PROFILE = SlotProfile(
    shapes=({"num_tokens": 16384, "hidden": 6144},),
)
_GLM53_DP_EXCHANGE_PROFILE = SlotProfile(shapes=tuple(
    {"num_tokens": tokens, "hidden": 6144} for tokens in (6, 32)
))

MODEL_PROFILES: dict[str, dict[str, SlotProfile]] = {
    "MiniMax-M3": {
        "moe.fused_experts": _M3_MOE_NVFP4_PROFILE,
        "moe.fused_experts_reduce": _M3_MOE_NVFP4_PROFILE,
    },
    "GLM-5.3": {
        "collective.all_gather_into_tensor": _GLM53_DP_EXCHANGE_PROFILE,
        "collective.all_reduce": _GLM53_ALL_REDUCE_PROFILE,
        "collective.reduce_scatter_tensor": _GLM53_DP_EXCHANGE_PROFILE,
        "linear.dense": _GLM53_DENSE_PROFILE,
        "moe.fused_routed_experts": _GLM53_ROUTED_MOE_PROFILE,
        "norm.fused_add_rmsnorm": _GLM53_NORM_PROFILE,
    },
}
MODEL_PROFILES["MiniMax-M3-NVFP4"] = MODEL_PROFILES["MiniMax-M3"]
MODEL_PROFILES["GLM-5.3-NVFP4"] = MODEL_PROFILES["GLM-5.3"]


def model_profile(model_key: Optional[str], slot_name: str) -> Optional[SlotProfile]:
    if not model_key:
        return None
    return MODEL_PROFILES.get(model_key, {}).get(slot_name)


def slot_for_model(slot_name: str, model_key: Optional[str] = None) -> SlotSpec:
    slot = get_slot(slot_name)
    profile = model_profile(model_key, slot_name)
    return specialize_slot(slot, profile) if profile else slot


def verification_call_descriptor(
    slot: SlotSpec,
    inputs: dict,
    *,
    dtype_name: str,
    architecture: Optional[str],
    graph_mode: str,
    tp_size: Optional[int],
    world_size: Optional[int],
) -> CallDescriptor:
    """Project validator inputs into the same fields as the live slot seam."""
    if slot.name == "moe.fused_experts":
        return _moe_call_descriptor(
            inputs["x"], inputs["topk_ids"], architecture=architecture,
            graph_mode=graph_mode, quant=str(inputs.get("__moe_quant__", "dense")),
            num_experts=int(inputs["w13"].shape[0]),
            intermediate_dim=int(inputs["w2"].shape[-1]),
            tp_size=tp_size, world_size=world_size,
        )
    if slot.name == "moe.fused_routed_experts":
        return _routed_moe_call_descriptor(
            inputs["x"], top_k=int(inputs["topk"]), architecture=architecture,
            graph_mode=graph_mode, quant=str(inputs.get("__moe_quant__", "dense")),
            num_experts=int(inputs["w13"].shape[0]),
            intermediate_dim=int(inputs["w2"].shape[-1]),
            tp_size=tp_size, world_size=world_size,
        )
    if slot.name == "linear.dense":
        x, weight = inputs["x"], inputs["weight"]
        return CallDescriptor(
            architecture=architecture, dtype=dtype_name, graph_mode=graph_mode,
            input_dim=int(weight.shape[1]), last_dim=int(x.shape[-1]),
            layout="weight_out_in_row_major", num_tokens=int(x.shape[0]),
            output_dim=int(weight.shape[0]),
            parallel_role=str(inputs.get("parallel_role", "replicated")),
            quant="dense", tp_size=int(inputs.get("local_tp_size", tp_size or 1)),
            world_size=int(world_size or tp_size or 1),
        )
    primary = next((inputs[name] for name in (
        "x", "q", "input", "input_tensor", "residual", "gemm_out"
    ) if name in inputs and torch.is_tensor(inputs[name])), None)
    fields = {"dtype": dtype_name, "architecture": architecture}
    if primary is not None and primary.dim() > 0:
        fields.update(last_dim=int(primary.shape[-1]),
                      num_tokens=int(primary.numel() // primary.shape[-1]))
    return CallDescriptor(fields)

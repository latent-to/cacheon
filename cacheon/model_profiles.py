"""Validator-owned model facts applied to the generic slot catalog."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from cacheon.capabilities import CallDescriptor
from cacheon.sparse_mla_contract import call_descriptor as _sparse_mla_descriptor
from cacheon.indexer_select_contract import call_descriptor as _indexer_select_descriptor
from cacheon.dense_contract import call_descriptor as _dense_descriptor
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

_MOE_SLOTS = ("moe.fused_experts",)
_ROUTED_MOE_SLOTS = ("moe.fused_routed_experts",)


def specialize_slot(slot: SlotSpec, profile: SlotProfile) -> SlotSpec:
    """Retarget one generic slot with validator-owned model facts."""
    repl: dict = {}
    if profile.quant == "nvfp4" and profile.shapes is None:
        raise ValueError(
            f"profile for {slot.name!r} sets quant={profile.quant!r} but no shapes; "
            "a quantized profile must carry the arena's per-rank verification shapes"
        )
    routed = slot.name in _ROUTED_MOE_SLOTS
    moe = routed or slot.name in _MOE_SLOTS
    if moe:
        def reference(inputs):
            if routed:
                return [_routed_moe_reference(inputs, profile.activation)]
            return [_moe_reference(
                inputs["x"], inputs["w13"], inputs["w2"], inputs["topk_ids"],
                inputs["topk_weights"], profile.activation,
            )]

        repl["invoke_reference"] = reference
        if not routed and slot.collective_partial is not None:
            repl["collective_partial"] = lambda inputs, prepared: reference(inputs)[0].float()
    if profile.correctness is not None:
        repl["correctness"] = profile.correctness
    if moe and profile.quant == "nvfp4":
        def quant_inputs(**kwargs):
            dense = slot.make_inputs(**kwargs)
            fused = profile.num_fused_shared_experts
            if not routed:
                tokens, top_k = dense["topk_ids"].shape
                experts = dense["w13"].shape[0]
                routed_k = top_k - fused
                generator = torch.Generator(device=kwargs["device"]).manual_seed(
                    int(kwargs["seed"]) + 17_171
                )
                ids = torch.rand(
                    tokens, experts - fused, generator=generator, device=kwargs["device"],
                ).topk(routed_k, dim=-1).indices.to(torch.int32)
                scores = torch.rand(tokens, routed_k, generator=generator, device=kwargs["device"])
                weights = profile.routed_weight_scale * scores / scores.sum(-1, keepdim=True)
                if fused:
                    shared_ids = torch.arange(experts - fused, experts, device=kwargs["device"], dtype=torch.int32).expand(tokens, fused)
                    ids = torch.cat((ids, shared_ids), dim=-1)
                    weights = torch.cat((weights, torch.ones_like(scores[:, :fused])), dim=-1)
                dense.update(topk_ids=ids, topk_weights=weights)
            dense.update(
                __moe_tp_size__=int(kwargs.get("world_size", 4)),
                __moe_ep_size__=1, __moe_ep_rank__=0, __moe_reduce_results__=False,
                __moe_num_fused_shared_experts__=fused,
                __moe_activation__=profile.activation.kind,
            )
            return _moe_nvfp4_verification_inputs(dense)

        repl["make_inputs"] = quant_inputs
        repl["invoke_prepare"] = lambda prepare_fn, i: prepare_fn(
            *_moe_prepare_args_from_inputs(i), *((i["topk"], i["routed_scaling"]) if routed else ())
        )
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
    shapes=tuple(
        {"num_tokens": tokens, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8}
        for tokens in (1, 8, 24, 32, 128, 16384)
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
    ) + tuple(
        # The FP32 router gate over 256 routed experts. The indexer's separate
        # 6144->128 key and FP32 6144->32 head projections exist only with indexer
        # fusion off; GLM runs fusion on, so those GEMMs are never issued.
        dict(num_tokens=tokens, input_dim=6144, output_dim=256, output_dtype="float32")
        for tokens in (1, 8, 24, 128, 16384)
    ) + tuple(
        dict(num_tokens=tokens, input_dim=k, output_dim=n, batch_size=64)
        for tokens in (6, 32, 4096) for k, n in ((192, 512), (512, 256))
    ),
)

_GLM53_NORM_PROFILE = SlotProfile(shapes=tuple(
    {"num_tokens": tokens, "hidden": 6144}
    for tokens in (6, 24, 32, 128, 4096, 16384)
) + tuple(dict(num_tokens=tokens, hidden=hidden, use_residual=False)
          for tokens in (6, 32, 4096) for hidden in (512, 2048, 6144)))
_GLM53_ALL_REDUCE_PROFILE = SlotProfile(
    shapes=({"num_tokens": 16384, "hidden": 6144},),
)
_GLM53_DP_EXCHANGE_PROFILE = SlotProfile(shapes=tuple(
    {"num_tokens": tokens, "hidden": 6144} for tokens in (6, 32)
))

# TP4 / attention-DP4 leaves all 64 query heads on each attention rank.
_GLM53_SPARSE_MLA_PROFILE = SlotProfile(shapes=tuple(
    dict(num_tokens=tokens, num_heads=64, value_dim=512, rope_dim=64,
         num_pages=pages, page_size=64, top_k=2048, query_chunk=chunk,
         input_dtype="bfloat16")
    for tokens, pages, chunk in (
        (6, 128, 1), (32, 1024, 1), (128, 128, 128), (16384, 1024, 16384),
    )
))

MODEL_PROFILES: dict[str, dict[str, SlotProfile]] = {
    "Qwen3.6-35B-A3B-BF16": {
        # SGLang Qwen3_5 at TP1: GDN qkvz/ba, gated full-attention qkv,
        # attention/GDN output, shared MLP and the unquantized router.
        "linear.dense": SlotProfile(shapes=tuple(
            dict(num_tokens=m, input_dim=k, output_dim=n, parallel_role=role, local_tp_size=1)
            for m in (1, 2, 8, 128, 4096)
            for k, n, role in ((2048, 12288, "column"), (2048, 64, "column"),
                               (2048, 9216, "column"), (4096, 2048, "row"),
                               (2048, 1024, "column"), (512, 2048, "row"),
                               (2048, 256, "replicated"))
        )),
        "moe.fused_experts": SlotProfile(shapes=tuple(
            dict(num_tokens=m, num_experts=256, hidden=2048, inter=512, topk=8)
            for m in (1, 2, 8, 128, 4096)
        )),
    },
    "MiniMax-M3": {
        "moe.fused_experts": _M3_MOE_NVFP4_PROFILE,
    },
    "GLM-5.3": {
        "attention.indexer_select": SlotProfile(shapes=tuple(
            dict(num_tokens=tokens, num_heads=32, head_dim=128, kv_len=length,
                 page_size=64, top_k=2048, num_init_tokens=0, num_local_tokens=0)
            for tokens, length in ((6, 8192), (32, 65536), (128, 8192), (512, 65536))
        )),
        "attention.sparse_mla": _GLM53_SPARSE_MLA_PROFILE,
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
    if slot.name in ("attention.sparse_mla", "attention.indexer_select"):
        descriptor = _sparse_mla_descriptor if slot.name == "attention.sparse_mla" else _indexer_select_descriptor
        return descriptor(
            inputs, architecture=architecture, graph_mode=graph_mode,
            tp_size=tp_size, world_size=world_size,
        )
    if slot.name == "linear.dense":
        return _dense_descriptor(
            inputs, architecture=architecture, graph_mode=graph_mode,
            parallel_role=str(inputs.get("parallel_role", "replicated")),
            tp_size=int(inputs.get("local_tp_size", 1)),
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

"""Carry the four qualified GLM kernels inside their decoder-layer boundary.

The stock layer owns the sequence and attention state. Private module copies
replace only its arithmetic calls; original modules and class methods remain
available to the node auditor. metadata/provenance.json identifies the unchanged kernels.
"""

import copy

import torch

from cacheon.moe_nvfp4_contract import prepare_args_from_layer
from glm53_baseline import dpo, exchange, fused_add_rmsnorm, moe_routed_experts


def prepare_norm(module):
    """Keep the champion's original plain/residual RMSNorm domain."""
    def normalize(x, residual=None, post_residual_addition=None, quant_linear=None):
        if (x.dtype != torch.bfloat16 or x.numel() == 0
                or post_residual_addition is not None or quant_linear is not None
                or getattr(module, "fp32_residual", False)
                or module.variance_size_override is not None
                or module.cast_x_before_out_mul):
            return module.forward(x, residual, post_residual_addition, quant_linear)
        out = torch.empty_like(x)
        updated = torch.empty_like(residual) if residual is not None else None
        fused_add_rmsnorm.fused_add_rmsnorm(
            x, residual, module.weight, module.variance_epsilon, out, updated)
        return out if residual is None else (out, updated)
    return normalize


def _copy_with_norms(module, copies):
    """Share weight storage, preserving aliases while replacing private norm calls."""
    from sglang.srt.layers.layernorm import RMSNorm

    if id(module) in copies:
        return copies[id(module)]
    replica = copy.copy(module)
    copies[id(module)] = replica
    replica._modules = {
        name: _copy_with_norms(child, copies) if child is not None else None
        for name, child in module._modules.items()
    }
    if isinstance(module, RMSNorm):
        replica.forward = prepare_norm(module)
    return replica


def _replace_experts(original, replica):
    """Retain the winner's finalized output and stock's shared-expert join."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.moe.topk import TopKOutputFormat

    prepared = None

    def routed(x, topk_output, pre_quant_input=None):
        nonlocal prepared
        if (topk_output.format != TopKOutputFormat.BYPASSED
                or not 1 <= x.shape[0] <= 16384):
            return original.experts.forward_impl(x, topk_output, pre_quant_input)
        config = topk_output.topk_config
        if prepared is None:
            prepared = moe_routed_experts.prepare(
                *prepare_args_from_layer(original.experts), config.top_k,
                config.routed_scaling_factor)
        out = torch.empty_like(x)
        moe_routed_experts.fused_routed_experts(
            x, topk_output.router_logits, config.correction_bias, prepared, out)
        if original.experts.reduce_results and original.experts.moe_tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    replica.experts.forward = routed
    replica.experts.forward_impl = routed
    # The old candidate finalizer returned routed.add_(shared), because the
    # champion had already finalized. This selects that same stock add path.
    replica.experts.supports_deferred_finalize = False

    def mlp(x, *args, **kwargs):
        if original._can_dual_stream_graph(x):
            # The stock custom-op registry points at the original module. Execute
            # its identical dual-stream body with our private expert copy instead.
            return replica.forward_normal_dual_stream(x, *args[1:], **kwargs)
        return type(original).forward(replica, x, *args, **kwargs)

    replica.forward = mlp


class _Deferred:
    """Projection input stays inside the enclosing layer call."""

    def __init__(self, x):
        self.x = x


def prepare(module):
    """Compose the measured GLM implementation without modifying the served tree."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers import communicator as comm
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if (parallel.tp_size != 4 or parallel.attn_dp_size != 4
            or parallel.attn_tp_size != 1 or module.config.hidden_size != 6144
            or parallel.enable_prefill_cp):
        raise ValueError("this baseline requires the commissioned GLM TP4/DP4 topology")
    replica = _copy_with_norms(module, {})
    original_comm = module.layer_communicator
    replica.layer_communicator = copy.copy(original_comm)
    communicator = replica.layer_communicator
    communicator._context = copy.copy(original_comm._context)
    communicator.input_layernorm = replica.input_layernorm
    communicator.post_attention_layernorm = replica.post_attention_layernorm
    communicator.qkv_latent_func = replica.self_attn.prepare_qkv_latent
    if isinstance(module.mlp, DeepseekV2MoE):
        _replace_experts(module.mlp, replica.mlp)
    group = get_tp_group()
    projection = module.self_attn.o_proj
    norm = module.post_attention_layernorm
    prepared_projection = {}
    batch = None
    gather_fn = original_comm._communicate_with_all_reduce_and_layer_norm_fn
    gathers = (getattr(gather_fn, "func", gather_fn)
               is comm.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual)

    def project(x, *args, **kwargs):
        if (gathers and batch.forward_mode.is_decode_or_idle()
                and batch.dp_padding_mode.is_max_len() and not batch.can_run_tbo
                and 0 < x.shape[0] <= 32 and x.dtype == torch.bfloat16
                and projection.weight.dtype == x.dtype
                and x.ndim == 2 and x.stride(1) == 1
                and projection.weight.shape[1] == x.shape[1]
                and projection.bias is None):
            return _Deferred(x), None
        return projection.forward(x, *args, **kwargs)

    replica.self_attn.o_proj.forward = project

    def prepare_mlp(hidden_states, residual, forward_batch, cache=None):
        if isinstance(hidden_states, _Deferred):
            x = hidden_states.x
            experts = getattr(module.mlp, "experts", None)
            method = getattr(experts, "quant_method", None)
            quantized = (method is not None and method.enable_flashinfer_trtllm_moe
                         and not method.quant_config.use_per_token_activation
                         and not module.mlp._can_dual_stream_graph(x))
            scale = (experts.w13_input_scale_quant if quantized
                     else torch.empty(0, device=x.device, dtype=torch.float32))
            if quantized not in prepared_projection:
                prepared_projection[quantized] = dpo.prepare(
                    projection.weight, norm.weight, norm.variance_epsilon, scale)
            rows, hidden = x.shape[0] * group.world_size, norm.weight.numel()
            out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
            updated = torch.empty_like(residual)
            packed = torch.empty((rows, hidden // 2) if quantized else (0,),
                                 dtype=torch.uint8, device=x.device)
            scales = torch.empty((rows, hidden // 16) if quantized else (0,),
                                 dtype=torch.uint8, device=x.device)
            dpo.project_gather_norm(x, residual, prepared_projection[quantized],
                                   out, updated, packed, scales, group.device_group)
            return out, updated
        if (gathers and forward_batch.dp_padding_mode.is_max_len()
                and not comm.get_attn_tp_context().input_scattered):
            if hidden_states.shape[0]:
                with comm.use_symmetric_memory(group, disabled=not comm.is_allocation_symmetric()):
                    hidden_states, residual = communicator.post_attention_layernorm(
                        hidden_states, residual)
            out = comm.get_global_dp_buffer(group)
            exchange.all_gather_into_tensor(hidden_states, out, group.device_group)
            return out, residual
        return type(original_comm).prepare_mlp(
            communicator, hidden_states, residual, forward_batch, cache)

    def postprocess(hidden_states, residual, forward_batch):
        scatters = (communicator._communicate_summable_tensor_pair_fn
                    is comm.CommunicateSummableTensorPairFn._scatter_hidden_states)
        if (scatters and communicator.allow_reduce_scatter
                and forward_batch.dp_padding_mode.is_max_len()
                and not comm.should_use_dp_reduce_scatterv()):
            out = comm.get_local_dp_buffer(group)
            exchange.reduce_scatter_tensor(hidden_states, out, group.device_group)
            return out, residual
        return type(original_comm).postprocess_layer(
            communicator, hidden_states, residual, forward_batch)

    communicator.prepare_mlp = prepare_mlp
    communicator.postprocess_layer = postprocess

    def layer(*args, **kwargs):
        nonlocal batch
        batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
        try:
            return type(module).forward(replica, *args, **kwargs)
        finally:
            batch = None

    return layer


def forward(prepared, *args, **kwargs):
    """Return the stock node structure from its prepared computation."""
    return prepared(*args, **kwargs)

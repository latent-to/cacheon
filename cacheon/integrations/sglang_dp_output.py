"""Bind a fused post-attention collective on stock SGLang0.5.18.

The adapter owns deferral and tensor flow; the selected bundle owns computation.
Ordinary dense, normalization and exchange callsites outside this bounded region
continue through their existing adapters. No SGLang source files are replaced.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os
import sys
import weakref

import torch

from cacheon.capabilities import collective_call_descriptor
from cacheon.dispatch import _arch_tag, _audit, _in_cuda_graph, _receipts
from cacheon.dp_output_projection_contract import SLOT, output_spec, reference
from cacheon.registry import REGISTRY
from cacheon.tensor_spec import allocate_output_spec, tensor_bindings, validate_tensor_bindings

_MODEL = "sglang.srt.models.deepseek_v2"
_COMM = "sglang.srt.layers.communicator"
_MOE = "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm"
_scope = ContextVar("cacheon_dp_output", default=None)
_patches = {}
_linears = weakref.WeakKeyDictionary()


@dataclass
class _Scope:
    layer: object
    batch: object
    quantized: tuple | None = None


@dataclass
class _Deferred:
    x: torch.Tensor
    linear: object
    implementation: object
    group: object
    quant_scale: torch.Tensor


def _quant_scale(state, x):
    from sglang.srt.layers.moe import get_moe_a2a_backend

    mlp = state.layer.mlp
    experts = getattr(mlp, "experts", None)
    method = getattr(experts, "quant_method", None)
    if (method is None or not getattr(method, "enable_flashinfer_trtllm_moe", False)
            or not get_moe_a2a_backend().is_none()
            or method.quant_config.use_per_token_activation
            or torch.compiler.is_compiling() or mlp._can_dual_stream_graph(x)
            or os.environ.get("FLASHINFER_NVFP4_4OVER6", "0") == "1"
            or os.environ.get("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "0").lower()
            in ("1", "true", "yes", "on")):
        return torch.empty(0, dtype=torch.float32, device=x.device)
    scale = experts.w13_input_scale_quant
    if scale.numel() != 1 or scale.dtype != torch.float32:
        raise RuntimeError("DP output preparation requires the scalar checkpoint scale")
    return scale


def _select(x, linear, registry):
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.layers.communicator import CommunicateWithAllReduceAndLayerNormFn

    state = _scope.get()
    if state is None:
        return None
    batch, parallel = state.batch, get_parallel()
    fn = state.layer.layer_communicator._communicate_with_all_reduce_and_layer_norm_fn
    if (getattr(fn, "func", fn)
            is not CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            or not batch.forward_mode.is_decode_or_idle()
            or not batch.dp_padding_mode.is_max_len() or batch.can_run_tbo
            or parallel.attn_tp_size != 1 or parallel.attn_dp_size != parallel.tp_size):
        return None
    weight = getattr(linear, "weight", None)
    if (not torch.is_tensor(weight) or weight.ndim != 2 or x.ndim != 2
            or weight.dtype != x.dtype or x.stride(1) != 1
            or weight.shape[1] != x.shape[1] or getattr(linear, "bias", None) is not None):
        return None
    group = get_tp_group()
    descriptor = collective_call_descriptor(
        dtype=str(x.dtype).removeprefix("torch."), architecture=_arch_tag(x.device.index or 0),
        graph_mode="cuda_graph" if _in_cuda_graph() else "eager", world_size=group.world_size,
        dimensions=dict(num_tokens=x.shape[0], input_dim=x.shape[1],
                        hidden_dim=weight.shape[0], last_dim=weight.shape[0]),
    )
    impl = registry.select(SLOT, descriptor).impl
    return None if impl is None else _Deferred(x, linear, impl, group, _quant_scale(state, x))


def _audit_rows(outputs, batch, group):
    """Compare real token rows using SGLang's original per-rank counts."""
    counts = batch.original_global_num_tokens_cpu
    rows = outputs[1].shape[0]
    if (not isinstance(counts, (list, tuple)) or len(counts) != group.world_size
            or any(type(n) is not int or not 0 <= n <= rows for n in counts)):
        raise RuntimeError("DP output audit requires valid original per-rank token counts")
    if all(n == rows for n in counts):
        return tuple(outputs)

    def gathered(tensor):
        return (torch.cat([tensor[i * rows:i * rows + n] for i, n in enumerate(counts)])
                if tensor.numel() else tensor)

    normalized, local, packed, scales = outputs
    return (gathered(normalized), local[:counts[group.rank_in_group]],
            gathered(packed), gathered(scales))


def _finish(deferred, residual, norm):
    state = _scope.get()
    if state is None:
        raise RuntimeError("DP output preparation escaped its decoder call")
    impl, group = deferred.implementation, deferred.group
    inputs = dict(x=deferred.x, residual=residual, weight=deferred.linear.weight,
                  gamma=norm.weight, epsilon=norm.variance_epsilon,
                  quant_scale=deferred.quant_scale, world_size=group.world_size)
    if residual is None or residual.shape != (deferred.x.shape[0], inputs["weight"].shape[0]):
        raise RuntimeError("DP output preparation received the wrong residual")
    cache = getattr(deferred.linear, "_cacheon_dp_output_prepared", None)
    if cache is None:
        cache = deferred.linear._cacheon_dp_output_prepared = {}
    parameters = (inputs["weight"], inputs["gamma"], inputs["quant_scale"])
    identity = tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype,
                      None if t.is_inference() else t._version) for t in parameters)
    key = (impl.bundle_id, impl.variant, id(impl.prepare), identity, inputs["epsilon"])
    if key not in cache:
        if impl.prepare is None:
            raise RuntimeError("selected DP output preparation has no prepare entry")
        cache[key] = _receipts.invoke(SLOT, impl.prepare, inputs["weight"], inputs["gamma"],
                                     inputs["epsilon"], inputs["quant_scale"], phase="prepare")
    allocation = allocate_output_spec(output_spec(inputs), fallback_dtype=deferred.x.dtype,
                                      fallback_device=deferred.x.device,
                                      inputs=(deferred.x, residual, *parameters))
    outputs = allocation.outputs
    tensors = (deferred.x, residual, *parameters, *outputs)
    bindings = tensor_bindings(tensors)
    audit = not _in_cuda_graph() and _audit.sampled()
    # SGLang's autotuning dummy batches have no scheduler token counts.
    if audit and state.batch.original_global_num_tokens_cpu is None:
        _audit.baseline_refused(SLOT)
        audit = False
    expected = reference(inputs, group.device_group, group.rank_in_group, group.world_size) if audit else None
    _receipts.invoke(SLOT, impl.entry, deferred.x, residual, cache[key], *outputs, group.device_group)
    validate_tensor_bindings(tensors, bindings, kind="DP output preparation input/output")
    if expected is not None:
        audited = _audit_rows(outputs, state.batch, group)
        if audited[0].numel():
            expected = _audit_rows(expected, state.batch, group)
            _audit.run(SLOT, audited, lambda: expected)
        else:
            _audit.baseline_refused(SLOT)
    _receipts.completed(SLOT)
    normalized, updated, packed, scales = outputs
    if deferred.quant_scale.numel():
        state.quantized = (normalized, deferred.quant_scale, packed, scales)
    return normalized, updated


def _wrap_init(original, registry):
    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        linear = self.self_attn.o_proj
        old_forward = linear.forward
        linear_ref = weakref.ref(linear)

        @wraps(old_forward)
        def project(x, *args, **kwargs):
            selected = _select(x, linear_ref(), registry)
            return (selected, None) if selected is not None else old_forward(x, *args, **kwargs)

        _linears[linear] = weakref.WeakMethod(old_forward)
        linear.forward = project
    return initialize


def _wrap_forward(original, registry):
    @wraps(original)
    def forward(self, *args, **kwargs):
        batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
        token = _scope.set(_Scope(self, batch))
        try:
            return original(self, *args, **kwargs)
        finally:
            _scope.reset(token)
    return forward


def _wrap_prepare_mlp(original, registry):
    @wraps(original)
    def prepare_mlp(self, hidden_states, residual, forward_batch, *args, **kwargs):
        if isinstance(hidden_states, _Deferred):
            return _finish(hidden_states, residual, self.post_attention_layernorm)
        return original(self, hidden_states, residual, forward_batch, *args, **kwargs)
    return prepare_mlp


def _wrap_fp4(original, registry):
    @wraps(original)
    def fp4(dispatch_output, quant_info, runner_config, *args, **kwargs):
        state = _scope.get()
        if state is not None and state.quantized is not None:
            hidden, scale, packed, scales = state.quantized
            if (dispatch_output.hidden_states.data_ptr() != hidden.data_ptr()
                    or dispatch_output.hidden_states.shape != hidden.shape
                    or quant_info.w13_input_scale_quant.data_ptr() != scale.data_ptr()
                    or dispatch_output.hidden_states_scale is not None):
                raise RuntimeError("DP output FP4 values reached a different MoE consumer")
            dispatch_output = dispatch_output._replace(
                hidden_states=packed, hidden_states_scale=scales.view(torch.float8_e4m3fn))
        return original(dispatch_output, quant_info, runner_config, *args, **kwargs)
    return fp4


def install(registry=REGISTRY):
    """Install fixed, reversible hooks only when this registered seam is enabled."""
    if os.environ.get("CACHEON_DP_OUTPUT_PROJECTION_SEAM") != "1":
        return
    layer = getattr(sys.modules.get(_MODEL), "DeepseekV2DecoderLayer", None)
    comm = getattr(sys.modules.get(_COMM), "LayerCommunicator", None)
    for owner, name, wrapper in (
        (layer, "__init__", _wrap_init), (layer, "forward", _wrap_forward),
        (comm, "prepare_mlp", _wrap_prepare_mlp),
        (sys.modules.get(_MOE), "fused_experts_none_to_flashinfer_trtllm_fp4", _wrap_fp4),
    ):
        if owner is not None and (owner, name) not in _patches:
            original = getattr(owner, name)
            _patches[owner, name] = original
            setattr(owner, name, wrapper(original, registry))


def uninstall():
    """Restore original methods and per-instance projection callables."""
    for linear, original_ref in list(_linears.items()):
        original = original_ref()
        if original is not None:
            linear.forward = original
    _linears.clear()
    for (owner, name), original in _patches.items():
        setattr(owner, name, original)
    _patches.clear()


def is_installed():
    """Report whether any of the registered module hooks is installed."""
    return bool(_patches)

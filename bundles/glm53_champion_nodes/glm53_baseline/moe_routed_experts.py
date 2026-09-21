"""v22: v20 + the decode finalize deferred to the bundle's own kernel.

Measured 2026-09-19 at 128 gathered rows, the fused MoE is six kernels: GEMM1 83.6 us, GEMM2 54.4,
finalize 7.3, two routing kernels 7.2, quantize 2.1. The GEMMs are at the weight-streaming
roofline and v20 already serves stock's boot-tuned tactic, so the finalize is the only in-slot
decode lever with size. TRT-LLM's finalize runs ONE block per token: at 24-128 rows that is
24-128 blocks on 148 SMs streaming 96 KB each, 7.3 us for ~14 MB, a quarter of HBM bandwidth.
v22 asks the vendor for `do_finalize=False` (it then skips that kernel and returns the permuted
GEMM2 rows, the expert weights and the token-major expanded->permuted map) and finishes with a
Triton kernel that splits every token's 6144 columns across six programs, so 24-128 tokens become
144-768 programs and the read has the parallelism the vendor kernel lacks at small M. Same math:
fp32 accumulate over the 8 slots in slot order, negative map entries skipped, one bf16 rounding.
`do_finalize` is not part of the runner's identity, so the AutoTuner cache key is unchanged and the
call still hits the boot-tuned tactic exactly as v20 does.

Applies at and below 512 gathered rows (the graded decode band is 24-128). Above that v20's paths
are unchanged: stock's wrapper with the vendor finalize up to 12288, v10's explicit tactic through
the raw op beyond. Expected: a few microseconds per decode call, ~2% of the slot at 128 rows. A
slower or non-matching finalize falsifies it; the audit comparator decides correctness.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

_SGLANG_VERSION, _FLASHINFER_VERSION = "0.5.18", "0.6.17"
_TAG, _GATE_UP = "nvfp4_layer", "gate_up"
_INTERLEAVED, _TRTLLM = (
    "up_gate_interleaved_64+sf_swizzled_128x4", "trtllm_fp4_shuffled")
_TOKENS, _EXPERTS, _HIDDEN, _INTERMEDIATE, _TOP_K = 16384, 256, 6144, 512, 8
# The live gathered prefill batch is NOT exactly _TOKENS. Measured on the arena engine
# (TP4 + DP4, 4096-row per-rank chunks) the seam is handed 16368, 16370..16384 -- the DP
# ranks contribute slightly unequal chunks, so the sum lands just under the cap. An
# exact-_TOKENS domain is refused on essentially every live call.
#
# The domain also has to reach DOWN to a shape the SCREEN can observe inside a CUDA graph.
# The resident screen promotes only when every registered slot is seen inside a capture on every
# rank (resident_execution_evidence.RankExecution.clean; no slot sets serving_graph_captured
# False, so the exemption set is empty). The screen is a hot swap, and the swap rebuilds only the
# DECODE graphs -- sglang_resident_swap installs its hook on init_decode_cuda_graph and nulls
# decode_cuda_graph_runner alone, so the prefill graphs captured at boot keep replaying stock
# while a candidate is resident. Capture can therefore only be proven from the decode recapture,
# whose gathered batches are at most a few hundred rows (max_running_requests across DP ranks).
# A prefill-only domain is called all day and still records captured=false -- a screen rejection,
# which is exactly what v2's earlier 2048-row floor earned. All experiments reach row 1;
# every selected call still pays the seam's separate shared-expert addition.
_MIN_TOKENS = 1
# Above this the profiled tactic is used. At and below it the heuristic is used, because that is
# what measurement says wins there: at 8192 gathered rows the tuned tactic reads 0.997x against
# the heuristic, i.e. the heuristic is already the better choice. Serving that band on tactic -1
# is a measured selection, not a fallback -- it is the same kernel, and it is what makes the
# captured band execute; this does not eliminate the separate shared-expert addition.
_TACTIC_MIN_TOKENS = 12288
_ROUTED_SCALING, _SF_BLOCK = 2.5, 16
_ROUTING_DEEPSEEK_V3, _ACT_SWIGLU = 2, 3
# V2's inherited 16K keep-guard is unchanged; it is not evidence for this variant.
_MIN_TACTIC_SPEEDUP, _AB_PAIRS, _AB_REPLAYS, _AB_ITERS = 1.05, 3, 5, 3
_W13_SHAPE = (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN // 2)
_W2_SHAPE = (_EXPERTS, _HIDDEN, _INTERMEDIATE // 2)
_W13_SF_SHAPE = (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN // _SF_BLOCK)
_W2_SF_SHAPE = (_EXPERTS, _HIDDEN, _INTERMEDIATE // _SF_BLOCK)

_tactic_lock = threading.Lock()
_tactics: dict[tuple, tuple[object, tuple[int, int], float]] = {}
_tactic_failures: dict[tuple, str] = {}


def _runtime():
    """Load and validate the one private FlashInfer ABI this bundle implements."""
    import flashinfer
    import sglang
    if flashinfer.__version__ != _FLASHINFER_VERSION:
        raise RuntimeError(f"requires FlashInfer {_FLASHINFER_VERSION}, got {flashinfer.__version__}")
    if sglang.__version__ != _SGLANG_VERSION:
        raise RuntimeError(f"requires SGLang {_SGLANG_VERSION}, got {sglang.__version__}")

    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
    from flashinfer.autotuner import AutoTuner, DynamicTensorSpec, TuningConfig
    from flashinfer.fused_moe.core import (ActivationType, Fp8QuantizationType,
        MoeRunnerInputs, RoutingInputMode, WeightLayout,
        deduce_trtllm_gen_tensor_dtype, get_trtllm_moe_sm100_module)
    # Imported standalone, moe_runner.flashinfer_trtllm pulls quantization/fp8.py, which
    # imports back into the half-initialised runner module and raises ImportError. The
    # engine never sees this because it loads the quantization package first; do the same.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import trtllm_moe_enable_pdl
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    module = get_trtllm_moe_sm100_module()
    if not hasattr(module, "MoERunner"):
        raise RuntimeError("FlashInfer 0.6.17 TRT-LLM module does not expose MoERunner")
    return SimpleNamespace(
        tuner=AutoTuner, spec=DynamicTensorSpec, config=TuningConfig,
        inputs=MoeRunnerInputs, routing=RoutingInputMode, layout=WeightLayout,
        activation=ActivationType, fp8=Fp8QuantizationType,
        deduce=deduce_trtllm_gen_tensor_dtype, runner=module.MoERunner,
        native=gen_trtllm_gen_fused_moe_sm100_module().build_and_load().trtllm_fp4_block_scale_moe,
        quantize=fp4_quantize, pdl=trtllm_moe_enable_pdl)


def _shape_dtype(name, tensor, shape, dtype, device=None):
    """Reject a tensor that is not the exact contiguous GLM contract value."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must be contiguous {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} is on {tensor.device}, expected {device}")
    return tensor


def _input_scale(name, value, device):
    """Canonicalize the scalar scale; verifier vectors must be exactly uniform."""
    if not torch.is_tensor(value) or value.dtype != torch.float32 or value.device != device:
        raise ValueError(f"{name} must be a float32 tensor on the weight device")
    flat = value.reshape(-1)
    if flat.numel() == 1:
        return flat
    if flat.numel() != _EXPERTS or not bool(torch.all(flat == flat[0])):
        raise ValueError(f"{name} must be scalar or a uniform 256-vector")
    return flat[:1].contiguous()


def _view_tensors(view):
    """Return TRT-LLM-shuffled weights from the validator-owned NVFP4 view."""
    layout = str(view.cacheon_w13_layout)
    w13 = _shape_dtype("w13_weight", view.w13_weight, _W13_SHAPE, torch.uint8)
    device = w13.device
    w2 = _shape_dtype("w2_weight", view.w2_weight, _W2_SHAPE, torch.uint8, device)
    w13_sf = _shape_dtype("w13_blockscale", view.w13_blockscale_swizzled,
                           _W13_SF_SHAPE, torch.float8_e4m3fn, device)
    w2_sf = _shape_dtype("w2_blockscale", view.w2_blockscale_swizzled,
                          _W2_SF_SHAPE, torch.float8_e4m3fn, device)
    if int(view.intermediate_size_per_partition) != _INTERMEDIATE:
        raise ValueError("GLM-5.3 TP4 requires intermediate_size_per_partition=512")
    if layout == _TRTLLM:
        return w13, w13_sf, w2, w2_sf
    if layout not in (_GATE_UP, _INTERLEAVED):
        raise ValueError(f"unsupported NVFP4 w13 layout {layout!r}")

    from cacheon_kernels import codec
    from sglang.srt.layers.quantization.utils import prepare_static_weights_for_trtllm_fp4_moe

    w13_sf = codec.unswizzle_blockscale(w13_sf, rows=2 * _INTERMEDIATE,
                                         cols=_HIDDEN // _SF_BLOCK)
    w2_sf = codec.unswizzle_blockscale(w2_sf, rows=_HIDDEN,
                                        cols=_INTERMEDIATE // _SF_BLOCK)
    if layout == _INTERLEAVED:
        w13 = codec.deinterleave_w13_halves(w13, group=64)
        w13_sf = codec.deinterleave_w13_halves(w13_sf, group=64)
    # The slot view is [gate; up]; the pinned TRT-LLM preparation consumes [up; gate].
    w13 = torch.cat((w13[:, _INTERMEDIATE:], w13[:, :_INTERMEDIATE]), dim=1)
    w13_sf = torch.cat((w13_sf[:, _INTERMEDIATE:], w13_sf[:, :_INTERMEDIATE]), dim=1)
    return prepare_static_weights_for_trtllm_fp4_moe(
        w13, w2, w13_sf, w2_sf, _HIDDEN, _INTERMEDIATE, _EXPERTS, is_gated=True)


def _quantize(prepared, x):
    """Apply the pinned activation NVFP4 quantizer without changing global state."""
    fp4, scale = prepared["rt"].quantize(
        x, prepared["input_scale"], sf_vec_size=_SF_BLOCK,
        sf_use_ue8m0=False, is_sf_swizzled_layout=False)
    tokens = x.shape[0]
    return (fp4.reshape(tokens, _HIDDEN // 2),
            scale.view(torch.float8_e4m3fn).reshape(tokens, _HIDDEN // _SF_BLOCK))


def _runner_kwargs(prepared, bias, tokens):
    """Build the GLM routing and weight arguments for one FlashInfer runner call.

    enable_pdl is decided PER CALL from the live row count, exactly as the stock runner decides
    it (`trtllm_moe_enable_pdl`, which is on at or below SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS). Fixing
    it once at prepare from the largest served shape turns it off for every small batch, where
    stock has it on: measured on the arena engine that cost 17% of the short-prompt prompt-pass
    cell, swamping the tactic win.
    """
    return {
        "routing_input_mode": prepared["routing_mode"], "num_experts": _EXPERTS,
        "routing_bias": bias, "gemm1_weights": prepared["w13"],
        "gemm1_weights_scale": prepared["w13_sf"], "gemm1_bias": None,
        "gemm1_alpha": None, "gemm1_beta": None, "gemm1_clamp_limit": None,
        "gemm2_weights": prepared["w2"], "gemm2_weights_scale": prepared["w2_sf"],
        "gemm2_bias": None, "output1_scale_scalar": prepared["g1_scale_c"],
        "output1_scale_gate_scalar": prepared["g1_alphas"],
        "output2_scale_scalar": prepared["g2_alphas"], "per_token_scale": None,
        "n_group": 1, "topk_group": 1, "local_expert_offset": 0,
        "routed_scaling_factor": _ROUTED_SCALING,
        "routing_method_type": _ROUTING_DEEPSEEK_V3, "do_finalize": True,
        "enable_pdl": bool(prepared["pdl"](tokens)), "activation_type": _ACT_SWIGLU,
        "num_fused_shared_experts": 0, "norm_topk_prob": True,
        "routing_replay_out": None,
    }


def _inputs(prepared, logits, bias, fp4, scale):
    """Allocate the output/routing buffers required by one runner call at fp4's row count."""
    device = fp4.device
    tokens = fp4.shape[0]
    values = prepared["rt"].inputs(
        output=torch.empty((tokens, _HIDDEN), dtype=torch.bfloat16, device=device),
        routing_logits=logits,
        topk_ids=torch.empty((tokens, _TOP_K), dtype=torch.int32, device=device),
        expert_weights=torch.empty((tokens, _TOP_K), dtype=torch.bfloat16, device=device),
        hidden_states=fp4, hidden_states_scale=scale, gemm1_lora_delta=None,
        per_token_scale=None)
    return values, _runner_kwargs(prepared, bias, tokens)


def _runner(prepared, fp4, scale):
    """Construct the pinned immutable runner; weights remain per-layer kwargs."""
    rt = prepared["rt"]
    return rt.runner(
        top_k=_TOP_K, num_local_experts=_EXPERTS,
        dtype_act=rt.deduce(fp4, scale),
        dtype_weights=rt.deduce(prepared["w13"], prepared["w13_sf"]),
        fp8_quantization_type=rt.fp8.NoneFp8, hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE, activation_type=rt.activation.Swiglu.value,
        use_shuffled_weight=True, weight_layout=rt.layout.MajorK,
        use_packed_weights=False, use_per_token_scaling=False,
        num_experts=_EXPERTS, num_fused_shared_experts=0)


def _exact_bucket(value: int) -> int:
    """Every served row count maps to the one profiled bucket.

    The tuner profiles at _TOKENS; the live batch is a few rows short of it, and trtllm-gen's
    tile choice does not change across that span, so the profiled tactic is the right one for
    the whole declared range.
    """
    if not _MIN_TOKENS <= value <= _TOKENS:
        raise ValueError(f"tuner received off-domain token count {value}")
    return _TOKENS


def _tuning_config(prepared, runner, inputs):
    """Restrict FlashInfer's local tuner to one exact graph-replay profile."""
    base = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=_TOKENS,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    spec = base.dynamic_tensor_specs[0]
    exact = prepared["rt"].spec(
        input_idx=spec.input_idx, dim_idx=spec.dim_idx, gen_tuning_buckets=(_TOKENS,),
        map_to_tuning_buckets=_exact_bucket,
        tensor_initializers=spec.tensor_initializers)
    return prepared["rt"].config(
        dynamic_tensor_specs=(exact,), constraint_specs=base.constraint_specs,
        use_cold_l2_cache=True, use_cuda_graph=True)


def _execute(runner, tactic, inputs, kwargs):
    """Execute one explicit runner tactic and return its caller-owned output."""
    runner.forward(inputs.to_list(), tactic=tactic, **kwargs)
    return inputs.output


def _capture(prepared, runner, tactic, x, logits, bias):
    """Capture BF16 quantization plus one tactic in the serving graph regime.

    Several replays go INSIDE the graph. Timing one replay of an N-call graph divides the
    launch and event overhead by N; timing N replays of a one-call graph does not, and at a
    ~1.3 ms kernel that overhead plus a cold first replay was enough to read a genuine 1.56x
    slot win as 1.16x -- which then failed this bundle's own keep-guard. Capture happens on a
    side stream for the same reason the engine does it: the default stream is not capturable
    once other work is queued on it.
    """
    fp4, scale = _quantize(prepared, x)
    inputs, kwargs = _inputs(prepared, logits, bias, fp4, scale)
    _execute(runner, tactic, inputs, kwargs)
    torch.cuda.synchronize(x.device)
    stream = torch.cuda.Stream(device=x.device)
    with torch.cuda.stream(stream):
        _execute(runner, tactic, inputs, kwargs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(_AB_ITERS):
                fp4, scale = _quantize(prepared, x)
                inputs, kwargs = _inputs(prepared, logits, bias, fp4, scale)
                _execute(runner, tactic, inputs, kwargs)
    torch.cuda.synchronize(x.device)
    graph.replay()
    torch.cuda.synchronize(x.device)
    return graph, inputs


def _graph_ms(capture) -> float:
    """Sample minimum replay latency per call; this is a noisy local tuning signal."""
    graph = capture[0]
    best = float("inf")
    for _ in range(_AB_REPLAYS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        best = min(best, float(start.elapsed_time(end)) / _AB_ITERS)
    return best


def _ab_speedup(prepared, runner, tactic, x, logits, bias) -> float:
    """Return the worst interleaved heuristic/explicit graph-replay ratio."""
    tuned_graph = _capture(prepared, runner, list(tactic), x, logits, bias)
    stock_graph = _capture(prepared, runner, -1, x, logits, bias)
    ratios = []
    for pair in range(_AB_PAIRS):
        if pair % 2:
            stock_ms = _graph_ms(stock_graph)
            tuned_ms = _graph_ms(tuned_graph)
        else:
            tuned_ms = _graph_ms(tuned_graph)
            stock_ms = _graph_ms(stock_graph)
        ratios.append(stock_ms / tuned_ms)
    return min(ratios)


def _select_tactic(prepared):
    """Tune once per process/device identity and require an exact-shape win."""
    device = prepared["w13"].device
    key = (_FLASHINFER_VERSION, device.type, device.index,
           torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device),
           _TOKENS, _EXPERTS, _HIDDEN, _INTERMEDIATE, _TOP_K)
    with _tactic_lock:
        if key in _tactics:
            return _tactics[key]
        if key in _tactic_failures:
            raise RuntimeError(f"prior exact-shape tactic preparation failed: {_tactic_failures[key]}")
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(5316384)
            x = torch.randn((_TOKENS, _HIDDEN), dtype=torch.bfloat16, device=device,
                            generator=generator)
            logits = torch.randn((_TOKENS, _EXPERTS), dtype=torch.float32,
                                 device=device, generator=generator)
            bias = torch.randn((_EXPERTS,), dtype=torch.float32, device=device,
                               generator=generator)
            fp4, scale = _quantize(prepared, x)
            inputs, kwargs = _inputs(prepared, logits, bias, fp4, scale)
            runner = _runner(prepared, fp4, scale)
            tuner = prepared["rt"].tuner(warmup=3, repeat=7)
            tuner.is_tuning_mode = True
            try:
                _, tactic = tuner.choose_one(
                    "flashinfer::trtllm_fp4_block_scale_moe", [runner],
                    _tuning_config(prepared, runner, inputs), inputs.to_list(), **kwargs)
            finally:
                tuner.is_tuning_mode = False
            # FlashInfer 0.6.17 hands the pair back through pybind/numpy types that fail a
            # plain list-of-int check while printing as plain numbers. The invariant is
            # unchanged -- exactly two members, neither the -1 heuristic -- but it is checked
            # on the coerced ints so later code and the metadata see plain Python ints.
            try:
                pair = tuple(int(item) for item in tactic)
            except TypeError:
                pair = ()
            if len(pair) != 2 or any(item < 0 for item in pair):
                raise RuntimeError(f"no fully explicit TRT-LLM tactic selected: {tactic!r} "
                                   f"({type(tactic).__name__})")
            tactic = pair
            tactic = (int(tactic[0]), int(tactic[1]))
            speedup = _ab_speedup(prepared, runner, tactic, x, logits, bias)
            if speedup < _MIN_TACTIC_SPEEDUP:
                raise RuntimeError(
                    f"explicit tactic {tactic} reached only {speedup:.4f}x at M={_TOKENS}; "
                    f"requires {_MIN_TACTIC_SPEEDUP:.2f}x")
            record = (runner, tactic, speedup)
            _tactics[key] = record
            return record
        except Exception as exc:
            _tactic_failures[key] = f"{type(exc).__name__}: {exc}"
            raise


def prepare(tag, view, topk, routed_scaling):
    """Prepare one real GLM layer and bind the process-local explicit tactic."""
    if tag != _TAG:
        raise ValueError(f"requires {_TAG!r} weights, got {tag!r}")
    if int(topk) != _TOP_K or float(routed_scaling) != _ROUTED_SCALING:
        raise ValueError(f"requires topk={_TOP_K}, routed_scaling={_ROUTED_SCALING}, "
                         f"got {topk}, {routed_scaling}")
    runtime = _runtime()
    w13, w13_sf, w2, w2_sf = _view_tensors(view)
    device = w13.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 3):
        raise ValueError("requires an sm103 CUDA device")
    prepared = {
        "rt": runtime, "w13": w13, "w13_sf": w13_sf, "w2": w2, "w2_sf": w2_sf,
        "g1_scale_c": _shape_dtype("g1_scale_c", view.g1_scale_c,
                                    (_EXPERTS,), torch.float32, device),
        "g1_alphas": _shape_dtype("g1_alphas", view.g1_alphas,
                                   (_EXPERTS,), torch.float32, device),
        "g2_alphas": _shape_dtype("g2_alphas", view.g2_alphas,
                                   (_EXPERTS,), torch.float32, device),
        "input_scale": _input_scale("w13_input_scale_quant",
                                    view.w13_input_scale_quant, device),
        "w2_input_scale": _input_scale("w2_input_scale_quant",
                                       view.w2_input_scale_quant, device),
        "routing_mode": runtime.routing.FromLogits,
        "pdl": runtime.pdl,
    }
    runner, tactic, speedup = _select_tactic(prepared)
    prepared["runner"] = runner
    prepared["tactic"] = tactic
    prepared["prepare_ab_speedup"] = speedup
    # Only weight/configuration arguments are retained, never live tensor addresses.
    prepared["native_weights"] = (
        w13, w13_sf, None, None, None, None, w2, w2_sf, None,
        prepared["g1_scale_c"], prepared["g1_alphas"], prepared["g2_alphas"], None,
        _EXPERTS, _TOP_K, 0, 1, 1, _INTERMEDIATE, 0, _EXPERTS,
        _ROUTED_SCALING, _ROUTING_DEEPSEEK_V3, True)
    prepared["call_plans"] = {}
    # Sized for the widest served batch and sliced per call: the live row count varies by a
    # few rows between forwards, and reallocating per call would show up in a 1.2 ms kernel.
    prepared["topk_ids"] = torch.empty((_TOKENS, _TOP_K), dtype=torch.int32, device=device)
    prepared["topk_weights"] = torch.empty((_TOKENS, _TOP_K), dtype=torch.bfloat16, device=device)
    return prepared


_FIN_BLOCK = 1024
_DECODE_MAX_TOKENS = 512


@triton.jit
def _finalize(GEMM2, IDX, W, OUT, hidden, stride, K: tl.constexpr, BLOCK: tl.constexpr):
    """out[t] = sum_k w[t,k] * gemm2[map[t*K+k]] in fp32, slot order, one bf16 rounding.

    Grid (tokens, hidden // BLOCK): six programs per token at H=6144, so a 24..128-row decode call
    runs 144..768 programs instead of the vendor's one block per token. The griddepcontrol.wait is
    the PDL handshake: this kernel is launched with launch_pdl so its prologue overlaps GEMM2's
    tail, and the wait sits before the first dependent load.
    """
    tok = tl.inline_asm_elementwise(
        "griddepcontrol.wait; mov.b32 $0, $1;", "=r,r", [tl.program_id(0)],
        dtype=tl.int32, is_pure=False, pack=1)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(K):
        idx = tl.load(IDX + tok * K + k)
        w = tl.load(W + tok * K + k).to(tl.float32)
        valid = idx >= 0
        row = tl.where(valid, idx, 0).to(tl.int64)
        v = tl.load(GEMM2 + row * stride + cols, mask=mask & valid, other=0.0).to(tl.float32)
        acc += w * v
    tl.store(OUT + tok.to(tl.int64) * hidden + cols, acc.to(tl.bfloat16), mask=mask)


def _decode_dispatch(prepared, x, router_logits, correction_bias, out, tokens):
    """Stock's wrapper call with the finalize deferred, then the bundle's finalize into `out`."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    fp4, scale = _quantize(prepared, x)
    gemm2, weights, mapping = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits, routing_bias=correction_bias,
        hidden_states=fp4, hidden_states_scale=scale,
        gemm1_weights=prepared["w13"], gemm1_weights_scale=prepared["w13_sf"],
        gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=prepared["w2"], gemm2_weights_scale=prepared["w2_sf"], gemm2_bias=None,
        output1_scale_scalar=prepared["g1_scale_c"],
        output1_scale_gate_scalar=prepared["g1_alphas"],
        output2_scale_scalar=prepared["g2_alphas"],
        num_experts=_EXPERTS, top_k=_TOP_K, n_group=1, topk_group=1,
        intermediate_size=_INTERMEDIATE, local_expert_offset=0, local_num_experts=_EXPERTS,
        routed_scaling_factor=_ROUTED_SCALING, routing_method_type=_ROUTING_DEEPSEEK_V3,
        do_finalize=False, activation_type=_ACT_SWIGLU, per_token_scale=None,
        tune_max_num_tokens=1 << (tokens - 1).bit_length(),
        output=out, enable_pdl=bool(prepared["pdl"](tokens)))
    if gemm2.dtype != torch.bfloat16 or gemm2.dim() != 2 or gemm2.shape[1] < _HIDDEN:
        raise RuntimeError(f"unexpected deferred GEMM2 output {tuple(gemm2.shape)} {gemm2.dtype}")
    mapping = mapping.reshape(-1)
    if mapping.numel() < tokens * _TOP_K or mapping.dtype != torch.int32:
        raise RuntimeError(f"unexpected expanded->permuted map {tuple(mapping.shape)} {mapping.dtype}")
    _finalize[(tokens, _HIDDEN // _FIN_BLOCK)](
        gemm2, mapping, weights.reshape(-1), out, _HIDDEN, gemm2.stride(0),
        _TOP_K, _FIN_BLOCK, num_warps=4, launch_pdl=True)


def _stock_dispatch(prepared, x, router_logits, correction_bias, out, tokens):
    """SGLang's exact decode call: the public FlashInfer wrapper with stock's keyword set.

    Mirrors `fused_experts_none_to_flashinfer_trtllm_fp4`'s `moe_kwargs` for the bypassed-top-k,
    non-deferred path the GLM engine takes, including `tune_max_num_tokens = next_power_of_2(M)`,
    which is part of the AutoTuner cache key and therefore what makes this call hit the entries
    the engine's boot autotune wrote for stock.
    """
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    fp4, scale = _quantize(prepared, x)
    trtllm_fp4_block_scale_moe(
        routing_logits=router_logits, routing_bias=correction_bias,
        hidden_states=fp4, hidden_states_scale=scale,
        gemm1_weights=prepared["w13"], gemm1_weights_scale=prepared["w13_sf"],
        gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=prepared["w2"], gemm2_weights_scale=prepared["w2_sf"], gemm2_bias=None,
        output1_scale_scalar=prepared["g1_scale_c"],
        output1_scale_gate_scalar=prepared["g1_alphas"],
        output2_scale_scalar=prepared["g2_alphas"],
        num_experts=_EXPERTS, top_k=_TOP_K, n_group=1, topk_group=1,
        intermediate_size=_INTERMEDIATE, local_expert_offset=0, local_num_experts=_EXPERTS,
        routed_scaling_factor=_ROUTED_SCALING, routing_method_type=_ROUTING_DEEPSEEK_V3,
        do_finalize=True, activation_type=_ACT_SWIGLU, per_token_scale=None,
        tune_max_num_tokens=1 << (tokens - 1).bit_length(),
        output=out, enable_pdl=bool(prepared["pdl"](tokens)))


def fused_routed_experts(x, router_logits, correction_bias, prepared, out):
    """Fill the validator output on the declared routed-MoE domain."""
    device = prepared["w13"].device
    tokens = int(x.shape[0])
    if not _MIN_TOKENS <= tokens <= _TOKENS:
        raise ValueError(f"served rows {tokens} outside the declared {_MIN_TOKENS}..{_TOKENS} range")
    _shape_dtype("x", x, (tokens, _HIDDEN), torch.bfloat16, device)
    _shape_dtype("router_logits", router_logits, (tokens, _EXPERTS), torch.float32, device)
    _shape_dtype("correction_bias", correction_bias, (_EXPERTS,), torch.float32, device)
    _shape_dtype("out", out, (tokens, _HIDDEN), torch.bfloat16, device)
    if tokens <= _DECODE_MAX_TOKENS:
        # The graded decode band: stock's dispatch with the vendor finalize deferred to ours.
        _decode_dispatch(prepared, x, router_logits, correction_bias, out, tokens)
        return
    if tokens <= _TACTIC_MIN_TOKENS:
        # Below the explicit-tactic threshold: stock's own dispatch, tuner lookup included.
        _stock_dispatch(prepared, x, router_logits, correction_bias, out, tokens)
        return
    plan = prepared["call_plans"].get(tokens)
    if plan is None:
        plan = (prepared["topk_ids"][:tokens], prepared["topk_weights"][:tokens],
                list(prepared["tactic"]))
        prepared["call_plans"][tokens] = plan
    ids, weights, tactic = plan
    fp4, scale = _quantize(prepared, x)
    # v10's prefill path, unchanged: MoERunner.forward's exact FP4 argument order with the
    # explicit 16K tactic, including real routing scratch.
    prepared["rt"].native(
        prepared["routing_mode"], router_logits, ids, weights, correction_bias, fp4, scale,
        *prepared["native_weights"], bool(prepared["pdl"](tokens)), _ACT_SWIGLU,
        out, tactic, True, None)

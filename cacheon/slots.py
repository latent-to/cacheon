"""Typed op-slot catalog — the submission ABI.

A *slot* is a replaceable, narrowly-typed region of the fixed model graph. The
validator owns this catalog; a miner may only target a slot that exists here, and
provides the small ``entry`` callable described by the slot's contract. Everything
around the slot (tensor allocation, the call site, the rest of the model) stays
validator-owned.

A slot comes in two ``kind``s, and the difference is only the *breadth* of the
typed boundary — the cheat-resistance story is identical for both: the validator
allocates the outputs, the miner only fills them, and the miner never produces the
final tokens/logprobs (so there is nothing to substitute, the attack that bites
whole-model submissions).

* ``"op"`` — a single fused op. ``silu_and_mul`` (``entry(x, out)``), ``rmsnorm``
  (``entry(x, weight, out, eps)``).
* ``"block"`` — a region that fuses several ops behind one tensor-in/tensor-out
  contract, for bigger wins. ``moe.fused_experts`` (dispatch + expert GEMMs +
  activation + combine behind one call) is the canonical one. A block has the *same
  shape* of contract as an op (named tensor inputs -> validator-allocated outputs),
  just wider — which is exactly why the seam / verify / registry machinery is
  unchanged. The breadth is bounded: a slot must stay strictly upstream of the
  logprobs/sampler, or the output-substitution attack reappears.

Each slot carries everything the validator needs to verify a submission without
trusting it: a trusted high-precision ``invoke_reference``, a deterministic input
generator, the standard shapes, per-dtype tolerances, explicit
``invoke_reference`` / ``invoke_entry`` (so non-uniform call shapes work), and a
``Correctness`` policy. The policy matters once a kernel legitimately changes
numerics (flash-style softmax reductions, fp8, MLA weight absorption): such kernels
are NOT bit-exact to the reference, so the gate is a *matched ratio* (>= rho of
elements within tolerance against high-precision ground truth) rather than
all-close — the deterministic-vs-low-precision tiering from FlashInfer-Bench. The
reference is always high-precision ground truth, never the stock kernel.

Some slots are a **(prepare, forward) pair**: a quantized / layout-sensitive kernel
(MoE experts, a quant GEMM) needs the *weights* in a custom layout, and that layout
transform is part of the kernel. Such a slot names a second miner callable via
``prepare`` — it runs ONCE at load on the raw checkpoint weights, the validator holds
the result, and ``entry`` (forward) consumes it each step as ``prepared``. A quantized
fused-MoE (repack the expert weights, interleave the FP4 block scales, then a fused
GEMM) fits *one* slot this way: the repack/interleave is ``prepare``, the kernel is
``forward``.

Adding a slot is a validator action (a code change here), never a miner action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from cacheon.artifact_abi import (
    COLLECTIVE_ALL_REDUCE_CALL_ABI,
    COLLECTIVE_AR_RESIDUAL_RMSNORM_CALL_ABI,
    COLLECTIVE_MOE_FINALIZE_AR_RMSNORM_CALL_ABI,
    RMSNORM_CALL_ABI,
    SILU_AND_MUL_CALL_ABI,
    SlotCallABI,
)
from cacheon.moe_nvfp4_contract import (
    prepare_args_from_inputs as _moe_prepare_args_from_inputs,
    prepare_args_from_layer as _moe_prepare_args_from_layer,
    verification_inputs as _moe_nvfp4_verification_inputs,
)
from cacheon.tensor_spec import OutputSpec, TensorSpec


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass(frozen=True)
class Correctness:
    """How ``verify`` compares the miner output to the reference.

    * ``"allclose"`` — every element must satisfy ``|a-e| <= atol + rtol*|e|``.
      Right for kernels meant to be numerically equivalent (a faster silu).
    * ``"matched_ratio"`` — at least ``min_ratio`` of elements must satisfy that
      bound. Right for kernels that legitimately differ from the reference at the
      ULP level (attention reorders the softmax reduction; fp8 / weight-absorbed
      forms shift a few elements). Calibrate ``min_ratio`` to the stock-vs-stock
      noise floor — the same discipline as the KL gate.
    * ``"cosine"`` — cosine similarity of the flattened output vs the HP reference
      must be >= ``min_cosine``, with an optional relative-L2-norm guard
      (``max_rel_norm_err``) to catch a kernel that gets the direction right but the
      scale wrong. This is the correct fidelity metric for **low-bit** kernels
      (FP4/FP8): element-wise tolerance is meaningless when every element carries
      ~6-12% quantization error, but the *direction* (and energy) of the block output
      is preserved — which is what actually drives the model's logits.
    * ``"topk_overlap"`` — for an MSA score sheet or direct index output. Values
      and index order do not matter; the selected block sets must meet
      ``min_overlap`` before the validator-owned attention consumes them.

    DESIGN NOTE — this is the *op-correctness* gate, a cheap **sanity** check ("is this
    even computing the slot's function?"), explicitly necessary-but-not-sufficient
    (verify.py). It is NOT the fidelity authority: the load-bearing anti-cheat gate is
    the end-to-end per-token **KL on the model's logits** (cacheon.eval), which is exactly
    where a temp-0 distributional metric belongs. The op-gate's only job is to never let
    through a kernel computing the WRONG function (e.g. plain SiLU on a swigluoai model:
    cosine 0.45) while never false-failing a kernel the e2e KL gate accepts (a faithful
    low-bit kernel: cosine 0.996). Hence: same-function reference + a validator-owned
    floor, never a per-element bound that the irreducible quant noise alone would trip.
    """

    mode: str = "allclose"  # "allclose" | "matched_ratio" | "cosine" | "topk_overlap"
    min_ratio: float = 1.0
    min_cosine: float = 0.0  # cosine mode: min cosine similarity vs the HP reference
    max_rel_norm_err: float = 0.0  # cosine mode: optional |‖a‖-‖e‖|/‖e‖ guard (0 = off)
    top_k: int = 0  # topk_overlap mode: the K of the selection (e.g. 16 blocks)
    min_overlap: float = 0.0  # topk_overlap mode: required mean per-row set overlap


@dataclass(frozen=True)
class Activation:
    """The gated-MLP activation a model's MoE/FFN uses — a MODEL fact (read from the
    model's config), NOT a miner choice. ``silu`` is the Qwen/Llama default
    ``silu(gate)*up``; ``swigluoai`` is the clamped GPT-OSS / MiniMax-M3 form
    ``g=min(gate,limit); u=clamp(up,-limit,limit); g*sigmoid(alpha*g)*(u+1)``."""

    kind: str = "silu"  # "silu" | "swigluoai"
    alpha: float = 1.702  # swigluoai sigmoid gain (config swiglu_alpha)
    limit: float = 7.0  # swigluoai clamp (config swiglu_limit)


_SILU = Activation("silu")


@dataclass(frozen=True)
class SlotSpec:
    name: str  # dotted slot id, e.g. "activation.silu_and_mul"
    entry: str  # required callable name the miner module must expose
    summary: str  # human-readable contract
    kind: str  # "op" (single fused op) | "block" (a region of several fused ops)

    make_inputs: Callable[..., dict]  # (**shape, dtype, device, seed) -> {name: tensor|scalar}
    out_shapes: Callable[[dict], Sequence[tuple]]  # (inputs) -> one shape per output the validator allocates
    invoke_reference: Callable[[dict], Sequence[torch.Tensor]]  # (inputs) -> expected outputs (HIGH PRECISION)
    invoke_entry: Callable[..., None]  # (entry, inputs, outs, prepared) -> None; writes each tensor in `outs`
    shapes: tuple[dict, ...]
    # Tensor inputs whose values may change between CUDA-graph replays while their
    # addresses/shapes remain fixed. Verification refreshes them in place and grades
    # every replay against a fresh trusted reference, so a captured graph that merely
    # copies a cached answer cannot pass. Model weights and prepare-time state are
    # deliberately absent. Python scalars are capture-static; a future slot that needs
    # one to vary within a graph bucket must tensorize it.
    graph_dynamic_inputs: tuple[str, ...] = ()
    # Is this slot's live serving seam inside the captured CUDA-graph region? The
    # validator owns this, not the miner: production captures the candidate
    # unconditionally, so a slot that serves from a captured region must PROVE
    # capture+replay to be crownable. Only prefill seams, which SGLang runs eager,
    # set this False.
    serving_graph_captured: bool = True
    # Additive typed-output ABI. Existing slots inherit dtype/device and remain
    # contiguous through ``out_shapes``.
    output_spec: Optional[Callable[[dict], OutputSpec]] = None
    # Optional 2nd miner callable for (prepare, forward) slots: `prepare` runs ONCE at
    # load on the raw weights (quant/layout transform); the validator holds the result
    # and passes it to `entry` each step as `prepared`. None -> a plain forward-only slot.
    prepare: Optional[str] = None
    invoke_prepare: Optional[Callable] = None  # (prepare_fn, inputs) -> prepared (None for forward-only)
    # Live seam: build the args for the miner's prepare() from the actual sglang layer
    # (validator-owned layer->contract mapping). The dispatcher calls
    # prepare(*prepare_from_layer(layer)); invoke_prepare mirrors the SAME call shape for
    # verify. This is how a slot carries more than two dense tensors (biases, the
    # interleaving flag, quant scales) without widening the generic contract. None ->
    # the dispatcher defaults to (layer.w13_weight.data, layer.w2_weight.data).
    prepare_from_layer: Optional[Callable] = None
    correctness: Correctness = field(default_factory=Correctness)
    tolerances: dict[torch.dtype, Tolerance] = field(default_factory=dict)
    # Collective slots (kind="collective") are verified DISTRIBUTED, so the single-process
    # invoke_reference/invoke_entry don't apply. These two hooks let cacheon.verify_collective
    # drive ANY collective slot (a bare all-reduce, OR a block that OWNS its trailing reduce
    # like moe.fused_experts_reduce) without hard-coding one contract:
    #   * collective_partial(inputs, prepared) -> the fp32 per-rank tensor whose cross-rank
    #     SUM is the trusted reference (x for all-reduce; the local experts' fp32 output for
    #     the MoE-overlap block).
    #   * invoke_collective(entry, inputs, out, group, prepared) -> call the miner kernel,
    #     handing it the process group; it fills `out` with the REDUCED result.
    collective_partial: Optional[Callable] = None
    invoke_collective: Optional[Callable] = None
    # Optional post-reduce transform for distributed verify: some collective slots do
    # trusted local math AFTER the cross-rank sum (e.g. residual-add + RMSNorm in
    # collective.ar_residual_rmsnorm). ``collective_finish(inputs, summed, prepared)``
    # maps the fp32 cross-rank SUM of ``collective_partial`` to the list of expected
    # outputs, one per ``out_shapes`` entry. None -> the reference is the sum itself and
    # the slot has exactly one output (the pre-existing all-reduce contract).
    collective_finish: Optional[Callable] = None
    # Per-slot end-to-end KL gate, calibrated to THIS slot's intrinsic noise floor (the
    # generic 5e-3 default is tuned for elementwise ops; attention sits ~6e-3 vs flash's
    # reordered softmax, so a flat 5e-3 false-fails a faithful attention kernel — README
    # calibration finding 6). None -> use the eval's generic threshold.
    kl_threshold: Optional[float] = None
    # Provider-neutral, declarative resource ABI for sealed AOT/native artifacts.
    # It is additive and intentionally appended after historical fields. A miner's
    # launch plan may only bind these validator-owned resources; it cannot name a
    # per-submission Python adapter. Every catalog slot points at the shared immutable
    # row in artifact_abi; provider support may still fail closed at build/runtime.
    call_abi: Optional[SlotCallABI] = None

    def tolerance_for(self, dtype: torch.dtype) -> Tolerance:
        if dtype in self.tolerances:
            return self.tolerances[dtype]
        if dtype in (torch.float16, torch.bfloat16):
            return Tolerance(atol=2e-2, rtol=2e-2)
        return Tolerance(atol=1e-4, rtol=1e-4)

    def output_contract(self, inputs: dict) -> OutputSpec:
        """Resolve one output declaration for both verify and live bindings."""
        if self.output_spec is not None:
            contract = self.output_spec(inputs)
            if not isinstance(contract, OutputSpec):
                raise TypeError(
                    f"slot {self.name!r} output_spec returned "
                    f"{type(contract).__name__}, expected OutputSpec"
                )
            return contract

        shapes = self.out_shapes(inputs)
        if isinstance(shapes, tuple) and (
            not shapes or isinstance(shapes[0], int)
        ):
            shapes = [shapes]
        return OutputSpec(
            tuple(
                TensorSpec(shape=tuple(shape), name=f"out[{index}]")
                for index, shape in enumerate(shapes)
            )
        )


_BF16_TOL = {
    torch.bfloat16: Tolerance(2e-2, 2e-2),
    torch.float16: Tolerance(1e-2, 1e-2),
    torch.float32: Tolerance(1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Slot (op): activation.silu_and_mul   (Qwen/Llama-class MLP)
#   x:(...,2d) -> out:(...,d) = silu(x[...,:d]) * x[...,d:]
#   contract: entry(x, out)
# ---------------------------------------------------------------------------


def _silu_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d].float()).to(x.dtype) * x[..., d:]


def _silu_inputs(*, num_tokens: int, d: int, dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, 2 * d, generator=g, device=device, dtype=torch.float32).to(dtype)
    return {"x": x}


SILU_AND_MUL = SlotSpec(
    name="activation.silu_and_mul",
    entry="silu_and_mul",
    summary="out = silu(x[...,:d]) * x[...,d:];  x:(...,2d) -> out:(...,d);  entry(x, out)",
    kind="op",
    make_inputs=_silu_inputs,
    out_shapes=lambda i: [(*i["x"].shape[:-1], i["x"].shape[-1] // 2)],
    invoke_reference=lambda i: [_silu_reference(i["x"])],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], outs[0]),
    graph_dynamic_inputs=("x",),
    shapes=(
        {"num_tokens": 1, "d": 1024},
        {"num_tokens": 8, "d": 1024},
        {"num_tokens": 128, "d": 4096},
        {"num_tokens": 4096, "d": 4096},
        {"num_tokens": 333, "d": 2880},
    ),
    correctness=Correctness("allclose"),
    tolerances=_BF16_TOL,
    call_abi=SILU_AND_MUL_CALL_ABI,
)


# ---------------------------------------------------------------------------
# Slot (op): norm.rmsnorm   (universal — every transformer, incl. GPT-OSS)
#   out = x / sqrt(mean(x^2, -1) + eps) * weight
#   contract: entry(x, weight, out, eps).  Validator owns the residual add.
# ---------------------------------------------------------------------------


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps)
    return (normed * weight.float()).to(x.dtype)


def _rmsnorm_inputs(*, num_tokens: int, hidden: int, dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn(hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    return {"x": x, "weight": w, "eps": 1e-6}


RMSNORM = SlotSpec(
    name="norm.rmsnorm",
    entry="rmsnorm",
    summary="out = x*rsqrt(mean(x^2,-1)+eps)*weight;  x:(...,H),weight:(H,) -> out:(...,H);  entry(x, weight, out, eps)",
    kind="op",
    make_inputs=_rmsnorm_inputs,
    out_shapes=lambda i: [tuple(i["x"].shape)],
    invoke_reference=lambda i: [_rmsnorm_reference(i["x"], i["weight"], i["eps"])],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], i["weight"], outs[0], i["eps"]),
    graph_dynamic_inputs=("x",),
    shapes=(
        {"num_tokens": 1, "hidden": 2880},
        {"num_tokens": 8, "hidden": 2880},
        {"num_tokens": 128, "hidden": 2880},
        {"num_tokens": 64, "hidden": 6144},
        {"num_tokens": 4096, "hidden": 4096},
        {"num_tokens": 333, "hidden": 1536},
    ),
    correctness=Correctness("allclose"),
    tolerances=_BF16_TOL,
    call_abi=RMSNORM_CALL_ABI,
)


# ---------------------------------------------------------------------------
# Slot (BLOCK, prepare+forward): moe.fused_experts
#   prepare(w13, w2) -> prepared              (weight layout; runs ONCE at load)
#   forward(x, topk_ids, topk_weights, prepared, out)   (per step)
#   x:(M,H)  w13:(E,2I,H)[gate;up]  w2:(E,H,I)  topk_ids/weights:(M,K) -> out:(M,H)
#   SwiGLU-MLP experts: out = sum_k topk_w * (silu(gate)*up) @ w2.T over each token's
#   top-k experts. The (prepare, forward) split is what lets a quantized / layout-
#   sensitive expert kernel fit one slot: a weight repack / FP4 block-scale interleave
#   is `prepare`, the fused GEMM is `forward`. The pure-torch example reorders
#   [gate;up]->[up;gate] in prepare to exercise the contract.
# ---------------------------------------------------------------------------


def _gated_activation(gate: torch.Tensor, up: torch.Tensor, act: Activation) -> torch.Tensor:
    """The fc1 -> intermediate activation. ``act`` (a MODEL fact) selects the form so the
    HP reference matches the model the kernel targets — using SiLU as the reference for a
    swigluoai model is the ratio-0.0 false-fail this fixes."""
    if act.kind == "swigluoai":
        g = gate.clamp(max=act.limit)
        u = up.clamp(min=-act.limit, max=act.limit)
        return g * torch.sigmoid(act.alpha * g) * (u + 1.0)
    return F.silu(gate) * up


def _moe_reference(x, w13, w2, topk_ids, topk_weights, act: Activation = _SILU):
    # x:(M,H) w13:(E,2I,H)[gate;up] w2:(E,H,I) topk_ids:(M,K) topk_weights:(M,K) -> (M,H)
    M, H = x.shape
    I = w13.shape[1] // 2
    K = topk_ids.shape[1]
    x32 = x.float()
    out = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    for k in range(K):
        e = topk_ids[:, k].long()
        wk = topk_weights[:, k].float()
        w13_e = w13[e].float()                          # (M,2I,H)
        w2_e = w2[e].float()                            # (M,H,I)
        fc1 = torch.einsum("mh,mih->mi", x32, w13_e)    # (M,2I)
        gate, up = fc1[:, :I], fc1[:, I:]
        act_out = _gated_activation(gate, up, act)      # (M,I)
        out += wk[:, None] * torch.einsum("mi,mhi->mh", act_out, w2_e)
    return out


def _raw_topk_weights(
    *, num_tokens: int, topk: int, generator: torch.Generator, device: str
) -> torch.Tensor:
    """Generate validator-owned raw routing multipliers, not probabilities.

    Sglang can hand the expert path unnormalized weights (for example when routing
    renormalization is disabled or routed scaling is applied). Keeping these values
    positive but deliberately below one guarantees that ``topk == 1`` still varies
    across graph replay cases instead of collapsing to a column of ones.
    """

    return 0.25 + 0.5 * torch.rand(
        num_tokens, topk, generator=generator, device=device, dtype=torch.float32
    )


def _moe_inputs(*, num_tokens: int, num_experts: int, hidden: int, inter: int, topk: int,
                dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * scale).to(dtype)

    ids = torch.randint(0, num_experts, (num_tokens, topk), generator=g, device=device).to(torch.int32)
    weights = _raw_topk_weights(
        num_tokens=num_tokens, topk=topk, generator=g, device=device
    )
    return {
        "x": rnd(num_tokens, hidden, scale=0.1),
        "w13": rnd(num_experts, 2 * inter, hidden, scale=0.05),
        "w2": rnd(num_experts, hidden, inter, scale=0.05),
        "topk_ids": ids,
        "topk_weights": weights,
    }


MOE_FUSED_EXPERTS = SlotSpec(
    name="moe.fused_experts",
    entry="fused_experts",
    prepare="prepare",
    summary=(
        "fused MoE experts — a (prepare, forward) PAIR.  prepare(w13, w2) -> prepared "
        "(weight layout, once at load);  forward(x, topk_ids, topk_weights, prepared, out).  "
        "x:(M,H) w13:(E,2I,H)[gate;up] w2:(E,H,I) -> out:(M,H);  SwiGLU-MLP experts."
    ),
    kind="block",
    make_inputs=_moe_inputs,
    out_shapes=lambda i: [(i["x"].shape[0], i["x"].shape[1])],
    invoke_reference=lambda i: [_moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"])],
    invoke_prepare=lambda prepare_fn, i: prepare_fn(i["w13"], i["w2"]),
    prepare_from_layer=_moe_prepare_args_from_layer,
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], i["topk_ids"], i["topk_weights"], prepared, outs[0]),
    graph_dynamic_inputs=("x", "topk_ids", "topk_weights"),
    shapes=(
        {"num_tokens": 4, "num_experts": 8, "hidden": 256, "inter": 128, "topk": 2},
        {"num_tokens": 16, "num_experts": 32, "hidden": 512, "inter": 256, "topk": 4},
        {"num_tokens": 8, "num_experts": 4, "hidden": 384, "inter": 192, "topk": 1},
        {"num_tokens": 33, "num_experts": 16, "hidden": 320, "inter": 192, "topk": 4},
    ),
    # A real fused-MoE kernel runs in fp8/fp4 with reordered reductions -> not bit-exact;
    # gate on a matched ratio vs the fp32 reference, calibrated to the stock noise floor.
    correctness=Correctness("matched_ratio", min_ratio=0.97),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot (BLOCK): moe.fused_routed_experts   (the FAT MoE slot: routing + experts
#   + combine in one contract)
#
# The boundary every served MoE engine of this class must have: FusedMoE.forward_impl
# receiving (hidden_states, router LOGITS). On runner backends that route inside the
# fused kernel (flashinfer_trtllm hands the seam a BypassedTopKOutput carrying
# router_logits only), the thin fused_experts contract cannot bind — this slot is
# that boundary. Measured 2026-08-30 (GLM-5.3 decode cell): routing + both expert
# GEMMs + finalize + input quant ≈ 53% of decode GPU time.
#
# Routing math mirrors the pinned engine exactly (topk.py biased_grouped_topk_impl,
# no group step at n_group=1): select on sigmoid(logits) + correction_bias, weight
# by the UNBIASED sigmoid scores gathered at the selection, renormalize with the
# +1e-20 epsilon, scale by routed_scaling. The near-tie discontinuity that makes a
# single output gate false-fail honest kernels (two experts tied at rank k) is
# killed at INPUT SYNTHESIS, not with a hybrid metric: the input builder boosts the
# selected experts' logits so every verification row has an unambiguous top-k, and
# real-serving tie behavior stays covered by the end-to-end quality gates.
# ---------------------------------------------------------------------------


def _routed_moe_inputs(*, num_tokens: int, num_experts: int, hidden: int, inter: int,
                       topk: int, routed_scaling: float = 1.0, dtype: torch.dtype,
                       device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * scale).to(dtype)

    logits = torch.randn(num_tokens, num_experts, generator=g, device=device,
                         dtype=torch.float32)
    bias = torch.randn(num_experts, generator=g, device=device,
                       dtype=torch.float32) * 0.1
    # Margin enforcement: lift the already-selected experts' logits so the
    # selection gap at rank `topk` is unambiguous for every row (the boost is
    # monotone in choice score, so the selected SET is unchanged).
    choice = torch.sigmoid(logits) + bias.unsqueeze(0)
    chosen = choice.topk(topk, dim=-1).indices
    logits = logits.scatter_add(
        1, chosen, torch.ones_like(chosen, dtype=torch.float32)
    )
    return {
        "x": rnd(num_tokens, hidden, scale=0.1),
        "w13": rnd(num_experts, 2 * inter, hidden, scale=0.05),
        "w2": rnd(num_experts, hidden, inter, scale=0.05),
        "router_logits": logits,
        "correction_bias": bias,
        "topk": int(topk),
        "routed_scaling": float(routed_scaling),
    }


def _routed_moe_topk(router_logits: torch.Tensor, correction_bias: torch.Tensor,
                     topk: int, routed_scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The pinned engine's routing head in fp32 (selection biased, weights unbiased)."""
    scores = torch.sigmoid(router_logits.float())
    choice = scores + correction_bias.float().unsqueeze(0)
    topk_ids = torch.topk(choice, k=topk, dim=-1).indices
    weights = scores.gather(1, topk_ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return topk_ids.to(torch.int32), (weights * routed_scaling).float()


def _routed_moe_reference(i: dict, act: Activation = _SILU) -> torch.Tensor:
    topk_ids, topk_weights = _routed_moe_topk(
        i["router_logits"], i["correction_bias"], i["topk"], i["routed_scaling"]
    )
    return _moe_reference(i["x"], i["w13"], i["w2"], topk_ids, topk_weights, act)


MOE_FUSED_ROUTED_EXPERTS = SlotSpec(
    name="moe.fused_routed_experts",
    entry="fused_routed_experts",
    prepare="prepare",
    summary=(
        "fused ROUTED MoE — routing + experts + combine, a (prepare, forward) PAIR. "
        "prepare(w13, w2, topk, routed_scaling) -> prepared (once at load); "
        "forward(x, router_logits, correction_bias, prepared, out). "
        "x:(M,H) router_logits:(M,E) correction_bias:(E,) w13:(E,2I,H)[gate;up] "
        "w2:(E,H,I) -> out:(M,H). Selection = topk(sigmoid(logits)+bias); weights = "
        "unbiased sigmoid scores renormalized (+1e-20) and scaled by routed_scaling."
    ),
    kind="block",
    make_inputs=_routed_moe_inputs,
    out_shapes=lambda i: [(i["x"].shape[0], i["x"].shape[1])],
    invoke_reference=lambda i: [_routed_moe_reference(i)],
    invoke_prepare=lambda prepare_fn, i: prepare_fn(
        i["w13"], i["w2"], i["topk"], i["routed_scaling"]
    ),
    # Live layers map through the same nvfp4-aware contract as the thin slot;
    # the dispatcher appends (top_k, routed_scaling) from the engine's TopKConfig.
    prepare_from_layer=_moe_prepare_args_from_layer,
    invoke_entry=lambda entry, i, outs, prepared: entry(
        i["x"], i["router_logits"], i["correction_bias"], prepared, outs[0]
    ),
    graph_dynamic_inputs=("x", "router_logits"),
    shapes=(
        {"num_tokens": 4, "num_experts": 8, "hidden": 256, "inter": 128, "topk": 2,
         "routed_scaling": 1.0},
        {"num_tokens": 16, "num_experts": 32, "hidden": 512, "inter": 256, "topk": 4,
         "routed_scaling": 2.5},
        {"num_tokens": 33, "num_experts": 16, "hidden": 320, "inter": 192, "topk": 4,
         "routed_scaling": 1.0},
    ),
    correctness=Correctness("matched_ratio", min_ratio=0.97),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Slot (COLLECTIVE): collective.all_reduce   (the TP comms waist)
#   x:(M,H) on each of `world_size` ranks -> out:(M,H) = sum over ranks.
#   contract: entry(x, out, group)  — miner owns the reduce algorithm + transport;
#   validator owns `out`, the process group, and the call site.
#
# Unlike op/block slots, a collective spans GPUs: the kernel needs the TP process group
# to move data across ranks, so it is verified DISTRIBUTED (cacheon.verify_collective,
# NOT verify_entry) against the trusted fp32 cross-rank sum. The reduce is mid-network
# (upstream of the sampler) — no output to substitute. Decode is comms-bound (~32–43%
# of GPU time at TP/EP scale, the largest single category), and it is *latency*-bound,
# so the lever is a lower-latency reduce or compute-comm overlap — both expressible here
# while staying inside the four invariants. WIDER SURFACE: handing the miner the
# communicator is more capability than "fill a tensor"; the invariants still bound it,
# but distributed verify + the end-to-end gate are MANDATORY (docs/architecture/slot-contract.md).
# ---------------------------------------------------------------------------


def _all_reduce_inputs(*, num_tokens: int, hidden: int, dtype: torch.dtype, device: str,
                       seed: int, rank: int = 0, world_size: int = 1) -> dict:
    # Each rank gets a DIFFERENT partial (seeded by rank); the all-reduce sums them.
    g = torch.Generator(device=device).manual_seed(seed + 1_000_003 * rank)
    x = (torch.randn(num_tokens, hidden, generator=g, device=device, dtype=torch.float32) * 0.1).to(dtype)
    return {"x": x}


COLLECTIVE_ALL_REDUCE = SlotSpec(
    name="collective.all_reduce",
    entry="all_reduce",
    summary=(
        "TP all-reduce (the comms waist): x:(M,H) per rank -> out:(M,H) = sum over ranks;  "
        "entry(x, out, group).  Validator owns out + the process group; verified DISTRIBUTED "
        "vs the fp32 cross-rank sum (cacheon.verify_collective)."
    ),
    kind="collective",
    make_inputs=_all_reduce_inputs,
    out_shapes=lambda i: [tuple(i["x"].shape)],
    # Collectives are verified distributed: the real reference is the fp32 sum ACROSS
    # ranks, which a single-process invoke_reference can't compute. These two are unused
    # for kind="collective" (kept non-None to satisfy the dataclass); verify_collective
    # drives the real verification.
    invoke_reference=lambda i: [i["x"]],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], outs[0], i.get("__group__")),
    graph_dynamic_inputs=("x",),
    # Distributed-verify hooks: the reference is the fp32 SUM of each rank's x.
    collective_partial=lambda i, prepared: i["x"].float(),
    invoke_collective=lambda entry, i, out, group, prepared: entry(i["x"], out, group),
    shapes=(
        {"num_tokens": 1, "hidden": 4096},
        {"num_tokens": 8, "hidden": 4096},
        {"num_tokens": 128, "hidden": 7168},
    ),
    # A different reduce algorithm/order (one-shot/NVLS/tree vs ring) is not bit-exact;
    # gate on matched_ratio vs the fp32 sum, with the end-to-end token/KL gate mandatory.
    correctness=Correctness("matched_ratio", min_ratio=0.99),
    tolerances=_BF16_TOL,
    call_abi=COLLECTIVE_ALL_REDUCE_CALL_ABI,
)


# ---------------------------------------------------------------------------
# Slot (COLLECTIVE): collective.ar_residual_rmsnorm   (the decode-epilogue waist)
#   Per rank: x:(M,H) is that rank's LOCAL partial (e.g. the un-reduced MoE/MLP output);
#   residual:(M,H) and weight:(H,) are REPLICATED (identical on every rank). The kernel
#   owns the whole fused epilogue:
#     new_residual = sum_over_ranks(x) + residual
#     norm_out     = rmsnorm(new_residual, weight, eps)
#   contract: entry(x, residual, weight, eps, out_norm, out_residual, group)
#
# This is sglang's OWN fusion waist: with --enable-flashinfer-allreduce-fusion the
# layer epilogues funnel through ONE module-level function
# (sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm,
# resolved per-call via a function-local import), so the seam is a module-attribute
# rebind of a real single call site — the validator owns the call site, both output
# buffers, and the process group; the miner owns the reduce algorithm + the fused
# residual/norm math. Mid-network, upstream of the sampler: nothing to substitute.
# The measured lever here is the fused AR+add+norm epilogue (two-shot Lamport beats
# flashinfer's own fused kernel at decode T on B300 — the 2026-07-02 campaign), and
# the future compute-comm overlap consumes pre-reduce exports at this same boundary.
# Wider capability than "fill a tensor" -> verified DISTRIBUTED (the fp32 cross-rank
# sum then trusted local add+norm via collective_finish), end-to-end gate mandatory.
# ---------------------------------------------------------------------------


def _ar_norm_inputs(*, num_tokens: int, hidden: int, dtype: torch.dtype, device: str,
                    seed: int, rank: int = 0, world_size: int = 1) -> dict:
    # x differs per rank (it is the local partial the reduce sums); residual + norm
    # weight are the SAME on every rank (replicated model state), so they are seeded
    # WITHOUT rank. Getting this split wrong is exactly the shared-expert/replication
    # bug class the M3 campaign flagged — keep it explicit.
    gx = torch.Generator(device=device).manual_seed(seed + 1_000_003 * rank)
    gs = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(num_tokens, hidden, generator=gx, device=device, dtype=torch.float32) * 0.1).to(dtype)
    residual = (torch.randn(num_tokens, hidden, generator=gs, device=device, dtype=torch.float32) * 0.1).to(dtype)
    weight = (torch.rand(hidden, generator=gs, device=device, dtype=torch.float32) * 0.5 + 0.75).to(dtype)
    return {"x": x, "residual": residual, "weight": weight, "eps": 1e-6}


def _ar_norm_reference_from_sum(inputs: dict, summed: "torch.Tensor", prepared) -> list:
    # Trusted fp32 math applied AFTER the cross-rank sum: residual add, then RMSNorm.
    new_residual = summed + inputs["residual"].float()
    var = new_residual.pow(2).mean(dim=-1, keepdim=True)
    norm_out = new_residual * torch.rsqrt(var + float(inputs["eps"])) * inputs["weight"].float()
    return [norm_out, new_residual]


COLLECTIVE_AR_RESIDUAL_RMSNORM = SlotSpec(
    name="collective.ar_residual_rmsnorm",
    entry="ar_residual_rmsnorm",
    summary=(
        "fused all-reduce + residual-add + RMSNorm (the decode-epilogue waist behind "
        "sglang's --enable-flashinfer-allreduce-fusion): x:(M,H) per-rank partial, "
        "residual/weight replicated -> out_residual = sum_over_ranks(x) + residual, "
        "out_norm = rmsnorm(out_residual, weight, eps).  "
        "entry(x, residual, weight, eps, out_norm, out_residual, group).  Validator owns "
        "both outputs + the group; verified DISTRIBUTED (fp32 cross-rank sum, then the "
        "trusted add+norm via collective_finish)."
    ),
    kind="collective",
    make_inputs=_ar_norm_inputs,
    # Two validator-allocated outputs: [norm_out, new_residual] — the stock chokepoint
    # returns exactly this pair.
    out_shapes=lambda i: [tuple(i["x"].shape), tuple(i["x"].shape)],
    # Single-process hooks are unused for kind="collective" (verify_collective drives the
    # real check); kept semantically correct for the world_size=1 degenerate case.
    invoke_reference=lambda i: _ar_norm_reference_from_sum(i, i["x"].float(), None),
    invoke_entry=lambda entry, i, outs, prepared: entry(
        i["x"], i["residual"], i["weight"], i["eps"], outs[0], outs[1], i.get("__group__")),
    graph_dynamic_inputs=("x", "residual"),
    collective_partial=lambda i, prepared: i["x"].float(),
    invoke_collective=lambda entry, i, outs, group, prepared: entry(
        i["x"], i["residual"], i["weight"], i["eps"], outs[0], outs[1], group),
    collective_finish=_ar_norm_reference_from_sum,
    shapes=(
        {"num_tokens": 4, "hidden": 4096},
        # DECODE-sized T at the ARENA hidden: the expected kernel class mode-switches on
        # T (one-shot small / two-shot large), and an H-gated kernel routes off-H shapes
        # to its reference — so without small-T AT the arena H, verify never exercises
        # the one-shot CUDA path at all. That exact hole shipped an engine-garbage
        # kernel past verify on 2026-07-07 (engine decode T=8 = one-shot, unverified).
        # A slot's shape set must cover every dispatch mode of its kernel class.
        {"num_tokens": 8, "hidden": 6144},
        {"num_tokens": 32, "hidden": 6144},
        {"num_tokens": 64, "hidden": 6144},
        {"num_tokens": 256, "hidden": 6144},
    ),
    # Reduce order + norm rounding differ across algorithms (one-shot/two-shot/ring);
    # gate on matched_ratio vs the fp32 composed reference, e2e token/KL gate mandatory.
    correctness=Correctness("matched_ratio", min_ratio=0.99),
    tolerances=_BF16_TOL,
    call_abi=COLLECTIVE_AR_RESIDUAL_RMSNORM_CALL_ABI,
)


# ---------------------------------------------------------------------------
# Slot (collective): collective.moe_finalize_ar_rmsnorm — the DEEP fused-epilogue
# waist: MoE finalize (gather permuted gemm2 rows, scale, sum over experts-per-token)
# + all-reduce + residual-add + RMSNorm in ONE kernel. This is the fe_export deep
# seam's kernel contract: the producer dep-patch exports flashinfer's pre-finalize
# pointers (gemm_output / row_map / scales) instead of launching the standalone
# finalize kernel, and the deferred-AR call site consumes them here — killing a
# ~17us/layer latency-bound kernel + a full [T,H] round-trip at decode.
#
# Verifiable WITHOUT flashinfer: the validator seeds synthetic pre-finalize tensors
# per rank, and finalize is LINEAR, so finalize-then-AR == AR-then-finalize:
# collective_partial = trusted fp32 LOCAL finalize per rank, verify sums across
# ranks, collective_finish = the same trusted add+norm as the shallow slot.
#
# ABI (matches fe_export.h, 2026-07-02 campaign):
#   gemm_out [T_exp*K, H]  per-rank partial (unfused gemm2 output, permuted rows)
#   row_map  [T_exp*K] i32 REPLICATED, K-MAJOR: slot (t, k) lives at t + k*T_exp
#   scales   [T_exp, K] f32 REPLICATED, T-MAJOR
#   residual [T, H], weight [H]  replicated; T <= T_exp (CUDA-graph batch padding:
#   the consume call may HEAD-TRIM — same data_ptr, offset-0 slice).
# ---------------------------------------------------------------------------


def _moe_fin_inputs(*, num_tokens: int, exp_tokens: int, topk: int, hidden: int,
                    dtype: torch.dtype, device: str, seed: int,
                    rank: int = 0, world_size: int = 1) -> dict:
    # gemm_out differs per rank (TP-sharded gemm2 emits per-rank partials the reduce
    # sums); routing (row_map/scales) and residual/weight are REPLICATED model/router
    # state -> seeded WITHOUT rank. Same replication-split discipline as the shallow
    # slot. num_tokens is jittered by verify; exp_tokens is clamped to keep T <= T_exp.
    exp_tokens = max(exp_tokens, num_tokens)
    rows = exp_tokens * topk
    gx = torch.Generator(device=device).manual_seed(seed + 1_000_003 * rank)
    gs = torch.Generator(device=device).manual_seed(seed)
    gemm_out = (torch.randn(rows, hidden, generator=gx, device=device,
                            dtype=torch.float32) * 0.1).to(dtype)
    row_map = torch.randperm(rows, generator=gs, device=device).to(torch.int32)
    scales = (torch.rand(exp_tokens, topk, generator=gs, device=device,
                         dtype=torch.float32) + 0.1) / topk
    residual = (torch.randn(num_tokens, hidden, generator=gs, device=device,
                            dtype=torch.float32) * 0.1).to(dtype)
    weight = (torch.rand(hidden, generator=gs, device=device,
                         dtype=torch.float32) * 0.5 + 0.75).to(dtype)
    return {"gemm_out": gemm_out, "row_map": row_map, "scales": scales,
            "residual": residual, "weight": weight, "eps": 1e-6}


def _moe_fin_local_finalize(inputs: dict, prepared=None) -> "torch.Tensor":
    # Trusted fp32 LOCAL finalize (this rank's partial): for token t,
    # acc[t] = sum_k scales[t,k] * gemm_out[row_map[t + k*T_exp]]; head-trim to T.
    t = inputs["residual"].shape[0]
    t_exp, k = inputs["scales"].shape
    per_k = inputs["gemm_out"].float()[inputs["row_map"].long().view(k, t_exp)]
    acc = (per_k * inputs["scales"].float().t().unsqueeze(-1)).sum(dim=0)
    return acc[:t]


COLLECTIVE_MOE_FINALIZE_AR_RMSNORM = SlotSpec(
    name="collective.moe_finalize_ar_rmsnorm",
    entry="moe_finalize_ar_rmsnorm",
    summary=(
        "DEEP fused MoE epilogue (the fe_export contract): finalize (gather permuted "
        "gemm2 rows via K-MAJOR row_map, scale by T-MAJOR scales, sum over K) + "
        "all-reduce + residual-add + RMSNorm, one kernel. gemm_out:(T_exp*K,H) per-rank "
        "partial; row_map/scales/residual/weight replicated; T<=T_exp (graph padding "
        "head-trim). entry(gemm_out, row_map, scales, residual, weight, eps, out_norm, "
        "out_residual, group). Validator owns both outputs + the group; verified "
        "DISTRIBUTED without flashinfer — finalize is linear, so the reference is "
        "trusted fp32 local finalize per rank -> cross-rank sum -> add+norm."
    ),
    kind="collective",
    make_inputs=_moe_fin_inputs,
    out_shapes=lambda i: [tuple(i["residual"].shape), tuple(i["residual"].shape)],
    invoke_reference=lambda i: _ar_norm_reference_from_sum(
        i, _moe_fin_local_finalize(i), None),
    invoke_entry=lambda entry, i, outs, prepared: entry(
        i["gemm_out"], i["row_map"], i["scales"], i["residual"], i["weight"], i["eps"],
        outs[0], outs[1], i.get("__group__")),
    graph_dynamic_inputs=("gemm_out", "row_map", "scales", "residual"),
    collective_partial=_moe_fin_local_finalize,
    invoke_collective=lambda entry, i, outs, group, prepared: entry(
        i["gemm_out"], i["row_map"], i["scales"], i["residual"], i["weight"], i["eps"],
        outs[0], outs[1], group),
    collective_finish=_ar_norm_reference_from_sum,
    shapes=(
        # K=5 = M3 (top4 + fused shared expert). Head-trim (T < T_exp) exercised in the
        # 1st/3rd shapes; num_tokens jitters per run, exp_tokens clamps to stay >= T.
        {"num_tokens": 8, "exp_tokens": 16, "topk": 5, "hidden": 4096},
        {"num_tokens": 64, "exp_tokens": 64, "topk": 5, "hidden": 6144},
        {"num_tokens": 224, "exp_tokens": 256, "topk": 5, "hidden": 6144},
    ),
    correctness=Correctness("matched_ratio", min_ratio=0.99),
    tolerances=_BF16_TOL,
    call_abi=COLLECTIVE_MOE_FINALIZE_AR_RMSNORM_CALL_ABI,
)


# This collective slot owns local experts and their one trailing TP reduction.


def _moe_reduce_inputs(*, num_tokens: int, num_experts: int, hidden: int, inter: int, topk: int,
                       dtype: torch.dtype, device: str, seed: int, rank: int = 0, world_size: int = 1) -> dict:
    # Tokens + routing are REPLICATED across ranks (seeded without rank), so every rank
    # runs the same tokens; the expert WEIGHTS are SHARDED (seeded WITH rank), so each
    # rank computes a different partial and the cross-rank reduce does real work.
    gx = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(num_tokens, hidden, generator=gx, device=device, dtype=torch.float32) * 0.1).to(dtype)
    ids = torch.randint(0, num_experts, (num_tokens, topk), generator=gx, device=device).to(torch.int32)
    weights = _raw_topk_weights(
        num_tokens=num_tokens, topk=topk, generator=gx, device=device
    )
    gw = torch.Generator(device=device).manual_seed(seed + 1_000_003 * rank)
    w13 = (torch.randn(num_experts, 2 * inter, hidden, generator=gw, device=device, dtype=torch.float32) * 0.05).to(dtype)
    w2 = (torch.randn(num_experts, hidden, inter, generator=gw, device=device, dtype=torch.float32) * 0.05).to(dtype)
    return {"x": x, "w13": w13, "w2": w2, "topk_ids": ids, "topk_weights": weights}


MOE_FUSED_EXPERTS_REDUCE = SlotSpec(
    name="moe.fused_experts_reduce",
    entry="fused_experts_reduce",
    prepare="prepare",
    summary=(
        "fused MoE experts that OWN the trailing TP all-reduce (the compute-comm overlap "
        "lever).  prepare(w13, w2) -> prepared;  "
        "forward(x, topk_ids, topk_weights, prepared, out, group) fills out with the "
        "SUM-over-ranks of the local expert output.  x:(M,H) -> out:(M,H);  verified DISTRIBUTED."
    ),
    kind="collective",
    make_inputs=_moe_reduce_inputs,
    out_shapes=lambda i: [(i["x"].shape[0], i["x"].shape[1])],
    # Single-process invoke_reference/entry are unused for kind="collective"; the real
    # reference is the cross-rank fp32 sum (collective_partial), driven by verify_collective.
    invoke_reference=lambda i: [_moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"])],
    invoke_entry=lambda entry, i, outs, prepared: None,
    graph_dynamic_inputs=("x", "topk_ids", "topk_weights"),
    invoke_prepare=lambda prepare_fn, i: prepare_fn(i["w13"], i["w2"]),
    prepare_from_layer=_moe_prepare_args_from_layer,
    # Reference partial = this rank's fp32 expert output (HP, from the RAW weights, NOT the
    # miner's `prepared`); the trusted cross-rank SUM is the full MoE output.
    collective_partial=lambda i, prepared: _moe_reference(
        i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"]).float(),
    invoke_collective=lambda entry, i, out, group, prepared: entry(
        i["x"], i["topk_ids"], i["topk_weights"], prepared, out, group),
    shapes=(
        {"num_tokens": 4, "num_experts": 8, "hidden": 256, "inter": 128, "topk": 2},
        {"num_tokens": 16, "num_experts": 32, "hidden": 512, "inter": 256, "topk": 4},
        {"num_tokens": 8, "num_experts": 4, "hidden": 384, "inter": 192, "topk": 1},
    ),
    correctness=Correctness("matched_ratio", min_ratio=0.97),
    tolerances=_BF16_TOL,
)


SLOTS: dict[str, SlotSpec] = {
    SILU_AND_MUL.name: SILU_AND_MUL,
    RMSNORM.name: RMSNORM,
    MOE_FUSED_EXPERTS.name: MOE_FUSED_EXPERTS,
    MOE_FUSED_ROUTED_EXPERTS.name: MOE_FUSED_ROUTED_EXPERTS,
    MOE_FUSED_EXPERTS_REDUCE.name: MOE_FUSED_EXPERTS_REDUCE,
    COLLECTIVE_ALL_REDUCE.name: COLLECTIVE_ALL_REDUCE,
    COLLECTIVE_AR_RESIDUAL_RMSNORM.name: COLLECTIVE_AR_RESIDUAL_RMSNORM,
    COLLECTIVE_MOE_FINALIZE_AR_RMSNORM.name: COLLECTIVE_MOE_FINALIZE_AR_RMSNORM,
}


def get_slot(name: str) -> SlotSpec:
    try:
        return SLOTS[name]
    except KeyError:
        known = ", ".join(sorted(SLOTS)) or "(none)"
        raise KeyError(f"unknown slot {name!r}; known slots: {known}") from None


def list_slots() -> list[str]:
    return sorted(SLOTS)


# ---------------------------------------------------------------------------
# Per-model slot policy — the VALIDATOR-OWNED specialization.
#
# A slot's default (above) is the generic case (SiLU experts, matched_ratio). But the
# *activation*, *quant format*, and the *correctness floor* for a given (model, slot) are
# MODEL/VALIDATOR facts, never miner choices: the validator controls the model it serves,
# reads swiglu_alpha/limit from its config, and calibrates the floor to the measured noise.
# A miner only NAMES which model it targets; the numbers below are the validator's.
#
# This is the precursor to a full per-model arena registry (docs: arenas). When that lands,
# these profiles move there; today it is the one validator-owned table that makes the MoE
# slot verifiable on a swigluoai/NVFP4 model (e.g. MiniMax-M3) without weakening the generic
# slot or letting a submission set its own gate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotProfile:
    """Validator-owned (model, slot) overrides. ``activation`` retargets the HP reference
    to the model's real activation; ``correctness`` (optional) swaps the op-sanity metric
    (e.g. cosine for a low-bit kernel). None fields keep the generic slot default.

    Every model fact a slot needs lives HERE, not in specialize_slot branches: the
    verification ``shapes`` (per-rank shards, at the arena TP), the fused-shared-expert
    count (0 when the arena serves shared experts unfused — read it from the live
    server_args, not the checkpoint), and the routed-weight scale the synthetic routing
    mimics. Rotating the arena adds a MODEL_PROFILES entry; it does not edit code."""

    activation: Activation = field(default_factory=Activation)
    correctness: Optional[Correctness] = None
    quant: Optional[str] = None
    shapes: Optional[tuple[dict, ...]] = None
    num_fused_shared_experts: int = 0
    routed_weight_scale: float = 1.0


_MOE_SLOTS = ("moe.fused_experts", "moe.fused_experts_reduce")
_ROUTED_MOE_SLOTS = ("moe.fused_routed_experts",)


def specialize_slot(slot: SlotSpec, profile: SlotProfile) -> SlotSpec:
    """Return a copy of ``slot`` retargeted by a validator ``profile``. Only rebinds the
    pieces the profile changes (the activation-bearing references + the correctness policy);
    everything else — inputs, shapes, seam wiring — is untouched. The module-level slot
    singletons are never mutated, so ``get_slot`` stays generic."""
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
            return [_moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"], _act)]

        repl["invoke_reference"] = _ref
        if slot.collective_partial is not None:  # the reduce block's distributed reference
            def _partial(i, prepared, _act=act):
                return _moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"], _act).float()

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
            # Distinct-expert routing draw (real routers never repeat an expert per
            # token, unlike the generic randint inputs). When the arena fuses shared
            # experts, the last `fused` expert ids are the shared ones by convention
            # and always ride with weight 1.0; `fused == 0` is a pure routed draw.
            routed = torch.rand(
                tokens, experts - fused, generator=generator, device=kwargs["device"]
            ).topk(routed_k, dim=-1).indices.to(torch.int32)
            scores = torch.rand(
                tokens, routed_k, generator=generator, device=kwargs["device"]
            )
            routed_weights = scale * scores / scores.sum(-1, keepdim=True)
            if fused:
                shared_ids = (
                    torch.arange(
                        experts - fused, experts,
                        device=kwargs["device"], dtype=torch.int32,
                    )
                    .expand(tokens, fused)
                )
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
    # Low-bit (NVFP4) experts: gate on cosine vs the same-function fp32 reference.
    # min_cosine = the measured NVFP4 representational floor (0.9958 at M3 shape,
    # m3_swigluoai_gate.py) with headroom; plain-SiLU scores 0.45 and is rejected.
    # No norm guard yet (max_rel_norm_err uncalibrated — TODO measure the floor).
    correctness=Correctness("cosine", min_cosine=0.985),
)
_M3_MOE_NVFP4_PROFILE = replace(
    _M3_MOE_PROFILE,
    quant="nvfp4",
    # Per-rank shards at the M3 arena's TP4 (moe_intermediate 3072 / 4). M3 serves
    # the shared expert FUSED into the routed kernel: 128 routed + 1 shared = 129,
    # top-4 routed + 1 = 5, routed weights scaled 2x.
    shapes=(
        {"num_tokens": 1, "num_experts": 129, "hidden": 6144, "inter": 768, "topk": 5},
        {"num_tokens": 8, "num_experts": 129, "hidden": 6144, "inter": 768, "topk": 5},
        {"num_tokens": 32, "num_experts": 129, "hidden": 6144, "inter": 768, "topk": 5},
    ),
    num_fused_shared_experts=1,
    routed_weight_scale=2.0,
)

_GLM53_MOE_NVFP4_PROFILE = SlotProfile(
    # GLM-5.3 experts are plain SiLU (config hidden_act=silu) — the Activation()
    # default. Shapes receipted from the served config + server_args 2026-08-30:
    # 256 routed experts, top-8, hidden 6144, moe_intermediate 2048 -> 512/rank at
    # the arena's TP4. The served path runs disable_shared_experts_fusion=True
    # (shared expert is unquantized and separate), so num_fused_shared_experts=0
    # and num_experts is 256, NOT 257 — read from the live engine, not the plan.
    # routed_weight_scale mirrors routed_scaling_factor=2.5 (norm_topk_prob=True).
    # correctness: measured at THIS shape by glm53_nvfp4_gate.py (2026-08-30,
    # E=256/inter 512/top-8/silu, M=2048): block floor with the served fp4xfp4
    # intermediate requant = cosine 0.9959 / rel_norm_err 0.0012; bar = floor
    # minus margin. The norm guard is load-bearing: cosine is scale-invariant,
    # and a kernel that drops routed_scaling ties the cosine floor exactly
    # (rejected only via rel_norm 0.60). Controls: swigluoai-act 0.44,
    # shuffled-routing 0.04, both cosine-rejected.
    correctness=Correctness("cosine", min_cosine=0.985, max_rel_norm_err=0.05),
    quant="nvfp4",
    shapes=(
        {"num_tokens": 1, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
        {"num_tokens": 8, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
        {"num_tokens": 32, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
        {"num_tokens": 128, "num_experts": 256, "hidden": 6144, "inter": 512, "topk": 8},
    ),
    num_fused_shared_experts=0,
    routed_weight_scale=2.5,
)

_GLM53_ROUTED_MOE_PROFILE = replace(
    _GLM53_MOE_NVFP4_PROFILE,
    # The fat slot owns the routing head too, so routed_scaling (2.5) is part of
    # the CONTRACT the miner implements, carried per shape into make_inputs.
    shapes=tuple(
        {**s, "routed_scaling": 2.5}
        for s in _GLM53_MOE_NVFP4_PROFILE.shapes
    ),
)

# model key (as a miner may declare it / as the validator keys its served model) -> {slot: profile}
MODEL_PROFILES: dict[str, dict[str, SlotProfile]] = {
    "MiniMax-M3": {
        # BOTH experts slots run the same swigluoai experts on M3 — the reduce-owning
        # block (the overlap target) just also owns the trailing all-reduce, and
        # specialize_slot retargets its distributed reference (collective_partial) too.
        # Registering only the plain slot would verify an M3 reduce kernel against a
        # SiLU reference and false-fail every honest submission.
        "moe.fused_experts": _M3_MOE_NVFP4_PROFILE,
        "moe.fused_experts_reduce": _M3_MOE_NVFP4_PROFILE,
    },
    "GLM-5.3": {
        "moe.fused_experts": _GLM53_MOE_NVFP4_PROFILE,
        "moe.fused_experts_reduce": _GLM53_MOE_NVFP4_PROFILE,
        "moe.fused_routed_experts": _GLM53_ROUTED_MOE_PROFILE,
    },
}
# NVFP4 builds carry a "-NVFP4" suffix in their declared model id; alias them.
MODEL_PROFILES["MiniMax-M3-NVFP4"] = MODEL_PROFILES["MiniMax-M3"]
MODEL_PROFILES["GLM-5.3-NVFP4"] = MODEL_PROFILES["GLM-5.3"]


def model_profile(model_key: Optional[str], slot_name: str) -> Optional[SlotProfile]:
    if not model_key:
        return None
    return MODEL_PROFILES.get(model_key, {}).get(slot_name)


def slot_for_model(slot_name: str, model_key: Optional[str] = None) -> SlotSpec:
    """``get_slot`` + the validator's per-model specialization. With no model key (or no
    registered profile) this is exactly ``get_slot`` — the generic slot — so existing
    callers and bundles are unchanged."""
    slot = get_slot(slot_name)
    prof = model_profile(model_key, slot_name)
    return specialize_slot(slot, prof) if prof else slot

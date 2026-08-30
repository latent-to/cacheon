"""CPU tests for the generalized (op + block) slot abstraction.

Covers: (1) the ``kind`` discriminator, (2) multi-input *block* slots (the MoE
pair slots) verify faithful pure-torch kernels, (3) the ``matched_ratio``
correctness mode FAILS broken kernels, and (4) backward-compat — the
single-op slots (silu) still verify under the generalized spec. torch-only; skipped
where torch is unavailable (e.g. the dev laptop); runs on the pod.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cacheon.sandbox import load_entry  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402
from cacheon.verify import verify_entry  # noqa: E402

MOE_BUNDLE = "examples/miner_moe_fused_experts_torch/kernels/moe.py"
ROUTED_MOE_BUNDLE = "examples/miner_moe_fused_routed_torch/kernels/moe_routed.py"


def test_slot_kind_discriminator():
    assert get_slot("moe.fused_experts").kind == "block"
    assert get_slot("moe.fused_routed_experts").kind == "block"
    assert get_slot("linear.dense").kind == "block"
    assert get_slot("collective.all_gather_into_tensor").kind == "collective"
    assert get_slot("collective.reduce_scatter_tensor").kind == "collective"
    assert get_slot("norm.fused_add_rmsnorm").kind == "block"
    assert get_slot("activation.silu_and_mul").kind == "op"
    assert get_slot("norm.rmsnorm").kind == "op"


def test_moe_prepare_forward_passes_correctness_cpu():
    # The (prepare, forward) PAIR: load BOTH miner callables; verify runs prepare (the
    # weight layout) then forward, and compares to the fp32 MoE reference.
    fwd = load_entry(MOE_BUNDLE, "fused_experts")
    prep = load_entry(MOE_BUNDLE, "prepare")
    slot = get_slot("moe.fused_experts")
    result = verify_entry(slot, fwd, prepare=prep, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: ratio={r.pass_ratio} {r.detail}" for r in result.shape_results
    )


def test_dense_prepare_forward_passes_correctness_cpu():
    def prepare(weight):
        return weight

    def dense(x, weight, out):
        torch.mm(x, weight.t(), out=out)

    result = verify_entry(
        get_slot("linear.dense"),
        dense,
        prepare=prepare,
        dtype=torch.float32,
        device="cpu",
        seed=0,
    )
    assert result.passed


def test_moe_broken_prepare_fails_cpu():
    # A `prepare` that forgets the [gate;up]->[up;gate] reorder -> forward swaps the
    # halves -> silu(up)*gate != silu(gate)*up -> wrong. The slot is only correct when
    # BOTH callables agree, so a bad prepare must fail just like a bad forward would.
    def broken_prepare(w13, w2):
        return {"w13": w13.contiguous(), "w2": w2.contiguous(), "inter": w13.shape[1] // 2}

    fwd = load_entry(MOE_BUNDLE, "fused_experts")
    slot = get_slot("moe.fused_experts")
    result = verify_entry(slot, fwd, prepare=broken_prepare, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_routed_moe_passes_correctness_cpu():
    # The FAT slot: the miner receives router LOGITS and owns routing + experts +
    # combine. The example implements the engine's biased-sigmoid gate faithfully.
    fwd = load_entry(ROUTED_MOE_BUNDLE, "fused_routed_experts")
    prep = load_entry(ROUTED_MOE_BUNDLE, "prepare")
    slot = get_slot("moe.fused_routed_experts")
    result = verify_entry(slot, fwd, prepare=prep, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: ratio={r.pass_ratio} {r.detail}" for r in result.shape_results
    )


def test_routed_moe_wrong_routing_fails_cpu():
    # Selecting on the UNBIASED scores (dropping the correction bias) picks a
    # different expert set on margin-boosted inputs -> wrong combine -> FAIL.
    # This pins that the slot actually grades the routing head, not just the GEMMs.
    faithful = load_entry(ROUTED_MOE_BUNDLE, "fused_routed_experts")
    prep = load_entry(ROUTED_MOE_BUNDLE, "prepare")

    def unbiased_routing(x, router_logits, correction_bias, prepared, out):
        faithful(x, router_logits, torch.zeros_like(correction_bias), prepared, out)

    slot = get_slot("moe.fused_routed_experts")
    result = verify_entry(
        slot, unbiased_routing, prepare=prep, dtype=torch.float32, device="cpu", seed=0
    )
    assert not result.passed


def test_routed_moe_margin_enforced_inputs():
    # Every verification row must have an unambiguous selection: the gap between
    # the k-th and (k+1)-th biased choice score is real, so top-k is stable and a
    # single output gate cannot false-fail an honest kernel on a near-tie.
    slot = get_slot("moe.fused_routed_experts")
    for shape in slot.shapes:
        i = slot.make_inputs(dtype=torch.float32, device="cpu", seed=7, **shape)
        choice = torch.sigmoid(i["router_logits"]) + i["correction_bias"].unsqueeze(0)
        sorted_choice = choice.sort(dim=-1, descending=True).values
        k = shape["topk"]
        gap = sorted_choice[:, k - 1] - sorted_choice[:, k]
        assert torch.all(gap > 0.01), f"ambiguous selection at {shape}"


def test_moe_is_prepare_forward_slot():
    slot = get_slot("moe.fused_experts")
    assert slot.kind == "block"
    assert slot.prepare == "prepare"          # names the 2nd miner callable
    assert slot.invoke_prepare is not None
    # forward-only slots have no prepare:
    assert get_slot("activation.silu_and_mul").prepare is None
    assert get_slot("activation.silu_and_mul").invoke_prepare is None


def test_silu_op_still_verifies_under_generalized_spec():
    # Backward-compat: the single-op path is unchanged by the multi-output/block work.
    def silu(x, out):
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d].float()).to(x.dtype) * x[..., d:])

    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, silu, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed


def test_fused_add_rmsnorm_block_verifies_both_outputs():
    def fused(x, residual, weight, eps, out_norm, out_residual):
        out_residual.copy_(x + residual)
        fp32 = out_residual.float()
        variance = fp32.square().mean(-1, keepdim=True)
        out_norm.copy_(
            (fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
        )

    result = verify_entry(
        get_slot("norm.fused_add_rmsnorm"),
        fused,
        dtype=torch.float32,
        device="cpu",
        seed=0,
    )
    assert result.passed


def test_matched_ratio_is_active_only_for_the_block_slot():
    # The op slots keep all-close; the block slots use matched_ratio. This guards
    # against accidentally loosening the op gates when the abstraction was generalized.
    assert get_slot("activation.silu_and_mul").correctness.mode == "allclose"
    assert get_slot("norm.rmsnorm").correctness.mode == "allclose"
    assert get_slot("moe.fused_experts").correctness.mode == "matched_ratio"
    assert get_slot("moe.fused_routed_experts").correctness.mode == "matched_ratio"
    assert get_slot("linear.dense").correctness.mode == "matched_ratio"
    assert get_slot("norm.fused_add_rmsnorm").correctness.mode == "matched_ratio"

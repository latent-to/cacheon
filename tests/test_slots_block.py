"""CPU contract checks for op and block slots."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cacheon.sandbox import load_entry  # noqa: E402
from cacheon.registry import eligibility_from_metadata  # noqa: E402
from cacheon.slots import get_slot, slot_for_model  # noqa: E402
from cacheon.verify import verify_entry  # noqa: E402

MOE_BUNDLE = "examples/miner_moe_fused_experts_torch/kernels/moe.py"
ROUTED_MOE_BUNDLE = "examples/miner_moe_fused_routed_torch/kernels/moe_routed.py"
DENSE_BUNDLE = "examples/miner_dense_torch/kernels/dense.py"
FUSED_NORM_BUNDLE = (
    "examples/miner_fused_add_rmsnorm_torch/kernels/fused_add_rmsnorm.py"
)
ROUTED_SHAPE = {
    "num_tokens": 2, "num_experts": 4, "hidden": 64, "inter": 64,
    "topk": 2, "routed_scaling": 2.5,
}


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
    fwd = load_entry(MOE_BUNDLE, "fused_experts")
    prep = load_entry(MOE_BUNDLE, "prepare")
    slot = get_slot("moe.fused_experts")
    result = verify_entry(slot, fwd, prepare=prep, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: ratio={r.pass_ratio} {r.detail}" for r in result.shape_results
    )


def test_dense_prepare_forward_passes_correctness_cpu():
    result = verify_entry(
        slot_for_model("linear.dense", "GLM-5.3-NVFP4"),
        load_entry(DENSE_BUNDLE, "dense"),
        prepare=load_entry(DENSE_BUNDLE, "prepare"),
        dtype=torch.float32,
        device="cpu",
        shapes=[{
            "num_tokens": 3, "input_dim": 8, "output_dim": 12,
            "parallel_role": "row", "local_tp_size": 4,
        }],
        eligibility=eligibility_from_metadata(
            {"capabilities": {
                "input_dim": 8, "num_tokens": 3, "output_dim": 12,
                "parallel_role": "row", "tp_size": 4, "world_size": 4,
            }},
            ("float32",),
        ),
        architecture="cpu",
        tp_size=4,
        world_size=4,
    )
    assert result.passed and result.shape_results[0].applicable


def test_moe_broken_prepare_fails_cpu():
    def broken_prepare(w13, w2):
        return {"w13": w13.contiguous(), "w2": w2.contiguous(), "inter": w13.shape[1] // 2}

    fwd = load_entry(MOE_BUNDLE, "fused_experts")
    slot = get_slot("moe.fused_experts")
    result = verify_entry(slot, fwd, prepare=broken_prepare, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_routed_moe_wrong_routing_fails_cpu():
    faithful = load_entry(ROUTED_MOE_BUNDLE, "fused_routed_experts")
    prep = load_entry(ROUTED_MOE_BUNDLE, "prepare")

    def wrong_routing(x, router_logits, correction_bias, prepared, out):
        forced = 100 * torch.arange(
            correction_bias.numel(), device=correction_bias.device
        )
        faithful(x, torch.zeros_like(router_logits), forced, prepared, out)

    slot = slot_for_model("moe.fused_routed_experts", "GLM-5.3-NVFP4")
    result = verify_entry(
        slot, wrong_routing, prepare=prep, dtype=torch.float32, device="cpu",
        shapes=[ROUTED_SHAPE],
    )
    assert not result.passed
    assert result.num_applicable == 1


def test_routed_moe_margin_enforced_inputs():
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
    assert get_slot("activation.silu_and_mul").prepare is None
    assert get_slot("activation.silu_and_mul").invoke_prepare is None


def test_silu_op_still_verifies_under_generalized_spec():
    def silu(x, out):
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d].float()).to(x.dtype) * x[..., d:])

    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, silu, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed


def test_fused_add_rmsnorm_block_verifies_both_outputs():
    result = verify_entry(
        slot_for_model("norm.fused_add_rmsnorm", "GLM-5.3-NVFP4"),
        load_entry(FUSED_NORM_BUNDLE, "fused_add_rmsnorm"),
        dtype=torch.float32,
        device="cpu",
        shapes=[{"num_tokens": 3, "hidden": 16}],
        eligibility=eligibility_from_metadata(
            {"min_num_tokens": 3, "max_num_tokens": 3}, ("float32",)
        ),
        architecture="cpu",
    )
    assert result.passed and result.shape_results[0].applicable


def test_matched_ratio_is_active_only_for_the_block_slot():
    # The op slots keep all-close; the block slots use matched_ratio. This guards
    # against accidentally loosening the op gates when the abstraction was generalized.
    assert get_slot("activation.silu_and_mul").correctness.mode == "allclose"
    assert get_slot("norm.rmsnorm").correctness.mode == "allclose"
    assert get_slot("moe.fused_experts").correctness.mode == "matched_ratio"
    assert get_slot("moe.fused_routed_experts").correctness.mode == "matched_ratio"
    assert get_slot("linear.dense").correctness.mode == "matched_ratio"
    assert get_slot("norm.fused_add_rmsnorm").correctness.mode == "matched_ratio"

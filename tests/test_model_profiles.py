"""Per-model activation, correctness, and override verification tests."""

from __future__ import annotations

import textwrap

import pytest

torch = pytest.importorskip("torch")

from cacheon.model_profiles import model_profile  # noqa: E402
from cacheon.slots import (  # noqa: E402
    SlotProfile,
    get_slot,
    slot_for_model,
    specialize_slot,
)
from cacheon.verify import verify_entry_from_source  # noqa: E402

_SMALL_SHAPE = {"num_tokens": 8, "num_experts": 4, "hidden": 64, "inter": 64, "topk": 2}


def test_slot_for_model_generic_unchanged():
    assert slot_for_model("moe.fused_experts", None) is get_slot("moe.fused_experts")
    assert slot_for_model("moe.fused_experts", "UnknownModel") is get_slot("moe.fused_experts")


def test_slot_for_model_m3_swaps_activation_and_correctness():
    generic = get_slot("moe.fused_experts")
    m3 = slot_for_model("moe.fused_experts", "MiniMax-M3")
    assert generic.correctness.mode == "matched_ratio"
    assert m3.correctness.mode == "cosine" and m3.correctness.min_cosine == 0.985
    # alias resolves to the same profile
    assert slot_for_model("moe.fused_experts", "MiniMax-M3-NVFP4").correctness.mode == "cosine"

    inp = generic.make_inputs(dtype=torch.float32, device="cpu", seed=0,
                              num_tokens=8, num_experts=4, hidden=32, inter=16, topk=2)
    ref_silu = generic.invoke_reference(inp)[0]
    ref_swig = m3.invoke_reference(inp)[0]
    assert not torch.allclose(ref_silu, ref_swig, atol=1e-3)


def test_glm53_registered_profiles_cover_the_measured_call_regimes():
    members = ("collective.all_gather_into_tensor", "collective.all_reduce",
               "collective.reduce_scatter_tensor", "linear.dense",
               "moe.fused_routed_experts", "norm.fused_add_rmsnorm")
    profiles = {name: slot_for_model(name, "GLM-5.3-NVFP4").shapes for name in members}
    assert all(model_profile("GLM-5.3-NVFP4", name) for name in members)
    assert profiles["collective.all_reduce"] == ({"num_tokens": 16384, "hidden": 6144},)
    assert {(s["num_tokens"], s["hidden"]) for s in profiles[members[0]]} == {
        (6, 6144), (32, 6144),
    }
    assert {s["num_tokens"] for s in profiles["norm.fused_add_rmsnorm"]} == {
        6, 24, 32, 128, 4096, 16384,
    }
    dense = profiles["linear.dense"]
    assert {(s["input_dim"], s["output_dim"]) for s in dense} == {
        (512, 6144), (2048, 4096), (2048, 16384), (3072, 6144),
        (6144, 128), (6144, 160), (6144, 1024), (6144, 2624), (6144, 6144),
        (16384, 6144), (6144, 32), (6144, 256), (192, 512), (512, 256),
    }
    assert {s["num_tokens"] for s in dense} >= {6, 24, 32, 128, 4096, 16384}
    assert {s["num_tokens"] for s in profiles["moe.fused_routed_experts"]} == {
        1, 8, 24, 32, 128, 16384,
    }


def test_slot_for_model_glm53_correctness_is_calibrated_cosine():
    # Regression for the false-FAIL fallback: with no profile correctness the
    # GLM NVFP4 slots inherited the generic elementwise matched_ratio gate,
    # which NVFP4 cannot pass. The registered gate is the measured floor
    # (glm53_nvfp4_gate.py 2026-08-30) WITH the energy guard — cosine alone
    # is scale-invariant and passes a kernel that drops routed_scaling.
    for key in ("GLM-5.3", "GLM-5.3-NVFP4"):
        c = slot_for_model("moe.fused_routed_experts", key).correctness
        assert c.mode == "cosine"
        assert c.min_cosine == 0.985
        assert c.max_rel_norm_err == 0.05


def test_glm53_profiles_cover_six_families_without_standalone_small_targets():
    from cacheon.model_profiles import MODEL_PROFILES
    from cacheon.target_catalog import default_target_catalog

    targets = ("moe.fused_routed_experts", "linear.dense", "norm.fused_add_rmsnorm",
               "collective.all_reduce", "collective.dp_attention_exchange.v1", "attention.sparse_mla.v1")
    catalog, profiles = default_target_catalog(), MODEL_PROFILES["GLM-5.3"]
    assert profiles is MODEL_PROFILES["GLM-5.3-NVFP4"]
    assert set(profiles) == {member for target in targets for member in catalog.require(target).members}
    assert catalog.validate_active_targets(targets) == tuple(sorted(targets))
    plain = [s for s in profiles["norm.fused_add_rmsnorm"].shapes if not s.get("use_residual", True)]
    assert {s["hidden"] for s in plain} == {512, 2048, 6144}
    dense = profiles["linear.dense"].shapes
    assert {(s["batch_size"], s["input_dim"], s["output_dim"])
            for s in dense if "batch_size" in s} == {(64, 192, 512), (64, 512, 256)}
    assert {s["output_dim"] for s in dense if s.get("output_dtype") == "float32"} == {32, 256}
    assert all(s["input_dtype"] == "bfloat16" for s in profiles["attention.sparse_mla"].shapes)


@pytest.mark.parametrize("extra, layout", [
    ({}, "weight_out_in_row_major"),
    ({"output_dtype": "float32"}, "weight_out_in_fp32_output"),
    ({"batch_size": 2}, "batched_weight_out_in_strided"),
])
def test_dense_profile_descriptor_matches_the_live_family(extra, layout):
    from cacheon.dense_contract import call_descriptor
    from cacheon.model_profiles import verification_call_descriptor

    slot = get_slot("linear.dense")
    inputs = slot.make_inputs(num_tokens=3, input_dim=7, output_dim=5,
                             dtype=torch.bfloat16, device="cpu", seed=7, **extra)
    live = call_descriptor(inputs, architecture="sm103", graph_mode="cuda_graph",
                           parallel_role="replicated", tp_size=1, world_size=4)
    verify = verification_call_descriptor(slot, inputs, dtype_name="bfloat16",
        architecture="sm103", graph_mode="cuda_graph", tp_size=4, world_size=4)
    assert live == verify and verify["layout"] == layout


def test_m3_specialized_routing_draw_follows_profile():
    shape = dict(num_tokens=8, num_experts=8, hidden=64, inter=32, topk=3,
                 dtype=torch.float32, device="cpu", seed=3)
    # M3 (fused shared expert): last id column pinned to experts-1 with weight 1.0,
    # routed weights normalized then scaled 2x — the pre-refactor behavior.
    m3 = slot_for_model("moe.fused_experts", "MiniMax-M3").make_inputs(**shape)
    assert torch.all(m3["topk_ids"][:, -1] == 7)
    assert torch.all(m3["topk_weights"][:, -1] == 1.0)
    assert torch.allclose(m3["topk_weights"][:, :-1].sum(-1),
                          torch.full((8,), 2.0), atol=1e-5)
    assert m3["__moe_num_fused_shared_experts__"] == 1
    assert m3["__moe_activation__"] == "swigluoai"
def test_quant_profile_without_shapes_is_rejected():
    bad = SlotProfile(quant="nvfp4")
    with pytest.raises(ValueError, match="no shapes"):
        specialize_slot(get_slot("moe.fused_experts"), bad)


# A CPU override bundle source: just the torch reference (the device @cute.jit epilogue is
# GPU-only and absent here — the loader returns None for it, the dense path uses this).
_OVERRIDE_SRC = textwrap.dedent("""
    import torch

    def gemm1_epilogue_ref(gate, up, alpha=1.702, limit=7.0):
        g = gate.clamp(max=limit)
        u = up.clamp(min=-limit, max=limit)
        return g * torch.sigmoid(alpha * g) * (u + 1.0)
""")


def _write_src(tmp_path):
    p = tmp_path / "swigluoai.py"
    p.write_text(_OVERRIDE_SRC)
    return str(p)


def test_override_verify_passes_with_m3_profile(tmp_path):
    src = _write_src(tmp_path)
    res = verify_entry_from_source(
        "moe.fused_experts", src, "gemm1_epilogue",
        override_point="gemm1_epilogue", model_key="MiniMax-M3",
        dtype_name="float32", device="cpu", shapes=[_SMALL_SHAPE],
    )
    assert res.passed, res.shape_results
    # cosine metric on the M3 profile
    assert all(r.metric == "cosine" for r in res.shape_results)


def test_override_verify_fails_generic(tmp_path):
    # Same swigluoai override, but the GENERIC slot reference is SiLU -> mismatch -> FAIL.
    # This is the control: the profile (not the kernel) is what makes it pass.
    src = _write_src(tmp_path)
    res = verify_entry_from_source(
        "moe.fused_experts", src, "gemm1_epilogue",
        override_point="gemm1_epilogue", model_key=None,
        dtype_name="float32", device="cpu", shapes=[_SMALL_SHAPE],
    )
    assert not res.passed


def test_override_missing_torch_reference_errors(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def gemm1_epilogue(gate, up):\n    return gate\n")  # no _ref
    with pytest.raises(ValueError, match="must ship a torch reference"):
        verify_entry_from_source(
            "moe.fused_experts", str(p), "gemm1_epilogue",
            override_point="gemm1_epilogue", model_key="MiniMax-M3",
            dtype_name="float32", device="cpu", shapes=[_SMALL_SHAPE],
        )

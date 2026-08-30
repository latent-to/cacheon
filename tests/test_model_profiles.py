"""Per-model activation, correctness, and override verification tests."""

from __future__ import annotations

import textwrap

import pytest

torch = pytest.importorskip("torch")

from cacheon.slots import (  # noqa: E402
    Activation,
    SlotProfile,
    _moe_reference,
    get_slot,
    slot_for_model,
    specialize_slot,
)
from cacheon.verify import verify_entry_from_source  # noqa: E402

_SMALL_SHAPE = {"num_tokens": 8, "num_experts": 4, "hidden": 64, "inter": 64, "topk": 2}


def test_slot_for_model_generic_unchanged():
    # No model key -> identical object to the generic slot (existing bundles untouched).
    assert slot_for_model("moe.fused_experts", None) is get_slot("moe.fused_experts")
    assert slot_for_model("moe.fused_experts", "UnknownModel") is get_slot("moe.fused_experts")


def test_slot_for_model_m3_swaps_activation_and_correctness():
    generic = get_slot("moe.fused_experts")
    m3 = slot_for_model("moe.fused_experts", "MiniMax-M3")
    assert generic.correctness.mode == "matched_ratio"
    assert m3.correctness.mode == "cosine" and m3.correctness.min_cosine == 0.985
    assert generic.call_abi is m3.call_abi is None
    # alias resolves to the same profile
    assert slot_for_model("moe.fused_experts", "MiniMax-M3-NVFP4").correctness.mode == "cosine"

    # the rebound reference computes swigluoai, not SiLU
    inp = generic.make_inputs(dtype=torch.float32, device="cpu", seed=0,
                              num_tokens=8, num_experts=4, hidden=32, inter=16, topk=2)
    ref_silu = generic.invoke_reference(inp)[0]
    ref_swig = m3.invoke_reference(inp)[0]
    assert not torch.allclose(ref_silu, ref_swig, atol=1e-3)


def test_slot_for_model_glm53_profile_resolves():
    glm = slot_for_model("moe.fused_experts", "GLM-5.3")
    generic = get_slot("moe.fused_experts")
    # Shapes come from the profile record (receipted from the served config):
    # 256 routed experts, top-8, per-rank inter 512 at TP4, no fused shared expert.
    assert glm.shapes[0]["num_experts"] == 256
    assert glm.shapes[0]["inter"] == 512
    assert all(s["topk"] == 8 for s in glm.shapes)
    assert glm.shapes != generic.shapes
    assert slot_for_model("moe.fused_experts", "GLM-5.3-NVFP4").shapes == glm.shapes
    # GLM experts are plain SiLU: on the same inputs the GLM reference equals the
    # generic SiLU reference and differs from M3's swigluoai one.
    inp = generic.make_inputs(dtype=torch.float32, device="cpu", seed=0,
                              num_tokens=8, num_experts=4, hidden=32, inter=16, topk=2)
    ref_generic = generic.invoke_reference(inp)[0]
    ref_glm = glm.invoke_reference(inp)[0]
    ref_m3 = slot_for_model("moe.fused_experts", "MiniMax-M3").invoke_reference(inp)[0]
    assert torch.allclose(ref_generic, ref_glm, atol=1e-6)
    assert not torch.allclose(ref_glm, ref_m3, atol=1e-3)


def test_specialized_routing_draw_follows_profile():
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
    # GLM (no fused shared expert): a pure distinct routed draw over all experts,
    # weights normalized then scaled by routed_scaling_factor 2.5.
    glm = slot_for_model("moe.fused_experts", "GLM-5.3").make_inputs(**shape)
    assert glm["topk_ids"].shape == (8, 3)
    for row in glm["topk_ids"]:
        assert len(set(row.tolist())) == 3  # distinct experts per token
    assert torch.allclose(glm["topk_weights"].sum(-1),
                          torch.full((8,), 2.5), atol=1e-5)
    assert glm["__moe_num_fused_shared_experts__"] == 0
    assert glm["__moe_activation__"] == "silu"


def test_quant_profile_without_shapes_is_rejected():
    bad = SlotProfile(quant="nvfp4")
    with pytest.raises(ValueError, match="no shapes"):
        specialize_slot(get_slot("moe.fused_experts"), bad)


def test_moe_reference_swigluoai_differs_from_silu():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 32, generator=g) * 0.1
    w13 = torch.randn(4, 32, 32, generator=g) * 0.05  # 2I=32 -> I=16
    w2 = torch.randn(4, 32, 16, generator=g) * 0.05
    ids = torch.randint(0, 4, (8, 2), generator=g).to(torch.int32)
    sc = torch.rand(8, 2, generator=g)
    w = (sc / sc.sum(1, keepdim=True)).float()
    silu = _moe_reference(x, w13, w2, ids, w)  # default SiLU
    swig = _moe_reference(x, w13, w2, ids, w, Activation("swigluoai", 1.702, 7.0))
    assert not torch.allclose(silu, swig, atol=1e-3)


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

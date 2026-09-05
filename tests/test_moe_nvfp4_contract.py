from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import cacheon.dispatch as dispatch  # noqa: E402
from cacheon.dispatch import make_moe_dispatcher  # noqa: E402
from cacheon.moe_nvfp4_contract import (  # noqa: E402
    NVFP4_GATE_UP_LAYOUT,
    NVFP4_PREPARE_TAG,
    dequantize_prepare_args,
    prepare_args_from_inputs,
    prepare_args_from_layer,
)
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from cacheon.sandbox import load_entry  # noqa: E402
from cacheon.slots import Activation, _moe_reference, slot_for_model  # noqa: E402
from cacheon.verify import verify_entry  # noqa: E402


SHAPE = {"num_tokens": 4, "num_experts": 4, "hidden": 64, "inter": 64, "topk": 2}
WEIGHT_FIELDS = (
    "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
    "g1_scale_c", "g1_alphas", "g2_alphas", "w13_input_scale_quant",
    "w2_input_scale_quant", "intermediate_size_per_partition",
)


@pytest.mark.parametrize("kind", ("silu", "swigluoai"))
def test_expert_grouped_reference_preserves_duplicate_routes_and_raw_weights(kind):
    generator = torch.Generator().manual_seed(19)
    x, w13, w2 = (torch.randn(*s, generator=generator) * 3 for s in
                  ((3, 4), (4, 6, 4), (4, 4, 3)))
    ids = torch.tensor([[1, 1], [2, 0], [1, 2]])  # repeated route and unused expert3
    weights = torch.tensor([[0.4, 0.7], [-0.2, 0.0], [1.5, 0.3]])
    expected = torch.zeros_like(x)
    for token in range(3):
        for choice in range(2):
            expert = int(ids[token, choice])
            gate, up = (w13[expert] @ x[token]).chunk(2)
            if kind == "silu":
                value = gate * torch.sigmoid(gate) * up
            else:
                gate, up = gate.clamp(max=7), up.clamp(-7, 7)
                value = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            expected[token] += weights[token, choice] * (w2[expert] @ value)
    actual = _moe_reference(x, w13, w2, ids, weights, Activation(kind, 1.702, 7.0))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def _candidate(corrupt: bool = False):
    seen = []

    def prepare(a, b):
        args = (a, b)
        seen.append(args)
        w13, w2 = dequantize_prepare_args(args)
        return w13, w2

    def entry(x, ids, weights, prepared, out):
        w13, w2 = prepared
        expected = _moe_reference(
            x, w13, w2, ids, weights, Activation("swigluoai", 1.702, 7.0)
        )
        out.copy_(torch.zeros_like(expected) if corrupt else expected)

    return prepare, entry, seen


def _live_layer(inputs, *, complete=True):
    layer = SimpleNamespace(
        **{name: inputs[name] for name in WEIGHT_FIELDS},
        w13_blockscale_swizzled=inputs["w13_weight_scale"],
        w2_blockscale_swizzled=inputs["w2_weight_scale"],
        moe_ep_size=1, moe_tp_size=4, reduce_results=False,
        num_fused_shared_experts=1,
        top_k=int(inputs["topk_ids"].shape[-1]),
    )
    if not complete:
        del layer.w13_blockscale_swizzled
    return layer


@pytest.mark.parametrize(("corrupt", "passed"), ((False, True), (True, False)))
def test_m3_nvfp4_verification_executes_the_quantized_contract(corrupt, passed):
    slot = slot_for_model("moe.fused_experts", "MiniMax-M3-NVFP4")
    prepare, entry, seen = _candidate(corrupt)
    result = verify_entry(
        slot,
        entry,
        prepare=prepare,
        dtype=torch.float32,
        device="cpu",
        shapes=[SHAPE],
        eligibility=Eligibility(
            dtypes=frozenset({"float32"}), quant=frozenset({"nvfp4"})
        ),
        architecture="cpu",
    )

    assert result.passed is passed and result.shape_results[0].applicable
    assert seen[0][0] == NVFP4_PREPARE_TAG
    view = seen[0][1]
    assert view.w13_weight.dtype == view.w2_weight.dtype == torch.uint8
    assert (view.cacheon_group_size, view.cacheon_w13_layout) == (
        16, NVFP4_GATE_UP_LAYOUT,
    )
    assert result.shape_results[0].case_descriptor.calls[0]["quant"] == "nvfp4"


@pytest.mark.parametrize("target", ("moe.fused_experts", "moe.fused_experts_reduce"))
@pytest.mark.parametrize("topk", (2, 3))
def test_live_layer_and_verifier_emit_the_same_nvfp4_prepare_schema(target, topk):
    slot = slot_for_model(target, "MiniMax-M3-NVFP4")
    inputs = slot.make_inputs(
        **(SHAPE | {"topk": topk}), dtype=torch.float32, device="cpu", seed=3
    )
    layer = _live_layer(inputs)
    layer.w13_weight_scale = torch.zeros_like(inputs["w13_weight_scale"])
    layer.w2_weight_scale = torch.zeros_like(inputs["w2_weight_scale"])
    verify_args = prepare_args_from_inputs(inputs)
    live_args = prepare_args_from_layer(layer)
    assert verify_args[0] == live_args[0] == NVFP4_PREPARE_TAG
    verify_view, live_view = verify_args[1], live_args[1]
    assert verify_view.moe_runner_config.top_k == live_view.moe_runner_config.top_k == topk
    assert (
        verify_view.moe_tp_size,
        verify_view.moe_ep_size,
        verify_view.num_fused_shared_experts,
    ) == (
        live_view.moe_tp_size,
        live_view.moe_ep_size,
        live_view.num_fused_shared_experts,
    ) == (4, 1, 1)
    assert all(
        getattr(verify_view, name).dtype == getattr(live_view, name).dtype
        and torch.equal(
            getattr(verify_view, name).float(), getattr(live_view, name).float()
        )
        for name in WEIGHT_FIELDS[:-1]
    )
    live_w13, live_w2 = dequantize_prepare_args(live_args)
    assert torch.equal(live_w13, inputs["w13"]) and torch.equal(live_w2, inputs["w2"])


def test_glm_routed_example_executes_the_nvfp4_prepare_contract():
    source = "examples/miner_moe_fused_routed_torch/kernels/moe_routed.py"
    slot = slot_for_model("moe.fused_routed_experts", "GLM-5.3-NVFP4")
    inputs = slot.make_inputs(
        num_tokens=2, num_experts=4, hidden=64, inter=64, topk=2,
        routed_scaling=2.5, dtype=torch.float32, device="cpu", seed=11,
    )
    assert prepare_args_from_inputs(inputs)[1].moe_runner_config.top_k == 2
    result = verify_entry(
        slot,
        load_entry(source, "fused_routed_experts"),
        prepare=load_entry(source, "prepare"),
        dtype=torch.float32,
        device="cpu",
        shapes=[{
            "num_tokens": 2, "num_experts": 4, "hidden": 64, "inter": 64,
            "topk": 2, "routed_scaling": 2.5,
        }],
        eligibility=Eligibility(
            dtypes=frozenset({"float32"}), quant=frozenset({"nvfp4"}),
            min_num_tokens=2, max_num_tokens=2,
        ),
        architecture="cpu",
        tp_size=4,
        world_size=4,
    )

    case = result.shape_results[0].case_descriptor
    assert result.passed and result.shape_results[0].applicable and case is not None
    assert dict(case.calls[0])["quant"] == "nvfp4"


@pytest.mark.parametrize(
    ("trtllm", "mma", "expected"),
    ((False, False, "gate_up"),
     (False, True, "up_gate_interleaved_64+sf_swizzled_128x4"),
     (True, False, "trtllm_fp4_shuffled")),
)
@pytest.mark.parametrize("activation", ("swigluoai", "silu"))
def test_live_nvfp4_prepare_identifies_the_backend_layout(trtllm, mma, expected, activation):
    slot = slot_for_model("moe.fused_experts", "MiniMax-M3-NVFP4")
    inputs = slot.make_inputs(**SHAPE, dtype=torch.float32, device="cpu", seed=3)
    layer = _live_layer(inputs)
    layer.quant_method = SimpleNamespace(enable_flashinfer_trtllm_moe=trtllm)
    layer.moe_runner_config = SimpleNamespace(activation=activation)
    if mma:
        layer.w13_blockscale_mma = inputs["w13_weight_scale"]

    _, view = prepare_args_from_layer(layer)
    assert view.cacheon_w13_layout == expected
    assert view.moe_runner_config.activation == activation
    assert view.w13_weight.data_ptr() == layer.w13_weight.data_ptr()
    assert view.w2_weight.data_ptr() == layer.w2_weight.data_ptr()


def test_m3_reduce_profile_uses_live_shape_topology_and_nvfp4_prepare():
    slot = slot_for_model("moe.fused_experts_reduce", "MiniMax-M3-NVFP4")
    assert slot.shapes[0] == {
        "num_tokens": 1, "num_experts": 129, "hidden": 6144,
        "inter": 768, "topk": 5,
    }
    inputs = slot.make_inputs(
        **SHAPE, dtype=torch.float32, device="cpu", seed=7, rank=0, world_size=4
    )
    assert torch.equal(inputs["topk_ids"][:, -1], torch.full((4,), 3, dtype=torch.int32))
    assert torch.all(inputs["topk_ids"][:, :-1] != 3)
    assert torch.allclose(inputs["topk_weights"][:, :-1].sum(-1), torch.full((4,), 2.0))
    assert torch.equal(inputs["topk_weights"][:, -1], torch.ones(4))
    tag, view = prepare_args_from_inputs(inputs)
    assert tag == NVFP4_PREPARE_TAG
    assert (view.moe_tp_size, view.moe_ep_size, view.num_fused_shared_experts) == (
        4, 1, 1,
    )


@pytest.mark.parametrize(
    ("quant", "complete", "routed"),
    ((frozenset(), True, False),
     (frozenset({"nvfp4"}), False, False),
     (frozenset({"nvfp4"}), True, True)),
)
def test_live_dispatch_selects_only_matching_finalized_nvfp4(
    monkeypatch, quant, complete, routed
):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    slot = slot_for_model("moe.fused_experts", "MiniMax-M3-NVFP4")
    inputs = slot.make_inputs(**SHAPE, dtype=torch.float32, device="cpu", seed=5)
    prepare, entry, prepared = _candidate()
    registry = KernelRegistry()
    registry.register(KernelImpl(
        slot="moe.fused_experts", bundle_id="candidate", entry=entry,
        prepare=prepare, eligibility=Eligibility(
            dtypes=frozenset({"float32"}), quant=quant,
        ),
    ))
    registry.enable()
    completed, stock = [], object()
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    wrapped = make_moe_dispatcher(lambda *_: stock, registry=registry,
                                  slots=("moe.fused_experts",))
    topk = SimpleNamespace(topk_ids=inputs["topk_ids"],
                           topk_weights=inputs["topk_weights"])
    output = wrapped(_live_layer(inputs, complete=complete), inputs["x"], topk)
    assert (output is not stock) is routed
    assert bool(prepared) is routed and bool(completed) is routed


def test_explicit_dense_quant_remains_applicable():
    source = "examples/miner_moe_fused_experts_torch/kernels/moe.py"
    result = verify_entry(
        slot_for_model("moe.fused_experts", None),
        load_entry(source, "fused_experts"),
        prepare=load_entry(source, "prepare"),
        dtype=torch.float32, device="cpu", shapes=[SHAPE], architecture="cpu",
        eligibility=Eligibility(
            dtypes=frozenset({"float32"}), quant=frozenset({"dense"})
        ),
    )
    assert result.passed and result.shape_results[0].applicable
    assert result.shape_results[0].case_descriptor.calls[0]["quant"] == "dense"

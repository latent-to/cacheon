from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import cacheon.dispatch as dispatch  # noqa: E402
from cacheon.dispatch import (  # noqa: E402
    make_moe_deferred_dispatcher,
    make_moe_deferred_finalize_dispatcher,
    make_moe_dispatcher,
)
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from cacheon.sandbox import load_entry  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402

MOE_BUNDLE = "examples/miner_moe_fused_experts_torch/kernels/moe.py"
_BASELINE = object()  # sentinel: the dispatcher returns this iff it fell back


@pytest.fixture(autouse=True)
def _single_moe_dp_rank(monkeypatch):
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)


def _baseline_forward(self, hidden_states, topk_output):
    return _BASELINE


class _Param:  # mimics torch.nn.Parameter's .data access the seam uses
    def __init__(self, t):
        self.data = t


def _fake_layer(inputs, *, moe_tp_size=1, moe_ep_size=1, reduce_results=False, **extra):
    layer = SimpleNamespace(
        w13_weight=_Param(inputs["w13"]),
        w2_weight=_Param(inputs["w2"]),
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        reduce_results=reduce_results,
    )
    for k, v in extra.items():  # e.g. w13_weight_scale to simulate a quantized layer
        setattr(layer, k, v)
    return layer


def _standard_topk_output(inputs):
    # Duck-typed StandardTopKOutput: carries explicit topk tensors.
    return SimpleNamespace(
        topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"], router_logits=None
    )


def _bypassed_topk_output():
    # BypassedTopKOutput has no topk_ids/topk_weights (routing not materialized).
    return SimpleNamespace(hidden_states=None, router_logits=None, topk_config=None)


def _inputs(seed=0):
    slot = get_slot("moe.fused_experts")
    shape = {"num_tokens": 16, "num_experts": 8, "hidden": 256, "inter": 128, "topk": 2}
    return slot.make_inputs(**shape, dtype=torch.float32, device="cpu", seed=seed)


def _registry(entry, prepare, *, quant=frozenset()):
    reg = KernelRegistry()
    reg.register(
        KernelImpl(
            slot="moe.fused_experts",
            bundle_id="test",
            entry=entry,
            prepare=prepare,
            eligibility=Eligibility(dtypes=frozenset({"float32"}), quant=quant),
        )
    )
    reg.enable()
    return reg


def _matched_ratio(actual, expected, atol=1e-4, rtol=1e-4):
    within = (actual.float() - expected.float()).abs() <= (atol + rtol * expected.float().abs())
    return within.float().mean().item()


def test_seam_routes_to_kernel_and_matches_reference(monkeypatch):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    slot = get_slot("moe.fused_experts")
    reference = slot.invoke_reference(inputs)[0]

    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))

    out = dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs))
    assert out is not _BASELINE, "seam should have routed to the miner kernel"
    assert tuple(out.shape) == (inputs["x"].shape[0], inputs["x"].shape[1])
    assert _matched_ratio(out, reference) >= slot.correctness.min_ratio


def test_disabled_when_env_flag_unset(monkeypatch):
    # Opt-in: with CACHEON_MOE_SEAM unset the seam is inert (trusts the baseline).
    monkeypatch.delenv("CACHEON_MOE_SEAM", raising=False)
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    assert dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_prepare_runs_once_and_is_memoized(monkeypatch):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    base_prepare = load_entry(MOE_BUNDLE, "prepare")
    calls = {"n": 0}

    def counting_prepare(w13, w2):
        calls["n"] += 1
        return base_prepare(w13, w2)

    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, counting_prepare))
    layer = _fake_layer(inputs)
    topk = _standard_topk_output(inputs)
    dispatched(layer, inputs["x"], topk)
    dispatched(layer, inputs["x"], topk)
    assert calls["n"] == 1, "prepare must run ONCE per layer (memoized), not per step"


def test_expert_parallel_falls_back(monkeypatch):
    # EP>1 adds an all-to-all the (M,H)->(M,H) contract doesn't model -> trust baseline.
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    layer = _fake_layer(inputs, moe_ep_size=2)
    assert dispatched(layer, inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_bypassed_routing_falls_back(monkeypatch):
    # Routing not materialized (no topk tensors) -> conservative fallback (no re-routing).
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    assert dispatched(_fake_layer(inputs), inputs["x"], _bypassed_topk_output()) is _BASELINE


ROUTED_MOE_BUNDLE = "examples/miner_moe_fused_routed_torch/kernels/moe_routed.py"


def _routed_inputs(seed=0):
    slot = get_slot("moe.fused_routed_experts")
    shape = {"num_tokens": 16, "num_experts": 8, "hidden": 256, "inter": 128,
             "topk": 2, "routed_scaling": 2.5}
    return slot.make_inputs(**shape, dtype=torch.float32, device="cpu", seed=seed)


def _routed_topk_output(inputs, **overrides):
    # Duck-typed BypassedTopKOutput + TopKConfig at the pin: routing NOT
    # materialized; the config carries the routing head's parameters.
    cfg = dict(
        top_k=inputs["topk"],
        correction_bias=inputs["correction_bias"],
        routed_scaling_factor=inputs["routed_scaling"],
        scoring_func="sigmoid",
        renormalize=True,
        apply_routed_scaling_factor_on_output=True,
        custom_routing_function=None,
        num_fused_shared_experts=0,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
    )
    cfg.update(overrides)
    return SimpleNamespace(
        hidden_states=inputs["x"],
        router_logits=inputs["router_logits"],
        topk_config=SimpleNamespace(**cfg),
    )


def _routed_registry(entry, prepare):
    reg = KernelRegistry()
    reg.register(
        KernelImpl(
            slot="moe.fused_routed_experts",
            bundle_id="test-routed",
            entry=entry,
            prepare=prepare,
            eligibility=Eligibility(dtypes=frozenset({"float32"})),
        )
    )
    reg.enable()
    return reg


def test_routed_moe_dispatches_on_bypassed_routing(monkeypatch):
    # The fat slot binds where the thin one cannot: router LOGITS at the seam.
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _routed_inputs()
    slot = get_slot("moe.fused_routed_experts")
    reference = slot.invoke_reference(inputs)[0]

    entry = load_entry(ROUTED_MOE_BUNDLE, "fused_routed_experts")
    prepare = load_entry(ROUTED_MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(
        _baseline_forward, registry=_routed_registry(entry, prepare)
    )
    out = dispatched(_fake_layer(inputs), inputs["x"], _routed_topk_output(inputs))
    assert out is not _BASELINE, "fat slot should have routed to the miner kernel"
    assert tuple(out.shape) == (inputs["x"].shape[0], inputs["x"].shape[1])
    assert _matched_ratio(out, reference) >= slot.correctness.min_ratio


def test_routed_moe_out_of_contract_config_stays_stock(monkeypatch):
    # A routing config outside the slot's fixed head (softmax scoring, grouped
    # selection, missing output-side scaling, fused shared experts) must serve stock —
    # the dispatcher never approximates routing semantics.
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _routed_inputs()
    entry = load_entry(ROUTED_MOE_BUNDLE, "fused_routed_experts")
    prepare = load_entry(ROUTED_MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(
        _baseline_forward, registry=_routed_registry(entry, prepare)
    )
    layer = _fake_layer(inputs)
    for bad in (
        {"scoring_func": "softmax"},
        {"apply_routed_scaling_factor_on_output": False},
        {"num_fused_shared_experts": 1},
        {"use_grouped_topk": True, "num_expert_group": 4, "topk_group": 2},
        {"renormalize": False},
        {"correction_bias": None},
    ):
        out = dispatched(layer, inputs["x"], _routed_topk_output(inputs, **bad))
        assert out is _BASELINE, f"config {bad} must stay stock"


def test_routed_moe_deferred_path_joins_stock_shared_output(monkeypatch):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _routed_inputs()
    entry = load_entry(ROUTED_MOE_BUNDLE, "fused_routed_experts")
    prepare = load_entry(ROUTED_MOE_BUNDLE, "prepare")
    deferred = make_moe_deferred_dispatcher(
        _baseline_forward, registry=_routed_registry(entry, prepare)
    )
    finalize = make_moe_deferred_finalize_dispatcher(
        lambda value, shared: (value, shared)
    )

    routed = get_slot("moe.fused_routed_experts").invoke_reference(inputs)[0]
    shared = torch.randn_like(routed)
    pending = deferred(
        _fake_layer(inputs), inputs["x"], _routed_topk_output(inputs)
    )
    out = finalize(pending, shared)
    assert _matched_ratio(out, routed + shared) == 1.0


def test_routed_moe_prepare_receives_routing_config(monkeypatch):
    # prepare gets (w13, w2, top_k, routed_scaling) — the static routing config
    # from the live TopKConfig — and is memoized per layer like the thin slot.
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _routed_inputs()
    base_prepare = load_entry(ROUTED_MOE_BUNDLE, "prepare")
    seen = []

    def spying_prepare(w13, w2, topk, routed_scaling):
        seen.append((topk, routed_scaling))
        return base_prepare(w13, w2, topk, routed_scaling)

    entry = load_entry(ROUTED_MOE_BUNDLE, "fused_routed_experts")
    dispatched = make_moe_dispatcher(
        _baseline_forward, registry=_routed_registry(entry, spying_prepare)
    )
    layer = _fake_layer(inputs)
    topk_output = _routed_topk_output(inputs)
    dispatched(layer, inputs["x"], topk_output)
    dispatched(layer, inputs["x"], topk_output)
    assert seen == [(inputs["topk"], inputs["routed_scaling"])]


def test_missing_prepare_aborts_selected_candidate(monkeypatch):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare=None))
    with pytest.raises(RuntimeError, match="selected MoE candidate.*has no prepare"):
        dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs))


def test_dense_layer_skips_quant_only_kernel(monkeypatch):
    # The other direction: a DENSE layer must NOT run a quant-only kernel (it expects
    # scales that aren't there) -> fall back.
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    reg = _registry(entry, prepare, quant=frozenset({"nvfp4"}))
    dispatched = make_moe_dispatcher(_baseline_forward, registry=reg)
    assert dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_non_2d_hidden_states_falls_back(monkeypatch):
    # The (M,H)->(M,H) contract assumes flattened 2D tokens; anything else -> baseline.
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    x3d = inputs["x"].unsqueeze(0)  # (1, M, H)
    assert dispatched(_fake_layer(inputs), x3d, _standard_topk_output(inputs)) is _BASELINE


def test_raising_selected_kernel_aborts(monkeypatch):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    inputs = _inputs()
    prepare = load_entry(MOE_BUNDLE, "prepare")

    def raising(x, topk_ids, topk_weights, prepared, out):
        raise RuntimeError("boom")

    reg = _registry(raising, prepare)
    dispatched = make_moe_dispatcher(_baseline_forward, registry=reg)
    with pytest.raises(RuntimeError, match="boom"):
        dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs))


def test_install_patches_forward_impl(monkeypatch):
    import sys
    from types import ModuleType

    from cacheon.integrations import sglang_moe
    def forward(self, hidden_states, topk_output):          # the router — must stay untouched
        return ("forward", hidden_states)

    def forward_impl(self, hidden_states, topk_output):     # the waist — must be patched
        return ("impl", hidden_states)

    def forward_deferred_finalize(self, hidden_states, topk_output):
        return ("deferred", hidden_states)

    def finalize_deferred(value, shared):
        return (value, shared)

    class FakeFusedMoE:
        pass

    FakeFusedMoE.forward = forward
    FakeFusedMoE.forward_impl = forward_impl
    FakeFusedMoE.forward_deferred_finalize = forward_deferred_finalize
    mod = ModuleType(sglang_moe._MODULE)
    mod.FusedMoE = FakeFusedMoE
    finalizer_mod = ModuleType(sglang_moe._FINALIZER_MODULE)
    setattr(finalizer_mod, sglang_moe._FINALIZER_FUNC, finalize_deferred)
    monkeypatch.setitem(sys.modules, sglang_moe._MODULE, mod)
    monkeypatch.setitem(sys.modules, sglang_moe._FINALIZER_MODULE, finalizer_mod)

    orig_forward = FakeFusedMoE.forward
    orig_impl = FakeFusedMoE.forward_impl
    orig_deferred = FakeFusedMoE.forward_deferred_finalize
    try:
        sglang_moe.install()
        assert sglang_moe.is_installed()
        assert FakeFusedMoE.forward_impl is not orig_impl          # patched
        assert FakeFusedMoE.forward_deferred_finalize is not orig_deferred
        assert getattr(finalizer_mod, sglang_moe._FINALIZER_FUNC) is not finalize_deferred
        assert FakeFusedMoE.forward is orig_forward                # router untouched
        assert FakeFusedMoE._cacheon_orig_forward_impl is orig_impl  # captured for fallback/uninstall
        sglang_moe.install()  # idempotent
        assert FakeFusedMoE._cacheon_orig_forward_impl is orig_impl
    finally:
        sglang_moe.uninstall()
    assert FakeFusedMoE.forward_impl is orig_impl
    assert FakeFusedMoE.forward_deferred_finalize is orig_deferred
    assert getattr(finalizer_mod, sglang_moe._FINALIZER_FUNC) is finalize_deferred
    assert not sglang_moe.is_installed()


def test_install_noop_without_forward_impl(monkeypatch):
    import sys
    from types import ModuleType

    from cacheon.integrations import sglang_moe

    class OldFusedMoE:
        def forward(self, hidden_states, topk_output):
            return ("forward", hidden_states)

    mod = ModuleType(sglang_moe._MODULE)
    mod.FusedMoE = OldFusedMoE
    monkeypatch.setitem(sys.modules, sglang_moe._MODULE, mod)

    sglang_moe.install()
    assert not sglang_moe.is_installed()
    assert not hasattr(OldFusedMoE, "_cacheon_orig_forward_impl")


def test_minimax_reduce_owns_immediate_and_deferred_ar_but_plain_cannot_forge(
    monkeypatch,
):
    import sys
    from types import ModuleType

    from cacheon.integrations import sglang_moe

    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_tp_device_group", lambda: group)
    monkeypatch.setattr(dispatch, "_in_cuda_graph", lambda: False)
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: False)
    completed = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)

    x = torch.randn(3, 8)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    weights = torch.full((3, 2), 0.5)
    topk = SimpleNamespace(topk_ids=ids, topk_weights=weights)

    class FakeFusedMoE:
        def forward_impl(self, hidden_states, _topk):
            return hidden_states

    experts = FakeFusedMoE()
    experts.w13_weight = _Param(torch.randn(4, 8, 8))
    experts.w2_weight = _Param(torch.randn(4, 8, 4))
    experts.moe_tp_size = 2
    experts.moe_ep_size = 1
    experts.reduce_results = False
    experts.num_fused_shared_experts = 1
    experts.num_local_experts = 4
    experts.intermediate_size_per_partition = 4

    model_mod = ModuleType(sglang_moe._MODEL_MODULE)
    stock_reduces = []

    def stock_reduce(output):
        stock_reduces.append(True)
        return output + 10

    model_mod.tensor_model_parallel_all_reduce = stock_reduce

    class FakeMiniMaxM3MoE:
        def forward_normal(
            self, hidden_states, should_allreduce_fusion=False, use_reduce_scatter=False
        ):
            out = self.experts.forward_impl(hidden_states, self.topk)
            if self.tp_size > 1 and not should_allreduce_fusion and not use_reduce_scatter:
                out = model_mod.tensor_model_parallel_all_reduce(out)
            return out

    class FakeMiniMaxM3DecoderLayer:
        def forward(self, hidden_states, deferred=False):
            out = self.mlp.forward_normal(hidden_states, deferred, False)
            if deferred:
                out._sglang_needs_allreduce_fusion = True
            return out, None

    model_mod.MiniMaxM3MoE = FakeMiniMaxM3MoE
    model_mod.MiniMaxM3DecoderLayer = FakeMiniMaxM3DecoderLayer
    layer_mod = ModuleType(sglang_moe._MODULE)
    layer_mod.FusedMoE = FakeFusedMoE
    monkeypatch.setitem(sys.modules, sglang_moe._MODULE, layer_mod)
    monkeypatch.setitem(sys.modules, sglang_moe._MODEL_MODULE, model_mod)

    registry = KernelRegistry()
    registry.register(KernelImpl(
        slot="moe.fused_experts_reduce", bundle_id="reduce", variant="default",
        entry=lambda x, _i, _w, _p, out, _g: out.copy_(x),
        prepare=lambda _w13, _w2: None,
        eligibility=Eligibility(dtypes=frozenset({"float32"})),
    ))
    registry.register(KernelImpl(
        slot="moe.fused_experts", bundle_id="plain", variant="default",
        entry=lambda x, _i, _w, _p, out: (
            out.copy_(x),
            setattr(out, dispatch._MOE_REDUCED_ATTR, dispatch._MOE_REDUCED_TOKEN),
        ),
        prepare=lambda _w13, _w2: None,
        eligibility=Eligibility(dtypes=frozenset({"float32"})),
    ))
    registry.enable()

    model = FakeMiniMaxM3MoE()
    model.experts, model.topk, model.tp_size = experts, topk, 2
    model.n_shared_experts, model.shared_experts = 1, None
    decoder = FakeMiniMaxM3DecoderLayer()
    decoder.mlp = model
    try:
        sglang_moe.install(registry)
        assert torch.equal(model.forward_normal(x), x)
        assert torch.equal(decoder.forward(x, True)[0], x)
        assert stock_reduces == [] and completed == [
            "moe.fused_experts_reduce", "moe.fused_experts_reduce"
        ]

        model.shared_experts = object()
        assert torch.equal(model.forward_normal(x), x + 10)
        assert stock_reduces == [True]
    finally:
        sglang_moe.uninstall()

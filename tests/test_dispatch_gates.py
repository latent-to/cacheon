"""Dispatcher conservatism gates: paths whose semantics the slot contracts don't
model must fall back to the trusted baseline instead of silently computing the
wrong thing (and framing an honest kernel for the resulting KL failure).

Pins two of them:
* rmsnorm: ``variance_size_override`` / ``cast_x_before_out_mul`` / ``fp32_residual`` -> baseline;
* moe.fused_experts_reduce: a TP layer with ``reduce_results=False`` defers its reduce
  downstream, so the reduce-OWNING kernel must not run (it would insert an extra reduce).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cacheon.dispatch import (  # noqa: E402
    make_moe_dispatcher,
    make_rmsnorm_dispatcher,
)
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402

_BASELINE = object()  # sentinel: the dispatcher fell back


class _Param:
    def __init__(self, t):
        self.data = t


# ---- rmsnorm: semantic overrides -------------------------------------------------------


def _rms_entry(x, weight, out, eps):
    v = (x.float() * x.float()).mean(-1, keepdim=True)
    out.copy_((x.float() * torch.rsqrt(v + eps)).to(x.dtype) * weight)


def _rms_registry():
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="norm.rmsnorm", bundle_id="t", entry=_rms_entry,
                            eligibility=Eligibility(dtypes=frozenset({"float32"}))))
    reg.enable()
    return reg


def _fused_rms_registry():
    def entry(x, residual, weight, eps, out_norm, out_residual):
        out_residual.copy_(x + residual)
        fp32 = out_residual.float()
        variance = fp32.square().mean(-1, keepdim=True)
        out_norm.copy_(
            (fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
        )

    reg = KernelRegistry()
    reg.register(
        KernelImpl(
            slot="norm.fused_add_rmsnorm",
            bundle_id="fused",
            entry=entry,
            eligibility=Eligibility(dtypes=frozenset({"float32"})),
        )
    )
    reg.enable()
    return reg


def _rms_self(**extra):
    s = SimpleNamespace(variance_epsilon=1e-6, weight=_Param(torch.ones(16)))
    for k, v in extra.items():
        setattr(s, k, v)
    return s


def test_rmsnorm_plain_layer_routes_to_kernel():
    dispatched = make_rmsnorm_dispatcher(lambda *a: _BASELINE, registry=_rms_registry())
    out = dispatched(_rms_self(), torch.randn(4, 16))
    assert out is not _BASELINE and torch.is_tensor(out)


def test_rmsnorm_residual_layer_routes_to_fused_kernel():
    dispatched = make_rmsnorm_dispatcher(
        lambda *_args: _BASELINE, registry=_fused_rms_registry()
    )
    x, residual = torch.randn(4, 16), torch.randn(4, 16)
    out, new_residual = dispatched(_rms_self(), x, residual)
    assert torch.equal(new_residual, x + residual)
    expected = get_slot("norm.fused_add_rmsnorm").invoke_reference(
        {"x": x, "residual": residual, "weight": torch.ones(16), "eps": 1e-6}
    )
    assert torch.allclose(out, expected[0])


def test_rmsnorm_forwards_v0518_quant_linear_to_stock():
    marker = object()
    calls = []

    def baseline(self, x, residual=None, post_residual_addition=None, *, quant_linear=None):
        calls.append((residual, post_residual_addition, quant_linear))
        return _BASELINE

    dispatched = make_rmsnorm_dispatcher(baseline, registry=_rms_registry())

    assert dispatched(_rms_self(), torch.randn(4, 16), quant_linear=marker) is _BASELINE
    assert calls == [(None, None, marker)]


def test_rmsnorm_omits_new_optional_argument_for_an_older_stock_signature():
    def baseline(self, x, residual=None, post_residual_addition=None):
        return _BASELINE

    dispatched = make_rmsnorm_dispatcher(baseline, registry=KernelRegistry())

    assert dispatched(_rms_self(), torch.randn(4, 16)) is _BASELINE


@pytest.mark.parametrize("attrs", [
    {"variance_size_override": 8},     # variance over a prefix of the hidden dim
    {"cast_x_before_out_mul": True},   # HF cast-before-multiply semantics
    {"fp32_residual": True},
])
def test_rmsnorm_semantic_overrides_fall_back(attrs):
    dispatched = make_rmsnorm_dispatcher(lambda *a: _BASELINE, registry=_rms_registry())
    assert dispatched(_rms_self(**attrs), torch.randn(4, 16)) is _BASELINE


# ---- moe.fused_experts_reduce: layers that defer their reduce --------------------------


def _moe_inputs():
    slot = get_slot("moe.fused_experts_reduce")
    shape = {"num_tokens": 8, "num_experts": 4, "hidden": 64, "inter": 32, "topk": 2}
    return slot.make_inputs(**shape, dtype=torch.float32, device="cpu", seed=0)


def _moe_layer(inputs, *, moe_tp_size, reduce_results):
    return SimpleNamespace(
        w13_weight=_Param(inputs["w13"]), w2_weight=_Param(inputs["w2"]),
        moe_tp_size=moe_tp_size, moe_ep_size=1, reduce_results=reduce_results,
    )


def _moe_reduce_registry(calls):
    def entry(x, topk_ids, topk_weights, prepared, out, group=None):
        calls.append("fired")
        out.zero_()

    reg = KernelRegistry()
    reg.register(KernelImpl(
        slot="moe.fused_experts_reduce", bundle_id="t", entry=entry,
        prepare=lambda w13, w2: (w13, w2),
        eligibility=Eligibility(dtypes=frozenset({"float32"})),
    ))
    reg.enable()
    return reg


def test_reduce_owning_kernel_skipped_when_tp_layer_defers_its_reduce(monkeypatch):
    import cacheon.dispatch as dispatch

    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    inputs = _moe_inputs()
    calls: list = []
    dispatched = make_moe_dispatcher(lambda *a: _BASELINE, registry=_moe_reduce_registry(calls))
    topk = SimpleNamespace(topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"])
    layer = _moe_layer(inputs, moe_tp_size=2, reduce_results=False)
    assert dispatched(layer, inputs["x"], topk) is _BASELINE
    assert calls == []  # the kernel never ran — an extra all-reduce would diverge from stock


def test_moe_forwards_v0518_pre_quant_input_to_stock(monkeypatch):
    monkeypatch.delenv("CACHEON_MOE_SEAM", raising=False)
    inputs = _moe_inputs()
    marker = object()
    calls = []

    def baseline(self, hidden_states, topk_output, *, pre_quant_input=None):
        calls.append(pre_quant_input)
        return _BASELINE

    dispatched = make_moe_dispatcher(baseline, registry=KernelRegistry())
    topk = SimpleNamespace(
        topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"]
    )

    assert dispatched(
        _moe_layer(inputs, moe_tp_size=1, reduce_results=False),
        inputs["x"],
        topk,
        pre_quant_input=marker,
    ) is _BASELINE
    assert calls == [marker]


def test_moe_omits_new_optional_argument_for_an_older_stock_signature(monkeypatch):
    monkeypatch.delenv("CACHEON_MOE_SEAM", raising=False)
    inputs = _moe_inputs()

    def baseline(self, hidden_states, topk_output):
        return _BASELINE

    dispatched = make_moe_dispatcher(baseline, registry=KernelRegistry())
    topk = SimpleNamespace(
        topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"]
    )

    assert dispatched(
        _moe_layer(inputs, moe_tp_size=1, reduce_results=False), inputs["x"], topk
    ) is _BASELINE


def test_reduce_owning_kernel_runs_when_layer_reduces(monkeypatch):
    import cacheon.dispatch as dispatch

    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        dispatch, "_tp_device_group", lambda: SimpleNamespace(size=lambda: 2)
    )
    inputs = _moe_inputs()
    calls: list = []
    dispatched = make_moe_dispatcher(lambda *a: _BASELINE, registry=_moe_reduce_registry(calls))
    topk = SimpleNamespace(topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"])
    layer = _moe_layer(inputs, moe_tp_size=2, reduce_results=True)
    out = dispatched(layer, inputs["x"], topk)
    assert out is not _BASELINE and calls == ["fired"]

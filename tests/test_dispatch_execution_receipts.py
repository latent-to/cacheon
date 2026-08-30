"""Control-flow receipts across the non-MSA serving dispatcher families."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import cacheon.dispatch as dispatch  # noqa: E402
import cacheon.dispatch_collective as exchange  # noqa: E402
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402


@pytest.fixture()
def events(monkeypatch):
    completed: list[str] = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: False)
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    return completed


@pytest.fixture()
def failures(monkeypatch):
    """``(slot, exception type)`` for every candidate raise the dispatcher receipted."""

    failed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dispatch._receipts, "failed",
        lambda slot, exc, **_details: failed.append((slot, type(exc).__name__)),
    )
    return failed


def _registry(slot, entry, *, prepare=None, dtype="float32"):
    registry = KernelRegistry()
    registry.register(
        KernelImpl(
            slot=slot,
            bundle_id="test",
            entry=entry,
            prepare=prepare,
            eligibility=Eligibility(dtypes=frozenset({dtype})),
        )
    )
    registry.enable()
    return registry


def _boom(*_args, **_kwargs):
    raise RuntimeError("candidate path failed")


def test_op_dispatchers_receipt_success_and_never_serve_stock(events, failures):
    completed = events
    baseline = object()
    silu = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: baseline,
        registry=_registry(
            "activation.silu_and_mul",
            lambda x, out: out.copy_(x[..., : x.shape[-1] // 2]),
        ),
    )
    assert silu(object(), torch.randn(2, 8)) is not baseline

    rms_self = SimpleNamespace(
        variance_epsilon=1e-6,
        weight=SimpleNamespace(data=torch.ones(8)),
    )
    rms = dispatch.make_rmsnorm_dispatcher(
        lambda *_: baseline,
        registry=_registry(
            "norm.rmsnorm", lambda x, _weight, out, _eps: out.copy_(x)
        ),
    )
    assert rms(rms_self, torch.randn(2, 8)) is not baseline
    assert completed == ["activation.silu_and_mul", "norm.rmsnorm"]

    # A candidate that raises takes the run down with it. Serving stock instead
    # would put stock inside a run that still carries the candidate's name.
    silu_bad = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: pytest.fail("stock served inside a candidate arm"),
        registry=_registry("activation.silu_and_mul", _boom),
    )
    rms_bad = dispatch.make_rmsnorm_dispatcher(
        lambda *_: pytest.fail("stock served inside a candidate arm"),
        registry=_registry("norm.rmsnorm", _boom),
    )
    for call in (
        lambda: silu_bad(object(), torch.randn(2, 8)),
        lambda: rms_bad(rms_self, torch.randn(2, 8)),
    ):
        with pytest.raises(RuntimeError, match="candidate path failed"):
            call()
    assert completed == ["activation.silu_and_mul", "norm.rmsnorm"]
    # The raise is receipted on the way out, naming the slot and the exception,
    # so the verdict can blame the candidate instead of the lane.
    assert failures == [
        ("activation.silu_and_mul", "RuntimeError"),
        ("norm.rmsnorm", "RuntimeError"),
    ]


def test_out_of_domain_call_serves_stock_and_mints_no_receipt(events):
    # A registered candidate whose declared domain excludes this call is not a
    # fallback: stock is the correct answer, and no receipt is minted, so the
    # evidence cannot claim the candidate ran.
    baseline = object()
    wrapped = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: baseline,
        registry=_registry(
            "activation.silu_and_mul",
            lambda x, out: out.copy_(x[..., : x.shape[-1] // 2]),
            dtype="float16",
        ),
    )
    assert wrapped(object(), torch.randn(2, 8)) is baseline
    assert events == []


def _moe_call(entry, *, slot="moe.fused_experts"):
    x = torch.randn(2, 4)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(data=torch.randn(2, 4, 4)),
        w2_weight=SimpleNamespace(data=torch.randn(2, 4, 2)),
        moe_tp_size=1,
        moe_ep_size=1,
        reduce_results=False,
    )
    topk = SimpleNamespace(
        topk_ids=torch.zeros(2, 1, dtype=torch.long),
        topk_weights=torch.ones(2, 1),
    )
    registry = _registry(slot, entry, prepare=lambda *_: object())
    wrapped = dispatch.make_moe_dispatcher(
        lambda *_: "stock", registry=registry, slots=("moe.fused_experts_reduce", slot)
    )
    return wrapped, layer, x, topk


def test_moe_records_success_but_never_falls_back_after_selection(
    events, failures, monkeypatch
):
    completed = events
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")

    def good_entry(x, _ids, _weights, _prepared, out):
        out.copy_(x)

    good, layer, x, topk = _moe_call(good_entry)
    assert torch.is_tensor(good(layer, x, topk))
    bad, layer, x, topk = _moe_call(_boom)
    with pytest.raises(RuntimeError, match="candidate path failed"):
        bad(layer, x, topk)
    assert completed == ["moe.fused_experts"]
    assert failures == [("moe.fused_experts", "RuntimeError")]


def test_moe_selected_audit_prelude_failure_aborts(events, monkeypatch):
    completed = events
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: True)
    monkeypatch.setattr(
        torch.Tensor,
        "clone",
        lambda _self: (_ for _ in ()).throw(RuntimeError("clone failed")),
    )

    def entry(x, _ids, _weights, _prepared, out):
        out.copy_(x)

    wrapped, layer, x, topk = _moe_call(entry)
    with pytest.raises(RuntimeError, match="clone failed"):
        wrapped(layer, x, topk)
    assert completed == []


def test_allreduce_dispatcher_receipts_and_topology_skip(events, monkeypatch):
    completed = events
    monkeypatch.setenv("CACHEON_COLLECTIVE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_allreduce_group_role", lambda *_args: "tp")
    x = torch.randn(2, 4)

    def good_entry(inp, out, _group):
        out.copy_(inp)

    good = dispatch.make_allreduce_dispatcher(
        lambda *_a, **_k: "stock",
        registry=_registry("collective.all_reduce", good_entry),
    )
    coordinator = SimpleNamespace(
        world_size=2, device_group=SimpleNamespace(size=lambda: 2)
    )
    assert torch.is_tensor(good(coordinator, x))
    bad = dispatch.make_allreduce_dispatcher(
        lambda *_a, **_k: "stock",
        registry=_registry("collective.all_reduce", _boom),
    )
    with pytest.raises(RuntimeError, match="candidate path failed"):
        bad(coordinator, x)
    # Single-rank is outside the slot contract (world_size > 1), so stock serves
    # it and no receipt is minted.
    assert good(SimpleNamespace(world_size=1, device_group=object()), x) == "stock"
    assert completed == ["collective.all_reduce"]


def test_compiled_collective_runtime_bodies_route_candidates(events, monkeypatch):
    monkeypatch.setenv("CACHEON_COLLECTIVE_SEAM", "1")
    monkeypatch.setattr(exchange, "_allreduce_group_role", lambda *_args: "tp")
    group = SimpleNamespace(size=lambda: 2)
    coordinator = SimpleNamespace(device_group=group, world_size=2)

    def entry(x, out, _group):
        out.copy_(x * 2)

    registry = _registry("collective.all_reduce", entry)
    inplace = exchange.make_allreduce_inplace_dispatcher(
        lambda *_args: pytest.fail("compiled stock in-place body ran"),
        registry=registry,
    )
    outplace = exchange.make_allreduce_outplace_dispatcher(
        lambda *_args: pytest.fail("compiled stock out-place body ran"),
        registry=registry,
    )
    x = torch.randn(2, 4)
    expected = x * 2
    assert inplace(coordinator, x) is None
    assert torch.equal(x, expected)
    assert torch.equal(outplace(coordinator, x / 2, "auto"), expected)
    assert events == ["collective.all_reduce", "collective.all_reduce"]


@pytest.mark.parametrize(
    "slot,input_rows,output_rows,factory",
    (
        (
            "collective.all_gather_into_tensor",
            2,
            4,
            exchange.make_all_gather_dispatcher,
        ),
        (
            "collective.reduce_scatter_tensor",
            4,
            2,
            exchange.make_reduce_scatter_dispatcher,
        ),
    ),
)
def test_compiled_exchange_runtime_bodies_route_candidates(
    events, monkeypatch, slot, input_rows, output_rows, factory
):
    monkeypatch.setenv("CACHEON_COLLECTIVE_SEAM", "1")
    monkeypatch.setattr(exchange, "_allreduce_group_role", lambda *_args: "attn_tp")
    group = SimpleNamespace(size=lambda: 2)
    coordinator = SimpleNamespace(device_group=group, world_size=2)

    def entry(x, out, _group):
        if output_rows > input_rows:
            out.copy_(x.repeat(output_rows // input_rows, 1))
        else:
            out.copy_(x[:output_rows])

    wrapped = factory(
        lambda *_args: pytest.fail("compiled stock exchange body ran"),
        registry=_registry(slot, entry),
    )
    output = torch.empty(output_rows, 4)
    assert wrapped(coordinator, output, torch.randn(input_rows, 4)) is None
    assert events == [slot]


def _fusion_baseline(x, residual, *_args, **_kwargs):
    return "stock", x + residual


def test_shallow_fusion_receipts(events, monkeypatch):
    completed = events
    monkeypatch.setenv("CACHEON_ARFUSION_SEAM", "1")
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(dispatch, "_arfusion_group", lambda _use_attn: group)
    monkeypatch.setattr(dispatch, "_arfusion_group_role", lambda _use_attn: "tp")
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    weight = torch.ones(4)

    def shallow_entry(x, residual, _weight, _eps, out_norm, out_residual, _group):
        out_norm.copy_(x)
        out_residual.copy_(residual)

    shallow = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.ar_residual_rmsnorm", shallow_entry),
    )
    assert torch.is_tensor(shallow(x, residual, weight)[0])
    shallow_bad = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.ar_residual_rmsnorm", _boom),
    )
    with pytest.raises(RuntimeError, match="candidate path failed"):
        shallow_bad(x, residual, weight)

    assert completed == ["collective.ar_residual_rmsnorm"]

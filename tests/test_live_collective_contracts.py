"""Offline/live parity for the non-deep collective bindings."""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")

import cacheon.dispatch as dispatch  # noqa: E402
from cacheon import receipts  # noqa: E402
from cacheon.registry import (  # noqa: E402
    Eligibility,
    KernelImpl,
    KernelRegistry,
    eligibility_from_metadata,
)
from cacheon.slots import get_slot  # noqa: E402
from cacheon.tensor_spec import OutputSpec, TensorSpec  # noqa: E402
from cacheon.verify_collective import _collective_descriptor  # noqa: E402


ALL_REDUCE = "collective.all_reduce"
MOE_PLAIN = "moe.fused_experts"
_STOCK = object()
_REAL_ALLREDUCE_GROUP_ROLE = dispatch._allreduce_group_role
_REAL_MOE_DP_WORLD_SIZE = dispatch._moe_data_parallel_world_size


class _Group:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class _Param:
    def __init__(self, data):
        self.data = data


class _RecordingRegistry(KernelRegistry):
    def __init__(self):
        super().__init__()
        self.selections = []

    def select(self, slot, descriptor):
        self.selections.append((slot, descriptor))
        return super().select(slot, descriptor)


@pytest.fixture(autouse=True)
def _quiet_runtime(monkeypatch):
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: False)
    monkeypatch.setattr(dispatch._receipts, "completed", lambda _slot: None)
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dispatch, "_allreduce_group_role", lambda _coord, _group: "tp")


def _register(
    slot,
    entry,
    *,
    prepare=None,
    eligibility=None,
    variant="default",
    registry=None,
):
    registry = registry or _RecordingRegistry()
    registry.register(
        KernelImpl(
            slot=slot,
            bundle_id="live-contract-test",
            variant=variant,
            entry=entry,
            prepare=prepare,
            eligibility=eligibility
            or Eligibility(dtypes=frozenset({"float32"})),
        )
    )
    registry.enable()
    return registry


def _offline(shape, *, slot_name=None, world_size=2, model_key=None):
    return _collective_descriptor(
        shape,
        dtype_name="float32",
        device="cpu",
        model_key=model_key,
        architecture=None,
        world_size=world_size,
        slot_name=slot_name,
    )


def _moe_call(num_tokens, *, hidden=8, inter=4, experts=4, top_k=2):
    x = torch.randn(num_tokens, hidden)
    ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    weights = torch.full((num_tokens, top_k), 1.0 / top_k)
    routed = SimpleNamespace(topk_ids=ids, topk_weights=weights)
    return x, routed


def _moe_layer(*, hidden=8, inter=4, experts=4, tp_size=2, reduce=False):
    return SimpleNamespace(
        w13_weight=_Param(torch.randn(experts, 2 * inter, hidden)),
        w2_weight=_Param(torch.randn(experts, hidden, inter)),
        moe_tp_size=tp_size,
        moe_ep_size=1,
        reduce_results=reduce,
        num_local_experts=experts,
        intermediate_size_per_partition=inter,
    )


def _assert_two_phase_parity(registry, slot, expected):
    selections = [row for row in registry.selections if row[0] == slot]
    assert len(selections) == 2
    assert all(row[1] == expected for row in selections)


@pytest.mark.parametrize("capturing", (False, True))
def test_allreduce_live_descriptor_exactly_matches_offline(
    monkeypatch, capturing
):
    monkeypatch.setenv("CACHEON_COLLECTIVE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_in_cuda_graph", lambda: capturing)
    group = _Group(2)
    registry = _register(
        ALL_REDUCE,
        lambda x, out, _group: out.copy_(x),
        eligibility=Eligibility(
            dtypes=frozenset({"float32"})
        ),
    )
    wrapped = dispatch.make_allreduce_dispatcher(
        lambda *_args, **_kwargs: _STOCK, registry=registry
    )

    x = torch.randn(3, 8)
    result = wrapped(SimpleNamespace(world_size=2, device_group=group), x)

    assert torch.equal(result, x)
    expected = _offline({"num_tokens": 3, "hidden": 8})
    if capturing:
        expected = expected.with_updates(graph_mode="cuda_graph")
    _assert_two_phase_parity(
        registry,
        ALL_REDUCE,
        expected,
    )


def test_collective_model_constraint_is_consistently_unavailable_until_arena_binding():
    descriptor = _collective_descriptor(
        {"num_tokens": 3, "hidden": 8},
        dtype_name="float32",
        device="cpu",
        model_key="validator/local/model-path",
        architecture=None,
        world_size=2,
        slot_name=ALL_REDUCE,
    )
    eligibility = eligibility_from_metadata(
        {"capabilities": {"model": "canonical/model"}}, ("float32",)
    )

    assert "model" not in descriptor
    match = eligibility.match(descriptor)
    assert not match.accepted
    assert any(mismatch.field == "model" for mismatch in match.mismatches)


@pytest.mark.parametrize("dp_size", (None, 2))
def test_moe_data_parallel_topology_serves_stock_without_miner_code(
    monkeypatch, dp_size
):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    monkeypatch.setattr(
        dispatch, "_moe_data_parallel_world_size", lambda: dp_size
    )
    prepared = []
    entered = []
    registry = _register(
        MOE_PLAIN,
        lambda *_args: entered.append(True),
        prepare=lambda *_args: prepared.append(True),
    )
    wrapped = dispatch.make_moe_dispatcher(
        lambda *_args: _STOCK,
        registry=registry,
        slots=(MOE_PLAIN,),
    )
    x, routed = _moe_call(3)

    assert wrapped(_moe_layer(), x, routed) is _STOCK
    assert prepared == entered == []
    assert registry.selections == []


def test_pinned_parallel_state_owns_dp_size_and_allreduce_role(monkeypatch):
    generic = _Group(2)
    ps = ModuleType("sglang.srt.distributed.parallel_state")
    ps.get_tp_group = lambda: SimpleNamespace(device_group=generic)
    ps.get_moe_data_parallel_world_size = lambda: 1
    distributed = ModuleType("sglang.srt.distributed")
    distributed.parallel_state = ps
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    monkeypatch.setitem(sys.modules, ps.__name__, ps)

    assert _REAL_MOE_DP_WORLD_SIZE() == 1
    ps.get_moe_data_parallel_world_size = lambda: 2
    assert _REAL_MOE_DP_WORLD_SIZE() == 2
    del ps.get_moe_data_parallel_world_size
    with pytest.raises(RuntimeError, match="cannot resolve MoE-DP topology"):
        _REAL_MOE_DP_WORLD_SIZE()

    tp_coord = ps.get_tp_group()
    assert _REAL_ALLREDUCE_GROUP_ROLE(tp_coord, generic) == "tp"
    dp_coord = SimpleNamespace(device_group=_Group(2))
    assert _REAL_ALLREDUCE_GROUP_ROLE(dp_coord, dp_coord.device_group) is None


def test_quantized_moe_rejects_incomplete_layout(monkeypatch):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    prepared = []
    entered = []
    registry = _register(
        MOE_PLAIN,
        lambda *_args: entered.append(True),
        prepare=lambda *_args: prepared.append(True),
        eligibility=Eligibility(
            dtypes=frozenset({"float32"}), quant=frozenset({"nvfp4"})
        ),
    )
    wrapped = dispatch.make_moe_dispatcher(
        lambda *_args: _STOCK, registry=registry, slots=(MOE_PLAIN,)
    )
    layer = _moe_layer()
    layer.w13_weight = _Param(layer.w13_weight.data.to(torch.uint8))
    layer.w13_weight_scale = _Param(torch.ones(1))
    x, routed = _moe_call(3)

    assert wrapped(layer, x, routed) is _STOCK
    assert prepared == entered == []
    assert registry.selections == []


def test_collective_moe_prepare_failure_is_receipted_without_stock(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.setattr(receipts, "_SCOPE", "")
    receipts.set_scope("prepare-failure")
    stock_calls = []

    def corrupt_then_raise(w13, w2):
        w13.zero_()
        w2.zero_()
        raise RuntimeError("prepare failed")

    registry = _register(
        MOE_PLAIN,
        lambda *_args: pytest.fail("entry must not run"),
        prepare=corrupt_then_raise,
    )
    wrapped = dispatch.make_moe_dispatcher(
        lambda *_args: stock_calls.append(True) or _STOCK,
        registry=registry,
        slots=(MOE_PLAIN,),
    )
    x, routed = _moe_call(3)

    with pytest.raises(RuntimeError, match="prepare failed"):
        wrapped(_moe_layer(), x, routed)
    assert stock_calls == []
    (failed,) = receipts.collect(tmp_path / "prepare-failure", "failed")
    assert failed["slot"] == MOE_PLAIN
    assert failed["phase"] == "prepare"
    assert failed["source"] == "test_live_collective_contracts.py"
    assert failed["line"] > 0


def test_two_moe_variants_prepare_once_and_gap_serves_stock(monkeypatch):
    monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
    prepared = {"small": 0, "large": 0}
    entered = []
    registry = _RecordingRegistry()

    for variant, tokens, value in (("small", 2, 2.0), ("large", 6, 6.0)):
        eligibility = eligibility_from_metadata(
            {"capabilities": {"num_tokens": tokens}},
            ("float32",),
        )

        def prepare(_w13, _w2, *, name=variant):
            prepared[name] += 1
            return name

        def entry(x, _ids, _weights, state, out, *, name=variant, fill=value):
            assert state == name
            entered.append(name)
            out.fill_(fill)

        _register(
            MOE_PLAIN,
            entry,
            prepare=prepare,
            eligibility=eligibility,
            variant=variant,
            registry=registry,
        )

    wrapped = dispatch.make_moe_dispatcher(
        lambda *_args: _STOCK,
        registry=registry,
        slots=(MOE_PLAIN,),
    )
    layer = _moe_layer()
    for tokens, value in ((2, 2.0), (6, 6.0), (2, 2.0), (6, 6.0)):
        x, routed = _moe_call(tokens)
        assert torch.equal(wrapped(layer, x, routed), torch.full_like(x, value))
    x, routed = _moe_call(4)
    assert wrapped(layer, x, routed) is _STOCK

    assert prepared == {"small": 1, "large": 1}
    assert entered == ["small", "large", "small", "large"]


def _typed_slot(slot_name, output_count):
    original = get_slot(slot_name)

    def output_spec(inputs):
        shape = tuple(inputs["x"].shape)
        return OutputSpec(
            tuple(
                TensorSpec(
                    shape,
                    dtype=torch.bfloat16,
                    stride_policy="strided",
                    stride_padding=3,
                    alignment_bytes=64,
                    name=f"typed[{index}]",
                )
                for index in range(output_count)
            )
        )

    return replace(original, output_spec=output_spec)


@pytest.mark.parametrize("slot", [ALL_REDUCE, MOE_PLAIN])
def test_live_candidate_observes_typed_strided_outputs(monkeypatch, slot):
    output_count = 1
    typed = _typed_slot(slot, output_count)
    monkeypatch.setattr(
        dispatch,
        "get_slot",
        lambda name: typed if name == slot else get_slot(name),
    )

    seen = []
    # The all-reduce ABI ends with the process group; the MoE ABI ends with the
    # validator-owned output. Locate the outputs from the slot's own trailing shape.
    trailing = 1 if slot == ALL_REDUCE else 0

    def check(*args):
        end = len(args) - trailing
        outputs = args[end - output_count:end]
        assert len(outputs) == output_count
        for out in outputs:
            seen.append((out.dtype, out.is_contiguous(), out.data_ptr() % 64))
            out.fill_(1)

    if slot == ALL_REDUCE:
        monkeypatch.setenv("CACHEON_COLLECTIVE_SEAM", "1")
        registry = _register(slot, check)
        wrapped = dispatch.make_allreduce_dispatcher(
            lambda *_args, **_kwargs: _STOCK, registry=registry
        )
        result = wrapped(
            SimpleNamespace(world_size=2, device_group=_Group(2)),
            torch.randn(3, 8),
        )
        outputs = (result,)
    else:
        monkeypatch.setenv("CACHEON_MOE_SEAM", "1")
        registry = _register(
            slot,
            check,
            prepare=lambda _w13, _w2: "prepared",
        )
        wrapped = dispatch.make_moe_dispatcher(
            lambda *_args: _STOCK,
            registry=registry,
            slots=(slot,),
        )
        x, routed = _moe_call(3)
        outputs = (wrapped(_moe_layer(), x, routed),)

    assert seen == [(torch.bfloat16, False, 0)] * output_count
    assert all(out.dtype == torch.bfloat16 and not out.is_contiguous() for out in outputs)


@pytest.mark.parametrize(
    "mutation",
    ("output_shape", "output_storage", "output_stride", "input_storage"),
)
def test_post_entry_binding_mutation_aborts_collective(monkeypatch, mutation):
    monkeypatch.setenv("CACHEON_COLLECTIVE_SEAM", "1")

    def mutate(x, out, _group):
        if mutation == "output_shape":
            out.resize_(1)
        elif mutation == "output_storage":
            out.set_(torch.empty_like(out))
            out.copy_(x)
        elif mutation == "output_stride":
            out.as_strided_(out.shape, (1, out.shape[0]))
            out.copy_(x)
        else:
            x.set_(x.detach().clone())
            out.copy_(x)

    registry = _register(ALL_REDUCE, mutate)
    wrapped = dispatch.make_allreduce_dispatcher(
        lambda *_args, **_kwargs: _STOCK, registry=registry
    )
    with pytest.raises(ValueError, match="validator-owned storage"):
        wrapped(
            SimpleNamespace(world_size=2, device_group=_Group(2)),
            torch.randn(3, 8),
        )


def test_cuda_graph_detector_supports_current_legacy_and_direct_capture(monkeypatch):
    current_name = (
        "sglang.srt.model_executor.runner_backend_utils."
        "tc_piecewise_cuda_graph"
    )
    legacy_name = "sglang.srt.compilation.piecewise_context_manager"
    current = ModuleType(current_name)
    legacy = ModuleType(legacy_name)
    current.is_in_tc_piecewise_cuda_graph = lambda: True
    legacy.is_in_piecewise_cuda_graph = lambda: False
    monkeypatch.setitem(sys.modules, current_name, current)
    monkeypatch.setitem(sys.modules, legacy_name, legacy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dispatch._in_cuda_graph()

    current.is_in_tc_piecewise_cuda_graph = lambda: False
    legacy.is_in_piecewise_cuda_graph = lambda: True
    assert dispatch._in_cuda_graph()

    legacy.is_in_piecewise_cuda_graph = lambda: False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert dispatch._in_cuda_graph()


def test_post_selection_allocation_failure_aborts_without_fired_receipt(monkeypatch):
    monkeypatch.setenv("CACHEON_COLLECTIVE_SEAM", "1")
    written = []
    monkeypatch.setattr(receipts, "write", lambda kind, payload, **kw: written.append(kind))
    monkeypatch.setattr(
        dispatch,
        "allocate_output_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("allocation failed")),
    )
    entered = []
    stock_calls = []
    registry = _register(ALL_REDUCE, lambda *_args: entered.append(True))
    wrapped = dispatch.make_allreduce_dispatcher(
        lambda *_args, **_kwargs: stock_calls.append(True) or _STOCK,
        registry=registry,
    )

    with pytest.raises(RuntimeError, match="allocation failed"):
        wrapped(
            SimpleNamespace(world_size=2, device_group=_Group(2)),
            torch.randn(3, 8),
        )
    assert entered == []
    assert stock_calls == []
    assert "fired" not in written

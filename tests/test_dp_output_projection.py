"""Fused output-preparation math and reversible stock-SGLang tensor flow."""

import sys
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from cacheon.dp_output_projection_contract import SLOT, make_inputs, output_spec, reference
from cacheon.integrations import sglang_dp_output as seam
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry


def test_bundle_declares_preparation_and_reviewed_native_build():
    from cacheon.manifest import load_manifest
    from cacheon.rebuild import parse_rebuild_plan
    from cacheon.target_catalog import resolve_intake_target, FEATURE_REBUILD_BUILD_CUDA_EXT
    from cacheon.slots import get_slot

    root = Path(__file__).resolve().parents[1] / "bundles/dp_output_projection"
    manifest = load_manifest(root)
    assert manifest.ops[0].prepare == get_slot(SLOT).prepare == "prepare"
    assert manifest.ops[0].cuda_sources == ("kernels/dp_projection_native.cu",)
    assert parse_rebuild_plan(root).steps[0].patcher_id == "cacheon.build-cuda-ext.v1"
    resolved = resolve_intake_target(manifest, observed_features={FEATURE_REBUILD_BUILD_CUDA_EXT})
    assert resolved.target_id == SLOT
    from cacheon.chain.reference_copy_policy import reference_copy_match, library_copy_match
    from cacheon.copy_fingerprint import fingerprint_submitted_delta
    fingerprint = fingerprint_submitted_delta(root)
    assert reference_copy_match(fingerprint) is None
    assert library_copy_match(fingerprint) is None


def test_projection_reference_keeps_both_bf16_rounding_points():
    inputs = make_inputs(num_tokens=1, input_dim=16, hidden=16, dtype=torch.bfloat16,
                         device="cpu", seed=7, quantize=False)
    inputs.update(x=torch.ones(1, 16, dtype=torch.bfloat16),
                  weight=torch.eye(16, dtype=torch.bfloat16),
                  residual=torch.ones(1, 16, dtype=torch.bfloat16),
                  gamma=torch.ones(16, dtype=torch.bfloat16), epsilon=0.)
    normalized, updated, packed, scales = reference(inputs, None, 0, 1)
    assert torch.equal(updated, torch.full_like(updated, 2))
    assert torch.equal(normalized, torch.ones_like(normalized))
    assert packed.numel() == scales.numel() == 0
    from cacheon.verify import _compare_outputs
    from cacheon.slots import get_slot
    slot = get_slot(SLOT)
    result = _compare_outputs([normalized, updated, packed, scales],
                              [normalized, updated, packed, scales],
                              tolerance_for=slot.tolerance_for, correctness=slot.correctness)
    assert result.passed
    assert [tuple(x.shape) for x in (normalized, updated, packed, scales)] == [
        x.shape for x in output_spec(inputs).outputs]


def test_fp4_reference_represents_constant_signed_blocks():
    from cacheon.dp_output_projection_contract import quantize_reference

    hidden = torch.cat((torch.full((1, 16), 6.), torch.full((1, 16), -6.))).bfloat16()
    packed, scales = quantize_reference(hidden, torch.tensor([1.]))
    assert torch.equal(packed[0], torch.full((8,), 0x77, dtype=torch.uint8))
    assert torch.equal(packed[1], torch.full((8,), 0xff, dtype=torch.uint8))
    assert torch.equal(scales.view(torch.float8_e4m3fn).float(), torch.ones(2, 1))


@pytest.fixture(scope="module")
def native_projection():
    """Exercise the declared native bundle when launched with four CUDA ranks."""
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires torchrun with four CUDA ranks")
    import torch.distributed as dist
    from cacheon.rebuild import apply_rebuild_plan
    from cacheon.sandbox import load_module

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    bundle = Path(__file__).resolve().parents[1] / "bundles/dp_output_projection"
    build_error = None
    if rank == 0:
        try:
            apply_rebuild_plan(bundle, phase="all")
        except Exception as error:
            build_error = error
    dist.barrier(device_ids=[rank])
    if build_error is not None:
        raise build_error
    if rank != 0:
        apply_rebuild_plan(bundle, phase="all")
    module = load_module(bundle / "kernels/projection.py")
    yield module, dist.group.WORLD, rank
    dist.destroy_process_group()


@pytest.mark.parametrize("factor", [1536., 2457.60009765625])
def test_native_quantizer_preserves_e4m3_scale_midpoints(native_projection, factor):
    from cacheon.dp_output_projection_contract import quantize_reference

    module, group, rank = native_projection
    device = torch.device("cuda", rank)
    # Zero projection and unit residual make normalized output exactly gamma.
    # Dense BF16 mantissas span both sides of, and exact, E4M3 scale midpoints.
    gamma = torch.arange(0x3B00, 0x3C80, dtype=torch.int16, device=device).view(
        torch.bfloat16).repeat_interleave(16)
    weight = torch.zeros(6144, 16384, dtype=torch.bfloat16, device=device)
    x = torch.zeros(1, 16384, dtype=torch.bfloat16, device=device)
    residual = torch.ones(1, 6144, dtype=torch.bfloat16, device=device)
    scale = torch.tensor([factor], dtype=torch.float32, device=device)
    prepared = module.prepare(weight, gamma, 0., scale)
    normalized = torch.empty(4, 6144, dtype=torch.bfloat16, device=device)
    updated = torch.empty_like(residual)
    packed = torch.empty(4, 3072, dtype=torch.uint8, device=device)
    scales = torch.empty(4, 384, dtype=torch.uint8, device=device)
    module.project_gather_norm(x, residual, prepared, normalized, updated, packed, scales, group)
    assert torch.equal(normalized, gamma.expand_as(normalized))
    assert torch.equal(updated, residual)
    expected_packed, expected_scales = quantize_reference(normalized, scale)
    assert torch.equal(scales, expected_scales)
    assert (packed == expected_packed).float().mean().item() >= .99


@pytest.fixture
def stock_flow(monkeypatch):
    class Dispatch(NamedTuple):
        hidden_states: torch.Tensor
        hidden_states_scale: torch.Tensor | None
        topk_output: object = None

    class Linear:
        def __init__(self):
            self.weight = torch.eye(16, dtype=torch.bfloat16)

        def forward(self, x):
            return x @ self.weight.T, None

    class Communicator:
        def __init__(self):
            self.post_attention_layernorm = SimpleNamespace(
                weight=torch.ones(16, dtype=torch.bfloat16), variance_epsilon=0.)

        def prepare_mlp(self, hidden, residual, batch):
            return hidden + residual, residual

    scale = torch.tensor([1.])
    calls = []
    moe = ModuleType(seam._MOE)

    def fp4(dispatch, quant_info, config):
        calls.append(dispatch)
        return dispatch.hidden_states

    moe.fused_experts_none_to_flashinfer_trtllm_fp4 = fp4

    class Layer:
        def __init__(self):
            self.self_attn = SimpleNamespace(o_proj=Linear())
            self.layer_communicator = Communicator()

        def forward(self, positions, hidden, forward_batch, residual):
            projected, _ = self.self_attn.o_proj.forward(hidden)
            normalized, updated = self.layer_communicator.prepare_mlp(projected, residual, forward_batch)
            moe.fused_experts_none_to_flashinfer_trtllm_fp4(
                Dispatch(normalized, None), SimpleNamespace(w13_input_scale_quant=scale), None)
            return normalized, updated

    model, comm = ModuleType(seam._MODEL), ModuleType(seam._COMM)
    model.DeepseekV2DecoderLayer, comm.LayerCommunicator = Layer, Communicator
    for name, value in [(seam._MODEL, model), (seam._COMM, comm), (seam._MOE, moe)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("CACHEON_DP_OUTPUT_PROJECTION_SEAM", "1")
    monkeypatch.setattr(seam._audit, "sampled", lambda: False)
    registry = KernelRegistry()
    prepares = []

    def prepare(weight, gamma, epsilon, quant_scale):
        prepares.append(weight)
        return dict(weight=weight, gamma=gamma, epsilon=epsilon, quant_scale=quant_scale)

    def entry(x, residual, prepared, normalized, updated, packed, scales, group):
        expected = reference(dict(x=x, residual=residual, world_size=1, **prepared), None, 0, 1)
        for actual, truth in zip((normalized, updated, packed, scales), expected):
            actual.copy_(truth)

    impl = KernelImpl(slot=SLOT, bundle_id="flow-test", entry=entry, prepare=prepare,
                      eligibility=Eligibility(dtypes=frozenset({"bfloat16"})))
    registry.register(impl)
    registry.enable()
    group = SimpleNamespace(world_size=1, rank_in_group=0, device_group=None)
    monkeypatch.setattr(seam, "_select", lambda x, linear, registry: seam._Deferred(x, linear, impl, group, scale))
    seam.install(registry)
    yield Layer, registry, calls, prepares, impl
    seam.uninstall()


@pytest.mark.parametrize("capture", [False, True])
def test_selection_reaches_bundle_in_eager_audit_and_graph_capture(monkeypatch, capture):
    def gather():
        pass

    group = SimpleNamespace(world_size=4)
    parallel = SimpleNamespace(attn_tp_size=1, attn_dp_size=4, tp_size=4)
    modules = {
        "sglang.srt.distributed": dict(get_tp_group=lambda: group),
        "sglang.srt.model_executor.runner": dict(get_is_capture_mode=lambda: capture),
        "sglang.srt.runtime_context": dict(get_parallel=lambda: parallel),
        seam._COMM: dict(CommunicateWithAllReduceAndLayerNormFn=SimpleNamespace(
            _gather_hidden_states_and_residual=gather)),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(seam, "_arch_tag", lambda device: "sm103")
    monkeypatch.setattr(seam, "_in_cuda_graph", lambda: capture)
    monkeypatch.setattr(seam, "_quant_scale", lambda state, x: torch.empty(0))
    registry = KernelRegistry()
    impl = KernelImpl(slot=SLOT, bundle_id="selection-test", entry=lambda *args: None,
                      eligibility=Eligibility(dtypes=frozenset({"bfloat16"})))
    registry.register(impl)
    registry.enable()
    layer = SimpleNamespace(layer_communicator=SimpleNamespace(
        _communicate_with_all_reduce_and_layer_norm_fn=gather))
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
                            dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
                            can_run_tbo=False)
    linear = SimpleNamespace(weight=torch.eye(16, dtype=torch.bfloat16), bias=None)
    x = torch.ones(2, 16, dtype=torch.bfloat16)
    token = seam._scope.set(seam._Scope(layer, batch))
    try:
        selected = seam._select(x, linear, registry)
        assert isinstance(selected, seam._Deferred)
        assert selected.implementation is impl and selected.group is group
    finally:
        seam._scope.reset(token)


def test_stock_methods_receive_fused_outputs_and_existing_fp4_dispatch(stock_flow):
    Layer, registry, calls, prepares, impl = stock_flow
    layer = Layer()
    x = torch.ones(1, 16, dtype=torch.bfloat16)
    for _ in range(2):
        normalized, updated = layer.forward(None, x, object(), x)
        assert torch.equal(normalized, x)
        assert torch.equal(updated, 2 * x)
        assert calls[-1].hidden_states.dtype == torch.uint8
        assert calls[-1].hidden_states_scale.dtype == torch.float8_e4m3fn
        assert seam._scope.get() is None
    assert len(prepares) == 1
    seam.uninstall()
    normalized, _ = layer.forward(None, x, object(), x)
    assert torch.equal(normalized, 2 * x)
    assert calls[-1].hidden_states_scale is None


def test_failed_bundle_clears_scope_and_does_not_call_stock(stock_flow):
    Layer, registry, calls, prepares, impl = stock_flow
    def fail(*args):
        raise RuntimeError("projection failed")
    impl.entry = fail
    x = torch.ones(1, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="projection failed"):
        Layer().forward(None, x, object(), x)
    assert not calls
    assert seam._scope.get() is None


@pytest.mark.parametrize("live_rows", [None, 0, 1, 3])
@pytest.mark.parametrize("corrupt", [False, True])
def test_live_audit_uses_real_rows_without_changing_outputs(stock_flow, monkeypatch, live_rows, corrupt):
    Layer, registry, calls, prepares, impl = stock_flow
    monkeypatch.setattr(seam._audit, "sampled", lambda: True)
    monkeypatch.setattr(seam._audit, "_stats", {})
    original = impl.entry

    def entry(x, residual, prepared, normalized, *args):
        original(x, residual, prepared, normalized, *args)
        if corrupt:
            normalized[:live_rows].add_(10)

    impl.entry = entry
    x = torch.ones(3, 16, dtype=torch.bfloat16)
    if live_rows is not None:
        x[live_rows:] = float("nan")
    normalized, _ = Layer().forward(
        None, x, SimpleNamespace(original_global_num_tokens_cpu=None if live_rows is None else [live_rows]), x)
    stats = seam._audit._stats[SLOT]
    assert normalized.shape == x.shape
    assert (torch.isfinite(normalized).all() if live_rows is None
            else torch.isnan(normalized[live_rows:]).all())
    assert stats["n"] == int(bool(live_rows))
    assert stats["baseline_refused"] == int(not live_rows)
    assert stats["violations"] == int(corrupt and bool(live_rows))


@pytest.mark.parametrize("rank", range(4))
def test_audit_rows_preserve_rank_order_and_local_residual(rank):
    counts = [0, 1, 3, 2]
    outputs = (torch.arange(192).reshape(12, 16), torch.arange(48).reshape(3, 16),
               torch.arange(96, dtype=torch.uint8).reshape(12, 8), torch.arange(12).reshape(12, 1))
    selected = seam._audit_rows(outputs, SimpleNamespace(original_global_num_tokens_cpu=counts),
                                SimpleNamespace(world_size=4, rank_in_group=rank))
    for index in (0, 2, 3):
        assert torch.equal(selected[index], outputs[index][[3, 6, 7, 8, 9, 10]])
    assert torch.equal(selected[1], outputs[1][:counts[rank]])


@pytest.mark.parametrize("counts", [None, [1], [-1, 1], [4, 1], [True, 1]])
def test_audit_rows_reject_invalid_original_counts(counts):
    outputs = (torch.ones(6, 16), torch.ones(3, 16), torch.empty(0), torch.empty(0))
    with pytest.raises(RuntimeError, match="original per-rank token counts"):
        seam._audit_rows(outputs, SimpleNamespace(original_global_num_tokens_cpu=counts),
                         SimpleNamespace(world_size=2, rank_in_group=0))

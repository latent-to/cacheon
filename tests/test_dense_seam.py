"""SGLang unquantized-linear adapter for the generic dense slot."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import cacheon.integrations.sglang_dense as dense_seam  # noqa: E402
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from cacheon.dense_contract import call_descriptor, dense_reference, make_dense_inputs  # noqa: E402
from cacheon.sandbox import load_entry  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402
from cacheon.verify import verify_entry  # noqa: E402
from support.graph_backend import FakeGraphBackend  # noqa: E402

_SOURCE = str(Path(__file__).resolve().parents[1] / "examples/miner_dense_torch/kernels/dense.py")
_ENTRY, _PREPARE = load_entry(_SOURCE, "dense"), load_entry(_SOURCE, "prepare")


class _Method:
    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)

    def apply_into(self, layer, x, output, bias=None):
        return output.copy_(self.apply(layer, x, bias))


def _registry(entry, prepare=lambda weight: weight):
    registry = KernelRegistry()
    registry.register(
        KernelImpl(
            slot="linear.dense",
            bundle_id="dense-test",
            entry=entry,
            prepare=prepare,
            eligibility=Eligibility(dtypes=frozenset({"float32", "bfloat16", "float16"})),
        )
    )
    registry.enable()
    return registry


@pytest.fixture()
def layer(monkeypatch):
    module = ModuleType(dense_seam._MODULE)
    module.UnquantizedLinearMethod = _Method
    monkeypatch.setitem(sys.modules, dense_seam._MODULE, module)
    monkeypatch.setattr(dense_seam, "_runtime_parallel_sizes", lambda: (1, 1))
    monkeypatch.setattr(dense_seam._audit, "sampled", lambda: False)
    yield SimpleNamespace(
        weight=torch.randn(7, 5),
        tp_size=1,
        gather_output=False,
    )
    dense_seam.uninstall()


def test_dense_adapter_routes_apply_and_apply_into_once_prepared(monkeypatch, layer):
    monkeypatch.setenv("CACHEON_DENSE_SEAM", "1")
    prepares = []

    def prepare(weight):
        prepares.append(weight)
        return weight

    def entry(x, weight, out):
        torch.mm(x, weight.t(), out=out)

    dense_seam.install(_registry(entry, prepare))
    method = _Method()
    x = torch.randn(3, 5)
    expected = torch.nn.functional.linear(x, layer.weight)
    assert torch.allclose(method.apply(layer, x), expected)
    output = torch.empty(3, 7)
    assert method.apply_into(layer, x, output) is output
    assert torch.allclose(output, expected)
    assert len(prepares) == 1
    assert prepares[0].data_ptr() == layer.weight.data_ptr()


def test_dense_adapter_stays_stock_outside_exact_domain(monkeypatch, layer):
    monkeypatch.delenv("CACHEON_DENSE_SEAM", raising=False)
    dense_seam.install(_registry(lambda *_args: pytest.fail("candidate fired")))
    x = torch.randn(3, 5)
    assert torch.allclose(_Method().apply(layer, x), x @ layer.weight.t())


def test_dense_adapter_candidate_error_is_not_stock_fallback(monkeypatch, layer):
    monkeypatch.setenv("CACHEON_DENSE_SEAM", "1")

    def broken(*_args):
        raise RuntimeError("dense candidate failed")

    dense_seam.install(_registry(broken))
    with pytest.raises(RuntimeError, match="dense candidate failed"):
        _Method().apply(layer, torch.randn(3, 5))
    with pytest.raises(RuntimeError, match="dense candidate failed"):
        _Method().apply_into(layer, torch.randn(3, 5), torch.empty(3, 7))


@pytest.mark.parametrize("method_name", ["apply", "apply_into"])
def test_dense_reuses_inference_workspace_outside_capture(monkeypatch, layer, method_name):
    monkeypatch.setenv("CACHEON_DENSE_SEAM", "1")
    prepares = []

    def prepare(weight):
        prepares.append(weight)
        return weight, torch.ones(1)

    def entry(x, prepared, out):
        weight, workspace = prepared
        workspace.zero_()
        torch.mm(x, weight.t(), out=out)

    dense_seam.install(_registry(entry, prepare))
    method = getattr(_Method(), method_name)
    x = torch.randn(3, 5)
    with torch.inference_mode():
        output = torch.empty(3, 7)
        args = (layer, x, output) if method_name == "apply_into" else (layer, x)
        method(*args)
    actual = method(*args)
    assert torch.allclose(actual, x @ layer.weight.t())
    assert len(prepares) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_family_literal_math_and_two_new_profiles(dtype):
    x = torch.tensor([[1., -2., 3.], [4., 0., -1.]], dtype=dtype)
    w = torch.tensor([[2., 1., -1.], [-3., 2., 4.]], dtype=dtype)
    out = torch.empty(2, 4)[:, ::2]
    _ENTRY(x, _PREPARE(w), out)
    assert out.tolist() == [[-3., 5.], [9., -16.]]
    assert torch.equal(dense_reference(dict(x=x, weight=w, output_dtype="float32"))[0], out)
    a = torch.tensor([[[1., -2.], [3., 4.]], [[2., 3.], [-1., 2.]]], dtype=dtype)
    b = torch.tensor([[[5., 6.], [-1., 2.]], [[-2., 1.], [4., 3.]]], dtype=dtype)
    result = torch.empty(2, 2, 2, dtype=dtype).transpose(0, 1)
    _ENTRY(a, _PREPARE(b.transpose(-1, -2)), result)
    assert result.tolist() == [[[7., 2.], [11., 26.]], [[8., 11.], [10., 5.]]]
    verify = verify_entry(get_slot("linear.dense"), _ENTRY, prepare=_PREPARE, dtype=dtype,
        device="cpu", seed=7, graph_replays=3, _graph_backend=FakeGraphBackend())
    assert verify.passed


def test_family_stale_activation_is_rejected_on_graph_replay():
    backend, saved = FakeGraphBackend(), None
    def stale(x, prepared, out):
        nonlocal saved
        if backend.phase == "capture":
            saved = x.clone()
        _ENTRY(saved if backend.phase == "replay" else x, prepared, out)
    assert not verify_entry(get_slot("linear.dense"), stale, prepare=_PREPARE,
        dtype=torch.float32, device="cpu", seed=7, graph_replays=3,
        _graph_backend=backend).passed


def test_family_install_uses_pinned_indexer_name_and_keeps_compiled_consumers(monkeypatch, layer):
    monkeypatch.setenv("CACHEON_DENSE_SEAM", "1")
    class Indexer:
        def _weights_proj_bf16_in_fp32_out(self, x):
            raise AssertionError("head projection used stock")
    class MoEGate:
        def forward(self, x, gemm_output_zero_allocator=None, forward_batch=None):
            raise AssertionError("router projection used stock")
    head_module = SimpleNamespace(Indexer=Indexer, _is_cuda=True)
    router_module = SimpleNamespace(MoEGate=MoEGate, _is_cuda=True, get_exec=lambda:
        SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=False)))
    bmm_module = SimpleNamespace(torch=torch)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.dsa.dsa_indexer", head_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.models.deepseek_v2", router_module)
    monkeypatch.setitem(sys.modules, dense_seam._BMM_MODULE, bmm_module)
    calls = []
    def entry(x, prepared, out):
        calls.append((x.ndim, out.dtype))
        _ENTRY(x, prepared, out)
    dense_seam.install(_registry(entry))
    dense_seam.install(_registry(entry))
    head, router = Indexer(), MoEGate()
    head.weights_proj = SimpleNamespace(weight=torch.ones(5, 3, dtype=torch.bfloat16))
    router.weight = torch.ones(5, 3, dtype=torch.bfloat16)
    compiled = torch.compile(lambda x: head._weights_proj_bf16_in_fp32_out(x) * .5,
                             dynamic=True, backend="eager")
    assert torch.equal(compiled(torch.ones(2, 3, dtype=torch.bfloat16)), torch.full((2, 5), 1.5))
    assert torch.equal(compiled(torch.full((4, 3), 2., dtype=torch.bfloat16)), torch.full((4, 5), 3.))
    assert router.forward(torch.ones(2, 3, dtype=torch.bfloat16)).dtype == torch.float32
    a, b = torch.ones(3, 2, 4).transpose(0, 1), torch.ones(2, 5, 4).transpose(1, 2)
    out = torch.empty(3, 2, 5).transpose(0, 1)
    original_bmm = torch.bmm
    assert bmm_module.torch.bmm(a, b, out=out) is out
    assert torch.equal(out, original_bmm(a, b))
    compiled_bmm = torch.compile(lambda x: bmm_module.torch.bmm(x, b), dynamic=True, backend="eager")
    assert torch.equal(compiled_bmm(a + 1), original_bmm(a + 1, b))
    assert calls == [(2, torch.float32)] * 3 + [(3, torch.float32)] * 2
    dense_seam.uninstall()
    assert bmm_module.torch is torch and torch.bmm is original_bmm


def test_family_prepare_invalidates_changed_weight_and_audit_copies_output(monkeypatch, layer):
    monkeypatch.setenv("CACHEON_DENSE_SEAM", "1")
    prepares = []
    def prepare(w):
        prepares.append(w.clone())
        return w.clone()
    dense_seam.install(_registry(_ENTRY, prepare))
    x = torch.ones(3, 5)
    _Method().apply(layer, x)
    layer.weight.add_(1)
    assert torch.allclose(_Method().apply(layer, x), x @ layer.weight.T)
    assert len(prepares) == 2
    seen = []
    monkeypatch.setattr(dense_seam._audit, "sampled", lambda: True)
    monkeypatch.setattr(dense_seam._audit, "run", lambda slot, actual, expected: seen.append(expected()))
    a, b, out = torch.ones(2, 3, 4), torch.ones(2, 4, 5), torch.empty(3, 2, 5).transpose(0, 1)
    dense_seam._make_bmm(torch.bmm, _registry(lambda x, w, out: out.zero_()))(a, b, out=out)
    assert torch.equal(seen[0], torch.full((2, 3, 5), 4.))
    dense_seam.uninstall()
    invocations = []
    def wrong(x, prepared, out):
        invocations.append(1)
        out.zero_()
    dense_seam.install(_registry(wrong))
    _Method().apply_into(layer, x, torch.empty(3, 7))
    assert invocations == [1]
    assert torch.equal(seen[-1], x @ layer.weight.T)


def test_family_new_consumers_propagate_failures_and_reject_output_rebinding(monkeypatch, layer):
    monkeypatch.setenv("CACHEON_DENSE_SEAM", "1")
    a, b = torch.ones(2, 3, 4), torch.ones(2, 4, 5)
    def broken(*args):
        raise RuntimeError("family candidate failed")
    def stock(*args, **kwargs):
        raise AssertionError("candidate failure retried stock")
    with pytest.raises(RuntimeError, match="family candidate failed"):
        dense_seam._make_bmm(stock, _registry(broken))(a, b)
    module = SimpleNamespace(_is_cuda=True)
    head = SimpleNamespace(weights_proj=SimpleNamespace(weight=torch.ones(5, 4)))
    with pytest.raises(RuntimeError, match="family candidate failed"):
        dense_seam._make_projection(stock, _registry(broken), module, "weights_proj")(head, a[0])
    def rebind(x, prepared, out):
        out.set_(torch.zeros_like(out))
    with pytest.raises(ValueError, match="storage"):
        dense_seam._make_bmm(stock, _registry(rebind))(a, b)


def test_family_descriptors_preserve_ordinary_and_identify_added_regimes():
    base = dict(num_tokens=3, input_dim=7, output_dim=5, dtype=torch.bfloat16, device="cpu", seed=1)
    ordinary = call_descriptor(make_dense_inputs(**base))
    projection = call_descriptor(make_dense_inputs(**base, output_dtype="float32"))
    batched = call_descriptor(make_dense_inputs(**base, batch_size=2))
    assert ordinary["layout"] == "weight_out_in_row_major"
    assert projection["layout"] == "weight_out_in_fp32_output"
    assert batched["layout"] == "batched_weight_out_in_strided" and batched["batch_size"] == 2

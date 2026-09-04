"""SGLang unquantized-linear adapter for the generic dense slot."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import cacheon.integrations.sglang_dense as dense_seam  # noqa: E402
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402


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
            eligibility=Eligibility(dtypes=frozenset({"float32"})),
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

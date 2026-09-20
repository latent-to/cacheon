"""Unfused reference arithmetic preserves the model's residual dtype boundary."""

import sys
from types import ModuleType

import pytest
import torch

from cacheon.integrations import sglang_nodes as nodes


class _RMSNorm(torch.nn.Module):
    """The pinned native path normalizes its FP32 sum before rounding the residual."""

    fp32_residual = False
    override_orig_dtype = None

    def __init__(self):
        super().__init__()
        self._forward_method = self.forward_native

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, x, residual=None, post_residual_addition=None, quant_linear=None):
        values = x.float() if residual is None else x.float() + residual.float()
        out = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        if residual is None:
            return out
        return out, values if self.fp32_residual else values.to(x.dtype)


@pytest.fixture
def norm(monkeypatch):
    module = ModuleType("sglang.srt.layers.layernorm")
    module.RMSNorm = _RMSNorm
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return _RMSNorm()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_native_twin_keeps_unfused_residual_rounding(norm, dtype):
    generator = torch.Generator().manual_seed(27)
    x = torch.randn(32, 16, generator=generator).to(dtype)
    residual = torch.randn(32, 16, generator=generator).to(dtype)
    summed = (x.double() + residual.double()).to(dtype)
    values = summed.double()
    expected = (values / (values.square().mean(-1, keepdim=True) + 1e-6).sqrt()).to(dtype)
    saved = norm._forward_method
    with nodes._native(norm):
        actual, updated = norm(x, residual)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(updated, summed)
    assert norm._forward_method == saved


def test_explicit_fp32_residual_and_plain_norm_keep_native_semantics(norm):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    residual = torch.full_like(x, 0.01)
    norm.fp32_residual = True
    expected = norm.forward_native(x, residual)
    with nodes._native(norm):
        actual = norm(x, residual)
        plain = norm(x)
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    assert actual[1].dtype == torch.float32
    assert torch.equal(plain, norm.forward_native(x))

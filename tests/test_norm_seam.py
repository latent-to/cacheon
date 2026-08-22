"""norm.rmsnorm — the seam must reach every norm class the pinned runtime uses.

This file exists because its absence let a real gap ship: the adapter patched
``RMSNorm`` only, MiniMax-M3 runs ``GemmaRMSNorm`` (a sibling class, not a subclass)
at every norm callsite, and a candidate for this slot therefore could not execute in
the production arena at all. Miners paid for a target nothing would ever call.

Covers the seam-table rows (so the compat canary asserts BOTH chokepoints against the
pinned runtime) and the adapter's install/uninstall over a fake layernorm module.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from cacheon.integrations import sglang_norm  # noqa: E402
from cacheon.registry import KernelRegistry  # noqa: E402
from cacheon.seams import SEAM_ADAPTERS  # noqa: E402

_MODULE = "sglang.srt.layers.layernorm"


def _fake_layernorm_module():
    class _Base:
        def forward_cuda(self, x, residual=None, post_residual_addition=None):
            return "stock-cuda"

        def forward_native(self, x, residual=None, post_residual_addition=None):
            return "stock-native"

    mod = ModuleType(_MODULE)
    mod.RMSNorm = type("RMSNorm", (_Base,), {})
    mod.GemmaRMSNorm = type("GemmaRMSNorm", (_Base,), {})
    return mod


@pytest.fixture
def fake_layernorm(monkeypatch):
    mod = _fake_layernorm_module()
    monkeypatch.setitem(sys.modules, _MODULE, mod)
    yield mod
    sglang_norm.uninstall()


def test_seam_table_asserts_both_norm_chokepoints():
    rows = {a.chokepoint: a for a in SEAM_ADAPTERS if a.target_module == _MODULE}
    assert set(rows) == {"RMSNorm.forward_cuda", "GemmaRMSNorm.forward_cuda"}
    assert all(a.slots == ("norm.rmsnorm",) for a in rows.values())
    assert all(a.integration == "sglang_norm" for a in rows.values())


def test_install_patches_both_classes_and_uninstall_restores(fake_layernorm):
    stock = {
        name: (cls.forward_cuda, cls.forward_native)
        for name, cls in (("RMSNorm", fake_layernorm.RMSNorm),
                          ("GemmaRMSNorm", fake_layernorm.GemmaRMSNorm))
    }

    sglang_norm.install(KernelRegistry())
    for name, cls in (("RMSNorm", fake_layernorm.RMSNorm),
                      ("GemmaRMSNorm", fake_layernorm.GemmaRMSNorm)):
        assert cls.forward_cuda is not stock[name][0], f"{name} left cold"
        assert cls.forward_native is not stock[name][1], f"{name} left cold"

    sglang_norm.install(KernelRegistry())  # idempotent: no double wrap
    assert fake_layernorm.GemmaRMSNorm._cacheon_orig_cuda is stock["GemmaRMSNorm"][0]

    sglang_norm.uninstall()
    for name, cls in (("RMSNorm", fake_layernorm.RMSNorm),
                      ("GemmaRMSNorm", fake_layernorm.GemmaRMSNorm)):
        assert cls.forward_cuda is stock[name][0]
        assert cls.forward_native is stock[name][1]


def test_install_skips_a_class_the_pinned_runtime_does_not_define(fake_layernorm):
    # An absent class is the compat canary's business, not a startup crash.
    del fake_layernorm.GemmaRMSNorm
    sglang_norm.install(KernelRegistry())
    assert getattr(fake_layernorm.RMSNorm, "_cacheon_patched", False)

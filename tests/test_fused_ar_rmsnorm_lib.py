"""cacheon_kernels.collective.fused_ar_rmsnorm — the portable library spine, CPU-only.

No GPU, no sglang. Pins two things:
* the measured dispatch constants survive refactors (one-shot/two-shot crossover 48,
  prefill fall-through 1024 — each was a real regression once);
* the module stays import-clean of sglang and the harness (Axiom 5).
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from cacheon_kernels.collective import fused_ar_rmsnorm as far  # noqa: E402


def test_measured_dispatch_constants():
    assert far.TWOSHOT_MIN == 48 and far.MAX_T == 1024
    assert far.mode_for(47) == 1
    assert far.mode_for(48) == 2
    assert far.mode_for(1024) == 2


def test_init_requires_eager_and_uninitialized_call_raises():
    with pytest.raises(RuntimeError, match="init"):
        x = torch.zeros(4, 8)
        far.ar_residual_rmsnorm(None, x, x, x[0], 1e-6, x.clone(), x.clone(), None)


def test_no_sglang_or_harness_imports():
    assert "cacheon_kernels.collective.fused_ar_rmsnorm" in sys.modules
    src = open(far.__file__).read()
    assert "import sglang" not in src and "from sglang" not in src
    assert "from cacheon." not in src and "import cacheon." not in src.replace("cacheon_kernels", "")

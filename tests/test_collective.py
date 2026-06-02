"""Distributed verification of the collective.all_reduce slot (CPU / gloo, 2 ranks).

Spawns 2 gloo ranks, runs the example all-reduce, and checks each rank's output equals
the trusted fp32 cross-rank sum. No GPU needed; torch-only (skipped where torch absent).
gloo has no bf16, so verify_collective uses fp32 on the CPU path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.verify_collective import verify_collective  # noqa: E402

ALLREDUCE_BUNDLE = "examples/miner_allreduce_torch/kernels/all_reduce.py"


def test_collective_kind_discriminator():
    assert get_slot("collective.all_reduce").kind == "collective"


def test_allreduce_faithful_passes_gloo_cpu():
    slot = get_slot("collective.all_reduce")
    res = verify_collective(slot, ALLREDUCE_BUNDLE, "all_reduce",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert res.passed, "\n".join(f"{r.shape}: {r.detail}" for r in res.shape_results)


def test_non_reducing_kernel_fails_gloo_cpu(tmp_path):
    # A "reduce" that returns the LOCAL partial (forgets to sum across ranks) must fail:
    # out = x_rank != sum_r(x_r). Distributed verify is what catches this — a single-rank
    # check never would.
    broken = tmp_path / "broken_allreduce.py"
    broken.write_text("def all_reduce(x, out, group=None):\n    out.copy_(x)  # BUG: no cross-rank sum\n")
    slot = get_slot("collective.all_reduce")
    res = verify_collective(slot, str(broken), "all_reduce",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert not res.passed


def test_verify_entry_rejects_collective():
    # Collective slots must be verified distributed, not via the single-process verify_entry.
    from optima.verify import verify_entry

    slot = get_slot("collective.all_reduce")
    with pytest.raises(ValueError, match="collective"):
        verify_entry(slot, lambda *a, **k: None, device="cpu")

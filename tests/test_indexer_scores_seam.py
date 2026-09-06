"""Pinned ragged/paged score mapping, API preservation and original restoration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cacheon import slots
from cacheon.indexer_scores_contract import SLOT, reference, slot_spec
from cacheon.integrations import sglang_indexer_scores as seam
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry
from cacheon.sandbox import load_entry

ENTRY = load_entry(str(Path(__file__).resolve().parents[1] /
                       "examples/miner_indexer_scores_torch/kernels/indexer_scores.py"), "indexer_scores")


def calls(paged, next_n=1):
    """Construct SoA cache bytes directly, independently of adapter view logic."""
    q = torch.ones(2 * next_n, 8, 128).to(torch.float8_e4m3fn)
    weights = torch.ones(2 * next_n, 8)
    if not paged:
        keys = torch.arange(128, dtype=torch.float32).repeat(5, 1).to(torch.float8_e4m3fn)
        return (q, (keys, torch.arange(1, 6, dtype=torch.float32)), weights,
                torch.tensor([0, 2], dtype=torch.int32), torch.tensor([3, 5], dtype=torch.int32))
    raw = torch.zeros(3, 64 * 132, dtype=torch.uint8)
    for page in range(3):
        raw[page, :64 * 128] = torch.full((64 * 128,), page + 1.).to(torch.float8_e4m3fn).view(torch.uint8)
        raw[page, 64 * 128:] = torch.arange(1, 65, dtype=torch.float32).view(torch.uint8)
    return (q.reshape(2, next_n, 8, 128), raw.view(3, 64, 1, 132), weights,
            torch.arange(1, 2 * next_n + 1, dtype=torch.int32).reshape(2, next_n) * 17,
            torch.tensor([[2, 0], [1, 2]], dtype=torch.int32), torch.zeros(5, 2, dtype=torch.int32), 128)


@pytest.fixture
def active(monkeypatch):
    recorded = []
    monkeypatch.setitem(slots.SLOTS, SLOT, slot_spec())
    monkeypatch.setenv("CACHEON_INDEXER_SCORES_SEAM", "1")
    monkeypatch.setattr(seam, "_dynamo_compiling", lambda: False)
    monkeypatch.setattr(seam, "_flashinfer_tuning", lambda: False)
    monkeypatch.setattr(seam, "_runtime_parallel_sizes", lambda: (4, 4))
    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: False))
    monkeypatch.setattr(seam, "_receipts", SimpleNamespace(is_invoking=lambda: False, invoke=lambda s, e, *a: e(*a), completed=recorded.append))
    return recorded


@pytest.mark.parametrize("paged,next_n", [(False, 1), (True, 1), (True, 2)])
def test_pinned_shapes_share_math_and_candidate_failure_is_not_retried(active, paged, next_n):
    args = calls(paged, next_n)
    normalize = seam._paged_inputs if paged else seam._ragged_inputs
    inputs = normalize(*args, clean_logits=False)
    if paged:
        assert inputs["key_pages"].untyped_storage().data_ptr() == args[1].untyped_storage().data_ptr()
        assert inputs["key_scales"].tolist() == [list(range(1, 65))] * 3
        assert inputs["key_pages"].float()[:, 0, 0].tolist() == [1., 2., 3.]
    expected = reference(inputs)[0]

    def stock(*a, **kw):
        active.append("stock")
        return torch.full_like(expected, -torch.inf)

    registry = KernelRegistry()
    registry.register(KernelImpl(slot=SLOT, bundle_id="scores", entry=ENTRY,
                                 eligibility=Eligibility(dtypes=frozenset({"float8_e4m3fn"}), quant=frozenset({"fp8_e4m3"}))))
    registry.enable()
    dispatch = seam._make_dispatch(stock, registry, paged=paged)
    assert torch.equal(dispatch(*args, clean_logits=False), expected)
    assert active == [SLOT]
    assert torch.isneginf(dispatch(*args, clean_logits=True)).all()
    assert active == [SLOT, "stock"]

    def broken(*a):
        raise RuntimeError("score candidate failed")

    registry = KernelRegistry()
    registry.register(KernelImpl(slot=SLOT, bundle_id="broken", entry=broken,
                                 eligibility=Eligibility(dtypes=frozenset({"float8_e4m3fn"}), quant=frozenset({"fp8_e4m3"}))))
    registry.enable()
    with pytest.raises(RuntimeError, match="score candidate failed"):
        seam._make_dispatch(stock, registry, paged=paged)(*args, clean_logits=False)
    assert active == [SLOT, "stock"]


def test_install_is_idempotent_and_restores_exact_originals(monkeypatch):
    originals = {name: lambda *a, **kw: None for name in seam._FUNCTIONS}
    module = SimpleNamespace(**originals)
    monkeypatch.setitem(seam.sys.modules, "deep_gemm", module)
    seam.install()
    installed = module.fp8_mqa_logits
    seam.install()
    assert seam.is_installed()
    assert module.fp8_mqa_logits is installed
    seam.uninstall()
    assert not seam.is_installed()
    assert all(getattr(module, name) is fn for name, fn in originals.items())


def test_sampled_audit_normalizes_only_unwritten_stock_cells(active, monkeypatch):
    args = calls(False)
    expected = reference(seam._ragged_inputs(*args, clean_logits=False))[0]
    stock_output = expected.clone()
    stock_output[0, 3:] = torch.nan
    stock_output[1, :2] = -torch.inf
    audited = []

    def audit(slot, outputs, baseline):
        audited.append((slot, outputs[0].clone(), baseline()))

    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: True, run=audit))
    registry = KernelRegistry()
    registry.register(KernelImpl(slot=SLOT, bundle_id="audited", entry=ENTRY,
                                 eligibility=Eligibility(quant=frozenset({"fp8_e4m3"}))))
    registry.enable()
    actual = seam._make_dispatch(lambda *a, **kw: stock_output, registry, paged=False)(*args, clean_logits=False)
    assert torch.equal(actual, expected)
    assert len(audited) == 1 and audited[0][0] == SLOT
    assert torch.equal(audited[0][1], audited[0][2])
    assert torch.isnan(stock_output[0, 3:]).all()

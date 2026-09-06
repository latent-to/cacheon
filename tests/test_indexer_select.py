"""Merged selection math, changing graph inputs and exact Indexer consumer binding."""

from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cacheon import slots
from cacheon.indexer_select_contract import DYNAMIC_INPUTS, SLOT, invoke_entry, reference, slot_spec
from cacheon.integrations import sglang_indexer_select as seam
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry
from cacheon.sandbox import load_entry
from cacheon.verify import verify_entry
from support.graph_backend import FakeGraphBackend

ENTRY = load_entry(str(Path(__file__).resolve().parents[1] /
                       "examples/miner_indexer_select_torch/kernels/indexer_select.py"), "indexer_select")
# The copied producer resolves this symbol from its isolated globals.
fused_q_indexer_rope_first_quant = None


def test_signed_heads_forced_tokens_empty_padding_and_ties():
    inputs = dict(q=torch.tensor([[[1., -1.], [-1., 1.]]] * 2).to(torch.float8_e4m3fn),
                  key_pages=torch.tensor([[[1., 0.], [0., 1.]], [[2., 1.], [1., 3.]]]).to(torch.float8_e4m3fn),
                  key_scales=torch.tensor([[2., 3.], [4., .5]]), weights=torch.tensor([[2., -3.]] * 2),
                  page_table=torch.tensor([[1, 0]], dtype=torch.int32), row_to_batch=torch.zeros(2, dtype=torch.int32),
                  lengths=torch.tensor([4, 0], dtype=torch.int32), page_offsets=torch.zeros(2, dtype=torch.int32),
                  positions=None, cos_sin_cache=None, q_scale_gate=1.0, top_k=6, num_init_tokens=1, num_local_tokens=1)
    for initial, local, tied, expected in ((1, 1, False, [2, 1, 0, 3, -1, -1]),
                                          (0, 0, False, [2, 0, 3, 1, -1, -1]),
                                          (0, 0, True, [2, 3, 0, 1, -1, -1])):
        inputs.update(num_init_tokens=initial, num_local_tokens=local)
        if tied:
            inputs["weights"].zero_()
        wanted = torch.tensor([expected, [-1] * 6], dtype=torch.int32)
        assert torch.equal(reference(inputs)[0], wanted)
        out = torch.full_like(wanted, 123)
        invoke_entry(ENTRY, inputs, [out])
        assert torch.equal(out, wanted)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16, torch.float16])
def test_two_geometries_and_graph_refresh(dtype):
    result = verify_entry(slot_spec(), ENTRY, dtype=dtype, device="cpu", seed=7,
                          graph_replays=3, _graph_backend=FakeGraphBackend())
    assert result.passed, result
    assert len(result.shape_results) == 2 and all(r.graph_replays == 3 for r in result.shape_results)


def test_changed_producer_call_structure_fails_before_binding():
    def changed(q, weights, gate, cache, positions):
        return fused_q_indexer_rope_first_quant(q, weights, gate, cache, positions)

    with pytest.raises(RuntimeError, match="expected two calls, got 1"):
        seam._bind_prepare(changed, SimpleNamespace())


@pytest.mark.parametrize("name,dtype", [(n, torch.float8_e4m3fn) for n in DYNAMIC_INPUTS[:8]] +
                                      [(n, torch.bfloat16) for n in DYNAMIC_INPUTS])
def test_stale_graph_input_is_rejected(name, dtype):
    backend, captured = FakeGraphBackend(), None

    def stale(*args):
        nonlocal captured
        args = list(args)
        index = DYNAMIC_INPUTS.index(name)
        if backend.phase == "capture":
            captured = args[index].clone()
        if backend.phase == "replay":
            args[index] = captured
        ENTRY(*args)

    assert not verify_entry(slot_spec(), stale, dtype=dtype, device="cpu", seed=7,
                            graph_replays=3, _graph_backend=backend).passed


@pytest.mark.parametrize("bad", ["input_mutation", "no_write"])
def test_bad_candidate_is_rejected(bad):
    def broken(*args):
        if bad == "input_mutation":
            ENTRY(*args)
            args[1].zero_()

    assert not verify_entry(slot_spec(), broken, dtype=torch.float8_e4m3fn, device="cpu", seed=7,
                            graph_replays=3, _graph_backend=FakeGraphBackend()).passed


@pytest.mark.parametrize("ragged,supplied", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("raw_query", [False, True])
def test_actual_indexer_class_mapping_output_identity_exception_and_restore(monkeypatch, ragged, supplied, raw_query):
    spec = slot_spec()
    full = spec.make_inputs(num_tokens=6 if ragged else 4, num_heads=2, head_dim=8,
                            kv_len=43, page_size=32, top_k=3, dtype=torch.bfloat16 if raw_query else torch.float8_e4m3fn, device="cpu", seed=5)
    count = 4 if ragged else 2
    batches = torch.tensor([0, 0, 1, 1] if ragged else [0, 1], dtype=torch.int32)
    offsets = torch.tensor([1, 1, 2, 2] if ragged else [0, 0], dtype=torch.int32)
    lengths = torch.tensor([3, 5, 4, 6] if ragged else [3, 4], dtype=torch.int32)
    cu_k, cu_q = torch.tensor([0, 64, 128]), torch.tensor([0, count // 2, count], dtype=torch.int32)
    logical = dict(full, q=full["q"][:count], weights=full["weights"][:count],
                   row_to_batch=batches, page_offsets=offsets, lengths=lengths,
                   positions=full["positions"][:count] if raw_query else None)
    expected = reference(logical)[0]
    raw = torch.cat((full["key_pages"].view(torch.uint8).flatten(1), full["key_scales"].view(torch.uint8)), 1)
    metadata = SimpleNamespace(topk_transform_method="PAGED", attn_metadata=SimpleNamespace(cu_seqlens_k=cu_k, cu_seqlens_q=cu_q),
        get_indexer_kvcache_range=lambda: (cu_k[batches] + offsets, cu_k[batches] + offsets + lengths),
        get_dsa_extend_len_cpu=lambda: [count // 2] * 2, get_seqlens_expanded=lambda: lengths,
        get_page_table_64=lambda: full["page_table"])

    def mapping(**kwargs):
        assert kwargs["num_rows"] == count and kwargs["cu_seqlens_q_topk"] is cu_q
        return (batches, kwargs["row_starts"] - cu_k[batches]) if ragged else (None, None)

    class Indexer:
        num_init_tokens = num_local_tokens = 1
        index_topk = 3
        _get_index_k_read_buffer = staticmethod(lambda pool, layer: raw)

        def _get_topk_paged(self, forward_batch, layer_id, q_fp8, weights, metadata):
            raise RuntimeError("original selection")

        def _get_topk_ragged(self, enable_dual_stream, forward_batch, layer_id, q_fp8, weights, metadata, topk_result=None):
            raise RuntimeError("original selection")

        def _fused_q_prepare_and_store(self, q, weights, gate, cache, positions, *, second=False):
            if second:
                return fused_q_indexer_rope_first_quant(q, weights, gate, cache, positions)
            return fused_q_indexer_rope_first_quant(q, weights, gate, cache, positions)

    native_calls = []

    def native(q, weights, gate, cache, positions):
        native_calls.append(positions)
        return q.to(torch.float8_e4m3fn), weights.float().unsqueeze(-1)

    module = SimpleNamespace(Indexer=Indexer, get_token_to_kv_pool=lambda: SimpleNamespace(page_size=32),
                             fused_q_indexer_rope_first_quant=native)
    originals = {n: getattr(Indexer, n) for n in (*seam._FUNCTIONS, seam._PREPARE)}
    monkeypatch.setitem(seam.sys.modules, seam._MODULE, module)
    monkeypatch.setitem(seam.sys.modules, "sglang.srt.layers.attention.dsa.dsa_topk_backend",
                        SimpleNamespace(TopkTransformMethod=SimpleNamespace(PAGED="PAGED"), _build_flashinfer_paged_args=mapping))
    monkeypatch.setitem(slots.SLOTS, SLOT, spec)
    monkeypatch.setenv("CACHEON_INDEXER_SELECT_SEAM", "1")
    monkeypatch.setattr(seam, "_dynamo_compiling", lambda: False)
    monkeypatch.setattr(seam, "_flashinfer_tuning", lambda: False)
    monkeypatch.setattr(seam, "_runtime_parallel_sizes", lambda: (4, 4))
    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: False))
    events = []

    def candidate(*args):
        assert args[1].untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert args[2].untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        ENTRY(*args)

    registry = KernelRegistry()
    registry.register(KernelImpl(slot=SLOT, bundle_id="select", entry=candidate,
                                 eligibility=Eligibility(quant=frozenset({"fp8_e4m3"}))))
    registry.enable()
    monkeypatch.setattr(seam, "_receipts", SimpleNamespace(is_invoking=lambda: False, invoke=lambda s, e, *a: e(*a), completed=events.append))
    seam.install(registry)
    installed = Indexer._get_topk_paged
    seam.install(registry)
    assert seam.is_installed() and installed is Indexer._get_topk_paged
    assert Indexer._fused_q_prepare_and_store.__code__ is originals[seam._PREPARE].__code__
    q, weights = full["q"], full["weights"].unsqueeze(-1)
    if raw_query:
        producer = Indexer()._fused_q_prepare_and_store
        q, weights = producer(q, full["weights"], full["q_scale_gate"], full["cos_sin_cache"], full["positions"], second=ragged)
        other_positions = full["positions"] + 1
        _, other = producer(q, full["weights"], full["q_scale_gate"], full["cos_sin_cache"], other_positions)
        assert weights[1] is full["positions"] and other[1] is other_positions
    common = (SimpleNamespace(attn_cp_metadata=None), 7, q, weights, metadata)
    args = ((True,) + common) if ragged else common
    destination = torch.full((full["q"].shape[0], 3), -7, dtype=torch.int32) if supplied else None
    kwargs = {"topk_result": destination} if supplied else {}
    method = Indexer()._get_topk_ragged if ragged else Indexer()._get_topk_paged
    result = method(*args, **kwargs)
    assert torch.equal(result[:count], expected) and (result[count:] == (-7 if supplied else -1)).all()
    assert (not supplied or result is destination) and events == [SLOT] and not native_calls
    audits = []

    @wraps(originals[seam._FUNCTIONS[1 if ragged else 0]])
    def audited_stock(*a, **kw):
        stock = torch.full((full["q"].shape[0], 3), -1, dtype=torch.int32)
        stock[:count] = expected
        return stock

    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: True,
                        run=lambda slot, actual, baseline: audits.append(torch.equal(actual[0], baseline()))))
    seam._make_dispatch(audited_stock, registry, module, ragged=ragged)(Indexer(), *args, **kwargs)
    assert audits == [True]
    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: False))
    monkeypatch.setattr(seam._receipts, "is_invoking", lambda: True)
    with pytest.raises(RuntimeError, match="original selection"):
        method(*args, **kwargs)
    monkeypatch.setattr(seam._receipts, "is_invoking", lambda: False)
    registry.disable()
    with pytest.raises(RuntimeError, match="original selection"):
        method(*args, **kwargs)
    registry.enable()
    native_before = len(native_calls)
    assert native_before == (3 if raw_query else 0)
    monkeypatch.setattr(seam._receipts, "invoke", lambda *a: (_ for _ in ()).throw(RuntimeError("candidate failure")))
    with pytest.raises(RuntimeError, match="candidate failure"):
        method(*args, **kwargs)
    assert events == [SLOT, SLOT] and len(native_calls) == native_before
    seam.uninstall()
    assert not seam.is_installed() and all(getattr(Indexer, n) is f for n, f in originals.items())

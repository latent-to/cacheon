"""CPU adapter routing at the pinned SGLang/FlashInfer call boundary."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cacheon.integrations import sglang_sparse_mla as seam  # noqa: E402
from cacheon.model_profiles import verification_call_descriptor  # noqa: E402
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from cacheon.seams import SEAM_ADAPTERS, seam_binding_environment  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402
from cacheon.sparse_mla_contract import DYNAMIC_INPUTS, SLOT, reference  # noqa: E402

_STOCK_CALLS = []


def _stock(
    query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank,
    qk_rope_head_dim, block_tables, seq_lens, max_seq_len, sparse_mla_top_k=0,
    out=None, bmm1_scale=1.0, bmm2_scale=1.0, sinks=None,
    skip_softmax_threshold_scale_factor=None, enable_pdl=None, backend="auto",
    is_var_seq=True, uses_shared_paged_kv_idx=True, lse=None, return_lse=False,
    cute_dsl_impl="auto", kv_scale_format="auto", cum_seq_lens_q=None,
    max_q_len=None, multi_ctas_kv_counter_buffer=None, sparse_mla_top_k_lens=None,
    enable_dcp=False, cp_world=1, cp_rank=0, causal_seqlens_kv_global=None,
):
    _STOCK_CALLS.append(query.shape)
    return torch.full((*query.shape[:-1], kv_lora_rank), -7, dtype=torch.bfloat16)


@pytest.fixture()
def runtime(monkeypatch):
    _STOCK_CALLS.clear()
    module = ModuleType(seam._MODULE)
    setattr(module, seam._FUNCTION, _stock)
    monkeypatch.setitem(sys.modules, seam._MODULE, module)
    monkeypatch.setenv("CACHEON_SPARSE_MLA_SEAM", "1")
    monkeypatch.setattr(seam, "_dynamo_compiling", lambda: False)
    monkeypatch.setattr(seam, "_flashinfer_tuning", lambda: False)
    monkeypatch.setattr(seam, "_in_cuda_graph", lambda: False)
    monkeypatch.setattr(seam, "_runtime_parallel_sizes", lambda: (4, 4))
    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: False))
    calls, completed = [], []

    def invoke(slot, entry, *args):
        calls.append(slot)
        return entry(*args)

    monkeypatch.setattr(seam, "_receipts", SimpleNamespace(
        invoke=invoke, completed=completed.append,
    ))
    yield module, calls, completed
    seam.uninstall()


def _registry(entry):
    registry = KernelRegistry()
    registry.register(KernelImpl(
        slot=SLOT, bundle_id="sparse-test", entry=entry,
        eligibility=Eligibility(
            dtypes=frozenset({"float32", "float16", "bfloat16", "float8_e4m3fn"}),
            quant=frozenset({"dense", "fp8_e4m3"}),
        ),
    ))
    registry.enable()
    return registry


def _faithful(q, cache, indices, lengths, out, value_dim, qk_scale, value_scale):
    out.copy_(reference(dict(
        q=q, kv_cache=cache, indices=indices, seq_lens=lengths,
        value_dim=value_dim, qk_scale=qk_scale, value_scale=value_scale,
    ))[0])


def _call(shape_index=0, input_dtype=None):
    slot = get_slot(SLOT)
    inputs = slot.make_inputs(
        **slot.shapes[shape_index], dtype=torch.float32, input_dtype=input_dtype,
        device="cpu", seed=13,
    )
    q, cache = inputs["q"], inputs["kv_cache"]
    return inputs, dict(
        query=q.unsqueeze(1), kv_cache=cache.unsqueeze(1),
        workspace_buffer=torch.zeros(8, dtype=torch.uint8), qk_nope_head_dim=8,
        kv_lora_rank=inputs["value_dim"], qk_rope_head_dim=q.shape[-1]-inputs["value_dim"],
        block_tables=inputs["indices"].unsqueeze(1), seq_lens=inputs["seq_lens"],
        max_seq_len=cache.shape[0]*cache.shape[1],
        sparse_mla_top_k=inputs["indices"].shape[1], backend="trtllm-gen",
        bmm1_scale=inputs["qk_scale"], bmm2_scale=inputs["value_scale"],
    )


@pytest.mark.parametrize("profile", [0, 1])
@pytest.mark.parametrize("input_dtype", [None, "float8_e4m3fn"])
def test_shared_core_dispatch_output_storage_and_offline_descriptor(runtime, profile, input_dtype):
    module, calls, completed = runtime
    registry = _registry(_faithful)
    descriptors = []
    select = registry.select

    def record(slot, descriptor):
        descriptors.append(descriptor)
        return select(slot, descriptor)

    registry.select = record
    seam.install(registry)
    seam.install(registry)
    inputs, kwargs = _call(profile, input_dtype)
    actual = getattr(module, seam._FUNCTION)(**kwargs)
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (
        inputs["q"].shape[0], 1, inputs["q"].shape[1], inputs["value_dim"],
    )
    torch.testing.assert_close(actual[:, 0], reference(inputs)[0])
    assert all(actual.data_ptr() != inputs[n].data_ptr() for n in DYNAMIC_INPUTS)
    assert calls == completed == [SLOT]
    assert not _STOCK_CALLS
    assert descriptors[0] == verification_call_descriptor(
        get_slot(SLOT), inputs, dtype_name="float32", architecture=None,
        graph_mode="eager", tp_size=4, world_size=4,
    )
    seam.uninstall()
    assert getattr(module, seam._FUNCTION) is _stock
    assert not seam.is_installed()


@pytest.mark.parametrize("change", [
    {"backend": "auto"}, {"sparse_mla_top_k": 0}, {"return_lse": True},
    {"sinks": []}, {"is_var_seq": False}, {"uses_shared_paged_kv_idx": False},
    {"bmm1_scale": torch.tensor(1.)}, {"enable_dcp": True},
    {"cum_seq_lens_q": torch.tensor([0, 4], dtype=torch.int32)},
    {"sparse_mla_top_k_lens": torch.ones(4, dtype=torch.int32)},
    {"out": torch.empty(4, 1, 2, 8, dtype=torch.bfloat16)},
])
def test_unsupported_domains_remain_preselection_stock(runtime, change):
    module, calls, completed = runtime
    seam.install(_registry(lambda *args: pytest.fail("out-of-domain candidate")))
    _, kwargs = _call()
    kwargs.update(change)
    getattr(module, seam._FUNCTION)(**kwargs)
    assert len(_STOCK_CALLS) == 1
    assert calls == completed == []


@pytest.mark.parametrize("reason", ["gate", "compile", "autotune", "disabled_registry"])
def test_inactive_or_runtime_tuning_never_counts_candidate_work(runtime, monkeypatch, reason):
    module, calls, completed = runtime
    registry = _registry(lambda *args: pytest.fail("inactive candidate"))
    if reason == "gate":
        monkeypatch.setenv("CACHEON_SPARSE_MLA_SEAM", "0")
    elif reason == "compile":
        monkeypatch.setattr(seam, "_dynamo_compiling", lambda: True)
    elif reason == "autotune":
        monkeypatch.setattr(seam, "_flashinfer_tuning", lambda: True)
    else:
        registry.disable()
    seam.install(registry)
    _, kwargs = _call()
    getattr(module, seam._FUNCTION)(**kwargs)
    assert len(_STOCK_CALLS) == 1 and calls == completed == []


@pytest.mark.parametrize("corruption", ["raise", "storage", "shape", "input_storage"])
def test_selected_candidate_failure_is_terminal_and_not_completed(runtime, corruption):
    module, calls, completed = runtime

    def broken(q, cache, indices, lengths, out, *rest):
        if corruption == "raise":
            raise RuntimeError("candidate exploded")
        if corruption == "storage":
            out.set_(out.clone())
        elif corruption == "input_storage":
            q.set_(q.clone())
        else:
            out.resize_(1)

    seam.install(_registry(broken))
    _, kwargs = _call()
    with pytest.raises((ValueError, RuntimeError)):
        getattr(module, seam._FUNCTION)(**kwargs)
    assert calls == [SLOT] and completed == []
    assert not _STOCK_CALLS


def test_sampled_audit_retains_stock_output_before_candidate_mutates_inputs(runtime, monkeypatch):
    module, calls, completed = runtime
    events = []

    def entry(q, cache, indices, lengths, out, *rest):
        q.zero_()
        out.fill_(-7)
        events.append("candidate")

    def audit(slot, outputs, baseline):
        events.append("audit")
        assert len(_STOCK_CALLS) == 1
        assert torch.equal(outputs[0], baseline())

    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: True, run=audit))
    seam.install(_registry(entry))
    _, kwargs = _call()
    getattr(module, seam._FUNCTION)(**kwargs)
    assert events == ["candidate", "audit"]
    assert calls == completed == [SLOT]


def test_single_seam_registry_owns_bootstrap_and_activation_gate():
    row, = [row for row in SEAM_ADAPTERS if row.name == "sparse_mla"]
    assert row.target_module == seam._MODULE and row.chokepoint == seam._FUNCTION
    assert row.slots == (SLOT,)
    assert seam_binding_environment(())["CACHEON_SPARSE_MLA_SEAM"] == "0"
    assert seam_binding_environment(("sparse_mla",))["CACHEON_SPARSE_MLA_SEAM"] == "1"

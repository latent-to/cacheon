"""Producer binding retains engine K writes and leaves vendor libraries callable."""

import ast
import inspect
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
import torch

from cacheon import receipts
from cacheon.integrations import sglang_sparse_mla as seam
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry
from cacheon.seams import SEAM_ADAPTERS, seam_binding_environment
from cacheon.slots import get_slot
from cacheon.sparse_mla_contract import SLOT, reference


def mla_quantize_and_rope_for_fp8(q, qr, k, kr, positions, cache, neox, value_dim, rope_dim):
    """Public fixture for the engine's joint Q/K producer; numerical oracle is separate."""
    import flashinfer.rope
    qr, kr, q, k = flashinfer.rope.mla_rope_quantize_fp8(q_rope=qr, k_rope=kr, q_nope=q, k_nope=k)
    return torch.cat((q.float(), qr.float()), -1).to(torch.float8_e4m3fn), k, kr


class _Producer:
    """Minimal public source retaining the pinned three expressions and engine write."""
    def _forward_trtllm(self, q, k, v, layer, forward_batch, seq_lens, save_kv_cache=True,
                       q_rope=None, k_rope=None, topk_indices=None, cos_sin_cache=None,
                       is_neox=False, llama_4_scaling=None, is_prefill=False):
        """Prepare and write K before the selected attention implementation."""
        import flashinfer.decode
        rope_positions = forward_batch.positions
        q, k, k_rope = mla_quantize_and_rope_for_fp8(
            q, q_rope, k.squeeze(1), k_rope.squeeze(1), rope_positions,
            cos_sin_cache, is_neox, self.kv_lora_rank, self.qk_rope_head_dim)
        if save_kv_cache:
            self.token_to_kv_pool.set_mla_kv_buffer(layer, forward_batch.out_cache_loc, k, k_rope)
        kv = self.token_to_kv_pool.get_key_buffer(layer.layer_id).unsqueeze(1)
        q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        q = q_all.view(q_all.shape[0], 1, q_all.shape[1], q_all.shape[2])
        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=kv, block_tables=topk_indices[:, None], seq_lens=seq_lens,
            kv_lora_rank=self.kv_lora_rank, bmm1_scale=0.5)
        return out


@pytest.fixture()
def runtime(monkeypatch):
    events, stock_queries = [], []
    def vendor(**call):
        events.append(("prepare", call["q_nope"].shape[1]))
        return tuple(call[k].to(torch.float8_e4m3fn) for k in ("q_rope", "k_rope", "q_nope", "k_nope"))
    def core(**call):
        events.append("stock")
        stock_queries.append(call["query"].clone())
        return torch.full((*call["query"].shape[:-1], call["kv_lora_rank"]), -7, dtype=torch.bfloat16)
    vendor_module = ModuleType("flashinfer")
    vendor_module.rope = SimpleNamespace(mla_rope_quantize_fp8=vendor)
    vendor_module.decode = SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=core)
    for name, module in (("flashinfer", vendor_module), ("flashinfer.rope", vendor_module.rope), ("flashinfer.decode", vendor_module.decode)):
        monkeypatch.setitem(sys.modules, name, module)
    module = SimpleNamespace(DeepseekSparseAttnBackend=_Producer, mla_quantize_and_rope_for_fp8=mla_quantize_and_rope_for_fp8,
        dsa_use_prefill_cp=lambda batch: getattr(batch, "cp", False),
        envs=SimpleNamespace(SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR=SimpleNamespace(get=lambda: None)))
    monkeypatch.setitem(sys.modules, seam._MODULE, module)
    monkeypatch.setenv("CACHEON_SPARSE_MLA_SEAM", "1")
    monkeypatch.setattr(seam, "_in_cuda_graph", lambda: False)
    monkeypatch.setattr(seam, "_runtime_parallel_sizes", lambda: (4, 4))
    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: False))
    yield SimpleNamespace(module=module, vendor=vendor_module, events=events, stock_queries=stock_queries)
    seam.uninstall()


def _registry(entry):
    registry = KernelRegistry()
    registry.register(KernelImpl(slot=SLOT, bundle_id="sparse-test", entry=entry,
                                 eligibility=Eligibility(quant=frozenset({"fp8_e4m3"}))))
    registry.enable()
    return registry


def _call(runtime, profile=0):
    inputs = get_slot(SLOT).make_inputs(**get_slot(SLOT).shapes[profile], dtype=torch.bfloat16, device="cpu", seed=13)
    inputs["qk_scale"] = 0.5
    q, cache = inputs["q"], inputs["kv_cache"]
    def write(layer, loc, k, kr):
        runtime.events.append("write")
        cache.view(torch.uint8).flatten(0, 1)[loc] = torch.cat((k.float(), kr.float()), -1).to(cache.dtype).view(torch.uint8)
    backend = SimpleNamespace(real_page_size=cache.shape[1], kv_cache_dim=cache.shape[2],
        kv_lora_rank=inputs["value_dim"], qk_rope_head_dim=inputs["q_rope"].shape[-1], dsa_index_topk=inputs["indices"].shape[-1],
        token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer: cache, set_mla_kv_buffer=write))
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=q.shape[1], head_dim=cache.shape[-1])
    batch = SimpleNamespace(positions=inputs["positions"], out_cache_loc=torch.arange(q.shape[0]))
    kwargs = dict(q=q, k=torch.ones(q.shape[0], 1, q.shape[-1]), v=None, layer=layer,
        forward_batch=batch, seq_lens=inputs["seq_lens"], q_rope=inputs["q_rope"],
        k_rope=torch.ones(q.shape[0], 1, inputs["q_rope"].shape[-1]), topk_indices=inputs["indices"],
        cos_sin_cache=inputs["cos_sin_cache"], is_neox=inputs["is_neox"], is_prefill=bool(profile))
    return inputs, lambda: _Producer._forward_trtllm(backend, **kwargs)


@pytest.mark.parametrize("profile", [0, 1])
def test_raw_query_storage_engine_write_order_and_uninstall(runtime, profile):
    inputs, call = _call(runtime, profile)
    outputs = []
    def entry(*args):
        assert runtime.events == [("prepare", 0), "write"]
        assert args[0].data_ptr() == inputs["q"].data_ptr() and args[1] is inputs["q_rope"]
        outputs.append(args[8])
        args[8].copy_(reference(inputs)[0])
    original = _Producer._forward_trtllm
    seam.install(_registry(entry))
    seam.install()
    actual = call()
    assert actual.data_ptr() == outputs[0].data_ptr() != inputs["q"].data_ptr()
    assert actual.shape == (*inputs["q"].shape[:1], 1, inputs["q"].shape[1], inputs["value_dim"])
    torch.testing.assert_close(actual[:, 0], reference(inputs)[0])
    assert seam.is_installed() and not runtime.stock_queries
    seam.uninstall()
    assert _Producer._forward_trtllm is original and not seam.is_installed()


@pytest.mark.parametrize("reason", ["gate", "disabled", "nested"])
def test_inactive_selection_retains_joint_preparation(runtime, monkeypatch, reason):
    inputs, call = _call(runtime)
    registry = _registry(lambda *args: pytest.fail("inactive candidate"))
    if reason == "gate":
        monkeypatch.setenv("CACHEON_SPARSE_MLA_SEAM", "0")
    elif reason == "disabled":
        registry.disable()
    else:
        monkeypatch.setattr(seam._receipts, "is_invoking", lambda: True)
    seam.install(registry)
    assert (call() == -7).all()
    assert runtime.events == [("prepare", inputs["q"].shape[1]), "write", "stock"]


@pytest.mark.parametrize("corruption", ["raise", "storage", "shape", "input_storage"])
def test_selected_failure_is_terminal(runtime, corruption):
    _, call = _call(runtime)
    def broken(*args):
        if corruption == "raise":
            raise RuntimeError("candidate exploded")
        if corruption == "shape":
            args[8].resize_(1)
        else:
            tensor = args[0] if corruption == "input_storage" else args[8]
            tensor.set_(tensor.clone())
    seam.install(_registry(broken))
    with pytest.raises((ValueError, RuntimeError)):
        call()
    assert runtime.events == [("prepare", 0), "write"]
    assert not receipts.is_invoking()


def test_audit_snapshots_stock_before_candidate_and_vendor_api_does_not_recurse(runtime, monkeypatch):
    inputs, call = _call(runtime)
    core = runtime.vendor.decode.trtllm_batch_decode_with_kv_cache_mla
    def entry(*args):
        assert runtime.events == [("prepare", 0), "write", ("prepare", inputs["q"].shape[1]), "stock"]
        args[0].zero_()
        args[8].copy_(core(query=args[0][:, None], kv_lora_rank=args[9])[:, 0])
    def audit(slot, outputs, baseline):
        assert runtime.stock_queries[0].float().count_nonzero() > 0
        assert torch.equal(outputs[0], baseline()) and slot == SLOT
    monkeypatch.setattr(seam, "_audit", SimpleNamespace(sampled=lambda: True, run=audit))
    seam.install(_registry(entry))
    assert (call() == -7).all()
    assert runtime.vendor.decode.trtllm_batch_decode_with_kv_cache_mla is core
    assert len(runtime.stock_queries) == 2 and not receipts.is_invoking()


def test_binding_changes_only_three_existing_statements(runtime, monkeypatch):
    original, compiled = _Producer._forward_trtllm, []
    before = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    def record(tree, *args):
        compiled.append(tree.body[0])
        return compile(tree, *args)
    monkeypatch.setattr(seam, "compile", record, raising=False)
    seam._bind_method(original, _registry(lambda *a: None), runtime.module)
    after = compiled[0].body[:1] + compiled[0].body[3:]
    assert len(after) == len(before.body)
    assert sum(ast.dump(a) != ast.dump(b) for a, b in zip(before.body, after)) == 3
    row, = [row for row in SEAM_ADAPTERS if row.name == "sparse_mla"]
    assert row.target_module == seam._MODULE and row.chokepoint == f"{seam._CLASS}.{seam._FUNCTION}"
    assert row.slots == (SLOT,)
    assert seam_binding_environment(())["CACHEON_SPARSE_MLA_SEAM"] == "0"
    assert seam_binding_environment(("sparse_mla",))["CACHEON_SPARSE_MLA_SEAM"] == "1"


@pytest.mark.parametrize("old,new", [("q_rope=None", "missing_q_rope=None"),
    ("q, k, k_rope =", "q, k ="), ("layer.head_dim)", "layer.other_dim)"),
    ("flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(", "flashinfer.decode.other_call(")])
def test_changed_signature_or_structure_fails_before_install(runtime, monkeypatch, old, new):
    source = textwrap.dedent(inspect.getsource(_Producer._forward_trtllm))
    monkeypatch.setattr(seam.inspect, "getsource", lambda _: source.replace(old, new))
    with pytest.raises(RuntimeError, match="changed"):
        seam.install(_registry(lambda *a: None))
    assert not seam.is_installed()

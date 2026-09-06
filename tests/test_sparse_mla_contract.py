"""Sparse-MLA math and replay orchestration, without claiming CUDA evidence."""

import math
from pathlib import Path

import pytest
import torch

from cacheon.manifest import load_manifest
from cacheon.model_profiles import model_profile, verification_call_descriptor
from cacheon.sandbox import load_entry
from cacheon.slots import get_slot
from cacheon.sparse_mla_contract import DYNAMIC_INPUTS, SLOT, call_descriptor, invoke_entry, query_reference, reference
from cacheon.target_catalog import default_target_catalog, resolve_target
from cacheon.verify import verify_entry
from support.graph_backend import FakeGraphBackend

BUNDLE = Path(__file__).resolve().parents[1] / "examples/miner_sparse_mla_torch"
ENTRY = load_entry(str(BUNDLE / "kernels/sparse_mla.py"), "sparse_mla")
ARGUMENTS = {"q": 0, "q_rope": 1, "positions": 2, "cos_sin_cache": 3,
             "kv_cache": 5, "indices": 6, "seq_lens": 7}


def _case():
    q = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) / 25
    return dict(q=q[..., :2], q_rope=q[..., 2:], positions=torch.arange(3),
                cos_sin_cache=torch.tensor([[1., 0.]] * 3), is_neox=False,
                kv_cache=torch.arange(256, dtype=torch.float32).reshape(2, 32, 4) / 150,
                indices=torch.tensor([[35, 3, 9, -1, 63], [1, 4, 8, -1, 40], [-1] * 5], dtype=torch.int32),
                seq_lens=torch.tensor([3, 2, 0], dtype=torch.int32),
                value_dim=2, qk_scale=0.7, value_scale=1.5)


def _run(inputs):
    out = torch.full((*inputs["q"].shape[:2], inputs["value_dim"]), torch.nan, dtype=torch.bfloat16)
    invoke_entry(ENTRY, inputs, [out])
    return out


def _scalar_oracle(inputs):
    """Use scalar sums and literal identity RoPE, independently of query_reference."""
    rows = inputs["kv_cache"].flatten(0, 1).double().tolist()
    query = torch.cat((inputs["q"], inputs["q_rope"]), -1).to(torch.float8_e4m3fn).double().tolist()
    output = []
    for q, ids, length in zip(query, inputs["indices"].tolist(), inputs["seq_lens"].tolist()):
        selected, heads = [rows[i] for i in ids[:length] if i >= 0], []
        for head in q:
            if not selected:
                heads.append([0.] * inputs["value_dim"])
                continue
            scores = [math.fsum(a * b for a, b in zip(head, key)) * inputs["qk_scale"] for key in selected]
            weights = [math.exp(v - max(scores)) for v in scores]
            heads.append([math.fsum(w * key[c] for w, key in zip(weights, selected))
                          / math.fsum(weights) * inputs["value_scale"] for c in range(inputs["value_dim"])])
        output.append(heads)
    return torch.tensor(output, dtype=torch.bfloat16)


def test_physical_pages_masked_rows_and_both_scales_match_scalar_math():
    inputs = _case()
    expected = _scalar_oracle(inputs)
    torch.testing.assert_close(reference(inputs)[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(_run(inputs), expected, atol=0, rtol=0)
    assert expected[2].count_nonzero() == 0
    inputs["indices"][0, 0] = 64
    with pytest.raises(ValueError, match="outside cache"):
        reference(inputs)


def test_future_history_and_indices_outside_active_prefix_cannot_leak():
    inputs = _case()
    before, cache = _run(inputs), inputs["kv_cache"].flatten(0, 1)
    for row in set(range(cache.shape[0])) - {35, 3, 9, 1, 4}:
        cache[row].fill_(10000)
    inputs["indices"][0, 3:] = 999999
    inputs["indices"][1, 2:] = 999999
    torch.testing.assert_close(_run(inputs), before, atol=0, rtol=0)
    torch.testing.assert_close(reference(inputs)[0], before, atol=0, rtol=0)
    cache[9].add_(2)
    after = _run(inputs)
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1:], before[1:], atol=0, rtol=0)


@pytest.mark.parametrize("storage_dtype", ["float32", "float16", "bfloat16", "float8_e4m3fn"])
def test_two_profiles_have_independent_math_and_mutable_graph_inputs(storage_dtype):
    slot = get_slot(SLOT)
    for index, shape in enumerate(slot.shapes):
        def generate(seed):
            return slot.make_inputs(**shape, dtype=torch.float32, input_dtype=storage_dtype,
                                    is_neox=bool(index), device="cpu", seed=seed)
        old = generate(7)
        pointers = {name: old[name].data_ptr() for name in DYNAMIC_INPUTS}
        for seed in (11, 19, 23):
            fresh = generate(seed)
            for name in DYNAMIC_INPUTS:
                assert not torch.equal(old[name].float(), fresh[name].float()), name
                old[name].copy_(fresh[name])
                assert old[name].data_ptr() == pointers[name]
            out = _run(old)
            torch.testing.assert_close(out, reference(old)[0], atol=0.02, rtol=0.02)
            assert out.dtype == torch.bfloat16


def test_ordinary_verifier_accepts_faithful_and_rejects_wrong_output():
    result = verify_entry(get_slot(SLOT), ENTRY, dtype=torch.float32, device="cpu", seed=7)
    assert result.passed and result.graph_required and not result.graph_verified
    assert not result.fully_verified
    def wrong(*args):
        args[8].zero_()
    assert not verify_entry(get_slot(SLOT), wrong, dtype=torch.float32, device="cpu", seed=7).passed


@pytest.mark.parametrize("stale_input", DYNAMIC_INPUTS)
def test_replay_orchestration_rejects_each_stale_dynamic_input(stale_input):
    backend, captured = FakeGraphBackend(), None
    def stale(*args):
        nonlocal captured
        args = list(args)
        if backend.phase == "capture":
            captured = args[ARGUMENTS[stale_input]].clone()
        if backend.phase == "replay":
            args[ARGUMENTS[stale_input]] = captured
        ENTRY(*args)
    result = verify_entry(get_slot(SLOT), stale, dtype=torch.float32, device="cpu", seed=7,
                          graph_replays=3, _graph_backend=backend)
    assert not result.passed and any(not r.passed for r in result.shape_results), stale_input


def test_faithful_replay_orchestration_and_profile_descriptor_agree():
    slot = get_slot(SLOT)
    result = verify_entry(slot, ENTRY, dtype=torch.bfloat16, device="cpu", seed=7,
                          graph_replays=3, _graph_backend=FakeGraphBackend())
    assert result.passed and all(r.graph_replays == 3 for r in result.shape_results)
    inputs = slot.make_inputs(**slot.shapes[0], dtype=torch.float32, device="cpu", seed=7)
    context = dict(architecture="sm103", graph_mode="cuda_graph", tp_size=4, world_size=4)
    assert verification_call_descriptor(slot, inputs, dtype_name="float32", **context) == call_descriptor(inputs, **context)
    profile = model_profile("GLM-5.3", SLOT)
    assert {s["query_chunk"] for s in profile.shapes} >= {1, 128, 16384}
    assert all(s["num_heads"] == 64 and s["value_dim"] == 512 for s in profile.shapes)
    assert resolve_target(load_manifest(BUNDLE)).target_id == SLOT
    assert default_target_catalog().target_spec_digest(SLOT)


@pytest.mark.parametrize("is_neox,rotated", [(False, [-2., 1., -4., 3.]), (True, [-3., -4., 1., 2.])])
def test_query_oracle_rotation_conventions_and_fp8_saturation(is_neox, rotated):
    inputs = dict(q=torch.tensor([[[1000., -1000.]]]), q_rope=torch.tensor([[[1., 2., 3., 4.]]]),
                  positions=torch.tensor([1]), cos_sin_cache=torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]]),
                  is_neox=is_neox)
    assert query_reference(inputs).float().tolist() == [[[448., -448., *rotated]]]

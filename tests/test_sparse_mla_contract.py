"""Sparse-MLA math and replay orchestration, without claiming CUDA evidence."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cacheon.manifest import load_manifest  # noqa: E402
from cacheon.model_profiles import model_profile, verification_call_descriptor  # noqa: E402
from cacheon.sandbox import load_entry  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402
from cacheon.sparse_mla_contract import (  # noqa: E402
    DYNAMIC_INPUTS, SLOT, call_descriptor, invoke_entry, reference,
)
from cacheon.target_catalog import default_target_catalog, resolve_target  # noqa: E402
from cacheon.verify import verify_entry  # noqa: E402
from support.graph_backend import FakeGraphBackend  # noqa: E402

BUNDLE = Path(__file__).resolve().parents[1] / "examples/miner_sparse_mla_torch"


@pytest.fixture()
def entry():
    return load_entry(str(BUNDLE / "kernels/sparse_mla.py"), "sparse_mla")


def _case():
    return dict(
        q=torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) / 25,
        kv_cache=torch.arange(256, dtype=torch.float32).reshape(2, 32, 4) / 150,
        indices=torch.tensor([[35, 3, 9, -1, 63], [1, 4, 8, -1, 40],
                              [-1, -1, -1, -1, -1]], dtype=torch.int32),
        seq_lens=torch.tensor([3, 2, 0], dtype=torch.int32),
        value_dim=2, qk_scale=0.7, value_scale=1.5,
    )


def _run(entry, inputs):
    out = torch.full(
        (inputs["q"].shape[0], inputs["q"].shape[1], inputs["value_dim"]),
        torch.nan, dtype=torch.bfloat16,
    )
    invoke_entry(entry, inputs, [out])
    return out


def _scalar_oracle(inputs):
    rows = inputs["kv_cache"].flatten(0, 1).double().tolist()
    output = []
    for q, ids, length in zip(
        inputs["q"].double().tolist(), inputs["indices"].tolist(),
        inputs["seq_lens"].tolist(),
    ):
        selected = [rows[i] for i in ids[:length] if i >= 0]
        heads = []
        for head in q:
            if not selected:
                heads.append([0.] * inputs["value_dim"])
                continue
            scores = [math.fsum(a * b for a, b in zip(head, key))
                      * inputs["qk_scale"] for key in selected]
            weights = [math.exp(v - max(scores)) for v in scores]
            heads.append([
                math.fsum(w * key[c] for w, key in zip(weights, selected))
                / math.fsum(weights) * inputs["value_scale"]
                for c in range(inputs["value_dim"])
            ])
        output.append(heads)
    return torch.tensor(output, dtype=torch.bfloat16)


def test_physical_pages_masked_rows_and_both_scales_match_scalar_math(entry):
    inputs = _case()
    expected = _scalar_oracle(inputs)
    torch.testing.assert_close(reference(inputs)[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(_run(entry, inputs), expected, atol=0, rtol=0)
    assert expected[2].count_nonzero() == 0


def test_future_history_and_indices_outside_active_prefix_cannot_leak(entry):
    inputs = _case()
    before = _run(entry, inputs)
    allowed = {35, 3, 9, 1, 4}
    cache = inputs["kv_cache"].flatten(0, 1)
    for row in set(range(cache.shape[0])) - allowed:
        cache[row].fill_(10000)
    # Inactive tails are not even valid addresses; a kernel must not gather them.
    inputs["indices"][0, 3:] = 999999
    inputs["indices"][1, 2:] = 999999
    torch.testing.assert_close(_run(entry, inputs), before, atol=0, rtol=0)
    torch.testing.assert_close(reference(inputs)[0], before, atol=0, rtol=0)
    cache[9].add_(2)
    after = _run(entry, inputs)
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1:], before[1:], atol=0, rtol=0)


@pytest.mark.parametrize("storage_dtype", ["float32", "float16", "bfloat16", "float8_e4m3fn"])
def test_two_profiles_have_independent_math_and_mutable_graph_inputs(entry, storage_dtype):
    slot = get_slot(SLOT)
    for shape in slot.shapes:
        old = slot.make_inputs(
            **shape, dtype=torch.float32, input_dtype=storage_dtype, device="cpu", seed=7,
        )
        out = _run(entry, old)
        torch.testing.assert_close(out, reference(old)[0], atol=0.02, rtol=0.02)
        assert out.dtype == torch.bfloat16
        pointers = {name: old[name].data_ptr() for name in DYNAMIC_INPUTS}
        for seed in (11, 19, 23):
            fresh = slot.make_inputs(
                **shape, dtype=torch.float32, input_dtype=storage_dtype,
                device="cpu", seed=seed,
            )
            for name in DYNAMIC_INPUTS:
                assert not torch.equal(old[name].float(), fresh[name].float()), name
                old[name].copy_(fresh[name])
                assert old[name].data_ptr() == pointers[name]
            torch.testing.assert_close(
                _run(entry, old), reference(old)[0], atol=0.02, rtol=0.02,
            )


def test_ordinary_verifier_accepts_faithful_and_rejects_wrong_output(entry):
    slot = get_slot(SLOT)
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=7)
    assert result.passed
    assert result.graph_required and not result.graph_verified
    assert not result.fully_verified

    def wrong(q, cache, indices, lengths, out, *scales):
        out.zero_()

    result = verify_entry(slot, wrong, dtype=torch.float32, device="cpu", seed=7)
    assert not result.passed


@pytest.mark.parametrize("stale_input", DYNAMIC_INPUTS)
def test_replay_orchestration_rejects_each_stale_dynamic_input(entry, stale_input):
    backend = FakeGraphBackend()
    position = DYNAMIC_INPUTS.index(stale_input)
    captured = None

    def stale(*args):
        nonlocal captured
        args = list(args)
        if backend.phase == "capture":
            captured = args[position].clone()
        if backend.phase == "replay":
            args[position] = captured
        entry(*args)

    result = verify_entry(
        get_slot(SLOT), stale, dtype=torch.float32, device="cpu", seed=7,
        graph_replays=3, _graph_backend=backend,
    )
    assert not result.passed, stale_input
    assert any(not r.passed for r in result.shape_results)


def test_faithful_replay_orchestration_and_profile_descriptor_agree(entry):
    slot = get_slot(SLOT)
    result = verify_entry(
        slot, entry, dtype=torch.bfloat16, device="cpu", seed=7,
        graph_replays=3, _graph_backend=FakeGraphBackend(),
    )
    assert result.passed
    assert all(r.graph_replays == 3 for r in result.shape_results)
    # The backend above rehearses orchestration only; no CUDA graph was run.
    inputs = slot.make_inputs(**slot.shapes[0], dtype=torch.float32, device="cpu", seed=7)
    context = dict(architecture="sm103", graph_mode="cuda_graph", tp_size=4, world_size=4)
    assert verification_call_descriptor(
        slot, inputs, dtype_name="float32", **context,
    ) == call_descriptor(inputs, **context)
    profile = model_profile("GLM-5.3", SLOT)
    assert {s["query_chunk"] for s in profile.shapes} >= {1, 128, 16384}
    assert all(s["num_heads"] == 64 and s["value_dim"] == 512 for s in profile.shapes)
    manifest = load_manifest(BUNDLE)
    assert resolve_target(manifest).target_id == SLOT
    assert default_target_catalog().target_spec_digest(SLOT)


def test_out_of_bounds_active_index_is_reference_input_error():
    inputs = _case()
    inputs["indices"][0, 0] = 64
    with pytest.raises(ValueError, match="outside cache"):
        reference(inputs)

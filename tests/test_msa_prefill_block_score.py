"""CPU contract tests for the batched, paged MSA prefill selector."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cacheon.artifact_abi import MSA_PREFILL_BLOCK_SCORE_CALL_ABI  # noqa: E402
from cacheon.capabilities import msa_prefill_call_descriptor  # noqa: E402
from cacheon.minimax_sparse_prefill_slot import INPUT_NAMES  # noqa: E402
from cacheon.registry import eligibility_from_metadata  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402
from cacheon.target_catalog import default_target_catalog  # noqa: E402
from cacheon.tensor_spec import validate_output_spec  # noqa: E402
from cacheon.verify import (  # noqa: E402
    _has_msa_prefill_probe_schema,
    _verification_call_descriptor,
    format_verify,
    verify_entry,
)

SLOT = get_slot("attention.msa_prefill_block_score")


def _candidate(*args, causal: bool = True, paged: bool = True) -> None:
    """Independent score-sheet oracle collapsed to selected block sets."""

    *values, out = args
    i = dict(zip(INPUT_NAMES, values, strict=True))
    out.fill_(-1)
    bq, bk, width = (int(i[name]) for name in
                     ("block_size_q", "block_size_k", "topk"))
    for request in range(i["seq_lens"].numel()):
        qs, qe = (int(i["cu_seqlens"][request + offset]) for offset in (0, 1))
        seq_len, prefix = int(i["seq_lens"][request]), int(i["prefix_lens"][request])
        sid = int(i["slot_ids"][request])
        physical = (i["req_to_token"][sid, :seq_len].long() if paged
                    else torch.arange(seq_len, device=out.device))
        keys = i["index_k_cache"][physical, 0].float()
        out_row = int(i["cu_seqblocks_q"][request])
        for qblock, row in enumerate(range(qs, qe, bq)):
            visible = min(seq_len, prefix + qblock * bq + 1) if causal else seq_len
            blocks = math.ceil(visible / bk)
            scores = i["q"][row].float() @ keys[:visible].T
            scores = torch.nn.functional.pad(
                scores, (0, blocks * bk - visible), value=float("-inf")
            ).view(i["q"].shape[1], blocks, bk).amax(-1)
            scores[:, :min(blocks, int(i["init_blocks"]))] = torch.inf
            scores[:, max(0, blocks - int(i["local_blocks"])):] = torch.finfo(scores.dtype).max
            take = min(width, blocks)
            out[:, out_row + qblock, :take] = scores.topk(take, dim=-1).indices.int()


def _faithful(*args) -> None:
    _candidate(*args)


def _wrong_selection(*args) -> None:
    _candidate(*args)
    args[-1][..., 0] = 999


def _acausal(*args) -> None:
    _candidate(*args, causal=False)


def _ignores_paging(*args) -> None:
    _candidate(*args, paged=False)


def _inputs(index: int = 0, *, seed: int = 0):
    return SLOT.make_inputs(**SLOT.shapes[index], dtype=torch.float32,
                            device="cpu", seed=seed)


def _production_eligibility(**overrides):
    capabilities = {
        "dtype": "float32", "architecture": "sm103", "head_dim": 16,
        "block_size": 4, "phase": "prefill", "layout": "paged",
        "graph_mode": "eager", "quant": "dense", "num_kv_heads": 1,
    }
    capabilities.update(overrides)
    return eligibility_from_metadata(
        {"graph_safe": False, "capabilities": capabilities},
        ("float32",), ("sm103",),
    )


def test_slot_is_same_target_with_v2_paged_selection_contract():
    assert SLOT.kind == "block" and SLOT.entry == "msa_prefill_block_score"
    assert SLOT.correctness.mode == "topk_overlap"
    inputs = _inputs()
    assert tuple(inputs) == INPUT_NAMES
    assert inputs["q"].shape[1:] == (3, 16)
    assert inputs["index_k_cache"].shape[1:] == (1, 16)
    assert torch.diff(inputs["cu_seqlens"]).tolist() == [7, 6]
    assert inputs["prefix_lens"].tolist() == [21, 20]
    assert inputs["slot_ids"].tolist() == [1, 0]
    identity = torch.arange(inputs["req_to_token"].shape[1])
    assert any(not torch.equal(row.cpu(), identity) for row in inputs["req_to_token"])


def test_output_is_validator_allocated_contiguous_int32_selection():
    inputs = _inputs(1)
    contract = SLOT.output_contract(inputs)
    spec = contract.outputs[0]
    expected = (
        inputs["q"].shape[1], int(inputs["all_seqblock_q"]), int(inputs["topk"])
    )
    assert spec.shape == expected == SLOT.out_shapes(inputs)[0]
    assert spec.dtype == torch.int32 and spec.stride_policy == "contiguous"
    out = torch.empty(expected, dtype=torch.int32)
    validate_output_spec(
        contract, [out], fallback_dtype=torch.float32, fallback_device="cpu",
        inputs=tuple(value for value in inputs.values() if torch.is_tensor(value)),
    )


def test_reference_preserves_padding_and_forces_initial_and_local_blocks():
    inputs = _inputs(1)
    output = SLOT.invoke_reference(inputs)[0]
    assert (output == -1).any(), "short causal rows must retain -1 padding"
    for request in range(inputs["seq_lens"].numel()):
        prefix = int(inputs["prefix_lens"][request])
        qblocks = (int(inputs["cu_seqblocks_q"][request]),
                   int(inputs["cu_seqblocks_q"][request + 1]))
        for qb, row in enumerate(range(*qblocks)):
            visible = prefix + qb * int(inputs["block_size_q"]) + 1
            blocks = math.ceil(visible / int(inputs["block_size_k"]))
            selected = set(output[0, row].tolist()) - {-1}
            forced = {
                *range(min(blocks, int(inputs["init_blocks"]))),
                *range(max(0, blocks - int(inputs["local_blocks"])), blocks),
            }
            assert forced <= selected
            assert len(selected) == min(int(inputs["topk"]), blocks)


def test_faithful_candidate_verifies_for_multi_request_head_and_ragged_shapes():
    result = verify_entry(
        SLOT, _faithful, dtype=torch.float32, device="cpu", graph_safe=False,
    )
    assert result.passed, format_verify(result)
    assert all(row.metric == "overlap" for row in result.shape_results)


@pytest.mark.parametrize("candidate", [_wrong_selection, _acausal, _ignores_paging])
def test_wrong_selection_causality_or_paging_fails(candidate):
    shape = SLOT.shapes[2] if candidate is not _wrong_selection else SLOT.shapes[0]
    result = verify_entry(
        SLOT, candidate, dtype=torch.float32, device="cpu", graph_safe=False,
        shapes=[shape],
    )
    assert not result.passed, format_verify(result)


def test_call_abi_matches_python_entry_order_without_score_slab():
    abi = MSA_PREFILL_BLOCK_SCORE_CALL_ABI
    expected = tuple(f"input.{name}" for name in INPUT_NAMES) + ("output.topk_idx",)
    assert SLOT.call_abi is abi and abi.call_args == expected
    names = {resource.name for resource in abi.resources}
    assert set(expected) <= names
    assert "input.index_k" not in names and "output.block_scores" not in names


def test_v2_catalog_and_verifier_descriptor_match_live_descriptor():
    contract = default_target_catalog().require(SLOT.name).contract_ref
    assert contract is not None
    assert contract.input_abi_id.endswith(".v2")
    assert contract.output_abi_id.endswith(".v2")
    assert contract.binding_family_id.endswith("prefill-selection.v2")
    inputs = _inputs()
    verified = _verification_call_descriptor(
        SLOT, inputs, dtype=torch.float32, device="cpu", architecture="sm103",
        tp_size=4, world_size=4, graph_mode="eager", model_key=None,
    )
    live = msa_prefill_call_descriptor(
        dtype="float32", architecture="sm103", head_dim=16, block_size=4,
        q_len=7, kv_len=28, top_k=5, batch_size=2, num_q_heads=3,
        q_block_size=4, init_blocks=0, local_blocks=1, num_tokens=13,
        num_kv_heads=1, tp_size=4, world_size=4,
    )
    assert verified == live
    assert _has_msa_prefill_probe_schema(
        SLOT, _production_eligibility(), list(SLOT.shapes)
    )


def test_v2_accepts_declared_131k_only_domain():
    result = verify_entry(
        SLOT, _faithful, dtype=torch.float32, device="cpu", architecture="sm103",
        eligibility=_production_eligibility(
            q_len=1, head_dim=128, block_size=128, kv_len=131072
        ),
        graph_safe=False,
    )
    assert result.passed and result.domain_coverage_complete, format_verify(result)
    assert any(
        row.applicable and row.shape.get("prefix_len_override") == 131071
        for row in result.shape_results
    )

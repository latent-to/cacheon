from __future__ import annotations

import threading
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.oci_session_protocol import BatchEvidence
from cacheon.eval.resident_count_execution_evidence import (
    ResidentCountExecutionEvidenceError,
    ResidentCountQualityExecutionEvidence,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionHold,
    execute_candidate_count_quality,
)
from cacheon.eval.resident_pair_binding import ResidentPairRuntimeBinding
from tests.test_resident_count_quality_execution import _fixture, _h


def _walk(value):
    yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from _walk(getattr(value, field.name))
    elif type(value) in (tuple, list):
        for row in value:
            yield from _walk(row)
    elif type(value) is dict:
        for key, row in value.items():
            yield from _walk(key)
            yield from _walk(row)


def _replace_binding(
    binding: ResidentPairRuntimeBinding, authority: str
) -> ResidentPairRuntimeBinding:
    if authority == "service_epoch_digest":
        return replace(binding, service_epoch_digest=_h("foreign-service-epoch"))
    lane = replace(
        binding.lanes[0], **{authority: _h(f"foreign-{authority}")}
    )
    return replace(binding, lanes=(lane, binding.lanes[1]))


def test_resident_count_execution_evidence_round_trips_canonically_without_paths() -> None:
    plan, judge, pair, _, _ = _fixture(6, barrier=False)
    execution = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=10**10
    )
    evidence = execution.evidence
    codec = ContinuationCodec((ResidentCountQualityExecutionEvidence,))
    reopened = codec.decode(codec.encode(evidence))

    assert reopened == evidence
    assert reopened.digest == evidence.digest
    assert reopened.regrade(plan, judge) == execution.observation
    assert evidence.to_dict()["pair_binding"]["service_epoch_digest"] == (
        plan.pair_binding.service_epoch_digest
    )
    assert not any(isinstance(value, Path) for value in _walk(evidence))
    lowered = repr(codec.encode(evidence)).lower()
    for forbidden in ("/users/", "/root/", "host_path", "pathlib"):
        assert forbidden not in lowered
    pair.close()


def test_raw_evidence_is_canonical_a_b_after_b_completes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, judge, pair, factory_a, _ = _fixture(6, barrier=True)
    b_completed = threading.Event()
    original_complete = pair._complete_result  # type: ignore[attr-defined]

    def observe_complete(work, result):
        original_complete(work, result)
        if result.request_slice.lane_id == "B":
            b_completed.set()

    monkeypatch.setattr(pair, "_complete_result", observe_complete)
    session_a = factory_a.sessions[0]
    original_execute = session_a.execute_batch_with_shape

    def delay_a(prompts, *, shape, canary=False):
        row = original_execute(prompts, shape=shape, canary=canary)
        assert b_completed.wait(2.0)
        return row

    session_a.execute_batch_with_shape = delay_a
    execution = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=10**10
    )
    slices = execution.evidence.request_slices

    assert tuple(row.request_slice.lane_id for row in pair.request_history) == (
        "B",
        "A",
    )
    assert tuple(row.lane_id for row in slices) == ("A", "B")
    assert slices[0].evaluation_id == slices[1].evaluation_id
    assert slices[0].request_id != slices[1].request_id
    assert execution.observation.execution_evidence_digest == execution.evidence.digest
    pair.close()


@pytest.mark.parametrize(
    "authority",
    (
        "service_epoch_digest",
        "stock_launch_digest",
        "allocation_digest",
        "executor_namespace_digest",
        "lane_digest",
    ),
)
def test_runtime_authority_tamper_fails_independent_regrade(authority: str) -> None:
    plan, judge, pair, _, _ = _fixture(4, barrier=False)
    execution = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=10**10
    )
    changed = replace(
        execution.evidence,
        pair_binding=_replace_binding(plan.pair_binding, authority),
    )

    with pytest.raises(ResidentCountQualityExecutionHold, match="commissioned plan"):
        changed.regrade(plan, judge)
    pair.close()


@pytest.mark.parametrize(
    "field",
    ("execution_plan_digest", "envelope_digest"),
)
def test_plan_or_envelope_tamper_fails_independent_regrade(field: str) -> None:
    plan, judge, pair, _, _ = _fixture(4, barrier=False)
    execution = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=10**10
    )
    changed = replace(execution.evidence, **{field: _h(f"foreign-{field}")})

    with pytest.raises(ResidentCountQualityExecutionHold, match="commissioned plan"):
        changed.regrade(plan, judge)
    pair.close()


def test_candidate_bundle_tamper_never_constructs_valid_raw_evidence() -> None:
    plan, judge, pair, _, _ = _fixture(4, barrier=False)
    evidence = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=10**10
    ).evidence

    with pytest.raises(ResidentCountExecutionEvidenceError, match="incomplete"):
        replace(evidence, candidate_bundle_digest=_h("foreign-candidate"))
    pair.close()


@pytest.mark.parametrize("tamper", ("order", "evaluation", "request", "swap"))
def test_slice_or_swap_tamper_never_constructs_valid_raw_evidence(tamper: str) -> None:
    plan, judge, pair, _, _ = _fixture(4, barrier=False)
    evidence = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=10**10
    ).evidence
    lane_a, lane_b = evidence.request_slices

    if tamper == "order":
        slices = (lane_b, lane_a)
    elif tamper == "evaluation":
        slices = (lane_a, replace(lane_b, evaluation_id="e" * 32))
    elif tamper == "request":
        slices = (lane_a, replace(lane_b, request_id=lane_a.request_id))
    else:
        activation, restoration = lane_a.new_swaps
        activation = replace(activation, swap_index=activation.swap_index + 7)
        slices = (
            replace(lane_a, new_swaps=(activation, restoration)),
            lane_b,
        )

    with pytest.raises(ResidentCountExecutionEvidenceError):
        replace(evidence, request_slices=slices)
    pair.close()


def test_raw_evidence_tamper_fails_independent_regrade() -> None:
    plan, judge, pair, _, _ = _fixture(4, barrier=False)
    evidence = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=10**10
    ).evidence
    lane_a, lane_b = evidence.request_slices
    batch = lane_a.new_batches[0]
    prompt = batch.evidence.prompts[0]
    changed_prompt = replace(
        prompt,
        output_ids=prompt.output_ids[:-1],
        top_logprobs=prompt.top_logprobs[:-1],
    )
    changed_batch = replace(
        batch,
        token_numerator=batch.token_numerator - 1,
        evidence=BatchEvidence((changed_prompt, *batch.evidence.prompts[1:])),
    )
    changed = replace(
        evidence,
        request_slices=(
            replace(lane_a, new_batches=(changed_batch,)),
            lane_b,
        ),
    )

    with pytest.raises(ResidentCountQualityExecutionHold, match="coverage"):
        changed.regrade(plan, judge)
    pair.close()

"""The prepared graph consumer requires the cases the collective verifier produces."""

from dataclasses import replace

import pytest

from cacheon.eval import b300_prepared_graph_probe as probe
from cacheon.capabilities import CallDescriptor
from cacheon.model_profiles import slot_for_model
from cacheon.verification_outcomes import GraphPhaseOutcome, VerificationCaseKind as Kind
from cacheon.verify_collective import _graph_capture_sequence, _temporal_sequence
from tests import test_b300_prepared_graph_probe as fixtures
from tests.test_b300_prepared_graph_probe import _request, _result, _shape

atomic = fixtures.atomic


def _domain_result(policy, slot, variant, shapes, *, selected=None):
    result = _result(policy, slot, variant, collective=True)
    result.shape_results = []
    applicable = []
    for index, shape in enumerate(shapes):
        accepted = selected is None or index in selected
        outcome = (GraphPhaseOutcome.graph_passed(policy.expected_graph_replays)
                   if accepted else GraphPhaseOutcome.not_applicable())
        row = _shape(policy, slot, variant, outcome, kind=Kind.COLLECTIVE_SINGLE,
                     ordinal=index, applicable=accepted)
        row.shape = shape
        row.case_descriptor = replace(row.case_descriptor, calls=(CallDescriptor({
            **dict(row.case_descriptor.calls[0]), "num_tokens": shape["num_tokens"],
        }),))
        result.shape_results.append(row)
        if accepted:
            applicable.append(shape)
    for kind, sequence in (
        (Kind.COLLECTIVE_TEMPORAL_EAGER, _temporal_sequence(applicable)),
        (Kind.COLLECTIVE_GRAPH_SEQUENCE, _graph_capture_sequence(applicable)),
    ):
        if sequence is not None:
            outcome = (GraphPhaseOutcome.eager_only_passed()
                       if kind is Kind.COLLECTIVE_TEMPORAL_EAGER
                       else GraphPhaseOutcome.graph_passed(policy.expected_graph_replays))
            result.shape_results.append(_shape(policy, slot, variant, outcome,
                                               kind=kind, ordinal=20))
    return result


@pytest.mark.parametrize("model,slot,selected", [
    ("GLM-5.3", "collective.all_reduce", None),
    ("GLM-5.3", "collective.all_gather_into_tensor", None),
    ("MiniMax-M3", "collective.all_reduce", None),
    ("MiniMax-M3", "collective.all_reduce", {0}),
])
def test_registered_and_domain_filtered_profiles_keep_their_graph_evidence(
    atomic, model, slot, selected,
):
    policy = replace(_request(atomic).policy, model_profile_key=model)
    shapes = list(slot_for_model(slot, model).shapes)
    result = _domain_result(policy, slot, "default", shapes, selected=selected)
    record = probe._variant_record(result, slot_id=slot, variant_id="default",
                                   policy=policy, collective=True)
    singles = [row for row in result.shape_results
               if row.case_descriptor.case_kind is Kind.COLLECTIVE_SINGLE]
    assert all(row.case_descriptor.digest in {s.descriptor_digest for s in record.shapes}
               for row in singles)
    assert sum(row.applicable for row in singles) == (len(shapes) if selected is None else 1)
    assert all(row.replay_count == policy.expected_graph_replays
               for row in record.shapes if row.applicable)
    assert not any(row.failed for row in record.shapes)


@pytest.mark.parametrize("kind,operation", [
    (Kind.COLLECTIVE_TEMPORAL_EAGER, "omit"),
    (Kind.COLLECTIVE_GRAPH_SEQUENCE, "omit"),
    (Kind.COLLECTIVE_TEMPORAL_EAGER, "duplicate"),
    (Kind.COLLECTIVE_GRAPH_SEQUENCE, "duplicate"),
])
def test_missing_or_duplicate_required_sequence_still_holds(atomic, kind, operation):
    policy = _request(atomic).policy
    slot = "collective.all_reduce"
    shapes = list(slot_for_model(slot, "MiniMax-M3").shapes)
    result = _domain_result(policy, slot, "default", shapes)
    sequence = next(row for row in result.shape_results if row.case_descriptor.case_kind is kind)
    if operation == "omit":
        result.shape_results.remove(sequence)
    else:
        result.shape_results.append(sequence)
    with pytest.raises(probe.PreparedGraphProbeIncompleteError, match="omitted or duplicated"):
        probe._variant_record(result, slot_id=slot, variant_id="default",
                              policy=policy, collective=True)


def test_same_token_count_requires_graph_sequence_but_no_temporal_transition(atomic):
    policy = _request(atomic).policy
    result = _domain_result(policy, "collective.all_reduce", "default",
                            [{"num_tokens": 8, "hidden": 4096},
                             {"num_tokens": 8, "hidden": 7168}])
    record = probe._variant_record(result, slot_id=result.slot, variant_id="default",
                                   policy=policy, collective=True)
    assert len(record.shapes) == 3
    assert all(row.case_descriptor.case_kind is not Kind.COLLECTIVE_TEMPORAL_EAGER
               for row in result.shape_results)


def test_production_probe_accepts_one_applicable_shape_per_variant(atomic, monkeypatch):
    request = _request(atomic)

    def verify(slot, _source, _entry, **kwargs):
        return _domain_result(request.policy, slot.name, kwargs["variant_name"],
                              [{"num_tokens": 16384, "hidden": 6144}])

    monkeypatch.setattr(probe, "_VERIFY_COLLECTIVE", verify)
    artifact = probe.execute_prepared_graph_probe(request, atomic.prepared.binding.tree.root)
    assert len(artifact.variants) == len(request.target_variants)
    assert all(len(row.shapes) == 1 and not row.shapes[0].failed for row in artifact.variants)


def test_singleton_candidate_failure_still_produces_failed_graph_evidence(atomic):
    policy = _request(atomic).policy
    result = _domain_result(policy, "collective.all_reduce", "default",
                            [{"num_tokens": 16384, "hidden": 6144}])
    row = result.shape_results[0]
    row.phase_outcome = GraphPhaseOutcome.capture_candidate_failed()
    row.passed, row.graph_replays = False, 0
    result.passed = result.graph_verified = False
    record = probe._variant_record(result, slot_id=result.slot, variant_id="default",
                                   policy=policy, collective=True)
    assert record.shapes[0].failed
    assert record.shapes[0].failure_is_candidate_attributable

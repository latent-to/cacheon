"""CPU contracts for the graph-only registered qualification terminal."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import cacheon.eval.b300_registered_qualification as registered
from cacheon.arena_service import ArenaCandidateBinding
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    publish_evidence,
    reopen_evidence,
)
from cacheon.eval.qualification import (
    GRAPH_EVIDENCE_DOMAIN,
    GRAPH_EVIDENCE_MEDIA_TYPE,
    GRAPH_EVIDENCE_SCHEMA,
    DiscoveryQualificationProfile,
    GraphVerificationEvidenceRef,
    GraphVerificationRawEvidence,
    QualificationDecision,
    reopen_graph_verification,
)
from cacheon.eval.qualification_graph_exit import (
    MAX_QUALIFICATION_GRAPH_EXIT_BYTES,
    QUALIFICATION_GRAPH_EXIT_DOMAIN,
    QUALIFICATION_GRAPH_EXIT_MEDIA_TYPE,
    QUALIFICATION_GRAPH_EXIT_SCHEMA,
    QualificationGraphExitError,
    QualificationGraphExitHold,
    assert_qualification_graph_exit_schema_safe,
    publish_qualification_graph_exit,
    reopen_qualification_graph_exit,
)
from cacheon.eval.qualification_intake import (
    GraphShapeObservation,
    GraphVariantObservation,
)
from cacheon.eval.qualification_runner import (
    DiscoveryCandidateQualificationAuthority,
)
from cacheon.stack_identity import canonical_json_bytes
from tests.test_b300_registered_qualification import (
    _graph_facts_for_members,
    _harness,
)
from tests.test_marginal_runtime import FUSED
from tests.test_qualification import _discovery_execution
from tests.test_qualification_runner import _reference


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _failure_facts(members: tuple[str, ...]) -> registered.B300FocusedGraphFacts:
    base = _graph_facts_for_members(members)
    observations = []
    for variant in base.variants:
        shapes = tuple(
            GraphShapeObservation(
                descriptor,
                True,
                True,
                index != 0,
                0 if index == 0 else base.expected_graph_replays,
                index != 0,
            )
            for index, descriptor in enumerate(variant.shape_descriptor_digests)
        )
        observations.append(
            GraphVariantObservation(
                variant.slot_id,
                variant.variant_id,
                True,
                True,
                shapes,
            )
        )
    return registered.B300FocusedGraphFacts(
        base.expected_graph_replays,
        base.variants,
        tuple(observations),
    )


def _plan(
    tmp_path: Path,
    *,
    source_fixture: Path | None = None,
    failure: bool,
):
    harness = _harness(tmp_path, source_fixture)
    factory = harness.factory
    if failure:
        facts = _failure_facts(harness.candidate.reservation.target_members)
        inputs = replace(
            harness.inputs,
            graph_facts_builder_digest=_h(
                "graph-failure:" + harness.candidate.reservation.target_id
            ),
            graph_facts_builder=lambda _candidate, _prepared: facts,
        )
        factory = registered.build_b300_registered_qualification_factory(inputs)
    value = factory.plan_builder(harness.cohort, b"g" * 32)
    return harness, value, value.candidates[0]


def _kwargs(harness, value, authority) -> dict[str, object]:
    return {
        "expected_plan": value,
        "expected_authority": authority,
        "expected_reservation": harness.candidate.reservation,
        "authenticated_request_digest": _h("authenticated-request"),
        "expected_candidate_binding": harness.candidate,
    }


def _raw(value, authority) -> GraphVerificationRawEvidence:
    return GraphVerificationRawEvidence.from_dict(
        json.loads(reopen_evidence(value.evidence_root, authority.graph_artifact_ref))
    )


def _with_raw(value, authority, raw: GraphVerificationRawEvidence):
    artifact = publish_evidence(
        value.evidence_root,
        canonical_json_bytes(raw.to_dict()),
        domain=GRAPH_EVIDENCE_DOMAIN,
        media_type=GRAPH_EVIDENCE_MEDIA_TYPE,
        schema=GRAPH_EVIDENCE_SCHEMA,
    )
    evidence_ref = GraphVerificationEvidenceRef(
        authority.graph_requirement.binding,
        authority.graph_requirement.digest,
        raw.digest,
    )
    changed = replace(
        authority,
        graph_artifact_ref=artifact,
        graph_evidence_ref=evidence_ref,
    )
    return replace(value, candidates=(changed,)), changed


def _publish_resigned(root: Path, row: dict[str, object]) -> EvidenceArtifactRef:
    return publish_evidence(
        root,
        canonical_json_bytes(row),
        domain=QUALIFICATION_GRAPH_EXIT_DOMAIN,
        media_type=QUALIFICATION_GRAPH_EXIT_MEDIA_TYPE,
        schema=QUALIFICATION_GRAPH_EXIT_SCHEMA,
        max_bytes=MAX_QUALIFICATION_GRAPH_EXIT_BYTES,
    )


@pytest.mark.parametrize("source_fixture", (None, FUSED), ids=("singleton", "atomic"))
def test_registered_profiles_use_one_fail_publish_reopen_path(
    tmp_path: Path,
    source_fixture: Path | None,
) -> None:
    harness, value, authority = _plan(
        tmp_path,
        source_fixture=source_fixture,
        failure=True,
    )
    expected_grade = reopen_graph_verification(
        value.evidence_root,
        authority.graph_artifact_ref,
        authority.graph_requirement,
        authority.graph_evidence_ref,
    )

    reference = publish_qualification_graph_exit(
        value.evidence_root,
        **_kwargs(harness, value, authority),
    )
    reopened = reopen_qualification_graph_exit(
        value.evidence_root,
        reference,
        **_kwargs(harness, value, authority),
    )

    assert expected_grade.decision is QualificationDecision.FAIL
    assert expected_grade.reason == "graph_capture_failed"
    assert reopened.graph_grade == expected_grade
    assert reopened.decision is QualificationDecision.FAIL
    assert reopened.terminal_reason == expected_grade.reason
    assert reopened.graph_requirement.binding.target_id == (
        harness.candidate.reservation.target_id
    )
    assert reopened.graph_requirement.binding.target_spec_digest == (
        authority.graph_requirement.binding.target_spec_digest
    )
    assert tuple(
        member.slot_id for member in reopened.graph_requirement.binding.members
    ) == harness.candidate.reservation.target_members


def test_graph_pass_is_continuation_and_publishes_no_exit(tmp_path: Path) -> None:
    harness, value, authority = _plan(tmp_path, failure=False)

    with pytest.raises(QualificationGraphExitError, match="PASS continues"):
        publish_qualification_graph_exit(
            value.evidence_root,
            **_kwargs(harness, value, authority),
        )

    assert not (value.evidence_root / QUALIFICATION_GRAPH_EXIT_DOMAIN).exists()


@pytest.mark.parametrize("condition", ("no_decision", "missing", "incomplete"))
def test_inconclusive_graph_states_hold_without_terminal_artifact(
    tmp_path: Path,
    condition: str,
) -> None:
    source = FUSED if condition == "incomplete" else None
    harness, value, authority = _plan(tmp_path, source_fixture=source, failure=False)
    if condition == "missing":
        missing = replace(
            authority.graph_artifact_ref,
            sha256=_h("missing-raw-graph-artifact"),
        )
        authority = replace(authority, graph_artifact_ref=missing)
        value = replace(value, candidates=(authority,))
    else:
        raw = _raw(value, authority)
        if condition == "incomplete":
            raw = replace(raw, members=raw.members[:-1])
        else:
            member = raw.members[0]
            variant = member.variants[0]
            shape = replace(variant.shapes[0], graph_replays=2)
            variant = replace(variant, shapes=(shape, *variant.shapes[1:]))
            member = replace(member, variants=(variant, *member.variants[1:]))
            raw = replace(raw, members=(member, *raw.members[1:]))
        value, authority = _with_raw(value, authority, raw)

    with pytest.raises(QualificationGraphExitHold):
        publish_qualification_graph_exit(
            value.evidence_root,
            **_kwargs(harness, value, authority),
        )

    assert not (value.evidence_root / QUALIFICATION_GRAPH_EXIT_DOMAIN).exists()


def _tamper(row: dict[str, object], field: str) -> None:
    if field in {
        "authenticated_request_digest",
        "qualification_authority_digest",
        "source_digest",
        "reservation_digest",
        "selected_delta_digest",
        "candidate_publication_binding_digest",
        "graph_requirement_digest",
        "graph_evidence_ref_digest",
    }:
        row[field] = _h("tampered:" + field)
    elif field == "target_spec":
        row["graph_requirement"]["binding"]["target_spec_digest"] = _h(field)  # type: ignore[index]
    elif field == "requirement":
        row["graph_requirement"]["expected_graph_replays"] = 4  # type: ignore[index]
    elif field == "graph_artifact_ref":
        row["graph_artifact_ref"]["sha256"] = _h(field)  # type: ignore[index]
    elif field == "graph_grade":
        row["graph_grade"]["raw_evidence_digest"] = _h(field)  # type: ignore[index]
    elif field == "terminal_reason":
        row[field] = "graph_replay_failed"
    else:  # pragma: no cover - the parametrization is closed below
        raise AssertionError(field)


@pytest.mark.parametrize(
    "field",
    (
        "authenticated_request_digest",
        "qualification_authority_digest",
        "source_digest",
        "reservation_digest",
        "selected_delta_digest",
        "candidate_publication_binding_digest",
        "target_spec",
        "requirement",
        "graph_requirement_digest",
        "graph_evidence_ref_digest",
        "graph_artifact_ref",
        "graph_grade",
        "terminal_reason",
    ),
)
def test_each_retained_binding_tamper_fails_reopen(
    tmp_path: Path,
    field: str,
) -> None:
    harness, value, authority = _plan(tmp_path, failure=True)
    kwargs = _kwargs(harness, value, authority)
    reference = publish_qualification_graph_exit(value.evidence_root, **kwargs)
    row = json.loads(reopen_evidence(value.evidence_root, reference))
    _tamper(row, field)
    resigned = _publish_resigned(value.evidence_root, row)

    with pytest.raises(QualificationGraphExitError):
        reopen_qualification_graph_exit(
            value.evidence_root,
            resigned,
            **kwargs,
        )


def test_canonically_resigned_headline_still_fails_independent_regrade(
    tmp_path: Path,
) -> None:
    harness, value, authority = _plan(tmp_path, failure=True)
    kwargs = _kwargs(harness, value, authority)
    reference = publish_qualification_graph_exit(value.evidence_root, **kwargs)
    row = json.loads(reopen_evidence(value.evidence_root, reference))
    row["graph_grade"]["reason"] = "graph_replay_failed"
    row["terminal_reason"] = "graph_replay_failed"
    resigned = _publish_resigned(value.evidence_root, row)

    with pytest.raises(QualificationGraphExitError, match="independently regrade"):
        reopen_qualification_graph_exit(
            value.evidence_root,
            resigned,
            **kwargs,
        )


def test_raw_cas_corruption_is_hold_not_candidate_fail(tmp_path: Path) -> None:
    harness, value, authority = _plan(tmp_path, failure=True)
    kwargs = _kwargs(harness, value, authority)
    reference = publish_qualification_graph_exit(value.evidence_root, **kwargs)
    raw_path = (
        value.evidence_root
        / authority.graph_artifact_ref.domain
        / authority.graph_artifact_ref.sha256[:2]
        / authority.graph_artifact_ref.sha256
    )
    raw_path.chmod(0o600)
    raw_path.write_bytes(b"x" * authority.graph_artifact_ref.size)
    raw_path.chmod(0o400)

    with pytest.raises(QualificationGraphExitHold, match="raw graph evidence"):
        reopen_qualification_graph_exit(
            value.evidence_root,
            reference,
            **kwargs,
        )


def test_foreign_root_with_only_terminal_artifact_is_hold(tmp_path: Path) -> None:
    harness, value, authority = _plan(tmp_path / "origin", failure=True)
    kwargs = _kwargs(harness, value, authority)
    reference = publish_qualification_graph_exit(value.evidence_root, **kwargs)
    payload = reopen_evidence(value.evidence_root, reference)
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    copied = publish_evidence(
        foreign,
        payload,
        domain=reference.domain,
        media_type=reference.media_type,
        schema=reference.schema,
    )
    assert copied == reference

    with pytest.raises(QualificationGraphExitHold, match="raw graph evidence"):
        reopen_qualification_graph_exit(foreign, copied, **kwargs)


def test_schema_is_path_free_bounded_and_publication_is_idempotent(
    tmp_path: Path,
) -> None:
    harness, value, authority = _plan(tmp_path, failure=True)
    kwargs = _kwargs(harness, value, authority)
    first = publish_qualification_graph_exit(value.evidence_root, **kwargs)
    second = publish_qualification_graph_exit(value.evidence_root, **kwargs)
    payload = reopen_evidence(value.evidence_root, first)
    row = json.loads(payload)

    assert first == second
    assert len(payload) < 16 << 10
    assert len(payload) <= MAX_QUALIFICATION_GRAPH_EXIT_BYTES
    assert str(harness.candidate.publication.root).encode() not in payload
    assert harness.candidate.reservation.hotkey.encode() not in payload
    forbidden = {
        "speed", "pair", "audit", "quality", "nll", "count",
        "settlement", "weight", "weights", "path",
    }

    def keys(value: object):
        if type(value) is dict:
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif type(value) is list:
            for item in value:
                yield from keys(item)

    assert not any(forbidden & set(key.lower().split("_")) for key in keys(row))
    assert_qualification_graph_exit_schema_safe()


def test_discovery_authority_is_rejected_without_guessing(
    tmp_path: Path,
) -> None:
    harness, value, _authority = _plan(tmp_path / "registered", failure=True)
    discovery_root = tmp_path / "discovery"
    discovery_root.mkdir()
    requirement, lifecycle = _discovery_execution(discovery_root)
    reference = _reference()
    profile = DiscoveryQualificationProfile(
        reference,
        _h("discovery-context"),
        _h("discovery-calibration"),
        requirement.digest,
        ("mean_nll", "task_score", "topk_kl"),
        "2",
        lifecycle.prepared.baseline_session_plan.max_new_tokens,
        lifecycle.prepared.baseline_session_plan.top_logprobs_num,
        1,
        _h("support-policy"),
        _h("hidden-task-policy"),
        _h("runtime-policy"),
        True,
        2,
    )
    discovery = DiscoveryCandidateQualificationAuthority(
        requirement.selected_delta_digest,
        profile,
        requirement,
    )

    with pytest.raises(QualificationGraphExitError, match="registered authority only"):
        publish_qualification_graph_exit(
            value.evidence_root,
            expected_plan=value,
            expected_authority=discovery,  # type: ignore[arg-type]
            expected_reservation=harness.candidate.reservation,
            authenticated_request_digest=_h("authenticated-request"),
            expected_candidate_binding=harness.candidate,
        )

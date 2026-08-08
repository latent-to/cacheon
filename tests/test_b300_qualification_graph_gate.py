"""CPU contracts for the pre-execution registered qualification graph gate."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

import cacheon.eval.b300_qualification_graph_gate as gate_module
from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldReason
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_graph_exit import (
    QUALIFICATION_GRAPH_EXIT_DOMAIN,
    QualificationGraphExitHold,
    reopen_qualification_graph_exit,
)
from cacheon.eval.qualification_intake import QualificationAuthorityManifest
from cacheon.eval.qualification_prebuilt_plan import (
    sealed_prebuilt_qualification_plan_factory,
)
from tests.test_marginal_runtime import FUSED
from tests.test_qualification_graph_exit import (
    _kwargs,
    _plan,
    _raw,
    _with_raw,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _factory(harness, plan):
    reference = _h("graph-gate-selection-reference")
    manifest = QualificationAuthorityManifest.seal(
        plan,
        reservations=(harness.candidate.reservation,),
        selection_secret_reference=reference,
    )
    return sealed_prebuilt_qualification_plan_factory(
        manifest,
        selection_secret_reference=reference,
        selection_secret=plan.selection_secret,
        plan=plan,
    )


def _run(harness, plan, factory):
    return gate_module.run_b300_qualification_graph_gate(
        factory,
        plan,
        evidence_root=plan.evidence_root,
        candidates=(harness.candidate,),
        authenticated_request_digest=_h("authenticated-graph-gate-request"),
    )


@pytest.mark.parametrize("source_fixture", (None, FUSED), ids=("singleton", "atomic"))
def test_pass_preserves_exact_prebuilt_plan_factory_and_raw_inventory(
    tmp_path: Path,
    source_fixture: Path | None,
) -> None:
    harness, plan, authority = _plan(
        tmp_path,
        source_fixture=source_fixture,
        failure=False,
    )
    factory = _factory(harness, plan)

    result = _run(harness, plan, factory)

    assert type(result) is gate_module.B300QualificationGraphGatePass
    assert result.plan is plan
    assert result.factory is factory
    assert factory.build() is plan
    assert result.supporting_evidence_refs == (authority.graph_artifact_ref,)
    assert tuple(
        member.slot_id
        for member in authority.graph_requirement.binding.members
    ) == harness.candidate.reservation.target_members


@pytest.mark.parametrize("source_fixture", (None, FUSED), ids=("singleton", "atomic"))
def test_fail_publishes_reopens_and_projects_one_nonretryable_batch(
    tmp_path: Path,
    source_fixture: Path | None,
) -> None:
    harness, plan, authority = _plan(
        tmp_path,
        source_fixture=source_fixture,
        failure=True,
    )
    factory = _factory(harness, plan)

    result = _run(harness, plan, factory)

    assert type(result) is gate_module.B300QualificationGraphGateFail
    assert result.plan is plan
    assert result.factory is factory
    assert result.batch.attempt_ref == result.graph_exit_ref
    assert result.supporting_evidence_refs == tuple(
        sorted(
            (authority.graph_artifact_ref, result.graph_exit_ref),
            key=lambda row: (
                row.domain,
                row.sha256,
                row.media_type,
                row.schema,
                row.size,
            ),
        )
    )
    assert len(result.batch.outcomes) == 1
    outcome = result.batch.outcomes[0]
    assert outcome.decision is QualificationDecision.FAIL
    assert outcome.retryable is False
    assert outcome.attempt_artifact_sha256 == result.graph_exit_ref.sha256
    assert outcome.report_digest == result.graph_exit.digest
    assert outcome.settlement_qualification is None
    reopened = reopen_qualification_graph_exit(
        plan.evidence_root,
        result.graph_exit_ref,
        **{
            **_kwargs(harness, plan, authority),
            "authenticated_request_digest": _h(
                "authenticated-graph-gate-request"
            ),
        },
    )
    assert reopened == result.graph_exit
    terminal_files = tuple(
        row
        for row in (plan.evidence_root / QUALIFICATION_GRAPH_EXIT_DOMAIN).rglob("*")
        if row.is_file()
    )
    assert len(terminal_files) == 1


@pytest.mark.parametrize(
    ("condition", "expected_reason"),
    (
        ("missing", RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE),
        ("incomplete", RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE),
    ),
)
def test_missing_and_incomplete_evidence_map_by_typed_state_without_exit(
    tmp_path: Path,
    condition: str,
    expected_reason: RemoteQualificationHoldReason,
) -> None:
    harness, plan, authority = _plan(tmp_path, failure=False)
    if condition == "missing":
        authority = replace(
            authority,
            graph_artifact_ref=replace(
                authority.graph_artifact_ref,
                sha256=_h("missing-raw-graph-evidence"),
            ),
        )
        plan = replace(plan, candidates=(authority,))
    else:
        raw = _raw(plan, authority)
        member = raw.members[0]
        variant = member.variants[0]
        shape = replace(variant.shapes[0], graph_replays=2)
        variant = replace(variant, shapes=(shape, *variant.shapes[1:]))
        member = replace(member, variants=(variant, *member.variants[1:]))
        raw = replace(raw, members=(member, *raw.members[1:]))
        plan, authority = _with_raw(plan, authority, raw)
    factory = _factory(harness, plan)

    result = _run(harness, plan, factory)

    assert type(result) is gate_module.B300QualificationGraphGateHold
    assert result.reason is expected_reason
    assert len(result.diagnostic_digest) == 64
    assert not (plan.evidence_root / QUALIFICATION_GRAPH_EXIT_DOMAIN).exists()


def test_exit_publication_typed_failure_maps_to_ambiguous_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, plan, _authority = _plan(tmp_path, failure=True)
    factory = _factory(harness, plan)

    def held(*_args, **_kwargs):
        raise QualificationGraphExitHold(
            "arbitrary text that the gate must never parse /private/path"
        )

    monkeypatch.setattr(gate_module, "publish_qualification_graph_exit", held)
    result = _run(harness, plan, factory)

    assert type(result) is gate_module.B300QualificationGraphGateHold
    assert result.reason is (
        RemoteQualificationHoldReason.GRAPH_EXIT_PUBLICATION_AMBIGUOUS
    )
    assert "/private/path" not in result.diagnostic_digest


def test_gate_rejects_candidate_or_evidence_root_substitution(tmp_path: Path) -> None:
    harness, plan, _authority = _plan(tmp_path, failure=False)
    factory = _factory(harness, plan)

    with pytest.raises(
        gate_module.B300QualificationGraphGateError,
        match="root, and candidate",
    ):
        gate_module.run_b300_qualification_graph_gate(
            factory,
            plan,
            evidence_root=tmp_path / "foreign-evidence",
            candidates=(harness.candidate,),
            authenticated_request_digest=_h("authenticated-graph-gate-request"),
        )

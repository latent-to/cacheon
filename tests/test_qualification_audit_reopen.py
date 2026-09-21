"""Durable causal regrade preserves the audit's three possible decisions."""

from dataclasses import replace

import pytest

from cacheon.eval import qualification_runner as runner
from cacheon.eval.evidence_store import publish_evidence
from cacheon.eval.qualification import QualificationDecision
from tests.test_qualification import _lifecycle
from tests.test_qualification_runner import (
    _REAL_PUBLISH_CAUSAL, _REAL_REOPEN_CAUSAL, _quality_verdict,
    _resident_case, _run_resident_harness,
)


@pytest.mark.parametrize("decision", tuple(QualificationDecision))
def test_causal_reopen_preserves_audit_decision(monkeypatch, tmp_path, decision):
    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)
    plan = _lifecycle(tmp_path / "resident-fixture")[0].plan
    harness.value.resident_speed_plan = plan
    harness.value.prepared.candidates[0].launch.digest = plan.candidate.launch.digest
    harness.value.expected_runtime_resource_policy_digest = plan.candidate.runtime_resource_policy_digest
    _run_resident_harness(harness, baseline)
    attempt = harness.published_attempt
    report = attempt.reports[0]
    audit = report.audit_witness
    receipts = audit.receipts
    if decision is QualificationDecision.FAIL:
        receipts = (replace(receipts[0], violations=1, worst_frac=0.0),)
    elif decision is QualificationDecision.NO_DECISION:
        receipts = ()
    grade, detail = runner._grade_audit(receipts, audit.policy)
    assert grade is decision
    audit = replace(audit, receipts=receipts, decision=grade, detail=detail)
    quality = _quality_verdict(QualificationDecision.PASS, 0, harness.calibration.digest)
    report = replace(
        report, audit_witness=audit, audit_evidence_digest=audit.digest,
        audit_decision=decision, decision=decision,
        retryable=decision is QualificationDecision.NO_DECISION,
        reason=runner._report_reason(QualificationDecision.PASS, quality, audit),
    )
    attempt = replace(
        attempt, reports=(report,),
        operational_timing=replace(attempt.operational_timing, audit_evidence_digest=audit.digest),
    )
    # Retained reports can carry non-PASS audits even though fresh qualification
    # now exits before T. Reopen the actual serialized record and its host grade.
    monkeypatch.setattr(runner, "publish_evidence", publish_evidence)
    root = tmp_path / "evidence"
    reference = _REAL_PUBLISH_CAUSAL(root, attempt)
    monkeypatch.setattr(runner, "_validate_reference_execution", lambda *_args: None)
    monkeypatch.setattr(runner, "reopen_calibration_evidence", lambda *_args, **_kwargs: harness.calibration)
    monkeypatch.setattr(runner, "score_reference_quality", lambda *_args, **_kwargs: quality)
    raw = report.raw_quality_binding
    binding_fields = (
        "qualification_identity_digest", "reference_manifest_digest", "calibration_digest",
        "selection_digest", "selected_prompt_digests", "t_session_digest", "t_request_sha256",
        "support_policy_digest", "hidden_task_plan_digest", "nll_tail_threshold",
        "topk_width", "hidden_tasks_per_prompt",
    )
    monkeypatch.setattr(
        runner, "expected_raw_binding",
        lambda *_args, **_kwargs: tuple(getattr(raw, field) for field in binding_fields),
    )
    harness.value.candidates[0].profile.tokens_per_prompt = raw.tokens_per_prompt
    reopened = _REAL_REOPEN_CAUSAL(root, reference, expected=harness.value)
    assert reopened == attempt
    assert reopened.reports[0].audit_decision is decision
    assert reopened.reports[0].decision is decision

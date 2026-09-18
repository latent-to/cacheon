"""Validator failures retain completed evidence instead of rejecting miners."""

from dataclasses import replace
import hashlib

import pytest

from cacheon.audit_gate import gate, infrastructure_failure
from cacheon.eval import qualification_runner as runner
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import QualificationContinuationError
from tests.test_qualification_runner import _resident_case, _run_resident_harness


def _d(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_recovery_excludes_repair_downtime_without_retiming_evidence():
    timing = runner.QualificationTimingWitness(
        *(_d(x) for x in ("policy", "speed", "audit", "reference")),
        120, 1.0, 30.0, 86400.0, 86430.0, 90000.0, 90030.0, 90031.0,
    )
    assert timing.speed_started_monotonic_s == 1.0
    assert timing.audit_started_monotonic_s == 86400.0
    assert runner.QualificationTimingWitness.from_dict(timing.to_dict()) == timing
    with pytest.raises(runner.QualificationRunnerError, match="total wall time"):
        replace(timing, max_qualification_seconds=60)
    with pytest.raises(runner.QualificationRunnerError, match="timing order"):
        replace(timing, audit_started_monotonic_s=2.0)


@pytest.mark.parametrize("failure", ["reference", "missing_rank", "empty", "under_covered"])
def test_audit_infrastructure_does_not_publish_miner_failure(monkeypatch, failure):
    harness, baseline, _reference, exits = _resident_case(monkeypatch)
    original = runner._run_slot_audits

    def unavailable(*args, **kwargs):
        witnesses, completed = original(*args, **kwargs)
        key, audit = next(iter(witnesses.items()))
        receipts = audit.receipts
        if failure == "reference":
            receipts = (replace(receipts[0], compare_errors=1), *receipts[1:])
        elif failure == "missing_rank":
            receipts = receipts[:-1]
        elif failure == "empty":
            receipts = ()
        else:
            receipts = (replace(receipts[0], n=0), *receipts[1:])
        passed, detail = gate(
            [row.to_gate_dict() for row in receipts],
            min_calls=audit.policy.minimum_calls,
            expected_slots=audit.policy.expected_slots,
            expected_member_count=audit.policy.expected_member_count,
        )
        assert not passed
        return {key: replace(audit, receipts=receipts,
                             decision=QualificationDecision.FAIL, detail=detail)}, completed

    monkeypatch.setattr(runner, "_run_slot_audits", unavailable)
    with pytest.raises(QualificationContinuationError, match="slot audit evidence unavailable"):
        _run_resident_harness(harness, baseline)
    assert not exits
    assert harness.published_attempt is None
    assert "reference" not in harness.calls


def test_complete_numerical_violation_is_not_infrastructure():
    receipt = {"slot": "a", "pid": 7, "rank": 0, "world_size": 1,
               "n": 10, "violations": 1, "compare_errors": 0, "worst_frac": 0.1}
    kwargs = dict(min_calls=4, expected_slots=("a",), expected_member_count=1)
    assert not gate([receipt], **kwargs)[0]
    assert infrastructure_failure([receipt], **kwargs) is None


def _correction(tmp_path, *, lane="right"):
    from cacheon.eval.continuation_codec import ContinuationCodec
    from cacheon.eval.oci_backend import EngineExecutionEvidence
    from cacheon.eval.oci_session_protocol import AuditReceiptFacts
    from cacheon.eval.qualification_recovery import _workload
    from tests.test_qualification_continuation import _typed_execution
    from tests.test_qualification_runner import _typed_resident_qualification_input

    value = _typed_resident_qualification_input(tmp_path / "authority", candidate_lane=lane)
    value.evidence_root.chmod(0o700)
    authority = value.resident_audit_plan
    execution = _typed_execution(authority.launch, authority.binding, value.model_mount,
                                 authority.plan, label="corrected", artifact_root=tmp_path / "native")
    receipt = AuditReceiptFacts(
        slot=authority.audit_policy.expected_slots[0], n=64, violations=0,
        baseline_refused=0, compare_errors=0, worst_frac=1.0, min_ratio=0.97,
        mode="matched_ratio", pid=10, rank=0, world_size=1,
    )
    session = replace(execution.session, audit_policy_digest=authority.audit_policy.digest,
                      batches=tuple(replace(batch, audit_receipts=(receipt,))
                                    for batch in execution.session.batches))
    execution = replace(execution, session=session,
                        resource_policy_digest=value.expected_runtime_resource_policy_digest)
    audit = runner.AuditWitness.from_execution(execution,
        selected_delta_digest=value.candidates[0].selected_delta_digest, policy=authority.audit_policy)
    old_receipt = replace(receipt, compare_errors=2)
    _, detail = gate([old_receipt.to_gate_dict()], min_calls=authority.audit_policy.minimum_calls,
                    expected_slots=authority.audit_policy.expected_slots, expected_member_count=1)
    old = replace(audit, receipts=(old_receipt,), decision=QualificationDecision.FAIL, detail=detail)
    payload = {
        "authority": runner.qualification_authority_digest(value), "source": value.prepared.source.digest,
        "speed": _d("retained-speed"), "original_final": None, "original_audit": old.to_dict(),
        "reason": "validator comparator repaired", "launch": authority.launch.to_dict(),
        "policy": authority.audit_policy.to_dict(),
        "execution": ContinuationCodec((EngineExecutionEvidence,)).encode(execution),
        "workload": _workload(authority.plan),
        "stack": value.prepared.candidates[0].arm.candidate.to_dict(),
    }
    return value, authority, execution, audit, old, payload


@pytest.mark.parametrize("lane", ["left", "right"])
def test_corrected_audit_regrades_under_original_envelope(tmp_path, lane):
    from cacheon.eval.qualification_recovery import _checked
    value, _authority, execution, audit, _old, payload = _correction(tmp_path, lane=lane)
    assert _checked(value, payload) == (audit, execution)


def test_recommissioning_preserves_all_bundle_bytes(tmp_path):
    from cacheon.engine_tree import _write_materialized_tree
    from cacheon.eval.marginal_runtime import _expected_contributions
    from cacheon.eval.qualification_recovery import _same_contributions
    from cacheon.stack_manifest import EvaluationStackManifest
    value, authority, *_ = _correction(tmp_path)
    candidate = value.prepared.candidates[0]
    raw = candidate.arm.candidate.to_dict()
    raw["arena_digest"] = _d("recommissioned-validator")
    for ref in raw["entries"].values():
        ref["attribution_digest"] = _d("standalone-audit-attribution")
    stack = EvaluationStackManifest.from_dict(raw)
    tree = candidate.binding.tree
    files = {row.path: (tree.root / row.path).read_bytes() for row in tree.files
             if row.path != "metadata/cacheon_engine_tree.json"}
    replacement = _write_materialized_tree(tmp_path / "corrected-tree", stack_digest=stack.digest,
        files=files, runtime_manifest=tree.runtime_manifest, contributions=_expected_contributions(stack))
    launch = replace(authority.launch, arena_digest=stack.arena_digest,
                     stack_digest=stack.digest, tree_digest=replacement.tree_digest)
    assert _same_contributions(value, launch, stack.to_dict())
    assert not _same_contributions(value, replace(launch, tree_digest=_d("changed-code")), stack.to_dict())


@pytest.mark.parametrize("tamper", ["model", "candidate", "policy", "workload", "threshold", "source"])
def test_correction_rejects_foreign_or_weaker_evidence(tmp_path, tamper):
    from cacheon.eval.qualification_recovery import _checked
    value, _authority, _execution, _audit, old, payload = _correction(tmp_path)
    if tamper in {"model", "candidate"}:
        field = "model_content_digest" if tamper == "model" else "tree_digest"
        payload["launch"][field] = _d("substituted")
    elif tamper == "policy":
        payload["policy"]["minimum_calls"] = 1
    elif tamper in {"workload", "source"}:
        payload[tamper] = _d("substituted")
    else:
        weaker = replace(old.receipts[0], min_ratio=0.99)
        payload["original_audit"] = replace(old, receipts=(weaker,)).to_dict()
    with pytest.raises(QualificationContinuationError):
        _checked(value, payload)


@pytest.mark.parametrize("replace_audit", [True, False])
def test_recovery_preserves_parent_and_is_idempotent(tmp_path, monkeypatch, replace_audit):
    from types import SimpleNamespace
    from cacheon.eval import qualification_recovery as recovery
    from cacheon.eval.qualification_continuation import QualificationContinuationStore, AuditContinuation
    from tests.test_qualification import _lifecycle

    value, authority, execution, audit, old, _payload = _correction(tmp_path)
    from tests.test_qualification_continuation import _typed_execution
    lifecycle, _delta, case, _calibration, _policy = _lifecycle(tmp_path / "retained")
    speed = lifecycle.crossover
    replacements = {}
    for name, arm in (("baseline_execution", lifecycle.plan.baseline),
                      ("candidate_execution", lifecycle.plan.candidate)):
        typed = _typed_execution(arm.launch, arm.binding, case.mount, arm.session_plan,
                                 label=name, artifact_root=tmp_path / name)
        replacements[name] = replace(typed, session=getattr(speed, name).session,
                                     resource_policy_digest=arm.runtime_resource_policy_digest)
    speed = replace(speed, **replacements)
    execution = replace(execution, device_receipts=tuple(
        replace(row, started_monotonic_s=row.started_monotonic_s + 100,
                completed_monotonic_s=row.completed_monotonic_s + 100)
        for row in execution.device_receipts))
    audit = runner.AuditWitness.from_execution(execution,
        selected_delta_digest=value.candidates[0].selected_delta_digest, policy=authority.audit_policy)
    scope = QualificationContinuationStore(tmp_path / "continuations").scope(
        request_digest=_d("request"), authority_digest=runner.qualification_authority_digest(value),
        source_digest=value.prepared.source.digest,
    )
    scope.record_resident_speed(speed)
    operation = _d("original-audit-operation")
    nonce = scope.arm_evaluator("audit", operation)
    old = old if replace_audit else audit
    scope.record_audit(AuditContinuation(nonce, operation, ((old.selected_delta_digest, old),),
                                        60.0, 80.0, 80.0))
    before = {p.name: p.read_bytes() for p in scope.directory.glob("*.json")}
    monkeypatch.setattr(recovery, "_speed", lambda *_: (
        speed, None, SimpleNamespace(evidence_digest=_d("retained-speed"))))
    monkeypatch.setattr(recovery, "_operation", lambda *_: operation)
    kwargs = dict(reason="validator failure repaired")
    if replace_audit:
        kwargs.update(authority=authority, execution=execution)
    reference = recovery.authorize_recovery(value, scope, **kwargs)
    assert recovery.authorize_recovery(value, scope, **kwargs) == reference
    assert {name: (scope.directory / name).read_bytes() for name in before} == before
    child = recovery.resolve_audit_recovery(value, scope)
    assert child.directory != scope.directory
    assert child.load_resident_speed() == speed
    assert child.load_audit(operation).audit_witnesses == ((audit.selected_delta_digest, audit),)
    assert child.load_final() is None
    assert child.load_quality() is None
    assert child.recovery_reference == reference


@pytest.mark.parametrize("failure", ["budget", "clock_reset"])
def test_unfinishable_resume_stops_before_audit_or_reference(monkeypatch, failure):
    harness, baseline, _reference, exits = _resident_case(monkeypatch)
    if failure == "clock_reset":
        harness.executor.manager.clock = lambda: 2.0
    else:
        original = runner.ResidentSpeedWitness.from_evidence
        def over_budget(*args):
            speed = original(*args)
            speed.started_monotonic_s = -20000.0
            return speed
        monkeypatch.setattr(runner.ResidentSpeedWitness, "from_evidence", over_budget)
    with pytest.raises(QualificationContinuationError, match="clock predates|exhausted"):
        _run_resident_harness(harness, baseline)
    assert not exits
    assert harness.published_attempt is None
    assert "reference" not in harness.calls

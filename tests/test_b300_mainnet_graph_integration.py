"""Caller integration for the B300 mainnet worker's pre-execution graph gate."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.eval.b300_mainnet_worker as worker_module
import tests.test_b300_mainnet_worker as mainnet_fixtures
from cacheon.arena_service import ArenaQualificationWork
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldReason
from cacheon.eval.b300_mainnet_worker import (
    B300MainnetWorker,
    B300RemoteQualificationRun,
)
from cacheon.eval.b300_resident_qualification import (
    B300ResidentQualificationError,
)
from cacheon.eval.b300_qualification_graph_gate import (
    B300QualificationGraphGateHold,
)
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
)
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import (
    QualificationContinuationError,
    QualificationContinuationStore,
)
from cacheon.eval.qualification_intake import QualificationPlanFactory
from tests.test_b300_qualification_graph_gate import _factory
from tests.test_qualification_graph_exit import _plan


executor_factory = mainnet_fixtures.executor_factory


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass
class _Case:
    worker: B300MainnetWorker
    authorities: object
    resident: object
    candidate: object
    receipt: object
    lease: EvaluationLease
    continuation: QualificationContinuationStore
    plan: object
    factory: QualificationPlanFactory
    authority: object
    work: ArenaQualificationWork


def _case(
    tmp_path: Path,
    executor_factory,
    *,
    failure: bool,
) -> _Case:
    authorities, resident, _builder = mainnet_fixtures._authorities(
        tmp_path / "worker",
        executor_factory,
    )
    manifest = mainnet_fixtures._manifest(authorities)
    readiness = mainnet_fixtures._readiness(manifest, authorities)
    harness, plan, authority = _plan(tmp_path / "graph", failure=failure)
    factory = _factory(harness, plan)
    candidate = harness.candidate
    receipt = mainnet_fixtures._promoted_receipt(manifest, candidate)
    lease = EvaluationLease(
        _h("graph-worker-lease:" + candidate.reservation.reservation_digest),
        1,
        "qualification",
        "graph-gate-worker-test",
        (
            EvaluationLeaseMember(
                candidate.reservation.reservation_digest,
                "promoted",
            ),
        ),
        20,
        40,
        40,
    )
    work = ArenaQualificationWork(
        factory,
        authorities.executor,
        authorities.entropy_provider,
        authorities.hidden_judge,
        time.monotonic() + 60.0,
        manifest.qualification_policy_digest,
        authorities.resident_baseline_executor,
    )
    worker = B300MainnetWorker(manifest, authorities, readiness)
    worker._bind_remote_qualification_graph_gate_root(plan.evidence_root)
    return _Case(
        worker,
        authorities,
        resident,
        candidate,
        receipt,
        lease,
        QualificationContinuationStore(tmp_path / "continuation"),
        plan,
        factory,
        authority,
        work,
    )


def _install_plan(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, object, object]]:
    calls: list[tuple[object, object, object]] = []

    def plan(candidates, receipts, *, state=None):
        calls.append((candidates, receipts, state))
        assert candidates == (case.candidate,)
        assert receipts == (case.receipt,)
        return case.work

    monkeypatch.setattr(case.worker.service, "plan_qualification", plan)
    return calls


def _run(case: _Case):
    return case.worker.run_remote_qualification(
        case.lease,
        (case.candidate,),
        (case.receipt,),
        screen_lane="primary",
        continuation_store=case.continuation,
        request_digest=_h("authenticated-worker-graph-request"),
    )


def _install_resident_bridge(monkeypatch):
    calls = []
    prefix = SimpleNamespace(
        speed_plan=object(),
        speed=object(),
        retirement=object(),
        count_result=None,
        count_checkpoint=None,
    )
    lifecycle = SimpleNamespace(closure=None)

    def resident_prefix(**kwargs):
        calls.append(kwargs)
        return prefix

    monkeypatch.setattr(
        worker_module,
        "run_b300_resident_qualification_prefix",
        resident_prefix,
    )
    monkeypatch.setattr(
        worker_module,
        "ResidentPairMarginalLifecycleEvidence",
        lambda *args: (calls.append(args), lifecycle)[1],
    )
    return calls, lifecycle


def test_graph_pass_reuses_one_plan_callback_and_exact_factory(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory, failure=False)
    plan_calls = _install_plan(case, monkeypatch)
    gate_calls = []
    intake_calls = []
    original_gate = worker_module.run_b300_qualification_graph_gate

    def gate(factory, plan, **kwargs):
        gate_calls.append((factory, plan, kwargs))
        return original_gate(factory, plan, **kwargs)

    def intake(factory, **kwargs):
        intake_calls.append((factory, kwargs))
        return mainnet_fixtures._systemic_batch(factory)

    monkeypatch.setattr(worker_module, "run_b300_qualification_graph_gate", gate)
    resident_calls, lifecycle = _install_resident_bridge(monkeypatch)
    monkeypatch.setattr(worker_module, "run_qualification_intake", intake)
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300RemoteQualificationRun
    assert len(plan_calls) == 1
    assert len(gate_calls) == 1
    assert gate_calls[0][0] is case.factory
    assert gate_calls[0][1] is case.plan
    assert len(intake_calls) == 1
    assert intake_calls[0][0] is case.factory
    assert len(resident_calls) == 2
    assert resident_calls[0]["plan"] is case.plan
    assert resident_calls[1][0] is case.plan.prepared
    assert intake_calls[0][1]["prebuilt_plan"] is case.plan
    assert intake_calls[0][1]["resident_pair_lifecycle"] is lifecycle
    assert result.supporting_evidence_refs == (
        case.authority.graph_artifact_ref,
    )
    assert case.resident.created == 0


def test_durable_resident_ambiguity_returns_authenticated_hold(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory, failure=False)
    _install_plan(case, monkeypatch)
    resident_calls, _lifecycle = _install_resident_bridge(monkeypatch)

    def interrupted(*_args, **_kwargs):
        raise QualificationContinuationError("durable resident state is partial")

    monkeypatch.setattr(worker_module, "run_qualification_intake", interrupted)
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300QualificationGraphGateHold
    assert result.reason is RemoteQualificationHoldReason.RESIDENT_EVIDENCE_UNAVAILABLE
    assert len(resident_calls) == 2


def test_resident_authority_mismatch_is_an_authenticated_hold(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory, failure=False)
    _install_plan(case, monkeypatch)
    intake_calls = []
    monkeypatch.setattr(
        worker_module,
        "run_b300_resident_qualification_prefix",
        lambda **_kwargs: (_ for _ in ()).throw(
            B300ResidentQualificationError("foreign pair authority")
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "run_qualification_intake",
        lambda *args, **kwargs: intake_calls.append((args, kwargs)),
    )
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300QualificationGraphGateHold
    assert result.reason is RemoteQualificationHoldReason.RESIDENT_EVIDENCE_UNAVAILABLE
    assert intake_calls == []


def test_graph_fail_returns_terminal_without_intake_pair_or_settlement(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory, failure=True)
    plan_calls = _install_plan(case, monkeypatch)
    intake_calls = []
    execute_calls = []

    def forbidden_intake(*args, **kwargs):
        intake_calls.append((args, kwargs))
        raise AssertionError("graph FAIL must stop before qualification intake")

    def forbidden_execute(*args, **kwargs):
        execute_calls.append((args, kwargs))
        raise AssertionError("graph FAIL must stop before resident execution")

    monkeypatch.setattr(worker_module, "run_qualification_intake", forbidden_intake)
    monkeypatch.setattr(OCIEngineExecutor, "execute", forbidden_execute)
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300RemoteQualificationRun
    assert len(plan_calls) == 1
    assert intake_calls == []
    assert execute_calls == []
    assert result.run.disposition == "completed"
    assert len(result.run.payload.outcomes) == 1
    outcome = result.run.payload.outcomes[0]
    assert outcome.decision is QualificationDecision.FAIL
    assert outcome.retryable is False
    assert outcome.settlement_qualification is None
    assert result.run.payload.retry_plan is None
    assert case.authority.graph_artifact_ref in result.supporting_evidence_refs
    assert result.run.payload.attempt_ref in result.supporting_evidence_refs
    assert case.resident.created == 0


def test_graph_missing_hold_returns_before_intake_or_resident_execution(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory, failure=False)
    changed_authority = replace(
        case.authority,
        graph_artifact_ref=replace(
            case.authority.graph_artifact_ref,
            sha256=_h("missing-worker-raw-graph"),
        ),
    )
    changed_plan = replace(case.plan, candidates=(changed_authority,))
    harness = type("Harness", (), {"candidate": case.candidate})()
    changed_factory = _factory(harness, changed_plan)
    case.plan = changed_plan
    case.factory = changed_factory
    case.authority = changed_authority
    case.work = replace(case.work, factory=changed_factory)
    plan_calls = _install_plan(case, monkeypatch)
    intake_calls = []
    execute_calls = []
    monkeypatch.setattr(
        worker_module,
        "run_qualification_intake",
        lambda *args, **kwargs: intake_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        OCIEngineExecutor,
        "execute",
        lambda *args, **kwargs: execute_calls.append((args, kwargs)),
    )
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300QualificationGraphGateHold
    assert result.reason is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
    assert len(plan_calls) == 1
    assert intake_calls == []
    assert execute_calls == []
    assert case.resident.created == 0


def test_provider_graph_hold_is_typed_and_never_becomes_no_decision(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory, failure=False)
    plan_calls = 0
    intake_calls = []

    def held(_candidates, _receipts, *, state=None):
        nonlocal plan_calls
        plan_calls += 1
        assert state is None
        raise B300QualificationGraphEvidenceHold("armed graph evidence is unavailable")

    monkeypatch.setattr(case.worker.service, "plan_qualification", held)
    monkeypatch.setattr(
        worker_module,
        "run_qualification_intake",
        lambda *args, **kwargs: intake_calls.append((args, kwargs)),
    )
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300QualificationGraphGateHold
    assert result.reason is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
    assert plan_calls == 1
    assert intake_calls == []
    assert case.resident.created == 0

"""Authenticated remote-product integration for the B300 graph gate."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.eval.b300_mainnet_worker as worker_module
import cacheon.eval.b300_qualification_graph_gate as gate_module
import cacheon.eval.b300_remote_qualification_adapter as adapter_module
import tests.test_b300_qualification_deployment as deployment_fixtures
import tests.test_b300_remote_qualification_adapter as adapter_fixtures
from cacheon.arena_service import ArenaQualificationWork
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteQualificationProduct,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    reopen_remote_response,
    seal_remote_response,
)
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldProduct,
    RemoteQualificationHoldReason,
)
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_graph_exit import QualificationGraphExitHold
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationPlanFactory,
)
from cacheon.eval.evidence_store import publish_evidence
from tests.test_b300_qualification_graph_gate import _factory
from tests.test_marginal_runtime import FUSED
from tests.test_qualification_graph_exit import _plan, _raw, _with_raw


configured = adapter_fixtures.configured


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass
class _RemoteCase:
    configured: adapter_fixtures._Configured
    adapter: adapter_module.B300RemoteQualificationAdapter
    candidate: object
    receipt: object
    plan: object
    factory: QualificationPlanFactory
    authority: object
    work: ArenaQualificationWork


def _case(
    configured: adapter_fixtures._Configured,
    *,
    failure: bool,
    source_fixture: Path | None = None,
) -> _RemoteCase:
    harness, plan, authority = _plan(
        configured.construction.evidence_root.parent,
        source_fixture=source_fixture,
        failure=failure,
        evidence_root=configured.construction.evidence_root,
    )
    assert plan.evidence_root == configured.construction.evidence_root
    factory = _factory(harness, plan)
    candidate = harness.candidate
    receipt = deployment_fixtures._receipt(
        configured.deployment.manifest.digest,
        candidate,
    )
    authorities = configured.deployment.authorities
    work = ArenaQualificationWork(
        factory,
        authorities.executor,
        authorities.entropy_provider,
        authorities.hidden_judge,
        time.monotonic() + 60.0,
        configured.deployment.manifest.qualification_policy_digest,
        authorities.resident_baseline_executor,
    )
    adapter = adapter_module.B300RemoteQualificationAdapter(
        configured.deployment,
        configured.construction,
        configured.readiness,
        adapter_module.B300WorkerBundleResolver((candidate.publication,)),
        configured.adapter.continuation_store,
        configured.adapter.worker,
    )
    return _RemoteCase(
        configured,
        adapter,
        candidate,
        receipt,
        plan,
        factory,
        authority,
        work,
    )


def _replace_plan(case: _RemoteCase, plan, authority) -> None:
    harness = type("Harness", (), {"candidate": case.candidate})()
    factory = _factory(harness, plan)
    case.plan = plan
    case.authority = authority
    case.factory = factory
    case.work = replace(case.work, factory=factory)


def _install_plan(
    case: _RemoteCase,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, object, object]]:
    calls: list[tuple[object, object, object]] = []
    worker = case.adapter.worker
    assert worker is not None

    def plan(candidates, receipts, *, state=None):
        calls.append((candidates, receipts, state))
        assert candidates == (case.candidate,)
        assert receipts == (case.receipt,)
        return case.work

    monkeypatch.setattr(worker.service, "plan_qualification", plan)
    return calls


def _request(case: _RemoteCase):
    body = adapter_fixtures._body(
        case.configured,
        candidate=case.candidate,
        receipt=case.receipt,
    )
    return adapter_fixtures._request(
        case.configured,
        candidate=case.candidate,
        body=body,
    )


def _transport(case: _RemoteCase):
    readiness = case.configured.readiness
    credential = RemoteWorkerCredential("worker-credential", b"c" * 32)
    identity = RemoteWorkerTransportIdentity(
        "fixed-spool",
        _h("endpoint"),
        REMOTE_EVALUATION_PROTOCOL_DIGEST,
        credential.digest,
        readiness.service_digest,
        readiness.digest,
    )
    return identity, credential


def _reopen_authenticated(case: _RemoteCase, request, product):
    identity, credential = _transport(case)
    response = seal_remote_response(request, product, identity, credential)
    return reopen_remote_response(request, response, identity, credential)


def test_full_pass_keeps_existing_product_and_adds_raw_graph_inventory(
    configured: adapter_fixtures._Configured,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(configured, failure=False)
    plan_calls = _install_plan(case, monkeypatch)
    intake_calls = []
    attempt = publish_evidence(
        configured.construction.evidence_root,
        b'{"complete":"qualification-pass"}',
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="cacheon.qualification.cohort-attempt.v1",
    )

    def intake(factory, **kwargs):
        intake_calls.append((factory, kwargs))
        reservation = factory.manifest.reservations[0]
        outcome = QualificationIntakeOutcome(
            reservation.reservation_digest,
            reservation.selected_delta_digest,
            factory.manifest.digest,
            QualificationDecision.PASS,
            "qualification_pass",
            False,
            attempt.sha256,
            _h("complete-qualification-report"),
        )
        return QualificationIntakeBatch(factory.manifest.digest, (outcome,), attempt)

    prefix = SimpleNamespace(
        speed_plan=object(),
        speed=object(),
        retirement=object(),
        count_result=None,
        count_checkpoint=None,
    )
    lifecycle = SimpleNamespace(closure=None)
    monkeypatch.setattr(worker_module, "run_qualification_intake", intake)
    monkeypatch.setattr(
        worker_module,
        "run_b300_resident_qualification_prefix",
        lambda **_kwargs: prefix,
    )
    monkeypatch.setattr(
        worker_module,
        "ResidentPairMarginalLifecycleEvidence",
        lambda *_args: lifecycle,
    )
    request = _request(case)
    try:
        product = case.adapter.run(request)
    finally:
        configured.adapter.close()

    assert type(product) is RemoteQualificationProduct
    assert len(plan_calls) == 1
    assert len(intake_calls) == 1
    assert intake_calls[0][1]["prebuilt_plan"] is case.plan
    assert intake_calls[0][1]["resident_pair_lifecycle"] is lifecycle
    assert intake_calls[0][0] is case.factory
    assert product.batch.outcomes[0].decision is QualificationDecision.PASS
    assert product.evidence_inventory == tuple(
        sorted(
            (attempt, case.authority.graph_artifact_ref),
            key=lambda row: (
                row.domain,
                row.sha256,
                row.media_type,
                row.schema,
            ),
        )
    )
    assert _reopen_authenticated(case, request, product) == product


def test_atomic_graph_fail_is_sealed_with_raw_and_exit_evidence_only(
    configured: adapter_fixtures._Configured,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(configured, failure=True, source_fixture=FUSED)
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
    request = _request(case)
    try:
        product = case.adapter.run(request)
    finally:
        configured.adapter.close()

    assert type(product) is RemoteQualificationProduct
    assert len(plan_calls) == 1
    assert intake_calls == []
    assert execute_calls == []
    assert len(product.batch.outcomes) == 1
    outcome = product.batch.outcomes[0]
    assert outcome.decision is QualificationDecision.FAIL
    assert outcome.retryable is False
    assert outcome.settlement_qualification is None
    assert product.batch.retry_plan is None
    assert product.batch.attempt_ref in product.evidence_inventory
    assert case.authority.graph_artifact_ref in product.evidence_inventory
    assert len(product.evidence_inventory) == 2
    assert tuple(
        row.slot_id for row in case.authority.graph_requirement.binding.members
    ) == case.candidate.reservation.target_members
    assert len(case.candidate.reservation.target_members) > 1
    assert _reopen_authenticated(case, request, product) == product


@pytest.mark.parametrize(
    ("condition", "expected_reason"),
    (
        ("missing", RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE),
        ("incomplete", RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE),
        (
            "publication",
            RemoteQualificationHoldReason.GRAPH_EXIT_PUBLICATION_AMBIGUOUS,
        ),
    ),
)
def test_each_typed_graph_hold_returns_normal_authenticated_response(
    configured: adapter_fixtures._Configured,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    expected_reason: RemoteQualificationHoldReason,
) -> None:
    case = _case(configured, failure=condition == "publication")
    if condition == "missing":
        authority = replace(
            case.authority,
            graph_artifact_ref=replace(
                case.authority.graph_artifact_ref,
                sha256=_h("missing-remote-raw-graph"),
            ),
        )
        _replace_plan(case, replace(case.plan, candidates=(authority,)), authority)
    elif condition == "incomplete":
        raw = _raw(case.plan, case.authority)
        member = raw.members[0]
        variant = member.variants[0]
        shape = replace(variant.shapes[0], graph_replays=2)
        variant = replace(variant, shapes=(shape, *variant.shapes[1:]))
        member = replace(member, variants=(variant, *member.variants[1:]))
        raw = replace(raw, members=(member, *raw.members[1:]))
        plan, authority = _with_raw(case.plan, case.authority, raw)
        _replace_plan(case, plan, authority)
    else:
        monkeypatch.setattr(
            gate_module,
            "publish_qualification_graph_exit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                QualificationGraphExitHold("typed publication ambiguity")
            ),
        )
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
    request = _request(case)
    try:
        product = case.adapter.run(request)
    finally:
        configured.adapter.close()

    assert type(product) is RemoteQualificationHoldProduct
    assert product.reason is expected_reason
    assert product.request_digest == request.digest
    assert product.reservation_digests == (
        case.candidate.reservation.reservation_digest,
    )
    assert product.selected_delta_digests == (
        case.candidate.reservation.selected_delta_digest,
    )
    assert product.candidate_digests == (case.candidate.digest,)
    assert len(plan_calls) == 1
    assert intake_calls == []
    assert execute_calls == []
    assert _reopen_authenticated(case, request, product) == product

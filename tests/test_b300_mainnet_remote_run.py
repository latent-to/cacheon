"""The B300 mainnet worker's remote run over one exact prebuilt qualification plan."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.eval.b300_mainnet_worker as worker_module
import tests.test_b300_mainnet_worker as mainnet_fixtures
from cacheon.arena_service import ArenaQualificationWork
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldReason,
    RemoteQualificationWorkerHold,
)
from cacheon.eval.b300_mainnet_worker import (
    B300MainnetWorker,
    B300RemoteQualificationRun,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import (
    QualificationContinuationError,
    QualificationContinuationStore,
)
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationPlanFactory,
)
from cacheon.eval.qualification_prebuilt_plan import (
    sealed_prebuilt_qualification_plan_factory,
)
from tests.test_b300_registered_qualification import _candidate_source, _harness


executor_factory = mainnet_fixtures.executor_factory


def _plan(tmp_path: Path, *, source_fixture: Path | None = None):
    harness = _harness(tmp_path, source_fixture)
    value = harness.factory.plan_builder(harness.cohort, b"g" * 32)
    return harness, value, value.candidates[0]


def _factory(harness, plan):
    reference = _h("remote-run-selection-reference")
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


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(label: str) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        "qualification.supporting", _h(label), 1, "application/json", label
    )


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
    source_fixture: Path | None = None,
) -> _Case:
    authorities, resident, _builder = mainnet_fixtures._authorities(
        tmp_path / "worker",
        executor_factory,
    )
    manifest = mainnet_fixtures._manifest(authorities)
    readiness = mainnet_fixtures._readiness(manifest, authorities)
    harness, plan, authority = _plan(tmp_path / "plan", source_fixture=source_fixture)
    factory = _factory(harness, plan)
    candidate = harness.candidate
    receipt = mainnet_fixtures._promoted_receipt(manifest, candidate)
    lease = EvaluationLease(
        _h("remote-run-lease:" + candidate.reservation.reservation_digest),
        1,
        "qualification",
        "remote-run-worker-test",
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
        request_digest=_h("authenticated-worker-remote-request"),
    )


def test_remote_run_reuses_one_plan_callback_and_exact_factory(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory)
    plan_calls = _install_plan(case, monkeypatch)
    intake_calls = []

    def intake(factory, **kwargs):
        intake_calls.append((factory, kwargs))
        return mainnet_fixtures._systemic_batch(factory)

    monkeypatch.setattr(worker_module, "run_qualification_intake", intake)
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300RemoteQualificationRun
    assert len(plan_calls) == 1
    assert len(intake_calls) == 1
    assert intake_calls[0][0] is case.factory
    assert intake_calls[0][1]["prebuilt_plan"] is case.plan
    assert result.supporting_evidence_refs == ()
    assert case.resident.created == 0


def test_native_rebuild_uses_dedicated_candidate_launch(
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _candidate_source(tmp_path / "native-source")
    (native / "kernels" / "native.cu").write_text(
        'extern "C" __global__ void noop() {}\n'
    )
    manifest = (native / "manifest.toml").read_text()
    (native / "manifest.toml").write_text(
        manifest.rstrip("\n") + '\ncuda_sources = ["kernels/native.cu"]\n'
    )
    (native / "rebuild.json").write_text(
        '{"steps":[{"type":"repo_python","path":"build_cuda_ext.py"}]}\n'
    )
    for path in sorted(native.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    case = _case(tmp_path, executor_factory, source_fixture=native)
    _install_plan(case, monkeypatch)
    intake_calls = []
    attempt_ref = EvidenceArtifactRef(
        "cacheon.qualification.cohort-attempt.v1",
        _h("native-direct-attempt"),
        1,
        "application/json",
        worker_module.ATTEMPT_SCHEMA_V3,
    )
    raw_quality_ref = _ref("native-direct-raw-quality")

    def intake(factory, **kwargs):
        intake_calls.append((factory, kwargs))
        reservation = factory.manifest.reservations[0]
        return QualificationIntakeBatch(
            factory.manifest.digest,
            (
                QualificationIntakeOutcome(
                    reservation.reservation_digest,
                    reservation.selected_delta_digest,
                    factory.manifest.digest,
                    QualificationDecision.FAIL,
                    "speed_regression",
                    False,
                    attempt_artifact_sha256=attempt_ref.sha256,
                    report_digest=_h("native-direct-report"),
                ),
            ),
            attempt_ref,
        )

    monkeypatch.setattr(worker_module, "run_qualification_intake", intake)
    monkeypatch.setattr(
        worker_module,
        "reopen_causal_qualification",
        lambda *_args, **_kwargs: SimpleNamespace(
            audit_recovery=None,
            reports=(
                SimpleNamespace(
                    raw_quality_artifact=raw_quality_ref,
                ),
            )
        ),
    )
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is B300RemoteQualificationRun
    assert len(intake_calls) == 1
    assert intake_calls[0][0] is case.factory
    kwargs = intake_calls[0][1]
    assert kwargs["prebuilt_plan"] is case.plan
    assert kwargs["executor"] is case.authorities.executor
    assert (
        kwargs["resident_baseline_executor"]
        is case.authorities.resident_baseline_executor
    )
    assert raw_quality_ref in result.supporting_evidence_refs


@pytest.mark.parametrize("failure", (QualificationContinuationError("durable resident state is partial"), worker_module.OuterSessionInfrastructureError("session response read timed out")))
def test_durable_resident_ambiguity_returns_authenticated_hold(
    failure,
    tmp_path: Path,
    executor_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, executor_factory)
    _install_plan(case, monkeypatch)

    def interrupted(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(worker_module, "run_qualification_intake", interrupted)
    try:
        result = _run(case)
    finally:
        case.worker.close()

    assert type(result) is RemoteQualificationWorkerHold
    assert result.reason is RemoteQualificationHoldReason.RESIDENT_EVIDENCE_UNAVAILABLE
    assert result.failure_type == type(failure).__name__
    assert str(failure) in result.failure_message

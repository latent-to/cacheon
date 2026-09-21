from __future__ import annotations

from types import SimpleNamespace

import pytest

import cacheon.eval.qualification_intake as intake
from cacheon.eval.candidate_failure_product import reopen_candidate_failure
from cacheon.eval.evidence_store import EvidenceArtifactRef, prepare_evidence_root
from cacheon.eval.oci_backend import OCIBackendError
from cacheon.eval.oci_outer_session import (
    OuterSessionCandidateError,
    OuterSessionProcessError,
    OuterSessionWorkerError,
)
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_runner import (
    QualificationRunnerError,
    SpeedStageDisposition,
)
from cacheon.eval.scoring import RawSpeedEvidenceError


def _d(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _reservation(index: int, delta: str | None = None) -> intake.QualificationReservation:
    return intake.QualificationReservation(
        _d(f"reservation-{index}"),
        _d(f"submission-{index}"),
        f"target.{index}",
        delta or _d(f"delta-{index}"),
        index,
        f"miner-{index}",
        100 + index,
        index,
        0,
        (f"target.{index}",),
    )


def _fake_plan(monkeypatch, *, count: int = 2):
    class FakePlan:
        pass

    monkeypatch.setattr(intake, "CausalQualificationInput", FakePlan)
    source = SimpleNamespace()
    source.digest = _d("source")
    plan = FakePlan()
    plan.selection_secret = b"s" * 32
    prepared_candidates = tuple(
        SimpleNamespace(
            arm=SimpleNamespace(
                digest=_d(f"arm-{index}"),
                selected_delta_digest=_d(f"delta-{index}"),
            ),
            launch=SimpleNamespace(digest=_d(f"launch-{index}")),
        )
        for index in range(count)
    )
    plan.prepared = SimpleNamespace(source=source, candidates=prepared_candidates)
    plan.commitment = SimpleNamespace(digest=_d("commitment"))
    plan.candidates = tuple(
        SimpleNamespace(selected_delta_digest=_d(f"delta-{index}"))
        for index in range(count)
    )
    plan.evidence_root = SimpleNamespace()
    plan.speed_stage_disposition = SpeedStageDisposition.TERMINAL
    monkeypatch.setattr(
        intake, "qualification_authority_digest", lambda _value: _d("authority")
    )
    reservations = tuple(_reservation(index) for index in range(count))
    manifest = intake.QualificationAuthorityManifest.seal(
        plan,
        reservations=reservations,
        selection_secret_reference=_d("secret-reference"),
    )
    return plan, manifest


def _factory(plan, manifest):
    return intake.QualificationPlanFactory(
        manifest,
        lambda reference: (
            plan.selection_secret
            if reference == manifest.selection_secret_reference
            else b""
        ),
        lambda secret: plan,
    )


def test_authority_manifest_roundtrip_contains_only_private_secret_reference(
    monkeypatch,
) -> None:
    plan, manifest = _fake_plan(monkeypatch)

    encoded = manifest.to_dict()
    assert intake.QualificationAuthorityManifest.from_dict(encoded) == manifest
    assert manifest.digest == intake.QualificationAuthorityManifest.from_dict(
        encoded
    ).digest
    assert plan.selection_secret.hex() not in str(encoded)
    assert encoded["selection_secret_reference"] == _d("secret-reference")


def test_plan_factory_reopens_exact_secret_and_public_authority(monkeypatch) -> None:
    plan, manifest = _fake_plan(monkeypatch)
    factory = _factory(plan, manifest)

    assert factory.build() is plan

    substituted = intake.QualificationPlanFactory(
        manifest, lambda _reference: b"x" * 32, lambda _secret: plan
    )
    with pytest.raises(intake.QualificationIntakeError, match="substituted"):
        substituted.build()

    changed = SimpleNamespace(**plan.__dict__)
    changed.selection_secret = plan.selection_secret
    with pytest.raises(intake.QualificationIntakeError, match="untyped plan"):
        intake.QualificationPlanFactory(
            manifest, lambda _reference: plan.selection_secret, lambda _secret: changed
        ).build()


def test_prebuilt_plan_is_handed_to_the_runner_unchanged(monkeypatch) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=1)
    calls = []

    def run(value, **kwargs):
        calls.append((value, kwargs))
        raise QualificationRunnerError("stop after authority handoff")

    monkeypatch.setattr(intake, "run_causal_qualification", run)
    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        resident_baseline_executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
        prebuilt_plan=plan,
    )

    assert len(calls) == 1
    assert calls[0][0] is plan
    assert result.outcomes[0].reason == "qualification_runner"


class _FakeReport:
    def __init__(self, delta: str, decision: QualificationDecision, index: int):
        self.selected_delta_digest = delta
        self.decision = decision
        self.reason = {
            QualificationDecision.PASS: "qualified",
            QualificationDecision.FAIL: "speed_regression",
            QualificationDecision.NO_DECISION: "speed_noise",
        }[decision]
        self.retryable = decision is QualificationDecision.NO_DECISION
        self.digest = _d(f"report-{index}")


class _FakeAttempt:
    pass


class _FakeStageExit:
    def __init__(
        self,
        manifest: intake.QualificationAuthorityManifest,
        decision: QualificationDecision,
    ) -> None:
        self.authority_digest = manifest.authority_digest
        self.source_digest = manifest.source_digest
        self.selected_delta_digest = manifest.reservations[0].selected_delta_digest
        self.stage = "speed"
        self.decision = decision
        self.reason = (
            "speed_noise"
            if decision is QualificationDecision.NO_DECISION
            else "speed_regression"
        )
        self.digest = _d(f"stage-exit-{decision.value}")


def _install_success_runner(monkeypatch, manifest, decisions):
    reference = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        _d("attempt-artifact"),
        1,
        "application/json",
        "cacheon.qualification.cohort-attempt.v1",
    )
    reports = tuple(
        _FakeReport(delta, decision, index)
        for index, (delta, decision) in enumerate(
            zip(manifest.candidate_deltas, decisions, strict=True)
        )
    )
    attempt = _FakeAttempt()
    attempt.authority_digest = manifest.authority_digest
    attempt.source_digest = manifest.source_digest
    attempt.reports = reports
    monkeypatch.setattr(intake, "CohortQualificationAttempt", _FakeAttempt)
    monkeypatch.setattr(intake, "CandidateQualificationReport", _FakeReport)
    monkeypatch.setattr(
        intake, "run_causal_qualification", lambda *_args, **_kwargs: reference
    )
    monkeypatch.setattr(
        intake, "reopen_causal_qualification", lambda *_args, **_kwargs: attempt
    )
    return reference


@pytest.mark.parametrize(
    ("decision", "reason", "expects_retry"),
    (
        (QualificationDecision.FAIL, "speed_regression", False),
        (QualificationDecision.NO_DECISION, "speed_noise", True),
    ),
)
def test_speed_stage_exit_projects_terminal_outcome_without_settlement(
    monkeypatch,
    decision: QualificationDecision,
    reason: str,
    expects_retry: bool,
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=1)
    reference = EvidenceArtifactRef(
        "qualification.stage-exit",
        _d(f"stage-exit-artifact-{decision.value}"),
        1,
        "application/json",
        intake.STAGE_EXIT_SCHEMA,
    )
    terminal = _FakeStageExit(manifest, decision)
    resident_baseline_executor = object()
    runner_kwargs = {}

    def run(*_args, **kwargs):
        runner_kwargs.update(kwargs)
        return reference

    monkeypatch.setattr(intake, "QualificationStageExit", _FakeStageExit)
    monkeypatch.setattr(intake, "run_causal_qualification", run)
    monkeypatch.setattr(
        intake,
        "reopen_qualification_stage_exit",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        intake,
        "reopen_causal_qualification",
        lambda *_args, **_kwargs: pytest.fail(
            "a terminal stage exit must not reopen a full qualification attempt"
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        resident_baseline_executor=resident_baseline_executor,
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert runner_kwargs["resident_baseline_executor"] is resident_baseline_executor
    assert result.attempt_ref == reference
    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    assert outcome.decision is decision
    assert outcome.reason == reason
    assert outcome.retryable is expects_retry
    assert outcome.attempt_artifact_sha256 == reference.sha256
    assert outcome.report_digest == terminal.digest
    assert outcome.failure_digest is None
    assert outcome.settlement_qualification is None
    if expects_retry:
        assert result.retry_plan is not None
        assert result.retry_plan.strategy == "requeue"
        assert result.retry_plan.reservation_groups == (
            (manifest.reservations[0].reservation_digest,),
        )
    else:
        assert result.retry_plan is None


def test_intake_rejects_calibration_observation_before_runner(
    monkeypatch,
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=1)
    plan.speed_stage_disposition = SpeedStageDisposition.CALIBRATION_OBSERVATION
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: pytest.fail(
            "economic intake must reject calibration authority before the runner"
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        resident_baseline_executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert result.attempt_ref is None
    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    assert outcome.decision is QualificationDecision.NO_DECISION
    assert outcome.reason == "qualification_plan"
    assert outcome.retryable is True
    assert outcome.report_digest is None
    assert outcome.failure_digest is not None
    assert outcome.settlement_qualification is None


def test_batch_service_projects_per_reservation_tristate_and_retry(monkeypatch) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=3)
    reference = _install_success_runner(
        monkeypatch,
        manifest,
        (
            QualificationDecision.PASS,
            QualificationDecision.FAIL,
            QualificationDecision.NO_DECISION,
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert [row.decision for row in result.outcomes] == [
        QualificationDecision.PASS,
        QualificationDecision.FAIL,
        QualificationDecision.NO_DECISION,
    ]
    assert all(row.attempt_artifact_sha256 == reference.sha256 for row in result.outcomes)
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "requeue"
    assert result.retry_plan.reservation_groups == (
        (manifest.reservations[2].reservation_digest,),
    )


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (RawSpeedEvidenceError("zero throughput"), "raw_speed_evidence"),
        (QualificationRunnerError("T died"), "qualification_runner"),
        (
            OuterSessionProcessError("session ended before a complete response"),
            "outer_session_process",
        ),
        (OCIBackendError("runtime post-drain unavailable"), "oci_backend"),
    ],
)
def test_cohort_failure_is_no_decision_with_deterministic_bisection(
    monkeypatch, failure, reason
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=3)

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(intake, "run_causal_qualification", fail)
    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert {row.decision for row in result.outcomes} == {
        QualificationDecision.NO_DECISION
    }
    assert all(row.retryable and row.report_digest is None for row in result.outcomes)
    assert all(row.reason == reason for row in result.outcomes)
    assert result.attempt_ref is None
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "bisect"
    assert result.retry_plan.reservation_groups == (
        (manifest.reservations[0].reservation_digest,),
        tuple(row.reservation_digest for row in manifest.reservations[1:]),
    )


def test_direct_singleton_candidate_error_publishes_terminal_failure(
    tmp_path, monkeypatch
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=1)
    plan.evidence_root = prepare_evidence_root(tmp_path / "evidence")
    worker_error = OuterSessionCandidateError(
        "batch: CandidateExecutionFailure: rank 0 failed",
        candidate_failure="rank 0 RuntimeError at kernels/moe.py:17: boom",
    )
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(worker_error),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert result.retry_plan is None
    assert result.outcomes[0].decision is QualificationDecision.FAIL
    assert result.outcomes[0].reason == "candidate_exception"
    assert result.attempt_ref is not None
    product = reopen_candidate_failure(plan.evidence_root, result.attempt_ref)
    assert product["failure"].endswith("kernels/moe.py:17: boom")


@pytest.mark.parametrize("source", ("baseline", "reference"))
def test_shared_worker_error_reaches_the_authenticated_adapter_unchanged(
    monkeypatch, source: str
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=2)
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OuterSessionWorkerError(f"{source} worker raised")
        ),
    )

    with pytest.raises(
        OuterSessionWorkerError,
        match=rf"{source} worker raised",
    ):
        intake.run_qualification_intake(
            _factory(plan, manifest),
            executor=object(),
            entropy_provider=lambda *_args: None,
            hidden_judge=lambda **_kwargs: None,
            deadline=100.0,
        )


def test_unexpected_controller_failure_still_propagates(monkeypatch) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=2)
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("controller invariant failed")
        ),
    )

    with pytest.raises(RuntimeError, match="controller invariant failed"):
        intake.run_qualification_intake(
            _factory(plan, manifest),
            executor=object(),
            entropy_provider=lambda *_args: None,
            hidden_judge=lambda **_kwargs: None,
            deadline=100.0,
        )


@pytest.mark.parametrize(
    "failure",
    [
        intake.QualificationIntakeError("secret record unavailable"),
        OSError("private store unavailable"),
    ],
)
def test_factory_infrastructure_failure_is_typed_no_decision(
    monkeypatch, failure
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=2)

    def fail_secret(_reference):
        raise failure

    factory = intake.QualificationPlanFactory(manifest, fail_secret, lambda _secret: plan)
    result = intake.run_qualification_intake(
        factory,
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert all(
        row.decision is QualificationDecision.NO_DECISION
        and row.reason == "qualification_plan"
        for row in result.outcomes
    )
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "bisect"


def test_outcomes_and_batches_cannot_claim_evidence_free_pass() -> None:
    with pytest.raises(intake.QualificationIntakeError, match="PASS/FAIL"):
        intake.QualificationIntakeOutcome(
            _d("reservation"),
            _d("delta"),
            _d("authority"),
            QualificationDecision.PASS,
            "qualified",
            False,
        )

    reference = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        _d("attempt"),
        1,
        "application/json",
        "cacheon.qualification.cohort-attempt.v1",
    )
    outcome = intake.QualificationIntakeOutcome(
        _d("reservation"),
        _d("delta"),
        _d("authority"),
        QualificationDecision.FAIL,
        "quality_failed",
        False,
        attempt_artifact_sha256=reference.sha256,
        report_digest=_d("report"),
    )
    with pytest.raises(intake.QualificationIntakeError, match="internally inconsistent"):
        intake.QualificationIntakeBatch(_d("authority"), (outcome,))


def test_single_pass_outcome_cannot_smuggle_a_settlement_candidate() -> None:
    reference = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        _d("attempt"),
        1,
        "application/json",
        "cacheon.qualification.cohort-attempt.v1",
    )
    with pytest.raises(
        intake.QualificationIntakeError, match="settlement qualification"
    ):
        intake.QualificationIntakeOutcome(
            _d("reservation"),
            _d("delta"),
            _d("authority"),
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=reference.sha256,
            report_digest=_d("report"),
            settlement_qualification=object(),  # type: ignore[arg-type]
        )

"""Finalized-intake projection for causal qualification evidence.

This module is deliberately narrower than the qualification runner.  It binds an
already prepared validator-owned plan to finalized reservation identities and
projects a completed attempt (or a retryable cohort failure) into per-reservation
three-way outcomes.  It does not fetch submissions, execute candidate code, settle
scores, or publish weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from cacheon._strict import NODE_ADDRESS, require_digest, require_identifier, require_int

if TYPE_CHECKING:
    from cacheon.settlement import SettlementQualification

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.candidate_failure_product import (
    candidate_failure_batch,
)
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_runner import (
    CandidateQualificationReport,
    CausalQualificationInput,
    CohortQualificationAttempt,
    QualificationStageExit,
    QualificationRunnerError,
    STAGE_EXIT_SCHEMA,
    SpeedStageDisposition,
    qualification_authority_digest,
    reopen_causal_qualification,
    reopen_qualification_stage_exit,
    run_causal_qualification,
)
from cacheon.eval.oci_backend import OCIBackendError
from cacheon.eval.oci_outer_session import (
    OuterSessionCandidateError,
    OuterSessionProcessError,
)
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from cacheon.eval.scoring import RawSpeedEvidenceError
from cacheon.stack_identity import canonical_digest


AUTHORITY_SCHEMA_VERSION = 1
_LANES = frozenset({"registered"})
_RETRY_STRATEGIES = frozenset({"requeue", "bisect"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class QualificationIntakeError(ValueError):
    """Finalized qualification authority or evidence is inconsistent."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=QualificationIntakeError)


def _identifier(value: object, field: str) -> str:
    return require_identifier(
        value, field=field, error=QualificationIntakeError, pattern=_IDENTIFIER
    )


def _integer(value: object, field: str) -> int:
    return require_int(value, field=field, error=QualificationIntakeError, minimum=0)


@dataclass(frozen=True)
class QualificationReservation:
    """One finalized submission in immutable cohort order."""

    reservation_digest: str
    submission_digest: str
    target_id: str
    selected_delta_digest: str
    arrival_order: int
    hotkey: str
    finalized_block: int
    finalized_event_index: int
    finalized_event_subindex: int
    target_members: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "reservation_digest",
            "submission_digest",
            "selected_delta_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
        object.__setattr__(
            self, "arrival_order", _integer(self.arrival_order, "arrival_order")
        )
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or self.hotkey.strip() != self.hotkey
            or len(self.hotkey) > 256
            or any(char in self.hotkey for char in "\x00\r\n")
        ):
            raise QualificationIntakeError("reservation hotkey is malformed")
        for field in (
            "finalized_block",
            "finalized_event_index",
            "finalized_event_subindex",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        members = tuple(self.target_members)
        if (
            not members
            or members != tuple(sorted(set(members)))
            or not all(isinstance(m, str) and NODE_ADDRESS.fullmatch(m) for m in members)
        ):
            raise QualificationIntakeError("reservation target members are not canonical")
        object.__setattr__(self, "target_members", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "arrival_order": self.arrival_order,
            "finalized_block": self.finalized_block,
            "finalized_event_index": self.finalized_event_index,
            "finalized_event_subindex": self.finalized_event_subindex,
            "hotkey": self.hotkey,
            "reservation_digest": self.reservation_digest,
            "selected_delta_digest": self.selected_delta_digest,
            "submission_digest": self.submission_digest,
            "target_id": self.target_id,
            "target_members": list(self.target_members),
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationReservation":
        fields = {
            "arrival_order",
            "finalized_block",
            "finalized_event_index",
            "finalized_event_subindex",
            "hotkey",
            "reservation_digest",
            "selected_delta_digest",
            "submission_digest",
            "target_id",
            "target_members",
        }
        if type(value) is not dict or set(value) != fields:
            raise QualificationIntakeError("reservation fields do not match the schema")
        if type(value["target_members"]) is not list:
            raise QualificationIntakeError("reservation target members are malformed")
        return cls(**{**value, "target_members": tuple(value["target_members"])})  # type: ignore[arg-type]


@dataclass(frozen=True)
class QualificationAuthorityManifest:
    """Public identity for one private, validator-owned qualification plan.

    ``selection_secret_reference`` names a record in a private secret store.  The
    secret bytes themselves never enter this serializable object.
    """

    lane: str
    authority_digest: str
    source_digest: str
    commitment_digest: str
    selection_secret_reference: str
    candidate_deltas: tuple[str, ...]
    reservations: tuple[QualificationReservation, ...]
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.lane not in _LANES:
            raise QualificationIntakeError("qualification lane is unsupported")
        for field in (
            "authority_digest",
            "source_digest",
            "commitment_digest",
            "selection_secret_reference",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        deltas = tuple(_digest(row, "candidate delta") for row in self.candidate_deltas)
        reservations = tuple(self.reservations)
        if (
            type(self.schema_version) is not int
            or self.schema_version != AUTHORITY_SCHEMA_VERSION
            or not deltas
            or len(set(deltas)) != len(deltas)
            or any(type(row) is not QualificationReservation for row in reservations)
            or len(reservations) != len(deltas)
            or tuple(row.selected_delta_digest for row in reservations) != deltas
            or len({row.reservation_digest for row in reservations}) != len(reservations)
            or len({row.arrival_order for row in reservations}) != len(reservations)
        ):
            raise QualificationIntakeError(
                "qualification reservations do not exactly bind the candidate order"
            )
        object.__setattr__(self, "candidate_deltas", deltas)
        object.__setattr__(self, "reservations", reservations)

    @classmethod
    def seal(
        cls,
        value: CausalQualificationInput,
        *,
        reservations: tuple[QualificationReservation, ...],
        selection_secret_reference: str,
    ) -> "QualificationAuthorityManifest":
        if type(value) is not CausalQualificationInput:
            raise QualificationIntakeError("qualification plan is not exactly typed")
        return cls(
            "registered",
            qualification_authority_digest(value),
            value.prepared.source.digest,
            value.commitment.digest,
            selection_secret_reference,
            tuple(row.selected_delta_digest for row in value.candidates),
            reservations,
        )

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.qualification.intake-authority", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_digest": self.authority_digest,
            "candidate_deltas": list(self.candidate_deltas),
            "commitment_digest": self.commitment_digest,
            "lane": self.lane,
            "reservations": [row.to_dict() for row in self.reservations],
            "schema_version": self.schema_version,
            "selection_secret_reference": self.selection_secret_reference,
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationAuthorityManifest":
        fields = {
            "authority_digest",
            "candidate_deltas",
            "commitment_digest",
            "lane",
            "reservations",
            "schema_version",
            "selection_secret_reference",
            "source_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise QualificationIntakeError("authority manifest fields do not match")
        if type(value["candidate_deltas"]) is not list or type(value["reservations"]) is not list:
            raise QualificationIntakeError("authority manifest arrays are malformed")
        return cls(
            lane=value["lane"],  # type: ignore[arg-type]
            authority_digest=value["authority_digest"],  # type: ignore[arg-type]
            source_digest=value["source_digest"],  # type: ignore[arg-type]
            commitment_digest=value["commitment_digest"],  # type: ignore[arg-type]
            selection_secret_reference=value["selection_secret_reference"],  # type: ignore[arg-type]
            candidate_deltas=tuple(value["candidate_deltas"]),  # type: ignore[arg-type]
            reservations=tuple(
                QualificationReservation.from_dict(row)
                for row in value["reservations"]  # type: ignore[union-attr]
            ),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


SecretLoader = Callable[[str], bytes]
PlanBuilder = Callable[[bytes], CausalQualificationInput]


@dataclass(frozen=True)
class QualificationPlanFactory:
    """Resolve a private secret and reconstruct one exact public authority manifest."""

    manifest: QualificationAuthorityManifest
    secret_loader: SecretLoader
    plan_builder: PlanBuilder

    def __post_init__(self) -> None:
        if type(self.manifest) is not QualificationAuthorityManifest:
            raise QualificationIntakeError("plan factory manifest is not exactly typed")
        if not callable(self.secret_loader) or not callable(self.plan_builder):
            raise QualificationIntakeError("plan factory authorities must be callable")

    def build(self) -> CausalQualificationInput:
        secret = self.secret_loader(self.manifest.selection_secret_reference)
        if type(secret) is not bytes or len(secret) < 32:
            raise QualificationIntakeError("private selection secret is unavailable")
        value = self.plan_builder(secret)
        if type(value) is not CausalQualificationInput:
            raise QualificationIntakeError("plan builder returned an untyped plan")
        if value.selection_secret != secret:
            raise QualificationIntakeError("plan builder substituted the private secret")
        observed = QualificationAuthorityManifest.seal(
            value,
            reservations=self.manifest.reservations,
            selection_secret_reference=self.manifest.selection_secret_reference,
        )
        if observed != self.manifest:
            raise QualificationIntakeError("rebuilt qualification authority differs")
        return value


@dataclass(frozen=True)
class QualificationRetryPlan:
    authority_manifest_digest: str
    strategy: str
    reservation_groups: tuple[tuple[str, ...], ...]
    failure_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_manifest_digest",
            _digest(self.authority_manifest_digest, "authority manifest"),
        )
        object.__setattr__(self, "failure_digest", _digest(self.failure_digest, "failure"))
        if self.strategy not in _RETRY_STRATEGIES:
            raise QualificationIntakeError("retry strategy is unsupported")
        groups = tuple(tuple(group) for group in self.reservation_groups)
        flat = tuple(row for group in groups for row in group)
        if (
            not groups
            or any(not group for group in groups)
            or any(_digest(row, "retry reservation") != row for row in flat)
            or len(set(flat)) != len(flat)
            or (self.strategy == "bisect" and len(groups) != 2)
        ):
            raise QualificationIntakeError("retry groups are malformed")
        object.__setattr__(self, "reservation_groups", groups)


@dataclass(frozen=True)
class QualificationIntakeOutcome:
    reservation_digest: str
    selected_delta_digest: str
    authority_manifest_digest: str
    decision: QualificationDecision
    reason: str
    retryable: bool
    attempt_artifact_sha256: str | None = None
    report_digest: str | None = None
    failure_digest: str | None = None
    settlement_qualification: SettlementQualification | None = None

    def __post_init__(self) -> None:
        for field in (
            "reservation_digest",
            "selected_delta_digest",
            "authority_manifest_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.decision) is not QualificationDecision:
            raise QualificationIntakeError("outcome decision is not typed")
        object.__setattr__(self, "reason", _identifier(self.reason, "reason"))
        if type(self.retryable) is not bool or self.retryable != (
            self.decision is QualificationDecision.NO_DECISION
        ):
            raise QualificationIntakeError("outcome retryability disagrees with decision")
        for field in ("attempt_artifact_sha256", "report_digest", "failure_digest"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _digest(value, field))
        if (self.report_digest is None) != (self.attempt_artifact_sha256 is None):
            raise QualificationIntakeError("outcome report and attempt coverage differ")
        if self.failure_digest is not None and self.report_digest is not None:
            raise QualificationIntakeError("outcome cannot be both report and failure based")
        report_based = self.report_digest is not None
        failure_based = self.failure_digest is not None
        if self.decision is QualificationDecision.NO_DECISION:
            if report_based == failure_based:
                raise QualificationIntakeError(
                    "NO_DECISION must retain exactly one report or failure product"
                )
        elif not report_based or failure_based:
            raise QualificationIntakeError(
                "PASS/FAIL requires a complete attempt and report product"
            )
        from cacheon.settlement import SettlementQualification

        if self.settlement_qualification is not None:
            if type(self.settlement_qualification) is not SettlementQualification:
                raise QualificationIntakeError(
                    "settlement qualification is not exactly typed"
                )
            if self.decision is not QualificationDecision.PASS:
                raise QualificationIntakeError(
                    "non-PASS outcome cannot carry a settlement qualification"
                )
            if (
                self.settlement_qualification.reservation_digest != self.reservation_digest
                or self.settlement_qualification.selected_delta_digest
                != self.selected_delta_digest
                or self.settlement_qualification.qualification_authority_digest
                != self.authority_manifest_digest
                or self.settlement_qualification.qualification_attempt_digest
                != self.attempt_artifact_sha256
                or self.settlement_qualification.qualification_report_digest
                != self.report_digest
            ):
                raise QualificationIntakeError(
                    "settlement qualification differs from qualification outcome"
                )


@dataclass(frozen=True)
class QualificationIntakeBatch:
    authority_manifest_digest: str
    outcomes: tuple[QualificationIntakeOutcome, ...]
    attempt_ref: EvidenceArtifactRef | None = None
    retry_plan: QualificationRetryPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_manifest_digest",
            _digest(self.authority_manifest_digest, "authority manifest"),
        )
        outcomes = tuple(self.outcomes)
        retry_ids = (
            tuple(
                reservation
                for group in self.retry_plan.reservation_groups
                for reservation in group
            )
            if self.retry_plan is not None
            else ()
        )
        no_decision_ids = tuple(
            row.reservation_digest
            for row in outcomes
            if row.decision is QualificationDecision.NO_DECISION
        )
        if (
            not outcomes
            or any(type(row) is not QualificationIntakeOutcome for row in outcomes)
            or any(row.authority_manifest_digest != self.authority_manifest_digest for row in outcomes)
            or len({row.reservation_digest for row in outcomes}) != len(outcomes)
            or (self.attempt_ref is not None and type(self.attempt_ref) is not EvidenceArtifactRef)
            or (self.retry_plan is not None and type(self.retry_plan) is not QualificationRetryPlan)
            or (
                self.retry_plan is not None
                and self.retry_plan.authority_manifest_digest
                != self.authority_manifest_digest
            )
            or (
                self.attempt_ref is None
                and any(row.attempt_artifact_sha256 is not None for row in outcomes)
            )
            or (
                self.attempt_ref is not None
                and any(
                    row.attempt_artifact_sha256 != self.attempt_ref.sha256
                    for row in outcomes
                )
            )
            or retry_ids != no_decision_ids
        ):
            raise QualificationIntakeError("qualification batch is internally inconsistent")
        object.__setattr__(self, "outcomes", outcomes)


def _failure_digest(manifest: QualificationAuthorityManifest, exc: BaseException) -> str:
    digest = canonical_digest(
        "cacheon.qualification.intake-failure",
        {
            "authority_manifest_digest": manifest.digest,
            "exception": type(exc).__name__,
            "message": str(exc)[:4096],
        },
    )
    # The digest binds the failure text but does not carry it, and the batch
    # crosses the trust boundary carrying only NO_DECISION plus this digest.
    # Pre-plan failures have no evidence root yet. Emit their exact text to the
    # worker diagnostic stream, which the OCI manager now retains and ``explain
    # --evidence-dir`` renders, instead of maintaining a second optional ledger.
    import sys

    print(
        "CACHEON-QUALIFICATION-INTAKE-FAILURE: "
        f"authority={manifest.digest[:16]} failure={digest[:16]} "
        f"{type(exc).__name__}: {str(exc)[:2048]}",
        flush=True,
        file=sys.stderr,
    )
    return digest


def _retry_plan(
    manifest: QualificationAuthorityManifest,
    reservations: tuple[QualificationReservation, ...],
    failure_digest: str,
    *,
    bisect: bool,
) -> QualificationRetryPlan:
    ids = tuple(row.reservation_digest for row in reservations)
    if bisect and len(ids) > 1:
        midpoint = len(ids) // 2
        groups = (ids[:midpoint], ids[midpoint:])
        strategy = "bisect"
    else:
        groups = tuple((row,) for row in ids)
        strategy = "requeue"
    return QualificationRetryPlan(manifest.digest, strategy, groups, failure_digest)


def _no_decision_batch(
    manifest: QualificationAuthorityManifest,
    exc: BaseException,
    *,
    reason: str,
) -> QualificationIntakeBatch:
    failure = _failure_digest(manifest, exc)
    outcomes = tuple(
        QualificationIntakeOutcome(
            row.reservation_digest,
            row.selected_delta_digest,
            manifest.digest,
            QualificationDecision.NO_DECISION,
            reason,
            True,
            failure_digest=failure,
        )
        for row in manifest.reservations
    )
    return QualificationIntakeBatch(
        manifest.digest,
        outcomes,
        retry_plan=_retry_plan(
            manifest,
            manifest.reservations,
            failure,
            bisect=manifest.lane == "registered",
        ),
    )


def _settlement_projection(
    reservation,
    prepared,
    report,
    authority,
    attempt_ref,
    attempt,
):
    if report.decision is not QualificationDecision.PASS:
        return None
    from cacheon.settlement import SettlementQualification

    return SettlementQualification.from_qualification(
        reservation_digest=reservation.reservation_digest,
        finalized_block=reservation.finalized_block,
        event_index=reservation.finalized_event_index,
        event_subindex=reservation.finalized_event_subindex,
        hotkey=reservation.hotkey,
        target_id=reservation.target_id,
        members=reservation.target_members,
        prepared=prepared,
        report=report,
        authority=authority,
        attempt_ref=attempt_ref,
        attempt=attempt,
    )


def run_qualification_intake(
    factory: QualificationPlanFactory,
    *,
    executor,
    resident_baseline_executor=None,
    entropy_provider,
    hidden_judge,
    deadline: float,
    continuation_store: QualificationContinuationStore | None = None,
    request_digest: str | None = None,
    prebuilt_plan: CausalQualificationInput | None = None,
) -> QualificationIntakeBatch:
    """Run, reopen, and project one finalized cohort without settlement authority.

    With a ``continuation_store``, durable speed/quality/final products are
    written and consumed at the runner's sealed boundaries.  A continuation
    error is a HOLD: it deliberately escapes the NO_DECISION retry mapping
    below, because retrying could re-execute an already-durable expensive
    stage.
    """

    if type(factory) is not QualificationPlanFactory:
        raise QualificationIntakeError("qualification factory is not exactly typed")
    if continuation_store is not None and type(continuation_store) is not (
        QualificationContinuationStore
    ):
        raise QualificationIntakeError("continuation store is not exactly typed")
    if (continuation_store is None) != (request_digest is None):
        raise QualificationIntakeError(
            "continuation store requires one exact authenticated request digest"
        )
    if request_digest is not None:
        request_digest = _digest(request_digest, "authenticated request digest")
    if (
        (
            prebuilt_plan is not None
            and type(prebuilt_plan) is not CausalQualificationInput
        )
    ):
        raise QualificationIntakeError(
            "prebuilt qualification intake authorities are not exact"
        )
    manifest = factory.manifest
    try:
        value = prebuilt_plan if prebuilt_plan is not None else factory.build()
        if prebuilt_plan is not None:
            observed = QualificationAuthorityManifest.seal(
                value,
                reservations=manifest.reservations,
                selection_secret_reference=manifest.selection_secret_reference,
            )
            if observed != manifest:
                raise QualificationIntakeError(
                    "prebuilt qualification authority differs"
                )
        if value.speed_stage_disposition is not SpeedStageDisposition.TERMINAL:
            raise QualificationIntakeError(
                "economic qualification cannot use calibration speed continuation"
            )
    except (QualificationIntakeError, QualificationRunnerError, OSError) as exc:
        return _no_decision_batch(manifest, exc, reason="qualification_plan")
    continuation = None
    if continuation_store is not None:
        continuation = continuation_store.scope(
            request_digest=request_digest,
            authority_digest=qualification_authority_digest(value),
            source_digest=value.prepared.source.digest,
        )
    try:
        runner_kwargs = {
            "executor": executor,
            "resident_baseline_executor": resident_baseline_executor,
            "entropy_provider": entropy_provider,
            "hidden_judge": hidden_judge,
            "deadline": deadline,
            "continuation": continuation,
        }
        reference = run_causal_qualification(value, **runner_kwargs)
        if type(reference) is not EvidenceArtifactRef:
            raise QualificationIntakeError("qualification runner returned no typed artifact")
        if reference.schema == STAGE_EXIT_SCHEMA:
            terminal = (
                reopen_qualification_stage_exit(
                    value.evidence_root, reference, expected=value
                )
            )
            if (
                type(terminal) is not QualificationStageExit
                or len(manifest.reservations) != 1
                or terminal.authority_digest != manifest.authority_digest
                or terminal.source_digest != manifest.source_digest
                or terminal.selected_delta_digest
                != manifest.reservations[0].selected_delta_digest
            ):
                raise QualificationIntakeError(
                    "qualification stage exit differs from intake authority"
                )
            reservation = manifest.reservations[0]
            settlement_qualification = (
                _settlement_projection(
                    reservation, value.prepared.candidates[0], terminal,
                    manifest, reference, value,
                )
                if terminal.decision is QualificationDecision.PASS else None
            )
            outcome = QualificationIntakeOutcome(
                reservation.reservation_digest,
                reservation.selected_delta_digest,
                manifest.digest,
                terminal.decision,
                terminal.reason,
                terminal.decision is QualificationDecision.NO_DECISION,
                attempt_artifact_sha256=reference.sha256,
                report_digest=terminal.digest,
                settlement_qualification=settlement_qualification,
            )
            retry_plan = None
            if terminal.decision is QualificationDecision.NO_DECISION:
                retry_failure = canonical_digest(
                    "cacheon.qualification.intake-stage-retry",
                    {
                        "artifact": reference.sha256,
                        "authority_manifest_digest": manifest.digest,
                        "report": terminal.digest,
                    },
                )
                retry_plan = _retry_plan(
                    manifest, (reservation,), retry_failure, bisect=False
                )
            return QualificationIntakeBatch(
                manifest.digest, (outcome,), reference, retry_plan
            )
        attempt = (
            reopen_causal_qualification(
                value.evidence_root, reference, expected=value
            )
        )
    except RawSpeedEvidenceError as exc:
        return _no_decision_batch(manifest, exc, reason="raw_speed_evidence")
    except OuterSessionProcessError as exc:
        return _no_decision_batch(manifest, exc, reason="outer_session_process")
    except OuterSessionCandidateError as exc:
        if len(manifest.reservations) != 1:
            return _no_decision_batch(manifest, exc, reason="candidate_worker")
        return candidate_failure_batch(manifest, value, exc)
    except OCIBackendError as exc:
        return _no_decision_batch(manifest, exc, reason="oci_backend")
    except QualificationRunnerError as exc:
        return _no_decision_batch(manifest, exc, reason="qualification_runner")

    if type(attempt) is not CohortQualificationAttempt:
        raise QualificationIntakeError("qualification attempt lane differs")
    if (
        attempt.authority_digest != manifest.authority_digest
        or attempt.source_digest != manifest.source_digest
        or len(attempt.reports) != len(manifest.reservations)
    ):
        raise QualificationIntakeError("qualification attempt differs from intake authority")
    expected_report_type = CandidateQualificationReport
    outcomes = []
    retry_reservations = []
    from cacheon.eval.marginal_runtime import PreparedMarginalRuntime

    prepared_candidates = (
        value.prepared.candidates
        if type(value.prepared) is PreparedMarginalRuntime
        else ()
    )
    for index, (reservation, report) in enumerate(
        zip(manifest.reservations, attempt.reports, strict=True)
    ):
        if (
            type(report) is not expected_report_type
            or report.selected_delta_digest != reservation.selected_delta_digest
        ):
            raise QualificationIntakeError("qualification report order differs")
        settlement_qualification = (
            _settlement_projection(
                reservation, prepared_candidates[index], report,
                manifest, reference, attempt,
            )
            if prepared_candidates else None
        )
        outcomes.append(
            QualificationIntakeOutcome(
                reservation.reservation_digest,
                reservation.selected_delta_digest,
                manifest.digest,
                report.decision,
                report.reason,
                report.retryable,
                attempt_artifact_sha256=reference.sha256,
                report_digest=report.digest,
                settlement_qualification=settlement_qualification,
            )
        )
        if report.decision is QualificationDecision.NO_DECISION:
            retry_reservations.append(reservation)
    retry_plan = None
    if retry_reservations:
        retry_failure = canonical_digest(
            "cacheon.qualification.intake-report-retry",
            {
                "attempt": reference.sha256,
                "authority_manifest_digest": manifest.digest,
                "reservations": [row.reservation_digest for row in retry_reservations],
            },
        )
        retry_plan = _retry_plan(
            manifest, tuple(retry_reservations), retry_failure, bisect=False
        )
    return QualificationIntakeBatch(
        manifest.digest, tuple(outcomes), reference, retry_plan
    )


__all__ = [
    "QualificationAuthorityManifest",
    "QualificationIntakeBatch",
    "QualificationIntakeError",
    "QualificationIntakeOutcome",
    "QualificationPlanFactory",
    "QualificationReservation",
    "QualificationRetryPlan",
    "run_qualification_intake",
]

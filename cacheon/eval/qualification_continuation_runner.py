"""Private continuation-aware quality-stage orchestration.

The public runner owns all authority types and supplies its seams at call time.
Keeping those seams explicit preserves the existing monkeypatch surface without
creating a second qualification registry or a circular module dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class QualificationContinuationRunnerSeams:
    """Runner-owned operations consumed by the extracted continuation stage."""

    qualification_decision: Any
    qualification_stage_exit_type: type[Any]
    quality_continuation_type: type[Any]
    audit_continuation_type: type[Any]
    selection_entropy_receipt_type: type[Any]
    qualification_runner_error: type[Exception]
    qualification_continuation_error: type[Exception]
    qualification_authority_digest: Callable[[Any], str]
    run_slot_audits: Callable[..., Any]
    selection_receipt_type: type[Any]
    cohort_trajectory_digest: Callable[[Any], str]
    lifecycle_causal_completion: Callable[[Any], float]
    canonical_digest: Callable[[str, Any], str]
    reference_request: Callable[..., Any]
    reference_session_plan_type: type[Any]
    publish_qualification_stage_exit: Callable[..., Any]
    reopen_qualification_stage_exit: Callable[..., Any]


@dataclass(frozen=True)
class QualificationContinuationStageResult:
    """Locals produced by the speed/audit/pristine-T continuation boundary."""

    terminal: bool
    terminal_reference: Any | None
    lifecycle: Any | None = None
    audit_witnesses: dict[str, Any] | None = None
    audit_started: float | None = None
    audit_completed: float | None = None
    teardown_before: Any | None = None
    entropy: Any | None = None
    entropy_observed: float | None = None
    selection: Any | None = None
    requests: tuple[Any, ...] | None = None
    plan: Any | None = None
    reference_execution: Any | None = None
    teardown_after: Any | None = None
    t_pre: Any | None = None
    t_post: Any | None = None


def _audit_operation_digest(value: Any, lifecycle: Any, seams: Any) -> str:
    return seams.canonical_digest(
        "cacheon.qualification.audit-operation.v1",
        {
            "authority": seams.qualification_authority_digest(value),
            "source": value.prepared.source.digest,
            "trajectory": seams.cohort_trajectory_digest(lifecycle),
        },
    )


def _t_operation_digest(value: Any, plan: Any, seams: Any) -> str:
    return seams.canonical_digest(
        "cacheon.qualification.pristine-t-operation.v1",
        {
            "authority": seams.qualification_authority_digest(value),
            "native_build": value.pristine_binding.native_build_spec.digest,
            "preflight": value.pristine_binding.runtime_preflight_receipt.sha256,
            "controller": value.pristine_binding.controller_distribution_digest,
            "launch": value.pristine_launch.digest,
            "plan": plan.digest,
        },
    )


def run_continuation_quality_stage(
    *,
    value: Any,
    executor: Any,
    entropy_provider: Callable[..., Any],
    deadline: float,
    make_id: Callable[[], str],
    continuation: Any | None,
    quality_state: Any | None,
    resident_pair_mode: bool,
    quality_reads: int,
    resident_lifecycle: Any | None,
    resident_speed_witness: Any | None,
    resident_accept: bool,
    seams: QualificationContinuationRunnerSeams,
) -> QualificationContinuationStageResult:
    """Resume or execute speed, audit, and pristine-T exactly as before."""

    lifecycle = resident_lifecycle
    def finish_resident_audit(
        audit_witnesses: dict[str, Any],
        audit_started: float,
        audit_completed: float,
        teardown: Any,
        *,
        passed: bool,
    ) -> QualificationContinuationStageResult:
        audit = audit_witnesses[value.candidates[0].selected_delta_digest]
        closure = (
            resident_lifecycle.closure
            if passed and resident_lifecycle is not None
            else None
        )
        if passed and (
            not resident_pair_mode
            or closure is None
            or resident_speed_witness is None
        ):
            raise seams.qualification_runner_error(
                "resident acceptance lacks its completed speed/count authority"
            )
        extra = (None, None, closure) if passed else ()
        terminal = seams.qualification_stage_exit_type(
            seams.qualification_authority_digest(value),
            value.prepared.source.digest,
            value.candidates[0].selected_delta_digest,
            "resident_accept" if passed else "audit",
            (
                seams.qualification_decision.PASS
                if passed
                else seams.qualification_decision.FAIL
            ),
            "qualified" if passed else "slot_audit_failed",
            resident_speed_witness,
            audit,
            audit_started,
            audit_completed,
            teardown.digest,
            *extra,
        )
        reference = seams.publish_qualification_stage_exit(
            value.evidence_root, terminal
        )
        seams.reopen_qualification_stage_exit(
            value.evidence_root,
            reference,
            expected=value,
            resident_pair_lifecycle=(
                resident_lifecycle if resident_pair_mode else None
            ),
        )
        if continuation is not None:
            continuation.record_final(reference)
        return QualificationContinuationStageResult(True, reference)

    if quality_state is not None:
        assert continuation is not None
        audit_operation = _audit_operation_digest(value, lifecycle, seams)
        audit_state = continuation.load_audit(audit_operation)
        if audit_state is None:
            raise seams.qualification_continuation_error(
                "quality continuation exists without durable audit completion"
            )
        audit_witnesses = dict(audit_state.audit_witnesses)
        if tuple(audit_witnesses) != tuple(
            row.selected_delta_digest for row in value.candidates
        ):
            raise seams.qualification_continuation_error(
                "quality continuation audit coverage differs from the sealed cohort"
            )
        if any(
            row.decision is not seams.qualification_decision.PASS
            for row in audit_witnesses.values()
        ):
            raise seams.qualification_continuation_error(
                "quality continuation carries a failed resident audit"
            )
        audit_started = float(audit_state.audit_started)
        audit_completed = float(audit_state.audit_completed)
        teardown_before = quality_state.teardown_before
        entropy = quality_state.entropy
        entropy_observed = float(quality_state.entropy_observed)
        requests = quality_state.requests
        reference_execution = quality_state.reference_execution
        teardown_after = quality_state.teardown_after
        selection = seams.selection_receipt_type.reveal(
            value.commitment,
            secret=value.selection_secret,
            entropy=entropy,
            sealed_cohort_trajectory_digest=seams.cohort_trajectory_digest(
                lifecycle
            ),
        )
        request_plan_digest = _request_plan_digest(
            value=value,
            lifecycle=lifecycle,
            selection=selection,
            resident_speed_witness=resident_speed_witness,
            seams=seams,
        )
        if len(requests) != len(value.candidates) * quality_reads or any(
            row.plan_digest != request_plan_digest for row in requests
        ):
            raise seams.qualification_continuation_error(
                "quality continuation requests differ from the sealed request plan"
            )
        plan = seams.reference_session_plan_type(
            value.candidates[0].profile.reference,
            value.pristine_stack,
            value.reference_engine_config.digest,
            value.reference_engine_config,
            value.reference_preflight,
            request_plan_digest,
            requests,
        )
        if quality_state.t_operation_digest != _t_operation_digest(
            value, plan, seams
        ):
            raise seams.qualification_continuation_error(
                "quality continuation differs from its pristine-T claim"
            )
        t_pre, t_post = reference_execution.device_receipts
        if resident_accept:
            return finish_resident_audit(
                audit_witnesses,
                audit_started,
                audit_completed,
                teardown_before,
                passed=True,
            )
    else:
        with executor.exclusive_transaction():
            audit_operation = _audit_operation_digest(value, lifecycle, seams)
            audit_state = (
                None if continuation is None
                else continuation.load_audit(audit_operation)
            )
            if audit_state is None:
                audit_completion: list[float] = []
                audit_nonce = (
                    "" if continuation is None
                    else continuation.arm_evaluator("audit", audit_operation)
                )
                audit_started = float(executor.manager.clock())
                def commit_audit(witnesses: Any, last_completed: float) -> None:
                    if audit_completion:
                        raise seams.qualification_runner_error("audit sink called twice")
                    completed = float(executor.manager.clock())
                    continuation.record_audit(seams.audit_continuation_type(
                        audit_nonce, audit_operation, tuple(witnesses.items()),
                        audit_started, completed, last_completed,
                    ))
                    audit_completion.append(completed)

                audit_witnesses, audit_last_completed = seams.run_slot_audits(
                    value, lifecycle, executor=executor, deadline=float(deadline),
                    completion_sink=commit_audit if continuation is not None else None,
                )
                if continuation is None:
                    audit_completed = float(executor.manager.clock())
                elif len(audit_completion) != 1:
                    raise seams.qualification_runner_error("audit sink was not called once")
                else:
                    audit_completed = audit_completion[0]
            else:
                audit_witnesses = dict(audit_state.audit_witnesses)
                audit_started = audit_state.audit_started
                audit_completed = audit_state.audit_completed
                audit_last_completed = audit_state.audit_last_completed
            if any(
                row.decision is not seams.qualification_decision.PASS
                for row in audit_witnesses.values()
            ):
                teardown = executor.prove_quiescent()
                if teardown.observed_monotonic_s < audit_last_completed:
                    raise seams.qualification_runner_error(
                        "audit-exit quiescence predates candidate teardown"
                    )
                return finish_resident_audit(
                    audit_witnesses,
                    audit_started,
                    audit_completed,
                    teardown,
                    passed=False,
                )
            teardown_before = executor.prove_quiescent()
            # Bind quiescence to the FINAL executed baseline (B'' under repeat
            # reads, B-prime otherwise) — baseline_after is mid-run in the
            # 5-leg shape.
            last_post = max(
                seams.lifecycle_causal_completion(lifecycle), audit_last_completed
            )
            if teardown_before.observed_monotonic_s < last_post:
                raise seams.qualification_runner_error(
                    "pre-T quiescence predates the final baseline teardown"
                )
            if resident_accept:
                return finish_resident_audit(
                    audit_witnesses,
                    audit_started,
                    audit_completed,
                    teardown_before,
                    passed=True,
                )
            entropy = entropy_provider(value.commitment, teardown_before)
            if type(entropy) is not seams.selection_entropy_receipt_type:
                raise seams.qualification_runner_error(
                    "entropy provider returned an untyped receipt"
                )
            entropy_observed = float(executor.manager.clock())
            if (
                not math.isfinite(entropy_observed)
                or entropy_observed < teardown_before.observed_monotonic_s
            ):
                raise seams.qualification_runner_error(
                    "entropy observation predates teardown"
                )
            selection = seams.selection_receipt_type.reveal(
                value.commitment,
                secret=value.selection_secret,
                entropy=entropy,
                sealed_cohort_trajectory_digest=seams.cohort_trajectory_digest(
                    lifecycle
                ),
            )
            request_plan_digest = _request_plan_digest(
                value=value,
                lifecycle=lifecycle,
                selection=selection,
                resident_speed_witness=resident_speed_witness,
                seams=seams,
            )
            session_id = make_id()
            request_rows: list[Any] = []
            for authority in value.candidates:
                for candidate_read in range(1, quality_reads + 1):
                    kwargs = {
                        "session_id": session_id,
                        "plan_digest": request_plan_digest,
                        "request_id": make_id(),
                        "nonce": make_id(),
                        "index": len(request_rows),
                    }
                    if candidate_read == 1:
                        # Preserve historical call/serialization behavior exactly.
                        request = seams.reference_request(
                            lifecycle, authority, selection, **kwargs
                        )
                    else:
                        request = seams.reference_request(
                            lifecycle,
                            authority,
                            selection,
                            candidate_read=candidate_read,
                            **kwargs,
                        )
                    request_rows.append(request)
            requests = tuple(request_rows)
            plan = seams.reference_session_plan_type(
                value.candidates[0].profile.reference,
                value.pristine_stack,
                value.reference_engine_config.digest,
                value.reference_engine_config,
                value.reference_preflight,
                request_plan_digest,
                requests,
            )
            t_operation = _t_operation_digest(value, plan, seams)
            t_nonce = (
                "" if continuation is None
                else continuation.arm_evaluator("t", t_operation)
            )
            quality_completion: list[tuple[Any, Any]] = []

            def close_reference(execution: Any) -> Any:
                completed = executor.prove_quiescent()
                t_before, t_after = execution.device_receipts
                if (
                    t_before.started_monotonic_s < entropy_observed
                    or t_after.completed_monotonic_s
                    > completed.observed_monotonic_s
                ):
                    raise seams.qualification_runner_error(
                        "pristine T does not lie between causal boundaries"
                    )
                return completed

            def commit_quality(execution: Any) -> None:
                if quality_completion:
                    raise seams.qualification_runner_error(
                        "pristine T invoked its completion sink more than once"
                    )
                completed = close_reference(execution)
                continuation.record_quality(seams.quality_continuation_type(
                    teardown_before=teardown_before,
                    entropy=entropy,
                    entropy_observed=entropy_observed,
                    requests=requests,
                    reference_execution=execution,
                    teardown_after=completed,
                    t_nonce=t_nonce,
                    t_operation_digest=t_operation,
                ))
                quality_completion.append((execution, completed))

            reference_execution = executor.execute_reference(
                value.pristine_launch,
                value.pristine_binding,
                value.model_mount,
                plan,
                deadline=float(deadline),
                completion_sink=(
                    commit_quality if continuation is not None else None
                ),
            )
            if continuation is not None and (
                len(quality_completion) != 1
                or quality_completion[0][0] is not reference_execution
            ):
                raise seams.qualification_runner_error(
                    "pristine T returned without its exact durable completion"
            )
            if continuation is None:
                teardown_after = close_reference(reference_execution)
            else:
                teardown_after = quality_completion[0][1]
            t_pre, t_post = reference_execution.device_receipts

    return QualificationContinuationStageResult(
        False,
        None,
        lifecycle,
        audit_witnesses,
        audit_started,
        audit_completed,
        teardown_before,
        entropy,
        entropy_observed,
        selection,
        requests,
        plan,
        reference_execution,
        teardown_after,
        t_pre,
        t_post,
    )


def _request_plan_digest(
    *,
    value: Any,
    lifecycle: Any,
    selection: Any,
    resident_speed_witness: Any | None,
    seams: QualificationContinuationRunnerSeams,
) -> str:
    payload = {
        "candidate_deltas": [
            row.selected_delta_digest for row in value.candidates
        ],
        "cohort_trajectory_digest": seams.cohort_trajectory_digest(lifecycle),
        "reference_manifest_digest": value.candidates[0].profile.reference.digest,
        "selection_digest": selection.digest,
    }
    if value.speed_evidence_policy.version != 1:
        payload["speed_evidence_policy"] = value.speed_evidence_policy.to_dict()
    if resident_speed_witness is not None:
        payload["resident_speed_evidence"] = resident_speed_witness.evidence_digest
    return seams.canonical_digest(
        "cacheon.qualification.reference-request-plan",
        payload,
    )


__all__ = [
    "QualificationContinuationRunnerSeams",
    "QualificationContinuationStageResult",
    "run_continuation_quality_stage",
]

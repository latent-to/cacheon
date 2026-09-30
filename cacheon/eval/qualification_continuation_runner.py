"""Resume the qualification owner's audit and pristine-T stages from durable work."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

from cacheon.audit_gate import infrastructure_failure


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


def _audit_operation_digest(value: Any, lifecycle: Any, owner: Any) -> str:
    return owner.canonical_digest(
        "cacheon.qualification.audit-operation.v1",
        {"authority": owner.qualification_authority_digest(value),
         "source": value.prepared.source.digest,
         "trajectory": owner.cohort_trajectory_digest(lifecycle)},
    )


def _t_operation_digest(value: Any, plan: Any, owner: Any) -> str:
    return owner.canonical_digest(
        "cacheon.qualification.pristine-t-operation.v1",
        {"authority": owner.qualification_authority_digest(value),
         "native_build": value.pristine_binding.native_build_spec.digest,
         "preflight": value.pristine_binding.runtime_preflight_receipt.sha256,
         "controller": value.pristine_binding.controller_distribution_digest,
         "launch": value.pristine_launch.digest, "plan": plan.digest},
    )


def run_continuation_quality_stage(
    *, value: Any, executor: Any, entropy_provider: Callable[..., Any],
    deadline: float, make_id: Callable[[], str], continuation: Any | None,
    quality_state: Any | None, resident_lifecycle: Any,
    resident_speed_witness: Any,
) -> QualificationContinuationStageResult:
    """Run only missing paid stages; fresh and resumed work share the same binding."""
    # Resolve the actual owner at call time: no second registry of its operations.
    from cacheon.eval import qualification_runner as owner

    lifecycle = resident_lifecycle
    speed = resident_speed_witness
    audit_operation = _audit_operation_digest(value, lifecycle, owner)
    audit_state = None if continuation is None else continuation.load_audit(audit_operation)
    if quality_state is not None and audit_state is None:
        raise owner.QualificationContinuationError(
            "quality continuation exists without durable audit completion"
        )
    if quality_state is None:
        # Charge completed work before launch, excluding downtime between durable
        # stages. An expired operator session must not buy a replacement B/C/B'.
        used = speed.completed_monotonic_s - speed.started_monotonic_s
        if audit_state is not None:
            used += audit_state.audit_completed - audit_state.audit_started
        remaining = value.resident_speed_plan.policy.max_qualification_seconds - used
        now = float(executor.manager.clock())
        last = max(speed.completed_monotonic_s,
                   audit_state.audit_completed if audit_state is not None else 0.0)
        if not math.isfinite(now) or now < last:
            raise owner.QualificationContinuationError(
                "current clock predates retained qualification; refusing new work"
            )
        deadline = min(deadline, now + remaining)
        if not math.isfinite(remaining) or deadline <= now:
            raise owner.QualificationContinuationError(
                "retained qualification exhausted its execution budget"
            )

    with executor.exclusive_transaction() if quality_state is None else nullcontext():
        if audit_state is None:
            completed_audits = []
            nonce = "" if continuation is None else continuation.arm_evaluator("audit", audit_operation)
            audit_started = float(executor.manager.clock())

            def commit_audit(witnesses: Any, last_completed: float) -> None:
                if completed_audits:
                    raise owner.QualificationRunnerError("audit sink called twice")
                record = owner.AuditContinuation(
                    nonce, audit_operation, tuple(witnesses.items()), audit_started,
                    float(executor.manager.clock()), last_completed,
                )
                continuation.record_audit(record)
                completed_audits.append(record)

            witnesses, last_completed = owner._run_slot_audits(
                value, lifecycle, executor=executor, deadline=float(deadline),
                completion_sink=commit_audit if continuation is not None else None,
            )
            if continuation is None:
                audit_state = owner.AuditContinuation(
                    nonce, audit_operation, tuple(witnesses.items()), audit_started,
                    float(executor.manager.clock()), last_completed,
                )
            elif len(completed_audits) != 1:
                raise owner.QualificationRunnerError("audit sink was not called once")
            else:
                audit_state = completed_audits[0]
        audit_witnesses = dict(audit_state.audit_witnesses)
        audit_started = audit_state.audit_started
        audit_completed = audit_state.audit_completed
        if tuple(audit_witnesses) != tuple(row.selected_delta_digest for row in value.candidates):
            raise owner.QualificationContinuationError(
                "quality continuation audit coverage differs from the sealed cohort"
            )
        for audit in audit_witnesses.values():
            unavailable = infrastructure_failure(
                [row.to_gate_dict() for row in audit.receipts],
                min_calls=audit.policy.minimum_calls,
                expected_slots=audit.policy.expected_slots,
                expected_member_count=audit.policy.expected_member_count,
            )
            if unavailable is not None:
                raise owner.QualificationContinuationError("slot audit evidence unavailable: " + unavailable)
        if any(row.decision is not owner.QualificationDecision.PASS for row in audit_witnesses.values()):
            if quality_state is not None:
                raise owner.QualificationContinuationError(
                    "quality continuation carries a failed resident audit"
                )
            teardown = executor.prove_quiescent()
            if teardown.observed_monotonic_s < audit_state.audit_last_completed:
                raise owner.QualificationRunnerError("audit-exit quiescence predates candidate teardown")
            audit = audit_witnesses[value.candidates[0].selected_delta_digest]
            # The stage exit re-checks this pair against the runner's own vocabulary.
            terminal = owner.QualificationStageExit(
                owner.qualification_authority_digest(value), value.prepared.source.digest,
                value.candidates[0].selected_delta_digest, "audit", audit.decision,
                "slot_audit_failed" if audit.decision is owner.QualificationDecision.FAIL else "audit_not_covered",
                speed, audit, audit_started, audit_completed, teardown.digest,
            )
            reference = owner.publish_qualification_stage_exit(value.evidence_root, terminal)
            owner.reopen_qualification_stage_exit(value.evidence_root, reference, expected=value)
            if continuation is not None:
                continuation.record_final(reference)
            return QualificationContinuationStageResult(True, reference)

        if quality_state is None:
            teardown_before = executor.prove_quiescent()
            # Bind quiescence to the FINAL executed baseline (B'' under repeat
            # reads, B-prime otherwise) — baseline_after is mid-run in the
            # 5-leg shape.
            last_post = max(owner._lifecycle_causal_completion(lifecycle), audit_state.audit_last_completed)
            if teardown_before.observed_monotonic_s < last_post:
                raise owner.QualificationRunnerError("pre-T quiescence predates the final baseline teardown")
            entropy = entropy_provider(value.commitment, teardown_before)
            if type(entropy) is not owner.SelectionEntropyReceipt:
                raise owner.QualificationRunnerError("entropy provider returned an untyped receipt")
            entropy_observed = float(executor.manager.clock())
            if not math.isfinite(entropy_observed) or entropy_observed < teardown_before.observed_monotonic_s:
                raise owner.QualificationRunnerError("entropy observation predates teardown")
        else:
            teardown_before = quality_state.teardown_before
            entropy = quality_state.entropy
            entropy_observed = float(quality_state.entropy_observed)
        selection = owner.SelectionReceipt.reveal(
            value.commitment, secret=value.selection_secret, entropy=entropy,
            sealed_cohort_trajectory_digest=owner.cohort_trajectory_digest(lifecycle),
        )
        if entropy != lifecycle.crossover.quality_entropy:
            raise owner.QualificationContinuationError("replay control selection entropy changed after execution")
        request_plan_digest = owner.canonical_digest(
            "cacheon.qualification.reference-request-plan",
            {"candidate_deltas": [row.selected_delta_digest for row in value.candidates],
             "cohort_trajectory_digest": owner.cohort_trajectory_digest(lifecycle),
             "reference_manifest_digest": value.candidates[0].profile.reference.digest,
             "selection_digest": selection.digest,
             "speed_evidence_policy": value.speed_evidence_policy.to_dict(),
             "resident_speed_evidence": speed.evidence_digest},
        )
        if quality_state is None:
            session_id = make_id()
            requests = tuple(owner._reference_request(
                lifecycle, authority, selection, session_id=session_id,
                plan_digest=request_plan_digest, request_id=make_id(), nonce=make_id(), index=index,
            ) for index, authority in enumerate(value.candidates))
        else:
            requests = quality_state.requests
            if len(requests) != len(value.candidates) or any(row.plan_digest != request_plan_digest for row in requests):
                raise owner.QualificationContinuationError(
                    "quality continuation requests differ from the sealed request plan"
                )
        plan = owner.ReferenceSessionPlan(
            value.candidates[0].profile.reference, value.pristine_stack,
            value.reference_engine_config.digest, value.reference_engine_config,
            value.reference_preflight, request_plan_digest, requests,
        )
        t_operation = _t_operation_digest(value, plan, owner)
        if quality_state is not None:
            if quality_state.t_operation_digest != t_operation:
                raise owner.QualificationContinuationError("quality continuation differs from its pristine-T claim")
            reference_execution = quality_state.reference_execution
            teardown_after = quality_state.teardown_after
        else:
            t_nonce = "" if continuation is None else continuation.arm_evaluator("t", t_operation)
            quality_completion = []

            def close_reference(execution: Any) -> Any:
                completed = executor.prove_quiescent()
                t_before, t_after = execution.device_receipts
                if (t_before.started_monotonic_s < entropy_observed
                    or t_after.completed_monotonic_s > completed.observed_monotonic_s):
                    raise owner.QualificationRunnerError("pristine T does not lie between causal boundaries")
                return completed

            def commit_quality(execution: Any) -> None:
                if quality_completion:
                    raise owner.QualificationRunnerError("pristine T invoked its completion sink more than once")
                completed = close_reference(execution)
                continuation.record_quality(owner.QualityContinuation(
                    teardown_before=teardown_before, entropy=entropy,
                    entropy_observed=entropy_observed, requests=requests,
                    reference_execution=execution, teardown_after=completed,
                    t_nonce=t_nonce, t_operation_digest=t_operation,
                ))
                quality_completion.append((execution, completed))

            reference_execution = executor.execute_reference(
                value.pristine_launch, value.pristine_binding, value.model_mount, plan,
                deadline=float(deadline),
                completion_sink=commit_quality if continuation is not None else None,
            )
            if continuation is not None and (
                len(quality_completion) != 1 or quality_completion[0][0] is not reference_execution
            ):
                raise owner.QualificationRunnerError("pristine T returned without its exact durable completion")
            teardown_after = close_reference(reference_execution) if continuation is None else quality_completion[0][1]
        t_pre, t_post = reference_execution.device_receipts

    return QualificationContinuationStageResult(
        False, None, lifecycle, audit_witnesses, audit_started, audit_completed,
        teardown_before, entropy, entropy_observed, selection, requests, plan,
        reference_execution, teardown_after, t_pre, t_post,
    )


__all__ = ["QualificationContinuationStageResult", "run_continuation_quality_stage"]

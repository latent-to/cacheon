from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.eval.qualification_runner as runner
from cacheon.eval.device_state import DeviceStateReceipt, DeviceStateSample
from cacheon.eval.qualification import declared_qualification_entropy_digest
from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
from cacheon.eval.oci_backend import OCIBackendError, OCIEngineExecutor
from cacheon.eval.oci_outer_session import OuterSessionWorkerError
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification import (
    GraphVerificationGrade,
    QualificationDecision,
    SelectionCommitment,
    SelectionEntropyReceipt,
)
from cacheon.eval.reference_protocol import ReferenceRoleInput, ReferenceTokenEvidence
from cacheon.eval.reference_quality import ReferenceQualityVerdict


_REAL_PUBLISH_CAUSAL = runner.publish_causal_qualification
_REAL_REOPEN_CAUSAL = runner.reopen_causal_qualification
_REAL_QUALIFICATION_AUTHORITY = runner.qualification_authority_digest


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _artifact(label: str) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        "reference-quality.raw",
        _d(label),
        2,
        "application/json",
        "cacheon.reference-quality-raw.v1",
    )


def _quiescence(sequence: int, observed: float) -> OCIQuiescenceReceipt:
    return OCIQuiescenceReceipt(
        "cacheon.oci-quiescence.v1",
        "runner",
        "1" * 32,
        _d("executor-namespace"),
        sequence,
        observed,
        (),
        (),
        (),
    )


def _graph_grade(decision: QualificationDecision, index: int) -> GraphVerificationGrade:
    reason = {
        QualificationDecision.PASS: "graph_verification_pass",
        QualificationDecision.FAIL: "graph_replay_failed",
        QualificationDecision.NO_DECISION: "graph_evidence_missing",
    }[decision]
    return GraphVerificationGrade(
        decision,
        reason,
        _d(f"graph-requirement-{index}"),
        _d(f"graph-reference-{index}"),
        _d(f"graph-raw-{index}"),
    )


def _quality_verdict(
    decision: QualificationDecision, index: int, calibration_digest: str
) -> ReferenceQualityVerdict:
    failed = ("mean_nll",) if decision is QualificationDecision.FAIL else ()
    overlap = ("mean_nll",) if decision is QualificationDecision.NO_DECISION else ()
    return ReferenceQualityVerdict(
        decision.value,
        failed,
        overlap,
        "1",
        _d(f"quality-evidence-{index}"),
        calibration_digest,
    )


class _Harness:
    def __init__(
        self,
        monkeypatch,
        *,
        graph: tuple[QualificationDecision, ...],
        speed: tuple[QualificationDecision, ...],
        quality: tuple[QualificationDecision, ...],
        audit: tuple[QualificationDecision, ...] | None = None,
        swap_exchanges: bool = False,
        fail_pre_t_quiescence: bool = False,
        exercise_judge_cache: bool = False,
        repeat: bool = False,
    ) -> None:
        assert len(graph) == len(speed)
        assert len(quality) == len(graph) * (2 if repeat else 1)
        audit = audit or (QualificationDecision.PASS,) * len(graph)
        assert len(audit) == len(graph)
        # This runner harness predates the typed authority constructor and
        # intentionally replaces the fully validated input boundary with
        # lightweight records. Preserve exact production type checks while
        # identifying only these registered-lane test doubles as that type.
        monkeypatch.setattr(
            runner, "CandidateQualificationAuthority", SimpleNamespace
        )
        self.calls: list[str] = []
        self.reference_calls = 0
        self.reference_request_counts: list[int] = []
        self.reference_session_plans: list[object] = []
        self.secret = b"s" * 32
        secret_commitment = hashlib.sha256(
            b"cacheon-selection-secret-v1\0" + self.secret
        ).hexdigest()
        self.commitment = SelectionCommitment(
            _d("source"),
            _d("reference"),
            _d("workload"),
            _d("entropy-source"),
            secret_commitment,
            (_d("prompt"),),
            1,
        )
        self.entropy = SelectionEntropyReceipt(
            self.commitment.entropy_source_digest,
            self.commitment.digest,
            _d("entropy"),
            _d("entropy-authority"),
        )
        self.calibration = SimpleNamespace(digest=_d("calibration"))
        self.grades = {
            _d(f"delta-{index}"): _graph_grade(decision, index)
            for index, decision in enumerate(graph)
        }
        reference = SimpleNamespace(
            digest=self.commitment.reference_manifest_digest,
            pristine_launch_digest=_d("pristine-launch"),
            hidden_corpus_commitment=_d("hidden-corpus"),
            hidden_judge_digest=_d("hidden-judge"),
        )
        hidden_task_policy = _d("hidden-task-policy")
        authorities = tuple(
            SimpleNamespace(
                selected_delta_digest=_d(f"delta-{index}"),
                profile=SimpleNamespace(
                    reference=reference,
                    digest=_d(f"profile-{index}"),
                    hidden_task_policy_digest=hidden_task_policy,
                ),
                graph_requirement=SimpleNamespace(digest=_d(f"requirement-{index}")),
                graph_artifact_ref=_artifact(f"graph-{index}"),
                graph_evidence_ref=SimpleNamespace(
                    digest=_d(f"graph-evidence-ref-{index}")
                ),
            )
            for index in range(len(graph))
        )
        lifecycle_candidates = tuple(
            SimpleNamespace(
                arm=SimpleNamespace(
                    selected_delta_digest=authority.selected_delta_digest,
                    digest=_d(f"arm-{index}"),
                    transition=SimpleNamespace(target_id=f"target.{index}"),
                ),
                candidate=SimpleNamespace(
                    launch=SimpleNamespace(digest=_d(f"candidate-launch-{index}"))
                ),
            )
            for index, authority in enumerate(authorities)
        )
        final_baseline = SimpleNamespace(
            device_receipts=(
                SimpleNamespace(completed_monotonic_s=1.0),
                SimpleNamespace(completed_monotonic_s=2.0),
            )
        )
        self.lifecycle = SimpleNamespace(
            candidates=lifecycle_candidates,
            baseline_after=final_baseline,
            # mirrors MarginalLifecycleEvidence.final_baseline (== baseline_after on
            # the historical 3-leg shape; B'' when repeat reads run)
            final_baseline=final_baseline,
        )
        prepared = SimpleNamespace(
            source=SimpleNamespace(digest=_d("source")),
            candidates=tuple(
                SimpleNamespace(
                    arm=row.arm,
                    launch=row.candidate.launch,
                    binding=SimpleNamespace(
                        launch_binding=SimpleNamespace(
                            native_build_spec=SimpleNamespace(
                                digest=_d(f"candidate-native-{index}")
                            ),
                            runtime_preflight_receipt=SimpleNamespace(
                                sha256=_d(f"candidate-preflight-{index}")
                            ),
                        )
                    ),
                )
                for index, row in enumerate(lifecycle_candidates)
            ),
            baseline_launch=SimpleNamespace(digest=_d("baseline-launch")),
            incumbent_binding=SimpleNamespace(
                launch_binding=SimpleNamespace(
                    runtime_preflight_receipt=SimpleNamespace(
                        sha256=_d("incumbent-preflight")
                    )
                )
            ),
        )
        self.value = SimpleNamespace(
            prepared=prepared,
            model_mount=SimpleNamespace(digest=_d("model-mount")),
            candidates=authorities,
            commitment=self.commitment,
            selection_secret=self.secret,
            evidence_root=SimpleNamespace(),
            pristine_stack=SimpleNamespace(digest=_d("pristine-stack")),
            pristine_launch=SimpleNamespace(digest=_d("pristine-launch")),
            pristine_binding=SimpleNamespace(
                native_build_spec=SimpleNamespace(digest=_d("pristine-native")),
                controller_distribution_digest=_d("pristine-controller"),
                runtime_preflight_receipt=SimpleNamespace(
                    sha256=_d("pristine-preflight")
                ),
            ),
            reference_engine_config=SimpleNamespace(digest=_d("reference-engine-config")),
            reference_preflight=SimpleNamespace(digest=_d("reference-preflight")),
            expected_launch_resource_policy_digest=_d("launch-policy"),
            expected_runtime_resource_policy_digest=_d("runtime-policy"),
            expected_device_policy_digest=_d("device-policy"),
            speed_evidence_policy=(
                runner.SpeedEvidencePolicy.repeat()
                if repeat
                else runner.SpeedEvidencePolicy.legacy()
            ),
            speed_stage_disposition=runner.SpeedStageDisposition.TERMINAL,
            audit_policies=tuple(
                runner.SlotAuditPolicy(
                    f"{700 + index:032x}",
                    250_000,
                    32,
                    ("norm.rmsnorm",),
                    1,
                )
                for index in range(len(authorities))
            ),
            calibration_threshold_policy=SimpleNamespace(
                digest=_d("calibration-threshold-policy")
            ),
            calibration_manifest=self.calibration,
            calibration_artifact_ref=_artifact("calibration"),
            calibration_context=SimpleNamespace(
                digest=_d("calibration-context"),
                workload_digest=_d("workload"),
            ),
        )

        self.executor = object.__new__(OCIEngineExecutor)
        self.executor.manager = SimpleNamespace(clock=lambda: 3.5)
        before, after = _quiescence(1, 3.0), _quiescence(2, 6.0)
        quiescence_count = 0
        @contextmanager
        def transaction(_executor):
            self.calls.append("transaction.enter")
            try:
                yield _executor
            finally:
                self.calls.append("transaction.exit")
        def prove_quiescent(_executor):
            nonlocal quiescence_count
            quiescence_count += 1
            self.calls.append(f"quiescence.{quiescence_count}")
            if fail_pre_t_quiescence and quiescence_count == 1:
                raise runner.QualificationRunnerError("pre-T quiescence failed")
            return before if quiescence_count == 1 else after
        def execute_reference(
            _executor, _launch, _binding, _mount, plan, *, deadline, completion_sink=None
        ):
            del deadline
            assert plan is not getattr(self.value, "resident_audit_plan", None)
            self.calls.append("reference")
            self.reference_calls += 1
            self.reference_request_counts.append(len(plan.requests))
            self.reference_session_plans.append(plan)
            requests = tuple(plan.requests)
            if swap_exchanges:
                requests = tuple(reversed(requests))
            exchanges = tuple(
                SimpleNamespace(
                    request=request,
                    request_sha256=request.sha256,
                    evidence_frame_sha256=_d(f"evidence-{request.index}"),
                )
                for request in requests
            )
            session = SimpleNamespace(
                exchanges=exchanges,
                digest=_d("reference-session"),
                session_id=requests[0].session_id,
                request_plan_digest=requests[0].plan_digest,
            )
            sample = DeviceStateSample(4.0, (), (), True, "idle")
            receipts = (
                DeviceStateReceipt(
                    "cacheon.device-state-receipt.v1", 10, "reference-launch", "pre",
                    (0,), _d("device-config"), self.value.expected_device_policy_digest,
                    4.0, 4.1, 1, (sample,),
                ),
                DeviceStateReceipt(
                    "cacheon.device-state-receipt.v1", 11, "reference-launch", "post",
                    (0,), _d("device-config"), self.value.expected_device_policy_digest,
                    4.2, 5.0, 1, (sample,),
                ),
            )
            evidence = SimpleNamespace(
                launch_digest=reference.pristine_launch_digest,
                runtime_identity=SimpleNamespace(
                    runtime_digest=_d("reference-runtime"),
                    base_engine_digest=_d("reference-base"),
                    validator_overlay_digest=_d("reference-overlay"),
                ),
                runtime_preflight_receipt_sha256=_d("reference-preflight-receipt"),
                arena_model_receipt_digest=_d("model-mount"),
                resource_policy_digest=self.value.expected_runtime_resource_policy_digest,
                prebuild=SimpleNamespace(build_spec_digest=_d("reference-build-spec")),
                native_publication_digest=_d("reference-publication"),
                runtime_argv_sha256=_d("reference-argv"),
                recovered_lease_ids=(),
                session=session,
                device_receipts=receipts,
            )
            if completion_sink is not None:
                completion_sink(evidence)
            return evidence
        monkeypatch.setattr(OCIEngineExecutor, "exclusive_transaction", transaction)
        monkeypatch.setattr(OCIEngineExecutor, "prove_quiescent", prove_quiescent)
        monkeypatch.setattr(OCIEngineExecutor, "execute_reference", execute_reference)
        def prevalidate(_value, *_args, **_kwargs):
            self.calls.append("prevalidate")
            return self.calibration, self.grades
        monkeypatch.setattr(runner, "_validate_pre_execution", prevalidate)
        def run_audits(value, _lifecycle, *, completion_sink=None, **_kwargs):
            self.calls.append("audit")
            witnesses = {}
            for index, (authority, prepared, policy, decision) in enumerate(
                zip(
                    value.candidates,
                    value.prepared.candidates,
                    value.audit_policies,
                    audit,
                    strict=True,
                )
            ):
                receipts = tuple(
                    runner.AuditReceiptFacts(
                        slot,
                        32,
                        1 if decision is QualificationDecision.FAIL else 0,
                        0,
                        0,
                        0.0 if decision is QualificationDecision.FAIL else 1.0,
                        0.995,
                        "allclose",
                        900 + index * 100 + rank,
                        rank,
                        policy.expected_member_count,
                    )
                    for slot in policy.expected_slots
                    for rank in range(policy.expected_member_count)
                )
                from cacheon.audit import gate

                passed, detail = gate(
                    [receipt.to_gate_dict() for receipt in receipts],
                    min_calls=policy.minimum_calls,
                    expected_slots=policy.expected_slots,
                    expected_member_count=policy.expected_member_count,
                )
                assert passed == (decision is QualificationDecision.PASS)
                witness = runner.AuditWitness(
                    authority.selected_delta_digest,
                    getattr(getattr(value, "resident_audit_plan", None), "launch",
                            prepared.launch).digest,
                    _d(f"audit-execution-{index}"),
                    f"{900 + index:032x}",
                    value.expected_runtime_resource_policy_digest,
                    policy,
                    receipts,
                    decision,
                    detail,
                )
                witnesses[authority.selected_delta_digest] = witness
            if completion_sink is not None:
                completion_sink(witnesses, 2.5)
            return witnesses, 2.5

        monkeypatch.setattr(runner, "_run_slot_audits", run_audits)
        monkeypatch.setattr(runner, "cohort_trajectory_digest", lambda _row: _d("cohort"))

        def entropy_provider(commitment, teardown):
            assert commitment is self.commitment
            assert teardown is before
            self.calls.append("entropy")
            return self.entropy

        self.entropy_provider = entropy_provider

        class BoundJudge:
            binding = runner.HiddenJudgeBinding(
                reference.hidden_corpus_commitment,
                reference.hidden_judge_digest,
                hidden_task_policy,
            )

            def __init__(self):
                self.calls = 0

            def __call__(self, *, prompt_digest, output_ids, task_digests):
                self.calls += 1
                return runner.HiddenJudgeReceipt(
                    self.binding.digest,
                    prompt_digest,
                    runner.hidden_judge_output_digest(prompt_digest, output_ids),
                    task_digests,
                    (True,) * len(task_digests),
                )

        self.hidden_judge = BoundJudge()

        def make_request(
            _lifecycle,
            authority,
            _selection,
            *,
            session_id,
            plan_digest,
            request_id,
            nonce,
            index,
            candidate_read=1,
        ):
            del request_id, nonce
            request_label = f"request-{authority.selected_delta_digest}"
            if candidate_read == 2:
                request_label += "-repeat"
            return SimpleNamespace(
                index=index,
                delta=authority.selected_delta_digest,
                session_id=session_id,
                plan_digest=plan_digest,
                sha256=_d(request_label),
            )

        monkeypatch.setattr(runner, "_reference_request", make_request)
        monkeypatch.setattr(runner, "request_sha256", lambda request: request.sha256)
        monkeypatch.setattr(
            runner,
            "ReferenceSessionPlan",
            lambda *_args: SimpleNamespace(digest=_d("reference-plan"), requests=_args[-1]),
        )
        raw_index = 0

        def raw_artifact(_lifecycle, authority, *_args, candidate_read=1):
            nonlocal raw_index
            self.calls.append(f"raw.{raw_index}")
            raw_index += 1
            if exercise_judge_cache:
                judge = _args[-1]
                for _ in range(3):
                    judge(
                        prompt_digest=_d("memo-prompt"),
                        output_ids=(1, 2),
                        task_digests=(_d("memo-task"),),
                    )
            binding = object.__new__(runner.ReferenceQualityRawBinding)
            binding_values = {
                "qualification_identity_digest": _d("qualification-identity"),
                "reference_manifest_digest": self.commitment.reference_manifest_digest,
                "calibration_digest": self.calibration.digest,
                "selection_digest": _d("selection"),
                "candidate_lifecycle_digest": _d("candidate-lifecycle"),
                "selected_trajectory_digest": _d("selected-trajectory"),
                "selected_trajectory_projection_digest": _d("trajectory-projection"),
                "selected_prompt_digests": (_d("prompt"),),
                "t_session_digest": _d("reference-session"),
                "t_request_sha256": _d(
                    f"request-{authority.selected_delta_digest}"
                    + ("-repeat" if candidate_read == 2 else "")
                ),
                "support_policy_digest": _d("support-policy"),
                "hidden_task_plan_digest": _d("hidden-task-plan"),
                "nll_tail_threshold": "1",
                "tokens_per_prompt": 1,
                "topk_width": 1,
                "hidden_tasks_per_prompt": 1,
            }
            for name, value in binding_values.items():
                object.__setattr__(binding, name, value)
            return SimpleNamespace(
                binding=binding,
                to_dict=lambda: {"candidate": authority.selected_delta_digest},
            )

        monkeypatch.setattr(runner, "_raw_artifact", raw_artifact)
        monkeypatch.setattr(
            runner,
            "validate_quality_binding",
            lambda *_args, **_kwargs: self.calls.append("quality.validate"),
        )
        publish_index = 0

        def publish(*_args, **_kwargs):
            nonlocal publish_index
            self.calls.append("quality.publish")
            result = _artifact(f"raw-quality-{publish_index}")
            publish_index += 1
            return result

        monkeypatch.setattr(runner, "publish_evidence", publish)
        monkeypatch.setattr(
            runner,
            "reopen_reference_quality_evidence",
            lambda *_args, **_kwargs: self.calls.append("quality.reopen") or object(),
        )
        quality_rows = iter(enumerate(quality))

        def score_quality(*_args, **_kwargs):
            index, decision = next(quality_rows)
            self.calls.append(f"quality.score.{index}")
            return _quality_verdict(decision, index, self.calibration.digest)

        monkeypatch.setattr(runner, "score_reference_quality", score_quality)
        monkeypatch.setattr(
            runner,
            "qualification_authority_digest",
            lambda _value: _d("qualification-authority"),
        )

        self.attempt_reference = EvidenceArtifactRef(
            runner.ATTEMPT_DOMAIN,
            _d("causal-attempt-artifact"),
            2,
            "application/json",
            runner.ATTEMPT_SCHEMA_V2 if repeat else runner.ATTEMPT_SCHEMA,
        )
        self.published_attempt = None

        def publish_attempt(root, attempt):
            assert root is self.value.evidence_root
            assert type(attempt) is runner.CohortQualificationAttempt
            self.calls.append("attempt.publish")
            self.published_attempt = attempt
            return self.attempt_reference

        def reopen_attempt(root, reference, *, expected):
            assert root is self.value.evidence_root
            assert reference == self.attempt_reference
            assert expected is self.value
            assert self.published_attempt is not None
            self.calls.append("attempt.reopen")
            return self.published_attempt

        monkeypatch.setattr(runner, "publish_causal_qualification", publish_attempt)
        monkeypatch.setattr(runner, "reopen_causal_qualification", reopen_attempt)

def _install_resident_runner_path(
    monkeypatch,
    harness: _Harness,
    *,
    speed_decision: QualificationDecision,
    escalated: bool,
):
    """Install only the resident orchestration seam around the existing runner harness."""

    assert len(harness.value.candidates) == 1
    harness.value.speed_evidence_policy = runner.SpeedEvidencePolicy.resident()
    harness.value.resident_audit_plan = SimpleNamespace(
        digest=_d("resident-audit-authority"),
        launch=SimpleNamespace(digest=_d("resident-audit-launch")),
    )
    harness.resident_speed_plans = []
    harness.value.resident_speed_plan = SimpleNamespace(
        digest=_d("resident-plan"),
        policy=SimpleNamespace(
            digest=_d("resident-policy"),
            max_qualification_seconds=7_200,
        ),
        baseline_lane_digest=_d("resident-baseline-lane"),
        candidate_lane_digest=_d("resident-candidate-lane"),
        baseline=SimpleNamespace(
            launch=SimpleNamespace(digest=_d("resident-baseline-launch")),
            runtime_resource_policy_digest=_d(
                "resident-baseline-runtime-policy"
            ),
        ),
        candidate=SimpleNamespace(
            launch=harness.value.prepared.candidates[0].launch,
            runtime_resource_policy_digest=(
                harness.value.expected_runtime_resource_policy_digest
            ),
        ),
    )
    harness.attempt_reference = EvidenceArtifactRef(
        runner.ATTEMPT_DOMAIN,
        _d("resident-causal-attempt-artifact"),
        2,
        "application/json",
        runner.ATTEMPT_SCHEMA_V3,
    )

    class FakeResidentCrossover:
        def __init__(self) -> None:
            self.escalated = escalated

    crossover = FakeResidentCrossover()

    class FakeResidentLifecycle:
        def __init__(self, prepared, plan, observed) -> None:
            assert prepared is harness.value.prepared
            assert plan is harness.value.resident_speed_plan
            assert observed is crossover
            self.candidates = harness.lifecycle.candidates
            self.final_baseline = harness.lifecycle.final_baseline

    class FakeResidentSpeedWitness:
        def __init__(self) -> None:
            plan = harness.value.resident_speed_plan
            self.selected_delta_digest = (
                harness.value.candidates[0].selected_delta_digest
            )
            self.candidate_launch_digest = plan.candidate.launch.digest
            self.calibration_digest = harness.calibration.digest
            self.calibration_context_digest = harness.value.calibration_context.digest
            self.workload_digest = harness.value.calibration_context.workload_digest
            self.baseline_runtime_resource_policy_digest = (
                plan.baseline.runtime_resource_policy_digest
            )
            self.candidate_runtime_resource_policy_digest = (
                plan.candidate.runtime_resource_policy_digest
            )
            self.plan_digest = plan.digest
            self.baseline_lane_digest = plan.baseline_lane_digest
            self.candidate_lane_digest = plan.candidate_lane_digest
            self.baseline_quiescence_digest = _d("resident-baseline-quiescence")
            self.candidate_quiescence_digest = _d("resident-candidate-quiescence")
            self.raw_crossover_digest = _d("resident-raw-crossover")
            self.evidence_digest = _d("resident-speed-evidence")
            self.started_monotonic_s = 1.0
            self.completed_monotonic_s = 3.0
            self.resident_policy = plan.policy
            roles = (
                ("B", "C", "B_prime", "C_prime", "B_double_prime")
                if escalated
                else ("B", "C", "B_prime")
            )
            self.rates = tuple(
                SimpleNamespace(
                    role=role,
                    lane_digest=(
                        plan.baseline_lane_digest
                        if role.startswith("B")
                        else plan.candidate_lane_digest
                    ),
                    launch_digest=(
                        plan.baseline.launch.digest
                        if role.startswith("B")
                        else plan.candidate.launch.digest
                    ),
                    session_id=("a" if role.startswith("B") else "b") * 32,
                )
                for role in roles
            )

        _scalars = (
            "selected_delta_digest",
            "candidate_launch_digest",
            "calibration_digest",
            "calibration_context_digest",
            "workload_digest",
            "baseline_runtime_resource_policy_digest",
            "candidate_runtime_resource_policy_digest",
            "plan_digest",
            "baseline_lane_digest",
            "candidate_lane_digest",
            "baseline_quiescence_digest",
            "candidate_quiescence_digest",
            "raw_crossover_digest",
            "evidence_digest",
        )

        @classmethod
        def from_evidence(cls, observed, plan):
            assert observed is crossover
            assert plan is harness.value.resident_speed_plan
            return cls()

        @property
        def policy(self):
            return runner.SpeedEvidencePolicy.resident()

        @property
        def has_repeat(self) -> bool:
            return len(self.rates) == 5

        def regrade(self, *_args, **_kwargs):
            reason = (
                None
                if speed_decision is QualificationDecision.PASS
                else "speed_noise"
                if speed_decision is QualificationDecision.NO_DECISION
                else "speed_threshold_not_met"
            )
            return speed_decision, "1.1000000000000001", reason

        def to_dict(self):
            return {
                **{name: getattr(self, name) for name in type(self)._scalars},
                "started_monotonic_s": format(self.started_monotonic_s, ".17g"),
                "completed_monotonic_s": format(
                    self.completed_monotonic_s, ".17g"
                ),
                "resident_policy": {"digest": self.resident_policy.digest},
                "rates": [dict(vars(row)) for row in self.rates],
            }

        @classmethod
        def from_dict(cls, value):
            witness = cls.__new__(cls)
            for name in cls._scalars:
                setattr(witness, name, value[name])
            witness.started_monotonic_s = float(value["started_monotonic_s"])
            witness.completed_monotonic_s = float(value["completed_monotonic_s"])
            witness.resident_policy = SimpleNamespace(
                digest=value["resident_policy"]["digest"],
                max_qualification_seconds=7_200,
            )
            witness.rates = tuple(
                SimpleNamespace(**row) for row in value["rates"]
            )
            return witness

        def __eq__(self, other):
            return type(other) is type(self) and self.to_dict() == other.to_dict()

    def run_resident(plan, *, baseline_executor, candidate_executor, model_mount, deadline):
        assert plan is harness.value.resident_speed_plan
        assert plan is not harness.value.resident_audit_plan
        harness.resident_speed_plans.append(plan)
        assert baseline_executor is resident_baseline_executor
        assert candidate_executor is harness.executor
        assert model_mount is harness.value.model_mount
        assert deadline == 100.0
        harness.calls.append("resident.speed")
        return crossover

    # The legacy cold lifecycle and legacy speed projection no longer exist in
    # the runner module at all; their absence IS the guarantee the two
    # forbidden_* monkeypatches used to assert here.
    monkeypatch.setattr(runner, "ResidentMarginalLifecycleEvidence", FakeResidentLifecycle)
    monkeypatch.setattr(runner, "ResidentSpeedWitness", FakeResidentSpeedWitness)
    monkeypatch.setattr(runner, "run_resident_crossover_speed", run_resident)

    published_stage_exits = []
    stage_reference = EvidenceArtifactRef(
        runner.STAGE_EXIT_DOMAIN,
        _d("resident-stage-exit"),
        2,
        "application/json",
        runner.STAGE_EXIT_SCHEMA,
    )

    def publish_stage(root, result):
        assert root is harness.value.evidence_root
        harness.calls.append("stage.publish")
        published_stage_exits.append(result)
        return stage_reference

    def reopen_stage(root, reference, *, expected, resident_pair_lifecycle=None):
        assert root is harness.value.evidence_root
        assert reference == stage_reference
        assert expected is harness.value
        assert resident_pair_lifecycle is None or isinstance(
            resident_pair_lifecycle, FakeResidentLifecycle
        )
        assert len(published_stage_exits) == 1
        harness.calls.append("stage.reopen")
        return published_stage_exits[0]

    monkeypatch.setattr(runner, "publish_qualification_stage_exit", publish_stage)
    monkeypatch.setattr(runner, "reopen_qualification_stage_exit", reopen_stage)

    clock_values = iter((3.1, 3.4, 3.5, 3.6))
    harness.executor.manager.clock = lambda: next(clock_values)
    resident_baseline_executor = object.__new__(OCIEngineExecutor)
    resident_baseline_executor.manager = SimpleNamespace(clock=lambda: 0.0)
    return resident_baseline_executor, stage_reference, published_stage_exits


def _run_resident_harness(harness: _Harness, resident_baseline_executor):
    ids = iter(f"{index + 1:032x}" for index in range(16))
    return runner.run_causal_qualification(
        harness.value,
        executor=harness.executor,
        resident_baseline_executor=resident_baseline_executor,
        entropy_provider=harness.entropy_provider,
        hidden_judge=harness.hidden_judge,
        deadline=100.0,
        id_factory=lambda: next(ids),
    )


def _resident_case(
    monkeypatch,
    *,
    speed_decision=QualificationDecision.PASS,
    escalated=False,
    quality=(QualificationDecision.PASS,),
    disposition=None,
    **harness_kwargs,
):
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=quality,
        **harness_kwargs,
    )
    if disposition is not None:
        harness.value.speed_stage_disposition = disposition
    baseline, stage_reference, exits = _install_resident_runner_path(
        monkeypatch, harness, speed_decision=speed_decision, escalated=escalated
    )
    return harness, baseline, stage_reference, exits


def test_resident_speed_fail_exits_before_audit_t_and_legacy_lifecycle(
    monkeypatch,
) -> None:
    harness, baseline, stage_reference, exits = _resident_case(
        monkeypatch, speed_decision=QualificationDecision.FAIL
    )

    reference = _run_resident_harness(harness, baseline)

    assert reference == stage_reference
    assert len(exits) == 1
    assert exits[0].stage == "speed"
    assert exits[0].decision is QualificationDecision.FAIL
    # The exit carries the witness's graded band reason, not a coarse
    # "speed_regression" invented from the bare decision enum.
    assert exits[0].reason == "speed_threshold_not_met"
    assert harness.calls == [
        "prevalidate",
        "resident.speed",
        "stage.publish",
        "stage.reopen",
    ]
    assert harness.reference_calls == 0


def test_resident_calibration_continuation_collects_audit_and_t_after_speed_fail(
    monkeypatch,
) -> None:
    harness, baseline, _stage_reference, exits = _resident_case(
        monkeypatch,
        speed_decision=QualificationDecision.FAIL,
        disposition=runner.SpeedStageDisposition.CALIBRATION_OBSERVATION,
    )

    reference = _run_resident_harness(harness, baseline)

    assert reference == harness.attempt_reference
    assert exits == []
    assert harness.reference_calls == 1
    assert "audit" in harness.calls
    assert "reference" in harness.calls
    assert "attempt.publish" in harness.calls
    assert "attempt.reopen" in harness.calls
    assert harness.published_attempt is not None
    report = harness.published_attempt.reports[0]
    assert report.speed_decision is QualificationDecision.FAIL
    assert report.decision is QualificationDecision.FAIL


def test_resident_audit_fail_exits_before_t(monkeypatch) -> None:
    harness, baseline, stage_reference, exits = _resident_case(
        monkeypatch, audit=(QualificationDecision.FAIL,)
    )

    reference = _run_resident_harness(harness, baseline)

    assert reference == stage_reference
    assert len(exits) == 1
    assert exits[0].stage == "audit"
    assert exits[0].decision is QualificationDecision.FAIL
    assert "audit" in harness.calls
    assert "reference" not in harness.calls
    assert "attempt.publish" not in harness.calls
    assert harness.reference_calls == 0


@pytest.mark.parametrize(
    ("escalated", "quality", "expected_requests"),
    (
        (False, (QualificationDecision.PASS,), 1),
        (
            True,
            (QualificationDecision.PASS, QualificationDecision.PASS),
            2,
        ),
    ),
)
def test_resident_pass_uses_adaptive_t_coverage_without_legacy_speed_projection(
    monkeypatch,
    escalated: bool,
    quality: tuple[QualificationDecision, ...],
    expected_requests: int,
) -> None:
    harness, baseline, _stage_reference, exits = _resident_case(
        monkeypatch, escalated=escalated, quality=quality, repeat=escalated
    )

    reference = _run_resident_harness(harness, baseline)

    assert reference == harness.attempt_reference
    assert reference.schema == runner.ATTEMPT_SCHEMA_V3
    assert exits == []
    assert harness.reference_request_counts == [expected_requests]
    assert harness.reference_calls == 1
    assert "resident.speed" in harness.calls
    assert "lifecycle" not in harness.calls
    assert not any(call.startswith("speed.") for call in harness.calls)
    assert harness.published_attempt is not None
    report = harness.published_attempt.reports[0]
    assert (report.repeat_quality is not None) is escalated


def test_resident_operational_timing_round_trip_and_total_budget() -> None:
    timing = runner.QualificationTimingWitness(
        _d("resident-policy"),
        _d("resident-speed"),
        _d("resident-audit"),
        _d("resident-t"),
        3_600,
        10.0,
        100.0,
        101.0,
        200.0,
        201.0,
        300.0,
        301.0,
    )
    assert runner.QualificationTimingWitness.from_dict(timing.to_dict()) == timing
    assert timing.digest

    # from_dict must never mutate the caller's payload: the publish/reopen
    # self-check compares attempt.to_dict() against the reopened payload, and an
    # in-place float decode of the .17g strings falsely fails a canonical
    # artifact ("qualification artifact is not semantically canonical",
    # 2026-07-24 B300 v9 calibration — the first v3 attempt publication with
    # operational_timing populated).
    payload = timing.to_dict()
    snapshot = {name: value for name, value in payload.items()}
    runner.QualificationTimingWitness.from_dict(payload)
    assert payload == snapshot

    with pytest.raises(runner.QualificationRunnerError, match="total wall time"):
        replace(
            timing,
            max_qualification_seconds=60,
            qualification_completed_monotonic_s=301.0,
        )


def test_repeat_report_cannot_drop_c_prime_quality(monkeypatch) -> None:
    # C-prime quality regresses on the escalated resident schedule; the
    # conservative aggregate fails, and a reopened payload cannot shed the
    # repeat-quality coverage that carried the regression.
    harness, baseline, _stage_reference, _exits = _resident_case(
        monkeypatch,
        escalated=True,
        quality=(QualificationDecision.PASS, QualificationDecision.FAIL),
        repeat=True,
    )
    _run_resident_harness(harness, baseline)
    report = harness.published_attempt.reports[0]

    assert harness.reference_request_counts == [2]
    assert report.repeat_quality is not None
    assert report.repeat_quality.quality_decision is QualificationDecision.FAIL
    assert report.quality_decision is QualificationDecision.FAIL
    assert report.decision is QualificationDecision.FAIL
    assert report.reason == "quality_repeat_regression"

    payload = report.to_dict()
    assert "repeat_quality" in payload
    del payload["repeat_quality"]
    with pytest.raises(
        runner.QualificationRunnerError, match="repeat quality coverage"
    ):
        runner.CandidateQualificationReport.from_dict(payload)


def test_pristine_reference_worker_error_remains_unattributed(monkeypatch) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)

    def fail_reference(*_args, **_kwargs):
        raise OuterSessionWorkerError("pristine teacher failed")

    monkeypatch.setattr(OCIEngineExecutor, "execute_reference", fail_reference)
    with pytest.raises(OuterSessionWorkerError, match="pristine teacher"):
        _run_resident_harness(harness, baseline)
    assert "resident.speed" in harness.calls
    assert "reference" not in harness.calls


@pytest.mark.parametrize(
    ("speed_decision", "retryable"),
    (
        (QualificationDecision.PASS, False),
        (QualificationDecision.FAIL, False),
        (QualificationDecision.NO_DECISION, True),
    ),
)
def test_candidate_headlines_recompute_pass_fail_and_no_decision(
    monkeypatch, speed_decision, retryable
) -> None:
    # A non-PASS speed verdict stage-exits under the terminal disposition;
    # the report-level headline only exists on the observation lane.
    harness, baseline, _stage_reference, _exits = _resident_case(
        monkeypatch,
        speed_decision=speed_decision,
        disposition=(
            None
            if speed_decision is QualificationDecision.PASS
            else runner.SpeedStageDisposition.CALIBRATION_OBSERVATION
        ),
    )
    _run_resident_harness(harness, baseline)
    report = harness.published_attempt.reports[0]
    assert report.decision is speed_decision
    assert report.retryable is retryable
    with pytest.raises(runner.QualificationRunnerError, match="headline"):
        runner.CandidateQualificationReport(
            **{
                **report.__dict__,
                "decision": (
                    QualificationDecision.FAIL
                    if speed_decision is not QualificationDecision.FAIL
                    else QualificationDecision.PASS
                ),
            }
        )


def test_slot_audit_violation_is_a_hard_nonretryable_qualification_fail(
    monkeypatch,
) -> None:
    # An audit violation terminates at the audit stage exit on every
    # disposition; the exit names the hard slot_audit_failed reason and
    # carries the violating receipt.
    harness, baseline, stage_reference, exits = _resident_case(
        monkeypatch, audit=(QualificationDecision.FAIL,)
    )
    reference = _run_resident_harness(harness, baseline)

    assert reference == stage_reference
    assert len(exits) == 1
    assert exits[0].stage == "audit"
    assert exits[0].decision is QualificationDecision.FAIL
    assert exits[0].reason == "slot_audit_failed"
    assert exits[0].audit_witness.receipts[0].violations == 1
    assert harness.published_attempt is None


def test_t_exchange_substitution_is_rejected(monkeypatch) -> None:
    # The escalated resident schedule issues two T requests; a swapped exchange
    # order must be rejected exactly as it was on the retired cold schedule.
    harness, baseline, _stage_reference, _exits = _resident_case(
        monkeypatch,
        escalated=True,
        quality=(QualificationDecision.PASS, QualificationDecision.PASS),
        repeat=True,
        swap_exchanges=True,
    )
    with pytest.raises(runner.QualificationRunnerError, match="request|exchange"):
        _run_resident_harness(harness, baseline)


def test_pre_t_quiescence_failure_prevents_reference_launch(monkeypatch) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(
        monkeypatch, fail_pre_t_quiescence=True
    )
    with pytest.raises(runner.QualificationRunnerError, match="quiescence"):
        _run_resident_harness(harness, baseline)
    assert harness.reference_calls == 0
    assert "entropy" not in harness.calls


def test_stale_hidden_judge_binding_is_rejected_before_b(monkeypatch) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)
    harness.hidden_judge.binding = runner.HiddenJudgeBinding(
        _d("stale-corpus"),
        _d("hidden-judge"),
        _d("hidden-task-policy"),
    )

    with pytest.raises(runner.QualificationRunnerError, match="hidden judge authority"):
        _run_resident_harness(harness, baseline)
    assert "resident.speed" not in harness.calls
    assert harness.reference_calls == 0


def test_identical_hidden_judge_inputs_are_memoized(monkeypatch) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(
        monkeypatch, exercise_judge_cache=True
    )
    _run_resident_harness(harness, baseline)
    assert harness.hidden_judge.calls == 1


@pytest.mark.parametrize("mislabeled", ("prompt", "output"))
def test_hidden_judge_receipt_cannot_relabel_prompt_or_output(mislabeled) -> None:
    prompt_digest = _d("judge-prompt")
    output_ids = (1,)
    binding = runner.HiddenJudgeBinding(
        _d("hidden-corpus"),
        _d("hidden-judge"),
        _d("hidden-task-policy"),
    )
    profile = SimpleNamespace(
        reference=SimpleNamespace(
            hidden_corpus_commitment=binding.hidden_corpus_commitment,
            hidden_judge_digest=binding.hidden_judge_digest,
        ),
        hidden_task_policy_digest=binding.hidden_task_policy_digest,
        hidden_tasks_per_prompt=1,
    )

    class MislabeledJudge:
        def __init__(self) -> None:
            self.binding = binding

        def __call__(self, *, prompt_digest, output_ids, task_digests):
            receipt_prompt = _d("other-prompt") if mislabeled == "prompt" else prompt_digest
            output_digest = runner.hidden_judge_output_digest(prompt_digest, output_ids)
            if mislabeled == "output":
                output_digest = _d("other-output")
            return runner.HiddenJudgeReceipt(
                self.binding.digest,
                receipt_prompt,
                output_digest,
                task_digests,
                (True,) * len(task_digests),
            )

    with pytest.raises(runner.QualificationRunnerError):
        runner._rollout(
            profile=profile,
            prompt_digest=prompt_digest,
            frame={"top_logprobs": (((-0.1, 1), (-1.0, 2)),)},
            role_input=ReferenceRoleInput(output_ids, ((1, 2),)),
            role_evidence=SimpleNamespace(
                tokens=(ReferenceTokenEvidence(-0.25, 1, (-0.1, -1.0)),)
            ),
            hidden_judge=MislabeledJudge(),
        )


def test_shared_manager_reservation_excludes_another_thread() -> None:
    lock = threading.RLock()
    first = object.__new__(OCIEngineExecutor)
    second = object.__new__(OCIEngineExecutor)
    first._lock = second._lock = lock
    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        with first.exclusive_transaction():
            entered.set()
            assert release.wait(timeout=2)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        with pytest.raises(OCIBackendError, match="active transaction"):
            with second.exclusive_transaction():
                raise AssertionError("another thread entered the reserved manager")
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_reports_and_attempt_expose_no_score_crown_or_settlement_fields(monkeypatch) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)
    _run_resident_harness(harness, baseline)
    attempt = harness.published_attempt

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for row in value.values() for key in keys(row)}
        if isinstance(value, list):
            return {key for row in value for key in keys(row)}
        return set()

    emitted = keys(attempt.to_dict())
    assert not ({"score", "crown", "crownable", "settlement", "winner"} & emitted)


def test_registered_authority_digest_versions_slot_audit_policy_and_report_wire(
    monkeypatch, tmp_path: Path
) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)
    _run_resident_harness(harness, baseline)
    attempt = harness.published_attempt
    value = harness.value
    captured: dict[str, object] = {}
    real_digest = runner.canonical_digest
    authority_domain = "cacheon.qualification.causal-authority.v3.eager-audit-v2"

    def capture(domain: str, payload: object) -> str:
        if domain == authority_domain:
            captured["payload"] = payload
        return real_digest(domain, payload)

    monkeypatch.setattr(runner, "CandidateQualificationAuthority", SimpleNamespace)
    monkeypatch.setattr(runner, "canonical_digest", capture)
    authority_digest = _REAL_QUALIFICATION_AUTHORITY(value)
    assert authority_digest == real_digest(authority_domain, captured["payload"])
    authority_payload = captured["payload"]
    assert set(authority_payload) == {
        "calibration",
        "candidates",
        "commitment",
        "incumbent_preflight",
        "model_mount",
        "policies",
        "reference",
        "slot_audit_policies",
        "source",
        "speed_evidence_policy",
        "resident_speed_plan",
        "resident_audit_authority",
    }
    assert set(authority_payload["candidates"][0]) == {
        "arm",
        "delta",
        "graph_artifact",
        "graph_evidence_ref",
        "graph_requirement",
        "launch",
        "native",
        "preflight",
        "profile",
        "target",
    }
    monkeypatch.setattr(runner, "canonical_digest", real_digest)

    report_fields = (
        "selected_delta_digest",
        "marginal_arm_digest",
        "candidate_launch_digest",
        "target_id",
        "profile_digest",
        "calibration_digest",
        "graph_grade_digest",
        "graph_decision",
        "speed_evidence_digest",
        "speed_decision",
        "speedup",
        "quality_evidence_digest",
        "quality_decision",
        "candidate_mean_teacher_nll",
        "raw_quality_artifact",
        "raw_quality_binding",
            "speed_witness",
            "t_request_sha256",
            "audit_evidence_digest",
            "audit_decision",
            "audit_witness",
            "decision",
        "reason",
        "retryable",
    )
    report = attempt.reports[0]
    assert tuple(report.to_dict()) == report_fields
    assert report.speed_witness.to_dict() == report.to_dict()["speed_witness"]
    assert report.audit_witness.to_dict() == report.to_dict()["audit_witness"]
    assert runner.canonical_json_bytes(report.to_dict()) == runner.canonical_json_bytes(
        {
            field: runner._encode_record(getattr(report, field))
            for field in report_fields
        }
    )
    assert tuple(attempt.to_dict()) == (
        "authority_digest",
        "source_digest",
        "cohort_trajectory_digest",
        "commitment",
        "teardown_before_t",
        "entropy",
        "entropy_observed_monotonic_s",
        "selection",
        "reference_execution",
        "teardown_after_t",
        "reports",
        "operational_timing",
    )

    monkeypatch.setattr(runner, "publish_evidence", publish_evidence)
    root = tmp_path / "registered-wire"
    reference = _REAL_PUBLISH_CAUSAL(root, attempt)
    assert (reference.domain, reference.schema) == (
        runner.ATTEMPT_DOMAIN,
        runner.ATTEMPT_SCHEMA_V3,
    )
    assert reopen_evidence(root, reference) == runner.canonical_json_bytes(
        attempt.to_dict()
    )


def _audit_witness(prefix, policy, receipts, session_id):
    execution = runner.EngineExecutionEvidence(
        "cacheon.oci-engine-execution.v1",
        _d(f"{prefix}-launch"),
        SimpleNamespace(),
        _d(f"{prefix}-preflight"),
        _d(f"{prefix}-model"),
        _d(f"{prefix}-runtime-policy"),
        SimpleNamespace(build_spec_digest=_d(f"{prefix}-build")),
        _d(f"{prefix}-publication"),
        _d(f"{prefix}-argv"),
        (),
        (),
        SimpleNamespace(
            audit_policy_digest=policy.digest,
            audit_receipts=receipts,
            session_id=session_id,
        ),
    )
    return runner.AuditWitness.from_execution(
        execution, selected_delta_digest=_d(f"{prefix}-delta"), policy=policy
    )


def test_audit_witness_canonicalizes_raw_protocol_floats_and_reopens() -> None:
    policy = runner.SlotAuditPolicy("a" * 32, 250_000, 32, ("norm.rmsnorm",), 1)
    receipt = runner.AuditReceiptFacts(
        "norm.rmsnorm", 32, 0, 0, 0, 1.0, 0.995, "allclose", 901, 0, 1
    )
    assert type(receipt.to_dict()["worst_frac"]) is float
    assert type(receipt.to_dict()["min_ratio"]) is float
    witness = _audit_witness("audit", policy, (receipt,), "1" * 32)
    durable = witness.to_dict()
    durable_receipt = durable["receipts"][0]
    assert durable_receipt == runner._record_dict(receipt)
    assert durable_receipt["worst_frac"] == "1"
    assert durable_receipt["min_ratio"] == "0.995"
    assert runner.AuditWitness.from_dict(durable) == witness

    # A durable authority record must never acquire a JSON float, even when it
    # spells the same numeric value as the canonical decimal string.
    float_tamper = witness.to_dict()
    float_tamper["receipts"][0]["worst_frac"] = 1.0
    with pytest.raises(
        runner.QualificationRunnerError,
        match="worst_frac is not a canonical decimal string",
    ):
        runner.AuditWitness.from_dict(float_tamper)

    spelling_tamper = witness.to_dict()
    spelling_tamper["receipts"][0]["worst_frac"] = "1.0"
    with pytest.raises(
        runner.QualificationRunnerError,
        match="worst_frac is not canonical",
    ):
        runner.AuditWitness.from_dict(spelling_tamper)


def test_audit_witness_grades_policy_bound_empty_receipts_as_fail() -> None:
    policy = runner.SlotAuditPolicy(
        "a" * 32, 250_000, 32, ("moe.fused_experts_reduce",), 4
    )
    witness = _audit_witness("empty-audit", policy, (), "3" * 32)

    assert witness.decision is QualificationDecision.FAIL
    assert witness.receipts == ()
    assert witness.detail == "no audit receipts (need >= 32 audited calls)"
    assert runner.AuditWitness.from_dict(witness.to_dict()) == witness


def test_audit_witness_host_regrade_does_not_import_torch(monkeypatch) -> None:
    """The trusted controller is intentionally lean and has no worker torch."""
    import sys

    import cacheon

    # Force both audit modules through a clean import boundary.  The old host
    # path imported cacheon.audit here and therefore raised ModuleNotFoundError
    # in the production controller venv before it could grade the receipts.
    for name in ("cacheon.audit", "cacheon.audit_gate"):
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.delattr(cacheon, name.rsplit(".", 1)[-1], raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    policy = runner.SlotAuditPolicy("b" * 32, 250_000, 32, ("norm.rmsnorm",), 1)
    receipt = runner.AuditReceiptFacts(
        "norm.rmsnorm", 32, 0, 0, 0, 1.0, 0.995, "allclose", 902, 0, 1
    )
    witness = _audit_witness("torch-free-audit", policy, (receipt,), "2" * 32)
    assert witness.decision is QualificationDecision.PASS
    assert runner.AuditWitness.from_dict(witness.to_dict()) == witness
    assert "cacheon.audit" not in sys.modules
    assert sys.modules["torch"] is None


def test_durable_attempt_roundtrip_rejects_nested_decision_tamper(
    monkeypatch, tmp_path
) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)
    _run_resident_harness(harness, baseline)
    attempt = harness.published_attempt
    root = tmp_path / "attempt-evidence"

    # Restore the real byte publisher only after the fake causal run is complete.
    monkeypatch.setattr(runner, "publish_evidence", publish_evidence)
    reference = _REAL_PUBLISH_CAUSAL(root, attempt)
    reopened = runner.CohortQualificationAttempt.from_dict(
        runner._canonical_payload(reopen_evidence(root, reference))
    )
    assert reopened == attempt

    tampered = attempt.to_dict()
    tampered["reports"][0]["decision"] = QualificationDecision.FAIL.value
    forged = publish_evidence(
        root,
        runner.canonical_json_bytes(tampered),
        domain=runner.ATTEMPT_DOMAIN,
        media_type="application/json",
        schema=runner.ATTEMPT_SCHEMA_V3,
    )
    with pytest.raises(runner.QualificationRunnerError, match="headline"):
        _REAL_REOPEN_CAUSAL(root, forged, expected=harness.value)


def test_reopen_rejects_self_consistent_speed_witness_arm_relabel(
    monkeypatch, tmp_path
) -> None:
    """A witness relabeled to another delta — with every nested digest kept
    wire-consistent — must still die on the EXTERNAL authority comparison.
    The report-level cohort identity is left untouched, so nothing before the
    per-report speed check can catch the substitution; the companion test in
    test_qualification.py proves the real witness's internal digest admits an
    equally self-consistent forgery at construction."""

    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)
    # Reopen's resident branch demands an exactly-typed ResidentCrossoverPlan
    # before it will even read the witness, so the harness's namespace plan is
    # replaced by a real plan built through the production crossover math; the
    # harness's candidate-launch and runtime-policy identities are realigned to
    # it so every honest cross-check passes and only the relabel can fail.
    from tests.test_qualification import _lifecycle as _resident_fixture

    plan = _resident_fixture(tmp_path / "resident-fixture")[0].plan
    harness.value.resident_speed_plan = plan
    harness.value.prepared.candidates[0].launch.digest = plan.candidate.launch.digest
    harness.value.expected_runtime_resource_policy_digest = (
        plan.candidate.runtime_resource_policy_digest
    )
    _run_resident_harness(harness, baseline)
    payload = harness.published_attempt.to_dict()
    report = payload["reports"][0]
    speed = report["speed_witness"]
    speed["selected_delta_digest"] = _d("relabeled-delta")
    speed["evidence_digest"] = _d("relabeled-evidence")
    report["speed_evidence_digest"] = speed["evidence_digest"]
    # The attempt-level timing witness cross-binds the report's speed evidence
    # digest; a competent forger rebinds it, and the reopen must still refuse.
    payload["operational_timing"]["speed_evidence_digest"] = speed["evidence_digest"]

    root = tmp_path / "speed-relabel"
    monkeypatch.setattr(runner, "publish_evidence", publish_evidence)
    forged = publish_evidence(
        root,
        runner.canonical_json_bytes(payload),
        domain=runner.ATTEMPT_DOMAIN,
        media_type="application/json",
        schema=runner.ATTEMPT_SCHEMA_V3,
    )
    monkeypatch.setattr(runner, "_validate_reference_execution", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "reopen_calibration_evidence",
        lambda *_args, **_kwargs: harness.calibration,
    )
    monkeypatch.setattr(
        runner, "reopen_reference_quality_evidence", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runner,
        "score_reference_quality",
        lambda *_args, **_kwargs: _quality_verdict(
            QualificationDecision.PASS, 0, harness.calibration.digest
        ),
    )
    with pytest.raises(
        runner.QualificationRunnerError, match="differs from its marginal arm"
    ):
        _REAL_REOPEN_CAUSAL(root, forged, expected=harness.value)


@pytest.mark.parametrize("tamper", ("launch", "causal_time"))
def test_reference_execution_witness_rejects_launch_and_causal_tamper(
    monkeypatch, tamper
) -> None:
    harness, baseline, _stage_reference, _exits = _resident_case(monkeypatch)
    _run_resident_harness(harness, baseline)
    attempt = harness.published_attempt
    witness = attempt.reference_execution
    identity = SimpleNamespace(
        runtime_digest=_d("reference-runtime"),
        base_engine_digest=_d("reference-base"),
        validator_overlay_digest=_d("reference-overlay"),
    )
    monkeypatch.setattr(runner, "runtime_identity_from_preflight", lambda _row: identity)
    harness.value.pristine_binding = SimpleNamespace(
        runtime_preflight_receipt=SimpleNamespace(
            sha256=witness.runtime_preflight_receipt_sha256
        ),
        native_build_spec=SimpleNamespace(digest=witness.build_spec_digest),
    )
    harness.value.pristine_launch = SimpleNamespace(
        digest=witness.launch_digest,
        hardware=SimpleNamespace(visible_gpu_count=1),
    )
    harness.value.model_mount = SimpleNamespace(digest=witness.arena_model_receipt_digest)
    harness.value.pristine_stack = SimpleNamespace(digest=_d("pristine-stack"))
    harness.value.reference_engine_config = SimpleNamespace(digest=_d("reference-config"))
    harness.value.reference_preflight = SimpleNamespace(digest=_d("reference-preflight"))
    runtime_digest = runner.canonical_digest(
        "cacheon.qualification.reference-runtime",
        {
            "base": identity.base_engine_digest,
            "runtime": identity.runtime_digest,
            "validator_overlay": identity.validator_overlay_digest,
        },
    )
    plan_digest = runner.canonical_digest(
        "cacheon.eval.reference-session-plan",
        {
            "engine_config_digest": harness.value.reference_engine_config.digest,
            "expected_preflight_digest": harness.value.reference_preflight.digest,
            "pristine_stack_digest": harness.value.pristine_stack.digest,
            "reference_manifest_digest": harness.value.candidates[0].profile.reference.digest,
            "request_plan_digest": witness.request_plan_digest,
            "request_sha256": list(witness.request_sha256),
        },
    )
    session_digest = runner.canonical_digest(
        "cacheon.eval.pristine-reference-session",
        {
            "exchanges": [
                {
                    "request_index": index,
                    "request_sha256": request,
                    "evidence_frame_sha256": evidence,
                }
                for index, (request, evidence) in enumerate(
                    zip(witness.request_sha256, witness.evidence_frame_sha256, strict=True)
                )
            ],
            "launch_digest": witness.launch_digest,
            "preflight_digest": harness.value.reference_preflight.digest,
            "reference_manifest_digest": harness.value.candidates[0].profile.reference.digest,
            "request_plan_digest": witness.request_plan_digest,
            "schema": "cacheon.pristine-reference-session.v1",
            "session_id": witness.session_id,
            "session_plan_digest": plan_digest,
        },
    )
    honest_witness = replace(
        witness,
        runtime_identity_digest=runtime_digest,
        plan_digest=plan_digest,
        session_digest=session_digest,
    )
    honest = replace(
        attempt,
        reference_execution=honest_witness,
        # The v3 operational-timing witness binds the reference session digest;
        # rebinding the witness must carry the timing binding with it.
        operational_timing=replace(
            attempt.operational_timing, reference_session_digest=session_digest
        ),
    )
    runner._validate_reference_execution(honest, harness.value)

    if tamper == "launch":
        forged_witness = replace(honest_witness, launch_digest=_d("other-launch"))
    else:
        post = list(honest_witness.device_receipts[1])
        post[8] = "7"
        forged_witness = replace(
            honest_witness,
            device_receipts=(honest_witness.device_receipts[0], tuple(post)),
        )
    with pytest.raises(runner.QualificationRunnerError, match="pristine T"):
        runner._validate_reference_execution(
            replace(honest, reference_execution=forged_witness), harness.value
        )


def _typed_resident_qualification_input(
    tmp_path: Path,
    *,
    candidate_lane: str = "right",
) -> runner.CausalQualificationInput:
    """Build the real registered-lane authority boundary for resident speed."""

    from cacheon.eval.calibration import (
        CalibrationContext,
        CalibrationThresholdPolicy,
    )
    from cacheon.eval.crossover_runtime import (
        ResidentArmPlan,
        ResidentCrossoverPlan,
        ResidentSpeedPolicy,
    )
    from cacheon.eval.marginal_runtime import (
        MaterializedArmBinding,
        prepare_marginal_runtime,
    )
    from cacheon.eval.oci_backend import expected_runtime_preflight
    from cacheon.eval.qualification import (
        GRAPH_EVIDENCE_DOMAIN,
        GRAPH_EVIDENCE_MEDIA_TYPE,
        GRAPH_EVIDENCE_SCHEMA,
        GraphVerificationEvidenceRef,
        QualificationProfile,
        ReferenceManifest,
    )
    from tests.test_calibration import _manifest as calibration_manifest
    from tests.test_marginal_runtime import _case
    from tests.test_qualification import _requirement

    if candidate_lane not in {"left", "right"}:
        raise AssertionError("test candidate lane is unsupported")
    baseline_lane = "left" if candidate_lane == "right" else "right"
    case = _case(tmp_path / "runtime")

    def logical_hardware(lane: str):
        return replace(
            case.launch.hardware,
            device_policy_digest=_d(f"typed-resident-{lane}-device-policy"),
        )

    def physical_hardware(lane: str):
        return replace(
            case.baseline_binding.launch_binding.physical_hardware,
            physical_gpu_ids=(("0",) if lane == "left" else ("1",)),
            device_policy_digest=_d(f"typed-resident-{lane}-device-policy"),
        )

    candidate_hardware = logical_hardware(candidate_lane)
    candidate_physical = physical_hardware(candidate_lane)
    candidate_launch_policy_digest = _d(
        f"typed-resident-{candidate_lane}-launch-policy"
    )
    baseline_launch_policy_digest = _d(
        f"typed-resident-{baseline_lane}-launch-policy"
    )
    incumbent_launch = replace(
        case.launch,
        hardware=candidate_hardware,
        resource_policy_digest=candidate_launch_policy_digest,
    )
    incumbent_binding = MaterializedArmBinding(
        case.baseline_binding.tree,
        replace(
            case.baseline_binding.launch_binding,
            physical_hardware=candidate_physical,
        ),
    )
    candidate_binding = MaterializedArmBinding(
        case.candidate_binding.tree,
        replace(
            case.candidate_binding.launch_binding,
            physical_hardware=candidate_physical,
        ),
    )
    baseline_session = replace(
        case.session,
        launch_digest=incumbent_launch.digest,
        expected_preflight=expected_runtime_preflight(
            incumbent_launch,
            case.preflight,
        ),
    )
    prepared = prepare_marginal_runtime(
        case.arm,
        catalog=case.catalog,
        expected_context=case.context,
        incumbent_launch=incumbent_launch,
        incumbent_binding=incumbent_binding,
        candidate_binding=candidate_binding,
        baseline_session_plan=baseline_session,
    )
    candidate = prepared.candidates[0]
    runtime_policy_digest = _d(
        f"typed-resident-{candidate_lane}-runtime-policy"
    )
    baseline_runtime_policy_digest = _d(
        f"typed-resident-{baseline_lane}-runtime-policy"
    )

    requirement = _requirement(atomic=False)
    requirement = replace(
        requirement,
        binding=replace(
            requirement.binding,
            marginal_arm_digest=candidate.arm.digest,
            candidate_launch_digest=candidate.launch.digest,
            contribution_ref_digest=candidate.arm.transition.replacement.digest,
            selected_delta_digest=candidate.arm.selected_delta_digest,
            target_id=candidate.arm.transition.target_id,
            target_spec_digest=candidate.arm.transition.target_spec_digest,
            catalog_digest=candidate.arm.candidate.catalog_digest,
        ),
    )
    workload_digest = runner.marginal_workload_digest(
        prepared.baseline_session_plan
    )
    reference = ReferenceManifest.from_pristine(
        case.incumbent,
        prepared.baseline_launch,
        prepared.incumbent_binding,
        workload_digest=workload_digest,
        tokenizer_digest=_d(f"typed-resident-{candidate_lane}-tokenizer"),
        hidden_corpus_commitment=_d(
            f"typed-resident-{candidate_lane}-hidden-corpus"
        ),
        hidden_judge_digest=_d(
            f"typed-resident-{candidate_lane}-hidden-judge"
        ),
        selection_policy_digest=_d(
            f"typed-resident-{candidate_lane}-selection-policy"
        ),
    )
    resident_baseline_launch = replace(
        prepared.baseline_launch,
        hardware=logical_hardware(baseline_lane),
        resource_policy_digest=baseline_launch_policy_digest,
    )
    resident_baseline_plan = replace(
        prepared.baseline_session_plan,
        launch_digest=resident_baseline_launch.digest,
        expected_preflight=expected_runtime_preflight(
            resident_baseline_launch,
            case.preflight,
        ),
    )
    baseline_arm = ResidentArmPlan(
        resident_baseline_launch,
        replace(
            prepared.incumbent_binding.launch_binding,
            physical_hardware=physical_hardware(baseline_lane),
        ),
        resident_baseline_plan,
        _d(f"typed-resident-{baseline_lane}-namespace"),
        baseline_runtime_policy_digest,
        _d(f"typed-resident-{baseline_lane}-device-configuration"),
    )
    candidate_arm = ResidentArmPlan(
        candidate.launch,
        candidate.binding.launch_binding,
        candidate.session_plan,
        _d(f"typed-resident-{candidate_lane}-namespace"),
        runtime_policy_digest,
        _d(f"typed-resident-{candidate_lane}-device-configuration"),
    )
    calibration_context = CalibrationContext(
        reference.measured_digest,
        reference.arena_digest,
        reference.runtime_digest,
        reference.base_engine_digest,
        reference.model_revision_digest,
        reference.model_manifest_digest,
        reference.model_content_digest,
        reference.logical_hardware_digest,
        reference.workload_digest,
        requirement.binding.verification_policy_digest,
    )
    calibration = replace(
        calibration_manifest(),
        context=calibration_context,
        raw_evidence_digest=_d(
            f"typed-resident-{candidate_lane}-calibration-raw"
        ),
    )
    threshold_policy = CalibrationThresholdPolicy.from_manifest(calibration)
    resident_plan = ResidentCrossoverPlan(
        candidate.arm.selected_delta_digest,
        baseline_arm,
        candidate_arm,
        ResidentSpeedPolicy.from_calibration(
            max_stage_seconds=60,
            calibration=calibration,
            context=calibration_context,
        ),
    )
    profile = QualificationProfile(
        reference,
        calibration_context.digest,
        calibration.digest,
        requirement.digest,
        tuple(row.name for row in calibration.quality_metrics),
        "2",
        prepared.baseline_session_plan.max_new_tokens,
        prepared.baseline_session_plan.top_logprobs_num,
        1,
        _d("typed-resident-support-policy"),
        _d("typed-resident-hidden-task-policy"),
        runtime_policy_digest,
        True,
        2,
    )
    graph_evidence_ref = GraphVerificationEvidenceRef(
        requirement.binding,
        requirement.digest,
        _d(f"typed-resident-{candidate_lane}-graph-raw"),
    )
    authority = runner.CandidateQualificationAuthority(
        candidate.arm.selected_delta_digest,
        profile,
        requirement,
        EvidenceArtifactRef(
            GRAPH_EVIDENCE_DOMAIN,
            _d(f"typed-resident-{candidate_lane}-graph-artifact"),
            1,
            GRAPH_EVIDENCE_MEDIA_TYPE,
            GRAPH_EVIDENCE_SCHEMA,
        ),
        graph_evidence_ref,
    )
    secret = (f"typed resident {candidate_lane} qualification secret").encode()
    commitment = SelectionCommitment.seal(
        source_plan_digest=prepared.source.digest,
        reference_manifest=reference,
        entropy_source_digest=declared_qualification_entropy_digest(
            reference.selection_policy_digest
        ),
        prompt_digests=runner._planned_prompt_digests(prepared),
        select_count=2,
        secret=secret,
    )
    audit_policy = runner.SlotAuditPolicy(
        "1" * 32,
        250_000,
        32,
        (requirement.binding.members[0].slot_id,),
        prepared.baseline_session_plan.engine_config.tp_size,
    )
    audit_prompt = candidate.session_plan.prompt_batches[0][0]
    resident_audit_plan = runner.ResidentAuditExecutionAuthority.derive(
        candidate.launch,
        candidate.binding.launch_binding,
        candidate.session_plan,
        audit_policy=audit_policy,
        prompt_batches=tuple(
            (audit_prompt,) for _ in range(audit_policy.minimum_calls + 1)
        ),
        max_new_tokens=2,
        top_logprobs_num=1,
        executor_namespace_digest=resident_plan.candidate.executor_namespace_digest,
        runtime_resource_policy_digest=resident_plan.candidate.runtime_resource_policy_digest,
        device_configuration_digest=resident_plan.candidate.device_configuration_digest,
    )
    return runner.CausalQualificationInput(
        prepared=prepared,
        model_mount=case.mount,
        candidates=(authority,),
        commitment=commitment,
        selection_secret=secret,
        evidence_root=tmp_path,
        calibration_threshold_policy=threshold_policy,
        calibration_manifest=calibration,
        calibration_context=calibration_context,
        calibration_artifact_ref=EvidenceArtifactRef(
            "qualification-calibration",
            _d(f"typed-resident-{candidate_lane}-calibration-artifact"),
            1,
            "application/json",
            "calibration-evidence-set-v1",
        ),
        pristine_stack=case.incumbent,
        pristine_launch=prepared.baseline_launch,
        pristine_binding=prepared.incumbent_binding.launch_binding,
        reference_engine_config=prepared.baseline_session_plan.engine_config,
        reference_preflight=prepared.baseline_session_plan.expected_preflight,
        expected_launch_resource_policy_digest=(
            prepared.baseline_launch.resource_policy_digest
        ),
        expected_runtime_resource_policy_digest=runtime_policy_digest,
        expected_device_policy_digest=(
            prepared.baseline_launch.hardware.device_policy_digest
        ),
        audit_policies=(audit_policy,),
        speed_evidence_policy=runner.SpeedEvidencePolicy.resident(),
        resident_speed_plan=resident_plan,
        resident_audit_plan=resident_audit_plan,
    )


def test_typed_resident_input_derives_calibration_from_candidate_reference(
    tmp_path: Path,
) -> None:
    value = _typed_resident_qualification_input(tmp_path)
    plan = value.resident_speed_plan
    assert plan is not None
    reference = value.candidates[0].profile.reference
    verification_policy = value.candidates[0].graph_requirement.binding
    assert (
        plan.baseline.runtime_resource_policy_digest
        != plan.candidate.runtime_resource_policy_digest
    )
    assert (
        plan.baseline.launch.resource_policy_digest
        != plan.candidate.launch.resource_policy_digest
    )
    assert value.calibration_context == runner.CalibrationContext(
        reference.measured_digest,
        reference.arena_digest,
        reference.runtime_digest,
        reference.base_engine_digest,
        reference.model_revision_digest,
        reference.model_manifest_digest,
        reference.model_content_digest,
        reference.logical_hardware_digest,
        reference.workload_digest,
        verification_policy.verification_policy_digest,
    )
    assert reference.logical_hardware_digest == plan.candidate.launch.hardware.digest
    assert reference.workload_digest == runner.marginal_workload_digest(
        plan.baseline.session_plan
    )
    assert plan.policy.calibration_context_digest == value.calibration_context.digest


def test_typed_resident_input_rejects_self_consistent_mismatched_context(
    tmp_path: Path,
) -> None:
    from cacheon.eval.calibration import CalibrationThresholdPolicy
    from cacheon.eval.crossover_runtime import ResidentSpeedPolicy

    value = _typed_resident_qualification_input(tmp_path)
    assert value.resident_speed_plan is not None
    mismatched = replace(
        value.calibration_context,
        logical_hardware_digest=_d("wrong-resident-candidate-hardware"),
    )
    calibration = replace(value.calibration_manifest, context=mismatched)
    policy = ResidentSpeedPolicy.from_calibration(
        max_stage_seconds=value.resident_speed_plan.policy.max_stage_seconds,
        max_qualification_seconds=(
            value.resident_speed_plan.policy.max_qualification_seconds
        ),
        calibration=calibration,
        context=mismatched,
    )
    profile = replace(
        value.candidates[0].profile,
        calibration_context_digest=mismatched.digest,
        calibration_digest=calibration.digest,
    )
    authority = replace(value.candidates[0], profile=profile)

    with pytest.raises(
        runner.QualificationRunnerError,
        match="resident speed plan differs from qualification authority",
    ):
        replace(
            value,
            candidates=(authority,),
            calibration_threshold_policy=CalibrationThresholdPolicy.from_manifest(
                calibration
            ),
            calibration_manifest=calibration,
            calibration_context=mismatched,
            resident_speed_plan=replace(
                value.resident_speed_plan,
                policy=policy,
            ),
        )


def test_typed_resident_input_accepts_exact_swapped_lane_authority(
    tmp_path: Path,
) -> None:
    primary = _typed_resident_qualification_input(
        tmp_path / "primary",
        candidate_lane="right",
    )
    reproduction = _typed_resident_qualification_input(
        tmp_path / "reproduction",
        candidate_lane="left",
    )
    assert primary.resident_speed_plan is not None
    assert reproduction.resident_speed_plan is not None
    primary_plan = primary.resident_speed_plan
    reproduction_plan = reproduction.resident_speed_plan

    assert (
        primary_plan.baseline_lane_digest
        == reproduction_plan.candidate_lane_digest
    )
    assert (
        primary_plan.candidate_lane_digest
        == reproduction_plan.baseline_lane_digest
    )
    assert (
        primary.calibration_context.logical_hardware_digest
        == primary_plan.candidate.launch.hardware.digest
    )
    assert (
        reproduction.calibration_context.logical_hardware_digest
        == reproduction_plan.candidate.launch.hardware.digest
    )
    assert primary.calibration_context != reproduction.calibration_context
    assert primary.calibration_artifact_ref != reproduction.calibration_artifact_ref
    assert (
        primary.calibration_threshold_policy.speed,
        primary.calibration_threshold_policy.quality_metrics,
        primary.calibration_threshold_policy.familywise_z,
        primary.calibration_manifest.controls,
    ) == (
        reproduction.calibration_threshold_policy.speed,
        reproduction.calibration_threshold_policy.quality_metrics,
        reproduction.calibration_threshold_policy.familywise_z,
        reproduction.calibration_manifest.controls,
    )


def test_calibration_speed_continuation_is_exact_and_authority_bound(
    tmp_path: Path,
) -> None:
    value = _typed_resident_qualification_input(tmp_path)
    assert value.speed_stage_disposition is runner.SpeedStageDisposition.TERMINAL

    calibration = replace(
        value,
        speed_stage_disposition=(
            runner.SpeedStageDisposition.CALIBRATION_OBSERVATION
        ),
    )
    assert runner.qualification_authority_digest(calibration) != (
        runner.qualification_authority_digest(value)
    )

    for malformed in ("calibration_observation", True):
        with pytest.raises(
            runner.QualificationRunnerError,
            match="causal qualification input is not exactly typed",
        ):
            replace(value, speed_stage_disposition=malformed)


def test_resident_slot_audit_executes_the_sealed_audit_only_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    value = _typed_resident_qualification_input(tmp_path)
    audit = value.resident_audit_plan
    assert audit is not None
    timed_baseline = SimpleNamespace(session=SimpleNamespace(session_id="a" * 32))
    timed_candidate = SimpleNamespace(session=SimpleNamespace(session_id="b" * 32))
    lifecycle = object.__new__(runner.ResidentMarginalLifecycleEvidence)
    object.__setattr__(
        lifecycle,
        "crossover",
        SimpleNamespace(
            baseline_execution=timed_baseline,
            candidate_execution=timed_candidate,
            baseline_quiescence=SimpleNamespace(observed_monotonic_s=7.0),
            candidate_quiescence=SimpleNamespace(observed_monotonic_s=8.0),
        ),
    )
    execution = SimpleNamespace(
        launch_digest=audit.launch.digest,
        session=SimpleNamespace(session_id="c" * 32),
        resource_policy_digest=value.expected_runtime_resource_policy_digest,
        device_receipts=(
            SimpleNamespace(started_monotonic_s=8.5, completed_monotonic_s=9.0),
        ),
    )

    class CapturingExecutor:
        def __init__(self) -> None:
            self.plans: list[object] = []
            self.manager = SimpleNamespace(
                namespace_digest=audit.executor_namespace_digest
            )
            self.config = SimpleNamespace(
                runtime=SimpleNamespace(
                    digest=audit.runtime_resource_policy_digest
                ),
                prebuild=SimpleNamespace(
                    policy=SimpleNamespace(
                        resource_policy_digest=audit.launch.resource_policy_digest
                    )
                ),
            )
            physical = audit.binding.physical_hardware
            self.device_policy = SimpleNamespace(
                physical_gpu_ids=tuple(map(int, physical.physical_gpu_ids)),
                configuration_sha256=audit.device_configuration_digest,
                policy_sha256=physical.device_policy_digest,
            )

        def execute(self, launch, binding, mount, plan, *, deadline):
            assert launch is audit.launch
            assert binding is audit.binding
            assert mount is value.model_mount
            assert deadline == 10.0
            self.plans.append(plan)
            return execution

    witness = object()

    def from_execution(observed, *, selected_delta_digest, policy):
        assert observed is execution
        assert selected_delta_digest == value.candidates[0].selected_delta_digest
        assert policy == value.audit_policies[0]
        return witness

    monkeypatch.setattr(
        runner.AuditWitness,
        "from_execution",
        staticmethod(from_execution),
    )
    executor = CapturingExecutor()
    expected_namespace = executor.manager.namespace_digest
    executor.manager.namespace_digest = _d("foreign-audit-namespace")
    with pytest.raises(
        runner.QualificationRunnerError, match="commissioned authority"
    ):
        runner._run_slot_audits(
            value, lifecycle, executor=executor, deadline=10.0
        )
    assert executor.plans == []
    executor.manager.namespace_digest = expected_namespace
    execution.device_receipts[0].started_monotonic_s = 7.5
    with pytest.raises(runner.QualificationRunnerError, match="preceded retirement"):
        runner._run_slot_audits(
            value, lifecycle, executor=executor, deadline=10.0
        )
    executor.plans.clear()
    execution.device_receipts[0].started_monotonic_s = 8.5
    execution.session.session_id = "a" * 32
    with pytest.raises(runner.QualificationRunnerError, match="reused a pair role"):
        runner._run_slot_audits(
            value, lifecycle, executor=executor, deadline=10.0
        )
    executor.plans.clear()
    execution.session.session_id = "c" * 32
    witnesses, completed = runner._run_slot_audits(
        value,
        lifecycle,
        executor=executor,  # type: ignore[arg-type]
        deadline=10.0,
    )

    assert executor.plans == [audit.plan]
    assert executor.plans[0] is not value.prepared.candidates[0].session_plan
    assert witnesses == {value.candidates[0].selected_delta_digest: witness}
    assert completed == 9.0

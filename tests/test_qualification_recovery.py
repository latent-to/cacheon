"""Validator failures retain completed evidence instead of rejecting miners."""

from dataclasses import replace
import hashlib
import json

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


def test_replacement_reference_uses_its_actual_launch_and_frozen_math(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tests.test_b300_registered_qualification as fixtures
    from tests.test_device_state import _policy
    from tests.test_oci_reference_session import _request
    from cacheon.eval import qualification_worker_checkpoint as checkpoint
    from cacheon.eval.evidence_store import publish_evidence
    from cacheon.eval.oci_backend import expected_runtime_preflight
    from cacheon.eval.oci_reference_session import ReferenceSessionPlan
    from cacheon.eval.qualification_continuation import QualificationContinuationStore
    from cacheon.stack_identity import canonical_json_bytes

    original_case, original_binding = fixtures._case, fixtures._local_binding

    def moved_binding(tree, native, launch, preflight):
        binding = original_binding(tree, native, launch, preflight, physical_id="2")
        return replace(binding, launch_binding=replace(binding.launch_binding,
            physical_hardware=replace(binding.launch_binding.physical_hardware,
                device_policy_digest=launch.hardware.device_policy_digest)))

    def moved_case(path):
        case = original_case(path)
        case.launch = replace(case.launch, hardware=replace(case.launch.hardware,
            device_policy_digest=_d("replacement-device-policy")))
        case.baseline_binding = moved_binding(case.baseline_tree,
            case.baseline_binding.launch_binding.native_build_spec, case.launch, case.preflight)
        case.session = replace(case.session, launch_digest=case.launch.digest,
            expected_preflight=expected_runtime_preflight(case.launch, case.preflight))
        return case

    first = fixtures._harness(tmp_path / "original")
    secret = b"replacement-regression-private-seed"
    value = first.factory.plan_builder(first.cohort, secret)
    with monkeypatch.context() as changed:
        changed.setattr(fixtures, "_case", moved_case)
        changed.setattr(fixtures, "_local_binding", moved_binding)
        second = fixtures._harness(tmp_path / "replacement")
        replacement = second.factory.plan_builder(second.cohort, b"replacement-reference-fresh-private-seed")
    scope = QualificationContinuationStore(tmp_path / "checkpoint").scope(
        request_digest=_d("original-request"), authority_digest=runner.qualification_authority_digest(value),
        source_digest=value.prepared.source.digest)
    executor = SimpleNamespace(device_policy=_policy())
    origins = []
    for clock, plan in (("old-boot", value), ("new-boot", replacement)):
        monkeypatch.setattr(checkpoint, "_clock_domain", lambda clock=clock: clock)
        origins.append(checkpoint._public_origin(checkpoint.capture_origin(plan, scope, executor, executor)))
    body = {"schema": checkpoint.RECOVERY_SCHEMA, "authority": runner.qualification_authority_digest(value),
            "origins": {"speed": origins[0]}, "replacement_boot": "new-boot", "replacement": origins[1]}
    reference = publish_evidence(value.evidence_root, canonical_json_bytes(body),
        domain=checkpoint.RECOVERY_DOMAIN, media_type="application/json", schema=checkpoint.RECOVERY_SCHEMA)
    _audit, actual = checkpoint.result_execution_authorities(value, reference, ("old-boot", "new-boot", "new-boot"))
    assert actual.pristine_launch.digest != value.pristine_launch.digest
    manifest = actual.candidates[0].profile.reference
    request = _request(manifest, session_id="1" * 32, plan_digest=_d("reference-request"), index=0)
    plan = ReferenceSessionPlan(manifest, actual.pristine_stack, actual.reference_engine_config.digest,
        actual.reference_engine_config, actual.reference_preflight, request.plan_digest, (request,))
    assert plan.reference.pristine_launch_digest == actual.pristine_launch.digest
    session = SimpleNamespace(reference_manifest_digest=manifest.digest, launch_digest=actual.pristine_launch.digest)
    frozen = value.candidates[0].profile.reference
    assert checkpoint.reference_execution_matches(frozen, manifest, session)
    assert not checkpoint.reference_execution_matches(frozen, None, session)
    assert not checkpoint.reference_execution_matches(frozen, replace(manifest, model_content_digest=_d("other-model")), session)
    with pytest.raises(QualificationContinuationError, match="clock domains"):
        checkpoint.result_execution_authorities(value, reference, ("new-boot", "new-boot", "new-boot"))


@pytest.mark.parametrize("profile", ["ordinary", "glm"])
@pytest.mark.parametrize("stage", ["speed", "speed_window_1"])
def test_recommissioned_arena_reconstructs_the_original_measured_work(tmp_path, monkeypatch, profile, stage):
    from types import SimpleNamespace
    import tests.test_b300_registered_qualification as fixtures
    import tests.test_marginal_runtime as runtime_fixtures
    from cacheon.eval import qualification_worker_checkpoint as checkpoint
    from cacheon.eval.qualification_continuation import QualificationContinuationStore
    from cacheon.eval.qualification_intake import QualificationAuthorityManifest
    from cacheon.eval.qualification_prebuilt_plan import sealed_prebuilt_qualification_plan_factory
    from cacheon.arena_service import ArenaQualificationWork
    from tests.test_device_state import _policy

    source = fixtures.FUSED if profile == "glm" else None
    first = fixtures._harness(tmp_path / "original", source)
    secret = b"recommissioned-arena-regression-seed"
    original = first.factory.plan_builder(first.cohort, secret)
    original_digest = runtime_fixtures._digest
    fixture_digest = fixtures._h
    with monkeypatch.context() as changed:
        changed.setattr(runtime_fixtures, "_digest", lambda value:
            original_digest("new-worker-" + value) if value in ("arena", "resources") else original_digest(value))
        changed.setattr(fixtures, "_h", lambda value: fixture_digest("new-worker-" + value)
            if value in ("resident-candidate-runtime-policy", "resident-baseline-runtime-policy") else fixture_digest(value))
        second = fixtures._harness(tmp_path / "replacement", source)
        replacement = second.factory.plan_builder(second.cohort, b"replacement-prebuilt-plan-private-seed")
    if stage != "speed":
        def portable(value):
            return replace(value, resident_speed_plan=replace(value.resident_speed_plan,
                policy=replace(value.resident_speed_plan.policy, version=18, max_stage_seconds=600)))
        original, replacement = portable(original), portable(replacement)
    assert replacement.prepared.source.digest != original.prepared.source.digest
    scope = QualificationContinuationStore(tmp_path / "checkpoint").scope(
        request_digest=_d("request"), authority_digest=runner.qualification_authority_digest(original),
        source_digest=original.prepared.source.digest)
    executor = SimpleNamespace(device_policy=_policy())
    monkeypatch.setattr(checkpoint, "_clock_domain", lambda: "original-boot")
    origin = checkpoint.capture_origin(original, scope, executor, executor)
    scope.checkpoint_origin = origin
    if stage == "speed":
        scope.record_resident_speed(_retained_speed(original, tmp_path / "speed"))
    else:
        from cacheon.eval import replay_checkpoint as progress
        from cacheon.eval.goodput_runtime import _orientation_plan
        from tests.test_replay_checkpoint import _execution
        plan = original.resident_speed_plan
        part = _orientation_plan(plan, plan.baseline.session_plan.replay.windows // 2)
        runs = tuple(_execution(original, arm, role, 1, role, tmp_path / "speed", 10.0)
            for arm, role in ((part.baseline, "incumbent"), (part.candidate, "candidate")))
        scope._record(stage, progress._record(plan, runs, 1, 0,
            part.baseline.session_plan.replay.windows, progress._origin(plan), 10.0, 1))
    pointer = json.loads((scope.checkpoint_root / (scope.request_digest + ".json")).read_bytes())
    store = QualificationContinuationStore(tmp_path / "replacement-checkpoint")
    imported = store.root / "imports"
    imported.mkdir()
    name = pointer["sha256"] + ".json.gz"
    (imported / name).write_bytes((scope.checkpoint_root / name).read_bytes())
    request = _d("replacement-request")
    (imported / (request + ".json")).write_text(json.dumps(pointer))
    manifest = QualificationAuthorityManifest.seal(replacement,
        reservations=(second.candidate.reservation,), selection_secret_reference=_d("replacement-secret"))
    factory = sealed_prebuilt_qualification_plan_factory(manifest, plan=replacement,
        selection_secret=replacement.selection_secret, selection_secret_reference=_d("replacement-secret"))
    work = ArenaQualificationWork(factory, executor, lambda: None, lambda: None, 1000.0,
                                  second.cohort.request.qualification_policy_digest, executor)
    recovered = checkpoint.prepare_work(work, store, request)
    assert recovered.factory.build().selection_secret == secret
    assert recovered.factory.build().commitment == original.commitment
    assert store.worker_replacements[request] == replacement
    if stage != "speed":
        resumed = store.scope(request_digest=request, authority_digest=runner.qualification_authority_digest(original),
                              source_digest=original.prepared.source.digest)
        checkpoint.configure_continuation(recovered.factory.build(), resumed, executor, executor)
        assert resumed.speed_windows == (scope._load(stage),)
        assert len(progress._reads(resumed.speed_windows, original.resident_speed_plan).incumbent) == 1
    restored = checkpoint.restore_origin(replacement, origin)
    assert runner.qualification_authority_digest(restored) == runner.qualification_authority_digest(original)
    assert restored.prepared.source == original.prepared.source
    assert restored.pristine_stack == original.pristine_stack
    assert restored.model_mount.model_root == replacement.model_mount.model_root
    assert restored.prepared.candidates[0].binding.tree.root != original.prepared.candidates[0].binding.tree.root
    assert checkpoint.restore_origin(replacement, origin).prepared.source == restored.prepared.source
    actual = replacement.candidates[0].profile.reference
    assert checkpoint.reference_execution_matches(original.candidates[0].profile.reference, actual,
        SimpleNamespace(reference_manifest_digest=actual.digest, launch_digest=actual.pristine_launch_digest))
    changed = json.loads(json.dumps(origin))
    changed["arms"][0]["launch"]["engine_config_digest"] = _d("different-engine-settings")
    with pytest.raises(QualificationContinuationError, match="executable envelope"):
        checkpoint.restore_origin(replacement, changed)


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
        graded, detail = gate(
            [row.to_gate_dict() for row in receipts],
            min_calls=audit.policy.minimum_calls,
            expected_slots=audit.policy.expected_slots,
            expected_member_count=audit.policy.expected_member_count,
        )
        assert graded != "PASS"
        return {key: replace(audit, receipts=receipts,
                             decision=QualificationDecision(graded), detail=detail)}, completed

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
    assert gate([receipt], **kwargs)[0] == "FAIL"
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


def _retained_speed(value, tmp_path):
    """Build typed completed replay evidence shared by the recovery regressions."""
    from cacheon.eval.crossover_runtime import ResidentCrossoverEvidence
    from cacheon.eval.oci_process import OCIQuiescenceReceipt
    from cacheon.eval.scoring import SpeedupVerdict, marginal_workload_digest
    from cacheon.eval.speed_verdict import SpeedStageDecision
    from tests.test_dashboard_replay import _witness
    from tests.test_qualification_continuation import _typed_execution

    # These typed fixtures exercise persistence; replay grading has separate tests.
    plan = value.resident_speed_plan
    runs = {(name, index): replace(
        _typed_execution(arm.launch, arm.binding, value.model_mount, arm.session_plan,
                         label=f"{name}-{index}", artifact_root=tmp_path / name),
        resource_policy_digest=arm.runtime_resource_policy_digest)
        for name, arm in (("baseline", plan.baseline), ("candidate", plan.candidate)) for index in (0, 1)}
    quiet = lambda arm, sequence: OCIQuiescenceReceipt(
        "cacheon.oci-quiescence.v1", "lane", "a" * 32, arm.executor_namespace_digest,
        sequence, 5.0 + sequence, (), (), ())
    verdict = SpeedupVerdict(1.1, 0.01, 1.0, True, True, 2)
    return ResidentCrossoverEvidence(
        plan.digest, plan.selected_delta_digest, plan.policy,
        marginal_workload_digest(plan.baseline.session_plan), plan.baseline_lane_digest,
        plan.candidate_lane_digest, runs["baseline", 1], runs["candidate", 1],
        quiet(plan.baseline, 1), quiet(plan.candidate, 2), (), verdict, verdict, False,
        SpeedStageDecision.PASS, "clear_pass", 0.0, 50.0,
        goodput=_witness(statistical=True).goodput,
        prior_executions=(runs["baseline", 0], runs["candidate", 0]),
    )


@pytest.mark.parametrize("replace_audit", [True, False])
def test_recovery_preserves_parent_and_is_idempotent(tmp_path, monkeypatch, replace_audit):
    from types import SimpleNamespace
    from cacheon.eval import qualification_recovery as recovery
    from cacheon.eval.qualification_continuation import QualificationContinuationStore, AuditContinuation

    value, authority, execution, audit, old, _payload = _correction(tmp_path)
    speed = _retained_speed(value, tmp_path)
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


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("change,new_boot", [
    ({}, False), ({}, True), ({"driver_version": "580.173.02"}, False),
    ({"driver_version": "580.173.02"}, True), ({"power_limit_mw": 1_100_000}, True),
])
def test_worker_replacement_restores_completed_work_or_restarts_partial_speed(
    tmp_path, monkeypatch, complete, change, new_boot,
):
    from types import SimpleNamespace
    import tests.test_b300_registered_qualification as fixtures
    from cacheon.eval import qualification_worker_checkpoint as checkpoint
    from cacheon.eval.qualification_continuation import QualificationContinuationStore
    from tests.test_device_state import _policy
    from tests.test_qualification_runner import _quiescence

    harness = fixtures._harness(tmp_path / "plan")
    value = harness.factory.plan_builder(harness.cohort, b"worker-replacement-regression-private-seed")
    speed = _retained_speed(value, tmp_path / "speed")
    executor = SimpleNamespace(device_policy=_policy())
    def scope(root, request, plan=value):
        return QualificationContinuationStore(root).scope(
            request_digest=_d(request), authority_digest=runner.qualification_authority_digest(plan),
            source_digest=plan.prepared.source.digest)

    original = scope(tmp_path / "original", "original-request")
    monkeypatch.setattr(checkpoint, "_clock_domain", lambda: "old-boot")
    checkpoint.configure_continuation(value, original, executor, executor)
    if complete:
        original.record_resident_speed(speed)
    else:
        original.record_speed_orientation(value.resident_speed_plan.digest, "old-boot", 0.0,
            speed.prior_executions, (_quiescence(1, 5.0), _quiescence(2, 6.0)))
    pointer = json.loads((original.checkpoint_root / (original.request_digest + ".json")).read_bytes())
    retained = original.checkpoint_root / (pointer["sha256"] + ".json.gz")
    raw = retained.read_bytes()
    replacement_value = (harness.factory.plan_builder(harness.cohort,
        b"replacement-request-fresh-selection-seed") if not complete and (new_boot or change) else value)
    replacement = scope(tmp_path / "replacement", "replacement-request", replacement_value)
    imported = replacement.directory.parent / "imports"
    imported.mkdir()
    (imported / retained.name).write_bytes(raw)
    (imported / (replacement.request_digest + ".json")).write_text(json.dumps({"sha256": pointer["sha256"]}))
    executor.device_policy = replace(executor.device_policy, expected_gpus=tuple(
        replace(gpu, **change) for gpu in executor.device_policy.expected_gpus))
    clock = "new-boot" if new_boot else "old-boot"
    monkeypatch.setattr(checkpoint, "_clock_domain", lambda: clock)
    checkpoint.configure_continuation(replacement_value, replacement, executor, executor)
    assert retained.read_bytes() == raw
    if complete:
        restored = replacement.worker_recovery["speed"]
        assert restored.continuation.load_resident_speed() == speed
        assert restored.origin["boot"] == "old-boot"
        assert replacement.worker_recovery_reference is not None
    elif new_boot or change:
        assert replacement.load_speed_orientation(value.resident_speed_plan.digest, clock) is None
        assert replacement.restored_stages == {}
    else:
        assert replacement.load_speed_orientation(value.resident_speed_plan.digest, clock) == (
            original.load_speed_orientation(value.resident_speed_plan.digest, clock))


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

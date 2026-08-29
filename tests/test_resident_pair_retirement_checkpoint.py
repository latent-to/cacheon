from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import MethodType

import pytest

import cacheon.eval.qualification_runner as runner_module
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteEvaluationDispatcherError,
    capture_remote_qualification_product,
    import_remote_qualification_evidence,
)
from cacheon.eval.crossover_runtime import (
    ResidentArmPlan,
    ResidentCrossoverPlan,
    ResidentSpeedPolicy,
    SpeedStageDecision,
)
from cacheon.eval.device_state import DeviceStateReceipt
from cacheon.eval.oci_backend import (
    ResidentEngineExecutionEvidence,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.oci_resident_session import ResidentBatchEvidence, ResidentSessionEvidence
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
    run_qualification_intake,
)
from cacheon.eval.qualification_runner import (
    HiddenJudgeBinding,
    STAGE_EXIT_SCHEMA_V2,
    STAGE_EXIT_SCHEMA_V3,
    _resident_closure_codec,
    qualification_authority_digest,
    reopen_qualification_stage_exit,
    run_causal_qualification,
)
from cacheon.eval.registered_resident_count_quality import (
    B300ResidentCountQualityCapability,
)
from cacheon.eval.resident_count_quality import (
    CountQualityPolicy,
    ResidentCountPromptObservation,
    ResidentCountQualityObservation,
    publish_resident_count_observation,
    seal_resident_count_stock_authority,
)
from cacheon.eval.resident_count_quality_execution import (
    execute_candidate_count_quality,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverPlan,
    run_resident_pair_crossover,
)
from cacheon.eval.resident_evaluation_pair import ResidentEvaluationPair
from cacheon.eval.resident_pair_binding import (
    ResidentPairLaneBinding,
    ResidentPairRuntimeBinding,
)
from cacheon.eval.resident_pair_quality_lifecycle import (
    ResidentPairMarginalLifecycleEvidence,
    ResidentPairQualityLifecycleError,
)
from cacheon.eval.resident_pair_retirement_checkpoint import (
    ResidentPairRetirementHold,
    build_resident_pair_retirement_checkpoint,
    regrade_resident_pair_retirement_checkpoint,
)
from tests import test_resident_pair_crossover as pair_fixtures
from tests import test_resident_count_quality_execution as count_fixtures
from tests import test_qualification_graph_exit as graph_fixtures
from tests.test_b300_resident_pair_factory import (
    _authority,
    _commissioned,
    _h,
)
from tests.test_marginal_runtime import FUSED


pytest_plugins = ("tests.test_b300_resident_pair_factory",)
_PAIR_FACTORY_CALL = pair_fixtures._Factory.__call__


def _arm(plan):
    return ResidentArmPlan(
        plan.stock_launch,
        plan.stock_binding,
        plan.speed_workload,
        plan.executor.manager.namespace_digest,
        plan.executor.config.runtime.digest,
        plan.executor.device_policy.configuration_sha256,
    )


def _count_row(session, prompts, outputs, *, canary):
    started, completed = session.clock.span(0.1)
    raw = BatchEvidence(
        tuple(
            PromptEvidence(outputs[prompt], tuple(() for _ in outputs[prompt]), 5)
            for prompt in prompts
        )
    )
    index = len(session.batch_rows)
    row = ResidentBatchEvidence(
        index,
        _h(f"{session.session_id}:count-request:{index}")[:32],
        _h(f"{session.session_id}:count-nonce:{index}")[:32],
        session.active_generation,
        session.active_slots,
        canary,
        started,
        completed,
        raw.observed_tokens,
        raw,
    )
    session.batch_rows.append(row)
    return row


def _count_template(profile, lane_a_digest, lane_b_digest):
    template, judge, unused_pair, factory_a, _factory_b = count_fixtures._fixture(
        total=4, barrier=False, profile=profile
    )
    unused_pair.close()
    admission = replace(
        template.admission,
        lane_a_allocation_digest=lane_a_digest,
        lane_b_allocation_digest=lane_b_digest,
    )
    envelope = replace(template.envelope, admission_policy_digest=admission.digest)
    return template, judge, factory_a.outputs, admission, envelope


def _run_prefix(coordinator, commissioned, harness, plan, capability, continuation):
    return coordinator.run_b300_resident_qualification_prefix(
        factory=commissioned.factory,
        capability=capability,
        candidate=harness.candidate,
        plan=plan,
        continuation=continuation,
        screen_lane="primary",
        deadline=time.monotonic() + 30.0,
    )


def _plan_scope(root, request_digest, plan):
    return QualificationContinuationStore(root).scope(
        request_digest=request_digest,
        authority_digest=qualification_authority_digest(plan),
        source_digest=plan.prepared.source.digest,
    )


def _authority_scope(root, authority):
    return QualificationContinuationStore(root).scope(
        request_digest=authority.authenticated_request_digest,
        authority_digest=authority.qualification_authority_digest,
        source_digest=_h("source"),
    )


def _forbidden_judge(profile, message):
    class Forbidden:
        binding = HiddenJudgeBinding(
            profile.reference.hidden_corpus_commitment,
            profile.reference.hidden_judge_digest,
            profile.hidden_task_policy_digest,
        )

        def __call__(self, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError(message)

    return Forbidden()


def _never_called(message):
    def refuse(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError(message)

    return refuse


def _readiness(plan, reference, name):
    hardware = plan.pristine_launch.hardware
    return WorkerReadiness(
        _h(f"{name}-ready"), 1, plan.pristine_stack.arena_digest, name,
        _h(f"{name}-provider"), plan.pristine_stack.runtime_digest,
        reference.worker_distribution_digest, reference.model_revision_digest,
        reference.model_manifest_digest, reference.model_content_digest,
        hardware.architecture, hardware.topology_class, hardware.topology_digest,
        hardware.visible_gpu_count, plan.reference_engine_config.tp_size,
        reference.workload_digest, _h(f"{name}-policy"))


def _capture(batch, authority, plan, reference, readiness, refs):
    return capture_remote_qualification_product(
        batch=batch, authority_manifest=authority,
        incumbent_stack=plan.pristine_stack,
        incumbent_tree_digest=reference.pristine_tree_digest,
        screen_lane="primary", service_digest=readiness.service_digest,
        readiness=readiness, evidence_root=plan.evidence_root,
        evidence_references=refs)


def _install_exact_lifetimes(monkeypatch, commissioned, lifetimes):
    original = pair_fixtures._Factory.__call__

    def execute(factory, driver):
        original(factory, driver)
        session = factory.sessions[-1]
        plan = commissioned.plans[0 if factory.lane_id == "A" else 1]
        completed = float(lifetimes.clock())
        launch_id = _h(f"launch:{factory.session_id}")[:32]

        def device(sequence, phase, timestamp):
            return DeviceStateReceipt(
                "cacheon.device-state-receipt.v1",
                sequence,
                launch_id,
                phase,
                plan.lane_policy.physical_gpu_ids,
                plan.lane_policy.device_configuration_digest,
                plan.lane_policy.device_policy_digest,
                float(timestamp - 0.01),
                float(timestamp),
                2,
                (),
            )

        return ResidentEngineExecutionEvidence(
            "cacheon.oci-resident-queue-execution.v1",
            plan.stock_launch.digest,
            runtime_identity_from_preflight(
                plan.stock_binding.runtime_preflight_receipt
            ),
            plan.stock_binding.runtime_preflight_receipt.sha256,
            commissioned.model_mount.digest,
            plan.executor.config.runtime.digest,
            None,
            _h(f"publication:{factory.session_id}"),
            _h(f"argv:{factory.session_id}"),
            (),
            (device(1, "pre", 0.9), device(2, "post", completed + 0.1)),
            ResidentSessionEvidence(
                factory.session_id,
                plan.stock_launch.digest,
                plan.resident_plan.expected_preflight,
                1.0,
                tuple(session.batch_rows),
                tuple(session.swap_receipts),
                completed,
            ),
        )

    monkeypatch.setattr(pair_fixtures._Factory, "__call__", execute)
    for plan in commissioned.plans:
        manager = plan.executor.manager

        def prove(plan=plan, manager=manager):
            return OCIQuiescenceReceipt(
                "cacheon.oci-quiescence.v1",
                manager.executor_id,
                manager.manager_instance_id,
                manager.namespace_digest,
                1,
                float(lifetimes.clock() + 1.0),
                (),
                (),
                (),
            )

        monkeypatch.setattr(plan.executor, "prove_quiescent", prove)


def _run_count_on_pair(borrowed, speed_plan, sessions, *, fast_lane):
    template, judge, outputs, admission, envelope = _count_template(
        f"retirement-{fast_lane.lower()}",
        borrowed.binding.lanes[0].allocation_digest,
        borrowed.binding.lanes[1].allocation_digest,
    )
    plan = replace(
        template,
        candidate_bundle_digest=speed_plan.candidate_bundle_digest,
        envelope=envelope,
        admission=admission,
        pair_binding=borrowed.binding,
    )
    release = threading.Event()
    fast_session = borrowed.binding.lookup(fast_lane).session_id

    def execute(session, prompts, *, shape, canary=False):
        if session.session_id != fast_session:
            assert release.wait(2.0)
        row = _count_row(session, prompts, outputs, canary=canary)
        if session.session_id == fast_session:
            release.set()
        return row

    for session in sessions:
        session.execute_batch_with_shape = MethodType(execute, session)
    result = execute_candidate_count_quality(
        plan, pair=borrowed.pair, judge=judge, deadline=time.monotonic() + 2.0
    )
    return plan, result.evidence


def _coordinator_case(
    tmp_path,
    monkeypatch,
    lifetimes,
    managed_executors,
    *,
    source_fixture=None,
    count_drop=0,
):
    from cacheon.eval import b300_resident_qualification as coordinator

    commissioned = _commissioned(tmp_path / "pair", lifetimes, managed_executors)
    _install_exact_lifetimes(monkeypatch, commissioned, lifetimes)
    harness, plan, _authority_row = graph_fixtures._plan(
        tmp_path / "qualification",
        failure=False,
        source_fixture=source_fixture,
    )
    plan = replace(plan, model_mount=commissioned.model_mount)
    # The pair-native grade starts at v6; the graph fixture still seals the
    # pre-pair v3 policy, so lift it and let the fixture's one-window reads
    # through the window gate.
    resident = plan.resident_speed_plan
    assert resident is not None
    plan = replace(
        plan,
        resident_speed_plan=replace(
            resident, policy=replace(resident.policy, version=6)
        ),
    )
    monkeypatch.setattr(
        ResidentSpeedPolicy, "read_window_scatter", lambda _policy, _row: 0.0
    )
    template, judge, outputs, admission, envelope = _count_template(
        "coordinator",
        commissioned.plans[0].lane_policy.digest,
        commissioned.plans[1].lane_policy.digest,
    )
    stock_rows = tuple(
        ResidentCountPromptObservation(
            ordinal,
            occurrence.prompt_digest,
            occurrence.task_digests,
            outputs[prompt],
            judge(
                prompt_digest=occurrence.prompt_digest,
                output_ids=outputs[prompt],
                task_digests=occurrence.task_digests,
            ),
        )
        for ordinal, (occurrence, prompt) in enumerate(
            zip(template.selected_occurrences, template.selected_prompts, strict=True)
        )
    )
    stock_observation = ResidentCountQualityObservation(
        "stock", envelope, _h("coordinator:stock-execution"), stock_rows
    )
    stock = seal_resident_count_stock_authority(
        plan.evidence_root,
        publish_resident_count_observation(plan.evidence_root, stock_observation),
        policy=CountQualityPolicy(2),
    )
    capability = B300ResidentCountQualityCapability(
        harness.inputs.catalog,
        envelope,
        template.prompt_batches,
        template.selected_ordinals,
        template.batch_shape,
        admission,
        stock,
        judge,
    )
    original_outputs = tuple(outputs[prompt] for prompt in template.selected_prompts)
    for index in range(count_drop):
        outputs[template.selected_prompts[index]] = original_outputs[index + 1]

    original_execute = pair_fixtures._Session.execute_batch_with_shape

    def execute(session, prompts, *, shape, canary=False):
        prompts = tuple(prompts)
        if shape == template.batch_shape and all(row in outputs for row in prompts):
            return _count_row(session, prompts, outputs, canary=canary)
        return original_execute(session, prompts, shape=shape, canary=canary)

    monkeypatch.setattr(
        pair_fixtures._Session, "execute_batch_with_shape", execute
    )

    def speed_plan(
        _plan,
        _factory,
        pair,
        retirement,
        *,
        candidate_bundle_digest,
        screen_lane,
    ):
        del _plan, _factory, screen_lane
        if pair is not None:
            lifetimes.factories[1].sessions[0].durations = (0.8,)
        binding = pair.binding if pair is not None else retirement.pair_binding
        crossover = ResidentCrossoverPlan(
            plan.candidates[0].selected_delta_digest,
            _arm(commissioned.plans[0]),
            _arm(commissioned.plans[1]),
            ResidentSpeedPolicy(
                60,
                0.005,
                2.0,
                0.01,
                _h("coordinator:calibration"),
                _h("coordinator:calibration-context"),
                version=6,
                min_windows=3,
                max_window_scatter=0.01,
                max_conditioning_slowdown=1.5,
            ),
        )
        return ResidentPairCrossoverPlan(
            candidate_bundle_digest, crossover, binding, "A", "B"
        )

    monkeypatch.setattr(coordinator, "_speed_plan", speed_plan)
    continuation = _plan_scope(
        tmp_path / "continuation", _h("coordinator:authenticated-request"), plan
    )
    return coordinator, commissioned, harness, plan, capability, continuation


@pytest.mark.parametrize("fast_lane", ("A", "B"))
def test_retirement_publishes_once_and_reopens_without_live_pair(
    tmp_path, monkeypatch, lifetimes, managed_executors, fast_lane
):
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    _install_exact_lifetimes(monkeypatch, commissioned, lifetimes)
    authority = _authority("retirement")
    commissioned.factory.open_request(authority, deadline=100.0)
    borrowed = commissioned.factory.borrow(authority)
    lifetimes.factories[1].sessions[0].durations = (0.8,)
    monkeypatch.setattr(
        ResidentSpeedPolicy, "read_window_scatter", lambda _policy, _row: 0.0
    )
    crossover = ResidentCrossoverPlan(
        _h("retirement:selected-delta"),
        _arm(commissioned.plans[0]),
        _arm(commissioned.plans[1]),
        ResidentSpeedPolicy(
            60,
            0.005,
            2.0,
            0.01,
            _h("calibration"),
            _h("calibration-context"),
            version=6,
            min_windows=3,
            max_window_scatter=0.01,
            max_conditioning_slowdown=1.5,
        ),
    )
    speed_plan = ResidentPairCrossoverPlan(
        _h("retirement:candidate"), crossover, borrowed.binding, "A", "B"
    )
    speed = run_resident_pair_crossover(
        speed_plan,
        pair=borrowed.pair,
        deadline=lifetimes.clock() + 60.0,
        clock=lifetimes.clock,
    )
    assert speed.decision is SpeedStageDecision.PASS
    count_plan, count = _run_count_on_pair(
        borrowed,
        speed_plan,
        tuple(factory.sessions[0] for factory in lifetimes.factories),
        fast_lane=fast_lane,
    )
    checkpoint = build_resident_pair_retirement_checkpoint(
        factory=commissioned.factory,
        authority=authority,
        speed_plan=speed_plan,
        speed=speed,
        count_plan=count_plan,
        count=count,
    )
    retirement_evidence, _proofs = commissioned.factory.retire_and_quiesce(
        authority, speed_plan.pair_binding
    )
    assert tuple(
        row.request_slice.lane_id
        for row in retirement_evidence.request_history[-2:]
    )[0] == fast_lane
    store_root = tmp_path / "continuations"
    continuation = _authority_scope(store_root, authority)
    continuation.record_resident_pair_retirement(checkpoint)
    continuation.record_resident_pair_retirement(checkpoint)

    reopened = _authority_scope(store_root, authority).load_resident_pair_retirement()
    assert regrade_resident_pair_retirement_checkpoint(
        reopened,
        factory=commissioned.factory,
        authority=authority,
        speed_plan=speed_plan,
        speed=speed,
        count_plan=count_plan,
        count=count,
    ) == checkpoint
    with pytest.raises(ResidentPairRetirementHold, match="another request"):
        regrade_resident_pair_retirement_checkpoint(
            reopened,
            factory=commissioned.factory,
            authority=replace(authority, target_profile_digest=_h("foreign-profile")),
            speed_plan=speed_plan,
            speed=speed,
            count_plan=count_plan,
            count=count,
        )


def _exact_lifecycle(harness, plan, capability, prefix, *, preserve_workload=False):
    resident = plan.resident_speed_plan
    assert resident is not None
    prepared = plan.prepared
    if not preserve_workload:
        warmup = resident.baseline.session_plan.warmup_count
        prompt_batches = resident.baseline.session_plan.prompt_batches[:warmup] + (
            resident.baseline.session_plan.prompt_batches[-1],
        ) * resident.policy.min_windows
        baseline_session = replace(
            resident.baseline.session_plan, prompt_batches=prompt_batches)
        candidate_session = replace(
            resident.candidate.session_plan, prompt_batches=prompt_batches)
        resident = replace(
            resident,
            baseline=replace(resident.baseline, session_plan=baseline_session),
            candidate=replace(resident.candidate, session_plan=candidate_session))
        prepared = replace(
            plan.prepared,
            baseline_session_plan=replace(
                plan.prepared.baseline_session_plan, prompt_batches=prompt_batches),
            candidates=(replace(
                plan.prepared.candidates[0], session_plan=candidate_session),))
    clock, activity = pair_fixtures._Clock(), pair_fixtures._Activity()

    class ExactFactory(pair_fixtures._Factory):
        def __call__(self, driver):
            return _PAIR_FACTORY_CALL(self, driver)

    pair = ResidentEvaluationPair(
        ExactFactory("c" * 32, resident.baseline.session_plan, (1.0,), clock, activity),
        ExactFactory("d" * 32, resident.baseline.session_plan, (0.8,), clock, activity),
        start_timeout_s=2.0,
        request_timeout_s=2.0,
        close_timeout_s=2.0,
        clock=clock)
    identities = pair.start()
    binding = ResidentPairRuntimeBinding(
        _h("exact-lifecycle-epoch"),
        tuple(
            ResidentPairLaneBinding(
                identity.lane_id,
                identity.session_id,
                _h(f"exact-stock:{identity.lane_id}"),
                resident.baseline_lane_digest if identity.lane_id == "A"
                else resident.candidate_lane_digest,
                _h(f"exact-allocation:{identity.lane_id}"),
                resident.baseline.executor_namespace_digest if identity.lane_id == "A"
                else resident.candidate.executor_namespace_digest,
            )
            for identity in identities))
    exact_plan = ResidentPairCrossoverPlan(
        harness.candidate.publication.content_hash, resident, binding, "A", "B")
    try:
        exact_speed = run_resident_pair_crossover(
            exact_plan, pair=pair, deadline=clock() + 120.0, clock=clock)
    finally:
        pair.close()
    count_result = replace(
        prefix.count_result,
        execution_plan_digest=_h("exact-count-plan"),
        pair_binding_digest=binding.digest,
        candidate_bundle_digest=exact_plan.candidate_bundle_digest,
        raw_execution_evidence_digest=_h("exact-count-evidence"))
    lanes = tuple(
        replace(
            lane,
            quiescence=replace(
                lane.quiescence,
                namespace_digest=binding.lookup(lane.lane_id).executor_namespace_digest,
                observed_monotonic_s=exact_speed.completed_monotonic_s + 1.0))
        for lane in prefix.retirement.lanes)
    retirement = replace(
        prefix.retirement,
        pair_binding=binding,
        speed_plan_digest=exact_plan.digest,
        speed_evidence_digest=exact_speed.digest,
        count_plan_digest=count_result.execution_plan_digest,
        count_evidence_digest=count_result.raw_execution_evidence_digest, lanes=lanes)
    count_checkpoint = replace(
        prefix.count_checkpoint,
        execution_plan_digest=count_result.execution_plan_digest,
        pair_binding_digest=count_result.pair_binding_digest,
        raw_execution_evidence_semantic_digest=count_result.raw_execution_evidence_digest)
    lifecycle = ResidentPairMarginalLifecycleEvidence(
        prepared, exact_plan, exact_speed, retirement, count_result,
        count_checkpoint, capability.stock_authority)
    return prepared, resident, lifecycle


@pytest.mark.parametrize("source_fixture", (None, FUSED), ids=("singleton", "atomic"))
def test_production_coordinator_reopens_without_new_pair_or_evaluator_work(
    tmp_path, monkeypatch, lifetimes, managed_executors, source_fixture
):
    coordinator, commissioned, harness, plan, capability, continuation = (
        _coordinator_case(
            tmp_path, monkeypatch, lifetimes, managed_executors,
            source_fixture=source_fixture))
    first = _run_prefix(
        coordinator, commissioned, harness, plan, capability, continuation)
    assert first.count_result.decision == "PASS"
    assert first.speed.escalated is False
    assert len(lifetimes.calls) == 2
    history = tuple(len(row.sessions[0].batch_rows) for row in lifetimes.factories)

    second = _run_prefix(
        coordinator, commissioned, harness, plan, capability, continuation)

    assert second == first
    assert len(lifetimes.calls) == 2
    assert tuple(len(row.sessions[0].batch_rows) for row in lifetimes.factories) == history

    if source_fixture is not None:
        return
    prepared, _resident, lifecycle = _exact_lifecycle(harness, plan, capability, first)
    assert lifecycle.candidates[0].candidate is prepared.candidates[0]
    assert lifecycle.role_names == ("B", "C")
    closure = lifecycle.closure
    assert closure is not None
    codec = _resident_closure_codec()
    assert codec.decode(codec.encode(closure)) == closure

    lanes = lifecycle.retirement.lanes
    premature = replace(
        lanes[0].quiescence,
        observed_monotonic_s=lifecycle.crossover.completed_monotonic_s - 0.01,
    )
    with pytest.raises(ResidentPairQualityLifecycleError, match="quiescence"):
        replace(
            lifecycle,
            retirement=replace(
                lifecycle.retirement,
                lanes=(replace(lanes[0], quiescence=premature), lanes[1]),
            ),
        )


def test_registered_count_fail_is_a_durable_terminal_without_audit_or_t(
    tmp_path, monkeypatch, lifetimes, managed_executors
):
    coordinator, commissioned, harness, plan, capability, continuation = (
        _coordinator_case(
            tmp_path, monkeypatch, lifetimes, managed_executors, count_drop=2
        )
    )
    prefix = _run_prefix(
        coordinator, commissioned, harness, plan, capability, continuation)
    assert prefix.count_result is not None
    assert prefix.count_result.decision == "FAIL"
    prepared, resident, lifecycle = _exact_lifecycle(
        harness, plan, capability, prefix, preserve_workload=True
    )
    assert prepared is plan.prepared
    assert resident is plan.resident_speed_plan
    assert lifecycle.closure is None
    assert lifecycle.count_result.decision == "FAIL"
    assert lifecycle.count_checkpoint is not None
    assert lifecycle.stock_authority == capability.stock_authority
    fail_continuation = _plan_scope(
        tmp_path / "count-fail-continuation",
        lifecycle.retirement.authenticated_request_digest,
        plan,
    )
    fail_continuation.record_resident_pair_speed(lifecycle.crossover)
    fail_continuation.record_resident_count_quality(lifecycle.count_checkpoint)
    fail_continuation.record_resident_pair_retirement(lifecycle.retirement)

    pair_calls = len(lifetimes.calls)
    kwargs = dict(
        executor=commissioned.plans[1].executor,
        entropy_provider=_never_called(
            "resident-count FAIL entered entropy selection"),
        hidden_judge=_forbidden_judge(
            plan.candidates[0].profile,
            "resident-count FAIL entered the hidden judge"),
        deadline=time.monotonic() + 30.0,
        continuation=fail_continuation,
        resident_pair_lifecycle=lifecycle,
    )
    reference = run_causal_qualification(plan, **kwargs)
    terminal = reopen_qualification_stage_exit(
        plan.evidence_root,
        reference,
        expected=plan,
        resident_pair_lifecycle=lifecycle,
    )
    assert reference.schema == STAGE_EXIT_SCHEMA_V2
    assert (terminal.stage, terminal.decision.value, terminal.reason) == (
        "resident_count",
        "FAIL",
        "resident_count_regression",
    )
    assert terminal.resident_count_result == lifecycle.count_result
    assert terminal.retirement_digest == lifecycle.retirement.digest
    assert len(lifetimes.calls) == pair_calls
    assert run_causal_qualification(plan, **kwargs) == reference
    assert len(lifetimes.calls) == pair_calls
    assert not (fail_continuation.directory / "audit_armed.json").exists()
    assert not (fail_continuation.directory / "t_armed.json").exists()


def test_v6_two_leg_pass_is_durable_importable_and_never_enters_pristine_t(
    tmp_path, monkeypatch, lifetimes, managed_executors
):
    coordinator, commissioned, harness, plan, capability, prefix_scope = (
        _coordinator_case(tmp_path, monkeypatch, lifetimes, managed_executors)
    )
    prefix = _run_prefix(
        coordinator, commissioned, harness, plan, capability, prefix_scope)
    assert prefix.count_result.decision == "PASS"
    _prepared, _resident, lifecycle = _exact_lifecycle(
        harness, plan, capability, prefix, preserve_workload=True
    )
    assert lifecycle.role_names == ("B", "C")
    assert lifecycle.crossover.decision is SpeedStageDecision.PASS
    assert lifecycle.closure is not None

    store = QualificationContinuationStore(tmp_path / "v6-pass-continuation")
    request_digest = lifecycle.retirement.authenticated_request_digest
    scope = store.scope(
        request_digest=request_digest,
        authority_digest=qualification_authority_digest(plan),
        source_digest=plan.prepared.source.digest,
    )
    scope.record_resident_pair_speed(lifecycle.crossover)
    scope.record_resident_count_quality(lifecycle.count_checkpoint)
    scope.record_resident_pair_retirement(lifecycle.retirement)

    executor = commissioned.plans[1].executor

    monkeypatch.setattr(
        runner_module,
        "_run_slot_audits",
        _never_called("v6 resident PASS entered the obsolete slot audit"),
    )
    profile = plan.candidates[0].profile
    authority = QualificationAuthorityManifest.seal(
        plan,
        reservations=(harness.candidate.reservation,),
        selection_secret_reference=_h("v6-pass-secret-reference"),
    )
    factory = QualificationPlanFactory(
        authority,
        lambda _reference: plan.selection_secret,
        lambda _secret: plan,
    )
    pair_calls = len(lifetimes.calls)
    kwargs = dict(
        executor=executor,
        entropy_provider=_never_called(
            "v6 resident PASS requested pristine-T entropy"),
        hidden_judge=_forbidden_judge(profile, "v6 resident PASS entered pristine T"),
        deadline=time.monotonic() + 30.0,
        continuation_store=store,
        request_digest=request_digest,
        prebuilt_plan=plan,
        resident_pair_lifecycle=lifecycle,
    )
    batch = run_qualification_intake(factory, **kwargs)
    outcome = batch.outcomes[0]
    assert batch.attempt_ref.schema == STAGE_EXIT_SCHEMA_V3
    assert (outcome.decision, outcome.reason, outcome.retryable) == (
        QualificationDecision.PASS,
        "qualified",
        False,
    )
    assert outcome.settlement_qualification is not None
    assert outcome.settlement_qualification.audit_policy is None
    assert (
        outcome.settlement_qualification.selection_evidence_digest
        == lifecycle.closure.count_result.digest
    )
    assert len(lifetimes.calls) == pair_calls
    assert not (scope.directory / "audit_armed.json").exists()
    assert not (scope.directory / "t_armed.json").exists()
    assert not (scope.directory / "quality.json").exists()
    assert run_qualification_intake(factory, **kwargs) == batch
    assert len(lifetimes.calls) == pair_calls

    closure = lifecycle.closure
    reference = profile.reference
    readiness = _readiness(plan, reference, "v6-pass")
    supports = (
        closure.count_checkpoint.raw_execution_evidence,
        closure.count_checkpoint.candidate_observation,
        closure.stock_authority.artifact,
    )
    product = _capture(batch, authority, plan, reference, readiness, supports)
    assert import_remote_qualification_evidence(
        product, tmp_path / "v6-pass-import"
    ) == product.evidence_inventory
    with pytest.raises(RemoteEvaluationDispatcherError, match="supporting evidence"):
        import_remote_qualification_evidence(
            _capture(batch, authority, plan, reference, readiness, supports[:-1]),
            tmp_path / "v6-pass-missing-stock",
        )


def test_count_checkpoint_without_retirement_holds_and_never_reopens_pair(
    tmp_path, monkeypatch, lifetimes, managed_executors
):
    coordinator, commissioned, harness, plan, capability, continuation = (
        _coordinator_case(tmp_path, monkeypatch, lifetimes, managed_executors)
    )
    monkeypatch.setattr(
        coordinator,
        "build_resident_pair_retirement_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            coordinator.ResidentPairRetirementHold("interrupted before retirement")
        ),
    )
    with pytest.raises(coordinator.B300ResidentQualificationHold):
        _run_prefix(coordinator, commissioned, harness, plan, capability, continuation)
    assert len(lifetimes.calls) == 2
    history = tuple(len(row.sessions[0].batch_rows) for row in lifetimes.factories)

    with pytest.raises(
        coordinator.B300ResidentQualificationHold,
        match="without exact pair retirement",
    ):
        _run_prefix(coordinator, commissioned, harness, plan, capability, continuation)
    assert len(lifetimes.calls) == 2
    assert tuple(len(row.sessions[0].batch_rows) for row in lifetimes.factories) == history

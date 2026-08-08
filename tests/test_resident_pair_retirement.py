from __future__ import annotations

import threading
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from cacheon.eval import native_artifact
from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.device_state import (
    DeviceStateReceipt,
    DeviceStateSample,
    GPUTelemetry,
)
from cacheon.eval.native_artifact import NativeArtifactPublication
from cacheon.eval.oci_backend import (
    CandidateFreeRuntimeIdentity,
    ResidentEngineExecutionEvidence,
)
from cacheon.eval.oci_prebuild import OCIPrebuildResult
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.oci_resident_session import (
    ResidentBatchEvidence,
    ResidentSessionEvidence,
)
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountLaneAdmission,
    ResidentCountQualityExecutionPlan,
    execute_candidate_count_quality,
)
from cacheon.eval.resident_pair_binding import (
    ResidentPairRuntimeBinding,
)
from cacheon.eval.resident_pair_crossover import run_resident_pair_crossover
from cacheon.eval.resident_pair_retirement import (
    ResidentPairRetirementError,
    ResidentPairRetirementEvidence,
    ResidentPairRetirementHold,
    build_resident_pair_retirement,
    regrade_resident_pair_retirement,
)
from tests.test_resident_count_quality_execution import (
    _fixture as _count_fixture,
    _h,
)
from tests.test_resident_pair_crossover import (
    _borderline_policy,
    _setup as _speed_setup,
)


def _walk(value):
    yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from _walk(getattr(value, field.name))
    elif type(value) in (tuple, list):
        for row in value:
            yield from _walk(row)
    elif type(value) is dict:
        for key, row in value.items():
            yield from _walk(key)
            yield from _walk(row)


@pytest.fixture
def cleanup_pairs():
    pairs = []
    yield pairs
    for pair in pairs:
        pair.close()


def _publication(root: Path, build_digest: str) -> NativeArtifactPublication:
    payload = native_artifact._identity_payload(build_digest, (), ())
    return NativeArtifactPublication(
        root / build_digest[:2] / build_digest,
        build_digest,
        native_artifact._publication_digest(payload),
        (),
        (),
    )


def _sample(at: float, physical_ids: tuple[int, ...]) -> DeviceStateSample:
    telemetry = tuple(
        GPUTelemetry(
            physical_id,
            f"GPU-{physical_id:08x}",
            "P0",
            31,
            0,
            0,
            1200,
            1600,
            100_000,
        )
        for physical_id in physical_ids
    )
    return DeviceStateSample(at, telemetry, (), True, "idle")


def _device_receipts(
    *,
    profile: str,
    lane_id: str,
    physical_ids: tuple[int, ...],
    ready: float,
    completed: float,
) -> tuple[DeviceStateReceipt, DeviceStateReceipt]:
    launch_id = f"launch-{profile}-{lane_id.lower()}"
    configuration = _h(f"{profile}-{lane_id}-device-configuration")
    policy = _h(f"{profile}-{lane_id}-device-policy")
    pre_start, pre_complete = ready - 0.4, ready - 0.1
    post_start, post_complete = completed + 0.1, completed + 0.4
    return (
        DeviceStateReceipt(
            "cacheon.device-state-receipt.v1",
            1,
            launch_id,
            "pre",
            physical_ids,
            configuration,
            policy,
            pre_start,
            pre_complete,
            1,
            (_sample(pre_complete - 0.05, physical_ids),),
        ),
        DeviceStateReceipt(
            "cacheon.device-state-receipt.v1",
            2,
            launch_id,
            "post",
            physical_ids,
            configuration,
            policy,
            post_start,
            post_complete,
            1,
            (_sample(post_complete - 0.05, physical_ids),),
        ),
    )


def _engine_execution(
    *,
    profile: str,
    lane_binding,
    session,
    artifact_root: Path,
    physical_ids: tuple[int, ...],
    completed: float,
) -> ResidentEngineExecutionEvidence:
    template_preflight = session.template.expected_preflight
    preflight = replace(
        template_preflight,
        launch_digest=lane_binding.stock_launch_digest,
        runtime_digest=_h(f"{profile}-{lane_binding.lane_id}-runtime"),
    )
    ready = 0.5
    session_evidence = ResidentSessionEvidence(
        lane_binding.session_id,
        lane_binding.stock_launch_digest,
        preflight,
        ready,
        tuple(session.batch_rows),
        tuple(session.swap_receipts),
        completed,
    )
    build_digest = _h(f"{profile}-{lane_binding.lane_id}-native-build")
    publication = _publication(artifact_root, build_digest)
    prebuild = OCIPrebuildResult(
        lane_binding.stock_launch_digest,
        build_digest,
        publication,
        1.0,
        _h(f"{profile}-{lane_binding.lane_id}-build-argv"),
    )
    return ResidentEngineExecutionEvidence(
        "cacheon.oci-resident-queue-execution.v1",
        lane_binding.stock_launch_digest,
        CandidateFreeRuntimeIdentity(
            preflight.runtime_digest,
            _h(f"{profile}-{lane_binding.lane_id}-base-engine"),
            _h(f"{profile}-{lane_binding.lane_id}-validator-overlay"),
        ),
        _h(f"{profile}-{lane_binding.lane_id}-runtime-preflight-receipt"),
        _h(f"{profile}-{lane_binding.lane_id}-arena-model-receipt"),
        _h(f"{profile}-{lane_binding.lane_id}-runtime-policy"),
        prebuild,
        publication.publication_digest,
        _h(f"{profile}-{lane_binding.lane_id}-runtime-argv"),
        (f"lease-{profile}-{lane_binding.lane_id.lower()}",),
        _device_receipts(
            profile=profile,
            lane_id=lane_binding.lane_id,
            physical_ids=physical_ids,
            ready=ready,
            completed=completed,
        ),
        session_evidence,
    )


def _count_plan(binding, candidate_digest: str, profile: str):
    base, judge, spare_pair, factory_a, _ = _count_fixture(
        4, barrier=False, profile=profile
    )
    spare_pair.close()
    admission = ResidentCountLaneAdmission(
        base.admission.lane_a_prompt_count,
        base.admission.lane_b_prompt_count,
        base.admission.engine_max_running_requests,
        binding.lanes[0].allocation_digest,
        binding.lanes[1].allocation_digest,
    )
    envelope = replace(
        base.envelope, admission_policy_digest=admission.digest
    )
    plan = ResidentCountQualityExecutionPlan(
        candidate_digest,
        envelope,
        base.prompt_batches,
        base.selected_ordinals,
        base.batch_shape,
        admission,
        binding,
    )
    return plan, judge, factory_a.outputs


def _install_count_reads(
    *,
    pair,
    sessions,
    count_plan,
    outputs,
    clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b_completed = threading.Event()
    original_complete = pair._complete_result  # type: ignore[attr-defined]

    def observe_complete(work, result):
        original_complete(work, result)
        if result.request_slice.lane_id == "B":
            b_completed.set()

    monkeypatch.setattr(pair, "_complete_result", observe_complete)

    for lane_id, session in zip(("A", "B"), sessions, strict=True):
        def execute(self, prompts, *, shape, canary=False, _lane=lane_id):
            assert shape == count_plan.batch_shape and not canary
            prompts = tuple(prompts)
            started, completed = clock.span(0.02)
            evidence = BatchEvidence(
                tuple(
                    PromptEvidence(
                        outputs[prompt], tuple(() for _ in outputs[prompt])
                    )
                    for prompt in prompts
                )
            )
            index = len(self.batch_rows)
            row = ResidentBatchEvidence(
                index,
                _h(f"count-{_lane}-request-{index}")[:32],
                _h(f"count-{_lane}-nonce-{index}")[:32],
                self.active_generation,
                self.active_slots,
                False,
                started,
                completed,
                evidence.observed_tokens,
                evidence,
            )
            self.batch_rows.append(row)
            if _lane == "A":
                assert b_completed.wait(2.0)
            return row

        session.execute_batch_with_shape = MethodType(execute, session)


def _case(
    tmp_path: Path,
    cleanup_pairs,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str,
    escalated: bool,
    baseline_lane: str,
):
    speed_kwargs = (
        {
            "baseline": (1.0, 1.0, 1.0),
            "candidate": (0.995, 0.98),
            "policy": _borderline_policy(),
        }
        if escalated
        else {"baseline": (1.0, 1.0), "candidate": (0.75,)}
    )
    base_speed, pair, clock, _, factory_a, factory_b = _speed_setup(
        tmp_path / "speed",
        cleanup_pairs,
        baseline_pair_lane=baseline_lane,
        **speed_kwargs,
    )
    binding = ResidentPairRuntimeBinding(
        _h(f"{profile}-service-epoch"),
        tuple(
            replace(
                row,
                stock_launch_digest=_h(f"{profile}-{row.lane_id}-stock-launch"),
                allocation_digest=_h(f"{profile}-{row.lane_id}-allocation"),
                executor_namespace_digest=_h(
                    f"{profile}-{row.lane_id}-executor-namespace"
                ),
            )
            for row in base_speed.pair_binding.lanes
        ),
    )
    candidate_digest = _h(f"{profile}-candidate")
    speed_plan = replace(
        base_speed,
        candidate_bundle_digest=candidate_digest,
        pair_binding=binding,
    )
    speed_evidence = run_resident_pair_crossover(
        speed_plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    count_plan, judge, outputs = _count_plan(
        binding, candidate_digest, profile
    )
    sessions = (factory_a.sessions[0], factory_b.sessions[0])
    _install_count_reads(
        pair=pair,
        sessions=sessions,
        count_plan=count_plan,
        outputs=outputs,
        clock=clock,
        monkeypatch=monkeypatch,
    )
    count_result = execute_candidate_count_quality(
        count_plan, pair=pair, judge=judge, deadline=10**10
    )
    assert tuple(
        row.request_slice.lane_id for row in pair.request_history[-2:]
    ) == ("B", "A")

    executions = {}
    for lane_binding, session, physical_ids in zip(
        binding.lanes, sessions, ((0, 1, 2, 3), (4, 5, 6, 7)), strict=True
    ):
        def finish(
            self,
            *,
            allow_empty=False,
            _binding=lane_binding,
            _physical=physical_ids,
        ):
            assert allow_empty
            self.finish_calls += 1
            self.closed = True
            tail = [
                row.response_completed_at for row in self.batch_rows
            ] + [row.completed_at for row in self.swap_receipts]
            completed = max(tail) + 0.1
            execution = _engine_execution(
                profile=profile,
                lane_binding=_binding,
                session=self,
                artifact_root=tmp_path / "native",
                physical_ids=_physical,
                completed=completed,
            )
            executions[_binding.lane_id] = execution
            return execution

        session.finish = MethodType(finish, session)

    retirement = pair.close()
    assert retirement is not None
    quiescence = {
        lane_id: OCIQuiescenceReceipt(
            "cacheon.oci-quiescence.v1",
            f"executor-{profile}-{lane_id.lower()}",
            _h(f"{profile}-{lane_id}-manager")[:32],
            binding.lookup(lane_id).executor_namespace_digest,
            1,
            float(executions[lane_id].device_receipts[1].completed_monotonic_s + 0.2),
            (),
            (),
            (),
        )
        for lane_id in ("A", "B")
    }
    closures = (
        (executions["B"], quiescence["B"]),
        (executions["A"], quiescence["A"]),
    )
    inputs = {
        "binding": binding,
        "speed_plan": speed_plan,
        "speed_evidence": speed_evidence,
        "count_plan": count_plan,
        "count_evidence": count_result.evidence,
        "count_observation": count_result.observation,
        "retirement": retirement,
        "lane_closures": closures,
    }
    product = build_resident_pair_retirement(**inputs)
    return SimpleNamespace(
        product=product,
        inputs=inputs,
        executions=executions,
        quiescence=quiescence,
        count_result=count_result,
        speed_evidence=speed_evidence,
    )


@pytest.mark.parametrize(
    ("profile", "escalated", "baseline_lane", "speed_reads"),
    (
        ("profile-one", False, "A", 3),
        ("profile-two", True, "B", 5),
    ),
)
def test_two_profiles_close_b_a_and_round_trip_path_free(
    tmp_path,
    cleanup_pairs,
    monkeypatch,
    profile,
    escalated,
    baseline_lane,
    speed_reads,
):
    case = _case(
        tmp_path,
        cleanup_pairs,
        monkeypatch,
        profile=profile,
        escalated=escalated,
        baseline_lane=baseline_lane,
    )
    product = case.product
    assert len(case.speed_evidence.request_slices) == speed_reads
    assert product.regrade(**case.inputs) is product
    assert regrade_resident_pair_retirement(product, **case.inputs) is product
    assert product.session_ids == tuple(
        row.session_id for row in case.inputs["binding"].identities
    )
    assert product.request_history_request_ids[-2:] == tuple(
        row.request_slice.request_id
        for row in case.inputs["retirement"].request_history[-2:]
    )
    assert product.canonical_request_ids[-2:] == tuple(
        row.request_id for row in case.count_result.evidence.request_slices
    )
    assert product.request_history_request_ids[-2:] == (
        product.canonical_request_ids[-1],
        product.canonical_request_ids[-2],
    )
    assert product.retirement_cutoff_monotonic_s == max(
        row.observed_monotonic_s for row in case.quiescence.values()
    )
    assert product.lane_a.close_count == product.lane_b.close_count == 1
    assert all(
        isinstance(row.prebuild.publication.root, Path)
        for row in case.executions.values()
    )
    assert not any(isinstance(value, Path) for value in _walk(product))

    codec = ContinuationCodec((ResidentPairRetirementEvidence,))
    encoded = codec.encode(product)
    reopened = codec.decode(encoded)
    assert reopened == product
    assert reopened.digest == product.digest
    assert reopened.regrade(**case.inputs) is reopened
    lowered = repr(encoded).lower()
    for forbidden in ("/users/", "/root/", "pathlib", "publication_root"):
        assert forbidden not in lowered


def test_regrade_rejects_foreign_missing_extra_and_mutated_authority(
    tmp_path, cleanup_pairs, monkeypatch
):
    case = _case(
        tmp_path,
        cleanup_pairs,
        monkeypatch,
        profile="tamper-profile",
        escalated=False,
        baseline_lane="A",
    )
    product, inputs = case.product, case.inputs
    changed = replace(product, count_observation_digest=_h("foreign-observation"))
    with pytest.raises(ResidentPairRetirementHold, match="projection differs"):
        changed.regrade(**inputs)

    for field in (
        "service_epoch_digest",
        "stock_launch_digest",
        "allocation_digest",
        "executor_namespace_digest",
    ):
        if field == "service_epoch_digest":
            foreign_binding = replace(
                inputs["binding"], **{field: _h(f"foreign-{field}")}
            )
        else:
            lane = replace(
                inputs["binding"].lanes[0],
                **{field: _h(f"foreign-{field}")},
            )
            foreign_binding = replace(
                inputs["binding"],
                lanes=(lane, inputs["binding"].lanes[1]),
            )
        with pytest.raises(ResidentPairRetirementHold, match="another pair"):
            product.regrade(**{**inputs, "binding": foreign_binding})

    foreign_speed = replace(
        inputs["speed_evidence"], plan_digest=_h("foreign-speed-evidence")
    )
    with pytest.raises(ResidentPairRetirementHold, match="speed evidence"):
        product.regrade(**{**inputs, "speed_evidence": foreign_speed})

    with pytest.raises(ResidentPairRetirementError, match="exactly two"):
        build_resident_pair_retirement(
            **{**inputs, "lane_closures": inputs["lane_closures"][:1]}
        )
    duplicate_closures = (
        inputs["lane_closures"][0],
        inputs["lane_closures"][0],
    )
    with pytest.raises(ResidentPairRetirementHold, match="duplicates"):
        build_resident_pair_retirement(
            **{**inputs, "lane_closures": duplicate_closures}
        )

    history = inputs["retirement"].request_history
    for changed_history in (
        history[:-1],
        history + (history[-1],),
        (history[0], history[-1], *history[1:-1]),
        (*history[:-2], history[-2], history[-2]),
    ):
        changed_retirement = replace(
            inputs["retirement"], request_history=changed_history
        )
        with pytest.raises(ResidentPairRetirementHold):
            build_resident_pair_retirement(
                **{**inputs, "retirement": changed_retirement}
            )

    foreign_observation = replace(
        inputs["count_observation"],
        execution_evidence_digest=_h("foreign-count-execution"),
    )
    with pytest.raises(ResidentPairRetirementHold, match="foreign"):
        build_resident_pair_retirement(
            **{**inputs, "count_observation": foreign_observation}
        )


def test_unclosed_nonidle_foreign_quiescence_and_paths_are_hold(
    tmp_path, cleanup_pairs, monkeypatch
):
    case = _case(
        tmp_path,
        cleanup_pairs,
        monkeypatch,
        profile="closure-profile",
        escalated=False,
        baseline_lane="A",
    )
    inputs = case.inputs
    lane_a = inputs["retirement"].lane_a
    unclosed = replace(lane_a, lifetime_evidence=("not-closed",))
    with pytest.raises(ResidentPairRetirementHold, match="lifetime product"):
        build_resident_pair_retirement(
            **{
                **inputs,
                "retirement": replace(inputs["retirement"], lane_a=unclosed),
            }
        )

    execution_a = case.executions["A"]
    pre, post = execution_a.device_receipts
    idle = post.samples[-1]
    busy_post = replace(
        post, samples=(replace(idle, idle=False, idle_reason="busy"),)
    )
    busy_execution = replace(
        execution_a, device_receipts=(pre, busy_post)
    )
    busy_retirement = replace(
        inputs["retirement"],
        lane_a=replace(
            inputs["retirement"].lane_a,
            lifetime_evidence=busy_execution,
        ),
    )
    busy_closures = (
        (busy_execution, case.quiescence["A"]),
        (case.executions["B"], case.quiescence["B"]),
    )
    with pytest.raises(ResidentPairRetirementHold, match="idle tail"):
        build_resident_pair_retirement(
            **{
                **inputs,
                "retirement": busy_retirement,
                "lane_closures": busy_closures,
            }
        )

    foreign_quiescence = replace(
        case.quiescence["A"], namespace_digest=_h("foreign-namespace")
    )
    with pytest.raises(ResidentPairRetirementHold, match="foreign"):
        build_resident_pair_retirement(
            **{
                **inputs,
                "lane_closures": (
                    (case.executions["A"], foreign_quiescence),
                    (case.executions["B"], case.quiescence["B"]),
                ),
            }
        )

    nonempty_quiescence = replace(case.quiescence["A"])
    object.__setattr__(
        nonempty_quiescence, "resource_entries", ("still-live",)
    )
    with pytest.raises(ResidentPairRetirementError, match="inconsistent"):
        build_resident_pair_retirement(
            **{
                **inputs,
                "lane_closures": (
                    (case.executions["A"], nonempty_quiescence),
                    (case.executions["B"], case.quiescence["B"]),
                ),
            }
        )

    path_sample = replace(post.samples[-1], idle_reason=tmp_path)
    path_execution = replace(
        execution_a,
        device_receipts=(pre, replace(post, samples=(path_sample,))),
    )
    path_retirement = replace(
        inputs["retirement"],
        lane_a=replace(
            inputs["retirement"].lane_a,
            lifetime_evidence=path_execution,
        ),
    )
    with pytest.raises((ResidentPairRetirementError, ResidentPairRetirementHold)):
        build_resident_pair_retirement(
            **{
                **inputs,
                "retirement": path_retirement,
                "lane_closures": (
                    (path_execution, case.quiescence["A"]),
                    (case.executions["B"], case.quiescence["B"]),
                ),
            }
        )


def test_production_has_no_target_or_path_literal():
    import cacheon.eval.resident_pair_retirement as module

    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("arnorm", "all_reduce", "/users/shiv", "host_path"):
        assert forbidden not in source

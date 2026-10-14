"""Completed paired windows survive new engines, physical lanes and host clocks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cacheon.eval import goodput_runtime as runtime, replay_checkpoint as progress
from cacheon.eval import qualification_worker_checkpoint as checkpoint
from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.crossover_runtime import _expected_lane_digest
from cacheon.eval.oci_outer_session import BatchExecutionEvidence
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from cacheon.eval.qualification_runner import qualification_authority_digest
from cacheon.eval.service_capacity import LoadRead, TurnRecord
from cacheon.eval.speed_verdict import SpeedStageDecision
from tests.test_b300_registered_qualification import _harness
from tests.test_device_state import _policy
from tests.test_qualification_continuation import _typed_execution
from tests.test_qualification_recovery import _d


def _execution(value, arm, role, count, label, root, start):
    execution = _typed_execution(arm.launch, arm.binding, value.model_mount, arm.session_plan,
                                 label=label, artifact_root=root / label)
    replay = arm.session_plan.replay
    work = replay.slice.expected_work(replay.loads[0])
    (session_id, (turns, inner)), = work.items()
    assert inner == 0
    batches, reads = [], []
    warmup = arm.session_plan.warmup_count
    prompt = PromptEvidence((1, 2), ((), ()), 5)
    for index in range(warmup + turns * count):
        begin = start + 2 * index + 1
        batches.append(BatchExecutionEvidence(index, _d(label + f"request{index}")[:32],
            _d(label + f"nonce{index}")[:32], begin, begin+1, 2, BatchEvidence((prompt,)),
            prompt_latencies=((0.1, 1.0),)))
    for window in range(count):
        rows = batches[warmup + turns*window:warmup + turns*(window+1)]
        records = tuple(TurnRecord(session_id, "main", i, round(row.request_started_at*1e9),
            round(row.request_started_at*1e9), round((row.request_started_at+0.1)*1e9),
            round(row.response_completed_at*1e9), 5, 2, "ok") for i, row in enumerate(rows))
        reads.append(LoadRead(role, window+1, _expected_lane_digest(arm), replay.loads[0], records))
    receipts = tuple(replace(row, selected_physical_gpu_ids=tuple(map(int, arm.binding.physical_hardware.physical_gpu_ids)),
        policy_sha256=arm.launch.hardware.device_policy_digest,
        configuration_sha256=arm.device_configuration_digest) for row in execution.device_receipts)
    receipts = (receipts[0], replace(receipts[1], event="final-warmup",
                started_monotonic_s=start+0.1, completed_monotonic_s=start+0.2),
                replace(receipts[2], started_monotonic_s=start+100, completed_monotonic_s=start+101))
    return replace(execution, schema="cacheon.oci-resident-engine-execution.v1",
        resource_policy_digest=arm.runtime_resource_policy_digest, device_receipts=receipts,
        session=replace(execution.session, ready_completed_at=start,
            batches=tuple(batches), replay_reads=tuple(reads), session_completed_at=start+99))


@pytest.mark.parametrize("kill_after", [1, 2, 3])
def test_resume_reuses_every_acknowledged_window_across_lanes_and_clocks(tmp_path, monkeypatch, kill_after):
    import tests.test_b300_registered_qualification as fixtures
    original_replay = fixtures.replay_plan
    monkeypatch.setattr(fixtures, "replay_plan", lambda root: original_replay(root, windows=4))
    harness = _harness(tmp_path / "plan")
    value = harness.factory.plan_builder(harness.cohort, b"portable-window-regression-private-seed")
    plan = replace(value.resident_speed_plan, policy=replace(value.resident_speed_plan.policy, version=18,
                                                           max_stage_seconds=600))
    value = replace(value, resident_speed_plan=plan)
    state = {"clock": 1000.0, "boot": "old-boot", "kill": kill_after}
    monkeypatch.setattr(checkpoint, "_clock_domain", lambda: state["boot"])
    scope = QualificationContinuationStore(tmp_path / "continuation").scope(
        request_digest=_d("request"), authority_digest=qualification_authority_digest(value),
        source_digest=value.prepared.source.digest)
    checkpoint.configure_continuation(value, scope, SimpleNamespace(device_policy=_policy()),
                                       SimpleNamespace(device_policy=_policy()))
    calls = []

    def fake_orientation(part, **kwargs):
        prior = kwargs["prior"]
        offset = 0 if prior is None else len(prior.incumbent)
        count = part.baseline.session_plan.replay.windows
        begin = state["clock"]
        calls.append((state["boot"], offset, count, part.baseline.binding.physical_hardware.physical_gpu_ids))
        for n in range(1, count+1):
            executions = tuple(_execution(value, arm, role, n, f'{state["boot"]}-{offset}-{role}',
                tmp_path / "raw", begin) for arm, role in ((part.baseline, "incumbent"), (part.candidate, "candidate")))
            replay = part.baseline.session_plan.replay
            expected = tuple((key, *work) for key, work in sorted(replay.slice.expected_work(replay.loads[0]).items()))
            reads = runtime.GoodputReadSet(
                (() if prior is None else prior.incumbent) + executions[0].session.replay_reads,
                (() if prior is None else prior.candidate) + executions[1].session.replay_reads, expected, 4,
                (() if prior is None else prior.orientations) + (kwargs["orientation"],)*n)
            grade = reads.grade(plan.policy.goodput)
            if n < count and grade.decision is SpeedStageDecision.NO_DECISION:
                kwargs["on_window"](executions, begin, part, n)
                if offset+n == state["kill"]:
                    raise RuntimeError("worker lost")
        state["clock"] += 150
        return dict(reads=reads, grade=grade, executions=executions, started=begin, quality=((), (), None))

    monkeypatch.setattr(runtime, "_run_orientation", fake_orientation)
    original_record = scope._record
    def save(stage, payload):
        original_record(stage, payload)
        if stage == f'speed_window_{state["kill"]}':
            raise RuntimeError("worker lost")
    monkeypatch.setattr(scope, "_record", save)
    def executor(arm):
        return SimpleNamespace(device_policy=_policy(), prove_quiescent=lambda: OCIQuiescenceReceipt(
            "cacheon.oci-quiescence.v1", "lane", "a"*32, arm.executor_namespace_digest,
            1, state["clock"]+10, (), (), ()))
    def run(current, continuation):
        return progress.run_pair(plan, baseline_executor=executor(current.baseline), candidate_executor=executor(current.candidate),
            model_mount=value.model_mount, deadline=10000, clock=lambda:state["clock"], quality_control=None,
            continuation=continuation, tokenizers=(None, None))
    with pytest.raises(RuntimeError, match="worker lost"):
        run(plan, scope)
    records = tuple(scope._load(f"speed_window_{i}") for i in range(1, kill_after+1))
    assert all(records)
    checkpoint.configure_continuation(value, scope, SimpleNamespace(device_policy=_policy()),
                                       SimpleNamespace(device_policy=_policy()))
    assert scope.speed_windows == records
    import json
    pointer = json.loads((scope.checkpoint_root / (scope.request_digest + ".json")).read_bytes())
    snapshot = checkpoint.read_snapshot(scope.checkpoint_root / (pointer["sha256"] + ".json.gz"), pointer["sha256"])
    assert set(snapshot["stages"]) == {f"speed_window_{i}" for i in range(1, kill_after+1)}
    saved_bytes = tuple((scope.directory / f"speed_window_{i}.json").read_bytes() for i in range(1, kill_after+1))
    moved = lambda arm, gpu: replace(arm, binding=replace(arm.binding,
        physical_hardware=replace(arm.binding.physical_hardware, physical_gpu_ids=(str(gpu),))))
    replacement = replace(plan, baseline=moved(plan.baseline, 10), candidate=moved(plan.candidate, 11))
    resumed = QualificationContinuationStore(tmp_path / "replacement").scope(
        request_digest=_d("new-request"), authority_digest=scope.authority_digest, source_digest=scope.source_digest)
    resumed.speed_windows = records
    resumed.replacement_execution = SimpleNamespace(resident_speed_plan=replacement, model_mount=value.model_mount)
    state.update(clock=10.0, boot="new-boot", kill=0)
    result = run(replacement, resumed)
    assert result.decision is SpeedStageDecision.FAIL
    assert result.replay_windows[:kill_after] == records
    assert result.regrade(plan) == result.final_verdict
    assert [call[1] for call in calls if call[0] == "new-boot"][0] == kill_after
    assert sum(call[2] for call in calls if call[0] == "new-boot") == 4-kill_after
    assert all(call[3] in (("10",), ("11",)) for call in calls if call[0] == "new-boot")
    assert saved_bytes == tuple((scope.directory / f"speed_window_{i}.json").read_bytes() for i in range(1, kill_after+1))
    codec = ContinuationCodec((type(result),))
    assert codec.decode(codec.encode(result)) == result
    with pytest.raises(Exception, match="repeated measured requests"):
        rows = list(result.replay_windows)
        import json
        from cacheon.stack_identity import canonical_json_bytes
        duplicate = json.loads(rows[0]); duplicate["ordinal"] = 2
        rows[1] = canonical_json_bytes(duplicate).decode()
        replace(result, replay_windows=tuple(rows)).regrade(plan)

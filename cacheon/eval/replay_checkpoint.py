"""Resume paired replay windows with their original host, engine and clock evidence.

This extends the existing speed continuation down to each completed paired
window. An open-engine snapshot makes no claim about teardown. The final pair
still closes through the ordinary executor before qualification can finish.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.oci_backend import EngineExecutionEvidence
from cacheon.eval.oci_outer_session import SessionExecutionEvidence
from cacheon.eval.scoring import CrossoverRuntimeError, marginal_workload_digest
from cacheon.eval.speed_verdict import SpeedStageDecision
from cacheon.stack_identity import canonical_json_bytes


def _drive_opened(controller, driver, *snapshot_args):
    """Keep opened-controller startup and failure inside the ordinary teardown path."""
    from cacheon.eval.oci_backend import OCIBackendError

    try:
        controller.start()
        bind_snapshot(controller, *snapshot_args)
        session = driver(controller)
        if not controller.closed:
            raise OCIBackendError("opened session driver returned without closing the engine")
        if type(session) is not SessionExecutionEvidence or session.session_id != controller.session_id:
            raise OCIBackendError("opened session driver returned another session receipt")
        return session
    finally:
        controller.abort()


def bind_snapshot(controller, conditioner, launch, identity, preflight, mount,
                  resource_policy, recovered, metadata):
    """Expose a host-owned prefix without closing the running engine."""
    prebuild, publication, argv, pre = metadata

    def snapshot():
        if controller.closed or not controller.replay_reads or controller.transport.has_pending_output():
            raise CrossoverRuntimeError("replay checkpoint is not at a completed window boundary")
        now = float(controller.clock())
        if not math.isfinite(now) or now < controller.last_host_time:
            raise CrossoverRuntimeError("replay checkpoint clock moved backwards")
        session = SessionExecutionEvidence(
            controller.session_id, controller.plan.launch_digest, controller.preflight,
            controller.ready_completed_at, tuple(controller.batch_rows), controller.plan.warmup_count,
            controller.plan.conditioning_count, controller.conditioning_started_at,
            controller.first_timed_completed_at,
            sum(row.token_numerator for row in controller.batch_rows[
                controller.conditioning_start_index:controller.plan.warmup_count + 1]),
            0.0, replay_reads=tuple(controller.replay_reads), checkpointed_at=now)
        return EngineExecutionEvidence("cacheon.oci-resident-engine-checkpoint.v1", launch.digest,
            identity, preflight.sha256, mount.digest, resource_policy, prebuild, publication, argv,
            recovered, (pre, conditioner.require_complete(), None), session)

    controller.checkpoint_execution = snapshot


def _origin(plan):
    from cacheon.eval.qualification_worker_checkpoint import _binding, _clock_domain

    return {"boot": _clock_domain(), "speed_policy": plan.policy.to_dict(), "arms": [
        {**_binding(arm.launch, arm.binding), "namespace": arm.executor_namespace_digest,
         "runtime_policy": arm.runtime_resource_policy_digest, "configuration": arm.device_configuration_digest}
        for arm in (plan.baseline, plan.candidate)]}


def _prefix(execution, count, turns):
    session = execution.session
    end = session.warmup_count + count * turns
    return replace(execution, schema="cacheon.oci-resident-engine-checkpoint.v1",
        device_receipts=(*execution.device_receipts[:2], None), session=replace(session,
            batches=session.batches[:end], replay_reads=session.replay_reads[:count],
            session_completed_at=0.0,
            checkpointed_at=session.checkpointed_at or session.session_completed_at))


def _record(plan, executions, ordinal, orientation, windows, origin, started, count):
    codec = ContinuationCodec((EngineExecutionEvidence,))
    replay = plan.baseline.session_plan.replay
    return canonical_json_bytes({
        "ordinal": ordinal, "orientation": orientation, "windows": windows, "origin": origin,
        "started": format(started, ".17g"), "plan": plan.digest,
        "executions": [codec.encode(_prefix(row, count, replay.slice.turns(replay.loads[0])))
                       for row in executions],
    }).decode()


def _read_record(raw, plan, ordinal):
    from cacheon.eval.goodput_runtime import _orientation_plan, _validate_replay_sessions
    from cacheon.eval.qualification_worker_checkpoint import restore_speed_plan

    row = json.loads(raw)
    if (canonical_json_bytes(row).decode() != raw
        or set(row) != {"ordinal", "orientation", "windows", "origin", "started", "plan", "executions"}
        or row["ordinal"] != ordinal
        or row["orientation"] != int(ordinal > plan.baseline.session_plan.replay.windows // 2)
        or type(row["windows"]) is not int or not 1 <= row["windows"] <= 5
        or not isinstance(row["origin"].get("boot"), str) or not row["origin"]["boot"]):
        raise CrossoverRuntimeError("replay checkpoint changed its paired window order")
    actual = restore_speed_plan(plan, row["origin"])
    old, current = actual.policy.to_dict(), plan.policy.to_dict()
    for key in ("calibration_digest", "calibration_context_digest"):
        old.pop(key); current.pop(key)
    if old != current:
        raise CrossoverRuntimeError("replay checkpoint changed its grading policy")
    if actual.digest != row["plan"]:
        raise CrossoverRuntimeError("replay checkpoint changed its execution plan")
    part = _orientation_plan(actual, row["windows"], swapped=bool(row["orientation"]))
    codec = ContinuationCodec((EngineExecutionEvidence,))
    executions = tuple(codec.decode(item) for item in row["executions"])
    if len(executions) != 2 or any(type(item) is not EngineExecutionEvidence for item in executions):
        raise CrossoverRuntimeError("replay checkpoint lacks its paired engine evidence")
    started = float(row["started"])
    for execution in executions:
        session = execution.session
        pre, active, post = execution.device_receipts
        if (not math.isfinite(started) or format(started, ".17g") != row["started"]
            or session.checkpointed_at is None or session.session_completed_at != 0
            or not started <= session.ready_completed_at <= session.checkpointed_at
            or session.checkpointed_at - started > plan.policy.max_stage_seconds
            or post is not None or pre.launch_id != active.launch_id or pre.sequence >= active.sequence
            or pre.phase != "pre" or active.event != "final-warmup"
            or not pre.completed_monotonic_s <= active.started_monotonic_s <= active.completed_monotonic_s
            <= session.checkpointed_at):
            raise CrossoverRuntimeError("replay checkpoint has invalid host/device chronology")
    _validate_replay_sessions([(exe.session.replay_reads, exe, arm, ()) for exe, arm in zip(
        executions, (part.baseline, part.candidate), strict=True)], plan.baseline.session_plan.replay,
        checkpoint=True)
    if len(executions[0].session.replay_reads) != len(executions[1].session.replay_reads):
        raise CrossoverRuntimeError("replay checkpoint has an unfinished paired window")
    return row, actual, part, executions


def _reads(records, plan):
    from cacheon.eval.goodput_runtime import GoodputReadSet

    if not records:
        return None
    pairs = [_read_record(raw, plan, i)[3] for i, raw in enumerate(records, 1)]
    replay = plan.baseline.session_plan.replay
    seen = set()
    count = replay.slice.turns(replay.loads[0])
    for pair in pairs:
        for execution in pair:
            for batch in execution.session.batches[-count:]:
                key = (execution.session.session_id, batch.request_id, batch.nonce)
                if key in seen:
                    raise CrossoverRuntimeError("portable replay repeated measured requests")
                seen.add(key)
    expected = tuple(sorted((root, main, inner) for root, (main, inner)
        in replay.slice.expected_work(replay.loads[0]).items()))
    return GoodputReadSet(tuple(pair[0].session.replay_reads[-1] for pair in pairs),
        tuple(pair[1].session.replay_reads[-1] for pair in pairs), expected, replay.windows,
        tuple(int(i >= replay.windows // 2) for i in range(len(records))))


def retained_seconds(records, *, before=None, boot=None):
    """Charge acknowledged execution once per engine lifetime, excluding downtime."""
    spans = {}
    codec = ContinuationCodec((EngineExecutionEvidence,))
    for raw in records:
        row = json.loads(raw)
        runs = tuple(codec.decode(value) for value in row["executions"])
        end = max(run.session.checkpointed_at for run in runs)
        if before is not None and row["origin"]["boot"] == boot and end >= before:
            continue
        key = (row["origin"]["boot"], *(run.session.session_id for run in runs))
        spans[key] = max(spans.get(key, 0.0), end-float(row["started"]))
    return math.fsum(spans.values())


def run_pair(plan, *, baseline_executor, candidate_executor, model_mount, deadline, clock,
             quality_control, continuation, tokenizers):
    """Run only unacknowledged windows through the ordinary paired replay driver."""
    from cacheon.eval.crossover_runtime import ResidentCrossoverEvidence, _expected_lane_digest
    from cacheon.eval.goodput_runtime import _orientation_plan, _run_orientation

    started = float(clock())
    records = list(getattr(continuation, "speed_windows", ()))
    prior = _reads(records, plan)
    deadline = min(deadline, started + plan.policy.max_stage_seconds - retained_seconds(records))
    if deadline <= started:
        raise CrossoverRuntimeError("retained replay exhausted its execution budget")
    if prior is not None and prior.grade(plan.policy.goodput).decision is not SpeedStageDecision.NO_DECISION:
        raise CrossoverRuntimeError("terminal replay prefix requires its complete speed-stage record")
    replacement = getattr(continuation, "replacement_execution", None)
    actual = plan if replacement is None else replacement.resident_speed_plan
    if replacement is not None:
        model_mount = replacement.model_mount
    origin = _origin(actual)
    windows = plan.baseline.session_plan.replay.windows
    for direction, limit in ((0, windows // 2), (1, windows)):
        if len(records) >= limit:
            continue
        offset = len(records)
        part = _orientation_plan(actual, limit-offset, swapped=bool(direction))
        executors = ((baseline_executor, candidate_executor) if direction == 0 else
                     (candidate_executor, baseline_executor))

        def save(executions, begin, segment, count):
            ordinal = offset + count
            raw = _record(actual, executions, ordinal, direction,
                          segment.baseline.session_plan.replay.windows, origin, begin, count)
            _read_record(raw, plan, ordinal)
            if ordinal != len(records)+1:
                raise CrossoverRuntimeError("paired replay repeated an acknowledged window")
            records.append(raw)
            if continuation is not None:
                continuation._record(f"speed_window_{ordinal}", raw)

        last = _run_orientation(part, baseline_executor=executors[0], candidate_executor=executors[1],
            model_mount=model_mount, deadline=deadline, clock=clock, quality_control=quality_control,
            tokenizers=tokenizers, full_plan=plan, prior=prior, orientation=direction, on_window=save)
        count = len(last["executions"][0].session.replay_reads)
        if last["grade"].decision is SpeedStageDecision.NO_DECISION:
            save(tuple(last["executions"]), last["started"], part, count)
            prior = last["reads"]
            continue
        # The terminal window is committed with its complete stage, after quality
        # controls and teardown. A death there repeats only that unacknowledged window.
        records.append(_record(actual, tuple(last["executions"]), offset+count, direction,
            part.baseline.session_plan.replay.windows, origin, last["started"], count))
        break
    grade = last["grade"]
    pairs, controls, entropy = last["quality"]
    result = ResidentCrossoverEvidence(plan.digest, plan.selected_delta_digest, plan.policy,
        marginal_workload_digest(plan.baseline.session_plan), _expected_lane_digest(part.baseline),
        _expected_lane_digest(part.candidate), *last["executions"],
        executors[0].prove_quiescent(), executors[1].prove_quiescent(), (), grade.verdict, grade.verdict,
        False, grade.decision, "clear_"+grade.decision.value.lower(), started, float(clock()),
        goodput=last["reads"], prompt_pairs=pairs, reference_inputs=controls, quality_entropy=entropy,
        replay_windows=tuple(records))
    result.regrade(plan)
    return result


def regrade(evidence, plan):
    """Regrade each saved window in its actual clock domain and execution binding."""
    from cacheon.eval.crossover_runtime import _expected_lane_digest
    from cacheon.eval.goodput_runtime import _validate_replay_sessions

    if (evidence.plan_digest != plan.digest or evidence.policy != plan.policy
        or evidence.selected_delta_digest != plan.selected_delta_digest
        or evidence.workload_digest != marginal_workload_digest(plan.baseline.session_plan)
        or len(evidence.replay_windows) != len(evidence.goodput.incumbent)):
        raise CrossoverRuntimeError("portable speed evidence names another sealed plan")
    rebuilt = _reads(evidence.replay_windows, plan)
    boot = json.loads(evidence.replay_windows[-1])["origin"]["boot"]
    elapsed = evidence.completed_monotonic_s-evidence.started_monotonic_s + retained_seconds(
        evidence.replay_windows, before=evidence.started_monotonic_s, boot=boot)
    if elapsed > plan.policy.max_stage_seconds:
        raise CrossoverRuntimeError("portable replay exceeded its combined execution budget")
    if rebuilt != evidence.goodput:
        raise CrossoverRuntimeError("portable speed reads differ from their host checkpoints")
    for index in range(1, len(evidence.replay_windows)):
        prefix = replace(rebuilt, incumbent=rebuilt.incumbent[:index], candidate=rebuilt.candidate[:index],
                         orientations=rebuilt.orientations[:index])
        if prefix.grade(plan.policy.goodput).decision is not SpeedStageDecision.NO_DECISION:
            raise CrossoverRuntimeError("portable replay ran past its sealed stopping rule")
    _, _, final, snapshots = _read_record(evidence.replay_windows[-1], plan, len(evidence.replay_windows))
    executions = (evidence.baseline_execution, evidence.candidate_execution)
    quiet = (evidence.baseline_quiescence, evidence.candidate_quiescence)
    for execution, snapshot, arm, receipt, lane in zip(executions, snapshots,
            (final.baseline, final.candidate), quiet,
            (evidence.baseline_lane_digest, evidence.candidate_lane_digest), strict=True):
        if (lane != _expected_lane_digest(arm) or receipt.namespace_digest != arm.executor_namespace_digest
            or (receipt.container_ids, receipt.lease_records, receipt.resource_entries) != ((), (), ())
            or execution.session.session_id != snapshot.session.session_id
            or execution.session.batches[:len(snapshot.session.batches)] != snapshot.session.batches
            or snapshot.session.checkpointed_at > execution.session.session_completed_at
            or execution.session.session_completed_at > receipt.observed_monotonic_s):
            raise CrossoverRuntimeError("portable final engine or quiescence differs from its windows")
    _validate_replay_sessions([(execution.session.replay_reads, execution, arm, controls)
        for execution, arm, controls in zip(executions, (final.baseline, final.candidate),
            (evidence.reference_inputs, ()), strict=True)], plan.baseline.session_plan.replay)
    grade = rebuilt.grade(plan.policy.goodput)
    if (grade.verdict != evidence.initial_verdict or grade.verdict != evidence.final_verdict
        or grade.decision is not evidence.decision):
        raise CrossoverRuntimeError("portable replay headline does not regrade")
    if grade.decision is SpeedStageDecision.PASS:
        from cacheon.eval.qualification_trajectories import validate_controls
        validate_controls(evidence, plan)
    return grade.verdict

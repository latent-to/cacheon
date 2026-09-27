"""Bind paired replay execution to the service scorer and existing qualification witness."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import math
from dataclasses import asdict, dataclass, replace

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.service_capacity import LoadRead, ServiceContract, ServiceVerdict, fixed_work_rate
from cacheon.eval.scoring import SpeedupVerdict
from cacheon.eval.speed_verdict import SpeedStageDecision


@dataclass(frozen=True)
class GoodputPolicy:
    """Frozen calibration inputs for a fixed-load comparison; no thresholds come from a candidate."""

    contract: ServiceContract
    required: float
    null_noise: float
    attainment_tolerance: float
    attainment_margin: float

    def __post_init__(self):
        values = (self.required, self.null_noise, self.attainment_tolerance, self.attainment_margin)
        if (type(self.contract) is not ServiceContract
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 1 < self.required < 2 or not 0 <= self.null_noise < 1
            or not 0 <= self.attainment_tolerance < 1 or not 0 <= self.attainment_margin < 1):
            raise ValueError("goodput calibration inputs are invalid")
        object.__setattr__(self, "contract", ServiceContract(**{
            name: float(value) for name, value in asdict(self.contract).items()
        }))
        for name in self.__dataclass_fields__:
            if name != "contract":
                object.__setattr__(self, name, float(getattr(self, name)))

    def to_dict(self):
        """Seal contract and calibration values as canonical decimal strings."""
        return {"contract": {key: format(value, ".17g") for key, value in asdict(self.contract).items()},
                **{key: format(getattr(self, key), ".17g") for key in self.__dataclass_fields__ if key != "contract"}}

    @classmethod
    def from_dict(cls, value):
        """Reopen only this exact policy type."""
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__) or type(value["contract"]) is not dict:
            raise ValueError("goodput policy fields differ")
        result = cls(ServiceContract(**{key: float(v) for key, v in value["contract"].items()}),
                     **{key: float(v) for key, v in value.items() if key != "contract"})
        if result.to_dict() != value:
            raise ValueError("goodput policy is not canonical")
        return result


@dataclass(frozen=True)
class GoodputReadSet:
    """The turn records replacing token-window rates in a replay speed witness."""

    incumbent: tuple[LoadRead, ...]
    candidate: tuple[LoadRead, ...]
    expected: tuple[tuple[str, int, int], ...]

    def __post_init__(self):
        if (not self.incumbent or len(self.incumbent) != len(self.candidate)
            or tuple(sorted(set(self.expected))) != self.expected
            or len({row[0] for row in self.expected}) != len(self.expected)):
            raise ValueError("goodput reads or fixed-work authority are incomplete")
        work = {root: (main, inner) for root, main, inner in self.expected}
        for baseline, candidate in zip(self.incumbent, self.candidate, strict=True):
            if (baseline.arm != "incumbent" or candidate.arm != "candidate"
                or (baseline.window, baseline.load) != (candidate.window, candidate.load)
                or baseline.lane == candidate.lane):
                raise ValueError("goodput reads are not a paired load window")
            fixed_work_rate(baseline, work)
            fixed_work_rate(candidate, work)

    def grade(self, policy: GoodputPolicy):
        """Call the data/scoring owner's single grade entrypoint; retain no second estimator."""
        from cacheon.eval import service_capacity
        from cacheon.eval.resident_schedule import ScheduleGrade

        result = service_capacity.grade(
            self.candidate, self.incumbent, policy.contract,
            {root: (main, inner) for root, main, inner in self.expected},
            required=policy.required, attainment_tolerance=policy.attainment_tolerance,
            attainment_margin=policy.attainment_margin,
        )
        if (type(result) is not ServiceVerdict or result.required != policy.required
            or result.ratio is None or not math.isfinite(result.ratio) or result.ratio <= 0):
            raise ValueError("service scorer returned another policy or result type")
        confident = result.decision is not SpeedStageDecision.NO_DECISION
        passed = result.decision is SpeedStageDecision.PASS
        verdict = SpeedupVerdict(
            result.ratio, policy.null_noise, result.required, passed, confident,
            len(self.incumbent), result.detail, len(self.candidate),
        )
        return ScheduleGrade(verdict, result.decision, False, None,
                             "goodput" if passed else None, format(result.ratio, ".17g"))

    def to_dict(self):
        """Retain raw turn stamps through the existing witness codec."""
        return ContinuationCodec((GoodputReadSet,)).encode(self)

    @classmethod
    def from_dict(cls, value):
        """Reopen fixed work and all source measurements, not a cached verdict."""
        result = ContinuationCodec((cls,)).decode(value)
        if type(result) is not cls:
            raise ValueError("goodput read set is not exactly typed")
        return result


def run_goodput_pair(plan, *, baseline_executor, candidate_executor, model_mount, deadline, clock, quality_control):
    """Run both sealed arms; neither lane tears down before the peer's complete fixed work."""
    from cacheon.eval.agent_replay import run_replay
    from cacheon.eval.crossover_runtime import ResidentCrossoverEvidence, _lane_digest
    from cacheon.eval.resident_schedule import ReadSchedule
    from cacheon.eval.scoring import marginal_workload_digest
    from cacheon.eval import service_capacity

    if not callable(getattr(service_capacity, "grade", None)) or not callable(quality_control):
        raise ValueError("goodput qualification scorer and quality producer must be installed before engine launch")

    started = float(clock())
    deadline = min(deadline, started + plan.policy.max_stage_seconds)
    if not math.isfinite(deadline) or deadline <= started:
        raise ValueError("goodput stage has no remaining execution budget")
    lanes = (_lane_digest(baseline_executor, plan.baseline),
             _lane_digest(candidate_executor, plan.candidate))
    schedule = ReadSchedule()
    replay = plan.baseline.session_plan.replay
    (load,) = replay.loads
    work = replay.slice.expected_work(load)
    expected = tuple(sorted((root, main, inner) for root, (main, inner) in work.items()))

    def execute(index, executor, arm):
        prefix, peer = ("incumbent", "candidate") if index == 0 else ("candidate", "incumbent")
        replay = replace(arm.session_plan.replay, arm=prefix, lane=lanes[index],
                         output_directory=arm.session_plan.replay.output_directory / prefix)
        session_plan = replace(arm.session_plan, replay=replay)

        def drive(controller):
            try:
                for _ in range(session_plan.warmup_count):
                    controller.execute_next()

                async def ready(load):
                    key = f"{load}:ready:"
                    schedule.put(key + prefix)
                    schedule.get(key + peer, deadline=deadline, clock=clock)

                read_plan = replace(replay, output_directory=replay.output_directory / controller.session_id)
                reads = asyncio.run(run_replay(controller, read_plan, before_read=ready))
                schedule.put(prefix, (reads, read_plan.output_directory))
                peer_reads, peer_directory = schedule.get(peer, deadline=deadline, clock=clock)
                if index == 0:
                    retained = GoodputReadSet(reads, peer_reads, expected)
                    grade = retained.grade(plan.policy.goodput)
                    schedule.put("read_set", retained)
                    schedule.put("grade", grade)
                    if grade.decision is SpeedStageDecision.PASS:
                        candidate, quiescence = schedule.get("candidate_closed", deadline=deadline, clock=clock)
                        schedule.put("quality", quality_control(
                            controller, candidate, read_plan.output_directory, peer_directory, quiescence,
                        ))
                return controller.finish(require_all=False)
            except BaseException as exc:
                schedule.fail(exc)
                raise
        try:
            execution = executor.execute_opened(arm.launch, arm.binding, model_mount, session_plan,
                                                deadline=deadline, driver=drive)
            if index == 1:
                schedule.put("candidate_closed", (execution, executor.prove_quiescent()))
            return execution
        except BaseException as exc:
            schedule.fail(exc)
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cacheon-goodput") as pool:
        futures = [pool.submit(execute, index, executor, arm) for index, (executor, arm) in enumerate((
            (baseline_executor, plan.baseline), (candidate_executor, plan.candidate),
        ))]
        executions, errors = [], []
        for future in futures:
            try:
                executions.append(future.result())
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise schedule.failure or errors[0]
    reads, grade = schedule.values["read_set"], schedule.values["grade"]
    pairs, controls, entropy = schedule.values.get("quality", ((), (), None))
    evidence = ResidentCrossoverEvidence(
        plan.digest, plan.selected_delta_digest, plan.policy,
        marginal_workload_digest(plan.baseline.session_plan), *lanes, *executions,
        baseline_executor.prove_quiescent(), candidate_executor.prove_quiescent(), (),
        grade.verdict, grade.verdict, False, grade.decision,
        "clear_" + grade.decision.value.lower(), started, float(clock()), goodput=reads,
        prompt_pairs=pairs, reference_inputs=controls, quality_entropy=entropy,
    )
    evidence.regrade(plan)
    return evidence


def regrade_goodput_execution(evidence, plan):
    """Verify complete replay records against the sealed workload and host pipe evidence."""
    from cacheon.eval.crossover_runtime import _validate_execution_binding, _expected_lane_digest
    from cacheon.eval.resident_measurement import CrossoverRuntimeError

    replay = plan.baseline.session_plan.replay
    (load,) = replay.loads
    expected = tuple(sorted((root, main, inner) for root, (main, inner) in replay.slice.expected_work(load).items()))
    if evidence.goodput.expected != expected:
        raise CrossoverRuntimeError("goodput fixed work differs from the sealed slice")
    for reads, execution, arm in (
        (evidence.goodput.incumbent, evidence.baseline_execution, plan.baseline),
        (evidence.goodput.candidate, evidence.candidate_execution, plan.candidate),
    ):
        _validate_execution_binding(execution, arm)
        session = execution.session
        if reads != session.replay_reads or len(reads) != 1:
            raise CrossoverRuntimeError("goodput reads differ from their completed engine session")
        read = reads[0]
        if (read.window, read.load, read.lane) != (replay.window, load, _expected_lane_digest(arm)):
            raise CrossoverRuntimeError("goodput read differs from its commissioned load or lane")
        warmup = arm.session_plan.warmup_count
        count = replay.slice.turns(load)
        rows = session.batches[warmup:warmup + count]
        controls = evidence.reference_inputs if arm is plan.baseline else ()
        if (len(session.batches) != warmup + count + len(controls)
            or tuple(row.batch_index for row in session.batches) != tuple(range(len(session.batches)))
            or len({row.request_id for row in session.batches}) != len(session.batches)
            or len({row.nonce for row in session.batches}) != len(session.batches)):
            raise CrossoverRuntimeError("goodput host requests are incomplete or repeated")
        rows = sorted(rows, key=lambda row: row.request_started_at)
        records = sorted(read.records, key=lambda row: row.request_start_ns)
        offset = records[0].request_start_ns - round(rows[0].request_started_at * 1e9)
        for row, record in zip(rows, records, strict=True):
            if len(row.evidence.prompts) != 1 or len(row.prompt_latencies) != 1 or row.audit_receipts:
                raise CrossoverRuntimeError("goodput request lacks its canonical token and timing evidence")
            prompt = row.evidence.prompts[0]
            if (record.request_start_ns != offset + round(row.request_started_at * 1e9)
                or record.first_token_ns != offset + round((row.request_started_at + row.prompt_latencies[0][0]) * 1e9)
                or record.request_end_ns != offset + round(row.response_completed_at * 1e9)
                or (record.prompt_tokens, record.output_tokens) != (prompt.prompt_tokens, len(prompt.output_ids))
                or row.token_numerator != record.output_tokens
                or row.response_completed_at > session.session_completed_at):
                raise CrossoverRuntimeError("goodput record does not regrade from its host request")
        for control, row in zip(controls, session.batches[warmup + count:], strict=True):
            if (len(row.evidence.prompts) != 1 or row.audit_receipts
                or control.roles[2].output_ids != row.evidence.prompts[0].output_ids
                or row.input_ids_sha256 != (hashlib.sha256(control.input_bytes).hexdigest(),)):
                raise CrossoverRuntimeError("incumbent quality control differs from its completed request")
    if evidence.decision is SpeedStageDecision.PASS:
        if not evidence.reference_inputs or not evidence.prompt_pairs or evidence.quality_entropy is None:
            raise CrossoverRuntimeError("passing replay lacks its selected incumbent quality controls")
        from cacheon.eval.qualification_trajectories import validate_controls
        validate_controls(evidence, plan)
    return evidence.goodput.grade(plan.policy.goodput)

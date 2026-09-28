"""Bind paired replay execution to the service scorer and existing qualification witness."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import math
from dataclasses import asdict, dataclass, field, replace

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.service_capacity import (
    LoadRead, ServiceContract, ServiceVerdict, continue_windows, fixed_work_rate,
)
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
    error_rate: float = field(default=0.0, metadata={"wire_optional": True})
    boot_noise: float = field(default=0.0, metadata={"wire_optional": True})

    def __post_init__(self):
        values = (self.required, self.null_noise, self.attainment_tolerance, self.attainment_margin,
                  self.error_rate, self.boot_noise)
        if (type(self.contract) is not ServiceContract
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not (self.required == 1.0 if self.error_rate else 1 < self.required < 2)
            or not 0 <= self.null_noise < 1 or not 0 <= self.boot_noise < 1
            or not 0 <= self.error_rate < 0.5
            or (self.error_rate > 0 and self.null_noise == self.boot_noise == 0)
            or (self.error_rate == 0 and self.boot_noise != 0)
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
                **{key: format(getattr(self, key), ".17g") for key in self.__dataclass_fields__
                   if key != "contract" and (self.error_rate or key not in ("error_rate", "boot_noise"))}}

    @classmethod
    def from_dict(cls, value):
        """Reopen only this exact policy type."""
        fields = set(cls.__dataclass_fields__)
        if type(value) is dict and "error_rate" not in value:
            fields -= {"error_rate", "boot_noise"}
        if type(value) is not dict or set(value) != fields or type(value["contract"]) is not dict:
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
    window_limit: int = field(default=0, metadata={"wire_optional": True})

    def __post_init__(self):
        if (not self.incumbent or len(self.incumbent) != len(self.candidate)
            or tuple(sorted(set(self.expected))) != self.expected
            or len({row[0] for row in self.expected}) != len(self.expected)):
            raise ValueError("goodput reads or fixed-work authority are incomplete")
        if (type(self.window_limit) is not int or self.window_limit < 0
            or (self.window_limit and not len(self.incumbent) <= self.window_limit <= 5)):
            raise ValueError("goodput read budget is invalid")
        work = {root: (main, inner) for root, main, inner in self.expected}
        for baseline, candidate in zip(self.incumbent, self.candidate, strict=True):
            if (baseline.arm != "incumbent" or candidate.arm != "candidate"
                or (baseline.window, baseline.load) != (candidate.window, candidate.load)
                or baseline.lane == candidate.lane):
                raise ValueError("goodput reads are not a paired load window")
            fixed_work_rate(baseline, work)
            fixed_work_rate(candidate, work)

    def grade(self, policy: GoodputPolicy, *, max_windows: int | None = None):
        """Call the data/scoring owner's single grade entrypoint; retain no second estimator."""
        from cacheon.eval import service_capacity
        from cacheon.eval.resident_schedule import ScheduleGrade

        statistic = {} if not policy.error_rate else dict(
            window_noise=policy.null_noise, boot_noise=policy.boot_noise,
            error_rate=policy.error_rate, max_windows=max_windows or self.window_limit or len(self.candidate),
        )
        grader = service_capacity.statistical_grade if statistic else service_capacity.grade
        result = grader(
            self.candidate, self.incumbent, policy.contract,
            {root: (main, inner) for root, main, inner in self.expected},
            **(statistic or {"required": policy.required}), attainment_tolerance=policy.attainment_tolerance,
            attainment_margin=policy.attainment_margin,
        )
        if (type(result) is not ServiceVerdict or (not policy.error_rate and result.required != policy.required)
            or not math.isfinite(result.ratio) or result.ratio <= 0):
            raise ValueError("service scorer returned another policy or result type")
        confident = result.decision is not SpeedStageDecision.NO_DECISION
        passed = result.decision is SpeedStageDecision.PASS
        verdict = SpeedupVerdict(
            result.ratio, result.standard_error if policy.error_rate else policy.null_noise, result.required, passed, confident,
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


def _orientation_plan(plan, windows, *, swapped=False):
    """Derive a shorter read and, when requested, exchange the sealed physical bindings."""
    def arm(source, lane):
        launch = replace(source.launch, hardware=lane.launch.hardware,
                         resource_policy_digest=lane.launch.resource_policy_digest)
        binding = replace(source.binding, physical_hardware=lane.binding.physical_hardware,
                          runtime_preflight_receipt=lane.binding.runtime_preflight_receipt)
        session = replace(source.session_plan, launch_digest=launch.digest,
                          expected_preflight=replace(source.session_plan.expected_preflight,
                                                     launch_digest=launch.digest),
                          replay=replace(source.session_plan.replay, windows=windows,
                              max_work_seconds=(source.session_plan.replay.max_work_seconds * windows
                                                // source.session_plan.replay.windows)))
        return replace(source, launch=launch, binding=binding, session_plan=session,
                       executor_namespace_digest=lane.executor_namespace_digest,
                       runtime_resource_policy_digest=lane.runtime_resource_policy_digest,
                       device_configuration_digest=lane.device_configuration_digest)
    baseline_lane, candidate_lane = ((plan.candidate, plan.baseline) if swapped
                                    else (plan.baseline, plan.candidate))
    return replace(plan, baseline=arm(plan.baseline, baseline_lane),
                   candidate=arm(plan.candidate, candidate_lane))


def run_goodput_pair(plan, *, baseline_executor, candidate_executor, model_mount, deadline, clock, quality_control):
    """Use one execution path for historical pairs and statistically scored lane swaps."""
    from transformers import AutoTokenizer

    # Concurrent lazy imports failed after both engines had booted (2026-09-28).
    # Resolve every tokenizer input before launching either lane; each thread owns one instance.
    tokenizers = tuple(AutoTokenizer.from_pretrained(
        arm.session_plan.replay.tokenizer_path, trust_remote_code=True, local_files_only=True,
    ) for arm in (plan.baseline, plan.candidate))
    kwargs = dict(model_mount=model_mount, deadline=deadline, clock=clock,
                  quality_control=quality_control, tokenizers=tokenizers)
    if not plan.policy.goodput.error_rate:
        return _run_orientation(plan, baseline_executor=baseline_executor,
                                candidate_executor=candidate_executor, **kwargs)
    from cacheon.eval.crossover_runtime import ResidentCrossoverEvidence, _expected_lane_digest

    windows = plan.baseline.session_plan.replay.windows
    if not 2 <= windows <= 5:
        raise ValueError("statistical replay requires two to five paired windows")
    split = windows // 2
    first = _run_orientation(_orientation_plan(plan, split), baseline_executor=baseline_executor,
                             candidate_executor=candidate_executor, full_plan=plan, **kwargs)
    swapped = _orientation_plan(plan, windows-split, swapped=True)
    last = _run_orientation(swapped, baseline_executor=candidate_executor,
                            candidate_executor=baseline_executor, full_plan=plan,
                            prior=first["reads"], **kwargs)
    grade = last["grade"]
    pairs, controls, entropy = last["quality"]
    evidence = ResidentCrossoverEvidence(
        plan.digest, plan.selected_delta_digest, plan.policy, first["workload"],
        _expected_lane_digest(swapped.baseline), _expected_lane_digest(swapped.candidate),
        *last["executions"], candidate_executor.prove_quiescent(), baseline_executor.prove_quiescent(), (),
        grade.verdict, grade.verdict, False, grade.decision, "clear_"+grade.decision.value.lower(),
        first["started"], float(clock()), goodput=last["reads"], prompt_pairs=pairs,
        reference_inputs=controls, quality_entropy=entropy, prior_executions=tuple(first["executions"]),
    )
    evidence.regrade(plan)
    return evidence


def _run_orientation(plan, *, baseline_executor, candidate_executor, model_mount, deadline, clock,
                     quality_control, tokenizers, full_plan=None, prior=None):
    """Keep each pair resident through its timed reads and any selected quality controls."""
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
    maximum = replay.windows if full_plan is None else full_plan.baseline.session_plan.replay.windows

    def read_set(baseline, candidate):
        return GoodputReadSet(
            (() if prior is None else prior.incumbent) + tuple(baseline),
            (() if prior is None else prior.candidate) + tuple(candidate), expected,
            maximum if plan.policy.goodput.error_rate else 0,
        )

    def execute(index, executor, arm):
        prefix, peer = ("incumbent", "candidate") if index == 0 else ("candidate", "incumbent")
        replay = replace(arm.session_plan.replay, arm=prefix, lane=lanes[index],
                         output_directory=arm.session_plan.replay.output_directory / prefix)
        session_plan = replace(arm.session_plan, replay=replay)

        def drive(controller):
            try:
                for _ in range(session_plan.warmup_count):
                    controller.execute_next()

                rates = {"incumbent": [], "candidate": []}
                completed = {"incumbent": [], "candidate": []}

                async def ready(load, window):
                    # Both lanes publish the finished window's rate, then decide identically whether
                    # the sealed sequential rule wants another; a split decision cannot pair and
                    # times out at the barrier instead of reading on alone.
                    proceed = True
                    if window > 1:
                        if plan.policy.goodput.error_rate:
                            completed[prefix].append(controller.replay_reads[-1])
                            schedule.put(f"{window}:read:{prefix}", completed[prefix][-1])
                            completed[peer].append(schedule.get(f"{window}:read:{peer}", deadline=deadline, clock=clock))
                            if prior is not None:
                                grade = read_set(completed["incumbent"], completed["candidate"]).grade(
                                    plan.policy.goodput, max_windows=maximum)
                                proceed = grade.decision is SpeedStageDecision.NO_DECISION
                        else:
                            rates[prefix].append(fixed_work_rate(controller.replay_reads[-1], work).rate)
                            schedule.put(f"{load}:{window - 1}:rate:{prefix}", rates[prefix][-1])
                            rates[peer].append(schedule.get(f"{load}:{window - 1}:rate:{peer}", deadline=deadline, clock=clock))
                            ratios = [c / i for c, i in zip(rates["candidate"], rates["incumbent"], strict=True)]
                            proceed = continue_windows(ratios, required=plan.policy.goodput.required,
                                                       null_noise=plan.policy.goodput.null_noise, max_windows=replay.windows)
                    key = f"{load}:{window}:{'ready' if proceed else 'stop'}:"
                    schedule.put(key + prefix)
                    schedule.get(key + peer, deadline=deadline, clock=clock)
                    return proceed

                read_plan = replace(replay, output_directory=replay.output_directory / controller.session_id)
                async def bounded_replay():
                    task = run_replay(controller, read_plan, tokenizer=tokenizers[index], before_read=ready)
                    return await asyncio.wait_for(task, timeout=read_plan.max_work_seconds or None)

                reads = asyncio.run(bounded_replay())
                schedule.put(prefix, (reads, read_plan.output_directory))
                peer_reads, peer_directory = schedule.get(peer, deadline=deadline, clock=clock)
                if index == 0:
                    retained = read_set(reads, peer_reads)
                    grade = retained.grade(plan.policy.goodput, max_windows=maximum)
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
    if full_plan is not None:
        return dict(reads=reads, grade=grade, executions=executions, quality=(pairs, controls, entropy),
                    started=started, workload=marginal_workload_digest(full_plan.baseline.session_plan))
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
    sessions = [
        (evidence.goodput.incumbent, evidence.baseline_execution, plan.baseline, evidence.reference_inputs),
        (evidence.goodput.candidate, evidence.candidate_execution, plan.candidate, ()),
    ]
    if plan.policy.goodput.error_rate:
        split = replay.windows // 2
        if evidence.goodput.window_limit != replay.windows or len(evidence.goodput.incumbent) <= split:
            raise CrossoverRuntimeError("statistical replay lacks its sealed budget or swapped reads")
        first = _orientation_plan(plan, split)
        last = _orientation_plan(plan, replay.windows-split, swapped=True)
        sessions = [
            (evidence.goodput.incumbent[:split], evidence.prior_executions[0], first.baseline, ()),
            (evidence.goodput.candidate[:split], evidence.prior_executions[1], first.candidate, ()),
            (evidence.goodput.incumbent[split:], evidence.baseline_execution, last.baseline, evidence.reference_inputs),
            (evidence.goodput.candidate[split:], evidence.candidate_execution, last.candidate, ()),
        ]
        completed = max(row.session.session_completed_at for row in evidence.prior_executions)
        if completed > min(evidence.baseline_execution.session.ready_completed_at,
                           evidence.candidate_execution.session.ready_completed_at):
            raise CrossoverRuntimeError("swapped read predates completion of its first orientation")
    for reads, execution, arm, controls in sessions:
        _validate_execution_binding(execution, arm)
        session = execution.session
        if reads != session.replay_reads or not 1 <= len(reads) <= arm.session_plan.replay.windows:
            raise CrossoverRuntimeError("goodput reads differ from their completed engine session")
        warmup = arm.session_plan.warmup_count
        count = replay.slice.turns(load)
        measured = count * len(reads)
        if (len(session.batches) != warmup + measured + len(controls)
            or tuple(row.batch_index for row in session.batches) != tuple(range(len(session.batches)))
            or len({row.request_id for row in session.batches}) != len(session.batches)
            or len({row.nonce for row in session.batches}) != len(session.batches)):
            raise CrossoverRuntimeError("goodput host requests are incomplete or repeated")
        for window, read in enumerate(reads, start=1):
            if (read.window, read.load, read.lane) != (window, load, _expected_lane_digest(arm)):
                raise CrossoverRuntimeError("goodput read differs from its commissioned load, window or lane")
            first = warmup + (window - 1) * count
            rows = sorted(session.batches[first:first + count], key=lambda row: row.request_started_at)
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
        for control, row in zip(controls, session.batches[warmup + measured:], strict=True):
            if (len(row.evidence.prompts) != 1 or row.audit_receipts
                or control.roles[2].output_ids != row.evidence.prompts[0].output_ids
                or row.input_ids_sha256 != (hashlib.sha256(control.input_bytes).hexdigest(),)):
                raise CrossoverRuntimeError("incumbent quality control differs from its completed request")
    # The lanes stopped where the sealed sequential rule stopped them, replayed from the retained rates.
    work = replay.slice.expected_work(load)
    ratios = [fixed_work_rate(cand, work).rate / fixed_work_rate(inc, work).rate
              for cand, inc in zip(evidence.goodput.candidate, evidence.goodput.incumbent, strict=True)]
    for read in range(1, len(ratios) + 1):
        if plan.policy.goodput.error_rate:
            prefix = replace(evidence.goodput, incumbent=evidence.goodput.incumbent[:read],
                             candidate=evidence.goodput.candidate[:read])
            again = prefix.grade(plan.policy.goodput).decision is SpeedStageDecision.NO_DECISION
        else:
            again = continue_windows(ratios[:read], required=plan.policy.goodput.required,
                                     null_noise=plan.policy.goodput.null_noise, max_windows=replay.windows)
        if again != (read < len(ratios)):
            raise CrossoverRuntimeError("goodput reads did not stop where the sealed sequential rule stops")
    if evidence.decision is SpeedStageDecision.PASS:
        if not evidence.reference_inputs or not evidence.prompt_pairs or evidence.quality_entropy is None:
            raise CrossoverRuntimeError("passing replay lacks its selected incumbent quality controls")
        from cacheon.eval.qualification_trajectories import validate_controls
        validate_controls(evidence, plan)
    return evidence.goodput.grade(plan.policy.goodput)

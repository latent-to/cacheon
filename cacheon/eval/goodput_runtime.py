"""Run replay engine lifetimes on one lane and grade them through the service scorer.

Version 19 gives one engine the whole node. A stock reference campaign is measured once per lane
identity and retained; each candidate reads whole lifetimes against it; a PASS adds one stock bracket
lifetime that is both the drift sentinel and the incumbent quality control. Paired read sets (versions
16-18) stay readable for the witnesses that retain them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.service_capacity import LoadRead, ServiceContract, ServiceVerdict, fixed_work_rate
from cacheon.eval.scoring import CrossoverRuntimeError, SpeedupVerdict, marginal_workload_digest
from cacheon.eval.speed_verdict import SpeedStageDecision
from cacheon.stack_identity import canonical_digest, canonical_json_bytes

# Version 19: stock lifetimes in one reference campaign, and the most candidate lifetimes a stage reads.
REFERENCE_BOOTS = 4
CANDIDATE_BOOTS = 2
REFERENCE_DOMAIN = "cacheon.qualification.stock-reference"
REFERENCE_SCHEMA = "cacheon.qualification.stock-reference.v1"
_REFERENCE_BYTES = 512 << 20


@dataclass(frozen=True)
class ScheduleGrade:
    """One grade of the retained windows: the headline verdict and the settled speedup text."""

    verdict: SpeedupVerdict
    decision: SpeedStageDecision
    settled_speedup: str


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
    # Statistical grades only: a first orientation reading this far slower ends the stage (0 = off).
    futility_margin: float = field(default=0.0, metadata={"wire_optional": True})
    # 1 grades service attainment once over all paired windows, 0 in every window; sealed only when set.
    attainment_pooled: float = field(default=0.0, metadata={"wire_optional": True})

    def __post_init__(self):
        values = (self.required, self.null_noise, self.attainment_tolerance, self.attainment_margin,
                  self.error_rate, self.boot_noise, self.futility_margin, self.attainment_pooled)
        if (type(self.contract) is not ServiceContract
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not (self.required == 1.0 if self.error_rate else 1 < self.required < 2)
            or not 0 <= self.null_noise < 1 or not 0 <= self.boot_noise < 1
            or not 0 <= self.error_rate < 0.5
            or (self.error_rate > 0 and self.null_noise == self.boot_noise == 0)
            or (self.error_rate == 0 and (self.boot_noise or self.futility_margin)) or not 0 <= self.futility_margin < 1
            or not 0 <= self.attainment_tolerance <= 1 or not 0 <= self.attainment_margin < 1
            or self.attainment_pooled not in (0, 1)):
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
                   if key != "contract" and (self.error_rate or key not in ("error_rate", "boot_noise"))
                   and (key not in ("futility_margin", "attainment_pooled") or getattr(self, key))}}

    @classmethod
    def from_dict(cls, value):
        """Reopen only this exact policy type."""
        fields = set(cls.__dataclass_fields__)
        if type(value) is dict and "error_rate" not in value:
            fields -= {"error_rate", "boot_noise"}
        if type(value) is dict and "futility_margin" not in value:
            fields -= {"futility_margin"}
        if type(value) is dict and "attainment_pooled" not in value:
            fields -= {"attainment_pooled"}
        if type(value) is not dict or set(value) != fields or type(value["contract"]) is not dict:
            raise ValueError("goodput policy fields differ")
        result = cls(ServiceContract(**{key: float(v) for key, v in value["contract"].items()}),
                     **{key: float(v) for key, v in value.items() if key != "contract"})
        if result.to_dict() != value:
            raise ValueError("goodput policy is not canonical")
        return result


@dataclass(frozen=True)
class GoodputReadSet:
    """The turn records replacing token-window rates in a replay speed witness.

    Paired policies hold one incumbent read per candidate read. Version 19 holds the stock reference
    campaign (``reference_boots`` lifetimes, then at most one stock sentinel lifetime) as the incumbent
    and up to ``boots`` candidate lifetimes, every lifetime ``window_limit`` windows on one lane.
    """

    incumbent: tuple[LoadRead, ...]
    candidate: tuple[LoadRead, ...]
    expected: tuple[tuple[str, int, int], ...]
    window_limit: int = field(default=0, metadata={"wire_optional": True})
    orientations: tuple[int, ...] = field(default=(), metadata={"wire_optional": True})
    boots: int = field(default=0, metadata={"wire_optional": True})
    reference_boots: int = field(default=0, metadata={"wire_optional": True})

    def __post_init__(self):
        if (not self.incumbent or not self.candidate
            or tuple(sorted(set(self.expected))) != self.expected
            or len({row[0] for row in self.expected}) != len(self.expected)):
            raise ValueError("goodput reads or fixed-work authority are incomplete")
        work = {root: (main, inner) for root, main, inner in self.expected}
        if self.boots or self.reference_boots:
            size = self.window_limit
            if (type(self.boots) is not int or type(self.reference_boots) is not int or self.orientations
                or not 2 <= self.boots <= CANDIDATE_BOOTS or self.reference_boots != REFERENCE_BOOTS
                or type(size) is not int or not 1 <= size <= 5
                or len(self.incumbent) not in (self.reference_boots * size, (self.reference_boots + 1) * size)
                or len(self.candidate) % size or len(self.candidate) > self.boots * size
                or any(read.window != index % size + 1 for reads in (self.incumbent, self.candidate)
                       for index, read in enumerate(reads))
                or any(read.arm != "incumbent" for read in self.incumbent)
                or any(read.arm != "candidate" for read in self.candidate)
                or len({(read.lane, read.load) for read in (*self.incumbent, *self.candidate)}) != 1):
                raise ValueError("reference read set is not whole lifetimes on one lane")
            for read in (*self.incumbent, *self.candidate):
                fixed_work_rate(read, work)
            return
        if len(self.incumbent) != len(self.candidate):
            raise ValueError("goodput reads or fixed-work authority are incomplete")
        if (type(self.window_limit) is not int or self.window_limit < 0
            or (self.window_limit and not len(self.incumbent) <= self.window_limit <= 5)):
            raise ValueError("goodput read budget is invalid")
        if self.orientations and self.orientations != tuple(
                int(i >= self.window_limit // 2) for i in range(len(self.incumbent))):
            raise ValueError("portable replay changed its sealed orientation order")
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

        work = {root: (main, inner) for root, main, inner in self.expected}
        service = dict(attainment_tolerance=policy.attainment_tolerance, attainment_margin=policy.attainment_margin)
        if self.boots:
            split = self.reference_boots * self.window_limit
            result = service_capacity.reference_grade(
                self.candidate, self.incumbent[:split], policy.contract, work, window_noise=policy.null_noise,
                boot_noise=policy.boot_noise, error_rate=policy.error_rate, boots=self.boots,
                futility_margin=policy.futility_margin, sentinel=self.incumbent[split:], **service,
            )
        elif policy.error_rate:
            result = service_capacity.statistical_grade(
                self.candidate, self.incumbent, policy.contract, work, window_noise=policy.null_noise,
                boot_noise=policy.boot_noise, error_rate=policy.error_rate,
                max_windows=max_windows or self.window_limit or len(self.candidate),
                futility_margin=policy.futility_margin, pooled_attainment=bool(policy.attainment_pooled),
                **({"orientations": self.orientations} if self.orientations else {}), **service,
            )
        else:
            result = service_capacity.grade(self.candidate, self.incumbent, policy.contract, work,
                                            required=policy.required, **service)
        if (type(result) is not ServiceVerdict or (not policy.error_rate and result.required != policy.required)
            or not math.isfinite(result.ratio) or result.ratio <= 0):
            raise ValueError("service scorer returned another policy or result type")
        confident = result.decision is not SpeedStageDecision.NO_DECISION
        passed = result.decision is SpeedStageDecision.PASS
        verdict = SpeedupVerdict(
            result.ratio, result.standard_error if policy.error_rate else policy.null_noise, result.required, passed, confident,
            len(self.incumbent), result.detail, len(self.candidate),
        )
        return ScheduleGrade(verdict, result.decision, format(result.ratio, ".17g"))

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


def _tokenizer(plan):
    # A lazy tokenizer import once failed after both engines had booted (2026-09-28); resolve it first.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(plan.candidate.session_plan.replay.tokenizer_path,
                                         trust_remote_code=True, local_files_only=True)


def _boot(executor, arm, lane, model_mount, *, prefix, deadline, tokenizer, after=None):
    """Run one engine lifetime's sealed windows; ``after`` runs on the still-open engine."""
    from cacheon.eval.agent_replay import run_replay

    replay = replace(arm.session_plan.replay, arm=prefix, lane=lane,
                     output_directory=arm.session_plan.replay.output_directory / prefix)
    session_plan = replace(arm.session_plan, replay=replay)
    held = {}

    def drive(controller):
        for _ in range(session_plan.warmup_count):
            controller.execute_next()
        held["directory"] = replay.output_directory / controller.session_id
        read_plan = replace(replay, output_directory=held["directory"])

        async def bounded_replay():
            task = run_replay(controller, read_plan, tokenizer=tokenizer)
            return await asyncio.wait_for(task, timeout=read_plan.max_work_seconds or None)

        asyncio.run(bounded_replay())
        if after is not None:
            held["after"] = after(controller, held["directory"])
        return controller.finish(require_all=False)

    execution = executor.execute_opened(arm.launch, arm.binding, model_mount, session_plan,
                                        deadline=deadline, driver=drive)
    return execution, executor.prove_quiescent(), held["directory"], held.get("after")


def _reference_pointer(plan, evidence_root) -> Path:
    identity = canonical_digest(REFERENCE_SCHEMA, {
        "boots": REFERENCE_BOOTS, "lane": plan.baseline_lane_digest, "launch": plan.baseline.launch.digest,
        "runtime": plan.baseline.binding.runtime_preflight_receipt.sha256,
        "workload": marginal_workload_digest(plan.baseline.session_plan),
    })
    # A dot name can never be an evidence domain directory.
    return Path(evidence_root) / ".stock-references" / f"{identity}.json"


def open_reference(plan, *, executor, model_mount, evidence_root, deadline):
    """Reopen the lane's current stock reference, or measure its campaign when none is current.

    The campaign runs before the speed clock starts. A missing or retired reference is measured
    again on this lane, never borrowed from another lane, image, launch or workload.
    """
    from cacheon.eval.crossover_runtime import _lane_digest
    from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
    from cacheon.eval.oci_backend import EngineExecutionEvidence

    codec = ContinuationCodec((EngineExecutionEvidence,))
    pointer = _reference_pointer(plan, evidence_root)
    lane = _lane_digest(executor, plan.baseline)
    if pointer.exists():
        record = json.loads(reopen_evidence(evidence_root, EvidenceArtifactRef.from_dict(
            json.loads(pointer.read_text())), max_bytes=_REFERENCE_BYTES))
        if record.get("identity") != pointer.stem:
            raise CrossoverRuntimeError("stock reference names another lane identity")
        executions = tuple(codec.decode(row) for row in record["executions"])
    else:
        tokenizer = _tokenizer(plan)
        executions = tuple(_boot(executor, plan.baseline, lane, model_mount, prefix="incumbent",
                                 deadline=deadline, tokenizer=tokenizer)[0] for _ in range(REFERENCE_BOOTS))
    if len(executions) != REFERENCE_BOOTS or any(type(row) is not EngineExecutionEvidence for row in executions):
        raise CrossoverRuntimeError("stock reference lacks its sealed lifetimes")
    _validate_replay_sessions([(row.session.replay_reads, row, plan.baseline, ()) for row in executions],
                              plan.baseline.session_plan.replay)
    if not pointer.exists():
        reference = publish_evidence(evidence_root, canonical_json_bytes({
            "identity": pointer.stem, "executions": [codec.encode(row) for row in executions],
        }), domain=REFERENCE_DOMAIN, media_type="application/json", schema=REFERENCE_SCHEMA,
            max_bytes=_REFERENCE_BYTES)
        pointer.parent.mkdir(mode=0o700, exist_ok=True)
        with pointer.open("x") as handle:
            handle.write(json.dumps(reference.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return executions


def _expected(replay):
    (load,) = replay.loads
    return tuple(sorted((root, main, inner) for root, (main, inner) in replay.slice.expected_work(load).items()))


def run_reference_speed(plan, *, executor, model_mount, reference, evidence_root, deadline, clock,
                        quality_control):
    """Read whole candidate lifetimes against the reference; a PASS adds the stock bracket lifetime."""
    from cacheon.eval.crossover_runtime import ResidentCrossoverEvidence, _lane_digest

    if not callable(quality_control):
        raise ValueError("the quality producer must be installed before engine launch")
    tokenizer = _tokenizer(plan)
    started = float(clock())
    deadline = min(deadline, started + plan.policy.max_stage_seconds)
    if not math.isfinite(deadline) or deadline <= started:
        raise ValueError("goodput stage has no remaining execution budget")
    lane = _lane_digest(executor, plan.candidate)
    replay = plan.candidate.session_plan.replay
    stock = tuple(read for row in reference for read in row.session.replay_reads)
    boots = []
    for _ in range(CANDIDATE_BOOTS):
        boots.append(_boot(executor, plan.candidate, lane, model_mount, prefix="candidate",
                           deadline=deadline, tokenizer=tokenizer))
        reads = GoodputReadSet(stock, tuple(read for row in boots for read in row[0].session.replay_reads),
                               _expected(replay), replay.windows, boots=CANDIDATE_BOOTS,
                               reference_boots=REFERENCE_BOOTS)
        grade = reads.grade(plan.policy.goodput)
        if grade.decision is not SpeedStageDecision.NO_DECISION:
            break
    candidate, candidate_quiet, candidate_directory, _ = boots[-1]
    bracket, bracket_quiet, quality = None, candidate_quiet, ((), (), None)
    if grade.decision is SpeedStageDecision.PASS:
        bracket, bracket_quiet, _, quality = _boot(
            executor, plan.baseline, lane, model_mount, prefix="incumbent", deadline=deadline, tokenizer=tokenizer,
            after=lambda controller, directory: quality_control(
                controller, candidate, directory, candidate_directory, candidate_quiet))
        reads = replace(reads, incumbent=stock + bracket.session.replay_reads)
        grade = reads.grade(plan.policy.goodput)
        if grade.decision is SpeedStageDecision.NO_DECISION:
            # The sentinel closed this reference; the next qualification measures a new campaign.
            pointer = _reference_pointer(plan, evidence_root)
            pointer.rename(pointer.with_name(f"{pointer.stem}.retired-{time.time_ns()}.json"))
    pairs, controls, entropy = quality
    evidence = ResidentCrossoverEvidence(
        plan.digest, plan.selected_delta_digest, plan.policy, marginal_workload_digest(plan.baseline.session_plan),
        lane, lane, bracket, candidate, bracket_quiet, candidate_quiet, (), grade.verdict, grade.verdict, False,
        grade.decision, "clear_" + grade.decision.value.lower(), started, float(clock()), goodput=reads,
        prompt_pairs=pairs, reference_inputs=controls, quality_entropy=entropy,
        prior_executions=(*reference, *(row[0] for row in boots[:-1])),
    )
    evidence.regrade(plan)
    return evidence


def regrade_reference_speed(evidence, plan):
    """Regrade a stage from its retained lifetimes: reference, candidate boots, then any stock bracket."""
    from cacheon.eval.crossover_runtime import _expected_lane_digest

    replay = plan.candidate.session_plan.replay
    lane = _expected_lane_digest(plan.candidate)
    reference = evidence.prior_executions[:REFERENCE_BOOTS]
    candidates = (*evidence.prior_executions[REFERENCE_BOOTS:], evidence.candidate_execution)
    bracket = evidence.baseline_execution
    quiet = (evidence.candidate_quiescence, *((evidence.baseline_quiescence,) if bracket else ()))
    if (evidence.plan_digest != plan.digest or evidence.policy != plan.policy
        or evidence.selected_delta_digest != plan.selected_delta_digest
        or evidence.workload_digest != marginal_workload_digest(plan.baseline.session_plan)
        or (evidence.baseline_lane_digest, evidence.candidate_lane_digest) != (lane, lane)
        or len(reference) != REFERENCE_BOOTS or len(candidates) > CANDIDATE_BOOTS
        or (bracket is None) != (evidence.baseline_quiescence == evidence.candidate_quiescence)
        or any(row.namespace_digest != plan.candidate.executor_namespace_digest
               or (row.container_ids, row.lease_records, row.resource_entries) != ((), (), ()) for row in quiet)):
        raise CrossoverRuntimeError("reference speed evidence names another sealed plan")
    _validate_replay_sessions([
        *((row.session.replay_reads, row, plan.baseline, ()) for row in reference),
        *((row.session.replay_reads, row, plan.candidate, ()) for row in candidates),
        *(((bracket.session.replay_reads, bracket, plan.baseline, evidence.reference_inputs),) if bracket else ()),
    ], replay)
    stage = (*candidates, *((bracket,) if bracket else ()))
    marks = [evidence.started_monotonic_s]
    for row, receipt in zip(stage, (*(None,) * (len(candidates) - 1), *quiet)):
        marks += [row.session.ready_completed_at, row.session.session_completed_at]
        marks += [] if receipt is None else [receipt.observed_monotonic_s]
    if marks + [evidence.completed_monotonic_s] != sorted(marks + [evidence.completed_monotonic_s]):
        raise CrossoverRuntimeError("reference speed lifetimes overlap or leave the stage clock")
    stock = tuple(read for row in reference for read in row.session.replay_reads)
    rebuilt = GoodputReadSet(stock + (bracket.session.replay_reads if bracket else ()),
                             tuple(read for row in candidates for read in row.session.replay_reads),
                             _expected(replay), replay.windows, boots=CANDIDATE_BOOTS, reference_boots=REFERENCE_BOOTS)
    if rebuilt != evidence.goodput:
        raise CrossoverRuntimeError("reference speed reads differ from their engine sessions")
    for count in range(1, len(candidates) + 1):
        prefix = replace(rebuilt, incumbent=stock, candidate=rebuilt.candidate[:count * replay.windows])
        decision = prefix.grade(plan.policy.goodput).decision
        if (decision is SpeedStageDecision.NO_DECISION) != (count < len(candidates)):
            raise CrossoverRuntimeError("reference speed did not stop where its sealed rule stops")
    if (bracket is not None) != (decision is SpeedStageDecision.PASS):
        raise CrossoverRuntimeError("a stock bracket follows exactly a passing comparison")
    grade = rebuilt.grade(plan.policy.goodput)
    if (grade.verdict != evidence.initial_verdict or grade.verdict != evidence.final_verdict
        or grade.decision is not evidence.decision):
        raise CrossoverRuntimeError("reference speed headline does not regrade")
    if bracket is not None:
        if not evidence.reference_inputs or not evidence.prompt_pairs or evidence.quality_entropy is None:
            raise CrossoverRuntimeError("stock bracket lacks its selected incumbent quality controls")
        from cacheon.eval.qualification_trajectories import validate_controls
        validate_controls(evidence, plan)
    elif evidence.reference_inputs or evidence.prompt_pairs or evidence.quality_entropy is not None:
        raise CrossoverRuntimeError("quality controls exist only on a stock bracket")
    return grade.verdict


def _validate_replay_sessions(sessions, replay):
    """Bind every retained read to the host requests of the engine session that produced it."""
    from cacheon.eval.crossover_runtime import _validate_execution_binding, _expected_lane_digest

    (load,) = replay.loads
    for reads, execution, arm, controls in sessions:
        _validate_execution_binding(execution, arm)
        session = execution.session
        if reads != session.replay_reads or len(reads) != arm.session_plan.replay.windows:
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

"""Pair-native serialized resident crossover evidence.

This module joins the standing two-session owner to the already-sealed
crossover workload and scoring policy.  It deliberately produces request-slice
evidence, not OCI lifetime/quiescence evidence: neither resident session is
finished by a speed read.
"""

from __future__ import annotations

import math
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Callable

from cacheon.eval.crossover_runtime import (
    CrossoverRuntimeError,
    ResidentCrossoverPlan,
    ResidentReadRate,
    ResidentSpeedPolicy,
    SpeedStageDecision,
    TimedWindow,
)
from cacheon.eval.continuation_codec import ContinuationCodec, ContinuationCodecError
from cacheon.eval.oci_resident_session import ResidentBatchEvidence, ResidentBatchShape
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationHandle,
    ResidentEvaluationPair,
    ResidentEvaluationPairError,
    ResidentLaneIdentity,
    ResidentRequestResult,
    ResidentRequestSlice,
)
from cacheon.eval.resident_pair_binding import (
    ResidentPairBindingError,
    ResidentPairRuntimeBinding,
)
from cacheon.eval.scoring import (
    RawSpeedEvidenceError,
    SpeedupVerdict,
    marginal_workload_digest,
    score_speedup,
)
from cacheon.eval.speed_verdict import (
    resident_speed_roles,
    speed_grade,
    v6_grade,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex

class ResidentPairCrossoverError(RuntimeError):
    """A commissioned pair-crossover input is invalid."""


class ResidentPairCrossoverHold(ResidentPairCrossoverError):
    """Evidence is incomplete or ambiguous and cannot become candidate FAIL."""

    decision = "HOLD"


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (TypeError, ValueError) as exc:
        raise ResidentPairCrossoverError(str(exc)) from None


def _hex32(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and value != "0" * 32
        and all(char in "0123456789abcdef" for char in value)
    )


def _shape(plan: ResidentCrossoverPlan) -> ResidentBatchShape:
    workload = plan.baseline.session_plan
    return ResidentBatchShape(
        workload.max_new_tokens,
        workload.top_logprobs_num,
        workload.temperature,
        workload.expected_prompt_tokens,
    )


def _prompt_digest(plan: ResidentCrossoverPlan) -> str:
    return canonical_digest(
        "cacheon.eval.resident-pair-prompt-batches.v1",
        plan.baseline.session_plan.prompt_batches,
    )


@dataclass(frozen=True)
class ResidentPairCrossoverPlan:
    """One candidate bound to the standing sessions and physical lane roles."""

    candidate_bundle_digest: str
    crossover_plan: ResidentCrossoverPlan
    pair_binding: ResidentPairRuntimeBinding
    baseline_pair_lane: str
    candidate_pair_lane: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_bundle_digest",
            _digest(self.candidate_bundle_digest, "candidate bundle digest"),
        )
        if (
            type(self.crossover_plan) is not ResidentCrossoverPlan
            or type(self.pair_binding) is not ResidentPairRuntimeBinding
            or (self.baseline_pair_lane, self.candidate_pair_lane)
            not in (("A", "B"), ("B", "A"))
        ):
            raise ResidentPairCrossoverError(
                "resident pair crossover identity or orientation is invalid"
            )
        baseline = self.pair_binding.lookup(self.baseline_pair_lane)
        candidate = self.pair_binding.lookup(self.candidate_pair_lane)
        if (
            baseline.lane_digest != self.crossover_plan.baseline_lane_digest
            or candidate.lane_digest != self.crossover_plan.candidate_lane_digest
        ):
            raise ResidentPairCrossoverError(
                "resident pair roles differ from the sealed crossover lanes"
            )

    @property
    def pair_identities(self) -> tuple[ResidentLaneIdentity, ResidentLaneIdentity]:
        """Compatibility projection; full runtime authority is ``pair_binding``."""

        return self.pair_binding.identities

    def session_id(self, lane_id: str) -> str:
        try:
            return self.pair_binding.lookup(lane_id).session_id
        except ResidentPairBindingError as exc:
            raise ResidentPairCrossoverError(str(exc)) from None

    @property
    def workload_digest(self) -> str:
        return marginal_workload_digest(self.crossover_plan.baseline.session_plan)

    @property
    def batch_shape(self) -> ResidentBatchShape:
        return _shape(self.crossover_plan)

    @property
    def prompt_batches_digest(self) -> str:
        return _prompt_digest(self.crossover_plan)

    @property
    def digest(self) -> str:
        crossover = self.crossover_plan
        return canonical_digest(
            "cacheon.eval.resident-pair-crossover-plan.v2",
            {
                "baseline_pair_lane": self.baseline_pair_lane,
                "baseline_physical_lane": crossover.baseline_lane_digest,
                "candidate_bundle": self.candidate_bundle_digest,
                "candidate_pair_lane": self.candidate_pair_lane,
                "candidate_physical_lane": crossover.candidate_lane_digest,
                "crossover_plan": crossover.digest,
                "pair_binding": {
                    "digest": self.pair_binding.digest,
                    "lanes": [
                        {
                            "allocation": row.allocation_digest,
                            "executor_namespace": row.executor_namespace_digest,
                            "lane": row.lane_id,
                            "lane_authority": row.lane_digest,
                            "session": row.session_id,
                            "stock_launch": row.stock_launch_digest,
                        }
                        for row in self.pair_binding.lanes
                    ],
                    "service_epoch": self.pair_binding.service_epoch_digest,
                },
                "prompt_batches": self.prompt_batches_digest,
                "selected_delta": crossover.selected_delta_digest,
                "workload": self.workload_digest,
            },
        )


@dataclass(frozen=True)
class ResidentPairCrossoverEvidence:
    """Raw request slices plus independently reproducible adaptive headlines."""

    plan_digest: str
    candidate_bundle_digest: str
    selected_delta_digest: str
    workload_digest: str
    prompt_batches_digest: str
    policy: ResidentSpeedPolicy
    pair_binding: ResidentPairRuntimeBinding
    orientation: tuple[str, str]
    physical_lane_digests: tuple[str, str]
    batch_shape: ResidentBatchShape
    request_slices: tuple[ResidentRequestSlice, ...]
    rates: tuple[ResidentReadRate, ...]
    initial_verdict: SpeedupVerdict
    final_verdict: SpeedupVerdict
    escalated: bool
    decision: SpeedStageDecision
    started_monotonic_s: float
    deadline_monotonic_s: float
    completed_monotonic_s: float

    def __post_init__(self) -> None:
        for field in (
            "plan_digest",
            "candidate_bundle_digest",
            "selected_delta_digest",
            "workload_digest",
            "prompt_batches_digest",
        ):
            _digest(getattr(self, field), field.replace("_", " "))
        values = (
            self.started_monotonic_s,
            self.deadline_monotonic_s,
            self.completed_monotonic_s,
        )
        if (
            type(self.policy) is not ResidentSpeedPolicy
            or type(self.pair_binding) is not ResidentPairRuntimeBinding
            or type(self.orientation) is not tuple
            or type(self.physical_lane_digests) is not tuple
            or len(self.physical_lane_digests) != 2
            or type(self.request_slices) is not tuple
            or type(self.rates) is not tuple
            or any(type(row) is not ResidentRequestSlice for row in self.request_slices)
            or any(type(row) is not ResidentReadRate for row in self.rates)
            or type(self.batch_shape) is not ResidentBatchShape
            or type(self.initial_verdict) is not SpeedupVerdict
            or type(self.final_verdict) is not SpeedupVerdict
            or type(self.escalated) is not bool
            or type(self.decision) is not SpeedStageDecision
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
            or not self.started_monotonic_s
            < self.completed_monotonic_s
            <= self.deadline_monotonic_s
        ):
            raise ResidentPairCrossoverError("resident pair evidence is malformed")
        for value in self.physical_lane_digests:
            _digest(value, "physical lane digest")

    @property
    def pair_identities(self) -> tuple[ResidentLaneIdentity, ResidentLaneIdentity]:
        """Session-only projection retained for readers of version-1 evidence."""

        return self.pair_binding.identities

    def regrade(self, plan: ResidentPairCrossoverPlan) -> SpeedupVerdict:
        return _regrade(self, plan)

    @property
    def digest(self) -> str:
        """Canonical digest of every retained request slice and headline."""

        try:
            payload = ContinuationCodec((ResidentPairCrossoverEvidence,)).encode(self)
        except ContinuationCodecError as exc:
            raise ResidentPairCrossoverError(
                f"resident pair evidence is not canonically encodable: {exc}"
            ) from None
        return canonical_digest(
            "cacheon.eval.resident-pair-crossover-evidence.v1", payload
        )


_LOCKS_GUARD = threading.Lock()
_PAIR_LOCKS: weakref.WeakKeyDictionary[ResidentEvaluationPair, threading.Lock] = (
    weakref.WeakKeyDictionary()
)


def _pair_lock(pair: ResidentEvaluationPair) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _PAIR_LOCKS.get(pair)
        if lock is None:
            lock = threading.Lock()
            _PAIR_LOCKS[pair] = lock
        return lock


def _now(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except BaseException as exc:
        raise ResidentPairCrossoverHold(f"resident speed host clock failed: {exc}") from None
    if not math.isfinite(value):
        raise ResidentPairCrossoverHold("resident speed host clock is not finite")
    return value


def _within_wall(clock: Callable[[], float], deadline: float) -> float:
    now = _now(clock)
    if now >= deadline:
        raise ResidentPairCrossoverHold("resident pair speed stage timed out")
    return now


def _validate_outputs(batch: ResidentBatchEvidence, prompts: tuple[str, ...], shape: ResidentBatchShape) -> bool:
    evidence = batch.evidence
    expected = len(prompts) * shape.max_new_tokens
    return (
        type(evidence) is BatchEvidence
        and type(evidence.prompts) is tuple
        and len(evidence.prompts) == len(prompts)
        and batch.token_numerator == expected
        and evidence.observed_tokens == expected
        and all(
            type(prompt) is PromptEvidence
            and type(prompt.output_ids) is tuple
            and len(prompt.output_ids) == shape.max_new_tokens
            and type(prompt.top_logprobs) is tuple
            and len(prompt.top_logprobs) == shape.max_new_tokens
            and all(
                type(position) is tuple
                and len(position) == shape.top_logprobs_num
                for position in prompt.top_logprobs
            )
            for prompt in evidence.prompts
        )
    )


def _rate_from_slice(
    role: str, request: ResidentRequestSlice, plan: ResidentPairCrossoverPlan
) -> ResidentReadRate:
    candidate = role.startswith("C")
    crossover = plan.crossover_plan
    arm = crossover.candidate if candidate else crossover.baseline
    lane = plan.candidate_pair_lane if candidate else plan.baseline_pair_lane
    template = arm.session_plan
    batches, swaps = request.new_batches, request.new_swaps
    # v7 puts the baseline through one stock-to-stock swap so both arms take the
    # same recapture path; v6 and earlier left the baseline unswapped.
    expected_swaps = 2 if candidate else int(crossover.policy.version >= 7)
    if (
        type(request) is not ResidentRequestSlice
        or request.bundle_digest != plan.candidate_bundle_digest
        or request.lane_id != lane
        or request.session_id != plan.session_id(lane)
        or request.expected_batch_count != len(template.prompt_batches)
        or request.expected_swap_count != expected_swaps
        or len(batches) != len(template.prompt_batches)
        or len(swaps) != expected_swaps
        or request.ending_bundle_digest is not None
        or request.ending_slots
    ):
        raise ResidentPairCrossoverHold(f"resident {role} request slice is incomplete")
    if candidate:
        activation, restoration = swaps
        baseline_restock = None
        valid_dispatch = (
            activation.bundle_digest == plan.candidate_bundle_digest
            and bool(activation.slots)
            and restoration.bundle_digest is None
            and not restoration.slots
            and activation.generation == request.starting_generation + 1
            and restoration.generation == activation.generation + 1
            and request.ending_generation == restoration.generation
        )
    elif expected_swaps:
        # v7 baseline: exactly one stock-to-stock swap, taken *before* the
        # batches so both arms enter measurement through the same recapture.
        # It must carry no bundle and no slots — otherwise the "baseline" ran
        # something — and it advances the generation exactly once, like the
        # candidate activation. It is a leading swap, never a restoration:
        # the trailing-clock check below applies only to restorations.
        activation = restoration = None
        (baseline_restock,) = swaps
        valid_dispatch = (
            baseline_restock.bundle_digest is None
            and not baseline_restock.slots
            and baseline_restock.generation == request.starting_generation + 1
            and request.ending_generation == baseline_restock.generation
        )
    else:
        activation = restoration = baseline_restock = None
        valid_dispatch = request.ending_generation == request.starting_generation
    if not valid_dispatch:
        raise ResidentPairCrossoverHold(f"resident {role} dispatch or stock restore is ambiguous")
    if candidate:
        # The candidate's reads are worth nothing unless its kernel actually ran
        # during them. Registering a slot is not running it: a bundle can load,
        # register, capture, and then never dispatch, and every such run still
        # produces a clean speed number. The restoration swap closes the
        # activation generation, so it carries that generation's per-rank
        # execution count.
        #
        # HOLD rather than FAIL, deliberately. An unproven execution and a
        # broken evidence path are different claims, and only the first is the
        # candidate's fault; holding is recoverable and requeue is capped, while
        # a wrong FAIL is permanent and lands on an honest miner.
        proven = restoration.execution.proves_execution(
            generation=activation.generation,
            expected_ranks=restoration.expected_ranks,
        )
        if not proven:
            detail = (
                f"resident {role} candidate has no proof its kernel executed "
                f"(generation {restoration.execution.prior_generation}, "
                f"ranks {restoration.execution.prior_execution_ranks} of "
                f"{restoration.expected_ranks})"
            )
            raise ResidentPairCrossoverHold(detail)
    previous = request.host_started_at
    seen: set[str] = set()
    # Whichever swap precedes the batches: the candidate's activation, or v7's
    # stock-to-stock baseline swap. Stock exposes no slots either way.
    leading = activation if candidate else baseline_restock
    expected_generation = (
        leading.generation if leading is not None else request.starting_generation
    )
    expected_slots = activation.slots if activation is not None else ()
    if leading is not None:
        if not request.host_started_at <= leading.requested_at < leading.completed_at:
            raise ResidentPairCrossoverHold(f"resident {role} activation clock is malformed")
        previous = leading.completed_at
    for batch, prompts in zip(batches, template.prompt_batches, strict=True):
        if (
            type(batch) is not ResidentBatchEvidence
            or batch.batch_index != batches[0].batch_index + len(seen) // 2
            or not _hex32(batch.request_id)
            or not _hex32(batch.nonce)
            or batch.request_id == batch.nonce
            or batch.request_id in seen
            or batch.nonce in seen
            or batch.generation != expected_generation
            or batch.active_slots != expected_slots
            or batch.canary
            or not previous <= batch.request_started_at < batch.response_completed_at
            or not _validate_outputs(batch, prompts, plan.batch_shape)
        ):
            raise ResidentPairCrossoverHold(f"resident {role} batch evidence is malformed")
        seen.update((batch.request_id, batch.nonce))
        previous = batch.response_completed_at
    if restoration is not None:
        if not previous <= restoration.requested_at < restoration.completed_at:
            raise ResidentPairCrossoverHold(f"resident {role} restoration clock is malformed")
        previous = restoration.completed_at
    if previous > request.host_completed_at:
        raise ResidentPairCrossoverHold(f"resident {role} host span is incomplete")
    conditioning = batches[
        template.warmup_count - template.conditioning_count : template.warmup_count
    ]
    timed = batches[template.warmup_count :]
    conditioning_seconds = timed[0].request_started_at - conditioning[0].request_started_at
    timed_seconds = timed[-1].response_completed_at - timed[0].request_started_at
    conditioning_tokens = sum(row.token_numerator for row in conditioning)
    timed_tokens = sum(row.token_numerator for row in timed)
    charged_seconds = conditioning_seconds + timed_seconds
    charged_tokens = conditioning_tokens + timed_tokens
    if not all(value > 0 and math.isfinite(value) for value in (conditioning_seconds, timed_seconds, charged_seconds)):
        raise ResidentPairCrossoverHold(f"resident {role} charged span is invalid")
    return ResidentReadRate(
        role,
        crossover.candidate_lane_digest if candidate else crossover.baseline_lane_digest,
        arm.launch.digest,
        request.session_id,
        batches[0].batch_index,
        batches[-1].batch_index,
        timed[0].batch_index,
        timed[-1].batch_index,
        conditioning_tokens,
        timed_tokens,
        charged_tokens,
        float(conditioning_seconds),
        float(timed_seconds),
        float(charged_seconds),
        float(charged_tokens / charged_seconds),
        tuple(
            TimedWindow(row.batch_index, row.token_numerator, float(row.elapsed_seconds))
            for row in timed
        )
        if crossover.policy.version >= 3
        else (),
    )


def _score(policy: ResidentSpeedPolicy, baselines: list[ResidentReadRate], candidates: list[ResidentReadRate]) -> SpeedupVerdict:
    try:
        return score_speedup(
            [policy.scored_tokens_per_second(row) for row in baselines],
            [policy.scored_tokens_per_second(row) for row in candidates],
            min_margin=policy.min_margin,
            k=policy.noise_multiplier,
            max_noise=policy.max_noise,
        )
    except (CrossoverRuntimeError, RawSpeedEvidenceError) as exc:
        raise ResidentPairCrossoverHold(f"resident speed measurement is unfit: {exc}") from None


def _initial_decision(verdict: SpeedupVerdict, margin: float) -> SpeedStageDecision | None:
    if not verdict.confident:
        return None
    if verdict.speedup <= verdict.required - margin:
        return SpeedStageDecision.FAIL
    if verdict.speedup >= verdict.required + margin:
        return SpeedStageDecision.PASS
    return None


def _regrade(evidence: ResidentPairCrossoverEvidence, plan: ResidentPairCrossoverPlan) -> SpeedupVerdict:
    if type(plan) is not ResidentPairCrossoverPlan:
        raise ResidentPairCrossoverError("resident pair crossover plan is not exact")
    crossover = plan.crossover_plan
    observed_roles = tuple(row.role for row in evidence.rates)
    if crossover.policy.version >= 6:
        expected_roles = resident_speed_roles(
            crossover.policy.version, len(evidence.rates)
        )
        schedule_valid = not evidence.escalated and bool(expected_roles)
    else:
        expected_roles = (
            ("B", "C", "B_prime", "C_prime", "B_double_prime")
            if evidence.escalated
            else ("B", "C", "B_prime")
        )
        schedule_valid = True
    if (
        not schedule_valid
        or evidence.plan_digest != plan.digest
        or evidence.candidate_bundle_digest != plan.candidate_bundle_digest
        or evidence.selected_delta_digest != crossover.selected_delta_digest
        or evidence.workload_digest != plan.workload_digest
        or evidence.prompt_batches_digest != plan.prompt_batches_digest
        or evidence.policy != crossover.policy
        or evidence.pair_binding != plan.pair_binding
        or evidence.orientation != (plan.baseline_pair_lane, plan.candidate_pair_lane)
        or evidence.physical_lane_digests
        != (crossover.baseline_lane_digest, crossover.candidate_lane_digest)
        or evidence.batch_shape != plan.batch_shape
        or evidence.deadline_monotonic_s - evidence.started_monotonic_s
        > crossover.policy.max_stage_seconds + 1e-9
        or observed_roles != expected_roles
        or len(evidence.request_slices) != len(expected_roles)
    ):
        raise ResidentPairCrossoverHold("resident pair evidence names another plan or schedule")
    recomputed: list[ResidentReadRate] = []
    previous_host = evidence.started_monotonic_s
    lane_tail: dict[str, tuple[int, int]] = {}
    request_ids: set[str] = set()
    for role, request in zip(expected_roles, evidence.request_slices, strict=True):
        if (
            request.host_started_at < previous_host
            or request.host_completed_at > evidence.completed_monotonic_s
            or request.request_id in request_ids
            or request.evaluation_id in request_ids
        ):
            raise ResidentPairCrossoverHold("resident pair reads overlap or are reordered")
        request_ids.update((request.request_id, request.evaluation_id))
        previous_host = request.host_completed_at
        row = _rate_from_slice(role, request, plan)
        tail = lane_tail.get(request.lane_id)
        if tail is not None and (
            request.starting_generation != tail[0]
            or row.first_batch_index != tail[1] + 1
        ):
            raise ResidentPairCrossoverHold("resident pair evidence contains interleaved lane work")
        lane_tail[request.lane_id] = (request.ending_generation, row.last_batch_index)
        recomputed.append(row)
    if tuple(recomputed) != evidence.rates:
        raise ResidentPairCrossoverHold("resident pair rates do not regrade from raw slices")
    baselines = [row for row in recomputed if row.role.startswith("B")]
    candidates = [row for row in recomputed if row.role.startswith("C")]
    policy = crossover.policy
    if policy.version >= 6:
        try:
            initial, final, decision = v6_grade(
                policy, recomputed[0], recomputed[1],
                recomputed[2] if len(recomputed) == 3 else None,
            )
        except (CrossoverRuntimeError, RawSpeedEvidenceError) as exc:
            raise ResidentPairCrossoverHold(
                f"resident speed measurement is unfit: {exc}"
            ) from None
    else:
        initial = _score(policy, baselines[:2], candidates[:1])
        try:
            conditioning_failed = policy.conditioning_regression(
                baselines[0], candidates[0]
            )
        except CrossoverRuntimeError as exc:
            raise ResidentPairCrossoverHold(str(exc)) from None
        disposition = SpeedStageDecision.FAIL if conditioning_failed else _initial_decision(
            initial, policy.min_margin
        )
        if disposition is None:
            if not evidence.escalated:
                raise ResidentPairCrossoverHold("borderline resident evidence omitted escalation")
            final = _score(policy, baselines, candidates)
            try:
                conditioning_failed = policy.conditioning_regression(
                    baselines[1], candidates[1]
                )
            except CrossoverRuntimeError as exc:
                raise ResidentPairCrossoverHold(str(exc)) from None
            if conditioning_failed:
                decision = SpeedStageDecision.FAIL
            elif not final.confident:
                raise ResidentPairCrossoverHold("post-escalation resident speed is nonconfident")
            else:
                decision = SpeedStageDecision.PASS if final.passed_speedup else SpeedStageDecision.FAIL
        else:
            if evidence.escalated:
                raise ResidentPairCrossoverHold("clear resident evidence added repeat reads")
            final, decision = initial, disposition
    if (
        evidence.initial_verdict != initial
        or evidence.final_verdict != final
        or evidence.decision is not decision
    ):
        raise ResidentPairCrossoverHold("resident pair speed headline does not regrade")
    return final


def run_resident_pair_crossover(
    plan: ResidentPairCrossoverPlan,
    *,
    pair: ResidentEvaluationPair,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> ResidentPairCrossoverEvidence:
    """Run the policy-sealed serialized pair schedule."""

    if type(plan) is not ResidentPairCrossoverPlan or type(pair) is not ResidentEvaluationPair or not callable(clock):
        raise ResidentPairCrossoverError("resident pair crossover authorities are not exact")
    started = _now(clock)
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline)):
        raise ResidentPairCrossoverError("resident pair crossover deadline is invalid")
    stage_deadline = min(float(deadline), started + plan.crossover_plan.policy.max_stage_seconds)
    if stage_deadline <= started:
        raise ResidentPairCrossoverHold("resident pair speed stage has no wall-clock budget")
    if pair.identities != plan.pair_binding.identities:
        raise ResidentPairCrossoverHold("standing pair sessions differ from the sealed plan")
    lock = _pair_lock(pair)
    if not lock.acquire(timeout=max(0.0, stage_deadline - _now(clock))):
        raise ResidentPairCrossoverHold("resident pair speed stage admission timed out")
    slices: list[ResidentRequestSlice] = []
    rates: list[ResidentReadRate] = []
    history_prefix = pair.request_history
    history_index = len(history_prefix)

    # Version 7 measures both arms through the same swap. Under v6 only the
    # candidate lane swapped, and a bundle audited `aot_invoked:0` — running
    # stock code on both lanes — still read 0.9-2.7% fast in the C role across
    # six runs and both physical orientations (2026-08-16). Position accounted
    # for 0.117% of that (B vs B_prime, same lane) and the physical lane for
    # none of it, because the sign did not flip when the lanes swapped roles.
    # The swap is the only remaining per-role difference, so the baseline lane
    # now takes `swap(None)`: the same registry clear and CUDA-graph recapture,
    # loading nothing. That equalises the recapture, not the candidate's module
    # load, so the inert controls decide whether it is sufficient.
    symmetric_swap = plan.crossover_plan.policy.version >= 7

    def read(role: str) -> None:
        nonlocal history_index, history_prefix
        _within_wall(clock, stage_deadline)
        candidate = role.startswith("C")
        lane = plan.candidate_pair_lane if candidate else plan.baseline_pair_lane
        template = plan.crossover_plan.baseline.session_plan

        def operation(handle: ResidentEvaluationHandle) -> tuple[ResidentBatchEvidence, ...]:
            if candidate:
                handle.swap(plan.candidate_bundle_digest)
            elif symmetric_swap:
                handle.swap(None)
            return tuple(
                handle.execute_batch_with_shape(prompts, shape=plan.batch_shape)
                for prompts in template.prompt_batches
            )

        try:
            result = pair.run_lane(
                lane,
                plan.candidate_bundle_digest,
                operation,
                expected_batch_count=len(template.prompt_batches),
                # A candidate activation declares its own stock restoration; a
                # symmetric baseline swap is already stock and needs none.
                expected_swap_count=2 if candidate else int(symmetric_swap),
                deadline=stage_deadline,
            )
        except ResidentEvaluationPairError as exc:
            raise ResidentPairCrossoverHold(f"resident {role} execution is on HOLD: {exc}") from None
        completed = _within_wall(clock, stage_deadline)
        history = pair.request_history
        if (
            type(result) is not ResidentRequestResult
            or not result.ok
            or type(result.value) is not tuple
            or result.value != result.request_slice.new_batches
            or history[:history_index] != history_prefix
            or history[history_index:] != (result,)
        ):
            raise ResidentPairCrossoverHold(f"resident {role} result or pair serialization is ambiguous")
        history_prefix = history
        history_index = len(history)
        slices.append(result.request_slice)
        rates.append(_rate_from_slice(role, result.request_slice, plan))
        if completed < result.request_slice.host_completed_at:
            raise ResidentPairCrossoverHold(f"resident {role} host clock escaped the stage wall")

    try:
        read("B")
        read("C")
        policy = plan.crossover_plan.policy
        if policy.version >= 6:
            try:
                initial, disposition = speed_grade(
                    policy, [rates[0]], [rates[1]], concluding=False
                )
            except (CrossoverRuntimeError, RawSpeedEvidenceError) as exc:
                raise ResidentPairCrossoverHold(
                    f"resident speed measurement is unfit: {exc}"
                ) from None
            if disposition is not None:
                final, decision = initial, disposition
            else:
                read("B_prime")
                final, decision = speed_grade(
                    policy, [rates[0], rates[2]], [rates[1]], concluding=True
                )
            escalated = False
        else:
            read("B_prime")
            initial = _score(policy, [rates[0], rates[2]], [rates[1]])
            try:
                conditioning_failed = policy.conditioning_regression(rates[0], rates[1])
            except CrossoverRuntimeError as exc:
                raise ResidentPairCrossoverHold(str(exc)) from None
            disposition = SpeedStageDecision.FAIL if conditioning_failed else _initial_decision(
                initial, policy.min_margin
            )
            escalated = disposition is None
            if escalated:
                read("C_prime")
                read("B_double_prime")
                final = _score(policy, [rates[0], rates[2], rates[4]], [rates[1], rates[3]])
                try:
                    conditioning_failed = policy.conditioning_regression(rates[2], rates[3])
                except CrossoverRuntimeError as exc:
                    raise ResidentPairCrossoverHold(str(exc)) from None
                if conditioning_failed:
                    decision = SpeedStageDecision.FAIL
                elif not final.confident:
                    raise ResidentPairCrossoverHold("post-escalation resident speed is nonconfident")
                else:
                    decision = SpeedStageDecision.PASS if final.passed_speedup else SpeedStageDecision.FAIL
            else:
                final, decision = initial, disposition
        completed = _now(clock)
        if completed > stage_deadline:
            raise ResidentPairCrossoverHold("resident pair speed stage timed out")
        evidence = ResidentPairCrossoverEvidence(
            plan.digest,
            plan.candidate_bundle_digest,
            plan.crossover_plan.selected_delta_digest,
            plan.workload_digest,
            plan.prompt_batches_digest,
            policy,
            plan.pair_binding,
            (plan.baseline_pair_lane, plan.candidate_pair_lane),
            (
                plan.crossover_plan.baseline_lane_digest,
                plan.crossover_plan.candidate_lane_digest,
            ),
            plan.batch_shape,
            tuple(slices),
            tuple(rates),
            initial,
            final,
            escalated,
            decision,
            started,
            stage_deadline,
            completed,
        )
        evidence.regrade(plan)
        return evidence
    finally:
        lock.release()


__all__ = [
    "ResidentPairCrossoverError",
    "ResidentPairCrossoverEvidence",
    "ResidentPairCrossoverHold",
    "ResidentPairCrossoverPlan",
    "run_resident_pair_crossover",
]

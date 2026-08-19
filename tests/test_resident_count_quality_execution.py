from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, replace

import pytest

from cacheon.eval.numeric_answer_judge import (
    NumericAnswerJudgeAuthority,
    derive_numeric_answer_prompt_occurrences,
    hidden_task_policy_digest,
    numeric_answer_prompt_plan_digest,
    numeric_hidden_judge_digest,
)
from cacheon.eval.oci_resident_session import (
    ResidentBatchEvidence,
    ResidentBatchShape,
    SwapReceipt,
)
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.qualification import ReferenceManifest
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.resident_count_quality import ResidentCountQualityEnvelope
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountLaneAdmission,
    ResidentCountQualityExecutionError,
    ResidentCountQualityExecutionHold,
    ResidentCountQualityExecutionPlan,
    ResidentCountQualityExecutionResult,
    execute_candidate_count_quality,
    resident_batch_shape_digest,
)
from cacheon.eval.resident_evaluation_pair import ResidentEvaluationPair
from cacheon.eval.resident_pair_binding import (
    ResidentPairLaneBinding,
    ResidentPairRuntimeBinding,
)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class _Limits:
    max_batches: int = 100
    max_swaps: int = 200


class _Session:
    def __init__(self, session_id: str, outputs: dict[str, tuple[int, ...]], barrier=None):
        self.session_id = session_id
        self.outputs = outputs
        self.barrier = barrier
        self.active_generation = 0
        self.active_bundle_digest = None
        self.active_slots = ()
        self.batch_rows = []
        self.swap_receipts = []
        self.closed = False
        self.plan = _Limits()
        self.finish_calls = 0
        self.prompt_batches = []

    def swap(self, bundle_digest):
        started = time.monotonic()
        self.active_generation += 1
        self.active_bundle_digest = bundle_digest
        self.active_slots = () if bundle_digest is None else ("registered.slot",)
        completed = max(time.monotonic(), started + 1e-9)
        row = SwapReceipt(
            len(self.swap_receipts),
            self.active_generation,
            bundle_digest,
            self.active_slots,
            started,
            completed,
        )
        self.swap_receipts.append(row)
        return row

    def execute_batch_with_shape(self, prompts, *, shape, canary=False):
        prompts = tuple(prompts)
        self.prompt_batches.append(prompts)
        if self.barrier is not None:
            self.barrier.wait(2.0)
        started = time.monotonic()
        evidence = BatchEvidence(
            tuple(
                PromptEvidence(
                    self.outputs[prompt],
                    tuple(() for _ in self.outputs[prompt]),
                )
                for prompt in prompts
            )
        )
        completed = max(time.monotonic(), started + 1e-9)
        row = ResidentBatchEvidence(
            len(self.batch_rows),
            _h(f"{self.session_id}-request-{len(self.batch_rows)}")[:32],
            _h(f"{self.session_id}-nonce-{len(self.batch_rows)}")[:32],
            self.active_generation,
            self.active_slots,
            canary,
            started,
            completed,
            evidence.observed_tokens,
            evidence,
        )
        self.batch_rows.append(row)
        return row

    def execute_batch(self, prompts, *, canary=False):
        return self.execute_batch_with_shape(
            prompts,
            shape=ResidentBatchShape(8, 0, 0.0),
            canary=canary,
        )

    def finish(self, *, allow_empty=False):
        self.finish_calls += 1
        self.closed = True
        return self.session_id, allow_empty


class _Factory:
    def __init__(self, session_id, outputs, barrier=None):
        self.session_id = session_id
        self.outputs = outputs
        self.barrier = barrier
        self.calls = 0
        self.sessions = []

    def __call__(self, driver):
        self.calls += 1
        session = _Session(self.session_id, self.outputs, self.barrier)
        self.sessions.append(session)
        return driver(session)


def _reference(binding, tokenizer, workload, profile):
    return ReferenceManifest(
        *(_h(f"{profile}-reference-{index}") for index in range(13)),
        workload,
        tokenizer,
        binding.hidden_corpus_commitment,
        binding.hidden_judge_digest,
        _h(f"{profile}-selection"),
    )


def _fixture(total=64, *, barrier=True, profile="profile-one"):
    prompts = tuple(f"{profile} prompt {index}" for index in range(total))
    prompt_batches = (prompts,)
    gold_ids = tuple((1000 + index,) for index in range(total))
    output_ids = tuple(
        tuple(2000 + index * 8 + token for token in range(8))
        for index in range(total)
    )
    accepted = (tuple((gold,) for gold in gold_ids),)
    corpus = _h(f"{profile}-corpus")
    binding = HiddenJudgeBinding(
        corpus,
        numeric_hidden_judge_digest(
            hidden_corpus_commitment=corpus,
            accepted_token_subsequences=accepted,
        ),
        hidden_task_policy_digest(),
    )
    decoded = {
        **{gold: f"#### {index}" for index, gold in enumerate(gold_ids)},
        **{output: f"answer is {index}" for index, output in enumerate(output_ids)},
    }
    tokenizer = _h(f"{profile}-tokenizer")
    workload = _h(f"{profile}-workload")
    authority = NumericAnswerJudgeAuthority(
        binding,
        tokenizer_digest=tokenizer,
        decoder=decoded.__getitem__,
        accepted_token_subsequences=accepted,
    )
    judge = authority.bind_prompt_plan(
        prompt_batches=prompt_batches,
        workload_digest=workload,
        hidden_tasks_per_prompt=1,
    )
    occurrences = derive_numeric_answer_prompt_occurrences(
        binding,
        prompt_batches=prompt_batches,
        workload_digest=workload,
        hidden_tasks_per_prompt=1,
    )
    session_ids = (
        _h(f"{profile}-session-a")[:32],
        _h(f"{profile}-session-b")[:32],
    )
    lanes = tuple(
        ResidentPairLaneBinding(
            lane_id,
            session_id,
            _h(f"{profile}-stock-launch-{lane_id}"),
            _h(f"{profile}-lane-{lane_id}"),
            _h(f"{profile}-allocation-{lane_id}"),
            _h(f"{profile}-executor-{lane_id}"),
        )
        for lane_id, session_id in zip(("A", "B"), session_ids, strict=True)
    )
    pair_binding = ResidentPairRuntimeBinding(
        _h(f"{profile}-service-epoch"), lanes
    )
    admission = ResidentCountLaneAdmission(
        total // 2,
        total - total // 2,
        256,
        lanes[0].allocation_digest,
        lanes[1].allocation_digest,
    )
    shape = ResidentBatchShape(8, 0, 0.0)
    envelope = ResidentCountQualityEnvelope(
        _reference(binding, tokenizer, workload, profile),
        binding,
        numeric_answer_prompt_plan_digest(occurrences),
        resident_batch_shape_digest(shape),
        admission.digest,
        total,
    )
    plan = ResidentCountQualityExecutionPlan(
        _h(f"{profile}-candidate"),
        envelope,
        prompt_batches,
        tuple(range(total)),
        shape,
        admission,
        pair_binding,
    )
    output_by_prompt = dict(zip(prompts, output_ids, strict=True))
    sync = threading.Barrier(2) if barrier else None
    factory_a = _Factory(session_ids[0], output_by_prompt, sync)
    factory_b = _Factory(session_ids[1], output_by_prompt, sync)
    pair = ResidentEvaluationPair(
        factory_a,
        factory_b,
        start_timeout_s=2.0,
        request_timeout_s=2.0,
        close_timeout_s=2.0,
    )
    pair.start()
    return plan, judge, pair, factory_a, factory_b


def test_execution_returns_raw_evidence_and_independently_regraded_observation() -> None:
    plan, judge, pair, factory_a, factory_b = _fixture()
    execution = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=time.monotonic() + 2.0
    )
    observation = execution.observation

    assert type(execution) is ResidentCountQualityExecutionResult
    assert execution.evidence.execution_plan_digest == plan.digest
    assert observation.execution_evidence_digest == execution.evidence.digest
    assert execution.evidence.regrade(plan, judge) == observation
    for request in execution.evidence.request_slices:
        assert (request.expected_batch_count, request.expected_swap_count) == (1, 2)
        assert [row.bundle_digest for row in request.new_swaps] == [
            plan.candidate_bundle_digest,
            None,
        ]
        assert request.ending_bundle_digest is None and request.ending_slots == ()
    assert observation.role == "candidate"
    assert observation.correct == observation.total == 64
    assert len(factory_a.sessions[0].prompt_batches[0]) == 32
    assert len(factory_b.sessions[0].prompt_batches[0]) == 32
    assert factory_a.calls == factory_b.calls == 1
    assert factory_a.sessions[0].finish_calls == 0
    assert factory_b.sessions[0].finish_calls == 0
    assert len(pair.request_history) == 2
    assert {row.request_slice.session_id for row in pair.request_history} == {
        identity.session_id for identity in plan.pair_binding.identities
    }
    pair.close()


def test_three_candidates_reuse_same_two_sessions_and_close_once() -> None:
    base, judge, pair, factory_a, factory_b = _fixture(6, barrier=False)
    sessions = pair.identities
    for index in range(3):
        plan = ResidentCountQualityExecutionPlan(
            _h(f"candidate-{index}"),
            base.envelope,
            base.prompt_batches,
            base.selected_ordinals,
            base.batch_shape,
            base.admission,
            base.pair_binding,
        )
        assert execute_candidate_count_quality(
            plan, pair=pair, judge=judge, deadline=time.monotonic() + 2.0
        ).observation.correct == 6
        assert pair.identities == sessions
        assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0
    retirement = pair.close()
    assert retirement is not None and pair.close() is retirement
    assert factory_a.calls == factory_b.calls == 1
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 1


def test_plan_binds_selection_shape_admission_and_capacity() -> None:
    plan, _, pair, _, _ = _fixture(4, barrier=False)
    pair.close()
    with pytest.raises(ResidentCountQualityExecutionError, match="selection differs"):
        ResidentCountQualityExecutionPlan(
            plan.candidate_bundle_digest,
            plan.envelope,
            plan.prompt_batches,
            (0, 1),
            plan.batch_shape,
            plan.admission,
            plan.pair_binding,
        )
    with pytest.raises(ResidentCountQualityExecutionError, match="admission capacity"):
        ResidentCountLaneAdmission(2, 3, 2, _h("a"), _h("b"))
    with pytest.raises(ResidentCountQualityExecutionError, match="distinct"):
        ResidentCountLaneAdmission(2, 2, 4, _h("same"), _h("same"))


def test_runtime_binding_mismatch_fails_before_pair_admission() -> None:
    plan, judge, pair, _, _ = _fixture(4, barrier=False)
    foreign_lane = replace(
        plan.pair_binding.lanes[0], session_id=_h("foreign-session")[:32]
    )
    foreign_binding = replace(
        plan.pair_binding,
        lanes=(foreign_lane, plan.pair_binding.lanes[1]),
    )
    foreign_plan = replace(plan, pair_binding=foreign_binding)

    with pytest.raises(ResidentCountQualityExecutionHold, match="sessions differ"):
        execute_candidate_count_quality(
            foreign_plan,
            pair=pair,
            judge=judge,
            deadline=time.monotonic() + 2.0,
        )
    assert pair.request_history == ()
    assert pair.fatal_error is None

    foreign_admission = replace(
        plan.admission, lane_a_allocation_digest=_h("foreign-allocation")
    )
    with pytest.raises(ResidentCountQualityExecutionError, match="allocations differ"):
        replace(plan, admission=foreign_admission)
    assert pair.request_history == ()
    pair.close()


def test_two_synthetic_runtime_profiles_share_the_generic_execution_path() -> None:
    products = []
    for profile in ("profile-one", "profile-two"):
        plan, judge, pair, factory_a, factory_b = _fixture(
            6, barrier=False, profile=profile
        )
        execution = execute_candidate_count_quality(
            plan, pair=pair, judge=judge, deadline=time.monotonic() + 2.0
        )
        products.append((plan, execution))
        assert execution.observation.correct == 6
        assert factory_a.calls == factory_b.calls == 1
        assert tuple(
            row.session_id for row in execution.evidence.pair_binding.lanes
        ) == tuple(row.session_id for row in plan.pair_binding.lanes)
        pair.close()

    first, second = products
    assert first[0].pair_binding.digest != second[0].pair_binding.digest
    assert first[0].envelope.digest != second[0].envelope.digest
    assert first[1].evidence.digest != second[1].evidence.digest


def test_pair_failure_is_hold_and_never_candidate_fail() -> None:
    plan, judge, pair, factory_a, _, = _fixture(4, barrier=False)
    factory_a.sessions[0].outputs.pop(plan.selected_prompts[0])
    with pytest.raises(ResidentCountQualityExecutionHold) as raised:
        execute_candidate_count_quality(
            plan, pair=pair, judge=judge, deadline=time.monotonic() + 2.0
        )
    assert raised.value.decision == "HOLD"
    assert pair.fatal_error is not None
    pair.close()


def test_executor_has_no_stock_callable_or_target_identity() -> None:
    plan, judge, pair, _, _ = _fixture(4, barrier=False)
    execution = execute_candidate_count_quality(
        plan, pair=pair, judge=judge, deadline=time.monotonic() + 2.0
    )
    observation = execution.observation
    assert observation.role == "candidate"
    text = plan.to_dict()
    assert "stock" not in repr(text).lower()
    for forbidden in ("msa", "arnorm", "all_reduce"):
        assert forbidden not in repr(text).lower()
    pair.close()


@pytest.mark.parametrize("tamper", ("token_count", "active_slots"))
def test_malformed_lane_batch_is_hold(tamper: str) -> None:
    plan, judge, pair, factory_a, _ = _fixture(4, barrier=False)
    session = factory_a.sessions[0]
    original = session.execute_batch_with_shape

    def execute_tampered(prompts, *, shape, canary=False):
        row = original(prompts, shape=shape, canary=canary)
        if tamper == "token_count":
            changed = replace(row, token_numerator=row.token_numerator - 1)
        else:
            changed = replace(row, active_slots=("wrong.slot",))
        session.batch_rows[-1] = changed
        return changed

    session.execute_batch_with_shape = execute_tampered
    with pytest.raises(ResidentCountQualityExecutionHold) as raised:
        execute_candidate_count_quality(
            plan, pair=pair, judge=judge, deadline=time.monotonic() + 2.0
        )
    assert raised.value.decision == "HOLD"
    pair.close()

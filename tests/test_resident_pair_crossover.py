from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cacheon.eval.crossover_runtime import ResidentSpeedPolicy, SpeedStageDecision
from cacheon.eval.oci_resident_session import (
    ResidentBatchEvidence,
    ResidentBatchShape,
    SwapReceipt,
)
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationPair,
    ResidentEvaluationPairError,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverHold,
    ResidentPairCrossoverPlan,
    run_resident_pair_crossover,
)
from tests.test_crossover_runtime import _rig


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Clock:
    def __init__(self) -> None:
        self.value = 1.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.value

    def span(self, seconds: float) -> tuple[float, float]:
        with self.lock:
            started = self.value
            self.value += seconds
            return started, self.value


class _Activity:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.overlap = False

    def enter(self) -> None:
        with self.lock:
            self.overlap |= self.active != 0
            self.active += 1

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _Session:
    def __init__(
        self,
        session_id: str,
        template,
        durations: tuple[float | tuple[float, ...], ...],
        clock: _Clock,
        activity: _Activity,
    ) -> None:
        self.session_id = session_id
        self.template = template
        self.durations = durations
        self.clock = clock
        self.activity = activity
        self.active_generation = 0
        self.active_bundle_digest = None
        self.active_slots = ()
        self.batch_rows = []
        self.swap_receipts = []
        self.closed = False
        self.plan = SimpleNamespace(max_batches=10_000, max_swaps=10_000)
        self.finish_calls = 0

    def swap(self, bundle_digest):
        started, completed = self.clock.span(0.01)
        self.active_generation += 1
        self.active_bundle_digest = bundle_digest
        self.active_slots = () if bundle_digest is None else ("registered.slot",)
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
        block = len(self.template.prompt_batches)
        index = len(self.batch_rows)
        read_index, local_index = divmod(index, block)
        assert shape == ResidentBatchShape(
            self.template.max_new_tokens,
            self.template.top_logprobs_num,
            self.template.temperature,
        )
        assert prompts == self.template.prompt_batches[local_index]
        selected = self.durations[min(read_index, len(self.durations) - 1)]
        windows = selected if isinstance(selected, tuple) else (selected,)
        duration = (
            0.1
            if local_index < self.template.warmup_count
            else windows[
                min(local_index - self.template.warmup_count, len(windows) - 1)
            ]
        )
        self.activity.enter()
        try:
            time.sleep(0.0001)
            started, completed = self.clock.span(duration)
        finally:
            self.activity.leave()
        positions = tuple(
            tuple((-0.5 - rank, rank) for rank in range(shape.top_logprobs_num))
            for _ in range(shape.max_new_tokens)
        )
        evidence = BatchEvidence(
            tuple(
                PromptEvidence(tuple(range(shape.max_new_tokens)), positions)
                for _ in prompts
            )
        )
        row = ResidentBatchEvidence(
            index,
            f"{index + 1:032x}",
            f"{index + 10001:032x}",
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
            shape=ResidentBatchShape(
                self.template.max_new_tokens,
                self.template.top_logprobs_num,
                self.template.temperature,
            ),
            canary=canary,
        )

    def finish(self, *, allow_empty=False):
        self.finish_calls += 1
        self.closed = True
        return self.session_id, allow_empty


class _Factory:
    def __init__(self, session_id, template, durations, clock, activity):
        self.session_id = session_id
        self.template = template
        self.durations = durations
        self.clock = clock
        self.activity = activity
        self.sessions = []

    def __call__(self, driver):
        session = _Session(
            self.session_id,
            self.template,
            self.durations,
            self.clock,
            self.activity,
        )
        self.sessions.append(session)
        return driver(session)


@pytest.fixture
def cleanup_pairs():
    pairs = []
    yield pairs
    for pair in pairs:
        pair.close()


def _setup(
    tmp_path,
    cleanup_pairs,
    *,
    baseline=(1.0,),
    candidate=(0.8,),
    policy=None,
    timed_batches=1,
):
    crossover, *_ = _rig(
        tmp_path,
        (1.0, 1.0),
        policy=policy,
        timed_batches=timed_batches,
    )
    clock, activity = _Clock(), _Activity()
    template = crossover.baseline.session_plan
    factory_a = _Factory("a" * 32, template, tuple(baseline), clock, activity)
    factory_b = _Factory("b" * 32, template, tuple(candidate), clock, activity)
    pair = ResidentEvaluationPair(
        factory_a,
        factory_b,
        start_timeout_s=2.0,
        request_timeout_s=2.0,
        close_timeout_s=2.0,
        clock=clock,
    )
    pair.start()
    cleanup_pairs.append(pair)
    plan = ResidentPairCrossoverPlan(
        _h("bundle"), crossover, pair.identities, "A", "B"
    )
    return plan, pair, clock, activity, factory_a, factory_b


@pytest.mark.parametrize(
    ("candidate_duration", "decision"),
    ((0.75, SpeedStageDecision.PASS), (1.25, SpeedStageDecision.FAIL)),
)
def test_exact_serial_b_c_b_prime_clear_decisions(
    tmp_path, cleanup_pairs, candidate_duration, decision
):
    plan, pair, clock, activity, factory_a, factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=(1.0, 1.0),
        candidate=(candidate_duration,),
    )
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    assert evidence.decision is decision
    assert not evidence.escalated
    assert tuple(row.role for row in evidence.rates) == ("B", "C", "B_prime")
    assert tuple(row.lane_id for row in evidence.request_slices) == ("A", "B", "A")
    assert tuple(len(row.new_swaps) for row in evidence.request_slices) == (0, 2, 0)
    assert all(
        row.ending_bundle_digest is None and not row.ending_slots
        for row in evidence.request_slices
    )
    assert evidence.regrade(plan) == evidence.final_verdict
    assert not activity.overlap
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0


def _borderline_policy(*, version=1):
    return ResidentSpeedPolicy(
        60,
        0.005,
        0.1,
        0.02 if version == 3 else 0.1,
        "8" * 64,
        "9" * 64,
        version=version,
        min_windows=3 if version == 3 else 0,
        max_window_scatter=0.01 if version == 3 else 0.0,
        max_conditioning_slowdown=1.5 if version == 3 else 0.0,
    )


def test_borderline_escalates_exactly_and_settles_pass(tmp_path, cleanup_pairs):
    plan, pair, clock, _, factory_a, factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=(1.0, 1.0, 1.0),
        candidate=(0.995, 0.98),
        policy=_borderline_policy(),
    )
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    assert evidence.escalated and evidence.decision is SpeedStageDecision.PASS
    assert tuple(row.role for row in evidence.rates) == (
        "B",
        "C",
        "B_prime",
        "C_prime",
        "B_double_prime",
    )
    assert tuple(row.lane_id for row in evidence.request_slices) == (
        "A",
        "B",
        "A",
        "B",
        "A",
    )
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0


@pytest.mark.parametrize("tamper", ("candidate_first", "cross_lane", "host_order"))
def test_independent_regrade_rejects_schedule_tampering(
    tmp_path, cleanup_pairs, tamper
):
    plan, pair, clock, *_ = _setup(tmp_path, cleanup_pairs)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    slices, rates = evidence.request_slices, evidence.rates
    if tamper == "candidate_first":
        changed = replace(
            evidence,
            request_slices=(slices[1], slices[0], slices[2]),
            rates=(rates[1], rates[0], rates[2]),
        )
    elif tamper == "cross_lane":
        changed_slice = replace(slices[1], lane_id="A")
        changed = replace(evidence, request_slices=(slices[0], changed_slice, slices[2]))
    else:
        changed_slice = replace(
            slices[1], host_started_at=slices[0].host_started_at
        )
        changed = replace(evidence, request_slices=(slices[0], changed_slice, slices[2]))
    with pytest.raises(ResidentPairCrossoverHold):
        changed.regrade(plan)


def test_incomplete_and_scattered_evidence_are_hold(tmp_path, cleanup_pairs):
    plan, pair, clock, *_ = _setup(tmp_path / "incomplete", cleanup_pairs)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    request = evidence.request_slices[0]
    bad_batch = replace(
        request.new_batches[0], token_numerator=request.new_batches[0].token_numerator - 1
    )
    bad_request = replace(
        request, new_batches=(bad_batch, *request.new_batches[1:])
    )
    changed = replace(
        evidence, request_slices=(bad_request, *evidence.request_slices[1:])
    )
    with pytest.raises(ResidentPairCrossoverHold):
        changed.regrade(plan)

    noisy_plan, noisy_pair, noisy_clock, *_ = _setup(
        tmp_path / "scatter",
        cleanup_pairs,
        baseline=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0)),
        candidate=((1.0, 1.0, 1.0),),
        policy=_borderline_policy(version=3),
        timed_batches=3,
    )
    with pytest.raises(ResidentPairCrossoverHold, match="unfit"):
        run_resident_pair_crossover(
            noisy_plan,
            pair=noisy_pair,
            deadline=noisy_clock() + 120.0,
            clock=noisy_clock,
        )


def test_post_escalation_nonconfidence_is_hold(tmp_path, cleanup_pairs):
    plan, pair, clock, *_ = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=(1.0, 1.0, 1.0),
        candidate=(0.995, 0.5),
        policy=_borderline_policy(),
    )
    with pytest.raises(ResidentPairCrossoverHold, match="post-escalation"):
        run_resident_pair_crossover(
            plan, pair=pair, deadline=clock() + 120.0, clock=clock
        )


def test_three_bundle_digests_reuse_sessions_without_finish(tmp_path, cleanup_pairs):
    base, pair, clock, _, factory_a, factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=(1.0,) * 6,
        candidate=(0.75,) * 3,
    )
    identities = pair.identities
    for index in range(3):
        plan = replace(base, candidate_bundle_digest=_h(f"bundle-{index}"))
        evidence = run_resident_pair_crossover(
            plan, pair=pair, deadline=clock() + 120.0, clock=clock
        )
        assert evidence.pair_identities == identities == pair.identities
    assert len({row.request_slice.bundle_digest for row in pair.request_history}) == 3
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0


def test_one_absolute_stage_deadline_binds_every_resident_read(
    tmp_path, cleanup_pairs, monkeypatch
):
    plan, pair, clock, *_ = _setup(tmp_path, cleanup_pairs)
    observed = []
    original = ResidentEvaluationPair.run_lane

    def bounded(self, *args, **kwargs):
        observed.append(kwargs.get("deadline"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ResidentEvaluationPair, "run_lane", bounded)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    assert observed == [evidence.deadline_monotonic_s] * 3

    called = False

    def operation(_handle):
        nonlocal called
        called = True

    with pytest.raises(ResidentEvaluationPairError, match="expired"):
        original(
            pair,
            "A",
            _h("expired"),
            operation,
            expected_batch_count=0,
            expected_swap_count=0,
            deadline=clock(),
        )
    assert not called


def test_concurrent_stages_are_pair_global_and_whole_stage_serialized(
    tmp_path, cleanup_pairs
):
    base, pair, clock, activity, factory_a, factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=(1.0,) * 50,
        candidate=(0.75,) * 30,
    )
    for round_index in range(10):
        barrier = threading.Barrier(2)
        outcomes, errors = [], []

        def worker(index):
            try:
                plan = replace(
                    base,
                    candidate_bundle_digest=_h(f"round-{round_index}-{index}"),
                )
                barrier.wait(2.0)
                outcomes.append(
                    run_resident_pair_crossover(
                        plan, pair=pair, deadline=clock() + 120.0, clock=clock
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)
            assert not thread.is_alive()
        assert not errors and len(outcomes) == 2
        tail = pair.request_history[-6:]
        bundles = [row.request_slice.bundle_digest for row in tail]
        assert bundles[:3] == [bundles[0]] * 3
        assert bundles[3:] == [bundles[3]] * 3
        assert bundles[0] != bundles[3]
        assert all(
            tuple(rate.role for rate in evidence.rates) == ("B", "C", "B_prime")
            for evidence in outcomes
        )
    assert not activity.overlap
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0


def test_production_has_no_target_literals():
    import cacheon.eval.resident_pair_crossover as module

    source = open(module.__file__, encoding="utf-8").read().lower()
    for forbidden in ("arnorm", "msa", "all_reduce"):
        assert forbidden not in source

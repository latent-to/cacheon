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
from cacheon.eval.resident_execution_evidence import (
    UNOBSERVED_EVIDENCE,
    ResidentExecutionEvidence,
)
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationPair,
    ResidentEvaluationPairError,
)
from cacheon.eval.resident_pair_binding import (
    ResidentPairLaneBinding,
    ResidentPairRuntimeBinding,
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
        executed_ranks: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.template = template
        self.durations = durations
        self.clock = clock
        self.activity = activity
        self.active_generation = 0
        self.applied_generation = -1
        self.expected_ranks = 4
        # None means "an honest candidate": every rank executed. A test that
        # wants an inert bundle sets this to the number of ranks that did.
        self.executed_ranks = executed_ranks
        self.active_bundle_digest = None
        self.active_slots = ()
        self.batch_rows = []
        self.swap_receipts = []
        self.closed = False
        self.plan = SimpleNamespace(max_batches=10_000, max_swaps=10_000)
        self.finish_calls = 0

    def swap(self, bundle_digest):
        started, completed = self.clock.span(0.01)
        # A swap closes the generation that was live and reports ITS execution
        # evidence, exactly as the seam does: the closing scope is final only
        # once the lane has swapped away from it. A generation that had a bundle
        # active is modelled as having executed on every rank, which is what an
        # honest candidate produces; `executed_ranks` lets a test say otherwise.
        closing = self.applied_generation
        if closing < 0:
            execution = UNOBSERVED_EVIDENCE
        else:
            execution = ResidentExecutionEvidence(
                closing,
                self.expected_ranks
                if self.active_bundle_digest is not None
                else 0,
            )
            if self.executed_ranks is not None and self.active_bundle_digest:
                execution = ResidentExecutionEvidence(closing, self.executed_ranks)
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
            execution,
            self.expected_ranks,
        )
        self.applied_generation = self.active_generation
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
    def __init__(
        self, session_id, template, durations, clock, activity, executed_ranks=None
    ):
        self.session_id = session_id
        self.template = template
        self.durations = durations
        self.clock = clock
        self.activity = activity
        self.executed_ranks = executed_ranks
        self.sessions = []

    def __call__(self, driver):
        session = _Session(
            self.session_id,
            self.template,
            self.durations,
            self.clock,
            self.activity,
            executed_ranks=self.executed_ranks,
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
    baseline_pair_lane="A",
    candidate_executed_ranks=None,
):
    crossover, *_ = _rig(
        tmp_path,
        (1.0, 1.0),
        policy=policy,
        timed_batches=timed_batches,
    )
    clock, activity = _Clock(), _Activity()
    template = crossover.baseline.session_plan
    if baseline_pair_lane not in ("A", "B"):
        raise ValueError("baseline pair lane must be A or B")
    candidate_pair_lane = "B" if baseline_pair_lane == "A" else "A"
    factory_a = _Factory(
        "a" * 32,
        template,
        tuple(baseline if baseline_pair_lane == "A" else candidate),
        clock,
        activity,
        executed_ranks=(
            None if baseline_pair_lane == "A" else candidate_executed_ranks
        ),
    )
    factory_b = _Factory(
        "b" * 32,
        template,
        tuple(baseline if baseline_pair_lane == "B" else candidate),
        clock,
        activity,
        executed_ranks=(
            None if baseline_pair_lane == "B" else candidate_executed_ranks
        ),
    )
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
    binding = ResidentPairRuntimeBinding(
        _h("service-epoch"),
        tuple(
            ResidentPairLaneBinding(
                identity.lane_id,
                identity.session_id,
                _h(f"stock-launch-{identity.lane_id}"),
                (
                    crossover.baseline_lane_digest
                    if identity.lane_id == baseline_pair_lane
                    else crossover.candidate_lane_digest
                ),
                _h(f"allocation-{identity.lane_id}"),
                _h(f"executor-namespace-{identity.lane_id}"),
            )
            for identity in pair.identities
        ),
    )
    plan = ResidentPairCrossoverPlan(
        _h("bundle"),
        crossover,
        binding,
        baseline_pair_lane,
        candidate_pair_lane,
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
    windowed = version >= 3
    return ResidentSpeedPolicy(
        60,
        0.005,
        0.1,
        0.002 if version >= 6 else 0.02 if version >= 2 else 0.1,
        "8" * 64,
        "9" * 64,
        version=version,
        min_windows=3 if windowed else 0,
        max_window_scatter=0.01 if windowed else 0.0,
        max_conditioning_slowdown=1.5 if windowed else 0.0,
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


@pytest.mark.parametrize(
    ("baseline_durations", "candidate_duration", "roles", "decision"),
    (
        (((1.0,) * 3,) * 2, 1.05, ("B", "C"), SpeedStageDecision.FAIL),
        (((1.0,) * 3,) * 2, 0.99, ("B", "C"), SpeedStageDecision.PASS),
        (
            ((1.0,) * 3, (1.0 / 1.002,) * 3),
            1.0 / 1.006,
            ("B", "C", "B_prime"),
            SpeedStageDecision.FAIL,
        ),
        (
            ((1.0,) * 3,) * 2,
            1.0 / 1.006,
            ("B", "C", "B_prime"),
            SpeedStageDecision.PASS,
        ),
    ),
)
def test_v6_runs_two_leg_clear_fail_or_three_leg_terminal(
    tmp_path, cleanup_pairs, baseline_durations, candidate_duration, roles, decision
):
    plan, pair, clock, activity, factory_a, factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=baseline_durations,
        candidate=((candidate_duration,) * 3,),
        policy=_borderline_policy(version=6),
        timed_batches=3,
    )

    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    assert evidence.decision is decision
    assert not evidence.escalated
    assert tuple(row.role for row in evidence.rates) == roles
    expected_lanes = tuple("B" if role == "C" else "A" for role in roles)
    assert tuple(row.lane_id for row in evidence.request_slices) == expected_lanes
    assert tuple(
        row.request_slice for row in pair.request_history
    ) == evidence.request_slices
    assert len(factory_a.sessions[0].batch_rows) == (
        len(plan.crossover_plan.baseline.session_plan.prompt_batches)
        * sum(role.startswith("B") for role in roles)
    )
    assert len(factory_b.sessions[0].batch_rows) == (
        len(plan.crossover_plan.baseline.session_plan.prompt_batches)
        * sum(role.startswith("C") for role in roles)
    )
    assert evidence.regrade(plan) == evidence.final_verdict
    assert not activity.overlap
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0


@pytest.mark.parametrize(
    ("candidate_duration", "decision"),
    ((1.05, SpeedStageDecision.FAIL), (0.99, SpeedStageDecision.PASS)),
)
def test_v7_swaps_both_arms_so_neither_role_is_measured_unswapped(
    tmp_path, cleanup_pairs, candidate_duration, decision
):
    """Under v7 the baseline takes a stock-to-stock swap of its own.

    v6 swapped only the candidate lane, which handed the candidate role a
    measured advantage on identical work: a bundle audited ``aot_invoked:0``
    read 0.9-2.7% fast in the C role across six runs and both physical
    orientations. Position explained 0.117% of it and the physical lane none,
    so the swap was the remaining asymmetry. Both arms must now traverse one.
    """

    plan, pair, clock, activity, factory_a, factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=((1.0,) * 3,) * 2,
        candidate=((candidate_duration,) * 3,),
        policy=_borderline_policy(version=7),
        timed_batches=3,
    )

    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    assert evidence.decision is decision
    assert tuple(row.role for row in evidence.rates) == ("B", "C")
    baseline_slice, candidate_slice = evidence.request_slices
    # The baseline is swapped exactly once, and that swap must be stock: no
    # bundle and no registered slots, or the "baseline" ran a candidate.
    assert baseline_slice.expected_swap_count == 1
    assert len(baseline_slice.new_swaps) == 1
    (restock,) = baseline_slice.new_swaps
    assert restock.bundle_digest is None
    assert not restock.slots
    # The candidate keeps activation plus its declared stock restoration.
    assert candidate_slice.expected_swap_count == 2
    assert len(candidate_slice.new_swaps) == 2
    # Neither arm ends holding a bundle.
    assert baseline_slice.ending_bundle_digest is None
    assert candidate_slice.ending_bundle_digest is None
    assert evidence.regrade(plan) == evidence.final_verdict
    assert not activity.overlap


@pytest.mark.parametrize(
    "executed_ranks, why",
    [
        (0, "inert bundle: registered its slot and never dispatched it"),
        (3, "partial: one rank of the group did not execute"),
    ],
)
def test_a_candidate_that_did_not_execute_cannot_win(
    tmp_path, cleanup_pairs, executed_ranks, why
):
    """A winning duration is not a win if the kernel never ran.

    This is the defect that produced a settlement candidate audited
    ``aot_invoked:0`` on all four ranks: the resident lane is launched stock, so
    the one-shot driver's execution gate never applied to it, and registration
    was the only thing the crossover could see. Registration is not execution.

    The candidate here is given a decisively winning duration precisely so the
    hold cannot be attributed to its speed.
    """

    plan, pair, clock, activity, factory_a, factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=((1.0,) * 3,) * 2,
        candidate=((0.5,) * 3,),
        policy=_borderline_policy(version=7),
        timed_batches=3,
        candidate_executed_ranks=executed_ranks,
    )

    with pytest.raises(ResidentPairCrossoverHold) as caught:
        run_resident_pair_crossover(
            plan, pair=pair, deadline=clock() + 120.0, clock=clock
        )
    assert "no proof its kernel executed" in str(caught.value), why


def test_v6_leaves_the_baseline_unswapped(tmp_path, cleanup_pairs):
    """v6 evidence stays verifiable at its own version after v7 lands.

    Version participates in policy equality, so a v6 record must keep grading
    under the procedure it was taken with rather than v7's.
    """

    plan, pair, clock, _activity, _factory_a, _factory_b = _setup(
        tmp_path,
        cleanup_pairs,
        baseline=((1.0,) * 3,) * 2,
        candidate=((1.05,) * 3,),
        policy=_borderline_policy(version=6),
        timed_batches=3,
    )

    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    baseline_slice, candidate_slice = evidence.request_slices
    assert baseline_slice.expected_swap_count == 0
    assert not baseline_slice.new_swaps
    assert candidate_slice.expected_swap_count == 2
    assert evidence.regrade(plan) == evidence.final_verdict


def test_v6_regrade_rejects_missing_b_prime_and_bad_windows(
    tmp_path, cleanup_pairs
):
    plan, pair, clock, *_ = _setup(
        tmp_path / "missing",
        cleanup_pairs,
        baseline=((1.0,) * 3,) * 2,
        # 1.006 is inside v6's sealed uncertainty band, so the
        # production runner appends B-prime. Removing that retained leg below
        # must fail independent regrade.
        candidate=((1.0 / 1.006,) * 3,),
        policy=_borderline_policy(version=6),
        timed_batches=3,
    )
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    with pytest.raises(ResidentPairCrossoverHold, match="omitted required B-prime"):
        replace(
            evidence,
            request_slices=evidence.request_slices[:2],
            rates=evidence.rates[:2],
        ).regrade(plan)

    bad_plan, bad_pair, bad_clock, *_ = _setup(
        tmp_path / "windows",
        cleanup_pairs,
        policy=_borderline_policy(version=6),
        timed_batches=2,
    )
    with pytest.raises(ResidentPairCrossoverHold, match="required timed windows"):
        run_resident_pair_crossover(
            bad_plan,
            pair=bad_pair,
            deadline=bad_clock() + 120.0,
            clock=bad_clock,
        )


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

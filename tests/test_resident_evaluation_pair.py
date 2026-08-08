"""CPU contracts for the two-lane resident evaluation coordinator."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from cacheon.eval.oci_resident_session import ResidentBatchEvidence, SwapReceipt
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationEpochFatal,
    ResidentLaneRequest,
    ResidentEvaluationPair,
    ResidentEvaluationPairError,
    ResidentEvaluationPairFailed,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
SLOT = "attention.msa_prefill_block_score"


def _batch_evidence() -> BatchEvidence:
    return BatchEvidence((PromptEvidence((1,), (((-0.5, 1),),)),))


class FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.active_generation = 0
        self.active_bundle_digest: str | None = None
        self.active_slots: tuple[str, ...] = ()
        self.batch_rows: list[ResidentBatchEvidence] = []
        self.swap_receipts: list[SwapReceipt] = []
        self.plan = SimpleNamespace(max_batches=100, max_swaps=100)
        self.finish_calls = 0
        self.finish_allow_empty: list[bool] = []
        self.closed = False
        self.fail_stock_restore = False
        self.terminal_batch_error: BaseException | None = None
        self._host_time = 1.0

    def swap(self, bundle_digest: str | None) -> SwapReceipt:
        if bundle_digest is None and self.fail_stock_restore:
            raise RuntimeError("stock restore\nfailed " + "x" * 800)
        self.active_generation += 1
        self.active_bundle_digest = bundle_digest
        self.active_slots = () if bundle_digest is None else (SLOT,)
        started = self._host_time
        self._host_time += 1.0
        row = SwapReceipt(
            len(self.swap_receipts),
            self.active_generation,
            bundle_digest,
            () if bundle_digest is None else (SLOT,),
            started,
            self._host_time,
        )
        self.swap_receipts.append(row)
        return row

    def execute_batch(self, prompts, *, canary: bool = False):
        assert tuple(prompts)
        assert not canary or self.active_bundle_digest is None
        if self.terminal_batch_error is not None:
            self.closed = True
            raise self.terminal_batch_error
        started = self._host_time
        self._host_time += 1.0
        index = len(self.batch_rows)
        row = ResidentBatchEvidence(
            index,
            f"{index + 1:032x}",
            f"{index + 101:032x}",
            self.active_generation,
            self.active_slots,
            canary,
            started,
            self._host_time,
            1,
            _batch_evidence(),
        )
        self.batch_rows.append(row)
        return row

    def finish(self, *, allow_empty: bool = False):
        self.finish_calls += 1
        self.finish_allow_empty.append(allow_empty)
        if not self.batch_rows and not allow_empty:
            raise RuntimeError("resident session executed no batches")
        self.closed = True
        return ("finished", self.session_id)


class FakeFactory:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.calls = 0
        self.sessions: list[FakeSession] = []

    def __call__(self, driver):
        self.calls += 1
        session = FakeSession(self.session_id)
        self.sessions.append(session)
        return driver(session)


def _pair():
    factory_a = FakeFactory("a" * 32)
    factory_b = FakeFactory("b" * 32)
    pair = ResidentEvaluationPair(
        factory_a,
        factory_b,
        start_timeout_s=5.0,
        request_timeout_s=5.0,
        close_timeout_s=5.0,
    )
    pair.start()
    return pair, factory_a, factory_b


def test_sessions_survive_requests_and_slices_bind_exact_deltas() -> None:
    pair, factory_a, factory_b = _pair()

    def candidate(handle):
        handle.swap(DIGEST_A)
        return handle.execute_batch(("prompt",))

    first = pair.run_lane(
        "A",
        DIGEST_A,
        candidate,
        expected_batch_count=1,
        expected_swap_count=2,
    )
    second = pair.run_lane(
        "A",
        DIGEST_B,
        lambda handle: (handle.swap(DIGEST_B), handle.execute_batch(("prompt",))),
        expected_batch_count=1,
        expected_swap_count=2,
    )

    assert first.ok and second.ok
    assert first.request_slice.session_id == second.request_slice.session_id == "a" * 32
    assert (
        first.request_slice.starting_generation,
        first.request_slice.ending_generation,
    ) == (0, 2)
    assert (
        second.request_slice.starting_generation,
        second.request_slice.ending_generation,
    ) == (2, 4)
    assert [row.batch_index for row in first.request_slice.new_batches] == [0]
    assert [row.batch_index for row in second.request_slice.new_batches] == [1]
    assert [row.swap_index for row in first.request_slice.new_swaps] == [0, 1]
    assert [row.swap_index for row in second.request_slice.new_swaps] == [2, 3]
    assert first.request_slice.new_swaps[-1].bundle_digest is None
    assert second.request_slice.new_swaps[-1].bundle_digest is None
    assert first.request_slice.ending_bundle_digest is None
    assert first.request_slice.ending_slots == ()
    assert first.request_slice.bundle_digest == DIGEST_A
    assert second.request_slice.bundle_digest == DIGEST_B
    assert first.request_slice.host_completed_at >= first.request_slice.host_started_at
    assert factory_a.calls == factory_b.calls == 1
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0

    retirement = pair.close()
    assert pair.close() is retirement
    assert retirement is pair.retirement_evidence
    assert retirement is not None
    assert retirement.lane_a.lifetime_evidence == ("finished", "a" * 32)
    assert retirement.lane_b.lifetime_evidence == ("finished", "b" * 32)
    assert retirement.request_history == pair.request_history == (first, second)
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 1
    assert factory_a.sessions[0].finish_allow_empty == [True]
    assert factory_b.sessions[0].finish_allow_empty == [True]


def test_lane_admissions_are_pair_global_and_never_overlap() -> None:
    pair, _, _ = _pair()
    entered_a, entered_b, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    outcomes = []

    def first(handle):
        entered_a.set()
        assert release.wait(5.0)
        return handle.execute_batch(("prompt",), canary=True)

    def second(handle):
        entered_b.set()
        return handle.execute_batch(("prompt",), canary=True)

    threads = [
        threading.Thread(
            target=lambda: outcomes.append(
                pair.run_lane(
                    "A",
                    DIGEST_A,
                    first,
                    expected_batch_count=1,
                    expected_swap_count=0,
                )
            )
        ),
        threading.Thread(
            target=lambda: outcomes.append(
                pair.run_lane(
                    "B",
                    DIGEST_A,
                    second,
                    expected_batch_count=1,
                    expected_swap_count=0,
                )
            )
        ),
    ]
    threads[0].start()
    assert entered_a.wait(2.0)
    threads[1].start()
    assert not entered_b.wait(0.1)
    release.set()
    for thread in threads:
        thread.join(5.0)
        assert not thread.is_alive()
    assert entered_b.is_set()
    assert len(outcomes) == 2 and all(result.ok for result in outcomes)
    pair.close()


def test_run_lanes_overlaps_owners_enqueues_both_then_returns_a_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, _, _ = _pair()
    barrier = threading.Barrier(2)
    b_completed = threading.Event()
    observations: list[tuple[str, str, object, object]] = []
    observation_lock = threading.Lock()

    for lane_id in ("A", "B"):
        lane_queue = pair._lanes[lane_id].work  # type: ignore[attr-defined]
        original_put = lane_queue.put

        def observed_put(
            item, block=True, timeout=None, *, lane_id=lane_id, original=original_put
        ):
            if hasattr(item, "evaluation_id"):
                with observation_lock:
                    observations.append(
                        ("enqueue", lane_id, item.evaluation_id, item.deadline)
                    )
            return original(item, block=block, timeout=timeout)

        monkeypatch.setattr(lane_queue, "put", observed_put)

    original_await = pair._await  # type: ignore[attr-defined]

    def observed_await(work):
        with observation_lock:
            observations.append(
                ("await", "?", work.evaluation_id, work.deadline)
            )
        return original_await(work)

    monkeypatch.setattr(pair, "_await", observed_await)
    original_complete = pair._complete_result  # type: ignore[attr-defined]

    def observed_complete(work, result):
        original_complete(work, result)
        if result.request_slice.lane_id == "B":
            b_completed.set()

    monkeypatch.setattr(pair, "_complete_result", observed_complete)

    def operation_a(handle):
        barrier.wait(2.0)
        assert b_completed.wait(2.0)
        return "A", handle.identity

    def operation_b(handle):
        barrier.wait(2.0)
        return "B", handle.identity

    result_a, result_b = pair.run_lanes(
        ResidentLaneRequest(DIGEST_A, operation_a, 0, 0),
        ResidentLaneRequest(DIGEST_B, operation_b, 0, 0),
    )

    assert result_a.ok and result_b.ok
    assert result_a.value == ("A", pair.identities[0])
    assert result_b.value == ("B", pair.identities[1])
    assert [item[:2] for item in observations[:4]] == [
        ("enqueue", "A"),
        ("enqueue", "B"),
        ("await", "?"),
        ("await", "?"),
    ]
    enqueued = observations[:2]
    assert enqueued[0][2:] == enqueued[1][2:]
    assert (
        result_a.request_slice.evaluation_id
        == result_b.request_slice.evaluation_id
        == enqueued[0][2]
    )
    assert result_a.request_slice.request_id != result_b.request_slice.request_id
    pair.close()


def test_run_lanes_validates_both_before_admitting_either() -> None:
    pair, _, _ = _pair()
    invoked = threading.Event()

    with pytest.raises(ResidentEvaluationPairError, match="count is invalid"):
        pair.run_lanes(
            ResidentLaneRequest(DIGEST_A, lambda _handle: invoked.set(), 0, 0),
            ResidentLaneRequest(DIGEST_B, lambda _handle: None, -1, 0),
        )

    assert not invoked.is_set()
    assert pair.request_history == ()
    assert pair.fatal_error is None
    pair.close()


def test_run_lanes_one_half_failure_latches_without_half_success() -> None:
    pair, _, _ = _pair()

    def fail(_handle):
        raise ValueError("lane B failed")

    with pytest.raises(ResidentEvaluationPairFailed, match="lane B failed"):
        pair.run_lanes(
            ResidentLaneRequest(DIGEST_A, lambda _handle: "unpublished", 0, 0),
            ResidentLaneRequest(DIGEST_B, fail, 0, 0),
        )

    assert pair.fatal_error is not None
    assert len(pair.request_history) == 2
    assert sum(result.ok for result in pair.request_history) == 1
    with pytest.raises(ResidentEvaluationPairFailed, match="lane B failed"):
        pair.run_lane(
            "A",
            DIGEST_A,
            lambda _handle: None,
            expected_batch_count=0,
            expected_swap_count=0,
        )
    pair.close()


def test_run_lanes_partial_enqueue_latches_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, _, _ = _pair()
    invoked = threading.Event()
    lane_b_queue = pair._lanes["B"].work  # type: ignore[attr-defined]
    original_put = lane_b_queue.put

    def fail_work_put(item, block=True, timeout=None):
        if hasattr(item, "evaluation_id"):
            raise RuntimeError("lane B queue rejected work")
        return original_put(item, block=block, timeout=timeout)

    monkeypatch.setattr(lane_b_queue, "put", fail_work_put)
    with pytest.raises(ResidentEvaluationPairFailed, match="admission failed"):
        pair.run_lanes(
            ResidentLaneRequest(DIGEST_A, lambda _handle: invoked.set(), 0, 0),
            ResidentLaneRequest(DIGEST_B, lambda _handle: None, 0, 0),
        )

    assert pair.fatal_error is not None
    assert not invoked.is_set()
    with pytest.raises(ResidentEvaluationPairFailed, match="queue rejected work"):
        pair.run_lane(
            "B",
            DIGEST_B,
            lambda _handle: None,
            expected_batch_count=0,
            expected_swap_count=0,
        )
    pair.close()


def test_run_lanes_timeout_uses_one_shared_wall_and_revokes_both() -> None:
    factory_a, factory_b = FakeFactory("a" * 32), FakeFactory("b" * 32)
    pair = ResidentEvaluationPair(
        factory_a,
        factory_b,
        start_timeout_s=5.0,
        request_timeout_s=0.2,
        close_timeout_s=5.0,
    )
    pair.start()
    release = threading.Event()
    barrier = threading.Barrier(2)
    handles = []

    def blocked(handle):
        handles.append(handle)
        barrier.wait(2.0)
        assert release.wait(5.0)

    started = time.monotonic()
    with pytest.raises(ResidentEvaluationPairFailed, match="timed out"):
        pair.run_lanes(
            ResidentLaneRequest(DIGEST_A, blocked, 0, 0),
            ResidentLaneRequest(DIGEST_B, blocked, 0, 0),
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.35
    assert len(handles) == 2
    assert pair.fatal_error is not None
    for handle in handles:
        with pytest.raises(ResidentEvaluationPairError, match="revoked"):
            handle.identity
    release.set()
    pair.close()


def test_three_run_lanes_calls_reuse_lifetimes_until_idempotent_close() -> None:
    pair, factory_a, factory_b = _pair()
    identities = pair.identities

    for _ in range(3):
        result_a, result_b = pair.run_lanes(
            ResidentLaneRequest(DIGEST_A, lambda handle: handle.identity, 0, 0),
            ResidentLaneRequest(DIGEST_B, lambda handle: handle.identity, 0, 0),
        )
        assert (result_a.value, result_b.value) == identities

    assert factory_a.calls == factory_b.calls == 1
    assert factory_a.sessions[0].finish_calls == 0
    assert factory_b.sessions[0].finish_calls == 0
    retirement = pair.close()
    assert retirement is not None and pair.close() is retirement
    assert factory_a.sessions[0].finish_calls == 1
    assert factory_b.sessions[0].finish_calls == 1


def test_local_operation_failure_is_returned_without_reload() -> None:
    pair, factory_a, _ = _pair()
    session_id = pair.identities[0].session_id

    def fail(handle):
        handle.swap(DIGEST_A)
        raise ValueError("candidate-local failure")

    failed = pair.run_lane(
        "A",
        DIGEST_A,
        fail,
        expected_batch_count=0,
        expected_swap_count=2,
    )
    assert not failed.ok
    assert failed.error is not None
    assert failed.error.error_type == "builtins.ValueError"
    assert failed.error.message == "candidate-local failure"
    assert not failed.error.epoch_fatal
    assert [row.bundle_digest for row in failed.request_slice.new_swaps] == [
        DIGEST_A,
        None,
    ]

    def recover(handle):
        identity = handle.identity
        handle.swap(DIGEST_B)
        handle.execute_batch(("prompt",))
        return identity

    recovered = pair.run_lane(
        "A",
        DIGEST_B,
        recover,
        expected_batch_count=1,
        expected_swap_count=2,
    )
    assert recovered.ok and recovered.value.session_id == session_id
    assert factory_a.calls == 1
    assert factory_a.sessions[0].finish_calls == 0
    pair.close()


def test_explicit_epoch_fatal_latches_without_silent_replacement() -> None:
    pair, factory_a, factory_b = _pair()

    def fatal(_handle):
        raise ResidentEvaluationEpochFatal("rank process was lost")

    result = pair.run_lane(
        "A",
        DIGEST_A,
        fatal,
        expected_batch_count=0,
        expected_swap_count=0,
    )
    assert result.error is not None and result.error.epoch_fatal
    with pytest.raises(ResidentEvaluationPairFailed, match="rank process was lost"):
        pair.run_lane(
            "B",
            DIGEST_B,
            lambda _handle: None,
            expected_batch_count=0,
            expected_swap_count=0,
        )
    assert factory_a.calls == factory_b.calls == 1
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 0
    pair.close()
    assert factory_a.sessions[0].finish_calls == factory_b.sessions[0].finish_calls == 1


def test_operation_handle_has_no_terminal_or_transport_surface() -> None:
    pair, _, _ = _pair()

    def inspect(handle):
        return {
            name: hasattr(handle, name)
            for name in ("finish", "abort", "transport", "session")
        }, handle.identity

    result = pair.run_lane(
        "A",
        DIGEST_A,
        inspect,
        expected_batch_count=0,
        expected_swap_count=0,
    )
    exposure, identity = result.value
    assert exposure == {
        "finish": False,
        "abort": False,
        "transport": False,
        "session": False,
    }
    assert identity == pair.identities[0]
    pair.close()


def test_operation_handle_refuses_another_bundle_without_touching_session() -> None:
    pair, factory_a, _ = _pair()
    result = pair.run_lane(
        "A",
        DIGEST_A,
        lambda handle: handle.swap(DIGEST_B),
        expected_batch_count=0,
        expected_swap_count=0,
    )
    assert result.error is not None and not result.error.epoch_fatal
    assert "cannot swap another bundle" in result.error.message
    assert result.request_slice.new_swaps == ()
    assert factory_a.sessions[0].active_bundle_digest is None
    assert factory_a.sessions[0].active_slots == ()
    pair.close()


def test_failed_stock_restore_is_bounded_epoch_fatal() -> None:
    pair, factory_a, factory_b = _pair()
    factory_a.sessions[0].fail_stock_restore = True

    result = pair.run_lane(
        "A",
        DIGEST_A,
        lambda handle: handle.swap(DIGEST_A),
        expected_batch_count=0,
        expected_swap_count=2,
    )
    assert result.error is not None and result.error.epoch_fatal
    assert "stock restoration failed" in result.error.message
    assert "\n" not in result.error.message
    assert len(result.error.message) <= 512
    assert result.request_slice.ending_bundle_digest == DIGEST_A
    assert result.request_slice.ending_slots == (SLOT,)
    with pytest.raises(ResidentEvaluationPairFailed, match="stock restoration failed"):
        pair.run_lane(
            "B",
            DIGEST_B,
            lambda _handle: None,
            expected_batch_count=0,
            expected_swap_count=0,
        )
    assert factory_a.calls == factory_b.calls == 1
    pair.close()


def test_start_then_close_retires_unused_lanes_and_caches_evidence() -> None:
    factory_a, factory_b = FakeFactory("a" * 32), FakeFactory("b" * 32)
    pair = ResidentEvaluationPair(
        factory_a,
        factory_b,
        start_timeout_s=5.0,
        request_timeout_s=5.0,
        close_timeout_s=5.0,
    )

    identities = pair.start()
    assert pair.start() == identities
    assert all(
        lane.thread is not None and not lane.thread.daemon
        for lane in pair._lanes.values()  # type: ignore[attr-defined]
    )

    retirement = pair.close()
    assert retirement is not None
    assert pair.close() is retirement
    assert retirement.request_history == ()
    assert retirement.lane_a.lifetime_evidence == ("finished", "a" * 32)
    assert retirement.lane_b.lifetime_evidence == ("finished", "b" * 32)
    assert factory_a.sessions[0].finish_allow_empty == [True]
    assert factory_b.sessions[0].finish_allow_empty == [True]


def test_close_before_start_is_idempotent_and_never_invokes_factories() -> None:
    factory_a, factory_b = FakeFactory("a" * 32), FakeFactory("b" * 32)
    pair = ResidentEvaluationPair(factory_a, factory_b)

    assert pair.close() is None
    assert pair.close() is None
    assert factory_a.calls == factory_b.calls == 0
    with pytest.raises(ResidentEvaluationPairError, match="closed"):
        pair.start()


def test_handle_is_opaque_and_revoked_before_request_returns() -> None:
    pair, factory_a, _ = _pair()
    captured = []

    def capture(handle):
        captured.append(handle)
        return handle

    result = pair.run_lane(
        "A",
        DIGEST_A,
        capture,
        expected_batch_count=0,
        expected_swap_count=0,
    )
    handle = captured[0]
    assert result.value is handle
    assert not hasattr(handle, "_ResidentEvaluationHandle__session")
    assert isinstance(handle._ResidentEvaluationHandle__token, str)
    before = (
        len(factory_a.sessions[0].batch_rows),
        len(factory_a.sessions[0].swap_receipts),
    )

    for use in (
        lambda: handle.identity,
        lambda: handle.swap(DIGEST_A),
        lambda: handle.execute_batch(("late",)),
    ):
        with pytest.raises(ResidentEvaluationPairError, match="revoked"):
            use()
    assert before == (
        len(factory_a.sessions[0].batch_rows),
        len(factory_a.sessions[0].swap_receipts),
    )
    pair.close()


def test_background_handle_use_is_rejected_on_the_live_request() -> None:
    pair, factory_a, _ = _pair()
    errors: list[BaseException] = []

    def operation(handle):
        def background() -> None:
            try:
                handle.swap(DIGEST_A)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=background)
        thread.start()
        thread.join(2.0)
        assert not thread.is_alive()

    result = pair.run_lane(
        "A",
        DIGEST_A,
        operation,
        expected_batch_count=0,
        expected_swap_count=0,
    )
    assert result.ok
    assert len(errors) == 1
    assert "left its lane thread" in str(errors[0])
    assert factory_a.sessions[0].swap_receipts == []
    pair.close()


def test_reentrant_pair_admission_and_close_fail_before_mutating_lifecycle() -> None:
    pair, _, _ = _pair()

    def operation(_handle):
        failures = []
        for action in (
            lambda: pair.run_lane(
                "B",
                DIGEST_A,
                lambda _nested: None,
                expected_batch_count=0,
                expected_swap_count=0,
            ),
            pair.close,
        ):
            try:
                action()
            except BaseException as exc:
                failures.append(exc)
        return failures

    result = pair.run_lane(
        "A",
        DIGEST_A,
        operation,
        expected_batch_count=0,
        expected_swap_count=0,
    )
    assert result.ok
    assert len(result.value) == 2
    assert all(
        isinstance(error, ResidentEvaluationPairError)
        and "reentrant" in str(error)
        for error in result.value
    )
    assert pair.run_lane(
        "B",
        DIGEST_B,
        lambda _handle: None,
        expected_batch_count=0,
        expected_swap_count=0,
    ).ok
    pair.close()


def test_request_timeout_revokes_and_retains_eventual_fatal_history() -> None:
    factory_a, factory_b = FakeFactory("a" * 32), FakeFactory("b" * 32)
    pair = ResidentEvaluationPair(
        factory_a,
        factory_b,
        start_timeout_s=5.0,
        request_timeout_s=0.05,
        close_timeout_s=5.0,
    )
    pair.start()
    entered, release = threading.Event(), threading.Event()
    captured = []

    def blocked(handle):
        captured.append(handle)
        entered.set()
        assert release.wait(5.0)

    with pytest.raises(ResidentEvaluationPairFailed, match="timed out"):
        pair.run_lane(
            "A",
            DIGEST_A,
            blocked,
            expected_batch_count=0,
            expected_swap_count=0,
        )
    assert entered.is_set()
    with pytest.raises(ResidentEvaluationPairError, match="revoked"):
        captured[0].swap(DIGEST_A)
    release.set()
    deadline = time.monotonic() + 2.0
    while not pair.request_history and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(pair.request_history) == 1
    eventual = pair.request_history[0]
    assert eventual.error is not None and eventual.error.epoch_fatal
    assert "timed out" in eventual.error.message
    assert eventual.request_slice.ending_bundle_digest is None
    with pytest.raises(ResidentEvaluationPairFailed, match="timed out"):
        pair.run_lane(
            "B",
            DIGEST_B,
            lambda _handle: None,
            expected_batch_count=0,
            expected_swap_count=0,
        )
    retirement = pair.close()
    assert retirement is not None
    assert retirement.request_history == (eventual,)


def test_declared_budget_exhaustion_is_epoch_fatal_before_callback() -> None:
    pair, factory_a, _ = _pair()
    invoked = threading.Event()
    factory_a.sessions[0].plan.max_batches = 0

    result = pair.run_lane(
        "A",
        DIGEST_A,
        lambda _handle: invoked.set(),
        expected_batch_count=1,
        expected_swap_count=0,
    )
    assert result.error is not None and result.error.epoch_fatal
    assert "lacks declared request budget" in result.error.message
    assert not invoked.is_set()
    assert result.request_slice.new_batches == ()
    pair.close()


def test_successful_callback_with_inexact_counts_latches_epoch() -> None:
    pair, _, _ = _pair()

    result = pair.run_lane(
        "A",
        DIGEST_A,
        lambda _handle: None,
        expected_batch_count=1,
        expected_swap_count=0,
    )
    assert result.error is not None and result.error.epoch_fatal
    assert "verb counts differ" in result.error.message
    with pytest.raises(ResidentEvaluationPairFailed, match="verb counts differ"):
        pair.run_lane(
            "B",
            DIGEST_B,
            lambda _handle: None,
            expected_batch_count=0,
            expected_swap_count=0,
        )
    pair.close()


def test_terminal_operation_preserves_original_diagnostic_and_error_type() -> None:
    from cacheon.eval.oci_outer_session import OuterSessionProtocolError

    pair, factory_a, _ = _pair()
    factory_a.sessions[0].terminal_batch_error = OuterSessionProtocolError(
        "original protocol diagnostic"
    )

    result = pair.run_lane(
        "A",
        DIGEST_A,
        lambda handle: handle.execute_batch(("prompt",), canary=True),
        expected_batch_count=1,
        expected_swap_count=0,
    )
    assert result.error is not None and result.error.epoch_fatal
    assert result.error.error_type.endswith("OuterSessionProtocolError")
    assert result.error.message == "original protocol diagnostic"
    assert "stock restoration failed" not in result.error.message
    with pytest.raises(
        ResidentEvaluationPairError, match="original protocol diagnostic"
    ):
        pair.close()

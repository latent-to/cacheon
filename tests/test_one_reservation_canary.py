from __future__ import annotations

import hashlib
from dataclasses import fields

import pytest

from cacheon.chain.one_reservation_canary import (
    CanaryCheckpoint,
    CanaryLeaseObservation,
    CanaryReservationPhase,
    CanaryStage,
    CanaryStageDisposition,
    CanaryStageReceipt,
    CanaryStoreObservation,
    CanaryTerminalOutcome,
    OneReservationCanaryBoundaries,
    OneReservationCanaryController,
    OneReservationCanaryError,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Harness:
    def __init__(
        self,
        *,
        expected: str,
        profile: str,
        phase: CanaryReservationPhase,
    ) -> None:
        self.expected = expected
        self.profile = profile
        self.phase = phase
        self.fifo_head: str | None = expected
        self.qualification_preview: tuple[str, ...] = ()
        self.active_lease: CanaryLeaseObservation | None = None
        self.screen_impl = lambda: (_ for _ in ()).throw(
            AssertionError("unexpected screen call")
        )
        self.qualification_impl = lambda: (_ for _ in ()).throw(
            AssertionError("unexpected qualification call")
        )
        self.calls: list[str] = []
        self.checkpoints: list[CanaryCheckpoint] = []
        self.observations = 0

    def observe(self) -> CanaryStoreObservation:
        self.observations += 1
        return CanaryStoreObservation(
            reservation_digest=self.expected,
            target_profile_digest=self.profile,
            phase=self.phase,
            fifo_head_reservation_digest=self.fifo_head,
            next_qualification_reservation_digests=self.qualification_preview,
            active_lease=self.active_lease,
        )

    def screen(
        self,
        observation: CanaryStoreObservation,
        checkpoint: CanaryCheckpoint,
    ) -> CanaryStageReceipt:
        assert observation.reservation_digest == checkpoint.expected_reservation_digest
        assert checkpoint.screen_claim_started
        self.calls.append("screen")
        return self.screen_impl()

    def qualification(
        self,
        observation: CanaryStoreObservation,
        checkpoint: CanaryCheckpoint,
    ) -> CanaryStageReceipt:
        assert observation.reservation_digest == checkpoint.expected_reservation_digest
        assert checkpoint.qualification_started
        self.calls.append("qualification")
        return self.qualification_impl()

    def persist(self, checkpoint: CanaryCheckpoint) -> None:
        assert type(checkpoint) is CanaryCheckpoint
        self.checkpoints.append(checkpoint)


def _lease(
    stage: CanaryStage,
    label: str,
    members: tuple[str, ...],
    request_label: str | None,
) -> CanaryLeaseObservation:
    return CanaryLeaseObservation(
        stage=stage,
        lease_id=_d(f"lease:{label}"),
        reservation_digests=members,
        request_id=None if request_label is None else _d(f"request:{request_label}"),
    )


def _stage_receipt(
    stage: CanaryStage,
    label: str,
    members: tuple[str, ...],
    *,
    disposition: CanaryStageDisposition = CanaryStageDisposition.COMPLETED,
    lease_label: str | None = None,
    request_label: str | None = None,
    expensive: tuple[str, ...] = (),
    reason: str | None = None,
) -> CanaryStageReceipt:
    no_work = disposition is CanaryStageDisposition.NO_WORK
    return CanaryStageReceipt(
        stage=stage,
        disposition=disposition,
        receipt_digest=_d(f"stage-receipt:{label}"),
        reservation_digests=() if no_work else members,
        lease_id=(
            None
            if no_work
            else _d(f"lease:{label if lease_label is None else lease_label}")
        ),
        request_id=(
            None
            if no_work
            else _d(f"request:{label if request_label is None else request_label}")
        ),
        expensive_stage_receipt_digests=tuple(_d(f"expensive:{x}") for x in expensive),
        reason=reason,
    )


def _controller(
    harness: _Harness,
    clock: _Clock,
    *,
    expected: str | None = None,
    deadline: float | None = None,
    max_ticks: int = 8,
    max_stage_receipts: int = 16,
    retained: CanaryCheckpoint | None = None,
) -> OneReservationCanaryController:
    return OneReservationCanaryController(
        boundaries=OneReservationCanaryBoundaries(
            observe_store=harness.observe,
            screen_once=harness.screen,
            qualification_once=harness.qualification,
            persist_checkpoint=harness.persist,
        ),
        expected_fifo_head_reservation_digest=(
            harness.expected if expected is None else expected
        ),
        deadline_monotonic=clock.value + 100 if deadline is None else deadline,
        max_ticks=max_ticks,
        max_stage_receipts=max_stage_receipts,
        monotonic=clock,
        retained_checkpoint=retained,
    )


@pytest.mark.parametrize("profile_label", ["target-profile-a", "target-profile-b"])
def test_green_screen_then_qualification_for_two_target_profiles(
    profile_label: str,
) -> None:
    expected = _d(f"reservation:{profile_label}")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d(profile_label),
        phase=CanaryReservationPhase.PUBLISHED,
    )

    def screen() -> CanaryStageReceipt:
        # Intent is durable before any existing screen stage may mutate the store.
        assert harness.checkpoints[-1].screen_claim_started
        assert harness.checkpoints[-1].ticks_used == 1
        harness.phase = CanaryReservationPhase.PROMOTED
        harness.fifo_head = None
        harness.qualification_preview = (expected,)
        return _stage_receipt(CanaryStage.SCREEN, "screen", (expected,))

    def qualification() -> CanaryStageReceipt:
        assert harness.checkpoints[-1].qualification_started
        assert harness.checkpoints[-1].ticks_used == 2
        harness.phase = CanaryReservationPhase.REPRODUCTION_PENDING
        harness.qualification_preview = ()
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            "qualification",
            (expected,),
            expensive=("baseline-b", "candidate-c", "recheck-b-prime"),
        )

    harness.screen_impl = screen
    harness.qualification_impl = qualification
    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.COMPLETED
    assert receipt.final_phase is CanaryReservationPhase.REPRODUCTION_PENDING
    assert harness.calls == ["screen", "qualification"]
    assert receipt.checkpoint.ticks_used == 2
    assert receipt.checkpoint.target_profile_digest == _d(profile_label)
    assert receipt.checkpoint.screen_request_id == _d("request:screen")
    assert receipt.checkpoint.qualification_request_id == _d("request:qualification")
    assert [row.stage for row in receipt.checkpoint.transitions] == [
        CanaryStage.SCREEN,
        CanaryStage.QUALIFICATION,
    ]
    assert receipt.to_dict()["schema"].endswith("receipt-v1")
    assert len(receipt.digest) == 64


def test_existing_exact_qualification_is_resumed_before_screen() -> None:
    expected = _d("reservation")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.fifo_head = _d("different-fresh-row")
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "same"
    )

    def qualification() -> CanaryStageReceipt:
        assert harness.checkpoints[-1].qualification_lease_id == _d(
            "lease:qualification"
        )
        assert harness.checkpoints[-1].qualification_request_id == _d("request:same")
        harness.phase = CanaryReservationPhase.QUALIFIED
        harness.active_lease = None
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            "qualification-complete",
            (expected,),
            lease_label="qualification",
            request_label="same",
        )

    harness.qualification_impl = qualification
    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.COMPLETED
    assert harness.calls == ["qualification"]
    assert receipt.checkpoint.qualification_request_id == _d("request:same")


def test_wrong_fifo_head_exits_before_any_mutating_call() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.PUBLISHED,
    )
    harness.fifo_head = _d("other")

    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.WRONG_FIFO_HEAD
    assert harness.calls == []
    assert receipt.checkpoint.ticks_used == 0


@pytest.mark.parametrize(
    "members",
    [
        (_d("other"),),
        (_d("expected"), _d("other")),
    ],
)
def test_wrong_or_extra_active_qualification_member_exits_before_mutation(
    members: tuple[str, ...],
) -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    # Substitute the parameter's expected placeholder with the real digest.
    members = tuple(expected if item == _d("expected") else item for item in members)
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", members, "request"
    )

    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.LEASE_DRIFT
    assert harness.calls == []


@pytest.mark.parametrize(
    "preview",
    [
        (_d("other"),),
        (_d("expected"), _d("other")),
    ],
)
def test_qualification_preview_must_be_exactly_one_expected_member(
    preview: tuple[str, ...],
) -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.PROMOTED,
    )
    harness.qualification_preview = tuple(
        expected if item == _d("expected") else item for item in preview
    )

    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.LEASE_DRIFT
    assert harness.calls == []


def test_stage_receipt_cannot_introduce_a_second_reservation() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.PUBLISHED,
    )

    def screen() -> CanaryStageReceipt:
        harness.phase = CanaryReservationPhase.PROMOTED
        return _stage_receipt(
            CanaryStage.SCREEN, "bad-screen", (expected, _d("second"))
        )

    harness.screen_impl = screen
    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.LEASE_DRIFT
    assert harness.calls == ["screen"]
    assert receipt.checkpoint.stage_receipt_digests == ()


def test_request_id_drift_terminates_same_request_resume() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "original"
    )

    def qualification() -> CanaryStageReceipt:
        harness.phase = CanaryReservationPhase.QUALIFIED
        harness.active_lease = None
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            "drifted",
            (expected,),
            lease_label="qualification",
            request_label="replacement",
        )

    harness.qualification_impl = qualification
    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.REQUEST_DRIFT
    assert receipt.checkpoint.qualification_request_id == _d("request:original")


def test_repeated_expensive_stage_receipt_fails_closed() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "same"
    )
    call = {"value": 0}

    def qualification() -> CanaryStageReceipt:
        call["value"] += 1
        if call["value"] == 1:
            return _stage_receipt(
                CanaryStage.QUALIFICATION,
                "progress-1",
                (expected,),
                disposition=CanaryStageDisposition.PROGRESSED,
                lease_label="qualification",
                request_label="same",
                expensive=("candidate-c",),
            )
        harness.phase = CanaryReservationPhase.QUALIFIED
        harness.active_lease = None
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            "progress-2",
            (expected,),
            lease_label="qualification",
            request_label="same",
            expensive=("candidate-c",),
        )

    harness.qualification_impl = qualification
    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.REPEATED_EXPENSIVE_STAGE
    assert harness.calls == ["qualification", "qualification"]
    assert receipt.checkpoint.expensive_stage_receipt_digests == (
        _d("expensive:candidate-c"),
    )


@pytest.mark.parametrize(
    ("disposition", "terminal"),
    [
        (CanaryStageDisposition.HOLD, CanaryTerminalOutcome.HOLD),
        (CanaryStageDisposition.REQUEUE, CanaryTerminalOutcome.REQUEUE),
    ],
)
def test_typed_hold_and_requeue_exit_without_retry(
    disposition: CanaryStageDisposition,
    terminal: CanaryTerminalOutcome,
) -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "same"
    )

    def qualification() -> CanaryStageReceipt:
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            disposition.value,
            (expected,),
            disposition=disposition,
            lease_label="qualification",
            request_label="same",
            reason="typed_worker_disposition",
        )

    harness.qualification_impl = qualification
    receipt = _controller(harness, clock).run()

    assert receipt.outcome is terminal
    assert harness.calls == ["qualification"]
    assert "typed_worker_disposition" in receipt.checkpoint.terminal_reason


def test_pre_request_qualification_hold_does_not_invent_request_identity() -> None:
    values = {
        "disposition": CanaryStageDisposition.HOLD,
        "lease_id": _d("qualification-lease"),
        "reason": "held_before_request_plan",
        "receipt_digest": _d("qualification-hold"),
        "request_id": None,
        "reservation_digests": (_d("reservation"),),
    }
    receipt = CanaryStageReceipt(stage=CanaryStage.QUALIFICATION, **values)
    assert receipt.request_id is None
    with pytest.raises(OneReservationCanaryError, match="request identity"):
        CanaryStageReceipt(stage=CanaryStage.SCREEN, **values)


def test_deadline_exits_before_mutation() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.PUBLISHED,
    )

    receipt = _controller(harness, clock, deadline=clock.value).run()

    assert receipt.outcome is CanaryTerminalOutcome.DEADLINE
    assert harness.calls == []
    assert receipt.checkpoint.ticks_used == 0


def test_max_ticks_stops_same_request_continuation() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "same"
    )

    def qualification() -> CanaryStageReceipt:
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            "progress",
            (expected,),
            disposition=CanaryStageDisposition.PROGRESSED,
            lease_label="qualification",
            request_label="same",
        )

    harness.qualification_impl = qualification
    receipt = _controller(harness, clock, max_ticks=1).run()

    assert receipt.outcome is CanaryTerminalOutcome.MAX_TICKS
    assert harness.calls == ["qualification"]
    assert receipt.checkpoint.ticks_used == 1


def test_stage_receipt_bound_counts_outer_and_expensive_receipts() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "same"
    )

    def qualification() -> CanaryStageReceipt:
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            "too-many-receipts",
            (expected,),
            lease_label="qualification",
            request_label="same",
            expensive=("one", "two"),
        )

    harness.qualification_impl = qualification
    receipt = _controller(harness, clock, max_stage_receipts=2).run()

    assert receipt.outcome is CanaryTerminalOutcome.MAX_STAGE_RECEIPTS
    assert receipt.checkpoint.stage_receipt_digests == ()


def test_restart_after_preclaim_checkpoint_never_calls_screen_again() -> None:
    expected = _d("expected")
    clock = _Clock()
    first = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.PUBLISHED,
    )

    def crash_after_intent() -> CanaryStageReceipt:
        assert first.checkpoints[-1].screen_claim_started
        raise RuntimeError("process died after durable intent")

    first.screen_impl = crash_after_intent
    with pytest.raises(RuntimeError, match="after durable intent"):
        _controller(first, clock).run()
    retained = first.checkpoints[-1]
    assert retained.screen_claim_started
    assert retained.ticks_used == 1

    restarted = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.PUBLISHED,
    )
    receipt = _controller(restarted, clock, retained=retained).run()

    assert receipt.outcome is CanaryTerminalOutcome.SECOND_CLAIM
    assert restarted.calls == []


def test_restart_from_completed_checkpoint_is_idempotent_and_does_no_work() -> None:
    expected = _d("expected")
    clock = _Clock()
    first = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.REPRODUCTION_PENDING,
    )
    original = _controller(first, clock).run()
    assert original.outcome is CanaryTerminalOutcome.COMPLETED
    assert first.calls == []

    restarted = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.REPRODUCTION_PENDING,
    )
    replay = _controller(
        restarted, clock, retained=original.checkpoint
    ).run()

    assert replay.outcome is CanaryTerminalOutcome.COMPLETED
    assert restarted.calls == []
    assert replay.checkpoint.digest == original.checkpoint.digest


def test_retained_request_and_target_profile_cannot_drift() -> None:
    expected = _d("expected")
    clock = _Clock()
    first = _Harness(
        expected=expected,
        profile=_d("profile-a"),
        phase=CanaryReservationPhase.QUALIFIED,
    )
    completed = _controller(first, clock).run()

    changed = _Harness(
        expected=expected,
        profile=_d("profile-b"),
        phase=CanaryReservationPhase.QUALIFIED,
    )
    receipt = _controller(changed, clock, retained=completed.checkpoint).run()

    assert receipt.outcome is CanaryTerminalOutcome.IDENTITY_DRIFT
    assert changed.calls == []


def test_qualification_callback_completion_is_not_store_completion() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "same"
    )

    def qualification() -> CanaryStageReceipt:
        # Deliberately leave durable state qualifying.
        return _stage_receipt(
            CanaryStage.QUALIFICATION,
            "lying-completion",
            (expected,),
            lease_label="qualification",
            request_label="same",
        )

    harness.qualification_impl = qualification
    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.HOLD
    assert receipt.checkpoint.terminal_reason == "qualification_receipt_not_complete_in_store"


def test_no_work_is_typed_and_never_inferred_as_success() -> None:
    expected = _d("expected")
    clock = _Clock()
    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFYING,
    )
    harness.active_lease = _lease(
        CanaryStage.QUALIFICATION, "qualification", (expected,), "same"
    )
    harness.qualification_impl = lambda: _stage_receipt(
        CanaryStage.QUALIFICATION,
        "idle",
        (expected,),
        disposition=CanaryStageDisposition.NO_WORK,
    )

    receipt = _controller(harness, clock).run()

    assert receipt.outcome is CanaryTerminalOutcome.NO_WORK
    assert receipt.final_phase is CanaryReservationPhase.QUALIFYING


def test_authority_surface_has_no_post_evaluation_stages() -> None:
    assert {field.name for field in fields(OneReservationCanaryBoundaries)} == {
        "observe_store",
        "screen_once",
        "qualification_once",
        "persist_checkpoint",
    }
    for forbidden in ("settlement", "incentive", "weights", "launch", "restart"):
        assert not hasattr(OneReservationCanaryBoundaries, forbidden)


def test_monotonic_clock_regression_fails_closed() -> None:
    expected = _d("expected")

    class _RegressingClock:
        values = iter((100.0, 99.0))

        def __call__(self) -> float:
            return next(self.values)

    harness = _Harness(
        expected=expected,
        profile=_d("profile"),
        phase=CanaryReservationPhase.QUALIFIED,
    )
    controller = OneReservationCanaryController(
        boundaries=OneReservationCanaryBoundaries(
            harness.observe,
            harness.screen,
            harness.qualification,
            harness.persist,
        ),
        expected_fifo_head_reservation_digest=expected,
        deadline_monotonic=200.0,
        max_ticks=2,
        max_stage_receipts=2,
        monotonic=_RegressingClock(),
    )

    with pytest.raises(OneReservationCanaryError, match="moved backwards"):
        controller.run()

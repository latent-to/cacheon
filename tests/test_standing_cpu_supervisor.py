from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from cacheon.chain.execution_disposition import (
    ExecutionDisposition,
    ExecutionOutcome,
)
from cacheon.chain.evaluation_recovery import RecoveryPhase
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationHold,
    RecoverableQualificationRequeue,
)
from cacheon.chain.standing_cpu_supervisor import (
    StandingCpuSupervisor,
    StandingCpuSupervisorError,
    SupervisorPhase,
    SupervisorStageResult,
    run_forever,
    settlement_stage,
    weights_stage,
)


def _d(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_tick_prefers_qualification_resume_before_new_screen_claim() -> None:
    calls: list[str] = []

    def qualification():
        calls.append("qualification")
        return SupervisorStageResult(
            stage="qualification",
            progressed=True,
            disposition="completed",
            request_id=_d("request"),
            lease_id=_d("lease"),
            lane_assignment="primary",
            worker_epoch="7",
            checkpoint_age_s=12.5,
        )

    def screen():
        calls.append("screen")
        raise AssertionError("screen must not run while qualification has work")

    supervisor = StandingCpuSupervisor(
        screen_once=screen,
        qualification_once=qualification,
        clock=_Clock(),
    )
    status = supervisor.tick()
    assert calls == ["qualification"]
    assert status.phase is SupervisorPhase.QUALIFICATION
    assert status.request_id == _d("request")
    assert status.lease_id == _d("lease")
    assert status.lane_assignment == "primary"
    assert status.worker_epoch == "7"
    assert status.checkpoint_age_s == 12.5
    assert status.to_dict()["schema"].endswith("status-v1")


def test_tick_runs_screen_when_qualification_is_idle() -> None:
    calls: list[str] = []

    def qualification():
        calls.append("qualification")
        return None

    def screen():
        calls.append("screen")
        return SupervisorStageResult(
            stage="screen",
            progressed=True,
            disposition="completed",
            lease_id=_d("screen-lease"),
            lane_assignment="reproduction",
        )

    status = StandingCpuSupervisor(
        screen_once=screen,
        qualification_once=qualification,
        clock=_Clock(),
    ).tick()
    assert calls == ["qualification", "screen"]
    assert status.phase is SupervisorPhase.SCREEN
    assert status.lease_id == _d("screen-lease")


def test_hold_and_requeue_products_are_visible_without_inventing_retry() -> None:
    hold = RecoverableQualificationHold(
        recovery_id=_d("recovery"),
        phase=RecoveryPhase.HELD,
        request_id=_d("held-request"),
        reason="marker_present",
    )
    status = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: hold,
        clock=_Clock(),
    ).tick()
    assert status.phase is SupervisorPhase.HOLD
    assert status.hold_reason == "marker_present"
    assert status.last_disposition == "hold"

    requeue = RecoverableQualificationRequeue(
        recovery_id=_d("recovery-2"),
        request_id=_d("requeue-request"),
        outcome=ExecutionOutcome(
            disposition=ExecutionDisposition.REQUEUE,
            decision="NO_DECISION",
            failure_code="adapter_start_failed",
        ),
    )
    status = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: requeue,
        clock=_Clock(),
    ).tick()
    assert status.phase is SupervisorPhase.QUALIFICATION
    assert status.request_id == _d("requeue-request")
    assert status.last_disposition == "requeue"
    assert status.hold_reason is None


def test_stall_becomes_visible_hold_not_idle_wait() -> None:
    clock = _Clock()
    supervisor = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: None,
        clock=clock,
        stall_timeout_s=30.0,
    )
    assert supervisor.tick().phase is SupervisorPhase.IDLE
    clock.advance(30.0)
    status = supervisor.tick()
    assert status.phase is SupervisorPhase.HOLD
    assert status.hold_reason == "supervisor_progress_stalled"


def test_exception_fails_closed_without_mapping_to_requeue() -> None:
    supervisor = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: (_ for _ in ()).throw(TimeoutError("boom")),
        clock=_Clock(),
    )
    with pytest.raises(StandingCpuSupervisorError, match="without a typed disposition"):
        supervisor.tick()
    assert supervisor.status().phase is SupervisorPhase.FAILED
    assert supervisor.status().last_disposition == "stage_error"


def test_run_forever_resumes_same_request_across_restart() -> None:
    """A process restart is a new supervisor; qualification stage resumes the request."""

    requests: list[str] = []
    state = {"crashes": 1, "qual_calls": 0}

    def qualification():
        state["qual_calls"] += 1
        request_id = _d("same-request")
        requests.append(request_id)
        if state["crashes"]:
            state["crashes"] -= 1
            raise RuntimeError("simulated supervisor process crash mid-qualification")
        return SupervisorStageResult(
            stage="qualification",
            progressed=True,
            disposition="completed",
            request_id=request_id,
            lease_id=_d("same-lease"),
        )

    screen_calls = {"n": 0}

    def screen():
        screen_calls["n"] += 1
        return None

    # First process: crash on first tick.
    first = StandingCpuSupervisor(
        screen_once=screen,
        qualification_once=qualification,
        clock=_Clock(),
    )
    with pytest.raises(StandingCpuSupervisorError, match="simulated supervisor"):
        first.tick()
    assert requests == [_d("same-request")]
    assert screen_calls["n"] == 0

    # Second process: same injectable stages resume the same request id.
    second = StandingCpuSupervisor(
        screen_once=screen,
        qualification_once=qualification,
        clock=_Clock(),
    )
    status = second.tick()
    assert status.request_id == _d("same-request")
    assert status.lease_id == _d("same-lease")
    assert requests == [_d("same-request"), _d("same-request")]
    assert state["qual_calls"] == 2
    assert screen_calls["n"] == 0


def test_run_forever_idle_poll_and_stop() -> None:
    clock = _Clock()
    supervisor = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: None,
        clock=clock,
        stall_timeout_s=10_000.0,
    )
    stop = threading.Event()
    seen: list[SupervisorPhase] = []

    def wait(seconds: float) -> bool:
        clock.advance(seconds)
        stop.set()
        return True

    run_forever(
        supervisor,
        stop,
        idle_poll_s=0.5,
        wait=wait,
        on_status=lambda status: seen.append(status.phase),
    )
    assert seen == [SupervisorPhase.IDLE]


def test_run_forever_prints_exact_stage_error(capsys) -> None:
    stop = threading.Event()
    supervisor = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: (_ for _ in ()).throw(TimeoutError("exact boom")),
        clock=_Clock(),
    )

    with pytest.raises(StandingCpuSupervisorError, match="exact boom"):
        run_forever(supervisor, stop)
    assert (
        "STANDING-CPU-SUPERVISOR-STAGE-ERROR: stage 'qualification' failed "
        "closed without a typed disposition: TimeoutError: exact boom"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("disposition", ["hold", "requeue"])
def test_run_forever_backs_off_typed_no_progress_after_screening(
    disposition: str,
) -> None:
    # A held or requeued qualification is not a unit of work: the screen stage
    # still runs on the same tick (2026-08-10 mainnet: one durably held
    # recovery starved the entire screen FIFO). Backoff engages only once no
    # stage anywhere progresses, and holds never advance the progress clock.
    clock = _Clock()
    stop = threading.Event()
    calls = {"qualification": 0, "screen": 0}
    waits: list[float] = []
    products = {
        "hold": RecoverableQualificationHold(
            recovery_id=_d("held-recovery"),
            phase=RecoveryPhase.HELD,
            request_id=_d("held-request"),
            reason="operator_hold",
        ),
        "requeue": RecoverableQualificationRequeue(
            recovery_id=_d("requeued-recovery"),
            request_id=_d("requeued-request"),
            outcome=ExecutionOutcome(
                disposition=ExecutionDisposition.REQUEUE,
                decision="NO_DECISION",
                failure_code="adapter_start_failed",
            ),
        ),
    }

    def qualification():
        calls["qualification"] += 1
        return products[disposition]

    def screen():
        calls["screen"] += 1
        return None

    def wait(seconds: float) -> bool:
        waits.append(seconds)
        clock.advance(seconds)
        if len(waits) == 2:
            stop.set()
            return True
        return False

    supervisor = StandingCpuSupervisor(
        screen_once=screen,
        qualification_once=qualification,
        clock=clock,
    )
    initial_progress = supervisor.status().last_progress_unix
    run_forever(
        supervisor,
        stop,
        idle_poll_s=0.25,
        wait=wait,
        restart_initial_backoff_s=2.0,
        restart_max_backoff_s=8.0,
    )

    assert calls == {"qualification": 2, "screen": 2}
    assert waits == [2.0, 4.0]
    assert supervisor.status().last_progress_unix == initial_progress
    assert supervisor.status().last_disposition == disposition


def test_held_qualification_does_not_starve_screen_progress() -> None:
    # The regression observed live on 2026-08-10: qualification durably HELD,
    # screens never claimed again, published count frozen. A progressing
    # screen must win the tick over a non-progressing qualification hold.
    hold = RecoverableQualificationHold(
        recovery_id=_d("held-recovery"),
        phase=RecoveryPhase.HELD,
        request_id=_d("held-request"),
        reason="transport_hold:worker_infrastructure_result",
    )
    screens: list[str] = []

    def screen():
        screens.append("claimed")
        return SupervisorStageResult(
            stage="screen",
            progressed=True,
            disposition="completed",
            lease_id=_d("screen-lease"),
            lane_assignment="primary",
        )

    supervisor = StandingCpuSupervisor(
        screen_once=screen,
        qualification_once=lambda: hold,
        clock=_Clock(),
    )
    for _ in range(3):
        status = supervisor.tick()
        assert status.phase is SupervisorPhase.SCREEN
        assert status.last_disposition == "completed"
    assert screens == ["claimed"] * 3

    # When the screen queue empties, the standing hold becomes visible again.
    supervisor_idle = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: hold,
        clock=_Clock(),
    )
    status = supervisor_idle.tick()
    assert status.last_disposition == "hold"
    assert status.hold_reason == "transport_hold:worker_infrastructure_result"


def test_settlement_runs_before_new_qualification_and_requires_commission() -> None:
    order: list[str] = []

    def settle():
        order.append("settlement")
        return SupervisorStageResult(
            stage="settlement",
            progressed=True,
            disposition="committed",
            lease_id=_d("settlement-lease"),
        )

    status = StandingCpuSupervisor(
        screen_once=lambda: order.append("screen") or None,
        qualification_once=lambda: order.append("qualification") or None,
        settle_once=settle,
        clock=_Clock(),
    )
    assert status.tick().phase is SupervisorPhase.SETTLEMENT
    assert order == ["settlement"]
    held = status.tick()
    assert held.phase is SupervisorPhase.HOLD
    assert held.hold_reason == "baseline_commission_required"
    assert order == ["settlement"]


def test_untyped_stage_product_is_rejected() -> None:
    with pytest.raises(StandingCpuSupervisorError, match="untyped product"):
        StandingCpuSupervisor(
            screen_once=lambda: SimpleNamespace(disposition="completed"),
            qualification_once=lambda: None,
            clock=_Clock(),
        ).tick()


def test_settlement_and_weights_stages_wire_into_supervisor() -> None:
    class _Store:
        def has_pending_settlement(self):
            return True

        def close(self):
            pass

    committed = {"lease": _d("settlement-plan")}

    import cacheon.chain.validator_loop as validator_loop

    original = validator_loop._settle_pending
    validator_loop._settle_pending = lambda store, **kwargs: committed
    try:
        settle = settlement_stage(
            open_store=lambda: (_Store(), (100, _d("finalized"))),
            finalized_block_provider=lambda: 100,
        )
        status = StandingCpuSupervisor(
            screen_once=lambda: None,
            qualification_once=lambda: None,
            settle_once=settle,
            clock=_Clock(),
        ).tick()
    finally:
        validator_loop._settle_pending = original
    assert status.phase is SupervisorPhase.SETTLEMENT
    assert status.lease_id == "lease"

    published = weights_stage(
        publish=lambda: SimpleNamespace(
            projection_digest=_d("projection"), status="confirmed"
        )
    )
    result = published()
    assert result is not None
    assert result.phase is SupervisorPhase.WEIGHTS
    assert result.disposition == "confirmed"

    assert weights_stage(publish=lambda: None)() is None


def test_python_m_entry_shares_one_module_identity_with_by_name_imports(tmp_path) -> None:
    """``python -m cacheon.chain.standing_cpu_supervisor`` runs the file as ``__main__``.

    The weights stage imports the supervisor by name when it is composed. On
    mainnet (2026-08-19) that loaded a second module copy whose
    ``SupervisorStageResult`` and ``StandingCpuSupervisorError`` were different
    classes, so every weights push that had already landed on serve-weights was
    then rejected as "an untyped product". The entry must alias itself so that
    the by-name import and the ``__main__`` module are one object.
    """

    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    import cacheon

    package_root = Path(cacheon.__file__).resolve().parents[1]
    child = textwrap.dedent(
        f"""
        import runpy, sys
        sys.argv = ["standing_cpu_supervisor", "--config", {str(tmp_path / "missing.json")!r}]
        try:
            runpy.run_module(
                "cacheon.chain.standing_cpu_supervisor", run_name="__main__", alter_sys=True
            )
        except SystemExit as exc:
            print("ENTRY_EXIT=" + str(exc.code))
        import cacheon.chain.standing_cpu_supervisor as by_name
        from cacheon.chain import standing_weights_stage
        print("BY_NAME_IS_MAIN=" + str(by_name.__name__ == "__main__"))
        try:
            standing_weights_stage.load_weights_config({str(tmp_path / "missing-weights.json")!r})
        except by_name.StandingCpuSupervisorError:
            print("STAGE_ERROR_IS_ENTRY_CLASS=True")
        except Exception as exc:
            print("STAGE_ERROR_IS_ENTRY_CLASS=False " + type(exc).__module__)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", child],
        cwd=package_root,
        env={**os.environ, "PYTHONPATH": str(package_root)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    lines = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    assert lines.get("ENTRY_EXIT") == "2", proc.stderr[-800:]
    assert lines.get("BY_NAME_IS_MAIN") == "True", proc.stdout
    assert lines.get("STAGE_ERROR_IS_ENTRY_CLASS") == "True", proc.stdout

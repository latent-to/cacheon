"""Crash matrix for standing composition (handoff aug6-2-1 §9 boundaries).

Each boundary must resume the same request/downstream phase, commit an already
completed product, or HOLD — never restart the expensive experiment.
"""

from __future__ import annotations

import hashlib

import pytest

from cacheon.chain.standing_cpu_supervisor import (
    StandingCpuSupervisor,
    StandingCpuSupervisorError,
    SupervisorPhase,
    SupervisorStageResult,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


# Ordered durable stages for one protected qualification request.
_STAGES = (
    "plan",
    "publish",
    "speed",
    "quality",
    "import",
    "commit",
)


class _ProtectedQualification:
    def __init__(self) -> None:
        self.index = 0
        self.request_id = _d("canary-request")
        self.lease_id = _d("canary-lease")
        self.counts = {name: 0 for name in _STAGES}
        self.crash_before: str | None = None

    @property
    def done(self) -> bool:
        return self.index >= len(_STAGES)

    def __call__(self):
        if self.done:
            return SupervisorStageResult(
                stage="qualification",
                progressed=True,
                disposition="done",
                request_id=self.request_id,
                lease_id=self.lease_id,
            )
        name = _STAGES[self.index]
        if self.crash_before == name:
            self.crash_before = None
            raise RuntimeError(f"simulated crash before {name}")
        self.counts[name] += 1
        self.index += 1
        return SupervisorStageResult(
            stage="qualification",
            progressed=True,
            disposition=name,
            request_id=self.request_id,
            lease_id=self.lease_id,
        )


def _run_until_done(qual: _ProtectedQualification) -> SupervisorStageResult:
    screen_calls = 0

    def screen():
        nonlocal screen_calls
        screen_calls += 1
        return None

    last = None
    # Enough ticks for one full path plus one crash recovery.
    for _ in range(20):
        supervisor = StandingCpuSupervisor(
            screen_once=screen,
            qualification_once=qual,
            clock=lambda: 1.0,
        )
        try:
            last = supervisor.tick()
        except StandingCpuSupervisorError:
            continue
        if qual.done and last.last_disposition == "done":
            assert screen_calls == 0
            return last
    raise AssertionError(f"did not complete; phase={qual.index} last={last}")


@pytest.mark.parametrize(
    ("crash_before", "label"),
    (
        ("plan", "cpu_crash_before_request_publication"),
        ("publish", "cpu_crash_after_plan_before_request_ready"),
        ("speed", "cpu_crash_while_pod_working"),
        ("quality", "crash_after_bcbp_checkpoint"),
        ("import", "crash_after_gsm8k_checkpoint"),
        ("commit", "crash_after_import_before_db_commit"),
    ),
)
def test_boundary_crash_resumes_same_request_without_duplicate_experiment(
    crash_before: str, label: str
) -> None:
    del label
    qual = _ProtectedQualification()
    qual.crash_before = crash_before
    status = _run_until_done(qual)
    assert status.request_id == _d("canary-request")
    assert status.lease_id == _d("canary-lease")
    # Each expensive/durable stage executes at most once across the crash.
    assert qual.counts == {name: 1 for name in _STAGES}


def test_final_product_resume_is_commit_only() -> None:
    qual = _ProtectedQualification()
    qual.index = _STAGES.index("commit")
    status = _run_until_done(qual)
    assert qual.counts["speed"] == 0
    assert qual.counts["quality"] == 0
    assert qual.counts["commit"] == 1
    assert status.last_disposition == "done"


def test_settlement_before_weights_crash_does_not_republish_settlement() -> None:
    settlements = {"n": 0}
    weights = {"n": 0}
    state = {"settled": False, "crash": True}

    def settle():
        if state["settled"]:
            return None
        settlements["n"] += 1
        state["settled"] = True
        if state["crash"]:
            state["crash"] = False
            raise RuntimeError("simulated crash after settlement before weights")
        return SupervisorStageResult(
            stage="settlement",
            progressed=True,
            disposition="committed",
            lease_id=_d("settlement"),
            phase=SupervisorPhase.SETTLEMENT,
        )

    def publish_weights():
        if not state["settled"]:
            return None
        weights["n"] += 1
        return SupervisorStageResult(
            stage="weights",
            progressed=True,
            disposition="confirmed",
            request_id=_d("projection"),
            phase=SupervisorPhase.WEIGHTS,
        )

    # First process crashes after settlement.
    first = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: None,
        settle_once=settle,
        weights_once=publish_weights,
        clock=lambda: 1.0,
    )
    with pytest.raises(StandingCpuSupervisorError, match="after settlement"):
        first.tick()
    assert settlements["n"] == 1
    assert weights["n"] == 0

    # Restart: settlement is idle (already committed); weights run once.
    second = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: None,
        settle_once=settle,
        weights_once=publish_weights,
        clock=lambda: 1.0,
    )
    status = second.tick()
    assert status.phase is SupervisorPhase.WEIGHTS
    assert settlements["n"] == 1
    assert weights["n"] == 1


def test_hold_never_becomes_silent_idle() -> None:
    status = StandingCpuSupervisor(
        screen_once=lambda: None,
        qualification_once=lambda: SupervisorStageResult(
            stage="qualification",
            progressed=True,
            disposition="hold",
            hold_reason="marker_present",
            phase=SupervisorPhase.HOLD,
        ),
        clock=lambda: 1.0,
    ).tick()
    assert status.phase is SupervisorPhase.HOLD
    assert status.hold_reason == "marker_present"

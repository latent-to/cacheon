"""Standing CPU supervisor: compose public FIFO stages into one forever loop.

Handoff (aug6-2-1 §5) requires one continuously running CPU service that
advances finalized work through screen and same-request qualification without
inventing retry policy, reclaiming terminal rows, or requiring a per-bundle
operator command.

This module owns only the public composition state machine and status surface.
Stage work is delegated to the existing dispatchers:

- screen FIFO → ``RemoteEvaluationDispatcher.dispatch_screen_once``
- qualification FIFO + same-request recovery →
  ``RecoverableQualificationDispatcher.dispatch_once``
- later gates attach settlement / incentives / weights as injectable stages

``chainops`` may launch this process and bind sealed paths; it must not
duplicate recovery or evidence semantics.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable


STATUS_SCHEMA = "cacheon-standing-cpu-supervisor-status-v1"
EVENT_SCHEMA = "cacheon-standing-cpu-supervisor-event-v1"
_TERMINAL_CLAIM_STATUSES = frozenset({"failed", "expired", "qualified"})


class StandingCpuSupervisorError(RuntimeError):
    """Supervisor authority or stage composition failed closed."""


class SupervisorPhase(str, Enum):
    """Coarse public phase for monitoring (not a second recovery authority)."""

    IDLE = "idle"
    SCREEN = "screen"
    QUALIFICATION = "qualification"
    SETTLEMENT = "settlement"
    INCENTIVE = "incentive"
    WEIGHTS = "weights"
    HOLD = "hold"
    FAILED = "failed"


@dataclass(frozen=True)
class SupervisorStatus:
    """One closed status snapshot for operator visibility."""

    phase: SupervisorPhase
    last_progress_unix: float
    request_id: str | None = None
    lease_id: str | None = None
    checkpoint_age_s: float | None = None
    worker_epoch: str | None = None
    lane_assignment: str | None = None
    hold_reason: str | None = None
    last_stage: str | None = None
    last_disposition: str | None = None

    def __post_init__(self) -> None:
        if type(self.phase) is not SupervisorPhase:
            raise StandingCpuSupervisorError("supervisor phase is not exactly typed")
        if (
            type(self.last_progress_unix) is bool
            or type(self.last_progress_unix) not in (int, float)
            or not math.isfinite(float(self.last_progress_unix))
            or float(self.last_progress_unix) < 0
        ):
            raise StandingCpuSupervisorError("last progress time is malformed")
        object.__setattr__(self, "last_progress_unix", float(self.last_progress_unix))
        for name in (
            "request_id",
            "lease_id",
            "worker_epoch",
            "lane_assignment",
            "hold_reason",
            "last_stage",
            "last_disposition",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or value.strip() != value
            ):
                raise StandingCpuSupervisorError(f"supervisor status {name} is malformed")
        if self.checkpoint_age_s is not None and (
            type(self.checkpoint_age_s) is bool
            or type(self.checkpoint_age_s) not in (int, float)
            or not math.isfinite(float(self.checkpoint_age_s))
            or float(self.checkpoint_age_s) < 0
        ):
            raise StandingCpuSupervisorError("checkpoint age is malformed")
        if self.checkpoint_age_s is not None:
            object.__setattr__(self, "checkpoint_age_s", float(self.checkpoint_age_s))

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_age_s": self.checkpoint_age_s,
            "hold_reason": self.hold_reason,
            "lane_assignment": self.lane_assignment,
            "last_disposition": self.last_disposition,
            "last_progress_unix": self.last_progress_unix,
            "last_stage": self.last_stage,
            "lease_id": self.lease_id,
            "phase": self.phase.value,
            "request_id": self.request_id,
            "schema": STATUS_SCHEMA,
            "worker_epoch": self.worker_epoch,
        }


@dataclass(frozen=True)
class SupervisorStageResult:
    """Normalized result from one injectable public stage."""

    stage: str
    progressed: bool
    disposition: str | None = None
    request_id: str | None = None
    lease_id: str | None = None
    lane_assignment: str | None = None
    worker_epoch: str | None = None
    checkpoint_age_s: float | None = None
    hold_reason: str | None = None
    phase: SupervisorPhase | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage, str)
            or not self.stage
            or self.stage.strip() != self.stage
            or type(self.progressed) is not bool
        ):
            raise StandingCpuSupervisorError("supervisor stage result is malformed")
        if self.phase is not None and type(self.phase) is not SupervisorPhase:
            raise StandingCpuSupervisorError("supervisor stage phase is not typed")


# Stage callables: zero-arg, return None (idle), SupervisorStageResult, or a
# legacy EvaluationRun / recoverable Hold|Requeue which the supervisor normalizes.
ScreenOnce = Callable[[], Any]
QualificationOnce = Callable[[], Any]
OptionalStage = Callable[[], Any]


@dataclass
class StandingCpuSupervisor:
    """Compose screen + recoverable qualification (+ later stages) with status."""

    screen_once: ScreenOnce
    qualification_once: QualificationOnce
    settle_once: OptionalStage | None = None
    incentive_once: OptionalStage | None = None
    weights_once: OptionalStage | None = None
    clock: Callable[[], float] = time.time
    stall_timeout_s: float = 3_600.0
    _status: SupervisorStatus = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.screen_once) or not callable(self.qualification_once):
            raise StandingCpuSupervisorError("screen and qualification stages are required")
        for name in ("settle_once", "incentive_once", "weights_once"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise StandingCpuSupervisorError(f"{name} is not callable")
        if (
            type(self.stall_timeout_s) is bool
            or type(self.stall_timeout_s) not in (int, float)
            or not math.isfinite(float(self.stall_timeout_s))
            or float(self.stall_timeout_s) <= 0
        ):
            raise StandingCpuSupervisorError("stall timeout must be a positive finite duration")
        object.__setattr__(self, "stall_timeout_s", float(self.stall_timeout_s))
        if not callable(self.clock):
            raise StandingCpuSupervisorError("clock is not callable")
        self._status = SupervisorStatus(
            phase=SupervisorPhase.IDLE,
            last_progress_unix=float(self.clock()),
        )

    def status(self) -> SupervisorStatus:
        return self._status

    def _observe(self, result: SupervisorStageResult) -> SupervisorStatus:
        now = float(self.clock())
        phase = result.phase
        if phase is None:
            if result.hold_reason:
                phase = SupervisorPhase.HOLD
            elif result.stage == "screen":
                phase = SupervisorPhase.SCREEN
            elif result.stage == "qualification":
                phase = SupervisorPhase.QUALIFICATION
            elif result.stage == "settlement":
                phase = SupervisorPhase.SETTLEMENT
            elif result.stage == "incentive":
                phase = SupervisorPhase.INCENTIVE
            elif result.stage == "weights":
                phase = SupervisorPhase.WEIGHTS
            else:
                phase = SupervisorPhase.IDLE
        progress = now if result.progressed else self._status.last_progress_unix
        self._status = SupervisorStatus(
            phase=phase,
            last_progress_unix=progress,
            request_id=result.request_id,
            lease_id=result.lease_id,
            checkpoint_age_s=result.checkpoint_age_s,
            worker_epoch=result.worker_epoch,
            lane_assignment=result.lane_assignment,
            hold_reason=result.hold_reason,
            last_stage=result.stage,
            last_disposition=result.disposition,
        )
        return self._status

    def _normalize(self, stage: str, raw: Any) -> SupervisorStageResult | None:
        if raw is None:
            return None
        if type(raw) is SupervisorStageResult:
            if raw.stage != stage:
                raise StandingCpuSupervisorError(
                    "stage result names a different stage than the caller"
                )
            return raw

        # Recoverable qualification HOLD / REQUEUE are first-class public products.
        from cacheon.chain.recoverable_qualification_dispatcher import (
            RecoverableQualificationHold,
            RecoverableQualificationRequeue,
        )

        if type(raw) is RecoverableQualificationHold:
            return SupervisorStageResult(
                stage=stage,
                progressed=True,
                disposition="hold",
                request_id=raw.request_id or None,
                lease_id=None,
                hold_reason=raw.reason,
                phase=SupervisorPhase.HOLD,
            )
        if type(raw) is RecoverableQualificationRequeue:
            return SupervisorStageResult(
                stage=stage,
                progressed=True,
                disposition="requeue",
                request_id=raw.request_id,
                lease_id=None,
                phase=SupervisorPhase.QUALIFICATION,
            )

        from cacheon.chain.evaluation_coordinator import EvaluationRun

        if type(raw) is EvaluationRun:
            lane = None
            try:
                lane = raw.lease.screen_lane  # type: ignore[attr-defined]
            except Exception:
                lane = None
            return SupervisorStageResult(
                stage=stage,
                progressed=True,
                disposition=raw.disposition,
                request_id=None,
                lease_id=raw.lease.lease_id,
                lane_assignment=lane if isinstance(lane, str) else None,
            )

        raise StandingCpuSupervisorError(
            f"stage {stage!r} returned an untyped product {type(raw).__name__}"
        )

    def _run_stage(self, stage: str, callback: Callable[[], Any]) -> SupervisorStageResult | None:
        try:
            raw = callback()
        except StandingCpuSupervisorError:
            raise
        except Exception as exc:
            # Never invent COMPLETE/REQUEUE/HOLD from exception class alone.
            self._status = SupervisorStatus(
                phase=SupervisorPhase.FAILED,
                last_progress_unix=self._status.last_progress_unix,
                request_id=self._status.request_id,
                lease_id=self._status.lease_id,
                checkpoint_age_s=self._status.checkpoint_age_s,
                worker_epoch=self._status.worker_epoch,
                lane_assignment=self._status.lane_assignment,
                hold_reason=None,
                last_stage=stage,
                last_disposition="stage_error",
            )
            raise StandingCpuSupervisorError(
                f"stage {stage!r} failed closed without a typed disposition: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        return self._normalize(stage, raw)

    def tick(self) -> SupervisorStatus:
        """Advance at most one unit of work, preferring protected qualification resume.

        Order:
        1. qualification (resume same-request recovery before claiming new screen work)
        2. screen FIFO
        3. optional settlement / incentive / weights stages (later handoff gates)
        """

        # Qualification first: recover active protected leases after restart.
        for stage, callback in (
            ("qualification", self.qualification_once),
            ("screen", self.screen_once),
            ("settlement", self.settle_once),
            ("incentive", self.incentive_once),
            ("weights", self.weights_once),
        ):
            if callback is None:
                continue
            result = self._run_stage(stage, callback)
            if result is not None:
                return self._observe(result)

        now = float(self.clock())
        stalled = now - self._status.last_progress_unix
        if stalled >= self.stall_timeout_s:
            self._status = replace(
                self._status,
                phase=SupervisorPhase.HOLD,
                hold_reason="supervisor_progress_stalled",
                last_stage="idle",
                last_disposition="hold",
            )
            return self._status
        self._status = replace(
            self._status,
            phase=SupervisorPhase.IDLE,
            hold_reason=None,
            last_stage="idle",
            last_disposition=None,
        )
        return self._status


@dataclass(frozen=True)
class FifoQueueTable:
    """Running table for fresh resubmissions only (handoff §10).

    Historical ``expired`` / ``failed`` terminal rows are reported separately and
    are never counted as pending/screening/qualifying work.
    """

    pending: int
    screening: int
    qualifying: int
    hold: int
    miner_pass: int
    miner_fail: int
    settled: int
    incentivized: int
    weight_published: int
    historical_terminal: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise StandingCpuSupervisorError(f"fifo table field {name} is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "historical_terminal": self.historical_terminal,
            "hold": self.hold,
            "incentivized": self.incentivized,
            "miner_fail": self.miner_fail,
            "miner_pass": self.miner_pass,
            "pending": self.pending,
            "qualifying": self.qualifying,
            "schema": "cacheon-standing-fifo-queue-table-v1",
            "screening": self.screening,
            "settled": self.settled,
            "weight_published": self.weight_published,
        }


def build_fifo_queue_table(
    *,
    status_counts: dict[str, int],
    hold: int = 0,
    settled: int = 0,
    incentivized: int = 0,
    weight_published: int = 0,
) -> FifoQueueTable:
    """Project intake status counts into the handoff running table.

    Terminal historical rows (``expired``, ``failed``) contribute only to
    ``historical_terminal`` — never to the active FIFO columns.
    """

    if type(status_counts) is not dict:
        raise StandingCpuSupervisorError("status_counts must be a dict")
    pending = int(status_counts.get("reserved", 0)) + int(
        status_counts.get("published", 0)
    ) + int(status_counts.get("deferred", 0))
    screening = int(status_counts.get("screening", 0))
    qualifying = int(status_counts.get("qualifying", 0)) + int(
        status_counts.get("promoted", 0)
    )
    miner_pass = int(status_counts.get("qualified", 0))
    miner_fail = 0  # fresh FAIL only; historical failed rows stay historical
    historical = int(status_counts.get("expired", 0)) + int(
        status_counts.get("failed", 0)
    )
    return FifoQueueTable(
        pending=pending,
        screening=screening,
        qualifying=qualifying,
        hold=hold,
        miner_pass=miner_pass,
        miner_fail=miner_fail,
        settled=settled,
        incentivized=incentivized,
        weight_published=weight_published,
        historical_terminal=historical,
    )


def refuse_terminal_reclaim(status: str) -> None:
    """Fail closed if a caller attempts to treat a terminal row as claimable work."""

    if status in _TERMINAL_CLAIM_STATUSES:
        raise StandingCpuSupervisorError(
            f"standing supervisor refuses to reclaim terminal status {status!r}"
        )


def settlement_stage(
    *,
    store_factory: Callable[[], Any],
    current_block: Callable[[], int],
    finalized_block_provider: Callable[[], int | tuple[int, str]],
) -> Callable[[], SupervisorStageResult | None]:
    """Wrap ``validator_loop._settle_pending`` as one injectable supervisor stage."""

    def once() -> SupervisorStageResult | None:
        from cacheon.chain.validator_loop import _settle_pending

        with store_factory() as store:
            if not store.has_pending_settlement():
                return None
            committed = _settle_pending(
                store,
                current_block=int(current_block()),
                finalized_block_provider=finalized_block_provider,
            )
        if not committed:
            return None
        lease_id = next(iter(committed))
        return SupervisorStageResult(
            stage="settlement",
            progressed=True,
            disposition="committed",
            lease_id=lease_id,
            phase=SupervisorPhase.SETTLEMENT,
        )

    return once


def incentive_stage(
    activate: Callable[[], Any],
) -> Callable[[], SupervisorStageResult | None]:
    """Wrap one-shot incentive activation; idle when the boundary is not ready.

    ``activate`` should call ``execute_selected_incentive_activation`` (or a test
    double).  Boundary-not-ready errors become idle ``None``; other failures
    fail closed through the supervisor without inventing a disposition.
    """

    def once() -> SupervisorStageResult | None:
        from cacheon.chain.incentive_activation import IncentiveActivationError

        try:
            result = activate()
        except IncentiveActivationError:
            return None
        campaign = getattr(result, "campaign_id", None)
        return SupervisorStageResult(
            stage="incentive",
            progressed=True,
            disposition="activated",
            request_id=campaign if isinstance(campaign, str) and campaign else None,
            phase=SupervisorPhase.INCENTIVE,
        )

    return once


def weights_stage(
    publish: Callable[[], Any],
) -> Callable[[], SupervisorStageResult | None]:
    """Wrap weight project/publish/readback; idle when nothing is due."""

    def once() -> SupervisorStageResult | None:
        result = publish()
        if result is None:
            return None
        digest = getattr(result, "projection_digest", None)
        status = getattr(result, "status", None)
        return SupervisorStageResult(
            stage="weights",
            progressed=True,
            disposition=status if isinstance(status, str) and status else "published",
            request_id=digest if isinstance(digest, str) and digest else None,
            phase=SupervisorPhase.WEIGHTS,
        )

    return once


def run_forever(
    supervisor: StandingCpuSupervisor,
    stop: threading.Event,
    *,
    idle_poll_s: float = 1.0,
    wait: Callable[[float], bool] | None = None,
    on_status: Callable[[SupervisorStatus], None] | None = None,
    restart_initial_backoff_s: float = 1.0,
    restart_max_backoff_s: float = 60.0,
) -> None:
    """Run supervisor ticks until ``stop`` is set; rebuild after stage failures."""

    if type(supervisor) is not StandingCpuSupervisor or not isinstance(
        stop, threading.Event
    ):
        raise StandingCpuSupervisorError("run_forever authorities are not exactly typed")
    if (
        type(idle_poll_s) is bool
        or type(idle_poll_s) not in (int, float)
        or not math.isfinite(float(idle_poll_s))
        or float(idle_poll_s) < 0
    ):
        raise StandingCpuSupervisorError("idle poll duration is malformed")
    waiter = stop.wait if wait is None else wait
    backoff = float(restart_initial_backoff_s)
    while not stop.is_set():
        try:
            status = supervisor.tick()
        except StandingCpuSupervisorError as exc:
            if on_status is not None:
                on_status(supervisor.status())
            if waiter(backoff):
                break
            backoff = min(float(restart_max_backoff_s), backoff * 2.0)
            # Surface the failure without inventing a reclaim/retry disposition.
            del exc
            continue
        backoff = float(restart_initial_backoff_s)
        if on_status is not None:
            on_status(status)
        if status.phase is SupervisorPhase.IDLE:
            if waiter(float(idle_poll_s)):
                break


def _emit_status(status: SupervisorStatus) -> None:
    from cacheon.chain.remote_worker_spool import spool_canonical_json

    payload = dict(status.to_dict())
    payload["event"] = "status"
    payload["schema"] = EVENT_SCHEMA
    payload["time_unix"] = int(time.time())
    sys.stdout.buffer.write(spool_canonical_json(payload) + b"\n")
    sys.stdout.buffer.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "absolute path to a closed standing-supervisor config that names "
            "the screen and recoverable-qualification authorities to compose"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry reserved for sealed composition; stages are loaded from config.

    The production config loader lands with private deploy (handoff §7). Until
    then, tests construct ``StandingCpuSupervisor`` directly with injectable
    stage callables.
    """

    args = build_parser().parse_args(argv)
    print(
        "STANDING-CPU-SUPERVISOR-ERROR: sealed config composition is not "
        f"wired for {args.config!r}; construct StandingCpuSupervisor in-process "
        "or wait for the private deploy gate",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVENT_SCHEMA",
    "STATUS_SCHEMA",
    "FifoQueueTable",
    "StandingCpuSupervisor",
    "StandingCpuSupervisorError",
    "SupervisorPhase",
    "SupervisorStageResult",
    "SupervisorStatus",
    "build_fifo_queue_table",
    "incentive_stage",
    "refuse_terminal_reclaim",
    "run_forever",
    "settlement_stage",
    "weights_stage",
]

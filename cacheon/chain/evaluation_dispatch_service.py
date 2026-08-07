"""Unattended polling around an already-authorized evaluation dispatcher.

This module deliberately owns only loop lifecycle.  Lease ownership, remote
transport, evidence import, and result commits remain the dispatcher's job.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cacheon.chain.evaluation_coordinator import EvaluationRun


SUPPORTED_EVALUATION_STAGES = frozenset({"screen", "qualification"})
MIN_POLL_DELAY_SECONDS = 0.001
MAX_POLL_DELAY_SECONDS = 60.0


class EvaluationDispatchServiceError(RuntimeError):
    """The dispatch loop's local configuration or lifecycle is invalid."""


class EvaluationDispatcher(Protocol):
    """The only dispatcher capability consumed by the unattended loop."""

    def dispatch_once(self, stage: str) -> EvaluationRun | None:
        """Dispatch at most one FIFO item for ``stage``."""


class InterruptibleStop(Protocol):
    """A ``threading.Event``-compatible, injected stop controller."""

    def is_set(self) -> bool:
        """Return whether termination has been requested."""

    def set(self) -> None:
        """Request termination."""

    def wait(self, timeout: float | None = None) -> bool:
        """Wait interruptibly and return whether termination was requested."""


@dataclass(frozen=True)
class EvaluationDispatchServiceConfig:
    """Explicit stages and bounded empty-FIFO polling delay."""

    stages: tuple[str, ...]
    empty_poll_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        delay = self.empty_poll_delay_seconds
        if (
            type(self.stages) is not tuple
            or not self.stages
            or any(type(stage) is not str for stage in self.stages)
            or len(set(self.stages)) != len(self.stages)
            or any(stage not in SUPPORTED_EVALUATION_STAGES for stage in self.stages)
            or type(delay) not in {int, float}
            or not math.isfinite(delay)
            or not MIN_POLL_DELAY_SECONDS <= float(delay) <= MAX_POLL_DELAY_SECONDS
        ):
            raise EvaluationDispatchServiceError(
                "dispatch service stages or empty poll delay are invalid"
            )


@dataclass(frozen=True)
class EvaluationDispatchServiceSnapshot:
    """Monotonic loop progress plus current lifecycle state."""

    revision: int
    dispatch_calls: int
    empty_polls: int
    completed_runs: int
    released_runs: int
    idle_waits: int
    in_flight: bool
    active_stage: str | None
    stop_requested: bool
    failed: bool


class EvaluationDispatchService:
    """Run explicit evaluation stages serially until stopped or failed.

    A complete pass over all configured stages with no returned work is the
    only state that triggers polling backoff.  Dispatcher exceptions terminate
    the loop immediately and are re-raised unchanged.
    """

    def __init__(
        self,
        *,
        dispatcher: EvaluationDispatcher,
        config: EvaluationDispatchServiceConfig,
        stop: InterruptibleStop,
    ) -> None:
        if (
            not callable(getattr(dispatcher, "dispatch_once", None))
            or type(config) is not EvaluationDispatchServiceConfig
            or not callable(getattr(stop, "is_set", None))
            or not callable(getattr(stop, "set", None))
            or not callable(getattr(stop, "wait", None))
        ):
            raise EvaluationDispatchServiceError(
                "dispatch service dependencies are not closed and explicit"
            )
        self._dispatcher = dispatcher
        self._config = config
        self._stop = stop
        self._state_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._revision = 0
        self._dispatch_calls = 0
        self._empty_polls = 0
        self._completed_runs = 0
        self._released_runs = 0
        self._idle_waits = 0
        self._in_flight = False
        self._active_stage: str | None = None
        self._failure: Exception | None = None

    @property
    def failure(self) -> Exception | None:
        """Return the original dispatcher exception, if the loop failed."""

        with self._state_lock:
            return self._failure

    def request_stop(self) -> None:
        """Interrupt an empty-FIFO wait and stop after any in-flight call."""

        self._stop.set()

    def snapshot(self) -> EvaluationDispatchServiceSnapshot:
        """Return an immutable, internally consistent progress snapshot."""

        with self._state_lock:
            return EvaluationDispatchServiceSnapshot(
                revision=self._revision,
                dispatch_calls=self._dispatch_calls,
                empty_polls=self._empty_polls,
                completed_runs=self._completed_runs,
                released_runs=self._released_runs,
                idle_waits=self._idle_waits,
                in_flight=self._in_flight,
                active_stage=self._active_stage,
                stop_requested=self._stop.is_set(),
                failed=self._failure is not None,
            )

    def run(self) -> EvaluationDispatchServiceSnapshot:
        """Poll until stopped; terminate and re-raise on the first exception."""

        if not self._run_lock.acquire(blocking=False):
            raise EvaluationDispatchServiceError("dispatch service is already running")
        empty_stages = 0
        stage_index = 0
        try:
            with self._state_lock:
                prior_failure = self._failure
            if prior_failure is not None:
                raise prior_failure
            while not self._stop.is_set():
                stage = self._config.stages[stage_index]
                stage_index = (stage_index + 1) % len(self._config.stages)
                with self._state_lock:
                    self._dispatch_calls += 1
                    self._in_flight = True
                    self._active_stage = stage
                    self._revision += 1
                try:
                    result = self._dispatcher.dispatch_once(stage)
                finally:
                    with self._state_lock:
                        self._in_flight = False
                        self._active_stage = None
                        self._revision += 1

                if result is None:
                    empty_stages += 1
                    with self._state_lock:
                        self._empty_polls += 1
                        self._revision += 1
                    if empty_stages == len(self._config.stages):
                        empty_stages = 0
                        with self._state_lock:
                            self._idle_waits += 1
                            self._revision += 1
                        self._stop.wait(
                            timeout=float(self._config.empty_poll_delay_seconds)
                        )
                    continue

                disposition = getattr(result, "disposition", None)
                if disposition not in {"completed", "released"}:
                    raise EvaluationDispatchServiceError(
                        "dispatcher returned a malformed evaluation run"
                    )
                empty_stages = 0
                with self._state_lock:
                    if disposition == "completed":
                        self._completed_runs += 1
                    else:
                        self._released_runs += 1
                    self._revision += 1
            return self.snapshot()
        except Exception as exc:
            with self._state_lock:
                if self._failure is None:
                    self._failure = exc
                    self._revision += 1
            raise
        finally:
            self._run_lock.release()


__all__ = [
    "EvaluationDispatchService",
    "EvaluationDispatchServiceConfig",
    "EvaluationDispatchServiceError",
    "EvaluationDispatchServiceSnapshot",
    "EvaluationDispatcher",
    "InterruptibleStop",
    "MAX_POLL_DELAY_SECONDS",
    "MIN_POLL_DELAY_SECONDS",
    "SUPPORTED_EVALUATION_STAGES",
]

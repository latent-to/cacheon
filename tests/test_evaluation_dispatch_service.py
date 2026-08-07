from __future__ import annotations

import builtins
import os
import threading
from dataclasses import dataclass

import pytest

from cacheon.chain.evaluation_dispatch_service import (
    EvaluationDispatchService,
    EvaluationDispatchServiceConfig,
    EvaluationDispatchServiceError,
)


@dataclass(frozen=True)
class _Run:
    disposition: str


class _StopAfterWait:
    def __init__(self) -> None:
        self.stopped = False
        self.waits: list[float | None] = []

    def is_set(self) -> bool:
        return self.stopped

    def set(self) -> None:
        self.stopped = True

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        self.stopped = True
        return True


class _SequenceDispatcher:
    def __init__(self, results: list[_Run | None]) -> None:
        self.results = list(results)
        self.stages: list[str] = []

    def dispatch_once(self, stage: str) -> _Run | None:
        self.stages.append(stage)
        return self.results.pop(0)


def test_empty_fifo_polls_every_explicit_stage_then_waits() -> None:
    stop = _StopAfterWait()
    dispatcher = _SequenceDispatcher([None, None])
    service = EvaluationDispatchService(
        dispatcher=dispatcher,
        config=EvaluationDispatchServiceConfig(
            stages=("screen", "qualification"), empty_poll_delay_seconds=0.25
        ),
        stop=stop,
    )

    snapshot = service.run()

    assert dispatcher.stages == ["screen", "qualification"]
    assert stop.waits == [0.25]
    assert snapshot.dispatch_calls == 2
    assert snapshot.empty_polls == 2
    assert snapshot.idle_waits == 1
    assert snapshot.stop_requested is True


def test_completed_and_released_runs_are_counted_distinctly() -> None:
    stop = _StopAfterWait()
    dispatcher = _SequenceDispatcher(
        [_Run("completed"), _Run("released"), None]
    )
    service = EvaluationDispatchService(
        dispatcher=dispatcher,
        config=EvaluationDispatchServiceConfig(stages=("qualification",)),
        stop=stop,
    )

    snapshot = service.run()

    assert snapshot.dispatch_calls == 3
    assert snapshot.completed_runs == 1
    assert snapshot.released_runs == 1
    assert snapshot.empty_polls == 1
    assert snapshot.revision > snapshot.dispatch_calls


def test_dispatcher_exception_is_exposed_and_never_retried() -> None:
    class DispatchFailure(RuntimeError):
        pass

    failure = DispatchFailure("transport failed")

    class FailingDispatcher:
        calls = 0

        def dispatch_once(self, stage: str) -> None:
            self.calls += 1
            raise failure

    dispatcher = FailingDispatcher()
    service = EvaluationDispatchService(
        dispatcher=dispatcher,
        config=EvaluationDispatchServiceConfig(stages=("screen",)),
        stop=threading.Event(),
    )

    with pytest.raises(DispatchFailure) as caught:
        service.run()

    assert caught.value is failure
    assert service.failure is failure
    assert service.snapshot().failed is True
    assert dispatcher.calls == 1

    with pytest.raises(DispatchFailure) as repeated:
        service.run()
    assert repeated.value is failure
    assert dispatcher.calls == 1


def test_malformed_result_permanently_fails_without_redispatch() -> None:
    dispatcher = _SequenceDispatcher([_Run("unknown"), _Run("completed")])
    service = EvaluationDispatchService(
        dispatcher=dispatcher,
        config=EvaluationDispatchServiceConfig(stages=("qualification",)),
        stop=threading.Event(),
    )

    with pytest.raises(
        EvaluationDispatchServiceError, match="malformed evaluation run"
    ) as first:
        service.run()

    assert service.failure is first.value
    assert service.snapshot().failed is True
    assert dispatcher.stages == ["qualification"]

    with pytest.raises(EvaluationDispatchServiceError) as repeated:
        service.run()
    assert repeated.value is first.value
    assert dispatcher.stages == ["qualification"]


def test_stop_request_interrupts_empty_fifo_wait_promptly() -> None:
    entered_wait = threading.Event()

    class ObservableEvent:
        def __init__(self) -> None:
            self.event = threading.Event()

        def is_set(self) -> bool:
            return self.event.is_set()

        def set(self) -> None:
            self.event.set()

        def wait(self, timeout: float | None = None) -> bool:
            entered_wait.set()
            return self.event.wait(timeout)

    stop = ObservableEvent()
    service = EvaluationDispatchService(
        dispatcher=_SequenceDispatcher([None]),
        config=EvaluationDispatchServiceConfig(
            stages=("screen",), empty_poll_delay_seconds=60.0
        ),
        stop=stop,
    )
    returned: list[object] = []
    thread = threading.Thread(target=lambda: returned.append(service.run()))
    thread.start()
    assert entered_wait.wait(timeout=1.0)

    service.request_stop()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(returned) == 1
    assert service.snapshot().stop_requested is True


@pytest.mark.parametrize(
    "stages,delay",
    [
        ((), 1.0),
        (["screen"], 1.0),
        (("screen", "screen"), 1.0),
        (("unknown",), 1.0),
        (("screen",), 0.0),
        (("screen",), 60.001),
        (("screen",), float("nan")),
        (("screen",), True),
    ],
)
def test_invalid_configuration_is_rejected(
    stages: object, delay: object
) -> None:
    with pytest.raises(EvaluationDispatchServiceError):
        EvaluationDispatchServiceConfig(  # type: ignore[arg-type]
            stages=stages, empty_poll_delay_seconds=delay
        )


def test_parallel_run_attempt_cannot_create_a_second_inflight_dispatch() -> None:
    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()

    class BlockingDispatcher:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def dispatch_once(self, stage: str) -> _Run:
            with self.lock:
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            entered.set()
            assert release.wait(timeout=1.0)
            with self.lock:
                self.active -= 1
            stop.set()
            return _Run("completed")

    dispatcher = BlockingDispatcher()
    service = EvaluationDispatchService(
        dispatcher=dispatcher,
        config=EvaluationDispatchServiceConfig(stages=("qualification",)),
        stop=stop,
    )
    first = threading.Thread(target=service.run)
    first.start()
    assert entered.wait(timeout=1.0)

    with pytest.raises(EvaluationDispatchServiceError, match="already running"):
        service.run()

    release.set()
    first.join(timeout=1.0)
    assert not first.is_alive()
    assert dispatcher.calls == 1
    assert dispatcher.max_active == 1


def test_service_reads_no_ambient_environment_paths_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = _StopAfterWait()
    service = EvaluationDispatchService(
        dispatcher=_SequenceDispatcher([None]),
        config=EvaluationDispatchServiceConfig(stages=("screen",)),
        stop=stop,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("ambient filesystem or environment access")

    # Restore the process globals before pytest itself renders the test result.
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(os, "getenv", forbidden)
        patch.setattr(os, "environ", forbidden)
        snapshot = service.run()
    assert snapshot.empty_polls == 1

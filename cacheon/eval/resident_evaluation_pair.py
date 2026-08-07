"""Service-owned two-lane residency; evaluation policy remains elsewhere.

Each factory starts one resident session and blocks in its driver. Requests
receive only a lane-thread-bound, revocable capability. The same two sessions
remain alive until the sole terminal authority,
:meth:`ResidentEvaluationPair.close`.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from cacheon.eval.oci_resident_session import (
    ResidentBatchEvidence,
    ResidentBatchShape,
    SwapReceipt,
)
from cacheon.stack_identity import require_sha256_hex


class ResidentEvaluationPairError(RuntimeError):
    """The pair API or a supplied resident lifetime is invalid."""


class ResidentEvaluationEpochFatal(RuntimeError):
    """Operation signal that permanently latches this service epoch."""


class ResidentEvaluationPairFailed(ResidentEvaluationPairError):
    """The epoch failed and cannot be silently replaced."""


@dataclass(frozen=True)
class ResidentLaneIdentity:
    lane_id: str
    session_id: str


@dataclass(frozen=True)
class ResidentRequestSlice:
    """Append-only session records produced by exactly one request."""

    request_id: str
    evaluation_id: str
    lane_id: str
    session_id: str
    bundle_digest: str
    expected_batch_count: int
    expected_swap_count: int
    starting_generation: int
    ending_generation: int
    ending_bundle_digest: str | None
    ending_slots: tuple[str, ...]
    new_batches: tuple[ResidentBatchEvidence, ...]
    new_swaps: tuple[SwapReceipt, ...]
    host_started_at: float
    host_completed_at: float

    def __post_init__(self) -> None:
        if (
            not _hex_id(self.request_id)
            or not _hex_id(self.evaluation_id)
            or not _hex_id(self.session_id)
        ):
            raise ResidentEvaluationPairError("request identity is invalid")
        if self.lane_id not in ("A", "B"):
            raise ResidentEvaluationPairError("request lane identity is invalid")
        _digest(self.bundle_digest)
        for name, value in (
            ("expected batch count", self.expected_batch_count),
            ("expected swap count", self.expected_swap_count),
        ):
            if type(value) is not int or value < 0:
                raise ResidentEvaluationPairError(f"request {name} is invalid")
        if (
            len(self.new_batches) > self.expected_batch_count
            or len(self.new_swaps) > self.expected_swap_count
        ):
            raise ResidentEvaluationPairError(
                "request exceeded its declared verb counts"
            )
        if (
            type(self.starting_generation) is not int
            or type(self.ending_generation) is not int
            or not 0 <= self.starting_generation <= self.ending_generation
        ):
            raise ResidentEvaluationPairError("request generation bounds are invalid")
        for row in self.new_batches:
            if type(row) is not ResidentBatchEvidence or not (
                self.starting_generation <= row.generation <= self.ending_generation
            ):
                raise ResidentEvaluationPairError("request batch slice is unbound")
        for row in self.new_swaps:
            if (
                type(row) is not SwapReceipt
                or not (
                    self.starting_generation
                    < row.generation
                    <= self.ending_generation
                )
                or row.bundle_digest not in (None, self.bundle_digest)
            ):
                raise ResidentEvaluationPairError("request swap slice is unbound")
        final = self.new_swaps[-1] if self.new_swaps else None
        if final is None:
            bound = (
                self.ending_generation == self.starting_generation
                and self.ending_bundle_digest is None
                and not self.ending_slots
            )
        else:
            bound = (
                final.generation == self.ending_generation
                and final.bundle_digest == self.ending_bundle_digest
                and final.slots == self.ending_slots
            )
        if not bound or (self.ending_bundle_digest is None) != (not self.ending_slots):
            raise ResidentEvaluationPairError("request ending dispatch is unbound")
        if self.ending_bundle_digest not in (None, self.bundle_digest):
            raise ResidentEvaluationPairError("request ended on another bundle")
        times = (self.host_started_at, self.host_completed_at)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in times
        ) or self.host_completed_at < self.host_started_at:
            raise ResidentEvaluationPairError("request host clock is invalid")


@dataclass(frozen=True)
class ResidentRequestFailure:
    error_type: str
    message: str
    epoch_fatal: bool


@dataclass(frozen=True)
class ResidentRequestResult:
    request_slice: ResidentRequestSlice
    value: object | None
    error: ResidentRequestFailure | None

    def __post_init__(self) -> None:
        if self.error is None and (
            len(self.request_slice.new_batches)
            != self.request_slice.expected_batch_count
            or len(self.request_slice.new_swaps)
            != self.request_slice.expected_swap_count
        ):
            raise ResidentEvaluationPairError(
                "successful request differs from its declared verb counts"
            )

    @property
    def ok(self) -> bool:
        return self.error is None


LifetimeEvidenceT = TypeVar("LifetimeEvidenceT")


@dataclass(frozen=True)
class ResidentLaneRetirementEvidence(Generic[LifetimeEvidenceT]):
    identity: ResidentLaneIdentity
    lifetime_evidence: LifetimeEvidenceT


@dataclass(frozen=True)
class ResidentEvaluationRetirementEvidence(Generic[LifetimeEvidenceT]):
    """Both factory-level lifetime products and the completed request history."""

    lane_a: ResidentLaneRetirementEvidence[LifetimeEvidenceT]
    lane_b: ResidentLaneRetirementEvidence[LifetimeEvidenceT]
    request_history: tuple[ResidentRequestResult, ...]

    def __post_init__(self) -> None:
        if self.lane_a.identity.lane_id != "A" or self.lane_b.identity.lane_id != "B":
            raise ResidentEvaluationPairError("retirement lanes are not canonical")


@dataclass
class _RequestCapability:
    identity: ResidentLaneIdentity
    session: Any
    bundle_digest: str
    request_id: str
    owner_thread_id: int
    batch_start: int
    swap_start: int
    expected_batch_count: int
    expected_swap_count: int


_CAPABILITY_LOCK = threading.Lock()
_CAPABILITIES: dict[str, _RequestCapability] = {}


def _issue_capability(
    identity: ResidentLaneIdentity,
    session: Any,
    work: _Work,
    *,
    batch_start: int,
    swap_start: int,
) -> str:
    token = uuid.uuid4().hex
    state = _RequestCapability(
        identity,
        session,
        work.bundle_digest,
        work.request_id,
        threading.get_ident(),
        batch_start,
        swap_start,
        work.expected_batch_count,
        work.expected_swap_count,
    )
    with _CAPABILITY_LOCK:
        _CAPABILITIES[token] = state
    return token


def _capability(token: str) -> _RequestCapability:
    with _CAPABILITY_LOCK:
        state = _CAPABILITIES.get(token)
        if state is None:
            raise ResidentEvaluationPairError("resident request handle is revoked")
        if state.owner_thread_id != threading.get_ident():
            raise ResidentEvaluationPairError(
                "resident request handle left its lane thread"
            )
        return state


def _revoke_capability(token: str | None) -> None:
    if token is not None:
        with _CAPABILITY_LOCK:
            _CAPABILITIES.pop(token, None)


class ResidentEvaluationHandle:
    """Opaque request capability with no resident session or terminal verbs."""

    __slots__ = ("__token",)

    def __init__(self, token: str) -> None:
        self.__token = token

    @property
    def identity(self) -> ResidentLaneIdentity:
        return _capability(self.__token).identity

    def swap(self, bundle_digest: str | None) -> SwapReceipt:
        state = _capability(self.__token)
        if bundle_digest is not None and bundle_digest != state.bundle_digest:
            raise ResidentEvaluationPairError("request cannot swap another bundle")
        used = len(state.session.swap_receipts) - state.swap_start
        # A candidate activation must leave one declared verb for exact stock
        # restoration, whether the callback or the coordinator performs it.
        required = 1 + int(bundle_digest is not None)
        if used < 0 or used + required > state.expected_swap_count:
            raise ResidentEvaluationEpochFatal(
                "request would exceed its declared swap count"
            )
        return state.session.swap(bundle_digest)

    def execute_batch(
        self, prompts: Sequence[str], *, canary: bool = False
    ) -> ResidentBatchEvidence:
        state = _capability(self.__token)
        used = len(state.session.batch_rows) - state.batch_start
        if used < 0 or used >= state.expected_batch_count:
            raise ResidentEvaluationEpochFatal(
                "request would exceed its declared batch count"
            )
        return state.session.execute_batch(prompts, canary=canary)

    def execute_batch_with_shape(
        self,
        prompts: Sequence[str],
        *,
        shape: ResidentBatchShape,
        canary: bool = False,
    ) -> ResidentBatchEvidence:
        """Issue a typed request-local read without exposing the resident owner."""

        state = _capability(self.__token)
        used = len(state.session.batch_rows) - state.batch_start
        if used < 0 or used >= state.expected_batch_count:
            raise ResidentEvaluationEpochFatal(
                "request would exceed its declared batch count"
            )
        return state.session.execute_batch_with_shape(
            prompts, shape=shape, canary=canary
        )


ResidentOperation = Callable[[ResidentEvaluationHandle], object]
ResidentLifetimeFactory = Callable[[Callable[[Any], object]], LifetimeEvidenceT]
_CLOSE = object()


@dataclass(frozen=True)
class ResidentLaneRequest:
    """One target-neutral request for a lane in :meth:`run_lanes`."""

    bundle_digest: str
    operation: ResidentOperation
    expected_batch_count: int
    expected_swap_count: int


@dataclass
class _Work:
    bundle_digest: str
    operation: ResidentOperation
    expected_batch_count: int
    expected_swap_count: int
    deadline: float
    evaluation_id: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    result: ResidentRequestResult | None = None
    internal_error: BaseException | None = None
    cancellation: ResidentEvaluationEpochFatal | None = None
    capability_token: str | None = None
    completed: bool = False


@dataclass
class _Lane(Generic[LifetimeEvidenceT]):
    lane_id: str
    factory: ResidentLifetimeFactory[LifetimeEvidenceT]
    work: queue.Queue[object] = field(default_factory=queue.Queue)
    ready: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    identity: ResidentLaneIdentity | None = None
    lifetime_evidence: LifetimeEvidenceT | None = None
    error: BaseException | None = None


class ResidentEvaluationPair(Generic[LifetimeEvidenceT]):
    """Own two resident model lifetimes across globally serialized requests."""

    def __init__(
        self,
        lane_a_factory: ResidentLifetimeFactory[LifetimeEvidenceT],
        lane_b_factory: ResidentLifetimeFactory[LifetimeEvidenceT],
        *,
        start_timeout_s: float = 1800.0,
        request_timeout_s: float = 3600.0,
        close_timeout_s: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not all(
            callable(value) for value in (lane_a_factory, lane_b_factory, clock)
        ):
            raise ResidentEvaluationPairError("pair authorities are not callable")
        for name, value in (
            ("start", start_timeout_s),
            ("request", request_timeout_s),
            ("close", close_timeout_s),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
                0 < float(value) <= 86_400
            ):
                raise ResidentEvaluationPairError(f"{name} timeout is invalid")
        self._lanes = {
            "A": _Lane("A", lane_a_factory),
            "B": _Lane("B", lane_b_factory),
        }
        self._timeouts = tuple(
            map(float, (start_timeout_s, request_timeout_s, close_timeout_s))
        )
        self._clock = clock
        self._state = threading.Lock()
        self._lifecycle = threading.Lock()
        self._admission = threading.Lock()
        self._fatal: BaseException | None = None
        self._history: list[ResidentRequestResult] = []
        self._retirement: ResidentEvaluationRetirementEvidence[
            LifetimeEvidenceT
        ] | None = None
        self._started = False
        self._ready = False
        self._closed = False

    @property
    def identities(self) -> tuple[ResidentLaneIdentity, ResidentLaneIdentity]:
        a, b = self._lanes["A"].identity, self._lanes["B"].identity
        if a is None or b is None:
            raise ResidentEvaluationPairError("resident pair is not ready")
        return a, b

    @property
    def fatal_error(self) -> BaseException | None:
        with self._state:
            return self._fatal

    @property
    def request_history(self) -> tuple[ResidentRequestResult, ...]:
        with self._state:
            return tuple(self._history)

    @property
    def retirement_evidence(
        self,
    ) -> ResidentEvaluationRetirementEvidence[LifetimeEvidenceT] | None:
        with self._state:
            return self._retirement

    def start(self) -> tuple[ResidentLaneIdentity, ResidentLaneIdentity]:
        """Start both non-daemon lane owners once and await both sessions."""

        self._reject_lane_reentrancy("start")
        with self._lifecycle:
            with self._state:
                if self._closed:
                    raise ResidentEvaluationPairError("resident pair is closed")
                if not self._started:
                    self._started = True
                    for lane in self._lanes.values():
                        lane.thread = threading.Thread(
                            target=self._run_lifetime,
                            args=(lane,),
                            name=f"cacheon-resident-eval-{lane.lane_id}",
                            daemon=False,
                        )
                        lane.thread.start()
            deadline = self._clock() + self._timeouts[0]
            for lane in self._lanes.values():
                if not lane.ready.wait(max(0.0, deadline - self._clock())):
                    failure = ResidentEvaluationPairFailed(
                        f"resident lane {lane.lane_id} did not become ready"
                    )
                    self._latch(failure)
                    raise failure
            fatal = self.fatal_error
            if fatal is not None:
                raise ResidentEvaluationPairFailed(
                    f"resident epoch failed: {_safe(fatal)}"
                ) from fatal
            for lane in self._lanes.values():
                if (
                    lane.error is not None
                    or lane.thread is None
                    or not lane.thread.is_alive()
                ):
                    failure = lane.error or ResidentEvaluationPairFailed(
                        f"resident lane {lane.lane_id} is not alive"
                    )
                    self._latch(failure)
                    raise ResidentEvaluationPairFailed(
                        f"resident lane {lane.lane_id} failed: {_safe(failure)}"
                    ) from failure
            identities = self.identities
            with self._state:
                if self._closed:
                    raise ResidentEvaluationPairError("resident pair is closed")
                self._ready = True
            return identities

    def run_lane(
        self,
        lane_id: str,
        bundle_digest: str,
        operation: ResidentOperation,
        *,
        expected_batch_count: int,
        expected_swap_count: int,
    ) -> ResidentRequestResult:
        """Admit one exact request; admissions never overlap across the pair."""

        digest = self._validate_request(
            lane_id,
            bundle_digest,
            operation,
            expected_batch_count,
            expected_swap_count,
        )
        self._reject_lane_reentrancy("request admission")
        with self._admission:
            evaluation_id = uuid.uuid4().hex
            work = _Work(
                digest,
                operation,
                expected_batch_count,
                expected_swap_count,
                self._clock() + self._timeouts[1],
                evaluation_id,
            )
            self._accept(self._lanes[lane_id], work)
            return self._await(work)

    def run_lanes(
        self,
        lane_a: ResidentLaneRequest,
        lane_b: ResidentLaneRequest,
    ) -> tuple[ResidentRequestResult, ResidentRequestResult]:
        """Run one A/B evaluation concurrently and publish only a full success."""

        requests = (lane_a, lane_b)
        if any(type(request) is not ResidentLaneRequest for request in requests):
            raise ResidentEvaluationPairError("paired lane requests are invalid")
        digest_a, digest_b = (
            self._validate_request(
                lane_id,
                request.bundle_digest,
                request.operation,
                request.expected_batch_count,
                request.expected_swap_count,
            )
            for lane_id, request in zip(("A", "B"), requests)
        )
        self._reject_lane_reentrancy("paired request admission")
        with self._admission:
            evaluation_id = uuid.uuid4().hex
            deadline = self._clock() + self._timeouts[1]
            work_a, work_b = (
                _Work(
                    digest,
                    request.operation,
                    request.expected_batch_count,
                    request.expected_swap_count,
                    deadline,
                    evaluation_id,
                )
                for digest, request in zip((digest_a, digest_b), requests)
            )
            works = (work_a, work_b)
            self._accept_pair(work_a, work_b)
            try:
                results = (self._await(work_a), self._await(work_b))
            except BaseException as exc:
                failure = self.fatal_error or ResidentEvaluationEpochFatal(
                    f"paired evaluation {evaluation_id} failed: {_safe(exc)}"
                )
                self._cancel_works(works, failure)
                raise
            for result in results:
                if result.error is not None:
                    failure = self.fatal_error or ResidentEvaluationEpochFatal(
                        "paired evaluation "
                        f"{evaluation_id} lane {result.request_slice.lane_id} "
                        f"failed: {result.error.message}"
                    )
                    self._latch(failure)
                    raise ResidentEvaluationPairFailed(_safe(failure)) from failure
            return results

    def close(
        self,
    ) -> ResidentEvaluationRetirementEvidence[LifetimeEvidenceT] | None:
        """Explicitly retire both epochs and return cached lifetime evidence."""

        self._reject_lane_reentrancy("close")
        with self._lifecycle:
            with self._state:
                if self._retirement is not None:
                    return self._retirement
                first = not self._closed
                self._closed = True
                self._ready = False
                started = self._started
                if first and started:
                    for lane in self._lanes.values():
                        lane.work.put(_CLOSE)
            if not started:
                return None
            deadline = self._clock() + self._timeouts[2]
            for lane in self._lanes.values():
                if lane.thread is not None:
                    lane.thread.join(max(0.0, deadline - self._clock()))
                if lane.thread is not None and lane.thread.is_alive():
                    raise ResidentEvaluationPairError(
                        f"resident lane {lane.lane_id} did not close in time"
                    )
            failures = [lane.error for lane in self._lanes.values() if lane.error]
            if failures:
                raise ResidentEvaluationPairError(
                    f"pair close failed: {_safe(failures[0])}"
                ) from failures[0]
            a, b = self._lanes["A"], self._lanes["B"]
            if (
                a.identity is None
                or b.identity is None
                or a.lifetime_evidence is None
                or b.lifetime_evidence is None
            ):
                raise ResidentEvaluationPairError(
                    "pair close produced incomplete lifetime evidence"
                )
            retirement = ResidentEvaluationRetirementEvidence(
                ResidentLaneRetirementEvidence(a.identity, a.lifetime_evidence),
                ResidentLaneRetirementEvidence(b.identity, b.lifetime_evidence),
                self.request_history,
            )
            with self._state:
                self._retirement = retirement
            return retirement

    def _accept(self, lane: _Lane[LifetimeEvidenceT], work: _Work) -> None:
        with self._state:
            self._require_admissible_locked()
            lane.work.put(work)

    def _accept_pair(self, work_a: _Work, work_b: _Work) -> None:
        accepted: list[_Work] = []
        admission_error: tuple[
            ResidentEvaluationEpochFatal, BaseException
        ] | None = None
        with self._state:
            self._require_admissible_locked()
            try:
                for lane, work in (
                    (self._lanes["A"], work_a),
                    (self._lanes["B"], work_b),
                ):
                    lane.work.put(work)
                    accepted.append(work)
            except BaseException as exc:
                failure = ResidentEvaluationEpochFatal(
                    f"paired evaluation admission failed: {_safe(exc)}"
                )
                if self._fatal is None:
                    self._fatal = failure
                admission_error = failure, exc
        if admission_error is not None:
            failure, cause = admission_error
            self._cancel_works(tuple(accepted), failure)
            raise ResidentEvaluationPairFailed(_safe(failure)) from cause

    def _require_admissible_locked(self) -> None:
        if not self._started or not self._ready or self._closed:
            state = "closed" if self._closed else "not ready"
            raise ResidentEvaluationPairError(f"resident pair is {state}")
        if self._fatal is not None:
            raise ResidentEvaluationPairFailed(
                f"resident epoch failed: {_safe(self._fatal)}"
            ) from self._fatal

    def _run_lifetime(self, lane: _Lane[LifetimeEvidenceT]) -> None:
        close_requested = False

        def driver(session: Any) -> object:
            nonlocal close_requested
            _validate_session(session)
            with self._state:
                lane.identity = ResidentLaneIdentity(lane.lane_id, session.session_id)
            lane.ready.set()
            while True:
                item = lane.work.get()
                if item is _CLOSE:
                    close_requested = True
                    return session.finish(allow_empty=True)
                if not isinstance(item, _Work):
                    raise ResidentEvaluationPairError("resident lane work is invalid")
                try:
                    result = self._execute(lane, session, item)
                except BaseException as exc:
                    self._complete_internal_error(item, exc)
                    raise
                else:
                    self._complete_result(item, result)
                    if bool(getattr(session, "closed", False)):
                        terminal = self.fatal_error or ResidentEvaluationEpochFatal(
                            f"resident lane {lane.lane_id} session became terminal"
                        )
                        raise terminal

        try:
            lane.lifetime_evidence = lane.factory(driver)
            if not close_requested:
                raise ResidentEvaluationPairFailed(
                    f"resident lane {lane.lane_id} exited before close"
                )
        except BaseException as exc:
            lane.error = exc
            lane.ready.set()
            if not close_requested:
                self._latch(exc)

    def _execute(
        self, lane: _Lane[LifetimeEvidenceT], session: Any, work: _Work
    ) -> ResidentRequestResult:
        batch_start, swap_start = len(session.batch_rows), len(session.swap_receipts)
        generation_start, started_at = session.active_generation, self._clock()
        value: object | None = None
        error: BaseException | None = None
        fatal = self.fatal_error
        if session.active_bundle_digest is not None or session.active_slots:
            error = ResidentEvaluationEpochFatal(
                f"resident lane {lane.lane_id} did not start on stock"
            )
        elif fatal is not None:
            error = ResidentEvaluationEpochFatal(
                f"resident epoch is failed: {_safe(fatal)}"
            )
        else:
            max_batches, max_swaps = _session_limits(session)
            remaining_batches = max_batches - batch_start
            remaining_swaps = max_swaps - swap_start
            if (
                remaining_batches < work.expected_batch_count
                or remaining_swaps < work.expected_swap_count
            ):
                error = ResidentEvaluationEpochFatal(
                    "resident lane "
                    f"{lane.lane_id} lacks declared request budget "
                    f"(batches {remaining_batches}/{work.expected_batch_count}, "
                    f"swaps {remaining_swaps}/{work.expected_swap_count})"
                )
        token: str | None = None
        if error is None:
            with work.lock:
                if work.cancellation is not None:
                    error = work.cancellation
                else:
                    if lane.identity is None:
                        raise ResidentEvaluationPairError(
                            "lane identity is unavailable"
                        )
                    token = _issue_capability(
                        lane.identity,
                        session,
                        work,
                        batch_start=batch_start,
                        swap_start=swap_start,
                    )
                    work.capability_token = token
            if token is not None:
                try:
                    value = work.operation(ResidentEvaluationHandle(token))
                except BaseException as exc:
                    error = exc
                finally:
                    _revoke_capability(token)
                    with work.lock:
                        if work.capability_token == token:
                            work.capability_token = None
        # Cancellation can race an already-started verb. It revokes subsequent
        # calls immediately; the lane then waits for that verb and seals its
        # eventual complete slice instead of publishing partial mutable lists.
        with work.lock:
            cancellation = work.cancellation
        terminal_before_restore = bool(getattr(session, "closed", False))
        if cancellation is not None and not (
            terminal_before_restore and error is not None
        ):
            error = cancellation
        if not terminal_before_restore:
            try:
                if session.active_bundle_digest is not None or session.active_slots:
                    session.swap(None)
                if session.active_bundle_digest is not None or session.active_slots:
                    raise ResidentEvaluationPairError(
                        "stock restoration was not exact"
                    )
            except BaseException as exc:
                error = ResidentEvaluationEpochFatal(
                    f"resident lane {lane.lane_id} stock restoration failed: "
                    f"{_safe(exc)}"
                )
        terminal = bool(getattr(session, "closed", False))
        if terminal and error is None:
            error = ResidentEvaluationEpochFatal(
                f"resident lane {lane.lane_id} session became terminal"
            )
        actual_batches = len(session.batch_rows) - batch_start
        actual_swaps = len(session.swap_receipts) - swap_start
        counts_exceeded = (
            actual_batches > work.expected_batch_count
            or actual_swaps > work.expected_swap_count
        )
        counts_inexact = (
            actual_batches != work.expected_batch_count
            or actual_swaps != work.expected_swap_count
        )
        if counts_exceeded or (error is None and counts_inexact):
            error = ResidentEvaluationEpochFatal(
                "resident lane "
                f"{lane.lane_id} request verb counts differ "
                f"(batches {actual_batches}/{work.expected_batch_count}, "
                f"swaps {actual_swaps}/{work.expected_swap_count})"
            )
        epoch_fatal = isinstance(error, ResidentEvaluationEpochFatal) or terminal
        request_slice = ResidentRequestSlice(
            work.request_id,
            work.evaluation_id,
            lane.lane_id,
            session.session_id,
            work.bundle_digest,
            work.expected_batch_count,
            work.expected_swap_count,
            generation_start,
            session.active_generation,
            session.active_bundle_digest,
            tuple(session.active_slots),
            tuple(session.batch_rows[batch_start:]),
            tuple(session.swap_receipts[swap_start:]),
            started_at,
            self._clock(),
        )
        failure = None if error is None else ResidentRequestFailure(
            _error_type(error), _safe(error), epoch_fatal
        )
        if epoch_fatal and error is not None:
            self._latch(error)
        return ResidentRequestResult(request_slice, value, failure)

    def _complete_result(self, work: _Work, result: ResidentRequestResult) -> None:
        with work.lock:
            if work.cancellation is not None and (
                result.error is None or not result.error.epoch_fatal
            ):
                result = ResidentRequestResult(
                    result.request_slice,
                    result.value,
                    ResidentRequestFailure(
                        _error_type(work.cancellation),
                        _safe(work.cancellation),
                        True,
                    ),
                )
            work.result = result
            work.completed = True
            with self._state:
                self._history.append(result)
        work.done.set()

    def _complete_internal_error(self, work: _Work, error: BaseException) -> None:
        with work.lock:
            work.internal_error = error
            work.completed = True
        self._latch(error)
        work.done.set()

    def _await(self, work: _Work) -> ResidentRequestResult:
        remaining = max(0.0, work.deadline - self._clock())
        if not work.done.wait(remaining):
            token: str | None = None
            with work.lock:
                if not work.completed:
                    failure = ResidentEvaluationEpochFatal(
                        f"resident request {work.request_id} timed out"
                    )
                    work.cancellation = failure
                    token = work.capability_token
                else:
                    failure = None
            if failure is not None:
                _revoke_capability(token)
                self._latch(failure)
                raise ResidentEvaluationPairFailed(_safe(failure)) from failure
        with work.lock:
            internal_error, result = work.internal_error, work.result
        if internal_error is not None:
            raise ResidentEvaluationPairFailed(
                f"resident coordinator failed: {_safe(internal_error)}"
            ) from internal_error
        if result is None:
            raise ResidentEvaluationPairFailed("resident request returned no result")
        return result

    def _cancel_works(
        self, works: tuple[_Work, ...], failure: BaseException
    ) -> None:
        if not isinstance(failure, ResidentEvaluationEpochFatal):
            failure = ResidentEvaluationEpochFatal(_safe(failure))
        tokens: list[str | None] = []
        for work in works:
            with work.lock:
                if not work.completed:
                    work.cancellation = failure
                    tokens.append(work.capability_token)
        for token in tokens:
            _revoke_capability(token)
        self._latch(failure)

    def _latch(self, error: BaseException) -> None:
        with self._state:
            if self._fatal is None:
                self._fatal = error

    def _reject_lane_reentrancy(self, action: str) -> None:
        current = threading.current_thread()
        if any(lane.thread is current for lane in self._lanes.values()):
            raise ResidentEvaluationPairError(
                f"resident lane cannot perform reentrant {action}"
            )

    @staticmethod
    def _validate_request(
        lane_id: str,
        bundle_digest: str,
        operation: ResidentOperation,
        expected_batch_count: int,
        expected_swap_count: int,
    ) -> str:
        if lane_id not in ("A", "B"):
            raise ResidentEvaluationPairError("resident lane must be A or B")
        if not callable(operation):
            raise ResidentEvaluationPairError("resident operation is not callable")
        for name, value in (
            ("expected batch count", expected_batch_count),
            ("expected swap count", expected_swap_count),
        ):
            if type(value) is not int or value < 0:
                raise ResidentEvaluationPairError(f"{name} is invalid")
        return _digest(bundle_digest)


def _hex_id(value: object) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(
        c in "0123456789abcdef" for c in value
    )


def _digest(value: object) -> str:
    try:
        return require_sha256_hex(value, field="bundle digest")
    except ValueError as exc:
        raise ResidentEvaluationPairError(str(exc)) from None


def _safe(error: BaseException) -> str:
    try:
        message = " ".join(str(error).split())
    except BaseException:
        message = "unprintable error"
    return (message or type(error).__name__)[:512]


def _error_type(error: BaseException) -> str:
    return f"{type(error).__module__}.{type(error).__qualname__}"[:256]


def _session_limits(session: Any) -> tuple[int, int]:
    plan = getattr(session, "plan", None)
    max_batches = getattr(plan, "max_batches", None)
    max_swaps = getattr(plan, "max_swaps", None)
    if (
        type(max_batches) is not int
        or max_batches < 0
        or type(max_swaps) is not int
        or max_swaps < 0
    ):
        raise ResidentEvaluationPairError("resident session plan budgets are invalid")
    return max_batches, max_swaps


def _validate_session(session: Any) -> None:
    bundle, slots = getattr(session, "active_bundle_digest", None), getattr(
        session, "active_slots", None
    )
    try:
        max_batches, max_swaps = _session_limits(session)
    except ResidentEvaluationPairError:
        raise ResidentEvaluationPairError("resident lane session is invalid") from None
    if (
        not _hex_id(getattr(session, "session_id", None))
        or type(getattr(session, "active_generation", None)) is not int
        or session.active_generation < 0
        or not isinstance(getattr(session, "batch_rows", None), list)
        or not isinstance(getattr(session, "swap_receipts", None), list)
        or len(session.batch_rows) > max_batches
        or len(session.swap_receipts) > max_swaps
        or (bundle is not None and _digest(bundle) != bundle)
        or not isinstance(slots, tuple)
        or any(not isinstance(slot, str) or not slot for slot in slots)
        or (bundle is None) != (not slots)
        or type(getattr(session, "closed", None)) is not bool
        or not all(
            callable(getattr(session, name, None))
            for name in ("swap", "execute_batch", "finish")
        )
    ):
        raise ResidentEvaluationPairError("resident lane session is invalid")

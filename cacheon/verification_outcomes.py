"""Typed verifier outcomes and the strict collective-rank evidence wire.

Diagnostic text remains useful to operators, but it is not a grading interface.
This module gives ordinary and distributed verifiers one causal, typed description
of which eager/capture/replay observations actually completed and who can safely be
held responsible for a failed observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from cacheon.capabilities import CallDescriptor


class PhaseDisposition(str, Enum):
    """Closed disposition vocabulary for one verifier execution phase."""

    NOT_RUN = "NOT_RUN"
    PASSED = "PASSED"
    CANDIDATE_FAILED = "CANDIDATE_FAILED"
    INFRASTRUCTURE_FAILED = "INFRASTRUCTURE_FAILED"


class VerificationCaseKind(str, Enum):
    """Closed verifier case vocabulary; sequence kinds are order-sensitive."""

    ORDINARY_SINGLE = "ordinary_single"
    COLLECTIVE_SINGLE = "collective_single"
    COLLECTIVE_TEMPORAL_EAGER = "collective_temporal_eager"
    COLLECTIVE_GRAPH_SEQUENCE = "collective_graph_sequence"


@dataclass(frozen=True)
class VerificationCaseDescriptor:
    """Canonical binding from one result row to its exact ordered live calls.

    ``CallDescriptor`` is the sole call representation: it already owns field
    canonicalization, so this class only enforces what a lone call cannot —
    the sealed execution context is present, exact, and identical across the
    ordered sequence, and the call count agrees with the case kind.
    """

    slot_id: str
    variant_id: str
    case_kind: VerificationCaseKind
    calls: tuple[CallDescriptor, ...]

    def __post_init__(self) -> None:
        for field_name in ("slot_id", "variant_id"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError(f"{field_name} must be a nonempty canonical string")
        if type(self.case_kind) is not VerificationCaseKind:
            raise TypeError("case_kind is not exact")
        calls = tuple(self.calls)
        if not calls:
            raise ValueError("verification case requires at least one call descriptor")
        single_kinds = {
            VerificationCaseKind.ORDINARY_SINGLE,
            VerificationCaseKind.COLLECTIVE_SINGLE,
        }
        if (self.case_kind in single_kinds) != (len(calls) == 1):
            raise ValueError("verification case kind disagrees with ordered call count")
        required = {"dtype", "architecture", "tp_size", "world_size", "graph_mode"}
        sealed_context = None
        for call in calls:
            if type(call) is not CallDescriptor:
                raise TypeError("ordered calls must be exact CallDescriptor objects")
            values = call.as_dict()
            if not required.issubset(values):
                raise ValueError("call descriptor is missing sealed execution context")
            if any(
                type(values[name]) is not str
                for name in ("dtype", "architecture", "graph_mode")
            ) or any(
                type(values[name]) is not int or values[name] < 1
                for name in ("tp_size", "world_size")
            ):
                raise ValueError("call descriptor execution context has invalid types")
            current_context = tuple((name, values[name]) for name in sorted(required))
            if sealed_context is None:
                sealed_context = current_context
            elif current_context != sealed_context:
                raise ValueError("ordered calls disagree on sealed execution context")
        object.__setattr__(self, "calls", calls)

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": "cacheon.verification-case-descriptor.v1",
            "slot_id": self.slot_id,
            "variant_id": self.variant_id,
            "case_kind": self.case_kind.value,
            "calls": [
                {"ordinal": ordinal, "descriptor": call.as_dict()}
                for ordinal, call in enumerate(self.calls)
            ],
        }

    @property
    def digest(self) -> str:
        raw = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class GraphPhaseOutcome:
    """Causal eager/capture/replay facts for one validator-controlled shape.

    ``replay_count`` preserves the verifier's historical meaning: execution
    failures count only earlier successful replays, while a numerically checked
    failing replay is included.  Infrastructure failure is deliberately
    incomplete and never candidate-attributable.
    """

    eager: PhaseDisposition
    capture: PhaseDisposition
    replay: PhaseDisposition
    replay_count: int
    observation_complete: bool

    def __post_init__(self) -> None:
        for name in ("eager", "capture", "replay"):
            if type(getattr(self, name)) is not PhaseDisposition:
                raise TypeError(f"{name} disposition is not exact")
        if type(self.replay_count) is not int or self.replay_count < 0:
            raise ValueError("replay_count must be a nonnegative integer")
        if type(self.observation_complete) is not bool:
            raise TypeError("observation_complete must be an exact boolean")
        if self.eager is not PhaseDisposition.PASSED and (
            self.capture is not PhaseDisposition.NOT_RUN
            or self.replay is not PhaseDisposition.NOT_RUN
            or self.replay_count
        ):
            raise ValueError("capture/replay cannot precede a passing eager phase")
        if self.capture is not PhaseDisposition.PASSED and (
            self.replay is not PhaseDisposition.NOT_RUN or self.replay_count
        ):
            raise ValueError("replay cannot precede a successful capture")
        if self.replay is PhaseDisposition.NOT_RUN and self.replay_count:
            raise ValueError("a non-run replay phase cannot carry replay coverage")
        if self.replay is PhaseDisposition.PASSED and self.replay_count < 1:
            raise ValueError("a passing replay phase requires replay coverage")
        dispositions = (self.eager, self.capture, self.replay)
        has_infrastructure_failure = (
            PhaseDisposition.INFRASTRUCTURE_FAILED in dispositions
        )
        has_candidate_failure = PhaseDisposition.CANDIDATE_FAILED in dispositions
        if has_infrastructure_failure and self.observation_complete:
            raise ValueError("infrastructure failure cannot be a complete observation")
        if has_candidate_failure and not self.observation_complete:
            raise ValueError("candidate failure must be a complete observation")
        if (
            self.observation_complete
            and self.capture is PhaseDisposition.PASSED
            and self.replay is PhaseDisposition.NOT_RUN
        ):
            raise ValueError("successful capture without replay is incomplete")

    @property
    def eager_passed(self) -> bool:
        return self.eager is PhaseDisposition.PASSED

    @property
    def capture_succeeded(self) -> bool:
        return self.capture is PhaseDisposition.PASSED

    @property
    def replay_passed(self) -> bool:
        return self.replay is PhaseDisposition.PASSED

    @property
    def failure_is_candidate_attributable(self) -> bool:
        return self.observation_complete and PhaseDisposition.CANDIDATE_FAILED in (
            self.eager,
            self.capture,
            self.replay,
        )

    @classmethod
    def unobserved(cls) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.NOT_RUN,
            PhaseDisposition.NOT_RUN,
            PhaseDisposition.NOT_RUN,
            0,
            False,
        )

    @classmethod
    def not_applicable(cls) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.NOT_RUN,
            PhaseDisposition.NOT_RUN,
            PhaseDisposition.NOT_RUN,
            0,
            True,
        )

    @classmethod
    def infrastructure_before_eager(cls) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.INFRASTRUCTURE_FAILED,
            PhaseDisposition.NOT_RUN,
            PhaseDisposition.NOT_RUN,
            0,
            False,
        )

    @classmethod
    def eager_only_passed(cls) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.PASSED,
            PhaseDisposition.NOT_RUN,
            PhaseDisposition.NOT_RUN,
            0,
            True,
        )

    @classmethod
    def eager_candidate_failed(cls) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.CANDIDATE_FAILED,
            PhaseDisposition.NOT_RUN,
            PhaseDisposition.NOT_RUN,
            0,
            True,
        )

    @classmethod
    def capture_infrastructure_failed(cls) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.PASSED,
            PhaseDisposition.INFRASTRUCTURE_FAILED,
            PhaseDisposition.NOT_RUN,
            0,
            False,
        )

    @classmethod
    def capture_candidate_failed(cls) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.PASSED,
            PhaseDisposition.CANDIDATE_FAILED,
            PhaseDisposition.NOT_RUN,
            0,
            True,
        )

    @classmethod
    def replay_infrastructure_failed(cls, replay_count: int) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.PASSED,
            PhaseDisposition.PASSED,
            PhaseDisposition.INFRASTRUCTURE_FAILED,
            replay_count,
            False,
        )

    @classmethod
    def replay_candidate_failed(cls, replay_count: int) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.PASSED,
            PhaseDisposition.PASSED,
            PhaseDisposition.CANDIDATE_FAILED,
            replay_count,
            True,
        )

    @classmethod
    def graph_passed(cls, replay_count: int) -> "GraphPhaseOutcome":
        return cls(
            PhaseDisposition.PASSED,
            PhaseDisposition.PASSED,
            PhaseDisposition.PASSED,
            replay_count,
            True,
        )


@dataclass
class _GraphCheck:
    check: Any
    outcome: GraphPhaseOutcome

    @property
    def replays(self) -> int:
        return self.outcome.replay_count


@dataclass(frozen=True)
class _GraphReplayCase:
    inputs: dict
    expected: list[Any]


class _CandidateGraphCallError(RuntimeError):
    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original


def _graph_exception_check(prefix, exc, outcome, output_check_type) -> _GraphCheck:
    original = exc.original if isinstance(exc, _CandidateGraphCallError) else exc
    return _GraphCheck(
        output_check_type(
            False,
            float("inf"),
            float("inf"),
            0.0,
            f"{prefix}: {type(original).__name__}: {original}",
            "ratio",
        ),
        outcome,
    )


class _CandidateCollectivePhaseError(RuntimeError):
    def __init__(self, phase: str, original: Exception, replay_count: int = 0):
        super().__init__(str(original))
        self.phase = phase
        self.original = original
        self.replay_count = replay_count


def _terminate_processes(processes, *, grace_s: float = 5.0) -> None:
    import signal
    import time

    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            if not process.is_alive() or process.pid is None:
                continue
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, sig)
                elif sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
            except OSError:
                process.terminate() if sig == signal.SIGTERM else process.kill()
        deadline = time.monotonic() + grace_s
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))


def _match_detail(match) -> str:
    return "; ".join(
        f"{row.field} {row.reason}: expected {row.expected}"
        + ("" if row.actual is None else f", got {row.actual!r}")
        for row in match.mismatches
    )


def conservative_phase_aggregate(
    outcomes: Iterable[GraphPhaseOutcome],
    *,
    expected_count: int,
    infrastructure_failure: bool = False,
) -> GraphPhaseOutcome:
    """Aggregate rank-local facts without turning partial rank state into blame."""

    rows = tuple(outcomes)
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    if type(infrastructure_failure) is not bool:
        raise TypeError("infrastructure_failure must be an exact boolean")
    if (
        infrastructure_failure
        or len(rows) != expected_count
        or any(type(row) is not GraphPhaseOutcome for row in rows)
    ):
        return GraphPhaseOutcome.infrastructure_before_eager()
    if any(not row.observation_complete for row in rows):
        if all(row.eager_passed for row in rows):
            if all(row.capture_succeeded for row in rows):
                return GraphPhaseOutcome.replay_infrastructure_failed(
                    min(row.replay_count for row in rows)
                )
            return GraphPhaseOutcome.capture_infrastructure_failed()
        return GraphPhaseOutcome.infrastructure_before_eager()
    eager = {row.eager for row in rows}
    if eager == {PhaseDisposition.NOT_RUN}:
        return GraphPhaseOutcome.not_applicable()
    if PhaseDisposition.CANDIDATE_FAILED in eager:
        return GraphPhaseOutcome.eager_candidate_failed()
    if eager != {PhaseDisposition.PASSED}:
        return GraphPhaseOutcome.infrastructure_before_eager()
    capture = {row.capture for row in rows}
    if capture == {PhaseDisposition.NOT_RUN}:
        return GraphPhaseOutcome.eager_only_passed()
    if PhaseDisposition.CANDIDATE_FAILED in capture:
        return GraphPhaseOutcome.capture_candidate_failed()
    if capture != {PhaseDisposition.PASSED}:
        return GraphPhaseOutcome.capture_infrastructure_failed()
    replay = {row.replay for row in rows}
    replay_counts = {row.replay_count for row in rows}
    if PhaseDisposition.CANDIDATE_FAILED in replay:
        return GraphPhaseOutcome.replay_candidate_failed(
            min(row.replay_count for row in rows)
        )
    if replay != {PhaseDisposition.PASSED} or len(replay_counts) != 1:
        return GraphPhaseOutcome.replay_infrastructure_failed(
            min(row.replay_count for row in rows)
        )
    return GraphPhaseOutcome.graph_passed(rows[0].replay_count)


@dataclass
class ShapeResult:
    shape: dict
    dtype: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    pass_ratio: float = 1.0
    detail: str = ""
    metric: str = "ratio"
    graph_replays: int = 0
    applicable: bool = True
    phase_outcome: GraphPhaseOutcome = field(
        default_factory=GraphPhaseOutcome.unobserved
    )
    case_descriptor: VerificationCaseDescriptor | None = None
    # {kernel_name: launches} observed on the device inside the candidate's own
    # invocation window. Diagnostic only, and empty off CUDA.
    kernels: dict = field(default_factory=dict)


@dataclass
class VerifyResult:
    slot: str
    dtype: str
    passed: bool
    shape_results: list[ShapeResult]
    graph_required: bool = False
    graph_verified: bool = False
    coverage_required: int = 0
    context_inapplicable: bool = False
    domain_coverage_complete: bool = True
    domain_coverage_detail: str = ""

    @property
    def num_failed(self) -> int:
        return sum(1 for row in self.shape_results if row.applicable and not row.passed)

    @property
    def num_applicable(self) -> int:
        return sum(1 for row in self.shape_results if row.applicable)

    @property
    def num_not_applicable(self) -> int:
        return len(self.shape_results) - self.num_applicable

    @property
    def coverage_sufficient(self) -> bool:
        return self.coverage_required == 0 or self.num_applicable >= self.coverage_required

    @property
    def fully_verified(self) -> bool:
        return self.passed and (not self.graph_required or self.graph_verified)


_VERDICT_VERSION = 2
_MAX_VERDICT_BYTES = 64 * 1024
_MAX_DETAIL_CHARS = 4096
_MAX_ERROR_CHARS = 16 * 1024
_VERDICT_FIELDS = frozenset(
    {
        "version",
        "rank",
        "world_size",
        "passed",
        "score",
        "max_abs",
        "detail",
        "metric",
        "error",
        "graph_replays",
        "phase_outcome",
    }
)
_PHASE_FIELDS = frozenset(
    {"eager", "capture", "replay", "replay_count", "observation_complete"}
)
_VERDICT_METRICS = frozenset({"ratio", "cosine", "overlap"})


class CollectiveVerdictError(RuntimeError):
    pass


@dataclass(frozen=True)
class _RankVerdict:
    version: int
    rank: int
    world_size: int
    passed: bool
    score: float
    max_abs: float | None
    detail: str
    metric: str
    error: str | None
    graph_replays: int
    phase_outcome: GraphPhaseOutcome


def _number(value: Any, lo: float, hi: float | None = None, *, integer=False):
    valid_type = type(value) is int if integer else (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    number = value if integer else float(value) if valid_type else float("nan")
    if (
        not valid_type
        or not math.isfinite(number)
        or number < lo
        or (hi is not None and number > hi)
    ):
        raise CollectiveVerdictError("numeric field is outside its permitted range")
    return number


def _regular_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise CollectiveVerdictError("verdict destination is not a regular file")
    return int(info.st_dev), int(info.st_ino)


def _bound_fd(path: Path, flags: int, identity: tuple[int, int]):
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or (int(info.st_dev), int(info.st_ino)) != identity
    ):
        os.close(fd)
        raise CollectiveVerdictError(
            "verdict destination inode changed or is not regular"
        )
    return fd, info


def _phase_to_wire(outcome: GraphPhaseOutcome) -> dict[str, object]:
    return {
        "eager": outcome.eager.value,
        "capture": outcome.capture.value,
        "replay": outcome.replay.value,
        "replay_count": outcome.replay_count,
        "observation_complete": outcome.observation_complete,
    }


def _write_rank_verdict(
    path: Path, verdict: _RankVerdict, expected_identity: tuple[int, int]
) -> None:
    value = {
        "version": verdict.version,
        "rank": verdict.rank,
        "world_size": verdict.world_size,
        "passed": verdict.passed,
        "score": verdict.score,
        "max_abs": verdict.max_abs,
        "detail": verdict.detail,
        "metric": verdict.metric,
        "error": verdict.error,
        "graph_replays": verdict.graph_replays,
        "phase_outcome": _phase_to_wire(verdict.phase_outcome),
    }
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not body or len(body) > _MAX_VERDICT_BYTES:
        raise CollectiveVerdictError("rank verdict exceeds the wire-size limit")
    fd, _ = _bound_fd(path, os.O_WRONLY | os.O_TRUNC, expected_identity)
    with os.fdopen(fd, "wb") as stream:
        if stream.write(body) != len(body):
            raise CollectiveVerdictError("short verdict write")
        stream.flush()
        os.fsync(stream.fileno())


def _reject_constant(value: str) -> None:
    raise CollectiveVerdictError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise CollectiveVerdictError("duplicate JSON key")
    return dict(pairs)


def _phase_from_wire(value: object) -> GraphPhaseOutcome:
    if type(value) is not dict or set(value) != _PHASE_FIELDS:
        raise CollectiveVerdictError("phase outcome fields do not match the exact schema")
    try:
        eager = PhaseDisposition(value["eager"])
        capture = PhaseDisposition(value["capture"])
        replay = PhaseDisposition(value["replay"])
    except (TypeError, ValueError) as exc:
        raise CollectiveVerdictError("phase outcome has an unsupported disposition") from exc
    replay_count = _number(value["replay_count"], 0, 64, integer=True)
    observation_complete = value["observation_complete"]
    if type(observation_complete) is not bool:
        raise CollectiveVerdictError("observation_complete must be a boolean")
    try:
        return GraphPhaseOutcome(
            eager, capture, replay, replay_count, observation_complete
        )
    except (TypeError, ValueError) as exc:
        raise CollectiveVerdictError(f"phase outcome is causally invalid: {exc}") from exc


def _read_rank_verdict(
    path: Path,
    *,
    expected_rank: int,
    expected_world_size: int,
    expected_identity: tuple[int, int],
) -> _RankVerdict:
    if _regular_identity(path) != expected_identity:
        raise CollectiveVerdictError("verdict destination inode changed")
    fd, info = _bound_fd(path, os.O_RDONLY, expected_identity)
    size = int(info.st_size)
    if not 1 <= size <= _MAX_VERDICT_BYTES:
        os.close(fd)
        raise CollectiveVerdictError(f"verdict size {size} is outside the limit")
    try:
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(_MAX_VERDICT_BYTES + 1)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (CollectiveVerdictError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectiveVerdictError(f"invalid verdict JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _VERDICT_FIELDS:
        raise CollectiveVerdictError("verdict fields do not match the exact schema")
    version = _number(
        value["version"], _VERDICT_VERSION, _VERDICT_VERSION, integer=True
    )
    rank = _number(value["rank"], 0, expected_world_size - 1, integer=True)
    world_size = _number(value["world_size"], 1, integer=True)
    if rank != expected_rank or world_size != expected_world_size:
        raise CollectiveVerdictError("rank identity mismatch")
    passed = value["passed"]
    if type(passed) is not bool:
        raise CollectiveVerdictError("passed must be a boolean")
    score = _number(value["score"], -1.0, 1.0)
    max_abs = None if value["max_abs"] is None else _number(value["max_abs"], 0.0)
    detail = value["detail"]
    if not isinstance(detail, str) or len(detail) > _MAX_DETAIL_CHARS:
        raise CollectiveVerdictError("detail must be a bounded string")
    metric = value["metric"]
    if metric not in _VERDICT_METRICS:
        raise CollectiveVerdictError(f"unsupported verdict metric {metric!r}")
    error = value["error"]
    if error is not None and (
        not isinstance(error, str) or len(error) > _MAX_ERROR_CHARS
    ):
        raise CollectiveVerdictError("error must be null or a bounded string")
    graph_replays = _number(value["graph_replays"], 0, 64, integer=True)
    phase_outcome = _phase_from_wire(value["phase_outcome"])
    if graph_replays != phase_outcome.replay_count:
        raise CollectiveVerdictError("graph_replays disagrees with typed replay_count")
    if passed and phase_outcome.failure_is_candidate_attributable:
        raise CollectiveVerdictError("passing verdict claims candidate phase failure")
    phase_success = (
        phase_outcome.observation_complete
        and phase_outcome.eager_passed
        and (
            phase_outcome.capture is PhaseDisposition.NOT_RUN
            or phase_outcome.replay_passed
        )
    )
    if phase_success and not passed:
        raise CollectiveVerdictError("failing verdict carries a successful phase outcome")
    if passed and (error is not None or max_abs is None):
        raise CollectiveVerdictError("passing verdict has missing output or an error")
    return _RankVerdict(
        version,
        rank,
        world_size,
        passed,
        score,
        max_abs,
        detail,
        metric,
        error,
        graph_replays,
        phase_outcome,
    )


__all__ = [
    "CollectiveVerdictError",
    "GraphPhaseOutcome",
    "PhaseDisposition",
    "ShapeResult",
    "VerificationCaseDescriptor",
    "VerificationCaseKind",
    "VerifyResult",
    "conservative_phase_aggregate",
]

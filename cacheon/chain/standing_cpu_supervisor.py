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
- later gates attach settlement / weights as injectable stages

``chainops`` may launch this process and bind sealed paths; it must not
duplicate recovery or evidence semantics.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from functools import partial
from cacheon.chain import sealed_config
from cacheon.chain.standing_weights_stage import (
    WeightsStageConfig,
    compose_weight_offer_push,
    load_weights_config,
)


STATUS_SCHEMA = "cacheon-standing-cpu-supervisor-status-v1"
EVENT_SCHEMA = "cacheon-standing-cpu-supervisor-event-v1"
CONFIG_SCHEMA = "cacheon-standing-supervisor-config-v1"
CONFIG_DOMAIN = "cacheon.chain.standing-supervisor-config.v1"
_STANDING_CONFIG_FIELDS = frozenset(
    {
        "enable_qualification",
        "enable_settlement",
        "enable_weights",
        "idle_poll_ms",
        "qualification_evidence_root",
        "qualification_incumbent_stack_path",
        "qualification_incumbent_tree_digest",
        "restart_initial_backoff_ms",
        "restart_max_backoff_ms",
        "schema",
        "screen_dispatcher_config",
        "settlement_network",
        "stall_timeout_ms",
        "weights_stage_config",
    }
)


class StandingCpuSupervisorError(RuntimeError):
    """Supervisor authority or stage composition failed closed."""


class SupervisorPhase(str, Enum):
    """Coarse public phase for monitoring (not a second recovery authority)."""

    IDLE = "idle"
    SCREEN = "screen"
    QUALIFICATION = "qualification"
    SETTLEMENT = "settlement"
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
    qualification_once: QualificationOnce | None
    settle_once: OptionalStage | None = None
    weights_once: OptionalStage | None = None
    clock: Callable[[], float] = time.time
    stall_timeout_s: float = 3_600.0
    _status: SupervisorStatus = field(init=False, repr=False)
    _last_tick_progressed: bool = field(init=False, repr=False)
    _commission_required: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.screen_once):
            raise StandingCpuSupervisorError("the screen stage is required")
        for name in ("qualification_once", "settle_once", "weights_once"):
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
        self._last_tick_progressed = False
        self._commission_required = False

    def status(self) -> SupervisorStatus:
        return self._status

    def _observe(self, result: SupervisorStageResult) -> SupervisorStatus:
        now = float(self.clock())
        self._last_tick_progressed = result.progressed
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
            CompletedQualificationHold,
            RecoverableQualificationHold,
            RecoverableQualificationRequeue,
        )

        if type(raw) is CompletedQualificationHold:
            return SupervisorStageResult(
                stage=stage,
                progressed=True,
                disposition="hold",
                request_id=raw.request_id,
                lease_id=raw.lease.lease_id,
                hold_reason=raw.reason,
                phase=SupervisorPhase.HOLD,
            )
        if type(raw) is RecoverableQualificationHold:
            return SupervisorStageResult(
                stage=stage,
                progressed=False,
                disposition="hold",
                request_id=raw.request_id or None,
                lease_id=None,
                hold_reason=raw.reason,
                phase=SupervisorPhase.HOLD,
            )
        if type(raw) is RecoverableQualificationRequeue:
            return SupervisorStageResult(
                stage=stage,
                progressed=False,
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
            self._last_tick_progressed = False
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
        """Advance one unit, settling before a new qualification may claim."""

        if self._commission_required:
            return self._observe(
                SupervisorStageResult(
                    stage="settlement",
                    progressed=False,
                    disposition="commission_required",
                    hold_reason="baseline_commission_required",
                )
            )
        deferred: SupervisorStageResult | None = None
        for stage, callback in (
            ("settlement", self.settle_once),
            ("qualification", self.qualification_once),
            ("screen", self.screen_once),
            ("weights", self.weights_once),
        ):
            if callback is None:
                continue
            result = self._run_stage(stage, callback)
            if result is None:
                continue
            if result.progressed:
                if stage == "settlement":
                    self._commission_required = True
                return self._observe(result)
            if deferred is None:
                deferred = result
        if deferred is not None:
            return self._observe(deferred)

        now = float(self.clock())
        self._last_tick_progressed = False
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


def settlement_stage(
    *,
    open_store: Callable[[], tuple[Any, tuple[int, str]]],
    finalized_block_provider: Callable[[], int | tuple[int, str]],
) -> Callable[[], SupervisorStageResult | None]:
    """Wrap ``validator_loop._settle_pending`` as one injectable supervisor stage."""

    def once() -> SupervisorStageResult | None:
        from cacheon.chain.validator_loop import _settle_pending

        store, point = open_store()
        try:
            if not store.has_pending_settlement():
                return None
            committed = _settle_pending(
                store,
                current_block=point[0],
                finalized_block_provider=finalized_block_provider,
            )
        finally:
            store.close()
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
    """Run until stopped; a stage exception exits before another paid dispatch."""

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
            print(
                f"STANDING-CPU-SUPERVISOR-STAGE-ERROR: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if on_status is not None:
                on_status(supervisor.status())
            raise
        if on_status is not None:
            on_status(status)
        if status.phase is SupervisorPhase.IDLE:
            backoff = float(restart_initial_backoff_s)
            if waiter(float(idle_poll_s)):
                break
        elif not supervisor._last_tick_progressed:
            if waiter(backoff):
                break
            backoff = min(float(restart_max_backoff_s), backoff * 2.0)
        else:
            backoff = float(restart_initial_backoff_s)


def _emit_status(status: SupervisorStatus) -> None:
    """Emit one operator status line.

    Status is monitoring output, not a sealed digest authority, so integer
    millisecond fields are used (canonical JSON forbids floats).
    """

    import json

    payload = dict(status.to_dict())
    payload["event"] = "status"
    payload["schema"] = EVENT_SCHEMA
    payload["time_unix"] = int(time.time())
    payload["last_progress_unix_ms"] = int(round(float(status.last_progress_unix) * 1000.0))
    del payload["last_progress_unix"]
    if status.checkpoint_age_s is not None:
        payload["checkpoint_age_ms"] = int(round(float(status.checkpoint_age_s) * 1000.0))
    payload.pop("checkpoint_age_s", None)
    sys.stdout.buffer.write(
        (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    sys.stdout.buffer.flush()


def _closed_config(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise StandingCpuSupervisorError(f"{label} fields are not closed")
    return value


_absolute_path = partial(sealed_config.absolute_path, error=StandingCpuSupervisorError)
_authority_file = partial(sealed_config.authority_file, error=StandingCpuSupervisorError)
_private_directory = partial(sealed_config.private_directory, error=StandingCpuSupervisorError)
_positive_int = partial(sealed_config.positive_int, error=StandingCpuSupervisorError)


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise StandingCpuSupervisorError(f"{label} must be a boolean")
    return value


@dataclass(frozen=True)
class StandingSupervisorConfig:
    """Closed standing-supervisor composition authority."""

    raw: dict[str, Any]
    screen_dispatcher_config: Path
    qualification_evidence_root: Path
    qualification_incumbent_stack_path: Path
    qualification_incumbent_tree_digest: str
    enable_weights: bool
    enable_settlement: bool
    enable_qualification: bool
    settlement_network: str
    weights_stage: WeightsStageConfig | None
    stall_timeout_s: float
    idle_poll_s: float
    restart_initial_backoff_s: float
    restart_max_backoff_s: float

    @property
    def digest(self) -> str:
        from cacheon.stack_identity import canonical_digest

        return canonical_digest(CONFIG_DOMAIN, self.raw)


def load_standing_config(path: str | os.PathLike[str]) -> StandingSupervisorConfig:
    """Strictly reopen one immutable standing-supervisor authority file."""

    from cacheon.chain.remote_worker_spool import load_json
    from cacheon.stack_identity import require_sha256_hex

    config_path = _absolute_path(os.fspath(path), "config path")
    _authority_file(config_path, "standing config")
    try:
        raw = load_json(config_path)
    except Exception as exc:
        raise StandingCpuSupervisorError(
            f"standing config cannot reopen: {exc}"
        ) from None
    row = _closed_config(raw, _STANDING_CONFIG_FIELDS, "standing supervisor config")
    if row["schema"] != CONFIG_SCHEMA:
        raise StandingCpuSupervisorError("standing supervisor config schema is unsupported")

    screen_path = _absolute_path(
        row["screen_dispatcher_config"], "screen_dispatcher_config"
    )
    _authority_file(screen_path, "screen dispatcher config")
    evidence_root = _absolute_path(
        row["qualification_evidence_root"], "qualification_evidence_root"
    )
    _private_directory(evidence_root, "qualification_evidence_root")
    incumbent_path = _absolute_path(
        row["qualification_incumbent_stack_path"],
        "qualification_incumbent_stack_path",
    )
    _authority_file(incumbent_path, "qualification incumbent stack")
    try:
        tree_digest = require_sha256_hex(
            row["qualification_incumbent_tree_digest"],
            field="qualification incumbent tree digest",
        )
    except (TypeError, ValueError) as exc:
        raise StandingCpuSupervisorError(str(exc)) from None

    enable_weights = _exact_bool(row["enable_weights"], "enable_weights")
    enable_settlement = _exact_bool(row["enable_settlement"], "enable_settlement")
    # Operator gate for the qualification stage: screens keep draining while a
    # qualification-side defect is being repaired, instead of the broken stage
    # consuming every claim window.
    enable_qualification = _exact_bool(
        row["enable_qualification"], "enable_qualification"
    )
    # Settlement is chain-independent arithmetic over already-durable PASS
    # pairs, but it is *clocked* by the finalized head: ``_settle_pending``
    # refuses a regressed clock and stamps the cohort lease with it. That read
    # is the authority this stage was waiting for, so the flag is only honored
    # together with the endpoint it reads from. No wallet and no chain write is
    # involved -- publication stays behind ``enable_weights``.
    settlement_network = row["settlement_network"]
    if type(settlement_network) is not str:
        raise StandingCpuSupervisorError("settlement network is malformed")
    if enable_settlement and not settlement_network.strip():
        raise StandingCpuSupervisorError(
            "enable_settlement requires settlement_network, the finalized-head "
            "endpoint that clocks the settlement cohort"
        )
    # A configured endpoint with the flag off is explicitly allowed, so a
    # commission can stage the endpoint and arming settlement stays a one-field
    # flip. Refusing it would push the endpoint back onto the operator at every
    # epoch, which is exactly how the 2026-08-16 expiry_blocks decision was
    # lost: a per-epoch artifact edited in place, then orphaned by the next
    # commission regenerating it from defaults.

    weights_stage_config = row["weights_stage_config"]
    if type(weights_stage_config) is not str:
        raise StandingCpuSupervisorError("weights_stage_config is malformed")
    if enable_weights and not weights_stage_config.strip():
        raise StandingCpuSupervisorError(
            "enable_weights requires weights_stage_config, the sealed eval "
            "push-weight-offer authority"
        )
    if not enable_weights and weights_stage_config:
        raise StandingCpuSupervisorError(
            "weights_stage_config is configured while enable_weights is false"
        )
    weights_stage = (
        load_weights_config(weights_stage_config) if enable_weights else None
    )

    stall_timeout_ms = _positive_int(
        row["stall_timeout_ms"], "stall_timeout_ms", maximum=86_400_000
    )
    idle_poll_ms = _positive_int(row["idle_poll_ms"], "idle_poll_ms", maximum=60_000)
    restart_initial_backoff_ms = _positive_int(
        row["restart_initial_backoff_ms"],
        "restart_initial_backoff_ms",
        maximum=600_000,
    )
    restart_max_backoff_ms = _positive_int(
        row["restart_max_backoff_ms"],
        "restart_max_backoff_ms",
        maximum=600_000,
    )
    if restart_initial_backoff_ms > restart_max_backoff_ms:
        raise StandingCpuSupervisorError(
            "restart initial backoff exceeds its maximum"
        )

    return StandingSupervisorConfig(
        raw=dict(row),
        screen_dispatcher_config=screen_path,
        qualification_evidence_root=evidence_root,
        qualification_incumbent_stack_path=incumbent_path,
        qualification_incumbent_tree_digest=tree_digest,
        enable_weights=enable_weights,
        enable_settlement=enable_settlement,
        enable_qualification=enable_qualification,
        settlement_network=settlement_network,
        weights_stage=weights_stage,
        stall_timeout_s=stall_timeout_ms / 1000.0,
        idle_poll_s=idle_poll_ms / 1000.0,
        restart_initial_backoff_s=restart_initial_backoff_ms / 1000.0,
        restart_max_backoff_s=restart_max_backoff_ms / 1000.0,
    )


def build_standing_supervisor(
    config: StandingSupervisorConfig,
) -> StandingCpuSupervisor:
    """Compose screen + recoverable qualification from sealed standing config."""

    from cacheon.chain.mainnet_screen_dispatcher import (
        build_dispatcher,
        load_config,
    )
    from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
    from cacheon.chain.recoverable_qualification_dispatcher import (
        RecoverableQualificationDispatcher,
    )
    from cacheon.chain.remote_worker_spool import load_json
    from cacheon.stack_manifest import EvaluationStackManifest

    if type(config) is not StandingSupervisorConfig:
        raise StandingCpuSupervisorError("standing supervisor config is not typed")

    screen_config = load_config(config.screen_dispatcher_config)
    screen_dispatcher = build_dispatcher(
        screen_config,
        store_factory=RecoverableFinalizedIntakeStore,
    )
    qualification_once = None
    if config.enable_qualification:
        try:
            incumbent_raw = load_json(config.qualification_incumbent_stack_path)
            incumbent = EvaluationStackManifest.from_dict(incumbent_raw)
        except Exception as exc:
            raise StandingCpuSupervisorError(
                f"qualification incumbent stack cannot reopen: {exc}"
            ) from None
        qualification_dispatcher = RecoverableQualificationDispatcher(
            coordinator=screen_dispatcher.coordinator,
            transport=screen_dispatcher.transport,
            credential=screen_dispatcher.credential,
            qualification_evidence_root=config.qualification_evidence_root,
            qualification_incumbent_stack=incumbent,
            qualification_incumbent_tree_digest=(
                config.qualification_incumbent_tree_digest
            ),
        )
        qualification_once = qualification_dispatcher.dispatch_once

    settle_once = None
    if config.enable_settlement:
        from cacheon import chain

        # One long-lived read-only connection for the whole process, the same
        # shape the intake loop uses. ``retry_forever`` keeps an endpoint blip
        # from tearing down a supervisor that is mid-qualification; the stage
        # itself is skipped whenever nothing is pending, so a slow head read
        # never sits in the screen/qualification path.
        subtensor = chain.connect(config.settlement_network, retry_forever=True)
        settle_once = settlement_stage(
            open_store=screen_dispatcher.coordinator._open_at_durable_cursor,
            finalized_block_provider=lambda: chain.read_finalized_head(subtensor),
        )

    weights_once = None
    if config.enable_weights:
        if config.weights_stage is None:
            raise StandingCpuSupervisorError(
                "enable_weights is set without a sealed weights stage"
            )
        weights_once = compose_weight_offer_push(
            config.weights_stage,
            store_factory=partial(
                RecoverableFinalizedIntakeStore,
                screen_config.intake_db,
                screen_config.policy,
                scope=screen_config.scope,
            ),
            scope=screen_config.scope,
        )

    return StandingCpuSupervisor(
        screen_once=screen_dispatcher.dispatch_screen_once,
        qualification_once=qualification_once,
        settle_once=settle_once,
        weights_once=weights_once,
        stall_timeout_s=config.stall_timeout_s,
    )


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
    """Load sealed standing config and run the composed forever loop."""

    args = build_parser().parse_args(argv)
    try:
        config = load_standing_config(args.config)
        supervisor = build_standing_supervisor(config)
    except StandingCpuSupervisorError as exc:
        print(
            f"STANDING-CPU-SUPERVISOR-ERROR: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            "STANDING-CPU-SUPERVISOR-ERROR: sealed composition failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    stop = threading.Event()

    def _stop(_signum: int, _frame: object | None) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    run_forever(
        supervisor,
        stop,
        idle_poll_s=config.idle_poll_s,
        restart_initial_backoff_s=config.restart_initial_backoff_s,
        restart_max_backoff_s=config.restart_max_backoff_s,
        on_status=_emit_status,
    )
    return 0


if __name__ == "__main__":
    # ``python -m`` executes this file as ``__main__``. The weights stage
    # imports this module by its canonical name at composition time; without
    # this alias that import loads a second copy whose SupervisorStageResult
    # and StandingCpuSupervisorError are different classes, and every landed
    # push is then rejected as "an untyped product" (mainnet, 2026-08-19).
    sys.modules.setdefault("cacheon.chain.standing_cpu_supervisor", sys.modules[__name__])
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "EVENT_SCHEMA",
    "STATUS_SCHEMA",
    "StandingCpuSupervisor",
    "StandingCpuSupervisorError",
    "StandingSupervisorConfig",
    "SupervisorPhase",
    "SupervisorStageResult",
    "SupervisorStatus",
    "build_standing_supervisor",
    "load_standing_config",
    "run_forever",
    "settlement_stage",
    "weights_stage",
]

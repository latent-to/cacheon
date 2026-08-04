#!/usr/bin/env python3
"""Standing CPU dispatcher for finalized, FIFO, remote screen work.

The intake-only validator remains the sole chain reader and durable cursor
advancer.  This process reopens that cursor read-only for every coordinator
operation, claims exactly one durable screen lease, and hands the resulting
typed request to the authenticated spool transport.  Qualification is not an
operation exposed by this daemon.

There is deliberately no provider import, command, argv, shell, environment,
or candidate-selected execution surface.  ``ArenaService`` still requires a
provider object, so the CPU installs a digest-exact remote-only proxy whose two
execution methods always fail closed if local code reaches them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sqlite3
import stat
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn
from urllib.parse import quote

from cacheon.arena_service import (
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    ScreenStagePolicy,
    ServingShape,
    WorkloadMixture,
    WorkloadRegime,
)
from cacheon.chain.evaluation_coordinator import (
    EvaluationCoordinator,
    EvaluationRun,
    WorkerReadiness,
)
from cacheon.chain.intake import IntakePolicy, IntakeScope
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteEvaluationDispatcher,
)
from cacheon.stack_identity import canonical_digest, require_sha256_hex

if __package__:
    from chainops import remote_worker_service as worker_service
else:  # Direct ``python chainops/mainnet_screen_dispatcher.py`` execution.
    import remote_worker_service as worker_service


CONFIG_SCHEMA = "cacheon-mainnet-screen-dispatcher-config-v1"
CONFIG_DOMAIN = "cacheon.chain.mainnet-screen-dispatcher-config.v1"
_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_OWNER_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_CONFIG_FIELDS = frozenset(
    {
        "arena_service_manifest",
        "credential_digest",
        "credential_path",
        "heartbeat_interval_ms",
        "heartbeat_join_timeout_ms",
        "idle_poll_ms",
        "intake_db",
        "intake_policy",
        "intake_scope",
        "lease_blocks",
        "lock_attempts",
        "lock_retry_delay_ms",
        "owner",
        "registration_digest",
        "registration_path",
        "response_timeout_seconds",
        "restart_initial_backoff_ms",
        "restart_max_backoff_ms",
        "schema",
        "spool_root",
        "transport_identity_digest",
        "transport_poll_seconds",
        "worker_readiness",
    }
)
_RUNTIME_FIELDS = frozenset(ArenaRuntimeIdentity.__dataclass_fields__)
_CAPACITY_FIELDS = frozenset(ArenaCapacityPolicy.__dataclass_fields__)
_POLICY_FIELDS = frozenset(IntakePolicy.__dataclass_fields__)
_SCOPE_FIELDS = frozenset(IntakeScope.__dataclass_fields__)
_READINESS_FIELDS = frozenset(WorkerReadiness.__dataclass_fields__)
_MANIFEST_FIELDS = frozenset(ArenaServiceManifest.__dataclass_fields__)
_WORKLOAD_FIELDS = frozenset(
    {"prompt_corpus_digest", "prompt_seed_scheme", "regimes"}
)
_REGIME_FIELDS = frozenset({"name", "phase", "shapes", "weight_ppm"})
_SHAPE_FIELDS = frozenset(ServingShape.__dataclass_fields__)
_SCREENS_FIELDS = frozenset({"crownable", "stages"})
_STAGE_FIELDS = frozenset(ScreenStagePolicy.__dataclass_fields__)


class MainnetScreenDispatcherError(RuntimeError):
    """The standing screen dispatcher cannot preserve its closed authority."""


def _closed(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise MainnetScreenDispatcherError(f"{label} fields are not closed")
    return value


def _positive_int(
    value: object,
    label: str,
    *,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value <= 0 or (
        maximum is not None and value > maximum
    ):
        raise MainnetScreenDispatcherError(f"{label} is outside its integer bounds")
    return value


def _digest(value: object, label: str) -> str:
    try:
        return require_sha256_hex(value, field=label)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MainnetScreenDispatcherError(str(exc)) from None


def _absolute_path(value: object, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not Path(value).is_absolute()
    ):
        raise MainnetScreenDispatcherError(f"{label} must be an absolute path")
    return Path(value)


def _authority_file(path: Path, label: str, *, secret: bool = False) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise MainnetScreenDispatcherError(f"{label} is unavailable: {exc}") from None
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
        or (secret and stat.S_IMODE(info.st_mode) & 0o077)
        or (not secret and stat.S_IMODE(info.st_mode) & 0o022)
    ):
        qualifier = "owner-only regular file" if secret else "owner-controlled regular file"
        raise MainnetScreenDispatcherError(f"{label} must be an {qualifier}")


def _private_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise MainnetScreenDispatcherError(f"{label} is unavailable: {exc}") from None
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
    ):
        raise MainnetScreenDispatcherError(
            f"{label} must be an owner-controlled mode-0700 directory"
        )


def _manifest_from_dict(value: object) -> ArenaServiceManifest:
    row = _closed(value, _MANIFEST_FIELDS, "arena service manifest")
    runtime = ArenaRuntimeIdentity(
        **_closed(row["runtime"], _RUNTIME_FIELDS, "arena runtime")
    )

    workload_row = _closed(row["workload"], _WORKLOAD_FIELDS, "arena workload")
    regimes_raw = workload_row["regimes"]
    if type(regimes_raw) is not list:
        raise MainnetScreenDispatcherError("arena workload regimes are not a list")
    regimes: list[WorkloadRegime] = []
    for index, raw_regime in enumerate(regimes_raw):
        regime = _closed(raw_regime, _REGIME_FIELDS, f"arena regime {index}")
        shapes_raw = regime["shapes"]
        if type(shapes_raw) is not list:
            raise MainnetScreenDispatcherError(
                f"arena regime {index} shapes are not a list"
            )
        shapes = tuple(
            ServingShape(
                **_closed(shape, _SHAPE_FIELDS, f"arena regime {index} shape")
            )
            for shape in shapes_raw
        )
        regimes.append(
            WorkloadRegime(
                name=regime["name"],
                phase=regime["phase"],
                weight_ppm=regime["weight_ppm"],
                shapes=shapes,
            )
        )
    workload = WorkloadMixture(
        workload_row["prompt_corpus_digest"],
        workload_row["prompt_seed_scheme"],
        tuple(regimes),
    )

    capacity = ArenaCapacityPolicy(
        **_closed(row["capacity"], _CAPACITY_FIELDS, "arena capacity")
    )
    screens_row = _closed(row["screens"], _SCREENS_FIELDS, "screen policy")
    if screens_row["crownable"] is not False or type(screens_row["stages"]) is not list:
        raise MainnetScreenDispatcherError(
            "screen policy must be explicitly non-crown and list its stages"
        )
    screens = NonCrownScreenPolicy(
        tuple(
            ScreenStagePolicy(
                **_closed(stage, _STAGE_FIELDS, "screen stage policy")
            )
            for stage in screens_row["stages"]
        )
    )
    manifest = ArenaServiceManifest(
        runtime=runtime,
        workload=workload,
        capacity=capacity,
        screens=screens,
        qualification_policy_digest=row["qualification_policy_digest"],
        provider_digest=row["provider_digest"],
        schema_version=row["schema_version"],
    )
    if manifest.to_dict() != row:
        raise MainnetScreenDispatcherError(
            "arena service manifest did not reopen byte-semantically"
        )
    return manifest


class RemoteOnlyArenaProvider:
    """Digest-exact ArenaService proxy with no local execution authority."""

    def __init__(self, provider_digest: str):
        self.provider_digest = _digest(provider_digest, "provider_digest")

    @staticmethod
    def _local_execution_disabled() -> NoReturn:
        raise MainnetScreenDispatcherError(
            "local arena provider execution is disabled; only authenticated "
            "remote screens are allowed"
        )

    def run_screen(self, _manifest, _stage, _candidate) -> NoReturn:
        self._local_execution_disabled()

    def build_qualification(self, _request, state=None) -> NoReturn:
        del state
        self._local_execution_disabled()


@dataclass(frozen=True)
class DispatcherConfig:
    raw: dict[str, Any]
    intake_db: Path
    spool_root: Path
    registration_path: Path
    credential_path: Path
    scope: IntakeScope
    policy: IntakePolicy
    manifest: ArenaServiceManifest
    readiness: WorkerReadiness
    owner: str
    lease_blocks: int
    heartbeat_interval_s: float
    heartbeat_join_timeout_s: float
    lock_attempts: int
    lock_retry_delay_s: float
    idle_poll_s: float
    response_timeout_seconds: int
    transport_poll_seconds: int
    restart_initial_backoff_s: float
    restart_max_backoff_s: float
    registration_digest: str
    transport_identity_digest: str
    credential_digest: str

    @property
    def digest(self) -> str:
        return canonical_digest(CONFIG_DOMAIN, self.raw)


def load_config(path: str | os.PathLike[str]) -> DispatcherConfig:
    """Strictly reopen one immutable deployment authority file."""

    config_path = _absolute_path(os.fspath(path), "config path")
    _authority_file(config_path, "config")
    try:
        raw = worker_service.load_json(config_path)
    except Exception as exc:
        raise MainnetScreenDispatcherError(f"config cannot reopen: {exc}") from None
    row = _closed(raw, _CONFIG_FIELDS, "dispatcher config")
    if row["schema"] != CONFIG_SCHEMA:
        raise MainnetScreenDispatcherError("dispatcher config schema is unsupported")

    intake_db = _absolute_path(row["intake_db"], "intake_db")
    spool_root = _absolute_path(row["spool_root"], "spool_root")
    registration_path = _absolute_path(
        row["registration_path"], "registration_path"
    )
    credential_path = _absolute_path(row["credential_path"], "credential_path")
    _authority_file(registration_path, "registration")
    _authority_file(credential_path, "credential", secret=True)
    _private_directory(spool_root, "spool_root")

    scope_raw = _closed(row["intake_scope"], _SCOPE_FIELDS, "intake scope")
    scope = IntakeScope(**scope_raw)
    if scope.to_dict() != scope_raw:
        raise MainnetScreenDispatcherError("intake scope did not reopen exactly")
    policy_raw = _closed(row["intake_policy"], _POLICY_FIELDS, "intake policy")
    policy = IntakePolicy(**policy_raw)
    if {name: getattr(policy, name) for name in policy.__dataclass_fields__} != policy_raw:
        raise MainnetScreenDispatcherError("intake policy did not reopen exactly")

    manifest = _manifest_from_dict(row["arena_service_manifest"])
    readiness_raw = _closed(
        row["worker_readiness"], _READINESS_FIELDS, "worker readiness"
    )
    readiness = WorkerReadiness(**readiness_raw)
    if readiness.to_dict() != readiness_raw:
        raise MainnetScreenDispatcherError("worker readiness did not reopen exactly")

    owner = row["owner"]
    if (
        not isinstance(owner, str)
        or not owner
        or len(owner) > 256
        or owner.strip() != owner
        or _OWNER_CONTROL.search(owner) is not None
    ):
        raise MainnetScreenDispatcherError("evaluation owner is malformed")

    lease_blocks = _positive_int(row["lease_blocks"], "lease_blocks")
    if lease_blocks > policy.expiry_blocks:
        raise MainnetScreenDispatcherError("lease exceeds the intake expiry policy")
    heartbeat_interval_ms = _positive_int(
        row["heartbeat_interval_ms"], "heartbeat_interval_ms", maximum=3_600_000
    )
    heartbeat_join_timeout_ms = _positive_int(
        row["heartbeat_join_timeout_ms"],
        "heartbeat_join_timeout_ms",
        maximum=3_600_000,
    )
    lock_attempts = _positive_int(row["lock_attempts"], "lock_attempts", maximum=1000)
    lock_retry_delay_ms = _positive_int(
        row["lock_retry_delay_ms"], "lock_retry_delay_ms", maximum=60_000
    )
    idle_poll_ms = _positive_int(row["idle_poll_ms"], "idle_poll_ms", maximum=60_000)
    response_timeout_seconds = _positive_int(
        row["response_timeout_seconds"],
        "response_timeout_seconds",
        maximum=worker_service.MAX_JOB_SECONDS,
    )
    transport_poll_seconds = _positive_int(
        row["transport_poll_seconds"], "transport_poll_seconds", maximum=30
    )
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
        raise MainnetScreenDispatcherError(
            "restart initial backoff exceeds its maximum"
        )

    return DispatcherConfig(
        raw=dict(row),
        intake_db=intake_db,
        spool_root=spool_root,
        registration_path=registration_path,
        credential_path=credential_path,
        scope=scope,
        policy=policy,
        manifest=manifest,
        readiness=readiness,
        owner=owner,
        lease_blocks=lease_blocks,
        heartbeat_interval_s=heartbeat_interval_ms / 1000.0,
        heartbeat_join_timeout_s=heartbeat_join_timeout_ms / 1000.0,
        lock_attempts=lock_attempts,
        lock_retry_delay_s=lock_retry_delay_ms / 1000.0,
        idle_poll_s=idle_poll_ms / 1000.0,
        response_timeout_seconds=response_timeout_seconds,
        transport_poll_seconds=transport_poll_seconds,
        restart_initial_backoff_s=restart_initial_backoff_ms / 1000.0,
        restart_max_backoff_s=restart_max_backoff_ms / 1000.0,
        registration_digest=_digest(
            row["registration_digest"], "registration_digest"
        ),
        transport_identity_digest=_digest(
            row["transport_identity_digest"], "transport_identity_digest"
        ),
        credential_digest=_digest(row["credential_digest"], "credential_digest"),
    )


class LiveFinalizedCursor:
    """Read the intake-only daemon's durable cursor without taking its flock."""

    def __init__(self, intake_db: Path, scope: IntakeScope):
        self.intake_db = intake_db
        self.scope = scope
        self._last: tuple[int, str] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _metadata_value(value: object, label: str) -> object:
        if not isinstance(value, str):
            raise MainnetScreenDispatcherError(f"durable {label} is not JSON text")

        def reject_constant(raw: str) -> NoReturn:
            raise MainnetScreenDispatcherError(
                f"durable {label} contains non-integer number {raw}"
            )

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, child in pairs:
                if key in result:
                    raise MainnetScreenDispatcherError(
                        f"durable {label} contains duplicate key {key}"
                    )
                result[key] = child
            return result

        try:
            return json.loads(
                value,
                object_pairs_hook=reject_duplicates,
                parse_float=reject_constant,
                parse_constant=reject_constant,
            )
        except MainnetScreenDispatcherError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MainnetScreenDispatcherError(
                f"durable {label} is malformed: {exc}"
            ) from None

    def _read(self) -> tuple[int, str]:
        _authority_file(self.intake_db, "intake database", secret=True)
        uri = f"file:{quote(str(self.intake_db), safe='/')}?mode=ro"
        try:
            database = sqlite3.connect(
                uri,
                uri=True,
                isolation_level=None,
                timeout=5.0,
            )
            database.execute("PRAGMA query_only=ON")
            database.execute("PRAGMA busy_timeout=5000")
            database.execute("BEGIN")
            rows = dict(
                database.execute(
                    "SELECT key,value FROM metadata WHERE key IN "
                    "('finalized_cursor','intake_scope')"
                ).fetchall()
            )
            database.execute("COMMIT")
        except sqlite3.Error as exc:
            raise MainnetScreenDispatcherError(
                f"live finalized cursor cannot reopen read-only: {exc}"
            ) from None
        finally:
            if "database" in locals():
                database.close()
        if set(rows) != {"finalized_cursor", "intake_scope"}:
            raise MainnetScreenDispatcherError(
                "live finalized cursor or intake scope is missing"
            )
        scope = self._metadata_value(rows["intake_scope"], "intake scope")
        if scope != self.scope.to_dict():
            raise MainnetScreenDispatcherError(
                "live intake database differs from the configured chain scope"
            )
        cursor = self._metadata_value(rows["finalized_cursor"], "finalized cursor")
        if (
            type(cursor) is not list
            or len(cursor) != 2
            or type(cursor[0]) is not int
            or cursor[0] < 0
            or not isinstance(cursor[1], str)
            or _BLOCK_HASH.fullmatch(cursor[1]) is None
        ):
            raise MainnetScreenDispatcherError("live finalized cursor is malformed")
        return cursor[0], cursor[1]

    def __call__(self) -> tuple[int, str]:
        with self._lock:
            current = self._read()
            previous = self._last
            if previous is not None and (
                current[0] < previous[0]
                or (current[0] == previous[0] and current[1] != previous[1])
            ):
                raise MainnetScreenDispatcherError(
                    "live finalized cursor regressed or changed hash"
                )
            self._last = current
            return current


def _registration(config: DispatcherConfig) -> dict[str, Any]:
    try:
        registration = worker_service.verify_registration(
            worker_service.load_json(config.registration_path)
        )
    except Exception as exc:
        raise MainnetScreenDispatcherError(
            f"worker registration cannot reopen: {exc}"
        ) from None
    expected = {
        "credential_digest": config.credential_digest,
        "registration_digest": config.registration_digest,
        "transport_identity_digest": config.transport_identity_digest,
        "worker_readiness_digest": config.readiness.digest,
    }
    for field, value in expected.items():
        if registration[field] != value:
            raise MainnetScreenDispatcherError(
                f"worker registration {field} differs from dispatcher config"
            )
    if (
        registration["worker_readiness"] != config.readiness.to_dict()
        or registration["service_identity"] != config.manifest.service_id
        or registration["credential_path"] != str(config.credential_path)
    ):
        raise MainnetScreenDispatcherError(
            "worker registration differs from configured readiness/service/credential"
        )
    return registration


def build_dispatcher(config: DispatcherConfig) -> RemoteEvaluationDispatcher:
    """Construct the exact CPU coordinator and durable spool adapter."""

    registration = _registration(config)
    provider = RemoteOnlyArenaProvider(config.manifest.provider_digest)
    service = ArenaService(config.manifest, provider)
    config.readiness.validate(service)
    cursor = LiveFinalizedCursor(config.intake_db, config.scope)
    # Do not advertise a reconstructed dispatcher until the independently
    # advancing intake authority has a present, correctly scoped live cursor.
    cursor()
    coordinator = EvaluationCoordinator(
        intake_db=config.intake_db,
        policy=config.policy,
        scope=config.scope,
        service=service,
        readiness=config.readiness,
        owner=config.owner,
        advance_finalized_cursor=cursor,
        lease_blocks=config.lease_blocks,
        heartbeat_interval_s=config.heartbeat_interval_s,
        heartbeat_join_timeout_s=config.heartbeat_join_timeout_s,
        lock_attempts=config.lock_attempts,
        lock_retry_delay_s=config.lock_retry_delay_s,
    )
    try:
        transport = worker_service.DurableSpoolAuthenticatedWorkerTransport(
            registration_path=config.registration_path,
            spool_root=config.spool_root,
            credential_path=config.credential_path,
            response_timeout_seconds=config.response_timeout_seconds,
            poll_seconds=config.transport_poll_seconds,
        )
    except Exception as exc:
        raise MainnetScreenDispatcherError(
            f"durable spool transport cannot reopen: {exc}"
        ) from None
    if (
        transport.registration != registration
        or transport.identity.digest != config.transport_identity_digest
        or transport.credential.digest != config.credential_digest
    ):
        raise MainnetScreenDispatcherError(
            "durable spool transport differs from pinned registration authority"
        )
    return RemoteEvaluationDispatcher(
        coordinator=coordinator,
        transport=transport,
        credential=transport.credential,
    )


def _event(event: str, config: DispatcherConfig, **fields: object) -> None:
    payload = {
        "config_digest": config.digest,
        "event": event,
        "schema": "cacheon-mainnet-screen-dispatcher-event-v1",
        "time_unix": int(time.time()),
        **fields,
    }
    sys.stdout.buffer.write(worker_service.canonical_json_bytes(payload) + b"\n")
    sys.stdout.buffer.flush()


def run_forever(
    config: DispatcherConfig,
    stop: threading.Event,
    *,
    dispatcher_factory: Callable[[DispatcherConfig], Any] = build_dispatcher,
    wait: Callable[[float], bool] | None = None,
) -> None:
    """Drain screen FIFO forever, rebuilding authority after bounded failures."""

    if type(config) is not DispatcherConfig or not isinstance(stop, threading.Event):
        raise MainnetScreenDispatcherError("daemon authority is not exactly typed")
    waiter = stop.wait if wait is None else wait
    backoff = config.restart_initial_backoff_s
    dispatcher = None
    while not stop.is_set():
        try:
            if dispatcher is None:
                dispatcher = dispatcher_factory(config)
                _event("dispatcher_ready", config)
            run = dispatcher.dispatch_screen_once()
            if run is not None and type(run) is not EvaluationRun:
                raise MainnetScreenDispatcherError(
                    "screen dispatcher returned an untyped result"
                )
        except Exception as exc:
            dispatcher = None
            _event(
                "dispatcher_restart",
                config,
                backoff_ms=int(backoff * 1000),
                error=str(exc)[:2048],
                error_type=type(exc).__name__,
            )
            if waiter(backoff):
                break
            backoff = min(config.restart_max_backoff_s, backoff * 2.0)
            continue

        backoff = config.restart_initial_backoff_s
        if run is None:
            if waiter(config.idle_poll_s):
                break
            continue
        _event(
            "screen_completed",
            config,
            disposition=run.disposition,
            lease_id=run.lease.lease_id,
            reservation_ids=list(run.lease.reservation_ids),
            result_digest=run.envelope.digest,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="absolute path to one closed cacheon-mainnet-screen-dispatcher-config-v1 JSON file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"MAINNET-SCREEN-DISPATCHER-ERROR: {exc}", file=sys.stderr)
        return 2

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    _event("daemon_started", config)
    try:
        run_forever(config, stop)
    except KeyboardInterrupt:
        stop.set()
    except Exception as exc:
        print(f"MAINNET-SCREEN-DISPATCHER-ERROR: {exc}", file=sys.stderr)
        return 2
    _event("daemon_stopped", config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

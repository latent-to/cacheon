"""Operate canonical durable FIFO evaluation leases from sealed file authority.

This module performs no evaluation, scheduling, or settlement.  Queue ordering and
every durable lease transition remain owned by :class:`FinalizedIntakeStore`.
"""

from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cacheon.chain.evaluation_leases import (
    EVALUATION_STAGES,
    EvaluationLease,
    EvaluationLeaseError,
    require_evaluation_owner,
)
from cacheon.chain.intake import (
    IntakeError,
    IntakePolicy,
    IntakeScope,
    is_lock_collision,
)
from cacheon.chain.recoverable_intake import (
    RecoverableFinalizedIntakeStore as FinalizedIntakeStore,
)
from cacheon.stack_identity import require_sha256_hex
from functools import partial
from cacheon.chain import sealed_config


CONFIG_SCHEMA = "cacheon-fifo-evaluation-lease-config-v1"
RESULT_SCHEMA = "cacheon-fifo-evaluation-lease-result-v1"
MAX_CONFIG_BYTES = 64 * 1024
MAX_LOCK_ATTEMPTS, MAX_LOCK_RETRY_DELAY_MS = 1_000, 60_000
MAX_LOCK_RETRY_TOTAL_MS = 60_000
_CONFIG_FIELDS = frozenset(
    {
        "intake_db",
        "intake_policy",
        "intake_scope",
        "lease_blocks",
        "lock_attempts",
        "lock_retry_delay_ms",
        "owner",
        "qualification_max_members",
        "schema",
        "stage",
    }
)
_POLICY_FIELDS = frozenset(IntakePolicy.__dataclass_fields__)
_SCOPE_FIELDS = frozenset(IntakeScope.__dataclass_fields__)


class FifoLeaseError(RuntimeError):
    """The sealed operator authority or requested lease operation is invalid."""


@dataclass(frozen=True)
class FifoLeaseConfig:
    intake_db: Path
    policy: IntakePolicy
    scope: IntakeScope
    owner: str
    stage: str
    lease_blocks: int
    qualification_max_members: int
    lock_attempts: int
    lock_retry_delay_ms: int


def _closed(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise FifoLeaseError(f"{label} fields are not closed")
    return value


_absolute_path = partial(sealed_config.absolute_path, error=FifoLeaseError)
_positive_int = partial(sealed_config.positive_int, error=FifoLeaseError)


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FifoLeaseError(f"config JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise FifoLeaseError(f"config JSON contains non-finite number {value}")


def _read_sealed_json(path: Path) -> object:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise FifoLeaseError(f"sealed config is unavailable: {exc}") from None
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o022
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
        ):
            raise FifoLeaseError(
                "sealed config must be an owner-controlled regular file"
            )
        payload = bytearray()
        while len(payload) <= MAX_CONFIG_BYTES:
            chunk = os.read(fd, min(16 * 1024, MAX_CONFIG_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > MAX_CONFIG_BYTES:
            raise FifoLeaseError("sealed config exceeds its byte bound")
    finally:
        os.close(fd)
    try:
        return json.loads(
            bytes(payload).decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except FifoLeaseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FifoLeaseError(f"sealed config is not strict JSON: {exc}") from None


def load_config(path: str | os.PathLike[str]) -> FifoLeaseConfig:
    """Reopen a closed, owner-controlled lease-operation config."""
    config_path = _absolute_path(os.fspath(path), "config path")
    row = _closed(_read_sealed_json(config_path), _CONFIG_FIELDS, "config")
    if row["schema"] != CONFIG_SCHEMA:
        raise FifoLeaseError("config schema is unsupported")

    policy_row = _closed(row["intake_policy"], _POLICY_FIELDS, "intake policy")
    scope_row = _closed(row["intake_scope"], _SCOPE_FIELDS, "intake scope")
    try:
        policy = IntakePolicy(**policy_row)
        scope = IntakeScope(**scope_row)
        owner = require_evaluation_owner(row["owner"])
    except (IntakeError, EvaluationLeaseError, TypeError) as exc:
        raise FifoLeaseError(f"config authority is malformed: {exc}") from None
    if (
        {name: getattr(policy, name) for name in policy.__dataclass_fields__}
        != policy_row
        or scope.to_dict() != scope_row
    ):
        raise FifoLeaseError("config authority did not reopen exactly")
    stage = row["stage"]
    if type(stage) is not str or stage not in EVALUATION_STAGES:
        raise FifoLeaseError("evaluation stage is unsupported")

    lease_blocks = _positive_int(row["lease_blocks"], "lease_blocks")
    if lease_blocks > policy.expiry_blocks:
        raise FifoLeaseError("lease_blocks exceeds the intake expiry policy")
    qualification_max_members = _positive_int(
        row["qualification_max_members"], "qualification_max_members"
    )
    if qualification_max_members > policy.max_cohort:
        raise FifoLeaseError(
            "qualification_max_members exceeds the intake cohort policy"
        )
    lock_attempts = _positive_int(
        row["lock_attempts"], "lock_attempts", maximum=MAX_LOCK_ATTEMPTS
    )
    lock_retry_delay_ms = _positive_int(
        row["lock_retry_delay_ms"],
        "lock_retry_delay_ms",
        maximum=MAX_LOCK_RETRY_DELAY_MS,
        allow_zero=True,
    )
    if (lock_attempts - 1) * lock_retry_delay_ms > MAX_LOCK_RETRY_TOTAL_MS:
        raise FifoLeaseError("configured lock retry wait exceeds 60 seconds")
    return FifoLeaseConfig(
        intake_db=_absolute_path(row["intake_db"], "intake_db"),
        policy=policy,
        scope=scope,
        owner=owner,
        stage=stage,
        lease_blocks=lease_blocks,
        qualification_max_members=qualification_max_members,
        lock_attempts=lock_attempts,
        lock_retry_delay_ms=lock_retry_delay_ms,
    )


def _open_store(config: FifoLeaseConfig) -> FinalizedIntakeStore:
    if type(config) is not FifoLeaseConfig:
        raise FifoLeaseError("lease config is not exactly typed")
    if not config.intake_db.exists():
        raise FifoLeaseError("configured intake database does not exist")
    for attempt in range(config.lock_attempts):
        try:
            return FinalizedIntakeStore(
                config.intake_db, config.policy, scope=config.scope
            )
        except IntakeError as exc:
            if not is_lock_collision(exc) or attempt + 1 == config.lock_attempts:
                raise
            if config.lock_retry_delay_ms:
                time.sleep(config.lock_retry_delay_ms / 1000.0)
    raise AssertionError("positive lock-attempt bound did not execute")


def _cursor(store: FinalizedIntakeStore) -> tuple[int, str]:
    point = store.finalized_cursor()
    if point is None:
        raise FifoLeaseError("durable finalized_cursor is absent")
    return point


def _lease_id(value: object) -> str:
    try:
        return require_sha256_hex(value, field="evaluation lease id")
    except (TypeError, ValueError) as exc:
        raise FifoLeaseError(str(exc)) from None


def _active_lease(
    store: FinalizedIntakeStore,
    config: FifoLeaseConfig,
    lease_id: object,
) -> EvaluationLease:
    exact_id = _lease_id(lease_id)
    matches = tuple(
        lease
        for lease in store.active_evaluation_leases()
        if lease.lease_id == exact_id
    )
    if len(matches) != 1:
        raise FifoLeaseError("exact active evaluation lease was not found")
    lease = matches[0]
    if lease.owner != config.owner or lease.stage != config.stage:
        raise FifoLeaseError("active lease differs from the sealed owner or stage")
    return lease


def _max_members(config: FifoLeaseConfig) -> int | None:
    return (
        config.qualification_max_members
        if config.stage == "qualification"
        else None
    )


def _lease_dict(lease: EvaluationLease) -> dict[str, object]:
    return {
        "claimed_block": lease.claimed_block,
        "expires_block": lease.expires_block,
        "generation": lease.generation,
        "initial_expires_block": lease.initial_expires_block,
        "lease_id": lease.lease_id,
        "members": [member.to_dict() for member in lease.members],
        "owner": lease.owner,
        "stage": lease.stage,
    }


def _result(
    operation: str,
    point: tuple[int, str],
    *,
    lease: EvaluationLease | None = None,
    **fields: object,
) -> dict[str, object]:
    return {
        "finalized_cursor": {"block": point[0], "block_hash": point[1]},
        "lease": None if lease is None else _lease_dict(lease),
        "operation": operation,
        "schema": RESULT_SCHEMA,
        **fields,
    }


def preview(config: FifoLeaseConfig) -> dict[str, object]:
    """Read the canonical next cohort without creating a lease."""
    store = _open_store(config)
    try:
        point = _cursor(store)
        reservation_ids = store.preview_evaluation_claim(
            stage=config.stage, max_members=_max_members(config)
        )
        return _result(
            "preview",
            point,
            reservation_ids=list(reservation_ids),
            stage=config.stage,
        )
    finally:
        store.close()


def claim(config: FifoLeaseConfig) -> dict[str, object]:
    """Atomically claim through the canonical durable FIFO lease API."""
    store = _open_store(config)
    try:
        point = _cursor(store)
        lease = store.claim_evaluation_lease(
            stage=config.stage,
            owner=config.owner,
            current_block=point[0],
            lease_blocks=config.lease_blocks,
            max_members=_max_members(config),
        )
        return _result("claim", point, lease=lease)
    finally:
        store.close()


def heartbeat(config: FifoLeaseConfig, lease_id: object) -> dict[str, object]:
    """Reopen one exact active lease by ID and CAS-extend its current version."""
    exact_id = _lease_id(lease_id)
    store = _open_store(config)
    try:
        point = _cursor(store)
        lease = _active_lease(store, config, exact_id)
        extended = store.heartbeat_evaluation_lease(
            lease,
            current_block=point[0],
            lease_blocks=config.lease_blocks,
        )
        return _result("heartbeat", point, lease=extended)
    finally:
        store.close()


def _release_reason(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > 2_048
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise FifoLeaseError("release reason is not bounded printable text")
    return value


def _result_digest(value: object) -> str:
    if value == "":
        return ""
    try:
        return require_sha256_hex(value, field="result_digest")
    except (TypeError, ValueError) as exc:
        raise FifoLeaseError(str(exc)) from None


def release(
    config: FifoLeaseConfig,
    lease_id: object,
    *,
    reason: object,
    result_digest: object = "",
) -> dict[str, object]:
    """CAS-release exact infrastructure work without consuming an attempt."""
    exact_id = _lease_id(lease_id)
    exact_reason = _release_reason(reason)
    exact_digest = _result_digest(result_digest)
    store = _open_store(config)
    try:
        point = _cursor(store)
        lease = _active_lease(store, config, exact_id)
        released = store.release_evaluation_lease(
            lease,
            current_block=point[0],
            reason=exact_reason,
            result_digest=exact_digest,
        )
        return _result(
            "release",
            point,
            lease=released,
            reason=exact_reason,
            result_digest=exact_digest,
        )
    finally:
        store.close()


def operate(
    config: FifoLeaseConfig,
    operation: str,
    *,
    lease_id: object = None,
    reason: object = None,
    result_digest: object = "",
) -> dict[str, object]:
    """Dispatch one closed operator verb without adding state-machine policy."""
    if operation == "preview":
        return preview(config)
    if operation == "claim":
        return claim(config)
    if operation == "heartbeat":
        return heartbeat(config, lease_id)
    if operation == "release":
        return release(
            config, lease_id, reason=reason, result_digest=result_digest
        )
    raise FifoLeaseError("evaluation lease operation is unsupported")


def canonical_json(value: object) -> str:
    """Encode a machine-readable operator result deterministically."""
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)

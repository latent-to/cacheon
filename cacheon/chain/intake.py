"""Durable finalized-arrival authority, separate from evaluation and settlement."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import fcntl
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from cacheon.chain.reserved_schema import (
    DebtPublicationError,
    FiniteDebtStoreError,
    IncentiveCompositionStoreError,
    ensure_debt_publication_schema,
    migrate_schema3_to4,
    migrate_schema4_to5,
)
from cacheon.chain.evaluation_lease_store import (
    EvaluationLeaseStoreError,
    EvaluationLeaseStoreMixin,
    configure_evaluation_lease_connection,
    ensure_evaluation_lease_schema,
)
from cacheon.chain.evaluation_leases import (
    EvaluationLease, EvaluationLeaseEvent, EvaluationLeaseMember,
)
from cacheon.copy_fingerprint import (
    SubmittedDeltaFingerprint, compare_submitted_deltas,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.stack_identity import canonical_digest, require_sha256_hex
from cacheon.chain.duplicate_replay import PriorVerdict, decide_replay
from cacheon.chain.eval_cost_credit import EVAL_COST_CREDITS_DDL

if TYPE_CHECKING:
    from cacheon.chain.weights import WeightProjection, WeightPublicationRecord
    from cacheon.settlement import (
        SettlementCandidate, SettlementEvidence, SettlementEvent, SettlementPlan,
    )
    from cacheon.stack_manifest import EvaluationStackManifest


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_ACTIVE = (
    "deferred", "reserved", "fetching", "transport_retry", "published", "screening",
    "promoted", "qualifying", "reproduction_pending",
)
_TERMINAL = ("failed", "expired", "qualified")
_STATUSES = frozenset((*_ACTIVE, *_TERMINAL, "held", "no_decision"))
_EXPLICITLY_EXPIRABLE = (
    "deferred", "reserved", "transport_retry", "published", "promoted",
    "reproduction_pending", "held", "no_decision",
)
_AUTOMATICALLY_EXPIRABLE = (
    "deferred", "reserved", "transport_retry", "published", "promoted",
    "reproduction_pending", "held", "no_decision",
)
_AUTOMATIC_EXPIRY_REASON = "finalized_block_sla_expired"
_VALIDATOR_DOWNTIME_REQUEUE_REASON = "validator_downtime_requeued"
# One refresh of the SLA anchor after a prior validator-downtime requeue
# re-expired (operator/SLA mismatch).  A third attempt still fails closed.
_VALIDATOR_DOWNTIME_REQUEUE_REFRESH_REASON = "validator_downtime_requeued_refresh"
_SCHEMA3_MIGRATION_HOLD_REASON = "schema3_reproduction_required"
# Durable "this row spent its automatic retry budget" mark. See
# FinalizedIntakeStore.mark_hold_retry_exhausted.
_EXHAUSTED_HOLD_SUFFIX = ":retry_budget_exhausted"
_SCHEMA3_ARCHIVE_REASON_PREFIX = "schema3_archived@"

# These domain separators are durable protocol identifiers, not product-facing
# package names. They were already committed to SQLite state before the
# Cacheon rename, so changing them would make an existing validator authority
# fail its own integrity checks on reopen.
_INTAKE_SCOPE_DOMAIN = "cacheon.chain.intake-scope"
_FINALIZED_PAYLOAD_DOMAIN = "cacheon.chain.finalized-payload"
_FINALIZED_ARRIVAL_DOMAIN = "cacheon.chain.finalized-arrival"
_EVALUATION_STACK_STATE_DOMAIN = "cacheon.chain.evaluation-stack-state"
_QUALIFICATION_RETRY_GROUP_DOMAIN = "cacheon.chain.qualification-retry-group"
_EVALUATION_STACK_GENESIS_DOMAIN = "cacheon.chain.evaluation-stack-genesis"
_SETTLEMENT_LEASE_DOMAIN = "cacheon.chain.settlement-lease"
_BURN_WEIGHT_AUTHORITY_DOMAIN = "cacheon.chain.burn-weight-authority"
_SETTLEMENT_STATE_DOMAIN = "cacheon.chain.settlement-state"


class IntakeError(RuntimeError):
    """Finalized arrival state is malformed, stale, or unsafe to advance."""


_LOCK_COLLISION_MESSAGE = "another intake controller owns this database"


def is_lock_collision(error: BaseException) -> bool:
    """Classify whether opening failed only because another controller holds the store."""
    return isinstance(error, IntakeError) and str(error) == _LOCK_COLLISION_MESSAGE


@dataclass(frozen=True)
class IntakeScope:
    genesis_hash: str
    netuid: int

    def __post_init__(self) -> None:
        if _BLOCK_HASH.fullmatch(self.genesis_hash or "") is None:
            raise IntakeError("intake genesis hash is malformed")
        if type(self.netuid) is not int or self.netuid < 0:
            raise IntakeError("intake netuid is malformed")

    def to_dict(self) -> dict[str, object]:
        return {"genesis_hash": self.genesis_hash, "netuid": self.netuid}

    @property
    def digest(self) -> str:
        return canonical_digest(_INTAKE_SCOPE_DOMAIN, self.to_dict())


@dataclass(frozen=True)
class IntakePolicy:
    epoch_blocks: int = 360
    cutoff_blocks: int = 30
    max_pending: int = 256
    max_per_hotkey_epoch: int = 16
    max_per_target_epoch: int = 64
    max_transport_retries: int = 3
    max_qualification_retries: int = 3
    max_cohort: int = 8
    # FIFO work should receive a verdict, not expire because evaluator service
    # time exceeded a daily capacity estimate. At the measured ~39 minutes per
    # reservation, the former 10,000-block (~33-hour) bound covered only about
    # 51 rows and 178 queued rows expired without verdicts. 500,000 blocks is
    # roughly 69 days: still a stale-state bound, no longer a routine outcome.
    expiry_blocks: int = 500_000

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(type(value) is not int or value <= 0 for value in values):
            raise IntakeError("intake policy bounds must be positive integers")
        if self.cutoff_blocks >= self.epoch_blocks:
            raise IntakeError("intake cutoff must be smaller than its epoch")
        if self.max_cohort > self.max_pending:
            raise IntakeError("cohort bound exceeds the pending queue bound")


@dataclass(frozen=True)
class FinalizedArrival:
    hotkey: str
    content_hash: str
    url: str
    block: int
    block_hash: str
    event_index: int
    event_subindex: int = 0
    payload_digest: str = ""
    invalid_reason: str = ""
    payment_block: int = 0
    payment_extrinsic_index: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or self.hotkey.strip() != self.hotkey
            or len(self.hotkey) > 256
            or any(char in self.hotkey for char in "\x00\r\n")
        ):
            raise IntakeError("arrival hotkey is malformed")
        identified = (
            isinstance(self.content_hash, str)
            and _HASH.fullmatch(self.content_hash) is not None
            and isinstance(self.url, str)
            and bool(self.url)
        )
        valid_reference = identified and not self.invalid_reason
        invalid_reference = (
            self.content_hash == ""
            and self.url == ""
            and isinstance(self.invalid_reason, str)
            and bool(self.invalid_reason)
            and len(self.invalid_reason) <= 2_048
        )
        attributed_invalid = (
            identified
            and isinstance(self.invalid_reason, str)
            and bool(self.invalid_reason)
            and len(self.invalid_reason) <= 2_048
        )
        if not (valid_reference or invalid_reference or attributed_invalid):
            raise IntakeError("arrival payload disposition is malformed")
        if type(self.block) is not int or self.block < 0:
            raise IntakeError("arrival block is malformed")
        if not isinstance(self.block_hash, str) or _BLOCK_HASH.fullmatch(self.block_hash) is None:
            raise IntakeError("arrival block hash is malformed")
        for field in ("event_index", "event_subindex"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise IntakeError(f"arrival {field} is malformed")
        if (
            type(self.payment_block) is not int
            or self.payment_block < 0
            or type(self.payment_extrinsic_index) is not int
            or self.payment_extrinsic_index < 0
        ):
            raise IntakeError("arrival eval-cost payment pointer is malformed")
        if self.payment_block == 0 and self.payment_extrinsic_index != 0:
            raise IntakeError("arrival eval-cost payment pointer is malformed")
        payload_digest = self.payload_digest or canonical_digest(
            _FINALIZED_PAYLOAD_DOMAIN,
            {"content_hash": self.content_hash, "url": self.url},
        )
        require_sha256_hex(payload_digest, field="payload_digest")
        object.__setattr__(self, "payload_digest", payload_digest)

    @property
    def valid(self) -> bool:
        return not self.invalid_reason

    @property
    def arrival_key(self) -> tuple[int, int, int, str, str]:
        return (
            self.block,
            self.event_index,
            self.event_subindex,
            self.hotkey,
            self.content_hash,
        )

    @property
    def reservation_id(self) -> str:
        return canonical_digest(
            _FINALIZED_ARRIVAL_DOMAIN,
            {
                "block": self.block,
                "block_hash": self.block_hash,
                "content_hash": self.content_hash,
                "event_index": self.event_index,
                "event_subindex": self.event_subindex,
                "hotkey": self.hotkey,
                "payload_digest": self.payload_digest,
                "url": self.url,
            },
        )


@dataclass(frozen=True)
class IntakeReservation:
    reservation_id: str
    arrival: FinalizedArrival
    admission_epoch: int
    status: str
    target_id: str
    target_members: tuple[str, ...]
    delta_fingerprint: SubmittedDeltaFingerprint | None
    transport_attempts: int
    publication_digest: str
    publication_root: str
    qualification_authority_digest: str
    qualification_evidence_digest: str
    arena_service_digest: str
    screen_lane: str
    screen_status: str
    screen_stage_count: int
    screen_attempts: int
    decision: str
    reason: str

    def __post_init__(self) -> None:
        require_sha256_hex(self.reservation_id, field="reservation_id")
        if self.reservation_id != self.arrival.reservation_id:
            raise IntakeError("reservation identity differs from finalized arrival")
        if type(self.admission_epoch) is not int or self.admission_epoch < 0:
            raise IntakeError("reservation epoch is malformed")
        if self.status not in _STATUSES:
            raise IntakeError("reservation status is unsupported")
        if tuple(self.target_members) != tuple(sorted(set(self.target_members))):
            raise IntakeError("reservation target members are not canonical")
        if self.delta_fingerprint is not None and (
            type(self.delta_fingerprint) is not SubmittedDeltaFingerprint
            or self.delta_fingerprint.target_id != self.target_id
            or self.delta_fingerprint.members != self.target_members
        ):
            raise IntakeError("reservation delta fingerprint differs from its target")
        if type(self.transport_attempts) is not int or self.transport_attempts < 0:
            raise IntakeError("reservation transport attempts are malformed")
        for field in (
            "publication_digest",
            "qualification_authority_digest",
            "qualification_evidence_digest",
            "arena_service_digest",
        ):
            value = getattr(self, field)
            if value and _HASH.fullmatch(value) is None:
                raise IntakeError(f"reservation {field} is malformed")
        if self.decision not in {"", "PASS", "FAIL", "NO_DECISION"}:
            raise IntakeError("reservation decision is unsupported")
        if self.screen_lane not in {"", "primary", "reproduction"}:
            raise IntakeError("reservation screen lane is unsupported")
        if self.screen_status not in {
            "", "running", "promote", "reject", "retry", "hold",
        }:
            raise IntakeError("reservation screen status is unsupported")
        if (
            type(self.screen_stage_count) is not int
            or self.screen_stage_count < 0
            or type(self.screen_attempts) is not int
            or self.screen_attempts < 0
        ):
            raise IntakeError("reservation screen counters are malformed")


@dataclass(frozen=True)
class EvaluationStackState:
    arena_digest: str
    generation: int
    manifest: EvaluationStackManifest
    tree_digest: str
    transition_event_id: str

    def __post_init__(self) -> None:
        from cacheon.stack_manifest import EvaluationStackManifest

        require_sha256_hex(self.arena_digest, field="arena_digest")
        require_sha256_hex(self.tree_digest, field="tree_digest")
        require_sha256_hex(self.transition_event_id, field="transition_event_id")
        if (
            type(self.generation) is not int
            or self.generation < 0
            or type(self.manifest) is not EvaluationStackManifest
            or self.manifest.arena_digest != self.arena_digest
        ):
            raise IntakeError("evaluation stack state is malformed")

    @property
    def digest(self) -> str:
        return canonical_digest(
            _EVALUATION_STACK_STATE_DOMAIN,
            {
                "arena_digest": self.arena_digest,
                "generation": self.generation,
                "stack_digest": self.manifest.digest,
                "tree_digest": self.tree_digest,
                "transition_event_id": self.transition_event_id,
            },
        )


@dataclass(frozen=True)
class SettlementLease:
    lease_id: str
    authority_digest: str
    generation: int
    expires_block: int
    stack: EvaluationStackState
    candidates: tuple[SettlementCandidate, ...]
    initial_event_sequence: int
    previous_event_digest: str

    def __post_init__(self) -> None:
        from cacheon.settlement import SettlementCandidate, SettlementQualification

        for field in ("lease_id", "authority_digest"):
            require_sha256_hex(getattr(self, field), field=field)
        if (
            type(self.generation) is not int
            or self.generation <= 0
            or type(self.expires_block) is not int
            or self.expires_block <= 0
            or type(self.stack) is not EvaluationStackState
            or type(self.initial_event_sequence) is not int
            or self.initial_event_sequence < 0
        ):
            raise IntakeError("settlement lease bounds are malformed")
        require_sha256_hex(
            self.previous_event_digest,
            field="previous_event_digest",
        ) if self.previous_event_digest else None
        candidates = tuple(self.candidates)
        if (
            not candidates
            or any(type(row) is not SettlementCandidate for row in candidates)
            or any(row.arena_digest != self.stack.arena_digest for row in candidates)
            or any(row.qualification_authority_digest != self.authority_digest for row in candidates)
        ):
            raise IntakeError("settlement lease candidates are inconsistent")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True)
class CrownedSettlement:
    """One active crown reopened from durable candidate, evidence, and event bytes."""

    candidate: SettlementCandidate
    evidence: SettlementEvidence
    event: SettlementEvent

    def __post_init__(self) -> None:
        from cacheon.settlement import (
            SettlementCandidate, SettlementEvidence, SettlementEvent,
            SettlementEventType,
        )

        if (
            type(self.candidate) is not SettlementCandidate
            or type(self.evidence) is not SettlementEvidence
            or type(self.event) is not SettlementEvent
            or self.event.event_type is not SettlementEventType.CROWN
            or self.evidence.candidate_digest != self.candidate.digest
            or self.event.candidate_digest != self.candidate.digest
            or self.event.target_id != self.candidate.target_id
        ):
            raise IntakeError("active crown authority is inconsistent")


class FinalizedIntakeStore(EvaluationLeaseStoreMixin):
    """Single SQLite authority for arrival order, admission, and qualification state."""

    def __init__(
        self,
        path: str | Path,
        policy: IntakePolicy = IntakePolicy(),
        *,
        scope: IntakeScope,
    ):
        if type(policy) is not IntakePolicy:
            raise IntakeError("intake store requires an exact IntakePolicy")
        if type(scope) is not IntakeScope:
            raise IntakeError("intake store requires an exact chain scope")
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise IntakeError("intake database path must not be a symlink")
        parent_existed = requested.parent.exists()
        requested.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            os.chmod(requested.parent, 0o700)
        try:
            parent_before = requested.parent.lstat()
            parent = requested.parent.resolve(strict=True)
            parent_after = parent.lstat()
        except OSError as exc:
            raise IntakeError(f"intake database parent is unavailable: {exc}") from None
        if (
            stat.S_ISLNK(parent_before.st_mode)
            or not stat.S_ISDIR(parent_before.st_mode)
            or (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
            or stat.S_IMODE(parent_after.st_mode) != 0o700
            or (hasattr(os, "geteuid") and parent_after.st_uid != os.geteuid())
        ):
            raise IntakeError(
                "intake database parent must be validator-owned mode 0700"
            )
        self.path = parent / requested.name
        self.policy = policy
        self.scope = scope
        if self.path.exists():
            info = self.path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                raise IntakeError("existing intake database has unsafe ownership or mode")
        previous_umask = os.umask(0o077)
        try:
            lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            lock_flags |= getattr(os, "O_NOFOLLOW", 0)
            self._lock_fd = os.open(str(self.path) + ".lock", lock_flags, 0o600)
            lock_info = os.fstat(self._lock_fd)
            if (
                not stat.S_ISREG(lock_info.st_mode)
                or lock_info.st_nlink != 1
                or stat.S_IMODE(lock_info.st_mode) != 0o600
                or lock_info.st_uid != os.geteuid()
            ):
                raise IntakeError("intake controller lock has an unsafe shape")
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise IntakeError(_LOCK_COLLISION_MESSAGE) from None
            self._db = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
            self._db.row_factory = sqlite3.Row
            self._evaluation_mutation_authority: set[str] = set()
            configure_evaluation_lease_connection(
                self._db, self._evaluation_mutation_authority
            )
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._create_schema()
            self._bind_scope()
        except Exception:
            if hasattr(self, "_db"):
                self._db.close()
            if hasattr(self, "_lock_fd"):
                os.close(self._lock_fd)
            raise
        finally:
            os.umask(previous_umask)
        os.chmod(self.path, 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                os.chmod(sidecar, 0o600)
        self._recover_interrupted()

    def __enter__(self) -> "FinalizedIntakeStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                block INTEGER NOT NULL,
                block_hash TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                event_subindex INTEGER NOT NULL,
                hotkey TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                url TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                invalid_reason TEXT NOT NULL DEFAULT '',
                admission_epoch INTEGER NOT NULL,
                status TEXT NOT NULL,
                target_id TEXT NOT NULL DEFAULT '',
                target_members_json TEXT NOT NULL DEFAULT '[]',
                delta_fingerprint_json TEXT NOT NULL DEFAULT '',
                transport_attempts INTEGER NOT NULL DEFAULT 0,
                publication_digest TEXT NOT NULL DEFAULT '',
                publication_root TEXT NOT NULL DEFAULT '',
                qualification_authority_digest TEXT NOT NULL DEFAULT '',
                qualification_authority_json TEXT NOT NULL DEFAULT '',
                qualification_evidence_digest TEXT NOT NULL DEFAULT '',
                arena_service_digest TEXT NOT NULL DEFAULT '',
                screen_lane TEXT NOT NULL DEFAULT '',
                screen_status TEXT NOT NULL DEFAULT '',
                screen_stage_count INTEGER NOT NULL DEFAULT 0,
                screen_attempts INTEGER NOT NULL DEFAULT 0,
                retry_group_digest TEXT NOT NULL DEFAULT '',
                retry_position INTEGER NOT NULL DEFAULT 0,
                decision TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                UNIQUE(block_hash, event_index, event_subindex, hotkey, content_hash)
            ) STRICT;
            CREATE INDEX IF NOT EXISTS reservations_order
                ON reservations(block, event_index, event_subindex, hotkey, content_hash);
            CREATE INDEX IF NOT EXISTS reservations_status
                ON reservations(status, admission_epoch, block, event_index, event_subindex);
            CREATE TABLE IF NOT EXISTS reservation_sla_resets (
                reservation_id TEXT PRIMARY KEY REFERENCES reservations(reservation_id),
                reset_block INTEGER NOT NULL CHECK(reset_block>=0),
                authority_digest TEXT NOT NULL,
                reason TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS qualification_dispositions (
                reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
                attempt_index INTEGER NOT NULL,
                authority_digest TEXT NOT NULL,
                authority_manifest_json TEXT NOT NULL,
                evidence_digest TEXT NOT NULL,
                attempt_ref_json TEXT NOT NULL,
                report_digest TEXT NOT NULL,
                failure_digest TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY(reservation_id, attempt_index)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS arena_screen_dispositions (
                reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
                attempt_index INTEGER NOT NULL,
                service_digest TEXT NOT NULL,
                candidate_digest TEXT NOT NULL,
                receipt_digest TEXT NOT NULL UNIQUE,
                receipt_json TEXT NOT NULL,
                decision TEXT NOT NULL,
                stage_count INTEGER NOT NULL,
                lane TEXT NOT NULL,
                PRIMARY KEY(reservation_id, attempt_index)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS settlement_qualifications (
                reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
                reproduction_index INTEGER NOT NULL,
                qualification_digest TEXT NOT NULL UNIQUE,
                qualification_json TEXT NOT NULL,
                attempt_ref_json TEXT NOT NULL,
                evidence_root TEXT NOT NULL,
                retained_block INTEGER NOT NULL DEFAULT 0 CHECK(retained_block>=0),
                PRIMARY KEY(reservation_id, reproduction_index)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS settlement_candidates (
                reservation_id TEXT PRIMARY KEY REFERENCES reservations(reservation_id),
                authority_digest TEXT NOT NULL,
                candidate_digest TEXT NOT NULL UNIQUE,
                candidate_json TEXT NOT NULL,
                evidence_root TEXT NOT NULL,
                reproduction_evidence_root TEXT NOT NULL DEFAULT '',
                settlement_evidence_digest TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                lease_id TEXT NOT NULL DEFAULT '',
                lease_generation INTEGER NOT NULL DEFAULT 0,
                lease_expires_block INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT ''
            ) STRICT;
            CREATE INDEX IF NOT EXISTS settlement_candidates_status
                ON settlement_candidates(status, authority_digest, reservation_id);
            CREATE TABLE IF NOT EXISTS settlement_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                reservation_id TEXT NOT NULL,
                arena_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                event_digest TEXT NOT NULL,
                event_json TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS evaluation_stacks (
                arena_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                stack_digest TEXT NOT NULL,
                tree_digest TEXT NOT NULL,
                stack_json TEXT NOT NULL,
                transition_event_id TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS standing_reward_claims (
                arena_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                claim_digest TEXT NOT NULL UNIQUE,
                claim_json TEXT NOT NULL,
                status TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(arena_id, target_id)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS discovery_bounty_claims (
                claim_digest TEXT PRIMARY KEY,
                proposal_digest TEXT NOT NULL UNIQUE,
                claim_json TEXT NOT NULL,
                status TEXT NOT NULL,
                event_id TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS weight_publications (
                record_digest TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                projection_digest TEXT NOT NULL,
                projection_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_block INTEGER NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS followed_weight_publications (
                record_digest TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                projection_digest TEXT NOT NULL,
                offer_digest TEXT NOT NULL,
                offer_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_block INTEGER NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS eval_cost_payments (
                payment_block INTEGER NOT NULL,
                payment_extrinsic_index INTEGER NOT NULL,
                reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
                content_hash TEXT NOT NULL,
                hotkey TEXT NOT NULL,
                amount_tao_rao INTEGER NOT NULL,
                PRIMARY KEY(payment_block, payment_extrinsic_index)
            ) STRICT;
            """
        )
        self._db.executescript(EVAL_COST_CREDITS_DDL)
        reservation_columns = {
            row["name"] for row in self._db.execute("PRAGMA table_info(reservations)")
        }
        additions = {
            "arena_service_digest": "TEXT NOT NULL DEFAULT ''",
            "screen_lane": "TEXT NOT NULL DEFAULT ''",
            "screen_status": "TEXT NOT NULL DEFAULT ''",
            "screen_stage_count": "INTEGER NOT NULL DEFAULT 0",
            "screen_attempts": "INTEGER NOT NULL DEFAULT 0",
            "eval_cost_payment_block": "INTEGER NOT NULL DEFAULT 0",
            "eval_cost_payment_extrinsic_index": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, declaration in additions.items():
            if name not in reservation_columns:
                self._db.execute(
                    f"ALTER TABLE reservations ADD COLUMN {name} {declaration}"
                )
        qualification_columns = {
            row["name"] for row in self._db.execute(
                "PRAGMA table_info(settlement_qualifications)"
            )
        }
        if "retained_block" not in qualification_columns:
            # Existing evidence predates a trustworthy progress timestamp.  Keep
            # zero as an explicit unknown sentinel; automatic expiry must not
            # invent a deadline for those rows.
            self._db.execute(
                "ALTER TABLE settlement_qualifications ADD COLUMN "
                "retained_block INTEGER NOT NULL DEFAULT 0 CHECK(retained_block>=0)"
            )
        settlement_columns = {
            row["name"] for row in self._db.execute(
                "PRAGMA table_info(settlement_candidates)"
            )
        }
        if "reproduction_evidence_root" not in settlement_columns:
            self._db.execute(
                "ALTER TABLE settlement_candidates ADD COLUMN "
                "reproduction_evidence_root TEXT NOT NULL DEFAULT ''"
            )
        try:
            ensure_evaluation_lease_schema(self._db)
        except EvaluationLeaseStoreError as exc:
            raise IntakeError(f"evaluation lease schema cannot open: {exc}") from None
        schema = self._db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()
        if schema is None:
            self._db.execute("INSERT INTO metadata(key,value) VALUES('schema','3')")
        elif schema["value"] in {"1", "2"}:
            # v1/v2 allowed one PASS to become settlement-pending.  Preserve all
            # rows for audit but fail them closed until a fresh two-PASS service
            # qualification is run under this schema.
            self._db.execute(
                "UPDATE settlement_candidates SET status='held',lease_id='',"
                "lease_expires_block=0,reason='schema3_reproduction_required'"
            )
            self._db.execute(
                "UPDATE reservations SET status='held',decision='NO_DECISION',"
                "reason='schema3_reproduction_required' WHERE reservation_id IN "
                "(SELECT reservation_id FROM settlement_candidates)"
            )
            self._db.execute("UPDATE metadata SET value='3' WHERE key='schema'")
        elif schema["value"] not in {"3", "4", "5", "6"}:
            raise IntakeError("intake database schema is unsupported")
        try:
            migrate_schema3_to4(self._db)
        except FiniteDebtStoreError as exc:
            raise IntakeError(f"intake schema-4 migration failed: {exc}") from None
        try:
            migrate_schema4_to5(self._db)
        except IncentiveCompositionStoreError as exc:
            raise IntakeError(f"intake schema-5 migration failed: {exc}") from None
        try:
            ensure_debt_publication_schema(self._db)
        except DebtPublicationError as exc:
            raise IntakeError(
                f"debt publication schema cannot open: {exc}"
            ) from None

    def _bind_scope(self) -> None:
        encoded = json.dumps(self.scope.to_dict(), separators=(",", ":"), sort_keys=True)
        row = self._db.execute(
            "SELECT value FROM metadata WHERE key='intake_scope'"
        ).fetchone()
        if row is None:
            self._db.execute(
                "INSERT INTO metadata(key,value) VALUES('intake_scope',?)", (encoded,)
            )
        elif row["value"] != encoded:
            raise IntakeError("intake database belongs to another chain scope")

    def _recover_interrupted(self) -> None:
        with self._transaction():
            self._db.execute(
                "UPDATE reservations SET status='held', decision='', "
                "reason='controller_restart_during_' || status "
                "WHERE status IN ('fetching','qualifying')"
            )
            self._db.execute(
                "UPDATE reservations SET status=CASE screen_lane "
                "WHEN 'reproduction' THEN 'reproduction_pending' ELSE 'published' END,"
                "decision='',screen_status='retry',"
                "reason='controller_restart_during_screening' WHERE status='screening'"
            )
            self._db.execute(
                "UPDATE settlement_candidates SET status='pending',lease_id='',"
                "lease_generation=lease_generation+1,lease_expires_block=0,"
                "reason='controller_restart_during_settlement' WHERE status='leased'"
            )

    def _transaction(self):
        store = self

        class Transaction:
            def __enter__(self):
                # Nested use composes under one outer transaction as a
                # SAVEPOINT rather than a second BEGIN (which SQLite
                # rejects).  Releasing the savepoint does not commit the
                # outer transaction.
                self._nested = store._db.in_transaction
                if self._nested:
                    self._savepoint = f"cacheon_nested_{id(self):x}"
                    store._db.execute(f"SAVEPOINT {self._savepoint}")
                else:
                    store._db.execute("BEGIN IMMEDIATE")
                return store._db

            def __exit__(self, exc_type, _exc, _tb):
                if self._nested:
                    if exc_type:
                        store._db.execute(f"ROLLBACK TO {self._savepoint}")
                    store._db.execute(f"RELEASE {self._savepoint}")
                else:
                    store._db.execute("ROLLBACK" if exc_type else "COMMIT")

        return Transaction()

    def _cursor(self) -> tuple[int, str] | None:
        row = self._db.execute(
            "SELECT value FROM metadata WHERE key='finalized_cursor'"
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError) as exc:
            raise IntakeError(f"finalized cursor is corrupt: {exc}") from None
        if (
            type(value) is not list
            or len(value) != 2
            or type(value[0]) is not int
            or value[0] < 0
            or not isinstance(value[1], str)
            or _BLOCK_HASH.fullmatch(value[1]) is None
        ):
            raise IntakeError("finalized cursor is malformed")
        return value[0], value[1]

    def finalized_cursor(self) -> tuple[int, str] | None:
        """Return the last atomically reserved finalized head, if any."""

        return self._cursor()

    def _eval_cost_payment_used(self, payment_block: int, payment_extrinsic_index: int) -> bool:
        row = self._db.execute(
            "SELECT 1 AS n FROM eval_cost_payments "
            "WHERE payment_block=? AND payment_extrinsic_index=?",
            (payment_block, payment_extrinsic_index),
        ).fetchone()
        return row is not None

    def _unspent_eval_cost_credit(self, hotkey: str) -> str:
        row = self._db.execute(
            "SELECT credit_id FROM eval_cost_credits "
            "WHERE hotkey=? AND reservation_id='' "
            "ORDER BY granted_at, credit_id LIMIT 1",
            (hotkey,),
        ).fetchone()
        return "" if row is None else row["credit_id"]

    def reserve_finalized(
        self,
        arrivals: Iterable[FinalizedArrival],
        *,
        finalized_block: int,
        finalized_block_hash: str,
        eval_cost_amount_tao_rao: int = 0,
    ) -> tuple[IntakeReservation, ...]:
        rows = tuple(arrivals)
        if any(type(row) is not FinalizedArrival for row in rows):
            raise IntakeError("finalized reservation input is not typed")
        if tuple(row.arrival_key for row in rows) != tuple(
            sorted({row.arrival_key for row in rows})
        ):
            raise IntakeError("finalized arrivals are duplicated or out of order")
        if type(finalized_block) is not int or finalized_block < 0:
            raise IntakeError("finalized block is malformed")
        if _BLOCK_HASH.fullmatch(finalized_block_hash or "") is None:
            raise IntakeError("finalized block hash is malformed")
        if (
            type(eval_cost_amount_tao_rao) is not int
            or eval_cost_amount_tao_rao < 0
        ):
            raise IntakeError("eval cost amount cannot be negative")
        if any(row.block > finalized_block for row in rows):
            raise IntakeError("unfinalized arrival reached durable intake")

        inserted: list[str] = []
        with self._transaction():
            # Admission capacity is finalized-chain state, not an operator-maintained
            # cache.  Apply the already-bound arrival-block SLA in the same write
            # transaction before counting unresolved rows.
            self._expire_stale_rows(finalized_block)
            # Capacity released by expiry belongs to the already-retained oldest
            # deferred arrivals before any newly observed arrival may compete.
            self._activate_deferred_rows()
            cursor = self._cursor()
            if cursor is not None and (
                finalized_block < cursor[0]
                or (finalized_block == cursor[0] and finalized_block_hash != cursor[1])
            ):
                raise IntakeError("finalized cursor regressed or changed hash")
            pending = self._db.execute(
                "SELECT COUNT(*) AS n FROM reservations WHERE status IN "
                "('reserved','fetching','transport_retry','published','screening',"
                "'promoted','qualifying','reproduction_pending','held','no_decision')"
            ).fetchone()["n"]
            for arrival in rows:
                existing = self._db.execute(
                    "SELECT * FROM reservations WHERE reservation_id=?",
                    (arrival.reservation_id,),
                ).fetchone()
                if existing is not None:
                    if self._row(existing).arrival != arrival:
                        raise IntakeError("reservation ID collision changed arrival bytes")
                    continue
                epoch = arrival.block // self.policy.epoch_blocks
                if arrival.block % self.policy.epoch_blocks >= (
                    self.policy.epoch_blocks - self.policy.cutoff_blocks
                ):
                    epoch += 1
                hotkey_count = self._db.execute(
                    "SELECT COUNT(*) AS n FROM reservations WHERE admission_epoch=? AND hotkey=?",
                    (epoch, arrival.hotkey),
                ).fetchone()["n"]
                status, reason = "reserved", ""
                # An operator-granted credit substitutes for exactly one
                # missing payment; every other invalid_reason keeps its
                # verdict, and a cited (even invalid) payment pointer is
                # never rescued.
                credit_id = ""
                if (
                    eval_cost_amount_tao_rao > 0
                    and arrival.payment_block == 0
                    and (
                        arrival.valid
                        or arrival.invalid_reason == "missing_eval_cost_payment"
                    )
                ):
                    credit_id = self._unspent_eval_cost_credit(arrival.hotkey)
                if not arrival.valid and not (
                    credit_id
                    and arrival.invalid_reason == "missing_eval_cost_payment"
                ):
                    status, reason = "failed", arrival.invalid_reason
                elif (
                    eval_cost_amount_tao_rao > 0
                    and arrival.payment_block == 0
                    and not credit_id
                ):
                    status, reason = "failed", "missing_eval_cost_payment"
                elif finalized_block - arrival.block >= self.policy.expiry_blocks:
                    status, reason = "expired", _AUTOMATIC_EXPIRY_REASON
                elif hotkey_count >= self.policy.max_per_hotkey_epoch:
                    status, reason = "failed", "hotkey_epoch_admission_limit"
                elif (
                    eval_cost_amount_tao_rao > 0
                    and arrival.payment_block > 0
                    and self._eval_cost_payment_used(
                        arrival.payment_block, arrival.payment_extrinsic_index
                    )
                ):
                    status, reason = "failed", "eval_cost_payment_used"
                elif pending >= self.policy.max_pending:
                    # GPU/worker backpressure is validator capacity, not a miner
                    # fault.  Retain the finalized arrival in FIFO order and
                    # promote it automatically when an active queue slot opens.
                    status, reason = "deferred", "pending_queue_deferred"
                else:
                    pending += 1
                self._db.execute(
                    "INSERT INTO reservations(reservation_id,block,block_hash,event_index,event_subindex,"
                    "hotkey,content_hash,url,payload_digest,invalid_reason,admission_epoch,status,reason,"
                    "eval_cost_payment_block,eval_cost_payment_extrinsic_index) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        arrival.reservation_id,
                        arrival.block,
                        arrival.block_hash,
                        arrival.event_index,
                        arrival.event_subindex,
                        arrival.hotkey,
                        arrival.content_hash,
                        arrival.url,
                        arrival.payload_digest,
                        arrival.invalid_reason,
                        epoch,
                        status,
                        reason,
                        arrival.payment_block,
                        arrival.payment_extrinsic_index,
                    ),
                )
                if (
                    eval_cost_amount_tao_rao > 0
                    and status in {"reserved", "deferred"}
                    and arrival.payment_block > 0
                ):
                    self._db.execute(
                        "INSERT INTO eval_cost_payments("
                        "payment_block,payment_extrinsic_index,reservation_id,"
                        "content_hash,hotkey,amount_tao_rao) VALUES(?,?,?,?,?,?)",
                        (
                            arrival.payment_block,
                            arrival.payment_extrinsic_index,
                            arrival.reservation_id,
                            arrival.content_hash,
                            arrival.hotkey,
                            eval_cost_amount_tao_rao,
                        ),
                    )
                elif (
                    eval_cost_amount_tao_rao > 0
                    and status in {"reserved", "deferred"}
                    and credit_id
                ):
                    # Mirror payment semantics: consumed only by an admitted
                    # row, inside the same transaction that peeked it.
                    spent = self._db.execute(
                        "UPDATE eval_cost_credits SET reservation_id=?, "
                        "spent_block=? WHERE credit_id=? AND reservation_id=''",
                        (arrival.reservation_id, finalized_block, credit_id),
                    )
                    if spent.rowcount != 1:
                        raise IntakeError(
                            "eval-cost credit disappeared during admission"
                        )
                inserted.append(arrival.reservation_id)
            cursor_value = json.dumps(
                [finalized_block, finalized_block_hash], separators=(",", ":")
            )
            self._db.execute(
                "INSERT INTO metadata(key,value) VALUES('finalized_cursor',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (cursor_value,),
            )
        return tuple(self.get(value) for value in inserted)

    def _row(self, row: sqlite3.Row) -> IntakeReservation:
        try:
            members = tuple(json.loads(row["target_members_json"]))
            fingerprint = (
                SubmittedDeltaFingerprint.from_dict(
                    json.loads(row["delta_fingerprint_json"])
                )
                if row["delta_fingerprint_json"]
                else None
            )
        except (TypeError, ValueError) as exc:
            raise IntakeError(f"reservation provenance is corrupt: {exc}") from None
        arrival = FinalizedArrival(
            row["hotkey"], row["content_hash"], row["url"], row["block"],
            row["block_hash"], row["event_index"], row["event_subindex"],
            row["payload_digest"], row["invalid_reason"],
            int(row["eval_cost_payment_block"] or 0),
            int(row["eval_cost_payment_extrinsic_index"] or 0),
        )
        return IntakeReservation(
            row["reservation_id"], arrival, row["admission_epoch"], row["status"],
            row["target_id"], members, fingerprint, row["transport_attempts"],
            row["publication_digest"], row["publication_root"],
            row["qualification_authority_digest"], row["qualification_evidence_digest"],
            row["arena_service_digest"], row["screen_lane"], row["screen_status"],
            row["screen_stage_count"], row["screen_attempts"],
            row["decision"], row["reason"],
        )

    def get(self, reservation_id: str) -> IntakeReservation:
        row = self._db.execute(
            "SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)
        ).fetchone()
        if row is None:
            raise IntakeError("unknown intake reservation")
        return self._row(row)

    def all(self) -> tuple[IntakeReservation, ...]:
        return tuple(self._row(row) for row in self._db.execute(
            "SELECT * FROM reservations ORDER BY block,event_index,event_subindex,hotkey,content_hash"
        ))

    def _activate_deferred_rows(self) -> tuple[str, ...]:
        active = self._db.execute(
            "SELECT COUNT(*) AS n FROM reservations WHERE status IN ("
            "'reserved','fetching','transport_retry','published','screening',"
            "'promoted','qualifying','reproduction_pending','held','no_decision')"
        ).fetchone()["n"]
        capacity = max(0, self.policy.max_pending - active)
        if capacity == 0:
            return ()
        ids = tuple(
            row["reservation_id"]
            for row in self._db.execute(
                "SELECT reservation_id FROM reservations WHERE status='deferred' "
                "ORDER BY block,event_index,event_subindex,hotkey,content_hash LIMIT ?",
                (capacity,),
            )
        )
        if ids:
            marks = ",".join("?" for _ in ids)
            cursor = self._db.execute(
                f"UPDATE reservations SET status='reserved',reason='' WHERE "
                f"status='deferred' AND reservation_id IN ({marks})",
                ids,
            )
            if cursor.rowcount != len(ids):
                raise IntakeError("deferred queue changed while activating capacity")
        return ids

    def pending(self, *, limit: int | None = None) -> tuple[IntakeReservation, ...]:
        bound = self.policy.max_cohort if limit is None else limit
        if type(bound) is not int or bound <= 0 or bound > self.policy.max_pending:
            raise IntakeError("pending reservation limit is invalid")
        with self._transaction():
            self._activate_deferred_rows()
            rows = tuple(self._db.execute(
                "SELECT * FROM reservations WHERE status IN ('reserved','transport_retry') "
                "AND transport_attempts < ? ORDER BY block,event_index,event_subindex,"
                "hotkey,content_hash LIMIT ?",
                (self.policy.max_transport_retries, bound),
            ))
        return tuple(self._row(row) for row in rows)

    def mark_fetching(self, reservation_id: str) -> IntakeReservation:
        with self._transaction():
            row = self.get(reservation_id)
            if row.status not in {"reserved", "transport_retry"}:
                raise IntakeError("only pending intake may begin transport")
            attempts = row.transport_attempts + 1
            status = "fetching" if attempts <= self.policy.max_transport_retries else "held"
            reason = "" if status == "fetching" else "transport_retry_limit"
            self._db.execute(
                "UPDATE reservations SET status=?,transport_attempts=?,reason=? WHERE reservation_id=?",
                (status, attempts, reason, reservation_id),
            )
        return self.get(reservation_id)

    def mark_transport_retry(self, reservation_id: str, reason: str) -> IntakeReservation:
        row = self.get(reservation_id)
        exhausted = row.transport_attempts >= self.policy.max_transport_retries
        return self._transition(
            reservation_id,
            {"fetching"},
            "held" if exhausted else "transport_retry",
            "",
            "transport_retry_limit" if exhausted else reason,
        )

    def mark_failed(self, reservation_id: str, reason: str) -> IntakeReservation:
        return self._transition(
            reservation_id, {"fetching", "published"}, "failed", "FAIL", reason
        )

    def release_manifest_compatibility_failure(
        self,
        reservation_id: str,
        *,
        expected_reason_digest: str,
    ) -> IntakeReservation:
        """Return one exact pre-publication rollout failure to durable FIFO.

        This is an operator recovery seam for a validator reader-compatibility
        defect, not a general terminal-result override.  It refuses rows with
        any publication, screen, qualification, lease, or settlement history;
        the caller must also bind the exact retained reason bytes.  The normal
        intake loop then reopens the content-addressed private tree, reruns the
        current manifest policy, and publishes through the ordinary path.
        """

        require_sha256_hex(
            expected_reason_digest,
            field="manifest compatibility reason digest",
        )
        with self._transaction():
            row = self.get(reservation_id)
            reason_digest = hashlib.sha256(row.reason.encode("utf-8")).hexdigest()
            if (
                row.status != "failed"
                or row.decision != "FAIL"
                or not row.reason.startswith("manifest:")
                or "unsupported abi_version" not in row.reason
                or reason_digest != expected_reason_digest
                or row.delta_fingerprint is not None
                or row.publication_digest
                or row.publication_root
                or row.qualification_authority_digest
                or row.qualification_evidence_digest
                or row.arena_service_digest
                or row.screen_attempts
            ):
                raise IntakeError(
                    "reservation is not the exact pre-publication compatibility failure"
                )
            covered = (
                self._db.execute(
                    "SELECT COUNT(*) AS n FROM arena_screen_dispositions "
                    "WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["n"]
                + self._db.execute(
                    "SELECT COUNT(*) AS n FROM qualification_dispositions "
                    "WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["n"]
                + self._db.execute(
                    "SELECT COUNT(*) AS n FROM settlement_qualifications "
                    "WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["n"]
                + self._db.execute(
                    "SELECT COUNT(*) AS n FROM settlement_candidates "
                    "WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["n"]
                + self._db.execute(
                    "SELECT COUNT(*) AS n FROM evaluation_lease_members "
                    "WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["n"]
            )
            if covered:
                raise IntakeError(
                    "compatibility failure already has downstream evaluation authority"
                )
            cursor = self._db.execute(
                "UPDATE reservations SET status='reserved',decision='',"
                "reason='manifest_compatibility_released' "
                "WHERE reservation_id=? AND status='failed' AND decision='FAIL'",
                (reservation_id,),
            )
            if cursor.rowcount != 1:
                raise IntakeError("compatibility release lost its exact terminal row")
        return self.get(reservation_id)

    def mark_held(self, reservation_id: str, reason: str) -> IntakeReservation:
        return self._transition(
            reservation_id,
            {
                "reserved", "fetching", "transport_retry", "published", "screening",
                "promoted", "qualifying", "reproduction_pending", "no_decision",
            },
            "held",
            "",
            reason,
        )

    def mark_published(
        self,
        reservation_id: str,
        *,
        delta_fingerprint: SubmittedDeltaFingerprint,
        publication_digest: str,
        publication_root: str | Path,
    ) -> IntakeReservation:
        if type(delta_fingerprint) is not SubmittedDeltaFingerprint:
            raise IntakeError("publication requires a typed submitted-delta fingerprint")
        target_id = delta_fingerprint.target_id
        members = delta_fingerprint.members
        require_sha256_hex(publication_digest, field="publication_digest")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status != "fetching":
                raise IntakeError("publication requires an active transport")
            count = self._db.execute(
                "SELECT COUNT(*) AS n FROM reservations WHERE admission_epoch=? AND target_id=?",
                (row.admission_epoch, target_id),
            ).fetchone()["n"]
            status, decision, reason = "published", "", ""
            if count >= self.policy.max_per_target_epoch:
                status, reason = "failed", "target_epoch_admission_limit"
            self._db.execute(
                "UPDATE reservations SET status=?,target_id=?,target_members_json=?,delta_fingerprint_json=?,"
                "publication_digest=?,publication_root=?,decision=?,reason=? WHERE reservation_id=?",
                (
                    status, target_id, json.dumps(members, separators=(",", ":")),
                    json.dumps(delta_fingerprint.to_dict(), separators=(",", ":"), sort_keys=True),
                    publication_digest, str(publication_root), decision, reason,
                    reservation_id,
                ),
            )
        return self.get(reservation_id)

    def screenable(self, *, limit: int | None = None) -> tuple[IntakeReservation, ...]:
        """Return validator-selected work awaiting a fresh non-crown screen."""

        bound = self.policy.max_cohort if limit is None else limit
        if type(bound) is not int or bound <= 0 or bound > self.policy.max_cohort:
            raise IntakeError("screen cohort limit is invalid")
        rows = self._db.execute(
            "SELECT r.* FROM reservations AS r WHERE status IN "
            "('published','reproduction_pending') AND NOT EXISTS ("
            "SELECT 1 FROM evaluation_lease_members AS em WHERE "
            "em.reservation_id=r.reservation_id AND em.active=1) ORDER BY "
            "CASE status WHEN 'reproduction_pending' THEN 0 ELSE 1 END,"
            "block,event_index,event_subindex,hotkey,content_hash LIMIT ?",
            (bound,),
        )
        return tuple(self._row(row) for row in rows)

    def retire_duplicate_screenables(
        self, *, service_digest: str, limit: int | None = None
    ) -> tuple[tuple[str, str], ...]:
        """Fail screenable rows whose exact bytes already lost under this arena.

        Wiring only; the rule lives in :mod:`cacheon.chain.duplicate_replay`,
        which documents why identical bytes under an identical arena may inherit
        a FAIL and why a PASS may not.

        Runs before work is claimed, so a duplicate costs neither a screen nor a
        qualification. ``screenable`` already excludes rows under an active
        lease, and ``decide_replay`` refuses the reproduction lane outright.
        """

        require_sha256_hex(service_digest, field="arena service digest")
        priors = tuple(
            PriorVerdict(
                reservation_id=row["reservation_id"],
                content_hash=row["content_hash"],
                arena_service_digest=row["arena_service_digest"],
                decision=row["decision"],
                reason=row["reason"],
            )
            for row in self._db.execute(
                "SELECT reservation_id,content_hash,arena_service_digest,decision,"
                "reason FROM reservations WHERE decision IN ('PASS','FAIL') "
                "AND arena_service_digest=?",
                (service_digest,),
            )
        )
        if not priors:
            return ()
        retired: list[tuple[str, str]] = []
        for row in self.screenable(limit=limit):
            decision = decide_replay(
                content_hash=row.arrival.content_hash,
                arena_service_digest=service_digest,
                screen_lane=row.screen_lane,
                priors=priors,
            )
            if not decision.replay:
                continue
            self.mark_failed(row.reservation_id, decision.reason)
            retired.append((row.reservation_id, decision.reason))
        return tuple(retired)

    def arena_queue_snapshot(self, *, current_block: int):
        from cacheon.arena_service import ArenaQueueSnapshot

        if type(current_block) is not int or current_block < 0:
            raise IntakeError("arena queue block is malformed")
        queued = self._db.execute(
            "SELECT COUNT(*) AS n,MIN(block) AS oldest FROM reservations AS r WHERE "
            "status IN ('published','reproduction_pending','promoted') AND NOT EXISTS ("
            "SELECT 1 FROM evaluation_lease_members AS em WHERE "
            "em.reservation_id=r.reservation_id AND em.active=1)"
        ).fetchone()
        active_screens = self._db.execute(
            "SELECT (SELECT COUNT(*) FROM reservations WHERE status='screening') + "
            "(SELECT COUNT(*) FROM evaluation_lease_members AS em JOIN "
            "evaluation_leases AS el USING(lease_id) WHERE em.active=1 "
            "AND el.stage='screen') AS n"
        ).fetchone()["n"]
        active_qualifications = self._db.execute(
            "SELECT (SELECT COUNT(*) FROM reservations WHERE status='qualifying') + "
            "(SELECT COUNT(*) FROM evaluation_lease_members AS em JOIN "
            "evaluation_leases AS el USING(lease_id) WHERE em.active=1 "
            "AND el.stage='qualification') AS n"
        ).fetchone()["n"]
        oldest = queued["oldest"]
        return ArenaQueueSnapshot(
            queued["n"],
            0 if oldest is None else max(0, current_block - oldest),
            active_screens,
            active_qualifications,
        )

    def begin_screen(
        self, reservation_id: str, *, service_digest: str
    ) -> IntakeReservation:
        require_sha256_hex(service_digest, field="arena service digest")
        with self._transaction():
            self._require_evaluation_mutation_authority(reservation_id)
            row = self.get(reservation_id)
            if row.status not in {"published", "reproduction_pending"}:
                raise IntakeError("only screenable intake may begin arena screening")
            lane = (
                "reproduction" if row.status == "reproduction_pending" else "primary"
            )
            attempts = self._db.execute(
                "SELECT COUNT(*) AS n FROM arena_screen_dispositions "
                "WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()["n"]
            self._db.execute(
                "UPDATE reservations SET status='screening',arena_service_digest=?,"
                "screen_lane=?,screen_status='running',screen_stage_count=0,"
                "screen_attempts=?,decision='',reason='' WHERE reservation_id=?",
                (service_digest, lane, attempts + 1, reservation_id),
            )
        return self.get(reservation_id)

    def apply_screen_receipt(
        self,
        reservation_id: str,
        *,
        candidate_digest: str,
        receipt,
    ) -> IntakeReservation:
        """Atomically retain one non-crown screen and its derived disposition."""

        from cacheon.arena_service import ArenaScreenReceipt, PromotionDecision

        require_sha256_hex(candidate_digest, field="screen candidate digest")
        if type(receipt) is not ArenaScreenReceipt:
            raise IntakeError("arena screen receipt is not exactly typed")
        encoded = json.dumps(
            receipt.to_dict(), separators=(",", ":"), sort_keys=True
        )
        with self._transaction():
            row = self.get(reservation_id)
            if (
                row.status != "screening"
                or row.arena_service_digest != receipt.service_digest
                or row.screen_attempts != receipt.screen_attempt
                or receipt.candidate_digest != candidate_digest
            ):
                raise IntakeError("arena screen receipt differs from active screening")
            attempt = row.screen_attempts - 1
            self._db.execute(
                "INSERT INTO arena_screen_dispositions(reservation_id,attempt_index,"
                "service_digest,candidate_digest,receipt_digest,receipt_json,decision,"
                "stage_count,lane) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    reservation_id, attempt, receipt.service_digest,
                    candidate_digest, receipt.digest, encoded, receipt.decision.value,
                    len(receipt.results), row.screen_lane,
                ),
            )
            if receipt.decision is PromotionDecision.PROMOTE:
                status, decision, reason = "promoted", "", "screen_promoted"
            elif receipt.decision is PromotionDecision.REJECT:
                status, decision, reason = "failed", "FAIL", "screen_rejected"
            elif receipt.decision is PromotionDecision.RETRY:
                status = (
                    "reproduction_pending"
                    if row.screen_lane == "reproduction"
                    else "published"
                )
                decision, reason = "", "screen_retry"
            else:
                status, decision, reason = "held", "", "screen_held"
            self._db.execute(
                "UPDATE reservations SET status=?,screen_status=?,screen_stage_count=?,"
                "decision=?,reason=? WHERE reservation_id=?",
                (
                    status, receipt.decision.value, len(receipt.results),
                    decision, reason, reservation_id,
                ),
            )
        return self.get(reservation_id)

    def demote_promoted_for_rescreen(
        self, reservation_id: str, *, reason: str
    ) -> "IntakeReservation":
        """Return one promoted reservation to the screen queue.

        A promoted row carries a screen receipt bound to the service identity
        that produced it. Once that identity is retired the receipt can no
        longer authorize qualification, and because the qualification selector
        is deterministic it would otherwise re-pick the same unusable head for
        as long as the row stays promoted. Demotion re-enters the row in the
        screen FIFO so a fresh receipt is produced under the live identity.
        Retained screen dispositions are append-only and are not disturbed."""

        if type(reason) is not str or not reason or len(reason) > 64:
            raise IntakeError("rescreen reason is malformed")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status != "promoted" or row.screen_status != "promote":
                raise IntakeError("only a promoted reservation may be rescreened")
            # Mirrors the screen-retry disposition, which is the proven inverse
            # of promotion and keeps the reproduction lane in its own queue.
            status = (
                "reproduction_pending"
                if row.screen_lane == "reproduction"
                else "published"
            )
            self._db.execute(
                "UPDATE reservations SET status=?,screen_status='',decision='',"
                "reason=? WHERE reservation_id=?",
                (status, reason, reservation_id),
            )
        return self.get(reservation_id)

    def latest_promoted_screen(self, reservation_id: str):
        from cacheon.arena_service import (
            ArenaScreenReceipt, PromotionDecision, ScreenGrade, ScreenStageResult,
        )

        row = self.get(reservation_id)
        if row.status != "promoted" or row.screen_status != "promote":
            raise IntakeError("reservation has no standing promoted screen")
        retained = self._db.execute(
            "SELECT receipt_digest,receipt_json,stage_count FROM "
            "arena_screen_dispositions WHERE reservation_id=? "
            "ORDER BY attempt_index DESC LIMIT 1",
            (reservation_id,),
        ).fetchone()
        if retained is None:
            raise IntakeError("promoted screen receipt is missing")
        try:
            raw = json.loads(retained["receipt_json"])
            results = tuple(
                ScreenStageResult(
                    item["stage"],
                    ScreenGrade(item["grade"]),
                    item["evidence_digest"],
                    item["elapsed_ms"],
                )
                for item in raw["results"]
            )
            receipt = ArenaScreenReceipt(
                raw["service_digest"], raw["candidate_digest"],
                raw["screen_attempt"], results,
                PromotionDecision(raw["decision"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"promoted screen receipt is corrupt: {exc}") from None
        if (
            receipt.digest != retained["receipt_digest"]
            or len(receipt.results) != retained["stage_count"]
            or receipt.decision is not PromotionDecision.PROMOTE
        ):
            raise IntakeError("promoted screen receipt differs from retained bytes")
        return receipt

    def promoted(self, *, limit: int | None = None) -> tuple[IntakeReservation, ...]:
        bound = self.policy.max_cohort if limit is None else limit
        if type(bound) is not int or bound <= 0 or bound > self.policy.max_cohort:
            raise IntakeError("promoted cohort limit is invalid")
        first = self._db.execute(
            "SELECT retry_group_digest,screen_lane FROM reservations AS r "
            "WHERE status='promoted' AND NOT EXISTS (SELECT 1 FROM "
            "evaluation_lease_members AS em WHERE em.reservation_id=r.reservation_id "
            "AND em.active=1) ORDER BY "
            "CASE screen_lane WHEN 'reproduction' THEN 0 ELSE 1 END,"
            "block,event_index,event_subindex,hotkey,content_hash LIMIT 1"
        ).fetchone()
        if first is None:
            return ()
        if first["screen_lane"] == "reproduction":
            rows = self._db.execute(
                "SELECT r.* FROM reservations AS r WHERE status='promoted' "
                "AND screen_lane='reproduction' AND NOT EXISTS (SELECT 1 FROM "
                "evaluation_lease_members AS em WHERE em.reservation_id=r.reservation_id "
                "AND em.active=1) ORDER BY block,event_index,"
                "event_subindex,hotkey,content_hash LIMIT 1"
            )
        elif first["retry_group_digest"]:
            rows = self._db.execute(
                "SELECT r.* FROM reservations AS r WHERE status='promoted' "
                "AND retry_group_digest=? AND NOT EXISTS (SELECT 1 FROM "
                "evaluation_lease_members AS em WHERE em.reservation_id=r.reservation_id "
                "AND em.active=1) ORDER BY retry_position LIMIT ?",
                (first["retry_group_digest"], bound),
            )
        else:
            rows = self._db.execute(
                "SELECT r.* FROM reservations AS r WHERE status='promoted' "
                "AND screen_lane='primary' AND retry_group_digest='' AND NOT EXISTS "
                "(SELECT 1 FROM evaluation_lease_members AS em WHERE "
                "em.reservation_id=r.reservation_id AND em.active=1) ORDER BY "
                "block,event_index,event_subindex,hotkey,content_hash LIMIT ?",
                (bound,),
            )
        return tuple(self._row(row) for row in rows)

    def settlement_blockers(self, reservation_id: str) -> tuple[IntakeReservation, ...]:
        candidate = self.get(reservation_id)
        if not candidate.target_members:
            raise IntakeError("candidate has no resolved target members")
        blockers: list[IntakeReservation] = []
        for row in self.all():
            if row.arrival.arrival_key >= candidate.arrival.arrival_key:
                break
            if row.status in _TERMINAL:
                continue
            if not row.target_members or set(row.target_members) & set(candidate.target_members):
                blockers.append(row)
        return tuple(blockers)

    def copy_predecessors(self, reservation_id: str) -> tuple[IntakeReservation, ...]:
        candidate = self.get(reservation_id)
        if candidate.delta_fingerprint is None:
            raise IntakeError("candidate has no submitted-delta fingerprint")
        matches: list[IntakeReservation] = []
        for row in self.all():
            if row.arrival.arrival_key >= candidate.arrival.arrival_key:
                break
            if (
                row.arrival.hotkey == candidate.arrival.hotkey
                or row.delta_fingerprint is None
            ):
                continue
            if compare_submitted_deltas(
                row.delta_fingerprint, candidate.delta_fingerprint
            ).authoritative:
                matches.append(row)
        return tuple(matches)

    def reconcile_copies(self) -> tuple[tuple[str, str], ...]:
        """Idempotently demote every unresolved later copy in finalized order."""

        dispositions = []
        for row in self.all():
            if row.delta_fingerprint is None or row.status in {"failed", "expired"}:
                continue
            predecessors = self.copy_predecessors(row.reservation_id)
            if predecessors:
                predecessor = predecessors[0]
                self.mark_copy(row.reservation_id, predecessor.reservation_id)
                dispositions.append((row.reservation_id, predecessor.reservation_id))
        return tuple(dispositions)

    def mark_copy(self, reservation_id: str, predecessor_id: str) -> IntakeReservation:
        predecessor = self.get(predecessor_id)
        candidate = self.get(reservation_id)
        if predecessor.arrival.arrival_key >= candidate.arrival.arrival_key:
            raise IntakeError("copy predecessor is not earlier in finalized order")
        if predecessor not in self.copy_predecessors(reservation_id):
            raise IntakeError("claimed predecessor is not an authoritative delta copy")
        return self._transition(
            reservation_id,
            {
                "published", "screening", "promoted", "qualifying",
                "reproduction_pending", "qualified", "held", "no_decision",
            },
            "failed",
            "FAIL",
            f"copy_of:{predecessor.reservation_id}",
        )

    def mark_reference_copy(
        self, reservation_id: str, reference_name: str
    ) -> IntakeReservation:
        """Demote an authoritative copy of a validator-published bundle."""

        if not isinstance(reference_name, str) or not re.fullmatch(
            r"[A-Za-z0-9._-]{1,80}", reference_name
        ):
            raise IntakeError("validator reference name is malformed")
        return self._transition(
            reservation_id,
            {
                "published", "screening", "promoted", "qualifying",
                "reproduction_pending", "qualified", "held", "no_decision",
            },
            "failed",
            "FAIL",
            f"copy_of:validator_reference:{reference_name}",
        )

    def mark_qualifying(
        self,
        reservation_id: str,
        authority_digest: str,
        authority_manifest: dict[str, object],
    ) -> IntakeReservation:
        require_sha256_hex(authority_digest, field="qualification_authority_digest")
        if type(authority_manifest) is not dict or not authority_manifest:
            raise IntakeError("qualification authority manifest is not a closed object")
        authority_json = json.dumps(
            authority_manifest, separators=(",", ":"), sort_keys=True
        )
        if len(authority_json.encode("utf-8")) > 1 << 20:
            raise IntakeError("qualification authority manifest is oversized")
        with self._transaction():
            self._require_evaluation_mutation_authority(reservation_id)
            row = self.get(reservation_id)
            if row.status != "promoted" or row.screen_status != "promote":
                raise IntakeError("only screen-promoted intake may enter qualification")
            self._db.execute(
                "UPDATE reservations SET status='qualifying',qualification_authority_digest=?,"
                "qualification_authority_json=?,"
                "decision='',reason='' WHERE reservation_id=?",
                (authority_digest, authority_json, reservation_id),
            )
        return self.get(reservation_id)

    def qualification_dispositions(self, reservation_id: str) -> tuple[dict[str, object], ...]:
        self.get(reservation_id)
        result = []
        for row in self._db.execute(
            "SELECT attempt_index,authority_digest,authority_manifest_json,evidence_digest,"
            "attempt_ref_json,report_digest,failure_digest,decision,reason "
            "FROM qualification_dispositions WHERE reservation_id=? ORDER BY attempt_index",
            (reservation_id,),
        ):
            value = dict(row)
            value["authority_manifest"] = json.loads(
                value.pop("authority_manifest_json")
            )
            attempt_ref_json = value.pop("attempt_ref_json")
            value["attempt_ref"] = (
                json.loads(attempt_ref_json) if attempt_ref_json else None
            )
            result.append(value)
        return tuple(result)

    def apply_qualification_batch(
        self,
        batch,
        *,
        current_finalized_block: int,
        evidence_root: str | Path | None = None,
    ) -> tuple[IntakeReservation, ...]:
        """Persist one typed cohort result and its retry groups atomically."""

        from cacheon.eval.qualification_intake import QualificationIntakeBatch
        from cacheon.eval.qualification import QualificationDecision
        from cacheon.settlement import SettlementCandidate, SettlementQualification

        if type(batch) is not QualificationIntakeBatch:
            raise IntakeError("qualification batch is not exactly typed")
        cursor = self._cursor()
        if (
            type(current_finalized_block) is not int
            or current_finalized_block < 0
            or (cursor is not None and current_finalized_block < cursor[0])
        ):
            raise IntakeError("qualification progress block is not finalized")
        if any(
            outcome.decision is QualificationDecision.PASS
            and type(outcome.settlement_qualification) is not SettlementQualification
            for outcome in batch.outcomes
        ):
            raise IntakeError(
                "PASS qualification lacks a settlement projection qualification"
            )
        root = None if evidence_root is None else Path(evidence_root)
        if any(
            outcome.decision is QualificationDecision.PASS
            for outcome in batch.outcomes
        ) and (
            root is None
            or not root.is_absolute()
            or root != Path(os.path.normpath(root))
        ):
            raise IntakeError("PASS qualification lacks a canonical evidence root")
        retry: dict[str, tuple[str, int, str]] = {}
        if batch.retry_plan is not None:
            for group_index, group in enumerate(batch.retry_plan.reservation_groups):
                group_digest = canonical_digest(
                    _QUALIFICATION_RETRY_GROUP_DOMAIN,
                    {
                        "authority_manifest_digest": batch.authority_manifest_digest,
                        "group_index": group_index,
                        "members": list(group),
                        "strategy": batch.retry_plan.strategy,
                    },
                )
                for position, reservation_id in enumerate(group):
                    retry[reservation_id] = (
                        group_digest,
                        position,
                        f"qualification_{batch.retry_plan.strategy}",
                    )
        with self._transaction():
            for outcome in batch.outcomes:
                reservation_id = outcome.reservation_digest
                row = self.get(reservation_id)
                if (
                    row.status != "qualifying"
                    or row.qualification_authority_digest
                    != batch.authority_manifest_digest
                    or row.delta_fingerprint is None
                    or row.delta_fingerprint.selected_delta_digest
                    != outcome.selected_delta_digest
                ):
                    raise IntakeError("qualification batch differs from active authority")
                authority_json = self._db.execute(
                    "SELECT qualification_authority_json FROM reservations WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["qualification_authority_json"]
                attempt_ref = (
                    batch.attempt_ref
                    if outcome.attempt_artifact_sha256 is not None
                    else None
                )
                attempt_json = (
                    json.dumps(
                        attempt_ref.to_dict(), separators=(",", ":"), sort_keys=True
                    )
                    if attempt_ref is not None
                    else ""
                )
                evidence = (
                    attempt_ref.sha256
                    if attempt_ref is not None
                    else outcome.failure_digest or ""
                )
                attempt = self._db.execute(
                    "SELECT COUNT(*) AS n FROM qualification_dispositions WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["n"]
                self._db.execute(
                    "INSERT INTO qualification_dispositions(reservation_id,attempt_index,"
                    "authority_digest,authority_manifest_json,evidence_digest,attempt_ref_json,"
                    "report_digest,failure_digest,decision,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        reservation_id, attempt, batch.authority_manifest_digest,
                        authority_json, evidence, attempt_json,
                        outcome.report_digest or "", outcome.failure_digest or "",
                        outcome.decision.value, outcome.reason,
                    ),
                )
                qualification = outcome.settlement_qualification
                if qualification is not None:
                    if (
                        type(qualification) is not SettlementQualification
                        or qualification.reservation_digest != reservation_id
                        or qualification.hotkey != row.arrival.hotkey
                        or (
                            qualification.finalized_block,
                            qualification.event_index,
                            qualification.event_subindex,
                        )
                        != (
                            row.arrival.block,
                            row.arrival.event_index,
                            row.arrival.event_subindex,
                        )
                        or qualification.target_id != row.target_id
                        or qualification.members != row.target_members
                        or qualification.selected_delta_digest
                        != row.delta_fingerprint.selected_delta_digest
                        or qualification.qualification_authority_digest
                        != batch.authority_manifest_digest
                        or qualification.qualification_attempt_digest != evidence
                        or qualification.qualification_report_digest
                        != outcome.report_digest
                    ):
                        raise IntakeError(
                            "settlement qualification differs from retained PASS"
                        )
                    self.evaluation_stack(qualification.arena_digest)
                    qualification_json = json.dumps(
                        qualification.to_dict(), separators=(",", ":"), sort_keys=True
                    )
                    if attempt_ref is None or root is None:
                        raise IntakeError("retained PASS evidence is incomplete")
                    retained = self._db.execute(
                        "SELECT reproduction_index,qualification_digest,qualification_json,"
                        "attempt_ref_json,evidence_root FROM settlement_qualifications "
                        "WHERE reservation_id=? ORDER BY reproduction_index",
                        (reservation_id,),
                    ).fetchall()
                    expected_lane = "primary" if not retained else "reproduction"
                    if row.screen_lane != expected_lane or len(retained) > 1:
                        raise IntakeError("qualification PASS used the wrong reproduction lane")
                    reproduction_index = len(retained)
                    self._db.execute(
                        "INSERT INTO settlement_qualifications(reservation_id,"
                        "reproduction_index,qualification_digest,qualification_json,"
                        "attempt_ref_json,evidence_root,retained_block) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (
                            reservation_id, reproduction_index, qualification.digest,
                            qualification_json, attempt_json, str(root),
                            current_finalized_block,
                        ),
                    )
                    if reproduction_index == 1:
                        try:
                            primary = SettlementQualification.from_dict(
                                json.loads(retained[0]["qualification_json"])
                            )
                            candidate = SettlementCandidate.from_reproductions(
                                primary, qualification
                            )
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            raise IntakeError(
                                f"independent reproduction is inconsistent: {exc}"
                            ) from None
                        if primary.digest != retained[0]["qualification_digest"]:
                            raise IntakeError("primary settlement qualification is corrupt")
                        candidate_json = json.dumps(
                            candidate.to_dict(), separators=(",", ":"), sort_keys=True
                        )
                        self._db.execute(
                            "INSERT INTO settlement_candidates(reservation_id,authority_digest,"
                            "candidate_digest,candidate_json,evidence_root,"
                            "reproduction_evidence_root,status) "
                            "VALUES(?,?,?,?,?,?, 'pending')",
                            (
                                reservation_id,
                                primary.qualification_authority_digest,
                                candidate.digest,
                                candidate_json,
                                retained[0]["evidence_root"],
                                str(root),
                            ),
                        )
                if reservation_id in retry:
                    group, position, reason = retry[reservation_id]
                    retry_status = (
                        "reproduction_pending"
                        if row.screen_lane == "reproduction"
                        else "published"
                    )
                    status = retry_status if (
                        attempt + 1 < self.policy.max_qualification_retries
                    ) else "held"
                    self._db.execute(
                        "UPDATE reservations SET status=?,decision=?,reason=?,"
                        "retry_group_digest=?,retry_position=?,qualification_authority_digest='',"
                        "qualification_authority_json='',qualification_evidence_digest='' "
                        "WHERE reservation_id=?",
                        (
                            status,
                            "" if status in {"published", "reproduction_pending"}
                            else "",
                            reason, group, position, reservation_id,
                        ),
                    )
                else:
                    if outcome.decision is QualificationDecision.PASS:
                        completed = self._db.execute(
                            "SELECT COUNT(*) AS n FROM settlement_qualifications "
                            "WHERE reservation_id=?",
                            (reservation_id,),
                        ).fetchone()["n"]
                        status = "qualified" if completed == 2 else "reproduction_pending"
                        decision = "PASS" if completed == 2 else ""
                        reason = (
                            outcome.reason if completed == 2 else "reproduction_pending"
                        )
                    elif outcome.decision is QualificationDecision.FAIL:
                        status, decision, reason = (
                            "failed", "FAIL", outcome.reason
                        )
                    else:
                        # A validator/infrastructure NO_DECISION is never a
                        # candidate failure.  Typed retry plans take the branch
                        # above; a terminal no-plan disposition remains explicit
                        # and operator-releasable rather than fabricating FAIL.
                        status, decision, reason = (
                            "no_decision", "NO_DECISION", outcome.reason
                        )
                    self._db.execute(
                        "UPDATE reservations SET status=?,decision=?,reason=?,"
                        "qualification_evidence_digest=?,retry_group_digest='',retry_position=0,"
                        "qualification_authority_digest='',qualification_authority_json='' "
                        "WHERE reservation_id=?",
                        (
                            status, decision, reason,
                            evidence, reservation_id,
                        ),
                    )
        return tuple(self.get(row.reservation_digest) for row in batch.outcomes)

    # ---- transactional settlement and evaluation-stack authority ----

    def _initialize_evaluation_stack_row(
        self,
        manifest: EvaluationStackManifest,
        *,
        tree_digest: str,
    ) -> None:
        """Initialize one stack inside the caller's transaction, idempotently."""

        from cacheon.stack_manifest import EvaluationStackManifest

        if type(manifest) is not EvaluationStackManifest:
            raise IntakeError("initial evaluation stack is not exactly typed")
        require_sha256_hex(tree_digest, field="tree_digest")
        arena = manifest.arena_digest
        genesis = canonical_digest(
            _EVALUATION_STACK_GENESIS_DOMAIN,
            {
                "arena_digest": arena,
                "stack_digest": manifest.digest,
                "tree_digest": tree_digest,
            },
        )
        encoded = json.dumps(
            manifest.to_dict(), separators=(",", ":"), sort_keys=True
        )
        existing = self._db.execute(
            "SELECT * FROM evaluation_stacks WHERE arena_id=?", (arena,)
        ).fetchone()
        if existing is None:
            self._db.execute(
                "INSERT INTO evaluation_stacks(arena_id,generation,stack_digest,"
                "tree_digest,stack_json,transition_event_id) VALUES(?,0,?,?,?,?)",
                (arena, manifest.digest, tree_digest, encoded, genesis),
            )
        elif existing["generation"] == 0 and (
            existing["stack_digest"] != manifest.digest
            or existing["tree_digest"] != tree_digest
            or existing["stack_json"] != encoded
        ):
            raise IntakeError("genesis qualification names another incumbent")

    def initialize_evaluation_stack(
        self,
        manifest: EvaluationStackManifest,
        *,
        tree_digest: str,
    ) -> EvaluationStackState:
        """Install one exact genesis incumbent, or reopen the identical state."""

        with self._transaction():
            self._initialize_evaluation_stack_row(manifest, tree_digest=tree_digest)
        state = self.evaluation_stack(manifest.arena_digest)
        if (
            state.manifest.digest != manifest.digest
            or state.tree_digest != tree_digest
        ):
            raise IntakeError("evaluation stack is already initialized differently")
        return state

    def evaluation_stack(self, arena_digest: str) -> EvaluationStackState:
        from cacheon.stack_manifest import EvaluationStackManifest

        require_sha256_hex(arena_digest, field="arena_digest")
        row = self._db.execute(
            "SELECT * FROM evaluation_stacks WHERE arena_id=?", (arena_digest,)
        ).fetchone()
        if row is None:
            raise IntakeError("evaluation stack is not initialized")
        try:
            manifest = EvaluationStackManifest.from_dict(json.loads(row["stack_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"evaluation stack state is corrupt: {exc}") from None
        if manifest.digest != row["stack_digest"]:
            raise IntakeError("evaluation stack digest differs from stored bytes")
        return EvaluationStackState(
            row["arena_id"], row["generation"], manifest, row["tree_digest"],
            row["transition_event_id"],
        )

    @staticmethod
    def _settlement_candidate(row: sqlite3.Row) -> SettlementCandidate:
        from cacheon.settlement import SettlementCandidate

        try:
            candidate = SettlementCandidate.from_dict(json.loads(row["candidate_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"settlement candidate is corrupt: {exc}") from None
        if candidate.digest != row["candidate_digest"]:
            raise IntakeError("settlement candidate digest differs from stored bytes")
        return candidate

    def _event_head(self) -> tuple[int, str]:
        row = self._db.execute(
            "SELECT sequence,event_digest FROM settlement_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return (0, "") if row is None else (row["sequence"] + 1, row["event_digest"])

    def _settlement_evidence_metadata(
        self,
        candidate: SettlementCandidate,
    ):
        from cacheon.settlement import SettlementEvidence, SettlementQualification

        row = self._db.execute(
            "SELECT sc.evidence_root,sc.reproduction_evidence_root,"
            "sc.candidate_digest,r.status,r.decision FROM settlement_candidates sc "
            "JOIN reservations r USING(reservation_id) WHERE sc.reservation_id=?",
            (candidate.reservation_digest,),
        ).fetchone()
        if (
            row is None
            or row["candidate_digest"] != candidate.digest
            or row["status"] != "qualified"
            or row["decision"] != "PASS"
            or not row["evidence_root"]
            or not row["reproduction_evidence_root"]
        ):
            raise IntakeError("settlement evidence no longer has standing authority")
        retained = tuple(
            self._db.execute(
                "SELECT reproduction_index,qualification_digest,qualification_json,"
                "attempt_ref_json,evidence_root FROM settlement_qualifications "
                "WHERE reservation_id=? ORDER BY reproduction_index",
                (candidate.reservation_digest,),
            )
        )
        if len(retained) != 2 or tuple(
            item["reproduction_index"] for item in retained
        ) != (0, 1):
            raise IntakeError("settlement candidate lacks two retained qualifications")
        qualifications = []
        references = []
        try:
            for item in retained:
                qualification = SettlementQualification.from_dict(
                    json.loads(item["qualification_json"])
                )
                reference = EvidenceArtifactRef.from_dict(
                    json.loads(item["attempt_ref_json"])
                )
                if (
                    qualification.digest != item["qualification_digest"]
                    or reference.sha256
                    != qualification.qualification_attempt_digest
                ):
                    raise IntakeError("retained reproduction identity differs")
                disposition = self._db.execute(
                    "SELECT authority_digest,report_digest,decision FROM "
                    "qualification_dispositions WHERE reservation_id=? "
                    "AND evidence_digest=?",
                    (
                        candidate.reservation_digest,
                        qualification.qualification_attempt_digest,
                    ),
                ).fetchone()
                if (
                    disposition is None
                    or disposition["decision"] != "PASS"
                    or disposition["authority_digest"]
                    != qualification.qualification_authority_digest
                    or disposition["report_digest"]
                    != qualification.qualification_report_digest
                ):
                    raise IntakeError("retained reproduction lost PASS authority")
                qualifications.append(qualification)
                references.append(reference)
        except IntakeError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"settlement reproduction is corrupt: {exc}") from None
        if tuple(qualifications) != (candidate.primary, candidate.reproduction):
            raise IntakeError("retained reproductions differ from settlement candidate")
        roots = (Path(retained[0]["evidence_root"]), Path(retained[1]["evidence_root"]))
        if roots != (
            Path(row["evidence_root"]), Path(row["reproduction_evidence_root"])
        ):
            raise IntakeError("settlement reproduction roots differ")
        receipt = SettlementEvidence.bind(
            candidate,
            primary_attempt_ref=references[0],
            reproduction_attempt_ref=references[1],
        )
        return roots, tuple(references), receipt

    def reopen_settlement_evidence(
        self,
        candidate: SettlementCandidate,
    ):
        """Reopen retained qualification bytes without duplicating their grader."""

        from cacheon.eval.evidence_store import EvidenceStoreError, reopen_evidence
        from cacheon.settlement import SettlementCandidate

        if type(candidate) is not SettlementCandidate:
            raise IntakeError("settlement evidence candidate is not exactly typed")
        roots, references, receipt = self._settlement_evidence_metadata(candidate)
        try:
            for root, reference in zip(roots, references, strict=True):
                reopen_evidence(root, reference)
        except (EvidenceStoreError, OSError) as exc:
            raise IntakeError(f"retained settlement evidence cannot reopen: {exc}") from None
        return receipt

    def _economic_blockers(
        self,
        candidate: SettlementCandidate,
        *,
        cohort_ids: frozenset[str],
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        for row in self.all():
            if row.arrival.arrival_key >= self.get(
                candidate.reservation_digest
            ).arrival.arrival_key:
                break
            if row.reservation_id in cohort_ids:
                continue
            if row.status in {"failed", "expired"}:
                continue
            if row.target_members and not (
                set(row.target_members) & set(candidate.members)
            ):
                continue
            if row.status == "qualified":
                economic = self._db.execute(
                    "SELECT status FROM settlement_candidates WHERE reservation_id=?",
                    (row.reservation_id,),
                ).fetchone()
                if economic is not None and economic["status"] in {
                    "crowned", "neutralized", "held", "discovery_bounty",
                    "duplicate_proposal", "review_pending", "reviewed_bounty",
                    "reviewed_promotion", "review_ineligible",
                    "review_expired",
                }:
                    continue
            blockers.append(row.reservation_id)
        return tuple(blockers)

    def has_pending_settlement(self) -> bool:
        """Return whether retained settlement work may currently be leased."""

        if (
            self._db.execute(
                "SELECT 1 FROM evaluation_leases WHERE state='active' "
                "AND stage='qualification' LIMIT 1"
            ).fetchone() is not None
        ):
            return False

        return self._db.execute(
            "SELECT 1 FROM settlement_candidates WHERE status='pending' LIMIT 1"
        ).fetchone() is not None

    def lease_settlement_cohort(
        self,
        *,
        current_block: int,
        lease_blocks: int = 30,
    ) -> SettlementLease | None:
        """Lease the oldest economically unblocked retained PASS cohort."""

        if (
            type(current_block) is not int
            or current_block < 0
            or type(lease_blocks) is not int
            or lease_blocks <= 0
        ):
            raise IntakeError("settlement lease bounds are malformed")
        with self._transaction():
            # Do not lease settlement work while a qualification
            # evaluation lease is active.
            if (
                self._db.execute(
                    "SELECT 1 FROM evaluation_leases WHERE state='active' "
                    "AND stage='qualification' LIMIT 1"
                ).fetchone() is not None
            ):
                return None
            # A stale unresolved predecessor must not retain economic priority
            # after the finalized-block SLA.  Do this atomically with leasing so
            # no caller can forget the liveness transition.
            self._expire_stale_rows(current_block)
            self._db.execute(
                "UPDATE settlement_candidates SET status='held',lease_id='',"
                "lease_expires_block=0,reason='intake_no_longer_qualified' "
                "WHERE status IN ('pending','leased') AND reservation_id IN "
                "(SELECT reservation_id FROM reservations WHERE status!='qualified')"
            )
            self._db.execute(
                "UPDATE settlement_candidates SET status='pending',lease_id='',"
                "lease_generation=lease_generation+1,lease_expires_block=0,"
                "reason='settlement_lease_expired' WHERE status='leased' "
                "AND lease_expires_block<=?",
                (current_block,),
            )
            pending = tuple(
                self._db.execute(
                    "SELECT sc.*,r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash "
                    "FROM settlement_candidates sc JOIN reservations r USING(reservation_id) "
                    "WHERE sc.status='pending' AND r.status='qualified' "
                    "ORDER BY r.block,r.event_index,"
                    "r.event_subindex,r.hotkey,r.content_hash"
                )
            )
            chosen: tuple[sqlite3.Row, ...] | None = None
            for first in pending:
                group = tuple(
                    row for row in pending
                    if row["authority_digest"] == first["authority_digest"]
                )
                candidates = tuple(self._settlement_candidate(row) for row in group)
                if len({row.arena_digest for row in candidates}) != 1:
                    raise IntakeError("one qualification authority spans multiple arenas")
                ids = frozenset(row.reservation_digest for row in candidates)
                if any(
                    self._economic_blockers(row, cohort_ids=ids)
                    for row in candidates
                ):
                    continue
                chosen = group
                break
            if chosen is None:
                return None
            candidates = tuple(self._settlement_candidate(row) for row in chosen)
            stack = self.evaluation_stack(candidates[0].arena_digest)
            generation = max(row["lease_generation"] for row in chosen) + 1
            expires = current_block + lease_blocks
            lease_id = canonical_digest(
                _SETTLEMENT_LEASE_DOMAIN,
                {
                    "authority_digest": chosen[0]["authority_digest"],
                    "candidates": [row.digest for row in candidates],
                    "generation": generation,
                    "incumbent_generation": stack.generation,
                    "lease_block": current_block,
                },
            )
            ids = tuple(row.reservation_digest for row in candidates)
            marks = ",".join("?" for _ in ids)
            cursor = self._db.execute(
                f"UPDATE settlement_candidates SET status='leased',lease_id=?,"
                f"lease_generation=?,lease_expires_block=?,reason='' "
                f"WHERE status='pending' AND reservation_id IN ({marks})",
                (lease_id, generation, expires, *ids),
            )
            if cursor.rowcount != len(ids):
                raise IntakeError("settlement cohort changed while leasing")
            sequence, previous = self._event_head()
        return SettlementLease(
            lease_id,
            chosen[0]["authority_digest"],
            generation,
            expires,
            stack,
            candidates,
            sequence,
            previous,
        )

    def commit_settlement(
        self,
        lease: SettlementLease,
        plan: SettlementPlan,
        evidence,
        *,
        current_block: int,
    ) -> EvaluationStackState:
        """Atomically commit one independently planned retained-evidence disposition."""

        from cacheon.economics import StandingRewardClaim, WEIGHT_PPM
        from cacheon.settlement import (
            SettlementEvidence,
            SettlementEventType,
            SettlementPlan,
            plan_settlement,
        )

        if type(lease) is not SettlementLease or type(plan) is not SettlementPlan:
            raise IntakeError("settlement commit is not exactly typed")
        receipts = tuple(evidence)
        if (
            type(current_block) is not int
            or current_block < 0
            or current_block >= lease.expires_block
            or len(receipts) != len(lease.candidates)
            or any(type(row) is not SettlementEvidence for row in receipts)
            or {row.candidate_digest for row in receipts}
            != {row.digest for row in lease.candidates}
        ):
            raise IntakeError("settlement evidence or lease deadline is invalid")
        expected = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        if expected.to_dict() != plan.to_dict():
            raise IntakeError("settlement plan differs from its leased authority")
        by_digest = {row.digest: row for row in lease.candidates}
        evidence_by_candidate = {row.candidate_digest: row for row in receipts}
        with self._transaction():
            if self._db.execute(
                "SELECT 1 FROM evaluation_leases WHERE state='active' "
                "AND stage='qualification' LIMIT 1"
            ).fetchone() is not None:
                raise IntakeError(
                    "settlement commit is fenced by active qualification"
                )
            # Re-evaluate the same finalized-block SLA at commit time.  Opening
            # retained evidence may cross the boundary after the lease was made.
            self._expire_stale_rows(current_block)
            current = self.evaluation_stack(lease.stack.arena_digest)
            if current != lease.stack or self._event_head() != (
                lease.initial_event_sequence, lease.previous_event_digest
            ):
                raise IntakeError("settlement incumbent or journal advanced")
            ids = tuple(row.reservation_digest for row in lease.candidates)
            cohort_ids = frozenset(ids)
            if any(
                self._economic_blockers(candidate, cohort_ids=cohort_ids)
                for candidate in lease.candidates
            ):
                raise IntakeError("settlement priority changed while evidence was open")
            for candidate in lease.candidates:
                _roots, _references, expected_receipt = (
                    self._settlement_evidence_metadata(candidate)
                )
                if expected_receipt != evidence_by_candidate[candidate.digest]:
                    raise IntakeError("settlement evidence changed after reopening")
            marks = ",".join("?" for _ in ids)
            active = tuple(
                self._db.execute(
                    f"SELECT sc.reservation_id,sc.candidate_digest FROM settlement_candidates sc "
                    f"JOIN reservations r USING(reservation_id) WHERE sc.status='leased' "
                    f"AND r.status='qualified' AND sc.lease_id=? AND sc.lease_generation=? "
                    f"AND sc.reservation_id IN ({marks})",
                    (lease.lease_id, lease.generation, *ids),
                )
            )
            if len(active) != len(ids) or {
                row["candidate_digest"] for row in active
            } != set(by_digest):
                raise IntakeError("settlement lease is stale or incomplete")

            commit_plan = plan

            # Retire/neutralize old families before installing the winner family.
            for event in commit_plan.events:
                if event.event_type in {
                    SettlementEventType.RETIREMENT,
                    SettlementEventType.NEUTRALIZATION,
                }:
                    self._db.execute(
                        "UPDATE standing_reward_claims SET status='inactive',event_id=? "
                        "WHERE arena_id=? AND target_id=?",
                        (event.digest, lease.stack.arena_digest, event.target_id),
                    )

            disposition: dict[str, str] = {}
            for event in commit_plan.events:
                candidate = by_digest[event.candidate_digest]
                event_json = json.dumps(
                    event.to_dict(), separators=(",", ":"), sort_keys=True
                )
                self._db.execute(
                    "INSERT INTO settlement_events(sequence,event_id,event_type,reservation_id,"
                    "arena_id,target_id,event_digest,event_json) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        event.sequence,
                        event.digest,
                        event.event_type.value,
                        candidate.reservation_digest,
                        candidate.arena_digest,
                        event.target_id,
                        event.digest,
                        event_json,
                    ),
                )
                if event.event_type is SettlementEventType.HOLD:
                    disposition[candidate.digest] = "held"
                elif event.event_type is SettlementEventType.CROWN:
                    assert candidate.candidate_manifest is not None
                    contribution = candidate.candidate_manifest.entries[candidate.target_id]
                    speedup_ppm = int(
                        (Decimal(candidate.speedup) * WEIGHT_PPM).to_integral_value(
                            rounding=ROUND_FLOOR
                        )
                    )
                    claim = StandingRewardClaim(
                        candidate.arena_digest,
                        candidate.target_id,
                        contribution.target_spec_digest,
                        contribution.digest,
                        candidate.hotkey,
                        speedup_ppm,
                        candidate.finalized_block,
                        evidence_by_candidate[candidate.digest].digest,
                    )
                    self._db.execute(
                        "INSERT INTO standing_reward_claims(arena_id,target_id,claim_digest,"
                        "claim_json,status,event_id) VALUES(?,?,?,?, 'active',?) "
                        "ON CONFLICT(arena_id,target_id) DO UPDATE SET "
                        "claim_digest=excluded.claim_digest,claim_json=excluded.claim_json,"
                        "status='active',event_id=excluded.event_id",
                        (
                            candidate.arena_digest,
                            candidate.target_id,
                            claim.digest,
                            json.dumps(claim.to_dict(), separators=(",", ":"), sort_keys=True),
                            event.digest,
                        ),
                    )
                    arrival = self._db.execute(
                        "SELECT block,block_hash,event_index,event_subindex,hotkey "
                        "FROM reservations WHERE reservation_id=?",
                        (candidate.reservation_digest,),
                    ).fetchone()
                    if (
                        arrival is None
                        or arrival["block"] != candidate.finalized_block
                        or arrival["event_index"] != candidate.event_index
                        or arrival["event_subindex"] != candidate.event_subindex
                        or arrival["hotkey"] != candidate.hotkey
                    ):
                        raise IntakeError(
                            "CROWN candidate differs from finalized arrival authority"
                        )
                    disposition[candidate.digest] = "crowned"

            if set(disposition) != set(by_digest):
                raise IntakeError("settlement plan did not dispose every leased candidate")
            for digest, status in disposition.items():
                candidate = by_digest[digest]
                self._db.execute(
                    "UPDATE settlement_candidates SET status=?,lease_id='',"
                    "lease_expires_block=0,reason=?,settlement_evidence_digest=? "
                    "WHERE reservation_id=?",
                    (
                        status,
                        "already_awarded"
                        if status == "duplicate_proposal"
                        else status,
                        evidence_by_candidate[digest].digest,
                        candidate.reservation_digest,
                    ),
                )

            if commit_plan.transition is not None:
                manifest = commit_plan.transition.manifest
                encoded = json.dumps(
                    manifest.to_dict(), separators=(",", ":"), sort_keys=True
                )
                transition_id = commit_plan.events[-1].digest
                cursor = self._db.execute(
                    "UPDATE evaluation_stacks SET generation=generation+1,stack_digest=?,"
                    "tree_digest=?,stack_json=?,transition_event_id=? WHERE arena_id=? "
                    "AND generation=? AND stack_digest=? AND tree_digest=?",
                    (
                        manifest.digest,
                        commit_plan.transition.after.tree_digest,
                        encoded,
                        transition_id,
                        lease.stack.arena_digest,
                        lease.stack.generation,
                        lease.stack.manifest.digest,
                        lease.stack.tree_digest,
                    ),
                )
                if cursor.rowcount != 1:
                    raise IntakeError("evaluation stack changed during settlement commit")
        return self.evaluation_stack(lease.stack.arena_digest)

    def active_reward_claims(self) -> tuple[tuple[object, ...], tuple[object, ...]]:
        """Reopen all active standing and discovery claims, or fail as one unit."""

        from cacheon.economics import DiscoveryBountyClaim, StandingRewardClaim

        standing = []
        for row in self._db.execute(
            "SELECT claim_digest,claim_json FROM standing_reward_claims "
            "WHERE status='active' ORDER BY arena_id,target_id"
        ):
            try:
                claim = StandingRewardClaim.from_dict(json.loads(row["claim_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntakeError(f"standing reward claim is corrupt: {exc}") from None
            if claim.digest != row["claim_digest"]:
                raise IntakeError("standing reward claim digest differs")
            standing.append(claim)
        discovery = []
        for row in self._db.execute(
            "SELECT claim_digest,claim_json FROM discovery_bounty_claims "
            "WHERE status='active' ORDER BY proposal_digest"
        ):
            try:
                claim = DiscoveryBountyClaim.from_dict(json.loads(row["claim_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntakeError(f"discovery reward claim is corrupt: {exc}") from None
            if claim.digest != row["claim_digest"]:
                raise IntakeError("discovery reward claim digest differs")
            discovery.append(claim)
        return tuple(standing), tuple(discovery)

    def reopen_active_crown(
        self, arena_digest: str, target_id: str
    ) -> CrownedSettlement:
        """Reopen the exact active CROWN needed by reviewed source promotion."""

        from cacheon.economics import StandingRewardClaim, WEIGHT_PPM
        from cacheon.settlement import SettlementEvent

        require_sha256_hex(arena_digest, field="arena_digest")
        if not isinstance(target_id, str) or not target_id:
            raise IntakeError("active crown target_id is malformed")
        claim_row = self._db.execute(
            "SELECT claim_digest,claim_json,event_id FROM standing_reward_claims "
            "WHERE arena_id=? AND target_id=? AND status='active'",
            (arena_digest, target_id),
        ).fetchone()
        if claim_row is None:
            raise IntakeError("active crown is not retained")
        try:
            claim = StandingRewardClaim.from_dict(json.loads(claim_row["claim_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"active crown claim is corrupt: {exc}") from None
        if claim.digest != claim_row["claim_digest"]:
            raise IntakeError("active crown claim digest differs")
        candidate_row = self._db.execute(
            "SELECT * FROM settlement_candidates WHERE settlement_evidence_digest=? "
            "AND status='crowned'",
            (claim.retained_evidence_digest,),
        ).fetchone()
        event_row = self._db.execute(
            "SELECT event_digest,event_json FROM settlement_events WHERE event_id=? "
            "AND event_type='CROWN'",
            (claim_row["event_id"],),
        ).fetchone()
        if candidate_row is None or event_row is None:
            raise IntakeError("active crown lacks retained settlement authority")
        candidate = self._settlement_candidate(candidate_row)
        evidence = self.reopen_settlement_evidence(candidate)
        try:
            event = SettlementEvent.from_dict(json.loads(event_row["event_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"active crown event is corrupt: {exc}") from None
        replacement = (
            None
            if candidate.candidate_manifest is None
            else candidate.candidate_manifest.entries.get(candidate.target_id)
        )
        current = self.evaluation_stack(arena_digest)
        speedup_ppm = int(
            (Decimal(candidate.speedup) * WEIGHT_PPM).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        if (
            event.digest != event_row["event_digest"]
            or replacement is None
            or current.manifest.entries.get(target_id) != replacement
            or claim.arena_digest != candidate.arena_digest
            or claim.target_id != candidate.target_id
            or claim.target_spec_digest != replacement.target_spec_digest
            or claim.contribution_digest != replacement.digest
            or claim.hotkey != candidate.hotkey
            or claim.speedup_ppm != speedup_ppm
            or claim.crowned_block != candidate.finalized_block
            or claim.retained_evidence_digest != evidence.digest
            or event.subject_digest != replacement.digest
            or event.from_stack_digest != candidate.incumbent_stack_digest
            or event.from_tree_digest != candidate.incumbent_tree_digest
            or event.to_stack_digest != candidate.incumbent_stack_digest
            or event.to_tree_digest != candidate.incumbent_tree_digest
            or event.reason != "qualified_win"
        ):
            raise IntakeError("active crown differs from retained settlement authority")
        return CrownedSettlement(candidate, evidence, event)

    def _reopen_claim_evidence(self, retained_digest: str, status: str):
        require_sha256_hex(retained_digest, field="retained_evidence_digest")
        row = self._db.execute(
            "SELECT * FROM settlement_candidates WHERE settlement_evidence_digest=? "
            "AND status=?",
            (retained_digest, status),
        ).fetchone()
        if row is None:
            raise IntakeError("active reward claim has no standing settlement candidate")
        candidate = self._settlement_candidate(row)
        receipt = self.reopen_settlement_evidence(candidate)
        if receipt.digest != retained_digest:
            raise IntakeError("active reward claim differs from reopened settlement evidence")
        return receipt

    def _bind_emissions_policy(self, policy_digest: str) -> None:
        require_sha256_hex(policy_digest, field="policy_digest")
        with self._transaction():
            row = self._db.execute(
                "SELECT value FROM metadata WHERE key='emissions_policy_digest'"
            ).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO metadata(key,value) VALUES('emissions_policy_digest',?)",
                    (policy_digest,),
                )
            elif row["value"] != policy_digest:
                raise IntakeError(
                    "emissions policy differs from the bound validator consensus state"
                )

    def build_weight_projection(
        self,
        *,
        policy,
        context,
        catalogs: Mapping[str, object],
        netuid: int,
    ) -> WeightProjection:
        """Build one global all-arena vector from the complete retained authority."""

        from cacheon.chain.weights import WeightProjection
        from cacheon.economics import (
            ArenaRewardAuthority,
            EmissionsPolicyManifest,
            GlobalRewardProjectionContext,
            project_global_rewards,
        )
        from cacheon.target_catalog import TargetCatalog

        if (
            type(policy) is not EmissionsPolicyManifest
            or type(context) is not GlobalRewardProjectionContext
            or not isinstance(catalogs, Mapping)
            or type(netuid) is not int
            or netuid < 0
        ):
            raise IntakeError("weight projection authority is malformed")
        standing, discovery = self.active_reward_claims()
        by_arena: dict[str, list[object]] = {}
        for claim in standing:
            by_arena.setdefault(claim.arena_digest, []).append(claim)
        states = self.evaluation_stacks()
        state_ids = {row.arena_digest for row in states}
        active_states = tuple(row for row in states if row.generation > 0)
        active_ids = {row.arena_digest for row in active_states}
        if set(by_arena) - state_ids:
            raise IntakeError("active reward claim belongs to an absent evaluation arena")
        if set(by_arena) - active_ids:
            raise IntakeError("active reward claim belongs to an uncrowned evaluation arena")
        if set(catalogs) - state_ids:
            raise IntakeError("reward catalog names an absent evaluation arena")
        if active_ids - set(catalogs):
            raise IntakeError("reward catalogs do not cover every crowned evaluation arena")
        for claim in standing:
            self._reopen_claim_evidence(claim.retained_evidence_digest, "crowned")
        for claim in discovery:
            self._reopen_claim_evidence(
                claim.retained_evidence_digest, "discovery_bounty"
            )
        authorities = []
        for state in active_states:
            catalog = catalogs[state.arena_digest]
            if type(catalog) is not TargetCatalog:
                raise IntakeError("reward catalog is not exactly typed")
            authorities.append(
                ArenaRewardAuthority(
                    catalog,
                    state.manifest,
                    state.generation,
                    tuple(by_arena.get(state.arena_digest, ())),
                )
            )
        projection = project_global_rewards(
            policy, context, tuple(authorities), discovery
        )
        self._bind_emissions_policy(policy.digest)
        evidence = tuple(
            sorted(
                {
                    claim.retained_evidence_digest
                    for claim in (*standing, *discovery)
                }
            )
        )
        return WeightProjection(
            context.chain_scope_digest,
            netuid,
            context.validator_hotkey,
            policy.digest,
            self.settlement_state_digest(),
            projection.digest,
            context.metagraph_digest,
            projection.arena_authority_digests,
            max((row.generation for row in active_states), default=0),
            context.current_block,
            len(standing),
            evidence,
            tuple(
                (row.hotkey, row.weight_ppm) for row in projection.weights
            ),
        )

    def build_burn_weight_projection(
        self,
        *,
        policy,
        context,
        netuid: int,
        burn_hotkey: str,
    ) -> WeightProjection:
        """Project the full pool to one designated hotkey while nothing is crowned.

        The all-uncrowned bootstrap deliberately fails closed in
        ``build_weight_projection`` because a crown is a payment claim and stock
        cannot hold one.  Directing the pool at the subnet owner's own burn
        registration is the explicit operator policy for that world, so it must
        become impossible the moment any real economic authority exists: this
        refuses on any active claim, any crowned arena, and any activated
        composition, and the projection digest-binds the empty settlement state
        it was derived from.
        """

        from cacheon.chain.weights import WEIGHT_PARTS, WeightProjection
        from cacheon.economics import (
            EmissionsPolicyManifest,
            GlobalRewardProjectionContext,
        )

        if (
            type(policy) is not EmissionsPolicyManifest
            or type(context) is not GlobalRewardProjectionContext
            or type(netuid) is not int
            or netuid < 0
            or not isinstance(burn_hotkey, str)
            or not burn_hotkey
            or burn_hotkey.strip() != burn_hotkey
            or len(burn_hotkey) > 256
        ):
            raise IntakeError("burn weight projection authority is malformed")
        if burn_hotkey not in context.eligible_hotkeys:
            raise IntakeError(
                "burn hotkey is not registered in the projection metagraph"
            )
        standing, discovery = self.active_reward_claims()
        if standing or discovery:
            raise IntakeError(
                "burn weights refused: active reward claims exist; "
                "project real weights instead"
            )
        if any(row.generation > 0 for row in self.evaluation_stacks()):
            raise IntakeError(
                "burn weights refused: a crowned evaluation arena exists; "
                "project real weights instead"
            )
        settlement_digest = self.settlement_state_digest()
        authority_digest = canonical_digest(
            _BURN_WEIGHT_AUTHORITY_DOMAIN,
            {
                "burn_hotkey": burn_hotkey,
                "chain_scope_digest": context.chain_scope_digest,
                "metagraph_digest": context.metagraph_digest,
                "netuid": netuid,
                "policy_digest": policy.digest,
                "settlement_state_digest": settlement_digest,
                "validator_hotkey": context.validator_hotkey,
            },
        )
        self._bind_emissions_policy(policy.digest)
        return WeightProjection(
            context.chain_scope_digest,
            netuid,
            context.validator_hotkey,
            policy.digest,
            settlement_digest,
            authority_digest,
            context.metagraph_digest,
            (authority_digest,),
            0,
            context.current_block,
            0,
            (),
            ((burn_hotkey, WEIGHT_PARTS),),
        )

    def build_subnet_owner_burn_weight_projection(
        self,
        *,
        policy,
        context,
        netuid: int,
        burn_hotkey: str,
        owner_coldkey: str,
        owner_hotkey: str,
        candidate_uids: tuple[int, ...],
    ) -> WeightProjection:
        """Project the full pool to the resolved subnet-owner burn sink.

        Bootstrap-only, like :meth:`build_burn_weight_projection`: this refuses
        on any active reward claim, any crowned evaluation arena, and any
        activated composition, and digest-binds the empty settlement state it
        was derived from. The burn hotkey is the chain-resolved subnet-owner
        identity rather than an operator-supplied one, and publication goes
        through the durable intent → pending → confirmed/held journal with
        ``require_current_crown=False``.
        """

        from cacheon.chain.weights import (
            SUBNET_OWNER_BURN_AUTHORITY,
            WEIGHT_PARTS,
            WeightProjection,
        )
        from cacheon.economics import (
            EmissionsPolicyManifest,
            GlobalRewardProjectionContext,
        )

        if (
            type(policy) is not EmissionsPolicyManifest
            or type(context) is not GlobalRewardProjectionContext
            or type(netuid) is not int
            or netuid < 0
            or not isinstance(burn_hotkey, str)
            or not burn_hotkey
            or burn_hotkey.strip() != burn_hotkey
            or len(burn_hotkey) > 256
            or not isinstance(owner_coldkey, str)
            or not owner_coldkey
            or owner_coldkey.strip() != owner_coldkey
            or len(owner_coldkey) > 256
            or not isinstance(owner_hotkey, str)
            or owner_hotkey.strip() != owner_hotkey
            or len(owner_hotkey) > 256
            or type(candidate_uids) is not tuple
            or not candidate_uids
            or any(type(uid) is not int or uid < 0 for uid in candidate_uids)
            or candidate_uids != tuple(sorted(set(candidate_uids)))
        ):
            raise IntakeError("subnet-owner burn weight projection authority is malformed")
        if burn_hotkey not in context.eligible_hotkeys:
            raise IntakeError(
                "subnet-owner burn hotkey is not registered in the projection metagraph"
            )
        standing, discovery = self.active_reward_claims()
        if standing or discovery:
            raise IntakeError(
                "subnet-owner burn weights refused: active reward claims exist; "
                "project real weights instead"
            )
        if any(row.generation > 0 for row in self.evaluation_stacks()):
            raise IntakeError(
                "subnet-owner burn weights refused: a crowned evaluation arena "
                "exists; project real weights instead"
            )
        settlement_digest = self.settlement_state_digest()
        authority_digest = canonical_digest(
            SUBNET_OWNER_BURN_AUTHORITY,
            {
                "burn_hotkey": burn_hotkey,
                "candidate_uids": list(candidate_uids),
                "chain_scope_digest": context.chain_scope_digest,
                "metagraph_digest": context.metagraph_digest,
                "netuid": netuid,
                "owner_coldkey": owner_coldkey,
                "owner_hotkey": owner_hotkey,
                "policy_digest": policy.digest,
                "settlement_state_digest": settlement_digest,
                "validator_hotkey": context.validator_hotkey,
            },
        )
        self._bind_emissions_policy(policy.digest)
        return WeightProjection(
            context.chain_scope_digest,
            netuid,
            context.validator_hotkey,
            policy.digest,
            settlement_digest,
            authority_digest,
            context.metagraph_digest,
            (authority_digest,),
            0,
            context.current_block,
            0,
            (),
            ((burn_hotkey, WEIGHT_PARTS),),
        )

    def settlement_state_digest(self) -> str:
        sequence, event = self._event_head()
        stacks = tuple(
            (row["arena_id"], row["generation"], row["stack_digest"], row["tree_digest"])
            for row in self._db.execute(
                "SELECT arena_id,generation,stack_digest,tree_digest "
                "FROM evaluation_stacks ORDER BY arena_id"
            )
        )
        candidates = tuple(
            (row["candidate_digest"], row["status"], row["lease_generation"])
            for row in self._db.execute(
                "SELECT candidate_digest,status,lease_generation FROM settlement_candidates "
                "ORDER BY reservation_id"
            )
        )
        return canonical_digest(
            _SETTLEMENT_STATE_DOMAIN,
            {
                "candidates": candidates,
                "event_head": event,
                "event_sequence": sequence,
                "stacks": stacks,
            },
        )

    def evaluation_stacks(self) -> tuple[EvaluationStackState, ...]:
        return tuple(
            self.evaluation_stack(row["arena_id"])
            for row in self._db.execute(
                "SELECT arena_id FROM evaluation_stacks ORDER BY arena_id"
            )
        )


    def requeue_validator_downtime_expired(
        self,
        reservation_ids: tuple[str, ...],
        *,
        authority_digest: str,
        current_block: int,
        allow_repeat_refresh: bool = False,
    ) -> tuple[IntakeReservation, ...]:
        """Readmit an exact validator-expired cohort with one fresh SLA window.

        Original arrival fields remain immutable and therefore continue to own
        FIFO ordering.  The one-row reset authority only changes the SLA anchor.
        A reservation may use the initial recovery once; if it then re-expires
        under automatic SLA (operator fault / mismatched window), exactly one
        refresh of that anchor is admitted before the budget fails closed.
        """

        if (
            type(reservation_ids) is not tuple
            or not reservation_ids
            or len(set(reservation_ids)) != len(reservation_ids)
        ):
            raise IntakeError("validator downtime requeue cohort is malformed")
        try:
            require_sha256_hex(
                authority_digest, field="validator downtime requeue authority"
            )
        except (TypeError, ValueError) as exc:
            raise IntakeError(str(exc)) from None
        self._require_evaluation_clock(current_block)
        restored: list[tuple[str, str, str]] = []
        with self._transaction():
            rows = tuple(self.get(reservation_id) for reservation_id in reservation_ids)
            for row in rows:
                self._require_evaluation_mutation_authority(row.reservation_id)
                if (
                    row.status != "expired"
                    or row.reason != _AUTOMATIC_EXPIRY_REASON
                ):
                    raise IntakeError(
                        "only an automatic validator-SLA expiry may be requeued"
                    )
                prior = self._db.execute(
                    "SELECT reason FROM reservation_sla_resets WHERE reservation_id=?",
                    (row.reservation_id,),
                ).fetchone()
                if prior is None:
                    reset_reason = _VALIDATOR_DOWNTIME_REQUEUE_REASON
                elif prior["reason"] == _VALIDATOR_DOWNTIME_REQUEUE_REASON:
                    # First requeue already burned; admit one refresh after
                    # the cohort re-expired under the automatic SLA again.
                    reset_reason = _VALIDATOR_DOWNTIME_REQUEUE_REFRESH_REASON
                elif (
                    allow_repeat_refresh
                    and prior["reason"]
                    == _VALIDATOR_DOWNTIME_REQUEUE_REFRESH_REASON
                ):
                    # Owner-escalated repeat: every prior window burned while
                    # the validator itself was down. Only an authority that
                    # explicitly carries the escalation reopens the window;
                    # the default budget stays fail-closed.
                    reset_reason = _VALIDATOR_DOWNTIME_REQUEUE_REFRESH_REASON
                else:
                    raise IntakeError(
                        "validator downtime requeue budget is already consumed"
                    )
                if self._db.execute(
                    "SELECT 1 FROM evaluation_lease_members WHERE reservation_id=? "
                    "AND active=1",
                    (row.reservation_id,),
                ).fetchone() is not None:
                    raise IntakeError(
                        "validator downtime requeue conflicts with an active lease"
                    )
                # Promote restore keys off the live receipt: after a service-
                # identity rotation a row may carry an older promote under the
                # retired service plus a fresh promote under the live one
                # (append-only dispositions).  Require the latest promote to
                # match the reservation's arena_service_digest.
                if row.screen_status == "promote":
                    latest = self._db.execute(
                        "SELECT decision,service_digest FROM arena_screen_dispositions "
                        "WHERE reservation_id=? ORDER BY attempt_index DESC LIMIT 1",
                        (row.reservation_id,),
                    ).fetchone()
                    if (
                        latest is None
                        or latest["decision"] != "promote"
                        or not row.arena_service_digest
                        or latest["service_digest"] != row.arena_service_digest
                    ):
                        raise IntakeError(
                            "validator downtime requeue screen authority is incomplete"
                        )
                    status = "promoted"
                    clear_screen = False
                elif (
                    row.screen_status == ""
                    and row.publication_digest
                    and row.publication_root
                ):
                    status = "published"
                    clear_screen = False
                elif (
                    row.screen_status in ("hold", "retry")
                    and row.publication_digest
                    and row.publication_root
                ):
                    # Mid-screen when the SLA expired: drop back to the
                    # pre-screen published queue so screening starts fresh.
                    # Disposition history stays append-only.
                    status = "published"
                    clear_screen = True
                else:
                    raise IntakeError(
                        "validator downtime requeue cannot restore this pipeline phase"
                    )
                restored.append(
                    (row.reservation_id, status, reset_reason, clear_screen)
                )
            for reservation_id, _status, reset_reason, _clear in restored:
                if reset_reason == _VALIDATOR_DOWNTIME_REQUEUE_REASON:
                    self._db.execute(
                        "INSERT INTO reservation_sla_resets(reservation_id,"
                        "reset_block,authority_digest,reason) VALUES(?,?,?,?)",
                        (
                            reservation_id,
                            current_block,
                            authority_digest,
                            reset_reason,
                        ),
                    )
                else:
                    self._db.execute(
                        "UPDATE reservation_sla_resets SET reset_block=?,"
                        "authority_digest=?,reason=? WHERE reservation_id=?",
                        (
                            current_block,
                            authority_digest,
                            reset_reason,
                            reservation_id,
                        ),
                    )
            for reservation_id, status, reset_reason, clear_screen in restored:
                if clear_screen:
                    self._db.execute(
                        "UPDATE reservations SET status=?,screen_status='',"
                        "screen_stage_count=0,decision='',reason=?,"
                        "retry_group_digest='',retry_position=0,"
                        "qualification_authority_digest='',"
                        "qualification_authority_json='',"
                        "qualification_evidence_digest='' WHERE reservation_id=?",
                        (status, reset_reason, reservation_id),
                    )
                else:
                    self._db.execute(
                        "UPDATE reservations SET status=?,decision='',reason=?,"
                        "retry_group_digest='',retry_position=0,"
                        "qualification_authority_digest='',"
                        "qualification_authority_json='',"
                        "qualification_evidence_digest='' WHERE reservation_id=?",
                        (status, reset_reason, reservation_id),
                    )
        return tuple(self.get(reservation_id) for reservation_id in reservation_ids)

    def _transition(
        self,
        reservation_id: str,
        expected: set[str],
        status: str,
        decision: str,
        reason: str,
        *,
        evidence_digest: str = "",
    ) -> IntakeReservation:
        if status not in _STATUSES or not isinstance(reason, str) or len(reason) > 2_048:
            raise IntakeError("intake transition is malformed")
        with self._transaction():
            self._require_evaluation_mutation_authority(reservation_id)
            row = self.get(reservation_id)
            if row.status not in expected:
                raise IntakeError(f"intake transition from {row.status!r} is forbidden")
            self._db.execute(
                "UPDATE reservations SET status=?,decision=?,reason=?,"
                "qualification_evidence_digest=? WHERE reservation_id=?",
                (status, decision, reason, evidence_digest, reservation_id),
            )
        return self.get(reservation_id)

    def _expire_stale_rows(self, current_block: int) -> tuple[str, ...]:
        """Expire SLA-old unresolved work inside the caller's transaction.

        The arrival-block SLA bounds admission, transport, and primary qualification
        work.  A first retained PASS resets the same SLA from its durable finalized
        progress block, so independent reproduction gets a full bounded window
        without regaining a permanent priority veto.  Legacy retained evidence with
        an unknown (zero) progress block remains fail-closed for explicit operator
        disposition.  Schema-v3 migration holds require their dedicated migration
        path instead.
        """

        threshold = current_block - self.policy.expiry_blocks
        if threshold < 0:
            return ()
        malformed = self._db.execute(
            "SELECT r.reservation_id FROM reservations AS r WHERE "
            "r.status='reproduction_pending' AND "
            "((SELECT COUNT(*) FROM settlement_qualifications AS q "
            "WHERE q.reservation_id=r.reservation_id)!=1 OR "
            "(SELECT COUNT(*) FROM settlement_qualifications AS q "
            "WHERE q.reservation_id=r.reservation_id "
            "AND q.reproduction_index=0)!=1) LIMIT 1"
        ).fetchone()
        if malformed is not None:
            raise IntakeError("reproduction-pending authority is inconsistent")
        malformed_block = self._db.execute(
            "SELECT reservation_id FROM settlement_qualifications WHERE "
            "retained_block<0 OR retained_block>? LIMIT 1",
            (current_block,),
        ).fetchone()
        if malformed_block is not None:
            raise IntakeError("retained qualification block is not finalized")
        placeholders = ",".join("?" for _ in _AUTOMATICALLY_EXPIRABLE)
        predicate = (
            f"r.status IN ({placeholders}) AND r.reason!=? AND NOT EXISTS ("
            "SELECT 1 FROM evaluation_lease_members AS em WHERE "
            "em.reservation_id=r.reservation_id AND em.active=1) AND ("
            "(COALESCE((SELECT s.reset_block FROM reservation_sla_resets AS s "
            "WHERE s.reservation_id=r.reservation_id),r.block)<=? AND NOT EXISTS ("
            "SELECT 1 FROM settlement_qualifications AS q "
            "WHERE q.reservation_id=r.reservation_id)) OR EXISTS ("
            "SELECT 1 FROM settlement_qualifications AS q "
            "WHERE q.reservation_id=r.reservation_id "
            "AND q.reproduction_index=0 AND q.retained_block>0 "
            "AND q.retained_block<=?))"
        )
        rows = tuple(
            row["reservation_id"]
            for row in self._db.execute(
                f"SELECT r.reservation_id FROM reservations AS r WHERE {predicate} "
                "ORDER BY r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash",
                (
                    *_AUTOMATICALLY_EXPIRABLE,
                    _SCHEMA3_MIGRATION_HOLD_REASON,
                    threshold,
                    threshold,
                ),
            )
        )
        if rows:
            self._db.execute(
                f"UPDATE reservations AS r SET status='expired',decision='NO_DECISION',"
                f"reason=? WHERE {predicate}",
                (
                    _AUTOMATIC_EXPIRY_REASON,
                    *_AUTOMATICALLY_EXPIRABLE,
                    _SCHEMA3_MIGRATION_HOLD_REASON,
                    threshold,
                    threshold,
                ),
            )
        return rows

    def expire_stale(self, *, current_block: int) -> tuple[IntakeReservation, ...]:
        """Apply finalized-block arrival/progress SLAs to eligible unresolved work."""

        if type(current_block) is not int or current_block < 0:
            raise IntakeError("automatic expiry block is malformed")
        with self._transaction():
            expired = self._expire_stale_rows(current_block)
        return tuple(self.get(reservation_id) for reservation_id in expired)

    def archive_schema3_migration_hold(
        self,
        reservation_id: str,
        *,
        current_finalized_block: int,
        reason: str,
    ) -> IntakeReservation:
        """Terminally archive one exact schema-v3 migration hold.

        This is deliberately narrower than generic expiry/release.  It preserves
        the retained candidate and qualification rows, cannot make them pending or
        crownable, and only removes the reservation's permanent queue/priority veto
        after an operator supplies a bounded audit reason at a finalized height.
        """

        if (
            type(current_finalized_block) is not int
            or current_finalized_block < 0
            or not isinstance(reason, str)
            or not reason
            or reason.strip() != reason
            or any(ord(char) < 32 or ord(char) == 127 for char in reason)
        ):
            raise IntakeError("schema3 archival authority is malformed")
        archive_reason = (
            f"{_SCHEMA3_ARCHIVE_REASON_PREFIX}{current_finalized_block}:{reason}"
        )
        if len(archive_reason) > 2_048:
            raise IntakeError("schema3 archival reason is oversized")

        with self._transaction():
            row = self.get(reservation_id)
            if (
                row.status != "held"
                or row.reason != _SCHEMA3_MIGRATION_HOLD_REASON
                or current_finalized_block < row.arrival.block
            ):
                raise IntakeError(
                    "only an exact schema3 reproduction migration hold may be archived"
                )
            candidate_row = self._db.execute(
                "SELECT * FROM settlement_candidates WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if candidate_row is None:
                raise IntakeError("schema3 migration hold lacks retained settlement authority")
            # Legacy candidate bytes may predate the current two-PASS parser.
            # Preserve them verbatim rather than pretending to regrade them; this
            # transition only removes priority and can never make them crownable.
            if (
                not candidate_row["candidate_json"]
                or require_sha256_hex(
                    candidate_row["candidate_digest"], field="candidate_digest"
                )
                != candidate_row["candidate_digest"]
                or candidate_row["status"] != "held"
                or candidate_row["reason"] != _SCHEMA3_MIGRATION_HOLD_REASON
                or candidate_row["lease_id"]
                or candidate_row["lease_expires_block"] != 0
                or candidate_row["settlement_evidence_digest"]
                or self._db.execute(
                    "SELECT 1 FROM settlement_events WHERE reservation_id=? LIMIT 1",
                    (reservation_id,),
                ).fetchone()
                is not None
            ):
                raise IntakeError(
                    "schema3 migration hold has settlement authority that cannot be archived"
                )
            reservation_update = self._db.execute(
                "UPDATE reservations SET status='expired',decision='NO_DECISION',"
                "reason=? WHERE reservation_id=? AND status='held' AND reason=?",
                (
                    archive_reason,
                    reservation_id,
                    _SCHEMA3_MIGRATION_HOLD_REASON,
                ),
            )
            candidate_update = self._db.execute(
                "UPDATE settlement_candidates SET reason=? WHERE reservation_id=? "
                "AND status='held' AND reason=? AND lease_id='' "
                "AND lease_expires_block=0 AND settlement_evidence_digest=''",
                (
                    archive_reason,
                    reservation_id,
                    _SCHEMA3_MIGRATION_HOLD_REASON,
                ),
            )
            if reservation_update.rowcount != 1 or candidate_update.rowcount != 1:
                raise IntakeError("schema3 migration hold changed during archival")
        return self.get(reservation_id)

    def auto_requeueable_holds(self, *, limit: int = 64) -> tuple[str, ...]:
        """Held rows parked by a bounded evaluation hold, oldest arrival first.

        An evaluation hold records that the stage produced no PASS/FAIL
        evidence -- worker infrastructure loss, a non-verdict batch, or a
        systemic release cap.  Both producers document ``release_hold`` as the
        reopen path (``_cap_systemic_releases`` and
        ``commit_remote_qualification_hold``), and on an autonomous validator
        no operator is there to call it, so every such park is a permanent
        queue leak.  Transport-retry limits, screen holds, and the schema3
        migration hold are authority fences, not evaluation holds; they are
        never listed here and ``release_hold`` refuses the last one outright.
        """

        if type(limit) is not int or isinstance(limit, bool) or limit < 1:
            raise IntakeError("hold reconciliation limit is invalid")
        rows = self._db.execute(
            "SELECT reservation_id FROM reservations WHERE status='held' AND ("
            "reason LIKE 'remote_qualification_hold:%' OR "
            "reason LIKE 'systemic_release_cap:%' OR "
            "reason LIKE 'auto_requeue_attempt_%') "
            "AND reason NOT LIKE ? "
            "ORDER BY block,event_index,event_subindex,hotkey,content_hash "
            "LIMIT ?",
            (f"%{_EXHAUSTED_HOLD_SUFFIX}", limit),
        ).fetchall()
        return tuple(row["reservation_id"] for row in rows)

    def mark_hold_retry_exhausted(self, reservation_id: str) -> IntakeReservation:
        """Stamp a held row that burned its whole automatic retry budget.

        Attempt counts are in-memory and a supervisor restart forgets them, so
        without a durable mark ``auto_requeueable_holds`` hands the same stuck
        row a fresh budget on every respawn -- and with a watchdog respawning
        the supervisor that is an unbounded loop, not a bounded retry.  A
        stamped row stays held for an operator, which is the correct end state
        once the bounded budget is genuinely spent.
        """

        with self._transaction():
            row = self.get(reservation_id)
            if row.status != "held":
                raise IntakeError("only a held reservation may be marked retry-exhausted")
            if row.reason.endswith(_EXHAUSTED_HOLD_SUFFIX):
                return row
            self._db.execute(
                "UPDATE reservations SET reason=? "
                "WHERE reservation_id=? AND status='held'",
                (f"{row.reason}{_EXHAUSTED_HOLD_SUFFIX}", reservation_id),
            )
        return self.get(reservation_id)

    def release_hold(self, reservation_id: str, *, reason: str) -> IntakeReservation:
        if not reason:
            raise IntakeError("hold release requires an operator reason")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status not in {"held", "no_decision"}:
                raise IntakeError("only held intake may be released")
            if row.reason == _SCHEMA3_MIGRATION_HOLD_REASON:
                raise IntakeError(
                    "legacy single-PASS settlement requires explicit archival migration"
                )
            reproductions = self._db.execute(
                "SELECT COUNT(*) AS n FROM settlement_qualifications "
                "WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()["n"]
            status = (
                "reproduction_pending" if reproductions == 1
                else "published" if row.publication_digest
                else "transport_retry"
            )
            attempts = row.transport_attempts if row.publication_digest else 0
            self._db.execute(
                "UPDATE reservations SET status=?,decision='',reason=?,"
                "transport_attempts=?,retry_group_digest='',retry_position=0,"
                "qualification_authority_digest='',qualification_authority_json='',"
                "qualification_evidence_digest='' "
                "WHERE reservation_id=?",
                (status, reason, attempts, reservation_id),
            )
        return self.get(reservation_id)


class SQLiteWeightPublicationJournal:
    """CAS journal adapter over the same exclusive control-plane SQLite authority."""

    def __init__(self, store: FinalizedIntakeStore, projection: WeightProjection) -> None:
        from cacheon.chain.weights import WeightProjection

        if type(store) is not FinalizedIntakeStore or type(projection) is not WeightProjection:
            raise IntakeError("weight publication journal authority is not exactly typed")
        self._require_legacy_v1_allowed(store)
        self.store = store
        self.projection = projection

    @staticmethod
    def _require_legacy_v1_allowed(store: FinalizedIntakeStore) -> None:
        if type(store) is not FinalizedIntakeStore:
            raise IntakeError(
                "weight publication journal store is not exactly typed"
            )

    @staticmethod
    def _read_head(
        store: FinalizedIntakeStore,
    ) -> tuple[str, str] | None:
        SQLiteWeightPublicationJournal._require_legacy_v1_allowed(store)
        row = store._db.execute(
            "SELECT value FROM metadata WHERE key='weight_publication_head'"
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"weight publication head is corrupt: {exc}") from None
        if type(value) is not dict or set(value) != {
            "projection_digest", "record_digest"
        }:
            raise IntakeError("weight publication head is malformed")
        require_sha256_hex(value["projection_digest"], field="projection_digest")
        require_sha256_hex(value["record_digest"], field="record_digest")
        return value["projection_digest"], value["record_digest"]

    def _head(self) -> tuple[str, str] | None:
        return self._read_head(self.store)

    @classmethod
    def reopen_from_head(
        cls,
        store: FinalizedIntakeStore,
    ) -> "SQLiteWeightPublicationJournal":
        """Reopen the exact retained projection bound to the verified journal head."""

        from cacheon.chain.weights import WeightProjection, WeightPublicationError

        if type(store) is not FinalizedIntakeStore:
            raise IntakeError("weight publication journal store is not exactly typed")
        head = cls._read_head(store)
        if head is None:
            raise IntakeError("weight publication journal has no retained head")
        row = store._db.execute(
            "SELECT projection_digest,projection_json FROM weight_publications "
            "WHERE record_digest=?",
            (head[1],),
        ).fetchone()
        if row is None or row["projection_digest"] != head[0]:
            raise IntakeError("weight publication head has no retained projection")
        try:
            projection = WeightProjection.from_dict(
                json.loads(row["projection_json"])
            )
        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
            WeightPublicationError,
        ) as exc:
            raise IntakeError(f"weight projection is corrupt: {exc}") from None
        if projection.digest != head[0]:
            raise IntakeError("retained weight projection differs from journal head")
        journal = cls(store, projection)
        try:
            record = journal.load()
        except WeightPublicationError as exc:
            raise IntakeError(f"weight publication record is corrupt: {exc}") from None
        if record is None or record.digest != head[1]:
            raise IntakeError("weight publication head cannot be reopened")
        return journal

    def load(self) -> WeightPublicationRecord | None:
        from cacheon.chain.weights import WeightPublicationRecord

        head = self._head()
        if head is None:
            return None
        row = self.store._db.execute(
            "SELECT record_json FROM weight_publications WHERE record_digest=?",
            (head[1],),
        ).fetchone()
        if row is None:
            raise IntakeError("weight publication head has no retained record")
        try:
            record = WeightPublicationRecord.from_dict(json.loads(row["record_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"weight publication record is corrupt: {exc}") from None
        if record.digest != head[1] or record.projection_digest != head[0]:
            raise IntakeError("weight publication head differs from retained record")
        seen: set[str] = set()
        current = record
        while True:
            if current.digest in seen:
                raise IntakeError("weight publication journal contains a cycle")
            seen.add(current.digest)
            prior = current.prior_record_digest
            if prior is None:
                break
            predecessor = self.store._db.execute(
                "SELECT record_json FROM weight_publications WHERE record_digest=?",
                (prior,),
            ).fetchone()
            if predecessor is None:
                raise IntakeError("weight publication predecessor is missing")
            try:
                current = WeightPublicationRecord.from_dict(
                    json.loads(predecessor["record_json"])
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntakeError(
                    f"weight publication predecessor is corrupt: {exc}"
                ) from None
            if current.digest != prior:
                raise IntakeError("weight publication predecessor digest differs")
        return record

    def compare_and_swap(
        self,
        expected_record_digest: str | None,
        replacement: WeightPublicationRecord,
    ) -> None:
        from cacheon.chain.weights import WeightPublicationRecord

        if type(replacement) is not WeightPublicationRecord:
            raise IntakeError("weight publication replacement is not exactly typed")
        if expected_record_digest is not None:
            require_sha256_hex(expected_record_digest, field="expected_record_digest")
        if replacement.prior_record_digest != expected_record_digest:
            raise IntakeError("weight publication replacement does not bind the CAS head")
        with self.store._transaction():
            head = self._head()
            observed = None if head is None else head[1]
            if observed != expected_record_digest:
                raise IntakeError("weight publication journal compare-and-swap failed")
            previous = self.store._db.execute(
                "SELECT sequence,projection_json FROM weight_publications "
                "WHERE projection_digest=? ORDER BY sequence DESC LIMIT 1",
                (replacement.projection_digest,),
            ).fetchone()
            if replacement.projection_digest == self.projection.digest:
                projection_json = json.dumps(
                    self.projection.to_dict(), separators=(",", ":"), sort_keys=True
                )
            elif previous is not None:
                projection_json = previous["projection_json"]
            else:
                raise IntakeError("publication record has no retained projection")
            sequence = self.store._db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 AS value FROM weight_publications"
            ).fetchone()["value"]
            record_json = json.dumps(
                replacement.to_dict(), separators=(",", ":"), sort_keys=True
            )
            updated_block = max(
                replacement.submit_block,
                replacement.confirmed_block,
                replacement.confirmed_last_update,
            )
            self.store._db.execute(
                "INSERT INTO weight_publications(record_digest,sequence,projection_digest,"
                "projection_json,record_json,status,updated_block) VALUES(?,?,?,?,?,?,?)",
                (
                    replacement.digest,
                    sequence,
                    replacement.projection_digest,
                    projection_json,
                    record_json,
                    replacement.status,
                    updated_block,
                ),
            )
            encoded = json.dumps(
                {
                    "projection_digest": replacement.projection_digest,
                    "record_digest": replacement.digest,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self.store._db.execute(
                "INSERT INTO metadata(key,value) VALUES('weight_publication_head',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (encoded,),
            )

    def retained_projection(self, projection_digest: str) -> WeightProjection:
        from cacheon.chain.weights import WeightProjection

        self._require_legacy_v1_allowed(self.store)
        require_sha256_hex(projection_digest, field="projection_digest")
        row = self.store._db.execute(
            "SELECT projection_json FROM weight_publications WHERE projection_digest=? "
            "ORDER BY sequence DESC LIMIT 1",
            (projection_digest,),
        ).fetchone()
        if row is None:
            raise IntakeError("weight projection is not retained")
        try:
            projection = WeightProjection.from_dict(json.loads(row["projection_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"weight projection is corrupt: {exc}") from None
        if projection.digest != projection_digest:
            raise IntakeError("retained weight projection digest differs")
        return projection


class SQLiteFollowerWeightPublicationJournal:
    """Signer-only journal for authenticated shared weight offers.

    This journal is deliberately separate from the legacy V1 publication
    journal. A follower owns chain signing and readback, but it does not need
    a replica of the evaluator's settlement state merely to retain
    publication intent.
    """

    _HEAD_KEY = "followed_weight_publication_head"

    def __init__(self, store: FinalizedIntakeStore, offer) -> None:
        from cacheon.chain.weight_share import CurrentWeightOffer

        if (
            type(store) is not FinalizedIntakeStore
            or type(offer) is not CurrentWeightOffer
        ):
            raise IntakeError("followed weight journal authority is not exactly typed")
        if (
            offer.projection.chain_scope_digest != store.scope.digest
            or offer.projection.netuid != store.scope.netuid
        ):
            raise IntakeError("followed weight offer differs from the journal scope")
        self.store = store
        self.offer = offer
        self.projection = offer.projection

    @staticmethod
    def _offer_from_json(encoded: str):
        from cacheon.chain.weight_share import CurrentWeightOffer, WeightShareError

        try:
            offer = CurrentWeightOffer.from_dict(json.loads(encoded))
        except (TypeError, ValueError, json.JSONDecodeError, WeightShareError) as exc:
            raise IntakeError(f"followed weight offer is corrupt: {exc}") from None
        return offer

    @classmethod
    def _read_head(
        cls,
        store: FinalizedIntakeStore,
    ) -> tuple[str, str] | None:
        row = store._db.execute(
            "SELECT value FROM metadata WHERE key=?",
            (cls._HEAD_KEY,),
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(
                f"followed weight publication head is corrupt: {exc}"
            ) from None
        if type(value) is not dict or set(value) != {
            "projection_digest",
            "record_digest",
        }:
            raise IntakeError("followed weight publication head is malformed")
        require_sha256_hex(value["projection_digest"], field="projection_digest")
        require_sha256_hex(value["record_digest"], field="record_digest")
        return value["projection_digest"], value["record_digest"]

    def _head(self) -> tuple[str, str] | None:
        return self._read_head(self.store)

    def load(self) -> WeightPublicationRecord | None:
        from cacheon.chain.weights import WeightPublicationRecord

        head = self._head()
        if head is None:
            return None
        row = self.store._db.execute(
            "SELECT projection_digest,offer_digest,offer_json,record_json "
            "FROM followed_weight_publications WHERE record_digest=?",
            (head[1],),
        ).fetchone()
        if row is None:
            raise IntakeError(
                "followed weight publication head has no retained record"
            )
        try:
            record = WeightPublicationRecord.from_dict(json.loads(row["record_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(
                f"followed weight publication record is corrupt: {exc}"
            ) from None
        offer = self._offer_from_json(row["offer_json"])
        if (
            record.digest != head[1]
            or record.projection_digest != head[0]
            or row["projection_digest"] != head[0]
            or offer.projection.digest != head[0]
            or offer.digest != row["offer_digest"]
        ):
            raise IntakeError(
                "followed weight publication head differs from retained authority"
            )
        seen: set[str] = set()
        current = record
        while True:
            if current.digest in seen:
                raise IntakeError("followed weight publication journal contains a cycle")
            seen.add(current.digest)
            prior = current.prior_record_digest
            if prior is None:
                break
            predecessor = self.store._db.execute(
                "SELECT record_json FROM followed_weight_publications "
                "WHERE record_digest=?",
                (prior,),
            ).fetchone()
            if predecessor is None:
                raise IntakeError(
                    "followed weight publication predecessor is missing"
                )
            try:
                current = WeightPublicationRecord.from_dict(
                    json.loads(predecessor["record_json"])
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntakeError(
                    f"followed weight publication predecessor is corrupt: {exc}"
                ) from None
            if current.digest != prior:
                raise IntakeError(
                    "followed weight publication predecessor digest differs"
                )
        return record

    @staticmethod
    def _require_monotonic_offer(current, proposed) -> None:
        from cacheon.chain.weight_share import LANE_LEGACY_V1

        if (
            proposed.projection.chain_scope_digest
            != current.projection.chain_scope_digest
            or proposed.projection.netuid != current.projection.netuid
        ):
            raise IntakeError("followed weight offer changed chain scope")
        if (
            proposed.projection.validator_hotkey
            != current.projection.validator_hotkey
        ):
            raise IntakeError("followed weight offer changed signer authority")
        if proposed.projection.effective_block < current.projection.effective_block:
            raise IntakeError("followed weight offer effective block regressed")
        if (
            proposed.projection.effective_block
            == current.projection.effective_block
            and proposed.digest != current.digest
        ):
            raise IntakeError(
                "followed weight offer conflicts at the retained effective block"
            )
        if current.lane != LANE_LEGACY_V1 and proposed.lane == LANE_LEGACY_V1:
            raise IntakeError("followed weight offer lane regressed to legacy V1")

    def compare_and_swap(
        self,
        expected_record_digest: str | None,
        replacement: WeightPublicationRecord,
    ) -> None:
        from cacheon.chain.weights import WeightPublicationRecord

        if type(replacement) is not WeightPublicationRecord:
            raise IntakeError(
                "followed weight publication replacement is not exactly typed"
            )
        if expected_record_digest is not None:
            require_sha256_hex(expected_record_digest, field="expected_record_digest")
        if replacement.prior_record_digest != expected_record_digest:
            raise IntakeError(
                "followed weight publication replacement does not bind the CAS head"
            )
        with self.store._transaction():
            head = self._head()
            observed = None if head is None else head[1]
            if observed != expected_record_digest:
                raise IntakeError(
                    "followed weight publication journal compare-and-swap failed"
                )
            previous = self.store._db.execute(
                "SELECT offer_digest,offer_json FROM followed_weight_publications "
                "WHERE projection_digest=? ORDER BY sequence DESC LIMIT 1",
                (replacement.projection_digest,),
            ).fetchone()
            if replacement.projection_digest == self.offer.projection.digest:
                offer = self.offer
                if previous is not None:
                    retained = self._offer_from_json(previous["offer_json"])
                    if (
                        retained.digest != previous["offer_digest"]
                        or retained.to_dict() != offer.to_dict()
                    ):
                        raise IntakeError(
                            "one followed projection has conflicting retained offers"
                        )
            elif previous is not None:
                offer = self._offer_from_json(previous["offer_json"])
                if offer.digest != previous["offer_digest"]:
                    raise IntakeError("retained followed weight offer digest differs")
            else:
                raise IntakeError(
                    "followed publication record has no retained offer"
                )
            if head is not None and offer.projection.digest != head[0]:
                current_row = self.store._db.execute(
                    "SELECT offer_digest,offer_json FROM followed_weight_publications "
                    "WHERE record_digest=?",
                    (head[1],),
                ).fetchone()
                if current_row is None:
                    raise IntakeError(
                        "followed weight publication head has no retained offer"
                    )
                current_offer = self._offer_from_json(current_row["offer_json"])
                if current_offer.digest != current_row["offer_digest"]:
                    raise IntakeError(
                        "retained followed weight offer digest differs"
                    )
                self._require_monotonic_offer(current_offer, offer)
            sequence = self.store._db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 AS value "
                "FROM followed_weight_publications"
            ).fetchone()["value"]
            record_json = json.dumps(
                replacement.to_dict(), separators=(",", ":"), sort_keys=True
            )
            offer_json = json.dumps(
                offer.to_dict(), separators=(",", ":"), sort_keys=True
            )
            updated_block = max(
                replacement.submit_block,
                replacement.confirmed_block,
                replacement.confirmed_last_update,
            )
            self.store._db.execute(
                "INSERT INTO followed_weight_publications("
                "record_digest,sequence,projection_digest,offer_digest,offer_json,"
                "record_json,status,updated_block) VALUES(?,?,?,?,?,?,?,?)",
                (
                    replacement.digest,
                    sequence,
                    replacement.projection_digest,
                    offer.digest,
                    offer_json,
                    record_json,
                    replacement.status,
                    updated_block,
                ),
            )
            encoded = json.dumps(
                {
                    "projection_digest": replacement.projection_digest,
                    "record_digest": replacement.digest,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self.store._db.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self._HEAD_KEY, encoded),
            )

    def retained_projection(self, projection_digest: str) -> WeightProjection:
        from cacheon.chain.weights import WeightProjection

        require_sha256_hex(projection_digest, field="projection_digest")
        row = self.store._db.execute(
            "SELECT offer_digest,offer_json FROM followed_weight_publications "
            "WHERE projection_digest=? ORDER BY sequence DESC LIMIT 1",
            (projection_digest,),
        ).fetchone()
        if row is None:
            raise IntakeError("followed weight projection is not retained")
        offer = self._offer_from_json(row["offer_json"])
        if (
            offer.digest != row["offer_digest"]
            or offer.projection.digest != projection_digest
        ):
            raise IntakeError("retained followed weight projection digest differs")
        projection = offer.projection
        if type(projection) is not WeightProjection:
            raise IntakeError("retained followed weight projection is untyped")
        return projection


__all__ = [
    "CrownedSettlement", "EvaluationStackState", "FinalizedArrival",
    "FinalizedIntakeStore", "IntakeError",
    "IntakePolicy", "IntakeReservation", "IntakeScope",
    "SQLiteFollowerWeightPublicationJournal", "SQLiteWeightPublicationJournal",
    "SettlementLease",
]

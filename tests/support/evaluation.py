"""Intake rows, arena service, and CPU coordinator fixtures shared by the evaluation suites.

The coordinator and remote-dispatcher suites each carried this block, and the
parallel-arena and worker-pool suites imported one of those test modules for
it. The copies differed only in arena and owner labels.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import cacheon.chain.evaluation_coordinator as coordinator_module
from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceManifest,
    Workload,
    WorkloadCell,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.evaluation_coordinator import (
    ClaimedQualificationEvaluation,
    EvaluationCoordinator,
    WorkerReadiness,
)
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.publication import publish_worker_bundle, reopen_worker_bundle
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.stack_identity import canonical_digest, sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest


SCOPE = IntakeScope("0x" + "0" * 64, 14)
POLICY = IntakePolicy(max_cohort=4, expiry_blocks=100)
BLOCK = 10


def h(label: str) -> str:
    return sha256_hex(label.encode())


def block_hash(block: int) -> str:
    return "0x" + f"{block:064x}"


def manifest() -> ArenaServiceManifest:
    return ArenaServiceManifest(
        ArenaRuntimeIdentity(
            arena_id="coordinator-test",
            runtime_digest=h("runtime"),
            base_engine_digest=h("engine"),
            validator_overlay_digest=h("overlay"),
            worker_distribution_digest=h("worker-distribution"),
            model_revision_digest=h("model-revision"),
            model_manifest_digest=h("model-manifest"),
            model_content_digest=h("model-content"),
            target_architecture="sm120",
            topology_class="tp4-test",
            topology_digest=h("topology"),
            gpu_count=4,
            tensor_parallel_size=4,
        ),
        Workload(
            h("corpus"),
            "test-seed-v1",
            (WorkloadCell("s8", 8192, 1024, 64, 8),),
        ),
        ArenaCapacityPolicy(32, 100, 4, 4),
        h("qualification-policy"),
        h("provider"),
    )


def incumbent(
    service: ArenaService,
    *,
    arena_digest: str | None = None,
    marker: str = "remote",
) -> EvaluationStackManifest:
    snapshot = {
        "composition_rules": [],
        "policy_version": "target-catalog.v1",
        "schema_version": 1,
        "targets": [{"marker": marker, "target_id": "target.0"}],
    }
    return EvaluationStackManifest(
        runtime_digest=service.manifest.runtime.runtime_digest,
        base_engine_digest=service.manifest.runtime.base_engine_digest,
        arena_digest=arena_digest or service.identity,
        catalog_snapshot=snapshot,
        catalog_digest=canonical_digest("cacheon.target-catalog", snapshot),
        entries={},
    )


class Provider:
    provider_digest = h("provider")

    def build_qualification(self, _request, state=None):  # pragma: no cover
        raise AssertionError("qualification work is built on the remote worker")


@dataclasses.dataclass
class Cursor:
    point: tuple[int, str]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def __call__(self) -> tuple[int, str]:
        with self._lock:
            return self.point

    def set(self, block: int) -> None:
        with self._lock:
            self.point = (block, block_hash(block))


def db_path(tmp_path: Path) -> Path:
    return tmp_path / "private" / "intake.sqlite3"


def store(tmp_path: Path) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(db_path(tmp_path), POLICY, scope=SCOPE)


def published_rows(tmp_path: Path, count: int, *, arenas=()):
    publications = []
    arrivals = []
    for index in range(count):
        source = tmp_path / f"source-{index}"
        source.mkdir(parents=True)
        leaf = source / "manifest.toml"
        leaf.write_text(f"bundle_id = 'candidate-{index}'\n")
        source.chmod(0o700)
        leaf.chmod(0o600)
        committed = content_hash(source)
        publication = publish_worker_bundle(
            source, tmp_path / "publications", committed
        )
        publications.append(publication)
        arrivals.append(
            FinalizedArrival(
                f"miner-{index}",
                committed,
                f"https://example.invalid/{index}",
                BLOCK,
                block_hash(BLOCK),
                index,
            )
        )
    with store(tmp_path) as intake:
        reserved = intake.reserve_finalized(
            tuple(arrivals),
            finalized_block=BLOCK,
            finalized_block_hash=block_hash(BLOCK),
        )
        result = []
        for index, (row, publication) in enumerate(
            zip(reserved, publications, strict=True)
        ):
            intake.mark_fetching(row.reservation_id)
            result.append(
                intake.mark_published(
                    row.reservation_id,
                    delta_fingerprint=SubmittedDeltaFingerprint(
                        "component",
                        f"target.{index}",
                        h(f"base:{index}"),
                        (f"slot.{index}",),
                        h(f"archive:{index}"),
                        h(f"selected:{index}"),
                        h(f"exact:{index}"),
                        (h(f"source:{index}"),),
                        (h(f"binary:{index}"),),
                    ),
                    publication_digest=publication.digest,
                    publication_root=publication.root,
                    competition_arena=arenas[index] if arenas else "",
                )
            )
        return tuple(result)


def coordinator(
    tmp_path: Path,
    service: ArenaService,
    cursor: Cursor,
    **changes,
) -> EvaluationCoordinator:
    readiness = WorkerReadiness.for_service(
        service, ready_receipt_digest=h("ready-receipt"), ready_epoch=7
    )
    options = dict(
        intake_db=db_path(tmp_path),
        policy=POLICY,
        scope=SCOPE,
        service=service,
        readiness=readiness,
        owner="cpu-coordinator-test",
        advance_finalized_cursor=cursor,
        lease_blocks=20,
        heartbeat_interval_s=10.0,
        heartbeat_join_timeout_s=1.0,
        lock_retry_delay_s=0.001,
    )
    options.update(changes)
    return EvaluationCoordinator(**options)


def claim_qualification(
    coordinator: EvaluationCoordinator,
) -> ClaimedQualificationEvaluation:
    """Durable qualification lease over published rows, materialized into the
    CPU authority DTO the recoverable dispatcher hands to transport."""
    intake, point = coordinator._open_at_durable_cursor()
    try:
        lease = intake.claim_evaluation_lease(
            stage="qualification",
            owner=coordinator.owner,
            current_block=point[0],
            lease_blocks=coordinator.lease_blocks,
            max_members=coordinator.qualification_max_members,
        )
        assert lease is not None
        reservations = tuple(intake.get(row) for row in lease.reservation_ids)
        attempts = tuple(
            intake.qualification_attempts(row.reservation_id) + 1 for row in reservations
        )
    finally:
        intake.close()
    publications = tuple(
        reopen_worker_bundle(
            row.publication_root,
            row.arrival.content_hash,
            expected_receipt_digest=row.publication_digest,
        )
        for row in reservations
    )
    authority = coordinator_module._qualification_reservations(reservations, publications)
    candidates = tuple(
        ArenaCandidateBinding(item, publication, attempt)
        for publication, item, attempt in zip(publications, authority, attempts, strict=True)
    )
    return ClaimedQualificationEvaluation(lease, reservations, publications, candidates)

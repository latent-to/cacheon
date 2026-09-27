from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import pytest

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
    EvaluationCoordinatorError,
    EvaluationResultEnvelope,
    WorkerReadiness,
)
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakeError,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.publication import publish_worker_bundle, reopen_worker_bundle
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.evidence_store import publish_evidence
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
)
from cacheon.stack_identity import canonical_digest, sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest


SCOPE = IntakeScope("0x" + "0" * 64, 14)
POLICY = IntakePolicy(max_cohort=4, expiry_blocks=100)
BLOCK = 10


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _block_hash(block: int) -> str:
    return "0x" + f"{block:064x}"


def _manifest() -> ArenaServiceManifest:
    runtime = ArenaRuntimeIdentity(
        arena_id="coordinator-test",
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("engine"),
        validator_overlay_digest=_h("overlay"),
        worker_distribution_digest=_h("worker-distribution"),
        model_revision_digest=_h("model-revision"),
        model_manifest_digest=_h("model-manifest"),
        model_content_digest=_h("model-content"),
        target_architecture="sm120",
        topology_class="tp4-test",
        topology_digest=_h("topology"),
        gpu_count=4,
        tensor_parallel_size=4,
    )
    workload = Workload(
        _h("corpus"),
        "test-seed-v1",
        (WorkloadCell("s8", 8192, 1024, 64, 8),),
    )
    return ArenaServiceManifest(
        runtime,
        workload,
        ArenaCapacityPolicy(32, 100, 4, 4),
        _h("qualification-policy"),
        _h("provider"),
    )


class _Provider:
    provider_digest = _h("provider")

    def build_qualification(self, request, state=None):
        raise AssertionError("qualification was not expected")


@dataclasses.dataclass
class _CursorAuthority:
    point: tuple[int, str]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def __call__(self) -> tuple[int, str]:
        with self._lock:
            return self.point

    def set(self, block: int) -> None:
        with self._lock:
            self.point = (block, _block_hash(block))


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "private" / "intake.sqlite3"


def _store(tmp_path: Path) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(_db_path(tmp_path), POLICY, scope=SCOPE)


def _published_rows(tmp_path: Path, count: int, *, arenas=()):
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
            source,
            tmp_path / "publications",
            committed,
        )
        publications.append(publication)
        arrivals.append(
            FinalizedArrival(
                f"miner-{index}",
                committed,
                f"https://example.invalid/{index}",
                BLOCK,
                _block_hash(BLOCK),
                index,
            )
        )
    with _store(tmp_path) as store:
        reserved = store.reserve_finalized(
            tuple(arrivals),
            finalized_block=BLOCK,
            finalized_block_hash=_block_hash(BLOCK),
        )
        result = []
        for index, (row, publication) in enumerate(zip(reserved, publications, strict=True)):
            store.mark_fetching(row.reservation_id)
            result.append(
                store.mark_published(
                    row.reservation_id,
                    delta_fingerprint=SubmittedDeltaFingerprint(
                        "component",
                        f"target.{index}",
                        _h(f"base:{index}"),
                        (f"slot.{index}",),
                        _h(f"archive:{index}"),
                        _h(f"selected:{index}"),
                        _h(f"exact:{index}"),
                        (_h(f"source:{index}"),),
                        (_h(f"binary:{index}"),),
                    ),
                    publication_digest=publication.digest,
                    publication_root=publication.root,
                    competition_arena=arenas[index] if arenas else "",
                )
            )
        return tuple(result)


def _coordinator(
    tmp_path: Path,
    service: ArenaService,
    cursor: _CursorAuthority,
    **changes,
) -> EvaluationCoordinator:
    readiness = changes.pop(
        "readiness",
        WorkerReadiness.for_service(
            service,
            ready_receipt_digest=_h("ready-receipt"),
            ready_epoch=7,
        ),
    )
    options = dict(
        intake_db=_db_path(tmp_path),
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


def _advance(tmp_path: Path, cursor: _CursorAuthority, block: int) -> None:
    with _store(tmp_path) as store:
        store.reserve_finalized(
            (),
            finalized_block=block,
            finalized_block_hash=_block_hash(block),
        )
    cursor.set(block)


def _claim_qualification(
    coordinator: EvaluationCoordinator,
) -> ClaimedQualificationEvaluation:
    """Test-local qualification claim over published rows (the recoverable
    dispatcher materializes the same DTO from its durable recovery lease)."""
    store, point = coordinator._open_at_durable_cursor()
    try:
        lease = store.claim_evaluation_lease(
            stage="qualification",
            owner=coordinator.owner,
            current_block=point[0],
            lease_blocks=coordinator.lease_blocks,
            max_members=coordinator.qualification_max_members,
        )
        assert lease is not None
        reservations = tuple(store.get(row) for row in lease.reservation_ids)
        attempts = tuple(
            store.qualification_attempts(row.reservation_id) + 1 for row in reservations
        )
    finally:
        store.close()
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


def _remote_incumbent(
    service: ArenaService,
    *,
    arena_digest: str | None = None,
    marker: str = "remote-commit",
) -> EvaluationStackManifest:
    snapshot = {
        "schema_version": 1,
        "policy_version": "target-catalog.v1",
        "targets": [{"target_id": "target.0", "marker": marker}],
        "composition_rules": [],
    }
    return EvaluationStackManifest(
        runtime_digest=service.manifest.runtime.runtime_digest,
        base_engine_digest=service.manifest.runtime.base_engine_digest,
        arena_digest=arena_digest or service.identity,
        catalog_snapshot=snapshot,
        catalog_digest=canonical_digest("cacheon.target-catalog", snapshot),
        entries={},
    )


def _remote_commit_product(
    tmp_path: Path,
    coordinator: EvaluationCoordinator,
    claim: ClaimedQualificationEvaluation,
):
    reservations = tuple(row.reservation for row in claim.candidates)
    authority = QualificationAuthorityManifest(
        "registered",
        _h("remote-commit-authority"),
        _h("remote-commit-source"),
        _h("remote-commit-commitment"),
        _h("remote-commit-secret"),
        tuple(row.selected_delta_digest for row in reservations),
        reservations,
    )
    evidence_root = tmp_path / "remote-cpu-evidence"
    attempt_ref = publish_evidence(
        evidence_root,
        b'{"remote":"attempt"}',
        domain="qualification-attempt",
        media_type="application/json",
        schema="cacheon.qualification.remote-commit-test.v1",
    )
    batch = QualificationIntakeBatch(
        authority.digest,
        tuple(
            QualificationIntakeOutcome(
                row.reservation_digest,
                row.selected_delta_digest,
                authority.digest,
                QualificationDecision.FAIL,
                "speed_regression",
                False,
                attempt_artifact_sha256=attempt_ref.sha256,
                report_digest=_h(f"remote-report:{row.reservation_digest}"),
            )
            for row in reservations
        ),
        attempt_ref,
    )
    envelope = EvaluationResultEnvelope.seal(
        claim.lease,
        coordinator.readiness,
        coordinator.service,
        batch,
    )
    return authority, batch, envelope, evidence_root, attempt_ref


def _commit(coordinator, claim, authority, batch, envelope, root, attempt_ref):
    return coordinator.commit_remote_qualification_result(
        claim,
        authority_manifest=authority,
        incumbent_stack=_remote_incumbent(coordinator.service),
        incumbent_tree_digest=_h("remote-tree"),
        batch=batch,
        envelope=envelope,
        evidence_root=root,
        evidence_inventory=(attempt_ref,),
    )


def test_readiness_mismatch_cannot_commit_or_mutate_the_lease(tmp_path: Path) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(tmp_path, service, cursor)
    claim = _claim_qualification(coordinator)
    product = _remote_commit_product(tmp_path, coordinator, claim)
    drifted = _coordinator(
        tmp_path,
        service,
        cursor,
        readiness=dataclasses.replace(
            coordinator.readiness, runtime_digest=_h("wrong-runtime")
        ),
    )

    with pytest.raises(EvaluationCoordinatorError, match="READY identity"):
        _commit(drifted, claim, *product)

    with _store(tmp_path) as store:
        assert store.active_evaluation_leases() == (claim.lease,)
        assert store.get(row.reservation_id).status == "published"
        assert store.qualification_dispositions(row.reservation_id) == ()


def test_ownership_opens_when_intake_advances_the_cursor_during_every_open(
    tmp_path: Path,
) -> None:
    """The store open blocks on the intake controller's flock. An intake pass
    that advances the durable cursor while the open waits makes every pre-open
    reading stale; ownership must still open by re-verifying the authority
    against the store's own durable cursor inside ownership (271 consecutive
    live failures, mainnet 2026-08-11)."""

    _published_rows(tmp_path, 1)
    service = ArenaService(_manifest(), _Provider())
    calls = 0

    def always_behind_the_open() -> tuple[int, str]:
        # Odd (pre-open) reads see the block the intake is about to leave;
        # even (post-open) reads agree with the store's durable cursor.
        nonlocal calls
        calls += 1
        if calls % 2:
            return BLOCK + 1, _block_hash(BLOCK + 1)
        return BLOCK, _block_hash(BLOCK)

    coordinator = _coordinator(
        tmp_path,
        service,
        always_behind_the_open,
        lock_attempts=3,
    )
    store, point = coordinator._open_at_durable_cursor()
    try:
        assert point == (BLOCK, _block_hash(BLOCK))
        assert store.finalized_cursor() == point
    finally:
        store.close()
    assert calls == 2

    # An authority that never agrees with the durable store still fails
    # closed after the full attempt budget.
    coordinator = _coordinator(
        tmp_path,
        service,
        _CursorAuthority((BLOCK + 1, _block_hash(BLOCK + 1))),
        lock_attempts=2,
    )
    with pytest.raises(
        EvaluationCoordinatorError, match="did not stabilize within retry bounds"
    ):
        coordinator._open_at_durable_cursor()


def test_expiry_reclaims_oldest_and_cross_lease_or_stale_envelope_cannot_commit(
    tmp_path: Path,
) -> None:
    first, second = _published_rows(tmp_path, 2)
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=2,
        qualification_max_members=1,
    )
    original = _claim_qualification(coordinator)
    assert original.lease.reservation_ids == (first.reservation_id,)
    authority, batch, old_envelope, root, attempt_ref = _remote_commit_product(
        tmp_path, coordinator, original
    )

    _advance(tmp_path, cursor, BLOCK + 2)
    reclaimed = _claim_qualification(coordinator)
    assert reclaimed.lease.reservation_ids == (first.reservation_id,)
    assert reclaimed.lease.generation == original.lease.generation + 1
    assert reclaimed.lease.lease_id != original.lease.lease_id
    with pytest.raises(EvaluationCoordinatorError, match="exact live lease"):
        _commit(coordinator, reclaimed, authority, batch, old_envelope, root, attempt_ref)

    fresh_envelope = EvaluationResultEnvelope.seal(
        reclaimed.lease,
        coordinator.readiness,
        service,
        batch,
    )
    _advance(tmp_path, cursor, BLOCK + 4)
    replacement = _claim_qualification(coordinator)
    assert replacement.lease.generation == reclaimed.lease.generation + 1
    with pytest.raises(EvaluationCoordinatorError, match="durable lease"):
        _commit(coordinator, reclaimed, authority, batch, fresh_envelope, root, attempt_ref)

    with _store(tmp_path) as store:
        assert store.qualification_dispositions(first.reservation_id) == ()
        assert store.get(second.reservation_id).status == "published"


def test_lock_collision_retries_transiently_and_commit_collision_fails_closed(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    calls = 0

    def transient_factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise IntakeError("another intake controller owns this database")
        return FinalizedIntakeStore(*args, **kwargs)

    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        store_factory=transient_factory,
    )
    claim = _claim_qualification(coordinator)
    assert calls == 3
    product = _remote_commit_product(tmp_path, coordinator, claim)

    def always_busy(*_args, **_kwargs):
        raise IntakeError("another intake controller owns this database")

    coordinator._store_factory = always_busy
    with pytest.raises(EvaluationCoordinatorError, match="did not stabilize"):
        _commit(coordinator, claim, *product)

    with _store(tmp_path) as store:
        assert store.active_evaluation_leases() == (claim.lease,)
        retained = store.get(row.reservation_id)
        assert (retained.status, store.qualification_dispositions(row.reservation_id)) == (
            "published",
            (),
        )


def test_remote_commit_reopens_complete_cpu_inventory_before_any_mutation(
    tmp_path: Path,
) -> None:
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = tuple(row.reservation_id for row in _published_rows(tmp_path, 1))
    coordinator = _coordinator(tmp_path, service, cursor)
    claim = _claim_qualification(coordinator)
    assert type(claim) is ClaimedQualificationEvaluation
    authority, batch, envelope, _root, attempt_ref = _remote_commit_product(
        tmp_path,
        coordinator,
        claim,
    )
    missing_root = tmp_path / "missing-cpu-evidence"
    missing_root.mkdir(mode=0o700)

    with pytest.raises(EvaluationCoordinatorError, match="cannot reopen from the CPU CAS"):
        _commit(coordinator, claim, authority, batch, envelope, missing_root, attempt_ref)

    with _store(tmp_path) as store:
        assert tuple(store.get(value).status for value in ids) == ("published",)
        assert store.qualification_dispositions(ids[0]) == ()
        assert store.active_evaluation_leases() == (claim.lease,)
        with pytest.raises(IntakeError, match="not initialized"):
            store.evaluation_stack(service.identity)


def test_remote_commit_accepts_completed_result_after_tree_advances(
    tmp_path: Path,
) -> None:
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = tuple(row.reservation_id for row in _published_rows(tmp_path, 1))
    coordinator = _coordinator(tmp_path, service, cursor)
    incumbent = _remote_incumbent(service)
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(
            incumbent,
            tree_digest=_h("authoritative-tree"),
        )
    claim = _claim_qualification(coordinator)
    assert type(claim) is ClaimedQualificationEvaluation
    authority, batch, envelope, root, attempt_ref = _remote_commit_product(
        tmp_path,
        coordinator,
        claim,
    )

    wrong_authority = dataclasses.replace(
        authority,
        source_digest=_h("wrong-remote-authority-source"),
    )
    with pytest.raises(EvaluationCoordinatorError, match="qualification result changed"):
        coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=wrong_authority,
            incumbent_stack=incumbent,
            incumbent_tree_digest=_h("authoritative-tree"),
            batch=batch,
            envelope=envelope,
            evidence_root=root,
            evidence_inventory=(attempt_ref,),
        )

    with pytest.raises(EvaluationCoordinatorError, match="differs from the CPU service"):
        coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=authority,
            incumbent_stack=_remote_incumbent(
                service,
                arena_digest=_h("wrong-service"),
                marker="wrong-service",
            ),
            incumbent_tree_digest=_h("authoritative-tree"),
            batch=batch,
            envelope=envelope,
            evidence_root=root,
            evidence_inventory=(attempt_ref,),
        )

    result = coordinator.commit_remote_qualification_result(
        claim,
        authority_manifest=authority,
        incumbent_stack=incumbent,
        incumbent_tree_digest=_h("prior-tree"),
        batch=batch,
        envelope=envelope,
        evidence_root=root,
        evidence_inventory=(attempt_ref,),
    )

    with _store(tmp_path) as store:
        retained = store.get(ids[0])
        stack = store.evaluation_stack(service.identity)
        dispositions = store.qualification_dispositions(ids[0])
        active = store.active_evaluation_leases()
    assert result == (retained,)
    assert retained.status == "failed"
    assert stack.tree_digest == _h("authoritative-tree")
    assert len(dispositions) == 1
    assert dispositions[0]["decision"] == "FAIL"
    assert active == ()


@pytest.mark.parametrize("lane", ["", "reproduction"])
def test_claimed_qualification_rejects_blank_or_mixed_screen_lanes(
    tmp_path: Path,
    lane: str,
) -> None:
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    _published_rows(tmp_path, 2)
    claim = _claim_qualification(_coordinator(tmp_path, service, cursor))
    assert type(claim) is ClaimedQualificationEvaluation
    reservations = (
        claim.reservations[0],
        dataclasses.replace(claim.reservations[1], screen_lane=lane),
    )

    with pytest.raises(EvaluationCoordinatorError, match="inconsistent"):
        ClaimedQualificationEvaluation(
            claim.lease,
            reservations,
            claim.publications,
            claim.candidates,
        )


def test_remote_commit_rejects_late_lease_without_consuming_attempt(
    tmp_path: Path,
) -> None:
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = tuple(row.reservation_id for row in _published_rows(tmp_path, 1))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=2,
    )
    claim = _claim_qualification(coordinator)
    assert type(claim) is ClaimedQualificationEvaluation
    product = _remote_commit_product(tmp_path, coordinator, claim)
    _advance(tmp_path, cursor, claim.lease.expires_block)

    with pytest.raises(EvaluationCoordinatorError, match="durable lease"):
        _commit(coordinator, claim, *product)

    with _store(tmp_path) as store:
        retained = store.get(ids[0])
        dispositions = store.qualification_dispositions(ids[0])
        events = store.evaluation_lease_events(lease_id=claim.lease.lease_id)
        active = store.active_evaluation_leases()
    assert retained.status == "published"
    assert dispositions == ()
    assert active == ()
    assert [row.event_type for row in events] == ["claimed", "expired"]


def test_admission_replays_a_prior_fail_onto_identical_bytes(tmp_path: Path) -> None:
    """A byte-identical resubmission of a FAILed bundle dies before any lease.

    The replay rule (``cacheon.chain.duplicate_replay``) used to run only in an
    operator sidecar sweeping every two minutes, which the one-second supervisor
    tick beat almost every time, so duplicates bought a full qualification
    anyway. Admission (``prepare_qualification_queue``) now applies it on the
    claiming store before any lease exists: the duplicate inherits the prior
    FAIL and costs no qualification. A prior PASS is never replayed.
    """

    source = tmp_path / "source-dup"
    source.mkdir(parents=True)
    leaf = source / "manifest.toml"
    leaf.write_text("bundle_id = 'candidate-dup'\n")
    source.chmod(0o700)
    leaf.chmod(0o600)
    committed = content_hash(source)
    publication = publish_worker_bundle(source, tmp_path / "publications", committed)
    fingerprint = SubmittedDeltaFingerprint(
        "component",
        "target.dup",
        _h("base:dup"),
        ("slot.dup",),
        _h("archive:dup"),
        _h("selected:dup"),
        _h("exact:dup"),
        (_h("source:dup"),),
        (_h("binary:dup"),),
    )
    with _store(tmp_path) as store:
        reserved = store.reserve_finalized(
            tuple(
                FinalizedArrival(
                    f"miner-{index}",
                    committed,
                    f"https://example.invalid/{index}",
                    BLOCK,
                    _block_hash(BLOCK),
                    index,
                )
                for index in range(2)
            ),
            finalized_block=BLOCK,
            finalized_block_hash=_block_hash(BLOCK),
        )
        first, second = (
            (
                store.mark_fetching(row.reservation_id),
                store.mark_published(
                    row.reservation_id,
                    delta_fingerprint=fingerprint,
                    publication_digest=publication.digest,
                    publication_root=publication.root,
                ),
            )[1]
            for row in reserved
        )

    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(tmp_path, service, cursor, qualification_max_members=1)
    claim = _claim_qualification(coordinator)
    assert claim.lease.reservation_ids == (first.reservation_id,)
    _commit(coordinator, claim, *_remote_commit_product(tmp_path, coordinator, claim))
    with _store(tmp_path) as store:
        prior = store.get(first.reservation_id)
        untouched = store.get(second.reservation_id)
    assert (prior.status, prior.decision) == ("failed", "FAIL")
    assert untouched.status == "published"

    # The identical bytes die at admission, before any lease exists.
    with _store(tmp_path) as store:
        retired = store.prepare_qualification_queue(service_digest=service.identity)
        replayed = store.get(second.reservation_id)
        leases = store._db.execute("SELECT COUNT(*) AS n FROM evaluation_leases").fetchone()
    assert retired == ((second.reservation_id, replayed.reason),)
    assert (replayed.status, replayed.decision) == ("failed", "FAIL")
    assert replayed.reason == f"duplicate_of:{prior.reservation_id[:16]}:{prior.reason}"
    assert leases["n"] == 1  # only the prior's own qualification lease

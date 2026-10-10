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
from tests.support.evaluation import (
    BLOCK, Cursor as _CursorAuthority, Provider as _Provider, block_hash as _block_hash,
    claim_qualification as _claim_qualification, coordinator as _coordinator, h as _h,
    incumbent as _incumbent, manifest as _manifest, published_rows as _published_rows,
    store as _store,
)


def _advance(tmp_path: Path, cursor: _CursorAuthority, block: int) -> None:
    with _store(tmp_path) as store:
        store.reserve_finalized(
            (),
            finalized_block=block,
            finalized_block_hash=_block_hash(block),
        )
    cursor.set(block)


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
        incumbent_stack=_incumbent(coordinator.service),
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
    incumbent = _incumbent(service)
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
            incumbent_stack=_incumbent(
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

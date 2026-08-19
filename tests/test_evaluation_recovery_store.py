from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain.evaluation_recovery import (
    EvaluationRecoveryHoldError,
    RecoveryAction,
    RecoveryPhase,
    RecoveryResolution,
    reviewed_legacy_screen_only_reason_digests,
)
from cacheon.chain.evaluation_lease_operator import (
    FifoLeaseConfig,
    release as operator_release,
)
from cacheon.chain.intake import (
    FinalizedArrival,
    IntakeError,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.stack_identity import sha256_hex


def _h(label: str) -> str:
    return sha256_hex(label.encode())


@dataclass(frozen=True)
class _Profile:
    label: str
    netuid: int
    target: str
    topology: str


PROFILES = (
    _Profile("alpha", 14, "collective.alpha_norm", "tp4"),
    _Profile("beta", 29, "attention.beta_projection", "tp8"),
)


def test_reviewed_legacy_release_reason_is_closed_and_digest_bound():
    # The one retained historical row on mainnet carries this exact shape; the
    # writer was retired, the reader must keep reopening the row.
    held = _h("held-reason")
    disposition = _h("reviewed-disposition")
    reason = f"operator_reviewed_legacy_screen_only:v1:{held}:{disposition}"
    assert reviewed_legacy_screen_only_reason_digests(reason) == (
        held,
        disposition,
    )
    assert reviewed_legacy_screen_only_reason_digests(reason + ":extra") is None
    assert reviewed_legacy_screen_only_reason_digests(
        reason.replace(disposition, disposition.upper())
    ) is None


def _store(
    tmp_path, profile: _Profile, **policy
) -> RecoverableFinalizedIntakeStore:
    return RecoverableFinalizedIntakeStore(
        tmp_path / profile.label / "private" / "intake.sqlite3",
        IntakePolicy(**policy),
        scope=IntakeScope("0x" + f"{profile.netuid:064x}", profile.netuid),
    )


def _arrival(profile: _Profile, index: int) -> FinalizedArrival:
    return FinalizedArrival(
        hotkey=f"{profile.label}-miner-{index}",
        content_hash=_h(f"{profile.label}:content:{index}"),
        url=f"https://example.com/{profile.label}/{index}.tar.gz",
        block=10,
        block_hash="0x" + f"{10:064x}",
        event_index=index,
    )


def _advance(store: RecoverableFinalizedIntakeStore, block: int) -> None:
    cursor = store.finalized_cursor()
    assert cursor is not None and block >= cursor[0]
    if block != cursor[0]:
        store.reserve_finalized(
            (),
            finalized_block=block,
            finalized_block_hash="0x" + f"{block:064x}",
        )


def _promoted(store: RecoverableFinalizedIntakeStore, profile: _Profile, index: int = 0):
    row = store.reserve_finalized(
        (_arrival(profile, index),),
        finalized_block=10,
        finalized_block_hash="0x" + f"{10:064x}",
    )[0]
    store.mark_fetching(row.reservation_id)
    published = store.mark_published(
        row.reservation_id,
        delta_fingerprint=SubmittedDeltaFingerprint(
            "component",
            profile.target,
            _h(f"{profile.label}:base"),
            (f"slot.{profile.label}",),
            _h(f"{profile.label}:archive"),
            _h(f"{profile.label}:selected"),
            _h(f"{profile.label}:exact"),
            (_h(f"{profile.label}:source"),),
            (_h(f"{profile.label}:binary:{profile.topology}"),),
        ),
        publication_digest=_h(f"{profile.label}:publication"),
        publication_root=f"/published/{profile.label}",
    )
    service = _h(f"{profile.label}:service")
    active = store.begin_screen(published.reservation_id, service_digest=service)
    candidate = _h(f"{profile.label}:candidate:{active.screen_attempts}")
    receipt = ArenaScreenReceipt(
        service,
        candidate,
        active.screen_attempts,
        tuple(
            ScreenStageResult(
                stage,
                ScreenGrade.PASS,
                _h(f"{profile.label}:screen:{stage}"),
                1,
            )
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )
    return store.apply_screen_receipt(
        published.reservation_id,
        candidate_digest=candidate,
        receipt=receipt,
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_claim_and_recovery_intent_are_atomic_and_target_neutral(
    tmp_path, monkeypatch, profile
):
    with _store(tmp_path, profile) as store:
        row = _promoted(store, profile)
        original = store._create_evaluation_recovery_locked

        def fail_after_lease_insert(lease):
            raise RuntimeError("fault after lease rows")

        with monkeypatch.context() as patch:
            patch.setattr(
                store, "_create_evaluation_recovery_locked", fail_after_lease_insert
            )
            with pytest.raises(RuntimeError, match="fault after lease"):
                store.claim_recoverable_qualification(
                    owner=f"{profile.label}-worker",
                    current_block=10,
                    lease_blocks=4,
                    max_members=1,
                )
        assert store.active_evaluation_leases() == ()
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_lease_members WHERE active=1"
        ).fetchone()["n"] == 0
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM evaluation_recoveries"
        ).fetchone()["n"] == 0
        assert store._create_evaluation_recovery_locked == original

        recovery = store.claim_recoverable_qualification(
            owner=f"{profile.label}-worker",
            current_block=10,
            lease_blocks=4,
            max_members=1,
        )
        assert recovery is not None
        assert recovery.lease.reservation_ids == (row.reservation_id,)
        assert recovery.phase is RecoveryPhase.CLAIMED
        assert recovery.action is RecoveryAction.PRE_RESIDENT_RELEASE
        assert [
            event.event_type.value
            for event in store.evaluation_recovery_events(recovery)
        ] == ["claimed"]
        path = store.path

    with RecoverableFinalizedIntakeStore(
        path,
        IntakePolicy(),
        scope=IntakeScope("0x" + f"{profile.netuid:064x}", profile.netuid),
    ) as reopened:
        assert reopened.pending_qualification_recovery() == recovery


def test_protected_qualification_rejects_every_generic_mutation(tmp_path):
    profile = PROFILES[0]
    with _store(tmp_path, profile) as store:
        _promoted(store, profile)
        recovery = store.claim_recoverable_qualification(
            owner="worker", current_block=10, lease_blocks=2, max_members=1
        )
        assert recovery is not None
        lease = recovery.lease

        with pytest.raises(IntakeError, match="generic heartbeat"):
            store.heartbeat_evaluation_lease(
                lease, current_block=10, lease_blocks=3
            )
        with pytest.raises(IntakeError, match="generic release"):
            store.release_evaluation_lease(
                lease, current_block=10, reason="operator_release"
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            store._db.execute(
                "UPDATE evaluation_leases SET state='released' WHERE lease_id=?",
                (lease.lease_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            store._db.execute(
                "UPDATE evaluation_leases SET expires_block=expires_block+1 "
                "WHERE lease_id=?",
                (lease.lease_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            store._db.execute(
                "UPDATE evaluation_leases SET owner='raw-owner' WHERE lease_id=?",
                (lease.lease_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            store._db.execute(
                "UPDATE evaluation_lease_members SET active=0 WHERE lease_id=?",
                (lease.lease_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            store._db.execute(
                "UPDATE evaluation_lease_members SET prior_status='published' "
                "WHERE lease_id=?",
                (lease.lease_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            store._db.execute(
                "DELETE FROM evaluation_lease_members WHERE lease_id=?",
                (lease.lease_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            store._db.execute(
                "DELETE FROM evaluation_leases WHERE lease_id=?",
                (lease.lease_id,),
            )

        _advance(store, 12)
        assert store.expire_evaluation_leases(current_block=12) == ()
        assert store.active_evaluation_leases() == (lease,)
        assert store.claim_recoverable_qualification(
            owner="other-worker", current_block=12, max_members=1
        ) is None


def test_sealed_operator_release_cannot_bypass_recovery_fence(tmp_path):
    profile = PROFILES[0]
    policy = IntakePolicy()
    scope = IntakeScope("0x" + f"{profile.netuid:064x}", profile.netuid)
    with _store(tmp_path, profile) as store:
        _promoted(store, profile)
        recovery = store.claim_recoverable_qualification(
            owner="operator-worker", current_block=10, max_members=1
        )
        assert recovery is not None
        path = store.path

    config = FifoLeaseConfig(
        path,
        policy,
        scope,
        "operator-worker",
        "qualification",
        30,
        1,
        1,
        0,
    )
    with pytest.raises(IntakeError, match="generic release"):
        operator_release(
            config,
            recovery.lease.lease_id,
            reason="operator_release",
        )


@pytest.mark.parametrize("profile", PROFILES)
def test_recovery_renewal_preserves_identity_but_cannot_commit_before_import(
    tmp_path, profile
):
    with _store(tmp_path, profile) as store:
        _promoted(store, profile)
        original = store.claim_recoverable_qualification(
            owner=f"{profile.label}-worker",
            current_block=10,
            lease_blocks=2,
            max_members=1,
        )
        assert original is not None
        _advance(store, 12)
        renewed, lease = store.renew_recovery_lease(
            original, current_block=12, lease_blocks=5
        )
        assert (
            lease.lease_id,
            lease.generation,
            lease.initial_expires_block,
            lease.expires_block,
        ) == (
            original.lease.lease_id,
            original.lease.generation,
            original.lease.initial_expires_block,
            17,
        )
        with pytest.raises(IntakeError, match="stale"):
            with store.accept_evaluation_result(
                original.lease,
                current_block=12,
                result_digest=_h(f"{profile.label}:stale-result"),
            ):
                raise AssertionError("stale recovery entered result transaction")

        with pytest.raises(IntakeError, match="requires imported evidence"):
            with store.accept_evaluation_result(
                lease,
                current_block=12,
                result_digest=_h(f"{profile.label}:result"),
            ):
                raise AssertionError("pre-import completion entered transaction")

        assert store.pending_qualification_recovery() == renewed
        assert store.active_evaluation_leases() == (lease,)
        assert store._db.execute(
            "SELECT active FROM evaluation_lease_members WHERE lease_id=?",
            (lease.lease_id,),
        ).fetchone()["active"] == 1
        assert [
            event.event_type.value
            for event in store.evaluation_recovery_events(renewed)
        ] == ["claimed", "renewed"]
        assert [
            event.event_type
            for event in store.evaluation_lease_events(lease_id=lease.lease_id)
        ] == ["claimed"]


def test_claimed_release_is_atomic_and_generic_qualification_release_is_impossible(
    tmp_path,
):
    profile = PROFILES[0]
    with _store(tmp_path, profile) as store:
        _promoted(store, profile, 0)
        first = store.claim_recoverable_qualification(
            owner="worker", current_block=10, lease_blocks=5, max_members=1
        )
        assert first is not None
        _advance(store, 11)
        store.release_pre_resident_recovery(
            first, current_block=11, reason="transport_proved_unpublished"
        )
        recovery_row = store._db.execute(
            "SELECT resolution,reason FROM evaluation_recoveries WHERE recovery_id=?",
            (first.recovery_id,),
        ).fetchone()
        lease_row = store._db.execute(
            "SELECT state,reason FROM evaluation_leases WHERE lease_id=?",
            (first.lease.lease_id,),
        ).fetchone()
        assert tuple(recovery_row) == (
            RecoveryResolution.PRE_RESIDENT_RELEASED.value,
            "transport_proved_unpublished",
        )
        assert tuple(lease_row) == ("released", "transport_proved_unpublished")
        assert [
            event.event_type.value
            for event in store.evaluation_recovery_events(first)
        ] == ["claimed", "pre_resident_released"]

    other = PROFILES[1]
    with _store(tmp_path, other) as store:
        _promoted(store, other)
        recovery = store.claim_recoverable_qualification(
            owner="worker", current_block=10, lease_blocks=5, max_members=1
        )
        assert recovery is not None
        with pytest.raises(IntakeError, match="generic release"):
            store.release_evaluation_lease(
                recovery.lease, current_block=10, reason="operator_release"
            )
        held = store.hold_recovery(
            recovery, current_block=10, reason="qualification_requires_review"
        )
        assert held.phase is RecoveryPhase.HELD
        assert held.action is RecoveryAction.HOLD


def test_result_completion_before_import_is_rejected_without_mutation(tmp_path):
    profile = PROFILES[0]
    with _store(tmp_path, profile) as store:
        row = _promoted(store, profile)
        recovery = store.claim_recoverable_qualification(
            owner="worker", current_block=10, lease_blocks=5, max_members=1
        )
        assert recovery is not None
        _advance(store, 11)

        with pytest.raises(IntakeError, match="requires imported evidence"):
            with store.accept_evaluation_result(
                recovery.lease,
                current_block=11,
                result_digest=_h("premature-result"),
            ):
                raise AssertionError("pre-import completion entered transaction")

        assert store.get(row.reservation_id).status == "promoted"
        assert store.qualification_dispositions(row.reservation_id) == ()
        assert store.active_evaluation_leases() == (recovery.lease,)
        assert store.pending_qualification_recovery() == recovery
        assert [
            event.event_type
            for event in store.evaluation_lease_events(
                lease_id=recovery.lease.lease_id
            )
        ] == ["claimed"]
        assert [
            event.event_type.value
            for event in store.evaluation_recovery_events(recovery)
        ] == ["claimed"]
        assert store._db.execute(
            "SELECT active FROM evaluation_lease_members WHERE lease_id=?",
            (recovery.lease.lease_id,),
        ).fetchone()["active"] == 1



def test_missing_recovery_for_active_qualification_is_typed_hold(tmp_path):
    profile = PROFILES[0]
    with _store(tmp_path, profile) as store:
        _promoted(store, profile)
        recovery = store.claim_recoverable_qualification(
            owner="worker", current_block=10, lease_blocks=2, max_members=1
        )
        assert recovery is not None
        path = store.path

    db = sqlite3.connect(path)
    try:
        db.execute("DROP TRIGGER evaluation_recovery_events_reject_delete")
        db.execute("DROP TRIGGER evaluation_recoveries_reject_delete")
        db.execute(
            "DELETE FROM evaluation_recovery_events WHERE recovery_id=?",
            (recovery.recovery_id,),
        )
        db.execute(
            "DELETE FROM evaluation_recoveries WHERE recovery_id=?",
            (recovery.recovery_id,),
        )
        db.commit()
    finally:
        db.close()

    with _store(tmp_path, profile) as reopened:
        lease = reopened.active_evaluation_leases()[0]
        with pytest.raises(EvaluationRecoveryHoldError, match="no recovery|HOLD"):
            reopened.pending_qualification_recovery()
        _advance(reopened, 12)
        with pytest.raises(EvaluationRecoveryHoldError, match="HOLD"):
            reopened.expire_evaluation_leases(current_block=12)
        with pytest.raises(EvaluationRecoveryHoldError, match="HOLD"):
            reopened.release_evaluation_lease(
                lease, current_block=12, reason="operator_release"
            )
        with pytest.raises(sqlite3.IntegrityError, match="protected qualification"):
            reopened._db.execute(
                "UPDATE evaluation_leases SET state='expired' WHERE lease_id=?",
                (lease.lease_id,),
            )

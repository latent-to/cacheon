from __future__ import annotations

import os
import json

import pytest

import cacheon.cli as cli
from cacheon import chain
from cacheon.arena_service import (
    SCREEN_STAGES, ArenaScreenReceipt, PromotionDecision, ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain.intake import (
    FinalizedArrival, FinalizedIntakeStore, IntakeError, IntakePolicy,
    IntakeScope, SQLiteWeightPublicationJournal,
)
from cacheon.chain.weights import WeightProjection, WeightPublicationRecord
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationRetryPlan,
)
from cacheon.economics import (
    EconomicsError,
    EmissionsPolicyManifest,
    GlobalRewardProjectionContext,
    MetagraphMember,
    StandingRewardClaim,
)
from cacheon.settlement import (
    SettlementCandidate, SettlementEventType, SettlementQualification,
    plan_settlement,
)
from cacheon.stack_identity import canonical_digest, sha256_hex
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import plan_marginal_arm
from cacheon.target_catalog import TargetCatalog, default_target_catalog


SCOPE = IntakeScope("0x" + "0" * 64, 307)
AUTHORITY = {"schema": "test-authority"}
ATTEMPT = EvidenceArtifactRef(
    "qualification.cohort-attempt",
    "9" * 64,
    1,
    "application/json",
    "cacheon.qualification.cohort-attempt.v1",
)
POLICY = EmissionsPolicyManifest(100, 20, 100_000)


def _arrival(index: int, *, hotkey: str = "miner", block: int = 10) -> FinalizedArrival:
    return FinalizedArrival(
        hotkey=hotkey,
        content_hash=f"{index + 1:064x}",
        url=f"https://example.com/{index}.tar.gz",
        block=block,
        block_hash="0x" + f"{block:064x}",
        event_index=index,
    )


def _store(tmp_path, **policy):
    return FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(**policy),
        scope=SCOPE,
    )


def test_default_expiry_preserves_fifo_backlog_for_roughly_sixty_nine_days():
    assert IntakePolicy().expiry_blocks == 500_000


def _fingerprint(
    target: str,
    member: str,
    marker: str = "a",
    *,
    selected_delta: str = "3" * 64,
):
    return SubmittedDeltaFingerprint(
        "component", target, "1" * 64, (member,), "2" * 64,
        selected_delta, "4" * 64, (marker * 64,), ("5" * 64,),
    )


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _audit_policy(label: str, slots: tuple[str, ...]) -> SlotAuditPolicy:
    return SlotAuditPolicy(_h(f"audit-seed:{label}")[:32], 100_000, 32, slots, 1)


def _promote(store: FinalizedIntakeStore, reservation_id: str) -> None:
    active = store.begin_screen(reservation_id, service_digest=_h("service"))
    candidate_digest = _h(f"candidate:{reservation_id}:{active.screen_attempts}")
    receipt = ArenaScreenReceipt(
        _h("service"),
        candidate_digest,
        active.screen_attempts,
        tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(stage), 1)
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )
    store.apply_screen_receipt(
        reservation_id, candidate_digest=candidate_digest, receipt=receipt
    )


def _bh(block: int) -> str:
    return "0x" + f"{block:064x}"


def _reserve(store, arrivals, *, block=10):
    return store.reserve_finalized(
        arrivals, finalized_block=block, finalized_block_hash=_bh(block)
    )


def _reserve_one(store, *, index=0, hotkey="miner", block=10):
    return _reserve(
        store, (_arrival(index, hotkey=hotkey, block=block),), block=block
    )[0]


def _publish(store, reservation_id, fingerprint, *, digest, root):
    store.mark_fetching(reservation_id)
    store.mark_published(
        reservation_id,
        delta_fingerprint=fingerprint,
        publication_digest=digest,
        publication_root=root,
    )


def _publish_pair(store, rows):
    for row, marker in zip(rows, ("a", "b"), strict=True):
        _publish(
            store,
            row.reservation_id,
            _fingerprint(f"target.{marker}", f"slot.{marker}", marker),
            digest=marker * 64,
            root=f"/published/{marker}",
        )
        _promote(store, row.reservation_id)
        store.mark_qualifying(row.reservation_id, "7" * 64, AUTHORITY)


def _settlement_plan(store, lease):
    plan = plan_settlement(
        lease.candidates,
        current_manifest=lease.stack.manifest,
        current_tree_digest=lease.stack.tree_digest,
        initial_event_sequence=lease.initial_event_sequence,
        previous_event_digest=lease.previous_event_digest,
    )
    evidence = tuple(
        store.reopen_settlement_evidence(row) for row in lease.candidates
    )
    return plan, evidence


def _context(*hotkeys):
    return GlobalRewardProjectionContext(
        SCOPE.digest,
        "validator",
        12,
        _bh(12),
        tuple(MetagraphMember(uid, hotkey) for uid, hotkey in enumerate(hotkeys)),
    )


def _staging_manifest(prefix, catalog):
    return EvaluationStackManifest(
        runtime_digest=_h(f"{prefix}-runtime"),
        base_engine_digest=_h(f"{prefix}-base"),
        arena_digest=_h(f"{prefix}-arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )


def _policy_metadata(store):
    return store._db.execute(
        "SELECT value FROM metadata WHERE key='emissions_policy_digest'"
    ).fetchone()


def _journal_projection():
    return WeightProjection(
        _h("scope"), 307, "validator", _h("policy"), _h("settlement"),
        _h("evaluation"), _h("metagraph"), (_h("arena-state"),),
        1, 10, 1, (_h("evidence"),), (("miner", 1_000_000),),
    )


def test_exact_manifest_compatibility_failure_can_return_to_fifo(tmp_path) -> None:
    reason = (
        "manifest:submission is not a registered component: "
        "unsupported abi_version 'pre-cutover'"
    )
    with _store(tmp_path) as store:
        row = _reserve_one(store)
        store.mark_fetching(row.reservation_id)
        terminal = store.mark_failed(row.reservation_id, reason)
        assert terminal.status == "failed"

        with pytest.raises(IntakeError, match="exact pre-publication"):
            store.release_manifest_compatibility_failure(
                row.reservation_id,
                expected_reason_digest=_h("wrong reason"),
            )

        released = store.release_manifest_compatibility_failure(
            row.reservation_id,
            expected_reason_digest=sha256_hex(reason.encode()),
        )
        assert released.status == "reserved"
        assert released.decision == ""
        assert released.reason == "manifest_compatibility_released"
        assert store.pending() == (released,)

        with pytest.raises(IntakeError, match="exact pre-publication"):
            store.release_manifest_compatibility_failure(
                row.reservation_id,
                expected_reason_digest=sha256_hex(reason.encode()),
            )


def _stack_context(catalog: TargetCatalog) -> EvaluationStackContext:
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    return EvaluationStackContext(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in targets
        },
    )


def _qualified_settlement_candidate(
    store: FinalizedIntakeStore,
    *,
    primary_only: bool = False,
    retained_block: int = 10,
) -> SettlementCandidate | str:
    catalog = default_target_catalog()
    target = "activation.silu_and_mul"
    incumbent = EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )
    replacement = ProposalContributionRef(
        target_id=target,
        target_spec_digest=catalog.target_spec_digest(target),
        artifact_digest=_h("artifact"),
        selected_payload_digest=_h("payload"),
        attribution_digest=_h("attribution"),
    )
    arm = plan_marginal_arm(
        incumbent,
        replacement,
        catalog=catalog,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h("candidate-tree"),
        expected_context=_stack_context(catalog),
    )
    store.initialize_evaluation_stack(
        incumbent, tree_digest=arm.baseline_before.tree_digest
    )
    evidence_root = store.path.parent / "evidence"
    primary_attempt = publish_evidence(
        evidence_root,
        b"retained primary qualification attempt",
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="cacheon.qualification.cohort-attempt.v1",
    )
    reproduction_attempt = publish_evidence(
        evidence_root,
        b"retained reproduction qualification attempt",
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="cacheon.qualification.cohort-attempt.v1",
    )
    row = _reserve_one(store)
    _publish(
        store,
        row.reservation_id,
        _fingerprint(target, target, selected_delta=arm.selected_delta_digest),
        digest="d" * 64,
        root="/published/candidate",
    )
    _promote(store, row.reservation_id)
    def qualification(marker: str, authority: str, attempt, speedup: str):
        audit_policy = _audit_policy(marker, (target,))
        return SettlementQualification(
            lane="registered",
            arena_digest=incumbent.arena_digest,
            reservation_digest=row.reservation_id,
            finalized_block=row.arrival.block,
            event_index=row.arrival.event_index,
            event_subindex=row.arrival.event_subindex,
            hotkey=row.arrival.hotkey,
            target_id=target,
            members=(target,),
            selected_delta_digest=arm.selected_delta_digest,
            qualification_authority_digest=authority,
            qualification_plan_digest=_h("plan-" + marker),
            qualification_attempt_digest=attempt.sha256,
            qualification_report_digest=_h("report-" + marker),
            selection_commitment_digest=_h("commitment-" + marker),
            selection_secret_commitment_digest=_h("secret-" + marker),
            selection_evidence_digest=_h("selection-" + marker),
            arm_digest=arm.digest,
            incumbent_stack_digest=arm.baseline_before.stack_digest,
            incumbent_tree_digest=arm.baseline_before.tree_digest,
            candidate_stack_digest=arm.challenger.stack_digest,
            candidate_tree_digest=arm.challenger.tree_digest,
            speedup=speedup,
            incumbent_manifest=incumbent,
            candidate_manifest=arm.candidate,
            audit_control_digest=audit_policy.control.digest,
            audit_policy=audit_policy,
            audit_evidence_digest=_h("audit-evidence-" + marker),
        )

    authorities = (_h("primary-authority"), _h("reproduction-authority"))
    qualifications = (
        qualification("primary", authorities[0], primary_attempt, "1.05"),
        qualification("reproduction", authorities[1], reproduction_attempt, "1.04"),
    )
    for index, (authority, attempt, settled) in enumerate(
        zip(
            authorities,
            (primary_attempt, reproduction_attempt),
            qualifications,
            strict=True,
        )
    ):
        if index:
            _promote(store, row.reservation_id)
        store.mark_qualifying(row.reservation_id, authority, AUTHORITY)
        outcome = QualificationIntakeOutcome(
            row.reservation_id,
            arm.selected_delta_digest,
            authority,
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=attempt.sha256,
            report_digest=settled.qualification_report_digest,
            settlement_qualification=settled,
        )
        store.apply_qualification_batch(
            QualificationIntakeBatch(authority, (outcome,), attempt),
            current_finalized_block=retained_block,
            evidence_root=evidence_root,
        )
        if index == 0:
            assert store.get(row.reservation_id).status == "reproduction_pending"
            assert store.lease_settlement_cohort(
                current_block=max(11, retained_block)
            ) is None
            if primary_only:
                return row.reservation_id
    return SettlementCandidate.from_reproductions(*qualifications)


def test_finalized_batch_is_reserved_atomically_before_transport(tmp_path):
    rows = (_arrival(0), _arrival(1, hotkey="other"))
    with _store(tmp_path) as store:
        reserved = _reserve(store, rows)
        assert tuple(row.arrival for row in reserved) == rows
        assert store.pending() == reserved
        assert oct(os.stat(store.path).st_mode & 0o777) == "0o600"
        for suffix in ("-wal", "-shm"):
            sidecar = store.path.with_name(store.path.name + suffix)
            if sidecar.exists():
                assert oct(os.stat(sidecar).st_mode & 0o777) == "0o600"

    with _store(tmp_path) as reopened:
        assert tuple(row.arrival for row in reopened.all()) == rows
        assert _reserve(reopened, rows) == ()


def test_malformed_payload_still_reserves_its_finalized_position(tmp_path):
    invalid = FinalizedArrival(
        "miner", "", "", 10, "0x" + f"{10:064x}", 4, 0,
        "9" * 64, "invalid_payload",
    )
    with _store(tmp_path) as store:
        row = _reserve(store, (invalid,))[0]
        assert row.status == "failed" and row.reason == "invalid_payload"
        assert store.pending() == ()


def test_store_rejects_a_symlink_database(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    target = tmp_path / "elsewhere"
    target.write_text("do not overwrite")
    (private / "intake.sqlite3").symlink_to(target)
    with pytest.raises(IntakeError, match="symlink"):
        FinalizedIntakeStore(private / "intake.sqlite3", scope=SCOPE)


def test_store_rejects_an_existing_nonprivate_parent(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o755)
    with pytest.raises(IntakeError, match="mode 0700"):
        FinalizedIntakeStore(parent / "intake.sqlite3", scope=SCOPE)


def test_store_binds_chain_scope_and_excludes_a_second_controller(tmp_path):
    path = tmp_path / "private" / "intake.sqlite3"
    with _store(tmp_path):
        with pytest.raises(IntakeError, match="another intake controller"):
            FinalizedIntakeStore(path, scope=SCOPE)
    with pytest.raises(IntakeError, match="another chain scope"):
        FinalizedIntakeStore(
            path, scope=IntakeScope("0x" + "1" * 64, SCOPE.netuid)
        )


def test_finalized_cursor_rejects_hash_change_or_regression(tmp_path):
    with _store(tmp_path) as store:
        _reserve_one(store)
        with pytest.raises(IntakeError, match="cursor"):
            store.reserve_finalized(
                (), finalized_block=10, finalized_block_hash="0x" + "f" * 64
            )
        with pytest.raises(IntakeError, match="cursor"):
            _reserve(store, (), block=9)


def test_cursor_rejection_rolls_back_automatic_expiry(tmp_path, monkeypatch):
    with _store(tmp_path, expiry_blocks=20) as store:
        row = _reserve_one(store)
        # Force the cursor check after the in-transaction expiry update to reject
        # this proposed head.  No partial liveness transition may survive.
        monkeypatch.setattr(store, "_cursor", lambda: (31, _bh(31)))
        with pytest.raises(IntakeError, match="cursor"):
            _reserve(store, (), block=30)
        assert store.get(row.reservation_id).status == "reserved"


def test_restart_holds_interrupted_work_instead_of_replaying(tmp_path):
    with _store(tmp_path) as store:
        row = _reserve_one(store)
        store.mark_fetching(row.reservation_id)
    with _store(tmp_path) as reopened:
        held = reopened.get(row.reservation_id)
        assert held.status == "held" and held.decision == ""  # a hold is not a verdict
        assert reopened.pending() == ()


def test_restart_applies_finalized_sla_before_admitting_a_new_arrival(tmp_path):
    with _store(tmp_path, max_pending=1, max_cohort=1, expiry_blocks=20) as store:
        stale = _reserve_one(store)
        store.mark_fetching(stale.reservation_id)

    with _store(
        tmp_path, max_pending=1, max_cohort=1, expiry_blocks=20
    ) as reopened:
        assert reopened.get(stale.reservation_id).status == "held"
        delayed, admitted = _reserve(
            reopened,
            (_arrival(2, hotkey="late-miner", block=10), _arrival(1, block=30)),
            block=30,
        )
        expired = reopened.get(stale.reservation_id)
        assert (expired.status, expired.decision, expired.reason) == (
            "expired", "NO_DECISION", "finalized_block_sla_expired",
        )
        assert (delayed.status, delayed.reason) == (
            "expired", "finalized_block_sla_expired",
        )
        assert admitted.status == "reserved"
        assert reopened.pending() == (admitted,)


def test_admission_bounds_and_epoch_cutoff_are_durable(tmp_path):
    policy = dict(
        max_per_hotkey_epoch=1,
        max_pending=2,
        max_cohort=2,
        epoch_blocks=100,
        cutoff_blocks=10,
    )
    with _store(tmp_path, **policy) as store:
        rows = (
            _arrival(0, block=95),
            _arrival(1, block=96),
            _arrival(2, hotkey="other", block=97),
        )
        result = _reserve(store, rows, block=100)
        assert [row.admission_epoch for row in result] == [1, 1, 1]
        assert [row.status for row in result] == ["reserved", "failed", "reserved"]


def test_unknown_older_and_overlapping_target_block_later_settlement(tmp_path):
    with _store(tmp_path) as store:
        first, second, third = _reserve(
            store, (_arrival(0), _arrival(1, hotkey="b"), _arrival(2, hotkey="c"))
        )
        for row, target, members in (
            (second, "target.a", ("slot.a",)),
            (third, "target.b", ("slot.b",)),
        ):
            _publish(
                store, row.reservation_id, _fingerprint(target, members[0]),
                digest="d" * 64, root=f"/published/{target}",
            )
        assert store.settlement_blockers(second.reservation_id) == (first,)
        assert store.settlement_blockers(third.reservation_id) == (first,)

        _publish(
            store, first.reservation_id, _fingerprint("target.a", "slot.a", "b"),
            digest="e" * 64, root="/published/first",
        )
        assert store.settlement_blockers(second.reservation_id) == (store.get(first.reservation_id),)
        assert store.settlement_blockers(third.reservation_id) == ()


def test_copy_decision_uses_only_durable_delta_fingerprints(tmp_path):
    with _store(tmp_path) as store:
        first, second = _reserve(
            store, (_arrival(0, hotkey="author"), _arrival(1, hotkey="copycat"))
        )
        for row in (first, second):
            _publish(
                store, row.reservation_id, _fingerprint("target.a", "slot.a"),
                digest="d" * 64, root=f"/published/{row.reservation_id}",
            )
        assert store.copy_predecessors(second.reservation_id) == (
            store.get(first.reservation_id),
        )
        copied = store.mark_copy(second.reservation_id, first.reservation_id)
        assert copied.status == "failed" and copied.decision == "FAIL"


def test_expiry_and_retry_release_are_explicit(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        row = _reserve_one(store)
        store.mark_fetching(row.reservation_id)
        store.mark_transport_retry(row.reservation_id, "host unavailable")
        with pytest.raises(IntakeError, match="not old enough"):
            store.expire(row.reservation_id, current_block=29, reason="operator expiry")
        expired = store.expire(row.reservation_id, current_block=30, reason="operator expiry")
        assert expired.status == "expired" and expired.decision == "NO_DECISION"


def test_transport_retry_exhaustion_becomes_an_explicit_hold(tmp_path):
    with _store(tmp_path, max_transport_retries=1) as store:
        row = _reserve_one(store)
        store.mark_fetching(row.reservation_id)
        held = store.mark_transport_retry(row.reservation_id, "host unavailable")
        assert held.status == "held"
        assert held.decision == ""  # a hold is not a verdict
        assert held.reason == "transport_retry_limit"
        released = store.release_hold(
            row.reservation_id, reason="operator granted one fresh transport budget"
        )
        assert released.status == "transport_retry"
        assert released.transport_attempts == 0
        assert store.pending() == (released,)


def test_finalized_sla_removes_old_blocker_but_preserves_settled_candidate(tmp_path):
    with _store(tmp_path, max_transport_retries=1, expiry_blocks=20) as store:
        blocker = _reserve_one(store, index=99, block=9)
        store.mark_fetching(blocker.reservation_id)
        blocker = store.mark_transport_retry(
            blocker.reservation_id, "host unavailable"
        )
        assert blocker.status == "held" and blocker.target_members == ()

        candidate = _qualified_settlement_candidate(store)
        assert store.lease_settlement_cohort(current_block=28) is None
        lease = store.lease_settlement_cohort(current_block=29)
        assert lease is not None and lease.candidates == (candidate,)
        expired = store.get(blocker.reservation_id)
        assert (expired.status, expired.reason) == (
            "expired", "finalized_block_sla_expired",
        )
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=30)
        assert store.get(candidate.reservation_digest).status == "qualified"
        assert store.expire_stale(current_block=100) == ()
        assert store.get(candidate.reservation_digest).status == "qualified"
        assert store.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        ).candidate == candidate


def test_finalized_sla_resets_on_retained_primary_and_survives_restart(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        reservation_id = _qualified_settlement_candidate(
            store, primary_only=True, retained_block=29
        )
        assert isinstance(reservation_id, str)

        # Arrival block 10 would expire at 30.  The primary PASS retained at 29
        # resets the same 20-block SLA, giving reproduction through block 48.
        assert store.expire_stale(current_block=30) == ()
        retained = store.get(reservation_id)
        assert retained.status == "reproduction_pending"
        progress = store._db.execute(
            "SELECT retained_block FROM settlement_qualifications "
            "WHERE reservation_id=? AND reproduction_index=0",
            (reservation_id,),
        ).fetchone()
        assert progress["retained_block"] == 29

    with _store(tmp_path, expiry_blocks=20) as reopened:
        assert reopened.expire_stale(current_block=48) == ()
        expired = reopened.expire_stale(current_block=49)
        assert tuple(row.reservation_id for row in expired) == (reservation_id,)
        assert (
            expired[0].status,
            expired[0].decision,
            expired[0].reason,
        ) == (
            "expired", "NO_DECISION", "finalized_block_sla_expired"
        )


def test_legacy_retained_primary_unknown_block_stays_manual(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        reservation_id = _qualified_settlement_candidate(
            store, primary_only=True
        )
        assert isinstance(reservation_id, str)
        # Simulate the exact additive migration input: an existing schema-3
        # qualification table from before retained progress was recorded.
        store._db.execute(
            "ALTER TABLE settlement_qualifications DROP COLUMN retained_block"
        )

    with _store(tmp_path, expiry_blocks=20) as reopened:
        progress = reopened._db.execute(
            "SELECT retained_block FROM settlement_qualifications "
            "WHERE reservation_id=? AND reproduction_index=0",
            (reservation_id,),
        ).fetchone()
        assert progress["retained_block"] == 0
        assert reopened.expire_stale(current_block=100) == ()
        # A retained_block=0 legacy row is unreachable by the automatic SLA;
        # the typed operator transition is the only terminalization path.
        expired = reopened.expire(
            reservation_id,
            current_block=100,
            reason="operator archived legacy retained PASS",
        )
        assert (expired.status, expired.reason) == (
            "expired", "operator archived legacy retained PASS"
        )


def test_schema3_migration_hold_survives_all_generic_expiry_paths(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        # Reopen through the real v2 -> v3 migration path.
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")

    with _store(tmp_path, expiry_blocks=20) as reopened:
        held = reopened.get(candidate.reservation_digest)
        assert (held.status, held.decision, held.reason) == (
            "held",
            "NO_DECISION",
            "schema3_reproduction_required",
        )
        assert reopened.expire_stale(current_block=100) == ()
        assert reopened.get(candidate.reservation_digest) == held
        with pytest.raises(IntakeError, match="archival migration"):
            reopened.expire(
                candidate.reservation_digest,
                current_block=100,
                reason="generic operator expiry",
            )
        with pytest.raises(IntakeError, match="archival migration"):
            reopened.release_hold(
                candidate.reservation_digest,
                reason="generic operator release",
            )


def test_schema3_archival_is_terminal_preserves_evidence_and_releases_priority(
    tmp_path,
):
    with _store(tmp_path, expiry_blocks=20) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")

    with _store(tmp_path, expiry_blocks=20) as reopened:
        legacy = reopened.get(candidate.reservation_digest)
        candidate_before = dict(
            reopened._db.execute(
                "SELECT * FROM settlement_candidates WHERE reservation_id=?",
                (candidate.reservation_digest,),
            ).fetchone()
        )
        qualifications_before = tuple(
            tuple(row)
            for row in reopened._db.execute(
                "SELECT * FROM settlement_qualifications WHERE reservation_id=? "
                "ORDER BY reproduction_index",
                (candidate.reservation_digest,),
            )
        )

        later = _reserve_one(reopened, index=1, block=11)
        _publish(
            reopened,
            later.reservation_id,
            _fingerprint(
                candidate.target_id, candidate.target_id, "b", selected_delta="6" * 64
            ),
            digest="e" * 64,
            root="/published/later",
        )
        assert reopened.settlement_blockers(later.reservation_id) == (legacy,)

        archived = reopened.archive_schema3_migration_hold(
            candidate.reservation_digest,
            current_finalized_block=11,
            reason="operator verified legacy evidence remains audit-only",
        )
        assert (archived.status, archived.decision) == ("expired", "NO_DECISION")
        assert archived.reason.startswith("schema3_archived@11:")
        assert reopened.settlement_blockers(later.reservation_id) == ()

        candidate_after = dict(
            reopened._db.execute(
                "SELECT * FROM settlement_candidates WHERE reservation_id=?",
                (candidate.reservation_digest,),
            ).fetchone()
        )
        assert candidate_after["status"] == "held"
        assert candidate_after["reason"] == archived.reason
        assert candidate_after["candidate_json"] == candidate_before["candidate_json"]
        assert candidate_after["candidate_digest"] == candidate_before["candidate_digest"]
        assert candidate_after["evidence_root"] == candidate_before["evidence_root"]
        assert candidate_after["reproduction_evidence_root"] == candidate_before[
            "reproduction_evidence_root"
        ]
        assert tuple(
            tuple(row)
            for row in reopened._db.execute(
                "SELECT * FROM settlement_qualifications WHERE reservation_id=? "
                "ORDER BY reproduction_index",
                (candidate.reservation_digest,),
            )
        ) == qualifications_before
        assert reopened.has_pending_settlement() is False
        assert reopened.lease_settlement_cohort(current_block=11) is None
        with pytest.raises(IntakeError, match="only held intake"):
            reopened.release_hold(
                candidate.reservation_digest,
                reason="must not restore crown eligibility",
            )
        with pytest.raises(IntakeError, match="active crown"):
            reopened.reopen_active_crown(candidate.arena_digest, candidate.target_id)
        with pytest.raises(IntakeError, match="exact schema3"):
            reopened.archive_schema3_migration_hold(
                candidate.reservation_digest,
                current_finalized_block=12,
                reason="must not archive twice",
            )


def test_schema3_archival_rejects_ordinary_or_inconsistent_holds(tmp_path):
    with _store(tmp_path) as store:
        ordinary = _reserve_one(store)
        ordinary = store.mark_held(ordinary.reservation_id, "ordinary operator hold")
        with pytest.raises(IntakeError, match="exact schema3"):
            store.archive_schema3_migration_hold(
                ordinary.reservation_id,
                current_finalized_block=10,
                reason="must not archive an ordinary hold",
            )
        assert store.get(ordinary.reservation_id) == ordinary

    other_root = tmp_path / "inconsistent"
    with _store(other_root) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")
    with _store(other_root) as reopened:
        reopened._db.execute(
            "UPDATE settlement_candidates SET status='pending' WHERE reservation_id=?",
            (candidate.reservation_digest,),
        )
        held = reopened.get(candidate.reservation_digest)
        with pytest.raises(IntakeError, match="settlement authority"):
            reopened.archive_schema3_migration_hold(
                candidate.reservation_digest,
                current_finalized_block=10,
                reason="must fail closed on inconsistent authority",
            )
        assert reopened.get(candidate.reservation_digest) == held


def test_schema3_archival_cli_uses_finalized_public_scope_without_a_wallet(
    tmp_path, monkeypatch, capsys
):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")

    class Subtensor:
        def get_block_hash(self, block):
            assert block == 0
            return SCOPE.genesis_hash

    monkeypatch.setattr(chain, "connect", lambda network: Subtensor())
    monkeypatch.setattr(
        chain, "read_finalized_head", lambda _subtensor: (12, _bh(12))
    )
    args = cli.build_parser().parse_args(
        [
            "chain-archive-schema3-hold",
            "--network",
            "mock",
            "--netuid",
            str(SCOPE.netuid),
            "--intake-db",
            str(tmp_path / "private" / "intake.sqlite3"),
            "--reservation-id",
            candidate.reservation_digest,
            "--reason",
            "reviewed before testnet restart",
        ]
    )
    assert args.func is cli.cmd_chain_archive_schema3_hold
    result = args.func(args)
    assert result == 0
    assert "retained evidence remains non-crownable" in capsys.readouterr().out
    with _store(tmp_path) as reopened:
        archived = reopened.get(candidate.reservation_digest)
        assert archived.status == "expired"
        assert archived.reason.startswith("schema3_archived@12:")


def test_qualification_batch_persists_dispositions_and_groups_atomically(tmp_path):
    with _store(tmp_path, max_cohort=2) as store:
        rows = _reserve(store, (_arrival(0), _arrival(1, hotkey="other")))
        _publish_pair(store, rows)
        failure = "6" * 64
        outcomes = tuple(
            QualificationIntakeOutcome(
                row.reservation_id,
                "3" * 64,
                "7" * 64,
                QualificationDecision.NO_DECISION,
                "shared_failure",
                True,
                failure_digest=failure,
            )
            for row in rows
        )
        retry = QualificationRetryPlan(
            "7" * 64,
            "bisect",
            tuple((row.reservation_id,) for row in rows),
            failure,
        )
        stored = store.apply_qualification_batch(
            QualificationIntakeBatch("7" * 64, outcomes, retry_plan=retry),
            current_finalized_block=10,
        )
        assert [row.status for row in stored] == ["published", "published"]
        # Re-screen both republished retries; the live promoted() cohort
        # selector must isolate the first retry group rather than merging
        # both groups back into one failing cohort.
        for row in rows:
            _promote(store, row.reservation_id)
        assert tuple(row.reservation_id for row in store.promoted()) == (
            rows[0].reservation_id,
        )
        assert store.qualification_dispositions(rows[0].reservation_id)[0][
            "authority_manifest"
        ] == AUTHORITY


def test_worker_failure_retry_holds_offender_without_stranding_peer(tmp_path):
    with _store(
        tmp_path, max_cohort=2, max_qualification_retries=2
    ) as store:
        offender, peer = _reserve(
            store, (_arrival(0), _arrival(1, hotkey="peer"))
        )
        _publish_pair(store, (offender, peer))

        first_failure = "6" * 64
        first_outcomes = tuple(
            QualificationIntakeOutcome(
                row.reservation_id,
                "3" * 64,
                "7" * 64,
                QualificationDecision.NO_DECISION,
                "candidate_worker",
                True,
                failure_digest=first_failure,
            )
            for row in (offender, peer)
        )
        first_retry = QualificationRetryPlan(
            "7" * 64,
            "bisect",
            ((offender.reservation_id,), (peer.reservation_id,)),
            first_failure,
        )
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "7" * 64, first_outcomes, retry_plan=first_retry
            ),
            current_finalized_block=10,
        )

        # The live screen selector picks the offender's isolated retry first
        # in finalized order.
        assert tuple(row.reservation_id for row in store.screenable(limit=1)) == (
            offender.reservation_id,
        )
        _promote(store, offender.reservation_id)
        store.mark_qualifying(offender.reservation_id, "8" * 64, AUTHORITY)
        singleton_failure = "9" * 64
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "8" * 64,
                (
                    QualificationIntakeOutcome(
                        offender.reservation_id,
                        "3" * 64,
                        "8" * 64,
                        QualificationDecision.NO_DECISION,
                        "candidate_worker",
                        True,
                        failure_digest=singleton_failure,
                    ),
                ),
                retry_plan=QualificationRetryPlan(
                    "8" * 64,
                    "requeue",
                    ((offender.reservation_id,),),
                    singleton_failure,
                ),
            ),
            current_finalized_block=10,
        )

        held = store.get(offender.reservation_id)
        assert held.status == "held"
        assert held.decision == ""  # a hold is not a verdict
        assert len(store.qualification_dispositions(offender.reservation_id)) == 2

        # Once the bounded offender is held, the peer's isolated group remains
        # runnable: the live screen selector now picks it, and it can retain an
        # independently evidenced terminal decision.
        assert tuple(row.reservation_id for row in store.screenable(limit=1)) == (
            peer.reservation_id,
        )
        _promote(store, peer.reservation_id)
        store.mark_qualifying(peer.reservation_id, "a" * 64, AUTHORITY)
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "a" * 64,
                (
                    QualificationIntakeOutcome(
                        peer.reservation_id,
                        "3" * 64,
                        "a" * 64,
                        QualificationDecision.FAIL,
                        "peer_completed",
                        False,
                        attempt_artifact_sha256=ATTEMPT.sha256,
                        report_digest="4" * 64,
                    ),
                ),
                ATTEMPT,
            ),
            current_finalized_block=10,
        )
        completed = store.get(peer.reservation_id)
        assert completed.status == "failed"
        assert completed.decision == "FAIL"
        assert completed.reason == "peer_completed"


def test_late_earlier_fingerprint_retroactively_identifies_a_qualified_copy(tmp_path):
    with _store(tmp_path) as store:
        first, later = _reserve(
            store, (_arrival(0, hotkey="author"), _arrival(1, hotkey="copycat"))
        )
        _publish(
            store, later.reservation_id, _fingerprint("target.a", "slot.a"),
            digest="b" * 64, root="/published/later",
        )
        _promote(store, later.reservation_id)
        store.mark_qualifying(later.reservation_id, "5" * 64, AUTHORITY)
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "5" * 64,
                (
                    QualificationIntakeOutcome(
                        later.reservation_id,
                        "3" * 64,
                        "5" * 64,
                        QualificationDecision.NO_DECISION,
                        "not_decided",
                        True,
                        failure_digest="4" * 64,
                    ),
                ),
                retry_plan=QualificationRetryPlan(
                    "5" * 64, "requeue", ((later.reservation_id,),), "4" * 64
                ),
            ),
            current_finalized_block=10,
        )

        _publish(
            store, first.reservation_id, _fingerprint("target.a", "slot.a"),
            digest="a" * 64, root="/published/first",
        )
        assert store.reconcile_copies() == (
            (later.reservation_id, first.reservation_id),
        )
        copied = store.get(later.reservation_id)
        assert copied.status == "failed" and copied.decision == "FAIL"


def test_pass_projection_settles_atomically_and_recovers_stack_and_claim(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        genesis = store.evaluation_stack(candidate.arena_digest)
        assert genesis.generation == 0
        assert genesis.manifest.digest == candidate.incumbent_stack_digest
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
        plan, evidence = _settlement_plan(store, lease)
        current = store.commit_settlement(
            lease, plan, evidence, current_block=11
        )
        assert current.generation == 1
        assert current.manifest.digest == candidate.candidate_stack_digest
        standing, discovery = store.active_reward_claims()
        assert discovery == ()
        assert len(standing) == 1
        assert standing[0].arena_digest == candidate.arena_digest
        assert standing[0].retained_evidence_digest == evidence[0].digest
        crown = store.reopen_active_crown(candidate.arena_digest, candidate.target_id)
        assert crown.candidate == candidate
        assert crown.evidence == evidence[0]
        assert crown.event.event_type is SettlementEventType.CROWN
        assert store.lease_settlement_cohort(current_block=12) is None

    with _store(tmp_path) as reopened:
        current = reopened.evaluation_stack(candidate.arena_digest)
        assert current.generation == 1
        assert current.manifest.digest == candidate.candidate_stack_digest
        assert reopened.active_reward_claims()[0][0].hotkey == candidate.hotkey
        crown = reopened.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        )
        assert crown.candidate == candidate
        reopened._db.execute(
            "UPDATE settlement_events SET event_json='{}' WHERE event_id=?",
            (crown.event.digest,),
        )
        with pytest.raises(IntakeError, match="event is corrupt"):
            reopened.reopen_active_crown(candidate.arena_digest, candidate.target_id)


def test_retained_pass_is_rewarded_even_when_settlement_holds_it(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        evidence = store.reopen_settlement_evidence(candidate)
        store._db.execute(
            "UPDATE settlement_candidates SET status='held',reason='incumbent_advanced',"
            "settlement_evidence_digest=? WHERE reservation_id=?",
            (evidence.digest, candidate.reservation_digest),
        )
        claims = store.passed_reward_claims()
        assert len(claims) == 1
        assert claims[0].hotkey == candidate.hotkey
        assert claims[0].retained_evidence_digest == evidence.digest

    with _store(tmp_path) as reopened:
        assert reopened.passed_reward_claims() == claims


def test_interrupted_settlement_lease_requeues_retained_evidence_without_gpu(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        first = store.lease_settlement_cohort(current_block=11, lease_blocks=10)
        assert first is not None
    with _store(tmp_path) as reopened:
        second = reopened.lease_settlement_cohort(current_block=12, lease_blocks=10)
        assert second is not None
        assert second.candidates == (candidate,)
        assert second.generation > first.generation
        assert second.lease_id != first.lease_id


def test_weight_projection_reopens_every_active_crown_and_holds_on_loss(tmp_path):
    catalog = default_target_catalog()
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        context = _context("validator", "miner")
        with pytest.raises(IntakeError, match="catalogs"):
            store.build_weight_projection(
                policy=POLICY,
                context=context,
                catalogs={},
                netuid=SCOPE.netuid,
            )
        assert _policy_metadata(store) is None
        legacy = POLICY.to_dict()
        legacy["policy_version"] = "cacheon.emissions.v1.1"
        store._db.execute(
            "INSERT INTO metadata(key,value) VALUES('emissions_policy_digest',?)",
            (canonical_digest("cacheon.economics.policy", legacy),),
        )
        projection = store.build_weight_projection(
            policy=POLICY,
            context=context,
            catalogs={candidate.arena_digest: catalog},
            netuid=SCOPE.netuid,
        )
        assert projection.crown_count == 1
        assert projection.weights_ppm == (("miner", 1_000_000),)
        assert _policy_metadata(store)["value"] == POLICY.digest
        pending = WeightPublicationRecord(
            projection.digest,
            "pending",
            submit_block=projection.effective_block,
            retry_after_block=projection.effective_block + 20,
            reason="sdk_result_unconfirmed",
        )
        journal = SQLiteWeightPublicationJournal(store, projection)
        journal.compare_and_swap(None, pending)
        standing = store.active_reward_claims()[0][0]
        orphan = StandingRewardClaim(
            _h("orphan-arena"),
            standing.target_id,
            standing.target_spec_digest,
            standing.contribution_digest,
            standing.hotkey,
            standing.speedup_ppm,
            standing.crowned_block,
            standing.retained_evidence_digest,
        )
        store._db.execute(
            "INSERT INTO standing_reward_claims(arena_id,target_id,claim_digest,"
            "claim_json,status,event_id) VALUES(?,?,?,?, 'active',?)",
            (
                orphan.arena_digest,
                orphan.target_id,
                orphan.digest,
                json.dumps(orphan.to_dict(), separators=(",", ":"), sort_keys=True),
                _h("orphan-event"),
            ),
        )
        with pytest.raises(IntakeError, match="absent evaluation arena"):
            store.build_weight_projection(
                policy=POLICY,
                context=context,
                catalogs={candidate.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )
        store._db.execute(
            "DELETE FROM standing_reward_claims WHERE arena_id=?",
            (orphan.arena_digest,),
        )
        with pytest.raises(IntakeError, match="emissions policy"):
            store.build_weight_projection(
                policy=EmissionsPolicyManifest(101, 20, 100_000),
                context=context,
                catalogs={candidate.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )

        artifact = (
            store.path.parent
            / "evidence"
            / evidence[0].primary_attempt_ref.domain
            / evidence[0].primary_attempt_ref.sha256[:2]
            / evidence[0].primary_attempt_ref.sha256
        )
        artifact.unlink()
        with pytest.raises(IntakeError, match="cannot reopen"):
            store.build_weight_projection(
                policy=POLICY,
                context=context,
                catalogs={candidate.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )
        retained = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert retained.projection == projection
        assert retained.load() == pending


def test_uncrowned_arena_is_staging_and_cannot_halt_a_crowned_arena(tmp_path):
    catalog = default_target_catalog()
    context = _context("validator", "miner")
    staging = _staging_manifest("staging", catalog)

    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        store.initialize_evaluation_stack(staging, tree_digest=_h("staging-tree"))

        projection = store.build_weight_projection(
            policy=POLICY,
            context=context,
            catalogs={candidate.arena_digest: catalog, staging.arena_digest: catalog},
            netuid=SCOPE.netuid,
        )
        assert projection.weights_ppm == (("miner", 1_000_000),)
        assert projection.crown_count == 1
        assert len(projection.arena_state_digests) == 1
        assert store.evaluation_stack(staging.arena_digest).generation == 0

    with _store(tmp_path) as reopened:
        # A restart must not reactivate a persisted generation-zero arena.  Its
        # catalog is optional because it has no economic authority yet.
        projection = reopened.build_weight_projection(
            policy=POLICY,
            context=context,
            catalogs={candidate.arena_digest: catalog},
            netuid=SCOPE.netuid,
        )
        assert projection.weights_ppm == (("miner", 1_000_000),)
        assert len(projection.arena_state_digests) == 1


def test_all_uncrowned_bootstrap_remains_an_explicit_fail_closed_policy(tmp_path):
    catalog = default_target_catalog()
    staging = _staging_manifest("bootstrap", catalog)
    context = _context("validator")
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(staging, tree_digest=_h("bootstrap-tree"))
        with pytest.raises(EconomicsError, match="typed arena authorities"):
            store.build_weight_projection(
                policy=POLICY,
                context=context,
                catalogs={staging.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )
        assert _policy_metadata(store) is None


def test_burn_weight_projection_covers_only_the_all_uncrowned_bootstrap(tmp_path):
    catalog = default_target_catalog()
    context = _context("owner-burn", "validator")
    staging = _staging_manifest("bootstrap", catalog)
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(staging, tree_digest=_h("bootstrap-tree"))
        with pytest.raises(IntakeError, match="not registered"):
            store.build_burn_weight_projection(
                policy=POLICY,
                context=context,
                netuid=SCOPE.netuid,
                burn_hotkey="stranger",
            )
        assert _policy_metadata(store) is None
        projection = store.build_burn_weight_projection(
            policy=POLICY,
            context=context,
            netuid=SCOPE.netuid,
            burn_hotkey="owner-burn",
        )
        assert projection.weights_ppm == (("owner-burn", 1_000_000),)
        assert projection.crown_count == 0
        assert projection.stack_generation == 0
        assert projection.evidence_digests == ()
        assert projection.settlement_state_digest == store.settlement_state_digest()
        again = store.build_burn_weight_projection(
            policy=POLICY,
            context=context,
            netuid=SCOPE.netuid,
            burn_hotkey="owner-burn",
        )
        assert again.digest == projection.digest
        journal = SQLiteWeightPublicationJournal(store, projection)
        pending = WeightPublicationRecord(
            projection.digest,
            "pending",
            submit_block=projection.effective_block,
            retry_after_block=projection.effective_block + 20,
            reason="sdk_result_unconfirmed",
        )
        journal.compare_and_swap(None, pending)

    with _store(tmp_path) as reopened:
        head = SQLiteWeightPublicationJournal.reopen_from_head(reopened)
        assert head.projection.digest == projection.digest
        assert head.projection.weights == {"owner-burn": 1.0}


def test_burn_weight_projection_refuses_any_real_economic_authority(tmp_path):
    with _store(tmp_path) as store:
        lease_candidate = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        assert lease_candidate.arena_digest in {
            row.arena_digest for row in store.evaluation_stacks()
        }
        context = _context("owner-burn", "validator", "miner")
        with pytest.raises(IntakeError, match="burn weights refused"):
            store.build_burn_weight_projection(
                policy=POLICY,
                context=context,
                netuid=SCOPE.netuid,
                burn_hotkey="owner-burn",
            )
        assert _policy_metadata(store) is None


def test_subnet_owner_burn_projection_binds_settlement_and_policy(tmp_path):
    context = _context("owner-burn", "validator")
    with _store(tmp_path) as store:
        projection = store.build_subnet_owner_burn_weight_projection(
            policy=POLICY,
            context=context,
            netuid=SCOPE.netuid,
            burn_hotkey="owner-burn",
            owner_coldkey="owner-ck",
            owner_hotkey="owner-burn",
            candidate_uids=(0,),
        )
        assert projection.weights_ppm == (("owner-burn", 1_000_000),)
        assert projection.crown_count == 0
        assert projection.stack_generation == 0
        assert projection.evidence_digests == ()
        assert projection.settlement_state_digest == store.settlement_state_digest()
        assert projection.policy_digest == POLICY.digest
        again = store.build_subnet_owner_burn_weight_projection(
            policy=POLICY,
            context=context,
            netuid=SCOPE.netuid,
            burn_hotkey="owner-burn",
            owner_coldkey="owner-ck",
            owner_hotkey="owner-burn",
            candidate_uids=(0,),
        )
        assert again.digest == projection.digest
        bound = _policy_metadata(store)
        assert bound is not None and bound[0] == POLICY.digest


def test_subnet_owner_burn_projection_refuses_any_real_economic_authority(
    tmp_path,
):
    with _store(tmp_path) as store:
        lease_candidate = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        assert lease_candidate.arena_digest in {
            row.arena_digest for row in store.evaluation_stacks()
        }
        context = _context("owner-burn", "validator", "miner")
        with pytest.raises(IntakeError, match="subnet-owner burn weights refused"):
            store.build_subnet_owner_burn_weight_projection(
                policy=POLICY,
                context=context,
                netuid=SCOPE.netuid,
                burn_hotkey="owner-burn",
                owner_coldkey="owner-ck",
                owner_hotkey="owner-burn",
                candidate_uids=(0,),
            )
        assert _policy_metadata(store) is None


def test_expired_settlement_lease_cannot_commit(tmp_path):
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11, lease_blocks=2)
        assert lease is not None
        plan, evidence = _settlement_plan(store, lease)
        with pytest.raises(IntakeError, match="deadline"):
            store.commit_settlement(lease, plan, evidence, current_block=13)
        assert store.evaluation_stack(lease.stack.arena_digest) == lease.stack


def test_pass_without_exact_settlement_projection_is_rejected_atomically(tmp_path):
    with _store(tmp_path) as store:
        row = _reserve_one(store)
        _publish(
            store, row.reservation_id, _fingerprint("target.a", "slot.a"),
            digest="d" * 64, root="/published/a",
        )
        _promote(store, row.reservation_id)
        store.mark_qualifying(row.reservation_id, "7" * 64, AUTHORITY)
        outcome = QualificationIntakeOutcome(
            row.reservation_id,
            "3" * 64,
            "7" * 64,
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=ATTEMPT.sha256,
            report_digest="4" * 64,
        )
        with pytest.raises(IntakeError, match="settlement projection"):
            store.apply_qualification_batch(
                QualificationIntakeBatch("7" * 64, (outcome,), ATTEMPT),
                current_finalized_block=10,
            )
        assert store.get(row.reservation_id).status == "qualifying"
        assert store.qualification_dispositions(row.reservation_id) == ()


def test_sqlite_weight_journal_is_cas_bound_and_restart_reopenable(tmp_path):
    projection = _journal_projection()
    intent = WeightPublicationRecord(
        projection.digest,
        "intent",
        submit_block=10,
        retry_after_block=20,
        reason="before_sdk_submission",
    )
    with _store(tmp_path) as store:
        journal = SQLiteWeightPublicationJournal(store, projection)
        assert journal.load() is None
        journal.compare_and_swap(None, intent)
        assert journal.load() == intent
        with pytest.raises(IntakeError, match="compare-and-swap"):
            journal.compare_and_swap(None, intent)

    with _store(tmp_path) as reopened:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(reopened)
        assert journal.projection == projection
        assert journal.load() == intent
        assert journal.retained_projection(projection.digest) == projection
        pending = WeightPublicationRecord(
            projection.digest,
            "pending",
            prior_record_digest=intent.digest,
            submit_block=10,
            retry_after_block=20,
            reason="sdk_result_unconfirmed",
        )
        journal.compare_and_swap(intent.digest, pending)
        assert journal.load() == pending
        assert reopened._db.execute(
            "SELECT COUNT(*) AS n FROM weight_publications"
        ).fetchone()["n"] == 2


def test_sqlite_weight_journal_reopen_rejects_corrupt_head_projection(tmp_path):
    projection = _journal_projection()
    pending = WeightPublicationRecord(
        projection.digest,
        "pending",
        submit_block=10,
        retry_after_block=20,
        reason="sdk_result_unconfirmed",
    )
    with _store(tmp_path) as store:
        journal = SQLiteWeightPublicationJournal(store, projection)
        journal.compare_and_swap(None, pending)
        store._db.execute(
            "UPDATE weight_publications SET projection_json='{}' "
            "WHERE record_digest=?",
            (pending.digest,),
        )
        with pytest.raises(IntakeError, match="projection is corrupt"):
            SQLiteWeightPublicationJournal.reopen_from_head(store)


def test_a_held_bundle_only_reopens_when_an_operator_releases_it(tmp_path):
    """A released qualification hold resumes after its retained screen."""

    with _store(tmp_path) as store:
        row = _reserve_one(store)
        _publish(
            store, row.reservation_id, _fingerprint("target.a", "slot.a"),
            digest="d" * 64, root="/published/a",
        )
        _promote(store, row.reservation_id)
        store.mark_held(
            row.reservation_id, "remote_qualification_hold:legacy_no_decision"
        )

        assert not hasattr(store, "auto_requeueable_holds")
        assert not hasattr(store, "mark_hold_retry_exhausted")
        assert store.get(row.reservation_id).status == "held"

        released = store.release_hold(
            row.reservation_id, reason="operator fixed the cause"
        )
        assert (released.status, released.screen_status) == ("promoted", "promote")

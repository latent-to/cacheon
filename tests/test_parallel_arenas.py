"""One intake database, independent commissions and recoverable GPU dispatch."""

from dataclasses import replace

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain.baseline_segments import commission_boundary
from cacheon.chain.intake import IntakeError
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from tests import test_evaluation_coordinator as fixture
from tests.test_baseline_segments import _manifest as stack_manifest
from tests.test_chain_intake import _qualified_settlement_candidate, _settlement_plan, _store


def _pair(tmp_path):
    rows = fixture._published_rows(tmp_path, 4, arenas=("", "qwen", "", "qwen"))
    glm = ArenaService(fixture._manifest(), fixture._Provider())
    qwen = ArenaService(replace(glm.manifest, runtime=replace(
        glm.manifest.runtime, arena_id="qwen", gpu_count=1,
        tensor_parallel_size=1, target_architecture="sm90",
        model_content_digest=fixture._h("qwen-weights"),
    )), fixture._Provider())
    cursor = fixture._CursorAuthority((fixture.BLOCK, fixture._block_hash(fixture.BLOCK)))
    coordinators = tuple(fixture._coordinator(
        tmp_path, service, cursor, accept_legacy_bundles=index == 0,
        owner=f"arena-{index}", qualification_max_members=1,
        store_factory=RecoverableFinalizedIntakeStore,
    ) for index, service in enumerate((glm, qwen)))
    return rows, coordinators


def test_two_dispatchers_keep_fifo_baselines_and_recovery_independent(tmp_path):
    rows, (glm, qwen) = _pair(tmp_path)
    # Screen both queues through the production claim/result path.
    for coordinator, expected in ((qwen, rows[1]), (glm, rows[0]),
                                  (qwen, rows[3]), (glm, rows[2])):
        result = fixture._run_screen(coordinator)
        assert result.lease.reservation_ids == (expected.reservation_id,)
    recoveries = []
    for coordinator in (glm, qwen):
        store, point = coordinator._open_at_durable_cursor()
        with store:
            incumbent = stack_manifest(coordinator.service.identity)
            assert commission_boundary(store, incumbent, tree_digest=fixture._h("tree")) is None
            recovery = store.claim_recoverable_qualification(
                owner=coordinator.owner, current_block=point[0], max_members=1,
            )
            assert recovery is not None
            recoveries.append(recovery)
            assert store.claim_recoverable_qualification(
                owner=coordinator.owner, current_block=point[0], max_members=1,
            ) is None
    assert recoveries[0].lease.reservation_ids == (rows[0].reservation_id,)
    assert recoveries[1].lease.reservation_ids == (rows[1].reservation_id,)
    store, point = qwen._open_at_durable_cursor()
    with store:
        held = store.hold_recovery(recoveries[1], reason="worker_unavailable", current_block=point[0])
        # A restart reopens this same request's state; it cannot select GLM's.
        assert store.pending_qualification_recovery() == held
        assert store.reservation_baseline_segment(rows[0].reservation_id).arena_digest == glm.service.identity
        assert store.reservation_baseline_segment(rows[2].reservation_id).arena_digest == glm.service.identity
    store, _ = glm._open_at_durable_cursor()
    with store:
        assert store.pending_qualification_recovery() == recoveries[0]
        assert len(store.active_evaluation_leases()) == 2
        assert store.arena_queue_snapshot(current_block=fixture.BLOCK).active_qualifications == 1


def test_default_alias_cannot_be_taken_by_second_arena(tmp_path):
    _, (glm, qwen) = _pair(tmp_path)
    store, _ = glm._open_at_durable_cursor()
    with store:
        assert store.publication_arena(glm.service.manifest.runtime.arena_id) == ""
        assert store.publication_arena("qwen") == "qwen"
        with pytest.raises(IntakeError, match="already belong"):
            store.select_arena("qwen", accept_legacy_bundles=True)
    qwen.accept_legacy_bundles = True
    with pytest.raises(Exception, match="already belong"):
        qwen.claim_screen()


def test_closed_targets_retire_before_claim_without_touching_glm(tmp_path):
    rows, (glm, qwen) = _pair(tmp_path)
    qwen.service = ArenaService(replace(qwen.service.manifest,
        closed_targets=(rows[1].target_id, rows[3].target_id)), fixture._Provider())
    qwen.readiness = fixture.WorkerReadiness.for_service(qwen.service,
        ready_receipt_digest=fixture._h("ready-receipt"), ready_epoch=7)
    assert qwen.claim_screen() is None
    store, _ = qwen._open_at_durable_cursor()
    with store:
        for row in (rows[1], rows[3]):
            parked = store.get(row.reservation_id)
            assert parked.decision == "NO_DECISION" and parked.screen_attempts == 0
            assert parked.status == "expired" and parked.reason == f"target_unavailable:{row.target_id}"
        assert not store.active_evaluation_leases()
    assert fixture._run_screen(glm).lease.reservation_ids == (rows[0].reservation_id,)


def test_same_target_crowns_and_rebuilds_independently_across_models(tmp_path):
    with _store(tmp_path) as store:
        store.select_arena("glm", accept_legacy_bundles=True)
        first = _qualified_settlement_candidate(store, marker="glm", speedups=("1.2", "1.19"))
        store.select_arena("qwen", accept_legacy_bundles=False)
        second = _qualified_settlement_candidate(
            store, index=1, marker="qwen", arena_marker="qwen", speedups=("1.05", "1.04"),
        )
        # One global committer drains both models; GLM's win sets no Qwen threshold.
        for expected in (first, second):
            lease = store.lease_settlement_cohort(current_block=11)
            assert lease is not None and lease.candidates == (expected,)
            assert not lease.lineage_tips
            plan, evidence = _settlement_plan(store, lease)
            store.commit_settlement(lease, plan, evidence, current_block=11)
        glm_tips = store.target_lineage_tips("")
        qwen_tips = store.target_lineage_tips("qwen")
        assert glm_tips.keys() == qwen_tips.keys() == {"activation.silu_and_mul"}
        assert glm_tips != qwen_tips
        store.backfill_target_lineage_tips()
        assert store.target_lineage_tips("") == glm_tips
        assert store.target_lineage_tips("qwen") == qwen_tips
        assert len(store.active_reward_claims()) == 2


def test_pre_namespace_lineage_migrates_without_changing_retained_evidence(tmp_path):
    with _store(tmp_path) as store:
        winner = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        tips = store.target_lineage_tips()
        retained = tuple(tuple(r) for r in store._db.execute("SELECT * FROM settlement_events"))
        # Recreate the historical column layout, preserving actual crowned rows.
        for table in ("target_lineage_tips", "target_lineage_nodes"):
            columns = ",".join(r["name"] for r in store._db.execute(f"PRAGMA table_info({table})")
                               if r["name"] != "competition_arena")
            store._db.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
            store._db.execute(f"CREATE TABLE {table} AS SELECT {columns} FROM {table}_old")
            store._db.execute(f"DROP TABLE {table}_old")
    with _store(tmp_path) as store:
        store.select_arena("glm", accept_legacy_bundles=True)
        assert store.target_lineage_tips() == tips
        assert tuple(tuple(r) for r in store._db.execute("SELECT * FROM settlement_events")) == retained
        assert store.evaluation_stack(winner.arena_digest).generation == 1
        store.select_arena("qwen", accept_legacy_bundles=False)
        assert store.target_lineage_tips() == {}

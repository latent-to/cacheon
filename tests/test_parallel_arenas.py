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


@pytest.mark.parametrize(("arena", "target"), (
    ("glm", "activation.silu_and_mul"), ("qwen", "norm.rmsnorm"),
))
@pytest.mark.parametrize("payment_kind", ("credit", "payment"))
def test_crown_cutoff_admits_commitments_once_before_screen(tmp_path, arena, target, payment_kind):
    from cacheon.chain.eval_cost_credit import grant_eval_cost_credit, list_eval_cost_credits
    from tests.test_chain_intake import _arrival, _bh, _fingerprint, _publish

    with _store(tmp_path) as store:
        store.select_arena(arena, accept_legacy_bundles=False)
        winner = _qualified_settlement_candidate(store, marker="winner")
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        # Neither commitment was in the transition's reservation snapshot.
        # Their chain blocks, not fetch/completion order, decide admission.
        late_arrival = _arrival(2, hotkey="late", block=12)
        if payment_kind == "credit":
            grant_eval_cost_credit(store.path, hotkey="late", amount_tao_rao=25)
        else:
            late_arrival = replace(late_arrival, payment_block=8, payment_extrinsic_index=4)
        early, late = store.reserve_finalized(
            (replace(_arrival(1, hotkey="early", block=11),
                     payment_block=8, payment_extrinsic_index=3), late_arrival),
            finalized_block=12, finalized_block_hash=_bh(12), eval_cost_amount_tao_rao=25,
        )
        if payment_kind == "credit":
            assert list_eval_cost_credits(store.path, hotkey="late")[0].reservation_id == late.reservation_id
        for row, marker in ((early, "a"), (late, "b")):
            _publish(store, row.reservation_id, _fingerprint(target, target, marker),
                     digest=marker * 64, root=tmp_path / marker)
        rejected = store.prepare_screen_queue(service_digest=winner.arena_digest)
        assert rejected == ((late.reservation_id, "baseline_closed_at_submission"),)
        assert store.get(early.reservation_id).status == "published"
        rejected_row = store.get(late.reservation_id)
        assert rejected_row.status == "expired" and rejected_row.screen_attempts == 0
        assert rejected_row.decision == "NO_DECISION"
        assert not store.active_evaluation_leases()
        if payment_kind == "credit":
            credit = list_eval_cost_credits(store.path, hotkey="late")[0]
            assert (credit.reservation_id, credit.spent_block) == ("", 0)
        else:
            assert store._db.execute("SELECT reservation_id FROM eval_cost_payments "
                                     "WHERE payment_extrinsic_index=4").fetchone() is None
        assert store.prepare_screen_queue(service_digest=winner.arena_digest) == ()
        assert store.get(winner.reservation_digest).decision == "PASS"

    # Reopening does not turn an earlier accepted commitment into a late one.
    with _store(tmp_path) as store:
        store.select_arena(arena, accept_legacy_bundles=False)
        assert store.prepare_screen_queue(service_digest=winner.arena_digest) == ()
        assert store.get(early.reservation_id).status == "published"
        fresh = store.reserve_finalized(
            (replace(late_arrival, block=13, block_hash=_bh(13)),),
            finalized_block=13, finalized_block_hash=_bh(13), eval_cost_amount_tao_rao=25,
        )[0]
        assert fresh.status == "reserved" and fresh.reason == ""
        if payment_kind == "credit":
            assert list_eval_cost_credits(store.path, hotkey="late")[0].reservation_id == fresh.reservation_id
        _publish(store, fresh.reservation_id, _fingerprint(target, target, "b"),
                 digest="b" * 64, root=tmp_path / "fresh")
        # A new commissioned service has its own open admission window.
        assert store.prepare_screen_queue(service_digest=fixture._h("new-commission")) == ()
        assert store.get(fresh.reservation_id).status == "published"
        # Another competition does not inherit this arena's crown cutoff.
        store.select_arena("other", accept_legacy_bundles=False)
        other = store.reserve_finalized(
            (_arrival(4, hotkey="other", block=14),),
            finalized_block=14, finalized_block_hash=_bh(14),
        )[0]
        _publish(store, other.reservation_id, _fingerprint(target, target, "d"),
                 digest="d" * 64, root=tmp_path / "other")
        assert store.prepare_screen_queue(service_digest=winner.arena_digest) == ()
        assert store.get(other.reservation_id).status == "published"

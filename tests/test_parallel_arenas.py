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
    recoveries = []
    for coordinator, queue in ((glm, (rows[0], rows[2])), (qwen, (rows[1], rows[3]))):
        store, point = coordinator._open_at_durable_cursor()
        with store:
            # Each arena's queue holds only its own rows, in finalized order.
            assert [row.reservation_id for row in store.claimable()] == [
                row.reservation_id for row in queue
            ]
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
        qwen._open_at_durable_cursor()


def test_closed_targets_retire_before_claim_without_touching_glm(tmp_path):
    rows, (glm, qwen) = _pair(tmp_path)
    qwen.service = ArenaService(replace(qwen.service.manifest,
        closed_targets=(rows[1].target_id, rows[3].target_id)), fixture._Provider())
    for coordinator in (qwen, glm):
        store, point = coordinator._open_at_durable_cursor()
        with store:
            store.prepare_qualification_queue(
                service_digest=coordinator.service.identity,
                closed_targets=coordinator.service.manifest.closed_targets,
            )
            recovery = store.claim_recoverable_qualification(
                owner=coordinator.owner, current_block=point[0], max_members=1,
            )
            if coordinator is qwen:
                assert recovery is None
                for row in (rows[1], rows[3]):
                    parked = store.get(row.reservation_id)
                    assert parked.decision == "NO_DECISION" and parked.arena_service_digest == ""
                    assert parked.status == "expired" and parked.reason == f"target_unavailable:{row.target_id}"
                assert not store.active_evaluation_leases()
    assert recovery.lease.reservation_ids == (rows[0].reservation_id,)


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
        assert glm_tips.keys() == qwen_tips.keys() == {"forward_pass"}
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
    ("glm", "forward_pass"), ("qwen", "prefix_cache"),
))
@pytest.mark.parametrize("payment_kind", ("credit", "payment"))
def test_crown_keeps_ancestor_admission_open_and_payment_bound(tmp_path, arena, target, payment_kind):
    from cacheon.chain.eval_cost_credit import grant_eval_cost_credit, list_eval_cost_credits
    from tests.test_chain_intake import _arrival, _bh, _fingerprint, _publish

    with _store(tmp_path) as store:
        store.select_arena(arena, accept_legacy_bundles=False)
        winner = _qualified_settlement_candidate(store, marker="winner")
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        # Both pre-crown and post-crown commitments use the commissioned baseline.
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
        assert store.prepare_qualification_queue(service_digest=winner.arena_digest) == ()
        assert store.get(early.reservation_id).status == "published"
        admitted = store.get(late.reservation_id)
        assert admitted.status == "published" and admitted.arena_service_digest == ""
        assert admitted.decision == "" and admitted.reason == ""
        assert not store.active_evaluation_leases()
        if payment_kind == "credit":
            credit = list_eval_cost_credits(store.path, hotkey="late")[0]
            assert credit.reservation_id == late.reservation_id
        else:
            assert store._db.execute("SELECT reservation_id FROM eval_cost_payments "
                                     "WHERE payment_extrinsic_index=4").fetchone()[0] == late.reservation_id
        assert store.prepare_qualification_queue(service_digest=winner.arena_digest) == ()
        assert store.get(winner.reservation_digest).decision == "PASS"

    # Restarting preserves both admissions and the late submission's payment.
    with _store(tmp_path) as store:
        store.select_arena(arena, accept_legacy_bundles=False)
        assert store.prepare_qualification_queue(service_digest=winner.arena_digest) == ()
        assert store.get(early.reservation_id).status == "published"
        assert store.get(late.reservation_id).status == "published"
        assert store.get(late.reservation_id).reason == ""
        if payment_kind == "credit":
            assert list_eval_cost_credits(store.path, hotkey="late")[0].reservation_id == late.reservation_id
        else:
            assert store._db.execute("SELECT reservation_id FROM eval_cost_payments "
                                     "WHERE payment_extrinsic_index=4").fetchone()[0] == late.reservation_id
        # Explicit baseline rotation also keeps unclaimed submissions eligible.
        assert store.prepare_qualification_queue(service_digest=fixture._h("new-commission")) == ()
        assert store.get(late.reservation_id).status == "published"
        # Another competition keeps its own admission queue.
        store.select_arena("other", accept_legacy_bundles=False)
        other = store.reserve_finalized(
            (_arrival(4, hotkey="other", block=14),),
            finalized_block=14, finalized_block_hash=_bh(14),
        )[0]
        _publish(store, other.reservation_id, _fingerprint(target, target, "d"),
                 digest="d" * 64, root=tmp_path / "other")
        assert store.prepare_qualification_queue(service_digest=winner.arena_digest) == ()
        assert store.get(other.reservation_id).status == "published"


@pytest.mark.parametrize(("crowns", "commit_block", "closed"), (
    (0, 20_000, False),       # An unbeaten commission does not expire.
    (1, 14_411, False),       # The 14,400-block boundary is inclusive.
    (1, 14_412, True),
    (4, 16, False),          # Baseline plus four newer winners is still five.
    (5, 15, False),          # Commitments in the fifth crown's block can drain.
    (5, 16, True),
    (5, 14_411, True),       # Five wins can close admission before the time limit.
    (5, 12, False),          # Later crowns do not punish delayed publication.
))
@pytest.mark.parametrize("payment_kind", ("credit", "payment"))
@pytest.mark.parametrize(("arena", "target"), (
    ("glm", "forward_pass"), ("qwen", "prefix_cache"),
))
def test_baseline_window_uses_the_earlier_bound_and_preserves_fees(
    tmp_path, crowns, commit_block, closed, payment_kind, arena, target,
):
    from cacheon.chain.eval_cost_credit import grant_eval_cost_credit, list_eval_cost_credits
    from tests.test_chain_intake import _arrival, _bh, _fingerprint, _publish

    service = fixture._h("arena")
    with _store(tmp_path) as store:
        store.select_arena(arena, accept_legacy_bundles=False)
        for index in range(crowns):
            winner = _qualified_settlement_candidate(
                store, index=index, marker=f"winner-{index}",
                speedups=(f"1.{index + 1}", f"1.{index + 1}"),
                initialize_stack=index == 0,
            )
            lease = store.lease_settlement_cohort(current_block=11 + index)
            plan, evidence = _settlement_plan(store, lease)
            store.commit_settlement(lease, plan, evidence, current_block=11 + index)
            assert store.target_lineage_tips()["forward_pass"].nodes[-1].artifact_digest == (
                winner.candidate_manifest.entries["forward_pass"].artifact_digest
            )
        arrival = _arrival(10, hotkey="late", block=commit_block)
        if payment_kind == "credit":
            grant_eval_cost_credit(store.path, hotkey="late", amount_tao_rao=25)
        else:
            arrival = replace(arrival, payment_block=5, payment_extrinsic_index=7)
        row = store.reserve_finalized(
            (arrival,), finalized_block=commit_block, finalized_block_hash=_bh(commit_block),
            eval_cost_amount_tao_rao=25,
        )[0]
        _publish(store, row.reservation_id, _fingerprint(target, target, "f"),
                 digest="f" * 64, root=tmp_path / "late")
        # A different commissioned service never inherits these crown counts.
        assert store.prepare_qualification_queue(service_digest=fixture._h("other-service")) == ()
        retired = store.prepare_qualification_queue(service_digest=service)
        assert retired == (((row.reservation_id, "baseline_closed_at_submission"),) if closed else ())
        result = store.get(row.reservation_id)
        assert result.status == ("expired" if closed else "published")
        assert result.decision == ("NO_DECISION" if closed else "")
        assert not store.active_evaluation_leases()
        if payment_kind == "credit":
            credit = list_eval_cost_credits(store.path, hotkey="late")[0]
            assert credit.reservation_id == ("" if closed else row.reservation_id)
            if closed:
                assert credit.spent_block == 0
        else:
            payment = store._db.execute("SELECT reservation_id FROM eval_cost_payments "
                                        "WHERE payment_extrinsic_index=7").fetchone()
            assert payment is None if closed else payment[0] == row.reservation_id

    with _store(tmp_path) as store:
        store.select_arena(arena, accept_legacy_bundles=False)
        assert store.prepare_qualification_queue(service_digest=service) == ()
        assert store.get(row.reservation_id).status == result.status


def test_closed_baseline_keeps_prior_claims_and_other_competitions(tmp_path):
    from tests.test_chain_intake import _arrival, _bh, _claim, _fingerprint, _publish

    with _store(tmp_path) as store:
        store.select_arena("glm", accept_legacy_bundles=False)
        winner = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        plan, evidence = _settlement_plan(store, lease)
        store.commit_settlement(lease, plan, evidence, current_block=11)
        for index, arena in enumerate(("glm", "qwen"), start=1):
            store.select_arena(arena, accept_legacy_bundles=False)
            row = store.reserve_finalized(
                (_arrival(index, hotkey=arena, block=20_000),),
                finalized_block=20_000, finalized_block_hash=_bh(20_000),
            )[0]
            _publish(store, row.reservation_id, _fingerprint("forward_pass", "forward_pass", "f"),
                     digest="f" * 64, root=tmp_path / arena)
            if arena == "glm":
                # Work admitted before this policy upgrade retains its first claim.
                _claim(store, row.reservation_id, service=winner.arena_digest)
                store.mark_held(row.reservation_id, "worker_unavailable")
                store.release_hold(row.reservation_id, reason="worker_recovered")
                assert store.get(row.reservation_id).arena_service_digest == winner.arena_digest
            assert store.prepare_qualification_queue(service_digest=winner.arena_digest) == ()
            assert store.get(row.reservation_id).status == "published"

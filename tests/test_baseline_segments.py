"""A retired arena's baseline segment must not wedge the live arena's qualification lane.

On 2026-09-06 a promoted reservation screened under one deployment was
re-queued after the validator redeployed on new worker bytes. Its segment
named the retired arena, the commission boundary refused to claim across it,
the selector could never pick it, and the rotated-cohort re-screen sat behind
that boundary. The rules under test: a screen under the live arena replaces a
retired arena's segment, the boundary rebinds evidence-free rows left on a
retired arena before it halts, and a same-arena generation boundary still
halts because those rows drain under the still-resident commission.
"""

from __future__ import annotations

from cacheon.chain.baseline_segments import commission_boundary
from cacheon.chain.screen_identity_rotation import rotated_reservation_ids
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.target_catalog import default_target_catalog
from tests.test_chain_intake import (
    _fingerprint,
    _h,
    _promote,
    _publish,
    _reserve_one,
    _store,
)
from tests.test_recoverable_qualification_dispatcher import (
    _Transport,
    _dispatcher,
    _fixtures,
    _store as _dispatcher_store,
)

RETIRED = _h("retired-arena")
LIVE = _h("live-arena")
ROTATED = "screen_receipt_service_rotated"


def _manifest(arena: str, *, runtime: str = "runtime") -> EvaluationStackManifest:
    catalog = default_target_catalog()
    return EvaluationStackManifest(
        runtime_digest=_h(runtime),
        base_engine_digest=_h("base"),
        arena_digest=arena,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )


def _published_row(store, *, index: int = 0, hotkey: str = "miner") -> str:
    row = _reserve_one(store, index=index, hotkey=hotkey)
    _publish(
        store,
        row.reservation_id,
        _fingerprint(f"target.{index}", f"slot.{index}"),
        digest=_h(f"publication:{index}"),
        root=f"/published/{index}",
    )
    return row.reservation_id


def _binding(store, reservation_id: str):
    return store._db.execute(
        "SELECT arena_id,binding_reason FROM reservation_baseline_segments "
        "WHERE reservation_id=?",
        (reservation_id,),
    ).fetchone()


def test_a_screen_under_the_live_arena_replaces_a_retired_arena_segment(tmp_path):
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(_manifest(RETIRED), tree_digest=_h("retired-tree"))
        rid = _published_row(store)
        _promote(store, rid, service=RETIRED)
        assert _binding(store, rid)["arena_id"] == RETIRED
        store.demote_promoted_for_rescreen(rid, reason=ROTATED)
        store.initialize_evaluation_stack(_manifest(LIVE), tree_digest=_h("live-tree"))

        _promote(store, rid, service=LIVE)

        assert store.reservation_baseline_segment(rid) == store.evaluation_stack(LIVE)
        assert _binding(store, rid)["binding_reason"] == (
            "begin_screen:rebound_from_" + RETIRED[:16]
        )
        # Retained screen dispositions are append-only: both receipts survive.
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM arena_screen_dispositions WHERE reservation_id=?",
            (rid,),
        ).fetchone()["n"] == 2


def test_a_crown_does_not_change_the_manually_commissioned_baseline(tmp_path):
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(_manifest(LIVE), tree_digest=_h("live-tree"))
        rid = _published_row(store)
        _promote(store, rid, service=LIVE)
        bound = store.reservation_baseline_segment(rid)
        advanced = _manifest(LIVE, runtime="runtime-advanced")
        assert advanced.digest != bound.manifest.digest
        store._db.execute(
            "UPDATE evaluation_stacks SET generation=1,stack_digest=?,tree_digest=?,"
            "stack_json=?,transition_event_id=? WHERE arena_id=?",
            (
                advanced.digest,
                _h("advanced-tree"),
                store._encoded_stack_manifest(
                    store.evaluation_stack(LIVE).__class__(
                        LIVE, 1, advanced, _h("advanced-tree"), _h("advanced-transition")
                    )
                ),
                _h("advanced-transition"),
                LIVE,
            ),
        )

        store.demote_promoted_for_rescreen(rid, reason=ROTATED)
        _promote(store, rid, service=LIVE)

        assert store.reservation_baseline_segment(rid) == bound
        assert _binding(store, rid)["binding_reason"] == "begin_screen"
        assert commission_boundary(store, bound.manifest, tree_digest=bound.tree_digest) is None
        assert store.reservation_baseline_segment(rid) == bound
        assert store.evaluation_stack(LIVE).manifest == advanced


def test_the_boundary_rebinds_evidence_free_rows_left_on_a_retired_arena(tmp_path):
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(_manifest(RETIRED), tree_digest=_h("retired-tree"))
        head = _published_row(store, index=0, hotkey="head")
        _promote(store, head, service=RETIRED)
        waiting = _published_row(store, index=1, hotkey="waiting")
        store._bind_reservation_baseline_segment(
            waiting, store.evaluation_stack(RETIRED), reason="backfill_current_stack"
        )
        passed = _published_row(store, index=2, hotkey="passed")
        _promote(store, passed, service=RETIRED)
        store._db.execute(
            "UPDATE reservations SET status='reproduction_pending' WHERE reservation_id=?",
            (passed,),
        )
        live = _manifest(LIVE)

        assert commission_boundary(store, live, tree_digest=_h("live-tree")) is None

        live_state = store.evaluation_stack(LIVE)
        assert live_state.generation == 0
        for rid in (head, waiting):
            assert store.reservation_baseline_segment(rid) == live_state
            assert _binding(store, rid)["binding_reason"] == "commissioned_incumbent"
        # A PASS half keeps its segment: the boundary stays visible to the operator.
        assert _binding(store, passed)["arena_id"] == RETIRED
        # The head's receipt came from the retired identity, so the claim path
        # re-screens it under the live one instead of qualifying on it.
        receipt = store.latest_promoted_screen(head)
        assert rotated_reservation_ids((store.get(head),), (receipt,), LIVE) == (head,)


def test_a_row_back_in_the_screen_queue_binds_itself_with_its_retry_group(tmp_path):
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(_manifest(LIVE), tree_digest=_h("live-tree"))
        rid = _published_row(store)
        store._db.execute(
            "UPDATE reservations SET retry_group_digest=? WHERE reservation_id=?",
            (_h("group"), rid),
        )

        _promote(store, rid, service=LIVE)

        assert store.reservation_baseline_segment(rid) == store.evaluation_stack(LIVE)


def test_the_dispatcher_claims_a_head_left_on_a_retired_arena(tmp_path):
    fixtures = _fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    live = authority.fixtures._incumbent(authority.service)
    retired = EvaluationStackManifest.from_dict({**live.to_dict(), "arena_digest": RETIRED})
    with _dispatcher_store(authority) as store:
        store._release_recovery(
            store.pending_qualification_recovery(), current_block=authority.fixtures.BLOCK,
            reason="unstarted_fixture",
        )
        store.initialize_evaluation_stack(
            retired, tree_digest=authority.fixtures._h("retired-tree")
        )
        head = store._db.execute(
            "SELECT reservation_id FROM reservations WHERE status='promoted' "
            "ORDER BY block,event_index,event_subindex,hotkey,content_hash LIMIT 1"
        ).fetchone()["reservation_id"]
        store._bind_reservation_baseline_segment(
            head, store.evaluation_stack(RETIRED), reason="begin_screen"
        )
        assert store.qualification_queue_baseline().arena_digest == RETIRED
    transport = _Transport(authority, fixtures, complete_on_publish=True)

    outcome = _dispatcher(authority, transport).dispatch_once()

    assert type(outcome).__name__ == "EvaluationRun"
    with _dispatcher_store(authority) as store:
        assert store.reservation_baseline_segment(head) == store.evaluation_stack(
            authority.service.identity
        )
        assert _binding(store, head)["binding_reason"] == "commissioned_incumbent"


def test_post_crown_misbound_work_qualifies_against_the_manual_incumbent(tmp_path):
    import json

    for profile in ("alpha", "beta"):
        fixtures = _fixtures()
        authority = fixtures._authority(tmp_path / profile, profile=profile, recoverable=True)
        original = authority.fixtures._incumbent(authority.service)
        original_tree = authority.fixtures._h("incumbent-tree")
        crowned = authority.fixtures._incumbent(authority.service, marker="crowned")
        reservation_id = authority.claim.lease.reservation_ids[0]
        with _dispatcher_store(authority) as store:
            store._release_recovery(
                store.pending_qualification_recovery(),
                current_block=authority.fixtures.BLOCK, reason="unstarted_fixture",
            )
            store.initialize_evaluation_stack(original, tree_digest=original_tree)
            store._db.execute(
                "UPDATE evaluation_stacks SET generation=1,stack_digest=?,tree_digest=?,"
                "stack_json=?,transition_event_id=? WHERE arena_id=?",
                (crowned.digest, authority.fixtures._h("crowned-tree"),
                 json.dumps(crowned.to_dict(), separators=(",", ":"), sort_keys=True),
                 authority.fixtures._h("crown-transition"), authority.service.identity),
            )
            store._db.execute("DELETE FROM reservation_baseline_segments")
            store._bind_reservation_baseline_segment(
                reservation_id, store.evaluation_stack(authority.service.identity),
                reason="old_begin_screen",
            )
        transport = _Transport(authority, fixtures, complete_on_publish=True)

        outcome = _dispatcher(authority, transport).dispatch_once()

        assert outcome.disposition == "completed"
        assert transport.plan.remote_request.body["incumbent_stack_digest"] == original.digest
        assert transport.plan.remote_request.body["incumbent_tree_digest"] == original_tree
        assert (transport.plans, transport.publications) == (1, 1)
        with _dispatcher_store(authority) as store:
            assert store.reservation_baseline_segment(reservation_id).manifest == original
            assert store.evaluation_stack(authority.service.identity).manifest == crowned
            assert store.pending_qualification_recovery() is None

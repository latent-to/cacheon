"""Declared baselines supersede the 2026-09-06 automatic retired-arena repair.

The old repair prevented a queue wedge by silently rebinding unmeasured work.
The declared-baseline contract instead stops before dispatch and preserves the
original baseline, including during restarts and commissioned-arena changes.
"""

import pytest

from cacheon.chain.baseline_segments import commission_boundary
from cacheon.chain.declared_baseline import current_baseline
from cacheon.chain.intake import IntakeError
from tests.test_chain_intake import _store, _h, _reserve_one
from tests.intake_fixtures import fixture_baseline
from cacheon.stack_manifest import EvaluationStackManifest


def test_commission_keeps_the_exact_queue_segment(tmp_path):
    with _store(tmp_path) as store:
        row = _reserve_one(store)
        before = store.reservation_baseline_segment(row.reservation_id)
        assert before == current_baseline(store)
        assert commission_boundary(store, before.manifest, tree_digest=before.tree_digest) is None
        assert store.reservation_baseline_segment(row.reservation_id) == before
        assert store.get(row.reservation_id).arrival.baseline_ref == before.manifest.digest
    with _store(tmp_path) as reopened:
        assert reopened.reservation_baseline_segment(row.reservation_id) == before


def test_a_different_arena_stops_before_rebinding(tmp_path):
    with _store(tmp_path) as store:
        row = _reserve_one(store)
        before = store.reservation_baseline_segment(row.reservation_id)
        raw = fixture_baseline().to_dict()
        raw["arena_digest"] = _h("new-arena")
        new = EvaluationStackManifest.from_dict(raw)
        boundary = commission_boundary(store, new, tree_digest=_h("new-tree"))
        assert boundary == (new.digest, before.manifest.digest, before.tree_digest)
        assert store.reservation_baseline_segment(row.reservation_id) == before
        assert current_baseline(store).manifest == new
        assert commission_boundary(store, before.manifest, tree_digest=before.tree_digest) is None
        assert current_baseline(store).manifest == new
        with pytest.raises(IntakeError, match="cannot be rebound"):
            store._bind_reservation_baseline_segment(row.reservation_id, current_baseline(store), reason="rebind")
        assert store.reservation_baseline_segment(row.reservation_id) == before


def test_same_manifest_cannot_change_its_tree(tmp_path):
    with _store(tmp_path) as store:
        row = _reserve_one(store)
        head = current_baseline(store)
        with pytest.raises(IntakeError, match="tree changed"):
            commission_boundary(store, head.manifest, tree_digest=_h("different-tree"))
        assert store.reservation_baseline_segment(row.reservation_id) == head
        assert current_baseline(store) == head

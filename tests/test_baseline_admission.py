"""Disclosed baseline admission and FIFO loading share retained crown authority."""

from dataclasses import replace

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.chain.baseline_admission import (
    admit_revealed_baselines, latest_revealed, promotion_target, revealed_baselines,
)
from cacheon.chain.baseline_segments import commission_boundary
from cacheon.chain.intake import IntakeError
from cacheon.chain.publication import publish_worker_bundle
from cacheon.manifest import ManifestError, load_manifest
from tests.test_chain_intake import (
    _arrival, _fingerprint, _h, _publish, _qualified_settlement_candidate,
    _reserve, _settlement_plan, _store,
)


def crown(store, *, index=0, marker="winner", target="forward_pass", retained=10):
    """Use the production settlement transaction to install a test crown."""
    candidate = _qualified_settlement_candidate(
        store, index=index, marker=marker, target=target, retained_block=retained,
        speedups=("1.2", "1.19") if index else ("1.05", "1.04"),
        initialize_stack=not index,
    )
    lease = store.lease_settlement_cohort(current_block=11)
    plan, evidence = _settlement_plan(store, lease)
    state = store.commit_settlement(lease, plan, evidence, current_block=11)
    return candidate, state


def clock(block):
    return {"unix": {10: 1000, 11: 2000, 12: 29799, 13: 29800, 14: 30800}[block],
            "estimated": False}


def published(store, tmp_path, index, block, baseline, target="forward_pass"):
    root = tmp_path / f"bundle-{index}"
    root.mkdir(mode=0o700)
    (root / "kernel.py").write_text("def run(x): return x\n")
    declaration = "" if baseline is None else f'baseline = "{baseline}"\n'
    (root / "manifest.toml").write_text(
        f'bundle_id = "test-{index}"\nabi_version = "cacheon-op-abi-v0"\n'
        f'[competition]\ntarget = "{target}"\nmode = "slot"\n{declaration}'
        '[[ops]]\nslot = "norm.rmsnorm"\nsource = "kernel.py"\nentry = "run"\n'
    )
    for path in root.iterdir():
        path.chmod(0o600)
    digest = content_hash(root)
    row = _reserve(store, (replace(_arrival(index, block=block), content_hash=digest),), block=block)[0]
    publication = publish_worker_bundle(root, tmp_path / "publications", digest)
    _publish(store, row.reservation_id, _fingerprint(target, target),
             digest=publication.digest, root=str(publication.root))
    return row.reservation_id


@pytest.mark.parametrize("target", ["forward_pass", "prefix_cache"])
def test_reveal_closes_old_admission_without_rebinding_the_old_queue(tmp_path, target):
    with _store(tmp_path, expiry_blocks=100000) as store:
        candidate, winner = crown(store, target=target)
        old = published(store, tmp_path, 1, 12, "stock", target)
        stale = published(store, tmp_path, 2, 13, "stock", target)
        new = published(store, tmp_path, 3, 13, candidate.reservation_digest, target)
        # Stock was commissioned before the crown; the engine can still be loaded there.
        incumbent, tree = candidate.incumbent_manifest, candidate.incumbent_tree_digest
        admit_revealed_baselines(store, incumbent, tree, clock, 10)
        assert store.get(stale).reason == "baseline_not_latest_revealed"
        assert store.get(stale).decision == "NO_DECISION"
        assert store.reservation_baseline_segment(new).manifest == winner.manifest
        assert store.reservation_baseline_segment(old).manifest == incumbent
        assert commission_boundary(store, incumbent, tree_digest=tree, preserve_revealed=True) is None
        assert promotion_target(store, incumbent) is None
        # Finishing the old queue exposes the new segment without measuring its winner again.
        store.mark_failed(old, "test_failure")
        assert commission_boundary(store, incumbent, tree_digest=tree, preserve_revealed=True) is not None
        assert promotion_target(store, incumbent) == winner
        assert commission_boundary(store, winner.manifest, tree_digest=winner.tree_digest,
                                   preserve_revealed=True) is None
        assert promotion_target(store, winner.manifest) is None
        assert store.get(new).status == "published"
        assert store.preview_evaluation_claim(stage="qualification", max_members=1) == (new,)


def test_delayed_admission_uses_finalized_submission_time_and_survives_restart(tmp_path):
    with _store(tmp_path, expiry_blocks=100000) as store:
        candidate, _ = crown(store)
        old = published(store, tmp_path, 1, 12, "stock")
        # The source has long since become public when this delayed fetch is checked.
        admit_revealed_baselines(store, candidate.incumbent_manifest,
                                candidate.incumbent_tree_digest, clock, 10)
        before = store.reservation_baseline_segment(old)
        assert store.get(old).status == "published"
    with _store(tmp_path, expiry_blocks=100000) as store:
        admit_revealed_baselines(store, candidate.incumbent_manifest,
                                candidate.incumbent_tree_digest, lambda _: pytest.fail("accepted row rechecked"), 10)
        assert store.reservation_baseline_segment(old) == before


@pytest.mark.parametrize("baseline", [None, "stock", "a" * 64])
def test_missing_stale_and_unknown_baselines_are_released_before_claim(tmp_path, baseline):
    with _store(tmp_path) as store:
        candidate, _ = crown(store)
        rid = published(store, tmp_path, 1, 13, baseline)
        admit_revealed_baselines(store, candidate.incumbent_manifest,
                                candidate.incumbent_tree_digest, clock, 10)
        row = store.get(rid)
        assert (row.status, row.decision, row.arena_service_digest) == ("expired", "NO_DECISION", "")
        assert not store.active_evaluation_leases()


def test_only_the_latest_public_winner_is_admissible_and_other_arenas_do_not_interfere(tmp_path):
    with _store(tmp_path) as store:
        first, old = crown(store)
        second, new = crown(store, index=1, marker="second", retained=11)
        history = revealed_baselines(store._db, store._competition_arena, clock, first.incumbent_manifest)
        assert latest_revealed(history, 29799) is None
        assert latest_revealed(history, 29800)[2] == old
        assert latest_revealed(history, 30799)[1] == first.reservation_digest
        assert latest_revealed(history, 30800)[2] == new
        stale = published(store, tmp_path, 2, 14, first.reservation_digest)
        current = published(store, tmp_path, 3, 14, second.reservation_digest)
        admit_revealed_baselines(store, first.incumbent_manifest,
                                first.incumbent_tree_digest, clock, 10)
        assert store.get(stale).reason == "baseline_not_latest_revealed"
        assert store.reservation_baseline_segment(current) == new
        store.select_arena("other", accept_legacy_bundles=False)
        assert revealed_baselines(store._db, store._competition_arena, clock, first.incumbent_manifest) == ()
        assert second.reservation_digest != first.reservation_digest


@pytest.mark.parametrize("status", ["fetching", "held", "no_decision", "qualifying"])
def test_unresolved_old_work_blocks_loading(tmp_path, status):
    with _store(tmp_path) as store:
        candidate, _ = crown(store)
        old = published(store, tmp_path, 1, 12, "stock")
        published(store, tmp_path, 2, 13, candidate.reservation_digest)
        admit_revealed_baselines(store, candidate.incumbent_manifest,
                                candidate.incumbent_tree_digest, clock, 10)
        store._db.execute("UPDATE reservations SET status=? WHERE reservation_id=?", (status, old))
        assert promotion_target(store, candidate.incumbent_manifest) is None


def test_missing_exact_times_are_infrastructure_errors_not_miner_losses(tmp_path):
    with _store(tmp_path) as store:
        candidate, _ = crown(store)
        rid = published(store, tmp_path, 1, 13, candidate.reservation_digest)
        with pytest.raises(IntakeError, match="timestamp"):
            admit_revealed_baselines(store, candidate.incumbent_manifest, candidate.incumbent_tree_digest,
                                    lambda _: {"unix": 1000, "estimated": True}, 10)
        assert store.get(rid).status == "published"


def test_declared_baseline_is_hash_bound_and_syntax_checked(tmp_path):
    with _store(tmp_path) as store:
        rid = published(store, tmp_path, 1, 10, "stock")
        root = tmp_path / "bundle-1"
        assert load_manifest(root).competition.baseline == "stock"
        original = content_hash(root)
        path = root / "manifest.toml"
        path.write_text(path.read_text().replace('baseline = "stock"', f'baseline = "{_h("winner")}"'))
        assert content_hash(root) != original
        assert store.get(rid).arrival.content_hash == original
        path.write_text(path.read_text().replace(_h("winner"), "bad"))
        with pytest.raises(ManifestError, match="baseline"):
            load_manifest(root)


def test_public_baseline_endpoint_matches_admission_at_disclosure(tmp_path, monkeypatch):
    import sqlite3
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from dashboard import disclosure

    with _store(tmp_path) as store:
        store.select_arena("test-arena", accept_legacy_bundles=True)
        candidate, _ = crown(store)
        path = store.path
    def connect():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con
    app = FastAPI()
    disclosure.install_disclosure_routes(app, connect, clock, lambda: tmp_path, lambda: tmp_path,
        lambda: {"worker_readiness": {"arena_id": "test-arena"},
                 "service_identity": candidate.incumbent_manifest.arena_digest})
    client = TestClient(app)
    monkeypatch.setattr(disclosure.time, "time", lambda: 29799)
    assert client.get("/api/baseline").json()["baseline"] == "stock"
    monkeypatch.setattr(disclosure.time, "time", lambda: 29800)
    baseline = client.get("/api/baseline").json()
    assert baseline["baseline"] == candidate.reservation_digest
    assert baseline["competition_arena"] == "test-arena"
    assert baseline["bundle_url"] == f"/api/submissions/{candidate.reservation_digest}/bundle.tar.gz"
    assert client.get("/api/baseline?arena=unknown").status_code == 404


def test_late_settlement_does_not_retroactively_close_admission(tmp_path):
    with _store(tmp_path) as store:
        candidate, _ = crown(store)
        store._db.execute("UPDATE target_lineage_nodes SET crowned_block=14")
        old = published(store, tmp_path, 1, 13, "stock")
        admit_revealed_baselines(store, candidate.incumbent_manifest,
                                candidate.incumbent_tree_digest, clock, 10)
        assert store.get(old).status == "published"
        assert not store.reservation_baseline_segment(old).manifest.entries

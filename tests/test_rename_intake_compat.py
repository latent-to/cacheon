"""Regression coverage for durable state created before the Cacheon rename."""

from __future__ import annotations

import json

from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeScope,
    SQLiteWeightPublicationJournal,
)
from cacheon.chain.weights import WeightProjection, WeightPublicationRecord
from cacheon.stack_identity import canonical_digest, sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _scope() -> IntakeScope:
    return IntakeScope("0x" + "1" * 64, 307)


def _store(path) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(path, IntakePolicy(), scope=_scope())


def test_cacheon_reopens_legacy_intake_and_evaluation_stack(tmp_path) -> None:
    """New writes use the same identities as rows emitted by Optima #72."""

    arrival_payload = {
        "content_hash": "2" * 64,
        "url": "https://example.com/candidate.tar.gz",
    }
    legacy_payload_digest = canonical_digest(
        "optima.chain.finalized-payload", arrival_payload
    )
    arrival = FinalizedArrival(
        hotkey="miner",
        content_hash=arrival_payload["content_hash"],
        url=arrival_payload["url"],
        block=10,
        block_hash="0x" + "3" * 64,
        event_index=4,
    )
    legacy_reservation_id = canonical_digest(
        "optima.chain.finalized-arrival",
        {
            "block": arrival.block,
            "block_hash": arrival.block_hash,
            "content_hash": arrival.content_hash,
            "event_index": arrival.event_index,
            "event_subindex": arrival.event_subindex,
            "hotkey": arrival.hotkey,
            "payload_digest": legacy_payload_digest,
            "url": arrival.url,
        },
    )
    assert arrival.payload_digest == legacy_payload_digest
    assert arrival.reservation_id == legacy_reservation_id
    assert _scope().digest == canonical_digest(
        "optima.chain.intake-scope", _scope().to_dict()
    )

    catalog_snapshot = {
        "composition_rules": [],
        "policy_version": "target-catalog.v1",
        "schema_version": 1,
        "targets": [],
    }
    legacy_catalog_digest = canonical_digest(
        "optima.target-catalog", catalog_snapshot
    )
    manifest = EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base-engine"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog_snapshot,
        catalog_digest=legacy_catalog_digest,
        entries={},
    )
    legacy_stack_digest = canonical_digest(
        "optima.stack.evaluation", manifest.to_dict()
    )
    tree_digest = _h("tree")
    legacy_genesis = canonical_digest(
        "optima.chain.evaluation-stack-genesis",
        {
            "arena_digest": manifest.arena_digest,
            "stack_digest": legacy_stack_digest,
            "tree_digest": tree_digest,
        },
    )
    assert manifest.digest == legacy_stack_digest

    path = tmp_path / "private" / "intake.sqlite3"
    with _store(path) as store:
        reserved = store.reserve_finalized(
            (arrival,),
            finalized_block=arrival.block,
            finalized_block_hash=arrival.block_hash,
        )
        assert reserved[0].reservation_id == legacy_reservation_id
        stack = store.initialize_evaluation_stack(manifest, tree_digest=tree_digest)
        assert stack.transition_event_id == legacy_genesis

        stored = store._db.execute(
            "SELECT stack_digest,stack_json,transition_event_id "
            "FROM evaluation_stacks WHERE arena_id=?",
            (manifest.arena_digest,),
        ).fetchone()
        assert stored["stack_digest"] == legacy_stack_digest
        assert json.loads(stored["stack_json"]) == manifest.to_dict()
        assert stored["transition_event_id"] == legacy_genesis

    with _store(path) as reopened:
        assert reopened.get(legacy_reservation_id).arrival == arrival
        stack = reopened.evaluation_stack(manifest.arena_digest)
        assert stack.manifest == manifest
        assert stack.manifest.digest == legacy_stack_digest
        assert stack.transition_event_id == legacy_genesis


def test_cacheon_reopens_legacy_weight_publication_journal(tmp_path) -> None:
    projection = WeightProjection(
        chain_scope_digest=_h("scope"),
        netuid=307,
        validator_hotkey="validator",
        policy_digest=_h("policy"),
        settlement_state_digest=_h("settlement"),
        evaluation_state_digest=_h("evaluation"),
        metagraph_digest=_h("metagraph"),
        arena_state_digests=(_h("arena-state"),),
        stack_generation=1,
        effective_block=10,
        crown_count=1,
        evidence_digests=(_h("evidence"),),
        weights_ppm=(("miner", 1_000_000),),
    )
    legacy_projection_digest = canonical_digest(
        "optima.chain.weight-projection", projection.to_dict()
    )
    assert projection.digest == legacy_projection_digest

    intent = WeightPublicationRecord(
        projection_digest=legacy_projection_digest,
        status="intent",
        submit_block=10,
        retry_after_block=20,
        reason="before_sdk_submission",
    )
    legacy_record_digest = canonical_digest(
        "optima.chain.weight-publication", intent.to_dict()
    )
    assert intent.digest == legacy_record_digest

    path = tmp_path / "private" / "intake.sqlite3"
    with _store(path) as store:
        journal = SQLiteWeightPublicationJournal(store, projection)
        journal.compare_and_swap(None, intent)
        raw = store._db.execute(
            "SELECT projection_digest,record_digest FROM weight_publications"
        ).fetchone()
        assert tuple(raw) == (legacy_projection_digest, legacy_record_digest)

    with _store(path) as reopened:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(reopened)
        assert journal.projection == projection
        assert journal.load() == intent
        assert journal.retained_projection(legacy_projection_digest) == projection

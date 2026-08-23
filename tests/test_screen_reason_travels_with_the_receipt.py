"""A rejected bundle must be told why, from anywhere -- not only from the pod.

A screen stage hashes ``{authority, candidate, facts, grade, publication,
reason, screen_attempt, service, stage}`` into one digest and stores no
payload. Every field but the authority is in the receipt or the reservation, so
the reason can be rebuilt and rehashed until it matches -- provided the
authority is known.

It was not. The report re-derived it by constructing a live screen adapter,
which produces the deployment's digest only when run on the commissioned
deployment. Off the pod it silently produced a *different* digest, no candidate
payload matched, and the report degraded to "failed at static" with no cause --
while every test passed, because the tests called the recovery with the
authority handed to them.

So these tests never pass the authority in. They store it the way production
stores it and read it back the way the report reads it.
"""

from __future__ import annotations

from cacheon.arena_service import (
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain.intake import FinalizedArrival, FinalizedIntakeStore, IntakeScope
from cacheon.chain.operator_status import _screen_dispositions
from cacheon.eval.screen_reason import SCREEN_EVIDENCE_SCHEMA, recover_screen_reason
from cacheon.stack_identity import canonical_digest


SCOPE = IntakeScope("0x" + "0" * 64, 307)
BLOCK = 40
BLOCK_HASH = "0x" + "4" * 64

SERVICE = "1" * 64
CANDIDATE = "2" * 64
AUTHORITY = "a" * 64


def _stage_evidence(publication_digest: str, reason: str, facts: dict) -> str:
    """The digest ``_stage_result`` would produce, built from the same fields."""

    return canonical_digest(
        SCREEN_EVIDENCE_SCHEMA,
        {
            "authority_digest": AUTHORITY,
            "candidate_digest": CANDIDATE,
            "facts": dict(sorted(facts.items())),
            "grade": "fail",
            "publication_digest": publication_digest,
            "reason": reason,
            "screen_attempt": 1,
            "service_digest": SERVICE,
            "stage": "static",
        },
    )


def _rejected_store(
    tmp_path,
    *,
    reason: str = "static_policy",
    facts: dict | None = None,
    record_authority: bool = True,
) -> tuple[FinalizedIntakeStore, str]:
    store = FinalizedIntakeStore(tmp_path / "state" / "intake.sqlite3", scope=SCOPE)
    arrival = FinalizedArrival(
        "miner-0", f"{1:064x}", "https://host.example/b.tar.gz", BLOCK, BLOCK_HASH, 0
    )
    store.reserve_finalized(
        (arrival,), finalized_block=BLOCK, finalized_block_hash=BLOCK_HASH
    )
    store._db.execute(
        "UPDATE reservations SET status='published',publication_digest=? "
        "WHERE reservation_id=?",
        ("9" * 64, arrival.reservation_id),
    )
    store.begin_screen(arrival.reservation_id, service_digest=SERVICE)
    evidence = _stage_evidence("9" * 64, reason, facts or {})
    store.apply_screen_receipt(
        arrival.reservation_id,
        candidate_digest=CANDIDATE,
        receipt=ArenaScreenReceipt(
            SERVICE,
            CANDIDATE,
            1,
            (ScreenStageResult("static", ScreenGrade.FAIL, evidence, 17),),
            PromotionDecision.REJECT,
        ),
        stage_authorities={"static": AUTHORITY} if record_authority else None,
    )
    return store, arrival.reservation_id


def _recovered(store, reservation_id, *, slots=()):
    """Recover using only what a reader gets back out of the database."""

    disposition = _screen_dispositions(store._db, reservation_id)[0]
    stage = disposition["stages"][0]
    authority = disposition["stage_authorities"].get("static")
    if not authority:
        return None
    return recover_screen_reason(
        stage=stage["stage"],
        grade=stage["grade"],
        evidence_digest=stage["evidence_digest"],
        authority_digest=authority,
        candidate_digest=disposition["candidate_digest"],
        publication_digest="9" * 64,
        service_digest=disposition["service_digest"],
        screen_attempt=disposition["attempt_index"] + 1,
        slots=slots,
    )


def test_the_reason_survives_the_round_trip_through_storage(tmp_path):
    store, reservation_id = _rejected_store(
        tmp_path, facts={"exception_type": "_CandidateStaticFailure"}
    )
    try:
        found = _recovered(store, reservation_id)
    finally:
        store.close()

    assert found is not None
    assert found.reason == "static_policy"
    assert found.sentence() == (
        "static: static_policy (exception_type _CandidateStaticFailure)"
    )


def test_without_the_recorded_authority_nothing_is_recovered(tmp_path):
    """The column is load-bearing, not decorative.

    This is the state every receipt written before it was added is still in,
    and the state the report was in all along.
    """

    store, reservation_id = _rejected_store(
        tmp_path,
        facts={"exception_type": "_CandidateStaticFailure"},
        record_authority=False,
    )
    try:
        assert _recovered(store, reservation_id) is None
    finally:
        store.close()


def test_a_quant_mismatch_recovers_from_the_target_members_alone(tmp_path):
    """The slot is always a member of the candidate's own target.

    That keeps the search space in the reservation row, so this reason needs no
    deployment either.
    """

    store, reservation_id = _rejected_store(
        tmp_path,
        reason="static_runtime_quant_mismatch",
        facts={"required_quant": "nvfp4", "slot": "moe.fused_experts"},
    )
    try:
        found = _recovered(
            store, reservation_id, slots=("moe.fused_experts", "attention.decode")
        )
    finally:
        store.close()

    assert found is not None
    assert found.reason == "static_runtime_quant_mismatch"
    assert dict(found.facts)["slot"] == "moe.fused_experts"


def test_a_wrong_authority_recovers_nothing_rather_than_the_wrong_reason(tmp_path):
    """The exact failure that shipped: a plausible digest that is not the one.

    Recovery is a preimage search, so a mismatched authority can only miss --
    it cannot produce a confident wrong answer.
    """

    store, reservation_id = _rejected_store(
        tmp_path, facts={"exception_type": "_CandidateStaticFailure"}
    )
    try:
        disposition = _screen_dispositions(store._db, reservation_id)[0]
        assert recover_screen_reason(
            stage="static",
            grade="fail",
            evidence_digest=disposition["stages"][0]["evidence_digest"],
            authority_digest="b" * 64,
            candidate_digest=CANDIDATE,
            publication_digest="9" * 64,
            service_digest=SERVICE,
            screen_attempt=1,
        ) is None
    finally:
        store.close()


def test_a_corrupt_authority_column_degrades_the_footnote_not_the_report(tmp_path):
    store, reservation_id = _rejected_store(tmp_path)
    try:
        store._db.execute(
            "UPDATE arena_screen_dispositions SET stage_authorities='{not json'"
        )
        disposition = _screen_dispositions(store._db, reservation_id)[0]
        assert disposition["stage_authorities"] == {}
        assert disposition["decision"] == "reject"
    finally:
        store.close()


def test_an_existing_database_gains_the_column_without_losing_its_rows(tmp_path):
    """Every receipt already on disk predates this column."""

    store, reservation_id = _rejected_store(tmp_path)
    path = store.path
    store._db.execute("ALTER TABLE arena_screen_dispositions DROP COLUMN "
                      "stage_authorities")
    store.close()

    reopened = FinalizedIntakeStore(path, scope=SCOPE)
    try:
        disposition = _screen_dispositions(reopened._db, reservation_id)[0]
        assert disposition["stage_authorities"] == {}
        assert disposition["receipt_digest"]
    finally:
        reopened.close()

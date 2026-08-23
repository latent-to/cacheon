"""A rejected bundle must be told why, from anywhere -- not only from the pod.

A screen stage used to hash its reason into the evidence digest and store
nothing else; the report then tried to rebuild the sentence by a preimage
search that needed the grading adapter's digest, which only the commissioned
deployment could produce. Off the pod it recovered nothing, and a FAIL whose
facts were not enumerable could never be recovered at all.

The reason now travels as a field of the signed stage result, so it is stored
by the same receipt write, re-verified by the same digest, and read back by the
same row reader -- with no second mechanism to drift.
"""

from __future__ import annotations

import pytest

from cacheon.arena_service import (
    ArenaScreenReceipt,
    ArenaServiceError,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain.intake import FinalizedArrival, FinalizedIntakeStore, IntakeScope
from cacheon.chain.miner_feedback import format_miner_submissions, miner_submissions
from cacheon.chain.operator_status import _screen_dispositions


SCOPE = IntakeScope("0x" + "0" * 64, 307)
BLOCK = 40
BLOCK_HASH = "0x" + "4" * 64

SERVICE = "1" * 64
CANDIDATE = "2" * 64
EVIDENCE = "3" * 64


def _rejected_store(tmp_path, *, reason: str) -> tuple[FinalizedIntakeStore, str]:
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
    store.apply_screen_receipt(
        arrival.reservation_id,
        candidate_digest=CANDIDATE,
        receipt=ArenaScreenReceipt(
            SERVICE,
            CANDIDATE,
            1,
            (ScreenStageResult("static", ScreenGrade.FAIL, EVIDENCE, 17, reason),),
            PromotionDecision.REJECT,
        ),
    )
    return store, arrival.reservation_id


def test_the_reason_survives_the_round_trip_through_storage(tmp_path):
    store, reservation_id = _rejected_store(
        tmp_path, reason="static_policy (_CandidateStaticFailure)"
    )
    try:
        stage = _screen_dispositions(store._db, reservation_id)[0]["stages"][0]
        path = store.path
    finally:
        store.close()

    assert stage == {
        "elapsed_ms": 17,
        "evidence_digest": EVIDENCE,
        "grade": "fail",
        "reason": "static_policy (_CandidateStaticFailure)",
        "stage": "static",
    }
    report = miner_submissions(path, hotkey="miner-0")
    assert report["submissions"][0]["screens"][0]["stages"][0]["reason"] == (
        "static_policy (_CandidateStaticFailure)"
    )
    text = format_miner_submissions(report)
    assert "screen[0] reject: failed at static" in text
    assert "    static: static_policy (_CandidateStaticFailure)" in text


def test_a_receipt_written_before_reasons_existed_keeps_its_digest(tmp_path):
    """Every receipt already on disk carries no reason and must verify as-is."""

    bare = ScreenStageResult("static", ScreenGrade.FAIL, EVIDENCE, 17)
    assert "reason" not in bare.to_dict()
    assert ScreenStageResult.from_dict(bare.to_dict()) == bare

    store, reservation_id = _rejected_store(tmp_path, reason="")
    try:
        disposition = _screen_dispositions(store._db, reservation_id)[0]
    finally:
        store.close()
    assert disposition["decision"] == "reject"
    assert "reason" not in disposition["stages"][0]


def test_the_reason_is_inside_the_signed_receipt(tmp_path):
    """Editing the stored reason changes the digest, so it cannot be rewritten."""

    stated = ScreenStageResult("static", ScreenGrade.FAIL, EVIDENCE, 17, "a")
    other = ScreenStageResult("static", ScreenGrade.FAIL, EVIDENCE, 17, "b")
    assert (
        ArenaScreenReceipt(SERVICE, CANDIDATE, 1, (stated,), PromotionDecision.REJECT).digest
        != ArenaScreenReceipt(SERVICE, CANDIDATE, 1, (other,), PromotionDecision.REJECT).digest
    )


@pytest.mark.parametrize(
    "reason",
    ["x" * 161, "non-ascii é", "newline\nreason", 7],
)
def test_an_unprintable_or_oversized_reason_is_refused(reason):
    with pytest.raises(ArenaServiceError):
        ScreenStageResult("static", ScreenGrade.FAIL, EVIDENCE, 17, reason)
    with pytest.raises(ArenaServiceError):
        ScreenStageResult.from_dict(
            {
                "elapsed_ms": 17,
                "evidence_digest": EVIDENCE,
                "grade": "fail",
                "reason": reason,
                "stage": "static",
            }
        )


def test_a_database_carrying_the_retired_authority_column_still_works(tmp_path):
    """Stores migrated by the previous deploy keep a ``stage_authorities``
    column nothing writes or reads any more; inserts and reports must not care."""

    store = FinalizedIntakeStore(tmp_path / "state" / "intake.sqlite3", scope=SCOPE)
    store._db.execute(
        "ALTER TABLE arena_screen_dispositions ADD COLUMN "
        "stage_authorities TEXT NOT NULL DEFAULT '{}'"
    )
    store.close()
    store, reservation_id = _rejected_store(tmp_path, reason="static_policy")
    try:
        disposition = _screen_dispositions(store._db, reservation_id)[0]
    finally:
        store.close()
    assert disposition["stages"][0]["reason"] == "static_policy"
    assert "stage_authorities" not in disposition

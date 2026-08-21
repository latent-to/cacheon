"""Closed-family intake disposal: parked without judgement, never charged.

A commissioned arena that cannot measure a registered family parks proposals
for it. The row leaves the queue as NO_DECISION -- never a FAIL that
byte-identical replay would echo after the family reopens -- and the cited
eval-cost payment pointer or admission credit is released in the same
transaction, so the closure costs the miner nothing.
"""

from __future__ import annotations

import pytest

from cacheon.chain.eval_cost_credit import grant_eval_cost_credit
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakeError,
    IntakePolicy,
    IntakeScope,
)

SCOPE = IntakeScope("0x" + "0" * 64, 307)
PAID = 25_000_000_000
TARGET = "attention.msa_prefill_block_score"


def _paid_arrival(digest: str, block: int) -> FinalizedArrival:
    return FinalizedArrival(
        hotkey="miner",
        content_hash=digest,
        url=f"https://example.com/{digest[:8]}.tar.gz",
        block=block,
        block_hash="0x" + f"{block:064x}",
        event_index=0,
        payment_block=8,
        payment_extrinsic_index=4,
    )


def _store(tmp_path) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    )


def _reserve(store, arrival: FinalizedArrival):
    return store.reserve_finalized(
        (arrival,),
        finalized_block=arrival.block,
        finalized_block_hash=arrival.block_hash,
        eval_cost_amount_tao_rao=PAID,
    )[0]


def test_disposal_parks_without_judgement_and_releases_the_payment(tmp_path) -> None:
    with _store(tmp_path) as store:
        first = _reserve(store, _paid_arrival("a" * 64, 10))
        assert first.status == "reserved"
        store.mark_fetching(first.reservation_id)

        parked = store.mark_target_unavailable(
            first.reservation_id, target_id=TARGET
        )
        assert parked.status == "expired"
        assert parked.decision == "NO_DECISION"
        assert parked.reason == f"target_unavailable:{TARGET}"
        assert (
            store._db.execute(
                "SELECT COUNT(*) FROM eval_cost_payments"
            ).fetchone()[0]
            == 0
        )

        # The released pointer admits a later submission with no
        # "eval_cost_payment_used" echo.
        retry = _reserve(store, _paid_arrival("b" * 64, 12))
        assert retry.status == "reserved"
        assert retry.reason == ""


def test_disposal_returns_an_admission_credit_unspent(tmp_path) -> None:
    with _store(tmp_path) as store:
        credit_id = grant_eval_cost_credit(
            store.path,
            hotkey="miner",
            coldkey="miner-cold",
            amount_tao_rao=PAID,
            note="closed-family regression fixture",
        )
        unpaid = FinalizedArrival(
            hotkey="miner",
            content_hash="c" * 64,
            url="https://example.com/c.tar.gz",
            block=10,
            block_hash="0x" + f"{10:064x}",
            event_index=0,
            invalid_reason="missing_eval_cost_payment",
        )
        first = _reserve(store, unpaid)
        assert first.status == "reserved"
        store.mark_fetching(first.reservation_id)
        store.mark_target_unavailable(first.reservation_id, target_id=TARGET)

        row = store._db.execute(
            "SELECT reservation_id, spent_block FROM eval_cost_credits "
            "WHERE credit_id=?",
            (credit_id,),
        ).fetchone()
        assert (row["reservation_id"], row["spent_block"]) == ("", 0)


def test_disposal_is_only_reachable_from_the_fetch_step(tmp_path) -> None:
    with _store(tmp_path) as store:
        first = _reserve(store, _paid_arrival("d" * 64, 10))
        with pytest.raises(IntakeError, match="forbidden"):
            store.mark_target_unavailable(first.reservation_id, target_id=TARGET)
        with pytest.raises(IntakeError, match="target id"):
            store.mark_target_unavailable(first.reservation_id, target_id="")

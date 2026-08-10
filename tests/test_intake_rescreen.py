"""A retired service identity must not wedge the qualification queue.

A promoted reservation carries a screen receipt bound to the service identity
that produced it. When the worker epoch rotates, that receipt can no longer
authorize qualification -- the dispatcher rejects the request as differing
provenance -- and because the qualification selector is deterministic, the same
unusable row is re-picked on every pass. Observed live on mainnet: the standing
supervisor crash-looped on one promoted row for over half an hour. Demotion
returns the row to the screen queue so a fresh receipt is produced under the
live identity, with no human in the loop.
"""

from __future__ import annotations

import pytest

from cacheon.chain.intake import IntakeError
from tests.test_chain_intake import (
    _arrival,
    _fingerprint,
    _promote,
    _store,
)

ROTATED = "screen_receipt_service_rotated"


def _promoted_row(store):
    row = store.reserve_finalized(
        (_arrival(0),),
        finalized_block=10,
        finalized_block_hash="0x" + f"{10:064x}",
    )[0]
    store.mark_fetching(row.reservation_id)
    store.mark_published(
        row.reservation_id,
        delta_fingerprint=_fingerprint("target", "target"),
        publication_digest="d" * 64,
        publication_root="/published/candidate",
    )
    _promote(store, row.reservation_id)
    return store.get(row.reservation_id)


def test_rotation_returns_the_row_to_the_screen_queue(tmp_path) -> None:
    with _store(tmp_path) as store:
        row = _promoted_row(store)
        assert row.status == "promoted" and row.screen_status == "promote"
        assert store.latest_promoted_screen(row.reservation_id) is not None

        back = store.demote_promoted_for_rescreen(row.reservation_id, reason=ROTATED)

        # 'published' is what the screen selector claims; 'promoted' is what the
        # qualification selector claims. The row must move between the two.
        assert back.status == "published"
        assert back.screen_status == ""
        assert back.reason == ROTATED
        with pytest.raises(IntakeError, match="no standing promoted screen"):
            store.latest_promoted_screen(row.reservation_id)


def test_a_rotated_row_recovers_without_a_human(tmp_path) -> None:
    with _store(tmp_path) as store:
        row = _promoted_row(store)
        store.demote_promoted_for_rescreen(row.reservation_id, reason=ROTATED)

        # Re-screening under the live identity restores qualification
        # eligibility; the campaign continues on its own.
        _promote(store, row.reservation_id)

        assert store.get(row.reservation_id).status == "promoted"
        assert store.latest_promoted_screen(row.reservation_id) is not None


def test_only_a_promoted_reservation_may_be_rescreened(tmp_path) -> None:
    with _store(tmp_path) as store:
        row = store.reserve_finalized(
            (_arrival(0),),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        with pytest.raises(IntakeError, match="only a promoted reservation"):
            store.demote_promoted_for_rescreen(row.reservation_id, reason=ROTATED)


def test_the_recorded_reason_is_bounded(tmp_path) -> None:
    with _store(tmp_path) as store:
        row = _promoted_row(store)
        for malformed in ("", "z" * 65):
            with pytest.raises(IntakeError, match="rescreen reason is malformed"):
                store.demote_promoted_for_rescreen(
                    row.reservation_id, reason=malformed
                )

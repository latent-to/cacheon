from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from dashboard.winners import (
    conservative_candidate_tokens_per_second,
    live_offer_shares,
    winner_reward,
    measured_baseline,
    prefill_summary,
    settlement_hold_notice,
    settlement_label,
)


def test_winners_view_separates_credit_from_measured_throughput() -> None:
    html = (
        Path(__file__).parents[1] / "dashboard" / "static" / "index.html"
    ).read_text()

    assert '"Gain vs baseline","Gain vs previous best","Measured baseline"' in html
    assert "baseline_tokens_per_second" in html
    assert "baseline_kind" in html
    assert "tokens_per_second" in html
    assert "sglang_tokens_per_second" not in html


def test_winner_keeps_measured_baseline_and_prefill_separate_from_credit() -> None:
    reads = [
        {"lanes": [{"role": "B", "tokens_per_second": 1200},
                   {"role": "B_prime", "tokens_per_second": 1190},
                   {"role": "C_prefill", "tokens_per_second": None}],
         "prefill": {"speedup": 1.12}},
        {"lanes": [], "prefill": {"speedup": 1.08}},
    ]
    assert measured_baseline(reads, {"incumbent_manifest": {"entries": {}}}) == {
        "baseline_tokens_per_second": 1190.0, "baseline_kind": "stock"}
    assert prefill_summary(reads) == {"prefill_speedup": 1.08}
    assert prefill_summary([]) == {"prefill_speedup": None}


def test_emission_columns_render_the_chain_alpha_symbol_not_tao() -> None:
    html = (
        Path(__file__).parents[1] / "dashboard" / "static" / "index.html"
    ).read_text()

    assert "emission_alpha_per_day" in html
    assert "emission_tao_per_day" not in html
    assert 'esc(d.emission_symbol) + "/day"' in html
    assert 'esc(d.emission_symbol) + "/d"' in html
    assert "τ/day" not in html
    assert "τ/d" not in html


def test_conservative_candidate_tokens_per_second_uses_slower_pass() -> None:
    speeds = [
        {"lanes": [{"role": "B", "tokens_per_second": 1900.0},
                   {"role": "C", "tokens_per_second": 2100.4}]},
        {"lanes": [{"role": "B", "tokens_per_second": 1935.2},
                   {"role": "C", "tokens_per_second": 2174.7}]},
    ]

    rate = conservative_candidate_tokens_per_second(speeds)

    assert rate == Decimal("2100.4")


def test_live_offer_shares_reads_the_served_vector(tmp_path: Path) -> None:
    offer = {
        "schema": "cacheon.current-weight-offer.v2",
        "offer": {
            "lane": "legacy_v1",
            "projection_digest": "ab" * 32,
            "projection": {
                "effective_block": 8994701,
                "crown_count": 4,
                "stack_generation": 2,
                "validator_hotkey": "5Vali",
                "weights_ppm": [["5Alice", 110958], ["5Bob", 889042]],
            },
        },
    }
    path = tmp_path / "current_weights.json"
    path.write_text(json.dumps(offer), encoding="utf-8")

    summary, shares = live_offer_shares(path)

    assert summary == {
        "lane": "legacy_v1",
        "projection_digest": "ab" * 32,
        "effective_block": 8994701,
        "crown_count": 4,
        "stack_generation": 2,
        "validator_hotkey": "5Vali",
    }
    assert shares == {"5Alice": Decimal("0.110958"), "5Bob": Decimal("0.889042")}
    assert sum(shares.values()) == Decimal(1)


def test_live_offer_shares_reports_absence_instead_of_a_vector(tmp_path: Path) -> None:
    assert live_offer_shares(tmp_path / "missing.json") == (None, {})
    broken = tmp_path / "broken.json"
    broken.write_text('{"offer": {"projection": {}}}', encoding="utf-8")
    assert live_offer_shares(broken) == (None, {})


def test_submission_snapshot_requires_the_exact_served_projection(tmp_path: Path) -> None:
    path, snapshot = tmp_path / "offer.json", tmp_path / "shares.json"
    path.write_text(json.dumps({"offer": {"projection_digest": "current",
        "projection": {"effective_block": 10, "weights_ppm": [["miner", 1_000_000]]}}}))
    report = {"projection_digest": "current", "submission_weights_ppm": {"a": 300_000, "b": 700_000}}
    snapshot.write_text(json.dumps(report))
    summary, shares = live_offer_shares(path, submission_path=snapshot)
    assert summary["submission_shares_available"] and shares == {"a": Decimal(".3"), "b": Decimal(".7")}
    assert live_offer_shares(path)[1] == {"miner": Decimal(1)}
    report["projection_digest"] = "another-offer"
    snapshot.write_text(json.dumps(report))
    summary, shares = live_offer_shares(path, submission_path=snapshot)
    assert not summary["submission_shares_available"] and not shares
    snapshot.unlink()
    assert live_offer_shares(path, submission_path=snapshot)[1] == {}


def test_winners_and_miners_render_the_served_weight_share() -> None:
    html = (
        Path(__file__).parents[1] / "dashboard" / "static" / "index.html"
    ).read_text()

    assert '"Submission weight share (served offer)"' in html
    assert '"Registered","Weight share","Subs"' in html
    assert "w.weight_share" in html and "m.weight_share" in html
    assert "Served weight offer" in html
    assert "follower journal" in html
    assert "Weight publications (latest)" not in html


def test_winner_share_uses_reservation_including_zero_and_missing_attribution() -> None:
    offer = {"submission_shares_available": True}
    shares = {"first": Decimal("0.3"), "second": Decimal("0.6")}
    row = {"reservation_id": "first", "hotkey": "same-miner", "waiting_for_queue": False}
    assert winner_reward(row, offer, shares) == {"weight_share": .3, "reward_claim_status": "earning"}
    row["reservation_id"] = "second"
    assert winner_reward(row, offer, shares)["weight_share"] == .6
    row["reservation_id"] = "unpaid"
    assert winner_reward(row, offer, shares) == {"weight_share": 0, "reward_claim_status": "not_earning"}
    assert winner_reward(row, {}, shares) == {"weight_share": None, "reward_claim_status": "attribution_unavailable"}
    row["waiting_for_queue"] = True
    assert winner_reward(row, offer, shares) == {"weight_share": None, "reward_claim_status": "waiting_for_queue"}



def test_a_stale_hold_reads_as_a_paid_pass() -> None:
    """2026-09-22: two paid Qwen passes showed "held" and the owner read that as unpaid."""

    class _Connection:
        def __init__(self, reason):
            self.reason = reason

        def execute(self, _query, _params):
            reason = self.reason

            class _Cursor:
                @staticmethod
                def fetchone():
                    return {"event_type": "HOLD", "event_json": json.dumps({"reason": reason}),
                            "sequence": 4}
            return _Cursor()

    # The candidates table only says "held"; the typed reason is in the journal.
    assert settlement_label(_Connection("stale_incumbent"), "a" * 64, "held", "held") == "passed"
    assert settlement_label(_Connection("conflict_lost"), "a" * 64, "held", "held") == "held"
    assert settlement_label(_Connection("stale_incumbent"), "a" * 64, "crowned", "crowned") == "crowned"

    notice = settlement_hold_notice(_Connection("stale_incumbent"), "a" * 64, {"status": "held", "reason": "held"})
    assert notice["title"] == "Passed evaluation — not the champion"
    assert "configured margin" in notice["message"]
    assert notice["reason"] == "stale_incumbent" and notice["event_sequence"] == 4
    assert settlement_hold_notice(_Connection("stale_incumbent"), "a" * 64, {"status": "crowned"}) is None

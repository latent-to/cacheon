from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from dashboard.winners import (
    conservative_candidate_tokens_per_second,
    live_offer_shares,
    measured_baseline,
    prefill_summary,
)


def test_winners_view_separates_credit_from_measured_throughput() -> None:
    html = (
        Path(__file__).parents[1] / "dashboard" / "static" / "index.html"
    ).read_text()

    assert '"Credited vs incumbent","Measured baseline"' in html
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


def test_winners_and_miners_render_the_served_weight_share() -> None:
    html = (
        Path(__file__).parents[1] / "dashboard" / "static" / "index.html"
    ).read_text()

    assert '"Weight share (served offer)"' in html
    assert '"Registered","Weight share","Subs"' in html
    assert "w.weight_share" in html and "m.weight_share" in html
    assert "Served weight offer" in html
    assert "follower journal" in html
    assert "Weight publications (latest)" not in html

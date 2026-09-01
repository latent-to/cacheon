from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from dashboard.winners import (
    conservative_candidate_tokens_per_second,
    cumulative_crown_speedups,
    estimated_sglang_tokens_per_second,
)


def _candidate(primary: str, reproduction: str) -> str:
    return json.dumps(
        {
            "primary": {"speedup": primary},
            "reproduction": {"speedup": reproduction},
        }
    )


def test_cumulative_winner_speedup_compounds_settled_gains_in_event_order() -> None:
    events = [
        {
            "sequence": 20,
            "reservation_id": "second",
            "target_id": "moe",
            "candidate_json": _candidate("1.20", "1.10"),
        },
        {
            "sequence": 10,
            "reservation_id": "first",
            "target_id": "moe",
            "candidate_json": _candidate("1.05", "1.04"),
        },
        {
            "sequence": 30,
            "reservation_id": "other-target",
            "target_id": "attention",
            "candidate_json": _candidate("1.03", "1.02"),
        },
    ]

    result = cumulative_crown_speedups(events)

    assert result == {
        "first": Decimal("1.04"),
        "second": Decimal("1.1440"),
        "other-target": Decimal("1.02"),
    }


def test_cumulative_winner_speedup_ignores_incomplete_non_crown_data() -> None:
    result = cumulative_crown_speedups(
        [
            {
                "sequence": 1,
                "reservation_id": "missing-reproduction",
                "target_id": "moe",
                "candidate_json": json.dumps(
                    {"primary": {"speedup": "1.2"}, "reproduction": {}}
                ),
            },
            {
                "sequence": 2,
                "reservation_id": "valid",
                "target_id": "moe",
                "candidate_json": _candidate("1.03", "1.02"),
            },
        ]
    )

    assert result == {"valid": Decimal("1.02")}


def test_winners_view_labels_relative_and_sglang_columns() -> None:
    html = (
        Path(__file__).parents[1] / "dashboard" / "static" / "index.html"
    ).read_text()

    assert '"Vs incumbent","Vs SGLang"' in html
    assert "cumulative_improvement_pct_over_sglang" in html
    assert "cumulative_speedup_over_sglang" in html
    assert "tokens_per_second" in html
    assert "sglang_tokens_per_second" in html


def test_conservative_candidate_tokens_per_second_uses_slower_pass() -> None:
    speeds = [
        {"lanes": [{"role": "B", "tokens_per_second": 1900.0},
                   {"role": "C", "tokens_per_second": 2100.4}]},
        {"lanes": [{"role": "B", "tokens_per_second": 1935.2},
                   {"role": "C", "tokens_per_second": 2174.7}]},
    ]

    rate = conservative_candidate_tokens_per_second(speeds)

    assert rate == Decimal("2100.4")
    assert estimated_sglang_tokens_per_second(rate, Decimal("1.23476")) == (
        Decimal("2100.4") / Decimal("1.23476")
    )

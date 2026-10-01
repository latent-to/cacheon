from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("fastapi")

from dashboard.app import submission_baseline  # noqa: E402
from dashboard.winners import reward_bars, reward_exclusion_notice  # noqa: E402


TARGET = "moe.fused_experts"


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE settlement_candidates(
            reservation_id TEXT PRIMARY KEY,
            candidate_json TEXT NOT NULL
        );
        CREATE TABLE settlement_qualifications(
            reservation_id TEXT NOT NULL,
            reproduction_index INTEGER NOT NULL,
            qualification_json TEXT NOT NULL
        );
        CREATE TABLE reservation_baseline_segments(
            reservation_id TEXT PRIMARY KEY,
            arena_id TEXT NOT NULL,
            stack_digest TEXT NOT NULL,
            tree_digest TEXT NOT NULL,
            stack_json TEXT NOT NULL
        );
        CREATE TABLE target_lineage_nodes(
            target_id TEXT NOT NULL,
            artifact_digest TEXT NOT NULL,
            parent_artifact_digest TEXT NOT NULL,
            winner_speedup TEXT NOT NULL,
            transition_event_id TEXT NOT NULL
        );
        CREATE TABLE settlement_events(
            event_id TEXT PRIMARY KEY,
            reservation_id TEXT NOT NULL
        );
        """
    )
    con.executemany(
        "INSERT INTO target_lineage_nodes VALUES(?,?,?,?,?)",
        (
            (TARGET, "B", "A", "1.1", "crown-b"),
            (TARGET, "C", "B", "1.1", "crown-c"),
        ),
    )
    con.executemany("INSERT INTO settlement_events VALUES(?,?)",
                    (("crown-b", "submission-b"), ("crown-c", "submission-c")))
    return con


def _candidate(con: sqlite3.Connection, reservation: str, baseline: str) -> None:
    doc = {
        "primary": {
            "arena_digest": "arena",
            "incumbent_stack_digest": "stack",
            "incumbent_tree_digest": "tree",
            "incumbent_manifest": {
                "arena_digest": "arena",
                "entries": (
                    {}
                    if not baseline
                    else {TARGET: {"artifact_digest": baseline}}
                ),
            },
        }
    }
    con.execute(
        "INSERT INTO settlement_candidates VALUES(?,?)",
        (reservation, json.dumps(doc)),
    )


BEST = {("arena", "stack"): {"speedup": 1.21, "reservation_id": "best"}}


def test_submission_baseline_reports_the_best_retained_pass_on_its_own_baseline() -> None:
    con = _db()
    _candidate(con, "uncle", "A")

    baseline = submission_baseline(con, "uncle", TARGET, bars=BEST)

    assert baseline["reward_bar"] == {"speedup": 1.21, "reservation_id": "best"}
    assert (baseline["evaluated"], baseline["assigned"], baseline["artifact_digest"]) == (True, False, "A")
    assert (baseline["arena_digest"], baseline["stack_digest"], baseline["tree_digest"]) == ("arena", "stack", "tree")
    assert baseline["reservation_id"] is None
    # The crown lineage (B then C, 1.21x over A) no longer sets any bar: pay never follows it.
    assert "threshold_speedup" not in baseline and "relationship" not in baseline
    assert submission_baseline(con, "uncle", TARGET, bars={("arena", "other"): BEST["arena", "stack"]})["reward_bar"] is None


def test_baseline_link_uses_the_evaluated_artifact() -> None:
    con = _db()
    _candidate(con, "candidate", "B")
    assert submission_baseline(con, "candidate", TARGET, bars={})["reservation_id"] == "submission-b"


def test_baseline_link_stays_in_submission_arena() -> None:
    con = _db()
    con.executescript("""
        ALTER TABLE target_lineage_nodes ADD COLUMN competition_arena TEXT DEFAULT 'one';
        CREATE TABLE reservations(reservation_id TEXT, competition_arena TEXT);
        INSERT INTO reservations VALUES('candidate', 'two');
        INSERT INTO target_lineage_nodes VALUES('moe.fused_experts','B','A','1.1','crown-two','two');
        INSERT INTO settlement_events VALUES('crown-two','submission-two');
    """)
    _candidate(con, "candidate", "B")
    assert submission_baseline(con, "candidate", TARGET, bars={})["reservation_id"] == "submission-two"


def test_base_engine_and_unmeasured_submissions_have_no_baseline_link() -> None:
    con = _db()
    _candidate(con, "stock", "")
    stock = submission_baseline(con, "stock", TARGET, bars={})
    assert (stock["kind"], stock["reservation_id"], stock["reward_bar"]) == ("stock", None, None)
    assert submission_baseline(con, "new", TARGET, bars=BEST) == {
        "evaluated": False, "assigned": False, "artifact_digest": "", "reward_bar": None}


def test_submission_baseline_shows_queued_assignment_before_measurement() -> None:
    con = _db()
    manifest = {"arena_digest": "arena", "entries": {TARGET: {"artifact_digest": "A"}}}
    con.execute("INSERT INTO reservation_baseline_segments VALUES(?,?,?,?,?)",
                ("queued", "arena", "stack", "tree", json.dumps(manifest)))

    baseline = submission_baseline(con, "queued", TARGET, bars=BEST)

    assert (baseline["assigned"], baseline["evaluated"], baseline["kind"]) == (True, False, "incumbent")
    assert baseline["reward_bar"]["reservation_id"] == "best"


def test_reward_bars_keep_the_best_eligible_pass_per_baseline(monkeypatch) -> None:
    from decimal import Decimal

    con = _db()
    for reservation in ("first", "better", "unpaid"):
        _candidate(con, reservation, "A")
    row = {"previous_best_speedup": Decimal("1.02"), "relative_speedup": Decimal("1.03"), "reward_eligible": True}
    monkeypatch.setattr("dashboard.winners.reward_comparisons", lambda con: {
        "first": row | {"previous_best_speedup": Decimal(1), "relative_speedup": Decimal("1.02")},
        "better": row, "unpaid": row | {"relative_speedup": Decimal("1.2"), "reward_eligible": False}})
    assert reward_bars(con) == {("arena", "stack"): {"speedup": pytest.approx(1.0506), "reservation_id": "better"}}
    assert submission_baseline(con, "unpaid", TARGET)["reward_bar"]["reservation_id"] == "better"


def test_operator_notice_names_a_zero_priced_pass_and_an_unpaid_excluded_hotkey(tmp_path, monkeypatch) -> None:
    offer, rule = tmp_path / "offer.json", tmp_path / "exclusions.json"
    offer.write_text(json.dumps({"offer": {"projection": {"effective_block": 7, "weights_ppm": [["paid", 5]]}}}))
    rule.write_text(json.dumps({
        "claims": [{"reservation_id": "zeroed", "reason": "grader defect, not miner misconduct"}],
        "records": [{"hotkey": "copier", "evidence": "same bytes"}, {"hotkey": "paid"}]}))
    assert reward_exclusion_notice("copier", offer, "zeroed") is None  # no rule file is configured
    monkeypatch.setenv("CACHEON_DASH_EXCLUSIONS", str(rule))
    claim = reward_exclusion_notice("anyone", offer, "zeroed")
    assert (claim["reason"], claim["message"], claim["offer_block"]) == (
        "operator_claim_exclusion", "grader defect, not miner misconduct", 7)
    assert reward_exclusion_notice("copier", offer, "other")["evidence"] == "same bytes"
    assert reward_exclusion_notice("paid", offer, "other") is None
    assert reward_exclusion_notice("stranger", offer, "other") is None

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("fastapi")

from dashboard.app import submission_baseline  # noqa: E402


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
        CREATE TABLE target_lineage_tips(
            target_id TEXT PRIMARY KEY,
            artifact_digest TEXT NOT NULL
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
    con.execute("INSERT INTO target_lineage_tips VALUES(?,?)", (TARGET, "C"))
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


def test_submission_baseline_shows_composed_ancestor_threshold() -> None:
    con = _db()
    _candidate(con, "uncle", "A")

    baseline = submission_baseline(con, "uncle", TARGET)

    assert baseline["relationship"] == "ancestor"
    assert baseline["evaluated"] is True
    assert baseline["assigned"] is False
    assert baseline["artifact_digest"] == "A"
    assert baseline["current_tip_artifact_digest"] == "C"
    assert baseline["threshold_speedup"] == pytest.approx(1.21)
    assert baseline["stack_digest"] == "stack"
    assert baseline["tree_digest"] == "tree"
    assert baseline["reservation_id"] is None


def test_baseline_link_uses_evaluated_artifact_not_current_tip() -> None:
    con = _db()
    _candidate(con, "candidate", "B")
    assert submission_baseline(con, "candidate", TARGET)["reservation_id"] == "submission-b"
    con.execute("DELETE FROM target_lineage_tips")
    assert submission_baseline(con, "candidate", TARGET)["reservation_id"] == "submission-b"


def test_baseline_link_stays_in_submission_arena() -> None:
    con = _db()
    con.executescript("""
        ALTER TABLE target_lineage_tips ADD COLUMN competition_arena TEXT DEFAULT 'one';
        ALTER TABLE target_lineage_nodes ADD COLUMN competition_arena TEXT DEFAULT 'one';
        CREATE TABLE reservations(reservation_id TEXT, competition_arena TEXT);
        INSERT INTO reservations VALUES('candidate', 'two');
        INSERT INTO target_lineage_nodes VALUES('moe.fused_experts','B','A','1.1','crown-two','two');
        INSERT INTO settlement_events VALUES('crown-two','submission-two');
    """)
    _candidate(con, "candidate", "B")
    assert submission_baseline(con, "candidate", TARGET)["reservation_id"] == "submission-two"


def test_base_engine_has_no_submission_link() -> None:
    con = _db()
    _candidate(con, "stock", "")
    assert submission_baseline(con, "stock", TARGET)["reservation_id"] is None


def test_submission_baseline_distinguishes_tip_side_branch_and_unmeasured() -> None:
    con = _db()
    _candidate(con, "current", "C")
    _candidate(con, "side", "X")

    assert submission_baseline(con, "current", TARGET)["relationship"] == "current_tip"
    assert (
        submission_baseline(con, "side", TARGET)["relationship"]
        == "outside_active_lineage"
    )
    assert submission_baseline(con, "new", TARGET) == {
        "evaluated": False,
        "assigned": False,
        "relationship": "not_evaluated",
        "artifact_digest": "",
        "current_tip_artifact_digest": "",
        "threshold_speedup": None,
    }


def test_submission_baseline_shows_queued_assignment_before_measurement() -> None:
    con = _db()
    manifest = {
        "arena_digest": "arena",
        "entries": {TARGET: {"artifact_digest": "A"}},
    }
    con.execute(
        "INSERT INTO reservation_baseline_segments VALUES(?,?,?,?,?)",
        ("queued", "arena", "stack", "tree", json.dumps(manifest)),
    )

    baseline = submission_baseline(con, "queued", TARGET)

    assert baseline["assigned"] is True
    assert baseline["evaluated"] is False
    assert baseline["relationship"] == "ancestor"
    assert baseline["threshold_speedup"] == pytest.approx(1.21)

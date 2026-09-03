from __future__ import annotations

import json
import sqlite3

import pytest

from dashboard.app import submission_baseline


TARGET = "moe.fused_experts_reduce"


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
            winner_speedup TEXT NOT NULL
        );
        """
    )
    con.executemany(
        "INSERT INTO target_lineage_nodes VALUES(?,?,?,?)",
        (
            (TARGET, "B", "A", "1.1"),
            (TARGET, "C", "B", "1.1"),
        ),
    )
    con.execute("INSERT INTO target_lineage_tips VALUES(?,?)", (TARGET, "C"))
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

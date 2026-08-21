"""The evaluation report regrades from verified artifact bytes, never prose."""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluation_report  # noqa: E402


def _policy_block() -> dict:
    # Float thresholds arrive as decimal strings, as real witnesses store them.
    return {
        "max_stage_seconds": 3600,
        "min_margin": "0.005",
        "noise_multiplier": "2",
        "max_noise": "0.02",
        "calibration_digest": "a" * 64,
        "calibration_context_digest": "b" * 64,
        "version": 8,
        "max_qualification_seconds": 7200,
        "min_windows": 6,
        "max_window_scatter": "0.05",
        "max_conditioning_slowdown": "1.25",
    }


def _read(role: str, seconds: float, lane: str) -> dict:
    return {
        "role": role,
        "lane_digest": lane,
        "windows": [{"tokens": 131072, "seconds": str(seconds + i * 0.01)} for i in range(6)],
        "conditioning_tokens": 131072,
        "conditioning_seconds": seconds,
        "timed_tokens": 786432,
        "timed_seconds": seconds * 6,
    }


BASELINE_LANE = "c" * 64
CANDIDATE_LANE = "d" * 64


def _stage_exit(candidate_seconds: float) -> bytes:
    # Resident bracket protocol: B and B_prime are the BASELINE bookends,
    # C is the CANDIDATE. Lane digests carry the ground truth.
    return json.dumps(
        {
            "decision": "FAIL",
            "reason": "speed_threshold_not_met",
            "stage": "speed",
            "speed_witness": {
                "resident_policy": _policy_block(),
                "baseline_lane_digest": BASELINE_LANE,
                "candidate_lane_digest": CANDIDATE_LANE,
                "rates": [
                    _read("B", 72.0, BASELINE_LANE),
                    _read("C", candidate_seconds, CANDIDATE_LANE),
                    _read("B_prime", 72.0, BASELINE_LANE),
                ],
            },
        }
    ).encode()


def _fixture(tmp_path: Path, artifact: bytes) -> tuple[Path, Path, str]:
    sha = hashlib.sha256(artifact).hexdigest()
    cas = tmp_path / "evidence" / evaluation_report.STAGE_EXIT_DOMAIN / sha[:2]
    cas.mkdir(parents=True)
    (cas / sha).write_bytes(artifact)
    db = tmp_path / "intake.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        create table reservations (
            reservation_id text, hotkey text, target_id text, block integer,
            arena_service_digest text, decision text, reason text);
        create table qualification_dispositions (
            reservation_id text, attempt_index integer, decision text,
            reason text, attempt_ref_json text);
        """
    )
    connection.execute(
        "insert into reservations values ('r1'||zeroblob(0), 'hk1', 't.x', 5, 'arena1', 'FAIL', 'speed_threshold_not_met')"
    )
    connection.execute(
        "insert into qualification_dispositions values ('r1', 0, 'FAIL', 'speed_threshold_not_met', ?)",
        (
            json.dumps(
                {"domain": evaluation_report.STAGE_EXIT_DOMAIN, "sha256": sha, "size": len(artifact)}
            ),
        ),
    )
    connection.commit()
    connection.close()
    return db, tmp_path / "evidence", sha


def test_report_regrades_with_the_deployed_grader(tmp_path: Path) -> None:
    db, root, _ = _fixture(tmp_path, _stage_exit(candidate_seconds=72.05))
    rows = evaluation_report.collect_rows(db, [root], "arena1")
    assert len(rows) == 1
    regrade = rows[0]["regrade"]
    assert regrade is not None, rows[0]["regrade_error"]
    assert regrade["grader_decision"]
    assert regrade["grader_required"] > 1.0
    assert regrade["windows"] == {"B": 6, "C": 6, "B_prime": 6}
    report = evaluation_report.render(rows, db_path=str(db), current_arena="arena1")
    assert f"{regrade['grader_speedup']:.5f}" in report
    assert "current contract" in report


def test_arm_orientation_follows_lane_digests_not_role_letters(tmp_path: Path) -> None:
    # The candidate (role C) is deliberately SLOWER (73.0s vs 72.0s baseline):
    # the grader's speedup must come out below one. A reversed arm mapping
    # would report ~1.014 instead — the exact defect this test pins.
    db, root, _ = _fixture(tmp_path, _stage_exit(candidate_seconds=73.0))
    rows = evaluation_report.collect_rows(db, [root], "arena1")
    regrade = rows[0]["regrade"]
    assert regrade is not None, rows[0]["regrade_error"]
    assert regrade["candidate_roles"] == ("C",)
    assert set(regrade["baseline_roles"]) == {"B", "B_prime"}
    assert regrade["grader_speedup"] < 1.0
    assert regrade["medians"]["C"] < regrade["medians"]["B"]


def test_tampered_artifact_bytes_are_refused(tmp_path: Path) -> None:
    artifact = _stage_exit(candidate_seconds=72.05)
    db, root, sha = _fixture(tmp_path, artifact)
    path = root / evaluation_report.STAGE_EXIT_DOMAIN / sha[:2] / sha
    path.write_bytes(artifact + b" ")
    rows = evaluation_report.collect_rows(db, [root], "arena1")
    assert rows[0]["regrade"] is None
    assert "differ" in rows[0]["regrade_error"]


def test_missing_evidence_is_reported_not_blank(tmp_path: Path) -> None:
    db, root, sha = _fixture(tmp_path, _stage_exit(candidate_seconds=72.05))
    (root / evaluation_report.STAGE_EXIT_DOMAIN / sha[:2] / sha).unlink()
    rows = evaluation_report.collect_rows(db, [root], "arena1")
    assert rows[0]["regrade_error"] == "evidence artifact not found"
    report = evaluation_report.render(rows, db_path=str(db), current_arena="arena1")
    assert "evidence artifact not found" in report


def test_superseded_arena_rows_are_tainted_not_condemned(tmp_path: Path) -> None:
    db, root, _ = _fixture(tmp_path, _stage_exit(candidate_seconds=72.05))
    connection = sqlite3.connect(db)
    connection.execute(
        "insert into reservations values ('r2', 'hk2',"
        " 'attention.msa_prefill_block_score', 4, 'oldarena', 'FAIL', 'speed_regression')"
    )
    connection.commit()
    connection.close()
    rows = evaluation_report.collect_rows(db, [root], "arena1")
    tainted = [r for r in rows if r["reservation"] == "r2"]
    assert tainted and tainted[0]["validity"].startswith("tainted:")
    assert "measured path" in tainted[0]["validity"]


def test_infra_reasons_are_not_kernel_verdicts(tmp_path: Path) -> None:
    db, root, _ = _fixture(tmp_path, _stage_exit(candidate_seconds=72.05))
    connection = sqlite3.connect(db)
    connection.execute(
        "insert into reservations values ('r3', 'hk3', 't.y', 3,"
        " 'arena1', 'NO_DECISION', 'finalized_block_sla_expired')"
    )
    connection.commit()
    connection.close()
    rows = evaluation_report.collect_rows(db, [root], "arena1")
    infra = [r for r in rows if r["reservation"] == "r3"]
    assert infra[0]["validity"] == "infrastructure — not a kernel verdict"


def test_policy_block_missing_fields_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy block lacks"):
        evaluation_report._policy_from_block({"version": 8})

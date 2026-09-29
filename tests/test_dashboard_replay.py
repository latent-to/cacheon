"""Replay reporting uses the retained scorer and never labels latency as token throughput."""

import json
import sqlite3
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cacheon.chain.baseline_band import qualification_speed_from_payload, retained_half_rates
from cacheon.eval.goodput_runtime import GoodputPolicy, GoodputReadSet
from cacheon.eval.qualification_runner import ResidentSpeedWitness, _resident_speed_projection_digest
from cacheon.eval.resident_speed_policy import ResidentSpeedPolicy
from cacheon.eval.service_capacity import LoadRead, ServiceContract
from dashboard.winners import candidate_measurement, measured_baseline
from tests.test_dashboard_metrics import _dashboard_db, client  # noqa: F401
from tests.test_service_capacity import _turn


def _witness(load=2, gains=(1.02, 1.04), *, statistical=False):
    contract = ServiceContract(.1, 5.0, .8)
    policy = ResidentSpeedPolicy(600, .01, 2., .02, "a" * 64, "b" * 64, 16,
                                goodput=GoodputPolicy(contract, 1.01, .001, .02, .0))
    if statistical:
        policy = replace(policy, version=17, min_margin=0,
                         goodput=GoodputPolicy(contract, 1., .001, .02, 0., .01, .001))
    expected = tuple((str(root), 3, 0) for root in sorted(range(load), key=str))
    arms = []
    for arm in ("incumbent", "candidate"):
        windows = []
        for window, (latency, gain) in enumerate(zip((4., 1.), gains), 1):
            rows = []
            for root, _, _ in expected:
                for turn in range(3):
                    row = _turn(root, "main", turn, start_s=100 * window + turn * 10,
                                ttft_s=.1, out=4)
                    seconds = 50. if turn == 0 else latency / (gain if arm == "candidate" else 1)
                    rows.append(replace(row, request_end_ns=row.credit_issued_ns + int(seconds * 1e9)))
            lane = ("candidate" if arm == "incumbent" else "incumbent") if statistical and window == 2 else arm
            windows.append(LoadRead(arm, window, lane, load, tuple(rows)))
        arms.append(tuple(windows))
    reads = GoodputReadSet(*arms, expected)
    excluded = {"resident_policy", "rates", "goodput", "started_monotonic_s",
                "completed_monotonic_s", "evidence_digest"}
    fields = {name: f"{index + 1:064x}" for index, name in enumerate(ResidentSpeedWitness.__dataclass_fields__)
              if name not in excluded}
    fields.update(resident_policy=policy, rates=(), goodput=reads, started_monotonic_s=1.,
                  completed_monotonic_s=301., calibration_digest=policy.calibration_digest,
                  calibration_context_digest=policy.calibration_context_digest)
    return ResidentSpeedWitness(**fields, evidence_digest=_resident_speed_projection_digest(**fields))


@pytest.mark.parametrize("load,gains,decision", [(2, (1.02, 1.04), "PASS"), (24, (1., 1.), "FAIL")])
def test_replay_report_uses_fastest_pass_score_and_preserves_workload(load, gains, decision):
    witness = _witness(load, gains)
    payload = json.dumps({"speed_witness": witness.to_dict()}).encode()
    speed = qualification_speed_from_payload(payload)
    assert "grading_error" not in speed
    assert speed["speedup"] == pytest.approx(1 / min(4 / gains[0], 1 / gains[1]))
    assert speed["grading"]["decision"] == decision
    assert speed["workload_digest"] == witness.workload_digest
    assert (speed["load"], speed["windows"]) == (load, 2)
    assert [(row["role"], row["window"]) for row in speed["lanes"]] == [("B", 1), ("B", 2), ("C", 1), ("C", 2)]
    assert all(row["warm_turns"] == load * 2 for row in speed["lanes"])
    assert speed["lanes"][0]["mean_warm_latency_s"] == 4.
    assert speed["lanes"][0]["attainment"] == pytest.approx(2 / 3)
    candidate = candidate_measurement([speed])
    baseline = measured_baseline([speed], {})
    assert candidate["tokens_per_second"] is None
    assert baseline["baseline_tokens_per_second"] is None
    assert baseline["baseline_mean_warm_latency_s"] == 1.
    assert baseline["baseline_mean_warm_latency_s"] / candidate["mean_warm_latency_s"] == pytest.approx(speed["speedup"])
    assert (speed["policy_version"], speed["score_basis"], speed["speed_stage_seconds"]) == (16, "fastest_pass_latency", 300.)
    assert [row["used_for_score"] for row in speed["lanes"]] == [False, True, False, True]
    assert speed["lanes"][0]["mean_ttft_s"] == pytest.approx(.1)
    assert speed["lanes"][0]["p95_ttft_s"] == pytest.approx(.1)
    assert speed["lanes"][0]["median_decode_tps"] == pytest.approx(3 / 3.9)
    assert all(row["unsuccessful_turns"] == 0 for row in speed["lanes"])


def test_statistical_replay_reports_its_elapsed_basis_without_fastest_pass_flags():
    speed = qualification_speed_from_payload(json.dumps({"speed_witness": _witness(statistical=True).to_dict()}).encode())
    assert "grading_error" not in speed
    assert speed["policy_version"] == 17 and speed["score_basis"] == "pooled_elapsed_orientation"
    assert all("used_for_score" not in row for row in speed["lanes"])
    assert speed["metric"] == "fixed_work_rate"
    assert speed["lanes"][0]["elapsed_s"] == 14
    assert speed["lanes"][0]["turns_per_second"] == pytest.approx(4 / 14)
    assert speed["grading"]["standard_error"] > 0
    assert speed["grading"]["lower_speedup"] == pytest.approx(speed["speedup"] / speed["grading"]["required_speedup"])
    baseline = measured_baseline([speed], {})["baseline_replay_turns_per_second"]
    candidate = candidate_measurement([speed])["replay_turns_per_second"]
    assert candidate / baseline == pytest.approx(speed["speedup"])


def test_corrupt_replay_witness_reports_its_original_error():
    wire = _witness().to_dict()
    wire["evidence_digest"] = "f" * 64
    speed = qualification_speed_from_payload(json.dumps({"speed_witness": wire}).encode())
    assert speed["lanes"] == []
    assert "digest does not recompute" in speed["grading_error"]
    assert "speedup" not in speed


def test_replay_measurements_reach_submission_and_winner_apis(tmp_path, client, monkeypatch):  # noqa: F811
    from cacheon.eval.evidence_store import prepare_evidence_root, publish_canonical_json_evidence

    target = "attention.indexer_select"
    monkeypatch.setattr("dashboard.winners.reward_comparisons", lambda con: {"example": {
        "previous_best_reservation_id": "earlier", "previous_best_speedup": Decimal("1.01"),
        "relative_speedup": Decimal("1.02"), "score_speedup": Decimal("1.02"),
        "reward_eligible": True, "grandfathered": False}})
    root = tmp_path / "retained"
    ref = publish_canonical_json_evidence(prepare_evidence_root(root),
        {"reports": [{"target_id": target, "speed_witness": _witness().to_dict()}]},
        domain="qualification.stage-exit", schema="cacheon.qualification.stage-exit.v1")
    db = tmp_path / "intake.sqlite3"
    _dashboard_db(db, json.dumps(ref.to_dict()), root, target)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE reservation_baseline_segments "
                    "(reservation_id TEXT,arena_id TEXT,stack_digest TEXT,tree_digest TEXT,stack_json TEXT)")
        con.execute("INSERT INTO reservation_baseline_segments VALUES(?,?,?,?,?)", (
            "example", "arena", "stack", "tree", json.dumps({"entries": {target: {"artifact_digest": "incumbent"}}})))
    store = SimpleNamespace(retained_pass_pairs=lambda: [
        ("example", "a" * 64, ["1.03"], [(0, json.dumps(ref.to_dict()))])])
    assert retained_half_rates(store, (root,)) == ()
    detail = client.get("/api/submissions/example").json()
    assert detail["tokens_per_second"] is None
    assert detail["mean_warm_latency_s"] == pytest.approx(1 / 1.04)
    assert detail["baseline_measurements"]["baseline_mean_warm_latency_s"] == 1.
    assert detail["baseline_measurements"]["baseline_kind"] == "incumbent"
    assert detail["qualification_attempts"][0]["speed"]["grading"]["decision"] == "PASS"
    with sqlite3.connect(db) as con:
        con.execute("UPDATE reservations SET status='qualified', decision='PASS'")
        con.execute("UPDATE qualification_dispositions SET decision='PASS'")
        con.execute("UPDATE settlement_qualifications SET reservation_id='example'")
        primary = {"speedup": "1.03", "target_id": target, "incumbent_manifest": {"entries": {}}}
        con.execute("INSERT INTO settlement_candidates VALUES(?,?,?,?)", (
            "example", "crowned", "", json.dumps({"primary": primary})))
        con.execute("INSERT INTO settlement_events VALUES(?,?,?,?)", (1, "CROWN", "example", target))
    winner = client.get("/api/winners").json()["items"][0]
    assert winner["tokens_per_second"] is None
    assert winner["mean_warm_latency_s"] == detail["mean_warm_latency_s"]
    assert winner["baseline_mean_warm_latency_s"] == 1.
    assert winner["speedup"] == 1.03
    assert client.get("/static/performance.js").status_code == 200

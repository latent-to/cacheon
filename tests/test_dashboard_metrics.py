"""Dashboard metrics retain workload boundaries and the correct prompt-pass units."""

import base64
import hashlib
import json
import sqlite3
import pytest

from cacheon.chain.baseline_band import qualification_evidence_roots, qualification_speed
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef, prepare_evidence_root, publish_canonical_json_evidence, reopen_evidence,
)
from cacheon.eval.resident_measurement import TimedWindow


def _rate(role, seconds, tokens, *, phase=False):
    windows = [TimedWindow(i, n, float(s)).to_dict()
               for i, (s, n) in enumerate(zip(seconds, tokens))]
    if phase:
        windows = [
            TimedWindow(0, 8, 2.0, 1024, ((.25, 1.25), (.75, 1.75))).to_dict(),
            TimedWindow(1, 32, 4.0, 2048, ((1., 3.),) * 4).to_dict(),
        ]
    return {
        "role": role,
        "timed_tokens": sum(w["tokens"] for w in windows),
        "timed_seconds": str(sum(float(w["seconds"]) for w in windows)),
        "conditioning_seconds": "1.0",
        "windows": windows,
        # The display must derive metrics from retained host timings.
        "cells": [{"mean_ttft_seconds": "999999"}],
    }


def _publish(root, rates, target="norm.fused_add_rmsnorm", *, reports=False):
    report = {"target_id": target, "speed_witness": {
        "resident_policy": {"version": 12, "prefill_min_margin": "0.05"},
        "rates": rates}}
    payload = {"reports": [report]} if reports else report
    reference = publish_canonical_json_evidence(
        prepare_evidence_root(root), payload, domain="qualification.stage-exit",
        schema="cacheon.qualification.stage-exit.v1")
    return json.dumps(reference.to_dict())


def _reads(phase=False):
    return [
        _rate("B", [2., 4.], [8, 32], phase=phase),
        _rate("C", [2., 4.], [8, 32], phase=phase),
        _rate("B_prime", [2., 4.], [8, 32], phase=phase),
        _rate("B_prefill", [20., 20., 40., 40., 40.], [128, 128, 24, 24, 24]),
        _rate("C_prefill", [19.88, 19.88, 39.76, 39.76, 39.76], [128, 128, 24, 24, 24]),
        _rate("B_prime_prefill", [20., 20., 40., 40., 40.], [128, 128, 24, 24, 24]),
    ]


def test_prefill_uses_prompts_per_second_without_rounding_away_the_gain(tmp_path):
    ref = _publish(tmp_path, _reads())
    speed = qualification_speed(ref, (tmp_path,))
    prefill = speed["lanes"][4]
    assert prefill["tokens_per_second"] is None
    assert prefill["prompts_per_second"] == pytest.approx(328 / 159.04, abs=1e-6)
    assert prefill["timed_seconds"] == 159.04
    assert prefill["cells"] == []
    assert speed["prefill"]["speedup"] == pytest.approx(160 / 159.04)
    assert speed["prefill"]["min_margin"] == "0.05"
    assert all(lane["cells"] == [] for lane in speed["lanes"])


def test_latency_cells_are_recomputed_and_keep_two_workloads_separate(tmp_path):
    speed = qualification_speed(_publish(tmp_path, _reads(phase=True)), (tmp_path,))
    cells = speed["lanes"][1]["cells"]
    assert [(c["input_tokens"], c["output_tokens"], c["concurrency"]) for c in cells] == [
        (1024, 4, 2), (2048, 8, 4)]
    assert [float(c["mean_ttft_seconds"]) for c in cells] == [.5, 1.]
    assert [float(c["mean_tpot_seconds"]) for c in cells] == pytest.approx([1 / 3, 2 / 7])
    assert [float(c["end_to_end_output_tokens_per_second"]) for c in cells] == [4., 8.]
    assert speed["lanes"][4]["cells"] == []


def test_rotated_evidence_and_target_selection_preserve_historical_reads(tmp_path):
    state = tmp_path / "state"
    recorded = tmp_path / "retained"
    staged = tmp_path / "stage" / "rotation" / "monday-config" / "qualification-evidence"
    staged.mkdir(parents=True)
    ref = _publish(recorded, _reads(), reports=True)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE settlement_qualifications(evidence_root TEXT)")
    con.execute("INSERT INTO settlement_qualifications VALUES(?)", (str(recorded),))
    roots = qualification_evidence_roots(state, (recorded,), con, stage_dir=tmp_path / "stage")
    assert roots == (recorded, staged)
    assert qualification_speed(ref, roots, "norm.fused_add_rmsnorm") is not None
    assert qualification_speed(ref, roots, "collective.dp_attention_exchange.v1") is None


def _dashboard_db(path, reference, root, target):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE metadata(key TEXT);
        CREATE TABLE reservations(reservation_id TEXT, status TEXT, decision TEXT,
            reason TEXT, hotkey TEXT, content_hash TEXT, target_id TEXT, block INTEGER,
            event_index INTEGER, admission_epoch INTEGER);
        CREATE TABLE arena_screen_dispositions(reservation_id TEXT, attempt_index INTEGER,
            decision TEXT, lane TEXT, stage_count INTEGER, receipt_json TEXT);
        CREATE TABLE qualification_dispositions(reservation_id TEXT, attempt_index INTEGER,
            decision TEXT, reason TEXT, attempt_ref_json TEXT);
        CREATE TABLE settlement_candidates(reservation_id TEXT, status TEXT, reason TEXT,
            candidate_json TEXT);
        CREATE TABLE settlement_qualifications(reservation_id TEXT, reproduction_index INTEGER,
            qualification_json TEXT, evidence_root TEXT, retained_block INTEGER);
        CREATE TABLE settlement_events(sequence INTEGER, event_type TEXT,
            reservation_id TEXT, target_id TEXT);
        CREATE TABLE evaluation_leases(lease_id TEXT, stage TEXT, state TEXT, generation INTEGER,
            claimed_block INTEGER, expires_block INTEGER, completed_block INTEGER, reason TEXT);
        CREATE TABLE evaluation_lease_members(reservation_id TEXT, lease_id TEXT);
    """)
    con.execute("INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("example", "failed", "FAIL", "speed_threshold_not_met", "miner", "bundle",
                 target, 9009700, 0, 1))
    con.execute("INSERT INTO qualification_dispositions VALUES(?,?,?,?,?)",
                ("example", 0, "FAIL", "speed_threshold_not_met", reference))
    con.execute("INSERT INTO settlement_qualifications VALUES(?,?,?,?,?)",
                ("older", 0, "{}", str(root), 1))
    con.commit()
    con.close()


@pytest.mark.parametrize("target,phase", [
    ("norm.fused_add_rmsnorm", False), ("collective.dp_attention_exchange.v1", True)])
def test_real_submission_api_exposes_prefill_and_optional_latency(tmp_path, monkeypatch, target, phase):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from dashboard import app

    evidence = tmp_path / "retained"
    ref = _publish(evidence, _reads(phase), target, reports=True)
    db = tmp_path / "intake.sqlite3"
    _dashboard_db(db, ref, evidence, target)
    monkeypatch.setattr(app, "DB_PATH", db)
    monkeypatch.setattr(app, "QUAL_EVIDENCE_STATE", tmp_path / "state")
    monkeypatch.setattr(app, "QUAL_EVIDENCE_EXTRA", ())
    monkeypatch.setattr(app, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(app, "SPOOL", tmp_path / "spool")
    monkeypatch.setattr(app, "OFFER_PATH", tmp_path / "offer.json")
    response = TestClient(app.app).get("/api/submissions/example")
    assert response.status_code == 200
    detail = response.json()
    assert detail["target_id"] == target
    assert detail["tokens_per_second"] == 6.7
    assert detail["baseline_measurements"]["baseline_tokens_per_second"] == 6.7
    speed = detail["qualification_attempts"][0]["speed"]
    assert speed["prefill"]["speedup"] == pytest.approx(160 / 159.04)
    assert bool(speed["lanes"][1]["cells"]) is phase

    con = sqlite3.connect(db)
    con.execute("UPDATE reservations SET status='qualified', decision='PASS'")
    con.execute("UPDATE qualification_dispositions SET decision='PASS'")
    con.execute("UPDATE settlement_qualifications SET reservation_id='example'")
    primary = {"speedup": "1.03", "target_id": target, "incumbent_manifest": {"entries": {}}}
    con.execute("INSERT INTO settlement_candidates VALUES(?,?,?,?)", (
        "example", "crowned", "", json.dumps({"primary": primary, "reproduction": primary})))
    con.execute("INSERT INTO settlement_events VALUES(?,?,?,?)", (1, "CROWN", "example", target))
    con.commit()
    con.close()
    winner = TestClient(app.app).get("/api/winners").json()["items"][0]
    assert winner["cumulative_speedup_over_sglang"] == 1.03
    assert winner["sglang_tokens_per_second"] is None
    assert winner["prefill_speedup"] == pytest.approx(160 / 159.04)


@pytest.mark.parametrize("target", ["norm.fused_add_rmsnorm", "collective.dp_attention_exchange.v1"])
def test_held_result_retains_metrics_without_a_disposition_and_deduplicates_import(
    tmp_path, monkeypatch, target,
):
    from fastapi.testclient import TestClient
    from dashboard import app

    root = tmp_path / "evidence"
    reference = _publish(root, _reads(), target)
    ref = EvidenceArtifactRef.from_dict(json.loads(reference))
    db = tmp_path / "intake.sqlite3"
    _dashboard_db(db, reference, root, target)
    con = sqlite3.connect(db)
    con.execute("DELETE FROM qualification_dispositions")
    con.execute("UPDATE reservations SET status='held', decision='', reason='remote_qualification_hold:legacy_no_decision'")
    con.commit()
    spool, request_id = tmp_path / "spool", "a" * 64
    carrier = spool / "outbox-retired" / request_id
    carrier.mkdir(parents=True)
    (carrier / "request.json").write_text(json.dumps({
        "request_id": request_id, "lease": {"members": [{"reservation_id": "example"}]}}))
    result = spool / "results-retired" / request_id
    (result / "blobs").mkdir(parents=True)
    response = json.dumps({"payload_kind": "remote_qualification_product", "payload": {
        "batch": {"attempt_ref": ref.to_dict(), "outcomes": [{"reservation_digest": "example",
            "decision": "NO_DECISION", "reason": "speed_noise"}]},
        "evidence": [{"reference": ref.to_dict(), "payload_base64":
            base64.b64encode(reopen_evidence(root, ref)).decode()}],
    }}).encode()
    digest = hashlib.sha256(response).hexdigest()
    blob = result / "blobs" / digest
    blob.write_bytes(response)
    (result / "result.json").write_text(json.dumps({
        "request_id": request_id, "state": "completed", "response_sha256": digest,
        "artifacts": [{"role": "adapter_result", "sha256": digest, "size": len(response)}]}))
    for name, value in {"DB_PATH": db, "QUAL_EVIDENCE_STATE": tmp_path / "state",
                        "QUAL_EVIDENCE_EXTRA": (), "SPOOL": spool, "LOG_ROOT": tmp_path / "logs",
                        "OFFER_PATH": tmp_path / "offer.json", "ENRICH": False}.items():
        monkeypatch.setattr(app, name, value)
    client = TestClient(app.app)
    detail = client.get("/api/submissions/example").json()
    assert detail["status"] == "held" and detail["decision"] == ""
    assert detail["tokens_per_second"] == 6.7
    (attempt,) = detail["qualification_attempts"]
    assert (attempt["decision"], attempt["reason"]) == ("NO_DECISION", "speed_noise")
    assert attempt["speed"]["prefill"]["speedup"] == pytest.approx(160 / 159.04)
    assert attempt["request_id"] == request_id
    con.execute("INSERT INTO qualification_dispositions VALUES(?,?,?,?,?)",
                ("example", 0, "NO_DECISION", "speed_noise", reference))
    con.commit()
    assert len(client.get("/api/submissions/example").json()["qualification_attempts"]) == 1
    con.execute("DELETE FROM qualification_dispositions")
    con.commit()
    con.close()
    blob.write_bytes(response + b" ")
    damaged = client.get("/api/submissions/example").json()
    assert damaged["qualification_attempts"] == []
    assert "differs from retained result" in damaged["forensics"][0]["qualification_error"]


def test_dashboard_explains_valid_boundary_uncertainty_from_the_shared_grader(tmp_path):
    from cacheon.chain.baseline_band import qualification_speed_from_payload
    from cacheon.eval.qualification_runner import ResidentSpeedWitness
    from tests.test_crossover_runtime import _rig, _speed
    from tests.test_prefill_lane import _policy

    plan, baseline, candidate, mount, _, _ = _rig(
        tmp_path, (0.994, 0.995), policy=_policy(), timed_batches=5,
        baseline_durations=(1., 1.006, 1., 1.))
    result = _speed(plan, baseline, candidate, mount)
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    speed = qualification_speed_from_payload(json.dumps({"speed_witness": witness.to_dict()}).encode())
    grade = speed["grading"]
    assert grade["decision"] == "NO_DECISION"
    assert grade["detail"] == "measurement uncertainty crosses the speed decision boundary"
    assert grade["measurement_valid"] and not grade["conditioning_failed"]
    assert grade["candidate_vs_before"] < 1.01 < grade["candidate_vs_after"]
    assert grade["required_speedup"] > grade["candidate_vs_before"]
    assert grade["baseline_drift"] < grade["max_noise"]
    assert speed["prefill"]["speedup"] < 1.05

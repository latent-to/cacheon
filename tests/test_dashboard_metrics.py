"""Dashboard metrics retain workload boundaries and the correct prompt-pass units."""

import base64
import hashlib
import json
import sqlite3
from decimal import Decimal
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from dashboard import app

    for name, value in {"DB_PATH": tmp_path / "intake.sqlite3",
                        "QUAL_EVIDENCE_STATE": tmp_path / "state", "QUAL_EVIDENCE_EXTRA": (),
                        "SPOOL": tmp_path / "spool", "LOG_ROOT": tmp_path / "logs",
                        "OFFER_PATH": tmp_path / "offer.json", "ENRICH": False}.items():
        monkeypatch.setattr(app, name, value)
    return TestClient(app.app)


@pytest.mark.parametrize("target", [
    "attention.indexer_select", "collective.dp_attention_exchange.v1",
])
@pytest.mark.parametrize("target_resolved", [False, True])
def test_payment_recovery_links_actual_evaluation_without_rewriting_rejection(tmp_path, client, target, target_resolved):
    db = tmp_path / "intake.sqlite3"
    _dashboard_db(db, "", tmp_path, target)
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE metadata ADD COLUMN value TEXT")
        for column in ("event_subindex", "invalid_reason", "transport_attempts", "screen_lane",
                       "screen_status", "screen_attempts", "retry_position",
                       "eval_cost_payment_block", "eval_cost_payment_extrinsic_index"):
            con.execute(f"ALTER TABLE reservations ADD COLUMN {column}")
        con.execute("UPDATE reservations SET reason='eval_cost_payment_invalid'")
        if not target_resolved:
            con.execute("UPDATE reservations SET target_id=''")
        con.execute("INSERT INTO reservations(reservation_id,status,hotkey,target_id) "
                    "VALUES('corrected','promoted','miner',?)", (target,))
        con.execute("INSERT INTO metadata VALUES('evaluation_recoveries',?)",
                    (json.dumps({"example": "corrected"}),))
        con.execute("INSERT INTO evaluation_leases(lease_id,stage,state) "
                    "VALUES('lease','qualification','active')")
        con.execute("INSERT INTO evaluation_lease_members VALUES('corrected','lease')")
    detail = client.get("/api/submissions/example").json()
    assert detail["status"] == "failed"
    assert detail["reason"] == "eval_cost_payment_invalid"
    assert detail["evaluation_recovery"]["reservation_id"] == "corrected"
    assert detail["evaluation_recovery"]["active_stage"] == "qualification"
    listed = client.get("/api/submissions?q=example").json()["items"]
    assert listed[0]["evaluation_recovery"] == detail["evaluation_recovery"]
    with sqlite3.connect(db) as con:
        con.execute("UPDATE evaluation_leases SET state='completed'")
        con.execute("UPDATE reservations SET status='failed',decision='FAIL',reason='candidate_slower' "
                    "WHERE reservation_id='corrected'")
    recovered = client.get("/api/submissions/example").json()["evaluation_recovery"]
    assert recovered["active_stage"] is None
    assert recovered["decision"] == "FAIL"
    assert recovered["reason"] == "candidate_slower"
    with sqlite3.connect(db) as con:
        con.execute("UPDATE reservations SET target_id='another-target' WHERE reservation_id='example'")
    assert client.get("/api/submissions/example").json()["evaluation_recovery"] is None
    with sqlite3.connect(db) as con:
        con.execute("UPDATE reservations SET target_id=? WHERE reservation_id='example'", (target,))
        con.execute("UPDATE reservations SET hotkey='another-miner' WHERE reservation_id='corrected'")
    assert client.get("/api/submissions/example").json()["evaluation_recovery"] is None


@pytest.mark.parametrize("target,phase", [
    ("norm.fused_add_rmsnorm", False), ("collective.dp_attention_exchange.v1", True)])
def test_real_submission_api_exposes_prefill_and_optional_latency(tmp_path, client, target, phase, monkeypatch):
    monkeypatch.setattr("dashboard.winners.reward_comparisons", lambda con: {"example": {
        "previous_best_reservation_id": "earlier", "previous_best_speedup": Decimal("1.01"),
        "relative_speedup": Decimal("1.02"), "score_speedup": Decimal("1.02"),
        "reward_eligible": True, "grandfathered": False}})
    evidence = tmp_path / "retained"
    ref = _publish(evidence, _reads(phase), target, reports=True)
    db = tmp_path / "intake.sqlite3"
    _dashboard_db(db, ref, evidence, target)
    response = client.get("/api/submissions/example")
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
    winner = client.get("/api/winners").json()["items"][0]
    assert winner["speedup"] == 1.03
    assert winner["relative_improvement_pct"] == 2.0
    assert winner["score_improvement_pct"] == 2.0
    assert winner["previous_best_reservation_id"] == "earlier"
    detail = client.get("/api/submissions/example").json()
    assert detail["settlement"]["relative_improvement_pct"] == 2.0
    assert winner["baseline_tokens_per_second"] == 6.7
    assert winner["baseline_kind"] == "stock"
    assert winner["prefill_speedup"] == pytest.approx(160 / 159.04)


@pytest.mark.parametrize("target", ["norm.fused_add_rmsnorm", "collective.dp_attention_exchange.v1"])
def test_held_result_retains_metrics_without_a_disposition_and_deduplicates_import(
    tmp_path, client, target,
):

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


@pytest.mark.parametrize("detailed", [False, True])
def test_graph_hold_cause_is_visible_without_inventing_a_timed_attempt(tmp_path, client, detailed):

    target, request_id = "collective.all_reduce", "b" * 64
    root, db, spool = tmp_path / "evidence", tmp_path / "intake.sqlite3", tmp_path / "spool"
    reference = _publish(root, _reads(), target)
    _dashboard_db(db, reference, root, target)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM qualification_dispositions")
        con.execute("UPDATE reservations SET status='held', decision='', reason='remote_qualification_hold:graph_evidence_unavailable'")
    carrier = spool / "outbox" / request_id
    carrier.mkdir(parents=True)
    (carrier / "request.json").write_text(json.dumps({
        "request_id": request_id, "lease": {"members": [{"reservation_id": "example"}]}}))
    result = spool / "results" / request_id
    (result / "blobs").mkdir(parents=True)
    message = "PreparedGraphProbeIncompleteError: omitted temporal-eager precondition" if detailed else ""
    response = json.dumps({"payload_kind": "remote_qualification_hold", "payload": {
        "reservation_digests": ["example"], "reason": "graph_evidence_unavailable",
        "failure_type": "PreparedGraphProbeIncompleteError" if detailed else "",
        "failure_message": message,
    }}).encode()
    digest = hashlib.sha256(response).hexdigest()
    (result / "blobs" / digest).write_bytes(response)
    (result / "result.json").write_text(json.dumps({
        "request_id": request_id, "state": "completed", "response_sha256": digest,
        "artifacts": [{"role": "adapter_result", "sha256": digest, "size": len(response)}]}))
    detail = client.get("/api/submissions/example").json()
    assert detail["status"] == "held" and detail["decision"] == ""
    assert detail["qualification_attempts"] == []
    hold = detail["forensics"][0]["qualification_hold"]
    assert hold["reason"] == "graph_evidence_unavailable"
    assert "failure_message" not in hold
    assert hold["failure_type"] == ("PreparedGraphProbeIncompleteError" if detailed else "")
    assert "qualification" not in detail["forensics"][0]


def test_winners_api_labels_a_stale_hold_as_a_pass(tmp_path, client, monkeypatch):
    """2026-09-22: the label needs the settlement journal, and the view had already closed its connection."""
    monkeypatch.setattr("dashboard.winners.reward_comparisons", lambda con: {"example": {
        "previous_best_reservation_id": "earlier", "previous_best_speedup": Decimal("1.01"),
        "relative_speedup": Decimal("1.02"), "score_speedup": Decimal("1.02"),
        "reward_eligible": True, "grandfathered": False}})
    db = tmp_path / "intake.sqlite3"
    _dashboard_db(db, "", tmp_path, "norm.fused_add_rmsnorm")
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE settlement_events ADD COLUMN event_json TEXT")
        for column in ("competition_arena", "screen_lane", "publication_root"):
            con.execute(f"ALTER TABLE reservations ADD COLUMN {column}")
        con.execute("UPDATE reservations SET status='qualified', decision='PASS', reason='qualified'")
        con.execute("INSERT INTO settlement_candidates VALUES('example','held','held',?)",
                    (json.dumps({"primary": {"target_id": "norm.fused_add_rmsnorm", "speedup": "1.05"}}),))
        con.execute("INSERT INTO settlement_qualifications VALUES('example',0,'{}',?,9009800)", (str(tmp_path),))
        con.execute("INSERT INTO settlement_events VALUES(4,'HOLD','example','norm.fused_add_rmsnorm',?)",
                    (json.dumps({"reason": "stale_incumbent"}),))
    winners = client.get("/api/winners").json()["items"]
    assert [w["settlement_status"] for w in winners] == ["passed"]
    assert winners[0]["reward_claim_status"] == "offer_unavailable"


@pytest.mark.parametrize("target", ["activation.silu_and_mul", "norm.rmsnorm"])
@pytest.mark.parametrize("score,earns", [("1.1", True), ("1.045", False)])
def test_waiting_winners_keep_metrics_without_payouts_until_queue_resolves(
    tmp_path, client, monkeypatch, target, score, earns,
):
    from dashboard import app
    from tests.test_chain_intake import _qualified_settlement_candidate, _reserve_one, _store

    with _store(tmp_path) as store:
        monkeypatch.setattr(app, "DB_PATH", store.path)
        first = _qualified_settlement_candidate(store, index=0, marker="first", speedups=("1.05", "1.05"))
        blocker = _reserve_one(store, index=1, hotkey="held")
        store.mark_held(blocker.reservation_id, "inspection")
        later = _qualified_settlement_candidate(
            store, index=2, marker="later", target=target, speedups=(score, score))
        # An existing hotkey payout must not be attributed to the waiting result.
        monkeypatch.setattr(app, "current_offer", lambda: ({}, {later.hotkey: Decimal("0.75")}))
        assert [claim.hotkey for claim in store.passed_reward_claims()] == [first.hotkey]
        changes = store._db.total_changes
        payload = client.get("/api/winners").json()
        assert [row["reservation_id"] for row in payload["items"]] == [first.reservation_digest]
        assert payload["pass_total"] == payload["waiting_total"] == 1
        waiting = payload["waiting_items"][0]
        assert waiting["reservation_id"] == later.reservation_digest
        assert waiting["waiting_for_queue"] and not waiting["reward_eligible"]
        assert waiting["weight_share"] is None and waiting["reward_claim_status"] == "waiting_for_queue"
        assert waiting["speedup"] == float(score)
        assert waiting["improvement_pct"] == pytest.approx((float(score) - 1) * 100)
        assert waiting["relative_improvement_pct"] is None
        assert store._db.total_changes == changes
        assert [claim.hotkey for claim in store.passed_reward_claims()] == [first.hotkey]

        store.expire(blocker.reservation_id, current_block=500010, reason="operator_terminal_expiry")
        assert len(store.passed_reward_claims()) == 1 + earns
        resolved = client.get("/api/winners").json()
        assert resolved["waiting_items"] == [] and resolved["waiting_total"] == 0
        assert resolved["pass_total"] == 1 + earns
        assert {row["reservation_id"] for row in resolved["items"]} == (
            {first.reservation_digest, later.reservation_digest} if earns else {first.reservation_digest})
        assert all(not row["waiting_for_queue"] for row in resolved["items"])
        if earns:
            winner = next(row for row in resolved["items"] if row["reservation_id"] == later.reservation_digest)
            assert winner["weight_share"] == .75 and winner["reward_claim_status"] == "earning"
            assert winner["improvement_pct"] == waiting["improvement_pct"]


@pytest.mark.parametrize("target,other_target", [
    ("activation.silu_and_mul", "norm.rmsnorm"),
    ("norm.rmsnorm", "activation.silu_and_mul"),
])
def test_resolving_earlier_winner_rescores_potential_winners_in_queue_order(
    tmp_path, client, monkeypatch, target, other_target,
):
    from decimal import ROUND_FLOOR
    from cacheon.chain.qualification_settlement import _reward_projection_inputs
    from dashboard import app
    from tests.test_chain_intake import _qualified_settlement_candidate, _store

    with _store(tmp_path) as store:
        monkeypatch.setattr(app, "DB_PATH", store.path)
        pending = []
        apply = store.apply_qualification_batch
        # Retain real qualification batches, then import them in reverse queue order.
        with monkeypatch.context() as patch:
            patch.setattr(store, "apply_qualification_batch", lambda batch, **kw: pending.append((batch, kw)))
            candidates = [_qualified_settlement_candidate(
                store, index=i, marker=str(i), target=slot, speedups=(score, score),
                check_single_pass=False, retained_block=20-i,
            ) for i, (slot, score) in enumerate([
                (target, "1.1"), (other_target, "1.105"), (target, "1.08"),
                (other_target, "1.12"), (target, "1.5")])]
        for batch, kwargs in reversed(pending[1:]):
            apply(batch, **kwargs)
        assert store.passed_reward_claims() == ()
        before = client.get("/api/winners").json()
        assert before["items"] == [] and before["waiting_total"] == 4
        evidence = list(store._db.execute("SELECT * FROM settlement_qualifications ORDER BY reservation_id"))

        batch, kwargs = pending[0]
        apply(batch, **kwargs)
        inputs = _reward_projection_inputs(store)
        after = client.get("/api/winners").json()
        winners = {row["reservation_id"]: row for row in after["items"]}
        assert after["waiting_items"] == [] and after["waiting_total"] == 0
        assert set(winners) == {candidates[i].reservation_digest for i in (0, 3, 4)}
        # The best earlier PASS is index 1, even though it missed the reward margin.
        # Neither the slower immediate predecessor nor the faster later PASS is the reference.
        winner = winners[candidates[3].reservation_digest]
        relative = Decimal("1.12") / Decimal("1.105")
        score_ppm = int((relative * 1_000_000).to_integral_value(rounding=ROUND_FLOOR))
        assert winner["previous_best_reservation_id"] == candidates[1].reservation_digest
        assert winner["relative_improvement_pct"] == pytest.approx(float((relative - 1) * 100))
        assert winner["score_improvement_pct"] == (score_ppm - 1_000_000) / 10_000
        claims = {claim.hotkey: claim for claim in inputs["earning_claims"]}
        assert set(claims) == {candidates[i].hotkey for i in (0, 3, 4)}
        assert inputs["score_speedups"][claims[candidates[3].hotkey].digest] == score_ppm
        assert claims[candidates[3].hotkey].speedup_ppm == 1_120_000
        original = next(row for row in before["waiting_items"] if row["reservation_id"] == winner["reservation_id"])
        assert winner["improvement_pct"] == original["improvement_pct"]
        assert all(store.get(c.reservation_digest).decision == "PASS" for c in candidates)
        assert evidence == list(store._db.execute(
            "SELECT * FROM settlement_qualifications WHERE reservation_id!=? ORDER BY reservation_id",
            (candidates[0].reservation_digest,)))

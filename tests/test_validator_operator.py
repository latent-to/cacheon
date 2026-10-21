"""Operator surfaces, end to end: what an operator grants, reads, restores and requeues on a live validator.

Each test builds one private deployment root with ``tests/support/e2e.py``, and every operator
command runs as its production subprocess over that root's own stores, beside the standing CPU
supervisor (``python -m cacheon.chain.standing_cpu_supervisor``) that qualifies and settles them:

- ``cacheon chain-eval-cost`` and ``chain-eval-cost-credit`` in an arena that charges the published fee;
- ``chain-reservation-status`` and ``chain-miner-report``, with the redacted chain audit, the retained
  evidence store and the remote spool attached;
- ``cacheon explain --evidence-dir`` over the pod's retained runtime stream;
- ``chain-snapshot`` while a writer holds the store, then ``chain-snapshot-verify`` into a fresh
  restore root;
- ``chain-evaluation-lease`` preview, requeue-expired and claim from sealed authority files after
  the qualification worker stayed down past the finalized-block SLA.

Simulated, beyond the boundaries ``tests/test_validator_end_to_end.py`` states: the S3 service behind
the snapshot commands. ``tests/support/simulated_s3`` is a boto3/botocore package backed by a
directory and placed on the ``PYTHONPATH`` of the snapshot children only, so the real CLI and the
production ``S3CompatibleObjectStore`` run unchanged; nothing in ``cacheon/`` changes. The candidate
failure and the runtime stream are pod outputs: ``pod_answer(raised=...)`` publishes the pod's
candidate-failure product with its own producer, and the stream is linked into the request's
worker log with ``record_oci_artifact`` and the production ``OCIStderrArtifactReceipt``, as the pod's
OCI process manager does.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.eval_cost import EvalCostPolicy
from cacheon.chain.evaluation_lease_operator import CONFIG_SCHEMA, REQUEUE_AUTHORITY_SCHEMA
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.eval.oci_process import STDERR_ARTIFACT_SCHEMA, OCIStderrArtifactReceipt
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.remote_run_forensics import bind_request, record_oci_artifact
from tests.support import e2e
from tests.support.chain import advance, amend
from tests.support.pod import pod_answer

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
PROFILE = e2e.PROFILES["h100-tp1"]
FAIL, PASS = QualificationDecision.FAIL, QualificationDecision.PASS
FEE = EvalCostPolicy()  # the published v1 fee
SIMULATED_S3 = e2e.REPO / "tests" / "support" / "simulated_s3"
MINERS = ("miner-raised", "miner-pass", "miner-twice", "miner-forged")
SUBMISSIONS = (("miner-raised", "miner-raised", 101), ("miner-pass", "miner-pass", 101),
               ("miner-twice", "miner-twice", 101), ("miner-twice-again", "miner-twice", 102),
               ("miner-forged", "miner-forged", 103))
RAISED = "rank 0 RuntimeError at kernels/forward.py:3: rank 0 boom"


def _cli(*argv: str, code: int = 0, env: dict[str, str] | None = None) -> str:
    return e2e.run("cacheon.cli", *argv, code=code, env=env)


def _json(*argv: str) -> dict:
    return json.loads(_cli(*argv))


def _s3(objects: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    """The snapshot children's simulated S3 SDK and the private bucket they name."""

    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(SIMULATED_S3), str(e2e.REPO))),
           "AWS_ACCESS_KEY_ID": "e2e-archive-access", "AWS_SECRET_ACCESS_KEY": "e2e-archive-secret"}
    return env, ("--object-store-endpoint", objects.as_uri(), "--object-store-bucket", "validator-archive")


def _runtime_stream(root: Path, carrier: Path, content_hash: str) -> None:
    """The pod retains the candidate's runtime stream and links it to the request's journal."""

    request_id = spool.load_json(carrier / "request.json")["request_id"]
    stream = ('Traceback (most recent call last):\n'
              f'  File "/cacheon/swap-intake/{content_hash}/kernels/forward.py", line 3, in forward\n'
              '    return module.forward(*args, **kwargs)\nRuntimeError: rank 0 boom\n').encode()
    artifact = e2e.private(root / "pod" / "oci") / f"runtime-1.{'0' * 32}.stderr"
    artifact.write_bytes(stream)
    artifact.chmod(0o600)
    digest, info = hashlib.sha256(stream).hexdigest(), artifact.stat()
    receipt = OCIStderrArtifactReceipt(STDERR_ARTIFACT_SCHEMA, "qualification-lane", "runtime-1", artifact,
                                       Path(f"{artifact}.json"), digest, digest, len(stream), len(stream), False,
                                       info.st_uid, info.st_gid, 0o600)
    receipt.receipt_path.write_bytes(receipt.receipt_bytes)
    with bind_request(e2e.private(carrier.parent.parent / "results" / request_id), request_id):
        record_oci_artifact(receipt)


def test_an_operator_answers_miners_from_the_live_stores_and_restores_them_elsewhere(
        tmp_path, monkeypatch, supervisors):
    root, key = e2e.deployment_root(tmp_path, "operator-reads", monkeypatch)
    db, audit = root / "validator" / "intake.sqlite3", root / "validator" / "chain-audit.jsonl"
    state, origin = e2e.chain(root, {}, MINERS, owner="subnet-owner-coldkey"), {}
    assert e2e.intake(root, state, origin, monkeypatch, eval_cost_policy=FEE) == []

    # The quote a miner reads is the fee intake charges; an operator credit stands in for one payment.
    assert f"amount_rao:   {FEE.amount_rao}" in _cli("chain-eval-cost", "--netuid", str(e2e.NETUID))
    credit = ("chain-eval-cost-credit", "--intake-db", str(db))
    assert "REFUSED: --hotkey is required" in _cli(*credit, code=2)
    for hotkey in MINERS:
        assert f"hotkey:         {hotkey}" in _cli(*credit, "--hotkey", hotkey, "--note", "make-good")
    reveals, hashes = {}, {}
    for name, hotkey, block in SUBMISSIONS:
        hashes[name], archive = e2e.package(e2e.miner_bundle(root, name, e2e.labelled(name)), key)
        if name == "miner-forged":  # the reveal names bytes its archive does not hold
            hashes[name] = hashlib.sha256(b"forged").hexdigest()
        reveals.setdefault(hotkey, []).append(list(e2e.reveal(hotkey, hashes[name], archive, origin, block)))
    amend(state, revealed=reveals, block=110)
    assert len(e2e.intake(root, state, origin, monkeypatch, eval_cost_policy=FEE)) == 4
    rows = {row["content_hash"]: row for row in e2e.store_rows(root, "SELECT * FROM reservations")}
    ids = {name: rows[hashes[name]]["reservation_id"] for name in hashes}
    assert [rows[hashes[name]]["status"] for name in MINERS] == ["published"] * 3 + ["failed"]
    again = rows[hashes["miner-twice-again"]]
    assert (again["status"], again["reason"]) == ("failed", "missing_eval_cost_payment")  # its one credit is spent
    listed = _cli(*credit, "--list").splitlines()
    assert sorted(line.split()[1] for line in listed) == sorted(MINERS)
    assert all(f" {FEE.amount_rao} rao " in line and " spent by " in line and "at block 110" in line
               and "# make-good" in line for line in listed)

    # The redacted chain journal keeps the rejection's class, never its detail or the private URL.
    assert json.loads(audit.read_text().splitlines()[-1])["rejected"] == {
        ids["miner-twice-again"]: "missing_eval_cost_payment", ids["miner-forged"]: "fetch"}
    assert "https://" not in audit.read_text() and rows[hashes["miner-forged"]]["reason"].startswith("fetch:")
    assert (audit.stat().st_mode & 0o777, audit.parent.stat().st_mode & 0o777) == (0o600, 0o700)

    # Reads before qualification: FIFO position, redacted chronology, exact selectors only.
    read = ("chain-reservation-status", "--json", "--intake-db", str(db))
    queued = _json(*read,"--miner-hotkey", "miner-pass", "--audit-log", str(audit))
    assert (queued["queue"]["position"], queued["queue"]["depth"]) == (2, 3), queued["queue"]
    assert [(event["field"], event["finalized_block"]) for event in queued["audit_events"]] == [
        ("reserved", 110), ("published", 110)], queued["audit_events"]
    assert "miner-pass.example" not in json.dumps(queued)  # the private fetch URL never leaves the store
    forged = _json(*read,"--content-hash", hashes["miner-forged"])["reservation"]["reason"]
    assert (forged["code"], forged["detail_redacted"]) == ("fetch:detail_redacted", True)
    assert "matches 2 reservations" in _cli(*read,"--miner-hotkey", "miner-twice", code=2)
    assert "no retained reservation" in _cli(*read,"--reservation-id", "f" * 64, code=2)
    corrupt = root / "corrupt-audit.jsonl"
    corrupt.write_text('{"event":"pass"}\n')
    assert "audit line 1 is corrupt" in _cli(*read,"--reservation-id", ids["miner-pass"],
                                             "--audit-log", str(corrupt), code=2)

    materialized, ready_path, ready = e2e.commission(root, PROFILE)
    sealed = e2e.seal_authority(root, materialized, ready_path, ready)
    objects = e2e.private(tmp_path / "objects")
    s3_env, bucket = _s3(objects)
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        raised = supervisor.until(e2e.waiting)["request_id"]
        (carrier,) = e2e.carriers(root)
        _runtime_stream(root, carrier, hashes["miner-raised"])
        pod_answer(root, sealed, carrier, FAIL, raised=RAISED)
        supervisor.until(lambda status: e2e.waiting(status) and status["request_id"] != raised)
        pod_answer(root, sealed, e2e.carriers(root)[1], PASS)
        supervisor.until(lambda status: status["last_stage"] == "settlement"
                         and status["last_disposition"] == "committed")
        supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == 3)
        # The reads and the snapshot take no controller lock: they run while a writer holds the store.
        with RecoverableFinalizedIntakeStore(db, sealed.policy, scope=sealed.scope):
            leased = _json(*read,"--reservation-id", ids["miner-twice"])
            snapshot = _cli("chain-snapshot", "--intake-db", str(db), "--audit-log", str(audit),
                            "--sealed-input", f"commission={root / 'pod' / 'commissioned'}", *bucket, env=s3_env)
        assert supervisor.stop(signal.SIGKILL) == -signal.SIGKILL
    assert (leased["queue"]["state"], leased["evaluation_lease"]["prior_status"]) == ("leased", "published")
    assert f"e2e-{root.name}" not in json.dumps(leased)  # the pair's owner name is not disclosed
    assert str(root / "validator" / "worker") not in json.dumps(leased)  # nor the private publication path

    # The miner's question, answered from the durable row, the retained attempt and the pod's stream.
    report = _cli("chain-miner-report", "--intake-db", str(db), "--miner-hotkey", "miner-raised",
                  "--evidence-root", str(root / "validator" / "evidence"), "--remote-spool-root", str(root / "spool"))
    assert "outcome: FAIL (fail_disposition)  reason: candidate_exception" in report, report
    assert "what happened: A typed rank receipt proves the selected candidate raised." in report
    assert "VERDICT  your kernel raised an exception" in report and RAISED in report
    assert f"remote request {raised[:16]}: completed" in report and "retained worker log:" in report
    assert "RAISED                     kernels/forward.py:3 in forward raised RuntimeError" in report
    explained = _cli("explain", "--evidence-dir", str(root / "pod" / "oci"))
    assert "kernels/forward.py:3 in forward raised RuntimeError" in explained and hashes["miner-raised"] in explained
    twice = _cli("chain-miner-report", "--intake-db", str(db), "--miner-hotkey", "miner-twice")
    assert "submissions: 2" in twice and "reason: missing_eval_cost_payment" in twice

    # Restore drill: the snapshot reopens elsewhere and answers exactly what the live store answers.
    manifest_key = snapshot.split("manifest_key:")[1].split()[0]
    assert "bundles:        3" in snapshot and "evidence:       1" in snapshot, snapshot
    restored = tmp_path / "restored"
    verified = _cli("chain-snapshot-verify", "--manifest-key", manifest_key, "--restore-root", str(restored),
                    *bucket, env=s3_env)
    assert "private archive semantic reopen: verified" in verified
    for name in ("miner-raised", "miner-pass"):
        live = _json(*read,"--reservation-id", ids[name], "--audit-log", str(audit))
        assert _json("chain-reservation-status", "--json", "--intake-db", str(restored / "intake.sqlite3"),
                     "--reservation-id", ids[name], "--audit-log", str(restored / "chain-audit.jsonl")) == live
    commissioned = root / "pod" / "commissioned"
    for path in (path for path in commissioned.rglob("*") if path.is_file()):
        assert (restored / "sealed" / "commission" / path.relative_to(commissioned)).read_bytes() == path.read_bytes()

    # An alias key binds nothing; altered archive bytes refuse both the restore and the next snapshot.
    blobs = objects / "validator-archive" / "cacheon" / "validator-archive" / "v1"
    alias = blobs / "aliases" / "latest.json"
    alias.parent.mkdir()
    alias.write_bytes((blobs / manifest_key).read_bytes())
    assert "does not bind its digest" in _cli(
        "chain-snapshot-verify", "--manifest-key", "aliases/latest.json", *bucket, code=2, env=s3_env)
    forward = blobs / "blobs" / "sha256" / hashlib.sha256(e2e.labelled("miner-pass").encode()).hexdigest()
    forward.write_bytes(forward.read_bytes().replace(b"label", b"LABEL"))
    assert "archive blob differs from its reference" in _cli(
        "chain-snapshot-verify", "--manifest-key", manifest_key, *bucket, code=2, env=s3_env)
    assert "content-addressed archive object conflicts" in _cli(
        "chain-snapshot", "--intake-db", str(db), *bucket, code=2, env=s3_env)


def test_work_left_to_expire_while_the_worker_was_down_is_requeued_from_sealed_authority(
        tmp_path, monkeypatch, supervisors):
    root, key = e2e.deployment_root(tmp_path, "operator-requeue", monkeypatch)
    miners, origin, reveals = ("miner-early", "miner-late"), {}, {}
    for hotkey in miners:  # legacy bundles, which name no arena: the operator's lease config has none
        bundle = e2e.miner_bundle(root, hotkey, e2e.labelled(hotkey))
        manifest = bundle / "manifest.toml"
        manifest.write_text(manifest.read_text().replace(f'\narena = "{root.name}"', ""))
        content_hash, archive = e2e.package(bundle, key)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin),)
    state = e2e.chain(root, reveals, miners)
    assert len(e2e.intake(root, state, origin, monkeypatch)) == 2
    materialized, ready_path, ready = e2e.commission(root, PROFILE)
    sealed = e2e.seal_authority(root, materialized, ready_path, ready, legacy=True)
    ids = {row["hotkey"]: row["reservation_id"] for row in e2e.store_rows(root, "SELECT * FROM reservations")}
    db = root / "validator" / "intake.sqlite3"

    # The qualification worker stays down past the finalized-block SLA; intake expires the queue.
    advance(state, e2e.REVEAL_BLOCK + sealed.policy.expiry_blocks)
    e2e.intake(root, state, origin, monkeypatch)
    assert {(row["status"], row["reason"]) for row in e2e.store_rows(root, "SELECT * FROM reservations")} == {
        ("expired", "finalized_block_sla_expired")}
    report = _cli("chain-miner-report", "--intake-db", str(db), "--miner-hotkey", "miner-early")
    assert "reason: finalized_block_sla_expired" in report and "This is NOT a judgement on the bundle." in report

    # The operator's sealed lease config takes its authority from the sealed dispatcher config.
    dispatcher = json.loads((root / "validator" / "dispatcher.json").read_text())
    config = root / "validator" / "lease-operator.json"
    e2e.sealed_json(config, {
        **{name: dispatcher[name] for name in ("intake_db", "intake_policy", "intake_scope", "lease_blocks")},
        "lock_attempts": 3, "lock_retry_delay_ms": 100, "owner": "operator-console",
        "qualification_max_members": 1, "schema": CONFIG_SCHEMA, "stage": "qualification"})
    lease = ("chain-evaluation-lease", "--config", str(config))
    assert _json(*lease, "preview")["reservation_ids"] == []
    authority = {"reason": "validator_worker_unavailable", "reservation_ids": [ids[hotkey] for hotkey in miners],
                 "retained_result_reservation_ids": [], "schema": REQUEUE_AUTHORITY_SCHEMA}
    escalated, requeue = root / "requeue-escalated.json", root / "requeue.json"
    e2e.sealed_json(escalated, {**authority, "allow_repeat_refresh": "yes"})
    e2e.sealed_json(requeue, authority)
    assert "must be boolean" in _cli(*lease, "requeue-expired", "--authority", str(escalated), code=2)
    requeued = _json(*lease, "requeue-expired", "--authority", str(requeue))
    assert requeued["requeued"] == [{"reservation_id": ids[hotkey], "status": "published"} for hotkey in miners]
    assert "only an automatic validator-SLA expiry may be requeued" in _cli(
        *lease, "requeue-expired", "--authority", str(requeue), code=2)

    # FIFO is the arrival order; an operator claim owns its row, and the pair claims past it.
    assert _json(*lease, "preview")["reservation_ids"] == [ids["miner-early"]]
    claimed = _json(*lease, "claim")["lease"]
    head = e2e.REVEAL_BLOCK + sealed.policy.expiry_blocks
    assert (claimed["owner"], claimed["claimed_block"], claimed["expires_block"]) == (
        "operator-console", head, head + dispatcher["lease_blocks"])
    assert [member["reservation_id"] for member in claimed["members"]] == [ids["miner-early"]]
    held = _json("chain-reservation-status", "--json", "--intake-db", str(db), "--reservation-id", ids["miner-early"])
    assert held["evaluation_lease"]["lease_id"] == claimed["lease_id"] and "operator-console" not in json.dumps(held)
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        supervisor.until(e2e.waiting)
        (carrier,) = e2e.carriers(root)
        assert spool.load_json(carrier / "request.json")["lease"]["members"][0]["reservation_id"] == ids["miner-late"]
        pod_answer(root, sealed, carrier, PASS)
        supervisor.until(lambda status: (status["last_stage"], status["last_disposition"]) == (
            "qualification", "completed"))
        supervisor.until(lambda status: status["phase"] == "idle")
        assert supervisor.stop(signal.SIGTERM) == 0
    # The later PASS is retained, but it settles only behind the earlier arrival the operator holds.
    assert [tuple(row) for row in e2e.store_rows(root, "SELECT reservation_id,status FROM settlement_candidates")] == [
        (ids["miner-late"], "pending")]
    with RecoverableFinalizedIntakeStore(db, sealed.policy, scope=sealed.scope) as store:
        assert store.get(ids["miner-late"]).decision == "PASS" and not store.has_pending_settlement()
        assert store.evaluation_stack(sealed.incumbent.arena_digest).manifest.digest == sealed.incumbent.digest

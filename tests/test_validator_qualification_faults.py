"""Qualification faults, end to end: a pod answer short of a verdict never fails the candidate.

Invariant (AGENTS.md): infrastructure, baseline, reference or incomplete evidence never
becomes a candidate FAIL. Each test builds one private deployment root with
``tests/support/e2e.py`` and drives the standing CPU supervisor process
(``python -m cacheon.chain.standing_cpu_supervisor``) through its sealed configs.

The GPU pod is played with the pod's own producers, after ``e2e.pod_request`` has verified
the authenticated request: the worker HOLD the pod adapter returns for a worker error
(``e2e.pod_hold``), the pod service's infrastructure result
(``infrastructure_result``, which signs a pre-resident refusal and leaves any other failure
unauthenticated), and the sealed verdict (``e2e.pod_answer``). A hostile pod or shuttle is
played by rewriting the finalized result before RESULT_READY, and a corrupted retained
attempt by truncating it in the validator's evidence store. A worker replacement is the
operator's: a fresh READY epoch, ``make-registration``, the dispatcher config re-pointed
(``e2e.replace_worker``). The pruning review is the operator's verdict file per
reservation, and the second pair is a second supervisor over the same intake database
with its own owner and spool.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.pruning_review import REVIEW_SCHEMA
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.remote_worker_pod_service import infrastructure_result
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.remote_run_forensics import append_event, journal_path
from cacheon.stack_identity import sha256_hex
from tests.support import e2e
from tests.support.chain import advance

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
PROFILE = e2e.PROFILES["h100-tp1"]
FAIL, PASS = QualificationDecision.FAIL, QualificationDecision.PASS
QUEUED = ("published", "", 0, "")


def _deploy(tmp_path: Path, monkeypatch, name: str, miners: tuple[str, ...]):
    """Distinct bundles revealed in FIFO order, fetched, then one commissioned worker."""

    root, key = e2e.deployment_root(tmp_path, name, monkeypatch)
    origin, reveals = {}, {}
    for hotkey in miners:
        content_hash, archive = e2e.package(e2e.miner_bundle(root, hotkey, e2e.labelled(hotkey)), key)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin),)
    state = e2e.chain(root, reveals, miners)
    assert len(e2e.intake(root, state, origin, monkeypatch)) == len(miners)
    materialized, ready_path, ready = e2e.commission(root, PROFILE)
    sealed = e2e.seal_authority(root, materialized, ready_path, ready)
    sealed.ready, sealed.state = ready, state
    return root, sealed


def _rows(root: Path, sealed) -> dict[str, tuple[str, str, int, str]]:
    with RecoverableFinalizedIntakeStore(root / "validator" / "intake.sqlite3", sealed.policy,
                                         scope=sealed.scope) as store:
        return {row.arrival.hotkey: (row.status, row.decision, store.qualification_attempts(row.reservation_id),
                                     row.reason) for row in store.all()}


def _infrastructure(root: Path, sealed, carrier: Path, failure_code: str) -> None:
    pod = e2e.pod_request(root, sealed, carrier)
    request_id = pod.request["request_id"]
    result = e2e.private(carrier.parent.parent / "results" / request_id)
    append_event(journal_path(result), request_id, "adapter.terminal", "no_decision")
    infrastructure_result(pod.request, result, failure_code, credential=pod.credential)
    spool.atomic_bytes(result / "RESULT_READY", (request_id + "\n").encode(), mode=0o400)


def _held(status: dict) -> bool:
    return status["phase"] == "hold"


def _reservation(carrier: Path) -> str:
    return spool.load_json(carrier / "request.json")["lease"]["members"][0]["reservation_id"]


def _settlement(root: Path, sealed) -> tuple[bool, bool]:
    """Whether a retained PASS waits to settle, and whether the evaluation stack still holds genesis."""

    with RecoverableFinalizedIntakeStore(root / "validator" / "intake.sqlite3", sealed.policy,
                                         scope=sealed.scope) as store:
        stack = store.evaluation_stack(sealed.incumbent.arena_digest).manifest
        return store.has_pending_settlement(), stack.digest == sealed.incumbent.digest


def test_refusals_and_holds_keep_the_queue_moving_and_a_stop_drains_before_settlement(
        tmp_path, monkeypatch, supervisors):
    miners = ("miner-refused", "miner-pass", "miner-held", "miner-next")
    root, sealed = _deploy(tmp_path, monkeypatch, "faults-queue", miners)
    hold = "remote_qualification_hold:qualification_worker_error"
    with e2e.relay(root, sealed):
        # An authenticated pre-resident refusal buys exactly one fresh request for the same work.
        supervisor = supervisors(root)
        refused = supervisor.until(e2e.waiting)["request_id"]
        _infrastructure(root, sealed, e2e.carriers(root)[0], "adapter_start_failed")
        assert supervisor.until(lambda status: status["last_disposition"] == "requeue")["request_id"] == refused
        fresh = supervisor.until(e2e.waiting)["request_id"]
        assert fresh != refused
        assert [_reservation(carrier) for carrier in e2e.carriers(root)] == [_reservation(e2e.carriers(root)[0])] * 2
        e2e.pod_answer(root, sealed, e2e.carriers(root)[1], FAIL)

        # A stop while the PASS runs keeps waiting for it, then exits before settling or claiming.
        supervisor.until(lambda status: e2e.waiting(status) and status["request_id"] != fresh)
        stopped = int(time.time())
        supervisor.process.send_signal(signal.SIGTERM)
        supervisor.until(lambda status: e2e.waiting(status) and status["time_unix"] > stopped)
        e2e.pod_answer(root, sealed, e2e.carriers(root)[2], PASS)
        assert supervisor.process.wait(timeout=60) == 0
        assert len(e2e.carriers(root)) == 3 and _settlement(root, sealed) == (True, True)

        # A truncated retained attempt fails settlement closed; restored from backup, it settles.
        (attempt,) = (path for path in (root / "validator" / "evidence").rglob("*")
                      if path.is_file() and b'"PASS"' in path.read_bytes())
        original = attempt.read_bytes()
        attempt.chmod(0o600)
        attempt.write_bytes(original[:-1])
        attempt.chmod(0o400)
        supervisor = supervisors(root)
        assert supervisor.until(lambda status: status["phase"] == "failed",
                                allow={"failed"})["last_stage"] == "settlement"
        assert supervisor.process.wait(timeout=60) == 1
        assert "retained settlement evidence cannot reopen" in (root / "supervisor.stderr").read_text()
        assert _settlement(root, sealed) == (True, True)
        attempt.chmod(0o600)
        attempt.write_bytes(original)
        attempt.chmod(0o400)

        # The restart settles the retained PASS first; a worker HOLD then parks its submission
        # without a verdict and without blocking the next claim.
        supervisor = supervisors(root)
        assert supervisor.until(lambda status: status["last_stage"] != "idle")["last_stage"] == "settlement"
        supervisor.until(e2e.waiting)
        e2e.pod_hold(root, sealed, e2e.carriers(root)[3])
        assert supervisor.until(_held, allow={"hold"})["hold_reason"] == hold
        supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == 5)
        e2e.pod_hold(root, sealed, e2e.carriers(root)[4])
        supervisor.until(lambda status: status["phase"] == "idle", allow={"hold"})
        assert supervisor.stop(signal.SIGTERM) == 0
    assert _settlement(root, sealed) == (False, False)
    assert _rows(root, sealed) == {
        "miner-refused": ("failed", "FAIL", 1, "speed_regression"),
        "miner-pass": ("qualified", "PASS", 1, "qualified"),
        "miner-held": ("held", "", 0, hold), "miner-next": ("held", "", 0, hold)}


def _forge_mac(result: Path) -> None:
    """A pod that bypassed finalization: canonical bytes whose response MAC is forged."""

    row = json.loads((result / "result.json").read_text())
    blob = result / "blobs" / row["response_sha256"]
    response = json.loads(blob.read_bytes())
    response["auth_tag"] = "0" * 64
    raw = spool.spool_canonical_json(response) + b"\n"
    digest = sha256_hex(raw)
    spool.atomic_bytes(result / "blobs" / digest, raw, mode=0o400)
    blob.unlink()
    row["artifacts"][0].update(sha256=digest, size=len(raw))
    row["response_sha256"] = digest
    (result / "result.json").unlink()
    spool.atomic_json(result / "result.json", row, mode=0o400)


def _flip_verdict(result: Path) -> None:
    """A shuttle that rewrote the retained FAIL as a PASS."""

    blob = result / "blobs" / json.loads((result / "result.json").read_text())["response_sha256"]
    raw = blob.read_bytes()
    assert raw.count(b'"FAIL"') == 1
    blob.unlink()
    blob.write_bytes(raw.replace(b'"FAIL"', b'"PASS"'))


FAULTS = {
    "forged-mac": ({"tamper": _forge_mac}, "transport_hold:result_tampered"),
    "flipped-verdict": ({"tamper": _flip_verdict}, "transport_hold:result_tampered"),
    "other-incumbent": ({"incumbent_tree_digest": sha256_hex(b"another incumbent tree")},
                        "transport_hold:remote_payload_changed"),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_an_unverifiable_verdict_holds_its_exact_request_across_restart(tmp_path, monkeypatch, supervisors, fault):
    options, reason = FAULTS[fault]
    root, sealed = _deploy(tmp_path, monkeypatch, f"faults-{fault}", ("miner-head", "miner-queued"))
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        request_id = supervisor.until(e2e.waiting)["request_id"]
        e2e.pod_answer(root, sealed, e2e.carriers(root)[0], FAIL, **options)
        held = supervisor.until(_held, allow={"hold"})
        assert (held["hold_reason"], held["request_id"]) == (reason, request_id)
        assert supervisor.stop(signal.SIGTERM) == 0
        supervisor = supervisors(root)  # The hold is durable: no rerun and no later claim.
        assert supervisor.until(_held, allow={"hold"})["hold_reason"] == reason
        assert supervisor.stop(signal.SIGTERM) == 0
    assert len(e2e.carriers(root)) == 1
    assert _rows(root, sealed) == {"miner-head": QUEUED, "miner-queued": QUEUED}


def test_lost_workers_never_rerun_or_fail_the_request_they_held(tmp_path, monkeypatch, supervisors):
    root, sealed = _deploy(tmp_path, monkeypatch, "faults-workers", ("miner-crashed", "miner-late", "miner-next"))

    # An unauthenticated worker failure may follow paid work: it holds the request.
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        crashed = supervisor.until(e2e.waiting)["request_id"]
        _infrastructure(root, sealed, e2e.carriers(root)[0], "adapter_exit_nonzero")
        assert supervisor.until(_held, allow={"hold"})["hold_reason"] == "transport_hold:worker_infrastructure_result"
        assert supervisor.stop(signal.SIGTERM) == 0
    assert _rows(root, sealed)["miner-crashed"] == QUEUED

    # Replacing the worker retires the unfinished carrier and requeues its work under the new epoch.
    sealed.registration = e2e.replace_worker(root, PROFILE, sealed.ready, "replacement-1")
    heartbeat = root / "spool" / "state" / "heartbeat.json"
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        fresh = supervisor.until(e2e.waiting)["request_id"]
        assert [path.name.split("-", 1)[1] for path in (root / "spool" / "retired").iterdir()] == [crashed]
        assert [path.name.split("-", 1)[1] for path in e2e.carriers(root)] == [fresh]
        e2e.pod_answer(root, sealed, e2e.carriers(root)[0], FAIL)
        late = supervisor.until(lambda status: e2e.waiting(status) and status["request_id"] != fresh)["request_id"]

    # A stale relay heartbeat fails the stage closed; the restart resumes the same request.
    with monkeypatch.context() as clock:
        clock.setattr(time, "time", lambda: 1_000_000_000)
        spool.atomic_json(heartbeat, spool.heartbeat_payload(sealed.registration, "running", None))
    assert supervisor.until(lambda status: status["phase"] == "failed", allow={"failed"})["request_id"] == late
    assert supervisor.process.wait(timeout=60) == 1
    assert "worker heartbeat is outside liveness bound" in (root / "supervisor.stderr").read_text()
    assert _rows(root, sealed)["miner-late"] == QUEUED
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        assert supervisor.until(e2e.waiting)["request_id"] == late
        assert supervisor.stop(signal.SIGKILL) == -signal.SIGKILL

    # The lease expires while the validator is down: the restart renews the same published request.
    advance(sealed.state, e2e.HEAD + 40)
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        assert supervisor.until(e2e.waiting)["request_id"] == late
        assert supervisor.stop(signal.SIGKILL) == -signal.SIGKILL
    assert len(e2e.carriers(root)) == 2

    # The pod finishes while the validator is down and the worker is replaced again:
    # the completed work commits under the authority that produced it, never rerun.
    e2e.pod_answer(root, sealed, e2e.carriers(root)[1], FAIL)
    sealed.registration = e2e.replace_worker(root, PROFILE, sealed.ready, "replacement-2")
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        following = supervisor.until(e2e.waiting)["request_id"]
        assert [path.name.split("-", 1)[1] for path in e2e.carriers(root)] == [fresh, late, following]
        e2e.pod_answer(root, sealed, e2e.carriers(root)[2], FAIL)
        supervisor.until(lambda status: status["phase"] == "idle")
        assert supervisor.stop(signal.SIGTERM) == 0
    assert {name: row[:3] for name, row in _rows(root, sealed).items()} == {
        "miner-crashed": ("failed", "FAIL", 1), "miner-late": ("failed", "FAIL", 1),
        "miner-next": ("failed", "FAIL", 1)}


def test_a_first_claim_waits_for_its_pruning_review_while_another_pair_holds_a_lease(
        tmp_path, monkeypatch, supervisors):
    """The Verda bypass (2026-10-18): a busy second pair emptied the capacity-one preview."""

    root, sealed = _deploy(tmp_path, monkeypatch, "faults-review", ("miner-first", "miner-pruner", "miner-third"))
    with RecoverableFinalizedIntakeStore(root / "validator" / "intake.sqlite3", sealed.policy,
                                         scope=sealed.scope) as store:
        rows = {row.arrival.hotkey: row for row in store.all()}
    reviews = e2e.private(root / "reviews")

    def review(hotkey: str, verdict: str) -> None:
        row = rows[hotkey]
        (reviews / f"{row.reservation_id}.json").write_text(json.dumps({
            "content_hash": row.arrival.content_hash, "reservation_id": row.reservation_id,
            "schema": REVIEW_SCHEMA, "verdict": verdict}))

    validator = root / "validator"
    standing = e2e.reseal(validator / "standing.json", validator / "standing.json", pruning_review_root=str(reviews))
    second = e2e.reseal(validator / "dispatcher.json", validator / "dispatcher-pair-2.json",
                        owner=f"e2e-{root.name}-pair-2", spool_root=str(e2e.private(root / "spool-pair-2")))
    second = e2e.reseal(standing, validator / "standing-pair-2.json", screen_dispatcher_config=str(second))
    review("miner-first", "pass")
    with e2e.relay(root, sealed), e2e.relay(root, sealed, "spool-pair-2"):
        pair_two = supervisors(root, second)
        pair_two.until(e2e.waiting)
        (leased,) = e2e.carriers(root, "spool-pair-2")
        pair_one = supervisors(root)
        for _ in range(3):  # The unreviewed head is never claimed while the other pair holds a lease.
            assert pair_one.until(lambda status: True)["phase"] == "idle"
        assert e2e.carriers(root) == []
        review("miner-pruner", "fail")
        review("miner-third", "pass")
        pair_one.until(e2e.waiting)
        (claimed,) = e2e.carriers(root)
        assert _reservation(claimed) == rows["miner-third"].reservation_id
        e2e.pod_answer(root, sealed, leased, FAIL)
        e2e.pod_answer(root, sealed, claimed, FAIL)
        for supervisor in (pair_two, pair_one):
            supervisor.until(lambda status: status["phase"] == "idle")
            assert supervisor.stop(signal.SIGTERM) == 0
    assert _rows(root, sealed) == {
        "miner-first": ("failed", "FAIL", 1, "speed_regression"),
        "miner-pruner": ("failed", "FAIL", 0, "routed_expert_pruning"),
        "miner-third": ("failed", "FAIL", 1, "speed_regression")}

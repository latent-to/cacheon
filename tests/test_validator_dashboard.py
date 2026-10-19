"""The read-only dashboard, end to end: the real app over HTTP, reading what production wrote.

Production names one plane per arena in ``CACHEON_DASH_SOURCES`` and runs ``python -m
dashboard.app`` beside the validator (``dashboard/README.md``). Here ``tests/support/e2e.py``
builds two commissioned arenas. In the free arena intake, the standing CPU supervisor and the
pod play (``tests/support/pod.py``) leave a PASS whose attempt carries a paired-replay speed
witness, a FAIL whose witness is slower and a worker HOLD; ``cacheon push-weight-offer`` pushes
the crowned offer to ``cacheon serve-weights``, and ``cacheon follow-weights`` commits it. In
the paid arena one miner pays with ``cacheon chain-submit --pay`` and one never pays. The
dashboard serves both planes on a local port, its chain enrichment off, and every read below
is an HTTP request to it.

Simulated boundaries, as in the other end-to-end modules: the chain and the pod. The chain
timestamps the dashboard's enrichment thread would cache are written into that cache, the only
clock the disclosure gate reads. ``/api/health`` and ``/api/arenas``, which the results monitor
(``cacheon.chain.bundle_result_monitor``) polls, read the process table from ``/proc``, so their
test runs only on a host that has one (the validator VM and CI's Linux job), not on a macOS
development host.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from nacl.public import PrivateKey

from cacheon.chain.baseline_band import replay_measurements
from cacheon.chain.bundle_result_monitor import check_results
from cacheon.chain.eval_cost import EvalCostPolicy
from cacheon.chain.fetch import fetch_bundle_from_local_file_for_testing
from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.chain.weight_share import AuthenticatedWeightOffer
from cacheon.eval.goodput_runtime import GoodputPolicy, GoodputReadSet
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_runner import ResidentSpeedWitness, _resident_speed_projection_digest
from cacheon.eval.resident_speed_policy import ResidentSpeedPolicy
from cacheon.eval.service_capacity import LoadRead, ServiceContract, TurnRecord
from tests.support import e2e
from tests.support import weight_roles as roles
from tests.support.chain import advance
from tests.support.pod import pod_answer, pod_hold

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
gateways = pytest.fixture(roles.gateway_processes, name="gateways")
PROFILE = e2e.PROFILES["h100-tp1"]
POLICY = ("--half-life-blocks", "100", "--discovery-lifetime-blocks", "20", "--discovery-pool-ppm", "100000")
DELAY = 8 * 3600  # source and raw logs are released eight hours after the result


def _witness(gain: float) -> dict:
    """A policy-17 paired-replay witness: two sessions of three turns, the candidate's warm turns ``gain`` faster."""

    contract = ServiceContract(.1, 5.0, .8)
    policy = replace(ResidentSpeedPolicy(600, .01, 2., .02, "a" * 64, "b" * 64, 16,
                                         goodput=GoodputPolicy(contract, 1.01, .001, .02, 0.)),
                     version=17, min_margin=0, goodput=GoodputPolicy(contract, 1., .001, .02, 0., .01, .001))
    expected = (("0", 3, 0), ("1", 3, 0))
    arms = []
    for arm in ("incumbent", "candidate"):
        windows = []
        for window, lane in ((1, arm), (2, "candidate" if arm == "incumbent" else "incumbent")):
            turns = []
            for session, _, _ in expected:
                for turn in range(3):
                    start = int((1e3 + 100 * window + 10 * turn) * 1e9)
                    seconds = 50. if turn == 0 else (4. if window == 1 else 1.) / (gain if arm == "candidate" else 1)
                    turns.append(TurnRecord(session, "main", turn, start, start, start + 10**8,
                                            start + int(seconds * 1e9), 1000, 4, "ok"))
            windows.append(LoadRead(arm, window, lane, 2, tuple(turns)))
        arms.append(tuple(windows))
    excluded = {"resident_policy", "rates", "goodput", "started_monotonic_s", "completed_monotonic_s",
                "evidence_digest"}
    fields = {name: f"{index + 1:064x}" for index, name in enumerate(ResidentSpeedWitness.__dataclass_fields__)
              if name not in excluded}
    fields.update(resident_policy=policy, rates=(), goodput=GoodputReadSet(*arms, expected), started_monotonic_s=1.,
                  completed_monotonic_s=301., calibration_digest=policy.calibration_digest,
                  calibration_context_digest=policy.calibration_context_digest)
    return ResidentSpeedWitness(**fields, evidence_digest=_resident_speed_projection_digest(**fields)).to_dict()


WITNESS = {"miner-pass": _witness(1.04), "miner-fail": _witness(.9)}


def _deploy(tmp_path: Path, monkeypatch, supervisors, gateways) -> SimpleNamespace:
    """Both arenas as production leaves them; the validator decryption key is the free arena's."""

    paid, paid_key = e2e.deployment_root(tmp_path, "arena-paid", monkeypatch)
    paid_state = e2e.chain(paid, {}, ("miner-paid", "miner-unpaid"), owner="subnet-owner-coldkey")
    origin = {}
    for hotkey, flags in (("miner-paid", ("--pay",)), ("miner-unpaid", ())):
        bundle = e2e.miner_bundle(paid, hotkey, e2e.labelled(hotkey))
        content_hash, archive = e2e.package(bundle, paid_key)
        url = f"https://{hotkey}.example/{content_hash}.sealed"
        origin[url] = archive
        e2e.run("cacheon.cli", "chain-submit", str(bundle), "--url", url, "--netuid", str(e2e.NETUID),
                "--network", str(paid_state), "--wallet", "miner", "--hotkey", hotkey, *flags, env=e2e.chain_env())
    advance(paid_state, e2e.HEAD + 10)  # the timelocks open
    assert len(e2e.intake(paid, paid_state, origin, monkeypatch,
                          eval_cost_policy=EvalCostPolicy(amount_rao=500_000_000))) == 1
    e2e.seal_authority(paid, *e2e.commission(paid, PROFILE))

    root, state, sealed = roles.weights_root(tmp_path, monkeypatch, "arena-free",
                                             {"miner-pass": 10, "miner-fail": 20, "miner-hold": 30}, profile=PROFILE)
    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        for index, hotkey in enumerate(("miner-pass", "miner-fail", "miner-hold")):
            supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == index + 1)
            carrier = e2e.carriers(root)[index]
            if hotkey == "miner-hold":
                pod_hold(root, sealed, carrier)
            else:
                decision = QualificationDecision.PASS if hotkey == "miner-pass" else QualificationDecision.FAIL
                pod_answer(root, sealed, carrier, decision, witness=WITNESS[hotkey])
        supervisor.until(lambda status: status["phase"] == "idle", allow={"hold"})
        assert supervisor.stop(signal.SIGTERM) == 0
    keys = roles.mint(root / "gateway" / "push.json", "eval")
    gateway = gateways(root, state, keys)
    assert "status=accepted" in e2e.run(
        "cacheon.cli", "push-weight-offer", "--intake-db", str(root / "validator" / "intake.sqlite3"),
        "--netuid", str(e2e.NETUID), "--network", str(state), "--url", gateway.url, "--push-credentials", str(keys),
        "--attribution-hotkey", "validator", *POLICY, env=e2e.chain_env())
    follow = roles.Follower(root, state, gateway.url)
    assert follow() == ("source", "committed", True)
    rows = {name: {row["hotkey"]: row for row in e2e.store_rows(path, "SELECT * FROM reservations")}
            for name, path in (("free", root), ("paid", paid))}
    return SimpleNamespace(free=root, paid=paid, sealed=sealed, gateway=gateway, follow=follow, rows=rows,
                           producer=roles.Producer(root, state, gateway.url, keys, follow.journal),
                           key=PrivateKey(bytes.fromhex((root / "bundle-decryption.key").read_text())))


def _source(tmp_path: Path, root: Path) -> dict:
    """One plane as the production sources file names it."""

    validator = root / "validator"
    return {"key": root.name, "slug": root.name, "label": root.name, "model": PROFILE.model,
            "paths": {"db": str(validator / "intake.sqlite3"), "mission": str(validator),
                      "audit": str(validator / "chain-audit.jsonl"), "spool": str(root / "spool"),
                      "heartbeat": str(root / "spool" / "state" / "heartbeat.json"),
                      "registration": str(root / "registration.json"), "logs": str(root / "logs"),
                      "evidence_state": str(validator / "evidence"), "stage": str(root / "stage")},
            "cache": str(tmp_path / f"enrichment-{root.name}.sqlite3"), "evidence_roots": [str(validator / "evidence")],
            "cutoff_reservation": "", "weights_included": True, "checkpoint": None,
            "processes": {"intake": ["chain-validate", str(validator / "intake.sqlite3")],
                          "supervisor": ["cacheon.chain.standing_cpu_supervisor", str(validator)],
                          "relay": ["cpu-serve", str(root / "spool")]}}


@pytest.fixture(name="dashboards")
def _dashboards():
    """Start ``python -m dashboard.app`` over a deployment; a start returns ``get(path, arena)``."""

    started = []

    def start(tmp_path: Path, deployed: SimpleNamespace):
        sources = tmp_path / "dashboard-sources.json"
        sources.write_text(json.dumps({"default": deployed.free.name,
                                       "weight_producer_config": str(deployed.producer.config),
                                       "sources": [_source(tmp_path, r) for r in (deployed.free, deployed.paid)]}))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "CACHEON_DASH_SOURCES": str(sources), "CACHEON_DASH_ENRICH": "0",
               "CACHEON_DASH_PORT": str(port), "CACHEON_DASH_OFFER": str(deployed.gateway.offer),
               "CACHEON_DASH_FOLLOW_JOURNAL": str(deployed.follow.journal)}
        log = tmp_path / "dashboard.log"
        started.append(subprocess.Popen([sys.executable, "-m", "dashboard.app"], cwd=e2e.REPO, env=env,
                                        stdout=log.open("w"), stderr=subprocess.STDOUT))

        def get(path: str, arena: str | None = deployed.free.name):
            query = "" if arena is None else ("&" if "?" in path else "?") + f"arena={arena}"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}{query}", timeout=30) as response:
                    code, body = response.status, response.read()
            except urllib.error.HTTPError as error:
                code, body = error.code, error.read()
            try:
                return code, json.loads(body)
            except ValueError:
                return code, body

        get.base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/bundle-encryption-key", timeout=1).close()
                return get
            except OSError:
                assert started[-1].poll() is None and time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)

    yield start
    for process in started:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)


def _stamp(tmp_path: Path, block: int, unix: int) -> None:
    """The chain timestamp of ``block``, as the enrichment thread caches it."""

    with sqlite3.connect(tmp_path / "enrichment-arena-free.sqlite3") as cache:
        cache.execute("INSERT OR REPLACE INTO block_times VALUES(?, ?)", (block, unix))


def test_the_dashboard_serves_both_arenas_from_what_production_wrote(
        tmp_path, monkeypatch, supervisors, gateways, dashboards):
    deployed = _deploy(tmp_path, monkeypatch, supervisors, gateways)
    get = dashboards(tmp_path, deployed)
    ids = {hotkey: row["reservation_id"] for hotkey, row in deployed.rows["free"].items()}

    # Planes: each arena's page, and no answer for an arena the sources file does not name.
    for path in ("/arena-free", "/arena-paid"):
        code, page = get(path, None)
        assert code == 200 and page.startswith(b"<!DOCTYPE html>")
    assert get("/arena-nope", None)[0] == 404
    assert get("/api/overview", "arena-nope") == (404, {"detail": "Unknown arena"})

    # The free arena reports what its intake store holds.
    code, listed = get("/api/submissions")
    assert code == 200 and {row["hotkey"]: (row["status"], row["decision"], row["reason"])
                            for row in listed["items"]} == {
        hotkey: (row["status"], row["decision"], row["reason"]) for hotkey, row in deployed.rows["free"].items()} == {
        "miner-pass": ("qualified", "PASS", "qualified"), "miner-fail": ("failed", "FAIL", "speed_regression"),
        "miner-hold": ("held", "", "remote_qualification_hold:qualification_worker_error")}
    overview = get("/api/overview")[1]
    assert {key: overview["counts"][key] for key in ("qualified", "failed", "held")} == {
        "qualified": 1, "failed": 1, "held": 1}
    assert overview["totals"]["crowned"] == 1 and overview["failure_reasons"] == [{"reason": "speed_regression", "n": 1}]

    # Each graded attempt reads back the replay witness the pod sealed, in its own units; the crown keeps the
    # pod's settled speedup beside it, measured on the commissioned stock baseline.
    details = {hotkey: get(f"/api/submissions/{rid}")[1] for hotkey, rid in ids.items()}
    for hotkey, decision, reason in (("miner-pass", "PASS", "qualified"), ("miner-fail", "FAIL", "speed_regression")):
        (attempt,) = details[hotkey]["qualification_attempts"]
        assert (attempt["decision"], attempt["reason"]) == (decision, reason)
        assert attempt["speed"] == json.loads(json.dumps(replay_measurements(WITNESS[hotkey])))
        assert attempt["speed"]["grading"]["decision"] == decision
    passed = details["miner-pass"]
    assert passed["result"]["detail"] == passed["qualification_attempts"][0]["speed"]["grading"]["detail"]
    assert {key: passed["settlement"][key] for key in ("status", "speedup_primary", "relative_improvement_pct")} == {
        "status": "crowned", "speedup_primary": 1.06, "relative_improvement_pct": 6.0}
    assert (passed["baseline"]["kind"], passed["baseline"]["stack_digest"]) == ("stock", deployed.sealed.incumbent.digest)
    # The HOLD names its cause without its source-bearing message and invents no timed attempt.
    held = details["miner-hold"]
    assert held["qualification_attempts"] == [] and held["forensics"][0]["qualification_hold"] == {
        "decision": "HOLD", "reason": "qualification_worker_error", "failure_type": "RuntimeError"}
    queue = get("/api/queue")[1]
    assert [(row["hotkey"], row["status"]) for row in queue["pending"]] == [("miner-hold", "held")]
    # The relay's pulse is not a worker observation.
    assert (queue["relay_heartbeat"]["reported_state"], queue["gpu_heartbeat"]["state"]) == ("running", "unknown")

    # Crowns, the settlement journal, and the served offer the follower committed.
    winners = get("/api/winners")[1]
    assert [(row["hotkey"], row["settlement_status"], row["speedup"], row["baseline_kind"])
            for row in winners["items"]] == [("miner-pass", "crowned", 1.06, "stock")]
    assert winners["items"][0]["result"] == passed["result"] and winners["waiting_items"] == []
    assert {row["hotkey"]: (row["crowned"], row["weight_share"]) for row in get("/api/miners")[1]["items"]} == {
        "miner-pass": (1, 1.0), "miner-fail": (0, None), "miner-hold": (0, None)}
    assert [(row["event_type"], row["reason"], row["reservation_id"]) for row in get("/api/events")[1]["items"]] == [
        ("STACK_TRANSITION", "incumbent_updated", ids["miner-pass"]), ("ADOPTION", "adopted", ids["miner-pass"]),
        ("CROWN", "qualified_win", ids["miner-pass"])]
    weights = get("/api/weights", None)[1]
    served = AuthenticatedWeightOffer.from_bytes(deployed.gateway.offer.read_bytes()).offer.projection.digest
    assert weights["offer"]["projection_digest"] == served
    assert [(row["hotkey"], row["weight_share"]) for row in weights["vector"]] == [("miner-pass", 1.0)]
    assert [(row["status"], row["projection_digest"]) for row in weights["items"]] == [
        ("confirmed", served), ("intent", served)]

    # Disclosure: source and raw logs wait for an exact result time plus eight hours; a HOLD never releases.
    run = passed["forensics"][0]["request_id"]
    log, bundle = f"/api/submissions/{ids['miner-pass']}/forensics/{run}.log", f"/api/submissions/{ids['miner-pass']}/bundle.tar.gz"
    block = passed["bundle_visibility"]["result_block"]
    assert passed["bundle_visibility"] == {"available": False, "release_at": None, "result_block": block}
    assert passed["url"] == "" and passed["forensics"][0]["worker_log"]["download_url"] is None
    assert get(log)[0] == get(bundle)[0] == 403
    stamped = int(time.time()) - DELAY + 600
    _stamp(tmp_path, block, stamped)
    assert get(f"/api/submissions/{ids['miner-pass']}")[1]["bundle_visibility"] == {
        "available": False, "release_at": stamped + DELAY, "result_block": block}
    assert get(bundle) == (403, {"detail": {"available": False, "release_at": stamped + DELAY, "result_block": block},
                                 "arena": get("/api/overview")[1]["arena"]})
    _stamp(tmp_path, block, int(time.time()) - DELAY - 60)
    assert get(f"/api/submissions/{ids['miner-pass']}")[1]["url"] == bundle + "?arena=arena-free"
    code, text = get(log)
    assert code == 200 and f"request_id={run}".encode() in text
    code, archive = get(bundle)
    assert code == 200
    (tmp_path / "released.tar.gz").write_bytes(archive)
    released = fetch_bundle_from_local_file_for_testing((tmp_path / "released.tar.gz").as_uri(),
                                                        deployed.rows["free"]["miner-pass"]["content_hash"],
                                                        tmp_path / "released")
    assert (released / "kernels" / "forward.py").read_text() == e2e.labelled("miner-pass")
    assert get(f"/api/submissions/{ids['miner-hold']}")[1]["bundle_visibility"]["result_block"] is None
    code, key = get("/api/bundle-encryption-key", None)
    assert code == 200 and key["public_key"] == bytes(deployed.key.public_key).hex()
    assert bytes(deployed.key).hex() not in json.dumps(key)

    # The paid arena: the covering payment, consumed by its submission; the miner who never paid was refused
    # before routing, so no arena's page claims it.
    paid = deployed.rows["paid"]
    assert (paid["miner-unpaid"]["status"], paid["miner-unpaid"]["reason"]) == ("failed", "missing_eval_cost_payment")
    code, listed = get("/api/submissions", "arena-paid")
    assert code == 200 and [(row["hotkey"], row["status"], row["payment"]["ref"]) for row in listed["items"]] == [
        ("miner-paid", "published", f"{e2e.HEAD}-0")]
    assert [(row["hotkey"], row["amount_tao"], row["reservation_id"], row["applied"])
            for row in get("/api/payments", "arena-paid")[1]["items"]] == [
        ("miner-paid", 0.5, paid["miner-paid"]["reservation_id"], True)]
    assert get("/api/payments")[1]["items"] == []
    assert [(row["source"], row["event_type"]) for row in get("/api/arena-events", None)[1]["items"]] == [
        ("arena-free", "STACK_TRANSITION"), ("arena-free", "ADOPTION"), ("arena-free", "CROWN")]


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="the dashboard reads the process table from /proc")
def test_health_arena_fees_and_the_results_monitor_on_the_validator_host(
        tmp_path, monkeypatch, supervisors, gateways, dashboards):
    deployed = _deploy(tmp_path, monkeypatch, supervisors, gateways)
    get = dashboards(tmp_path, deployed)
    health = get("/api/health")[1]
    assert health["db"] == {"ok": True} and health["worker_epoch"] == deployed.sealed.registration["worker_epoch"]
    assert [row["up"] for row in health["processes"]] == [False, False, False]  # every role has exited
    arenas = get("/api/arenas", None)[1]
    assert [(row["key"], row["fees"], row["target_weight_ppm"]) for row in arenas["items"]] == [
        ("arena-free", {"count": 0, "tao": 0.0, "by_fee": []}, 1_000_000),
        ("arena-paid", {"count": 1, "tao": 0.5, "by_fee": [{"fee_tao": 0.5, "count": 1}]}, 0)]
    assert (arenas["fees_count"], arenas["fees_total_tao"]) == (1, 0.5)
    with FinalizedIntakeStore(deployed.paid / "validator" / "intake.sqlite3", scope=e2e.scope(deployed.paid)) as store:
        assert store._db.execute("SELECT COUNT(*) FROM eval_cost_payments").fetchone()[0] == 1

    # The results monitor reads this API: the first poll seeds history, a new outcome is announced once,
    # and a failed send is announced again on the next poll.
    ids = {hotkey: row["reservation_id"] for hotkey, row in deployed.rows["free"].items()}
    state, messages, saved = {}, [], []
    poll = lambda notify: check_results(get.base, state, notify=notify,  # noqa: E731
                                        persist=lambda: saved.append(json.dumps(state)))
    poll(messages.append)
    assert messages == [] and sorted(state["bundle_results"]["arena-free"]) == sorted(ids.values())
    del state["bundle_results"]["arena-free"][ids["miner-pass"]]

    def unreachable(message):
        raise OSError("webhook unreachable")

    with pytest.raises(OSError):
        poll(unreachable)
    poll(messages.append)
    poll(messages.append)
    (message,) = messages
    assert message.splitlines()[0] == "**Cacheon · arena-free · PASS**"
    assert message.endswith(f"/arena-free?submission={ids['miner-pass']})")

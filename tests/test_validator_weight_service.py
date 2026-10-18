"""The shared weight service, end to end: eval offers, the gateway, and the follower signer.

Production splits weight publication into three roles
(``docs/validator-guide/settlement-and-weights.md``). The eval-side producer,
``python -m cacheon.chain.weight_offer_service`` or the one-shot ``cacheon push-weight-offer``,
projects the V1 offer from the intake store and pushes it with rotatable HMAC credentials from
``cacheon mint-push-credentials``. The gateway, ``cacheon serve-weights``, keeps the authenticated
offer and serves it, signed by its authority hotkey, to hotkeys holding validator_permit. The
follower, ``cacheon follow-weights``, is the only role that signs chain weights. Each role runs
here as its own process against one simulated chain (``tests/support/chain.py``), whose hotkeys
sign with ``SimulatedKeypair``; crowns come from the standing supervisor settling PASSes the pod
sealed, as in ``tests/test_validator_weights.py``.

Test-supplied: the chain's head, registrations and permits; an intake pass holding the store
(opened in-process) or its database volume going missing; the bytes an attacker writes into the
gateway's storage; and, once, twelve hours passing, written as the follower's saved-source
observation time. Assertions read the commits the chain received, the gateway's stored offer, and
each role's own output.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.chain.ssh_worker_transport import _dispatcher_heartbeat
from cacheon.chain.weight_push_auth import (
    PushCredential, PushCredentialSet, mint_push_credential, write_push_credentials,
)
from cacheon.chain.weight_share import (
    AuthenticatedWeightOffer, CurrentWeightOffer, authenticate_weight_offer, read_current_weight_offer,
)
from cacheon.eval.qualification import QualificationDecision
from tests.support import e2e
from tests.support.chain import advance, amend

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
POLICY = ("--half-life-blocks", "100", "--discovery-lifetime-blocks", "20", "--discovery-pool-ppm", "100000")
# Each PASS's speedup over the commissioned baseline and its reveal block.
FIRST, TIE, FASTER = ("1.06", 10), ("1.06", 20), ("1.1", 82)


def _commits(state: Path) -> list[list]:
    """What the chain received: [hotkey, commit_block, uids] rows."""

    return [row[:3] for row in json.loads(state.read_text())["commits"]]


def _mint(path: Path, credential: str, *flags: str) -> Path:
    assert f"active_id={credential}" in e2e.run(
        "cacheon.cli", "mint-push-credentials", "--path", str(path), "--credential-id", credential, *flags)
    return path


class _Gateway:
    """``cacheon serve-weights`` on one local port, signing as the subnet owner's hotkey."""

    def __init__(self, root: Path, state: Path, credentials: Path | None) -> None:
        self.root, self.state, self.credentials = root, state, credentials
        self.offer = e2e.private(root / "gateway") / "offer.json"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.url, self.process = f"http://127.0.0.1:{self.port}", None

    def start(self) -> _Gateway:
        push = ("--push-credentials", str(self.credentials)) if self.credentials else ()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "cacheon.cli", "serve-weights", "--netuid", str(e2e.NETUID),
             "--network", str(self.state), "--hotkey", "subnet-owner", "--weight-offer-path", str(self.offer),
             "--port", str(self.port), *push],
            cwd=e2e.REPO, env={**e2e.chain_env(), "PYTHONUNBUFFERED": "1"}, text=True,
            stdout=subprocess.PIPE, stderr=(self.root / "gateway.stderr").open("a"))
        assert "serving current weights" in self.process.stdout.readline()
        return self

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=30)

    def stored(self) -> tuple[int, dict[str, int]]:
        """The effective block and hotkey shares (ppm) of the authenticated offer the gateway holds."""

        projection = AuthenticatedWeightOffer.from_bytes(self.offer.read_bytes()).offer.projection
        return projection.effective_block, dict(projection.weights_ppm)


@pytest.fixture(name="gateways")
def _gateways():
    started: list[_Gateway] = []
    yield lambda *args: started.append(_Gateway(*args).start()) or started[-1]
    for gateway in started:
        if gateway.process.poll() is None:
            gateway.process.kill()
            gateway.process.wait(timeout=30)


class _Follower:
    """``cacheon follow-weights`` for one signer; a call returns (mode, status, submitted)."""

    def __init__(self, root: Path, state: Path, url: str, hotkey: str = "") -> None:
        self.state, self.output = state, ""
        self.journal = e2e.private(root / f"follower-{hotkey or 'validator'}") / "journal.sqlite3"
        wallet = ("--hotkey", hotkey) if hotkey else ("--wallet-path", str(state))
        self.argv = ("cacheon.cli", "follow-weights", "--url", url, "--journal-db", str(self.journal),
                     "--netuid", str(e2e.NETUID), "--network", str(state), "--refresh-blocks", "10", *wallet)

    def __call__(self, *flags: str, block: int | None = None) -> tuple[str, str, bool]:
        if block is not None:
            advance(self.state, block)
        done = subprocess.run([sys.executable, "-m", *self.argv, *flags], cwd=e2e.REPO, env=e2e.chain_env(),
                              capture_output=True, text=True, timeout=300)
        self.output = done.stdout + done.stderr
        assert done.returncode == 0, self.output
        mode, status, submitted = re.search(
            r"mode=(\w+) projection=\w+ status=(\w+) submitted=(\w+)", done.stdout).groups()
        return mode, status, submitted == "True"

    def age(self, seconds: int) -> None:
        """The saved source was first observed ``seconds`` earlier."""

        key = "follower_last_authenticated_offer"
        with closing(sqlite3.connect(self.journal)) as db, db:
            saved = json.loads(db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0])
            saved["observed_at"] -= seconds
            db.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps(saved, sort_keys=True), key))


def _push(root: Path, state: Path, url: str, credentials: Path, *flags: str, code: int = 0) -> str:
    return e2e.run("cacheon.cli", "push-weight-offer", "--intake-db", str(root / "validator" / "intake.sqlite3"),
                   "--netuid", str(e2e.NETUID), "--network", str(state), "--url", url,
                   "--push-credentials", str(credentials), "--attribution-hotkey", "validator", *POLICY, *flags,
                   env=e2e.chain_env(), code=code)


class _Producer:
    """``python -m cacheon.chain.weight_offer_service`` over the root's sealed dispatcher config."""

    def __init__(self, root: Path, state: Path, url: str, credentials: Path, journal: Path, *,
                 service: dict | None = None, stage: dict | None = None) -> None:
        validator = root / "validator"
        self.config = validator / "offer-service.json"
        e2e.sealed_json(validator / "weights-stage.json", {
            "attribution_hotkey": "validator", "burn_hotkey": "burn-sink", "confirmation_journal": str(journal),
            "discovery_lifetime_blocks": 20, "discovery_pool_ppm": 100_000, "fallback_endpoint": "",
            "half_life_blocks": 100, "network": f"wss://{state}", "push_credentials": str(credentials),
            "push_url": url, "refresh_blocks": 10, "schema": "cacheon-standing-weights-config-v2", **(stage or {})})
        e2e.sealed_json(self.config, {
            "max_consecutive_failures": 2, "poll_ms": 100, "restart_initial_backoff_ms": 100,
            "restart_max_backoff_ms": 200, "schema": "cacheon-weight-offer-service-config-v1",
            "screen_dispatcher_config": str(validator / "dispatcher.json"),
            "weights_stage_config": str(validator / "weights-stage.json"), **(service or {})})
        self.argv = ("cacheon.chain.weight_offer_service", "--config", str(self.config))

    def __call__(self, code: int = 0) -> str:
        """One ``--once`` pass."""

        return e2e.run(*self.argv, "--once", env=e2e.chain_env(), code=code)

    def serve(self):
        """The service loop, and a reader that returns its next report line with the given prefix."""

        process = subprocess.Popen([sys.executable, "-m", *self.argv], cwd=e2e.REPO, env=e2e.chain_env(),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line.strip()) for line in process.stdout], daemon=True).start()

        def next_line(prefix: str) -> str:
            deadline = time.monotonic() + 30
            while not (line := lines.get(timeout=30)).startswith(prefix):
                assert time.monotonic() < deadline, f"the service never reported {prefix!r}"
            return line

        return process, next_line


def _forge(gateway: _Gateway, hotkey: str) -> None:
    """Someone with write access to the gateway's storage, but not its push secret, redirects the vector."""

    stored = AuthenticatedWeightOffer.from_bytes(gateway.offer.read_bytes())
    forged = CurrentWeightOffer.from_legacy_projection(
        replace(stored.offer.projection, weights_ppm=((hotkey, 1_000_000),)))
    impostor = PushCredential(stored.credential_id, "an-attacker-chosen-secret-" * 2)
    gateway.offer.write_bytes(authenticate_weight_offer(forged, impostor).to_bytes())


def _answer(supervisor, root: Path, sealed, index: int, speedup: str) -> None:
    supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == index + 1)
    e2e.pod_answer(root, sealed, e2e.carriers(root)[index], QualificationDecision.PASS, speedup=speedup)


def _shares(gateway: _Gateway) -> dict[str, float]:
    return {hotkey: ppm / 1_000_000 for hotkey, ppm in gateway.stored()[1].items()}


def test_crowns_reach_the_chain_only_through_the_follower(tmp_path, monkeypatch, supervisors, gateways):
    root, state, sealed = e2e.weights_root(tmp_path, monkeypatch, "service-crowns", {
        "miner-first": FIRST[1], "miner-tie": TIE[1], "miner-faster": FASTER[1]}, profile=e2e.PROFILES["h100-tp1"])
    keys = _mint(root / "gateway" / "push.json", "eval")
    gateway = gateways(root, state, keys)
    follow = _Follower(root, state, gateway.url)
    # First boot with nothing offered: the follower burns the pool to the subnet owner (uid 1).
    assert follow() == ("burn", "committed", True) and _commits(state) == [["validator", 100, [1]]]
    produce = _Producer(root, state, gateway.url, keys, follow.journal)
    # While an intake pass owns the store and no reward checkpoint exists, a pass is skipped, not failed.
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=e2e.scope(root)):
        assert "weight-offer-service busy" in produce(code=3)

    relay = e2e.private(root / "spool" / "state") / "heartbeat.json"
    with _dispatcher_heartbeat(sealed.registration, relay, lambda: None):
        supervisor = supervisors(root)
        _answer(supervisor, root, sealed, 0, FIRST[0])
        supervisor.until(lambda status: status["last_stage"] == "settlement")
        supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == 2)
        stopped = int(time.time())
        supervisor.process.send_signal(signal.SIGTERM)
        supervisor.until(lambda status: e2e.waiting(status) and status["time_unix"] > stopped)
        _answer(supervisor, root, sealed, 1, TIE[0])
        assert supervisor.process.wait(timeout=60) == 0

    # The crown reaches the gateway; the follower takes it one block after its own burn, then holds it.
    assert "weight-offer-service pushed" in produce() and gateway.stored() == (100, {"miner-first": 1_000_000})
    assert follow() == ("source", "committed", False)
    assert follow(block=101) == ("source", "committed", True) and _commits(state)[-1] == ["validator", 101, [3]]
    assert follow(block=105) == ("source", "committed", False)
    # With the intake database gone, the reward checkpoint keeps the offer at the live head.
    database = root / "validator" / "intake.sqlite3"
    database.rename(database.with_suffix(".offline"))
    advance(state, 150)
    assert "weight-offer-service pushed" in produce()
    database.with_suffix(".offline").rename(database)
    assert gateway.stored() == (150, {"miner-first": 1_000_000})
    assert follow() == ("source", "committed", True) and _commits(state)[-1] == ["validator", 150, [3]]

    with _dispatcher_heartbeat(sealed.registration, relay, lambda: None):
        supervisor = supervisors(root)
        _answer(supervisor, root, sealed, 2, FASTER[0])
        supervisor.until(lambda status: status["last_stage"] == "settlement" and len(e2e.carriers(root)) == 3)
        assert supervisor.stop(signal.SIGTERM) == 0
    # Each claim decays from the first follower commit that paid it: miner-first from 101, miner-faster from 200.
    advance(state, 200)
    assert "weight-offer-service pushed" in produce() and _shares(gateway) == e2e.reward_shares(
        miner_first=e2e.reward_credit(FIRST, None, 200, 101), miner_faster=e2e.reward_credit(FASTER, FIRST, 200, None))
    assert follow() == ("source", "committed", True) and _commits(state)[-1][:2] == ["validator", 200]
    advance(state, 250)
    assert "weight-offer-service pushed" in produce() and _shares(gateway) == e2e.reward_shares(
        miner_first=e2e.reward_credit(FIRST, None, 250, 101), miner_faster=e2e.reward_credit(FASTER, FIRST, 250, 200))

    # As a service it pushes once per refresh_blocks // 2 finalized blocks; two failed pushes stop it.
    service, report = produce.serve()
    try:
        assert report("weight-offer-service").startswith("weight-offer-service pushed")
        assert report("weight-offer-service idle") == "weight-offer-service idle refresh window not reached"
        advance(state, 255)
        report("weight-offer-service pushed")
        gateway.stop()
        advance(state, 260)
        assert "(2 consecutive)" in report("weight-offer-service failed") + report("weight-offer-service failed")
        assert service.wait(timeout=30) == 1 and "WEIGHT-OFFER-SERVICE-ERROR" in service.stderr.read()
    finally:
        service.kill()
        service.wait(timeout=30)
    assert gateway.stored()[0] == 255


def test_the_gateway_serves_only_authenticated_offers_to_permitted_validators(tmp_path, monkeypatch, gateways):
    root, state, _ = e2e.weights_root(tmp_path, monkeypatch, "service-gateway", {"miner-queued": e2e.REVEAL_BLOCK})
    keys = _mint(root / "gateway" / "push.json", "eval-1")
    assert keys.stat().st_mode & 0o777 == 0o600
    eval_keys = shutil.copy(keys, root / "eval-push.json")
    gateway = gateways(root, state, keys)
    follow = _Follower(root, state, gateway.url)

    # Eval pushes and never signs; the gateway keeps one offer per finalized block.
    push = (root, state, gateway.url, eval_keys, "--burn-hotkey")
    assert "status=accepted credential=eval-1" in _push(*push, "burn-sink")
    assert "conflicts at the current effective block" in _push(*push, "subnet-owner", code=1)
    assert gateway.stored() == (100, {"burn-sink": 1_000_000}) and _commits(state) == []
    # Only a hotkey holding validator_permit may read it, and only the pinned authority's signature counts.
    amend(state, unpermitted=["miner-queued"])
    miner = _Follower(root, state, gateway.url, "miner-queued")
    assert miner("--dry-run") == ("burn", "dry_run", False) and "hold validator_permit" in miner.output
    assert follow("--expected-authority", "burn-sink", "--dry-run") == ("burn", "dry_run", False)
    assert "is not the pinned authority" in follow.output
    assert follow("--dry-run") == ("source", "dry_run", False) and _commits(state) == []
    assert "pinning subnet-owner hotkey subnet-owner" in follow.output  # unpinned, it pins the subnet owner
    assert follow() == ("source", "committed", True) and _commits(state) == [["validator", 100, [2]]]

    # Rotation retires the eval host's old secret; its replacement is accepted.
    _mint(keys, "eval-2", "--retire-active")
    gateway.stop()
    gateway.start()
    advance(state, 105)
    assert "push credential is unknown or retired" in _push(*push, "burn-sink", code=1)
    shutil.copy(keys, eval_keys)
    assert "status=accepted credential=eval-2" in _push(*push, "burn-sink")
    assert gateway.stored() == (105, {"burn-sink": 1_000_000})
    # Bytes forged into its storage without the secret fail the gateway's check; the follower keeps its vector.
    _forge(gateway, "miner-queued")
    assert follow(block=110) == ("retained", "committed", True) and _commits(state)[-1] == ["validator", 110, [2]]
    assert "stored weight offer authentication failed" in follow.output


def test_the_follower_rides_out_a_lost_gateway_then_burns_to_the_subnet_owner(tmp_path, monkeypatch, gateways):
    root, state, _ = e2e.weights_root(tmp_path, monkeypatch, "service-outage", {"miner-queued": e2e.REVEAL_BLOCK})
    keys = _mint(root / "gateway" / "push.json", "eval")
    gateway = gateways(root, state, keys)
    follow = _Follower(root, state, gateway.url)
    moved = ["validator", "subnet-owner", "newcomer", "miner-queued", "burn-sink"]  # the sink re-registered at uid 4
    gone = [*moved[:4], "late-joiner"]  # and then deregistered
    _push(root, state, gateway.url, keys, "--burn-hotkey", "burn-sink")
    assert follow() == ("source", "committed", True) and _commits(state) == [["validator", 100, [2]]]

    # The gateway goes down: each new follower process re-signs its saved vector, on the refresh cadence only.
    gateway.stop()
    assert follow(block=105) == ("retained", "committed", False) and "Connection refused" in follow.output
    assert follow(block=110) == ("retained", "committed", True) and _commits(state)[-1] == ["validator", 110, [2]]
    # The vector is by hotkey: re-registered, the sink is paid at its new uid from the next refresh.
    amend(state, registrations=[[111, moved]])
    assert follow(block=120) == ("retained", "committed", True) and _commits(state)[-1] == ["validator", 120, [4]]
    # Twelve hours after the source was first observed it no longer counts: the pool burns to the subnet owner.
    follow.age(12 * 60 * 60)
    assert follow(block=121) == ("burn", "committed", True) and _commits(state)[-1] == ["validator", 121, [1]]

    # Back with a fresh offer, the gateway is followed one block after the follower's own burn.
    gateway.start()
    _push(root, state, gateway.url, keys, "--burn-hotkey", "burn-sink")
    assert follow() == ("source", "committed", False)
    assert follow(block=122) == ("source", "committed", True) and _commits(state)[-1] == ["validator", 122, [4]]
    # Past the freshness window the offer is refused; a submission lost before inclusion is retried next pass.
    amend(state, lost_submissions=1)
    assert follow(block=132) == ("retained", "released", False) and "retry next pass" in follow.output
    assert "exceeds the initial freshness window" in follow.output
    assert follow() == ("retained", "committed", True) and _commits(state)[-1] == ["validator", 132, [4]]
    # A recipient that left the subnet cannot be paid from the saved vector: the pool burns.
    amend(state, registrations=[[111, moved], [133, gone]])
    assert follow(block=133) == ("burn", "committed", True) and _commits(state)[-1] == ["validator", 133, [1]]


def test_a_signer_shares_only_its_confirmed_vector_through_a_read_only_gateway(tmp_path, monkeypatch, gateways):
    root, state, _ = e2e.weights_root(tmp_path, monkeypatch, "service-sidecar", {"miner-queued": e2e.REVEAL_BLOCK},
                                      peers=("validator-peer",))
    gateway = gateways(root, state, None)
    sign = ("cacheon.cli", "set-weights", "--intake-db", str(root / "validator" / "intake.sqlite3"),
            "--netuid", str(e2e.NETUID), "--network", str(state), "--wallet-path", str(state), *POLICY,
            "--refresh-blocks", "10", "--burn-hotkey", "burn-sink", "--weight-offer-path", str(gateway.offer))

    # The signer shares the vector it submitted, never a dry run.
    e2e.run(*sign, "--dry-run", env=e2e.chain_env())
    assert not gateway.offer.exists()
    assert "status=pending" in e2e.run(*sign, env=e2e.chain_env()) and _commits(state) == [["validator", 100, [2]]]
    offer = read_current_weight_offer(gateway.offer).projection
    assert (offer.effective_block, dict(offer.weights_ppm)) == (100, {"burn-sink": 1_000_000})
    # A peer validator follows it; the gateway, holding no push credentials, takes no pushes.
    peer = _Follower(root, state, gateway.url, "validator-peer")
    assert peer(block=103) == ("source", "committed", True) and _commits(state)[-1] == ["validator-peer", 103, [2]]
    keys = _mint(root / "eval-push.json", "eval")
    assert "weight push is not enabled on this server" in _push(
        root, state, gateway.url, keys, "--burn-hotkey", "burn-sink", code=1)


@pytest.mark.parametrize("service,stage,refusal", [
    (None, {}, "weight offer service config is unavailable"),
    ("group-writable", {}, "must be an owner-controlled regular file"),
    ({"schema": "cacheon-weight-offer-service-config-v0"}, {}, "service config schema is unsupported"),
    ({"surplus": 1}, {}, "fields are not closed"),
    ({"restart_initial_backoff_ms": 300}, {}, "initial backoff exceeds its maximum"),
    ({}, {"schema": "cacheon-standing-weights-config-v1"}, "weights stage config schema is unsupported"),
    ({}, {"network": "finney"}, "weights network must be an explicit wss:// URL"),
])
def test_the_offer_service_refuses_an_unsealed_composition(tmp_path, service, stage, refusal):
    validator = e2e.private(tmp_path / "service-refusals" / "validator")
    e2e.sealed_json(validator / "dispatcher.json", {})
    keys = validator / "push.json"
    write_push_credentials(keys, PushCredentialSet((mint_push_credential(credential_id="eval"),)))
    journal = validator / "journal.sqlite3"
    journal.touch(mode=0o600)
    produce = _Producer(validator.parent, tmp_path / "chain.json", "http://127.0.0.1:9", keys, journal,
                        service=service if isinstance(service, dict) else {}, stage=stage)
    if service is None:
        produce.config.unlink()
    elif service == "group-writable":
        produce.config.chmod(0o460)
    assert refusal in produce(code=2)

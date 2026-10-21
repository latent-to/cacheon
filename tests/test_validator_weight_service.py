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
import shutil
import signal
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.chain.weight_push_auth import (
    PushCredential, PushCredentialSet, mint_push_credential, write_push_credentials,
)
from cacheon.chain.weight_share import (
    AuthenticatedWeightOffer, CurrentWeightOffer, authenticate_weight_offer, read_current_weight_offer,
)
from cacheon.eval.qualification import QualificationDecision
from tests.support import e2e
from tests.support import weight_roles as roles
from tests.support.pod import pod_answer
from tests.support.chain import advance, amend

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
gateways = pytest.fixture(roles.gateway_processes, name="gateways")
POLICY = ("--half-life-blocks", "100", "--discovery-lifetime-blocks", "20", "--discovery-pool-ppm", "100000")
# Each PASS's speedup over the commissioned baseline and its reveal block.
FIRST, TIE, FASTER = ("1.06", 10), ("1.06", 20), ("1.1", 82)


def _commits(state: Path) -> list[list]:
    """What the chain received: [hotkey, commit_block, uids] rows."""

    return [row[:3] for row in json.loads(state.read_text())["commits"]]


def _push(root: Path, state: Path, url: str, credentials: Path, *flags: str, code: int = 0) -> str:
    return e2e.run("cacheon.cli", "push-weight-offer", "--intake-db", str(root / "validator" / "intake.sqlite3"),
                   "--netuid", str(e2e.NETUID), "--network", str(state), "--url", url,
                   "--push-credentials", str(credentials), "--attribution-hotkey", "validator", *POLICY, *flags,
                   env=e2e.chain_env(), code=code)


def _forge(gateway: roles.Gateway, hotkey: str) -> None:
    """Someone with write access to the gateway's storage, but not its push secret, redirects the vector."""

    stored = AuthenticatedWeightOffer.from_bytes(gateway.offer.read_bytes())
    forged = CurrentWeightOffer.from_legacy_projection(
        replace(stored.offer.projection, weights_ppm=((hotkey, 1_000_000),)))
    impostor = PushCredential(stored.credential_id, "an-attacker-chosen-secret-" * 2)
    gateway.offer.write_bytes(authenticate_weight_offer(forged, impostor).to_bytes())


def _answer(supervisor, root: Path, sealed, index: int, speedup: str) -> None:
    supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == index + 1)
    pod_answer(root, sealed, e2e.carriers(root)[index], QualificationDecision.PASS, speedup=speedup)


def test_crowns_reach_the_chain_only_through_the_follower(tmp_path, monkeypatch, supervisors, gateways):
    root, state, sealed = roles.weights_root(tmp_path, monkeypatch, "service-crowns", {
        "miner-first": FIRST[1], "miner-tie": TIE[1], "miner-faster": FASTER[1]}, profile=e2e.PROFILES["h100-tp1"])
    keys = roles.mint(root / "gateway" / "push.json", "eval")
    gateway = gateways(root, state, keys)
    follow = roles.Follower(root, state, gateway.url)
    # First boot with nothing offered: the follower burns the pool to the subnet owner (uid 1).
    assert follow() == ("burn", "committed", True) and _commits(state) == [["validator", 100, [1]]]
    produce = roles.Producer(root, state, gateway.url, keys, follow.journal)
    # While an intake pass owns the store and no reward checkpoint exists, a pass is skipped, not failed.
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=e2e.scope(root)):
        assert "weight-offer-service busy" in produce(code=3)

    with e2e.relay(root, sealed):
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

    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        _answer(supervisor, root, sealed, 2, FASTER[0])
        supervisor.until(lambda status: status["last_stage"] == "settlement" and len(e2e.carriers(root)) == 3)
        assert supervisor.stop(signal.SIGTERM) == 0
    # Each claim decays from the first follower commit that paid it: miner-first from 101, miner-faster from 200.
    advance(state, 200)
    assert "weight-offer-service pushed" in produce() and gateway.shares() == roles.reward_shares(
        miner_first=roles.reward_credit(FIRST, None, 200, 101),
        miner_faster=roles.reward_credit(FASTER, FIRST, 200, None))
    assert follow() == ("source", "committed", True) and _commits(state)[-1][:2] == ["validator", 200]
    advance(state, 250)
    assert "weight-offer-service pushed" in produce() and gateway.shares() == roles.reward_shares(
        miner_first=roles.reward_credit(FIRST, None, 250, 101),
        miner_faster=roles.reward_credit(FASTER, FIRST, 250, 200))

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
    root, state, _ = roles.weights_root(tmp_path, monkeypatch, "service-gateway", {"miner-queued": e2e.REVEAL_BLOCK})
    keys = roles.mint(root / "gateway" / "push.json", "eval-1")
    assert keys.stat().st_mode & 0o777 == 0o600
    eval_keys = shutil.copy(keys, root / "eval-push.json")
    gateway = gateways(root, state, keys)
    follow = roles.Follower(root, state, gateway.url)

    # Eval pushes and never signs; the gateway keeps one offer per finalized block.
    push = (root, state, gateway.url, eval_keys, "--burn-hotkey")
    assert "status=accepted credential=eval-1" in _push(*push, "burn-sink")
    assert "conflicts at the current effective block" in _push(*push, "subnet-owner", code=1)
    assert gateway.stored() == (100, {"burn-sink": 1_000_000}) and _commits(state) == []
    # Only a hotkey holding validator_permit may read it, and only the pinned authority's signature counts.
    amend(state, unpermitted=["miner-queued"])
    miner = roles.Follower(root, state, gateway.url, "miner-queued")
    assert miner("--dry-run") == ("burn", "dry_run", False) and "hold validator_permit" in miner.output
    assert follow("--expected-authority", "burn-sink", "--dry-run") == ("burn", "dry_run", False)
    assert "is not the pinned authority" in follow.output
    assert follow("--dry-run") == ("source", "dry_run", False) and _commits(state) == []
    assert "pinning subnet-owner hotkey subnet-owner" in follow.output  # unpinned, it pins the subnet owner
    assert follow() == ("source", "committed", True) and _commits(state) == [["validator", 100, [2]]]

    # Rotation retires the eval host's old secret; its replacement is accepted.
    roles.mint(keys, "eval-2", "--retire-active")
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
    root, state, _ = roles.weights_root(tmp_path, monkeypatch, "service-outage", {"miner-queued": e2e.REVEAL_BLOCK})
    keys = roles.mint(root / "gateway" / "push.json", "eval")
    gateway = gateways(root, state, keys)
    follow = roles.Follower(root, state, gateway.url)
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
    root, state, _ = roles.weights_root(tmp_path, monkeypatch, "service-sidecar", {"miner-queued": e2e.REVEAL_BLOCK},
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
    peer = roles.Follower(root, state, gateway.url, "validator-peer")
    assert peer(block=103) == ("source", "committed", True) and _commits(state)[-1] == ["validator-peer", 103, [2]]
    keys = roles.mint(root / "eval-push.json", "eval")
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
    produce = roles.Producer(validator.parent, tmp_path / "chain.json", "http://127.0.0.1:9", keys, journal,
                        service=service if isinstance(service, dict) else {}, stage=stage)
    if service is None:
        produce.config.unlink()
    elif service == "group-writable":
        produce.config.chmod(0o460)
    assert refusal in produce(code=2)

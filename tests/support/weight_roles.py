"""The weight roles as processes, and the reward arithmetic their vectors are checked against.

Production splits weight publication into three roles
(``docs/validator-guide/settlement-and-weights.md``): the eval-side offer producer
(``python -m cacheon.chain.weight_offer_service``), the gateway (``cacheon serve-weights``) and the
follower signer (``cacheon follow-weights``); the legacy V1 signer (``cacheon set-weights``) projects
and signs in one process. Each runs here as its own process against one simulated chain file;
``weights_root`` builds the deployment root they read.
"""

from __future__ import annotations

import json
import math
import queue
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from cacheon.chain.weight_share import AuthenticatedWeightOffer
from tests.support import e2e
from tests.support.chain import advance, amend


def weights_root(tmp_path: Path, monkeypatch, name: str, reveals: dict[str, int], *, profile=None,
                 peers: tuple[str, ...] = ()):
    """Distinct bundles revealed at the given blocks and fetched; a signed vector reveals three blocks later.

    UIDs run validator, subnet owner, burn sink, ``peers``, then ``reveals``. With a ``profile`` the
    arena is commissioned and the standing composition sealed.
    """

    root, key = e2e.deployment_root(tmp_path, name, monkeypatch)
    origin, revealed = {}, {}
    for hotkey, block in reveals.items():
        content_hash, archive = e2e.package(e2e.miner_bundle(root, hotkey, e2e.labelled(hotkey)), key)
        revealed[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin, block),)
    state = e2e.chain(root, revealed, ("subnet-owner", "burn-sink", *peers, *reveals), owner="subnet-owner-coldkey")
    amend(state, reveal_delay=3)
    assert len(e2e.intake(root, state, origin, monkeypatch)) == len(reveals)
    if profile is None:
        return root, state, None
    materialized, ready_path, ready = e2e.commission(root, profile)
    return root, state, e2e.seal_authority(root, materialized, ready_path, ready)


def reward_credit(record: tuple[str, int], earlier: tuple[str, int] | None, block: int,
                  decay_start: int | None) -> float:
    """``docs/reference/emissions-policy.md`` at half-life 100: ln(s) x stall bonus x decay from the confirmed start.

    A record is the PASS's speedup and reveal block; ``earlier`` is the best earlier rewarded record.
    """

    (c, b), (q, p) = record, earlier or ("1", record[1])
    s = math.floor(float(c) / float(q) * 1_000_000) / 1_000_000
    age = 0 if decay_start is None else block - decay_start
    return math.log(s) * (1 + math.sqrt((b - p) / 1800)) * 2 ** (-age / 100)


def reward_shares(**credits: float) -> dict[str, float]:
    total = sum(credits.values())
    return {hotkey.replace("_", "-"): pytest.approx(credit / total, abs=2e-6) for hotkey, credit in credits.items()}


class Signer:
    """``cacheon set-weights``, the legacy V1 signer, on one root; a call returns the journal records it appended."""

    POLICY = ("--half-life-blocks", "100", "--discovery-lifetime-blocks", "20",
              "--discovery-pool-ppm", "100000", "--refresh-blocks", "10")

    def __init__(self, root: Path, state: Path) -> None:
        self.root, self.state, self.seen, self.output = root, state, 0, ""

    def argv(self, *flags: str) -> tuple[str, ...]:
        return ("cacheon.cli", "set-weights", "--intake-db", str(self.root / "validator" / "intake.sqlite3"),
                "--netuid", str(e2e.NETUID), "--network", str(self.state), "--wallet-path", str(self.state),
                *self.POLICY, *flags)

    def __call__(self, *flags: str, block: int | None = None, code: int = 0) -> list[tuple[str, str]]:
        if block is not None:
            advance(self.state, block)
        self.output = e2e.run(*self.argv(*flags), env=e2e.chain_env(), code=code)
        return self.appended()

    def journal(self) -> list[tuple[str, str]]:
        return [(row["status"], json.loads(row["record_json"])["reason"]) for row in e2e.store_rows(
            self.root, "SELECT status, record_json FROM weight_publications ORDER BY sequence")]

    def appended(self) -> list[tuple[str, str]]:
        journal = self.journal()
        appended, self.seen = journal[self.seen:], len(journal)
        return appended

    def shares(self) -> dict[str, float]:
        """The vector the last call signed or would sign, by hotkey."""

        (rows,) = re.findall(r"SET-WEIGHTS-PAYLOAD .* rows=(\[.*\])", self.output)
        return {hotkey: float(weight) for weight, hotkey in re.findall(r"'weight': ([^,]+), 'hotkey': '([^']+)'", rows)}


def mint(path: Path, credential: str, *flags: str) -> Path:
    assert f"active_id={credential}" in e2e.run(
        "cacheon.cli", "mint-push-credentials", "--path", str(path), "--credential-id", credential, *flags)
    return path


class Gateway:
    """``cacheon serve-weights`` on one local port, signing as the subnet owner's hotkey."""

    def __init__(self, root: Path, state: Path, credentials: Path | None) -> None:
        self.root, self.state, self.credentials = root, state, credentials
        self.offer = e2e.private(root / "gateway") / "offer.json"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.url, self.process = f"http://127.0.0.1:{self.port}", None

    def start(self) -> Gateway:
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

    def shares(self) -> dict[str, float]:
        return {hotkey: ppm / 1_000_000 for hotkey, ppm in self.stored()[1].items()}


def gateway_processes():
    """Fixture body: start gateways; stop any a failed assertion left running."""

    started: list[Gateway] = []
    yield lambda *args: started.append(Gateway(*args).start()) or started[-1]
    for gateway in started:
        if gateway.process.poll() is None:
            gateway.process.kill()
            gateway.process.wait(timeout=30)


class Follower:
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


class Producer:
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

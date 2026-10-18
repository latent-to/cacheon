"""Crowns, rewards and weight publication, end to end.

Each test builds one private deployment root with ``tests/support/e2e.py``. Crowns come
from the standing CPU supervisor process (``python -m cacheon.chain.standing_cpu_supervisor``)
settling PASSes the GPU pod sealed with ``e2e.pod_answer``; weights come from the legacy V1
signer, ``python -m cacheon.cli set-weights``, run as a subprocess against the simulated
chain. That chain plays commit-reveal: a signed vector is timelocked for ``reveal_delay``
blocks, then becomes the validator's sparse row with ``last_update`` at its reveal block.
Chain events the signer must survive are chain state written with
``tests.support.chain.amend``: an out-of-band vector from the same validator key, a
submission lost before inclusion, a late reveal, a UID re-registration, and finality
moving while a sparse-row read or a submission is in flight.

Assertions read what production retained: the settlement candidates and events, the
publication journal (``weight_publications``) and the commits the chain received.
"""

from __future__ import annotations

import json
import math
import re
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from cacheon.chain.ssh_worker_transport import _dispatcher_heartbeat
from cacheon.eval.qualification import QualificationDecision
from tests.support import e2e
from tests.support.chain import advance, amend, write_state

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
PROFILE = e2e.PROFILES["h100-tp1"]
BURN, RECONCILE = "--burn-to-subnet-owner", ("--reconcile-only", "--validator-hotkey", "validator")
POLICY = ("--half-life-blocks", "100", "--discovery-lifetime-blocks", "20",
          "--discovery-pool-ppm", "100000", "--refresh-blocks", "10")
INTENT, PENDING = ("intent", "before_sdk_submission"), ("pending", "sdk_result_unconfirmed")
CONFIRMED, BOUND = ("confirmed", "authoritative_readback"), ("confirmed", "preexisting_authoritative_readback")
DEADLINE, REHELD = ("held", "publication_readback_deadline_expired"), ("held", "projection_changed_while_unresolved")
MEMBERS = ["validator", "subnet-owner", "burn-sink", "miner-queued"]


def _deploy(tmp_path: Path, monkeypatch, name: str, reveals: dict[str, int], *, commission: bool):
    """Distinct bundles revealed at the given blocks and fetched; uid 1 is the subnet owner's."""

    root, key = e2e.deployment_root(tmp_path, name, monkeypatch)
    origin, revealed = {}, {}
    for hotkey, block in reveals.items():
        body = (f"def forward(module, *args, **kwargs):\n    label = {hotkey!r}\n"
                "    return module.forward(*args, **kwargs)\n")
        content_hash, archive = e2e.package(e2e.miner_bundle(root, hotkey, body), key)
        revealed[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin, block),)
    state = e2e.chain(root, revealed, ("subnet-owner", "burn-sink", *reveals), owner="subnet-owner-coldkey")
    amend(state, reveal_delay=3)
    assert len(e2e.intake(root, state, origin, monkeypatch)) == len(reveals)
    if not commission:
        return root, state, None
    materialized, ready_path, ready = e2e.commission(root, PROFILE)
    return root, state, e2e.seal_authority(root, materialized, ready_path, ready)


def _rows(root: Path, query: str) -> list[sqlite3.Row]:
    with closing(sqlite3.connect(f"file:{root / 'validator' / 'intake.sqlite3'}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return db.execute(query).fetchall()


def _commits(state: Path) -> list[list]:
    """What the chain received: [hotkey, commit_block, uids, uint16 values, reveal_block] rows."""

    return json.loads(state.read_text())["commits"]


def _until(predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "the signer never reached the expected state"
        time.sleep(0.1)


def _out_of_band(state: Path, uid: int) -> None:
    """The validator key publishes another vector from elsewhere, revealed at the head."""

    head = json.loads(state.read_text())["block"]
    amend(state, commits=[*_commits(state), ["validator", head, [uid], [65_535], head]])


class _Signer:
    """``cacheon set-weights`` against one root; a call returns the journal records it appended."""

    def __init__(self, root: Path, state: Path) -> None:
        self.root, self.state, self.seen, self.output = root, state, 0, ""

    def argv(self, *flags: str) -> tuple[str, ...]:
        return ("cacheon.cli", "set-weights", "--intake-db", str(self.root / "validator" / "intake.sqlite3"),
                "--netuid", str(e2e.NETUID), "--network", str(self.state), "--wallet-path", str(self.state),
                *POLICY, *flags)

    def __call__(self, *flags: str, block: int | None = None, code: int = 0) -> list[tuple[str, str]]:
        if block is not None:
            advance(self.state, block)
        self.output = e2e.run(*self.argv(*flags), env=e2e.chain_env(), code=code)
        return self.appended()

    def journal(self) -> list[tuple[str, str]]:
        return [(row["status"], json.loads(row["record_json"])["reason"]) for row in _rows(
            self.root, "SELECT status, record_json FROM weight_publications ORDER BY sequence")]

    def appended(self) -> list[tuple[str, str]]:
        journal = self.journal()
        appended, self.seen = journal[self.seen:], len(journal)
        return appended

    def shares(self) -> dict[str, float]:
        """The vector the last call signed or would sign, by hotkey."""

        (rows,) = re.findall(r"SET-WEIGHTS-PAYLOAD .* rows=(\[.*\])", self.output)
        return {hotkey: float(weight) for weight, hotkey in re.findall(r"'weight': ([^,]+), 'hotkey': '([^']+)'", rows)}


# Finalized order, the speedup each PASS measured against the commissioned baseline, and its reveal block.
RECORDS = {"miner-first": ("1.06", 10), "miner-tie": ("1.06", 20), "miner-faster": ("1.1", 82),
           "miner-marginal": ("1.1164", 90), "miner-short": ("1.11", 95)}


def _credit(hotkey: str, earlier: str | None, block: int, decay_start: int | None) -> float:
    """``docs/reference/emissions-policy.md``: ln(s) x stall bonus x decay from the confirmed start."""

    (c, b), (q, p) = RECORDS[hotkey], RECORDS[earlier] if earlier else ("1", RECORDS[hotkey][1])
    s = math.floor(float(c) / float(q) * 1_000_000) / 1_000_000
    age = 0 if decay_start is None else block - decay_start
    return math.log(s) * (1 + math.sqrt((b - p) / 1800)) * 2 ** (-age / 100)


def _shares(**credits: float) -> dict[str, float]:
    total = sum(credits.values())
    return {hotkey.replace("_", "-"): pytest.approx(credit / total, abs=2e-6) for hotkey, credit in credits.items()}


def _pass(supervisor, root: Path, sealed, index: int) -> None:
    supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == index + 1)
    speedup = list(RECORDS.values())[index][0]
    e2e.pod_answer(root, sealed, e2e.carriers(root)[index], QualificationDecision.PASS, speedup=speedup)


def test_crowns_and_rewards_follow_the_finalized_record(tmp_path, monkeypatch, supervisors):
    root, state, sealed = _deploy(tmp_path, monkeypatch, "weights-record",
                                  {hotkey: block for hotkey, (_, block) in RECORDS.items()}, commission=True)
    sign = _Signer(root, state)
    # Before any crown a real vector cannot be projected at all; only a burn can be published.
    assert sign(code=1) == [] and "global projection requires typed arena authorities" in sign.output
    relay =e2e.private(root / "spool" / "state") / "heartbeat.json"
    with _dispatcher_heartbeat(sealed.registration, relay, lambda: None):
        supervisor = supervisors(root)
        _pass(supervisor, root, sealed, 0)
        supervisor.until(lambda status: status["last_stage"] == "settlement")
        # A stop while the next PASS runs keeps waiting for it, then exits before settling it.
        supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(root)) == 2)
        stopped = int(time.time())
        supervisor.process.send_signal(signal.SIGTERM)
        supervisor.until(lambda status: e2e.waiting(status) and status["time_unix"] > stopped)
        _pass(supervisor, root, sealed, 1)
        assert supervisor.process.wait(timeout=60) == 0

    # The first crown ends the subnet-owner bootstrap, even under --watch; its vector confirms at 103.
    assert sign("--watch", "--interval", "1", BURN, code=1) == [] and "active reward claims exist" in sign.output
    assert sign() == [INTENT, PENDING] and sign.shares() == {"miner-first": 1.0}
    assert sign(block=103) == [CONFIRMED]

    with _dispatcher_heartbeat(sealed.registration, relay, lambda: None):
        supervisor = supervisors(root)
        supervisor.until(lambda status: status["last_stage"] == "settlement")
        for index in (2, 3, 4):
            _pass(supervisor, root, sealed, index)
            supervisor.until(lambda status: status["last_stage"] == "settlement")
        assert supervisor.stop(signal.SIGTERM) == 0

    # Finalized order settles: a tie keeps the earlier crown, and each later crown must beat the last.
    hotkeys = {row["reservation_id"]: row["hotkey"] for row in _rows(root, "SELECT * FROM reservations")}
    assert {hotkeys[row["reservation_id"]]: row["status"] for row in _rows(
        root, "SELECT * FROM settlement_candidates")} == {
        "miner-first": "crowned", "miner-tie": "held", "miner-faster": "crowned",
        "miner-marginal": "crowned", "miner-short": "held"}
    events = [(hotkeys[row["reservation_id"]], row["event_digest"], json.loads(row["event_json"]))
              for row in _rows(root, "SELECT * FROM settlement_events ORDER BY sequence")]
    crowned = ("ADOPTION", "adopted"), ("STACK_TRANSITION", "incumbent_updated")
    assert [(event["event_type"], hotkey, event["reason"]) for hotkey, _, event in events] == [
        ("CROWN", "miner-first", "qualified_win"), *((kind, "miner-first", why) for kind, why in crowned),
        ("HOLD", "miner-tie", "lost_potential"),
        ("CROWN", "miner-faster", "qualified_ancestor_win"), ("RETIREMENT", "miner-faster", "superseded"),
        *((kind, "miner-faster", why) for kind, why in crowned),
        ("CROWN", "miner-marginal", "qualified_ancestor_win"), ("RETIREMENT", "miner-marginal", "superseded"),
        *((kind, "miner-marginal", why) for kind, why in crowned),
        ("HOLD", "miner-short", "lost_potential")]
    assert [event["previous_event_digest"] for *_, event in events] == ["", *(digest for _, digest, _ in events[:-1])]
    assert [events[index][2]["subject_digest"] for index in (5, 9)] == [  # each retires the crown it replaced
        events[index][2]["subject_digest"] for index in (0, 4)]

    # Only records 1.5% over the best earlier rewarded PASS earn: the marginal crown is unpaid.
    # A new claim's decay waits for the first confirmed vector that pays it.
    assert sign("--dry-run", block=200) == [] and sign.shares() == _shares(
        miner_first=_credit("miner-first", None, 200, 103),
        miner_faster=_credit("miner-faster", "miner-first", 200, None))
    assert sign() == [INTENT, PENDING]
    assert sign(block=203) == [CONFIRMED]
    for block in (203, 303):
        assert sign("--dry-run", block=block) == [] and sign.shares() == _shares(
            miner_first=_credit("miner-first", None, block, 103),
            miner_faster=_credit("miner-faster", "miner-first", block, 203))


def test_burn_bootstrap_publishes_holds_and_releases_through_the_journal(tmp_path, monkeypatch):
    root, state, _ = _deploy(tmp_path, monkeypatch, "weights-burn", {"miner-queued": e2e.REVEAL_BLOCK},
                             commission=False)
    sign = _Signer(root, state)

    # Nothing is crowned: the full pool goes to the subnet owner or a registered burn hotkey.
    assert sign(BURN, "--dry-run") == [] and sign.shares() == {"subnet-owner": 1.0}
    assert sign("--burn-hotkey", "burn-sink", "--dry-run") == [] and sign.shares() == {"burn-sink": 1.0}
    assert sign("--burn-hotkey", "burn-sink") == [INTENT, PENDING] and sign.shares() == {"burn-sink": 1.0}
    assert sign(BURN, code=1) == [] and "differs from the resolved subnet-owner burn projection" in sign.output
    assert sign("--burn-hotkey", "burn-sink") == [] and len(_commits(state)) == 1  # pending is never re-signed
    # Revealed at 103 and reconciled without a signer at 113: confirmed, and already due a refresh.
    assert sign(*RECONCILE, block=113, code=3) == [CONFIRMED] and "refresh_due=True" in sign.output
    assert sign(BURN) == [INTENT, PENDING] and sign.shares() == {"subnet-owner": 1.0}
    assert sign(BURN, block=116) == [CONFIRMED]
    assert sign(BURN) == [BOUND]  # a fresh projection of the same vector binds to the readback

    # The row changes under a confirmed projection: it holds until the validator's operator releases it.
    assert sign("--release-hold", "nothing is held", code=1) == [] and "only a retained" in sign.output
    _out_of_band(state, uid=2)
    assert sign(BURN, code=2) == [("held", "confirmed_vector_changed_on_chain")]
    assert sign(BURN, code=2) == []
    assert sign("--release-hold", "why", "--validator-hotkey", "miner-queued", code=1) == []
    assert "public validator hotkey" in sign.output
    assert sign(*RECONCILE, "--half-life-blocks", "99", code=1) == [] and "supplied emissions policy" in sign.output
    assert sign("--release-hold", "out-of-band vector replaced") == [("released", "out-of-band vector replaced")]
    # The signer's own --watch loop waits out the chain's rate limit, then signs once.
    log = tmp_path / "watch.log"
    with log.open("w") as output:
        watch = subprocess.Popen([sys.executable, "-m", *sign.argv("--watch", "--interval", "1", BURN)],
                                 cwd=e2e.REPO, env=e2e.chain_env(), stdout=output, stderr=subprocess.STDOUT)
    try:
        _until(lambda: "rate limit until block 117" in log.read_text())
        advance(state, 117)
        _until(lambda: sign.journal()[-1] == PENDING)
    finally:
        watch.terminate()
        watch.wait(timeout=30)
    assert sign.appended() == [INTENT, PENDING] and len(_commits(state)) == 4
    assert sign(BURN, block=120) == [CONFIRMED]
    assert sign(*RECONCILE, block=130, code=1) == [] and "refresh-due; a submission is required" in sign.output

    # A reveal later than the readback deadline holds until the operator releases it. Released before
    # it lands it lacks a post-submit readback; landing while the signer reads, it confirms unsigned.
    amend(state, reveal_delay=15)
    assert sign(BURN) == [INTENT, PENDING]
    assert sign(BURN, block=140, code=2) == [DEADLINE]
    assert sign(BURN, block=143, code=2) == [REHELD]  # still timelocked
    assert sign("--release-hold", "late reveal", block=144) == [("released", "late reveal")]
    assert sign(*RECONCILE, code=1) == [] and "lacks post-submit authoritative readback" in sign.output
    amend(state, finalize={"weights": 145})
    assert sign(BURN) == [BOUND] and len(_commits(state)) == 5
    # Landing while still held proves nothing either; released, it confirms without a signer.
    assert sign(BURN, block=155) == [INTENT, PENDING]
    assert sign(BURN, block=165, code=2) == [DEADLINE]
    assert sign(BURN, block=171, code=2) == [REHELD]  # revealed at 170
    assert sign("--release-hold", "late reveal landed") == [("released", "late reveal landed")]
    assert sign(*RECONCILE) == [BOUND] and len(_commits(state)) == 6

    # A submission lost before inclusion holds at the deadline, then releases itself, whatever another
    # hotkey still has timelocked. A vector that reveals at once confirms in the run that signed it.
    amend(state, reveal_delay=3, lost_submissions=1)
    assert sign(BURN, block=181) == [INTENT, ("pending", "sdk_exception:ConnectionError")]
    amend(state, commits=[*_commits(state), ["burn-sink", 185, [1], [65_535], 999]])
    assert sign(BURN, block=191, code=2) == [DEADLINE]
    assert sign(BURN) == [("released", "attempt_absent_from_finalized_chain"), INTENT, PENDING]
    assert sign(BURN, block=194) == [CONFIRMED] and len(_commits(state)) == 8
    amend(state, reveal_delay=0)
    assert sign(BURN, block=204) == [INTENT, PENDING, ("confirmed", "post_submit_authoritative_readback")]


def test_finality_and_registration_changes_never_sign_a_stale_uid_mapping(tmp_path, monkeypatch):
    root, state, _ = _deploy(tmp_path, monkeypatch, "weights-races", {"miner-queued": e2e.REVEAL_BLOCK},
                             commission=False)
    sign = _Signer(root, state)
    moved = [*MEMBERS[:1], "newcomer", *MEMBERS[2:], "subnet-owner"]  # the owner re-registered at uid 4
    again = [*moved[:4], "late-joiner", "subnet-owner"]  # and then at uid 5

    # Finality advances while the first sparse row is read: the signer re-reads and signs at 102.
    amend(state, finalize={"weights": 102})
    assert sign(BURN) == [INTENT, PENDING] and _commits(state)[-1][:3] == ["validator", 102, [1]]
    assert sign(BURN, block=105) == [CONFIRMED]
    # Another signer's identical commit reveals while the refresh reads the row: bound, not re-signed.
    amend(state, commits=[*_commits(state), ["validator", 112, [1], [65_535], 116]], finalize={"weights": 116})
    assert sign(BURN, block=115) == [BOUND] and len(_commits(state)) == 2
    # A lagging endpoint reports an older finalized head before signing: nothing is journaled or signed.
    amend(state, finalize={"weights": 125})
    assert sign(BURN, block=126, code=1) == [] and "regressed before signing" in sign.output
    # The owner's UID is reassigned between the first read and signing: refused before intent.
    amend(state, registrations=[[127, moved]], finalize={"weights": 127})
    assert sign(BURN, block=126, code=1) == [] and "UID mapping changed before signing" in sign.output
    assert len(_commits(state)) == 2
    # Signed for uid 4 at 127; while it is included the owner moves again, so it cannot confirm.
    amend(state, registrations=[[127, moved], [129, again]], finalize={"set_weights": 129})
    assert sign(BURN, code=2) == [INTENT, PENDING, ("held", "post_submit_uid_mapping_changed")]
    assert _commits(state)[-1][:3] == ["validator", 127, [4]]
    # Released and re-signed for uid 5; a re-registration before its readback holds it again.
    assert sign("--release-hold", "owner re-registered") == [("released", "owner re-registered")]
    assert sign(BURN) == [INTENT, PENDING] and _commits(state)[-1][:3] == ["validator", 129, [5]]
    amend(state, registrations=[[127, moved], [129, again], [131, moved]])
    assert sign(BURN, block=131, code=2) == [("held", "metagraph_uid_mapping_changed")]


@pytest.mark.parametrize("flags,refusal", [
    (("--reconcile-only", "--dry-run", *POLICY), "--reconcile-only cannot be combined with --dry-run"),
    (("--reconcile-only", *POLICY), "--reconcile-only requires --validator-hotkey"),
    (("--release-hold", "why", "--dry-run", *POLICY), "--release-hold cannot be combined with --dry-run"),
    (("--burn-hotkey", "burn-sink", "--release-hold", "why", *POLICY), "--burn-hotkey cannot be combined"),
    ((BURN, "--burn-hotkey", "burn-sink", *POLICY), "--burn-to-subnet-owner cannot be combined with --burn-hotkey"),
    ((BURN, "--weight-offer-path", "offer.json", *POLICY), "cannot be combined with --weight-offer-path"),
    (("--watch", "--dry-run", *POLICY), "--watch requires the signer path"),
    ((), "--half-life-blocks is required"),
    ((BURN,), "--half-life-blocks is required with --burn-to-subnet-owner"),
])
def test_contradictory_modes_are_refused_before_the_chain_or_journal(tmp_path, flags, refusal):
    state = write_state(tmp_path / "chain.json", hotkeys=MEMBERS, revealed={}, block=e2e.HEAD, validator="validator")
    before = state.read_text()
    output = e2e.run("cacheon.cli", "set-weights", "--intake-db", str(tmp_path / "intake.sqlite3"),
                     "--netuid", str(e2e.NETUID), "--network", str(state), "--wallet-path", str(state),
                     *flags, env=e2e.chain_env(), code=1)
    assert refusal in output and state.read_text() == before and not (tmp_path / "intake.sqlite3").exists()

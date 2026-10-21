"""Static arena allocation, end to end: one offer producer pays several arena stores by frozen terms.

``docs/validator-guide/chain-loop.md`` (static arena percentages) and
``docs/reference/emissions-policy.md``: each arena keeps its own intake store, settled by its own
standing supervisor. The single offer producer (``python -m cacheon.chain.weight_offer_service``)
names an allocation file whose sources are the arenas' sealed dispatcher configs, registers it with
a projection before activation, and from then on prices each PASS by the history row its finalized
arrival falls in. Offers reach the chain only through the gateway (``cacheon serve-weights``) and
the follower (``cacheon follow-weights``), whose journal starts each claim's decay. The roots are
built with ``tests/support/e2e.py`` and the roles run from ``tests/support/weight_roles.py``: the
primary arena on the H100 TP1 profile and an arena added later on B300 TP4, on one chain and one
validator key, each listener admitting every reveal.

Test-supplied: the chain head, the miners' archives, the GPU pods (``pod_answer``), the
operator's edits to the allocation file, an intake pass holding a store (opened in-process) and a
store's volume going missing. Assertions read the gateway's stored offer, the allocation report
the producer retained under ``weight-allocation-evidence``, the dashboard's reading of both, and
the commits the chain received.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.chain.weight_share import AuthenticatedWeightOffer
from cacheon.eval.evidence_store import EvidenceArtifactRef, reopen_evidence
from cacheon.eval.qualification import QualificationDecision
from dashboard.winners import live_offer_shares
from tests.support import e2e
from tests.support import weight_roles as roles
from tests.support.chain import advance, amend
from tests.support.pod import pod_answer

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
gateways = pytest.fixture(roles.gateway_processes, name="gateways")
# Each PASS's speedup over its arena's commissioned baseline and its reveal block.
A, A2, B = ("1.06", 10), ("1.1", 82), ("1.05", 145)
ACTIVATION, OPENED = 120, 140  # the allocation's activation block, and the row that gives arena-b its share


def _qualify(supervisors, root: Path, sealed, speedup: str) -> None:
    """The arena's supervisor (re)opens its oldest request; the pod PASSes it and it settles at the head."""

    with e2e.relay(root, sealed):
        supervisor = supervisors(root)
        supervisor.until(e2e.waiting)
        pod_answer(root, sealed, e2e.carriers(root)[-1], QualificationDecision.PASS, speedup=speedup)
        supervisor.until(lambda status: status["phase"] == "idle")
        assert supervisor.stop(signal.SIGTERM) == 0


def _allocate(path: Path, sources: dict[str, Path], *rows: tuple) -> None:
    """The operator atomically replaces the owner-controlled allocation file."""

    path.unlink(missing_ok=True)
    e2e.sealed_json(path, {"activation_block": ACTIVATION, "burn_hotkey": "burn-sink",
                           "sources": {name: str(config) for name, config in sources.items()},
                           "history": [{"from_block": block, "weights_ppm": dict(zip(sources, weights)), **bonus}
                                       for block, weights, *extra in rows for bonus in (extra[0] if extra else {},)]})


def _rewrite(path: Path, data: bytes) -> None:
    """Disk corruption of one read-only retained artifact, or its restore from backup."""

    path.chmod(0o600)
    path.write_bytes(data)
    path.chmod(0o400)


def _offer(gateway):
    return AuthenticatedWeightOffer.from_bytes(gateway.offer.read_bytes()).offer.projection


def _report(root: Path, projection) -> dict:
    """The allocation report the producer retained beside the primary store for this offer."""

    ref = EvidenceArtifactRef.from_dict(projection.to_dict()["allocation_evidence"])
    return json.loads(reopen_evidence(root / "validator" / "weight-allocation-evidence", ref))


def _shares(block: int, a2_decay_start: int | None) -> dict[str, float]:
    """Both arenas paid: arena-a's pool split by decayed credit, arena-b's single claim at its whole term.

    Both arena-a claims arrived before activation, at the full term, so arena-a requests one million
    ppm and arena-b its 400,000; the 1.4 million requested is scaled to one million. Each claim decays
    from the follower's first commit that paid it: miner-a's at 101, miner-a2's at ``a2_decay_start``.
    """

    a, a2 = roles.reward_credit(A, None, block, 101), roles.reward_credit(A2, A, block, a2_decay_start)
    shares = {"miner-a": a / (a + a2) / 1.4, "miner-a2": a2 / (a + a2) / 1.4, "miner-b": 0.4 / 1.4}
    return {hotkey: pytest.approx(share, abs=2e-6) for hotkey, share in shares.items()}


def test_one_offer_pays_each_arena_by_the_terms_frozen_at_arrival(tmp_path, monkeypatch, supervisors, gateways):
    primary, key = e2e.deployment_root(tmp_path, "arena-a", monkeypatch)
    added = e2e.private(tmp_path / "arena-b")
    e2e.private(added / "validator")
    origin, reveals = {}, {}
    for hotkey, (_, block) in (("miner-a", A), ("miner-a2", A2)):
        content_hash, archive = e2e.package(e2e.miner_bundle(primary, hotkey, e2e.labelled(hotkey)), key)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin, block),)
    state = e2e.chain(primary, reveals, ("subnet-owner", "burn-sink", "miner-a", "miner-a2", "miner-b"),
                      owner="subnet-owner-coldkey")
    amend(state, reveal_delay=3)
    assert len(e2e.intake(primary, state, origin, monkeypatch)) == 2
    sealed_a = e2e.seal_authority(primary, *e2e.commission(primary, e2e.PROFILES["h100-tp1"]))
    with e2e.relay(primary, sealed_a):
        supervisor = supervisors(primary)
        supervisor.until(e2e.waiting)
        pod_answer(primary, sealed_a, e2e.carriers(primary)[0], QualificationDecision.PASS, speedup=A[0])
        supervisor.until(lambda status: e2e.waiting(status) and len(e2e.carriers(primary)) == 2)
        assert supervisor.stop(signal.SIGKILL) == -signal.SIGKILL  # miner-a2's evaluation outlives this process

    # Registered before activation, the producer serves the primary store's own projection.
    allocation = primary / "validator" / "allocation.json"
    sources = {"arena-a": primary / "validator" / "dispatcher.json"}
    _allocate(allocation, sources, (0, (1_000_000,)), (ACTIVATION, (600_000,)))
    keys = roles.mint(primary / "gateway" / "push.json", "eval")
    gateway = gateways(primary, state, keys)
    follow = roles.Follower(primary, state, gateway.url)
    assert follow() == ("burn", "committed", True)  # the signer boots first, with nothing offered yet
    produce = roles.Producer(primary, state, gateway.url, keys, follow.journal,
                             stage={"arena_allocation_path": str(allocation)})
    assert "weight-offer-service pushed" in produce() and _offer(gateway).allocation_evidence is None
    assert gateway.shares() == {"miner-a": 1.0} and follow(block=101) == ("source", "committed", True)

    # Activated, it reports the combined projection; the earlier arrival keeps its full term.
    advance(state, 125)
    assert "weight-offer-service pushed" in produce() and gateway.shares() == {"miner-a": 1.0}
    report = _report(primary, _offer(gateway))
    assert (report["arena_weights_ppm"], report["burned_ppm"]) == ({"arena-a": 1_000_000}, 0)
    assert list(report["submission_terms"].values()) == [["arena-a", 1_000_000]]
    assert "submission_stall_bonus_ppm" not in report  # full-strength reports keep their encoding

    # A second arena is commissioned with its own store and added with zero weight in every accepted row.
    sealed_b = e2e.seal_authority(added, *e2e.commission(added, e2e.PROFILES["b300-tp4"]), state=state)
    assert len(e2e.intake(added, state, origin, monkeypatch)) == 2  # every listener admits every reveal
    sources["arena-b"] = added / "validator" / "dispatcher.json"
    accepted = ((0, (1_000_000, 0)), (ACTIVATION, (600_000, 0)),
                (OPENED, (600_000, 400_000), {"stall_bonus_ppm": 250_000}))
    _allocate(allocation, sources, *accepted)
    before = _offer(gateway)
    advance(state, 130)
    assert "weight-offer-service pushed" in produce() and gateway.shares() == {"miner-a": 1.0}
    assert _offer(gateway).policy_digest != before.policy_digest
    assert set(_report(primary, _offer(gateway))["sources"]) == {"arena-a", "arena-b"}

    # Terms and stall-bonus strength freeze at finalized arrival: miner-b arrives after arena-b's share
    # opened, while miner-a2, revealed before activation, finishes qualifying only now.
    late_hash, late_archive = e2e.package(e2e.miner_bundle(added, "miner-b", e2e.labelled("miner-b")), key)
    revealed = json.loads(state.read_text())["revealed"]
    amend(state, block=150, revealed={**revealed, "miner-b": [list(
        e2e.reveal("miner-b", late_hash, late_archive, origin, B[1]))]})
    assert e2e.intake(primary, state, origin, monkeypatch) == e2e.intake(added, state, origin, monkeypatch) == [
        late_hash]
    _qualify(supervisors, added, sealed_b, B[0])
    _qualify(supervisors, primary, sealed_a, A2[0])
    assert "weight-offer-service pushed" in produce() and gateway.shares() == _shares(150, None)
    offer = _offer(gateway)
    report = _report(primary, offer)
    assert (report["arena_weights_ppm"], report["burned_ppm"]) == ({"arena-a": 714_286, "arena-b": 285_714}, 0)
    assert sorted(report["submission_terms"].values()) == [
        ["arena-a", 1_000_000], ["arena-a", 1_000_000], ["arena-b", 400_000]]
    assert sorted(report["submission_stall_bonus_ppm"].values()) == [250_000, 1_000_000, 1_000_000]
    # The dashboard reads the same split per submission from the retained report.
    summary, by_submission = live_offer_shares(
        gateway.offer, submission_roots=[added / "validator" / "weight-allocation-evidence",
                                         primary / "validator" / "weight-allocation-evidence"])
    assert summary["submission_shares_available"] and {
        rid: int(share * 1_000_000) for rid, share in by_submission.items()} == report["submission_weights_ppm"]

    # Accepted rows cannot change, a new row must start after both stores' finalized intake, and a
    # malformed file is refused whole: weights are integer ppm, and the bonus full strength before activation.
    bonus = "stall bonus must be integer ppm, with full strength before activation"
    for rows, refusal in ((((0, (1_000_000, 0)), (ACTIVATION, (600_000, 0)), (OPENED, (500_000, 500_000))),
                           "accepted allocation history or source authority changed"),
                          ((*accepted, (150, (100_000, 900_000))),
                           "allocation updates must precede their future submission boundary"),
                          ((*accepted, (300, (True, 900_000))), "arena weights must be nonnegative integer ppm"),
                          ((*accepted, (300, (100_000, 900_000), {"stall_bonus_ppm": 1_000_001})), bonus),
                          (((0, (1_000_000, 0), {"stall_bonus_ppm": 250_000}), *accepted[1:]), bonus)):
        _allocate(allocation, sources, *rows)
        assert refusal in produce(code=2) and _offer(gateway) == offer
    _allocate(allocation, sources, *accepted, (300, (100_000, 900_000)))
    advance(state, 155)
    assert "weight-offer-service pushed" in produce() and gateway.shares() == _shares(155, None)

    # A missing or busy source store pushes from the reward checkpoint, recomputed at the live head:
    # miner-a keeps decaying while miner-a2, not yet paid on chain, does not.
    database = added / "validator" / "intake.sqlite3"
    database.rename(database.with_suffix(".offline"))
    advance(state, 250)
    assert "weight-offer-service pushed" in produce() and gateway.shares() == _shares(250, None)
    database.with_suffix(".offline").rename(database)
    with FinalizedIntakeStore(database, scope=sealed_b.scope):
        advance(state, 260)
        assert "weight-offer-service pushed" in produce() and gateway.shares() == _shares(260, None)
    # With every intake stopped at 150, the live projection keeps projecting retained rewards at the head.
    advance(state, 400)
    assert "weight-offer-service pushed" in produce() and gateway.shares() == _shares(400, None)
    report = _report(primary, _offer(gateway))
    assert {name: row["cursor"][0] for name, row in report["sources"].items()} == {"arena-a": 150, "arena-b": 150}
    # The combined vector reaches the chain through the follower; the claims it newly pays start decaying.
    assert follow() == ("source", "committed", True)
    assert json.loads(state.read_text())["commits"][-1][:3] == ["validator", 400, [3, 4, 5]]
    # Corrupt retained evidence in either store stops the live offer; the gateway keeps the last one.
    (attempt,) = (path for path in (added / "validator" / "evidence").rglob("*")
                  if path.is_file() and b'"PASS"' in path.read_bytes())
    original, offer = attempt.read_bytes(), _offer(gateway)
    _rewrite(attempt, original[:-1])
    advance(state, 410)
    assert "retained settlement evidence cannot reopen" in produce(code=2) and _offer(gateway) == offer
    _rewrite(attempt, original)
    # A recipient that left the subnet is paid to the validator's own attribution, not burned.
    hotkeys = json.loads(state.read_text())["hotkeys"]
    amend(state, registrations=[[420, [hotkey for hotkey in hotkeys if hotkey != "miner-b"]]])
    advance(state, 420)
    expected = _shares(420, 400)
    expected["validator"] = expected.pop("miner-b")
    assert "weight-offer-service pushed" in produce() and gateway.shares() == expected

    # Dropping the allocation from an activated producer does not bring back the single-store projection.
    for config in (primary / "validator" / "weights-stage.json", produce.config):
        config.unlink()
    produce = roles.Producer(primary, state, gateway.url, keys, follow.journal)
    assert "activated arena allocation requires its configured producer" in produce(code=2)

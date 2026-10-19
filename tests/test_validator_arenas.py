"""Parallel arenas and evaluation pairs over one intake store, end to end.

``docs/validator-guide/arena-service.md`` (Registry and CLI boundary): one finalized intake feeds
one shared store; each arena runs the standing supervisor
(``python -m cacheon.chain.standing_cpu_supervisor``) over its own commissioned manifest, READY,
registration, spool and evidence; within an arena each further pair is another supervisor with its
own owner and spool. Pairs only qualify (``enable_settlement`` false) and one economics supervisor
(``enable_qualification`` false) settles every arena. Each arena is built with
``tests/support/e2e.py``: the default arena, which legacy bundles map to, on the H100 TP1 profile,
and a named arena on the B300 TP4 profile sharing its chain, validator key and intake store. A
recommissioned arena keeps its name on new worker bytes, under a new root over the same store.

Test-supplied: the chain, the miners' archives and the GPU pods, played with ``pod_answer``
and ``pod_hold``. Assertions read what production retained: the carriers in each spool, the
reservations, settlement candidates and events in the shared store, and the vector the V1 signer
(``cacheon set-weights --dry-run``) would publish.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.chain import remote_worker_spool as spool
from cacheon.eval.qualification import QualificationDecision
from tests.support import e2e
from tests.support import weight_roles as roles
from tests.support.chain import amend
from tests.support.pod import pod_answer, pod_hold

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
H100, B300 = e2e.PROFILES["h100-tp1"], e2e.PROFILES["b300-tp4"]
PASS, FAIL = QualificationDecision.PASS, QualificationDecision.FAIL
# Finalized order: each miner's arena, reveal block and the speedup its PASS measures.
MINERS = {"a-first": ("arena-a", 80, "1.06"), "b-first": ("arena-b", 81, "1.05"),
          "a-second": ("arena-a", 82, "1.1"), "b-second": ("arena-b", 83, None), "a-third": ("arena-a", 84, None)}
LATE = ("a-late", ("arena-a", 105, "1.2"))  # finalized after the first crowns
CROWNED = (("ADOPTION", "adopted"), ("STACK_TRANSITION", "incumbent_updated"))


def _deploy(tmp_path: Path, monkeypatch, miners: dict[str, tuple]):
    """Both arenas' bundles revealed and fetched into the shared store, then each arena commissioned."""

    root, key = e2e.deployment_root(tmp_path, "arena-a", monkeypatch)
    roots = {"arena-a": root, "arena-b": e2e.private(tmp_path / "arena-b")}
    e2e.private(roots["arena-b"] / "validator")
    origin, reveals = {}, {}
    for hotkey, (arena, block, _) in miners.items():
        content_hash, archive = e2e.package(e2e.miner_bundle(roots[arena], hotkey, e2e.labelled(hotkey)), key)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin, block),)
    state = e2e.chain(root, reveals, tuple(miners))
    assert len(e2e.intake(root, state, origin, monkeypatch)) == len(miners)
    sealed = {}
    for arena, profile in (("arena-a", H100), ("arena-b", B300)):
        materialized, ready_path, ready = e2e.commission(roots[arena], profile)
        sealed[arena] = e2e.seal_authority(roots[arena], materialized, ready_path, ready, state=state,
                                           intake_db=root / "validator" / "intake.sqlite3", legacy=arena == "arena-a")
    return roots, state, sealed, key, origin


def _hotkeys(store: Path) -> dict[str, str]:
    return {row["reservation_id"]: row["hotkey"] for row in e2e.store_rows(store, "SELECT * FROM reservations")}


def _claimed(root: Path, store: Path, spool_name: str = "spool") -> list[str]:
    """The miners whose work this spool's carriers hold, in claim order."""

    hotkeys = _hotkeys(store)
    return [hotkeys[spool.load_json(carrier / "request.json")["lease"]["members"][0]["reservation_id"]]
            for carrier in e2e.carriers(root, spool_name)]


def _events(store: Path) -> list[tuple[str, str, str]]:
    hotkeys = _hotkeys(store)
    return [(event["event_type"], hotkeys[row["reservation_id"]], event["reason"])
            for row in e2e.store_rows(store, "SELECT * FROM settlement_events ORDER BY sequence")
            for event in (json.loads(row["event_json"]),)]


def _answer(root: Path, sealed, pair, decision: QualificationDecision, spool_name: str = "spool",
            speedup: str = "1.06") -> None:
    """The pod seals the pair's open request; the pair imports it and moves on."""

    request = pair.until(e2e.waiting)["request_id"]
    pod_answer(root, sealed, e2e.carriers(root, spool_name)[-1], decision, speedup=speedup)
    pair.until(lambda status: status["request_id"] != request or status["phase"] == "idle")


def test_pairs_and_arenas_qualify_at_once_and_settle_each_arrival_prefix(tmp_path, monkeypatch, supervisors):
    roots, state, sealed, key, origin = _deploy(tmp_path, monkeypatch, MINERS)
    root_a, root_b = roots["arena-a"], roots["arena-b"]
    validator_a, validator_b = root_a / "validator", root_b / "validator"
    economics = e2e.reseal(validator_a / "standing.json", validator_a / "economics.json", enable_qualification=False)
    pair_one = e2e.reseal(validator_a / "standing.json", validator_a / "standing.json", enable_settlement=False)
    dispatcher_two = e2e.reseal(validator_a / "dispatcher.json", validator_a / "dispatcher-pair-2.json",
                                owner="e2e-arena-a-pair-2", spool_root=str(e2e.private(root_a / "spool-pair-2")))
    pair_two = e2e.reseal(pair_one, validator_a / "standing-pair-2.json", screen_dispatcher_config=str(dispatcher_two),
                          qualification_evidence_root=str(e2e.private(validator_a / "evidence-pair-2")))
    named_pair = e2e.reseal(validator_b / "standing.json", validator_b / "standing.json", enable_settlement=False)

    with (e2e.relay(root_a, sealed["arena-a"]), e2e.relay(root_a, sealed["arena-a"], "spool-pair-2"),
          e2e.relay(root_b, sealed["arena-b"])):
        # Both default-arena pairs and the named arena's pair claim at once, each its arena's oldest work.
        pairs = {"spool": supervisors(root_a, pair_one), "spool-pair-2": supervisors(root_a, pair_two)}
        named = supervisors(root_b, named_pair)
        held = {name: pair.until(e2e.waiting)["request_id"] for name, pair in pairs.items()}
        named.until(e2e.waiting)
        assert sorted(_claimed(root_a, root_a) + _claimed(root_a, root_a, "spool-pair-2")) == ["a-first", "a-second"]
        assert _claimed(root_b, root_a) == ["b-first"]
        # A pair killed mid-request reopens its own request on restart, and nothing else.
        assert pairs["spool-pair-2"].stop(signal.SIGKILL) == -signal.SIGKILL
        pairs["spool-pair-2"] = supervisors(root_a, pair_two)
        assert pairs["spool-pair-2"].until(e2e.waiting)["request_id"] == held["spool-pair-2"]
        assert len(e2e.carriers(root_a)) == len(e2e.carriers(root_a, "spool-pair-2")) == 1
        holder = {_claimed(root_a, root_a, name)[0]: name for name in pairs}

        # The later default-arena PASS completes first and waits for a-first; the named arena's settles at once.
        settle = supervisors(root_a, economics)
        name = holder["a-second"]
        _answer(root_a, sealed["arena-a"], pairs[name], PASS, name, MINERS["a-second"][2])
        _answer(root_b, sealed["arena-b"], named, PASS, speedup=MINERS["b-first"][2])
        settle.until(lambda status: status["last_stage"] == "settlement")
        assert _events(root_a) == [("CROWN", "b-first", "qualified_win"),
                                   *((kind, "b-first", why) for kind, why in CROWNED)]
        # a-first completes the prefix: both settle in arrival order, in the default arena's own lineage.
        _answer(root_a, sealed["arena-a"], pairs[holder["a-first"]], PASS, holder["a-first"], MINERS["a-first"][2])
        settle.until(lambda status: status["last_stage"] == "settlement" and len(_events(root_a)) == 10)
        # a-third's worker HOLD parks it without a verdict and frees its pair; then the validator pauses.
        name = holder["a-second"]
        pairs[name].until(e2e.waiting)
        pod_hold(root_a, sealed["arena-a"], e2e.carriers(root_a, name)[-1])
        pairs[name].until(lambda status: status["phase"] == "idle", allow={"hold"})
        _answer(root_b, sealed["arena-b"], named, FAIL)
        for supervisor in (*pairs.values(), named, settle):
            assert supervisor.stop(signal.SIGTERM) == 0
    assert _events(root_a)[3:] == [
        ("CROWN", "a-first", "qualified_win"), *((kind, "a-first", why) for kind, why in CROWNED),
        ("CROWN", "a-second", "qualified_ancestor_win"), ("RETIREMENT", "a-second", "superseded"),
        *((kind, "a-second", why) for kind, why in CROWNED)]
    # One signer pays both arenas' crowns; each earns over its own arena's earlier record only.
    sign = roles.Signer(root_a, state)
    record = {hotkey: (speedup, block) for hotkey, (_, block, speedup) in (*MINERS.items(), LATE)}
    credits = {"a_first": roles.reward_credit(record["a-first"], None, 110, None),
               "a_second": roles.reward_credit(record["a-second"], record["a-first"], 110, None),
               "b_first": roles.reward_credit(record["b-first"], None, 110, None)}

    # A commitment finalized after the crowns, while the validator is paused, is claimed on resume;
    # its PASS waits behind the held a-third, earning and settling nothing until that resolves.
    late_hash, late_archive = e2e.package(e2e.miner_bundle(root_a, LATE[0], e2e.labelled(LATE[0])), key)
    chain = json.loads(state.read_text())
    amend(state, block=110, hotkeys=[*chain["hotkeys"], LATE[0]], revealed={**chain["revealed"], LATE[0]: [
        list(e2e.reveal(LATE[0], late_hash, late_archive, origin, LATE[1][1]))]})
    assert e2e.intake(root_a, state, origin, monkeypatch) == [late_hash]
    pair_config = {"spool": pair_one, "spool-pair-2": pair_two}[name]
    with e2e.relay(root_a, sealed["arena-a"], name):
        pair, settle = supervisors(root_a, pair_config), supervisors(root_a, economics)
        _answer(root_a, sealed["arena-a"], pair, PASS, name, LATE[1][2])
        resumed = int(time.time())
        settle.until(lambda status: status["phase"] == "idle" and status["time_unix"] > resumed)
        for supervisor in (pair, settle):
            assert supervisor.stop(signal.SIGTERM) == 0
    assert len(_events(root_a)) == 10
    assert sign("--dry-run") == [] and sign.shares() == roles.reward_shares(**credits)

    # The operator returns a-third to its queue; its FAIL completes the prefix and a-late settles and earns.
    (third,) = (row for row in e2e.store_rows(root_a, "SELECT * FROM reservations") if row["hotkey"] == "a-third")
    assert third["status"] == "held" and "status=published" in e2e.run(
        "cacheon.cli", "chain-release-hold", "--netuid", str(e2e.NETUID), "--network", str(state),
        "--intake-db", str(root_a / "validator" / "intake.sqlite3"), "--reservation-id", third["reservation_id"],
        "--reason", "worker repaired", env=e2e.chain_env())
    with e2e.relay(root_a, sealed["arena-a"], name):
        pair, settle = supervisors(root_a, pair_config), supervisors(root_a, economics)
        _answer(root_a, sealed["arena-a"], pair, FAIL, name)
        settle.until(lambda status: status["last_stage"] == "settlement" and len(_events(root_a)) == 14)
        for supervisor in (pair, settle):
            assert supervisor.stop(signal.SIGTERM) == 0
    assert _events(root_a)[10:] == [("CROWN", "a-late", "qualified_ancestor_win"),
                                    ("RETIREMENT", "a-late", "superseded"),
                                    *((kind, "a-late", why) for kind, why in CROWNED)]
    assert {row["hotkey"]: (row["status"], row["decision"]) for row in e2e.store_rows(
        root_a, "SELECT * FROM reservations")} == {
        "a-first": ("qualified", "PASS"), "a-second": ("qualified", "PASS"), "b-first": ("qualified", "PASS"),
        "a-third": ("failed", "FAIL"), "b-second": ("failed", "FAIL"), "a-late": ("qualified", "PASS")}
    assert sign("--dry-run") == [] and sign.shares() == roles.reward_shares(
        **credits, a_late=roles.reward_credit(record["a-late"], record["a-second"], 110, None))

    # The legacy alias is the default arena's: the named arena cannot take it over.
    legacy = e2e.reseal(validator_b / "dispatcher.json", validator_b / "dispatcher-legacy.json",
                        accept_legacy_bundles=True)
    claimant = supervisors(root_b, e2e.reseal(named_pair, validator_b / "standing-legacy.json",
                                              screen_dispatcher_config=str(legacy)))
    assert claimant.until(lambda status: status["phase"] == "failed", allow={"failed"})
    assert claimant.process.wait(timeout=60) == 1
    assert "legacy bundles already belong to another arena" in (root_b / "supervisor.stderr").read_text()


def test_a_recommissioned_arena_claims_the_work_its_retired_commission_held(tmp_path, monkeypatch, supervisors):
    """2026-09-06: a submission held under the retired commission wedged the queue after a redeploy."""

    roots, state, sealed, *_ = _deploy(tmp_path / "retired", monkeypatch, {"a-held": ("arena-a", 90, None)})
    retired, store = roots["arena-a"], roots["arena-a"]
    with e2e.relay(retired, sealed["arena-a"]):
        supervisor = supervisors(retired)
        supervisor.until(e2e.waiting)
        pod_hold(retired, sealed["arena-a"], e2e.carriers(retired)[0])
        supervisor.until(lambda status: status["phase"] == "hold", allow={"hold"})
        assert supervisor.stop(signal.SIGTERM) == 0
    (held,) = e2e.store_rows(store, "SELECT * FROM reservations")
    assert held["status"] == "held"

    # The operator redeploys the arena on new worker bytes, then returns the held submission to its queue.
    live = e2e.private(tmp_path / "live" / "arena-a")
    e2e.private(live / "validator")
    materialized, ready_path, ready = e2e.commission(live, SimpleNamespace(**{**vars(H100), "sglang": "0.5.20"}))
    commissioned = e2e.seal_authority(live, materialized, ready_path, ready, state=state,
                                      intake_db=store / "validator" / "intake.sqlite3", legacy=True)
    assert commissioned.incumbent.arena_digest != sealed["arena-a"].incumbent.arena_digest
    assert "status=published" in e2e.run(
        "cacheon.cli", "chain-release-hold", "--netuid", str(e2e.NETUID), "--network", str(state),
        "--intake-db", str(store / "validator" / "intake.sqlite3"), "--reservation-id", held["reservation_id"],
        "--reason", "operator redeployed the worker", env=e2e.chain_env())
    # The live commission claims it and measures it against its own incumbent (``pod_request`` checks).
    with e2e.relay(live, commissioned):
        supervisor = supervisors(live)
        supervisor.until(e2e.waiting)
        pod_answer(live, commissioned, e2e.carriers(live)[0], FAIL)
        supervisor.until(lambda status: status["phase"] == "idle")
        assert supervisor.stop(signal.SIGTERM) == 0
    (row,) = e2e.store_rows(store, "SELECT * FROM reservations")
    assert (row["status"], row["decision"]) == ("failed", "FAIL")
    (segment,) = e2e.store_rows(store, "SELECT * FROM reservation_baseline_segments")
    assert (segment["arena_id"], segment["binding_reason"]) == (
        commissioned.incumbent.arena_digest, "commissioned_incumbent")

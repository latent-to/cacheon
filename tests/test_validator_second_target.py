"""A second registered target, end to end: the prefix cache crowns beside the forward-pass crown.

The arena registers ``forward_pass`` and ``prefix_cache`` with radix caching on, as the tracked
arena prompts do. ``tests/support/e2e.py`` builds the deployment root, the standing CPU
supervisor process (``python -m cacheon.chain.standing_cpu_supervisor``) qualifies and settles,
and ``tests/support/pod.py`` plays the pod. After the first crown the operator moves the baseline
to the durable crowned stack (``e2e.rotate``: the pod re-declares it through the commission's own
``_commissioned_stock_authority`` over the crowned entries, and the CPU is re-pointed at it). Then:

- a cache bundle measured on that baseline crowns beside the model crown, and the stack holds
  both entries;
- a two-node forward-pass bundle (``model.layers.*.mlp`` and ``model.norm``) measured on the
  same, now stale, baseline replaces the model crown while the cache crown stays: settlement
  composes the winner onto the current stack and assembles the composed tree from the
  validator's retained publications, and the pod's next rotation reproduces that tree identity
  (the race that dropped the cache crown before 2026-10-20);
- each settlement journals CROWN, ADOPTION and STACK_TRANSITION, the replacement also
  RETIREMENT, and none NEUTRALIZATION;
- the rotated baseline reopens both contributions in one engine tree under separate namespaces;
- ``set-weights --dry-run`` projects all three standing claims, each paid the log of its speedup
  over the best earlier rewarded PASS on the same measured baseline (the two-node bundle over the
  cache PASS), with the stall bonus counted from the previous crown's reveal.
"""

from __future__ import annotations

import json
import signal

import pytest

from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.eval.qualification import QualificationDecision
from cacheon.manifest import load_manifest
from tests.support import e2e
from tests.support import weight_roles as roles
from tests.support.chain import advance
from tests.support.pod import pod_answer

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
# Finalized order: reveal block, the head that finalizes it, and the speedup its PASS measures on its baseline.
RECORDS = {"miner-model": (90, 100, "1.06"), "miner-cache": (105, 110, "1.04"), "miner-nodes": (115, 120, "1.1")}
CROWNED = (("ADOPTION", "adopted"), ("STACK_TRANSITION", "incumbent_updated"))


def _nodes(root):
    """A forward-pass bundle that replaces two nodes from one source."""

    bundle = e2e.miner_bundle(root, "miner-nodes", "def mlp(module, *args, **kwargs):\n"
                              "    return module.forward(*args, **kwargs)\n\n\n"
                              "def norm(module, *args, **kwargs):\n    return module.forward(*args, **kwargs)\n")
    manifest = bundle / "manifest.toml"
    manifest.write_text(manifest.read_text().replace('entry = "forward"', 'entry = "mlp"')
                        + '\n[[ops]]\nslot = "model.norm"\nsource = "kernels/forward.py"\nentry = "norm"\n')
    return bundle


def _stack(root, sealed) -> dict[str, str]:
    """The durable evaluation stack, each target named by its crowned miner."""

    hotkeys = {row["content_hash"]: row["hotkey"] for row in e2e.store_rows(root, "SELECT * FROM reservations")}
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=sealed.scope) as store:
        stack = store.evaluation_stack(sealed.incumbent.arena_digest)
    return {target: hotkeys[ref.artifact_digest] for target, ref in stack.manifest.entries.items()}


def test_a_prefix_cache_crown_stands_beside_the_forward_pass_crown_and_outlives_its_replacement(
        tmp_path, monkeypatch, supervisors):
    root, key = e2e.deployment_root(tmp_path, "second-target", monkeypatch)
    bundles = {"miner-model": e2e.miner_bundle(root, "miner-model", e2e.labelled("miner-model")),
               "miner-cache": e2e.cache_bundle(root, "miner-cache"), "miner-nodes": _nodes(root)}
    reveals, origin = {}, {}
    for hotkey, (block, _, _) in RECORDS.items():
        content_hash, archive = e2e.package(bundles[hotkey], key)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin, block=block),)
    state = e2e.chain(root, reveals, tuple(RECORDS))
    materialized, ready_path, ready = e2e.commission(root, e2e.PROFILES["h100-tp1"], ("forward_pass", "prefix_cache"))
    sealed = e2e.seal_authority(root, materialized, ready_path, ready)
    assert sealed.manifest["closed_targets"] == []

    stacks, baselines = [], []
    for hotkey, (_, head, speedup) in RECORDS.items():
        if len(stacks) == 1:  # the operator moves the baseline to the first crown; the third PASS is measured on it too
            sealed = e2e.rotate(root, sealed)
        advance(state, head)
        assert len(e2e.intake(root, state, origin, monkeypatch)) == 1
        baselines.append(sealed.incumbent.digest)
        with e2e.relay(root, sealed):
            supervisor = supervisors(root)
            supervisor.until(e2e.waiting)
            pod_answer(root, sealed, e2e.carriers(root)[-1], QualificationDecision.PASS, speedup=speedup)
            supervisor.until(lambda status: status["last_stage"] == "settlement"
                             and status["last_disposition"] == "committed")
            assert supervisor.stop(signal.SIGTERM) == 0
        stacks.append(_stack(root, sealed))
    assert stacks == [{"forward_pass": "miner-model"}, {"forward_pass": "miner-model", "prefix_cache": "miner-cache"},
                      {"forward_pass": "miner-nodes", "prefix_cache": "miner-cache"}]

    hotkeys = {row["reservation_id"]: row["hotkey"] for row in e2e.store_rows(root, "SELECT * FROM reservations")}
    events = [(hotkeys[row["reservation_id"]], json.loads(row["event_json"]))
              for row in e2e.store_rows(root, "SELECT * FROM settlement_events ORDER BY sequence")]
    assert [(event["event_type"], hotkey, event["target_id"], event["reason"]) for hotkey, event in events] == [
        ("CROWN", "miner-model", "forward_pass", "qualified_win"),
        *((kind, "miner-model", "forward_pass", why) for kind, why in CROWNED),
        ("CROWN", "miner-cache", "prefix_cache", "qualified_win"),
        *((kind, "miner-cache", "prefix_cache", why) for kind, why in CROWNED),
        ("CROWN", "miner-nodes", "forward_pass", "qualified_win"),
        ("RETIREMENT", "miner-nodes", "forward_pass", "superseded"),
        *((kind, "miner-nodes", "forward_pass", why) for kind, why in CROWNED)]
    # Each crown advances from the current stack, while the PASS it crowns was measured on its own baseline:
    # the third on the stale {model} baseline while the stack already held the cache crown.
    transitions = [event["to_stack_digest"] for _, event in events if event["event_type"] == "STACK_TRANSITION"]
    assert [event["from_stack_digest"] for _, event in events if event["event_type"] == "CROWN"] == [baselines[0], *transitions[:2]]
    measured = {hotkeys[row["reservation_id"]]: json.loads(row["qualification_json"])["incumbent_stack_digest"]
                for row in e2e.store_rows(root, "SELECT reservation_id, qualification_json FROM settlement_qualifications")}
    assert [measured[hotkey] for hotkey in RECORDS] == baselines and baselines[1] == baselines[2] != baselines[0]

    # The pod's rotation reproduces the validator's composition identity. This CPU proof does not execute
    # the combined stack; the rotated baseline reopens both contributions under two namespaces.
    rotated = e2e.rotate(root, sealed)
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=sealed.scope) as store:
        durable = store.evaluation_stack(sealed.incumbent.arena_digest)
    assert (rotated.incumbent.digest, rotated.incumbent_tree.tree_digest) == (durable.manifest.digest, durable.tree_digest)
    ops = load_manifest(rotated.incumbent_tree.root).ops
    assert {op.slot for op in ops} == {"model.layers.*.mlp", "model.norm", "tree_cache"}
    assert len({op.source.split("/")[1] for op in ops}) == 2

    sign = roles.Signer(root, state)
    assert sign("--dry-run") == [] and sign.shares() == roles.reward_shares(
        miner_model=roles.reward_credit(("1.06", 90), None, 120, None),
        miner_cache=roles.reward_credit(("1.04", 105), ("1", 90), 120, None),
        miner_nodes=roles.reward_credit(("1.1", 115), ("1.04", 105), 120, None))

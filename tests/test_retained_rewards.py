"""Historical arena policy remains payable through the durable weight consumer."""

from copy import deepcopy
from dataclasses import replace

import pytest

from cacheon.economics import EconomicsError
from cacheon.stack_identity import canonical_digest
from cacheon.target_catalog import TargetCatalog
from tests import test_chain_intake as intake
from tests.test_economics import (
    _catalog, _claim, _global_context, _policy, _project, _sealed_stack, _slot,
)


def _snapshot(schema=1):
    snapshot = _catalog().snapshot()
    if schema == 1:
        snapshot.update(schema_version=1, policy_version="target-catalog.v1", composition_rules=[])
        for row in snapshot["targets"]:
            row.pop("conflicts_with")
            row["compatible_with"] = []
    return snapshot


def _project_snapshot(snapshot, targets=("slot.a", "slot.b")):
    stack = _sealed_stack(_catalog(), snapshot, targets)
    return _project(_policy(), stack, _global_context(), tuple(
        _claim(stack, target, ("alice", "bob", "carol")[index], 1_100_000 + index * 100_000)
        for index, target in enumerate(targets)
    ))


def _rows(snapshot):
    return {row["target_id"]: row for row in snapshot["targets"]}


def _compatible_snapshot():
    snapshot = _snapshot()
    rows = _rows(snapshot)
    for target, other in (("slot.a", "slot.b"), ("slot.b", "slot.a")):
        rows[target]["compatible_with"] = [other]
        rows[target]["contract_ref"]["binding_family_id"] = "joint.binding.v1"
    snapshot["composition_rules"] = [{
        "schema_version": 1,
        "rule_id": "joint.first.v1",
        "target_ids": ["slot.a", "slot.b"],
        "precedence": ["slot.b", "slot.a"],
        "mode": "first_applicable",
        "binding_family_id": "joint.binding.v1",
        # Historical v1 validates format, not a recomputed binding hash.
        "binding_contract_digest": "0" * 64,
    }]
    return snapshot


def test_historical_active_composition_keeps_one_global_reward_normalization():
    result = _project_snapshot(_compatible_snapshot())
    assert result.weights_by_hotkey == {"alice": 343_297, "bob": 656_703}


@pytest.mark.parametrize("schema", (1, 2))
@pytest.mark.parametrize("failure", ("requires", "transitive-displacement", "cycle", "members"))
def test_active_sealed_relationships_remain_enforced(schema, failure):
    snapshot = _snapshot(schema)
    rows = _rows(snapshot)
    if failure == "requires":
        rows["slot.a"]["requires"] = ["atomic.ab"]
    elif failure == "transitive-displacement":
        rows["slot.a"]["displaces"] = ["atomic.ab"]
        rows["atomic.ab"]["displaces"] = ["slot.b"]
    elif failure == "cycle":
        rows["slot.a"]["requires"] = ["atomic.ab"]
        rows["atomic.ab"]["requires"] = ["slot.a"]
    else:
        rows["slot.b"]["members"] = ["slot.a"]
    with pytest.raises(EconomicsError, match="active reward families"):
        _project_snapshot(snapshot)


def test_active_v2_conflicts_and_unknown_policy_are_rejected():
    snapshot = _snapshot(2)
    _rows(snapshot)["slot.a"]["conflicts_with"] = ["slot.b"]
    with pytest.raises(EconomicsError, match="conflicts_with"):
        _project_snapshot(snapshot)
    snapshot["policy_version"] = "target-catalog.v0"
    with pytest.raises(EconomicsError, match="catalog policy"):
        _project_snapshot(snapshot)


@pytest.mark.parametrize("failure", ("missing", "duplicate", "asymmetric", "family", "precedence"))
def test_historical_active_composition_cannot_be_reinterpreted(failure):
    snapshot = _compatible_snapshot()
    rule = snapshot["composition_rules"][0]
    if failure == "missing":
        snapshot["composition_rules"].clear()
    elif failure == "duplicate":
        snapshot["composition_rules"].append(deepcopy(rule))
    elif failure == "asymmetric":
        _rows(snapshot)["slot.a"]["compatible_with"] = []
    elif failure == "family":
        rule["binding_family_id"] = "other.binding.v1"
    else:
        rule["precedence"] = ["slot.a", "slot.a"]
    with pytest.raises(EconomicsError, match="active reward families"):
        _project_snapshot(snapshot)


def _commissioning_catalog(schema, target_id="activation.silu_and_mul"):
    """Construct the same small fixture as a historical v1 or present v2 issuer."""
    target = _slot(target_id)
    target = replace(target, contract_ref=replace(
        target.contract_ref, reference_id=f"model.schema{schema}.reference.v1"
    ))
    catalog = TargetCatalog((target,))
    if schema == 1:
        snapshot = catalog.snapshot()
        snapshot.update(schema_version=1, policy_version="target-catalog.v1", composition_rules=[])
        snapshot["targets"][0].pop("conflicts_with")
        snapshot["targets"][0]["compatible_with"] = []
        # This fixture represents the old issuer before settlement, never a
        # migration or rewrite of retained claim/evidence bytes.
        catalog._snapshot = snapshot
        catalog._target_snapshots = {target.target_id: snapshot["targets"][0]}
        catalog._digest = canonical_digest("cacheon.target-catalog", snapshot)
    return catalog


def test_mixed_catalogs_reopen_and_pay_through_the_store_without_requalification(tmp_path, monkeypatch):
    context = intake._context("validator", "minerold", "minernew")
    with intake._store(tmp_path) as store:
        for index, (schema, marker, target) in enumerate((
            (1, "old", "activation.silu_and_mul"), (2, "new", "norm.rmsnorm"),
        )):
            catalog = _commissioning_catalog(schema, target)
            monkeypatch.setattr(intake, "default_target_catalog", lambda: catalog)
            intake._qualified_settlement_candidate(
                store, index=index, marker=marker, arena_marker=marker, check_single_pass=False,
                target=target, speedups=("1.05", "1.04") if schema == 1 else ("1.1", "1.09"),
            )
            lease = store.lease_settlement_cohort(current_block=11)
            plan, evidence = intake._settlement_plan(store, lease)
            store.commit_settlement(lease, plan, evidence, current_block=11)
        store.initialize_evaluation_stack(
            intake._staging_manifest("next-glm", _commissioning_catalog(2)),
            tree_digest=intake._h("next-glm-tree"),
        )
        projection = store.build_weight_projection(
            policy=intake.POLICY, context=context, netuid=intake.SCOPE.netuid
        )
        assert projection.crown_count == 2
        weights = dict(projection.weights_ppm)
        assert weights["minernew"] > weights["minerold"] > 0
        assert sum(weights.values()) == 1_000_000
        assert len(projection.arena_state_digests) == 2

    with intake._store(tmp_path) as reopened:
        again = reopened.build_weight_projection(
            policy=intake.POLICY, context=context, netuid=intake.SCOPE.netuid
        )
        assert again == projection
        # Lost retained evidence still stops the entire vector after restart.
        ref = evidence[0].primary_attempt_ref
        (reopened.path.parent / "evidence" / ref.domain / ref.sha256[:2] / ref.sha256).unlink()
        with pytest.raises(intake.IntakeError, match="cannot reopen"):
            reopened.build_weight_projection(
                policy=intake.POLICY, context=context, netuid=intake.SCOPE.netuid
            )

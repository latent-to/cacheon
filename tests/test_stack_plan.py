from __future__ import annotations

from dataclasses import replace

import pytest

from cacheon.stack_identity import sha256_hex
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import StackPlanError, plan_marginal_arm
from cacheon.target_catalog import TargetCatalog, default_target_catalog


FORWARD = "forward_pass"
CACHE = "prefix_cache"


def test_a_transition_preserves_the_incumbent_of_the_other_target():
    # Model and cache replacements are distinct stack entries (2026-09-28): a
    # cache candidate must keep the incumbent model kernels, and vice versa.
    catalog = default_target_catalog()
    entries = {FORWARD: _ref(catalog, FORWARD, "model")}
    incumbent = _stack(catalog, entries)
    arm = _plan(incumbent, _ref(catalog, CACHE, "cache"), catalog,
                _context(catalog, (FORWARD, CACHE)))
    assert set(arm.candidate.entries) == {FORWARD, CACHE}
    assert not arm.transition.displaced
    assert arm.candidate.entries[FORWARD] == entries[FORWARD]
    assert incumbent.entries == entries
    assert arm.baseline_before.stack_digest == arm.baseline_after.stack_digest == incumbent.digest
    next_arm = _plan(arm.candidate, _ref(catalog, FORWARD, "replacement"), catalog,
                     _context(catalog, (FORWARD, CACHE)))
    assert next_arm.candidate.entries[CACHE] == arm.candidate.entries[CACHE]
    assert not next_arm.transition.displaced


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _context(
    catalog: TargetCatalog, target_ids: tuple[str, ...]
) -> EvaluationStackContext:
    del target_ids  # expected context always binds the complete catalog
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    return EvaluationStackContext(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in targets
        },
    )


def _stack(
    catalog: TargetCatalog,
    entries: dict[str, ProposalContributionRef] | None = None,
) -> EvaluationStackManifest:
    return EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries=entries or {},
    )


def _ref(
    catalog: TargetCatalog,
    target: str,
    label: str,
    *,
    payload: str | None = None,
) -> ProposalContributionRef:
    return ProposalContributionRef(
        target_id=target,
        target_spec_digest=catalog.target_spec_digest(target),
        artifact_digest=_h(f"artifact:{label}"),
        selected_payload_digest=_h(f"payload:{payload or label}"),
        attribution_digest=_h(f"attribution:{label}"),
    )


def _plan(
    incumbent: EvaluationStackManifest,
    replacement: ProposalContributionRef,
    catalog: TargetCatalog,
    context: EvaluationStackContext,
    *,
    incumbent_tree: str = "tree:b",
    candidate_tree: str | None = None,
):
    return plan_marginal_arm(
        incumbent,
        replacement,
        catalog=catalog,
        incumbent_tree_digest=_h(incumbent_tree),
        candidate_tree_digest=_h(
            candidate_tree or f"tree:c:{replacement.selected_delta_digest}"
        ),
        expected_context=context,
    )


@pytest.mark.parametrize(
    "initial_target,replacement_target,expected_removed",
    [
        (None, FORWARD, ()),
        (FORWARD, FORWARD, ()),
    ],
)
def test_registered_stock_and_same_target_transitions(
    initial_target, replacement_target, expected_removed
):
    catalog = default_target_catalog()
    targets = tuple(filter(None, (initial_target, replacement_target)))
    context = _context(catalog, targets)
    entries = (
        {}
        if initial_target is None
        else {initial_target: _ref(catalog, initial_target, "incumbent")}
    )
    incumbent = _stack(catalog, entries)

    arm = _plan(
        incumbent,
        _ref(catalog, replacement_target, "replacement"),
        catalog,
        context,
    )

    assert tuple(ref.target_id for ref in arm.transition.displaced) == expected_removed
    assert set(arm.candidate.entries) == {replacement_target}
    assert arm.baseline_before == arm.baseline_after
    assert arm.baseline_before is not arm.baseline_after
    assert arm.baseline_before.stack_digest == incumbent.digest
    assert arm.challenger.stack_digest == arm.candidate.digest
    assert arm.transition.prior is entries.get(replacement_target)


def test_planning_rejects_a_catalog_outside_the_frozen_stack_context():
    catalog = default_target_catalog()
    context = _context(catalog, (FORWARD,))
    incumbent = _stack(catalog)
    narrow = TargetCatalog((catalog.require(FORWARD),))

    with pytest.raises(StackPlanError, match="catalog does not match"):
        _plan(
            incumbent,
            _ref(catalog, FORWARD, "replacement"),
            narrow,
            context,
        )


def test_stale_target_spec_and_selected_payload_noop_reject():
    catalog = default_target_catalog()
    context = _context(catalog, (FORWARD,))
    prior = _ref(catalog, FORWARD, "prior", payload="same")
    incumbent = _stack(catalog, {FORWARD: prior})
    padded_alias = ProposalContributionRef(
        target_id=FORWARD,
        target_spec_digest=prior.target_spec_digest,
        artifact_digest=_h("different padding"),
        selected_payload_digest=prior.selected_payload_digest,
        attribution_digest=_h("different attribution"),
    )
    stale = replace(padded_alias, target_spec_digest=_h("stale spec"))

    with pytest.raises(StackPlanError, match="target-spec digest is stale"):
        _plan(incumbent, stale, catalog, context)
    with pytest.raises(StackPlanError, match="no executable delta"):
        _plan(incumbent, padded_alias, catalog, context)


def test_marginal_plan_rejects_equal_tree_and_detects_incumbent_rebase():
    catalog = default_target_catalog()
    context = _context(catalog, (FORWARD, CACHE))
    incumbent = _stack(catalog)
    replacement = _ref(catalog, FORWARD, "model")
    with pytest.raises(StackPlanError, match="tree digests must differ"):
        _plan(
            incumbent,
            replacement,
            catalog,
            context,
            candidate_tree="tree:b",
        )
    _plan(incumbent, replacement, catalog, context)


def test_plan_schema_version_is_type_exact():
    catalog = default_target_catalog()
    context = _context(catalog, (FORWARD, CACHE))
    model = _plan(_stack(catalog), _ref(catalog, FORWARD, "model"), catalog, context)
    with pytest.raises(StackPlanError, match="schema_version"):
        replace(model, schema_version=True)

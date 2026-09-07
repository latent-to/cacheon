from __future__ import annotations

from dataclasses import replace

import pytest

from cacheon.stack_identity import sha256_hex
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import (
    CohortPlan,
    RollbackPlan,
    StackPlanError,
    StaleStackPlanError,
    plan_marginal_arm,
)
from cacheon.target_catalog import (
    FEATURE_ENTRY,
    TargetCatalog,
    TargetKind,
    TargetSpec,
    default_target_catalog,
)


ROUTED = "moe.fused_routed_experts"
SILU = "activation.silu_and_mul"
NORM = "norm.rmsnorm"
ALLREDUCE = "collective.all_reduce"


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
        (None, ROUTED, ()),
        (ROUTED, ROUTED, ()),
        ("moe.fused_experts", "moe.fused_experts_reduce", ("moe.fused_experts",)),
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


def test_subset_challenger_replaces_wide_incumbent_instead_of_being_shadowed():
    catalog = default_target_catalog()
    for wide, narrow in (
        ("moe.fused_experts_reduce", "moe.fused_experts"),
        ("norm.fused_add_rmsnorm", "norm.rmsnorm"),
        ("collective.ar_residual_rmsnorm", "collective.all_reduce"),
    ):
        incumbent = _stack(catalog, {wide: _ref(catalog, wide, "wide")})
        arm = _plan(
            incumbent,
            _ref(catalog, narrow, "narrow"),
            catalog,
            _context(catalog, (wide, narrow)),
        )
        assert tuple(row.target_id for row in arm.transition.displaced) == (wide,)
        assert set(arm.candidate.entries) == {narrow}


def test_planning_rejects_a_catalog_outside_the_frozen_stack_context():
    catalog = default_target_catalog()
    context = _context(catalog, (ROUTED,))
    incumbent = _stack(catalog)
    narrow = TargetCatalog(
        (
            replace(
                catalog.require(ROUTED),
                displaces=frozenset(),
                conflicts_with=frozenset(),
            ),
        )
    )

    with pytest.raises(StackPlanError, match="catalog does not match"):
        _plan(
            incumbent,
            _ref(catalog, ROUTED, "replacement"),
            narrow,
            context,
        )


def _dependency_catalog() -> tuple[TargetCatalog, str]:
    base = default_target_catalog()
    atomic_id = "atomic.silu_allreduce"
    specs = [
        base.require(SILU),
        replace(base.require(NORM), requires=frozenset({SILU})),
        base.require(ALLREDUCE),
        TargetSpec(
            target_id=atomic_id,
            kind=TargetKind.ATOMIC,
            members=(SILU, ALLREDUCE),
            displaces=frozenset({SILU, ALLREDUCE}),
            allowed_features=frozenset({FEATURE_ENTRY}),
            atomic_semantics_id="silu-allreduce-atomic.v1",
        ),
    ]
    return TargetCatalog(specs), atomic_id


def test_atomic_and_member_challengers_remove_every_overlapping_incumbent():
    catalog, atomic_id = _dependency_catalog()
    atomic_incumbent = _stack(
        catalog, {atomic_id: _ref(catalog, atomic_id, "atomic incumbent")}
    )
    for challenger in (SILU, ALLREDUCE):
        arm = _plan(
            atomic_incumbent,
            _ref(catalog, challenger, f"challenger:{challenger}"),
            catalog,
            _context(catalog, (atomic_id, challenger)),
        )
        assert tuple(row.target_id for row in arm.transition.displaced) == (atomic_id,)
        assert set(arm.candidate.entries) == {challenger}

    members = {target: _ref(catalog, target, target) for target in (SILU, ALLREDUCE)}
    arm = _plan(
        _stack(catalog, members),
        _ref(catalog, atomic_id, "atomic challenger"),
        catalog,
        _context(catalog, (atomic_id, SILU, ALLREDUCE)),
    )
    assert {row.target_id for row in arm.transition.displaced} == set(members)
    assert set(arm.candidate.entries) == {atomic_id}


def test_stock_does_not_satisfy_active_only_dependency():
    catalog, _ = _dependency_catalog()
    context = _context(catalog, (SILU, NORM))
    with pytest.raises(StackPlanError, match="stock does not satisfy requires"):
        _plan(_stack(catalog), _ref(catalog, NORM, "norm"), catalog, context)

    incumbent = _stack(catalog, {SILU: _ref(catalog, SILU, "silu")})
    arm = _plan(
        incumbent, _ref(catalog, NORM, "norm"), catalog, context
    )
    assert set(arm.candidate.entries) == {SILU, NORM}


def test_displacement_rejects_stranded_active_dependent():
    catalog, atomic_id = _dependency_catalog()
    context = _context(catalog, (SILU, NORM, atomic_id))
    incumbent = _stack(
        catalog,
        {
            SILU: _ref(catalog, SILU, "silu"),
            NORM: _ref(catalog, NORM, "norm"),
        },
    )
    with pytest.raises(StackPlanError, match="stock does not satisfy requires"):
        _plan(
            incumbent,
            _ref(catalog, atomic_id, "atomic"),
            catalog,
            context,
        )


def test_stale_target_spec_and_selected_payload_noop_reject():
    catalog = default_target_catalog()
    context = _context(catalog, (ROUTED,))
    prior = _ref(catalog, ROUTED, "prior", payload="same")
    incumbent = _stack(catalog, {ROUTED: prior})
    padded_alias = ProposalContributionRef(
        target_id=ROUTED,
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
    context = _context(catalog, (ROUTED, SILU))
    incumbent = _stack(catalog)
    replacement = _ref(catalog, ROUTED, "routed")
    with pytest.raises(StackPlanError, match="tree digests must differ"):
        _plan(
            incumbent,
            replacement,
            catalog,
            context,
            candidate_tree="tree:b",
        )
    _plan(incumbent, replacement, catalog, context)


def _two_arms():
    catalog = default_target_catalog()
    context = _context(catalog, (ROUTED, SILU))
    incumbent = _stack(catalog)
    routed = _plan(incumbent, _ref(catalog, ROUTED, "routed"), catalog, context)
    silu = _plan(incumbent, _ref(catalog, SILU, "silu"), catalog, context)
    return catalog, context, incumbent, routed, silu


def test_cohort_order_is_entropy_derived_and_authority_remains_distinct():
    _, context, incumbent, routed, silu = _two_arms()
    entropy = _h("post-seal entropy")
    authority = (silu.transition.replacement, routed.transition.replacement)
    first = CohortPlan.seal(
        (routed, silu),
        entropy_digest=entropy,
        authority_order=authority,
        catalog=default_target_catalog(),
        expected_context=context,
    )
    second = CohortPlan.seal(
        (silu, routed),
        entropy_digest=entropy,
        authority_order=authority,
        catalog=default_target_catalog(),
        expected_context=context,
    )

    assert first.digest == second.digest
    assert first.execution_order == second.execution_order
    assert first.authority_order == authority
    assert set(first.execution_order) == {
        routed.selected_delta_digest,
        silu.selected_delta_digest,
    }
    assert first.reopen(
        catalog=default_target_catalog(), expected_context=context
    ) is first
    with pytest.raises(StackPlanError, match="sealed entropy"):
        replace(first, execution_order=tuple(reversed(first.execution_order)))


@pytest.mark.parametrize(
    "case",
    ["duplicate_delta", "duplicate_tree", "duplicate_authority", "missing_authority"],
)
def test_cohort_rejects_duplicate_work_and_invalid_authority(case):
    catalog, context, incumbent, routed, silu = _two_arms()
    arms = (routed, silu)
    authority = (routed.transition.replacement, silu.transition.replacement)
    message = ""
    if case == "duplicate_delta":
        alias = _ref(catalog, ROUTED, "padding alias", payload="routed")
        # _ref's payload label matches the original selected payload while the
        # whole artifact and attribution identities differ.
        duplicate = _plan(
            incumbent,
            alias,
            catalog,
            _context(catalog, (ROUTED, SILU)),
            candidate_tree="tree:alias",
        )
        arms = (routed, duplicate)
        authority = (routed.transition.replacement, duplicate.transition.replacement)
        message = "duplicate selected deltas"
    elif case == "duplicate_tree":
        duplicate_tree = _plan(
            incumbent,
            silu.transition.replacement,
            catalog,
            _context(catalog, (ROUTED, SILU)),
            candidate_tree=f"tree:c:{routed.selected_delta_digest}",
        )
        arms = (routed, duplicate_tree)
        message = "duplicate candidate trees"
    elif case == "duplicate_authority":
        authority = (routed.transition.replacement, routed.transition.replacement)
        message = "duplicate contributions"
    else:
        authority = (routed.transition.replacement,)
        message = "every cohort contribution exactly once"

    with pytest.raises(StackPlanError, match=message):
        CohortPlan.seal(
            arms,
            entropy_digest=_h("entropy"),
            authority_order=authority,
            catalog=catalog,
            expected_context=context,
        )


def test_rollback_is_exact_and_rejects_stale_stack_or_tree():
    catalog, context, _, routed, _ = _two_arms()
    rollback = RollbackPlan.from_arm(
        routed, catalog=catalog, expected_context=context
    )

    restored, restored_tree = rollback.reconstruct(
        routed.candidate,
        tree_digest=routed.challenger.tree_digest,
        source_arm=routed,
        catalog=catalog,
        expected_context=context,
    )
    assert restored.digest == routed.incumbent.digest
    assert restored_tree == routed.baseline_before.tree_digest
    with pytest.raises(StaleStackPlanError, match="current stack is stale"):
        rollback.reconstruct(
            routed.incumbent,
            tree_digest=routed.challenger.tree_digest,
            source_arm=routed,
            catalog=catalog,
            expected_context=context,
        )
    with pytest.raises(StaleStackPlanError, match="current tree is stale"):
        rollback.reconstruct(
            routed.candidate,
            tree_digest=_h("wrong tree"),
            source_arm=routed,
            catalog=catalog,
            expected_context=context,
        )

    forged_manifest = _stack(catalog)
    forged = replace(
        rollback,
        restored=replace(
            rollback.restored,
            stack_digest=forged_manifest.digest,
            tree_digest=_h("forged tree"),
        ),
        restored_manifest=forged_manifest,
    )
    with pytest.raises(StackPlanError, match="does not reopen"):
        forged.reconstruct(
            routed.candidate,
            tree_digest=routed.challenger.tree_digest,
            source_arm=routed,
            catalog=catalog,
            expected_context=context,
        )


def test_plan_schema_versions_are_type_exact():
    catalog, context, _, routed, sdpa = _two_arms()
    cohort = CohortPlan.seal(
        (routed, sdpa),
        entropy_digest=_h("entropy"),
        authority_order=(routed.transition.replacement, sdpa.transition.replacement),
        catalog=catalog,
        expected_context=context,
    )
    rollback = RollbackPlan.from_arm(
        routed, catalog=catalog, expected_context=context
    )
    for record in (routed, cohort, rollback):
        with pytest.raises(StackPlanError, match="schema_version"):
            replace(record, schema_version=True)

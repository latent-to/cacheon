from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.settlement import (
    ResidentLaneOrientation,
    SettlementCandidate,
    SettlementError,
    SettlementEvidence,
    SettlementEvent,
    SettlementEventType,
    SettlementQualification,
    TargetLineage,
    TargetLineageNode,
    plan_settlement,
)
from cacheon.stack_identity import canonical_digest, sha256_hex
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import plan_marginal_arm
from cacheon.target_catalog import TargetCatalog, default_target_catalog


ROUTED = "moe.fused_routed_experts"
SILU = "activation.silu_and_mul"
RESIDENT_SPEED_POLICY_DIGEST = canonical_digest(
    "cacheon.qualification.speed-evidence-policy",
    {
        "candidate_reads": 0,
        "estimator": "resident-adaptive-bcbp-v1",
        "version": 3,
    },
)


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _audit_policy(label: str, slots: tuple[str, ...]) -> SlotAuditPolicy:
    return SlotAuditPolicy(_h(f"audit-seed:{label}")[:32], 100_000, 32, slots, 1)


def _context(catalog: TargetCatalog) -> EvaluationStackContext:
    rows = catalog.snapshot()["targets"]
    assert isinstance(rows, list)
    return EvaluationStackContext(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in rows
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


def _ref(catalog: TargetCatalog, target: str, label: str) -> ProposalContributionRef:
    return ProposalContributionRef(
        target_id=target,
        target_spec_digest=catalog.target_spec_digest(target),
        artifact_digest=_h(f"artifact:{label}"),
        selected_payload_digest=_h(f"payload:{label}"),
        attribution_digest=_h(f"attribution:{label}"),
    )


def _candidate(
    incumbent: EvaluationStackManifest,
    replacement: ProposalContributionRef,
    catalog: TargetCatalog,
    *,
    label: str,
    speedup: str = "1.05",
    block: int = 10,
    event: int = 0,
) -> SettlementCandidate:
    plan = plan_marginal_arm(
        incumbent,
        replacement,
        catalog=catalog,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h(f"candidate-tree:{label}"),
        expected_context=_context(catalog),
    )
    members = catalog.require(replacement.target_id).members
    audit_slots = tuple(sorted(members))
    primary_audit = _audit_policy(f"primary:{label}", audit_slots)
    primary = SettlementQualification(
        lane="registered",
        arena_digest=incumbent.arena_digest,
        reservation_digest=_h(f"reservation:{label}"),
        finalized_block=block,
        event_index=event,
        event_subindex=0,
        hotkey=f"miner-{label}",
        target_id=replacement.target_id,
        members=tuple(sorted(members)),
        selected_delta_digest=plan.selected_delta_digest,
        qualification_authority_digest=_h(f"authority:{label}"),
        qualification_plan_digest=_h(f"plan-authority:{label}"),
        qualification_attempt_digest=_h(f"attempt:{label}"),
        qualification_report_digest=_h(f"report:{label}"),
        selection_commitment_digest=_h(f"selection-commitment:{label}"),
        selection_secret_commitment_digest=_h(f"selection-secret:{label}"),
        selection_evidence_digest=_h(f"selection-evidence:{label}"),
        arm_digest=plan.digest,
        incumbent_stack_digest=plan.baseline_before.stack_digest,
        incumbent_tree_digest=plan.baseline_before.tree_digest,
        candidate_stack_digest=plan.challenger.stack_digest,
        candidate_tree_digest=plan.challenger.tree_digest,
        speedup=speedup,
        incumbent_manifest=incumbent,
        candidate_manifest=plan.candidate,
        audit_control_digest=primary_audit.control.digest,
        audit_policy=primary_audit,
        audit_evidence_digest=_h(f"audit-evidence:{label}"),
    )
    reproduction_audit = _audit_policy(f"reproduction:{label}", audit_slots)
    reproduction = replace(
        primary,
        qualification_authority_digest=_h(f"reproduction-authority:{label}"),
        qualification_plan_digest=_h(f"reproduction-plan-authority:{label}"),
        qualification_attempt_digest=_h(f"reproduction-attempt:{label}"),
        qualification_report_digest=_h(f"reproduction-report:{label}"),
        selection_commitment_digest=_h(f"reproduction-selection-commitment:{label}"),
        selection_secret_commitment_digest=_h(f"reproduction-selection-secret:{label}"),
        selection_evidence_digest=_h(f"reproduction-selection-evidence:{label}"),
        audit_policy=reproduction_audit,
        audit_evidence_digest=_h(f"reproduction-audit-evidence:{label}"),
        speedup=("1.04" if speedup == "1.05" else speedup),
    )
    return SettlementCandidate.from_reproductions(primary, reproduction)


def test_candidate_json_round_trip_and_digest_are_canonical() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(_stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a")
    reopened = SettlementCandidate.from_dict(candidate.to_dict())
    assert reopened == candidate
    assert reopened.digest == candidate.digest
    assert candidate.speedup == "1.04"
    with pytest.raises(SettlementError, match="canonical decimal"):
        replace(candidate.primary, speedup="1.050")
    with pytest.raises(SettlementError, match="target/delta"):
        replace(candidate.primary, selected_delta_digest=_h("other"))


def _resident_orientation(
    baseline: str = "lane-a",
    candidate: str = "lane-b",
    *,
    control: str = "resident-lane-control",
) -> ResidentLaneOrientation:
    return ResidentLaneOrientation(
        RESIDENT_SPEED_POLICY_DIGEST,
        _h(control),
        _h(baseline),
        _h(candidate),
    )


def test_resident_reproduction_requires_exact_physical_lane_role_swap() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "resident"), catalog,
        label="resident",
    )
    primary_orientation = _resident_orientation()
    reproduction_orientation = _resident_orientation("lane-b", "lane-a")
    primary = replace(
        candidate.primary,
        speed_evidence_policy_digest=(
            primary_orientation.speed_evidence_policy_digest
        ),
        resident_lane_orientation=primary_orientation,
    )
    reproduction = replace(
        candidate.reproduction,
        speed_evidence_policy_digest=(
            reproduction_orientation.speed_evidence_policy_digest
        ),
        resident_lane_orientation=reproduction_orientation,
    )

    resident = SettlementCandidate.from_reproductions(primary, reproduction)
    assert SettlementCandidate.from_dict(resident.to_dict()) == resident
    assert resident.primary.resident_lane_orientation is not None
    assert resident.reproduction.resident_lane_orientation is not None
    assert resident.primary.resident_lane_orientation.digest != (
        resident.reproduction.resident_lane_orientation.digest
    )

    with pytest.raises(SettlementError, match="did not swap physical TP lane"):
        SettlementCandidate.from_reproductions(
            primary,
            replace(
                candidate.reproduction,
                speed_evidence_policy_digest=(
                    primary_orientation.speed_evidence_policy_digest
                ),
                resident_lane_orientation=primary_orientation,
            ),
        )
    with pytest.raises(SettlementError, match="did not swap physical TP lane"):
        SettlementCandidate.from_reproductions(
            primary,
            replace(
                candidate.reproduction,
                speed_evidence_policy_digest=(
                    reproduction_orientation.speed_evidence_policy_digest
                ),
                resident_lane_orientation=replace(
                    reproduction_orientation,
                    control_digest=_h("other-control"),
                ),
            ),
        )
    with pytest.raises(SettlementError, match="physical lane orientation"):
        replace(
            candidate.reproduction,
            speed_evidence_policy_digest=(
                primary_orientation.speed_evidence_policy_digest
            ),
        )


def test_auditless_resident_acceptance_round_trips_exact_wire_shape() -> None:
    """A v6 resident acceptance carries orientation and no audit witness.

    This is the exact production shape that wedged mainnet request
    ``0fb58834…`` on 2026-08-15: ``to_dict`` omitted the audit triple while
    emitting ``resident_lane_orientation`` and ``from_dict`` refused it.
    """

    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "resident"), catalog, label="resident"
    )
    orientation = _resident_orientation()
    fields = SettlementQualification.__dataclass_fields__  # type: ignore[attr-defined]
    auditless = replace(
        candidate.primary,
        speed_evidence_policy_digest=orientation.speed_evidence_policy_digest,
        resident_lane_orientation=orientation,
        audit_control_digest=fields["audit_control_digest"].default,
        audit_policy=None,
        audit_evidence_digest=fields["audit_evidence_digest"].default,
    )
    wire = auditless.to_dict()
    assert "resident_lane_orientation" in wire
    assert not {"audit_policy", "audit_control_digest", "audit_evidence_digest"} & set(wire)
    reopened = SettlementQualification.from_dict(wire)
    assert reopened == auditless
    assert reopened.digest == auditless.digest


def test_auditless_resident_pair_becomes_settlement_candidate() -> None:
    """Two auditless resident acceptances on swapped lanes form a candidate.

    Exact production shape of mainnet reservation ``87982705…`` on
    2026-08-15: primary and lane-swapped reproduction both PASS without an
    audit witness; ``from_reproductions`` must accept the pair, while an
    auditless pair without lane orientation stays legacy-only.
    """

    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "resident"), catalog, label="resident"
    )
    fields = SettlementQualification.__dataclass_fields__  # type: ignore[attr-defined]
    auditless = {
        "audit_control_digest": fields["audit_control_digest"].default,
        "audit_policy": None,
        "audit_evidence_digest": fields["audit_evidence_digest"].default,
    }
    primary_orientation = _resident_orientation()
    reproduction_orientation = _resident_orientation("lane-b", "lane-a")
    primary = replace(
        candidate.primary,
        speed_evidence_policy_digest=primary_orientation.speed_evidence_policy_digest,
        resident_lane_orientation=primary_orientation,
        **auditless,
    )
    reproduction = replace(
        candidate.reproduction,
        speed_evidence_policy_digest=(
            reproduction_orientation.speed_evidence_policy_digest
        ),
        resident_lane_orientation=reproduction_orientation,
        **auditless,
    )
    pair = SettlementCandidate.from_reproductions(primary, reproduction)
    assert pair.primary.audit_policy is None
    assert pair.reproduction.audit_policy is None
    assert SettlementCandidate.from_dict(pair.to_dict()) == pair
    assert pair.speedup == min(primary.speedup, reproduction.speedup, key=Decimal)
    with pytest.raises(SettlementError, match="requires two audited"):
        SettlementCandidate.from_reproductions(
            replace(primary, resident_lane_orientation=None,
                    speed_evidence_policy_digest=candidate.primary.speed_evidence_policy_digest),
            replace(reproduction, resident_lane_orientation=None,
                    speed_evidence_policy_digest=candidate.reproduction.speed_evidence_policy_digest),
        )


def test_resident_lane_orientation_is_registered_and_nonoverlapping() -> None:
    with pytest.raises(SettlementError, match="reused one physical TP lane"):
        _resident_orientation("lane-a", "lane-a")
    with pytest.raises(SettlementError, match="all-zero"):
        ResidentLaneOrientation(
            "0" * 64,
            _h("resident-lane-control"),
            _h("lane-a"),
            _h("lane-b"),
        )

    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "resident-policy"), catalog,
        label="resident-policy",
    )
    with pytest.raises(SettlementError, match="physical lane orientation"):
        replace(
            candidate.primary,
            speed_evidence_policy_digest=RESIDENT_SPEED_POLICY_DIGEST,
        )


def test_resident_extension_preserves_legacy_settlement_bytes_and_digests() -> None:
    # These bytes include the current target-catalog digest. Catalog changes are
    # reviewed target identity epochs; the settlement schema remains unchanged.
    # Epoch 2026-09-01: priority fallback composition was replaced by explicit
    # target conflict/displacement and symmetric challenger transitions.
    # Epoch 2026-09-05: the merged GLM catalog retires the artifact-provider
    # registry and preserves the exclusive target composition policy.
    # Historical records are unaffected: they embed their own catalog snapshot.
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    assert "resident_lane_orientation" not in candidate.primary.to_dict()
    assert "resident_lane_orientation" not in candidate.reproduction.to_dict()
    assert candidate.primary.digest == (
        "63c65128ace99e51f95e666aaaf1e8242eeb4b2413bfba535384321fc40af561"
    )
    assert candidate.digest == (
        "a28d61de8b177eff8022f489e346ee543ef12c0235d069f510aa6c9b4b0691d7"
    )


def test_single_pass_or_reused_evidence_cannot_become_settlement_candidate() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    with pytest.raises(SettlementError, match="reuses primary"):
        SettlementCandidate.from_reproductions(candidate.primary, candidate.primary)
    with pytest.raises(SettlementError, match="reproduction identity"):
        SettlementCandidate.from_reproductions(
            candidate.primary,
            replace(candidate.reproduction, hotkey="different-miner"),
        )


@pytest.mark.parametrize(
    "field",
    (
        "qualification_authority_digest",
        "qualification_plan_digest",
        "qualification_attempt_digest",
        "qualification_report_digest",
        "selection_commitment_digest",
        "selection_secret_commitment_digest",
        "selection_evidence_digest",
    ),
)
def test_each_reproduction_authority_and_evidence_identity_must_be_distinct(
    field: str,
) -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    reproduced = replace(
        candidate.reproduction,
        **{field: getattr(candidate.primary, field)},
    )
    with pytest.raises(SettlementError, match="reuses primary"):
        SettlementCandidate.from_reproductions(candidate.primary, reproduced)


def test_conservative_speed_uses_slower_independent_reproduction() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog,
        label="a", speedup="1.09",
    )
    slower = replace(candidate.reproduction, speedup="1.03")
    reproduced = SettlementCandidate.from_reproductions(candidate.primary, slower)
    assert reproduced.speedup == "1.03"


def test_reproduction_must_use_the_same_speed_evidence_policy() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    mismatched = replace(
        candidate.reproduction,
        speed_evidence_policy_digest=_h("repeat-speed-policy"),
    )
    with pytest.raises(SettlementError, match="contribution identity"):
        SettlementCandidate.from_reproductions(candidate.primary, mismatched)


@pytest.mark.parametrize("field", ("sample_rate_ppm", "minimum_calls"))
def test_reproduction_must_use_the_same_seed_independent_audit_control(
    field: str,
) -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    assert candidate.reproduction.audit_policy is not None
    changed_policy = replace(
        candidate.reproduction.audit_policy,
        **{
            field: getattr(candidate.reproduction.audit_policy, field) + 1,
        },
    )
    mismatched = replace(
        candidate.reproduction,
        audit_control_digest=changed_policy.control.digest,
        audit_policy=changed_policy,
    )
    with pytest.raises(SettlementError, match="contribution identity"):
        SettlementCandidate.from_reproductions(candidate.primary, mismatched)


def test_reproduction_requires_distinct_audit_seed_and_evidence() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    assert candidate.primary.audit_policy is not None
    reused_seed = replace(
        candidate.reproduction,
        audit_policy=replace(
            candidate.reproduction.audit_policy,
            validator_seed=candidate.primary.audit_policy.validator_seed,
        ),
    )
    with pytest.raises(SettlementError, match="slot-audit authority"):
        SettlementCandidate.from_reproductions(candidate.primary, reused_seed)
    reused_evidence = replace(
        candidate.reproduction,
        audit_evidence_digest=candidate.primary.audit_evidence_digest,
    )
    with pytest.raises(SettlementError, match="slot-audit authority"):
        SettlementCandidate.from_reproductions(candidate.primary, reused_evidence)


def test_new_candidate_rejects_legacy_auditless_qualification_but_reopens_history() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    legacy_primary = replace(
        candidate.primary,
        audit_control_digest=SettlementQualification.__dataclass_fields__[  # type: ignore[index]
            "audit_control_digest"
        ].default,
        audit_policy=None,
        audit_evidence_digest=SettlementQualification.__dataclass_fields__[  # type: ignore[index]
            "audit_evidence_digest"
        ].default,
    )
    legacy_reproduction = replace(
        candidate.reproduction,
        audit_control_digest=legacy_primary.audit_control_digest,
        audit_policy=None,
        audit_evidence_digest=legacy_primary.audit_evidence_digest,
    )
    with pytest.raises(SettlementError, match="requires two audited"):
        SettlementCandidate.from_reproductions(
            legacy_primary, legacy_reproduction
        )
    historical = SettlementCandidate(legacy_primary, legacy_reproduction)
    assert SettlementCandidate.from_dict(historical.to_dict()) == historical


def test_settlement_evidence_binds_both_retained_attempts() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, ROUTED, "a"), catalog, label="a"
    )
    primary_ref = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        candidate.primary.qualification_attempt_digest,
        1,
        "application/json",
        "cacheon.qualification.cohort-attempt.v1",
    )
    reproduction_ref = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        candidate.reproduction.qualification_attempt_digest,
        1,
        "application/json",
        "cacheon.qualification.cohort-attempt.v1",
    )
    evidence = SettlementEvidence.bind(
        candidate,
        primary_attempt_ref=primary_ref,
        reproduction_attempt_ref=reproduction_ref,
    )
    assert SettlementEvidence.from_dict(evidence.to_dict()) == evidence
    with pytest.raises(SettlementError, match="differ from the candidate"):
        SettlementEvidence.bind(
            candidate,
            primary_attempt_ref=reproduction_ref,
            reproduction_attempt_ref=primary_ref,
        )


def test_highest_speedup_wins_and_events_form_hash_chain() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    early = _candidate(
        incumbent, _ref(catalog, ROUTED, "early"), catalog,
        label="early", speedup="1.04", block=10,
    )
    late = _candidate(
        incumbent, _ref(catalog, ROUTED, "late"), catalog,
        label="late", speedup="1.06", block=11,
    )
    plan = plan_settlement(
        (early, late), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"), initial_event_sequence=7,
    )
    assert plan.winner_candidate_digest == late.digest
    assert plan.transition is not None
    assert plan.transition.manifest == late.candidate_manifest
    assert [row.event_type for row in plan.events] == [
        SettlementEventType.HOLD,
        SettlementEventType.CROWN,
        SettlementEventType.ADOPTION,
        SettlementEventType.STACK_TRANSITION,
    ]
    assert plan.events[0].reason == "conflict_lost"
    assert [row.sequence for row in plan.events] == [7, 8, 9, 10]
    for prior, current in zip(plan.events, plan.events[1:]):
        assert current.previous_event_digest == prior.digest
    assert SettlementEvent.from_dict(plan.events[1].to_dict()) == plan.events[1]


def test_equal_speedup_uses_finalized_order_not_input_order() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    first = _candidate(
        incumbent, _ref(catalog, ROUTED, "first"), catalog,
        label="first", speedup="1.05", block=10, event=2,
    )
    second = _candidate(
        incumbent, _ref(catalog, ROUTED, "second"), catalog,
        label="second", speedup="1.05", block=11, event=0,
    )
    plan = plan_settlement(
        (second, first), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    assert plan.winner_candidate_digest == first.digest


def test_stale_candidate_holds_without_stack_change() -> None:
    catalog = default_target_catalog()
    old = _stack(catalog)
    candidate = _candidate(old, _ref(catalog, ROUTED, "old"), catalog, label="old")
    current_ref = _ref(catalog, SILU, "current")
    current = _stack(catalog, {SILU: current_ref})
    plan = plan_settlement(
        (candidate,), current_manifest=current, current_tree_digest=_h("other-tree")
    )
    assert plan.transition is None
    assert plan.before == plan.after
    assert [row.event_type for row in plan.events] == [SettlementEventType.HOLD]
    assert plan.events[0].reason == "stale_incumbent"


def test_nonoverlapping_loser_is_held_for_requalification() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    routed = _candidate(
        incumbent, _ref(catalog, ROUTED, "routed"), catalog,
        label="routed", speedup="1.07",
    )
    silu = _candidate(
        incumbent, _ref(catalog, SILU, "silu"), catalog,
        label="silu", speedup="1.06",
    )
    plan = plan_settlement(
        (silu, routed), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    hold = next(row for row in plan.events if row.event_type is SettlementEventType.HOLD)
    assert hold.candidate_digest == silu.digest
    assert hold.reason == "incumbent_advanced"


def test_replacement_retires_prior() -> None:
    catalog = default_target_catalog()
    prior = _ref(catalog, ROUTED, "prior")
    incumbent = _stack(catalog, {ROUTED: prior})
    replacement = _candidate(
        incumbent, _ref(catalog, ROUTED, "next"), catalog, label="next"
    )
    replaced = plan_settlement(
        (replacement,), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    assert SettlementEventType.RETIREMENT in {
        row.event_type for row in replaced.events
    }


def _tip(candidate: SettlementCandidate) -> TargetLineage:
    assert candidate.candidate_manifest is not None
    incumbent = candidate.incumbent_manifest.entries.get(candidate.target_id)
    return TargetLineage(
        (
            TargetLineageNode(
                candidate.candidate_manifest.entries[
                    candidate.target_id
                ].artifact_digest,
                "" if incumbent is None else incumbent.artifact_digest,
                candidate.speedup,
                _h(f"transition:{candidate.digest}"),
            ),
        )
    )


def _lineage(*candidates: SettlementCandidate) -> TargetLineage:
    nodes = []
    for candidate in candidates:
        assert candidate.candidate_manifest is not None
        incumbent = candidate.incumbent_manifest.entries.get(
            candidate.target_id
        )
        nodes.append(
            TargetLineageNode(
                candidate.candidate_manifest.entries[
                    candidate.target_id
                ].artifact_digest,
                "" if incumbent is None else incumbent.artifact_digest,
                candidate.speedup,
                _h(f"transition:{candidate.digest}"),
            )
        )
    return TargetLineage(tuple(nodes))


def test_pretransition_stale_sibling_above_last_winner_can_crown() -> None:
    catalog = default_target_catalog()
    parent = _stack(catalog, {ROUTED: _ref(catalog, ROUTED, "parent")})
    last_winner = _candidate(
        parent, _ref(catalog, ROUTED, "winner"), catalog, label="winner", speedup="1.05"
    )
    better_sibling = _candidate(
        parent, _ref(catalog, ROUTED, "better"), catalog, label="better",
        speedup="1.09", block=11,
    )
    planned = plan_settlement(
        (better_sibling,),
        current_manifest=parent,
        current_tree_digest=_h("incumbent-tree"),
        lineage_tips={ROUTED: _tip(last_winner)},
        pretransition_reservations=frozenset(
            {better_sibling.reservation_digest}
        ),
    )
    assert planned.winner_candidate_digest == better_sibling.digest
    crown = next(
        event for event in planned.events
        if event.event_type is SettlementEventType.CROWN
    )
    assert crown.reason == "qualified_pretransition_ancestor_win"


def test_faster_stale_sibling_becomes_the_next_baseline_and_threshold() -> None:
    catalog = default_target_catalog()
    parent = _stack(catalog, {ROUTED: _ref(catalog, ROUTED, "parent")})
    first = _candidate(
        parent, _ref(catalog, ROUTED, "first"), catalog,
        label="first", speedup="1.05",
    )
    faster = _candidate(
        parent, _ref(catalog, ROUTED, "faster"), catalog,
        label="faster", speedup="1.09",
    )
    accepted = plan_settlement(
        (faster,),
        current_manifest=parent,
        current_tree_digest=_h("incumbent-tree"),
        lineage_tips={ROUTED: _tip(first)},
        pretransition_reservations=frozenset({faster.reservation_digest}),
    )
    assert accepted.winner_candidate_digest == faster.digest
    assert accepted.after == faster.challenger

    # Once accepted, the faster sibling owns the lineage and its score is the
    # threshold. Another sibling from the old parent cannot clear it with a
    # score that only beat the former winner.
    middle_sibling = _candidate(
        parent, _ref(catalog, ROUTED, "middle"), catalog,
        label="middle", speedup="1.07",
    )
    rejected = plan_settlement(
        (middle_sibling,),
        current_manifest=parent,
        current_tree_digest=_h("incumbent-tree"),
        lineage_tips={ROUTED: _tip(faster)},
        pretransition_reservations=frozenset(
            {middle_sibling.reservation_digest}
        ),
    )
    assert rejected.winner_candidate_digest == ""
    assert rejected.events[0].reason == "stale_incumbent"

    # A fresh challenger measured against the faster sibling is current and
    # uses ordinary marginal settlement, not the stale-sibling exception.
    assert faster.candidate_manifest is not None
    current_challenger = _candidate(
        faster.candidate_manifest,
        _ref(catalog, ROUTED, "current-child"),
        catalog,
        label="current-child",
        speedup="1.02",
    )
    current = plan_settlement(
        (current_challenger,),
        current_manifest=faster.candidate_manifest,
        current_tree_digest=_h("incumbent-tree"),
        lineage_tips={ROUTED: _tip(faster)},
    )
    assert current.winner_candidate_digest == current_challenger.digest


def test_pretransition_uncle_must_beat_composed_tip_score_from_ancestor() -> None:
    catalog = default_target_catalog()
    a = _stack(catalog, {ROUTED: _ref(catalog, ROUTED, "A")})
    b = _candidate(
        a, _ref(catalog, ROUTED, "B"), catalog, label="B", speedup="1.1"
    )
    assert b.candidate_manifest is not None
    c = _candidate(
        b.candidate_manifest,
        _ref(catalog, ROUTED, "C"),
        catalog,
        label="C",
        speedup="1.1",
    )
    assert c.candidate_manifest is not None
    active = _lineage(b, c)
    assert active.threshold_from(a.entries[ROUTED].artifact_digest) == (
        Decimal("1.21"),
        active.nodes[0].transition_event_id,
    )

    equal_d = _candidate(
        a, _ref(catalog, ROUTED, "D-equal"), catalog,
        label="D-equal", speedup="1.21",
    )
    equal_plan = plan_settlement(
        (equal_d,),
        current_manifest=c.candidate_manifest,
        current_tree_digest=c.candidate_tree_digest,
        lineage_tips={ROUTED: active},
        pretransition_reservations=frozenset({equal_d.reservation_digest}),
    )
    assert equal_plan.winner_candidate_digest == ""

    faster_d = _candidate(
        a, _ref(catalog, ROUTED, "D-faster"), catalog,
        label="D-faster", speedup="1.22",
    )
    faster_plan = plan_settlement(
        (faster_d,),
        current_manifest=c.candidate_manifest,
        current_tree_digest=c.candidate_tree_digest,
        lineage_tips={ROUTED: active},
        pretransition_reservations=frozenset(
            {faster_d.reservation_digest}
        ),
    )
    assert faster_plan.winner_candidate_digest == faster_d.digest
    assert faster_plan.after == faster_d.challenger


@pytest.mark.parametrize(
    ("speedup", "pretransition"),
    (("1.05", True), ("1.04", True), ("1.09", False)),
)
def test_stale_sibling_must_be_strictly_better_and_pretransition(
    speedup: str, pretransition: bool,
) -> None:
    catalog = default_target_catalog()
    parent = _stack(catalog, {ROUTED: _ref(catalog, ROUTED, "parent")})
    last_winner = _candidate(
        parent, _ref(catalog, ROUTED, "winner"), catalog, label="winner",
        speedup="1.05",
    )
    stale = _candidate(
        parent, _ref(catalog, ROUTED, f"stale:{speedup}:{pretransition}"),
        catalog, label=f"stale:{speedup}:{pretransition}", speedup=speedup,
    )
    planned = plan_settlement(
        (stale,),
        current_manifest=parent,
        current_tree_digest=_h("incumbent-tree"),
        lineage_tips={ROUTED: _tip(last_winner)},
        pretransition_reservations=(
            frozenset({stale.reservation_digest})
            if pretransition
            else frozenset()
        ),
    )
    assert planned.winner_candidate_digest == ""
    assert [event.event_type for event in planned.events] == [
        SettlementEventType.HOLD
    ]
    assert planned.events[0].reason == "stale_incumbent"


def test_lineage_tip_naming_the_incumbent_still_crowns() -> None:
    catalog = default_target_catalog()
    parent = _stack(catalog, {ROUTED: _ref(catalog, ROUTED, "parent")})
    last_winner = _candidate(
        parent, _ref(catalog, ROUTED, "winner"), catalog, label="winner"
    )
    assert last_winner.candidate_manifest is not None
    current = last_winner.candidate_manifest
    candidate = _candidate(
        current, _ref(catalog, ROUTED, "next"), catalog, label="next"
    )
    crowned = plan_settlement(
        (candidate,),
        current_manifest=current,
        current_tree_digest=_h("incumbent-tree"),
        lineage_tips={ROUTED: _tip(last_winner)},
    )
    assert crowned.winner_candidate_digest == candidate.digest
    assert crowned.after == candidate.challenger

    with pytest.raises(SettlementError, match="lineage tip artifact"):
        TargetLineageNode(
            "not-a-digest", parent.entries[ROUTED].artifact_digest,
            "1.05", _h("transition"),
        )


def test_genesis_target_without_a_lineage_tip_crowns() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    candidate = _candidate(incumbent, _ref(catalog, ROUTED, "first"), catalog, label="first")

    # No prior crown for this target, so nothing constrains its lineage yet.
    planned = plan_settlement(
        (candidate,),
        current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
        lineage_tips={
            SILU: TargetLineage(
                (
                    TargetLineageNode(
                        _h("artifact:other-target"), "", "1.01",
                        _h("transition"),
                    ),
                )
            ),
        },
    )
    assert planned.winner_candidate_digest == candidate.digest


def test_duplicate_reservation_is_rejected() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    candidate = _candidate(
        incumbent, _ref(catalog, ROUTED, "one"), catalog, label="one"
    )
    with pytest.raises(SettlementError, match="duplicates"):
        plan_settlement(
            (candidate, candidate), current_manifest=incumbent,
            current_tree_digest=_h("incumbent-tree"),
        )

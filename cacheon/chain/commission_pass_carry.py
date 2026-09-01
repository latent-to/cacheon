"""Carry a durable PASS across a source-only arena commission change."""

from __future__ import annotations

from dataclasses import replace


def _manifest_without_arena(manifest: object) -> dict[str, object]:
    value = manifest.to_dict()  # type: ignore[attr-defined]
    value.pop("arena_digest")
    return value


def carry_primary_pass_forward(primary: object, reproduction: object):
    """Rebind one prior PASS when only commission-derived identities changed."""

    from cacheon.settlement import (
        ResidentLaneOrientation,
        SettlementError,
        SettlementQualification,
    )

    if (
        type(primary) is not SettlementQualification
        or type(reproduction) is not SettlementQualification
    ):
        raise SettlementError("commission carry requires two exact qualifications")
    if primary.reproduction_identity == reproduction.reproduction_identity:
        return primary
    stable = (
        "lane", "reservation_digest", "finalized_block", "event_index",
        "event_subindex", "hotkey", "target_id", "members",
        "selected_delta_digest", "proposal_digest", "speed_evidence_policy_digest",
        "audit_control_digest",
    )
    if (
        primary.arena_digest == reproduction.arena_digest
        or any(getattr(primary, field) != getattr(reproduction, field) for field in stable)
        or _manifest_without_arena(primary.incumbent_manifest)
        != _manifest_without_arena(reproduction.incumbent_manifest)
        or primary.candidate_manifest is None
        or reproduction.candidate_manifest is None
        or _manifest_without_arena(primary.candidate_manifest)
        != _manifest_without_arena(reproduction.candidate_manifest)
    ):
        raise SettlementError(
            "independent reproduction differs from the primary reproduction identity"
        )
    orientation = reproduction.resident_lane_orientation
    carried_orientation = None
    if orientation is not None:
        if primary.resident_lane_orientation is None:
            raise SettlementError(
                "independent reproduction has incomplete resident lane orientation"
            )
        carried_orientation = ResidentLaneOrientation(
            orientation.speed_evidence_policy_digest,
            orientation.control_digest,
            orientation.candidate_physical_lane_digest,
            orientation.baseline_physical_lane_digest,
        )
    return replace(
        primary,
        arena_digest=reproduction.arena_digest,
        arm_digest=reproduction.arm_digest,
        incumbent_stack_digest=reproduction.incumbent_stack_digest,
        incumbent_tree_digest=reproduction.incumbent_tree_digest,
        candidate_stack_digest=reproduction.candidate_stack_digest,
        candidate_tree_digest=reproduction.candidate_tree_digest,
        incumbent_manifest=reproduction.incumbent_manifest,
        candidate_manifest=reproduction.candidate_manifest,
        resident_lane_orientation=carried_orientation,
    )


__all__ = ["carry_primary_pass_forward"]

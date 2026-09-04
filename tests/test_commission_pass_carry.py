from __future__ import annotations

from dataclasses import replace

import pytest

from cacheon.chain.commission_pass_carry import carry_primary_pass_forward
from cacheon.settlement import SettlementCandidate, SettlementError
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.target_catalog import default_target_catalog
from tests.test_settlement import (
    MSA,
    _candidate,
    _h,
    _ref,
    _resident_orientation,
    _stack,
)


def _rotate_manifest(
    manifest: EvaluationStackManifest, arena: str, *, runtime: str | None = None
) -> EvaluationStackManifest:
    value = manifest.to_dict()
    value["arena_digest"] = arena
    if runtime is not None:
        value["runtime_digest"] = runtime
    return EvaluationStackManifest.from_dict(value)


def _cross_commission_pair(*, runtime: str | None = None):
    catalog = default_target_catalog()
    pair = _candidate(
        _stack(catalog), _ref(catalog, MSA, "carry"), catalog, label="carry"
    )
    old_orientation = _resident_orientation("old-a", "old-b", control="old")
    primary = replace(
        pair.primary,
        speed_evidence_policy_digest=old_orientation.speed_evidence_policy_digest,
        resident_lane_orientation=old_orientation,
    )
    arena = _h("new-arena")
    incumbent = _rotate_manifest(pair.reproduction.incumbent_manifest, arena, runtime=runtime)
    assert pair.reproduction.candidate_manifest is not None
    candidate = _rotate_manifest(pair.reproduction.candidate_manifest, arena, runtime=runtime)
    new_orientation = _resident_orientation("new-b", "new-a", control="new")
    reproduction = replace(
        pair.reproduction,
        arena_digest=arena,
        arm_digest=_h("new-arm"),
        incumbent_stack_digest=incumbent.digest,
        incumbent_tree_digest=_h("new-incumbent-tree"),
        candidate_stack_digest=candidate.digest,
        candidate_tree_digest=_h("new-candidate-tree"),
        incumbent_manifest=incumbent,
        candidate_manifest=candidate,
        speed_evidence_policy_digest=new_orientation.speed_evidence_policy_digest,
        resident_lane_orientation=new_orientation,
    )
    return primary, reproduction


def test_pass_carries_across_source_only_commission_identity_change() -> None:
    primary, reproduction = _cross_commission_pair()

    carried = carry_primary_pass_forward(primary, reproduction)
    candidate = SettlementCandidate.from_reproductions(carried, reproduction)

    assert carried.speedup == primary.speedup
    assert carried.qualification_attempt_digest == primary.qualification_attempt_digest
    assert carried.reproduction_identity == reproduction.reproduction_identity
    assert candidate.speedup == min(primary.speedup, reproduction.speedup)


def test_pass_does_not_carry_across_runtime_change() -> None:
    primary, reproduction = _cross_commission_pair(runtime=_h("other-runtime"))
    with pytest.raises(SettlementError, match="reproduction identity"):
        carry_primary_pass_forward(primary, reproduction)

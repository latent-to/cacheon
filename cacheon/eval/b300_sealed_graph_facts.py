"""Path-free registry for commissioned B300 focused graph facts.

The registry is a narrow authority seam between an offline/private verifier and
registered qualification.  It stores already-typed graph facts under identities
derived only from trusted public candidate and prepared-runtime objects.  It
does not execute candidate code, interpret verifier output, grade evidence, or
infer attribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from cacheon._strict import require_digest, require_identifier
from cacheon.arena_service import ArenaCandidateBinding
from cacheon.eval.b300_registered_qualification_inputs import B300FocusedGraphFacts
from cacheon.eval.marginal_runtime import PreparedCandidateRuntime
from cacheon.stack_identity import canonical_digest
from cacheon.stack_plan import MarginalArmPlan


REGISTRY_SCHEMA_VERSION = 1
REGISTRY_POLICY_VERSION = "cacheon.eval.b300-sealed-graph-facts-policy.v1"
IDENTITY_DOMAIN = "cacheon.eval.b300-sealed-graph-facts-identity"
FACTS_DOMAIN = "cacheon.eval.b300-sealed-graph-facts"
REGISTRY_DOMAIN = "cacheon.eval.b300-sealed-graph-facts-registry"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class B300SealedGraphFactsError(ValueError):
    """A sealed fact or lookup differs from its commissioned identity."""


def _digest(value: object, field_name: str) -> str:
    return require_digest(
        value,
        field=field_name,
        error=B300SealedGraphFactsError,
    )


def _identifier(value: object, field_name: str) -> str:
    return require_identifier(
        value,
        field=field_name,
        error=B300SealedGraphFactsError,
        pattern=_IDENTIFIER,
    )


def _facts_data(facts: B300FocusedGraphFacts) -> dict[str, object]:
    """Return exact, path-free primitives for one typed focused-facts value."""

    if type(facts) is not B300FocusedGraphFacts:
        raise B300SealedGraphFactsError(
            "sealed graph facts must be an exact B300FocusedGraphFacts"
        )
    return {
        "expected_graph_replays": facts.expected_graph_replays,
        "variants": [row.to_dict() for row in facts.variants],
        "observations": [
            {
                "context_applicable": row.context_applicable,
                "domain_coverage_complete": row.domain_coverage_complete,
                "shapes": [
                    {
                        "applicable": shape.applicable,
                        "capture_succeeded": shape.capture_succeeded,
                        "descriptor_digest": shape.descriptor_digest,
                        "eager_passed": shape.eager_passed,
                        "replay_count": shape.replay_count,
                        "replay_passed": shape.replay_passed,
                    }
                    for shape in row.shapes
                ],
                "slot_id": row.slot_id,
                "variant_id": row.variant_id,
            }
            for row in facts.observations
        ],
    }


def focused_graph_facts_digest(facts: B300FocusedGraphFacts) -> str:
    """Canonical identity for every exact typed field in focused graph facts."""

    return canonical_digest(FACTS_DOMAIN, _facts_data(facts))


@dataclass(frozen=True)
class B300SealedGraphFactsIdentity:
    """Complete path-free lookup identity for one candidate/prepared pair."""

    candidate_binding_digest: str
    reservation_digest: str
    target_id: str
    target_members: tuple[str, ...]
    selected_delta_digest: str
    publication_content_hash: str
    publication_digest: str
    publication_receipt_digest: str
    target_spec_digest: str
    prepared_contribution_digest: str
    prepared_arm_digest: str
    prepared_launch_digest: str
    materialized_stack_digest: str
    materialized_tree_digest: str

    def __post_init__(self) -> None:
        for name in (
            "candidate_binding_digest",
            "reservation_digest",
            "selected_delta_digest",
            "publication_content_hash",
            "publication_digest",
            "publication_receipt_digest",
            "target_spec_digest",
            "prepared_contribution_digest",
            "prepared_arm_digest",
            "prepared_launch_digest",
            "materialized_stack_digest",
            "materialized_tree_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
        members = self.target_members
        if (
            type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or any(_identifier(row, "target member") != row for row in members)
        ):
            raise B300SealedGraphFactsError(
                "sealed target members must be a nonempty canonical tuple"
            )

    @classmethod
    def derive(
        cls,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
    ) -> "B300SealedGraphFactsIdentity":
        """Re-derive the complete registry key from exact trusted objects."""

        if type(candidate) is not ArenaCandidateBinding:
            raise B300SealedGraphFactsError(
                "graph-facts candidate must be an exact ArenaCandidateBinding"
            )
        if type(prepared) is not PreparedCandidateRuntime:
            raise B300SealedGraphFactsError(
                "graph-facts runtime must be an exact PreparedCandidateRuntime"
            )
        arm = prepared.arm
        if type(arm) is not MarginalArmPlan:
            raise B300SealedGraphFactsError(
                "sealed registered graph facts require one marginal arm"
            )
        reservation = candidate.reservation
        publication = candidate.publication
        tree = prepared.binding.tree
        trusted = prepared.binding.launch_binding
        launch = prepared.launch
        replacement = arm.transition.replacement
        if (
            arm.transition.target_id != reservation.target_id
            or arm.selected_delta_digest != reservation.selected_delta_digest
            or replacement.artifact_digest != publication.content_hash
            or tree.stack_digest != launch.stack_digest
            or tree.tree_digest != launch.tree_digest
            or trusted.materialized_tree_root != tree.root
            or trusted.native_build_spec.tree_digest != tree.tree_digest
            or trusted.native_build_spec.digest != launch.native_build_spec_digest
        ):
            raise B300SealedGraphFactsError(
                "candidate and prepared runtime do not form one exact graph-facts identity"
            )
        return cls(
            candidate_binding_digest=candidate.digest,
            reservation_digest=reservation.reservation_digest,
            target_id=reservation.target_id,
            target_members=reservation.target_members,
            selected_delta_digest=reservation.selected_delta_digest,
            publication_content_hash=publication.content_hash,
            publication_digest=publication.digest,
            publication_receipt_digest=publication.publication_digest,
            target_spec_digest=arm.transition.target_spec_digest,
            prepared_contribution_digest=replacement.digest,
            prepared_arm_digest=arm.digest,
            prepared_launch_digest=launch.digest,
            materialized_stack_digest=tree.stack_digest,
            materialized_tree_digest=tree.tree_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_binding_digest": self.candidate_binding_digest,
            "materialized_stack_digest": self.materialized_stack_digest,
            "materialized_tree_digest": self.materialized_tree_digest,
            "prepared_arm_digest": self.prepared_arm_digest,
            "prepared_contribution_digest": self.prepared_contribution_digest,
            "prepared_launch_digest": self.prepared_launch_digest,
            "publication_content_hash": self.publication_content_hash,
            "publication_digest": self.publication_digest,
            "publication_receipt_digest": self.publication_receipt_digest,
            "reservation_digest": self.reservation_digest,
            "selected_delta_digest": self.selected_delta_digest,
            "target_id": self.target_id,
            "target_members": list(self.target_members),
            "target_spec_digest": self.target_spec_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(IDENTITY_DOMAIN, self.to_dict())


@dataclass(frozen=True)
class B300SealedGraphFactsEntry:
    """One commissioned raw-evidence identity and its already-typed facts."""

    identity: B300SealedGraphFactsIdentity
    facts: B300FocusedGraphFacts
    raw_evidence_digest: str

    def __post_init__(self) -> None:
        if type(self.identity) is not B300SealedGraphFactsIdentity:
            raise B300SealedGraphFactsError(
                "sealed graph-facts identity is not exactly typed"
            )
        if type(self.facts) is not B300FocusedGraphFacts:
            raise B300SealedGraphFactsError(
                "sealed graph facts are not exactly typed"
            )
        object.__setattr__(
            self,
            "raw_evidence_digest",
            _digest(self.raw_evidence_digest, "raw_evidence_digest"),
        )
        observed_members = tuple(sorted({row.slot_id for row in self.facts.variants}))
        if observed_members != self.identity.target_members:
            raise B300SealedGraphFactsError(
                "sealed graph facts differ from the candidate target members"
            )

    @classmethod
    def seal(
        cls,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
        facts: B300FocusedGraphFacts,
        *,
        raw_evidence_digest: str,
    ) -> "B300SealedGraphFactsEntry":
        return cls(
            B300SealedGraphFactsIdentity.derive(candidate, prepared),
            facts,
            raw_evidence_digest,
        )

    @property
    def facts_digest(self) -> str:
        return focused_graph_facts_digest(self.facts)

    def to_dict(self) -> dict[str, object]:
        return {
            "facts": _facts_data(self.facts),
            "facts_digest": self.facts_digest,
            "identity": self.identity.to_dict(),
            "identity_digest": self.identity.digest,
            "raw_evidence_digest": self.raw_evidence_digest,
        }


@dataclass(frozen=True)
class B300SealedGraphFactsRegistry:
    """Callable GraphFactsBuilder backed by a closed commissioned registry."""

    verification_policy_digest: str
    entries: tuple[B300SealedGraphFactsEntry, ...]
    policy_version: str = REGISTRY_POLICY_VERSION
    schema_version: int = REGISTRY_SCHEMA_VERSION
    _by_identity: Mapping[str, B300SealedGraphFactsEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verification_policy_digest",
            _digest(self.verification_policy_digest, "verification_policy_digest"),
        )
        if self.policy_version != REGISTRY_POLICY_VERSION:
            raise B300SealedGraphFactsError(
                "sealed graph-facts policy version is unsupported"
            )
        if (
            type(self.schema_version) is not int
            or self.schema_version != REGISTRY_SCHEMA_VERSION
        ):
            raise B300SealedGraphFactsError(
                "sealed graph-facts registry schema is unsupported"
            )
        rows = self.entries
        if (
            type(rows) is not tuple
            or not rows
            or any(type(row) is not B300SealedGraphFactsEntry for row in rows)
        ):
            raise B300SealedGraphFactsError(
                "sealed graph-facts entries must be one nonempty exact tuple"
            )
        ordered = tuple(sorted(rows, key=lambda row: row.identity.digest))
        by_identity: dict[str, B300SealedGraphFactsEntry] = {}
        for row in ordered:
            key = row.identity.digest
            prior = by_identity.get(key)
            if prior is not None:
                kind = "duplicate" if prior == row else "ambiguous"
                raise B300SealedGraphFactsError(
                    f"sealed graph-facts registry contains a {kind} identity"
                )
            by_identity[key] = row
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "_by_identity", MappingProxyType(by_identity))

    def __call__(
        self,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
    ) -> B300FocusedGraphFacts:
        identity = B300SealedGraphFactsIdentity.derive(candidate, prepared)
        row = self._by_identity.get(identity.digest)
        if row is None or row.identity != identity:
            raise B300SealedGraphFactsError(
                "no sealed graph facts match the exact candidate and prepared runtime"
            )
        return row.facts

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [row.to_dict() for row in self.entries],
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "verification_policy_digest": self.verification_policy_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(REGISTRY_DOMAIN, self.to_dict())


__all__ = [
    "B300SealedGraphFactsEntry",
    "B300SealedGraphFactsError",
    "B300SealedGraphFactsIdentity",
    "B300SealedGraphFactsRegistry",
    "FACTS_DOMAIN",
    "IDENTITY_DOMAIN",
    "REGISTRY_DOMAIN",
    "REGISTRY_POLICY_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "focused_graph_facts_digest",
]

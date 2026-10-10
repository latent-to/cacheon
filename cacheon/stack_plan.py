"""Pure planning for one-target evaluation-stack transitions.

The types in this module describe immutable B/C/B-prime arms and sealed candidate
cohorts.  They do not launch engines, interpret measurements, select winners, or
mutate incumbent state.
"""

from __future__ import annotations

from dataclasses import dataclass

from cacheon.stack_identity import canonical_digest
from cacheon.stack_manifest import (
    ContributionRef,
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.target_catalog import TargetCatalog, TargetResolutionError
from cacheon._strict import require_digest


_PLAN_SCHEMA_VERSION = 1
_PLAN_POLICY_VERSION = "stack-plan.v2"


class StackPlanError(ValueError):
    """A requested stack transition is not one registered marginal delta."""


def _digest(value: object, *, field: str) -> str:
    return require_digest(value, field=field, error=StackPlanError)


def _ref_dict(ref: ContributionRef) -> dict[str, object]:
    return ref.to_dict()


def _require_ref(value: object, *, field: str) -> ContributionRef:
    if not isinstance(value, ProposalContributionRef):
        raise StackPlanError(f"{field} must be a contribution ref")
    return value


@dataclass(frozen=True)
class StackArmIdentity:
    """Content identity of one complete materialized engine arm."""

    stack_digest: str
    tree_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stack_digest",
            _digest(self.stack_digest, field="arm stack_digest"),
        )
        object.__setattr__(
            self,
            "tree_digest",
            _digest(self.tree_digest, field="arm tree_digest"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "stack_digest": self.stack_digest,
            "tree_digest": self.tree_digest,
        }


@dataclass(frozen=True)
class TargetTransition:
    """The exact registered target replacement represented by one C arm."""

    target_id: str
    target_spec_digest: str
    replacement: ContributionRef
    prior: ContributionRef | None
    displaced: tuple[ContributionRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise StackPlanError("transition target_id must be a non-empty string")
        object.__setattr__(
            self,
            "target_spec_digest",
            _digest(self.target_spec_digest, field="transition target_spec_digest"),
        )
        object.__setattr__(
            self, "replacement", _require_ref(self.replacement, field="replacement")
        )
        if self.prior is not None:
            object.__setattr__(self, "prior", _require_ref(self.prior, field="prior"))
        object.__setattr__(
            self,
            "displaced",
            tuple(_require_ref(ref, field="displaced entry") for ref in self.displaced),
        )
        if self.replacement.target_id != self.target_id:
            raise StackPlanError("replacement target does not match transition target")
        if self.replacement.target_spec_digest != self.target_spec_digest:
            raise StackPlanError("replacement target-spec digest does not match transition")
        if self.prior is not None:
            if self.prior.target_id != self.target_id:
                raise StackPlanError("prior contribution does not match transition target")
            if self.displaced:
                raise StackPlanError(
                    "same-target replacement cannot also displace active targets"
                )
            if self.prior.selected_delta_digest == self.replacement.selected_delta_digest:
                raise StackPlanError("same-target replacement has no executable delta")
        displaced_ids = tuple(ref.target_id for ref in self.displaced)
        if displaced_ids != tuple(sorted(displaced_ids)):
            raise StackPlanError("displaced contributions must be target-sorted")
        if len(set(displaced_ids)) != len(displaced_ids):
            raise StackPlanError("displaced contributions contain duplicate targets")
        if self.target_id in displaced_ids:
            raise StackPlanError("transition cannot displace its replacement target")

    @property
    def selected_delta_digest(self) -> str:
        return self.replacement.selected_delta_digest

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
            "replacement": _ref_dict(self.replacement),
            "prior": None if self.prior is None else _ref_dict(self.prior),
            "displaced": [_ref_dict(ref) for ref in self.displaced],
        }


@dataclass(frozen=True)
class MarginalArmPlan:
    """One exact target transition over a frozen incumbent stack."""

    incumbent: EvaluationStackManifest
    candidate: EvaluationStackManifest
    transition: TargetTransition
    baseline_before: StackArmIdentity
    challenger: StackArmIdentity
    baseline_after: StackArmIdentity
    schema_version: int = _PLAN_SCHEMA_VERSION
    policy_version: str = _PLAN_POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.incumbent, EvaluationStackManifest) or not isinstance(
            self.candidate, EvaluationStackManifest
        ):
            raise StackPlanError("marginal arm stacks must be evaluation manifests")
        if not isinstance(self.transition, TargetTransition):
            raise StackPlanError("marginal arm transition is invalid")
        if not all(
            isinstance(arm, StackArmIdentity)
            for arm in (self.baseline_before, self.challenger, self.baseline_after)
        ):
            raise StackPlanError("marginal arm identities are invalid")
        if type(self.schema_version) is not int or self.schema_version != _PLAN_SCHEMA_VERSION:
            raise StackPlanError("marginal arm schema_version must be 1")
        if self.policy_version != _PLAN_POLICY_VERSION:
            raise StackPlanError("marginal arm policy_version is unsupported")
        if self.baseline_before != self.baseline_after:
            raise StackPlanError("B and B-prime must bind the same exact incumbent")
        if self.baseline_before.stack_digest != self.incumbent.digest:
            raise StackPlanError("baseline arm does not bind incumbent stack")
        if self.challenger.stack_digest != self.candidate.digest:
            raise StackPlanError("challenger arm does not bind candidate stack")
        if self.challenger.tree_digest == self.baseline_before.tree_digest:
            raise StackPlanError("challenger and incumbent tree digests must differ")
        incumbent_entries = self.incumbent.entries
        candidate_entries = self.candidate.entries
        target_id = self.transition.target_id
        if incumbent_entries.get(target_id) != self.transition.prior:
            raise StackPlanError("transition prior does not match incumbent entry")
        if candidate_entries.get(target_id) != self.transition.replacement:
            raise StackPlanError("transition replacement does not match candidate entry")
        displaced = {
            ref.target_id: ref for ref in self.transition.displaced
        }
        if any(incumbent_entries.get(target) != ref for target, ref in displaced.items()):
            raise StackPlanError("transition displaced entries do not match incumbent")
        expected_targets = (set(incumbent_entries) - set(displaced)) | {target_id}
        if set(candidate_entries) != expected_targets:
            raise StackPlanError("candidate entries do not match transition target set")
        for active_id, incumbent_ref in incumbent_entries.items():
            if active_id not in displaced and active_id != target_id:
                if candidate_entries.get(active_id) != incumbent_ref:
                    raise StackPlanError(
                        f"candidate changed unrelated target {active_id!r}"
                    )

    @property
    def selected_delta_digest(self) -> str:
        return self.transition.selected_delta_digest

    @property
    def contribution_digest(self) -> str:
        return self.transition.replacement.digest

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "incumbent_stack_digest": self.incumbent.digest,
            "candidate_stack_digest": self.candidate.digest,
            "transition": self.transition.to_dict(),
            "baseline_before": self.baseline_before.to_dict(),
            "challenger": self.challenger.to_dict(),
            "baseline_after": self.baseline_after.to_dict(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.stack.marginal-arm-plan", self.to_dict())

    def reopen(
        self,
        *,
        catalog: TargetCatalog,
        expected_context: EvaluationStackContext,
    ) -> "MarginalArmPlan":
        """Reconstruct and compare the complete registered transition."""

        expected = plan_marginal_arm(
            self.incumbent,
            self.transition.replacement,
            catalog=catalog,
            incumbent_tree_digest=self.baseline_before.tree_digest,
            candidate_tree_digest=self.challenger.tree_digest,
            expected_context=expected_context,
        )
        if expected.to_dict() != self.to_dict():
            raise StackPlanError("marginal arm does not reopen to its declared transition")
        return self


def _candidate_transition(
    incumbent: EvaluationStackManifest,
    replacement: ContributionRef,
    *,
    catalog: TargetCatalog,
    expected_context: EvaluationStackContext,
) -> tuple[EvaluationStackManifest, TargetTransition]:

    if not isinstance(incumbent, EvaluationStackManifest):
        raise TypeError("incumbent must be an EvaluationStackManifest")
    if not isinstance(catalog, TargetCatalog):
        raise TypeError("catalog must be a TargetCatalog")
    replacement = _require_ref(replacement, field="replacement")
    incumbent.validate_against(expected_context)
    if (
        catalog.digest != incumbent.catalog_digest
        or catalog.digest != expected_context.catalog_digest
        or catalog.snapshot() != incumbent.catalog_snapshot
        or catalog.snapshot() != expected_context.catalog_snapshot
    ):
        raise StackPlanError("planning catalog does not match the frozen stack context")
    target_id = replacement.target_id
    try:
        catalog.require(target_id)
        expected_spec = catalog.target_spec_digest(target_id)
        catalog.validate_active_targets(incumbent.entries)
    except TargetResolutionError as exc:
        raise StackPlanError(f"invalid registered transition: {exc}") from exc
    if replacement.target_spec_digest != expected_spec:
        raise StackPlanError(
            f"replacement target-spec digest is stale for {target_id!r}"
        )

    active = incumbent.entries
    prior = active.get(target_id)
    # Targets never share a node (disjoint roots), so a transition replaces its
    # own entry and nothing else; ``displaced`` stays empty on the wire.
    remove: tuple[str, ...] = ()
    displaced: tuple[ContributionRef, ...] = ()

    transition = TargetTransition(
        target_id=target_id,
        target_spec_digest=expected_spec,
        replacement=replacement,
        prior=prior,
        displaced=displaced,
    )
    try:
        candidate = incumbent.with_contribution(replacement, remove=remove)
        catalog.validate_active_targets(candidate.entries)
        candidate.validate_against(expected_context)
    except (ValueError, TargetResolutionError) as exc:
        raise StackPlanError(f"invalid marginal transition: {exc}") from exc

    expected_targets = (set(active) - set(remove)) | {target_id}
    if set(candidate.entries) != expected_targets:
        raise StackPlanError("candidate changed entries outside the target transition")
    for active_id, incumbent_ref in active.items():
        if active_id not in remove and active_id != target_id:
            if candidate.entries.get(active_id) != incumbent_ref:
                raise StackPlanError(
                    f"candidate changed unrelated target {active_id!r}"
                )
    if candidate.digest == incumbent.digest:
        raise StackPlanError("marginal transition does not change the stack")
    return candidate, transition


def plan_candidate_stack(
    incumbent: EvaluationStackManifest,
    replacement: ContributionRef,
    *,
    catalog: TargetCatalog,
    expected_context: EvaluationStackContext,
) -> EvaluationStackManifest:
    """Construct the exact C stack before its engine tree is materialized."""

    candidate, _ = _candidate_transition(
        incumbent,
        replacement,
        catalog=catalog,
        expected_context=expected_context,
    )
    return candidate


def plan_marginal_arm(
    incumbent: EvaluationStackManifest,
    replacement: ContributionRef,
    *,
    catalog: TargetCatalog,
    incumbent_tree_digest: str,
    candidate_tree_digest: str,
    expected_context: EvaluationStackContext,
) -> MarginalArmPlan:
    """Bind an exact C transition to independently materialized tree identities."""

    candidate, transition = _candidate_transition(
        incumbent,
        replacement,
        catalog=catalog,
        expected_context=expected_context,
    )

    incumbent_tree = _digest(
        incumbent_tree_digest, field="incumbent tree_digest"
    )
    candidate_tree = _digest(candidate_tree_digest, field="candidate tree_digest")
    return MarginalArmPlan(
        incumbent=incumbent,
        candidate=candidate,
        transition=transition,
        baseline_before=StackArmIdentity(incumbent.digest, incumbent_tree),
        challenger=StackArmIdentity(candidate.digest, candidate_tree),
        baseline_after=StackArmIdentity(incumbent.digest, incumbent_tree),
    )

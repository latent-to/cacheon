"""Validator-owned qualification construction for all registered targets.

This module closes the target-independent part of the B300 qualification
builder.  A finalized candidate supplies only an
``ArenaCandidateBinding``.  It cannot name a Python module, command, output
directory, graph policy, workload, calibration, reference, resident lane, or
evidence store.

The construction deliberately stops at three validator capabilities which are
deployment facts rather than data that can be inferred from a target ID:

* a source resolver for already-active incumbent contributions;
* a reviewed binding factory for a newly materialized candidate tree; and
* focused graph observations from the validator's commissioned verifier.

All three capabilities have explicit source digests and are supplied in
process.  The generic layer independently inspects the immutable publication,
derives the proposal ref and marginal stack transition, materializes the C
tree below a fixed private root, prepares B/C/B-prime, constructs the catalog-
bound graph requirement, publishes raw graph evidence, builds the quality and
resident-v3 authorities, and returns the exact ``CausalQualificationInput``
consumed by :mod:`cacheon.eval.b300_qualification_deployment`.

There is no FE campaign/profile identity and no expected-submission allowlist
in this module.  The supported registry is exactly the complete registered
catalog: singleton and atomic targets share this one target-neutral path.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.engine_tree import (
    inspect_contribution,
    materialize_engine_tree,
    reopen_materialized_engine_tree,
)
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationCohort,
    B300RegisteredProfileAuthority,
)
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
    B300QualificationGraphEvidenceStoreError,
)
from cacheon.eval.crossover_runtime import (
    ResidentArmPlan,
    ResidentCrossoverPlan,
    ResidentSpeedPolicy,
)
from cacheon.eval.engine_launch import TrustedLaunchBinding
from cacheon.eval.marginal_runtime import (
    MaterializedArmBinding,
    PreparedCandidateRuntime,
    prepare_marginal_runtime,
)
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.eval.qualification import (
    GraphVerificationBinding,
    GraphVerificationMemberBinding,
    GraphVerificationRequirement,
    QualificationProfile,
    ReferenceManifest,
    SelectionCommitment,
)
from cacheon.eval.qualification_intake import (
    GraphMemberObservation,
    GraphVerificationObservation,
    publish_graph_observation,
)
from cacheon.eval.qualification_runner import (
    CandidateQualificationAuthority,
    CausalQualificationInput,
    SpeedEvidencePolicy,
    SpeedStageDisposition,
    _planned_prompt_digests,
)
from cacheon.eval.resident_audit_authority import (
    ResidentAuditExecutionAuthority,
)
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.stack_identity import canonical_digest
from cacheon.stack_manifest import (
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import MarginalArmPlan, plan_candidate_stack, plan_marginal_arm

from cacheon.eval.b300_registered_qualification_inputs import (
    ATTRIBUTION_SCHEMA,
    AUDIT_SEED_DOMAIN,
    FACTORY_SCHEMA,
    ORDINARY_B300_TARGET_IDS,
    POLICY_SCHEMA,
    PRODUCTION_AUTHORITY_BLOCKERS,
    RESOLVER_SCHEMA,
    REGISTERED_B300_TARGET_IDS,
    B300FocusedGraphFacts,
    B300MemberContractProjection,
    B300QualificationBlocker,
    B300RegisteredQualificationError,
    B300RegisteredQualificationInputs,
    B300RegisteredQualificationPolicy,
    B300RegisteredTargetProjection,
    _digest,
    registered_b300_member_contract_projection,
    registered_b300_profile_resolver_digest,
)


@dataclass(frozen=True)
class B300RegisteredQualificationComponents:
    """Values plugged directly into B300QualificationConstructionAuthority."""

    profiles: tuple[B300RegisteredProfileAuthority, ...]
    builder_source_digest: str
    plan_builder: Callable[[B300QualificationCohort, bytes], CausalQualificationInput]

    def __post_init__(self) -> None:
        if (
            type(self.profiles) is not tuple
            or tuple(row.target_id for row in self.profiles)
            != REGISTERED_B300_TARGET_IDS
            or any(type(row) is not B300RegisteredProfileAuthority for row in self.profiles)
            or not callable(self.plan_builder)
        ):
            raise B300RegisteredQualificationError(
                "registered qualification components are not canonical"
            )
        object.__setattr__(
            self,
            "builder_source_digest",
            _digest(self.builder_source_digest, "builder source digest"),
        )


class _CandidateSourceResolver:
    def __init__(
        self,
        *,
        candidate_artifact_digest: str,
        candidate_root: Path,
        fallback: object,
    ) -> None:
        self._candidate_artifact_digest = candidate_artifact_digest
        self._candidate_root = candidate_root
        self._fallback = fallback

    def resolve_proposal(self, artifact_digest: str) -> str | Path:
        if artifact_digest == self._candidate_artifact_digest:
            return self._candidate_root
        return self._fallback.resolve_proposal(artifact_digest)

    def resolve_integrated(self, source_tree_digest: str) -> str | Path:
        return self._fallback.resolve_integrated(source_tree_digest)


class B300RegisteredQualificationFactory:
    """Closed registered-target registry plus deterministic candidate plan builder."""

    def __init__(self, inputs: B300RegisteredQualificationInputs) -> None:
        if type(inputs) is not B300RegisteredQualificationInputs:
            raise B300RegisteredQualificationError(
                "registered qualification inputs are not exact"
            )
        self._inputs = inputs
        self._projection = registered_b300_member_contract_projection(inputs.catalog)
        self._profiles = tuple(
            self._profile_row(row) for row in self._projection
        )

    @property
    def profiles(self) -> tuple[B300RegisteredProfileAuthority, ...]:
        return self._profiles

    @property
    def builder_source_digest(self) -> str:
        return self._inputs.builder_source_digest

    @property
    def components(self) -> B300RegisteredQualificationComponents:
        return B300RegisteredQualificationComponents(
            self.profiles,
            self.builder_source_digest,
            self.plan_builder,
        )

    def profile_for(self, target_id: str) -> B300RegisteredProfileAuthority:
        for row in self.profiles:
            if row.target_id == target_id:
                return row
        raise B300RegisteredQualificationError(
            f"registered qualification target {target_id!r} is unsupported"
        )

    def _profile_row(
        self, target: B300RegisteredTargetProjection
    ) -> B300RegisteredProfileAuthority:
        inputs = self._inputs
        target_id = target.target_id
        resolver_digest = registered_b300_profile_resolver_digest(
            target,
            builder_source_digest=inputs.builder_source_digest,
        )

        def resolve(
            candidate: ArenaCandidateBinding,
            prepared: PreparedCandidateRuntime,
            *,
            expected_target: str = target_id,
        ) -> CandidateQualificationAuthority:
            if candidate.reservation.target_id != expected_target:
                raise B300RegisteredQualificationError(
                    "profile resolver received another registered target"
                )
            return self._candidate_authority(candidate, prepared)

        return B300RegisteredProfileAuthority(
            target_id,
            target.target_spec_digest,
            resolver_digest,
            resolve,
        )

    def _contribution(
        self, candidate: ArenaCandidateBinding
    ) -> ProposalContributionRef:
        inputs = self._inputs
        reservation = candidate.reservation
        target_id = reservation.target_id
        self.profile_for(target_id)
        spec = inputs.catalog.require(target_id)
        if tuple(reservation.target_members) != tuple(sorted(spec.members)):
            raise B300RegisteredQualificationError(
                "finalized reservation members differ from the registered target"
            )
        try:
            inspected = inspect_contribution(
                candidate.publication.root,
                catalog=inputs.catalog,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"immutable candidate publication failed inspection: {exc}"
            ) from None
        if (
            inspected.target_id != target_id
            or inspected.target_spec_digest
            != inputs.catalog.target_spec_digest(target_id)
            or inspected.selected_delta_digest != reservation.selected_delta_digest
            or candidate.publication.digest != reservation.submission_digest
        ):
            raise B300RegisteredQualificationError(
                "candidate publication differs from finalized target/delta authority"
            )
        attribution = canonical_digest(
            ATTRIBUTION_SCHEMA,
            {
                "finalized_block": reservation.finalized_block,
                "finalized_event_index": reservation.finalized_event_index,
                "finalized_event_subindex": reservation.finalized_event_subindex,
                "hotkey": reservation.hotkey,
                "reservation_digest": reservation.reservation_digest,
                "submission_digest": reservation.submission_digest,
            },
        )
        contribution = ProposalContributionRef(
            target_id,
            inspected.target_spec_digest,
            candidate.publication.content_hash,
            inspected.selected_payload_digest,
            attribution,
        )
        if contribution.selected_delta_digest != reservation.selected_delta_digest:
            raise B300RegisteredQualificationError(
                "derived proposal contribution differs from finalized selected delta"
            )
        return contribution

    def _candidate_binding(
        self,
        candidate: ArenaCandidateBinding,
        contribution: ProposalContributionRef,
    ) -> tuple[MarginalArmPlan, MaterializedArmBinding]:
        inputs = self._inputs
        candidate_stack = plan_candidate_stack(
            inputs.incumbent_stack,
            contribution,
            catalog=inputs.catalog,
            expected_context=inputs.expected_context,
        )
        destination = (
            inputs.materialization_root
            / candidate.digest[:2]
            / candidate.digest
        )
        try:
            if destination.exists() or destination.is_symlink():
                tree = reopen_materialized_engine_tree(destination)
            else:
                resolver = _CandidateSourceResolver(
                    candidate_artifact_digest=contribution.artifact_digest,
                    candidate_root=candidate.publication.root,
                    fallback=inputs.source_resolver,
                )
                tree = materialize_engine_tree(
                    candidate_stack,
                    context=inputs.expected_context,
                    catalog=inputs.catalog,
                    resolver=resolver,
                    destination=destination,
                )
        except (OSError, TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"candidate stack materialization failed: {exc}"
            ) from None
        if tree.root != destination or tree.stack_digest != candidate_stack.digest:
            raise B300RegisteredQualificationError(
                "candidate materialization destination or stack identity drifted"
            )
        arm = plan_marginal_arm(
            inputs.incumbent_stack,
            contribution,
            catalog=inputs.catalog,
            incumbent_tree_digest=inputs.incumbent_binding.tree.tree_digest,
            candidate_tree_digest=tree.tree_digest,
            expected_context=inputs.expected_context,
        )
        try:
            trusted = inputs.candidate_binding_builder(tree)
        except Exception as exc:
            raise B300RegisteredQualificationError(
                "validator candidate binding factory failed"
            ) from exc
        if type(trusted) is not TrustedLaunchBinding or trusted.materialized_tree_root != tree.root:
            raise B300RegisteredQualificationError(
                "candidate binding factory returned another materialized tree"
            )
        return arm, MaterializedArmBinding(tree, trusted)

    def _candidate_authority(
        self,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
    ) -> CandidateQualificationAuthority:
        inputs = self._inputs
        reservation = candidate.reservation
        target_id = reservation.target_id
        self.profile_for(target_id)
        if type(prepared) is not PreparedCandidateRuntime or type(
            prepared.arm
        ) is not MarginalArmPlan:
            raise B300RegisteredQualificationError(
                "registered profile resolver requires one prepared marginal arm"
            )
        arm = prepared.arm
        if (
            arm.transition.target_id != target_id
            or arm.transition.target_spec_digest
            != inputs.catalog.target_spec_digest(target_id)
            or arm.selected_delta_digest != reservation.selected_delta_digest
            or arm.transition.replacement.artifact_digest
            != candidate.publication.content_hash
        ):
            raise B300RegisteredQualificationError(
                "prepared candidate differs from finalized publication authority"
            )
        spec = inputs.catalog.require(target_id)
        target_projection = next(
            (row for row in self._projection if row.target_id == target_id),
            None,
        )
        if (
            target_projection is None
            or target_projection.target_spec_digest
            != arm.transition.target_spec_digest
            or target_projection.members != tuple(reservation.target_members)
            or target_projection.members != tuple(spec.members)
        ):
            raise B300RegisteredQualificationError(
                "registered target differs from its ordered member authority"
            )
        try:
            facts = inputs.graph_facts_builder(candidate, prepared)
        except B300QualificationGraphEvidenceHold:
            raise
        except B300QualificationGraphEvidenceStoreError:
            raise B300QualificationGraphEvidenceHold(
                "commissioned graph evidence is unavailable or unauthenticated"
            ) from None
        except Exception as exc:
            raise B300RegisteredQualificationError(
                "validator focused graph authority failed"
            ) from exc
        if type(facts) is not B300FocusedGraphFacts:
            raise B300RegisteredQualificationError(
                "focused graph authority returned untyped facts"
            )
        fact_members = tuple(sorted({row.slot_id for row in facts.variants}))
        observation_members = tuple(
            sorted({row.slot_id for row in facts.observations})
        )
        if (
            fact_members != target_projection.members
            or observation_members != target_projection.members
        ):
            raise B300RegisteredQualificationError(
                "focused graph authority returned another or incomplete member domain"
            )
        members = tuple(
            GraphVerificationMemberBinding(
                row.slot_id,
                row.target_spec_digest,
                row.contract_digest,
                row.verification_profile_id,
            )
            for row in target_projection.member_contracts
        )
        binding = GraphVerificationBinding(
            arm.digest,
            prepared.launch.digest,
            arm.transition.replacement.digest,
            arm.selected_delta_digest,
            target_id,
            arm.transition.target_spec_digest,
            inputs.catalog.digest,
            members,
            inputs.policy.verification_policy_digest,
        )
        requirement = GraphVerificationRequirement(
            binding,
            facts.variants,
            facts.expected_graph_replays,
        )
        observation = GraphVerificationObservation(
            requirement.digest,
            tuple(
                GraphMemberObservation(
                    member_id,
                    tuple(
                        row
                        for row in facts.observations
                        if row.slot_id == member_id
                    ),
                )
                for member_id in target_projection.members
            ),
        )
        try:
            product = publish_graph_observation(
                inputs.evidence_root,
                requirement,
                observation,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"focused graph evidence failed publication/reopen: {exc}"
            ) from None
        profile = QualificationProfile(
            inputs.reference_manifest,
            inputs.calibration_context.digest,
            inputs.calibration_manifest.digest,
            requirement.digest,
            tuple(row.name for row in inputs.calibration_manifest.quality_metrics),
            inputs.policy.nll_tail_threshold,
            inputs.policy.tokens_per_prompt,
            inputs.policy.topk_width,
            inputs.policy.hidden_tasks_per_prompt,
            inputs.policy.support_policy_digest,
            inputs.policy.hidden_task_policy_digest,
            inputs.candidate_runtime_resource_policy_digest,
            inputs.policy.hidden_tasks_required,
            inputs.policy.select_count,
        )
        return CandidateQualificationAuthority(
            reservation.selected_delta_digest,
            profile,
            requirement,
            product.artifact_ref,
            product.evidence_ref,
        )

    def plan_builder(
        self,
        cohort: B300QualificationCohort,
        secret: bytes,
    ) -> CausalQualificationInput:
        inputs = self._inputs
        if type(cohort) is not B300QualificationCohort:
            raise B300RegisteredQualificationError(
                "registered plan builder requires an exact one-candidate cohort"
            )
        if type(secret) is not bytes or len(secret) < 32:
            raise B300RegisteredQualificationError(
                "registered plan builder requires a 256-bit private secret"
            )
        candidate = cohort.candidate
        contribution = self._contribution(candidate)
        arm, candidate_binding = self._candidate_binding(candidate, contribution)
        try:
            prepared = prepare_marginal_runtime(
                arm,
                catalog=inputs.catalog,
                expected_context=inputs.expected_context,
                incumbent_launch=inputs.incumbent_launch,
                incumbent_binding=inputs.incumbent_binding,
                candidate_binding=candidate_binding,
                baseline_session_plan=inputs.baseline_session_plan,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"candidate B/C/B-prime runtime failed to prepare: {exc}"
            ) from None
        if len(prepared.candidates) != 1:
            raise B300RegisteredQualificationError(
                "resident-v3 plan did not prepare exactly one registered candidate"
            )
        prepared_candidate = prepared.candidates[0]
        authority = self.profile_for(
            candidate.reservation.target_id
        ).resolve(candidate, prepared_candidate)
        candidate_resident_arm = ResidentArmPlan(
            prepared_candidate.launch,
            prepared_candidate.binding.launch_binding,
            prepared_candidate.session_plan,
            inputs.candidate_executor_namespace_digest,
            inputs.candidate_runtime_resource_policy_digest,
            inputs.candidate_device_configuration_digest,
        )
        resident_plan = ResidentCrossoverPlan(
            candidate.reservation.selected_delta_digest,
            inputs.resident_baseline_arm,
            candidate_resident_arm,
            inputs.resident_speed_policy,
        )
        audit_seed = hashlib.sha256(
            AUDIT_SEED_DOMAIN
            + secret
            + bytes.fromhex(candidate.reservation.selected_delta_digest)
        ).hexdigest()[:32]
        audit_policy = SlotAuditPolicy(
            audit_seed,
            inputs.policy.audit_sample_rate_ppm,
            inputs.policy.audit_minimum_calls,
            tuple(sorted(candidate.reservation.target_members)),
            prepared_candidate.session_plan.engine_config.tp_size,
        )
        audit_prompt = min(
            (
                prompt
                for batch in prepared_candidate.session_plan.prompt_batches
                for prompt in batch
            ),
            key=lambda prompt: (
                len(prompt),
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            ),
        )
        audit_authority = ResidentAuditExecutionAuthority.derive(
            prepared_candidate.launch,
            prepared_candidate.binding.launch_binding,
            prepared_candidate.session_plan,
            audit_policy=audit_policy,
            prompt_batches=tuple(
                (audit_prompt,)
                for _ in range(inputs.policy.audit_minimum_calls + 1)
            ),
            max_new_tokens=inputs.policy.audit_max_new_tokens,
            top_logprobs_num=inputs.policy.audit_toplogprobs_num,
            executor_namespace_digest=inputs.candidate_executor_namespace_digest,
            runtime_resource_policy_digest=(
                inputs.candidate_runtime_resource_policy_digest
            ),
            device_configuration_digest=(
                inputs.candidate_device_configuration_digest
            ),
        )
        commitment = SelectionCommitment.seal(
            source_plan_digest=prepared.source.digest,
            reference_manifest=inputs.reference_manifest,
            entropy_source_digest=inputs.reference_manifest.selection_policy_digest,
            prompt_digests=_planned_prompt_digests(prepared),
            select_count=inputs.policy.select_count,
            secret=secret,
        )
        return CausalQualificationInput(
            prepared=prepared,
            model_mount=inputs.model_mount,
            candidates=(authority,),
            commitment=commitment,
            selection_secret=secret,
            evidence_root=inputs.evidence_root,
            calibration_threshold_policy=inputs.calibration_threshold_policy,
            calibration_manifest=inputs.calibration_manifest,
            calibration_context=inputs.calibration_context,
            calibration_artifact_ref=inputs.calibration_artifact_ref,
            pristine_stack=inputs.pristine_stack,
            pristine_launch=inputs.pristine_launch,
            pristine_binding=inputs.pristine_binding.launch_binding,
            reference_engine_config=inputs.pristine_session_plan.engine_config,
            reference_preflight=inputs.pristine_session_plan.expected_preflight,
            expected_launch_resource_policy_digest=(
                inputs.incumbent_launch.resource_policy_digest
            ),
            expected_runtime_resource_policy_digest=(
                inputs.candidate_runtime_resource_policy_digest
            ),
            expected_device_policy_digest=(
                inputs.incumbent_launch.hardware.device_policy_digest
            ),
            audit_policies=(audit_policy,),
            speed_evidence_policy=SpeedEvidencePolicy.resident(),
            resident_speed_plan=resident_plan,
            resident_audit_plan=audit_authority,
            speed_stage_disposition=SpeedStageDisposition.TERMINAL,
        )


def build_b300_registered_qualification_factory(
    inputs: B300RegisteredQualificationInputs,
) -> B300RegisteredQualificationFactory:
    """Return the complete registered catalog and resident-v3 plan builder."""

    return B300RegisteredQualificationFactory(inputs)


__all__ = [
    "B300FocusedGraphFacts",
    "B300MemberContractProjection",
    "B300QualificationBlocker",
    "B300RegisteredQualificationComponents",
    "B300RegisteredQualificationError",
    "B300RegisteredQualificationFactory",
    "B300RegisteredQualificationInputs",
    "B300RegisteredQualificationPolicy",
    "B300RegisteredTargetProjection",
    "FACTORY_SCHEMA",
    "ORDINARY_B300_TARGET_IDS",
    "POLICY_SCHEMA",
    "PRODUCTION_AUTHORITY_BLOCKERS",
    "RESOLVER_SCHEMA",
    "REGISTERED_B300_TARGET_IDS",
    "build_b300_registered_qualification_factory",
    "registered_b300_member_contract_projection",
    "registered_b300_profile_resolver_digest",
]

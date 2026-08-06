"""Validator-owned qualification construction for ordinary registered targets.

This module closes the target-independent part of the B300 qualification
builder.  A finalized candidate supplies only an
``ArenaCandidateBinding``.  It cannot name a Python module, command, output
directory, graph policy, workload, calibration, reference, resident lane, or
evidence store.

The construction deliberately stops at three validator capabilities which are
deployment facts rather than data that can be inferred from a target ID:

* a source resolver for already-active incumbent contributions;
* a reviewed binding factory for a newly materialized candidate tree; and
* focused graph observations from the validator's target-specific verifier.

All three capabilities have explicit source digests and are supplied in
process.  The generic layer independently inspects the immutable publication,
derives the proposal ref and marginal stack transition, materializes the C
tree below a fixed private root, prepares B/C/B-prime, constructs the catalog-
bound graph requirement, publishes raw graph evidence, builds the quality and
resident-v3 authorities, and returns the exact ``CausalQualificationInput``
consumed by :mod:`cacheon.eval.b300_qualification_deployment`.

There is no FE campaign/profile identity and no expected-submission allowlist
in this module.  The supported registry is exactly the eleven live singleton
targets in ``target_catalog.SINGLETON_TARGET_IDS``.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable

from cacheon._strict import require_digest
from cacheon.arena_service import ArenaCandidateBinding
from cacheon.engine_tree import (
    MaterializedEngineTree,
    inspect_contribution,
    materialize_engine_tree,
    reopen_materialized_engine_tree,
)
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationCohort,
    B300RegisteredProfileAuthority,
)
from cacheon.eval.calibration import (
    CalibrationContext,
    CalibrationManifest,
    CalibrationThresholdPolicy,
    reopen_calibration_evidence,
)
from cacheon.eval.crossover_runtime import (
    ResidentArmPlan,
    ResidentCrossoverPlan,
    ResidentSpeedPolicy,
)
from cacheon.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.marginal_runtime import (
    MaterializedArmBinding,
    PreparedCandidateRuntime,
    prepare_marginal_runtime,
)
from cacheon.eval.oci_backend import TrustedArenaModelMountReceipt
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.eval.qualification import (
    GraphVariantRequirement,
    GraphVerificationBinding,
    GraphVerificationMemberBinding,
    GraphVerificationRequirement,
    QualificationProfile,
    ReferenceManifest,
    SelectionCommitment,
)
from cacheon.eval.qualification_intake import (
    GraphMemberObservation,
    GraphVariantObservation,
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
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.stack_identity import canonical_digest
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import MarginalArmPlan, plan_candidate_stack, plan_marginal_arm
from cacheon.target_catalog import SINGLETON_TARGET_IDS, TargetCatalog


ORDINARY_B300_TARGET_IDS = tuple(sorted(SINGLETON_TARGET_IDS))
POLICY_SCHEMA = "cacheon.eval.b300-registered-qualification-policy.v1"
FACTORY_SCHEMA = "cacheon.eval.b300-registered-qualification-factory.v1"
RESOLVER_SCHEMA = "cacheon.eval.b300-registered-profile-resolver.v1"
ATTRIBUTION_SCHEMA = "cacheon.eval.b300-finalized-proposal-attribution.v1"
AUDIT_SEED_DOMAIN = b"cacheon-b300-registered-audit-seed-v1\0"


class B300RegisteredQualificationError(RuntimeError):
    """A submitted bundle differs from sealed ordinary-target authority."""


def _digest(value: object, field: str) -> str:
    return require_digest(
        value,
        field=field,
        error=B300RegisteredQualificationError,
    )


def _positive(value: object, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise B300RegisteredQualificationError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _canonical_private_root(value: object, field: str) -> Path:
    try:
        root = Path(value)  # type: ignore[arg-type]
    except TypeError:
        raise B300RegisteredQualificationError(f"{field} is not path-like") from None
    posix = PurePosixPath(root.as_posix())
    if (
        not root.is_absolute()
        or not posix.is_absolute()
        or root != Path(posix.as_posix())
        or "." in posix.parts
        or ".." in posix.parts
    ):
        raise B300RegisteredQualificationError(
            f"{field} is not one canonical absolute path"
        )
    try:
        before = root.lstat()
        resolved = root.resolve(strict=True)
        after = resolved.lstat()
    except OSError as exc:
        raise B300RegisteredQualificationError(
            f"{field} is unavailable: {exc}"
        ) from None
    owner = os.geteuid() if hasattr(os, "geteuid") else after.st_uid
    if (
        root != resolved
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or after.st_uid != owner
        or stat.S_IMODE(after.st_mode) != 0o700
    ):
        raise B300RegisteredQualificationError(
            f"{field} must be a validator-owned, nonsymlink mode-0700 directory"
        )
    return root


@dataclass(frozen=True)
class B300QualificationBlocker:
    """One production capability that must be commissioned outside this layer."""

    blocker_id: str
    missing_authority: str
    donor_coordinate: str


# These are deliberately code coordinates, not claims that the private donor is
# production authority.  They identify the largest existing construction
# pattern while making the missing ordinary-target commissioning work explicit.
PRODUCTION_AUTHORITY_BLOCKERS = (
    B300QualificationBlocker(
        "ordinary-focused-graph-facts",
        (
            "reviewed focused graph-observation producers for all eleven ordinary "
            "targets, including attention.msa_prefill_block_score"
        ),
        (
            "experiments/minimax_m3/frontier_2026-07-13/"
            "b300_testnet_joined_probe.py:4686"
        ),
    ),
    B300QualificationBlocker(
        "ordinary-runtime-binding",
        (
            "a commissioned ordinary-target B300 runtime case and candidate-tree "
            "binding factory for each physical qualification orientation"
        ),
        (
            "experiments/minimax_m3/frontier_2026-07-13/"
            "b300_testnet_joined_probe.py:5105"
        ),
    ),
    B300QualificationBlocker(
        "typed-frozen-reference-calibration",
        (
            "typed pristine-reference, prompt, hidden-judge, and frozen calibration "
            "authorities that reopen for the exact ordinary workload/lane context"
        ),
        (
            "experiments/minimax_m3/frontier_2026-07-13/"
            "b300_testnet_joined_probe.py:5148"
        ),
    ),
)


@dataclass(frozen=True)
class B300RegisteredQualificationPolicy:
    """Candidate-independent policy for the exact ordinary target registry."""

    catalog_digest: str
    target_spec_digests: tuple[tuple[str, str], ...]
    verification_policy_digest: str
    nll_tail_threshold: str
    tokens_per_prompt: int
    topk_width: int
    hidden_tasks_per_prompt: int
    support_policy_digest: str
    hidden_task_policy_digest: str
    hidden_tasks_required: bool
    select_count: int
    audit_sample_rate_ppm: int = 1_000_000
    audit_minimum_calls: int = 32
    audit_max_new_tokens: int = 2
    audit_toplogprobs_num: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "catalog_digest", _digest(self.catalog_digest, "catalog digest")
        )
        rows = tuple(self.target_spec_digests)
        if (
            type(self.target_spec_digests) is not tuple
            or tuple(target for target, _ in rows) != ORDINARY_B300_TARGET_IDS
            or len({target for target, _ in rows}) != len(rows)
        ):
            raise B300RegisteredQualificationError(
                "registered target-spec rows do not exactly cover the eleven ordinary targets"
            )
        checked = tuple(
            (target, _digest(digest, f"{target} target-spec digest"))
            for target, digest in rows
        )
        object.__setattr__(self, "target_spec_digests", checked)
        for field in (
            "verification_policy_digest",
            "support_policy_digest",
            "hidden_task_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not isinstance(self.nll_tail_threshold, str):
            raise B300RegisteredQualificationError(
                "NLL tail threshold must be a canonical decimal string"
            )
        # QualificationProfile performs the complete decimal/range validation;
        # these integers are validated here before any candidate is inspected.
        object.__setattr__(
            self,
            "tokens_per_prompt",
            _positive(self.tokens_per_prompt, "tokens per prompt"),
        )
        object.__setattr__(
            self,
            "topk_width",
            _positive(self.topk_width, "top-k width", minimum=0),
        )
        object.__setattr__(
            self,
            "hidden_tasks_per_prompt",
            _positive(
                self.hidden_tasks_per_prompt,
                "hidden tasks per prompt",
                minimum=0,
            ),
        )
        if (
            type(self.hidden_tasks_required) is not bool
            or self.hidden_tasks_required != (self.hidden_tasks_per_prompt > 0)
        ):
            raise B300RegisteredQualificationError(
                "hidden-task policy and count disagree"
            )
        object.__setattr__(
            self, "select_count", _positive(self.select_count, "select count", minimum=2)
        )
        rate = _positive(
            self.audit_sample_rate_ppm,
            "audit sample rate",
        )
        if rate > 1_000_000:
            raise B300RegisteredQualificationError(
                "audit sample rate exceeds one million ppm"
            )
        object.__setattr__(self, "audit_sample_rate_ppm", rate)
        for field in (
            "audit_minimum_calls",
            "audit_max_new_tokens",
            "audit_toplogprobs_num",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))

    @classmethod
    def seal(
        cls,
        catalog: TargetCatalog,
        *,
        verification_policy_digest: str,
        nll_tail_threshold: str,
        tokens_per_prompt: int,
        topk_width: int,
        hidden_tasks_per_prompt: int,
        support_policy_digest: str,
        hidden_task_policy_digest: str,
        hidden_tasks_required: bool,
        select_count: int,
        audit_sample_rate_ppm: int = 1_000_000,
        audit_minimum_calls: int = 32,
        audit_max_new_tokens: int = 2,
        audit_toplogprobs_num: int = 1,
    ) -> "B300RegisteredQualificationPolicy":
        if type(catalog) is not TargetCatalog:
            raise B300RegisteredQualificationError(
                "qualification target catalog is not exact"
            )
        rows = []
        for target_id in ORDINARY_B300_TARGET_IDS:
            spec = catalog.require(target_id)
            if spec.members != (target_id,) or spec.contract_ref is None:
                raise B300RegisteredQualificationError(
                    f"ordinary target {target_id!r} is not one singleton contract"
                )
            rows.append((target_id, catalog.target_spec_digest(target_id)))
        return cls(
            catalog.digest,
            tuple(rows),
            verification_policy_digest,
            nll_tail_threshold,
            tokens_per_prompt,
            topk_width,
            hidden_tasks_per_prompt,
            support_policy_digest,
            hidden_task_policy_digest,
            hidden_tasks_required,
            select_count,
            audit_sample_rate_ppm,
            audit_minimum_calls,
            audit_max_new_tokens,
            audit_toplogprobs_num,
        )

    def require_catalog(self, catalog: TargetCatalog) -> None:
        if type(catalog) is not TargetCatalog or catalog.digest != self.catalog_digest:
            raise B300RegisteredQualificationError(
                "registered qualification catalog is stale"
            )
        observed = tuple(
            (target, catalog.target_spec_digest(target))
            for target in ORDINARY_B300_TARGET_IDS
        )
        if observed != self.target_spec_digests:
            raise B300RegisteredQualificationError(
                "registered ordinary target specifications are stale"
            )
        for target_id in ORDINARY_B300_TARGET_IDS:
            spec = catalog.require(target_id)
            if spec.members != (target_id,) or spec.contract_ref is None:
                raise B300RegisteredQualificationError(
                    f"ordinary target {target_id!r} lost its singleton declared-math contract"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_max_new_tokens": self.audit_max_new_tokens,
            "audit_minimum_calls": self.audit_minimum_calls,
            "audit_sample_rate_ppm": self.audit_sample_rate_ppm,
            "audit_toplogprobs_num": self.audit_toplogprobs_num,
            "catalog_digest": self.catalog_digest,
            "hidden_task_policy_digest": self.hidden_task_policy_digest,
            "hidden_tasks_per_prompt": self.hidden_tasks_per_prompt,
            "hidden_tasks_required": self.hidden_tasks_required,
            "nll_tail_threshold": self.nll_tail_threshold,
            "select_count": self.select_count,
            "support_policy_digest": self.support_policy_digest,
            "target_spec_digests": [list(row) for row in self.target_spec_digests],
            "tokens_per_prompt": self.tokens_per_prompt,
            "topk_width": self.topk_width,
            "verification_policy_digest": self.verification_policy_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(POLICY_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class B300FocusedGraphFacts:
    """Raw, target-local graph facts before candidate/catalog identity is added."""

    expected_graph_replays: int
    variants: tuple[GraphVariantRequirement, ...]
    observations: tuple[GraphVariantObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_graph_replays",
            _positive(self.expected_graph_replays, "expected graph replays", minimum=2),
        )
        variants = tuple(self.variants)
        observations = tuple(self.observations)
        if (
            not variants
            or any(type(row) is not GraphVariantRequirement for row in variants)
            or any(type(row) is not GraphVariantObservation for row in observations)
            or tuple((row.slot_id, row.variant_id) for row in variants)
            != tuple((row.slot_id, row.variant_id) for row in observations)
            or tuple((row.slot_id, row.variant_id) for row in variants)
            != tuple(sorted((row.slot_id, row.variant_id) for row in variants))
        ):
            raise B300RegisteredQualificationError(
                "focused graph facts lack canonical requirement/observation coverage"
            )
        for required, observed in zip(variants, observations, strict=True):
            descriptors = tuple(row.descriptor_digest for row in observed.shapes)
            applicable = tuple(
                row.descriptor_digest for row in observed.shapes if row.applicable
            )
            if (
                descriptors != required.shape_descriptor_digests
                or applicable != required.applicable_shape_descriptor_digests
                or observed.context_applicable != required.context_applicable
            ):
                raise B300RegisteredQualificationError(
                    "focused graph observations differ from the declared shape domain"
                )
        object.__setattr__(self, "variants", variants)
        object.__setattr__(self, "observations", observations)


CandidateBindingBuilder = Callable[[MaterializedEngineTree], TrustedLaunchBinding]
GraphFactsBuilder = Callable[
    [ArenaCandidateBinding, PreparedCandidateRuntime], B300FocusedGraphFacts
]


@dataclass(frozen=True)
class B300RegisteredQualificationInputs:
    """Exact runtime, measurement, calibration, and prompt authorities.

    The two callables are reviewed validator capabilities.  Their source
    digests participate in the factory/profile identities; neither is selected
    from candidate metadata.
    """

    catalog: TargetCatalog
    policy: B300RegisteredQualificationPolicy
    expected_context: EvaluationStackContext
    incumbent_stack: EvaluationStackManifest
    incumbent_binding: MaterializedArmBinding
    incumbent_launch: EngineLaunchSpec
    baseline_session_plan: SessionExecutionPlan
    model_mount: TrustedArenaModelMountReceipt
    materialization_root: Path
    source_resolver_digest: str
    source_resolver: object
    candidate_binding_builder_digest: str
    candidate_binding_builder: CandidateBindingBuilder
    graph_facts_builder_digest: str
    graph_facts_builder: GraphFactsBuilder
    evidence_root: Path
    reference_manifest: ReferenceManifest
    calibration_threshold_policy: CalibrationThresholdPolicy
    calibration_manifest: CalibrationManifest
    calibration_context: CalibrationContext
    calibration_artifact_ref: EvidenceArtifactRef
    pristine_stack: EvaluationStackManifest
    pristine_binding: MaterializedArmBinding
    pristine_launch: EngineLaunchSpec
    pristine_session_plan: SessionExecutionPlan
    resident_baseline_arm: ResidentArmPlan
    resident_speed_policy: ResidentSpeedPolicy
    candidate_executor_namespace_digest: str
    candidate_runtime_resource_policy_digest: str
    candidate_device_configuration_digest: str

    def __post_init__(self) -> None:
        if type(self.catalog) is not TargetCatalog or type(
            self.policy
        ) is not B300RegisteredQualificationPolicy:
            raise B300RegisteredQualificationError(
                "registered catalog or policy is not exact"
            )
        self.policy.require_catalog(self.catalog)
        if type(self.expected_context) is not EvaluationStackContext:
            raise B300RegisteredQualificationError(
                "evaluation-stack context is not exact"
            )
        if (
            self.expected_context.catalog_digest != self.catalog.digest
            or self.expected_context.catalog_snapshot != self.catalog.snapshot()
        ):
            raise B300RegisteredQualificationError(
                "evaluation-stack context differs from the ordinary target catalog"
            )
        if type(self.incumbent_stack) is not EvaluationStackManifest:
            raise B300RegisteredQualificationError("incumbent stack is not exact")
        try:
            self.incumbent_stack.validate_against(self.expected_context)
        except (TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"incumbent stack is stale: {exc}"
            ) from None
        if (
            type(self.incumbent_binding) is not MaterializedArmBinding
            or type(self.incumbent_launch) is not EngineLaunchSpec
            or type(self.baseline_session_plan) is not SessionExecutionPlan
            or self.incumbent_binding.tree.stack_digest != self.incumbent_stack.digest
            or self.incumbent_launch.stack_digest != self.incumbent_stack.digest
            or self.incumbent_launch.tree_digest
            != self.incumbent_binding.tree.tree_digest
            or self.baseline_session_plan.launch_digest != self.incumbent_launch.digest
            or self.baseline_session_plan.audit_policy is not None
            or self.baseline_session_plan.engine_config.disable_cuda_graph is not False
        ):
            raise B300RegisteredQualificationError(
                "incumbent B runtime is not one exact graph-on, audit-free authority"
            )
        if type(self.model_mount) is not TrustedArenaModelMountReceipt:
            raise B300RegisteredQualificationError("model mount is not exact")
        object.__setattr__(
            self,
            "materialization_root",
            _canonical_private_root(
                self.materialization_root, "candidate materialization root"
            ),
        )
        object.__setattr__(
            self,
            "evidence_root",
            _canonical_private_root(self.evidence_root, "qualification evidence root"),
        )
        for field in (
            "source_resolver_digest",
            "candidate_binding_builder_digest",
            "graph_facts_builder_digest",
            "candidate_executor_namespace_digest",
            "candidate_runtime_resource_policy_digest",
            "candidate_device_configuration_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not all(
            callable(value)
            for value in (
                getattr(self.source_resolver, "resolve_proposal", None),
                getattr(self.source_resolver, "resolve_integrated", None),
                self.candidate_binding_builder,
                self.graph_facts_builder,
            )
        ):
            raise B300RegisteredQualificationError(
                "materialization/binding/graph capabilities are not callable"
            )
        if type(self.pristine_stack) is not EvaluationStackManifest:
            raise B300RegisteredQualificationError("pristine T stack is not exact")
        try:
            self.pristine_stack.validate_against(self.expected_context)
        except (TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"pristine T stack is stale: {exc}"
            ) from None
        if self.pristine_stack.entries:
            raise B300RegisteredQualificationError(
                "pristine T contains proposal contributions"
            )
        if (
            type(self.pristine_binding) is not MaterializedArmBinding
            or type(self.pristine_launch) is not EngineLaunchSpec
            or type(self.pristine_session_plan) is not SessionExecutionPlan
            or self.pristine_binding.tree.stack_digest != self.pristine_stack.digest
            or self.pristine_launch.stack_digest != self.pristine_stack.digest
            or self.pristine_launch.tree_digest != self.pristine_binding.tree.tree_digest
            or self.pristine_session_plan.launch_digest != self.pristine_launch.digest
            or self.pristine_session_plan.audit_policy is not None
            or self.pristine_launch.resource_policy_digest
            != self.incumbent_launch.resource_policy_digest
        ):
            raise B300RegisteredQualificationError(
                "pristine T launch/binding/session differs from the sealed empty stack"
            )
        if type(self.reference_manifest) is not ReferenceManifest:
            raise B300RegisteredQualificationError(
                "pristine reference manifest is not exact"
            )
        try:
            expected_reference = ReferenceManifest.from_pristine(
                self.pristine_stack,
                self.pristine_launch,
                self.pristine_binding,
                workload_digest=marginal_workload_digest(self.baseline_session_plan),
                tokenizer_digest=self.reference_manifest.tokenizer_digest,
                hidden_corpus_commitment=(
                    self.reference_manifest.hidden_corpus_commitment
                ),
                hidden_judge_digest=self.reference_manifest.hidden_judge_digest,
                selection_policy_digest=(
                    self.reference_manifest.selection_policy_digest
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"pristine T failed to reopen: {exc}"
            ) from None
        if expected_reference != self.reference_manifest:
            raise B300RegisteredQualificationError(
                "reference manifest differs from pristine T or the frozen workload"
            )
        if not all(
            type(value) is expected
            for value, expected in (
                (self.calibration_threshold_policy, CalibrationThresholdPolicy),
                (self.calibration_manifest, CalibrationManifest),
                (self.calibration_context, CalibrationContext),
                (self.calibration_artifact_ref, EvidenceArtifactRef),
            )
        ):
            raise B300RegisteredQualificationError(
                "calibration authority is not exactly typed"
            )
        expected_context = CalibrationContext(
            self.reference_manifest.digest,
            self.reference_manifest.arena_digest,
            self.reference_manifest.runtime_digest,
            self.reference_manifest.base_engine_digest,
            self.reference_manifest.model_revision_digest,
            self.reference_manifest.model_manifest_digest,
            self.reference_manifest.model_content_digest,
            self.reference_manifest.logical_hardware_digest,
            self.reference_manifest.workload_digest,
            self.policy.verification_policy_digest,
            self.reference_manifest.controller_distribution_digest,
        )
        if expected_context != self.calibration_context:
            raise B300RegisteredQualificationError(
                "calibration context differs from declared math/reference/workload"
            )
        try:
            reopened = reopen_calibration_evidence(
                self.evidence_root,
                self.calibration_artifact_ref,
                expected_threshold_policy=self.calibration_threshold_policy,
                expected_manifest=self.calibration_manifest,
                expected_context=self.calibration_context,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"frozen calibration evidence failed to reopen: {exc}"
            ) from None
        if reopened != self.calibration_manifest or not reopened.thresholds_frozen:
            raise B300RegisteredQualificationError(
                "qualification calibration is not one frozen authority"
            )
        if (
            type(self.resident_baseline_arm) is not ResidentArmPlan
            or type(self.resident_speed_policy) is not ResidentSpeedPolicy
            or self.resident_speed_policy.version != 3
            or marginal_workload_digest(self.resident_baseline_arm.session_plan)
            != marginal_workload_digest(self.baseline_session_plan)
            or self.resident_baseline_arm.executor_namespace_digest
            == self.candidate_executor_namespace_digest
        ):
            raise B300RegisteredQualificationError(
                "resident-v3 baseline/candidate lane authority is inconsistent"
            )
        try:
            expected_speed = ResidentSpeedPolicy.from_calibration(
                max_stage_seconds=self.resident_speed_policy.max_stage_seconds,
                max_qualification_seconds=(
                    self.resident_speed_policy.max_qualification_seconds
                ),
                calibration=self.calibration_manifest,
                context=self.calibration_context,
                version=3,
                min_windows=self.resident_speed_policy.min_windows,
                max_window_scatter=self.resident_speed_policy.max_window_scatter,
                max_conditioning_slowdown=(
                    self.resident_speed_policy.max_conditioning_slowdown
                ),
            )
        except (TypeError, ValueError) as exc:
            raise B300RegisteredQualificationError(
                f"resident-v3 policy is not derived from calibration: {exc}"
            ) from None
        if expected_speed != self.resident_speed_policy:
            raise B300RegisteredQualificationError(
                "resident-v3 speed policy differs from frozen calibration"
            )
        if (
            self.policy.tokens_per_prompt
            != self.baseline_session_plan.max_new_tokens
            or self.policy.topk_width
            != self.baseline_session_plan.top_logprobs_num
        ):
            raise B300RegisteredQualificationError(
                "profile token/support authorities differ from the frozen workload"
            )

    @property
    def builder_source_digest(self) -> str:
        return canonical_digest(
            FACTORY_SCHEMA,
            {
                "binding_builder_digest": self.candidate_binding_builder_digest,
                "calibration_digest": self.calibration_manifest.digest,
                "graph_facts_builder_digest": self.graph_facts_builder_digest,
                "policy_digest": self.policy.digest,
                "reference_manifest_digest": self.reference_manifest.digest,
                "source_resolver_digest": self.source_resolver_digest,
                "speed_policy_digest": self.resident_speed_policy.digest,
            },
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
            != ORDINARY_B300_TARGET_IDS
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
    """Closed ordinary-target registry plus deterministic candidate plan builder."""

    def __init__(self, inputs: B300RegisteredQualificationInputs) -> None:
        if type(inputs) is not B300RegisteredQualificationInputs:
            raise B300RegisteredQualificationError(
                "registered qualification inputs are not exact"
            )
        self._inputs = inputs
        self._profiles = tuple(
            self._profile_row(target_id) for target_id in ORDINARY_B300_TARGET_IDS
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
            f"ordinary qualification target {target_id!r} is unsupported"
        )

    def _profile_row(self, target_id: str) -> B300RegisteredProfileAuthority:
        inputs = self._inputs
        spec = inputs.catalog.require(target_id)
        contract = spec.contract_ref
        assert contract is not None
        resolver_digest = canonical_digest(
            RESOLVER_SCHEMA,
            {
                "builder_source_digest": inputs.builder_source_digest,
                "contract_digest": inputs.catalog.contract_digest(target_id),
                "target_id": target_id,
                "target_spec_digest": inputs.catalog.target_spec_digest(target_id),
                "verification_profile_id": contract.verification_profile_id,
            },
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
            inputs.catalog.target_spec_digest(target_id),
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
        contract = spec.contract_ref
        if contract is None or spec.members != (target_id,):
            raise B300RegisteredQualificationError(
                "ordinary target lacks one independent singleton math contract"
            )
        try:
            facts = inputs.graph_facts_builder(candidate, prepared)
        except Exception as exc:
            raise B300RegisteredQualificationError(
                "validator focused graph authority failed"
            ) from exc
        if type(facts) is not B300FocusedGraphFacts or any(
            row.slot_id != target_id for row in facts.variants
        ):
            raise B300RegisteredQualificationError(
                "focused graph authority returned another target or untyped facts"
            )
        member = GraphVerificationMemberBinding(
            target_id,
            inputs.catalog.target_spec_digest(target_id),
            inputs.catalog.contract_digest(target_id),
            contract.verification_profile_id,
        )
        binding = GraphVerificationBinding(
            arm.digest,
            prepared.launch.digest,
            arm.transition.replacement.digest,
            arm.selected_delta_digest,
            target_id,
            arm.transition.target_spec_digest,
            inputs.catalog.digest,
            (member,),
            inputs.policy.verification_policy_digest,
        )
        requirement = GraphVerificationRequirement(
            binding,
            facts.variants,
            facts.expected_graph_replays,
        )
        observation = GraphVerificationObservation(
            requirement.digest,
            (GraphMemberObservation(target_id, facts.observations),),
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
                "ordinary plan builder requires an exact singleton cohort"
            )
        if type(secret) is not bytes or len(secret) < 32:
            raise B300RegisteredQualificationError(
                "ordinary plan builder requires a 256-bit private secret"
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
                "resident-v3 plan did not prepare exactly one ordinary candidate"
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
        audit_plan = replace(
            prepared_candidate.session_plan,
            prompt_batches=tuple(
                (audit_prompt,)
                for _ in range(inputs.policy.audit_minimum_calls + 1)
            ),
            warmup_count=1,
            conditioning_count=1,
            max_new_tokens=inputs.policy.audit_max_new_tokens,
            top_logprobs_num=inputs.policy.audit_toplogprobs_num,
            audit_policy=audit_policy,
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
            resident_audit_plan=audit_plan,
            speed_stage_disposition=SpeedStageDisposition.TERMINAL,
        )


def build_b300_registered_qualification_factory(
    inputs: B300RegisteredQualificationInputs,
) -> B300RegisteredQualificationFactory:
    """Return the closed eleven-target registry and resident-v3 plan builder."""

    return B300RegisteredQualificationFactory(inputs)


__all__ = [
    "B300FocusedGraphFacts",
    "B300QualificationBlocker",
    "B300RegisteredQualificationComponents",
    "B300RegisteredQualificationError",
    "B300RegisteredQualificationFactory",
    "B300RegisteredQualificationInputs",
    "B300RegisteredQualificationPolicy",
    "FACTORY_SCHEMA",
    "ORDINARY_B300_TARGET_IDS",
    "POLICY_SCHEMA",
    "PRODUCTION_AUTHORITY_BLOCKERS",
    "RESOLVER_SCHEMA",
    "build_b300_registered_qualification_factory",
]

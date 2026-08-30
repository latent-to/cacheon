"""Sealed policy, blocker, graph-fact, and input authorities for registered
B300 qualification.

These are the closed types the qualification factory validates and derives
plans from.  ``b300_registered_qualification`` composes and re-exports every
name, so import paths are unchanged.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
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
    registered_b300_target_ids,
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
from cacheon.target_catalog import TargetCatalog, TargetKind


POLICY_SCHEMA = "cacheon.eval.b300-registered-qualification-policy.v2"
FACTORY_SCHEMA = "cacheon.eval.b300-registered-qualification-factory.v2"
RESOLVER_SCHEMA = "cacheon.eval.b300-registered-profile-resolver.v2"
ATTRIBUTION_SCHEMA = "cacheon.eval.b300-finalized-proposal-attribution.v1"
AUDIT_SEED_DOMAIN = b"cacheon-b300-registered-audit-seed-v1\0"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class B300RegisteredQualificationError(RuntimeError):
    """A submitted bundle differs from sealed registered-target authority."""


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


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise B300RegisteredQualificationError(f"{field} is not canonical")
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
# pattern while making the missing registered-target commissioning work explicit.
PRODUCTION_AUTHORITY_BLOCKERS = (
    B300QualificationBlocker(
        "registered-focused-graph-facts",
        (
            "reviewed focused graph-observation producers for every registered "
            "catalog target, including multi-member atomic targets"
        ),
        (
            "experiments/minimax_m3/frontier_2026-07-13/"
            "b300_testnet_joined_probe.py:4686"
        ),
    ),
    B300QualificationBlocker(
        "registered-runtime-binding",
        (
            "a commissioned registered-target B300 runtime case and candidate-tree "
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
            "authorities that reopen for the exact registered workload/lane context"
        ),
        (
            "experiments/minimax_m3/frontier_2026-07-13/"
            "b300_testnet_joined_probe.py:5148"
        ),
    ),
)


@dataclass(frozen=True)
class B300MemberContractProjection:
    """One singleton member's declared-math authority."""

    slot_id: str
    target_spec_digest: str
    contract_digest: str
    verification_profile_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _identifier(self.slot_id, "member slot ID"))
        for field in ("target_spec_digest", "contract_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self,
            "verification_profile_id",
            _identifier(self.verification_profile_id, "verification profile ID"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_digest": self.contract_digest,
            "slot_id": self.slot_id,
            "target_spec_digest": self.target_spec_digest,
            "verification_profile_id": self.verification_profile_id,
        }


@dataclass(frozen=True)
class B300RegisteredTargetProjection:
    """Catalog-derived target identity plus exact ordered member contracts."""

    target_id: str
    target_spec_digest: str
    kind: str
    members: tuple[str, ...]
    member_contracts: tuple[B300MemberContractProjection, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target ID"))
        object.__setattr__(
            self,
            "target_spec_digest",
            _digest(self.target_spec_digest, "target-spec digest"),
        )
        if self.kind not in {TargetKind.SLOT.value, TargetKind.ATOMIC.value}:
            raise B300RegisteredQualificationError("target kind is not registered")
        members = tuple(self.members)
        contracts = tuple(self.member_contracts)
        if (
            type(self.members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or any(_identifier(row, "target member") != row for row in members)
            or type(self.member_contracts) is not tuple
            or any(type(row) is not B300MemberContractProjection for row in contracts)
            or tuple(row.slot_id for row in contracts) != members
            or (
                self.kind == TargetKind.SLOT.value
                and members != (self.target_id,)
            )
            or (self.kind == TargetKind.ATOMIC.value and len(members) < 2)
        ):
            raise B300RegisteredQualificationError(
                "registered target member-contract projection is not canonical"
            )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "member_contracts", contracts)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "member_contracts": [row.to_dict() for row in self.member_contracts],
            "members": list(self.members),
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
        }


def registered_b300_member_contract_projection(
    catalog: TargetCatalog,
    target_ids: tuple[str, ...],
) -> tuple[B300RegisteredTargetProjection, ...]:
    """Project one arena's registered targets into ordered graph authorities."""

    if type(catalog) is not TargetCatalog:
        raise B300RegisteredQualificationError(
            "qualification target catalog is not exact"
        )
    try:
        observed_ids = registered_b300_target_ids(catalog, target_ids)
    except (TypeError, ValueError) as exc:
        raise B300RegisteredQualificationError(
            f"registered B300 target catalog is malformed: {exc}"
        ) from None
    projection = []
    for target_id in observed_ids:
        spec = catalog.require(target_id)
        member_rows = []
        for member_id in spec.members:
            member_spec = catalog.require(member_id)
            contract = member_spec.contract_ref
            if (
                member_spec.kind is not TargetKind.SLOT
                or member_spec.members != (member_id,)
                or contract is None
            ):
                raise B300RegisteredQualificationError(
                    f"registered member {member_id!r} lacks a singleton contract"
                )
            member_rows.append(
                B300MemberContractProjection(
                    member_id,
                    catalog.target_spec_digest(member_id),
                    catalog.contract_digest(member_id),
                    contract.verification_profile_id,
                )
            )
        projection.append(
            B300RegisteredTargetProjection(
                target_id,
                catalog.target_spec_digest(target_id),
                spec.kind.value,
                spec.members,
                tuple(member_rows),
            )
        )
    return tuple(projection)


def registered_b300_profile_resolver_digest(
    target: B300RegisteredTargetProjection,
    *,
    builder_source_digest: str,
) -> str:
    """Bind one resolver to its builder and full target/member projection."""

    if type(target) is not B300RegisteredTargetProjection:
        raise B300RegisteredQualificationError(
            "profile resolver target projection is not exact"
        )
    return canonical_digest(
        RESOLVER_SCHEMA,
        {
            "builder_source_digest": _digest(
                builder_source_digest, "builder source digest"
            ),
            "target_authority": target.to_dict(),
        },
    )


@dataclass(frozen=True)
class B300RegisteredQualificationPolicy:
    """Candidate-independent policy for the exact registered target catalog."""

    catalog_digest: str
    model_profile_key: str
    target_spec_digests: tuple[tuple[str, str], ...]
    target_contract_projection: tuple[B300RegisteredTargetProjection, ...]
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
        object.__setattr__(
            self,
            "model_profile_key",
            _identifier(self.model_profile_key, "model profile key"),
        )
        rows = tuple(self.target_spec_digests)
        projection = tuple(self.target_contract_projection)
        registered_ids = tuple(target for target, _ in rows)
        if (
            type(self.target_spec_digests) is not tuple
            or not registered_ids
            or registered_ids != tuple(sorted(set(registered_ids)))
            or len({target for target, _ in rows}) != len(rows)
            or type(self.target_contract_projection) is not tuple
            or any(type(row) is not B300RegisteredTargetProjection for row in projection)
            or tuple(row.target_id for row in projection) != registered_ids
            or tuple((row.target_id, row.target_spec_digest) for row in projection)
            != rows
        ):
            raise B300RegisteredQualificationError(
                "registered target authorities do not exactly cover the catalog"
            )
        checked = tuple(
            (target, _digest(digest, f"{target} target-spec digest"))
            for target, digest in rows
        )
        object.__setattr__(self, "target_spec_digests", checked)
        object.__setattr__(self, "target_contract_projection", projection)
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
        registered_target_ids: tuple[str, ...],
        model_profile_key: str,
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
        projection = registered_b300_member_contract_projection(
            catalog, registered_target_ids
        )
        rows = tuple((row.target_id, row.target_spec_digest) for row in projection)
        return cls(
            catalog.digest,
            model_profile_key,
            rows,
            projection,
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
            for target in self.registered_target_ids
        )
        if observed != self.target_spec_digests:
            raise B300RegisteredQualificationError(
                "registered target specifications are stale"
            )
        if (
            registered_b300_member_contract_projection(
                catalog, self.registered_target_ids
            )
            != self.target_contract_projection
        ):
            raise B300RegisteredQualificationError(
                "registered target member-contract authority is stale"
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
            "model_profile_key": self.model_profile_key,
            "nll_tail_threshold": self.nll_tail_threshold,
            "select_count": self.select_count,
            "support_policy_digest": self.support_policy_digest,
            "target_contract_projection": [
                row.to_dict() for row in self.target_contract_projection
            ],
            "target_spec_digests": [list(row) for row in self.target_spec_digests],
            "tokens_per_prompt": self.tokens_per_prompt,
            "topk_width": self.topk_width,
            "verification_policy_digest": self.verification_policy_digest,
        }

    @property
    def registered_target_ids(self) -> tuple[str, ...]:
        return tuple(target for target, _ in self.target_spec_digests)

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
        ):
            raise B300RegisteredQualificationError(
                "focused graph facts lack canonical requirement/observation coverage"
            )
        variant_keys = tuple((row.slot_id, row.variant_id) for row in variants)
        observation_keys = tuple(
            (row.slot_id, row.variant_id) for row in observations
        )
        if (
            variant_keys != observation_keys
            or variant_keys != tuple(sorted(variant_keys))
            or len(set(variant_keys)) != len(variant_keys)
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
    [ArenaCandidateBinding, PreparedCandidateRuntime, str], B300FocusedGraphFacts
]


# Construction token: only the commissioner (and its tests) may build the
# inputs; every other caller goes through build_commissioned_b300_qualification_service.
_COMMISSION_SEAL = object()


@dataclass(frozen=True)
class SealedIncumbentBundle:
    """The one swappable incumbent contribution the v7 baseline read injects.

    Derived at commission from the durable incumbent stack entry and its
    resolver-verified source manifest — never from a runtime swap
    acknowledgement.  Absent at genesis and whenever the incumbent stack
    cannot be realized by a single swap (multiple entries, a non-swappable
    bundle, or a non-proposal reference); every candidate then routes to the
    version-8 two-process schedule, whose baseline boots the incumbent tree.
    """

    target_id: str
    bundle_digest: str
    slots: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bundle_digest",
            require_digest(
                self.bundle_digest,
                field="incumbent bundle digest",
                error=B300RegisteredQualificationError,
            ),
        )
        slots = self.slots
        if (
            type(self.target_id) is not str
            or not self.target_id
            or type(slots) is not tuple
            or not slots
            or any(type(slot) is not str or not slot for slot in slots)
            or list(slots) != sorted(set(slots))
        ):
            raise B300RegisteredQualificationError(
                "sealed incumbent bundle requires a target and sorted distinct slots"
            )


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
    incumbent_bundle: SealedIncumbentBundle | None = None
    seal: object = dc_field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Construction is sealed to the commissioner. Every object here is
        # built and validated at its own boundary (sealed config parsing,
        # B300QualificationCapabilities, the calibration loader, the typed
        # constructors); this pass keeps only the cross-object invariants
        # that no single boundary can see.
        if self.seal is not _COMMISSION_SEAL:
            raise B300RegisteredQualificationError(
                "registered qualification inputs are built only by the commissioner"
            )
        self.policy.require_catalog(self.catalog)
        if (
            self.expected_context.catalog_digest != self.catalog.digest
            or self.expected_context.catalog_snapshot != self.catalog.snapshot()
        ):
            raise B300RegisteredQualificationError(
                "evaluation-stack context differs from the registered target catalog"
            )
        for label, stack in (
            ("incumbent", self.incumbent_stack),
            ("pristine T", self.pristine_stack),
        ):
            try:
                stack.validate_against(self.expected_context)
            except (TypeError, ValueError) as exc:
                raise B300RegisteredQualificationError(
                    f"{label} stack is stale: {exc}"
                ) from None
        if (
            self.incumbent_binding.tree.stack_digest != self.incumbent_stack.digest
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
        if self.pristine_stack.entries:
            raise B300RegisteredQualificationError(
                "pristine T contains proposal contributions"
            )
        if (
            self.pristine_binding.tree.stack_digest != self.pristine_stack.digest
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
            self.resident_speed_policy.version < 3
            or marginal_workload_digest(self.resident_baseline_arm.session_plan)
            != marginal_workload_digest(self.baseline_session_plan)
            or self.resident_baseline_arm.executor_namespace_digest
            == self.candidate_executor_namespace_digest
        ):
            raise B300RegisteredQualificationError(
                "resident-v3 baseline/candidate lane authority is inconsistent"
            )
        if self.incumbent_bundle is not None:
            if type(self.incumbent_bundle) is not SealedIncumbentBundle:
                raise B300RegisteredQualificationError(
                    "sealed incumbent bundle authority is not exactly typed"
                )
            declared = self.incumbent_stack.entries.get(
                self.incumbent_bundle.target_id
            )
            if (
                declared is None
                or getattr(declared, "artifact_digest", None)
                != self.incumbent_bundle.bundle_digest
            ):
                raise B300RegisteredQualificationError(
                    "sealed incumbent bundle differs from the incumbent stack entry"
                )
        if (
            self.policy.tokens_per_prompt
            != self.baseline_session_plan.quality_tokens_per_prompt
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

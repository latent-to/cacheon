"""Closed B300 qualification composition for registered submitted bundles.

This is a qualification composition root, not a screening implementation or
plugin loader.  A remote request names only an exact
:class:`ArenaQualificationRequest` and retained ``primary``/``reproduction``
stage; it cannot select code, paths, profiles, executors, judges, or deadlines.

``B300RegisteredProfileAuthority`` binds the candidate-dynamic graph and
quality authority behind one sealed resolver per registered target.  This
module independently compares its exact typed result with the plan builder's
result.  No boolean "profile passed" callback is accepted.

Production resident speed-policy v3 permits one registered candidate at a
time, so larger cohorts fail closed and leases use ``max_members=1``.  That is
one marginal candidate, not one semantic member: an atomic target remains one
indivisible transition while binding every ordered member contract.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from cacheon._strict import require_digest
from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaQualificationRequest,
    ArenaScreenReceipt,
    ArenaServiceManifest,
    PromotionDecision,
)
from cacheon.eval.b300_arena_provider import (
    B300ArenaProviderError,
    B300DeploymentAuthorities,
    B300ScreenDeploymentAuthorities,
    b300_arena_provider_digest,
)
from cacheon.eval.b300_qualification_graph_store_io import B300QualificationGraphEvidenceHold
from cacheon.eval.marginal_runtime import PreparedCandidateRuntime
from cacheon.eval.oci_backend import OCIEngineExecutor
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
)
from cacheon.eval.qualification_prebuilt_plan import sealed_prebuilt_qualification_plan_factory
from cacheon.eval.qualification_runner import (
    CandidateQualificationAuthority,
    CausalQualificationInput,
    HiddenJudgeBinding,
    SpeedStageDisposition,
)
from cacheon.stack_identity import canonical_digest
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.stack_plan import MarginalArmPlan
from cacheon.target_catalog import TargetCatalog, default_target_catalog


def registered_b300_target_ids(catalog: TargetCatalog) -> tuple[str, ...]:
    """Return the exact canonical registered IDs carried by one catalog."""

    if type(catalog) is not TargetCatalog:
        raise TypeError("registered B300 catalog is not exact")
    rows = catalog.snapshot().get("targets")
    target_ids = (
        tuple(row.get("target_id") for row in rows)
        if isinstance(rows, list) and all(type(row) is dict for row in rows)
        else ()
    )
    checked = tuple(row for row in target_ids if isinstance(row, str))
    if (
        not checked
        or checked != target_ids
        or checked != tuple(sorted(set(checked)))
    ):
        raise ValueError("registered B300 catalog target rows are not canonical")
    return checked


REGISTERED_B300_TARGET_IDS = registered_b300_target_ids(default_target_catalog())
QUALIFICATION_SPEED_EVIDENCE_POLICY = (
    "resident-v3-one-candidate-registered-target.v2"
)
CONSTRUCTION_SCHEMA = "cacheon.eval.b300-qualification-construction.v2"
POLICY_SCHEMA = "cacheon.eval.b300-qualification-policy.v2"
REGISTRY_SCHEMA = "cacheon.eval.b300-qualification-profile-registry.v2"
COHORT_SCHEMA = "cacheon.eval.b300-qualification-cohort.v2"
SELECTION_REFERENCE_SCHEMA = (
    "cacheon.eval.b300-qualification-selection-secret-reference.v1"
)
_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_STAGES = frozenset({"primary", "reproduction"})


class B300QualificationDeploymentError(RuntimeError):
    """A qualification request differs from sealed deployment authority."""


def _digest(value: object, field: str) -> str:
    return require_digest(
        value,
        field=field,
        error=B300QualificationDeploymentError,
    )


def _target(value: object, field: str) -> str:
    if not isinstance(value, str) or _TARGET.fullmatch(value) is None:
        raise B300QualificationDeploymentError(f"{field} is not canonical")
    return value


def _evidence_root(value: object) -> Path:
    try:
        root = Path(value)  # type: ignore[arg-type]
    except TypeError:
        raise B300QualificationDeploymentError(
            "qualification evidence root is not path-like"
        ) from None
    posix = PurePosixPath(root.as_posix())
    if (
        not root.is_absolute()
        or not posix.is_absolute()
        or "." in posix.parts
        or ".." in posix.parts
        or root != Path(posix.as_posix())
        or root.is_symlink()
    ):
        raise B300QualificationDeploymentError(
            "qualification evidence root is not one canonical absolute authority"
        )
    return root


ProfileResolver = Callable[
    [ArenaCandidateBinding, PreparedCandidateRuntime],
    CandidateQualificationAuthority,
]


@dataclass(frozen=True)
class B300RegisteredProfileAuthority:
    """One validator-registered target's candidate-specific profile resolver.

    A graph requirement necessarily binds a candidate launch and selected delta,
    so it cannot be a static catalog row.  The resolver is deployment-owned and
    is selected only after the candidate's finalized registered target has been
    checked against the immutable :class:`TargetCatalog`.
    """

    target_id: str
    target_spec_digest: str
    resolver_digest: str
    resolver: ProfileResolver

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _target(self.target_id, "target ID"))
        for field in ("target_spec_digest", "resolver_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not callable(self.resolver):
            raise B300QualificationDeploymentError(
                "registered profile resolver is not callable"
            )

    def resolve(
        self,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
    ) -> CandidateQualificationAuthority:
        try:
            value = self.resolver(candidate, prepared)
        except B300QualificationGraphEvidenceHold:
            raise
        except Exception as exc:
            raise B300QualificationDeploymentError(
                "registered profile authority failed"
            ) from exc
        if type(value) is not CandidateQualificationAuthority:
            raise B300QualificationDeploymentError(
                "registered profile resolver returned an untyped authority"
            )
        return value


@dataclass(frozen=True)
class B300QualificationCohort:
    """Path-free identity for one exact promoted registered candidate.

    ``ArenaCandidateBinding`` contains a trusted pod-local immutable publication
    root, but its digest deliberately excludes that root.  No root or other
    control field is accepted separately by this boundary.
    """

    request: ArenaQualificationRequest
    screen_lane: str

    def __post_init__(self) -> None:
        if type(self.request) is not ArenaQualificationRequest:
            raise B300QualificationDeploymentError(
                "qualification request is not exactly typed"
            )
        if self.screen_lane not in _STAGES:
            raise B300QualificationDeploymentError(
                "qualification screen lane is unsupported"
            )
        # Resident speed-policy v3 is a one-candidate contract in
        # CausalQualificationInput.  An atomic candidate may bind multiple
        # semantic members, but it is never split into multiple candidates.
        if len(self.request.candidates) != 1:
            raise B300QualificationDeploymentError(
                "resident v3 qualification requires one submitted bundle"
            )
        candidate = self.request.candidates[0]
        receipt = self.request.screen_receipts[0]
        if (
            type(candidate) is not ArenaCandidateBinding
            or type(receipt) is not ArenaScreenReceipt
            or receipt.candidate_digest != candidate.digest
            or receipt.screen_attempt != candidate.screen_attempt
            or receipt.service_digest != self.request.service_digest
            or receipt.decision is not PromotionDecision.PROMOTE
        ):
            raise B300QualificationDeploymentError(
                "qualification cohort lacks exact promoted coverage"
            )

    @property
    def candidate(self) -> ArenaCandidateBinding:
        return self.request.candidates[0]

    @property
    def receipt(self) -> ArenaScreenReceipt:
        return self.request.screen_receipts[0]

    @property
    def digest(self) -> str:
        return canonical_digest(
            COHORT_SCHEMA,
            {
                "candidate_digest": self.candidate.digest,
                "qualification_policy_digest": (
                    self.request.qualification_policy_digest
                ),
                "screen_lane": self.screen_lane,
                "screen_receipt_digest": self.receipt.digest,
                "service_digest": self.request.service_digest,
            },
        )


SecretLoader = Callable[[str], bytes]
QualificationPlanBuilder = Callable[
    [B300QualificationCohort, bytes], CausalQualificationInput
]
QualificationDeadlineProvider = Callable[[B300QualificationCohort], float]


@dataclass(frozen=True)
class B300QualificationConstructionAuthority:
    """All non-executor authorities for one registered-target qualification.

    The identity properties intentionally exclude the host evidence path and
    Python callable representations.  Their deployment identities are supplied
    explicitly and digest-bound; the exact root and callable objects remain
    private in-process capabilities.
    """

    catalog: TargetCatalog
    profiles: tuple[B300RegisteredProfileAuthority, ...]
    incumbent_stack: EvaluationStackManifest
    incumbent_tree_digest: str
    pristine_stack: EvaluationStackManifest
    pristine_tree_digest: str
    evidence_root: Path
    evidence_policy_digest: str
    builder_source_digest: str
    selection_store_digest: str
    secret_loader: SecretLoader
    plan_builder: QualificationPlanBuilder
    entropy_provider_digest: str
    entropy_provider: object
    hidden_judge: object
    deadline_policy_digest: str
    deadline_provider: QualificationDeadlineProvider

    def __post_init__(self) -> None:
        if type(self.catalog) is not TargetCatalog:
            raise B300QualificationDeploymentError(
                "qualification target catalog is not exact"
            )
        rows = tuple(self.profiles)
        try:
            catalog_target_ids = registered_b300_target_ids(self.catalog)
        except (TypeError, ValueError):
            catalog_target_ids = ()
        if (
            type(self.profiles) is not tuple
            or any(type(row) is not B300RegisteredProfileAuthority for row in rows)
            or catalog_target_ids != REGISTERED_B300_TARGET_IDS
            or tuple(row.target_id for row in rows) != REGISTERED_B300_TARGET_IDS
        ):
            raise B300QualificationDeploymentError(
                "registered qualification profiles do not exactly cover the catalog"
            )
        try:
            from cacheon.eval.b300_registered_qualification_inputs import (
                registered_b300_member_contract_projection,
                registered_b300_profile_resolver_digest,
            )

            projection = registered_b300_member_contract_projection(self.catalog)
            expected_profiles = tuple(
                (
                    target.target_id,
                    target.target_spec_digest,
                    registered_b300_profile_resolver_digest(
                        target,
                        builder_source_digest=self.builder_source_digest,
                    ),
                )
                for target in projection
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise B300QualificationDeploymentError(
                f"registered qualification profile authority is invalid: {exc}"
            ) from None
        if tuple(
            (row.target_id, row.target_spec_digest, row.resolver_digest)
            for row in rows
        ) != expected_profiles:
            raise B300QualificationDeploymentError(
                "registered qualification target or member authority is stale"
            )
        for row in rows:
            try:
                self.catalog.require(row.target_id)
                expected_spec = self.catalog.target_spec_digest(row.target_id)
            except Exception as exc:
                raise B300QualificationDeploymentError(
                    f"qualification target {row.target_id!r} is not registered"
                ) from exc
            if row.target_spec_digest != expected_spec:
                raise B300QualificationDeploymentError(
                    f"qualification profile for {row.target_id!r} is stale"
                )
        object.__setattr__(self, "profiles", rows)
        if type(self.incumbent_stack) is not EvaluationStackManifest:
            raise B300QualificationDeploymentError(
                "incumbent qualification stack is not exact"
            )
        if (
            self.incumbent_stack.catalog_digest != self.catalog.digest
            or self.incumbent_stack.catalog_snapshot != self.catalog.snapshot()
        ):
            raise B300QualificationDeploymentError(
                "incumbent stack differs from the qualification target catalog"
            )
        object.__setattr__(
            self,
            "incumbent_tree_digest",
            _digest(self.incumbent_tree_digest, "incumbent tree digest"),
        )
        if type(self.pristine_stack) is not EvaluationStackManifest:
            raise B300QualificationDeploymentError(
                "pristine qualification stack is not exact"
            )
        if (
            self.pristine_stack.runtime_digest
            != self.incumbent_stack.runtime_digest
            or self.pristine_stack.base_engine_digest
            != self.incumbent_stack.base_engine_digest
            or self.pristine_stack.arena_digest != self.incumbent_stack.arena_digest
            or self.pristine_stack.catalog_digest != self.catalog.digest
            or self.pristine_stack.catalog_snapshot != self.catalog.snapshot()
            or self.pristine_stack.entries
        ):
            raise B300QualificationDeploymentError(
                "pristine T differs from the empty sealed evaluation context"
            )
        object.__setattr__(
            self,
            "pristine_tree_digest",
            _digest(self.pristine_tree_digest, "pristine tree digest"),
        )
        object.__setattr__(self, "evidence_root", _evidence_root(self.evidence_root))
        for field in (
            "evidence_policy_digest",
            "builder_source_digest",
            "selection_store_digest",
            "entropy_provider_digest",
            "deadline_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not all(
            callable(value)
            for value in (
                self.secret_loader,
                self.plan_builder,
                self.entropy_provider,
                self.hidden_judge,
                self.deadline_provider,
            )
        ):
            raise B300QualificationDeploymentError(
                "qualification construction authorities are not callable"
            )
        if type(getattr(self.hidden_judge, "binding", None)) is not HiddenJudgeBinding:
            raise B300QualificationDeploymentError(
                "qualification hidden judge lacks an exact sealed binding"
            )

    @property
    def profile_registry_digest(self) -> str:
        return canonical_digest(
            REGISTRY_SCHEMA,
            {
                "catalog_digest": self.catalog.digest,
                "profiles": [
                    {
                        "resolver_digest": row.resolver_digest,
                        "target_id": row.target_id,
                        "target_spec_digest": row.target_spec_digest,
                    }
                    for row in self.profiles
                ],
            },
        )

    @property
    def qualification_builder_digest(self) -> str:
        return canonical_digest(
            CONSTRUCTION_SCHEMA,
            {
                "builder_source_digest": self.builder_source_digest,
                "evidence_policy_digest": self.evidence_policy_digest,
                "profile_registry_digest": self.profile_registry_digest,
                "selection_store_digest": self.selection_store_digest,
                "speed_evidence_policy": QUALIFICATION_SPEED_EVIDENCE_POLICY,
            },
        )

    @property
    def qualification_policy_digest(self) -> str:
        binding = self.hidden_judge.binding
        assert type(binding) is HiddenJudgeBinding
        return canonical_digest(
            POLICY_SCHEMA,
            {
                "builder_digest": self.qualification_builder_digest,
                "deadline_policy_digest": self.deadline_policy_digest,
                "entropy_provider_digest": self.entropy_provider_digest,
                "hidden_judge_binding_digest": binding.digest,
                "profile_registry_digest": self.profile_registry_digest,
                "speed_evidence_policy": QUALIFICATION_SPEED_EVIDENCE_POLICY,
            },
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.b300-qualification-authority.v2",
            {
                "builder_digest": self.qualification_builder_digest,
                "incumbent_stack_digest": self.incumbent_stack.digest,
                "incumbent_tree_digest": self.incumbent_tree_digest,
                "policy_digest": self.qualification_policy_digest,
                "pristine_stack_digest": self.pristine_stack.digest,
                "pristine_tree_digest": self.pristine_tree_digest,
            },
        )

    def profile_for(self, target_id: str) -> B300RegisteredProfileAuthority:
        expected = _target(target_id, "qualification target ID")
        for row in self.profiles:
            if row.target_id == expected:
                return row
        raise B300QualificationDeploymentError(
            f"qualification target {expected!r} is unsupported"
        )

    def selection_secret_reference(self, cohort: B300QualificationCohort) -> str:
        if type(cohort) is not B300QualificationCohort:
            raise B300QualificationDeploymentError(
                "selection reference requires an exact cohort"
            )
        return canonical_digest(
            SELECTION_REFERENCE_SCHEMA,
            {
                "cohort_digest": cohort.digest,
                "construction_digest": self.digest,
                "selection_store_digest": self.selection_store_digest,
            },
        )


def _executor_ids(executor: OCIEngineExecutor, role: str) -> tuple[str, ...]:
    if type(executor) is not OCIEngineExecutor:
        raise B300QualificationDeploymentError(
            f"{role} qualification executor is not exact"
        )
    gpus = tuple(executor.device_policy.expected_gpus)
    ids = tuple(str(gpu.physical_id) for gpu in gpus)
    if (
        len(gpus) != 4
        or ids != tuple(sorted(set(ids), key=int))
        or any("B300" not in gpu.name.upper() for gpu in gpus)
        or len({gpu.uuid for gpu in gpus}) != 4
    ):
        raise B300QualificationDeploymentError(
            f"{role} qualification executor is not one canonical B300 TP4 lane"
        )
    return ids


def _executor_pair(
    candidate_executor: OCIEngineExecutor,
    resident_baseline_executor: OCIEngineExecutor,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidate_ids = _executor_ids(candidate_executor, "candidate")
    baseline_ids = _executor_ids(resident_baseline_executor, "resident baseline")
    candidate_uuids = {
        row.uuid for row in candidate_executor.device_policy.expected_gpus
    }
    baseline_uuids = {
        row.uuid for row in resident_baseline_executor.device_policy.expected_gpus
    }
    if (
        candidate_executor is resident_baseline_executor
        or candidate_executor.manager is resident_baseline_executor.manager
        or candidate_executor.manager.namespace_digest
        == resident_baseline_executor.manager.namespace_digest
        or set(candidate_ids).intersection(baseline_ids)
        or candidate_uuids.intersection(baseline_uuids)
    ):
        raise B300QualificationDeploymentError(
            "candidate and resident-baseline TP4 executors overlap"
        )
    return candidate_ids, baseline_ids


def _executor_matches_resident_arm(
    executor: OCIEngineExecutor,
    arm: object,
    *,
    role: str,
) -> None:
    try:
        launch = arm.launch  # type: ignore[attr-defined]
        binding = arm.binding  # type: ignore[attr-defined]
        physical = binding.physical_hardware
        policy = executor.device_policy
        config = executor.config
        manager = executor.manager
        matches = (
            tuple(map(str, policy.physical_gpu_ids)) == physical.physical_gpu_ids
            and policy.policy_sha256 == launch.hardware.device_policy_digest
            and physical.device_policy_digest == policy.policy_sha256
            and manager.namespace_digest == arm.executor_namespace_digest
            and config.prebuild.policy.resource_policy_digest
            == launch.resource_policy_digest
            and config.runtime.digest == arm.runtime_resource_policy_digest
            and policy.configuration_sha256 == arm.device_configuration_digest
        )
    except (AttributeError, TypeError, ValueError):
        matches = False
    if not matches:
        raise B300QualificationDeploymentError(
            f"{role} executor differs from the sealed resident arm"
        )


def _validate_profile_binding(
    authority: CandidateQualificationAuthority,
    candidate: ArenaCandidateBinding,
    prepared: PreparedCandidateRuntime,
    construction: B300QualificationConstructionAuthority,
) -> None:
    reservation = candidate.reservation
    graph = authority.graph_requirement.binding
    target = construction.catalog.require(reservation.target_id)
    expected_spec = construction.catalog.target_spec_digest(reservation.target_id)
    expected_members = []
    for member_id in target.members:
        member_spec = construction.catalog.require(member_id)
        contract = member_spec.contract_ref
        if contract is None or member_spec.members != (member_id,):
            raise B300QualificationDeploymentError(
                "registered target member lacks singleton contract authority"
            )
        expected_members.append(
            (
                member_id,
                construction.catalog.target_spec_digest(member_id),
                construction.catalog.contract_digest(member_id),
                contract.verification_profile_id,
            )
        )
    observed_members = tuple(
        (
            row.slot_id,
            row.target_spec_digest,
            row.contract_digest,
            row.verification_profile_id,
        )
        for row in graph.members
    )
    arm = prepared.arm
    if (
        type(arm) is not MarginalArmPlan
        or authority.selected_delta_digest != reservation.selected_delta_digest
        or graph.target_id != reservation.target_id
        or graph.target_spec_digest != expected_spec
        or graph.catalog_digest != construction.catalog.digest
        or graph.selected_delta_digest != reservation.selected_delta_digest
        or graph.marginal_arm_digest != arm.digest
        or graph.candidate_launch_digest != prepared.launch.digest
        or graph.contribution_ref_digest != arm.transition.replacement.digest
        or observed_members != tuple(expected_members)
        or tuple(reservation.target_members) != tuple(target.members)
    ):
        raise B300QualificationDeploymentError(
            "qualification profile/graph authority differs from the registered target"
        )
    profile = authority.profile
    judge = construction.hidden_judge.binding
    if (
        profile.reference.hidden_corpus_commitment
        != judge.hidden_corpus_commitment
        or profile.reference.hidden_judge_digest != judge.hidden_judge_digest
        or profile.hidden_task_policy_digest
        != judge.hidden_task_policy_digest
    ):
        raise B300QualificationDeploymentError(
            "qualification profile differs from the sealed hidden judge"
        )


def _validate_plan(
    value: object,
    cohort: B300QualificationCohort,
    secret: bytes,
    construction: B300QualificationConstructionAuthority,
    candidate_executor: OCIEngineExecutor,
    resident_baseline_executor: OCIEngineExecutor,
) -> CausalQualificationInput:
    if type(value) is not CausalQualificationInput:
        raise B300QualificationDeploymentError(
            "qualification builder returned an untyped causal plan"
        )
    candidate = cohort.candidate
    reservation = candidate.reservation
    prepared_rows = tuple(value.prepared.candidates)
    authorities = tuple(value.candidates)
    if (
        type(secret) is not bytes
        or len(secret) < 32
        or value.selection_secret != secret
        or type(value.prepared.source) is not MarginalArmPlan
        or len(prepared_rows) != 1
        or len(authorities) != 1
        or type(prepared_rows[0]) is not PreparedCandidateRuntime
        or type(authorities[0]) is not CandidateQualificationAuthority
        or value.speed_evidence_policy.version != 3
        or value.speed_stage_disposition is not SpeedStageDisposition.TERMINAL
        or value.resident_speed_plan is None
        or value.resident_audit_plan is None
        or value.evidence_root != construction.evidence_root
        or value.pristine_stack != construction.pristine_stack
        or value.pristine_launch.stack_digest != construction.pristine_stack.digest
        or value.pristine_launch.tree_digest != construction.pristine_tree_digest
    ):
        raise B300QualificationDeploymentError(
            "qualification plan differs from resident-v3 deployment authority"
        )
    prepared = prepared_rows[0]
    arm = prepared.arm
    assert type(arm) is MarginalArmPlan
    if (
        arm != value.prepared.source
        or arm.incumbent != construction.incumbent_stack
        or arm.baseline_before.tree_digest != construction.incumbent_tree_digest
        or value.prepared.incumbent_binding.tree.stack_digest
        != construction.incumbent_stack.digest
        or value.prepared.incumbent_binding.tree.tree_digest
        != construction.incumbent_tree_digest
        or arm.transition.target_id != reservation.target_id
        or arm.transition.target_spec_digest
        != construction.catalog.target_spec_digest(reservation.target_id)
        or arm.selected_delta_digest != reservation.selected_delta_digest
        or arm.transition.replacement.artifact_digest
        != candidate.publication.content_hash
    ):
        raise B300QualificationDeploymentError(
            "qualification marginal arm differs from finalized intake or incumbent"
        )
    registered = construction.profile_for(reservation.target_id)
    expected_authority = registered.resolve(candidate, prepared)
    observed_authority = authorities[0]
    if observed_authority != expected_authority:
        raise B300QualificationDeploymentError(
            "qualification builder substituted the registered profile authority"
        )
    _validate_profile_binding(
        observed_authority,
        candidate,
        prepared,
        construction,
    )
    resident = value.resident_speed_plan
    _executor_matches_resident_arm(
        candidate_executor,
        resident.candidate,
        role="candidate",
    )
    _executor_matches_resident_arm(
        resident_baseline_executor,
        resident.baseline,
        role="resident baseline",
    )
    return value


def _factory_builder(
    construction: B300QualificationConstructionAuthority,
    candidate_executor: OCIEngineExecutor,
    resident_baseline_executor: OCIEngineExecutor,
    *,
    screen_lane: str,
):
    if screen_lane not in _STAGES:
        raise B300QualificationDeploymentError(
            "qualification screen lane is unsupported"
        )

    def build(
        request: ArenaQualificationRequest,
        state: object | None,
    ) -> QualificationPlanFactory:
        # ``state`` is the only generic extension field in ArenaService's
        # provider seam.  Qualification deployment accepts no path, command, or
        # other operator control through it.
        if state is not None:
            raise B300QualificationDeploymentError(
                "qualification request supplied forbidden control state"
            )
        cohort = B300QualificationCohort(request, screen_lane)
        if (
            request.qualification_policy_digest
            != construction.qualification_policy_digest
            or request.service_digest != construction.incumbent_stack.arena_digest
        ):
            raise B300QualificationDeploymentError(
                "qualification request differs from the sealed policy or incumbent arena"
            )
        candidate = cohort.candidate
        construction.profile_for(candidate.reservation.target_id)
        reference = construction.selection_secret_reference(cohort)

        def load_secret(observed: str) -> bytes:
            if observed != reference:
                raise B300QualificationDeploymentError(
                    "selection secret reference was substituted"
                )
            try:
                secret = construction.secret_loader(observed)
            except Exception as exc:
                raise B300QualificationDeploymentError(
                    "private selection authority is unavailable"
                ) from exc
            if type(secret) is not bytes or len(secret) < 32:
                raise B300QualificationDeploymentError(
                    "private selection authority returned no 256-bit secret"
                )
            return secret

        def plan(secret: bytes) -> CausalQualificationInput:
            try:
                value = construction.plan_builder(cohort, secret)
            except B300QualificationGraphEvidenceHold:
                raise
            except Exception as exc:
                raise B300QualificationDeploymentError(
                    "sealed qualification plan construction failed"
                ) from exc
            return _validate_plan(
                value,
                cohort,
                secret,
                construction,
                candidate_executor,
                resident_baseline_executor,
            )

        # A public manifest cannot be fabricated from only a target ID: it
        # binds the candidate launch, graph authority and selection commitment.
        # Build the exact private plan once to derive that public identity, then
        # require QualificationPlanFactory to reproduce it on every reopen.
        secret = load_secret(reference)
        first = plan(secret)
        manifest = QualificationAuthorityManifest.seal(
            first,
            reservations=(candidate.reservation,),
            selection_secret_reference=reference,
        )
        return sealed_prebuilt_qualification_plan_factory(
            manifest, selection_secret_reference=reference,
            selection_secret=secret, plan=first,
        )

    return build


def _deadline_provider(
    construction: B300QualificationConstructionAuthority,
    *,
    screen_lane: str,
):
    def deadline(
        request: ArenaQualificationRequest,
        state: object | None,
    ) -> float:
        if state is not None:
            raise B300QualificationDeploymentError(
                "qualification deadline received forbidden control state"
            )
        cohort = B300QualificationCohort(request, screen_lane)
        try:
            value = construction.deadline_provider(cohort)
        except Exception as exc:
            raise B300QualificationDeploymentError(
                "qualification deadline authority failed"
            ) from exc
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise B300QualificationDeploymentError(
                "qualification deadline authority returned an invalid value"
            )
        return float(value)

    return deadline


@dataclass(frozen=True)
class B300QualificationDeployment:
    """One service-compatible full authority bundle and its role orientation."""

    manifest: ArenaServiceManifest
    authorities: B300DeploymentAuthorities
    screen_lane: str
    construction_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.manifest) is not ArenaServiceManifest
            or type(self.authorities) is not B300DeploymentAuthorities
            or self.screen_lane not in _STAGES
            or self.authorities.qualification_stage != self.screen_lane
        ):
            raise B300QualificationDeploymentError(
                "qualification deployment is not exactly typed"
            )
        object.__setattr__(
            self,
            "construction_digest",
            _digest(self.construction_digest, "qualification construction digest"),
        )
        if (
            self.manifest.runtime != self.authorities.runtime_identity
            or self.manifest.qualification_policy_digest
            != self.authorities.qualification_policy_digest
            or self.manifest.provider_digest
            != b300_arena_provider_digest(self.authorities)
        ):
            raise B300QualificationDeploymentError(
                "qualification authorities drifted from the screen service manifest"
            )


def compose_b300_qualification_deployment(
    *,
    manifest: ArenaServiceManifest,
    screen_authorities: B300ScreenDeploymentAuthorities,
    construction: B300QualificationConstructionAuthority,
    candidate_executor: OCIEngineExecutor,
    resident_baseline_executor: OCIEngineExecutor,
    screen_lane: str,
) -> B300QualificationDeployment:
    """Compose one exact full worker authority from validator-owned inputs.

    The screen worker must have declared the *same* qualification capability in
    advance.  A screen-only service whose declaration predicted overlapping
    executors, another profile registry, or another orientation is rejected; it
    cannot be upgraded in place by an operator flag.
    """

    if type(manifest) is not ArenaServiceManifest:
        raise B300QualificationDeploymentError(
            "qualification service manifest is not exact"
        )
    if type(screen_authorities) is not B300ScreenDeploymentAuthorities:
        raise B300QualificationDeploymentError(
            "screen deployment authorities are not exact"
        )
    if type(construction) is not B300QualificationConstructionAuthority:
        raise B300QualificationDeploymentError(
            "qualification construction authority is not exact"
        )
    if screen_lane not in _STAGES:
        raise B300QualificationDeploymentError(
            "qualification screen lane is unsupported"
        )
    _executor_pair(candidate_executor, resident_baseline_executor)
    if (
        manifest.runtime != screen_authorities.runtime_identity
        or manifest.provider_digest
        != b300_arena_provider_digest(screen_authorities)
        or construction.incumbent_stack.runtime_digest
        != manifest.runtime.runtime_digest
        or construction.incumbent_stack.base_engine_digest
        != manifest.runtime.base_engine_digest
        or construction.incumbent_stack.arena_digest != manifest.digest
        or manifest.qualification_policy_digest
        != construction.qualification_policy_digest
        or screen_authorities.qualification.qualification_policy_digest
        != construction.qualification_policy_digest
        or screen_authorities.qualification.qualification_builder_digest
        != construction.qualification_builder_digest
    ):
        raise B300QualificationDeploymentError(
            "screen manifest did not predeclare this qualification construction"
        )
    try:
        authorities = B300DeploymentAuthorities(
            runtime_identity=screen_authorities.runtime_identity,
            screen_handlers=screen_authorities.screen_handlers,
            resident_screen_factory=screen_authorities.resident_screen_factory,
            qualification_policy_digest=construction.qualification_policy_digest,
            qualification_builder_digest=(
                construction.qualification_builder_digest
            ),
            qualification_factory_builder=_factory_builder(
                construction,
                candidate_executor,
                resident_baseline_executor,
                screen_lane=screen_lane,
            ),
            executor=candidate_executor,
            resident_baseline_executor=resident_baseline_executor,
            entropy_provider_digest=construction.entropy_provider_digest,
            entropy_provider=construction.entropy_provider,
            hidden_judge=construction.hidden_judge,
            deadline_policy_digest=construction.deadline_policy_digest,
            deadline_provider=_deadline_provider(
                construction,
                screen_lane=screen_lane,
            ),
            qualification_lane_pair=screen_authorities.qualification.lane_pair,
            qualification_stage=screen_lane,
        )
    except B300ArenaProviderError as exc:
        raise B300QualificationDeploymentError(
            f"qualification executor orientation differs from declaration: {exc}"
        ) from exc
    # This exact equality is the key screen->qualification handoff.  The
    # provider-level declaration must be stable across the primary/reproduction
    # physical role swap; otherwise its manifest cannot validate both attempts.
    if authorities.qualification != screen_authorities.qualification:
        raise B300QualificationDeploymentError(
            "full qualification executors differ from the screen declaration"
        )
    return B300QualificationDeployment(
        manifest,
        authorities,
        screen_lane,
        construction.digest,
    )


__all__ = [
    "B300QualificationCohort",
    "B300QualificationConstructionAuthority",
    "B300QualificationDeployment",
    "B300QualificationDeploymentError",
    "B300RegisteredProfileAuthority",
    "COHORT_SCHEMA",
    "CONSTRUCTION_SCHEMA",
    "POLICY_SCHEMA",
    "QUALIFICATION_SPEED_EVIDENCE_POLICY",
    "REGISTERED_B300_TARGET_IDS",
    "REGISTRY_SCHEMA",
    "SELECTION_REFERENCE_SCHEMA",
    "compose_b300_qualification_deployment",
    "registered_b300_target_ids",
]

"""Registered-target gate for immutable resident fixed-stock count quality.

The registered authority in this module contains identities only.  It stores no
path, resident executor, model callable, or aggregate count.  The gate consumes
one already-completed candidate execution, independently regrades its raw A/B
evidence, reopens the sealed stock observation, and compares exact counts under
the registered policy.  Any authority or evidence ambiguity is a HOLD; only a
successfully reopened comparison may carry a candidate PASS or FAIL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cacheon.eval.b300_qualification_lanes import B300QualificationLanePair
from cacheon.eval.count_quality import CountQualityError, CountQualityPolicy
from cacheon.eval.engine_launch import EngineLaunchSpec
from cacheon.eval.marginal_runtime import MaterializedArmBinding
from cacheon.eval.numeric_answer_judge import (
    NumericAnswerHiddenJudge,
    derive_numeric_answer_prompt_occurrences,
    numeric_answer_prompt_plan_digest,
)
from cacheon.eval.oci_resident_session import ResidentBatchShape
from cacheon.eval.qualification import ReferenceManifest
from cacheon.eval.resident_count_quality import (
    ResidentCountQualityError,
    ResidentCountQualityResult,
    ResidentCountQualityStockAuthority,
    compare_resident_count_quality,
    reopen_resident_count_stock,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountLaneAdmission,
    ResidentCountQualityExecutionPlan,
    ResidentCountQualityExecutionResult,
    resident_batch_shape_digest,
    regrade_candidate_count_quality_execution,
)
from cacheon.stack_identity import StackIdentityError, canonical_digest, require_sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.target_catalog import TargetCatalog, TargetResolutionError


REGISTERED_RESIDENT_COUNT_QUALITY_AUTHORITY_SCHEMA = (
    "cacheon.eval.registered-resident-count-quality-authority.v1"
)
REGISTERED_RESIDENT_COUNT_QUALITY_RESULT_SCHEMA = (
    "cacheon.eval.registered-resident-count-quality-result.v1"
)

_TARGET_ID_RE = re.compile(r"^[0-9A-Za-z._\-]+$")


class RegisteredResidentCountQualityError(ValueError):
    """A registered count-quality authority or result is malformed."""


class RegisteredResidentCountQualityHold(RegisteredResidentCountQualityError):
    """Evidence is absent, foreign, or ambiguous and cannot convict a candidate."""

    decision = "HOLD"


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (StackIdentityError, TypeError, ValueError) as exc:
        raise RegisteredResidentCountQualityError(str(exc)) from None


def _target_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or _TARGET_ID_RE.fullmatch(value) is None
    ):
        raise RegisteredResidentCountQualityError(
            "registered count-quality target ID is not canonical"
        )
    return value


@dataclass(frozen=True)
class B300ResidentCountQualityBuilderContext:
    """Public stock identities supplied to the private count builder."""

    catalog: TargetCatalog
    pristine_stack: EvaluationStackManifest
    pristine_launch: EngineLaunchSpec
    pristine_binding: MaterializedArmBinding
    evidence_root: Path
    lane_pair: B300QualificationLanePair
    engine_max_running_requests: int

    def __post_init__(self) -> None:
        root = Path(self.evidence_root)
        if (
            type(self.catalog) is not TargetCatalog
            or type(self.pristine_stack) is not EvaluationStackManifest
            or type(self.pristine_launch) is not EngineLaunchSpec
            or type(self.pristine_binding) is not MaterializedArmBinding
            or type(self.lane_pair) is not B300QualificationLanePair
            or not root.is_absolute()
            or not root.is_dir()
            or root.is_symlink()
            or type(self.engine_max_running_requests) is not int
            or self.engine_max_running_requests < 1
            or self.pristine_stack.entries
            or self.pristine_launch.stack_digest != self.pristine_stack.digest
        ):
            raise RegisteredResidentCountQualityError(
                "resident count builder context is not one stock commission"
            )
        object.__setattr__(self, "evidence_root", root)


@dataclass(frozen=True)
class B300ResidentCountQualityCapability:
    """Static GSM8K/count authority shared by every registered target."""

    catalog: TargetCatalog
    envelope: object
    prompt_batches: tuple[tuple[str, ...], ...]
    selected_ordinals: tuple[int, ...]
    batch_shape: ResidentBatchShape
    admission: ResidentCountLaneAdmission
    stock_authority: ResidentCountQualityStockAuthority
    judge: NumericAnswerHiddenJudge

    def __post_init__(self) -> None:
        from cacheon.eval.resident_count_quality import ResidentCountQualityEnvelope

        if (
            type(self.catalog) is not TargetCatalog
            or type(self.envelope) is not ResidentCountQualityEnvelope
            or type(self.prompt_batches) is not tuple
            or not self.prompt_batches
            or any(
                type(batch) is not tuple
                or not batch
                or any(type(prompt) is not str or not prompt for prompt in batch)
                for batch in self.prompt_batches
            )
            or type(self.selected_ordinals) is not tuple
            or type(self.batch_shape) is not ResidentBatchShape
            or type(self.admission) is not ResidentCountLaneAdmission
            or type(self.stock_authority) is not ResidentCountQualityStockAuthority
            or type(self.judge) is not NumericAnswerHiddenJudge
        ):
            raise RegisteredResidentCountQualityError(
                "resident count capability is not exactly typed"
            )
        occurrences = derive_numeric_answer_prompt_occurrences(
            self.envelope.judge_binding,
            prompt_batches=self.prompt_batches,
            workload_digest=self.envelope.reference.workload_digest,
            hidden_tasks_per_prompt=1,
        )
        selected = tuple(self.selected_ordinals)
        if (
            selected != tuple(sorted(set(selected)))
            or any(
                type(index) is not int or not 0 <= index < len(occurrences)
                for index in selected
            )
            or len(selected) != self.envelope.expected_prompt_count
            or numeric_answer_prompt_plan_digest(
                tuple(occurrences[index] for index in selected)
            )
            != self.envelope.prompt_plan_digest
            or resident_batch_shape_digest(self.batch_shape)
            != self.envelope.generation_shape_digest
            or self.admission.digest != self.envelope.admission_policy_digest
            or self.stock_authority.envelope_digest != self.envelope.digest
            or self.judge.binding != self.envelope.judge_binding
            or self.judge.tokenizer_digest != self.envelope.reference.tokenizer_digest
        ):
            raise RegisteredResidentCountQualityError(
                "resident count capability authorities disagree"
            )
        object.__setattr__(self, "selected_ordinals", selected)

    def validate(self, context: B300ResidentCountQualityBuilderContext) -> None:
        """Reopen stock and bind every runtime-dependent commission identity."""

        if type(context) is not B300ResidentCountQualityBuilderContext:
            raise RegisteredResidentCountQualityError("resident count context is not exact")
        reference = self.envelope.reference
        expected_reference = ReferenceManifest.from_pristine(
            context.pristine_stack,
            context.pristine_launch,
            context.pristine_binding,
            workload_digest=reference.workload_digest,
            tokenizer_digest=reference.tokenizer_digest,
            hidden_corpus_commitment=reference.hidden_corpus_commitment,
            hidden_judge_digest=reference.hidden_judge_digest,
            selection_policy_digest=reference.selection_policy_digest,
        )
        if (
            self.catalog.digest != context.catalog.digest
            or reference != expected_reference
            or (
                self.admission.lane_a_allocation_digest,
                self.admission.lane_b_allocation_digest,
            )
            != (context.lane_pair.lane_a.digest, context.lane_pair.lane_b.digest)
            or self.admission.engine_max_running_requests
            != context.engine_max_running_requests
        ):
            raise RegisteredResidentCountQualityError(
                "resident count capability differs from the B300 commission"
            )
        reopen_resident_count_stock(
            context.evidence_root,
            self.stock_authority,
            expected_envelope=self.envelope,
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.b300-resident-count-quality-capability.v1",
            {
                "admission": self.admission.digest,
                "catalog": self.catalog.digest,
                "envelope": self.envelope.digest,
                "judge": self.judge.binding.digest,
                "selected_ordinals": list(self.selected_ordinals),
                "shape": resident_batch_shape_digest(self.batch_shape),
                "stock": self.stock_authority.digest,
            },
        )


@dataclass(frozen=True)
class RegisteredResidentCountQualityAuthority:
    """Path-free registered profile for one exact target and candidate plan.

    Pair and candidate-bundle identities are deliberately not duplicated here.
    They remain owned by the supplied execution plan and raw execution result;
    ``execution_plan_digest`` seals that plan without inventing another profile.
    """

    target_id: str
    catalog_digest: str
    target_spec_digest: str
    execution_envelope_digest: str
    execution_plan_digest: str
    fixed_stock_authority_digest: str
    judge_binding_digest: str
    tokenizer_digest: str
    policy: CountQualityPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _target_id(self.target_id))
        for field_name in (
            "catalog_digest",
            "target_spec_digest",
            "execution_envelope_digest",
            "execution_plan_digest",
            "fixed_stock_authority_digest",
            "judge_binding_digest",
            "tokenizer_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name.replace("_", " ")),
            )
        if type(self.policy) is not CountQualityPolicy:
            raise RegisteredResidentCountQualityError(
                "registered count-quality policy is not exact"
            )

    @classmethod
    def register(
        cls,
        catalog: TargetCatalog,
        target_id: str,
        *,
        plan: ResidentCountQualityExecutionPlan,
        stock_authority: ResidentCountQualityStockAuthority,
        judge: NumericAnswerHiddenJudge,
        policy: CountQualityPolicy,
    ) -> "RegisteredResidentCountQualityAuthority":
        """Seal one live catalog/profile tuple without target-specific branching."""

        if type(catalog) is not TargetCatalog:
            raise RegisteredResidentCountQualityError(
                "registered count-quality catalog is not exact"
            )
        exact_target = _target_id(target_id)
        if (
            type(plan) is not ResidentCountQualityExecutionPlan
            or type(stock_authority) is not ResidentCountQualityStockAuthority
            or type(judge) is not NumericAnswerHiddenJudge
            or type(policy) is not CountQualityPolicy
        ):
            raise RegisteredResidentCountQualityError(
                "registered count-quality profile inputs are not exact"
            )
        try:
            target = catalog.require(exact_target)
        except TargetResolutionError as exc:
            raise RegisteredResidentCountQualityError(str(exc)) from None
        if target.target_id != exact_target:
            raise RegisteredResidentCountQualityError(
                "registered count-quality target is not canonical"
            )
        if (
            stock_authority.envelope_digest != plan.envelope.digest
            or stock_authority.policy != policy
            or judge.binding != plan.envelope.judge_binding
            or judge.tokenizer_digest != plan.envelope.reference.tokenizer_digest
        ):
            raise RegisteredResidentCountQualityError(
                "registered count-quality profile authorities disagree"
            )
        return cls(
            exact_target,
            catalog.digest,
            catalog.target_spec_digest(exact_target),
            plan.envelope.digest,
            plan.digest,
            stock_authority.digest,
            judge.binding.digest,
            judge.tokenizer_digest,
            policy,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_digest": self.catalog_digest,
            "execution_envelope_digest": self.execution_envelope_digest,
            "execution_plan_digest": self.execution_plan_digest,
            "fixed_stock_authority_digest": self.fixed_stock_authority_digest,
            "judge_binding_digest": self.judge_binding_digest,
            "policy": self.policy.to_dict(),
            "schema": REGISTERED_RESIDENT_COUNT_QUALITY_AUTHORITY_SCHEMA,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
            "tokenizer_digest": self.tokenizer_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            REGISTERED_RESIDENT_COUNT_QUALITY_AUTHORITY_SCHEMA,
            self.to_dict(),
        )


@dataclass(frozen=True)
class RegisteredResidentCountQualityResult:
    """Closed receipt for one independently regraded fixed-stock comparison."""

    target_id: str
    catalog_digest: str
    target_spec_digest: str
    profile_digest: str
    execution_envelope_digest: str
    execution_plan_digest: str
    pair_binding_digest: str
    candidate_bundle_digest: str
    raw_execution_evidence_digest: str
    fixed_stock_authority_digest: str
    stock_observation_digest: str
    candidate_observation_digest: str
    policy_digest: str
    count_quality_result_digest: str
    count_quality_result: ResidentCountQualityResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _target_id(self.target_id))
        for field_name in (
            "catalog_digest",
            "target_spec_digest",
            "profile_digest",
            "execution_envelope_digest",
            "execution_plan_digest",
            "pair_binding_digest",
            "candidate_bundle_digest",
            "raw_execution_evidence_digest",
            "fixed_stock_authority_digest",
            "stock_observation_digest",
            "candidate_observation_digest",
            "policy_digest",
            "count_quality_result_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name.replace("_", " ")),
            )
        result = self.count_quality_result
        if type(result) is not ResidentCountQualityResult or (
            result.digest != self.count_quality_result_digest
            or result.stock_observation_digest != self.stock_observation_digest
            or result.candidate_observation_digest
            != self.candidate_observation_digest
            or result.policy.digest != self.policy_digest
        ):
            raise RegisteredResidentCountQualityError(
                "registered count-quality result does not recompute"
            )

    @property
    def decision(self) -> str:
        return self.count_quality_result.verdict.decision

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_bundle_digest": self.candidate_bundle_digest,
            "candidate_observation_digest": self.candidate_observation_digest,
            "catalog_digest": self.catalog_digest,
            "count_quality_result": self.count_quality_result.to_dict(),
            "count_quality_result_digest": self.count_quality_result_digest,
            "execution_envelope_digest": self.execution_envelope_digest,
            "execution_plan_digest": self.execution_plan_digest,
            "fixed_stock_authority_digest": self.fixed_stock_authority_digest,
            "pair_binding_digest": self.pair_binding_digest,
            "policy_digest": self.policy_digest,
            "profile_digest": self.profile_digest,
            "raw_execution_evidence_digest": self.raw_execution_evidence_digest,
            "schema": REGISTERED_RESIDENT_COUNT_QUALITY_RESULT_SCHEMA,
            "stock_observation_digest": self.stock_observation_digest,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            REGISTERED_RESIDENT_COUNT_QUALITY_RESULT_SCHEMA,
            self.to_dict(),
        )


def evaluate_registered_resident_count_quality(
    evidence_root: str | Path,
    *,
    catalog: TargetCatalog,
    target_id: str,
    authority: RegisteredResidentCountQualityAuthority,
    plan: ResidentCountQualityExecutionPlan,
    execution: ResidentCountQualityExecutionResult,
    stock_authority: ResidentCountQualityStockAuthority,
    judge: NumericAnswerHiddenJudge,
) -> RegisteredResidentCountQualityResult:
    """Reopen stock and independently regrade one completed candidate execution."""

    try:
        if type(catalog) is not TargetCatalog:
            raise RegisteredResidentCountQualityHold(
                "registered count-quality catalog is not exact"
            )
        exact_target = _target_id(target_id)
        if (
            type(authority) is not RegisteredResidentCountQualityAuthority
            or type(plan) is not ResidentCountQualityExecutionPlan
            or type(execution) is not ResidentCountQualityExecutionResult
            or type(stock_authority) is not ResidentCountQualityStockAuthority
            or type(judge) is not NumericAnswerHiddenJudge
            or not isinstance(evidence_root, (str, Path))
        ):
            raise RegisteredResidentCountQualityHold(
                "registered count-quality gate inputs are not exact"
            )
        try:
            target = catalog.require(exact_target)
            live_spec_digest = catalog.target_spec_digest(exact_target)
        except TargetResolutionError as exc:
            raise RegisteredResidentCountQualityHold(str(exc)) from None
        if exact_target != authority.target_id or target.target_id != exact_target:
            raise RegisteredResidentCountQualityHold(
                "registered count-quality target differs from profile authority"
            )
        if catalog.digest != authority.catalog_digest:
            raise RegisteredResidentCountQualityHold(
                "registered count-quality catalog authority is stale"
            )
        if live_spec_digest != authority.target_spec_digest:
            raise RegisteredResidentCountQualityHold(
                "registered count-quality target-spec authority is stale"
            )
        if (
            plan.envelope.digest != authority.execution_envelope_digest
            or plan.digest != authority.execution_plan_digest
        ):
            raise RegisteredResidentCountQualityHold(
                "registered count-quality execution plan was substituted"
            )
        if (
            stock_authority.digest != authority.fixed_stock_authority_digest
            or stock_authority.envelope_digest
            != authority.execution_envelope_digest
            or stock_authority.policy != authority.policy
        ):
            raise RegisteredResidentCountQualityHold(
                "registered fixed-stock or policy authority was substituted"
            )
        if (
            judge.binding.digest != authority.judge_binding_digest
            or judge.binding != plan.envelope.judge_binding
            or judge.tokenizer_digest != authority.tokenizer_digest
            or judge.tokenizer_digest != plan.envelope.reference.tokenizer_digest
        ):
            raise RegisteredResidentCountQualityHold(
                "registered numeric-judge authority was substituted"
            )

        regraded = regrade_candidate_count_quality_execution(
            execution.evidence,
            plan=plan,
            judge=judge,
        )
        if execution.observation != regraded or execution.observation.digest != regraded.digest:
            raise RegisteredResidentCountQualityHold(
                "candidate observation differs from independent raw regrade"
            )
        stock = reopen_resident_count_stock(
            evidence_root,
            stock_authority,
            expected_envelope=plan.envelope,
        )
        comparison = compare_resident_count_quality(
            stock,
            regraded,
            judge=judge,
            policy=authority.policy,
        )
        return RegisteredResidentCountQualityResult(
            exact_target,
            catalog.digest,
            live_spec_digest,
            authority.digest,
            plan.envelope.digest,
            plan.digest,
            plan.pair_binding.digest,
            plan.candidate_bundle_digest,
            execution.evidence.digest,
            stock_authority.digest,
            stock.digest,
            regraded.digest,
            authority.policy.digest,
            comparison.digest,
            comparison,
        )
    except RegisteredResidentCountQualityHold:
        raise
    except (
        CountQualityError,
        ResidentCountQualityError,
        RegisteredResidentCountQualityError,
        TargetResolutionError,
        TypeError,
        ValueError,
    ) as exc:
        raise RegisteredResidentCountQualityHold(
            f"registered resident count quality is on HOLD: {exc}"
        ) from None


__all__ = [
    "B300ResidentCountQualityBuilderContext",
    "B300ResidentCountQualityCapability",
    "REGISTERED_RESIDENT_COUNT_QUALITY_AUTHORITY_SCHEMA",
    "REGISTERED_RESIDENT_COUNT_QUALITY_RESULT_SCHEMA",
    "RegisteredResidentCountQualityAuthority",
    "RegisteredResidentCountQualityError",
    "RegisteredResidentCountQualityHold",
    "RegisteredResidentCountQualityResult",
    "evaluate_registered_resident_count_quality",
]

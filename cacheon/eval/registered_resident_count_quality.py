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

from cacheon.eval.count_quality import CountQualityError, CountQualityPolicy
from cacheon.eval.numeric_answer_judge import NumericAnswerHiddenJudge
from cacheon.eval.resident_count_quality import (
    ResidentCountQualityError,
    ResidentCountQualityResult,
    ResidentCountQualityStockAuthority,
    compare_resident_count_quality,
    reopen_resident_count_stock,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionPlan,
    ResidentCountQualityExecutionResult,
    regrade_candidate_count_quality_execution,
)
from cacheon.stack_identity import StackIdentityError, canonical_digest, require_sha256_hex
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
    "REGISTERED_RESIDENT_COUNT_QUALITY_AUTHORITY_SCHEMA",
    "REGISTERED_RESIDENT_COUNT_QUALITY_RESULT_SCHEMA",
    "RegisteredResidentCountQualityAuthority",
    "RegisteredResidentCountQualityError",
    "RegisteredResidentCountQualityHold",
    "RegisteredResidentCountQualityResult",
    "evaluate_registered_resident_count_quality",
]

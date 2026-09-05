"""Sealed qualification commission identity and digest prediction.

``B300QualificationConstructionAuthority``'s policy digest must appear in a
service manifest *before* any candidate work exists, while the registered
factory's own ``builder_source_digest`` property deliberately binds the
frozen calibration and reference manifests -- values that themselves embed
the service digest.  Declaring the factory-derived identity inside the
manifest would therefore be circular.  The sealed commission block instead
carries one *reviewed* builder source identity (plus the other capability
digests), and the helpers below predict the exact construction digests from
only the sealed block, the target catalog, and the sealed prompt identity.
The commissioning composer later re-derives the same digests from the real
construction authority; any drift fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cacheon.eval.b300_qualification_deployment import (
    CONSTRUCTION_SCHEMA as QUALIFICATION_CONSTRUCTION_SCHEMA,
    POLICY_SCHEMA as QUALIFICATION_POLICY_SCHEMA,
    QUALIFICATION_SPEED_EVIDENCE_POLICY,
    REGISTRY_SCHEMA as QUALIFICATION_REGISTRY_SCHEMA,
)
from cacheon.eval.b300_registered_qualification_inputs import (
    B300RegisteredQualificationError,
    _digest,
    registered_b300_member_contract_projection,
    registered_b300_profile_resolver_digest,
)
from cacheon.eval.calibration import (
    CalibrationContext,
    CalibrationEvidenceSet,
    CalibrationError,
    CalibrationManifest,
    CalibrationThresholdPolicy,
    decimal_value,
    derive_calibration_manifest,
)
from cacheon.eval.qualification import declared_qualification_entropy_digest
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.stack_identity import canonical_digest
from cacheon.target_catalog import TargetCatalog


QUALIFICATION_COMMISSION_SCHEMA = "cacheon-private-b300-qualification-commission-v3"
QUALIFICATION_DEADLINE_MAXIMUM_SECONDS = 14_400
CALIBRATION_PACKAGE_SCHEMA = "cacheon-private-b300-calibration-pair-v1"
QUALIFICATION_STAGES = frozenset({"primary", "reproduction"})

QUALIFICATION_EVIDENCE_POLICY_DIGEST = canonical_digest(
    "cacheon.eval.b300-qualification-evidence-policy.v1",
    {
        "calibration": "frozen-reopened",
        "evidence_root": "content-addressed",
        "graph_observations": "published-and-reopened",
    },
)

_COMMISSION_FIELDS = frozenset(
    {
        "builder_source_digest",
        "candidate_binding_builder_digest",
        "graph_facts_builder_digest",
        "policy",
        "resident_speed",
        "resident_count_quality_builder_digest",
        "schema",
        "selection_store_digest",
        "session",
        "source_resolver_digest",
        "support_policy_digest",
        "verification_policy_digest",
    }
)
_COMMISSION_POLICY_FIELDS = frozenset(
    {
        "audit_minimum_calls",
        "hidden_tasks_per_prompt",
        "hidden_tasks_required",
        "nll_tail_threshold",
        "select_count",
        "tokens_per_prompt",
        "topk_width",
    }
)
_COMMISSION_SESSION_FIELDS = frozenset(
    {"conditioning_count", "temperature", "warmup_count"}
)
_COMMISSION_SPEED_FIELDS = frozenset(
    {
        "max_conditioning_slowdown",
        "max_qualification_seconds",
        "max_stage_seconds",
        "max_window_scatter",
        "min_windows",
    }
)
_CALIBRATION_RECORD_FIELDS = frozenset(
    {
        "evidence",
        "manifest",
        "measurement_authority",
        "threshold_policy",
    }
)
_CALIBRATION_MEASUREMENT_FIELDS = frozenset(
    {
        "context_digest",
        "logical_hardware_digest",
        "projection_sha256",
        "raw_quality_artifact_sha256",
        "raw_quality_binding_digest",
        "report_digest",
        "source_attempt_digest",
        "source_attempt_ref_sha256",
        "transform",
    }
)
_CALIBRATION_TRANSFORMS = frozenset(
    {
        "validator-owned-hidden-task-fail.v1",
        "validator-owned-teacher-nll-fail.v1",
    }
)


class B300QualificationCommissionError(RuntimeError):
    """Sealed qualification commissioning failed closed."""


@dataclass(frozen=True)
class B300QualificationCapabilities:
    """Validator-private callables plus their sealed reviewed identities.

    ``incumbent_entries`` declares the durable evaluation stack's crowned
    contributions (empty at genesis). The commission constructs the measured
    baseline from these entries; the CPU dispatcher's pinned incumbent and the
    durable commit both refuse any product whose declared entries do not
    reproduce the durable stack digest, so a wrong declaration fails closed
    end to end.
    """

    secret_loader: Callable[[str], bytes]
    entropy_provider: object
    hidden_judge: object
    source_resolver: object
    source_resolver_digest: str
    graph_facts_builder: object
    graph_facts_builder_digest: str
    resident_count_quality_builder: object
    resident_count_quality_builder_digest: str
    incumbent_entries: dict[str, object]

    def __post_init__(self) -> None:
        from cacheon.stack_manifest import ProposalContributionRef

        if type(self.incumbent_entries) is not dict or any(
            type(target) is not str
            or type(ref) is not ProposalContributionRef
            for target, ref in self.incumbent_entries.items()
        ):
            raise B300QualificationCommissionError(
                "incumbent entries are not an exact target-to-contribution mapping"
            )
        if (
            not callable(self.secret_loader)
            or not callable(self.entropy_provider)
            or not (
                callable(self.hidden_judge)
                or callable(getattr(self.hidden_judge, "bind_prompt_plan", None))
            )
            or not callable(self.graph_facts_builder)
            or not callable(self.resident_count_quality_builder)
            or not callable(getattr(self.source_resolver, "resolve_proposal", None))
        ):
            raise B300QualificationCommissionError(
                "qualification capabilities are not callable"
            )
        if type(getattr(self.hidden_judge, "binding", None)) is not HiddenJudgeBinding:
            raise B300QualificationCommissionError(
                "hidden judge capability lacks an exact sealed binding"
            )
        for field in (
            "source_resolver_digest",
            "graph_facts_builder_digest",
            "resident_count_quality_builder_digest",
        ):
            value = getattr(self, field)
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise B300QualificationCommissionError(
                    f"capability {field} is not one SHA-256 identity"
                )


def parse_sealed_calibration_package(
    value: object,
    context: CalibrationContext,
    stage: str,
) -> tuple[
    CalibrationThresholdPolicy,
    CalibrationManifest,
    CalibrationEvidenceSet,
]:
    """Reopen one closed two-orientation calibration package."""

    stages = value.get("stages") if type(value) is dict else None
    if (
        type(value) is not dict
        or set(value) != {"schema", "stages"}
        or value.get("schema") != CALIBRATION_PACKAGE_SCHEMA
        or type(stages) is not dict
        or set(stages) != QUALIFICATION_STAGES
        or any(
            type(stages[name]) is not dict
            or set(stages[name]) != _CALIBRATION_RECORD_FIELDS
            for name in QUALIFICATION_STAGES
        )
        or type(context) is not CalibrationContext
        or stage not in QUALIFICATION_STAGES
    ):
        raise B300QualificationCommissionError(
            "sealed calibration package is not one closed frozen authority"
        )
    try:
        parsed = {}
        templates = {}
        source_digests = {}
        for name in QUALIFICATION_STAGES:
            record = stages[name]
            threshold = CalibrationThresholdPolicy.from_dict(
                record["threshold_policy"]
            )
            manifest = CalibrationManifest.from_dict(record["manifest"])
            evidence = CalibrationEvidenceSet.from_dict(record["evidence"])
            source = record["measurement_authority"]
            if (
                threshold.status != "frozen"
                or type(source) is not dict
                or set(source) != _CALIBRATION_MEASUREMENT_FIELDS
                or source["context_digest"] != threshold.context.digest
                or source["logical_hardware_digest"]
                != threshold.context.logical_hardware_digest
                or source["transform"] not in _CALIBRATION_TRANSFORMS
            ):
                raise ValueError(
                    f"{name} calibration measurement authority is not exact"
                )
            for field in _CALIBRATION_MEASUREMENT_FIELDS - {"transform"}:
                _digest(source[field], f"{name} calibration {field}")
            derived = derive_calibration_manifest(
                threshold, evidence.observations
            )
            if (
                evidence.threshold_policy_digest != threshold.digest
                or evidence.configured_manifest_digest != manifest.digest
                or derived != manifest
                or manifest.context != threshold.context
            ):
                raise ValueError(f"{name} calibration evidence was relabeled")
            template = threshold.to_dict()
            del template["context"]
            parsed[name] = (threshold, manifest, evidence)
            templates[name] = template
            source_digests[name] = canonical_digest(
                "cacheon.private.b300-calibration-measurement-authority.v1",
                source,
            )
        primary = parsed["primary"]
        reproduction = parsed["reproduction"]
        if (
            templates["primary"] != templates["reproduction"]
            or primary[0].context == reproduction[0].context
            or primary[1].raw_evidence_digest
            == reproduction[1].raw_evidence_digest
            or source_digests["primary"] == source_digests["reproduction"]
        ):
            raise ValueError("lane calibration authorities are not independent")
        if parsed[stage][0].context != context:
            raise ValueError(
                f"{stage} calibration context differs from the commissioned lane"
            )
    except (
        B300RegisteredQualificationError,
        CalibrationError,
        TypeError,
        ValueError,
    ) as exc:
        raise B300QualificationCommissionError(
            f"sealed calibration package is invalid: {exc}"
        ) from None
    return parsed[stage]


def _commission_int(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise B300RegisteredQualificationError(
            f"sealed qualification {field} must be an integer >= {minimum}"
        )
    return value


def _commission_decimal(value: object, field: str) -> str:
    if type(value) is not str:
        raise B300RegisteredQualificationError(
            f"sealed qualification {field} must be a canonical decimal string"
        )
    try:
        decimal_value(value)
    except CalibrationError:
        raise B300RegisteredQualificationError(
            f"sealed qualification {field} must be a canonical decimal string"
        ) from None
    return value


def sealed_qualification_commission(value: object) -> dict[str, object]:
    """Validate one closed sealed qualification commission block."""

    if (
        type(value) is not dict
        or set(value) != _COMMISSION_FIELDS
        or value.get("schema") != QUALIFICATION_COMMISSION_SCHEMA
    ):
        raise B300RegisteredQualificationError(
            "sealed qualification commission block is not closed"
        )
    for field in (
        "builder_source_digest",
        "candidate_binding_builder_digest",
        "graph_facts_builder_digest",
        "resident_count_quality_builder_digest",
        "selection_store_digest",
        "source_resolver_digest",
        "support_policy_digest",
        "verification_policy_digest",
    ):
        _digest(value.get(field), f"sealed qualification {field}")
    policy = value.get("policy")
    if (
        type(policy) is not dict
        or set(policy) != _COMMISSION_POLICY_FIELDS
        or not isinstance(policy.get("nll_tail_threshold"), str)
        or type(policy.get("hidden_tasks_required")) is not bool
    ):
        raise B300RegisteredQualificationError(
            "sealed qualification policy block is not closed"
        )
    _commission_int(policy.get("tokens_per_prompt"), "tokens_per_prompt", minimum=1)
    _commission_int(policy.get("topk_width"), "topk_width", minimum=0)
    _commission_int(
        policy.get("hidden_tasks_per_prompt"), "hidden_tasks_per_prompt", minimum=0
    )
    _commission_int(policy.get("select_count"), "select_count", minimum=2)
    _commission_int(policy.get("audit_minimum_calls"), "audit_minimum_calls", minimum=1)
    session = value.get("session")
    if type(session) is not dict or set(session) != _COMMISSION_SESSION_FIELDS:
        raise B300RegisteredQualificationError(
            "sealed qualification session block is not closed"
        )
    _commission_int(session.get("warmup_count"), "warmup_count", minimum=0)
    _commission_int(session.get("conditioning_count"), "conditioning_count", minimum=0)
    _commission_decimal(session.get("temperature"), "temperature")
    speed = value.get("resident_speed")
    if type(speed) is not dict or set(speed) != _COMMISSION_SPEED_FIELDS:
        raise B300RegisteredQualificationError(
            "sealed qualification resident-speed block is not closed"
        )
    _commission_int(speed.get("max_stage_seconds"), "max_stage_seconds", minimum=1)
    _commission_int(
        speed.get("max_qualification_seconds"),
        "max_qualification_seconds",
        minimum=1,
    )
    _commission_int(speed.get("min_windows"), "min_windows", minimum=0)
    _commission_decimal(speed.get("max_window_scatter"), "max_window_scatter")
    _commission_decimal(
        speed.get("max_conditioning_slowdown"), "max_conditioning_slowdown"
    )
    return value


def declared_qualification_deadline_digest() -> str:
    """The one tracked lease-bounded monotonic deadline policy identity."""

    return canonical_digest(
        "cacheon.eval.b300-declared-deadline-policy.v1",
        {
            "maximum_seconds": QUALIFICATION_DEADLINE_MAXIMUM_SECONDS,
            "source": "lease-bounded-monotonic",
        },
    )


def sealed_qualification_profile_rows(
    catalog: TargetCatalog,
    *,
    registered_target_ids: tuple[str, ...],
    builder_source_digest: str,
) -> tuple[tuple[str, str, str], ...]:
    """(target, spec digest, resolver digest) rows for one reviewed identity."""

    if type(catalog) is not TargetCatalog:
        raise B300RegisteredQualificationError(
            "qualification target catalog is not exact"
        )
    reviewed = _digest(builder_source_digest, "reviewed builder source digest")
    rows = []
    for target in registered_b300_member_contract_projection(
        catalog, registered_target_ids
    ):
        resolver_digest = registered_b300_profile_resolver_digest(
            target,
            builder_source_digest=reviewed,
        )
        rows.append(
            (target.target_id, target.target_spec_digest, resolver_digest)
        )
    return tuple(rows)


def predicted_qualification_registry_digest(
    catalog: TargetCatalog,
    *,
    registered_target_ids: tuple[str, ...],
    builder_source_digest: str,
) -> str:
    rows = sealed_qualification_profile_rows(
        catalog,
        registered_target_ids=registered_target_ids,
        builder_source_digest=builder_source_digest,
    )
    return canonical_digest(
        QUALIFICATION_REGISTRY_SCHEMA,
        {
            "catalog_digest": catalog.digest,
            "profiles": [
                {
                    "resolver_digest": resolver_digest,
                    "target_id": target_id,
                    "target_spec_digest": spec_digest,
                }
                for target_id, spec_digest, resolver_digest in rows
            ],
        },
    )


def predicted_qualification_builder_digest(
    catalog: TargetCatalog,
    *,
    registered_target_ids: tuple[str, ...],
    builder_source_digest: str,
    selection_store_digest: str,
    resident_count_quality_builder_digest: str,
) -> str:
    return canonical_digest(
        QUALIFICATION_CONSTRUCTION_SCHEMA,
        {
            "builder_source_digest": _digest(
                builder_source_digest, "reviewed builder source digest"
            ),
            "evidence_policy_digest": QUALIFICATION_EVIDENCE_POLICY_DIGEST,
            "profile_registry_digest": predicted_qualification_registry_digest(
                catalog,
                registered_target_ids=registered_target_ids,
                builder_source_digest=builder_source_digest,
            ),
            "resident_count_quality_builder_digest": _digest(
                resident_count_quality_builder_digest,
                "resident count quality builder digest",
            ),
            "selection_store_digest": _digest(
                selection_store_digest, "selection store digest"
            ),
            "speed_evidence_policy": QUALIFICATION_SPEED_EVIDENCE_POLICY,
        },
    )


def predicted_qualification_policy_digest(
    catalog: TargetCatalog,
    *,
    registered_target_ids: tuple[str, ...],
    builder_source_digest: str,
    selection_store_digest: str,
    hidden_judge_binding_digest: str,
    selection_policy_digest: str,
    resident_count_quality_builder_digest: str,
) -> str:
    return canonical_digest(
        QUALIFICATION_POLICY_SCHEMA,
        {
            "builder_digest": predicted_qualification_builder_digest(
                catalog,
                registered_target_ids=registered_target_ids,
                builder_source_digest=builder_source_digest,
                selection_store_digest=selection_store_digest,
                resident_count_quality_builder_digest=(
                    resident_count_quality_builder_digest
                ),
            ),
            "deadline_policy_digest": declared_qualification_deadline_digest(),
            "entropy_provider_digest": declared_qualification_entropy_digest(
                selection_policy_digest
            ),
            "hidden_judge_binding_digest": _digest(
                hidden_judge_binding_digest, "hidden judge binding digest"
            ),
            "profile_registry_digest": predicted_qualification_registry_digest(
                catalog,
                registered_target_ids=registered_target_ids,
                builder_source_digest=builder_source_digest,
            ),
            "speed_evidence_policy": QUALIFICATION_SPEED_EVIDENCE_POLICY,
        },
    )


__all__ = [
    "B300QualificationCapabilities",
    "B300QualificationCommissionError",
    "CALIBRATION_PACKAGE_SCHEMA",
    "QUALIFICATION_COMMISSION_SCHEMA",
    "QUALIFICATION_DEADLINE_MAXIMUM_SECONDS",
    "QUALIFICATION_EVIDENCE_POLICY_DIGEST",
    "QUALIFICATION_STAGES",
    "QUALIFICATION_SPEED_EVIDENCE_POLICY",
    "declared_qualification_deadline_digest",
    "declared_qualification_entropy_digest",
    "predicted_qualification_builder_digest",
    "predicted_qualification_policy_digest",
    "predicted_qualification_registry_digest",
    "parse_sealed_calibration_package",
    "sealed_qualification_commission",
    "sealed_qualification_profile_rows",
]

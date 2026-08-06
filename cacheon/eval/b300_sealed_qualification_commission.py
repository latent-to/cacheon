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

from cacheon.eval.b300_qualification_deployment import (
    CONSTRUCTION_SCHEMA as QUALIFICATION_CONSTRUCTION_SCHEMA,
    POLICY_SCHEMA as QUALIFICATION_POLICY_SCHEMA,
    REGISTRY_SCHEMA as QUALIFICATION_REGISTRY_SCHEMA,
)
from cacheon.eval.b300_registered_qualification_inputs import (
    ORDINARY_B300_TARGET_IDS,
    RESOLVER_SCHEMA,
    B300RegisteredQualificationError,
    _digest,
)
from cacheon.stack_identity import canonical_digest
from cacheon.target_catalog import TargetCatalog


QUALIFICATION_COMMISSION_SCHEMA = "cacheon-private-b300-qualification-commission-v1"
QUALIFICATION_SPEED_EVIDENCE_POLICY = "resident-v3-singleton"
QUALIFICATION_DEADLINE_MAXIMUM_SECONDS = 14_400

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


def _commission_int(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise B300RegisteredQualificationError(
            f"sealed qualification {field} must be an integer >= {minimum}"
        )
    return value


def _commission_number(value: object, field: str) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise B300RegisteredQualificationError(
            f"sealed qualification {field} must be a finite number"
        )
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        raise B300RegisteredQualificationError(
            f"sealed qualification {field} must be a finite non-negative number"
        )
    return number


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
    _commission_number(session.get("temperature"), "temperature")
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
    _commission_number(speed.get("max_window_scatter"), "max_window_scatter")
    _commission_number(
        speed.get("max_conditioning_slowdown"), "max_conditioning_slowdown"
    )
    return value


def declared_qualification_entropy_digest(selection_policy_digest: str) -> str:
    """The declared entropy identity for one sealed selection policy."""

    return canonical_digest(
        "cacheon.eval.b300-declared-entropy-provider.v1",
        {
            "selection_policy_digest": _digest(
                selection_policy_digest, "selection policy digest"
            )
        },
    )


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
    builder_source_digest: str,
) -> tuple[tuple[str, str, str], ...]:
    """(target, spec digest, resolver digest) rows for one reviewed identity."""

    if type(catalog) is not TargetCatalog:
        raise B300RegisteredQualificationError(
            "qualification target catalog is not exact"
        )
    reviewed = _digest(builder_source_digest, "reviewed builder source digest")
    rows = []
    for target_id in ORDINARY_B300_TARGET_IDS:
        spec = catalog.require(target_id)
        contract = spec.contract_ref
        if spec.members != (target_id,) or contract is None:
            raise B300RegisteredQualificationError(
                f"ordinary target {target_id!r} is not one singleton contract"
            )
        resolver_digest = canonical_digest(
            RESOLVER_SCHEMA,
            {
                "builder_source_digest": reviewed,
                "contract_digest": catalog.contract_digest(target_id),
                "target_id": target_id,
                "target_spec_digest": catalog.target_spec_digest(target_id),
                "verification_profile_id": contract.verification_profile_id,
            },
        )
        rows.append(
            (target_id, catalog.target_spec_digest(target_id), resolver_digest)
        )
    return tuple(rows)


def predicted_qualification_registry_digest(
    catalog: TargetCatalog,
    *,
    builder_source_digest: str,
) -> str:
    rows = sealed_qualification_profile_rows(
        catalog, builder_source_digest=builder_source_digest
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
    builder_source_digest: str,
    selection_store_digest: str,
) -> str:
    return canonical_digest(
        QUALIFICATION_CONSTRUCTION_SCHEMA,
        {
            "builder_source_digest": _digest(
                builder_source_digest, "reviewed builder source digest"
            ),
            "evidence_policy_digest": QUALIFICATION_EVIDENCE_POLICY_DIGEST,
            "profile_registry_digest": predicted_qualification_registry_digest(
                catalog, builder_source_digest=builder_source_digest
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
    builder_source_digest: str,
    selection_store_digest: str,
    hidden_judge_binding_digest: str,
    selection_policy_digest: str,
) -> str:
    return canonical_digest(
        QUALIFICATION_POLICY_SCHEMA,
        {
            "builder_digest": predicted_qualification_builder_digest(
                catalog,
                builder_source_digest=builder_source_digest,
                selection_store_digest=selection_store_digest,
            ),
            "deadline_policy_digest": declared_qualification_deadline_digest(),
            "entropy_provider_digest": declared_qualification_entropy_digest(
                selection_policy_digest
            ),
            "hidden_judge_binding_digest": _digest(
                hidden_judge_binding_digest, "hidden judge binding digest"
            ),
            "profile_registry_digest": predicted_qualification_registry_digest(
                catalog, builder_source_digest=builder_source_digest
            ),
            "speed_evidence_policy": QUALIFICATION_SPEED_EVIDENCE_POLICY,
        },
    )


__all__ = [
    "QUALIFICATION_COMMISSION_SCHEMA",
    "QUALIFICATION_DEADLINE_MAXIMUM_SECONDS",
    "QUALIFICATION_EVIDENCE_POLICY_DIGEST",
    "QUALIFICATION_SPEED_EVIDENCE_POLICY",
    "declared_qualification_deadline_digest",
    "declared_qualification_entropy_digest",
    "predicted_qualification_builder_digest",
    "predicted_qualification_policy_digest",
    "predicted_qualification_registry_digest",
    "sealed_qualification_commission",
    "sealed_qualification_profile_rows",
]

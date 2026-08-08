"""Qualification authority derivation for the commissioned B300 screen service.

The screen deployment owns the sealed files and OCI configuration factory.
This bridge owns only the optional qualification block's validation and the
path-free identities derived from those already-sealed inputs.  It deliberately
does not grant a qualification capability or select a target, model, lane, or
reservation.
"""

from __future__ import annotations

from collections.abc import Callable

from cacheon._strict import require_digest
from cacheon.eval.b300_arena_provider import (
    B300DeclaredQualificationAuthorities,
    B300QualificationLanePair,
    b300_executor_role_policy_digest,
)
from cacheon.eval.b300_registered_qualification_inputs import (
    B300RegisteredQualificationError,
)
from cacheon.eval.b300_sealed_qualification_commission import (
    declared_qualification_deadline_digest,
    declared_qualification_entropy_digest,
    predicted_qualification_builder_digest,
    predicted_qualification_policy_digest,
    sealed_qualification_commission,
)
from cacheon.eval.oci_backend import OCIBackendConfig
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.stack_identity import canonical_digest
from cacheon.target_catalog import TargetCatalog


class B300ScreenQualificationBridgeError(RuntimeError):
    """The optional sealed qualification block is invalid or inconsistent."""


def derive_b300_screen_qualification(
    *,
    authority: dict[str, object],
    authority_refs: dict[str, dict[str, str]],
    prompt_identity: dict[str, str],
    catalog: TargetCatalog,
    lane_pair: B300QualificationLanePair,
    backend_config_factory: Callable[[str], OCIBackendConfig],
) -> tuple[B300DeclaredQualificationAuthorities, dict[str, object] | None]:
    """Derive the path-free qualification declaration and sealed block.

    The backend factory is invoked only after the optional commission block has
    passed validation and digest prediction.  This preserves the deployment's
    fail-closed ordering while keeping path construction in its owning module.
    """

    qualification_builder_digest = require_digest(
        authority.get("qualification_builder_digest"),
        field="qualification builder digest",
        error=B300ScreenQualificationBridgeError,
    )
    hidden_binding = HiddenJudgeBinding(
        prompt_identity["hidden_corpus_commitment"],
        prompt_identity["hidden_judge_digest"],
        prompt_identity["hidden_task_policy_digest"],
    )
    qualification_commission: dict[str, object] | None
    raw_commission = authority.get("qualification")
    if raw_commission is not None:
        try:
            qualification_commission = sealed_qualification_commission(
                raw_commission
            )
            predicted_builder = predicted_qualification_builder_digest(
                catalog,
                builder_source_digest=qualification_commission[
                    "builder_source_digest"
                ],
                selection_store_digest=qualification_commission[
                    "selection_store_digest"
                ],
            )
            qualification_policy_digest = predicted_qualification_policy_digest(
                catalog,
                builder_source_digest=qualification_commission[
                    "builder_source_digest"
                ],
                selection_store_digest=qualification_commission[
                    "selection_store_digest"
                ],
                hidden_judge_binding_digest=hidden_binding.digest,
                selection_policy_digest=prompt_identity["selection_policy_digest"],
            )
        except B300RegisteredQualificationError as exc:
            raise B300ScreenQualificationBridgeError(
                f"sealed qualification commission is invalid: {exc}"
            ) from None
        if qualification_builder_digest != predicted_builder:
            raise B300ScreenQualificationBridgeError(
                "sealed qualification builder digest differs from the tracked"
                " construction identity"
            )
    else:
        qualification_commission = None
        qualification_policy_digest = canonical_digest(
            "cacheon.eval.b300-declared-qualification-policy.v1",
            {
                "builder_digest": qualification_builder_digest,
                "calibration_package_sha256": authority_refs[
                    "calibration_package"
                ]["sha256"],
                "calibration_projection_sha256": authority_refs[
                    "calibration_projection_receipt"
                ]["sha256"],
                "hidden_judge_binding_digest": hidden_binding.digest,
                "prompt_authority_sha256": prompt_identity["sha256"],
            },
        )

    candidate_config = backend_config_factory("b300-qualification-candidate")
    baseline_config = backend_config_factory("b300-qualification-resident")
    declared = B300DeclaredQualificationAuthorities(
        qualification_policy_digest=qualification_policy_digest,
        qualification_builder_digest=qualification_builder_digest,
        candidate_executor_policy_digest=b300_executor_role_policy_digest(
            candidate_config, role="candidate"
        ),
        resident_baseline_executor_policy_digest=b300_executor_role_policy_digest(
            baseline_config, role="resident_baseline"
        ),
        lane_pair=lane_pair,
        entropy_provider_digest=declared_qualification_entropy_digest(
            prompt_identity["selection_policy_digest"]
        ),
        hidden_judge_binding_digest=hidden_binding.digest,
        deadline_policy_digest=declared_qualification_deadline_digest(),
    )
    return declared, qualification_commission


__all__ = [
    "B300ScreenQualificationBridgeError",
    "derive_b300_screen_qualification",
]

"""One sealed in-process qualification plan without repeated construction.

Deployment must construct a private plan once to seal its public authority.
This helper retains that exact immutable plan for the lifetime of the resulting
factory, so later validation and intake calls cannot repeat graph generation or
another expensive construction callback.  Recommissioning after restart still
constructs and validates a fresh in-process factory from durable authority.
"""

from __future__ import annotations

import hmac

from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
)
from cacheon.eval.qualification_runner import CausalQualificationInput


class PrebuiltQualificationPlanError(RuntimeError):
    """A retained plan differs from the public factory authority."""


def sealed_prebuilt_qualification_plan_factory(
    manifest: QualificationAuthorityManifest,
    *,
    selection_secret_reference: str,
    selection_secret: bytes,
    plan: CausalQualificationInput,
) -> QualificationPlanFactory:
    """Return a factory which reuses one already sealed immutable plan."""

    if (
        type(manifest) is not QualificationAuthorityManifest
        or type(plan) is not CausalQualificationInput
        or type(selection_secret_reference) is not str
        or not selection_secret_reference
        or type(selection_secret) is not bytes
        or len(selection_secret) < 32
        or type(plan.selection_secret) is not bytes
        or not hmac.compare_digest(plan.selection_secret, selection_secret)
        or manifest.selection_secret_reference != selection_secret_reference
    ):
        raise PrebuiltQualificationPlanError(
            "prebuilt qualification plan differs from its private authority"
        )
    expected = QualificationAuthorityManifest.seal(
        plan,
        reservations=manifest.reservations,
        selection_secret_reference=selection_secret_reference,
    )
    if expected != manifest:
        raise PrebuiltQualificationPlanError(
            "prebuilt qualification plan differs from its public manifest"
        )

    def load_secret(observed_reference: str) -> bytes:
        if observed_reference != selection_secret_reference:
            raise PrebuiltQualificationPlanError(
                "prebuilt qualification secret reference was substituted"
            )
        return selection_secret

    def load_plan(observed_secret: bytes) -> CausalQualificationInput:
        if type(observed_secret) is not bytes or not hmac.compare_digest(
            observed_secret,
            selection_secret,
        ):
            raise PrebuiltQualificationPlanError(
                "prebuilt qualification secret was substituted"
            )
        return plan

    return QualificationPlanFactory(manifest, load_secret, load_plan)


__all__ = [
    "PrebuiltQualificationPlanError",
    "sealed_prebuilt_qualification_plan_factory",
]

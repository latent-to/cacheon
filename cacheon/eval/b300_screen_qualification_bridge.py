"""Qualification authority derivation for the commissioned B300 screen service.

The screen deployment owns the sealed files and OCI configuration factory.
This bridge owns only the optional qualification block's validation and the
path-free identities derived from those already-sealed inputs.  It deliberately
does not grant a qualification capability or select a target, model, lane, or
reservation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cacheon._strict import require_digest
from cacheon.eval.b300_arena_provider import (
    B300DeclaredQualificationAuthorities,
    B300QualificationLanePair,
    b300_executor_role_policy_digest,
)
from cacheon.eval.b300_mainnet_worker import B300MainnetWorker
from cacheon.eval.b300_registered_qualification_inputs import (
    B300RegisteredQualificationError,
)
from cacheon.eval.b300_sealed_qualification_commission import (
    B300QualificationCommissionError,
    declared_qualification_deadline_digest,
    declared_qualification_entropy_digest,
    predicted_qualification_builder_digest,
    predicted_qualification_policy_digest,
    sealed_qualification_commission,
)
from cacheon.eval.oci_backend import OCIBackendConfig, OCIEngineExecutor
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.stack_identity import canonical_digest
from cacheon.target_catalog import TargetCatalog

if TYPE_CHECKING:
    from cacheon.eval.b300_remote_worker_adapter import (
        B300RemoteQualificationCommission,
    )


QUALIFICATION_EXECUTOR_ID = "b300-qualification-lane"


class B300ScreenQualificationBridgeError(RuntimeError):
    """The optional sealed qualification block is invalid or inconsistent."""


@dataclass
class CommissionedB300QualificationService:
    """One screen owner plus both sealed qualification orientations."""

    worker: B300MainnetWorker
    commission: "B300RemoteQualificationCommission"
    reproduction_commission: "B300RemoteQualificationCommission"
    _executors: tuple[OCIEngineExecutor, ...]
    _screen_composition: object
    _reproduction_worker: B300MainnetWorker | None = None
    _lock: object = field(default_factory=threading.RLock)
    _closed: bool = False

    def __post_init__(self) -> None:
        commissions = (self.commission, self.reproduction_commission)
        if (
            not callable(getattr(self._screen_composition, "close", None))
            or type(self._executors) is not tuple
            or len(self._executors) != 2
            or any(type(row) is not OCIEngineExecutor for row in self._executors)
            or len({id(row.manager) for row in self._executors}) != 2
            or tuple(row.deployment.screen_lane for row in commissions)
            != ("primary", "reproduction")
            or commissions[0].deployment.manifest != commissions[1].deployment.manifest
            or commissions[0].readiness != commissions[1].readiness
            or type(self.worker) is not B300MainnetWorker
            or self.worker.service.manifest != self.commission.deployment.manifest
            or self.worker.readiness != self.commission.readiness
            or self.worker._remote_qualification_lane != "primary"
        ):
            raise B300QualificationCommissionError(
                "commissioned service does not own both qualification orientations"
            )

    def adapter_for(self, publications, continuation_store, screen_lane: str):
        with self._lock:
            if self._closed:
                raise B300QualificationCommissionError(
                    "commissioned qualification service is closed"
                )
            if screen_lane == "primary":
                commission, worker = self.commission, self.worker
            elif screen_lane == "reproduction":
                commission = self.reproduction_commission
                worker = self._reproduction_worker
                if worker is None:
                    worker = B300MainnetWorker(
                        commission.deployment.manifest,
                        commission.deployment.authorities,
                        commission.readiness,
                    )
                    self._reproduction_worker = worker
            else:
                raise B300QualificationCommissionError(
                    "qualification stage must be primary or reproduction"
                )
            return commission.adapter_for(
                publications,
                continuation_store,
                worker=worker,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        closers = (
            self.worker.close,
            *(
                ()
                if self._reproduction_worker is None
                else (self._reproduction_worker.close,)
            ),
            self._screen_composition.close,
            *(executor.manager.close for executor in self._executors),
        )
        for closer in closers:
            try:
                closer()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure


def derive_b300_screen_qualification(
    *,
    authority: dict[str, object],
    prompt_identity: dict[str, str],
    catalog: TargetCatalog,
    registered_target_ids: tuple[str, ...],
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
                registered_target_ids=registered_target_ids,
                builder_source_digest=qualification_commission[
                    "builder_source_digest"
                ],
                selection_store_digest=qualification_commission[
                    "selection_store_digest"
                ],
                resident_count_quality_builder_digest=qualification_commission[
                    "resident_count_quality_builder_digest"
                ],
            )
            qualification_policy_digest = predicted_qualification_policy_digest(
                catalog,
                registered_target_ids=registered_target_ids,
                builder_source_digest=qualification_commission[
                    "builder_source_digest"
                ],
                selection_store_digest=qualification_commission[
                    "selection_store_digest"
                ],
                hidden_judge_binding_digest=hidden_binding.digest,
                selection_policy_digest=prompt_identity["selection_policy_digest"],
                resident_count_quality_builder_digest=qualification_commission[
                    "resident_count_quality_builder_digest"
                ],
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
            "cacheon.eval.b300-declared-qualification-policy.v2",
            {
                "builder_digest": qualification_builder_digest,
                "hidden_judge_binding_digest": hidden_binding.digest,
                "prompt_authority_sha256": prompt_identity["sha256"],
            },
        )

    lane_config = backend_config_factory(QUALIFICATION_EXECUTOR_ID)
    declared = B300DeclaredQualificationAuthorities(
        qualification_policy_digest=qualification_policy_digest,
        qualification_builder_digest=qualification_builder_digest,
        candidate_executor_policy_digest=b300_executor_role_policy_digest(
            lane_config, role="candidate"
        ),
        resident_baseline_executor_policy_digest=b300_executor_role_policy_digest(
            lane_config, role="resident_baseline"
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
    "CommissionedB300QualificationService",
    "QUALIFICATION_EXECUTOR_ID",
    "derive_b300_screen_qualification",
]

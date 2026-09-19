"""Golden builder for the Refactor D request-plan family.

Consumed by ``tests/test_refactor_d_goldens.py`` (the pinned document, the
regeneration helper and the byte/digest assertions live there). The builder
maps one JSON-able ``inputs`` dict to ``(canonical_bytes, digest, extras)``
using only the production producers and readers and synthetic identities. The
family-specific negatives that a refactor must keep live next to their builder.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaScreenReceipt,
    PromotionDecision,
    ScreenGrade,
    ScreenStageResult,
)
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.evaluation_recovery_plan import (
    EvaluationRecoveryPlanError,
    decode_recovery_request_plan,
    encode_recovery_request_plan,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    seal_remote_request,
)
from cacheon.chain.remote_worker_artifact_recovery import PlannedQualificationArtifact
from cacheon.chain.remote_worker_request_plan import (
    DOMAIN_QUALIFICATION_REQUEST_PLAN,
    SCHEMA_QUALIFICATION_REQUEST_PLAN,
    QualificationRequestPlan,
    RemoteWorkerError,
    _lease_dict,
)
from cacheon.chain.remote_worker_spool import (
    DOMAIN_REQUEST,
    SCHEMA_REQUEST,
    spool_canonical_json,
    spool_digest,
)
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.stack_identity import canonical_digest


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# request_plan: QualificationRequestPlan.to_dict() -> spool_canonical_json bytes,
# plan_digest / request_id, and the SQLite recovery BLOB codec round trip
# (cacheon/chain/remote_worker_request_plan.py, evaluation_recovery_plan.py).
# --------------------------------------------------------------------------- #

REQUEST_PLAN_INPUTS: dict[str, dict[str, Any]] = {
    "primary": {"screen_lane": "primary", "candidates": 1, "expires_block": 1},
    "reproduction": {"screen_lane": "reproduction", "candidates": 1, "expires_block": 1},
    "renewed_lease": {"screen_lane": "primary", "candidates": 1, "expires_block": 9},
    "two_member": {"screen_lane": "primary", "candidates": 2, "expires_block": 1},
}

_CREATED_AT_UNIX = 1_700_000_000
_WORKER_EPOCH = "1" * 32


def _candidate(index: int, service_digest: str) -> tuple[QualificationReservation, dict[str, Any]]:
    label = f"candidate-{index}"
    publication = {
        "address_digest": _h(f"{label}:address"),
        "content_hash": _h(f"{label}:content"),
        "directories": [],
        "files": [{"path": "bundle.bin", "sha256": _h(f"{label}:file"), "size": 1}],
        "publication_digest": _h(f"{label}:publication"),
        "schema": "cacheon.worker-bundle-publication.v1",
    }
    wire_digest = canonical_digest("cacheon.chain.worker-bundle-publication", publication)
    reservation = QualificationReservation(
        reservation_digest=_h(f"{label}:reservation"),
        submission_digest=wire_digest,
        target_id="target-a",
        selected_delta_digest=_h(f"{label}:delta"),
        arrival_order=index,
        hotkey=f"golden-hotkey-{index}",
        finalized_block=1 + index,
        finalized_event_index=index,
        finalized_event_subindex=0,
        target_members=("component",),
    )
    binding = reservation.to_dict()
    binding.pop("arrival_order")
    candidate_digest = canonical_digest(
        "cacheon.arena.candidate-binding",
        {"publication_digest": wire_digest, "reservation": binding, "screen_attempt": 1},
    )
    receipt = ArenaScreenReceipt(
        service_digest=service_digest,
        candidate_digest=candidate_digest,
        screen_attempt=1,
        results=tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(f"{label}:evidence-{stage}"), 1)
            for stage in SCREEN_STAGES
        ),
        decision=PromotionDecision.PROMOTE,
    )
    return reservation, {
        "candidate_digest": candidate_digest,
        "publication": publication,
        "reservation": reservation.to_dict(),
        "screen_receipt": receipt.to_dict(),
    }


def _plan_and_lease(inputs: dict[str, Any]) -> tuple[QualificationRequestPlan, EvaluationLease]:
    service_digest = _h("service")
    policy_digest = _h("qualification-policy")
    rows = [_candidate(index, service_digest) for index in range(int(inputs["candidates"]))]
    body = {
        "candidates": [row for _, row in rows],
        "kind": "qualification_work",
        "qualification_policy_digest": policy_digest,
        "schema_version": 2,
        "screen_lane": inputs["screen_lane"],
        "service_digest": service_digest,
    }
    lease = EvaluationLease(
        lease_id=_h("lease"),
        generation=1,
        stage="qualification",
        owner="validator",
        members=tuple(
            EvaluationLeaseMember(reservation.reservation_digest, "promoted")
            for reservation, _ in rows
        ),
        claimed_block=0,
        initial_expires_block=1,
        expires_block=int(inputs["expires_block"]),
    )
    readiness = WorkerReadiness(
        ready_receipt_digest=_h("ready-receipt"),
        ready_epoch=0,
        service_digest=service_digest,
        arena_id="golden-arena",
        provider_digest=_h("provider"),
        runtime_digest=_h("runtime"),
        worker_distribution_digest=_h("worker-distribution"),
        model_revision_digest=_h("model-revision"),
        model_manifest_digest=_h("model-manifest"),
        model_content_digest=_h("model-content"),
        target_architecture="sm120",
        topology_class="tp4-golden",
        topology_digest=_h("topology"),
        gpu_count=4,
        tensor_parallel_size=4,
        workload_digest=_h("workload"),
        qualification_policy_digest=policy_digest,
    )
    credential = RemoteWorkerCredential("golden-credential", b"x" * 32)
    transport = RemoteWorkerTransportIdentity(
        transport_id="golden-transport",
        endpoint_identity_digest=_h("endpoint"),
        protocol_digest=REMOTE_EVALUATION_PROTOCOL_DIGEST,
        credential_digest=credential.digest,
        service_digest=service_digest,
        worker_readiness_digest=readiness.digest,
    )
    request = seal_remote_request(
        lease,
        readiness,
        f"{readiness.arena_id}@{service_digest}",
        transport,
        credential,
        body,
    )
    artifacts = [
        PlannedQualificationArtifact("qualification_payload", _h("payload"), 10).to_dict(),
        *(
            PlannedQualificationArtifact(
                "candidate_publication", _h(f"candidate-{index}:artifact"), 20 + index
            ).to_dict()
            for index in range(len(rows))
        ),
    ]
    common = {
        "artifacts": artifacts,
        "created_at_unix": _CREATED_AT_UNIX,
        "deadline_unix": _CREATED_AT_UNIX + 3600,
        "lease": _lease_dict(lease),
        "queued_at_unix_ns": _CREATED_AT_UNIX * 1_000_000_000,
        "worker_epoch": _WORKER_EPOCH,
    }
    outer = {
        **common,
        "ready_receipt_digest": request.ready_receipt_digest,
        "schema": SCHEMA_REQUEST,
        "service_identity": request.service_identity,
        "worker_readiness_digest": request.worker_readiness_digest,
    }
    unsigned = {
        **common,
        "credential_digest": credential.digest,
        "registration_digest": _h("registration"),
        "remote_request": request.to_dict(),
        "request_id": spool_digest(DOMAIN_REQUEST, outer),
        "schema": SCHEMA_QUALIFICATION_REQUEST_PLAN,
        "transport_identity_digest": transport.digest,
    }
    plan_digest = spool_digest(DOMAIN_QUALIFICATION_REQUEST_PLAN, unsigned)
    return QualificationRequestPlan.from_dict({**unsigned, "plan_digest": plan_digest}), lease


def build_request_plan(inputs: dict[str, Any]) -> tuple[bytes, str, dict[str, Any]]:
    plan, lease = _plan_and_lease(inputs)
    raw = spool_canonical_json(plan.to_dict())
    blob, plan_digest, request_id = encode_recovery_request_plan(plan, expected_lease=lease)
    reopened = decode_recovery_request_plan(
        blob,
        expected_lease=lease,
        expected_plan_digest=plan_digest,
        expected_request_id=request_id,
    )
    assert reopened == plan and blob == raw
    return raw, plan.plan_digest, {
        "request_id": plan.request_id,
        "remote_request_digest": plan.remote_request.digest,
        "lease": _lease_dict(plan.lease),
        "sqlite_blob_sha256": hashlib.sha256(blob).hexdigest(),
    }


def test_request_plan_refuses_reordered_artifact_roles() -> None:
    plan, _ = _plan_and_lease(REQUEST_PLAN_INPUTS["primary"])
    value = plan.to_dict()
    value["artifacts"] = list(reversed(value["artifacts"]))
    with pytest.raises(RemoteWorkerError, match="artifact roles are incomplete or reordered"):
        QualificationRequestPlan.from_dict(value)


def test_request_plan_blob_is_bound_to_its_lease_identity() -> None:
    plan, lease = _plan_and_lease(REQUEST_PLAN_INPUTS["primary"])
    blob, plan_digest, request_id = encode_recovery_request_plan(plan, expected_lease=lease)
    other = EvaluationLease(
        lease_id=lease.lease_id,
        generation=lease.generation + 1,
        stage=lease.stage,
        owner=lease.owner,
        members=lease.members,
        claimed_block=lease.claimed_block,
        initial_expires_block=lease.initial_expires_block,
        expires_block=lease.expires_block,
    )
    with pytest.raises(EvaluationRecoveryPlanError, match="retained request plan identity changed"):
        decode_recovery_request_plan(
            blob,
            expected_lease=other,
            expected_plan_digest=plan_digest,
            expected_request_id=request_id,
        )


FAMILIES: dict[str, tuple[dict[str, dict[str, Any]], Any]] = {
    "request_plan": (REQUEST_PLAN_INPUTS, build_request_plan),
}

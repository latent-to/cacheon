"""Golden builders for the Refactor D request-plan and graph-store families.

Consumed by ``tests/test_refactor_d_goldens.py`` (the pinned document, the
regeneration helper and the byte/digest assertions live there). Each builder
maps one JSON-able ``inputs`` dict to ``(canonical_bytes, digest, extras)``
using only the production producers and readers, synthetic identities, and a
scratch directory. The family-specific negatives that a refactor must keep
live next to their builder.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import cacheon.eval.b300_qualification_graph_evidence_store as graph_store_module
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
from cacheon.eval.b300_qualification_capabilities import (
    StructuredGraphShapeRecord,
    StructuredGraphVariantRecord,
)
from cacheon.eval.b300_qualification_graph_evidence_store import (
    B300QualificationGraphEvidenceStore,
    B300QualificationGraphGenerationOutput,
)
from cacheon.eval.b300_qualification_graph_provider import (
    B300QualificationGraphArtifact,
    B300QualificationGraphBinding,
)
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
    B300QualificationGraphEvidenceStoreError,
)
from cacheon.eval.continuation_codec import ContinuationCodec, ContinuationCodecError
from cacheon.eval.count_quality import (
    CountQualityEvidence,
    CountQualityPolicy,
    score_count_quality,
)
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.eval.registered_resident_count_quality import (
    RegisteredResidentCountQualityResult,
)
from cacheon.eval.resident_count_quality import (
    ResidentCountQualityError,
    ResidentCountQualityResult,
)
from cacheon.stack_identity import canonical_digest, canonical_json_bytes


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@contextlib.contextmanager
def _scratch() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="cacheon-refactor-d-")).resolve(strict=True)
    try:
        yield root
    finally:
        for parent, dirs, files in os.walk(root):
            for name in dirs + files:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(parent, name), stat.S_IRWXU)
        shutil.rmtree(root, ignore_errors=True)


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


# --------------------------------------------------------------------------- #
# graph_store: B300QualificationGraphEvidenceStore durable attempt/index records
# (cacheon/eval/b300_qualification_graph_evidence_store.py). The attempt nonce
# is a CSPRNG draw in production; it is pinned here through the module's own
# `secrets` name so the record layout is byte-comparable.
# --------------------------------------------------------------------------- #

GRAPH_STORE_INPUTS: dict[str, dict[str, Any]] = {
    "armed": {"stop_after": "arm"},
    "output_no_terminal": {"stop_after": "output"},
    "terminal_no_index": {"stop_after": "terminal"},
    "complete": {"stop_after": ""},
}

_GENERATION_ONE = f"{1:016d}"


class _FixedSecrets:
    @staticmethod
    def token_hex(nbytes: int) -> str:
        return "ab" * nbytes


class _StopAfterDurable(Exception):
    pass


def _graph_binding() -> B300QualificationGraphBinding:
    label = "golden-target"
    return B300QualificationGraphBinding(
        reservation_digest=_h(f"{label}:reservation"),
        reservation_identity_digest=_h(f"{label}:reservation-identity"),
        candidate_binding_digest=_h(f"{label}:candidate"),
        screen_attempt=1,
        target_id=label,
        target_members=("member-a",),
        target_spec_digest=_h(f"{label}:spec"),
        selected_delta_digest=_h(f"{label}:delta"),
        publication_content_hash=_h(f"{label}:content"),
        publication_address_digest=_h(f"{label}:address"),
        publication_digest=_h(f"{label}:publication"),
        publication_receipt_digest=_h(f"{label}:receipt"),
        prepared_arm_digest=_h(f"{label}:arm"),
        prepared_contribution_digest=_h(f"{label}:contribution"),
        prepared_launch_digest=_h(f"{label}:launch"),
        materialized_stack_digest=_h(f"{label}:stack"),
        materialized_tree_digest=_h(f"{label}:tree"),
        trusted_tree_identity_digest=_h(f"{label}:trusted-tree"),
        native_build_spec_digest=_h(f"{label}:native-build"),
    )


def _graph_artifact(
    binding: B300QualificationGraphBinding, policy: str
) -> B300QualificationGraphArtifact:
    variants = tuple(
        StructuredGraphVariantRecord(
            member,
            "commissioned",
            True,
            True,
            (
                StructuredGraphShapeRecord(
                    _h(f"{binding.digest}:{member}:shape"), True, True, True, 3, True, True, False
                ),
            ),
        )
        for member in binding.target_members
    )
    return B300QualificationGraphArtifact(binding, policy, 2, variants)


def _deadline() -> float:
    return time.monotonic() + 30.0


def build_graph_store(inputs: dict[str, Any]) -> tuple[bytes, str, dict[str, Any]]:
    stop_after = str(inputs["stop_after"])
    policy = _h("golden-policy")
    binding = _graph_binding()

    def produce(exact, token, _deadline):
        return B300QualificationGraphGenerationOutput(token, _graph_artifact(exact, policy))

    def boundary(kind: str, phase: str) -> None:
        if kind == stop_after and phase == "parents_fsynced":
            raise _StopAfterDurable(kind)

    saved = (graph_store_module.secrets, graph_store_module._publication_boundary)
    graph_store_module.secrets = _FixedSecrets  # type: ignore[assignment]
    graph_store_module._publication_boundary = boundary  # type: ignore[assignment]
    try:
        with _scratch() as root:
            store = B300QualificationGraphEvidenceStore(root / "evidence", policy)
            with contextlib.suppress(_StopAfterDurable):
                store.probe_once(binding, produce, deadline=_deadline())
            attempt_dir = root / "evidence" / "attempts" / policy / binding.digest
            index_path = root / "evidence" / "indexes" / policy / f"{binding.digest}.json"
            records = {
                path.name: path.read_bytes().decode("utf-8")
                for path in sorted(attempt_dir.glob(f"{_GENERATION_ONE}.*.json"))
            }
            if index_path.exists():
                records["index"] = index_path.read_bytes().decode("utf-8")
            try:
                store.reopen(binding, deadline=_deadline())
                reopen_outcome = "reference"
            except B300QualificationGraphEvidenceHold:
                reopen_outcome = "hold"
            graph_store_module._publication_boundary = saved[1]
            healed = store.probe_once(binding, produce, deadline=_deadline())
            healed_files = sorted(
                path.name for path in attempt_dir.glob(f"{_GENERATION_ONE}.*.json")
            ) + (["index"] if index_path.exists() else [])
    finally:
        graph_store_module.secrets, graph_store_module._publication_boundary = saved

    last = records["index"] if "index" in records else records[sorted(records)[-1]]
    artifact_bytes = _graph_artifact(binding, policy).canonical_bytes
    assert hashlib.sha256(artifact_bytes).hexdigest() == healed.sha256
    return last.encode("utf-8"), binding.digest, {
        "records_after_stop": records,
        "reopen_after_stop": reopen_outcome,
        "files_after_heal": healed_files,
        "healed_reference": healed.to_dict(),
        "artifact_canonical": artifact_bytes.decode("utf-8"),
    }


def test_graph_store_refuses_an_index_without_a_terminal_record() -> None:
    policy = _h("golden-policy")
    binding = _graph_binding()
    with _scratch() as root:
        store = B300QualificationGraphEvidenceStore(root / "evidence", policy)
        store.arm(binding, deadline=_deadline())
        index_path = root / "evidence" / "indexes" / policy / f"{binding.digest}.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(b"{}")
        with pytest.raises(
            B300QualificationGraphEvidenceStoreError, match="no authenticated terminal attempt"
        ):
            store.reopen(binding, deadline=_deadline())


# --------------------------------------------------------------------------- #
# count_closure: RegisteredResidentCountQualityResult (the count result nested
# in resident_count stage exits and resident-pair closures) through the same
# ContinuationCodec root qualification_runner registers for it
# (cacheon/eval/registered_resident_count_quality.py, continuation_codec.py).
# --------------------------------------------------------------------------- #

COUNT_CLOSURE_INPUTS: dict[str, dict[str, Any]] = {
    "pass": {"stock_correct": 8, "candidate_correct": 8, "total": 10, "threshold_drop": 1},
    "fail": {"stock_correct": 8, "candidate_correct": 6, "total": 10, "threshold_drop": 1},
}


def _count_result(inputs: dict[str, Any]) -> ResidentCountQualityResult:
    policy = CountQualityPolicy(regression_threshold_drop=int(inputs["threshold_drop"]))
    evidence = CountQualityEvidence(
        stock_observation_digest=_h("stock-observation"),
        candidate_observation_digest=_h("candidate-observation"),
        stock_correct=int(inputs["stock_correct"]),
        candidate_correct=int(inputs["candidate_correct"]),
        total=int(inputs["total"]),
    )
    return ResidentCountQualityResult(
        stock_observation_digest=evidence.stock_observation_digest,
        candidate_observation_digest=evidence.candidate_observation_digest,
        evidence=evidence,
        policy=policy,
        verdict=score_count_quality(evidence, policy),
    )


def _registered(inputs: dict[str, Any]) -> RegisteredResidentCountQualityResult:
    count = _count_result(inputs)
    return RegisteredResidentCountQualityResult(
        target_id="golden-target",
        catalog_digest=_h("catalog"),
        target_spec_digest=_h("target-spec"),
        profile_digest=_h("profile"),
        execution_envelope_digest=_h("envelope"),
        execution_plan_digest=_h("plan"),
        pair_binding_digest=_h("pair-binding"),
        candidate_bundle_digest=_h("candidate-bundle"),
        raw_execution_evidence_digest=_h("raw-execution"),
        fixed_stock_authority_digest=_h("stock-authority"),
        stock_observation_digest=count.stock_observation_digest,
        candidate_observation_digest=count.candidate_observation_digest,
        policy_digest=count.policy.digest,
        count_quality_result_digest=count.digest,
        count_quality_result=count,
    )


def build_count_closure(inputs: dict[str, Any]) -> tuple[bytes, str, dict[str, Any]]:
    registered = _registered(inputs)
    codec = ContinuationCodec((RegisteredResidentCountQualityResult,))
    encoded = codec.encode(registered)
    raw = canonical_json_bytes(encoded)
    decoded = codec.decode(json.loads(raw.decode("utf-8")))
    assert decoded == registered and decoded.digest == registered.digest
    return raw, registered.digest, {
        "codec_type": encoded["type"],
        "decision": registered.count_quality_result.verdict.decision,
        "count_result_digest": registered.count_quality_result.digest,
    }


def test_count_result_refuses_a_stale_verdict() -> None:
    passing = _count_result(COUNT_CLOSURE_INPUTS["pass"])
    failing_evidence = _count_result(COUNT_CLOSURE_INPUTS["fail"]).evidence
    with pytest.raises(ResidentCountQualityError):
        ResidentCountQualityResult(
            stock_observation_digest=failing_evidence.stock_observation_digest,
            candidate_observation_digest=failing_evidence.candidate_observation_digest,
            evidence=failing_evidence,
            policy=passing.policy,
            verdict=passing.verdict,
        )


def test_count_closure_codec_refuses_a_foreign_type_key() -> None:
    codec = ContinuationCodec((RegisteredResidentCountQualityResult,))
    encoded = codec.encode(_registered(COUNT_CLOSURE_INPUTS["pass"]))
    with pytest.raises(ContinuationCodecError, match="not a registered codec root"):
        codec.decode({"type": "not.a.registered.Root", "value": encoded["value"]})


FAMILIES: dict[str, tuple[dict[str, dict[str, Any]], Any]] = {
    "request_plan": (REQUEST_PLAN_INPUTS, build_request_plan),
    "graph_store": (GRAPH_STORE_INPUTS, build_graph_store),
    "count_closure": (COUNT_CLOSURE_INPUTS, build_count_closure),
}

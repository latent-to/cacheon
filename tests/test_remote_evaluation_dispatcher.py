from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.chain.evaluation_coordinator as coordinator_module
import cacheon.chain.remote_qualification_evidence as remote_evidence_module
from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceManifest,
    Workload,
    WorkloadCell,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.evaluation_coordinator import (
    ClaimedQualificationEvaluation,
    EvaluationCoordinator,
    WorkerReadiness,
)
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.publication import publish_worker_bundle, reopen_worker_bundle
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationDispatcher,
    RecoverableQualificationDispatcherError,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    capture_remote_qualification_product,
    import_remote_qualification_evidence,
    qualification_batch_from_dict,
    qualification_batch_to_dict,
    remote_qualification_product_from_dict,
    remote_qualification_product_to_dict,
)
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.evidence_store import publish_evidence, reopen_evidence
from cacheon.eval.candidate_failure_product import (
    candidate_failure_digest,
    publish_candidate_failure,
)
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationReservation,
    QualificationRetryPlan,
)
from cacheon.stack_identity import canonical_digest, sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest
from tests.support.evaluation import (
    BLOCK, POLICY, SCOPE, Cursor as _Cursor, Provider as _Provider,
    block_hash as _block_hash, claim_qualification as _claim_qualification,
    coordinator as _coordinator, db_path as _db_path, h as _h, incumbent as _incumbent,
    manifest as _manifest, published_rows as _published_rows, store as _store,
)


def _transport_identity(
    coordinator: EvaluationCoordinator,
    credential: RemoteWorkerCredential,
    *,
    endpoint: str = "worker-endpoint-a",
) -> RemoteWorkerTransportIdentity:
    return RemoteWorkerTransportIdentity(
        "test-spool-v1",
        _h(endpoint),
        REMOTE_EVALUATION_PROTOCOL_DIGEST,
        credential.digest,
        coordinator.service.identity,
        coordinator.readiness.digest,
        1 << 20,
    )


def _authority_for_request(request: RemoteEvaluationRequest) -> QualificationAuthorityManifest:
    reservations = tuple(
        QualificationReservation.from_dict(row["reservation"])
        for row in request.body["candidates"]
    )
    return QualificationAuthorityManifest(
        "registered",
        _h("remote-qualification-authority"),
        _h("remote-qualification-source"),
        _h("remote-qualification-commitment"),
        _h("remote-qualification-secret-reference"),
        tuple(row.selected_delta_digest for row in reservations),
        reservations,
    )


def _failed_batch(
    authority: QualificationAuthorityManifest,
    attempt_ref,
) -> QualificationIntakeBatch:
    return QualificationIntakeBatch(
        authority.digest,
        tuple(
            QualificationIntakeOutcome(
                row.reservation_digest,
                row.selected_delta_digest,
                authority.digest,
                QualificationDecision.FAIL,
                "speed_regression",
                False,
                attempt_artifact_sha256=attempt_ref.sha256,
                report_digest=_h(f"report:{row.reservation_digest}"),
            )
            for row in authority.reservations
        ),
        attempt_ref,
    )


def test_qualification_batch_wire_roundtrip_is_exact_and_closed() -> None:
    authority = _h("authority")
    reservation = _h("reservation")
    failure = _h("failure")
    batch = QualificationIntakeBatch(
        authority,
        (
            QualificationIntakeOutcome(
                reservation,
                _h("selected"),
                authority,
                QualificationDecision.NO_DECISION,
                "oci_backend",
                True,
                failure_digest=failure,
            ),
        ),
        None,
        QualificationRetryPlan(
            authority, "requeue", ((reservation,),), failure
        ),
    )

    wire = qualification_batch_to_dict(batch)
    assert qualification_batch_from_dict(wire) == batch
    with pytest.raises(RemoteEvaluationDispatcherError, match="fields are not closed"):
        qualification_batch_from_dict({**wire, "command": "ignored"})


def test_dispatcher_rejects_drifted_transport_identity_before_claim(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _Cursor((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path, service, cursor, store_factory=RecoverableFinalizedIntakeStore
    )
    credential = RemoteWorkerCredential("qualification-key-v1", b"q" * 32)
    # Every transport method fails the test if the dispatcher reaches it.
    transport = SimpleNamespace(
        identity=dataclasses.replace(
            _transport_identity(coordinator, credential),
            worker_readiness_digest=_h("another-ready-epoch"),
        ),
        **dict.fromkeys(RecoverableQualificationDispatcher._TRANSPORT_METHODS, pytest.fail),
    )

    with pytest.raises(RecoverableQualificationDispatcherError, match="differs from CPU authority"):
        RecoverableQualificationDispatcher(
            coordinator=coordinator,
            transport=transport,
            credential=credential,
            qualification_evidence_root=tmp_path / "cpu-evidence",
            qualification_incumbent_stack=_incumbent(service),
            qualification_incumbent_tree_digest=_h("incumbent-tree"),
        )

    with _store(tmp_path) as store:
        assert store.active_evaluation_leases() == ()
        assert store.get(row.reservation_id).status == "published"


def test_remote_qualification_product_closes_inventory_bytes_and_bounds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = ArenaService(_manifest(), _Provider())
    readiness = WorkerReadiness.for_service(
        service,
        ready_receipt_digest=_h("product-ready"),
        ready_epoch=9,
    )
    reservation = QualificationReservation(
        _h("product-reservation"),
        _h("product-submission"),
        "target.0",
        _h("product-selected"),
        0,
        "product-miner",
        BLOCK,
        0,
        0,
        ("slot.0",),
    )
    authority = QualificationAuthorityManifest(
        "registered",
        _h("product-authority"),
        _h("product-source"),
        _h("product-commitment"),
        _h("product-secret"),
        (reservation.selected_delta_digest,),
        (reservation,),
    )
    pod_root = tmp_path / "product-pod-evidence"
    reference = publish_evidence(
        pod_root,
        b"exact evidence",
        domain="qualification-attempt",
        media_type="application/json",
        schema="cacheon.qualification.product-test.v1",
    )
    product = capture_remote_qualification_product(
        batch=_failed_batch(authority, reference),
        authority_manifest=authority,
        incumbent_stack=_incumbent(service),
        incumbent_tree_digest=_h("product-tree"),
        screen_lane="primary",
        service_digest=service.identity,
        readiness=readiness,
        evidence_root=pod_root,
        evidence_references=(reference,),
    )
    wire = remote_qualification_product_to_dict(product)
    assert remote_qualification_product_from_dict(wire) == product
    cpu_root = tmp_path / "product-cpu-evidence"
    assert import_remote_qualification_evidence(product, cpu_root) == (reference,)
    assert reopen_evidence(cpu_root, reference) == b"exact evidence"

    missing = {**wire, "evidence": []}
    with pytest.raises(RemoteEvaluationDispatcherError, match="authority is malformed"):
        remote_qualification_product_from_dict(missing)

    tampered = json.loads(json.dumps(wire))
    tampered["evidence"][0]["payload_base64"] = "dGFtcGVyZWQ="
    with pytest.raises(RemoteEvaluationDispatcherError, match="differs from its bounded"):
        remote_qualification_product_from_dict(tampered)

    duplicated = json.loads(json.dumps(wire))
    duplicated["evidence_inventory"].append(duplicated["evidence_inventory"][0])
    duplicated["evidence"].append(duplicated["evidence"][0])
    with pytest.raises(RemoteEvaluationDispatcherError, match="duplicate"):
        remote_qualification_product_from_dict(duplicated)

    with pytest.raises(RemoteEvaluationDispatcherError, match="duplicated"):
        capture_remote_qualification_product(
            batch=product.batch,
            authority_manifest=authority,
            incumbent_stack=product.incumbent_stack,
            incumbent_tree_digest=product.incumbent_tree_digest,
            screen_lane="primary",
            service_digest=service.identity,
            readiness=readiness,
            evidence_root=pod_root,
            evidence_references=(reference, reference),
        )

    monkeypatch.setattr(
        remote_evidence_module,
        "_MAX_REMOTE_EVIDENCE_ARTIFACT_BYTES",
        4,
    )
    with pytest.raises(RemoteEvaluationDispatcherError, match="cannot be captured"):
        capture_remote_qualification_product(
            batch=product.batch,
            authority_manifest=authority,
            incumbent_stack=product.incumbent_stack,
            incumbent_tree_digest=product.incumbent_tree_digest,
            screen_lane="primary",
            service_digest=service.identity,
            readiness=readiness,
            evidence_root=pod_root,
            evidence_references=(reference,),
        )


def test_remote_candidate_failure_product_is_reopened_against_outcomes(
    tmp_path: Path,
) -> None:
    service = ArenaService(_manifest(), _Provider())
    readiness = WorkerReadiness.for_service(
        service, ready_receipt_digest=_h("failure-ready"), ready_epoch=10
    )
    reservation = QualificationReservation(
        _h("failure-reservation"), _h("failure-submission"), "target.0",
        _h("failure-selected"), 0, "failure-miner", BLOCK, 0, 0, ("slot.0",),
    )
    authority = QualificationAuthorityManifest(
        "registered", _h("failure-authority"), _h("failure-source"),
        _h("failure-commitment"), _h("failure-secret"),
        (reservation.selected_delta_digest,), (reservation,),
    )
    pod_root = tmp_path / "failure-pod-evidence"
    reference, failure = publish_candidate_failure(
        pod_root,
        authority_manifest_digest=authority.digest,
        source_digest=authority.source_digest,
        culprit_reservation_digest=reservation.reservation_digest,
        selected_delta_digest=reservation.selected_delta_digest,
        target_id=reservation.target_id,
        arm_digest=_h("failure-arm"),
        launch_digest=_h("failure-launch"),
        failure_kind="candidate_exception",
        failure="rank 0 RuntimeError at kernels/fail.py:17: boom",
    )
    outcome = QualificationIntakeOutcome(
        reservation.reservation_digest,
        reservation.selected_delta_digest,
        authority.digest,
        QualificationDecision.FAIL,
        "candidate_exception",
        False,
        attempt_artifact_sha256=reference.sha256,
        report_digest=candidate_failure_digest(failure),
    )
    batch = QualificationIntakeBatch(authority.digest, (outcome,), reference)
    product = capture_remote_qualification_product(
        batch=batch,
        authority_manifest=authority,
        incumbent_stack=_incumbent(service),
        incumbent_tree_digest=_h("failure-tree"),
        screen_lane="primary",
        service_digest=service.identity,
        readiness=readiness,
        evidence_root=pod_root,
        evidence_references=(),
    )
    assert import_remote_qualification_evidence(
        product, tmp_path / "failure-cpu-evidence"
    ) == (reference,)

    wrong = dataclasses.replace(
        product,
        batch=QualificationIntakeBatch(
            authority.digest,
            (dataclasses.replace(outcome, reason="speed_regression"),),
            reference,
        ),
    )
    with pytest.raises(RemoteEvaluationDispatcherError, match="differs from its qualification"):
        import_remote_qualification_evidence(
            wrong, tmp_path / "failure-cpu-invalid"
        )

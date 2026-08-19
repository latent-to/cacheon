"""CPU-only contracts for the closed B300 remote qualification adapter."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import cacheon.eval.b300_remote_qualification_adapter as adapter_module
import tests.test_b300_qualification_deployment as deployment_fixtures
import tests.test_b300_sealed_qualification_commission as authority_fixtures
from cacheon.chain.evaluation_coordinator import (
    EvaluationResultEnvelope,
    EvaluationRun,
    WorkerReadiness,
)
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.publication import WorkerBundlePublication
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteEvaluationDispatcherError,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    seal_remote_request,
    verify_remote_request,
)
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldReason,
    verify_remote_qualification_hold_request,
)
from cacheon.eval.b300_mainnet_worker import B300RemoteQualificationRun
from cacheon.eval.b300_qualification_deployment import (
    B300QualificationConstructionAuthority,
    B300QualificationDeployment,
    compose_b300_qualification_deployment,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence
from cacheon.eval.oci_backend import OCIBackendConfig, OCIEngineExecutor
from cacheon.eval.oci_outer_session import OuterSessionWorkerError
from cacheon.eval.oci_prebuild import OCIPrebuildConfig
from cacheon.eval.device_state import DeviceStatePolicy
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _executor(
    tmp_path: Path,
    *,
    role: str,
    lane: str,
) -> OCIEngineExecutor:
    runtime = deployment_fixtures._runtime_policy()
    root = tmp_path / f"executor-{role}-{lane}"
    config = OCIBackendConfig(
        OCIPrebuildConfig(
            docker_binary="/usr/bin/docker",
            recovery_root=root / "recovery",
            publication_root=root / "publications",
            seccomp_profile=root / "seccomp.json",
            executor_id=role,
            policy=deployment_fixtures._prebuild_policy(runtime),
        ),
        runtime,
    )
    offset = 0 if lane == "A" else 4
    return OCIEngineExecutor(
        config,
        DeviceStatePolicy(
            expected_gpus=tuple(
                deployment_fixtures._gpu(index)
                for index in range(offset, offset + 4)
            ),
            required_consecutive_idle_samples=2,
            poll_interval_s=0.05,
            ready_poll_interval_s=0.05,
            drain_timeout_s=2.0,
            maximum_samples=8,
        ),
    )


def _readiness(deployment: B300QualificationDeployment) -> WorkerReadiness:
    manifest = deployment.manifest
    runtime = manifest.runtime
    return WorkerReadiness(
        ready_receipt_digest=_h("ready-receipt"),
        ready_epoch=7,
        service_digest=manifest.digest,
        arena_id=runtime.arena_id,
        provider_digest=manifest.provider_digest,
        runtime_digest=runtime.runtime_digest,
        worker_distribution_digest=runtime.worker_distribution_digest,
        model_revision_digest=runtime.model_revision_digest,
        model_manifest_digest=runtime.model_manifest_digest,
        model_content_digest=runtime.model_content_digest,
        target_architecture=runtime.target_architecture,
        topology_class=runtime.topology_class,
        topology_digest=runtime.topology_digest,
        gpu_count=runtime.gpu_count,
        tensor_parallel_size=runtime.tensor_parallel_size,
        workload_digest=manifest.workload.digest,
        qualification_policy_digest=manifest.qualification_policy_digest,
    )


@dataclass
class _Configured:
    deployment: B300QualificationDeployment
    construction: B300QualificationConstructionAuthority
    readiness: WorkerReadiness
    candidate: object
    receipt: object
    adapter: adapter_module.B300RemoteQualificationAdapter
    executors: tuple[OCIEngineExecutor, OCIEngineExecutor]


def _construction(
    tmp_path: Path,
) -> B300QualificationConstructionAuthority:
    runtime = deployment_fixtures._runtime()
    catalog, incumbent = deployment_fixtures._incumbent(runtime, _h("arena"))
    builder_source = _h("builder-source")
    evidence_root = tmp_path / "evidence"
    count_quality = authority_fixtures._resident_count_quality(
        catalog,
        evidence_root,
    )
    return B300QualificationConstructionAuthority(
        catalog=catalog,
        profiles=deployment_fixtures._profiles(catalog, builder_source),
        incumbent_stack=incumbent,
        incumbent_tree_digest=_h("incumbent-tree"),
        pristine_stack=incumbent,
        pristine_tree_digest=_h("pristine-tree"),
        evidence_root=evidence_root,
        evidence_policy_digest=_h("evidence-policy"),
        builder_source_digest=builder_source,
        selection_store_digest=_h("selection-store"),
        resident_count_quality_builder_digest=_h("resident-count-quality-builder"),
        resident_count_quality=count_quality,
        secret_loader=lambda _reference: b"s" * 32,
        plan_builder=lambda _cohort, _secret: object(),
        entropy_provider_digest=_h("entropy-provider"),
        entropy_provider=lambda *_args: None,
        hidden_judge=deployment_fixtures._Judge(),
        deadline_policy_digest=_h("deadline-policy"),
        deadline_provider=lambda _cohort: 1_000_000.0,
    )


def _bind_construction(
    construction: B300QualificationConstructionAuthority,
    manifest,
) -> B300QualificationConstructionAuthority:
    _catalog, empty_stack = deployment_fixtures._incumbent(
        manifest.runtime,
        manifest.digest,
    )
    return replace(
        construction,
        incumbent_stack=empty_stack,
        pristine_stack=empty_stack,
    )


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    construction = _construction(tmp_path)
    candidate_executor = _executor(tmp_path, role="candidate", lane="A")
    baseline_executor = _executor(tmp_path, role="resident-baseline", lane="B")
    lane_pair = deployment_fixtures._lane_pair(
        candidate_executor,
        baseline_executor,
    )
    screen = deployment_fixtures._screen_authorities(
        construction,
        candidate_executor,
        baseline_executor,
        lane_pair,
    )
    manifest = deployment_fixtures._manifest(screen)
    construction = _bind_construction(
        construction,
        manifest,
    )
    resident_pair_factory, pair_executors = (
        authority_fixtures._resident_pair_factory(
            tmp_path / "pair",
            monkeypatch,
            manifest.digest,
        )
    )
    deployment = compose_b300_qualification_deployment(
        manifest=manifest,
        screen_authorities=screen,
        construction=construction,
        candidate_executor=candidate_executor,
        resident_baseline_executor=baseline_executor,
        resident_pair_factory=resident_pair_factory,
        screen_lane="primary",
    )
    readiness = _readiness(deployment)
    candidate = deployment_fixtures._bundle(tmp_path / "bundle", 0)
    receipt = deployment_fixtures._receipt(manifest.digest, candidate)
    resolver = adapter_module.B300WorkerBundleResolver((candidate.publication,))
    adapter = adapter_module.B300RemoteQualificationAdapter(
        deployment,
        construction,
        readiness,
        resolver,
        QualificationContinuationStore(tmp_path / "continuation"),
    )
    result = _Configured(
        deployment,
        construction,
        readiness,
        candidate,
        receipt,
        adapter,
        (candidate_executor, baseline_executor, *pair_executors),
    )
    yield result
    for executor in result.executors:
        executor.manager.close()


def _body(
    configured: _Configured,
    *,
    candidate=None,
    receipt=None,
    service_digest: str | None = None,
    policy_digest: str | None = None,
    screen_lane: str = "primary",
) -> dict[str, object]:
    candidate = configured.candidate if candidate is None else candidate
    receipt = configured.receipt if receipt is None else receipt
    manifest = configured.deployment.manifest
    return {
        "candidates": [
            {
                "candidate_digest": candidate.digest,
                "publication": candidate.publication.to_dict(),
                "reservation": candidate.reservation.to_dict(),
                "screen_receipt": receipt.to_dict(),
            }
        ],
        "kind": "qualification_work",
        "qualification_policy_digest": (
            manifest.qualification_policy_digest
            if policy_digest is None
            else policy_digest
        ),
        "schema_version": 2,
        "screen_lane": screen_lane,
        "service_digest": (
            manifest.digest if service_digest is None else service_digest
        ),
    }


def _request(
    configured: _Configured,
    *,
    body: dict[str, object] | None = None,
    readiness: WorkerReadiness | None = None,
    candidate=None,
    owner: str = "cpu-qualification-supervisor",
):
    readiness = configured.readiness if readiness is None else readiness
    candidate = configured.candidate if candidate is None else candidate
    body = _body(configured, candidate=candidate) if body is None else body
    lease = EvaluationLease(
        _h(f"lease-{candidate.reservation.reservation_digest}-{owner}"),
        3,
        "qualification",
        owner,
        (
            EvaluationLeaseMember(
                candidate.reservation.reservation_digest,
                "promoted",
            ),
        ),
        100,
        120,
        120,
    )
    credential = RemoteWorkerCredential("worker-credential", b"c" * 32)
    transport = RemoteWorkerTransportIdentity(
        "fixed-spool",
        _h("endpoint"),
        REMOTE_EVALUATION_PROTOCOL_DIGEST,
        credential.digest,
        readiness.service_digest,
        readiness.digest,
    )
    request = seal_remote_request(
        lease,
        readiness,
        f"{readiness.arena_id}@{readiness.service_digest}",
        transport,
        credential,
        body,
    )
    verify_remote_request(request, transport, credential)
    return request


def _authority(candidate) -> QualificationAuthorityManifest:
    reservation = candidate.reservation
    return QualificationAuthorityManifest(
        "registered",
        _h("qualification-authority"),
        _h("source"),
        _h("commitment"),
        _h("selection-reference"),
        (reservation.selected_delta_digest,),
        (reservation,),
    )


def _patch_worker_result(
    monkeypatch: pytest.MonkeyPatch,
    configured: _Configured,
    reference: EvidenceArtifactRef,
    *,
    drift_lease: bool = False,
) -> list[tuple[object, str]]:
    calls: list[tuple[object, str]] = []

    def run(
        self,
        lease,
        candidates,
        receipts,
        *,
        screen_lane,
        continuation_store,
        request_digest,
    ):
        del receipts
        calls.append((continuation_store, request_digest))
        candidate = candidates[0]
        manifest = _authority(candidate)
        outcome = QualificationIntakeOutcome(
            candidate.reservation.reservation_digest,
            candidate.reservation.selected_delta_digest,
            manifest.digest,
            QualificationDecision.PASS,
            "qualification_pass",
            False,
            reference.sha256,
            _h("qualification-report"),
        )
        batch = QualificationIntakeBatch(
            manifest.digest,
            (outcome,),
            reference,
        )
        returned_lease = replace(lease, owner="drifted-worker") if drift_lease else lease
        envelope = EvaluationResultEnvelope.seal(
            returned_lease,
            self.readiness,
            self.service,
            batch,
        )
        evaluation = EvaluationRun(
            returned_lease,
            envelope,
            batch,
            "completed",
        )
        return B300RemoteQualificationRun(evaluation, manifest, screen_lane)

    monkeypatch.setattr(
        adapter_module.B300MainnetWorker,
        "run_remote_qualification",
        run,
    )
    return calls


def test_success_captures_every_typed_batch_reference_without_paths(
    configured: _Configured,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"sealed":"qualification-attempt"}'
    reference = publish_evidence(
        configured.construction.evidence_root,
        payload,
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="cacheon.qualification.cohort-attempt.v1",
    )
    calls = _patch_worker_result(monkeypatch, configured, reference)

    request = _request(configured)
    product = configured.adapter.run(request)

    assert product.evidence_inventory == (reference,)
    assert product.evidence[0].payload == payload
    assert product.screen_lane == "primary"
    assert product.service_digest == configured.deployment.manifest.digest
    assert product.incumbent_stack == configured.construction.incumbent_stack
    assert str(configured.candidate.publication.root) not in str(product.to_dict())
    assert str(configured.construction.evidence_root) not in str(product.to_dict())
    assert calls == [(configured.adapter.continuation_store, request.digest)]


def test_worker_control_error_is_an_exact_terminal_hold(
    configured: _Configured,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "audit_execute: RuntimeError: exact retained worker failure"

    def fail(*_args, **_kwargs):
        raise OuterSessionWorkerError(message)

    monkeypatch.setattr(
        adapter_module.B300MainnetWorker,
        "run_remote_qualification",
        fail,
    )
    request = _request(configured)

    product = configured.adapter.run(request)

    assert product.reason is RemoteQualificationHoldReason.QUALIFICATION_WORKER_ERROR
    assert product.failure_type == "OuterSessionWorkerError"
    assert product.failure_message == message
    assert product.schema_version == 2
    verify_remote_qualification_hold_request(product, request)


def test_publication_identity_and_configured_root_substitution_fail_closed(
    configured: _Configured,
    tmp_path: Path,
) -> None:
    other = deployment_fixtures._bundle(tmp_path / "other-bundle", 9)
    other_receipt = deployment_fixtures._receipt(
        configured.deployment.manifest.digest,
        other,
    )
    substituted_request = _request(
        configured,
        candidate=other,
        body=_body(configured, candidate=other, receipt=other_receipt),
    )
    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="absent from the fixed pod mapping",
    ):
        configured.adapter.run(substituted_request)

    publication = configured.candidate.publication
    substituted_root = (
        tmp_path
        / "attacker-controlled"
        / publication.address_digest[:2]
        / publication.address_digest
    )
    substituted = replace(publication, root=substituted_root)
    resolver = adapter_module.B300WorkerBundleResolver((substituted,))
    adapter = adapter_module.B300RemoteQualificationAdapter(
        configured.deployment,
        configured.construction,
        configured.readiness,
        resolver,
        configured.adapter.continuation_store,
    )
    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="failed to reopen",
    ):
        adapter.run(_request(configured))


@pytest.mark.parametrize("drift", ("lane", "policy", "service", "readiness"))
def test_request_service_policy_lane_and_readiness_drift_fail_closed(
    configured: _Configured,
    drift: str,
) -> None:
    body = _body(configured)
    readiness = configured.readiness
    if drift == "lane":
        body = _body(configured, screen_lane="reproduction")
    elif drift == "policy":
        body = _body(configured, policy_digest=_h("drifted-policy"))
    elif drift == "service":
        service = _h("drifted-service")
        receipt = replace(configured.receipt, service_digest=service)
        body = _body(
            configured,
            receipt=receipt,
            service_digest=service,
        )
        readiness = replace(readiness, service_digest=service)
    else:
        readiness = replace(readiness, ready_epoch=readiness.ready_epoch + 1)
    request = _request(configured, body=body, readiness=readiness)

    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="differs from deployment",
    ):
        configured.adapter.run(request)


def test_worker_lease_drift_is_rejected(
    configured: _Configured,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = publish_evidence(
        configured.construction.evidence_root,
        b"lease-drift",
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="cacheon.qualification.cohort-attempt.v1",
    )
    _patch_worker_result(monkeypatch, configured, reference, drift_lease=True)

    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="changed the sealed lease",
    ):
        configured.adapter.run(_request(configured))


@pytest.mark.parametrize("damage", ("missing", "tampered"))
def test_evidence_omission_and_tamper_fail_closed(
    configured: _Configured,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    if damage == "missing":
        payload = b"missing-evidence"
        reference = EvidenceArtifactRef(
            "qualification.cohort-attempt",
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            "application/json",
            "cacheon.qualification.cohort-attempt.v1",
        )
    else:
        payload = b"original-evidence"
        reference = publish_evidence(
            configured.construction.evidence_root,
            payload,
            domain="qualification.cohort-attempt",
            media_type="application/json",
            schema="cacheon.qualification.cohort-attempt.v1",
        )
        artifact = (
            configured.construction.evidence_root
            / reference.domain
            / reference.sha256[:2]
            / reference.sha256
        )
        artifact.chmod(0o600)
        artifact.write_bytes(b"x" * len(payload))
        artifact.chmod(0o400)
    _patch_worker_result(monkeypatch, configured, reference)

    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="could not be captured",
    ):
        configured.adapter.run(_request(configured))


@pytest.mark.parametrize("field", ("publication_root", "control_state"))
def test_remote_request_cannot_supply_paths_or_control_state(
    configured: _Configured,
    field: str,
) -> None:
    body = _body(configured)
    if field == "publication_root":
        body["candidates"][0][field] = "/tmp/request-selected"  # type: ignore[index]
    else:
        body[field] = {"evidence_root": "/tmp/request-selected"}

    with pytest.raises(RemoteEvaluationDispatcherError, match="not closed|malformed"):
        _request(configured, body=body)


def test_adapter_rejects_construction_and_ready_authority_drift(
    configured: _Configured,
) -> None:
    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="construction differs",
    ):
        adapter_module.B300RemoteQualificationAdapter(
            configured.deployment,
            replace(
                configured.construction,
                incumbent_tree_digest=_h("drifted-tree"),
            ),
            configured.readiness,
            configured.adapter.publications,
            configured.adapter.continuation_store,
        )
    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="READY authority differs",
    ):
        adapter_module.B300RemoteQualificationAdapter(
            configured.deployment,
            configured.construction,
            replace(configured.readiness, workload_digest=_h("drifted-workload")),
            configured.adapter.publications,
            configured.adapter.continuation_store,
        )


def test_publication_resolver_is_canonical_and_path_bound(
    configured: _Configured,
    tmp_path: Path,
) -> None:
    second = deployment_fixtures._bundle(tmp_path / "resolver-second", 11).publication
    rows = (configured.candidate.publication, second)
    if rows == tuple(sorted(rows, key=lambda row: row.digest)):
        rows = tuple(reversed(rows))
    with pytest.raises(
        adapter_module.B300RemoteQualificationAdapterError,
        match="canonical fixed mapping",
    ):
        adapter_module.B300WorkerBundleResolver(rows)

    assert isinstance(configured.candidate.publication, WorkerBundlePublication)
    assert str(configured.candidate.publication.root) not in str(
        configured.candidate.publication.to_dict()
    )


def test_commission_adapter_for_canonicalizes_cohort_order(
    configured: _Configured,
    tmp_path: Path,
) -> None:
    """Cohort wire order is arrival order; adapter_for must digest-sort it.

    Mainnet 2026-08-15: every 4-member cohort was refused pre-resident because
    the commission handed FIFO-ordered publications to the canonical resolver.
    A singleton tuple is trivially sorted, so no single-candidate path could
    catch this.
    """

    from cacheon.eval.b300_remote_worker_adapter import (
        B300RemoteQualificationCommission,
    )

    second = deployment_fixtures._bundle(tmp_path / "cohort-second", 11).publication
    rows = (configured.candidate.publication, second)
    unsorted_rows = tuple(sorted(rows, key=lambda row: row.digest, reverse=True))
    commission = B300RemoteQualificationCommission(
        configured.deployment,
        configured.construction,
        configured.readiness,
    )
    derived = commission.adapter_for(
        unsorted_rows,
        configured.adapter.continuation_store,
    )
    assert derived.publications.publications == tuple(
        sorted(rows, key=lambda row: row.digest)
    )

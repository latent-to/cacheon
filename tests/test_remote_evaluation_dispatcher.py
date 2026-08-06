from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import pytest

import cacheon.chain.remote_evaluation_dispatcher as remote_dispatcher_module
import cacheon.chain.remote_qualification_evidence as remote_evidence_module
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    ScreenGrade,
    ScreenStagePolicy,
    ScreenStageResult,
    ServingShape,
    WorkloadMixture,
    WorkloadRegime,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.evaluation_coordinator import EvaluationCoordinator, WorkerReadiness
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakeError,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.publication import publish_worker_bundle
from cacheon.chain.remote_evaluation_dispatcher import (
    AuthenticatedRemoteEvaluationResponse,
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteEvaluationDispatcher,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    RemoteQualificationProduct,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    capture_remote_qualification_product,
    import_remote_qualification_evidence,
    qualification_batch_from_dict,
    qualification_batch_to_dict,
    remote_qualification_product_from_dict,
    remote_qualification_product_to_dict,
    seal_remote_response,
    verify_remote_request,
)
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.evidence_store import publish_evidence, reopen_evidence
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


SCOPE = IntakeScope("0x" + "0" * 64, 14)
POLICY = IntakePolicy(max_cohort=4, expiry_blocks=100)
BLOCK = 10


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _block_hash(block: int) -> str:
    return "0x" + f"{block:064x}"


def _manifest() -> ArenaServiceManifest:
    return ArenaServiceManifest(
        ArenaRuntimeIdentity(
            arena_id="remote-dispatch-test",
            runtime_digest=_h("runtime"),
            base_engine_digest=_h("engine"),
            validator_overlay_digest=_h("overlay"),
            worker_distribution_digest=_h("worker-distribution"),
            model_revision_digest=_h("model-revision"),
            model_manifest_digest=_h("model-manifest"),
            model_content_digest=_h("model-content"),
            target_architecture="sm120",
            topology_class="tp4-test",
            topology_digest=_h("topology"),
            gpu_count=4,
            tensor_parallel_size=4,
        ),
        WorkloadMixture(
            _h("corpus"),
            "test-seed-v1",
            (
                WorkloadRegime(
                    "decode", "decode", 500_000, (ServingShape(128, 32, 1, 1),)
                ),
                WorkloadRegime(
                    "prefill",
                    "long_prefill",
                    500_000,
                    (ServingShape(1024, 8, 1, 1),),
                ),
            ),
        ),
        ArenaCapacityPolicy(32, 100, 4, 4, 4, 3, 3, 3),
        NonCrownScreenPolicy(
            tuple(ScreenStagePolicy(stage, 1_000) for stage in SCREEN_STAGES)
        ),
        _h("qualification-policy"),
        _h("provider"),
    )


def _incumbent(service: ArenaService, *, marker: str = "remote") -> EvaluationStackManifest:
    snapshot = {
        "composition_rules": [],
        "policy_version": "target-catalog.v1",
        "schema_version": 1,
        "targets": [{"marker": marker, "target_id": "target.0"}],
    }
    return EvaluationStackManifest(
        runtime_digest=service.manifest.runtime.runtime_digest,
        base_engine_digest=service.manifest.runtime.base_engine_digest,
        arena_digest=service.identity,
        catalog_snapshot=snapshot,
        catalog_digest=canonical_digest("cacheon.target-catalog", snapshot),
        entries={},
    )


class _Provider:
    provider_digest = _h("provider")

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.qualification_calls = 0

    def run_screen(self, _manifest, stage, candidate):
        self.calls.append(stage.stage)
        return ScreenStageResult(
            stage.stage,
            ScreenGrade.PASS,
            _h(f"screen:{stage.stage}:{candidate.digest}"),
            1,
        )

    def build_qualification(self, _request, state=None):  # pragma: no cover
        self.qualification_calls += 1
        raise AssertionError("qualification is not part of the screen fixture")


@dataclasses.dataclass
class _Cursor:
    point: tuple[int, str]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def __call__(self) -> tuple[int, str]:
        with self._lock:
            return self.point

    def set(self, block: int) -> None:
        with self._lock:
            self.point = (block, _block_hash(block))


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "private" / "intake.sqlite3"


def _store(tmp_path: Path) -> FinalizedIntakeStore:
    return FinalizedIntakeStore(_db_path(tmp_path), POLICY, scope=SCOPE)


def _published_rows(tmp_path: Path, count: int):
    publications = []
    arrivals = []
    for index in range(count):
        source = tmp_path / f"source-{index}"
        source.mkdir(parents=True)
        leaf = source / "manifest.toml"
        leaf.write_text(f"bundle_id = 'candidate-{index}'\n")
        source.chmod(0o700)
        leaf.chmod(0o600)
        committed = content_hash(source)
        publication = publish_worker_bundle(
            source, tmp_path / "publications", committed
        )
        publications.append(publication)
        arrivals.append(
            FinalizedArrival(
                f"miner-{index}",
                committed,
                f"https://example.invalid/{index}",
                BLOCK,
                _block_hash(BLOCK),
                index,
            )
        )
    with _store(tmp_path) as store:
        reserved = store.reserve_finalized(
            tuple(arrivals),
            finalized_block=BLOCK,
            finalized_block_hash=_block_hash(BLOCK),
        )
        result = []
        for index, (row, publication) in enumerate(
            zip(reserved, publications, strict=True)
        ):
            store.mark_fetching(row.reservation_id)
            result.append(
                store.mark_published(
                    row.reservation_id,
                    delta_fingerprint=SubmittedDeltaFingerprint(
                        "component",
                        f"target.{index}",
                        _h(f"base:{index}"),
                        (f"slot.{index}",),
                        _h(f"archive:{index}"),
                        _h(f"selected:{index}"),
                        _h(f"exact:{index}"),
                        (_h(f"source:{index}"),),
                        (_h(f"binary:{index}"),),
                    ),
                    publication_digest=publication.digest,
                    publication_root=publication.root,
                )
            )
        return tuple(result)


def _advance(tmp_path: Path, cursor: _Cursor, block: int) -> None:
    with _store(tmp_path) as store:
        store.reserve_finalized(
            (), finalized_block=block, finalized_block_hash=_block_hash(block)
        )
    cursor.set(block)


def _coordinator(
    tmp_path: Path,
    service: ArenaService,
    cursor: _Cursor,
    **changes,
) -> EvaluationCoordinator:
    readiness = WorkerReadiness.for_service(
        service, ready_receipt_digest=_h("ready-receipt"), ready_epoch=7
    )
    options = dict(
        intake_db=_db_path(tmp_path),
        policy=POLICY,
        scope=SCOPE,
        service=service,
        readiness=readiness,
        owner="remote-cpu-dispatch-test",
        advance_finalized_cursor=cursor,
        lease_blocks=20,
        heartbeat_interval_s=10.0,
        heartbeat_join_timeout_s=1.0,
        lock_retry_delay_s=0.001,
    )
    options.update(changes)
    return EvaluationCoordinator(**options)


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


class _Transport:
    def __init__(
        self,
        coordinator: EvaluationCoordinator,
        credential: RemoteWorkerCredential,
        *,
        hook=None,
        forged: bool = False,
        fail: bool = False,
        endpoint: str = "worker-endpoint-a",
    ) -> None:
        self.coordinator = coordinator
        self.credential = credential
        self.identity = _transport_identity(
            coordinator, credential, endpoint=endpoint
        )
        self.hook = hook
        self.forged = forged
        self.fail = fail
        self.requests: list[RemoteEvaluationRequest] = []

    def run_screen(self, request, *, job):
        parsed = RemoteEvaluationRequest.from_dict(request.to_dict())
        verify_remote_request(parsed, self.identity, self.credential)
        assert parsed.lease_id == job.lease.lease_id
        assert parsed.members == job.lease.members
        self.requests.append(parsed)
        if self.hook is not None:
            self.hook(parsed, job)
        if self.fail:
            raise RuntimeError("worker transport disappeared")
        receipt = self.coordinator.service.screen(job.candidate)
        response = seal_remote_response(
            parsed, receipt, self.identity, self.credential
        )
        response = AuthenticatedRemoteEvaluationResponse.from_dict(
            response.to_dict()
        )
        if self.forged:
            response = dataclasses.replace(response, auth_tag="f" * 64)
        return response

    def run_qualification(self, request, *, job, work, prepared):  # pragma: no cover
        raise AssertionError("qualification is not part of the screen fixture")


def _dispatcher(
    coordinator: EvaluationCoordinator,
    transport: _Transport,
    credential: RemoteWorkerCredential,
) -> RemoteEvaluationDispatcher:
    return RemoteEvaluationDispatcher(
        coordinator=coordinator,
        transport=transport,
        credential=credential,
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


def test_remote_screen_claim_is_fifo_closed_authenticated_and_committed(
    tmp_path: Path,
) -> None:
    first, second = _published_rows(tmp_path, 2)
    service = ArenaService(_manifest(), _Provider())
    cursor = _Cursor((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(tmp_path, service, cursor)
    credential = RemoteWorkerCredential("screen-key-v1", b"s" * 32)
    lock_checks = []

    def prove_closed_wire_and_unlocked(request, job) -> None:
        with _store(tmp_path) as other:
            lock_checks.append(other.finalized_cursor())
        wire = json.dumps(request.to_dict(), sort_keys=True)
        assert str(job.publication.root) not in wire
        assert not ({"command", "argv", "env", "shell"} & _all_keys(request.to_dict()))

    transport = _Transport(coordinator, credential, hook=prove_closed_wire_and_unlocked)

    result = _dispatcher(coordinator, transport, credential).dispatch_screen_once()

    assert result is not None and result.disposition == "completed"
    assert result.lease.reservation_ids == (first.reservation_id,)
    assert lock_checks == [(BLOCK, _block_hash(BLOCK))]
    assert len(transport.requests) == 1
    with _store(tmp_path) as store:
        assert store.get(first.reservation_id).status == "promoted"
        assert store.get(second.reservation_id).status == "published"
        events = store.evaluation_lease_events(lease_id=result.lease.lease_id)
    assert [row.event_type for row in events] == ["claimed", "completed"]


def _all_keys(value: object) -> set[str]:
    if type(value) is dict:
        return set(value) | {
            key
            for item in value.values()
            for key in _all_keys(item)
        }
    if type(value) is list:
        return {key for item in value for key in _all_keys(item)}
    return set()


@pytest.mark.parametrize("mode", ["exception", "forged-response"])
def test_remote_failure_releases_without_attempt_and_replacement_reclaims(
    tmp_path: Path,
    mode: str,
) -> None:
    first, second = _published_rows(tmp_path, 2)
    service = ArenaService(_manifest(), _Provider())
    cursor = _Cursor((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(tmp_path, service, cursor)
    credential = RemoteWorkerCredential("screen-key-v1", b"s" * 32)
    failed = _Transport(
        coordinator,
        credential,
        fail=mode == "exception",
        forged=mode == "forged-response",
    )

    with pytest.raises(RemoteEvaluationDispatcherError, match="remote_screen_infrastructure"):
        _dispatcher(coordinator, failed, credential).dispatch_screen_once()

    with _store(tmp_path) as store:
        retained = store.get(first.reservation_id)
        events = store.evaluation_lease_events(reservation_id=first.reservation_id)
    assert (retained.status, retained.screen_attempts) == ("published", 0)
    assert [row.event_type for row in events] == ["claimed", "released"]

    replacement = _Transport(
        coordinator, credential, endpoint="replacement-worker-endpoint"
    )
    # A replacement pod has a new pinned endpoint identity and therefore a new
    # dispatcher/transport identity, but reclaims the same oldest durable row.
    result = _dispatcher(coordinator, replacement, credential).dispatch_screen_once()
    assert result is not None
    assert result.lease.reservation_ids == (first.reservation_id,)
    assert result.lease.generation == events[0].generation + 1
    with _store(tmp_path) as store:
        assert store.get(first.reservation_id).status == "promoted"
        assert store.get(second.reservation_id).status == "published"


def test_remote_screen_heartbeat_advances_while_transport_is_blocked(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _Cursor((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=3,
        heartbeat_interval_s=0.01,
    )
    credential = RemoteWorkerCredential("screen-key-v1", b"s" * 32)
    entered = threading.Event()
    release = threading.Event()

    def block_transport(_request, _job) -> None:
        entered.set()
        assert release.wait(5)

    transport = _Transport(coordinator, credential, hook=block_transport)
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(
                _dispatcher(coordinator, transport, credential).dispatch_screen_once()
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            outcome.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(5)
    _advance(tmp_path, cursor, BLOCK + 1)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            with _store(tmp_path) as store:
                events = store.evaluation_lease_events(
                    reservation_id=row.reservation_id
                )
        except IntakeError as exc:
            assert str(exc) == "another intake controller owns this database"
        else:
            if any(event.event_type == "heartbeat" for event in events):
                break
        threading.Event().wait(0.01)
    else:  # pragma: no cover
        pytest.fail("remote lease heartbeat did not advance")
    release.set()
    thread.join(5)

    assert not thread.is_alive()
    assert len(outcome) == 1 and not isinstance(outcome[0], BaseException)
    with _store(tmp_path) as store:
        assert store.get(row.reservation_id).status == "promoted"
        events = store.evaluation_lease_events(
            lease_id=outcome[0].lease.lease_id  # type: ignore[union-attr]
        )
    assert [event.event_type for event in events] == [
        "claimed",
        "heartbeat",
        "completed",
    ]


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
    coordinator = _coordinator(tmp_path, service, cursor)
    credential = RemoteWorkerCredential("screen-key-v1", b"s" * 32)
    transport = _Transport(coordinator, credential)
    transport.identity = dataclasses.replace(
        transport.identity, worker_readiness_digest=_h("another-ready-epoch")
    )

    with pytest.raises(RemoteEvaluationDispatcherError, match="differs from CPU authority"):
        _dispatcher(coordinator, transport, credential)

    with _store(tmp_path) as store:
        assert store.active_evaluation_leases() == ()
        assert store.get(row.reservation_id).screen_attempts == 0


def test_remote_qualification_is_path_free_imported_and_committed_without_cpu_plan(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _Cursor((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(tmp_path, service, cursor)
    screened = coordinator.run_screen_once()
    assert screened is not None and screened.disposition == "completed"

    credential = RemoteWorkerCredential("qualification-key-v1", b"q" * 32)
    pod_root = tmp_path / "pod-evidence"
    attempt_ref = publish_evidence(
        pod_root,
        b'{"attempt":"pod-only"}',
        domain="qualification-attempt",
        media_type="application/json",
        schema="cacheon.qualification.test-attempt.v1",
    )
    incumbent = _incumbent(service)
    incumbent_tree_digest = _h("incumbent-tree")

    class QualificationTransport:
        def __init__(self) -> None:
            self.identity = _transport_identity(coordinator, credential)
            self.requests: list[RemoteEvaluationRequest] = []

        def run_screen(self, request, *, job):  # pragma: no cover
            raise AssertionError("screen is not part of the qualification transport")

        def run_qualification(self, request):
            parsed = RemoteEvaluationRequest.from_dict(request.to_dict())
            verify_remote_request(parsed, self.identity, credential)
            self.requests.append(parsed)
            assert parsed.body["screen_lane"] == "primary"
            assert "authority_manifest" not in parsed.body
            authority = _authority_for_request(parsed)
            batch = _failed_batch(authority, attempt_ref)
            product = capture_remote_qualification_product(
                batch=batch,
                authority_manifest=authority,
                incumbent_stack=incumbent,
                incumbent_tree_digest=incumbent_tree_digest,
                screen_lane=parsed.body["screen_lane"],
                service_digest=service.identity,
                readiness=coordinator.readiness,
                evidence_root=pod_root,
                evidence_references=(attempt_ref,),
            )
            wire = json.dumps(product.to_dict(), sort_keys=True)
            assert str(pod_root) not in wire
            return AuthenticatedRemoteEvaluationResponse.from_dict(
                seal_remote_response(
                    parsed,
                    product,
                    self.identity,
                    credential,
                ).to_dict()
            )

    transport = QualificationTransport()
    cpu_root = tmp_path / "cpu-evidence"
    result = RemoteEvaluationDispatcher(
        coordinator=coordinator,
        transport=transport,
        credential=credential,
        qualification_evidence_root=cpu_root,
        qualification_incumbent_stack=incumbent,
        qualification_incumbent_tree_digest=incumbent_tree_digest,
    ).dispatch_qualification_once()

    assert result is not None and result.disposition == "completed"
    assert result.lease.reservation_ids == (row.reservation_id,)
    assert provider.qualification_calls == 0
    assert reopen_evidence(cpu_root, attempt_ref) == b'{"attempt":"pod-only"}'
    with _store(tmp_path) as store:
        retained = store.get(row.reservation_id)
        stack = store.evaluation_stack(service.identity)
        events = store.evaluation_lease_events(lease_id=result.lease.lease_id)
    assert (retained.status, retained.decision) == ("failed", "FAIL")
    assert stack.manifest == _incumbent(service)
    assert stack.tree_digest == _h("incumbent-tree")
    assert [event.event_type for event in events] == ["claimed", "completed"]
    assert len(transport.requests) == 1


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


@pytest.mark.parametrize("drift", ["epoch", "service", "lane", "incumbent"])
def test_remote_qualification_rejects_signed_product_outside_request_authority(
    tmp_path: Path,
    drift: str,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _Cursor((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(tmp_path, service, cursor)
    assert coordinator.run_screen_once() is not None
    credential = RemoteWorkerCredential("qualification-drift-key", b"d" * 32)
    pod_root = tmp_path / f"pod-evidence-{drift}"
    reference = publish_evidence(
        pod_root,
        b"drift evidence",
        domain="qualification-attempt",
        media_type="application/json",
        schema="cacheon.qualification.drift-test.v1",
    )
    expected_incumbent = _incumbent(service)
    expected_tree_digest = _h("drift-tree")

    class DriftTransport:
        identity = _transport_identity(coordinator, credential)

        def run_screen(self, request, *, job):  # pragma: no cover
            raise AssertionError

        def run_qualification(self, request):
            parsed = RemoteEvaluationRequest.from_dict(request.to_dict())
            authority = _authority_for_request(parsed)
            incumbent = expected_incumbent
            product = capture_remote_qualification_product(
                batch=_failed_batch(authority, reference),
                authority_manifest=authority,
                incumbent_stack=incumbent,
                incumbent_tree_digest=expected_tree_digest,
                screen_lane="primary",
                service_digest=service.identity,
                readiness=coordinator.readiness,
                evidence_root=pod_root,
                evidence_references=(reference,),
            )
            if drift == "epoch":
                product = dataclasses.replace(
                    product,
                    ready_epoch=product.ready_epoch + 1,
                )
            elif drift == "lane":
                product = dataclasses.replace(product, screen_lane="reproduction")
            elif drift == "service":
                wrong_service = _h("another-service")
                wrong_stack = EvaluationStackManifest(
                    runtime_digest=incumbent.runtime_digest,
                    base_engine_digest=incumbent.base_engine_digest,
                    arena_digest=wrong_service,
                    catalog_snapshot=incumbent.catalog_snapshot,
                    catalog_digest=incumbent.catalog_digest,
                    entries=incumbent.entries,
                )
                product = dataclasses.replace(
                    product,
                    service_digest=wrong_service,
                    incumbent_stack=wrong_stack,
                )
            else:
                product = dataclasses.replace(
                    product,
                    incumbent_stack=_incumbent(service, marker="pod-substituted"),
                )
            return seal_remote_response(
                parsed,
                product,
                self.identity,
                credential,
            )

    dispatcher = RemoteEvaluationDispatcher(
        coordinator=coordinator,
        transport=DriftTransport(),
        credential=credential,
        qualification_evidence_root=tmp_path / f"cpu-evidence-{drift}",
        qualification_incumbent_stack=expected_incumbent,
        qualification_incumbent_tree_digest=expected_tree_digest,
    )
    with pytest.raises(
        RemoteEvaluationDispatcherError,
        match="remote_qualification_infrastructure",
    ):
        dispatcher.dispatch_qualification_once()

    with _store(tmp_path) as store:
        retained = store.get(row.reservation_id)
        events = store.evaluation_lease_events(reservation_id=row.reservation_id)
        dispositions = store.qualification_dispositions(row.reservation_id)
    assert retained.status == "promoted"
    assert dispositions == ()
    assert [event.event_type for event in events][-2:] == ["claimed", "released"]

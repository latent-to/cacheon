from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.chain.evaluation_coordinator as coordinator_module
import cacheon.eval.qualification_intake as qualification_intake_module
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCapacityPolicy,
    ArenaQualificationWork,
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
from cacheon.chain.evaluation_coordinator import (
    ClaimedQualificationEvaluation,
    EvaluationCoordinator,
    EvaluationCoordinatorError,
    EvaluationResultEnvelope,
    WorkerReadiness,
)
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakeError,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.publication import publish_worker_bundle
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationPlanFactory,
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
    runtime = ArenaRuntimeIdentity(
        arena_id="coordinator-test",
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
    )
    workload = WorkloadMixture(
        _h("corpus"),
        "test-seed-v1",
        (
            WorkloadRegime(
                "decode",
                "decode",
                500_000,
                (ServingShape(128, 32, 1, 1),),
            ),
            WorkloadRegime(
                "prefill",
                "long_prefill",
                500_000,
                (ServingShape(1024, 8, 1, 1),),
            ),
        ),
    )
    return ArenaServiceManifest(
        runtime,
        workload,
        ArenaCapacityPolicy(32, 100, 4, 4, 4, 3, 3, 3),
        NonCrownScreenPolicy(
            tuple(ScreenStagePolicy(stage, 1_000) for stage in SCREEN_STAGES)
        ),
        _h("qualification-policy"),
        _h("provider"),
    )


class _Provider:
    provider_digest = _h("provider")

    def __init__(self, *, screen_hook=None, qualification_builder=None):
        self.screen_hook = screen_hook
        self.qualification_builder = qualification_builder
        self.screen_calls: list[str] = []
        self.qualification_calls = 0

    def run_screen(self, _manifest, stage, candidate):
        self.screen_calls.append(stage.stage)
        if self.screen_hook is not None:
            self.screen_hook(stage, candidate)
        return ScreenStageResult(
            stage.stage,
            ScreenGrade.PASS,
            _h(f"screen:{stage.stage}:{candidate.digest}"),
            1,
        )

    def build_qualification(self, request, state=None):
        self.qualification_calls += 1
        assert state is None
        if self.qualification_builder is None:
            raise AssertionError("qualification was not expected")
        return self.qualification_builder(request)


@dataclasses.dataclass
class _CursorAuthority:
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
            source,
            tmp_path / "publications",
            committed,
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


def _coordinator(
    tmp_path: Path,
    service: ArenaService,
    cursor: _CursorAuthority,
    **changes,
) -> EvaluationCoordinator:
    readiness = changes.pop(
        "readiness",
        WorkerReadiness.for_service(
            service,
            ready_receipt_digest=_h("ready-receipt"),
            ready_epoch=7,
        ),
    )
    options = dict(
        intake_db=_db_path(tmp_path),
        policy=POLICY,
        scope=SCOPE,
        service=service,
        readiness=readiness,
        owner="cpu-coordinator-test",
        advance_finalized_cursor=cursor,
        lease_blocks=20,
        heartbeat_interval_s=10.0,
        heartbeat_join_timeout_s=1.0,
        lock_retry_delay_s=0.001,
    )
    options.update(changes)
    return EvaluationCoordinator(**options)


def _advance(tmp_path: Path, cursor: _CursorAuthority, block: int) -> None:
    with _store(tmp_path) as store:
        store.reserve_finalized(
            (),
            finalized_block=block,
            finalized_block_hash=_block_hash(block),
        )
    cursor.set(block)


def test_readiness_mismatch_creates_no_lease_or_worker_call(tmp_path: Path) -> None:
    row = _published_rows(tmp_path, 1)[0]
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    readiness = dataclasses.replace(
        WorkerReadiness.for_service(
            service,
            ready_receipt_digest=_h("ready"),
            ready_epoch=1,
        ),
        runtime_digest=_h("wrong-runtime"),
    )
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        readiness=readiness,
    )

    with pytest.raises(EvaluationCoordinatorError, match="READY identity"):
        coordinator.run_screen_once()

    with _store(tmp_path) as store:
        assert store.active_evaluation_leases() == ()
        assert store.get(row.reservation_id).screen_attempts == 0
    assert provider.screen_calls == []


def test_screen_is_fifo_and_provider_runs_without_controller_lock(
    tmp_path: Path,
) -> None:
    first, second = _published_rows(tmp_path, 2)
    lock_checks = []

    def prove_unlocked(_stage, _candidate) -> None:
        with _store(tmp_path) as other:
            lock_checks.append(other.finalized_cursor())

    provider = _Provider(screen_hook=prove_unlocked)
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))

    result = _coordinator(tmp_path, service, cursor).run_screen_once()

    assert result is not None and result.disposition == "completed"
    assert result.lease.reservation_ids == (first.reservation_id,)
    assert len(lock_checks) == len(SCREEN_STAGES)
    with _store(tmp_path) as store:
        assert store.get(first.reservation_id).status == "promoted"
        assert store.get(second.reservation_id).status == "published"


def test_intake_advances_while_worker_is_blocked_and_heartbeat_cas_extends(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    entered = threading.Event()
    release = threading.Event()
    blocked_once = False

    def block_worker(_stage, _candidate) -> None:
        nonlocal blocked_once
        if blocked_once:
            return
        blocked_once = True
        entered.set()
        assert release.wait(5)

    provider = _Provider(screen_hook=block_worker)
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=3,
        heartbeat_interval_s=0.01,
    )
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(coordinator.run_screen_once())
        except BaseException as exc:  # pragma: no cover - asserted below
            outcome.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(5)
    # The synchronous worker is still blocked, but the coordinator closed its
    # flock-backed store.  An independent intake owner can advance exact finality.
    _advance(tmp_path, cursor, BLOCK + 1)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            with _store(tmp_path) as store:
                observed = store.evaluation_lease_events(
                    reservation_id=row.reservation_id
                )
        except IntakeError as exc:
            assert str(exc) == "another intake controller owns this database"
        else:
            if any(event.event_type == "heartbeat" for event in observed):
                break
        threading.Event().wait(0.01)
    else:  # pragma: no cover - deterministic failure aid
        pytest.fail("heartbeat did not observe the advanced durable cursor")
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


def test_heartbeat_retries_after_full_transient_lock_attempt_budget(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=3,
        heartbeat_interval_s=0.01,
        lock_attempts=3,
        lock_retry_delay_s=0.001,
    )
    claim = coordinator.claim_screen()
    assert claim is not None
    _advance(tmp_path, cursor, BLOCK + 1)
    calls = 0

    def contended_factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= coordinator.lock_attempts:
            raise IntakeError("another intake controller owns this database")
        return FinalizedIntakeStore(*args, **kwargs)

    coordinator._store_factory = contended_factory
    heartbeat = coordinator_module._LeaseHeartbeat(coordinator, claim.lease)
    heartbeat.start()
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
    else:  # pragma: no cover - deterministic failure aid
        pytest.fail("heartbeat did not recover from transient intake ownership")
    lease, error = heartbeat.stop()
    assert error is None
    assert calls >= coordinator.lock_attempts + 1
    assert lease.expires_block == BLOCK + 4
    with _store(tmp_path) as store:
        events = store.evaluation_lease_events(lease_id=lease.lease_id)
    assert [event.event_type for event in events] == ["claimed", "heartbeat"]


def test_heartbeat_retries_after_full_transient_cursor_mismatch_budget(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=3,
        heartbeat_interval_s=0.01,
        lock_attempts=3,
        lock_retry_delay_s=0.001,
    )
    claim = coordinator.claim_screen()
    assert claim is not None
    _advance(tmp_path, cursor, BLOCK + 1)
    calls = 0

    def stale_then_live_cursor() -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls <= coordinator.lock_attempts:
            return BLOCK, _block_hash(BLOCK)
        return cursor()

    coordinator.advance_finalized_cursor = stale_then_live_cursor
    heartbeat = coordinator_module._LeaseHeartbeat(coordinator, claim.lease)
    heartbeat.start()
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
    else:  # pragma: no cover - deterministic failure aid
        pytest.fail("heartbeat did not recover from transient cursor mismatch")
    lease, error = heartbeat.stop()

    assert error is None
    assert calls >= coordinator.lock_attempts + 1
    assert lease.expires_block == BLOCK + 4
    with _store(tmp_path) as store:
        events = store.evaluation_lease_events(lease_id=lease.lease_id)
    assert [event.event_type for event in events] == ["claimed", "heartbeat"]


def test_transient_heartbeat_contention_does_not_admit_an_expired_result(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=3,
        heartbeat_interval_s=0.01,
        lock_attempts=3,
        lock_retry_delay_s=0.001,
    )
    claim = coordinator.claim_screen()
    assert claim is not None
    receipt = service.screen(claim.candidate)
    exhausted = threading.Event()
    calls = 0

    def always_contended(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls >= coordinator.lock_attempts:
            exhausted.set()
        raise IntakeError("another intake controller owns this database")

    coordinator._store_factory = always_contended
    heartbeat = coordinator_module._LeaseHeartbeat(coordinator, claim.lease)
    heartbeat.start()
    assert exhausted.wait(2)
    _advance(tmp_path, cursor, claim.lease.expires_block)
    lease, error = heartbeat.stop()

    assert error is None
    assert calls >= coordinator.lock_attempts
    assert lease == claim.lease
    coordinator._store_factory = FinalizedIntakeStore
    envelope = EvaluationResultEnvelope.seal(
        lease,
        coordinator.readiness,
        service,
        receipt,
    )
    with pytest.raises(EvaluationCoordinatorError, match="after lease expiry"):
        coordinator.commit_screen_result(claim, receipt, envelope)

    with _store(tmp_path) as store:
        retained = store.get(row.reservation_id)
        events = store.evaluation_lease_events(lease_id=lease.lease_id)
    assert (retained.status, retained.screen_attempts) == ("published", 0)
    assert [event.event_type for event in events] == ["claimed", "expired"]


def test_provider_exception_releases_without_consuming_screen_attempt(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]

    def fail(_stage, _candidate) -> None:
        raise RuntimeError("worker exploded")

    service = ArenaService(_manifest(), _Provider(screen_hook=fail))
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    with pytest.raises(EvaluationCoordinatorError, match="screen_provider_exception"):
        _coordinator(tmp_path, service, cursor).run_screen_once()

    with _store(tmp_path) as store:
        retained = store.get(row.reservation_id)
        events = store.evaluation_lease_events(reservation_id=row.reservation_id)
    assert (retained.status, retained.screen_attempts) == ("published", 0)
    assert [event.event_type for event in events] == ["claimed", "released"]


def test_expiry_reclaims_oldest_and_cross_lease_or_stale_envelope_cannot_commit(
    tmp_path: Path,
) -> None:
    first, second = _published_rows(tmp_path, 2)
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=2,
    )
    original = coordinator.claim_screen()
    assert original is not None and original.lease.reservation_ids == (
        first.reservation_id,
    )
    receipt = service.screen(original.candidate)
    old_envelope = EvaluationResultEnvelope.seal(
        original.lease,
        coordinator.readiness,
        service,
        receipt,
    )

    _advance(tmp_path, cursor, BLOCK + 2)
    reclaimed = coordinator.claim_screen()
    assert reclaimed is not None
    assert reclaimed.lease.reservation_ids == (first.reservation_id,)
    assert reclaimed.lease.generation == original.lease.generation + 1
    assert reclaimed.lease.lease_id != original.lease.lease_id
    with pytest.raises(EvaluationCoordinatorError, match="exact live lease"):
        coordinator.commit_screen_result(reclaimed, receipt, old_envelope)

    fresh_receipt = service.screen(reclaimed.candidate)
    fresh_envelope = EvaluationResultEnvelope.seal(
        reclaimed.lease,
        coordinator.readiness,
        service,
        fresh_receipt,
    )
    _advance(tmp_path, cursor, BLOCK + 4)
    with pytest.raises(EvaluationCoordinatorError, match="durable lease"):
        coordinator.commit_screen_result(reclaimed, fresh_receipt, fresh_envelope)

    with _store(tmp_path) as store:
        assert store.get(first.reservation_id).screen_attempts == 0
        assert store.get(second.reservation_id).status == "published"


def test_lock_collision_retries_transiently_and_commit_collision_fails_closed(
    tmp_path: Path,
) -> None:
    row = _published_rows(tmp_path, 1)[0]
    service = ArenaService(_manifest(), _Provider())
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    calls = 0

    def transient_factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise IntakeError("another intake controller owns this database")
        return FinalizedIntakeStore(*args, **kwargs)

    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        store_factory=transient_factory,
    )
    claim = coordinator.claim_screen()
    assert claim is not None and calls == 3
    receipt = service.screen(claim.candidate)
    envelope = EvaluationResultEnvelope.seal(
        claim.lease,
        coordinator.readiness,
        service,
        receipt,
    )

    def always_busy(*_args, **_kwargs):
        raise IntakeError("another intake controller owns this database")

    coordinator._store_factory = always_busy
    with pytest.raises(EvaluationCoordinatorError, match="did not stabilize"):
        coordinator.commit_screen_result(claim, receipt, envelope)

    with _store(tmp_path) as store:
        assert store.active_evaluation_leases() == (claim.lease,)
        retained = store.get(row.reservation_id)
        assert (retained.status, retained.screen_attempts) == (
            "published",
            0,
        )


def _promote_all(
    tmp_path: Path,
    service: ArenaService,
    cursor: _CursorAuthority,
    count: int,
) -> tuple[str, ...]:
    ids = tuple(row.reservation_id for row in _published_rows(tmp_path, count))
    coordinator = _coordinator(tmp_path, service, cursor)
    for expected in ids:
        result = coordinator.run_screen_once()
        assert result is not None and result.lease.reservation_ids == (expected,)
    return ids


def _qualification_fixture_builder(
    tmp_path: Path,
    service: ArenaService,
    monkeypatch,
    *,
    inside_accept: list[bool] | None = None,
):
    class FakePlan:
        pass

    @dataclasses.dataclass(frozen=True)
    class Baseline:
        tree_digest: str

    @dataclasses.dataclass(frozen=True)
    class Arm:
        incumbent: EvaluationStackManifest
        baseline_before: Baseline

    monkeypatch.setattr(
        qualification_intake_module,
        "CausalQualificationInput",
        FakePlan,
    )
    monkeypatch.setattr(
        qualification_intake_module,
        "qualification_authority_digest",
        lambda _plan: _h("qualification-authority"),
    )
    snapshot = {
        "schema_version": 1,
        "policy_version": "target-catalog.v1",
        "targets": [{"target_id": "target.0", "marker": "coordinator"}],
        "composition_rules": [],
    }
    incumbent = EvaluationStackManifest(
        runtime_digest=service.manifest.runtime.runtime_digest,
        base_engine_digest=service.manifest.runtime.base_engine_digest,
        arena_digest=service.identity,
        catalog_snapshot=snapshot,
        catalog_digest=canonical_digest("cacheon.target-catalog", snapshot),
        entries={},
    )
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()

    def build(request):
        if inside_accept is not None:
            assert inside_accept == [False]
        with _store(tmp_path) as unlocked:
            assert unlocked.finalized_cursor() == (BLOCK, _block_hash(BLOCK))
        reservations = tuple(row.reservation for row in request.candidates)
        manifest = QualificationAuthorityManifest(
            "registered",
            _h("qualification-authority"),
            _h("qualification-source"),
            _h("qualification-commitment"),
            _h("qualification-secret-ref"),
            tuple(row.selected_delta_digest for row in reservations),
            reservations,
        )
        plan = FakePlan()
        plan.selection_secret = b"s" * 32
        plan.prepared = SimpleNamespace(
            source=SimpleNamespace(digest=manifest.source_digest),
            candidates=tuple(
                SimpleNamespace(
                    arm=Arm(incumbent, Baseline(_h("baseline-tree")))
                )
                for _ in reservations
            ),
        )
        plan.commitment = SimpleNamespace(digest=manifest.commitment_digest)
        plan.candidates = tuple(
            SimpleNamespace(selected_delta_digest=row.selected_delta_digest)
            for row in reservations
        )
        plan.evidence_root = evidence_root
        factory = QualificationPlanFactory(
            manifest,
            lambda reference: (
                plan.selection_secret
                if reference == manifest.selection_secret_reference
                else b""
            ),
            lambda _secret: plan,
        )
        return ArenaQualificationWork(
            factory,
            object(),
            lambda *_args: None,
            lambda **_kwargs: None,
            10.0,
            service.manifest.qualification_policy_digest,
        )

    return build


def _no_decision_batch(work, reason: str) -> QualificationIntakeBatch:
    failure = _h(f"failure:{reason}")
    outcomes = tuple(
        QualificationIntakeOutcome(
            row.reservation_digest,
            row.selected_delta_digest,
            work.factory.manifest.digest,
            QualificationDecision.NO_DECISION,
            reason,
            True,
            failure_digest=failure,
        )
        for row in work.factory.manifest.reservations
    )
    ids = tuple(row.reservation_digest for row in work.factory.manifest.reservations)
    groups = (
        (ids[0],)
        if len(ids) == 1
        else (ids[: len(ids) // 2], ids[len(ids) // 2 :])
    )
    strategy = "requeue" if len(ids) == 1 else "bisect"
    return QualificationIntakeBatch(
        work.factory.manifest.digest,
        outcomes,
        retry_plan=QualificationRetryPlan(
            work.factory.manifest.digest,
            strategy,
            groups,
            failure,
        ),
    )


def _remote_incumbent(
    service: ArenaService,
    *,
    arena_digest: str | None = None,
    marker: str = "remote-commit",
) -> EvaluationStackManifest:
    snapshot = {
        "schema_version": 1,
        "policy_version": "target-catalog.v1",
        "targets": [{"target_id": "target.0", "marker": marker}],
        "composition_rules": [],
    }
    return EvaluationStackManifest(
        runtime_digest=service.manifest.runtime.runtime_digest,
        base_engine_digest=service.manifest.runtime.base_engine_digest,
        arena_digest=arena_digest or service.identity,
        catalog_snapshot=snapshot,
        catalog_digest=canonical_digest("cacheon.target-catalog", snapshot),
        entries={},
    )


def _remote_commit_product(
    tmp_path: Path,
    coordinator: EvaluationCoordinator,
    claim: ClaimedQualificationEvaluation,
):
    reservations = tuple(row.reservation for row in claim.candidates)
    authority = QualificationAuthorityManifest(
        "registered",
        _h("remote-commit-authority"),
        _h("remote-commit-source"),
        _h("remote-commit-commitment"),
        _h("remote-commit-secret"),
        tuple(row.selected_delta_digest for row in reservations),
        reservations,
    )
    evidence_root = tmp_path / "remote-cpu-evidence"
    attempt_ref = publish_evidence(
        evidence_root,
        b'{"remote":"attempt"}',
        domain="qualification-attempt",
        media_type="application/json",
        schema="cacheon.qualification.remote-commit-test.v1",
    )
    batch = QualificationIntakeBatch(
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
                report_digest=_h(f"remote-report:{row.reservation_digest}"),
            )
            for row in reservations
        ),
        attempt_ref,
    )
    envelope = EvaluationResultEnvelope.seal(
        claim.lease,
        coordinator.readiness,
        coordinator.service,
        batch,
    )
    return authority, batch, envelope, evidence_root, attempt_ref


def test_systemic_qualification_releases_whole_cohort_without_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = _promote_all(tmp_path, service, cursor, 2)
    provider.qualification_builder = _qualification_fixture_builder(
        tmp_path,
        service,
        monkeypatch,
    )

    def systemic(factory, **_kwargs):
        with _store(tmp_path) as unlocked:
            assert unlocked.finalized_cursor() == cursor()
        work = SimpleNamespace(factory=factory)
        return _no_decision_batch(work, "oci_backend")

    monkeypatch.setattr(coordinator_module, "run_qualification_intake", systemic)
    result = _coordinator(tmp_path, service, cursor).run_qualification_once()

    assert result is not None and result.disposition == "released"
    assert result.lease.reservation_ids == ids
    with _store(tmp_path) as store:
        assert tuple(store.get(value).status for value in ids) == (
            "promoted",
            "promoted",
        )
        assert all(store.qualification_dispositions(value) == () for value in ids)
        event = store.evaluation_lease_events(lease_id=result.lease.lease_id)[-1]
    assert event.event_type == "released"
    assert event.result_digest == result.envelope.digest
    assert event.reason == "systemic_qualification:oci_backend"


def test_candidate_worker_batch_commits_atomically_and_no_remote_work_runs_in_accept(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inside_accept = [False]
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = _promote_all(tmp_path, service, cursor, 2)
    provider.qualification_builder = _qualification_fixture_builder(
        tmp_path,
        service,
        monkeypatch,
        inside_accept=inside_accept,
    )
    original_accept = FinalizedIntakeStore.accept_evaluation_result

    @contextlib.contextmanager
    def observed_accept(self, *args, **kwargs):
        with original_accept(self, *args, **kwargs) as rows:
            inside_accept[0] = True
            try:
                yield rows
            finally:
                inside_accept[0] = False

    monkeypatch.setattr(
        FinalizedIntakeStore,
        "accept_evaluation_result",
        observed_accept,
    )

    def candidate_worker(factory, **_kwargs):
        assert inside_accept == [False]
        with _store(tmp_path) as unlocked:
            assert unlocked.finalized_cursor() == cursor()
        return _no_decision_batch(
            SimpleNamespace(factory=factory),
            "candidate_worker",
        )

    monkeypatch.setattr(
        coordinator_module,
        "run_qualification_intake",
        candidate_worker,
    )
    original_cursor = coordinator_module.EvaluationCoordinator._open_at_durable_cursor

    def observed_open(self, *args, **kwargs):
        assert inside_accept == [False]
        return original_cursor(self, *args, **kwargs)

    monkeypatch.setattr(
        coordinator_module.EvaluationCoordinator,
        "_open_at_durable_cursor",
        observed_open,
    )
    result = _coordinator(tmp_path, service, cursor).run_qualification_once()

    assert result is not None and result.disposition == "completed"
    with _store(tmp_path) as store:
        assert tuple(store.get(value).status for value in ids) == (
            "published",
            "published",
        )
        assert all(len(store.qualification_dispositions(value)) == 1 for value in ids)
        assert store.active_evaluation_leases() == ()


def test_partial_qualification_payload_is_rejected_before_any_cohort_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = _promote_all(tmp_path, service, cursor, 2)
    provider.qualification_builder = _qualification_fixture_builder(
        tmp_path,
        service,
        monkeypatch,
    )
    coordinator = _coordinator(tmp_path, service, cursor)
    claim = coordinator.claim_qualification()
    assert type(claim) is ClaimedQualificationEvaluation
    work = service.plan_qualification(
        claim.candidates,
        claim.screen_receipts,
        state=None,
    )
    prepared = work.factory.build()
    batch = _no_decision_batch(work, "candidate_worker")
    envelope = EvaluationResultEnvelope.seal(
        claim.lease,
        coordinator.readiness,
        service,
        batch,
    )
    object.__setattr__(batch, "outcomes", batch.outcomes[:1])

    with pytest.raises(EvaluationCoordinatorError, match="exact finalized cohort"):
        coordinator.commit_qualification_result(
            claim,
            work,
            prepared,
            batch,
            envelope,
        )

    with _store(tmp_path) as store:
        assert tuple(store.get(value).status for value in ids) == (
            "promoted",
            "promoted",
        )
        assert all(store.qualification_dispositions(value) == () for value in ids)
        assert store.active_evaluation_leases() == (claim.lease,)


def test_remote_commit_reopens_complete_cpu_inventory_before_any_mutation(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = _promote_all(tmp_path, service, cursor, 1)
    coordinator = _coordinator(tmp_path, service, cursor)
    claim = coordinator.claim_qualification()
    assert type(claim) is ClaimedQualificationEvaluation
    authority, batch, envelope, _root, attempt_ref = _remote_commit_product(
        tmp_path,
        coordinator,
        claim,
    )
    missing_root = tmp_path / "missing-cpu-evidence"
    missing_root.mkdir(mode=0o700)

    with pytest.raises(EvaluationCoordinatorError, match="cannot reopen from the CPU CAS"):
        coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=authority,
            incumbent_stack=_remote_incumbent(service),
            incumbent_tree_digest=_h("remote-tree"),
            batch=batch,
            envelope=envelope,
            evidence_root=missing_root,
            evidence_inventory=(attempt_ref,),
        )

    with _store(tmp_path) as store:
        assert tuple(store.get(value).status for value in ids) == ("promoted",)
        assert store.qualification_dispositions(ids[0]) == ()
        assert store.active_evaluation_leases() == (claim.lease,)
        with pytest.raises(IntakeError, match="not initialized"):
            store.evaluation_stack(service.identity)


def test_remote_commit_rejects_wrong_incumbent_and_existing_tree_atomically(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = _promote_all(tmp_path, service, cursor, 1)
    coordinator = _coordinator(tmp_path, service, cursor)
    incumbent = _remote_incumbent(service)
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(
            incumbent,
            tree_digest=_h("authoritative-tree"),
        )
    claim = coordinator.claim_qualification()
    assert type(claim) is ClaimedQualificationEvaluation
    authority, batch, envelope, root, attempt_ref = _remote_commit_product(
        tmp_path,
        coordinator,
        claim,
    )

    wrong_authority = dataclasses.replace(
        authority,
        source_digest=_h("wrong-remote-authority-source"),
    )
    with pytest.raises(EvaluationCoordinatorError, match="qualification result changed"):
        coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=wrong_authority,
            incumbent_stack=incumbent,
            incumbent_tree_digest=_h("authoritative-tree"),
            batch=batch,
            envelope=envelope,
            evidence_root=root,
            evidence_inventory=(attempt_ref,),
        )

    with pytest.raises(EvaluationCoordinatorError, match="differs from the CPU service"):
        coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=authority,
            incumbent_stack=_remote_incumbent(
                service,
                arena_digest=_h("wrong-service"),
                marker="wrong-service",
            ),
            incumbent_tree_digest=_h("authoritative-tree"),
            batch=batch,
            envelope=envelope,
            evidence_root=root,
            evidence_inventory=(attempt_ref,),
        )

    with pytest.raises(EvaluationCoordinatorError, match="durable lease"):
        coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=authority,
            incumbent_stack=incumbent,
            incumbent_tree_digest=_h("wrong-tree"),
            batch=batch,
            envelope=envelope,
            evidence_root=root,
            evidence_inventory=(attempt_ref,),
        )

    with _store(tmp_path) as store:
        retained = store.get(ids[0])
        stack = store.evaluation_stack(service.identity)
        dispositions = store.qualification_dispositions(ids[0])
        active = store.active_evaluation_leases()
    assert retained.status == "promoted"
    assert stack.tree_digest == _h("authoritative-tree")
    assert dispositions == ()
    assert active == (claim.lease,)


@pytest.mark.parametrize("lane", ["", "reproduction"])
def test_claimed_qualification_rejects_blank_or_mixed_screen_lanes(
    tmp_path: Path,
    lane: str,
) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    _promote_all(tmp_path, service, cursor, 2)
    claim = _coordinator(tmp_path, service, cursor).claim_qualification()
    assert type(claim) is ClaimedQualificationEvaluation
    reservations = (
        claim.reservations[0],
        dataclasses.replace(claim.reservations[1], screen_lane=lane),
    )

    with pytest.raises(EvaluationCoordinatorError, match="inconsistent"):
        ClaimedQualificationEvaluation(
            claim.lease,
            reservations,
            claim.publications,
            claim.candidates,
            claim.screen_receipts,
        )


def test_remote_commit_rejects_late_lease_without_consuming_attempt(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    cursor = _CursorAuthority((BLOCK, _block_hash(BLOCK)))
    ids = _promote_all(tmp_path, service, cursor, 1)
    coordinator = _coordinator(
        tmp_path,
        service,
        cursor,
        lease_blocks=2,
    )
    claim = coordinator.claim_qualification()
    assert type(claim) is ClaimedQualificationEvaluation
    authority, batch, envelope, root, attempt_ref = _remote_commit_product(
        tmp_path,
        coordinator,
        claim,
    )
    _advance(tmp_path, cursor, claim.lease.expires_block)

    with pytest.raises(EvaluationCoordinatorError, match="durable lease"):
        coordinator.commit_remote_qualification_result(
            claim,
            authority_manifest=authority,
            incumbent_stack=_remote_incumbent(service),
            incumbent_tree_digest=_h("late-tree"),
            batch=batch,
            envelope=envelope,
            evidence_root=root,
            evidence_inventory=(attempt_ref,),
        )

    with _store(tmp_path) as store:
        retained = store.get(ids[0])
        dispositions = store.qualification_dispositions(ids[0])
        events = store.evaluation_lease_events(lease_id=claim.lease.lease_id)
        active = store.active_evaluation_leases()
    assert retained.status == "promoted"
    assert dispositions == ()
    assert active == ()
    assert [row.event_type for row in events] == ["claimed", "expired"]

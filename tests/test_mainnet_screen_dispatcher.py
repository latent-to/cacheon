from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaScreenReceipt,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    PromotionDecision,
    ScreenGrade,
    ScreenStagePolicy,
    ScreenStageResult,
    Workload,
    WorkloadCell,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain import mainnet_screen_dispatcher as dispatcher_module
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.evaluation_coordinator import EvaluationResultEnvelope, WorkerReadiness
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.publication import publish_worker_bundle
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteEvaluationDispatcher,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
)
from cacheon.chain.remote_worker_registration import verify_registration
from cacheon.copy_fingerprint import SubmittedDeltaFingerprint
from cacheon.stack_identity import canonical_json_bytes, sha256_hex

BLOCK = 10
SCOPE = IntakeScope("0x" + "0" * 64, 14)
POLICY = IntakePolicy(max_cohort=4, expiry_blocks=100)


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _block_hash(block: int) -> str:
    return "0x" + f"{block:064x}"


def _manifest() -> ArenaServiceManifest:
    return ArenaServiceManifest(
        runtime=ArenaRuntimeIdentity(
            arena_id="mainnet-remote-screen-test",
            runtime_digest=_h("runtime"),
            base_engine_digest=_h("engine"),
            validator_overlay_digest=_h("overlay"),
            worker_distribution_digest=_h("worker-distribution"),
            model_revision_digest=_h("model-revision"),
            model_manifest_digest=_h("model-manifest"),
            model_content_digest=_h("model-content"),
            target_architecture="sm120",
            topology_class="tp8-test",
            topology_digest=_h("topology"),
            gpu_count=8,
            tensor_parallel_size=8,
        ),
        workload=Workload(
            _h("prompt-corpus"),
            "mainnet-test-seed-v1",
            (WorkloadCell("s8", 8192, 1024, 64, 8),),
        ),
        capacity=ArenaCapacityPolicy(32, 100, 4, 8, 4, 3, 3, 3),
        screens=NonCrownScreenPolicy(
            tuple(ScreenStagePolicy(stage, 1_000) for stage in SCREEN_STAGES)
        ),
        qualification_policy_digest=_h("qualification-policy"),
        provider_digest=_h("remote-provider"),
    )


def _private_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o400)


def _setup_authority(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    intake_db = private / "intake.sqlite3"
    with FinalizedIntakeStore(intake_db, POLICY, scope=SCOPE) as store:
        store.reserve_finalized(
            (),
            finalized_block=BLOCK,
            finalized_block_hash=_block_hash(BLOCK),
        )

    spool_root = tmp_path / "spool"
    spool_root.mkdir(mode=0o700)
    credential_path = tmp_path / "credential.secret"
    secret = b"s" * 48
    _private_file(credential_path, secret)
    known_hosts = tmp_path / "known_hosts"
    _private_file(known_hosts, b"example.invalid ssh-ed25519 AAAATEST\n")

    manifest = _manifest()
    readiness = WorkerReadiness.for_service(
        dispatcher_module.ArenaService(
            manifest,
            dispatcher_module.RemoteOnlyArenaProvider(manifest.provider_digest),
        ),
        ready_receipt_digest=_h("ready-receipt"),
        ready_epoch=7,
    )
    credential = RemoteWorkerCredential("mainnet-screen-test", secret)
    transport = RemoteWorkerTransportIdentity(
        transport_id="lium-b300-test",
        endpoint_identity_digest=_h("endpoint"),
        protocol_digest=REMOTE_EVALUATION_PROTOCOL_DIGEST,
        credential_digest=credential.digest,
        service_digest=manifest.digest,
        worker_readiness_digest=readiness.digest,
        max_response_bytes=1 << 20,
    )
    registration: dict[str, object] = {
        "adapter_sha256": _h("adapter"),
        "created_at_unix": 1,
        "credential_digest": credential.digest,
        "credential_file_sha256": spool.file_sha256(credential_path),
        "credential_id": credential.credential_id,
        "credential_path": str(credential_path),
        "known_hosts_path": str(known_hosts),
        "known_hosts_sha256": spool.file_sha256(known_hosts),
        "lane_devices": list(range(readiness.gpu_count)),
        "lane_digest": _h("lane"),
        "pod_host": "example.invalid",
        "pod_port": 2222,
        "pod_user": "root",
        "python_executable": sys.executable,
        "python_executable_sha256": spool.file_sha256(
            Path(sys.executable).resolve()
        ),
        "ready_receipt_digest": readiness.ready_receipt_digest,
        "ready_receipt_file_sha256": _h("ready-receipt-file"),
        "remote_service_sha256": _h("remote-service"),
        "schema": spool.SCHEMA_REGISTRATION,
        "service_identity": manifest.service_id,
        "transport_identity": transport.to_dict(),
        "transport_identity_digest": transport.digest,
        "worker_epoch": "a" * 32,
        "worker_readiness": readiness.to_dict(),
        "worker_readiness_digest": readiness.digest,
    }
    registration["registration_digest"] = spool.spool_digest(
        spool.DOMAIN_REGISTRATION, registration
    )
    verify_registration(registration)
    registration_path = tmp_path / "registration.json"
    _private_file(
        registration_path,
        spool.spool_canonical_json(registration) + b"\n",
    )

    policy = {name: getattr(POLICY, name) for name in POLICY.__dataclass_fields__}
    config: dict[str, object] = {
        "arena_service_manifest": manifest.to_dict(),
        "credential_digest": credential.digest,
        "credential_path": str(credential_path),
        "heartbeat_interval_ms": 10_000,
        "heartbeat_join_timeout_ms": 1_000,
        "idle_poll_ms": 10,
        "intake_db": str(intake_db),
        "intake_policy": policy,
        "intake_scope": SCOPE.to_dict(),
        "lease_blocks": 20,
        "lock_attempts": 3,
        "lock_retry_delay_ms": 1,
        "owner": "mainnet-remote-screen-test",
        "registration_digest": registration["registration_digest"],
        "registration_path": str(registration_path),
        "response_timeout_seconds": 30,
        "restart_initial_backoff_ms": 10,
        "restart_max_backoff_ms": 40,
        "schema": dispatcher_module.CONFIG_SCHEMA,
        "spool_root": str(spool_root),
        "transport_identity_digest": transport.digest,
        "transport_poll_seconds": 1,
        "worker_readiness": readiness.to_dict(),
    }
    config_path = tmp_path / "dispatcher.json"
    _private_file(
        config_path,
        spool.spool_canonical_json(config) + b"\n",
    )
    return config_path, config


def test_builds_exact_screen_only_dispatcher_over_live_durable_cursor(
    tmp_path: Path,
) -> None:
    config_path, _ = _setup_authority(tmp_path)
    config = dispatcher_module.load_config(config_path)

    dispatcher = dispatcher_module.build_dispatcher(config)

    assert type(dispatcher) is RemoteEvaluationDispatcher
    assert dispatcher.coordinator.policy == POLICY
    assert dispatcher.coordinator.scope == SCOPE
    assert dispatcher.coordinator.advance_finalized_cursor() == (
        BLOCK,
        _block_hash(BLOCK),
    )
    assert dispatcher.transport.identity.digest == config.transport_identity_digest
    assert dispatcher.credential.digest == config.credential_digest
    assert callable(dispatcher.transport.qualification_publication_resolver)
    assert dispatcher.dispatch_screen_once() is None
    provider = dispatcher.coordinator.service._provider
    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="local arena provider execution is disabled",
    ):
        provider.run_screen(None, None, None)
    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="local arena provider execution is disabled",
    ):
        provider.build_qualification(None)


def test_composed_qualification_claim_is_pinned_singleton_fifo(tmp_path: Path) -> None:
    config_path, raw = _setup_authority(tmp_path)
    intake_db = Path(raw["intake_db"])
    rows = tuple(
        _published_intake_row(tmp_path, intake_db, label=label)[0]
        for label in ("first", "second")
    )
    dispatcher = dispatcher_module.build_dispatcher(dispatcher_module.load_config(config_path))
    coordinator = dispatcher.coordinator
    passing = tuple(
        ScreenStageResult(stage, ScreenGrade.PASS, _h(stage), 1) for stage in SCREEN_STAGES
    )
    for row in rows:
        claim = coordinator.claim_screen()
        assert claim is not None and claim.reservation == row
        receipt = ArenaScreenReceipt(
            coordinator.service.identity,
            claim.candidate.digest,
            claim.candidate.screen_attempt,
            passing,
            PromotionDecision.PROMOTE,
        )
        envelope = EvaluationResultEnvelope.seal(
            claim.lease, coordinator.readiness, coordinator.service, receipt
        )
        coordinator.commit_screen_result(claim, receipt, envelope)
    # Mainnet 2026-08-15: the v3 execution core refuses multi-candidate
    # requests at the deployment factory, so the dispatcher pins singleton
    # claims instead of deriving min(policy.max_cohort, capacity).
    assert coordinator.qualification_max_members == 1
    store, point = coordinator._open_at_durable_cursor()
    try:
        lease = store.claim_evaluation_lease(
            stage="qualification",
            owner=coordinator.owner,
            current_block=point[0],
            lease_blocks=coordinator.lease_blocks,
            max_members=coordinator.qualification_max_members,
        )
    finally:
        store.close()
    assert lease is not None
    assert lease.reservation_ids == (rows[0].reservation_id,)


def _published_intake_row(tmp_path: Path, intake_db: Path, *, label: str):
    source = tmp_path / f"source-{label}"
    source.mkdir(parents=True, mode=0o700)
    leaf = source / "manifest.toml"
    leaf.write_text(f"bundle_id = '{label}'\n")
    leaf.chmod(0o600)
    committed = content_hash(source)
    publication = publish_worker_bundle(
        source,
        tmp_path / "publications",
        committed,
    )
    with FinalizedIntakeStore(intake_db, POLICY, scope=SCOPE) as store:
        reserved = store.reserve_finalized(
            (
                FinalizedArrival(
                    f"miner-{label}",
                    committed,
                    f"https://example.invalid/{label}",
                    BLOCK,
                    _block_hash(BLOCK),
                    0,
                ),
            ),
            finalized_block=BLOCK,
            finalized_block_hash=_block_hash(BLOCK),
        )
        store.mark_fetching(reserved[0].reservation_id)
        row = store.mark_published(
            reserved[0].reservation_id,
            delta_fingerprint=SubmittedDeltaFingerprint(
                "component",
                f"target.{label}",
                _h(f"base:{label}"),
                (f"slot.{label}",),
                _h(f"archive:{label}"),
                _h(f"selected:{label}"),
                _h(f"exact:{label}"),
                (_h(f"source:{label}"),),
                (_h(f"binary:{label}"),),
            ),
            publication_digest=publication.digest,
            publication_root=publication.root,
        )
    return row, publication


def _synthetic_qualification_request(
    reservation_id: str,
    publication_dict: dict[str, object],
) -> RemoteEvaluationRequest:
    body = {
        "candidates": [
            {
                "publication": publication_dict,
                "reservation": {"reservation_digest": reservation_id},
            }
        ]
    }
    body_bytes = canonical_json_bytes(body)
    request = object.__new__(RemoteEvaluationRequest)
    object.__setattr__(request, "stage", "qualification")
    object.__setattr__(request, "body_bytes", body_bytes)
    return request


def test_qualification_publication_resolver_reopens_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    intake_db = private / "intake.sqlite3"
    row, publication = _published_intake_row(tmp_path, intake_db, label="match")
    other_source = tmp_path / "source-other"
    other_source.mkdir(mode=0o700)
    other_leaf = other_source / "manifest.toml"
    other_leaf.write_text("bundle_id = 'other'\n")
    other_leaf.chmod(0o600)
    other = publish_worker_bundle(
        other_source,
        tmp_path / "publications",
        content_hash(other_source),
    )
    resolver = dispatcher_module.make_qualification_publication_resolver(
        intake_db=intake_db,
        policy=POLICY,
        scope=SCOPE,
        store_factory=FinalizedIntakeStore,
    )

    resolved = resolver(
        _synthetic_qualification_request(row.reservation_id, publication.to_dict())
    )
    assert len(resolved) == 1
    assert resolved[0].to_dict() == publication.to_dict()

    with pytest.raises(
        RemoteEvaluationDispatcherError,
        match="differs from authenticated work",
    ):
        resolver(_synthetic_qualification_request(row.reservation_id, other.to_dict()))


def test_qualification_publication_resolver_releases_store_before_tree_reopen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reservation_digest = _h("resolver-unlock")
    publication_dict = {
        "content_hash": _h("content"),
        "digest": _h("publication"),
        "root": str(tmp_path / "publication"),
    }
    state = {"closed": False}

    class _Store:
        def get(self, observed: str) -> object:
            assert observed == reservation_digest
            return SimpleNamespace(
                arrival=SimpleNamespace(content_hash=publication_dict["content_hash"]),
                publication_digest=publication_dict["digest"],
                publication_root=publication_dict["root"],
            )

        def close(self) -> None:
            state["closed"] = True

    class _Publication:
        def to_dict(self) -> dict[str, object]:
            return publication_dict

    def reopen(*_args: object, **_kwargs: object) -> object:
        assert state["closed"] is True
        return _Publication()

    monkeypatch.setattr(dispatcher_module, "reopen_worker_bundle", reopen)
    resolver = dispatcher_module.make_qualification_publication_resolver(
        intake_db=tmp_path / "intake.sqlite3",
        policy=POLICY,
        scope=SCOPE,
        store_factory=lambda *_args, **_kwargs: _Store(),
    )

    resolved = resolver(
        _synthetic_qualification_request(reservation_digest, publication_dict)
    )

    assert len(resolved) == 1


def test_default_dispatcher_reopens_recovery_connection_before_screen_claim(
    tmp_path: Path,
) -> None:
    config_path, raw = _setup_authority(tmp_path)
    intake_db = Path(raw["intake_db"])
    row, _ = _published_intake_row(tmp_path, intake_db, label="recovery-screen")

    # Persist the recovery triggers, then close the commissioning connection.
    # Their authorizing SQLite function is connection-local and must be
    # re-registered by the dispatcher's default store factory.
    with RecoverableFinalizedIntakeStore(intake_db, POLICY, scope=SCOPE):
        pass

    dispatcher = dispatcher_module.build_dispatcher(
        dispatcher_module.load_config(config_path)
    )
    claim = dispatcher.coordinator.claim_screen()

    assert claim is not None
    assert claim.lease.stage == "screen"
    assert claim.lease.reservation_ids == (row.reservation_id,)
    assert claim.reservation.reservation_id == row.reservation_id


def test_config_and_cli_are_closed_and_digest_pinned(tmp_path: Path) -> None:
    config_path, raw = _setup_authority(tmp_path)
    extra = dict(raw)
    extra["candidate_command"] = ["python", "candidate.py"]
    config_path.chmod(0o600)
    config_path.write_bytes(spool.spool_canonical_json(extra) + b"\n")
    config_path.chmod(0o400)

    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="fields are not closed",
    ):
        dispatcher_module.load_config(config_path)

    credential_path = Path(raw["credential_path"])
    credential_path.chmod(0o644)
    raw_config = dict(raw)
    config_path.chmod(0o600)
    config_path.write_bytes(spool.spool_canonical_json(raw_config) + b"\n")
    config_path.chmod(0o400)
    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="credential must be an owner-only regular file",
    ):
        dispatcher_module.load_config(config_path)
    credential_path.chmod(0o400)

    raw["transport_identity_digest"] = _h("another-transport")
    config_path.chmod(0o600)
    config_path.write_bytes(spool.spool_canonical_json(raw) + b"\n")
    config_path.chmod(0o400)
    config = dispatcher_module.load_config(config_path)
    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="transport_identity_digest differs",
    ):
        dispatcher_module.build_dispatcher(config)


def test_live_cursor_rejects_missing_regression_and_scope_drift(
    tmp_path: Path,
) -> None:
    config_path, _ = _setup_authority(tmp_path)
    config = dispatcher_module.load_config(config_path)
    cursor = dispatcher_module.LiveFinalizedCursor(config.intake_db, config.scope)
    assert cursor() == (BLOCK, _block_hash(BLOCK))

    database = sqlite3.connect(config.intake_db)
    database.execute(
        "UPDATE metadata SET value=? WHERE key='finalized_cursor'",
        (json.dumps([BLOCK - 1, _block_hash(BLOCK - 1)]),),
    )
    database.commit()
    database.close()
    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="regressed or changed hash",
    ):
        cursor()

    database = sqlite3.connect(config.intake_db)
    database.execute("DELETE FROM metadata WHERE key='finalized_cursor'")
    database.commit()
    database.close()
    fresh = dispatcher_module.LiveFinalizedCursor(config.intake_db, config.scope)
    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="cursor or intake scope is missing",
    ):
        fresh()

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    ScreenStagePolicy,
    ServingShape,
    WorkloadMixture,
    WorkloadRegime,
)
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.intake import FinalizedIntakeStore, IntakePolicy, IntakeScope
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteEvaluationDispatcher,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
)
from cacheon.stack_identity import sha256_hex
from chainops import mainnet_screen_dispatcher as dispatcher_module
from chainops import remote_worker_service as worker_service


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
        workload=WorkloadMixture(
            _h("prompt-corpus"),
            "mainnet-test-seed-v1",
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

    spool = tmp_path / "spool"
    spool.mkdir(mode=0o700)
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
        "credential_file_sha256": worker_service.file_sha256(credential_path),
        "credential_id": credential.credential_id,
        "credential_path": str(credential_path),
        "known_hosts_path": str(known_hosts),
        "known_hosts_sha256": worker_service.file_sha256(known_hosts),
        "pod_host": "example.invalid",
        "pod_port": 2222,
        "pod_user": "root",
        "ready_receipt_digest": readiness.ready_receipt_digest,
        "ready_receipt_file_sha256": _h("ready-receipt-file"),
        "remote_service_sha256": _h("remote-service"),
        "schema": worker_service.SCHEMA_REGISTRATION,
        "service_identity": manifest.service_id,
        "transport_identity": transport.to_dict(),
        "transport_identity_digest": transport.digest,
        "worker_epoch": "a" * 32,
        "worker_readiness": readiness.to_dict(),
        "worker_readiness_digest": readiness.digest,
    }
    registration["registration_digest"] = worker_service.semantic_digest(
        worker_service.DOMAIN_REGISTRATION, registration
    )
    worker_service.verify_registration(registration)
    registration_path = tmp_path / "registration.json"
    _private_file(
        registration_path,
        worker_service.canonical_json_bytes(registration) + b"\n",
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
        "spool_root": str(spool),
        "transport_identity_digest": transport.digest,
        "transport_poll_seconds": 1,
        "worker_readiness": readiness.to_dict(),
    }
    config_path = tmp_path / "dispatcher.json"
    _private_file(
        config_path,
        worker_service.canonical_json_bytes(config) + b"\n",
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


def test_config_and_cli_are_closed_and_digest_pinned(tmp_path: Path) -> None:
    config_path, raw = _setup_authority(tmp_path)
    extra = dict(raw)
    extra["candidate_command"] = ["python", "candidate.py"]
    config_path.chmod(0o600)
    config_path.write_bytes(worker_service.canonical_json_bytes(extra) + b"\n")
    config_path.chmod(0o400)

    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="fields are not closed",
    ):
        dispatcher_module.load_config(config_path)
    with pytest.raises(SystemExit):
        dispatcher_module.build_parser().parse_args(
            ["--config", str(config_path), "--stage", "qualification"]
        )

    credential_path = Path(raw["credential_path"])
    credential_path.chmod(0o644)
    raw_config = dict(raw)
    config_path.chmod(0o600)
    config_path.write_bytes(worker_service.canonical_json_bytes(raw_config) + b"\n")
    config_path.chmod(0o400)
    with pytest.raises(
        dispatcher_module.MainnetScreenDispatcherError,
        match="credential must be an owner-only regular file",
    ):
        dispatcher_module.load_config(config_path)
    credential_path.chmod(0o400)

    raw["transport_identity_digest"] = _h("another-transport")
    config_path.chmod(0o600)
    config_path.write_bytes(worker_service.canonical_json_bytes(raw) + b"\n")
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


def test_daemon_rebuilds_dispatcher_with_bounded_backoff(
    tmp_path: Path,
) -> None:
    config_path, _ = _setup_authority(tmp_path)
    config = dispatcher_module.load_config(config_path)
    stop = threading.Event()
    factory_calls = []
    waits = []

    class FailingDispatcher:
        def dispatch_screen_once(self):
            raise RuntimeError("worker epoch disappeared")

    class StoppingDispatcher:
        def dispatch_screen_once(self):
            stop.set()
            return None

    def factory(_config):
        factory_calls.append(len(factory_calls))
        return FailingDispatcher() if len(factory_calls) == 1 else StoppingDispatcher()

    def wait(seconds: float) -> bool:
        waits.append(seconds)
        return stop.is_set()

    dispatcher_module.run_forever(
        config,
        stop,
        dispatcher_factory=factory,
        wait=wait,
    )

    assert factory_calls == [0, 1]
    assert waits == [config.restart_initial_backoff_s, config.idle_poll_s]

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain import mainnet_screen_dispatcher as dispatcher_module
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.standing_cpu_supervisor import (
    CONFIG_SCHEMA,
    StandingCpuSupervisor,
    StandingCpuSupervisorError,
    build_standing_supervisor,
    load_standing_config,
    main,
    refuse_terminal_reclaim,
)
from cacheon.stack_identity import canonical_digest, sha256_hex
from cacheon.stack_manifest import EvaluationStackManifest


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _private_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o400)


def _screen_fixtures():
    path = Path(__file__).with_name("test_mainnet_screen_dispatcher.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_standing_screen_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _incumbent(service: ArenaService) -> EvaluationStackManifest:
    snapshot = {
        "composition_rules": [],
        "policy_version": "target-catalog.v1",
        "schema_version": 1,
        "targets": [{"marker": "standing", "target_id": "target.0"}],
    }
    return EvaluationStackManifest(
        runtime_digest=service.manifest.runtime.runtime_digest,
        base_engine_digest=service.manifest.runtime.base_engine_digest,
        arena_digest=service.identity,
        catalog_snapshot=snapshot,
        catalog_digest=canonical_digest("cacheon.target-catalog", snapshot),
        entries={},
    )


def _setup(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    fixtures = _screen_fixtures()
    screen_root = tmp_path / "screen"
    screen_root.mkdir(mode=0o700)
    screen_config_path, _ = fixtures._setup_authority(screen_root)
    private = tmp_path / "standing-private"
    private.mkdir(mode=0o700)
    evidence = private / "qual-evidence"
    evidence.mkdir(mode=0o700)

    manifest = fixtures._manifest()
    service = ArenaService(
        manifest,
        dispatcher_module.RemoteOnlyArenaProvider(manifest.provider_digest),
    )
    incumbent = _incumbent(service)
    incumbent_path = private / "incumbent-stack.json"
    _private_file(
        incumbent_path,
        spool.spool_canonical_json(incumbent.to_dict()) + b"\n",
    )

    standing: dict[str, object] = {
        "enable_incentive": False,
        "enable_settlement": False,
        "enable_weights": False,
        "idle_poll_ms": 25,
        "qualification_evidence_root": str(evidence),
        "qualification_incumbent_stack_path": str(incumbent_path),
        "qualification_incumbent_tree_digest": _h("incumbent-tree"),
        "restart_initial_backoff_ms": 10,
        "restart_max_backoff_ms": 40,
        "schema": CONFIG_SCHEMA,
        "screen_dispatcher_config": str(screen_config_path),
        "stall_timeout_ms": 120_000,
    }
    standing_path = private / "standing.json"
    _private_file(
        standing_path,
        spool.spool_canonical_json(standing) + b"\n",
    )
    return standing_path, standing


def test_load_standing_config_closed_and_weights_disabled(tmp_path: Path) -> None:
    standing_path, raw = _setup(tmp_path)
    config = load_standing_config(standing_path)
    assert config.enable_weights is False
    assert config.enable_settlement is False
    assert config.idle_poll_s == 0.025
    assert config.raw == raw


def test_malformed_standing_config_fail_closed(tmp_path: Path) -> None:
    standing_path, raw = _setup(tmp_path)
    bad = dict(raw)
    bad["schema"] = "not-a-schema"
    standing_path.chmod(0o600)
    standing_path.write_bytes(spool.spool_canonical_json(bad) + b"\n")
    standing_path.chmod(0o400)
    with pytest.raises(StandingCpuSupervisorError, match="schema"):
        load_standing_config(standing_path)


def test_enable_weights_refused_until_sealed(tmp_path: Path) -> None:
    standing_path, raw = _setup(tmp_path)
    bad = dict(raw)
    bad["enable_weights"] = True
    standing_path.chmod(0o600)
    standing_path.write_bytes(spool.spool_canonical_json(bad) + b"\n")
    standing_path.chmod(0o400)
    with pytest.raises(StandingCpuSupervisorError, match="enable_weights"):
        load_standing_config(standing_path)


def test_build_standing_supervisor_omits_weights(tmp_path: Path) -> None:
    standing_path, _ = _setup(tmp_path)
    config = load_standing_config(standing_path)
    supervisor = build_standing_supervisor(config)
    assert type(supervisor) is StandingCpuSupervisor
    assert supervisor.weights_once is None
    assert supervisor.settle_once is None
    assert supervisor.incentive_once is None
    assert callable(supervisor.screen_once)
    assert callable(supervisor.qualification_once)


def test_main_returns_2_on_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert main(["--config", str(missing)]) == 2


def test_terminal_reclaim_still_refused() -> None:
    with pytest.raises(StandingCpuSupervisorError, match="refuses to reclaim"):
        refuse_terminal_reclaim("expired")
    with pytest.raises(StandingCpuSupervisorError, match="refuses to reclaim"):
        refuse_terminal_reclaim("failed")

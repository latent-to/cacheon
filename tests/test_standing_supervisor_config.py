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
        "enable_qualification": True,
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
        "settlement_network": "",
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


def _rewrite(standing_path: Path, row: dict[str, object]) -> None:
    standing_path.chmod(0o600)
    standing_path.write_bytes(spool.spool_canonical_json(row) + b"\n")
    standing_path.chmod(0o400)


def test_enable_settlement_refused_without_its_finalized_clock(tmp_path: Path) -> None:
    """The flag alone is not the authority: settlement needs a head to read.

    ``_settle_pending`` stamps every cohort lease with the finalized head and
    refuses a regressed clock, so an endpoint-less settlement stage could not
    grade at all.  Enabling the flag without ``settlement_network`` is the
    misconfiguration that used to be spelled "not sealed".
    """

    standing_path, raw = _setup(tmp_path)
    bad = dict(raw)
    bad["enable_settlement"] = True
    _rewrite(standing_path, bad)
    with pytest.raises(StandingCpuSupervisorError, match="settlement_network"):
        load_standing_config(standing_path)


def test_settlement_network_may_be_staged_while_settlement_is_disabled(
    tmp_path: Path,
) -> None:
    """A commission stages the endpoint; arming stays a one-field flip.

    Refusing a staged endpoint would force the operator to re-supply it at
    every epoch, and a per-epoch artifact that must be re-edited by hand is
    precisely how the 2026-08-16 ``expiry_blocks`` decision was lost twice:
    edited in place, then orphaned when the next commission regenerated the
    file from defaults. The builder still installs no stage while the flag is
    off, so nothing runs that the config does not claim.
    """

    standing_path, raw = _setup(tmp_path)
    row = dict(raw)
    row["settlement_network"] = "wss://example.invalid"
    _rewrite(standing_path, row)
    config = load_standing_config(standing_path)
    assert config.enable_settlement is False
    assert config.settlement_network == "wss://example.invalid"
    assert build_standing_supervisor(config).settle_once is None


def test_enable_settlement_with_its_clock_loads(tmp_path: Path) -> None:
    """The sealed pair opens, and carries the endpoint the builder will dial."""

    standing_path, raw = _setup(tmp_path)
    row = dict(raw)
    row["enable_settlement"] = True
    row["settlement_network"] = "wss://example.invalid"
    _rewrite(standing_path, row)
    config = load_standing_config(standing_path)
    assert config.enable_settlement is True
    assert config.settlement_network == "wss://example.invalid"
    assert config.enable_weights is False


def test_backoff_ordering_fail_closed(tmp_path: Path) -> None:
    standing_path, raw = _setup(tmp_path)
    bad = dict(raw)
    bad["restart_initial_backoff_ms"] = 80
    standing_path.chmod(0o600)
    standing_path.write_bytes(spool.spool_canonical_json(bad) + b"\n")
    standing_path.chmod(0o400)
    with pytest.raises(StandingCpuSupervisorError, match="exceeds its maximum"):
        load_standing_config(standing_path)


def test_group_writable_config_fail_closed(tmp_path: Path) -> None:
    standing_path, _ = _setup(tmp_path)
    standing_path.chmod(0o664)
    with pytest.raises(
        StandingCpuSupervisorError, match="owner-controlled regular file"
    ):
        load_standing_config(standing_path)


def test_symlinked_config_fail_closed(tmp_path: Path) -> None:
    standing_path, _ = _setup(tmp_path)
    link = standing_path.parent / "standing-link.json"
    link.symlink_to(standing_path)
    with pytest.raises(
        StandingCpuSupervisorError, match="owner-controlled regular file"
    ):
        load_standing_config(link)


def test_build_standing_supervisor_omits_weights(tmp_path: Path) -> None:
    standing_path, _ = _setup(tmp_path)
    config = load_standing_config(standing_path)
    supervisor = build_standing_supervisor(config)
    assert type(supervisor) is StandingCpuSupervisor
    assert supervisor.weights_once is None
    assert supervisor.settle_once is None
    assert callable(supervisor.screen_once)
    assert callable(supervisor.qualification_once)


def test_enabled_settlement_is_actually_wired_into_the_supervisor(
    tmp_path: Path, monkeypatch
) -> None:
    """The flag has to reach ``settle_once``, not just survive validation.

    Until 2026-08-18 the composed supervisor passed ``settle_once=None``
    unconditionally while the loader refused the flag outright, so
    ``settlement_stage`` -- fully written -- could never run: mainnet screened
    and qualified for weeks and settled nothing. A loader that accepts the flag
    without a builder that installs the stage would reproduce exactly that.
    """

    from cacheon import chain

    dialed: list[str] = []

    def _fake_connect(network: str, **_kwargs: object) -> object:
        dialed.append(network)
        return object()

    monkeypatch.setattr(chain, "connect", _fake_connect)

    standing_path, raw = _setup(tmp_path)
    row = dict(raw)
    row["enable_settlement"] = True
    row["settlement_network"] = "wss://example.invalid"
    settling_path = standing_path.parent / "standing-settling.json"
    _private_file(settling_path, spool.spool_canonical_json(row) + b"\n")

    supervisor = build_standing_supervisor(load_standing_config(settling_path))
    assert callable(supervisor.settle_once)
    assert dialed == ["wss://example.invalid"]
    # Publication remains a separate authority behind its own flag.
    assert supervisor.weights_once is None


def test_disabled_qualification_gates_the_stage_and_screens_still_claim(
    tmp_path: Path,
) -> None:
    # Operator gate: while a qualification-side defect is under repair, the
    # stage is held without touching screens (observed need 2026-08-10: a
    # deterministic arm-boot failure was consuming every claim window).
    standing_path, raw = _setup(tmp_path)
    gated = dict(raw)
    gated["enable_qualification"] = False
    gated_path = standing_path.parent / "standing-gated.json"
    _private_file(
        gated_path,
        spool.spool_canonical_json(gated) + b"\n",
    )
    config = load_standing_config(gated_path)
    assert config.enable_qualification is False
    supervisor = build_standing_supervisor(config)
    assert supervisor.qualification_once is None
    assert callable(supervisor.screen_once)
    status = supervisor.tick()
    assert status.last_stage in ("screen", "idle")


def test_main_returns_2_on_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert main(["--config", str(missing)]) == 2


def test_terminal_reclaim_still_refused() -> None:
    with pytest.raises(StandingCpuSupervisorError, match="refuses to reclaim"):
        refuse_terminal_reclaim("expired")
    with pytest.raises(StandingCpuSupervisorError, match="refuses to reclaim"):
        refuse_terminal_reclaim("failed")
